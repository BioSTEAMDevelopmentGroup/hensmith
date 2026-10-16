# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Targeting: what any heat exchanger network of a set of process streams can
achieve, before (or instead of) synthesizing one. `PinchTargets` holds the
problem table of the streams and derives from it the composite and grand
composite curves, the multiple-utility targets priced by biosteam
(`PinchTargets.utility_targets`) and the balanced composite curves;
`plot_composite_curves` and `plot_grand_composite_curve` draw them.
"""
import math
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from warnings import warn
import numpy as np
import biosteam as bst
from ._curves import _T_EQ
from ._planner import _approach_contributions, _contribution_value
from .hxn_synthesis import (_problem_table, _pinch_cut, _format_H,
    _arrange_heat_utilities, _process_streams, _explicit_contributions)

__all__ = ('PinchTargets', 'plot_composite_curves',
           'plot_grand_composite_curve')

#: Kinds of the constraint positions of the grand composite curve (see
#: `_positions`): the heat arriving at a grid temperature (before its point
#: loads), the heat leaving it (after them), and the heat just above and
#: just below a utility level strictly between two grid temperatures.
_ARRIVING, _LEAVING, _ABOVE, _BELOW = 0, 1, 2, 3

UtilityTarget = namedtuple('UtilityTarget', (
    'ID', 'agent', 'heating', 'level', 'duty', 'utility_duty', 'flow', 'cost',
    'T_in', 'T_out', 'T_utility_in', 'T_utility_out'))

UtilityTarget.__doc__ = """
One utility of `PinchTargets.utility_targets`: an agent placed on the grand
composite curve and priced by biosteam as a single heat utility over the
temperatures it serves.

Unlike biosteam's signed `HeatUtility.duty`, `duty` is a magnitude: whether
the utility heats or cools the process is the `heating` flag.

Attributes
----------
ID : str or None
    ID of the agent; None for a remainder that no agent can serve.
agent : UtilityAgent or None
    The agent; None for an unserved remainder.
heating : bool
    True for a heating utility (heat given to cold streams).
level : float
    Shifted temperature level of the utility on the grand composite curve
    [K]; +inf for fuel and for unserved heating, -inf for unserved cooling.
duty : float
    Process-side heat [kJ/hr] (a magnitude).
utility_duty : float
    Utility-side heat, ``duty`` divided by the agent's heat transfer
    efficiency [kJ/hr], from biosteam; nan when unserved.
flow : float
    Flow rate of the agent [kmol/hr], from biosteam; nan when unserved.
cost : float
    Cost of the utility [USD/hr], from biosteam; nan when unserved.
T_in, T_out : float
    Process temperatures at which the utility was priced [K]: the inlet and
    outlet of the stretch of the process it serves.
T_utility_in, T_utility_out : float
    Inlet and outlet temperature of the agent [K], from the priced
    `HeatUtility`; nan when unserved.

"""

def _stream_knots(grid, j, shifted):
    """
    The knots of stream j on the table's grid, ascending in temperature, as
    ``(T, Hl, Hr)``: real temperatures (shifted ones if `shifted`), and the
    left (low-enthalpy) and right (high-enthalpy) enthalpy at each, measured
    from the stream's cold end.
    """
    k_hi, k_lo = grid['k_hi'][j], grid['k_lo'][j]
    sl = slice(k_lo, k_hi - 1 if k_hi > 0 else None, -1)
    T = grid['Ts'][sl] if shifted else grid['Ts'][sl] + grid['shift'][j]
    H0 = grid['Hl'][j, k_lo]
    return T, grid['Hl'][j][sl] - H0, grid['Hr'][j][sl] - H0

def _member_values(member, R):
    """
    Left and right enthalpy of a composite member ``(T, Hl, Hr)`` at each
    temperature of `R`: its knot values at a knot (within `_T_EQ`), else its
    chord from ``Hr[i]`` at ``T[i]`` to ``Hl[i+1]`` at ``T[i+1]``; 0 below its
    range and its total above it.
    """
    T, Hl, Hr = member
    n = T.size
    il = np.searchsorted(T, R - _T_EQ, 'left')
    ir = np.searchsorted(T, R + _T_EQ, 'right')
    if n > 1:
        i1 = np.clip(il, 1, n - 1)
        i0 = i1 - 1
        x0 = T[i0]
        y0 = Hr[i0]
        chord = y0 + (Hl[i1] - y0) * (R - x0) / (T[i1] - x0)
        outside = np.where(il == 0, Hl[0], np.where(il >= n, Hr[-1], chord))
    else:
        outside = np.where(il == 0, Hl[0], Hr[-1])
    match = ir > il
    hl = np.where(match, Hl[np.minimum(il, n - 1)], outside)
    hr = np.where(match, Hr[np.maximum(ir - 1, 0)], outside)
    return hl, hr

def _composite(members, inv_h=None):
    """
    Composite of `members` (each ``(T, Hl, Hr)``, ascending T, enthalpies
    from 0, see `_stream_knots`), as segments ``(Ha, Hb, Ta, Tb, w)`` in
    ascending temperature: slopes from ``(Ha, Ta)`` to ``(Hb, Tb)`` and flats
    (``Ta == Tb``) of positive heat. Each member is linear in H within a
    segment (on a flat in proportion to its own flat), so a stretch [x, y] of
    a segment carries ``sum(q_m * inv_h[m]) = (y - x) * w`` (`inv_h`: one
    weight per member, 1 by default).
    """
    if not members:
        empty = np.empty(0)
        return empty, empty, empty, empty, empty
    inv_h = np.ones(len(members)) if inv_h is None else np.asarray(inv_h, float)
    values = np.unique(np.concatenate([m[0] for m in members]))
    # temperatures closer than _T_EQ are one knot, represented by the
    # highest member of the cluster (as in `_problem_table`)
    cluster = np.concatenate([[0], np.cumsum(np.diff(values) > _T_EQ)])
    R = values[np.flatnonzero(np.append(np.diff(cluster) > 0, True))]
    n = R.size
    hl = np.empty((len(members), n))
    hr = np.empty((len(members), n))
    for m, member in enumerate(members):
        hl[m], hr[m] = _member_values(member, R)
    Cl = hl.sum(axis=0)
    Cr = hr.sum(axis=0)
    # interleaved: the flat at R[0], the slope to R[1], the flat at R[1], ...
    Ha = np.empty(2 * n - 1)
    Hb = np.empty(2 * n - 1)
    Ta = np.empty(2 * n - 1)
    Tb = np.empty(2 * n - 1)
    d = np.empty((len(members), 2 * n - 1))
    Ha[0::2], Hb[0::2], Ta[0::2], Tb[0::2] = Cl, Cr, R, R
    d[:, 0::2] = hr - hl
    Ha[1::2], Hb[1::2], Ta[1::2], Tb[1::2] = Cr[:-1], Cl[1:], R[:-1], R[1:]
    d[:, 1::2] = hl[:, 1:] - hr[:, :-1]
    keep = Hb > Ha
    Ha, Hb, Ta, Tb = Ha[keep], Hb[keep], Ta[keep], Tb[keep]
    w = (inv_h @ d[:, keep]) / (Hb - Ha)
    return Ha, Hb, Ta, Tb, w

def _curve(segments):
    """The ``(T, H)`` polyline through `segments` (see `_composite`)."""
    Ha, Hb, Ta, Tb, _ = segments
    if not Ha.size: return np.empty(0), np.empty(0)
    start = np.ones(Ha.size, dtype=bool)
    start[1:] = (Ha[1:] != Hb[:-1]) | (Ta[1:] != Tb[:-1])
    keep = np.column_stack([start, np.ones(Ha.size, dtype=bool)]).ravel()
    T = np.column_stack([Ta, Tb]).ravel()[keep]
    H = np.column_stack([Ha, Hb]).ravel()[keep]
    return T, H

def _snap(Ts, level):
    """`level`, or the first grid temperature within `_T_EQ` of it."""
    k = np.flatnonzero(np.abs(Ts - level) <= _T_EQ)
    return float(Ts[k[0]]) if k.size else level

def _positions(Ts, arr, lea, levels):
    """
    The constraint positions of the grand composite curve (`arr`, `lea`:
    the heat arriving at and leaving each grid temperature `Ts`) for the
    utility `levels`, top to bottom, as arrays ``(T, G, kind, k)``: per grid
    index k the arriving and then the leaving heat, and per level l strictly
    between ``Ts[k]`` and ``Ts[k+1]`` (in descending l, after the leaving
    heat at ``Ts[k]``) the heat just above and just below l, both the linear
    value of the curve at l (`_ABOVE`, `_BELOW`).
    """
    n = Ts.size
    T = np.repeat(Ts, 2)
    G = np.column_stack([arr, lea]).ravel()
    kind = np.tile([_ARRIVING, _LEAVING], n)
    k = np.repeat(np.arange(n), 2)
    inner = sorted({l for l in levels if Ts[-1] < l < Ts[0]
                    and not np.any(Ts == l)}, reverse=True)
    if inner:
        l = np.array(inner)
        ki = np.searchsorted(-Ts, -l, 'right') - 1 # Ts[ki] > l > Ts[ki + 1]
        Gl = lea[ki] + (arr[ki + 1] - lea[ki]) * (Ts[ki] - l) / (Ts[ki] - Ts[ki + 1])
        index = np.repeat(2 * (ki + 1), 2)
        T = np.insert(T, index, np.repeat(l, 2))
        G = np.insert(G, index, np.repeat(Gl, 2))
        kind = np.insert(kind, index, np.tile([_ABOVE, _BELOW], l.size))
        k = np.insert(k, index, np.repeat(ki, 2))
    return T, G, kind, k

def _gap(positions, level):
    """
    Index of the first position below the gap in which a utility at the
    (snapped) `level` acts: 0 above the grid, the number of positions below
    it, the leaving heat at a grid temperature, else the position just below
    the level.
    """
    T, _, kind, _ = positions
    if level > T[0]: return 0
    if level < T[-1]: return T.size
    i = np.flatnonzero((T == level) & ((kind == _LEAVING) | (kind == _BELOW)))
    return int(i[0])

def _balance(positions, heating, cooling):
    """
    The balanced heat flow at every position: the curve minus the heating
    utilities ``(level, q)`` acting below each position and the cooling ones
    acting above it.
    """
    B = positions[1].copy()
    for level, q in heating: B[:_gap(positions, level)] -= q
    for level, q in cooling: B[_gap(positions, level):] -= q
    return B

@contextmanager
def _isolated_pricing(agents):
    """
    Price heat utilities of `agents` without side effects: each agent's
    `utility_stream_dump` is replaced by a new empty list (so every pricing
    gets fresh utility streams, and the costs do not depend on what was
    priced before) and new streams are registered in a temporary flowsheet;
    the original dumps are restored on exit.
    """
    dumps = {id(a): (a, a.utility_stream_dump) for a in agents}
    try:
        for a, _ in dumps.values(): a.utility_stream_dump = []
        with bst.Flowsheet('_hensmith_targets'):
            yield
    finally:
        for a, dump in dumps.values(): a.utility_stream_dump = dump

def _unit_label(unit):
    """``unit.ID``, or ``'<owner.ID> - <unit.ID>'`` for an auxiliary unit."""
    return unit.ID if unit.owner is unit else f'{unit.owner.ID} - {unit.ID}'

def _utility_member(u):
    """The composite member ``(T, Hl, Hr)`` of a non-fuel `UtilityTarget`:
    isothermal at the agent's temperature for a latent agent, else linear
    between the agent's inlet and outlet temperatures."""
    T_a, T_b = sorted((u.T_utility_in, u.T_utility_out))
    if u.agent.T_limit is None or T_b - T_a <= _T_EQ:
        T = u.agent.T if u.agent.T_limit is None else T_b
        return np.array([T]), np.array([0.]), np.array([u.duty])
    return (np.array([T_a, T_b]), np.array([0., u.duty]),
            np.array([0., u.duty]))


class PinchTargets:
    """
    Pinch analysis targets of a set of process streams: the problem table
    (`hensmith.hxn_synthesis.problem_table`) and what follows from it, the
    composite and grand composite curves, the targets of several utilities
    (`utility_targets`) priced by biosteam, and the balanced composite
    curves. Build it from heat utilities with `from_heat_utilities` (as
    `hensmith.hxn_synthesis.synthesize_network` prepares them) or from
    streams as `problem_table` takes them.

    Parameters
    ----------
    streams_inlet : list[Stream]
        Inlet of each process stream.
    streams_quenched : list[Stream]
        Corresponding outlets, re-flashed at their enthalpy.
    is_hot : Sequence[bool]
        True where the stream is cooled.
    T_min_app : float, optional
        Minimum approach temperature [K]. Defaults to 5.
    approach_contributions : Sequence[float or None], optional
        Approach temperature contribution of each stream [K], in stream
        order, as `problem_table` takes them: None or nan (unspecified:
        ``T_min_app / 2``), or a finite number > 0 (`T_min_app` must then be
        finite and > 0 too). All unspecified is the same as None.
    labels : list[str], optional
        Display name of each stream. Defaults to the IDs of `streams_inlet`.
    curves : list[StreamCurve], optional
        Prebuilt temperature-enthalpy curves of the streams (built here if
        not given).

    Attributes
    ----------
    T_min_app : float
        Minimum approach temperature [K], as given.
    approach_contributions : tuple[float or None] or None
        Approach temperature contribution of each stream [K] in stream order
        (None: ``T_min_app / 2``), or None if no stream has one.
    is_hot : numpy.ndarray
        (N,) bool, True for hot (cooled) streams.
    labels : list[str]
        Display name of each stream.
    streams_inlet, streams_quenched : list[Stream]
        The process streams.
    curves : list[StreamCurve]
        Temperature-enthalpy curve of each stream (independent of
        `T_min_app`).
    table : ProblemTable
        The problem table.
    grid : dict
        The table's grid (see `hensmith.hxn_synthesis._problem_table`).
    shift : numpy.ndarray
        (N,) shift of each stream [K]: its real temperature at shifted
        temperature T* is ``T* + shift[j]``.
    Q_hot, Q_cold : float
        Minimum hot and cold utility [kJ/hr], process side (the table's
        `hot_util_load` and `cold_util_load`).
    pinch_T : float
        Shifted pinch temperature [K].
    cut : str
        'below' or 'above': the side of the pinch to which the point loads
        at the pinch temperature belong (see
        `hensmith.hxn_synthesis._pinch_cut`).
    heat_utilities : list[HeatUtility] or None
        The heat utilities the streams were taken from, in stream order;
        None when built from streams.
    units : list[Unit] or None
        The unit of each stream (``hu.unit``), in stream order; None when
        built from streams.
    force_ideal_thermo : bool
        Whether the stream copies use ideal thermodynamics (False when built
        from streams).

    Notes
    -----
    *Composite curves.* `composite_curves` adds up the streams on the
    table's own grid, whose knots hold every breakpoint of every stream, so
    the composites reproduce the table to rounding: the cold composite
    starts at the cold utility target and ends the hot utility target to
    the right of the hot one. On the shifted scale the composites touch at
    the pinch (if there is hot utility). On the real scale they are at
    least `T_min_app` apart, or, with approach contributions, at least the
    smallest hot and the smallest cold contribution apart (unspecified ones
    ``T_min_app / 2``); the shifted composites then carry the pinch.

    *Grand composite curve.* `grand_composite_curve` is the heat cascaded
    through each shifted grid temperature with the minimum hot utility, both
    as it arrives (before the point loads at that temperature) and as it
    leaves (after them). It is non-negative and zero at the pinch.

    *Multiple utilities.* `utility_targets` places biosteam's agents on the
    grand composite curve, heating and cooling separately. biosteam lets a
    heating agent serve real temperatures below ``R = agent.T - agent.dT -
    dT`` and a cooling agent real temperatures above ``R = agent.T +
    agent.dT + dT`` (``dT = bst.HeatUtility.dT``, read at call time). On the
    shifted scale a stream with shift s may take an agent's heat only up to
    ``R - s``; the agent's level is the highest single level that serves no
    stream above its own limit (lowest for cooling): ``R`` for heating and
    ``R - T_min_app`` for cooling without approach contributions, which is
    biosteam's rule; with contributions it can be conservative for streams
    with slack. Non-fuel agents are placed in list order (biosteam's first
    feasible agent: for the default lists the lowest steam and the warmest
    coolant first), each as much as the curve above its level (below it, for
    cooling) allows given the agents already placed; what remains goes to
    the first fuel agent, or is left unserved with a RuntimeWarning (nan
    cost). A level equal to a grid temperature serves that temperature's
    point loads (biosteam's own rule is strict; they differ only at exact
    equality).

    Each agent is priced as one biosteam heat utility over the stretch it
    serves, from its level to the next zero of the balanced curve (the
    pinch, or a utility pinch), at real temperatures: shifted ones plus the
    largest shift of the cold streams there (heating) or the smallest of the
    hot streams (cooling), exactly the streams' own temperatures without
    approach contributions and an approximation with them. So a sensible
    agent's outlet follows the temperatures it actually serves, and biosteam
    divides the duty by the agent's heat transfer efficiency. Pricing uses
    fresh utility streams in a temporary flowsheet: costs are
    deterministic, and the flowsheet's registry and the agents' stream dumps
    are untouched.

    *Balanced composite curves.* `balanced_composite_curves` adds the
    utilities to the real composites (heating ones to the hot composite,
    cooling ones to the cold one), so both end at the same heat: a latent
    agent as a flat at its temperature, a sensible one as a line between its
    inlet and outlet temperatures, and fuel as a furnace at infinite
    temperature. `balanced_grand_composite_curve` is the cascade net of the
    utilities, zero at the process pinch and at every utility pinch.

    Examples
    --------
    One hot stream cooled 400 -> 300 K and one cold stream heated 330 ->
    440 K, both with a heat capacity flow rate of 1 kW/K (1000 kmol/hr of a
    pseudo-component with a molar heat capacity of 3.6 J/mol/K):

    >>> import biosteam as bst, thermosteam as tmo
    >>> from hensmith import PinchTargets
    >>> fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1.,
    ...                      Cn=3.6, default=True)
    >>> bst.settings.set_thermo([fluid])
    >>> bst.main_flowsheet.set_flowsheet('targeting_example')
    >>> H1 = bst.HXutility('H1', ins=bst.Stream('H1_in', Fluid=1000., T=400.),
    ...                    T=300., rigorous=False)
    >>> C1 = bst.HXutility('C1', ins=bst.Stream('C1_in', Fluid=1000., T=330.),
    ...                    T=440., rigorous=False)
    >>> H1.simulate(); C1.simulate()
    >>> hus = [H1.heat_utilities[0], C1.heat_utilities[0]]
    >>> targets = PinchTargets.from_heat_utilities(hus, T_min_app=10.)
    >>> targets
    <PinchTargets: 2 streams, T_min_app = 10 K, Q_hot = 1.800e+05 kJ/hr, Q_cold = 1.440e+05 kJ/hr>
    >>> targets.labels, targets.pinch_T
    (['C1', 'H1'], 390.0)

    The composite curves [K, kW], on the knots of the table's grid (340 K
    is the cold inlet and 390 K the hot inlet, each shifted to the other
    side): the cold one starts at the cold utility target, 40 kW, and ends
    50 kW (the hot utility target) to the right of the hot one.

    >>> hot_T, hot_H, cold_T, cold_H = targets.composite_curves()
    >>> hot_T, hot_H / 3600
    (array([300., 340., 400.]), array([  0.,  40., 100.]))
    >>> cold_T, cold_H / 3600
    (array([330., 390., 440.]), array([ 40., 100., 150.]))

    The utility targets with biosteam's default agents [kW]: low pressure
    steam serves the cold stream up to its level, 407.189 K, and medium
    pressure steam the rest; cooling water serves the hot stream down to its
    level (302.372 K shifted) and chilled water the rest.

    >>> utilities = targets.utility_targets()
    >>> [(u.ID, round(u.duty / 3600, 3)) for u in utilities]
    [('low_pressure_steam', 17.189), ('medium_pressure_steam', 32.811), ('cooling_water', 27.628), ('chilled_water', 12.372)]
    >>> [(round(u.T_in, 3), round(u.T_out, 3)) for u in utilities]
    [(390.0, 407.189), (407.189, 440.0), (340.0, 312.372), (312.372, 300.0)]
    >>> round(sum(u.cost for u in utilities), 4)  # USD/hr
    1.656

    The balanced composite curves end at the same heat:

    >>> hot_T, hot_H, cold_T, cold_H = targets.balanced_composite_curves(utilities)
    >>> round(hot_H[-1] / 3600, 6), round(cold_H[-1] / 3600, 6)
    (150.0, 150.0)

    The same streams at another minimum approach temperature (the curves
    are reused):

    >>> targets.at(20.)
    <PinchTargets: 2 streams, T_min_app = 20 K, Q_hot = 2.160e+05 kJ/hr, Q_cold = 1.800e+05 kJ/hr>

    """
    def __init__(self, streams_inlet, streams_quenched, is_hot, T_min_app=5.,
                 approach_contributions=None, labels=None, curves=None):
        N = len(streams_inlet)
        if not N:
            raise ValueError('PinchTargets needs at least one process stream')
        if len(streams_quenched) != N:
            raise ValueError(f'{len(streams_quenched)} outlets given for '
                             f'{N} inlets')
        is_hot = np.asarray(is_hot, dtype=bool)
        if is_hot.shape != (N,):
            raise ValueError(f'{is_hot.size} is_hot values given for {N} '
                             'streams')
        x = approach_contributions
        if x is not None:
            if (not isinstance(x, (Sequence, np.ndarray))
                    or isinstance(x, (str, bytes))):
                raise TypeError('approach_contributions must be a sequence '
                                'with one value per stream, not '
                                f'{type(x).__name__!r}')
            x = [_contribution_value(v, f'stream {j}') for j, v in enumerate(x)]
            if all(v is None for v in x):
                x = None
            else: # checks T_min_app, then the length
                _approach_contributions(N, T_min_app, x)
                x = tuple(x)
        if labels is None:
            labels = [s.ID for s in streams_inlet]
        elif len(labels) != N:
            raise ValueError(f'{len(labels)} labels given for {N} streams')
        if x is None:
            table, curves, grid = _problem_table(
                streams_inlet, streams_quenched, is_hot, T_min_app, curves)
        else:
            table, curves, grid = _problem_table(
                streams_inlet, streams_quenched, is_hot, T_min_app, curves,
                approach_contributions=list(x))
        self.T_min_app = T_min_app
        self.approach_contributions = x
        self.is_hot = is_hot
        self.labels = list(labels)
        self.streams_inlet = streams_inlet
        self.streams_quenched = streams_quenched
        self.curves = curves
        self.table = table
        self.grid = grid
        self.shift = grid['shift']
        self.Q_hot = table.hot_util_load
        self.Q_cold = table.cold_util_load
        self.pinch_T = table.pinch_T
        self.cut = _pinch_cut(table)
        self.heat_utilities = None
        self.units = None
        self.force_ideal_thermo = False
        #: Grid index of the pinch.
        self._k_pinch = int(np.flatnonzero(table.Ts == table.pinch_T)[0])
        #: Threshold of every heat comparison [kJ/hr], the table's own.
        self._tol = 1e-9 * float(np.abs(
            np.array([c.H_out - c.H_in for c in curves])).sum())

    @classmethod
    def from_heat_utilities(cls, hus, T_min_app=5., approach_contributions=None,
                            force_ideal_thermo=False, sort_hus_by_T=False):
        """
        Targets of the process streams behind heat utilities, prepared
        exactly as `hensmith.hxn_synthesis.synthesize_network` prepares them
        (so the targets equal the synthesis's, bit for bit).

        Parameters
        ----------
        hus : Iterable[HeatUtility]
            Heat utilities of heat exchangers (``hu.unit``). Zero-duty ones
            are dropped.
        T_min_app : float, optional
            Minimum approach temperature [K]. Defaults to 5.
        approach_contributions : Mapping, Sequence or callable, optional
            Approach temperature contributions [K] in any form
            `synthesize_network` takes (a sequence is aligned with `hus` as
            given, zero-duty heat utilities included). Defaults to None.
        force_ideal_thermo : bool, optional
            Copy the streams with ideal thermodynamics. Defaults to False.
        sort_hus_by_T : bool, optional
            Order the streams by inlet temperature, as the synthesis's
            option. Defaults to False.

        Returns
        -------
        PinchTargets
            Streams in stream order (heating utilities, i.e. cold streams,
            first), labeled by unit (``'<owner> - <unit>'`` for an
            auxiliary unit).

        """
        hus = list(hus)
        values = _explicit_contributions(hus, approach_contributions, T_min_app)
        arranged, N_heating = _arrange_heat_utilities(hus, sort_hus_by_T)
        if values is None:
            contributions = None
        else: # in stream order
            value_of = dict(zip(map(id, hus), values))
            contributions = [value_of[id(hu)] for hu in arranged]
        is_hot = [False] * N_heating + [True] * (len(arranged) - N_heating)
        return cls._from_exchangers([hu.unit for hu in arranged], is_hot,
                                    T_min_app, contributions,
                                    force_ideal_thermo, None, arranged)

    @classmethod
    def _from_exchangers(cls, hxs, is_hot, T_min_app, approach_contributions,
                         force_ideal_thermo, labels=None, heat_utilities=None):
        """Targets of the process streams of the exchangers `hxs` (stream
        order, contributions resolved in stream order), labeled by unit by
        default."""
        streams_inlet, streams_quenched = _process_streams(hxs, force_ideal_thermo)
        if labels is None: labels = [_unit_label(hx) for hx in hxs]
        self = cls(streams_inlet, streams_quenched, is_hot, T_min_app,
                   approach_contributions, labels)
        self.units = list(hxs)
        self.heat_utilities = None if heat_utilities is None else list(heat_utilities)
        self.force_ideal_thermo = bool(force_ideal_thermo)
        return self

    def at(self, T_min_app, scale_contributions=False):
        """
        The targets of the same streams at another minimum approach
        temperature, reusing their curves (no flash).

        Parameters
        ----------
        T_min_app : float
            Minimum approach temperature [K].
        scale_contributions : bool, optional
            Scale the specified approach contributions by ``T_min_app /
            self.T_min_app`` (the contribution scaling of contribution-based
            supertargeting, Ahmad, Linnhoff & Smith 1990); by default they
            stay as they are. Unspecified ones are ``T_min_app / 2`` either
            way. Defaults to False.

        Returns
        -------
        PinchTargets
            With the same streams, curves, labels, units and heat utilities.

        """
        c = self.approach_contributions
        if c is not None and scale_contributions:
            _approach_contributions(len(c), T_min_app, c) # checks T_min_app
            c = [None if x is None else x * T_min_app / self.T_min_app
                 for x in c]
        new = type(self)(self.streams_inlet, self.streams_quenched,
                         self.is_hot, T_min_app, c, self.labels, self.curves)
        new.heat_utilities = self.heat_utilities
        new.units = self.units
        new.force_ideal_thermo = self.force_ideal_thermo
        return new

    def __repr__(self):
        N = self.is_hot.size
        return (f'<{type(self).__name__}: {N} stream{"s" if N != 1 else ""}, '
                f'T_min_app = {self.T_min_app:g} K, Q_hot = {self.Q_hot:.3e} '
                f'kJ/hr, Q_cold = {self.Q_cold:.3e} kJ/hr>')

    # %% Curves

    def _members(self, hot, shifted=False):
        """The composite members of the hot (or cold) streams."""
        return [_stream_knots(self.grid, j, shifted)
                for j in np.flatnonzero(self.is_hot == hot)]

    def composite_curves(self, shifted=False):
        """
        The hot and cold composite curves.

        Parameters
        ----------
        shifted : bool, optional
            Return shifted temperatures (each stream's real temperature
            minus its shift; the scale of the problem table) instead of real
            ones. Defaults to False.

        Returns
        -------
        hot_T, hot_H, cold_T, cold_H : numpy.ndarray
            Temperatures [K] and cumulative heat [kJ/hr] of each composite,
            ascending; the hot one starts at 0 and the cold one at the cold
            utility target. Empty if there is no stream of that kind.

        """
        hot_T, hot_H = _curve(_composite(self._members(True, shifted)))
        cold_T, cold_H = _curve(_composite(self._members(False, shifted)))
        return hot_T, hot_H, cold_T, cold_H + self.Q_cold

    def _gcc(self):
        """The grid temperatures and the heat cascaded with the hot utility
        target arriving at and leaving each, ``(Ts, arr, lea)``."""
        table = self.table
        leaving = table.residual + table.hot_util_load
        arriving = leaving - table.point_H.sum(axis=0)
        return table.Ts, arriving, leaving

    def grand_composite_curve(self):
        """
        The grand composite curve.

        Returns
        -------
        T : numpy.ndarray
            Shifted grid temperatures [K], descending, each twice.
        H : numpy.ndarray
            Heat cascaded with the hot utility target [kJ/hr]: at each grid
            temperature the heat arriving (before its point loads) and then
            the heat leaving it (after them). It starts at the hot utility
            target and ends at the cold one, to rounding, and is zero at the
            pinch.

        """
        Ts, arr, lea = self._gcc()
        return np.repeat(Ts, 2), np.column_stack([arr, lea]).ravel()

    # %% Utilities

    def _side_heat(self):
        """Each stream's heat above and below the pinch [kJ/hr] (all below
        for a threshold problem)."""
        table = self.table
        k = self._k_pinch
        interval_H = np.abs(table.interval_H)
        point_H = np.abs(table.point_H)
        above = interval_H[:, :k].sum(axis=1) + point_H[:, :k].sum(axis=1)
        below = interval_H[:, k:].sum(axis=1) + point_H[:, k+1:].sum(axis=1)
        if self.cut == 'above': above = above + point_H[:, k]
        else: below = below + point_H[:, k]
        if self.Q_hot == 0.: above, below = 0. * above, below + above
        return above, below

    def _levels(self, agents, heating):
        """
        ``[(agent, level)]``: each agent's shifted level, the highest
        (lowest for cooling) that serves no stream beyond its own real
        limit (see Notes of the class).
        """
        dT = bst.HeatUtility.dT
        grid = self.grid
        Ts, s = grid['Ts'], self.shift
        above, below = self._side_heat()
        if heating:
            streams = [j for j in np.flatnonzero(~self.is_hot) if above[j] > self._tol]
            ends = Ts[grid['k_hi']]
        else:
            streams = [j for j in np.flatnonzero(self.is_hot) if below[j] > self._tol]
            ends = Ts[grid['k_lo']]
        levels = []
        for agent in agents:
            if heating:
                R = agent.T - agent.dT - dT
                reached = [R - s[j] for j in streams if ends[j] > R - s[j]]
                if reached: L = min(reached)
                elif streams: L = R - min(s[j] for j in streams)
                else: L = R
            else:
                R = agent.T + agent.dT + dT
                reached = [R - s[j] for j in streams if ends[j] < R - s[j]]
                if reached: L = max(reached)
                elif streams: L = R - max(s[j] for j in streams)
                else: L = R - self.T_min_app
            levels.append((agent, float(L)))
        return levels

    def _allocate(self, levels, Q, heating):
        """
        Greedy list-order allocation of `Q` to the agents of `levels` on the
        grand composite curve: ``([(level, q, agent)], remaining)``.
        """
        Ts, arr, lea = self._gcc()
        tol = self._tol
        allocated = []
        remaining = Q
        for agent, level in levels:
            if remaining <= tol: break
            level = _snap(Ts, level)
            positions = _positions(Ts, arr, lea,
                                   [level] + [l for l, _, _ in allocated])
            done = [(l, q) for l, q, _ in allocated]
            if heating:
                B = _balance(positions, done, ())
                bound = B[:_gap(positions, level)]
            else:
                B = _balance(positions, (), done)
                bound = B[_gap(positions, level):]
            q = min(remaining, bound.min()) if bound.size else remaining
            if q > tol:
                allocated.append((level, q, agent))
                remaining -= q
        return allocated, remaining

    def _offset(self, lo, hi, heating):
        """The pricing offset of a utility serving shifted [lo, hi]: the
        largest shift of the cold streams overlapping it (heating; 0 if
        none) or the smallest of the hot ones (cooling; `T_min_app`)."""
        grid = self.grid
        Ts = grid['Ts']
        overlap = (Ts[grid['k_hi']] > lo) & (Ts[grid['k_lo']] < hi)
        overlap &= ~self.is_hot if heating else self.is_hot
        if not overlap.any(): return 0. if heating else self.T_min_app
        s = self.shift[overlap]
        return float(s.max() if heating else s.min())

    def utility_targets(self, heating_agents=None, cooling_agents=None):
        """
        Multiple-utility targets: the agents placed on the grand composite
        curve and priced by biosteam (see Notes of the class).

        Parameters
        ----------
        heating_agents : Sequence[UtilityAgent], optional
            Heating agents in order of preference; fuel agents (``isfuel``)
            take the remainder, the first one only. Defaults to
            ``bst.HeatUtility.heating_agents`` (read at call time).
        cooling_agents : Sequence[UtilityAgent], optional
            Cooling agents in order of preference. Defaults to
            ``bst.HeatUtility.cooling_agents``.

        Returns
        -------
        list[UtilityTarget]
            The heating utilities in allocation order (fuel, or an unserved
            remainder, last), then the cooling ones. Agents with no duty are
            left out.

        Warns
        -----
        RuntimeWarning
            If no agent can serve part of the hot or cold utility target.

        """
        if heating_agents is None: heating_agents = bst.HeatUtility.heating_agents
        if cooling_agents is None: cooling_agents = bst.HeatUtility.cooling_agents
        heating_agents = list(heating_agents)
        cooling_agents = list(cooling_agents)
        tol = self._tol
        heat, remaining = self._allocate(
            self._levels([a for a in heating_agents if not a.isfuel], True),
            self.Q_hot, True)
        if remaining > tol:
            fuels = [a for a in heating_agents if a.isfuel]
            heat.append((math.inf, remaining, fuels[0] if fuels else None))
            if not fuels:
                warn(f'no heating agent can serve {remaining:.6g} kJ/hr of the '
                     'hot utility target; it is left unserved', RuntimeWarning)
        cool, remaining = self._allocate(
            self._levels(cooling_agents, False), self.Q_cold, False)
        if remaining > tol:
            cool.append((-math.inf, remaining, None))
            warn(f'no cooling agent can serve {remaining:.6g} kJ/hr of the '
                 'cold utility target; it is left unserved', RuntimeWarning)
        # the stretch each agent serves: from its level to the next zero of
        # the balanced curve, the pinch or a utility pinch
        Ts, arr, lea = self._gcc()
        positions = _positions(Ts, arr, lea,
                               [l for l, _, _ in heat + cool if math.isfinite(l)])
        zero = _balance(positions, [(l, q) for l, q, _ in heat],
                        [(l, q) for l, q, _ in cool]) <= tol
        T = positions[0]
        utilities = []
        with _isolated_pricing([a for *_, a in heat + cool if a is not None]):
            for level, q, agent in heat:
                g = _gap(positions, level)
                z = np.flatnonzero(zero[g:])
                lo = float(T[g + z[0]]) if z.size else self.pinch_T
                hi = min(level, float(Ts[0]))
                o = self._offset(lo, hi, True)
                utilities.append(self._utility(agent, True, level, q,
                                               lo + o, hi + o))
            for level, q, agent in cool:
                g = _gap(positions, level)
                z = np.flatnonzero(zero[:g])
                hi = float(T[z[-1]]) if z.size else self.pinch_T
                lo = max(level, float(Ts[-1]))
                o = self._offset(lo, hi, False)
                utilities.append(self._utility(agent, False, level, q,
                                               hi + o, lo + o))
        return utilities

    @staticmethod
    def _utility(agent, heating, level, q, T_in, T_out):
        """The `UtilityTarget` of `agent` serving `q` from `T_in` to `T_out`
        (process temperatures), priced as one biosteam heat utility (inside
        `_isolated_pricing`)."""
        if agent is None:
            nan = math.nan
            return UtilityTarget(None, None, heating, level, q, nan, nan, nan,
                                 T_in, T_out, nan, nan)
        hu = bst.HeatUtility()
        hu(q if heating else -q, T_in, T_out, agent=agent)
        return UtilityTarget(agent.ID, agent, heating, level, q, abs(hu.duty),
                             abs(hu.flow), hu.cost, T_in, T_out,
                             hu.inlet_utility_stream.T,
                             hu.outlet_utility_stream.T)

    # %% Balanced curves

    def _balanced_segments(self, utilities, inv_h=None, inv_h_utilities=None):
        """
        The hot and cold segments (`_composite`) of the balanced composite
        curves on real temperatures: the streams plus the served utilities,
        fuel as a last hot segment at infinite temperature (a weight of 0,
        no area). `inv_h` weighs the streams (N) and `inv_h_utilities` the
        `utilities` (1 by default).
        """
        if inv_h is None: inv_h = np.ones(self.is_hot.size)
        if inv_h_utilities is None: inv_h_utilities = [1.] * len(utilities)
        sides = []
        for hot in (True, False):
            members = self._members(hot)
            weights = list(inv_h[self.is_hot == hot])
            fuel = 0.
            for u, weight in zip(utilities, inv_h_utilities):
                if u.heating != hot or u.agent is None: continue
                if u.agent.isfuel:
                    fuel += u.duty
                    continue
                members.append(_utility_member(u))
                weights.append(weight)
            segments = _composite(members, weights)
            if fuel:
                end = segments[1][-1] if segments[1].size else 0.
                segments = tuple(np.append(x, v) for x, v in zip(
                    segments, (end, end + fuel, math.inf, math.inf, 0.)))
            sides.append(segments)
        return sides

    def balanced_composite_curves(self, utilities=None):
        """
        The balanced composite curves (see Notes of the class).

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            Defaults to `utility_targets` with the default agents.

        Returns
        -------
        hot_T, hot_H, cold_T, cold_H : numpy.ndarray
            Real temperatures [K] and cumulative heat [kJ/hr] of each
            composite, ascending from 0; fuel is a last hot stretch at
            ``T = inf``. Both end at the same heat unless part of a target is
            unserved.

        """
        if utilities is None: utilities = self.utility_targets()
        hot, cold = self._balanced_segments(utilities)
        return (*_curve(hot), *_curve(cold))

    def _balanced_positions(self, utilities):
        """The constraint positions with the levels of every allocated
        finite level of `utilities`, and the balanced heat flow at each."""
        Ts, arr, lea = self._gcc()
        positions = _positions(Ts, arr, lea, [u.level for u in utilities
                                              if math.isfinite(u.level)])
        B = _balance(positions,
                     [(u.level, u.duty) for u in utilities if u.heating],
                     [(u.level, u.duty) for u in utilities if not u.heating])
        return positions, B

    def balanced_grand_composite_curve(self, utilities=None):
        """
        The balanced grand composite curve: the grand composite curve net of
        the utilities.

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            Defaults to `utility_targets` with the default agents.

        Returns
        -------
        T : numpy.ndarray
            Shifted temperatures [K], descending: the grid temperatures
            (twice each, arriving and leaving heat) and the utility levels
            between them (twice each, just above and just below).
        B : numpy.ndarray
            Heat cascaded net of the utilities [kJ/hr], non-negative: zero at
            the pinch and at every utility pinch, at the top the duty of the
            fuel and of levels above the grid, at the bottom that of cooling
            levels below it.

        """
        if utilities is None: utilities = self.utility_targets()
        positions, B = self._balanced_positions(utilities)
        return positions[0].copy(), B


def plot_composite_curves(targets, kind='real', utilities=None, ax=None,
                          file=None, dpi=300):
    """
    Draw the composite curves of `targets`: real, shifted, or balanced with
    the utilities.

    Parameters
    ----------
    targets : PinchTargets
    kind : str, optional
        'real' (the default) or 'shifted' for `PinchTargets.composite_curves`
        with the heat recovered shaded and the utility targets labeled
        (shifted ones with the pinch marked), or 'balanced' for
        `PinchTargets.balanced_composite_curves` with each utility labeled.
    utilities : list[UtilityTarget], optional
        Utilities of the balanced curves. Defaults to
        `PinchTargets.utility_targets` with the default agents.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if not given.
    file : str, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution used when saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Temperatures are shown in degC. Artists carry stable gids: 'Hot
    composite' and 'Cold composite' (or 'Hot balanced composite' and 'Cold
    balanced composite'), 'Heat recovered', 'Hot utility', 'Cold utility',
    'Pinch', 'Fuel' and 'Utility:<agent ID>'. Fuel, at infinite temperature
    on the balanced curves, is labeled rather than drawn.

    Examples
    --------
    The streams of `PinchTargets`:

    >>> import biosteam as bst, thermosteam as tmo
    >>> from hensmith import PinchTargets, plot_composite_curves
    >>> fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1.,
    ...                      Cn=3.6, default=True)
    >>> bst.settings.set_thermo([fluid])
    >>> bst.main_flowsheet.set_flowsheet('targeting_example')
    >>> H1 = bst.HXutility('H1', ins=bst.Stream('H1_in', Fluid=1000., T=400.),
    ...                    T=300., rigorous=False)
    >>> C1 = bst.HXutility('C1', ins=bst.Stream('C1_in', Fluid=1000., T=330.),
    ...                    T=440., rigorous=False)
    >>> H1.simulate(); C1.simulate()
    >>> targets = PinchTargets.from_heat_utilities(
    ...     [H1.heat_utilities[0], C1.heat_utilities[0]], T_min_app=10.)
    >>> fig, ax = plot_composite_curves(targets, kind='shifted')
    >>> [line.get_gid() for line in ax.get_lines()]
    ['Hot composite', 'Cold composite', 'Pinch']
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    import matplotlib.pyplot as plt
    if kind not in ('real', 'shifted', 'balanced'):
        raise ValueError("kind must be 'real', 'shifted' or 'balanced', not "
                         f'{kind!r}')
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
    else:
        fig = ax.figure
    hot_color, cold_color = '#d62728', '#2e6db4'
    text_kwargs = dict(fontsize=8)
    if kind == 'balanced':
        if utilities is None: utilities = targets.utility_targets()
        hot_T, hot_H, cold_T, cold_H = targets.balanced_composite_curves(utilities)
        finite = np.isfinite(hot_T)
        hot_T, hot_H = hot_T[finite], hot_H[finite]
        ax.plot(hot_H, hot_T - 273.15, color=hot_color, lw=2,
                label='hot balanced composite', gid='Hot balanced composite')
        ax.plot(cold_H, cold_T - 273.15, color=cold_color, lw=2,
                label='cold balanced composite', gid='Cold balanced composite')
        for u in utilities:
            if u.agent is None: continue
            label = f'{u.ID} {_format_H(u.duty)} kJ/hr'
            if u.agent.isfuel: # at the end of the finite hot curve
                x = hot_H[-1] if hot_H.size else 0.
                y = (hot_T[-1] if hot_T.size else targets.grid['Ts'][0]) - 273.15
                ax.annotate(f'{u.ID} (furnace) {_format_H(u.duty)} kJ/hr',
                            xy=(x, y), xytext=(0, 4), textcoords='offset points',
                            ha='right', va='bottom', color=hot_color, gid='Fuel',
                            **text_kwargs)
                continue
            T, H = (hot_T, hot_H) if u.heating else (cold_T, cold_H)
            if u.agent.T_limit is None: # beside the agent's flat
                T_u = u.agent.T
                if u.heating: i = np.searchsorted(T, T_u - 1e-6)
                else: i = np.searchsorted(T, T_u + 1e-6) - 1
                x = H[min(max(i, 0), H.size - 1)]
            else:
                T_u = 0.5 * (u.T_utility_in + u.T_utility_out)
                x = np.interp(T_u, T, H)
            ax.annotate(label, xy=(x, T_u - 273.15),
                        xytext=(-6 if u.heating else 6, 0),
                        textcoords='offset points',
                        ha='right' if u.heating else 'left', va='center',
                        color=hot_color if u.heating else cold_color,
                        gid='Utility:' + u.ID, **text_kwargs)
        ax.set_ylabel('T [°C]')
    else:
        shifted = kind == 'shifted'
        hot_T, hot_H, cold_T, cold_H = targets.composite_curves(shifted)
        if cold_H.size and hot_H.size and cold_H[0] < hot_H[-1]:
            ax.axvspan(cold_H[0], hot_H[-1], color='0.92', lw=0, zorder=0,
                       label='heat recovered', gid='Heat recovered')
        ax.plot(hot_H, hot_T - 273.15, color=hot_color, lw=2,
                label='hot composite', gid='Hot composite')
        ax.plot(cold_H, cold_T - 273.15, color=cold_color, lw=2,
                label='cold composite', gid='Cold composite')
        T_all = np.concatenate([hot_T, cold_T]) - 273.15
        H_hot_end = hot_H[-1] if hot_H.size else 0.
        H_cold_end = cold_H[-1] if cold_H.size else targets.Q_cold
        ax.margins(y=0.12) # headroom for the utility labels
        ax.text(0.5 * (H_hot_end + H_cold_end), T_all.max() + 1.,
                f'hot utility {_format_H(targets.Q_hot)} kJ/hr', ha='center',
                va='bottom', gid='Hot utility', **text_kwargs)
        ax.text(0.5 * targets.Q_cold, T_all.min() - 1.,
                f'cold utility {_format_H(targets.Q_cold)} kJ/hr', ha='left',
                va='top', gid='Cold utility', **text_kwargs)
        if shifted:
            # where the composites touch: the cold utility plus the cold
            # streams' heat below the pinch
            below = targets._side_heat()[1]
            x = targets.Q_cold + below[~targets.is_hot].sum()
            ax.plot([x], [targets.pinch_T - 273.15], 'o', mfc='w', mec='k',
                    ms=7, zorder=4, label='pinch', gid='Pinch')
        ax.set_ylabel('shifted T [°C]' if shifted else 'T [°C]')
    ax.set_xlabel('H [kJ·h$^{-1}$]')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    if file: fig.savefig(file, dpi=dpi, bbox_inches='tight')
    return fig, ax

def plot_grand_composite_curve(targets, utilities=None, ax=None, file=None,
                               dpi=300):
    """
    Draw the grand composite curve of `targets` with the utility targets as
    a staircase of levels.

    Parameters
    ----------
    targets : PinchTargets
    utilities : list[UtilityTarget] or False, optional
        Utilities to draw. Defaults to `PinchTargets.utility_targets` with
        the default agents; False draws the curve only.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if not given.
    file : str, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution used when saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Shifted temperatures are shown in degC. Heating utilities are drawn by
    ascending level from the temperature axis, each a horizontal segment at
    its level as long as its duty, joined by dotted verticals; cooling
    utilities likewise by descending level. Fuel is drawn dashed at the top
    of the grid and an unserved remainder in gray. Artists carry stable
    gids: 'GCC', 'Pinch', 'Utility:<agent ID>' (the segments,
    'Utility:unserved' for a remainder) and 'Label:<agent ID>'.

    Examples
    --------
    The streams of `PinchTargets`:

    >>> import biosteam as bst, thermosteam as tmo
    >>> from hensmith import PinchTargets, plot_grand_composite_curve
    >>> fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1.,
    ...                      Cn=3.6, default=True)
    >>> bst.settings.set_thermo([fluid])
    >>> bst.main_flowsheet.set_flowsheet('targeting_example')
    >>> H1 = bst.HXutility('H1', ins=bst.Stream('H1_in', Fluid=1000., T=400.),
    ...                    T=300., rigorous=False)
    >>> C1 = bst.HXutility('C1', ins=bst.Stream('C1_in', Fluid=1000., T=330.),
    ...                    T=440., rigorous=False)
    >>> H1.simulate(); C1.simulate()
    >>> targets = PinchTargets.from_heat_utilities(
    ...     [H1.heat_utilities[0], C1.heat_utilities[0]], T_min_app=10.)
    >>> fig, ax = plot_grand_composite_curve(targets)
    >>> sorted(line.get_gid() for line in ax.get_lines()
    ...        if (line.get_gid() or '').startswith('Utility:'))
    ['Utility:chilled_water', 'Utility:cooling_water', 'Utility:low_pressure_steam', 'Utility:medium_pressure_steam']
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.2))
    else:
        fig = ax.figure
    hot_color, cold_color = '#d62728', '#2e6db4'
    T, H = targets.grand_composite_curve()
    ax.plot(H, T - 273.15, color='k', lw=2, label='grand composite curve',
            gid='GCC')
    ax.axvline(0, color='k', ls='--', lw=0.8)
    if targets.Q_hot > 0.:
        x, y = 0., targets.pinch_T
    else: # threshold problem
        i = int(np.argmin(H))
        x, y = H[i], T[i]
    ax.plot([x], [y - 273.15], 'o', mfc='w', mec='k', ms=7, zorder=4,
            gid='Pinch')
    if utilities is None: utilities = targets.utility_targets()
    if utilities is not False:
        Ts = targets.grid['Ts']
        heating = sorted((u for u in utilities if u.heating),
                         key=lambda u: u.level)
        cooling = sorted((u for u in utilities if not u.heating),
                         key=lambda u: -u.level)
        for steps, color, end in ((heating, hot_color, Ts[0]),
                                  (cooling, cold_color, Ts[-1])):
            x = 0.
            previous = None
            for u in steps:
                y = (u.level if math.isfinite(u.level) else end) - 273.15
                ID = 'unserved' if u.ID is None else u.ID
                if u.agent is None: style = dict(color='0.5', ls='-')
                elif u.agent.isfuel: style = dict(color=color, ls='--')
                else: style = dict(color=color, ls='-')
                if previous is not None:
                    ax.plot([x, x], [previous, y], color=style['color'],
                            ls=':', lw=0.8)
                ax.plot([x, x + u.duty], [y, y], lw=1.5, gid='Utility:' + ID,
                        **style)
                ax.text(x + u.duty, y, ' ' + ID, ha='left', va='center',
                        fontsize=7, color=style['color'], gid='Label:' + ID)
                x += u.duty
                previous = y
    ax.set_xlabel('heat cascaded [kJ·h$^{-1}$]')
    ax.set_ylabel('shifted T [°C]')
    ax.grid(alpha=0.3)
    if file: fig.savefig(file, dpi=dpi, bbox_inches='tight')
    return fig, ax
