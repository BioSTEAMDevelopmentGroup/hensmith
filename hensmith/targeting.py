# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Targeting: what a heat exchanger network of a set of process streams can
achieve, before (or instead of) synthesizing one (the energy targets bound
every network, the others are estimates). `PinchTargets` holds the
problem table of the streams and derives from it the composite and grand
composite curves, the multiple-utility targets priced by biosteam
(`PinchTargets.utility_targets`), the balanced composite curves, and the
area, units, capital cost and total annualized cost targets
(`PinchTargets.summary`); `supertarget` sweeps the minimum approach
temperature for the least total annualized cost, and `network_report`
compares a synthesized network with its targets. `plot_composite_curves`,
`plot_grand_composite_curve` and `plot_supertargets` draw them.
"""
import math
import os
from collections import namedtuple
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral, Real
from warnings import catch_warnings, filterwarnings, warn
import numpy as np
from matplotlib.legend import Legend
from matplotlib.patches import Patch
from matplotlib.path import Path
import biosteam as bst
from ._curves import StreamCurve, _T_EQ, _copy
from ._planner import _approach_contributions, _given_contributions
from .hxn_synthesis import (_problem_table, _pinch_cut, _format_H,
    _arrange_heat_utilities, _process_streams, _explicit_contributions,
    _check_key_levels, _nearest_key_value, _key_universe, _unknown_keys,
    _unknown_key_hints, _held_owners, _parent, _top, _with_auxiliaries)

__all__ = ('PinchTargets', 'supertarget', 'Supertargets', 'network_report',
           'plot_composite_curves', 'plot_grand_composite_curve',
           'plot_supertargets')

#: Kinds of the constraint positions of the grand composite curve (see
#: `_positions`): the heat arriving at a grid temperature (before its point
#: loads), the heat leaving it (after them), and the heat just above and
#: just below a utility level strictly between two grid temperatures.
_ARRIVING, _LEAVING, _ABOVE, _BELOW = 0, 1, 2, 3

#: How far short of a grid temperature a utility level is placed to serve
#: none of that temperature's point loads [K]: beyond `_T_EQ`, so that the
#: level is not snapped back onto it.
_T_SHORT = 2 * _T_EQ

#: Square feet per square meter, biosteam's own factor in `HX._design`.
_FT2_PER_M2 = 10.763

#: Default film coefficient [kW/m²/K]. A pair gives U = 0.5, biosteam's
#: heuristic overall heat transfer coefficient (except condensing against
#: boiling).
_H_DEFAULT = 1.0

#: Btu per kJ, biosteam's factor in `HX._cost` (furnace).
_BTU_PER_KJ = 0.947817

#: Default economics: interest rate, years of annualization and operating
#: hours per year (a capital recovery factor of 0.16275).
_INTEREST, _YEARS, _OPERATING_HOURS = 0.10, 10, 8000.

#: Default minimum approach temperatures of `supertarget` [K].
_T_MIN_APPS = np.arange(1., 41.)

#: A vertical temperature difference at or below this is no driving force
#: [K].
_DT_ZERO = 1e-9

#: The options of `PinchTargets.summary` (and `supertarget`).
_SUMMARY_OPTIONS = ('heating_agents', 'cooling_agents', 'h', 'h_utilities',
                    'P', 'heat_exchanger_type', 'material', 'interest',
                    'years', 'operating_hours', 'tea')

#: The options of `PinchTargets.summary` whose agents `h_utilities` may key.
_OPTION_AGENTS = 'heating_agents or cooling_agents'

#: The frames that a warning of `_warn` skips: those of hensmith itself.
_PACKAGE_FRAMES = (os.path.dirname(os.path.abspath(__file__)) + os.sep,)

#: The difference between the flows of an exchanger's first inlet and
#: outlet, relative to the larger, beyond which its inlet has changed since
#: its unit was simulated (see `_stale_labels`): far above a legitimate one,
#: as of the total condenser of a `biosteam.MESHDistillation` (about 0.1 %).
_STALE_FLOW = 1e-2

UtilityTarget = namedtuple('UtilityTarget', (
    'ID', 'agent', 'heating', 'level', 'duty', 'utility_duty', 'flow', 'cost',
    'T_in', 'T_out', 'T_utility_in', 'T_utility_out', 'shift'),
    defaults=(None,))

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
    Process temperatures of the stretch it serves [K]: the real temperatures
    of the streams it serves there, the coldest and the hottest (heating) or
    the hottest and the coldest (cooling). biosteam prices the utility from
    them, a sensible agent from the inlet that keeps its line ``dT`` from
    the process (see Notes of `PinchTargets`).
T_utility_in, T_utility_out : float
    Inlet and outlet temperature of the agent [K], from the priced
    `HeatUtility`; nan when unserved.
shift : tuple[float] or None
    `PinchTargets.shift` of the targets that placed it [K], in stream order:
    `level` is on their shifted temperatures, and only targets with the same
    shifts take it (their balanced curves and their area, units and capital
    targets raise a ValueError for any other). None, the default, for a
    utility built by hand, which is not checked.

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
    the original dumps, and biosteam's registry of flowsheets, are restored
    on exit.
    """
    dumps = {id(a): (a, a.utility_stream_dump) for a in agents}
    flowsheets = bst.Flowsheet.flowsheet.__dict__
    ID = '_hensmith_targets'
    missing = object()
    previous = flowsheets.get(ID, missing)
    try:
        for a, _ in dumps.values(): a.utility_stream_dump = []
        with bst.Flowsheet(ID):
            yield
    finally:
        for a, dump in dumps.values(): a.utility_stream_dump = dump
        if previous is missing: flowsheets.pop(ID, None)
        else: flowsheets[ID] = previous

def _overcounted_units(hus, owners=None):
    """
    The units given more heat utilities with duty in `hus` than they hold,
    ``[(unit, given, listed), ...]`` in order of first occurrence: those
    whose heat utilities `hus` holds more often than any unit on their
    owner chain (`unit` and up along `hensmith.hxn_synthesis._parent` with
    `owners`) lists them in its `heat_utilities`. biosteam lists the heat
    utilities of a unit in parallel (``Unit.parallel``, as a
    `biosteam.AbstractStirredTankReactor` sets for its recirculation heat
    exchanger) once per unit in parallel in its owner's, each a stream of
    the synthesis; a heat utility given more often than that is given
    twice, as by selections from overlapping unit lists.
    """
    given = {}
    for hu in hus:
        if hu.duty: given.setdefault(id(hu.unit), [hu.unit, 0])[1] += 1
    over = []
    for unit, n in given.values():
        if n == 1: continue
        listed, above = 0, unit
        while above is not None:
            listed = max(listed, sum([hu.unit is unit and bool(hu.duty)
                                      for hu in above.heat_utilities]))
            above = _parent(above, owners)
        if n > listed: over.append((unit, n, listed))
    return over

def _duty_off_stream(hu):
    """Whether the duty of the heat exchanger's heat utility `hu`
    (``hu.unit_duty``) differs from that of its process stream (``outs[0].H
    - ins[0].H`` of its unit) beyond rounding, as for the heat utilities
    that a heat exchanger network simulated with
    ``replace_unit_heat_utilities`` writes its utilities over or empties
    (and, by 0.1 %, that of biosteam's `MESHDistillation` total
    condenser)."""
    unit = hu.unit
    dH = unit.outs[0].H - unit.ins[0].H
    return abs(dH - hu.unit_duty) > 1e-6 * max(abs(dH), abs(hu.unit_duty))

def _unit_label(unit, owners=None):
    """``unit.ID``, or ``'<owner.ID> - <unit.ID>'`` for an auxiliary unit,
    with its top owner as keys take it (along
    `hensmith.hxn_synthesis._parent` with `owners`, which may make a unit
    the owner of an auxiliary it holds without one), and ``owner.ID`` alone
    for an auxiliary whose ID is empty."""
    owner = _top(unit, owners)
    if owner is unit: return unit.ID
    return f'{owner.ID} - {unit.ID}' if unit.ID else owner.ID

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

def _sensible(agent):
    """Whether biosteam prices `agent` by its sensible heat: a non-fuel
    agent with a temperature limit."""
    return agent is not None and not agent.isfuel and bool(agent.T_limit)

def _euler(regions):
    """Euler's S - 1 per connected component of the members of `regions`
    (sets; members of one region are connected)."""
    components = []
    for members in regions:
        if not members: continue
        component = set(members)
        for other in [c for c in components if c & component]:
            component |= other
            components.remove(other)
        components.append(component)
    return sum(len(c) - 1 for c in components)

def _film_value(x, label):
    """A film coefficient [kW/m²/K]: ``float(x)`` for a finite real number
    > 0 (not a bool); a ValueError naming `label` otherwise."""
    if isinstance(x, Real) and not isinstance(x, (bool, np.bool_)):
        value = float(x)
        if math.isfinite(value) and value > 0.: return value
    raise ValueError(f'{label} must be a finite number > 0 kW/m2/K, not {x!r}')

def _utility_films(utilities, h_utilities, agents=(), where='agents'):
    """
    The film coefficient [kW/m²/K] of each of `utilities` from
    `h_utilities`: None (`_H_DEFAULT`), a number, or a mapping from agent
    IDs or agents to numbers (`_H_DEFAULT` for an agent it leaves out). An
    ID key must name an agent of `utilities`, of `agents` (the option
    `where` names in the error) or of biosteam's default lists.
    """
    return _agent_films([u.agent for u in utilities], h_utilities, agents,
                        where)

def _agent_films(of, h_utilities, agents=(), where='agents'):
    """`_utility_films` of the agents `of` (None for an unserved utility)."""
    if h_utilities is None: return [_H_DEFAULT] * len(of)
    if not isinstance(h_utilities, Mapping):
        return [_film_value(h_utilities, 'h_utilities')] * len(of)
    known = {a.ID for a in (*agents, *bst.HeatUtility.heating_agents,
                            *bst.HeatUtility.cooling_agents)}
    known.update(a.ID for a in of if a is not None)
    values = {} # agent ID -> (key, value)
    unknown = []
    for key, value in h_utilities.items():
        if isinstance(key, bst.UtilityAgent): ID = key.ID
        elif isinstance(key, str): ID = key
        else:
            raise TypeError('h_utilities keys must be agent IDs (str) or '
                            f'UtilityAgent objects, not {type(key).__name__!r} '
                            f'({key!r})')
        value = _film_value(value, f'h_utilities of key {key!r}')
        if ID not in known and not isinstance(key, bst.UtilityAgent):
            unknown.append(key)
        other, other_value = values.setdefault(ID, (key, value))
        if other_value != value:
            raise ValueError(f'h_utilities keys {other!r} and {key!r} give agent '
                             f'{ID!r} different values: {other_value!r} and '
                             f'{value!r}')
    if unknown:
        raise ValueError('h_utilities keys that name no agent of the '
                         f"utilities, of {where}, or of biosteam's lists: "
                         f"{', '.join(map(repr, unknown))} (key another agent "
                         'by the UtilityAgent itself)')
    return [values[a.ID][1] if a is not None and a.ID in values
            else _H_DEFAULT for a in of]

def _T_at(segments, i, H):
    """Temperature of the segments `i` (see `_composite`) at heat `H`."""
    Ha, Hb, Ta, Tb = (x[i] for x in segments[:4])
    with np.errstate(invalid='ignore'): # inf - inf on a furnace's flat
        T = Ta + (Tb - Ta) * (H - Ha) / (Hb - Ha)
    return np.where(Ta == Tb, Ta, T)

def _bath(hot, cold):
    """
    Area [m²] of vertical heat transfer between the `hot` and `cold`
    segments (`_composite`, weighted by 1/h [m²·K/kW]): Bath's formula,
    exact on the stretches between the breakpoints of both curves, where
    both and every member are linear. A hot stretch at infinite temperature
    (fuel) takes no area. inf with a RuntimeWarning at a driving force of
    `_DT_ZERO` or less (a zero or negative one).
    """
    if not (hot[0].size and cold[0].size): return 0.
    end = min(hot[1][-1], cold[1][-1])
    X = np.unique(np.concatenate([hot[0], hot[1], cold[0], cold[1]]))
    X = X[(X >= 0.) & (X <= end)]
    x, y = X[:-1], X[1:]
    keep = y - x > 1e-12 * end
    x, y = x[keep], y[keep]
    m = 0.5 * (x + y)
    i = np.searchsorted(hot[0], m, 'right') - 1
    j = np.searchsorted(cold[0], m, 'right') - 1
    Th_x, Th_y = _T_at(hot, i, x), _T_at(hot, i, y)
    finite = np.isfinite(Th_x) # not fuel
    x, y, i, j = x[finite], y[finite], i[finite], j[finite]
    d1 = Th_x[finite] - _T_at(cold, j, x)
    d2 = Th_y[finite] - _T_at(cold, j, y)
    zero = np.flatnonzero(np.minimum(d1, d2) <= _DT_ZERO)
    if zero.size:
        k = zero[0]
        _warn('vertical heat transfer at a non-positive driving force between '
              f'H = {x[k]:.6g} and {y[k]:.6g} kJ/hr')
        return math.inf
    LMTD = _lmtd(d1, d2)
    return float(np.sum((y - x) * (hot[4][i] + cold[4][j]) / 3600. / LMTD))

def _lmtd(d1, d2):
    """
    Log-mean temperature differences [K] of the positive driving forces `d1`
    and `d2` (arrays), to machine precision when they are nearly equal: the
    log of their ratio is ``log1p((d1 - d2) / d2)`` for a ratio between 0.5
    and 1.5 (where the difference is exact), and the LMTD is `d1` at
    equality.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (d1 - d2) / d2
        log = np.where(np.abs(r) < .5, np.log1p(r), np.log(d1 / d2))
        return np.where(r == 0., d1, (d1 - d2) / log)

def _check_exchanger(heat_exchanger_type, material):
    """Raise a ValueError unless biosteam knows the shell and tube
    `heat_exchanger_type` and `material`."""
    from biosteam.units.heat_exchange import Cb_dict
    from biosteam.units.design_tools.specification_factors import (
        shell_and_tube_material_factor_coefficients as materials)
    types = [i for i in Cb_dict if i != 'Furnace']
    if heat_exchanger_type not in types:
        raise ValueError(f'heat_exchanger_type must be one of {types}, not '
                         f'{heat_exchanger_type!r}')
    if material not in materials:
        raise ValueError(f'material must be one of {list(materials)}, not '
                         f'{material!r}')

def _exchanger_costs(A, P=101325., heat_exchanger_type='Floating head',
                     material='Carbon steel/carbon steel'):
    """
    ``(purchase, installed, type)`` [USD] of one heat exchanger of area `A`
    [ft²] at pressure `P` [Pa], as biosteam's `HX._cost` and
    `Unit._load_costs` cost it: a double pipe below 150 ft², else a shell
    and tube of `heat_exchanger_type` and `material` with 20 ft tubes.
    ``(inf, inf, None)`` for an infinite area (or one beyond the
    correlations), ``(nan, nan, None)`` for nan.
    """
    from biosteam.units.heat_exchange import (
        Cb_dict, p2, compute_double_pipe_purchase_cost)
    from biosteam.units.design_tools.specification_factors import (
        compute_shell_and_tube_material_factor,
        shell_and_tube_material_factor_coefficients)
    _check_exchanger(heat_exchanger_type, material)
    P = _pressure(P)
    if math.isnan(A): return math.nan, math.nan, None
    if math.isinf(A): return math.inf, math.inf, None
    P_psi = P * 14.7 / 101325
    L = 20 # ft, as biosteam assumes
    try:
        if A < 150: # double pipe
            p = P_psi / 600
            F_p = 0.8510 + 0.1292 * p + 0.0198 * p**2
            F_m = 2
            if A < 2.1718:
                F_l = A / 2.1718
                A = 2.1718
            else:
                F_l = 1
            kind = 'Double pipe'
            C_b = compute_double_pipe_purchase_cost(A, bst.CE)
        else: # shell and tube
            F_m = compute_shell_and_tube_material_factor(
                A, *shell_and_tube_material_factor_coefficients[material])
            F_l = np.polyval(p2, L) # L == 20 is not > 20 in biosteam
            p = P_psi / 100
            F_p = 0.9803 + 0.018 * p + 0.0017 * p**2
            kind = heat_exchanger_type
            C_b = Cb_dict[kind](A, bst.CE)
    except OverflowError:
        return math.inf, math.inf, None
    F = F_l * F_p * F_m
    return C_b * F, C_b * (bst.HX._F_BM_default[kind] + F - 1.), kind

def _furnace_costs(Q):
    """``(purchase, installed)`` [USD] of a furnace of utility duty `Q`
    [kJ/hr], as biosteam's `HX._cost` costs it (500 psi)."""
    from biosteam.units.heat_exchange import compute_furnace_purchase_cost
    C_b = compute_furnace_purchase_cost(Q * _BTU_PER_KJ, bst.CE)
    S = 500 / 500
    F_P = 0.986 - 0.0035 * S + 0.0175 * S * S
    return C_b * F_P, C_b * (bst.HX._F_BM_default['Furnace'] + F_P - 1.)

def _shell_split(A, M, P=101325., heat_exchanger_type='Floating head',
                 material='Carbon steel/carbon steel'):
    """
    ``(purchase, installed, k, type)`` [USD] of the area `A` [m²] in `M` >= 1
    shells, k of them sharing it equally, each of biosteam's `type`
    (`_exchanger_costs`): Linnhoff and Ahmad's k = `M`, unless that puts the
    shells below 150 ft² where the area fills k = ``A // 150 ft²`` >= 1
    shells of at least 150 ft² and these, with the ``M - k`` others of no
    area (each costed as biosteam costs a vanishing double pipe), cost more
    to install. k = `M` for an infinite or nan `A`.
    """
    def shells(k): # k shells of A / k
        p, i, kind = _exchanger_costs(A / k * _FT2_PER_M2, P,
                                      heat_exchanger_type, material)
        return k * p, k * i, k, kind
    equal = shells(M)
    if not math.isfinite(A * _FT2_PER_M2): return equal
    k = min(int(A * _FT2_PER_M2 // 150.), M)
    while k and A / k * _FT2_PER_M2 < 150.: k -= 1 # rounding
    if not k or k == M: return equal
    p, i, k, kind = shells(k)
    p0, i0, _ = _exchanger_costs(0., P, heat_exchanger_type, material)
    p += (M - k) * p0
    i += (M - k) * i0
    return (p, i, k, kind) if i > equal[1] else equal

def _real(x, name):
    """``float(x)`` of a finite real number (not a bool); a ValueError
    naming `name` otherwise."""
    if isinstance(x, Real) and not isinstance(x, (bool, np.bool_)):
        value = float(x)
        if math.isfinite(value): return value
    raise ValueError(f'{name} must be a finite number, not {x!r}')

def _pressure(P):
    """``float(P)`` of the exchangers' pressure `P` [Pa], a finite number
    > 0 (not a bool); a ValueError naming P otherwise."""
    P = _real(P, 'P')
    if P <= 0.: raise ValueError(f'P must be > 0 Pa, not {P!r}')
    return P

def _check_T_min_app(T_min_app):
    """A ValueError naming `T_min_app` unless it is a finite number >= 0
    [K] (not a bool), the rule of `supertarget`; > 0 with approach
    contributions is checked with them (`_given_contributions`)."""
    if _real(T_min_app, 'T_min_app') < 0.:
        raise ValueError(f'T_min_app must be >= 0 K, not {T_min_app!r}')

def _agents(agents, heating):
    """`agents` as a list (None stays None: biosteam's lists, read when
    used), each ID resolved among ``bst.HeatUtility.heating_agents`` (or
    ``cooling_agents``) as they are now; a ValueError for an unknown ID or
    an agent of the other kind (a heating agent is ``isheating_agent`` or
    ``isfuel``, a cooling one ``iscooling_agent`` and not ``isfuel``)."""
    if agents is None: return None
    kind, other = ('heating', 'cooling') if heating else ('cooling', 'heating')
    if isinstance(agents, (str, bst.UtilityAgent)):
        raise TypeError(f'{kind}_agents must be a sequence of agents (or '
                        f'their IDs), not {type(agents).__name__!r}')
    known = bst.HeatUtility.heating_agents if heating else bst.HeatUtility.cooling_agents
    resolved = []
    for agent in agents:
        if isinstance(agent, str):
            match = [a for a in known if a.ID == agent]
            if not match:
                raise ValueError(
                    f'no {kind} agent {agent!r} in bst.HeatUtility.{kind}_'
                    f"agents ({', '.join(a.ID for a in known)}); pass the "
                    'UtilityAgent itself for another one')
            agent = match[0]
        if heating: right = agent.isfuel or agent.isheating_agent
        else: right = agent.iscooling_agent and not agent.isfuel
        if not right:
            raise ValueError(f'{agent.ID!r} is not a {kind} agent; pass it in '
                             f'{other}_agents')
        resolved.append(agent)
    return resolved

def _key_agents(agents):
    """The `agents` of `PinchTargets.area_target` as a list (None is
    none), each ID resolved among biosteam's heating and cooling agents as
    they are now; a ValueError for an unknown ID."""
    if agents is None: return []
    if isinstance(agents, (str, bst.UtilityAgent)):
        raise TypeError('agents must be a sequence of agents (or their IDs), '
                        f'not {type(agents).__name__!r}')
    known = [*bst.HeatUtility.heating_agents, *bst.HeatUtility.cooling_agents]
    resolved = []
    for agent in agents:
        if isinstance(agent, str):
            match = [a for a in known if a.ID == agent]
            if not match:
                raise ValueError(
                    f'no agent {agent!r} in bst.HeatUtility.heating_agents or '
                    'cooling_agents; pass the UtilityAgent itself for another '
                    'one')
            agent = match[0]
        elif not isinstance(agent, bst.UtilityAgent):
            raise TypeError('agents must be UtilityAgent objects or their IDs '
                            f'(str), not {type(agent).__name__!r} ({agent!r})')
        resolved.append(agent)
    return resolved

def _economics(interest=None, years=None, operating_hours=None, tea=None):
    """
    ``(crf, hours)``: the capital recovery factor [1/yr] and the operating
    hours per year. Each keyword takes precedence over `tea` (its `IRR`,
    ``duration[1] - duration[0]`` and `operating_hours`, unless None), which
    takes precedence over the defaults.
    """
    if tea is not None:
        if interest is None: interest = tea.IRR
        if years is None: years = tea.duration[1] - tea.duration[0]
        if operating_hours is None: operating_hours = tea.operating_hours
    interest = _real(_INTEREST if interest is None else interest, 'interest')
    years = _real(_YEARS if years is None else years, 'years')
    hours = _real(_OPERATING_HOURS if operating_hours is None
                  else operating_hours, 'operating_hours')
    if interest <= -1.:
        raise ValueError(f'interest must be > -1, not {interest!r}')
    if years <= 0.:
        raise ValueError(f'years must be > 0, not {years!r}')
    if hours < 0.:
        raise ValueError(f'operating_hours must be >= 0, not {hours!r}')
    if interest == 0.: return 1. / years, hours
    fn = (1. + interest) ** years
    return interest * fn / (fn - 1.), hours


class PinchTargets:
    """
    Pinch analysis targets of a set of process streams: the problem table
    (`hensmith.hxn_synthesis.problem_table`) and what follows from it, the
    composite and grand composite curves, the targets of several utilities
    (`utility_targets`) priced by biosteam, the balanced composite curves,
    and the targets of area, units, capital cost and total annualized cost
    (`summary`). Build it from heat utilities with `from_heat_utilities` (as
    `hensmith.hxn_synthesis.synthesize_network` prepares them), from a
    facility with `HeatExchangerNetwork.targets`, or from streams as
    `problem_table` takes them.

    Parameters
    ----------
    streams_inlet : list[Stream]
        Inlet of each process stream.
    streams_quenched : list[Stream]
        Corresponding outlets, re-flashed at their enthalpy.
    is_hot : Sequence[bool]
        True where the stream is cooled, False where it is heated (a
        ValueError is raised for a stream with duty marked otherwise).
    T_min_app : float, optional
        Minimum approach temperature [K], a finite number >= 0 (> 0 with
        approach contributions), or a ValueError is raised. Defaults to 5.
    approach_contributions : Sequence[float or None], optional
        Approach temperature contribution of each stream [K], in stream
        order, as `problem_table` takes them: None or nan (unspecified:
        ``T_min_app / 2``), or a finite number > 0 (`T_min_app` must then be
        finite and > 0 too). All unspecified is the same as None.
    labels : list[str], optional
        Display name of each stream, which the curves built here also take
        (to name their streams in their warnings). Defaults to the IDs of
        `streams_inlet`.
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
        The table's grid: 'Ts', its shifted temperatures [K] (descending,
        n); 'shift', as `shift`; 'contributions', each stream's approach
        temperature contribution [K] (N); 'Hl' and 'Hr', (N, n) each
        stream's enthalpy [kJ/hr] at every grid temperature, the left (low
        enthalpy) and right (high enthalpy) limit, which differ only at a
        point load of the stream; 'k_hi' and 'k_lo', (N,) the grid indices
        of each stream's own upper and lower temperature.
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
        at the pinch temperature belong. It is 'below' when the zero heat
        flow of the cascade is the flow arriving at `pinch_T` (before those
        point loads), 'above' when it is the flow leaving it (after them),
        and always 'below' for a threshold problem with no hot utility.
    heat_utilities : list[HeatUtility] or None
        The heat utilities the streams were taken from, in stream order;
        None when built from streams. For `HeatExchangerNetwork.targets`
        after a simulation they are the facility's `original_heat_utils`,
        which with `replace_unit_heat_utilities` hold the network's
        utilities (the streams are read from the units).
    units : list[Unit] or None
        The unit of each stream (``hu.unit``), in stream order; None when
        built from streams.
    key_units : list[Unit] or None
        Further units that keys of `h` may name (with their owners) without
        a stream of theirs, a key that names only such units then having no
        effect: for `HeatExchangerNetwork.targets`, every unit available to
        the network, its nested auxiliary units and the exchangers they hold
        through their heat utilities only, which approach contribution keys
        may name; for `from_heat_utilities`, its `key_units` and theirs;
        else None (settable).
        An auxiliary that one of them, or a unit of `units`, holds without
        an owner when the targets are made (for `HeatExchangerNetwork.targets`,
        at the network's synthesis) stays that unit's for keys and labels,
        as the facility takes it (see Notes of `HeatExchangerNetwork`), even
        once the unit has rebuilt its auxiliaries.
    force_ideal_thermo : bool
        Whether the stream copies use ideal thermodynamics (False when built
        from streams).
    system : System or None
        The system of the facility of `HeatExchangerNetwork.targets`, else
        None (settable). If it has operating hours, `summary` (and so
        `supertarget`) uses them unless `operating_hours` or `tea` is given.

    Notes
    -----
    *Composite curves.* `composite_curves` adds up the streams on the
    table's own grid, whose knots hold every breakpoint of every stream, so
    the composites reproduce the table to rounding: the cold composite
    starts at the cold utility target and ends the hot utility target
    beyond the end of the hot one. On the shifted scale the composites touch at
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
    ``R - s`` (down to it, for cooling); the agent's level is the highest
    single level, up to the highest of these limits, that serves no stream
    above its own limit (the lowest, down to the lowest limit, for
    cooling): ``R`` for heating and ``R - T_min_app`` for cooling without
    approach contributions, which is biosteam's rule. A level serves only
    what lies below it (above it, for cooling) and above the highest zero
    of the curve balanced with the agents already placed (below the lowest
    one): the pinch, a utility pinch or another process pinch, which its
    heat never passes. So a stream reaching beyond its limit bounds the
    level at that limit, one lying wholly beyond it only at its own end
    (just short of it, if it has a point load there), and one with no heat
    on the level's side of that zero not at all. With contributions the
    level can be conservative for streams with slack. Non-fuel agents are
    placed in list order (biosteam's first feasible agent: for the default
    lists the lowest steam and the warmest coolant first), each as much as
    the curve above its level (below it, for cooling) allows given the
    agents already placed; what remains goes to the first fuel agent, or is
    left unserved with a RuntimeWarning (nan cost). The order is biosteam's
    preference, not price: the utility cost target is the cost of the
    agents placed this way, not the least utility cost, and an agent later
    in the list can be cheaper per kJ (with biosteam's default prices,
    natural gas is cheaper than any steam for a furnace heating from below
    about 565 K), so a network, which serves each stream with one agent at
    its outlet, can cost less than the target as well as more (see Notes of
    `network_report`). A level equal to a grid
    temperature serves that temperature's point loads (biosteam's own rule
    is strict; they differ only at exact equality), except where a sensible
    agent's line would have no driving force left: at a point load at
    ``agent.T - dT`` (``agent.T + dT`` for cooling), a stream's limit when
    ``agent.dT`` is 0 (`UtilityAgent`'s default), the agent would leave at
    its inlet temperature at an unbounded flow, so its level stays just
    short of that point load, which the next agent serves, as in biosteam.
    For the same reason a stream that ends at that temperature bounds the
    level of such an agent at that end, as one reaching beyond its limit
    does: were another stream (of another approach contribution) to set the
    level beyond that end, the agent would have given heat above it (taken
    it below, for cooling) and its line would reach its inlet temperature
    there; at the level it has given none.

    Each agent is priced as one biosteam heat utility over the stretch it
    serves, from its level (or the end of the grid, if that is nearer) to
    the next zero of the balanced curve (the pinch, or a utility pinch). Its
    `T_in` and `T_out` are the real temperatures of the streams with heat
    in that stretch (the cold streams for heating, the hot ones for
    cooling): the coldest and the hottest for heating, the hottest and the
    coldest for cooling. A point load at an end of the stretch counts where
    the stretch holds it, as the balanced curve does: a boiler at the pinch
    with the cut 'above', or a condenser at it with the cut 'below', is
    served at its saturation temperature. They are the stretch's ends
    without approach contributions (where a stream reaches them), and
    `T_out` is within the agent's limit. biosteam prices
    a latent agent and fuel from them (the flue leaves at least ``dT``
    above `T_in`), but checks a sensible agent's outlet only against
    `T_in`, while its line must keep ``dT`` from the process over the whole
    stretch (cooling water serving a condenser at the cold end of its
    stretch must leave ``dT`` below the condensing temperature). At each
    position p of the stretch, the agent takes at least ``x = q - B(p)`` of
    its duty q below p (gives it above p, for heating), B being the
    balanced heat flow at p. Its outlet is therefore at most
    ``agent.T + min_p (T(p) - dT - agent.T) * q / x`` for cooling and at
    least ``agent.T - min_p (agent.T - dT - T(p)) * q / x`` for heating,
    T(p) being the lowest real temperature of the served hot streams at p
    (the highest of the cold ones): its line stays inside the grand
    composite curve, as a non-isothermal utility's does in pinch analysis.
    biosteam is given the inlet that gives this outlet, and divides the
    duty by the agent's heat transfer efficiency. Pricing uses fresh
    utility streams in a temporary flowsheet: costs are deterministic, and
    the flowsheet's registry and the agents' stream dumps are untouched.

    *Balanced composite curves.* `balanced_composite_curves` adds the
    utilities to the real composites (heating ones to the hot composite,
    cooling ones to the cold one), so both end at the same heat: a latent
    agent as a flat at its temperature, a sensible one as a line between its
    inlet and outlet temperatures, and fuel as a furnace at infinite
    temperature. An unserved remainder is left out: unserved cooling, the
    coldest member of the cold composite, as a gap at its start (which
    keeps the curves aligned), unserved heating as one at the end of the
    hot composite. `balanced_grand_composite_curve` is the cascade net of the
    utilities, zero at the process pinch and at every utility pinch.

    *Area target.* `area_target` is Bath's formula [Townsend84]_ on the
    balanced composite curves: vertical heat transfer, counter-current (a
    correction factor of 1), each stretch between breakpoints of either
    curve at its log-mean temperature difference, ``A = sum_i (1 /
    LMTD_i) sum_j q_ij / h_j``, over the real temperatures (with approach
    contributions too) and with a film coefficient h per stream and per
    utility (1 kW/m²/K by default, so a pair has biosteam's heuristic
    overall coefficient of 0.5 kW/m²/K; condensing against boiling, where
    biosteam takes 1, is not reproduced). The sum is exact on the
    piecewise-linear curves, a stream's share of a stretch being linear in
    the heat. Fuel takes no area, and the area is inf (with a
    RuntimeWarning) at a zero or negative driving force, as at
    ``T_min_app = 0``.

    *Units target.* `units_target` applies Euler's N = S - 1 to each region
    of the balanced grand composite curve between its zeros (the process
    pinch and the utility pinches), S counting the streams with heat in the
    region and the utilities acting in it. Where the curve stays zero, each
    piece between grid temperatures is balanced on its own: on each side of
    the pinch such pieces count S - 1 per group of them linked by common
    streams, so a stream across several counts once. It is an estimate, not
    a bound: it assumes one component and no loop per region, so a balanced
    subset of streams needs fewer units and each loop of matches one more.

    *Capital and total annualized cost.* `capital_cost_target` takes as
    many shells as the units target (less one furnace for all fuel) and
    costs each with biosteam's correlations at one pressure (biosteam takes
    each exchanger's highest inlet pressure: its pressure factor is up to
    about 12 % higher for high pressure steam heaters): a double pipe below
    150 ft² (for carbon steel at 1 atm and biosteam's default CE of 567.5,
    the purchase cost jumps from about 5,486 USD just below 150 ft² to
    20,138 USD for a floating head at 150 ft², and the installed cost, the
    one used here, from about 8,055 to 64,658 USD; the floating head
    correlation, not monotone below about 126 ft², is not used there), a
    shell and tube of any area above. Linnhoff and Ahmad divide the area
    target equally over the shells [Linnhoff90]_: with a cost law of
    economies of scale, such as a + bA^c, that is the costliest way to
    divide it, and each added shell adds cost. biosteam's jump breaks this:
    the equal division can bring every shell below 150 ft² although the area
    would fill some shell and tubes, so that a utility of a few watts that
    adds units would cut the target by most of it, where a network just adds
    a small exchanger and keeps its large ones. So when the equal division
    puts the N shells below 150 ft² and the area fills k = A // 150 ft² >= 1
    shells of at least 150 ft², the target takes the costlier to install of
    the N double pipes and of k equal shell and tubes with N - k shells of
    no area (each costed as biosteam costs a vanishing double pipe, about
    1,305 USD installed). This is the least change to the equal division
    that never falls when a unit is added (or some area, except where
    biosteam's correlations do: for shells whose cost rises faster than
    their area, beyond about 35,000 ft² (3,300 m²) for the default floating
    head of carbon steel and from about 15,000 ft² by type and material,
    and for the 'Double pipe' `heat_exchanger_type`, cheaper at 150 ft² than
    below). Where it departs from the equal division it tends to be high, as
    networks keep only part of their area in shell and tubes (the equal
    division tends to be low there): an estimate, not a bound. The furnace
    is costed on the fuel's utility duty at biosteam's default furnace
    pressure of 500 psi (`P` sets the pressure of the shells only).
    `summary` adds the total annualized cost, the installed (bare module)
    cost times the capital recovery factor plus the utility cost times the
    operating hours: no Lang factor and no working capital.

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

    The area target [m²] with film coefficients of 1 kW/m²/K, the units
    target (the balanced grand composite curve is zero at the utility
    pinches of low pressure steam and cooling water and from the pinch down
    to 330 K, which leaves five regions of two streams or utilities each),
    and the capital cost target [USD]. The 24.14 m² (260 ft²) fill one
    floating head, which with four shells of no area costs more to install
    than five equal double pipes of 52 ft² (about 34,000 USD), so the target
    takes it:

    >>> round(targets.area_target(utilities), 2)
    24.14
    >>> targets.units_target(utilities)
    {'above': 2, 'below': 3, 'total': 5}
    >>> capital = targets.capital_cost_target(utilities)
    >>> capital['N_shells'], capital['N_area_shells'], capital['type']
    (5, 1, 'Floating head')
    >>> round(capital['installed_cost'])
    72819

    All targets, with the total annualized cost [USD/yr] at 10 % interest
    over 10 years and 8000 operating hours a year:

    >>> summary = targets.summary()
    >>> summary['shell_type']
    'Floating head'
    >>> {key: round(value, 2) for key, value in summary.items()
    ...  if key != 'shell_type'}
    {'T_min_app': 10.0, 'Q_hot': 180000.0, 'Q_cold': 144000.0,
     'utility_cost': 1.66, 'area': 24.14, 'N_units': 5, 'N_shells': 5,
     'N_area_shells': 1, 'shell_area': 24.14, 'purchase_cost': 21054.85,
     'installed_cost': 72819.1, 'annualized_capital_cost': 11850.97,
     'operating_cost': 13247.77, 'TAC': 25098.74}

    The same streams at another minimum approach temperature (the curves
    are reused):

    >>> targets.at(20.)
    <PinchTargets: 2 streams, T_min_app = 20 K, Q_hot = 2.160e+05 kJ/hr, Q_cold = 1.800e+05 kJ/hr>

    References
    ----------
    .. [Townsend84] Townsend, D. W., & Linnhoff, B. (1984). Surface area
        targets for heat exchanger networks. IChemE 11th Annual Research
        Meeting, Bath, UK.
    .. [Linnhoff90] Linnhoff, B., & Ahmad, S. (1990). Cost optimum heat
        exchanger networks - 1. Minimum energy and capital using simple
        models for capital cost. Computers & Chemical Engineering, 14(7),
        729-750.

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
        _check_T_min_app(T_min_app)
        x = approach_contributions
        if x is not None:
            if (not isinstance(x, (Sequence, np.ndarray))
                    or isinstance(x, (str, bytes))):
                raise TypeError('approach_contributions must be a sequence '
                                'with one value per stream, not '
                                f'{type(x).__name__!r}')
            # the length, each value, then T_min_app if a value is given;
            # every value unspecified is the default path (None)
            x = _given_contributions(N, T_min_app, x)
            if x is not None: x = tuple(x)
        if labels is None:
            labels = [s.ID for s in streams_inlet]
        elif len(labels) != N:
            raise ValueError(f'{len(labels)} labels given for {N} streams')
        if curves is None: # named by the labels in their warnings
            curves = [StreamCurve(si, so, hot, label=label) for si, so, hot, label
                      in zip(streams_inlet, streams_quenched, is_hot, labels)]
        if x is None:
            table, curves, grid = _problem_table(
                streams_inlet, streams_quenched, is_hot, T_min_app, curves)
        else:
            table, curves, grid = _problem_table(
                streams_inlet, streams_quenched, is_hot, T_min_app, curves,
                approach_contributions=list(x))
        dH = np.array([c.H_out - c.H_in for c in curves])
        tol = 1e-9 * float(np.abs(dH).sum())
        for j in np.flatnonzero(np.where(is_hot, dH > tol, dH < -tol)):
            error = ValueError(f'stream {labels[j]!r} is '
                               f"{'heated' if dH[j] > 0. else 'cooled'} "
                               f'({dH[j]:+.6g} kJ/hr) but marked '
                               f"{'hot' if is_hot[j] else 'cold'}: is_hot "
                               'must be True for a cooled stream and False '
                               'for a heated one')
            # which stream, for `from_heat_utilities` to say it in terms of
            # its heat utility, which gives its side
            error._hensmith_reversed = (int(j), float(dH[j]))
            raise error
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
        self.key_units = None
        self.force_ideal_thermo = False
        self.system = None
        #: The holders of auxiliaries held without an owner when the
        #: targets were made (see `hensmith.hxn_synthesis._held_owners`).
        self._holders = None
        #: Grid index of the pinch.
        self._k_pinch = int(np.flatnonzero(table.Ts == table.pinch_T)[0])
        #: Threshold of every heat comparison [kJ/hr], the table's own.
        self._tol = tol

    @classmethod
    def from_heat_utilities(cls, hus, T_min_app=5., approach_contributions=None,
                            force_ideal_thermo=False, sort_hus_by_T=False,
                            key_units=None):
        """
        Targets of the process streams behind heat utilities, prepared
        exactly as `hensmith.hxn_synthesis.synthesize_network` prepares them
        (so the targets equal the synthesis's, bit for bit).

        Parameters
        ----------
        hus : Iterable[HeatUtility]
            Heat utilities of heat exchangers (``hu.unit``, whose first
            inlet and outlet are the process stream), such as
            ``bst.process_tools.heat_exchanger_utilities_from_units(sys.units)``
            (those a heat exchanger network may take, ``hu.hxn_ok``, as
            `HeatExchangerNetwork` selects them), not ``sys.heat_utilities``
            (sums by agent, of no unit). Zero-duty ones are dropped. Each
            counts once per occurrence, a stream each, as in the synthesis:
            biosteam lists the heat utility of a unit in parallel
            (``Unit.parallel``, as the recirculation heat exchanger of a
            `biosteam.AbstractStirredTankReactor`, such as a fermentor,
            with ``parallel['self']`` reactors) once per unit in its
            owner's ``heat_utilities``, and so in this selection. After a
            simulation of a `HeatExchangerNetwork` with
            ``replace_unit_heat_utilities=True`` they hold the network's
            utilities, not the process duties, and those of the streams it
            serves completely are empty (left out of biosteam's selection):
            take `HeatExchangerNetwork.targets` then, or build the targets
            before that simulation.
        T_min_app : float, optional
            Minimum approach temperature [K], finite and >= 0 (> 0 with
            approach contributions). Defaults to 5.
        approach_contributions : Mapping, Sequence or callable, optional
            Approach temperature contributions [K] in any form
            `synthesize_network` takes (a sequence is aligned with `hus` as
            given, zero-duty heat utilities included). Defaults to None.
        force_ideal_thermo : bool, optional
            Copy the streams with ideal thermodynamics. Defaults to False.
        sort_hus_by_T : bool, optional
            Order the streams by inlet temperature, as the synthesis's
            option. Defaults to False.
        key_units : Iterable[Unit], optional
            Further units at hand, such as ``sys.units``: the keys of
            `approach_contributions` and of `h` (see `area_target`) may name
            them, their nested auxiliary units and the exchangers they hold
            through their heat utilities only (the targets' `key_units`),
            whether or not they hold a heat utility of `hus`, as the keys of
            a `HeatExchangerNetwork` with these units may. A
            `biosteam.MultistageCompressor` K1 among them owns the stages it
            holds without an owner, so ``'K1'`` covers its intercoolers and
            ``('K1', 'K1_H1')`` names one (as a unit owns an exchanger it
            holds through its heat utilities only). Without them, only the
            units of `hus` and their owners are at hand, and ``'K1_H1'`` is
            that intercooler's key. Defaults to None.

        Returns
        -------
        PinchTargets
            Streams in stream order (heating utilities, i.e. cold streams,
            first), labeled by unit (``'<owner> - <unit>'`` for an
            auxiliary unit, with its top owner as keys take it).

        Raises
        ------
        ValueError
            If a heat utility belongs to no unit, or one with duty is not a
            heat exchanger's (``hu.hxn_ok`` False, as for the heat utilities
            that reactors, mixers or compressors add with
            ``Unit.add_heat_utility``: their unit's first inlet and outlet
            need not be the stream they heat or cool). If the heat utilities
            with duty of a unit are given more often than its
            ``heat_utilities`` or those of any of its owners list them, as
            when selections from overlapping unit lists (such as a unit and
            an auxiliary unit of it) give the same heat utility twice: its
            stream would count once more per extra one. If the stream of a
            heat utility, from its unit's inlet now to its outlet as last
            simulated, runs against the heat utility (cooled for a heating
            one, or heated for a cooling one): its inlet has changed since
            its unit was simulated (as the second warning below, given
            before this error, may say too).

        Warns
        -----
        RuntimeWarning
            If a heat utility is one that a `HeatExchangerNetwork`
            simulated with ``replace_unit_heat_utilities=True`` has written
            its utilities over (and not given back yet, as its next
            simulation does), with a duty (``hu.unit_duty``) other than its
            process stream's (``outs[0].H - ins[0].H`` of its unit, which
            the targets use) beyond rounding: the warning names that
            network, whose `HeatExchangerNetwork.targets` are the process's.
            A duty that differs from its stream's for another reason warns
            of nothing, the targets being the streams' (the total condenser
            of a `biosteam.MESHDistillation` with ``full_condenser=True``
            reports a duty about 0.1 % off the enthalpy change of its own
            streams, for instance).
        RuntimeWarning
            If the first inlet and outlet of a heat utility's unit differ in
            flow (``ins[0].mol`` and ``outs[0].mol``) by more than 1 % of
            the larger: its inlet has changed since its unit was simulated
            (a feed changed, its unit not simulated since), and the targets
            take its stream from that new inlet to the outlet of the former
            state, a state of no process: simulate the system (or the
            units) first. A change of temperature alone keeps the flows
            equal and warns of nothing: the targets are then of the new
            inlet and of the outlet as last simulated, which is the
            outlet's state now if the unit sets it by temperature (as a
            `biosteam.HXutility` with ``T`` does).

        """
        hus = list(hus)
        _check_T_min_app(T_min_app)
        selection = ('take bst.process_tools.heat_exchanger_utilities_from_'
                     'units(sys.units), the heat utilities a heat exchanger '
                     'network may take (before any simulation of a '
                     'HeatExchangerNetwork with replace_unit_heat_utilities='
                     'True, which writes its utilities over them; take its '
                     'targets() after one)')
        if any(hu.unit is None for hu in hus):
            raise ValueError('every heat utility must belong to a heat '
                             'exchanger (hu.unit), not be one of '
                             'System.heat_utilities (sums by agent, of no '
                             f'unit): {selection}')
        others = [hu for hu in hus if hu.duty and not hu.hxn_ok]
        if others:
            names = ', '.join(dict.fromkeys(repr(_unit_label(hu.unit))
                                            for hu in others))
            raise ValueError('every heat utility must be a heat exchanger\'s '
                             f'(hu.hxn_ok), not those of {names}, whose first '
                             'inlet and outlet need not be the stream they '
                             f'heat or cool: {selection}')
        if key_units is not None: key_units = _with_auxiliaries(key_units)
        holders = _held_owners([hu.unit for hu in hus] + (key_units or []))
        # a heat utility counts once per occurrence, as in the synthesis
        # (biosteam repeats those of a unit in parallel): only more
        # occurrences than its unit and owners list are given twice
        over = _overcounted_units(hus, holders)
        if over:
            names = ', '.join(f'{_unit_label(unit, holders)!r} ({n} given, '
                              f'{m} listed)' for unit, n, m in over)
            raise ValueError(f'the heat utilities of {names} are given more '
                             'often than their units, or the owners of '
                             'these, list them in their heat_utilities (which '
                             'repeat those of a unit in parallel, by '
                             'Unit.parallel, once per unit, each a stream): '
                             'their streams would count once more per extra '
                             'one, as when selections from overlapping unit '
                             'lists (such as a unit and an auxiliary unit of '
                             'it) give the same heat utility twice; give the '
                             'heat utilities of units none of which owns '
                             'another, such as bst.process_tools.heat_'
                             'exchanger_utilities_from_units(sys.units)')
        # only those that a network has written over: a duty may differ from
        # its stream's for another reason (biosteam's MESHDistillation total
        # condenser, 0.1 % off), which the targets, the streams', ignore
        replaced = [hu for hu in hus if hu.hxn_ok and _duty_off_stream(hu)]
        if replaced:
            from ._heat_exchanger_network import _replacing_networks
            networks = _replacing_networks(replaced)
            replaced = [hu for hu in replaced if id(hu) in networks]
        if replaced:
            names = ', '.join(dict.fromkeys(repr(_unit_label(hu.unit))
                                            for hu in replaced))
            IDs = list(dict.fromkeys(networks[id(hu)].ID for hu in replaced))
            _warn(f'the heat utilities of {names} have duties (hu.unit_duty) '
                  'other than their process streams\' (outs[0].H - ins[0].H), '
                  'which the targets use: they hold the utilities of '
                  f"HeatExchangerNetwork {', '.join(map(repr, IDs))}, "
                  'simulated with replace_unit_heat_utilities=True, and the '
                  'streams it serves completely have none and are left out; '
                  f"take {' or '.join(f'{ID}.targets()' for ID in IDs)} "
                  'instead')
        values = _explicit_contributions(hus, approach_contributions, T_min_app,
                                         key_units)
        arranged, N_heating = _arrange_heat_utilities(hus, sort_hus_by_T)
        if values is None:
            contributions = None
        else: # in stream order
            value_of = dict(zip(map(id, hus), values))
            contributions = [value_of[id(hu)] for hu in arranged]
        is_hot = [False] * N_heating + [True] * (len(arranged) - N_heating)
        hxs = [hu.unit for hu in arranged]
        labels = [_unit_label(hx, holders) for hx in hxs]
        # before the targets, whose streams a changed inlet may reverse
        _warn_stale(_stale_labels(hxs, labels))
        try:
            self = cls._from_exchangers(hxs, is_hot, T_min_app, contributions,
                                        force_ideal_thermo, labels, arranged,
                                        holders)
        except ValueError as error:
            reversed_ = getattr(error, '_hensmith_reversed', None)
            if reversed_ is None: raise
            j, dH = reversed_
            hu = arranged[j]
            raise ValueError(
                f'the heat utility of {labels[j]!r} '
                f"{'heats' if hu.duty > 0. else 'cools'} (hu.duty = "
                f'{hu.duty:+.6g} kJ/hr), but its stream, from its unit\'s '
                'inlet now to its outlet as last simulated, is '
                f"{'heated' if dH > 0. else 'cooled'} ({dH:+.6g} kJ/hr): its "
                'inlet has changed since its unit was simulated (a feed '
                'changed, or a recycle converged loosely?); simulate the '
                'system (or its units) first, converging any recycle more '
                'tightly') from None
        self.key_units = key_units
        return self

    @classmethod
    def _from_exchangers(cls, hxs, is_hot, T_min_app, approach_contributions,
                         force_ideal_thermo, labels=None, heat_utilities=None,
                         holders=None):
        """Targets of the process streams of the exchangers `hxs` (stream
        order, contributions resolved in stream order), labeled by unit by
        default, with the holders `holders` of auxiliaries held without an
        owner (see `hensmith.hxn_synthesis._held_owners`), as they were when
        the contributions were resolved."""
        _check_T_min_app(T_min_app)
        streams_inlet, streams_quenched = _process_streams(hxs, force_ideal_thermo)
        if labels is None: labels = [_unit_label(hx, holders) for hx in hxs]
        self = cls(streams_inlet, streams_quenched, is_hot, T_min_app,
                   approach_contributions, labels)
        self.units = list(hxs)
        self.heat_utilities = None if heat_utilities is None else list(heat_utilities)
        self.force_ideal_thermo = bool(force_ideal_thermo)
        self._holders = holders
        return self

    def at(self, T_min_app, scale_contributions=False):
        """
        The targets of the same streams at another minimum approach
        temperature, reusing their curves (no flash).

        Parameters
        ----------
        T_min_app : float
            Minimum approach temperature [K], finite and >= 0 (> 0 with
            approach contributions).
        scale_contributions : bool, optional
            Scale the specified approach contributions by ``T_min_app /
            self.T_min_app`` (the contribution scaling of contribution-based
            supertargeting, Ahmad, Linnhoff & Smith 1990); by default they
            stay as they are. Unspecified ones are ``T_min_app / 2`` either
            way. Defaults to False.

        Returns
        -------
        PinchTargets
            With the same streams, curves, labels, units, key units, heat
            utilities and system.

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
        new.key_units = self.key_units
        new.force_ideal_thermo = self.force_ideal_thermo
        new.system = self.system
        new._holders = self._holders
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
        for a threshold problem, whose pinch is at the grid's top with the
        cut 'below')."""
        table = self.table
        k = self._k_pinch
        interval_H = np.abs(table.interval_H)
        point_H = np.abs(table.point_H)
        above = interval_H[:, :k].sum(axis=1) + point_H[:, :k].sum(axis=1)
        below = interval_H[:, k:].sum(axis=1) + point_H[:, k+1:].sum(axis=1)
        if self.cut == 'above': above = above + point_H[:, k]
        else: below = below + point_H[:, k]
        return above, below

    def _level(self, agent, heating, j, reach):
        """
        The shifted level of `agent` serving the streams `j` (`_served`) of
        which those of the mask `reach` (`_in_reach`) are in its reach: the
        highest (lowest for cooling), up to the highest of the shifted limits
        of `j` (down to the lowest), that serves no stream in reach beyond
        its own real limit nor, if the agent is sensible, one at ``agent.T -
        dT`` real (``agent.T + dT`` for cooling) anywhere but at the level
        (see Notes of the class).
        """
        dT = bst.HeatUtility.dT
        grid = self.grid
        Ts = grid['Ts']
        s = self.shift[j]
        top, bottom = Ts[grid['k_hi'][j]], Ts[grid['k_lo'][j]]
        # the point load at the end a level beyond the stream would reach
        # first: its bottom (heating) or its top (cooling)
        k = grid['k_lo' if heating else 'k_hi'][j]
        point = np.abs(self.table.point_H[j, k]) > self._tol
        # a stream in reach reaching beyond its limit caps the level at the
        # limit, or, lying wholly beyond it, at its own end (just short of it
        # if a point load is there): a level serves only what lies below it
        # (above it, for cooling). So does one ending at its limit, for a
        # sensible agent whose dT is 0 (the stream's end at agent.T - dT
        # real; agent.T + dT for cooling): the agent, having given heat above
        # that end (taken it below), would have to leave at its inlet
        # temperature there, at an unbounded flow (see `_line_limit`); at the
        # level, where it has given none, it keeps dT
        sensible = _sensible(agent)
        if heating:
            R = agent.T - agent.dT - dT
            limit = R - s
            beyond = top > limit
            if sensible: beyond |= top + s >= agent.T - dT - _T_EQ
            caps = np.maximum(limit, bottom - _T_SHORT * point)[reach & beyond]
            if j.size: L = caps.min(initial=limit.max())
            else: L = R
        else:
            R = agent.T + agent.dT + dT
            limit = R - s
            beyond = bottom < limit
            if sensible: beyond |= bottom + s <= agent.T + dT + _T_EQ
            caps = np.minimum(limit, top + _T_SHORT * point)[reach & beyond]
            if j.size: L = caps.max(initial=limit.min())
            else: L = R - self.T_min_app
        if reach.any() and sensible:
            L = self._short_of_stuck_loads(agent, heating, L, j[reach], s[reach])
        return float(L)

    def _in_reach(self, j, positions, B, heating):
        """
        Mask of the streams `j` (`_served`) that the heat of a next agent
        can reach given the balanced heat flow `B` at the `positions` of the
        agents already placed: those with heat above the highest zero of `B`
        above the pinch temperature (below the lowest one below it, for
        cooling), a utility pinch or another process pinch; all of them if
        there is no such zero. A level at or beyond that zero gets no duty,
        and the heat of a level short of it never passes it.
        """
        T = positions[0]
        pinch_T = self.pinch_T
        if heating: z = np.flatnonzero((B <= self._tol) & (T > pinch_T))
        else: z = np.flatnonzero((B <= self._tol) & (T < pinch_T))
        if not z.size: return np.ones(j.size, dtype=bool)
        # the heat between the virtual top and the zero (heating), or the
        # zero and the virtual bottom (cooling): position p is column p + 1
        H = self._position_H(positions)[j]
        if heating: heat = np.abs(H[:, z[0] + 1] - H[:, 0])
        else: heat = np.abs(H[:, -1] - H[:, z[-1] + 1])
        return heat > self._tol

    def _short_of_stuck_loads(self, agent, heating, L, j, s):
        """
        The level `L` of the sensible `agent`, or just short of it
        (`_T_SHORT`) if it would serve a point load of the streams `j`
        (shifts `s`) at it that leaves the agent's line no driving force:
        at ``agent.T - dT`` real (``agent.T + dT`` for cooling), a stream's
        own limit when ``agent.dT`` is 0. The agent would leave at its inlet
        temperature, at an unbounded flow; biosteam's rule, strict there,
        does not take it either.
        """
        Ts = self.grid['Ts']
        k = np.flatnonzero(np.abs(Ts - L) <= _T_EQ)
        if not k.size: return L
        dT = bst.HeatUtility.dT
        T = Ts[k][None, :] + s[:, None] # real
        load = np.abs(self.table.point_H[np.ix_(j, k)]) > self._tol
        if heating:
            if (load & (T >= agent.T - dT - _T_EQ)).any(): return L - _T_SHORT
        elif (load & (T <= agent.T + dT + _T_EQ)).any(): return L + _T_SHORT
        return L

    def _allocate(self, agents, Q, heating):
        """
        Greedy list-order allocation of `Q` to `agents` on the grand
        composite curve: ``([(level, q, agent)], remaining)``. Each agent's
        level (`_level`) is set by the streams in reach given the agents
        placed before it (`_in_reach`).
        """
        Ts, arr, lea = self._gcc()
        tol = self._tol
        served = self._served(heating)
        allocated = []
        remaining = Q
        for agent in agents:
            if remaining <= tol: break
            done = [(l, q) for l, q, _ in allocated]
            positions = _positions(Ts, arr, lea, [l for l, _ in done])
            B = (_balance(positions, done, ()) if heating
                 else _balance(positions, (), done))
            reach = self._in_reach(served, positions, B, heating)
            level = _snap(Ts, self._level(agent, heating, served, reach))
            positions = _positions(Ts, arr, lea, [level] + [l for l, _ in done])
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

    def _served(self, heating):
        """Indices of the streams a heating agent can serve (the cold streams
        with heat above the pinch) or a cooling one (the hot streams with
        heat below it)."""
        above, below = self._side_heat()
        if heating: return np.flatnonzero(~self.is_hot & (above > self._tol))
        return np.flatnonzero(self.is_hot & (below > self._tol))

    def _position_H(self, positions):
        """Each stream's enthalpy [kJ/hr] at the `positions` (`_positions`),
        between a virtual top and a virtual bottom one: (N, P + 2), position
        p in column p + 1. The heat of a stream between two columns is the
        heat it has between those positions, point loads included where the
        arriving and the leaving position of their grid temperature both lie
        between them."""
        T, _, kind, k = positions
        Ts, Hl, Hr = self.grid['Ts'], self.grid['Hl'], self.grid['Hr']
        H = np.where(kind == _ARRIVING, Hr[:, k], Hl[:, k])
        inner = np.flatnonzero((kind == _ABOVE) | (kind == _BELOW))
        if inner.size:
            ki = k[inner]
            f = (T[inner] - Ts[ki + 1]) / (Ts[ki] - Ts[ki + 1])
            H[:, inner] = Hr[:, ki + 1] + (Hl[:, ki] - Hr[:, ki + 1]) * f
        return np.column_stack([Hr[:, 0], H, Hl[:, -1]])

    def _pinch_position(self, positions):
        """Index of the process pinch among the `positions`: the heat
        arriving at the pinch temperature if the cut is 'below', the heat
        leaving it if 'above'."""
        T, _, kind, _ = positions
        return int(np.flatnonzero(
            (T == self.grid['Ts'][self._k_pinch])
            & (kind == (_ARRIVING if self.cut == 'below' else _LEAVING)))[0])

    def _ends(self, H, a, b, lo, hi, heating):
        """
        ``((T_in, T_out), (top, bottom, shift))`` of a utility serving the
        heat between columns `a` and `b` of `H` (`_position_H`), shifted [lo,
        hi]: the real temperatures [K] of the streams it serves there (the
        cold ones with heat there above the tolerance, or the hot ones for
        cooling; any heat if no stream has more), the coldest and the hottest
        (heating; the hottest and the coldest for cooling), and their shifted
        ends [K] and shifts [K]. A stream served there only at a point load
        at an end of the stretch counts at that point load's temperature.
        """
        heat = np.abs(H[:, b] - H[:, a])
        heat[self.is_hot if heating else ~self.is_hot] = 0.
        j = np.flatnonzero(heat > self._tol)
        if not j.size: j = np.flatnonzero(heat > 0.)
        if not j.size:
            raise RuntimeError(f'no stream has heat in the stretch [{lo:.6g}, '
                               f'{hi:.6g}] K (shifted) of a utility')
        grid = self.grid
        Ts, s = grid['Ts'], self.shift[j]
        top, bottom = Ts[grid['k_hi'][j]], Ts[grid['k_lo'][j]]
        cold = float(np.min(np.maximum(bottom, lo) + s))
        hot = float(np.max(np.minimum(top, hi) + s))
        return ((cold, hot) if heating else (hot, cold)), (top, bottom, s)

    def _line_limit(self, agent, heating, q, T, B, served):
        """
        The limit [K] on the outlet of the sensible `agent` serving `q` from
        the positions of its stretch (shifted temperatures `T` and balanced
        heat flow `B`): the lowest outlet (heating; the highest for cooling)
        at which its line, linear in its heat, stays `dT` from the `served`
        streams present at each position (see Notes of the class); None if
        no position bounds it.
        """
        top, bottom, s = served
        # the heat the agent has at least given above each position (heating)
        # or taken below it (cooling): the rest of the heat flowing there
        x = q - B
        present = (bottom - _T_EQ <= T[:, None]) & (T[:, None] <= top + _T_EQ)
        bound = (x > self._tol) & present.any(axis=1)
        if not bound.any(): return None
        dT = bst.HeatUtility.dT
        x, T, present = x[bound], T[bound], present[bound]
        if heating:
            T_process = T + np.where(present, s, -math.inf).max(axis=1)
            return agent.T - float(np.min((agent.T - dT - T_process) * q / x))
        T_process = T + np.where(present, s, math.inf).min(axis=1)
        return agent.T + float(np.min((T_process - dT - agent.T) * q / x))

    def utility_targets(self, heating_agents=None, cooling_agents=None):
        """
        Multiple-utility targets: the agents placed on the grand composite
        curve and priced by biosteam (see Notes of the class). They are
        placed in biosteam's order of preference, not by price, so their
        total cost is not the least utility cost of a network.

        Parameters
        ----------
        heating_agents : Sequence[UtilityAgent or str], optional
            Heating agents (``isheating_agent``) in order of preference, or
            the IDs of agents of ``bst.HeatUtility.heating_agents``; fuel
            agents (``isfuel``) take the remainder, the first one only.
            Defaults to ``bst.HeatUtility.heating_agents`` (read at call
            time).
        cooling_agents : Sequence[UtilityAgent or str], optional
            Cooling agents (``iscooling_agent``, not fuels) in order of
            preference, or the IDs of agents of
            ``bst.HeatUtility.cooling_agents``. Defaults to
            ``bst.HeatUtility.cooling_agents``.

        Returns
        -------
        list[UtilityTarget]
            The heating utilities in allocation order (fuel, or an unserved
            remainder, last), then the cooling ones. Agents with no duty are
            left out.

        Raises
        ------
        ValueError
            If an ID names no agent of biosteam's list, or an agent is not
            of the kind of its list.

        Warns
        -----
        RuntimeWarning
            If no agent can serve part of the hot or cold utility target.

        """
        heating_agents = _agents(heating_agents, True)
        cooling_agents = _agents(cooling_agents, False)
        if heating_agents is None: heating_agents = list(bst.HeatUtility.heating_agents)
        if cooling_agents is None: cooling_agents = list(bst.HeatUtility.cooling_agents)
        tol = self._tol
        heat, remaining = self._allocate(
            [a for a in heating_agents if not a.isfuel], self.Q_hot, True)
        if remaining > tol:
            fuels = [a for a in heating_agents if a.isfuel]
            heat.append((math.inf, remaining, fuels[0] if fuels else None))
            if not fuels:
                _warn(f'no heating agent can serve {remaining:.6g} kJ/hr of '
                      'the hot utility target; it is left unserved')
        cool, remaining = self._allocate(cooling_agents, self.Q_cold, False)
        if remaining > tol:
            cool.append((-math.inf, remaining, None))
            _warn(f'no cooling agent can serve {remaining:.6g} kJ/hr of the '
                  'cold utility target; it is left unserved')
        # the stretch each agent serves: from its level, clipped to the grid,
        # to the next zero of the balanced curve, the pinch or a utility pinch
        Ts, arr, lea = self._gcc()
        positions = _positions(Ts, arr, lea,
                               [l for l, _, _ in heat + cool if math.isfinite(l)])
        B = _balance(positions, [(l, q) for l, q, _ in heat],
                     [(l, q) for l, q, _ in cool])
        zero = B <= tol
        T = positions[0]
        # each stream's enthalpy at the positions: its heat in a stretch, the
        # point loads of a grid temperature at its end included only where
        # the stretch holds them (as the balanced curve does)
        H = self._position_H(positions)
        pinch = self._pinch_position(positions)
        dT = bst.HeatUtility.dT
        utilities = []
        with _isolated_pricing([a for *_, a in heat + cool if a is not None]):
            for level, q, agent in heat:
                # positions g to end - 1 (columns g to end of H)
                g = _gap(positions, level)
                z = np.flatnonzero(zero[g:])
                end = g + int(z[0]) + 1 if z.size else pinch + 1
                lo = float(T[end - 1])
                hi = min(level, float(Ts[0]))
                (T_in, T_out), served = self._ends(H, g, end, lo, hi, True)
                T_priced = T_in
                if _sensible(agent):
                    limit = self._line_limit(agent, True, q, T[g:end], B[g:end],
                                             served)
                    if limit is not None: T_priced = max(T_in, limit - dT)
                utilities.append(self._utility(agent, True, level, q, T_in,
                                               T_out, T_priced))
            for level, q, agent in cool:
                # positions start to g - 1 (columns start + 1 to g + 1 of H)
                g = _gap(positions, level)
                z = np.flatnonzero(zero[:g])
                start = int(z[-1]) if z.size else pinch
                hi = float(T[start])
                lo = max(level, float(Ts[-1]))
                (T_in, T_out), served = self._ends(H, start + 1, g + 1, lo, hi,
                                                   False)
                T_priced = T_in
                if _sensible(agent):
                    limit = self._line_limit(agent, False, q, T[start:g],
                                             B[start:g], served)
                    if limit is not None: T_priced = min(T_in, limit + dT)
                utilities.append(self._utility(agent, False, level, q, T_in,
                                               T_out, T_priced))
        return utilities

    def _utility(self, agent, heating, level, q, T_in, T_out, T_priced):
        """The `UtilityTarget` of `agent` serving `q` from `T_in` to `T_out`
        (process temperatures), priced as one biosteam heat utility from
        `T_priced` to `T_out` (inside `_isolated_pricing`)."""
        shift = tuple(self.shift.tolist())
        if agent is None:
            nan = math.nan
            return UtilityTarget(None, None, heating, level, q, nan, nan, nan,
                                 T_in, T_out, nan, nan, shift)
        hu = bst.HeatUtility()
        hu(q if heating else -q, T_priced, T_out, agent=agent)
        return UtilityTarget(agent.ID, agent, heating, level, q, abs(hu.duty),
                             abs(hu.flow), hu.cost, T_in, T_out,
                             hu.inlet_utility_stream.T,
                             hu.outlet_utility_stream.T, shift)

    # %% Balanced curves

    def _check_utilities(self, utilities):
        """Raise a ValueError unless the heating duties of `utilities`
        (unserved ones included) add up to `Q_hot` and the cooling ones to
        `Q_cold`, as those of `utility_targets` do (to the remainders of at
        most `_tol` that it leaves out, and round-off), and every utility's
        `shift` is None or `shift`: the duties alone do not tell targets of
        another shift apart when they do not depend on it (streams of one
        kind only, or a threshold problem below its threshold), and their
        levels would then be read on the wrong shifted temperatures."""
        for heating, Q, name in ((True, self.Q_hot, 'Q_hot'),
                                 (False, self.Q_cold, 'Q_cold')):
            total = math.fsum([u.duty for u in utilities
                               if bool(u.heating) == heating])
            if not abs(total - Q) <= 2. * self._tol + 1e-9 * abs(Q):
                kind = 'heating' if heating else 'cooling'
                raise ValueError(
                    f'the {kind} duties of utilities add up to {total:.6g} '
                    f'kJ/hr, not {name} = {Q:.6g} kJ/hr: give the '
                    'utility_targets() of these targets (those of other '
                    'targets, e.g. of another T_min_app, or a part of them '
                    'do not balance their curves)')
        for u in utilities:
            if u.shift is not None and not np.array_equal(u.shift, self.shift):
                raise ValueError(
                    f'utility {u.ID!r} was placed on the shifted temperatures '
                    'of other targets (its shift is not that of these '
                    'targets): give the utility_targets() of these targets '
                    '(those of other targets, e.g. of another T_min_app, do '
                    'not balance their curves)')

    def _balanced_segments(self, utilities, inv_h=None, inv_h_utilities=None):
        """
        The hot and cold segments (`_composite`) of the balanced composite
        curves on real temperatures: the streams plus the served utilities,
        fuel as a last hot segment at infinite temperature (a weight of 0,
        no area). `inv_h` weighs the streams (N) and `inv_h_utilities` the
        `utilities` (1 by default). The `utilities` are checked
        (`_check_utilities`).
        """
        self._check_utilities(utilities)
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
            `utility_targets` of these targets, with any agents: a
            ValueError is raised unless the heating duties (unserved ones
            included) add up to `Q_hot` and the cooling ones to `Q_cold`,
            and the `shift` of each is None or these targets' `shift` (the
            utilities of other targets, e.g. of another `T_min_app`, or a
            part of them do not balance the curves). Defaults to
            `utility_targets` with the default agents.

        Returns
        -------
        hot_T, hot_H, cold_T, cold_H : numpy.ndarray
            Real temperatures [K] and cumulative heat [kJ/hr] of each
            composite, ascending: the hot one from 0, with fuel as a last
            stretch at ``T = inf``, and the cold one from the unserved
            cooling (0 if none), the coldest of its members, which is left
            out. The curves are thus aligned for vertical heat transfer, and
            end at the same heat unless part of the heating target is
            unserved (the hot one then ends short of the cold one by it).

        """
        if utilities is None: utilities = self.utility_targets()
        hot, cold = self._balanced_segments(utilities)
        hot_T, hot_H = _curve(hot)
        cold_T, cold_H = _curve(cold)
        unserved = sum([u.duty for u in utilities
                        if u.agent is None and not u.heating], 0.)
        if unserved: cold_H = cold_H + unserved
        return hot_T, hot_H, cold_T, cold_H

    def _balanced_positions(self, utilities):
        """The constraint positions with the levels of every allocated
        finite level of `utilities` (checked, see `_check_utilities`), and
        the balanced heat flow at each."""
        self._check_utilities(utilities)
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
            `utility_targets` of these targets, with any agents: a
            ValueError is raised unless the heating duties (unserved ones
            included) add up to `Q_hot` and the cooling ones to `Q_cold`,
            and the `shift` of each is None or these targets' `shift` (the
            utilities of other targets, e.g. of another `T_min_app`, or a
            part of them do not balance the curves). Defaults to
            `utility_targets` with the default agents.

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

    # %% Supertargets

    def _film_coefficients(self, h):
        """The film coefficient of each stream [kW/m²/K] from `h` (see
        `area_target`), an (N,) array."""
        N = self.is_hot.size
        if h is None: return np.full(N, _H_DEFAULT)
        if isinstance(h, Mapping): return self._film_mapping(h)
        if isinstance(h, (Sequence, np.ndarray)) and not isinstance(h, (str, bytes)):
            values = list(h)
            if len(values) != N:
                raise ValueError(f'{len(values)} film coefficients h given for '
                                 f'{N} streams')
            return np.array([_film_value(v, f'h[{j}]')
                             for j, v in enumerate(values)])
        return np.full(N, _film_value(h, 'h'))

    def _film_mapping(self, h):
        """
        `_film_coefficients` of a mapping whose keys are stream indices or
        name units as approach contribution keys do (see
        `hensmith.hxn_synthesis._mapping_contributions`): the key of the
        lowest level on the owner chain of a stream's unit wins, and a stream
        index is below its unit. Keys of the same level for a unit must give
        it the same value. Unit keys may also name `key_units` (and their
        owners) and the auxiliaries that units may build but do not hold
        now, as approach contribution keys of the facility may, and then
        have no effect. An auxiliary that a unit at hand held without an
        owner when the targets were made, or holds so now, is that unit's
        (see `hensmith.hxn_synthesis._key_universe`).
        """
        N = self.is_hot.size
        units = self.units or ()
        universe, live, owners = _key_universe(
            [*units, *(self.key_units or ())], self._holders)
        items = []
        unknown = []
        for key, value in h.items():
            if isinstance(key, bst.HeatUtility):
                raise TypeError('a heat utility cannot be a key of h (biosteam '
                                'replaces the heat utilities of a unit at every '
                                "simulation): use a stream index, its unit "
                                "(hu.unit), the unit's ID or (owner.ID, "
                                'unit.ID)')
            if isinstance(key, Integral) and not isinstance(key, (bool, np.bool_)):
                if not 0 <= key < N: unknown.append(key)
            elif not (isinstance(key, (bst.Unit, str))
                      or isinstance(key, tuple) and len(key) == 2
                      and all(isinstance(i, str) for i in key)):
                raise TypeError('h keys must be stream indices (int), units, '
                                'unit IDs (str) or (owner.ID, unit.ID) tuples, '
                                f'not {type(key).__name__!r} ({key!r})')
            elif _unknown_keys([key], universe, owners):
                unknown.append(key)
            items.append((key, _film_value(value, f'h of key {key!r}')))
        if unknown:
            raise ValueError('h keys that name no stream: '
                             f"{', '.join(map(repr, unknown))} (a key is a "
                             'stream index or, for targets of heat utilities, '
                             'a unit, the ID of a unit without an owner, or '
                             '(owner.ID, unit.ID)'
                             f'{_unknown_key_hints(unknown, universe, owners)}'
                             ')')
        unit_items = [(k, v) for k, v in items if not isinstance(k, Integral)]
        _check_key_levels(unit_items, universe, live, owners, 'h')
        by_index = {k: v for k, v in items if isinstance(k, Integral)}
        values = np.full(N, _H_DEFAULT)
        for j in range(N): # a stream index is below its unit
            if j in by_index: values[j] = by_index[j]
            elif unit_items: # only targets of heat utilities have units
                values[j] = _nearest_key_value(unit_items, units[j], live,
                                               owners, _H_DEFAULT)
        return values

    def _area(self, utilities, h, h_utilities):
        """`area_target` with resolved film coefficients (`h`, an (N,)
        array; `h_utilities`, one per utility)."""
        self._check_utilities(utilities)
        if any(u.agent is None for u in utilities): return math.nan
        hot, cold = self._balanced_segments(
            utilities, 1. / h, [1. / x for x in h_utilities])
        return _bath(hot, cold)

    def area_target(self, utilities=None, h=None, h_utilities=None,
                    agents=None):
        """
        The area target [m²] of vertical heat transfer between the balanced
        composite curves: Bath's formula (see Notes of the class).

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            `utility_targets` of these targets, with any agents: a
            ValueError is raised unless the heating duties (unserved ones
            included) add up to `Q_hot` and the cooling ones to `Q_cold`,
            and the `shift` of each is None or these targets' `shift` (the
            utilities of other targets, e.g. of another `T_min_app`, or a
            part of them do not balance the curves). Defaults to
            `utility_targets` with the default agents.
        h : float, Sequence[float] or Mapping, optional
            Film coefficient of the streams [kW/m²/K]: one for all, one per
            stream in stream order, or a mapping from stream indices (int)
            or, for targets of heat utilities, keys that name units as
            approach contribution keys do (a unit, the ID of a unit without
            an owner, or ``(owner.ID, unit.ID)``; an auxiliary unit takes the
            value of its owner's key unless a key names it). The nearest key
            wins, a stream index first; streams left out take 1. Labels are
            not keys. A key that names no stream raises a ValueError, unless
            it names one of `key_units` (for `HeatExchangerNetwork.targets`,
            any unit available to the network, as for its approach
            contribution keys) or an auxiliary that a unit may build but
            does not hold now: it then has no effect, so that a key of a
            unit idle in a Monte Carlo sample does not fail. Defaults to 1
            for every stream.
        h_utilities : float or Mapping, optional
            Film coefficient of the utilities [kW/m²/K]: one for all, or a
            mapping from agent IDs or agents (the others take 1). An ID must
            name an agent of `utilities`, of `agents` or of biosteam's
            lists, or a ValueError is raised: key another agent by the
            UtilityAgent itself, or pass it in `agents`. Defaults to 1.
        agents : Sequence[UtilityAgent or str], optional
            Agents whose IDs `h_utilities` may key as well, or the IDs of
            agents of biosteam's lists (which it may key anyway): the
            `heating_agents` and `cooling_agents` that `utilities` were
            priced with, so that one `h_utilities` serves at every
            `T_min_app`, an agent that serves nothing at one included, as
            for `summary`. They change nothing else. Defaults to none.

        Returns
        -------
        float
            The area [m²]: inf (with a RuntimeWarning) at a zero or
            negative driving force, nan if part of a utility target is
            unserved.

        """
        if utilities is None: utilities = self.utility_targets()
        return self._area(utilities, self._film_coefficients(h),
                          _utility_films(utilities, h_utilities,
                                         _key_agents(agents)))

    def units_target(self, utilities=None):
        """
        An estimate of the number of units: Euler's S - 1 per region of the
        balanced grand composite curve between its zeros (see Notes of the
        class).

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            `utility_targets` of these targets, with any agents: a
            ValueError is raised unless the heating duties (unserved ones
            included) add up to `Q_hot` and the cooling ones to `Q_cold`,
            and the `shift` of each is None or these targets' `shift` (the
            utilities of other targets, e.g. of another `T_min_app`, or a
            part of them do not balance the curves). Defaults to
            `utility_targets` with the default agents.

        Returns
        -------
        dict
            ``{'above': int, 'below': int, 'total': int}``: the units of the
            regions above the process pinch, below it, and both.

        """
        if utilities is None: utilities = self.utility_targets()
        positions, B = self._balanced_positions(utilities)
        P = positions[0].size
        tol = self._tol
        # each stream's cumulative enthalpy at every position, between the
        # virtual top and bottom ones
        H = self._position_H(positions)
        cuts = [-1, *np.flatnonzero(B <= tol).tolist(), P]
        gaps = [_gap(positions, u.level) for u in utilities if u.duty > tol]
        pinch = self._pinch_position(positions)
        # the members of each region (streams j, utilities -1 - i), grouped:
        # consecutive regions of one gap between two zeros (where the curve
        # stays zero, each piece balanced on its own) on one side of the
        # pinch form one group
        groups = [] # [above, [members of each region], zero]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            heat = np.abs(H[:, c1 + 1] - H[:, c0 + 1])
            members = {*np.flatnonzero(heat > tol).tolist(),
                       *(-1 - i for i, g in enumerate(gaps) if c0 < g <= c1)}
            side = c1 <= pinch
            zero = 0 <= c0 and c1 == c0 + 1 < P
            if zero and groups and groups[-1][2] and groups[-1][0] == side:
                groups[-1][1].append(members)
            else:
                groups.append([side, [members], zero])
        above = below = 0
        for side, regions, _ in groups:
            N = _euler(regions)
            if side: above += N
            else: below += N
        return {'above': above, 'below': below, 'total': above + below}

    def _capital(self, utilities, h, h_utilities, P, heat_exchanger_type,
                 material):
        """`capital_cost_target` with resolved film coefficients (see
        `_area`), and the units target."""
        _check_exchanger(heat_exchanger_type, material)
        P = _pressure(P)
        A = self._area(utilities, h, h_utilities)
        N = self.units_target(utilities)['total']
        fuel = [u for u in utilities if u.agent is not None and u.agent.isfuel]
        N_shells = N - (1 if fuel else 0) # one furnace for all fuel
        if A > 0. and N_shells < 1: N_shells = 1
        elif A == 0.: N_shells = 0
        if N_shells:
            purchase, installed, N_area_shells, kind = _shell_split(
                A, N_shells, P, heat_exchanger_type, material)
            shell_area = A / N_area_shells
        else: # no area, or nan
            shell_area = purchase = installed = A
            N_area_shells = 0
            kind = None
        furnace_duty = sum((u.utility_duty for u in fuel), 0.)
        if fuel:
            furnace_purchase, furnace_installed = _furnace_costs(furnace_duty)
            purchase += furnace_purchase
            installed += furnace_installed
        return {'area': A, 'N_shells': N_shells,
                'N_area_shells': N_area_shells, 'shell_area': shell_area,
                'type': kind, 'furnace_duty': furnace_duty,
                'purchase_cost': purchase, 'installed_cost': installed}, N

    def capital_cost_target(self, utilities=None, h=None, h_utilities=None,
                            P=101325., heat_exchanger_type='Floating head',
                            material='Carbon steel/carbon steel', agents=None):
        """
        The capital cost target: the area target in as many shells as the
        units target (less a furnace), split equally over all of them or
        over fewer if that costs more (the others then have no area), costed
        with biosteam's correlations, plus a furnace for the fuel (see Notes
        of the class).

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            `utility_targets` of these targets, with any agents: a
            ValueError is raised unless the heating duties (unserved ones
            included) add up to `Q_hot` and the cooling ones to `Q_cold`,
            and the `shift` of each is None or these targets' `shift` (the
            utilities of other targets, e.g. of another `T_min_app`, or a
            part of them do not balance the curves). Defaults to
            `utility_targets` with the default agents.
        h, h_utilities : optional
            Film coefficients, as `area_target` takes them.
        P : float, optional
            Pressure of every shell [Pa], > 0. Defaults to 101325.
        heat_exchanger_type : str, optional
            biosteam's type of the shell and tube exchangers ('Floating
            head', 'Fixed head', 'U tube', 'Kettle vaporizer' or 'Double
            pipe'). Defaults to 'Floating head'.
        material : str, optional
            biosteam's shell and tube material. Defaults to 'Carbon
            steel/carbon steel'.
        agents : Sequence[UtilityAgent or str], optional
            Agents whose IDs `h_utilities` may key as well, as `area_target`
            takes them.

        Returns
        -------
        dict
            'area' [m²]; 'N_shells', the number of shells; 'N_area_shells',
            how many of them share the area equally (the others have none);
            'shell_area', the area of each of those [m²]; 'type', biosteam's
            type of each ('Double pipe' below 150 ft², None without shells
            or with an infinite or nan area); 'furnace_duty', the utility
            duty of the fuel [kJ/hr];
            'purchase_cost' and 'installed_cost' (bare module) [USD]. inf
            with an infinite area, nan with an unserved utility.

        """
        if utilities is None: utilities = self.utility_targets()
        return self._capital(utilities, self._film_coefficients(h),
                             _utility_films(utilities, h_utilities,
                                            _key_agents(agents)), P,
                             heat_exchanger_type, material)[0]

    def _check_options(self, options):
        """The options of `summary` (agents as lists of agents, the
        operating hours of `system` filled in if it has them and neither
        `operating_hours` nor `tea` is given), after raising for any that
        `summary` would reject, before pricing anything."""
        unknown = [i for i in options if i not in _SUMMARY_OPTIONS]
        if unknown:
            raise TypeError(f"unexpected options: {', '.join(map(repr, unknown))} "
                            f'(the options are {", ".join(_SUMMARY_OPTIONS)})')
        options = dict(options)
        for key in ('heating_agents', 'cooling_agents'):
            if options.get(key) is not None:
                options[key] = _agents(options[key], key == 'heating_agents')
        if options.get('operating_hours') is None and options.get('tea') is None:
            hours = getattr(self.system, 'operating_hours', None)
            if hours is not None: options['operating_hours'] = hours
        get = options.get
        _economics(get('interest'), get('years'), get('operating_hours'),
                   get('tea'))
        _check_exchanger(get('heat_exchanger_type', 'Floating head'),
                         get('material', 'Carbon steel/carbon steel'))
        _pressure(get('P', 101325.))
        self._film_coefficients(get('h'))
        _utility_films((), get('h_utilities'),
                       [*(get('heating_agents') or ()),
                        *(get('cooling_agents') or ())], _OPTION_AGENTS)
        return options

    def summary(self, **options):
        """
        The targets of energy, area, units, capital and total annualized
        cost (TAC), pricing the utilities once.

        Parameters
        ----------
        **options
            `heating_agents` and `cooling_agents` (as `utility_targets`
            takes them); `h` and `h_utilities` (as `area_target` takes
            them); `P`, `heat_exchanger_type` and `material` (as
            `capital_cost_target` takes them); `interest` (default 0.10),
            `years` (default 10) and `operating_hours` (per year, default
            8000), each taken from `tea` when not given and `tea` is (its
            `IRR`, ``duration[1] - duration[0]`` and `operating_hours`
            unless None). Without `operating_hours` and `tea`, the
            operating hours are those of `system` if it has them, as for
            targets of `HeatExchangerNetwork.targets` in a system with
            operating hours (a TEA's, for instance), so that they match
            `network_report`.

        Returns
        -------
        dict
            'T_min_app' [K]; 'Q_hot' and 'Q_cold' [kJ/hr], process side;
            'utility_cost' [USD/hr]; 'area' [m²]; 'N_units', the units
            target; 'N_shells', 'N_area_shells', 'shell_area' [m²] and
            'shell_type', the 'type' of `capital_cost_target`, as it gives
            them; 'purchase_cost' and 'installed_cost' [USD];
            'annualized_capital_cost', the installed cost times the capital
            recovery factor, 'operating_cost', the utility cost times the
            operating hours, and their sum 'TAC' [USD/yr].

        """
        options = self._check_options(options)
        get = options.get
        heating_agents = get('heating_agents')
        cooling_agents = get('cooling_agents')
        crf, hours = _economics(get('interest'), get('years'),
                                get('operating_hours'), get('tea'))
        utilities = self.utility_targets(heating_agents, cooling_agents)
        h_utilities = _utility_films(
            utilities, get('h_utilities'),
            [*(heating_agents or ()), *(cooling_agents or ())], _OPTION_AGENTS)
        capital, N = self._capital(
            utilities, self._film_coefficients(get('h')), h_utilities,
            get('P', 101325.), get('heat_exchanger_type', 'Floating head'),
            get('material', 'Carbon steel/carbon steel'))
        utility_cost = sum((u.cost for u in utilities), 0.)
        annualized = crf * capital['installed_cost']
        operating = hours * utility_cost
        return {'T_min_app': self.T_min_app, 'Q_hot': self.Q_hot,
                'Q_cold': self.Q_cold, 'utility_cost': utility_cost,
                'area': capital['area'], 'N_units': N,
                'N_shells': capital['N_shells'],
                'N_area_shells': capital['N_area_shells'],
                'shell_area': capital['shell_area'],
                'shell_type': capital['type'],
                'purchase_cost': capital['purchase_cost'],
                'installed_cost': capital['installed_cost'],
                'annualized_capital_cost': annualized,
                'operating_cost': operating, 'TAC': annualized + operating}


def _least_TAC(records):
    """Index of the record of least TAC (nan counts as inf)."""
    TAC = np.array([r['TAC'] for r in records], dtype=float)
    return int(np.argmin(np.where(np.isnan(TAC), math.inf, TAC)))

def _constant_TAC(records, where_finite=False):
    """Whether the records are at least two, each with a finite TAC (or, if
    `where_finite`, have at least two finite TACs, the others left out),
    and their TACs the same to within round-off (a relative 1e-12)."""
    TAC = np.array([r['TAC'] for r in records], dtype=float)
    finite = np.isfinite(TAC)
    if where_finite: TAC = TAC[finite]
    elif not finite.all(): return False
    return (TAC.size > 1
            and float(np.ptp(TAC)) <= 1e-12 * float(np.abs(TAC).max()))

def supertarget(targets, T_min_apps=None, refine=3, n_refine=8,
                scale_contributions=False, **options):
    """
    Sweep the minimum approach temperature for the least total annualized
    cost (TAC) target: supertargeting [Ahmad90]_.

    Parameters
    ----------
    targets : PinchTargets
        The streams (their curves are reused: no flash).
    T_min_apps : Sequence[float], optional
        Minimum approach temperatures to evaluate [K], finite and >= 0 (> 0
        with approach contributions); sorted and made unique. Defaults to 1,
        2, ..., 40 K.
    refine : int, optional
        Rounds of refinement around the least TAC. Defaults to 3.
    n_refine : int, optional
        New temperatures evaluated per round of refinement. Defaults to 8.
    scale_contributions : bool, optional
        Scale the specified approach contributions of `targets` with the
        minimum approach temperature (see `PinchTargets.at`). Defaults to
        False.
    **options
        Options of `PinchTargets.summary`.

    Returns
    -------
    Supertargets

    Warns
    -----
    RuntimeWarning
        Once for the sweep, if some minimum approach temperatures leave part
        of a utility target unserved (no agent given can serve it: their
        TAC is nan), instead of the warning of
        `PinchTargets.utility_targets` at each.
    RuntimeWarning
        If no minimum approach temperature of the sweep has a finite TAC
        (each has an unserved utility or an infinite area): the sweep is
        then not refined, and its optimum is the smallest one, meaningless.

    Notes
    -----
    Each minimum approach temperature is evaluated with
    ``targets.at(T_min_app, scale_contributions).summary(**options)``. Only
    the streams without an approach contribution follow the minimum approach
    temperature (``T_min_app / 2`` each), unless `scale_contributions`
    scales the specified ones by ``T_min_app / targets.T_min_app`` too, the
    contribution scaling of [Ahmad90]_; with every stream specified, the
    sweep needs it. The streams' curves are reused, but each point builds
    its own problem table (about that of a synthesis, which reads every
    stream's enthalpy on its grid), so on a system of many streams with
    rigorous thermodynamics the default sweep (40 temperatures and three
    refinements) can take longer than a synthesis: a coarser `T_min_apps`
    or a smaller `refine` keeps it short.

    The TAC is not smooth (the units target jumps, so do biosteam's
    correlations at 150 ft², and the pinch can move), so the sweep is a
    grid search: each round of refinement evaluates `n_refine` new
    temperatures between the neighbors of the least TAC so far, never on
    it: with it, they cut that bracket into ``n_refine + 2`` gaps, shared by
    its two sides in proportion to their widths, each side evenly spaced
    (all on its one side at a bound of the sweep). On an evenly spaced grid,
    after r rounds the spacing near the optimum is then about the grid's
    times ``(2 / (n_refine + 2))**r``, (1/5)**r for the defaults. A least
    TAC at the smallest or largest temperature evaluated is flagged
    (`Supertargets.at_bound`). A sweep whose TACs are all finite and the
    same (to a relative 1e-12) is not refined: its targets do not depend on
    the minimum approach temperature over the sweep (for streams that
    recover no heat, only heaters, say), its optimum is merely the smallest
    temperature, and `Supertargets.constant` flags it. Nor is one whose
    finite TACs (at least two) are the same, the others not (an unserved
    utility or an infinite area): its optimum is merely the smallest
    temperature of that flat stretch, so its repr says 'TAC constant where
    finite' rather than 'at the bound'.

    The capital target drops by about a shell and tube wherever the
    shrinking area fills one fewer of 150 ft² (see Notes of
    `PinchTargets`), so the least TAC often lies just past such a step, at
    the first temperature evaluated beyond it, rather than at a smooth
    trade-off of energy and capital. The records' 'N_area_shells' and
    'shell_type' show the steps (`Supertargets.to_frame`). A network has
    such steps too, but where its own exchangers cross 150 ft², which the
    target, with its equal shells, places only roughly.

    Where the targets do not depend on the minimum approach temperature
    (below the threshold of a threshold problem, where neither the utilities
    nor the balanced composite curves change), the TACs differ only by the
    discretization of the streams' curves on the problem table's grid, by
    a tiny relative amount, and where the optimum lies in that flat stretch
    means nothing physically. Where the discretization trend is steady, it
    lies at an end of the stretch (for the quickstart system of the
    tutorial, the warmest point evaluated, just short of the threshold):
    compare the TACs of its neighbors before reading anything into its
    position.

    Unless `operating_hours` or `tea` is given, the operating hours are
    those of ``targets.system`` if it has them (see `PinchTargets.summary`).

    Examples
    --------
    One hot stream cooled 400 -> 300 K and one cold stream heated 290 ->
    390 K, both 1 kW/K. Up to 10 K they balance with no utility in a single
    exchanger of 20 m² (215 ft², a shell and tube); above it the utilities
    add two units, but the area still fills one shell and tube until it
    falls below 150 ft² (13.94 m²) just below 15 K, where three double pipes
    take it. So on a coarse grid the least total annualized cost (TAC) is at
    15 K:

    >>> import biosteam as bst, thermosteam as tmo
    >>> from hensmith import PinchTargets, supertarget
    >>> fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1.,
    ...                      Cn=3.6, default=True)
    >>> bst.settings.set_thermo([fluid])
    >>> bst.main_flowsheet.set_flowsheet('supertargeting_example')
    >>> H1 = bst.HXutility('H1', ins=bst.Stream('H1_in', Fluid=1000., T=400.),
    ...                    T=300., rigorous=False)
    >>> C1 = bst.HXutility('C1', ins=bst.Stream('C1_in', Fluid=1000., T=290.),
    ...                    T=390., rigorous=False)
    >>> H1.simulate(); C1.simulate()
    >>> targets = PinchTargets.from_heat_utilities(
    ...     [H1.heat_utilities[0], C1.heat_utilities[0]], T_min_app=10.)
    >>> grid = [5., 10., 15., 20., 25., 30.]
    >>> sweep = supertarget(targets, grid, refine=0)
    >>> sweep
    <Supertargets: 6 points, T_min_app 5-30 K, optimum 15 K (TAC 4.949e+03 USD/yr)>
    >>> sweep['N_units'], sweep['area'].round(2)
    (array([1, 1, 3, 3, 4, 4]), array([20.  , 20.  , 13.88, 11.07,  9.73,  9.17]))
    >>> sweep['N_area_shells'], [r['shell_type'] for r in sweep.records[1:3]]
    (array([1, 1, 3, 3, 4, 4]), ['Floating head', 'Double pipe'])

    Refining around it finds the least TAC at the first temperature
    evaluated past that step, where the TAC falls to under half that of its
    neighbor below: the step decides the optimum, not a trade-off of energy
    and capital.

    >>> sweep = supertarget(targets, grid)
    >>> sweep
    <Supertargets: 30 points, T_min_app 5-30 K, optimum 14.96 K (TAC 4.936e+03 USD/yr)>
    >>> i = sweep.records.index(sweep.optimum)
    >>> [(r['shell_type'], round(r['area'], 2), round(r['TAC']))
    ...  for r in sweep.records[i - 1:i + 1]]
    [('Floating head', 13.95, 12573), ('Double pipe', 13.91, 4936)]

    References
    ----------
    .. [Ahmad90] Ahmad, S., Linnhoff, B., & Smith, R. (1990). Cost optimum
        heat exchanger networks - 2. Targets and design for detailed capital
        cost models. Computers & Chemical Engineering, 14(7), 751-767.

    """
    if T_min_apps is None: T_min_apps = _T_MIN_APPS
    try:
        values = list(T_min_apps)
    except TypeError:
        values = []
    if not values or any(np.ndim(x) for x in values):
        raise ValueError('T_min_apps must be a non-empty sequence of '
                         'temperatures [K]')
    # the rule of PinchTargets.at: finite numbers, not bools or strings
    T = np.array([_real(x, 'each of T_min_apps') for x in values])
    contributions = targets.approach_contributions
    if contributions is None:
        if np.any(T < 0.):
            raise ValueError(f'T_min_apps must be >= 0 K, not {T_min_apps!r}')
    elif np.any(T <= 0.):
        raise ValueError('T_min_apps must be > 0 K with approach '
                         f'contributions, not {T_min_apps!r}')
    for value, name, low in ((refine, 'refine', 0), (n_refine, 'n_refine', 1)):
        if (not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
                or value < low):
            raise ValueError(f'{name} must be an int >= {low}, not {value!r}')
    if (contributions is not None and not scale_contributions
            and all(c is not None for c in contributions)):
        raise ValueError('every stream has an approach contribution, so '
                         'T_min_app changes nothing: pass '
                         'scale_contributions=True to scale them with it')
    options = targets._check_options(options)
    def evaluate(T_min_app):
        with catch_warnings(): # counted once for the sweep instead
            filterwarnings('ignore', r'no (heating|cooling) agent can serve',
                           RuntimeWarning)
            return targets.at(T_min_app, scale_contributions).summary(**options)
    records = [evaluate(float(x)) for x in np.unique(T)]
    for _ in range(refine):
        if not any(math.isfinite(r['TAC']) for r in records):
            break # nothing to refine around
        if _constant_TAC(records, where_finite=True):
            break # a flat line (where finite): nothing to refine either
        records.sort(key=lambda r: r['T_min_app'])
        i = _least_TAC(records)
        lo = records[max(i - 1, 0)]['T_min_app']
        hi = records[min(i + 1, len(records) - 1)]['T_min_app']
        if hi - lo <= 1e-6: break
        # the new temperatures go on both sides of the least TAC, never on
        # it (a bracket's midpoint can be it): with it they cut the bracket
        # into n_refine + 2 gaps, shared by its sides in proportion to their
        # widths (all on its one side at a bound of the sweep), so each side
        # is spaced evenly and the two about alike
        x = records[i]['T_min_app']
        k = math.floor((n_refine + 2) * (x - lo) / (hi - lo) + 0.5) - 1
        k = min(max(k, 0), n_refine) # how many go below it
        candidates = np.concatenate((
            np.linspace(lo, x, k + 2)[1:-1],
            np.linspace(x, hi, n_refine - k + 2)[1:-1]))
        evaluated = np.array([r['T_min_app'] for r in records])
        for T_new in candidates:
            if np.abs(evaluated - T_new).min() > 1e-12:
                records.append(evaluate(float(T_new)))
    sweep = Supertargets(targets, records, options, scale_contributions)
    unserved = int(np.isnan(sweep['utility_cost']).sum())
    if unserved:
        _warn(f'{unserved} of the {len(records)} minimum approach '
              'temperatures evaluated leave part of a utility target '
              'unserved (no agent given can serve it), so their TAC is nan')
    if not np.isfinite(sweep['TAC']).any():
        _warn('no minimum approach temperature of the sweep has a finite TAC '
              '(each has an unserved utility or an infinite area), so it is '
              'not refined and its optimum, the smallest, is meaningless')
    return sweep


class Supertargets:
    """
    The targets of a sweep of the minimum approach temperature, as
    `supertarget` returns them.

    Parameters
    ----------
    targets : PinchTargets
        The streams swept.
    records : list[dict]
        `PinchTargets.summary` at each minimum approach temperature.
    options : dict, optional
        Options of `PinchTargets.summary` of the records.
    scale_contributions : bool, optional
        Whether the approach contributions were scaled (see
        `PinchTargets.at`). Defaults to False.

    Attributes
    ----------
    targets : PinchTargets
        The streams swept.
    options : dict
        Options of `PinchTargets.summary` of the records.
    scale_contributions : bool
        Whether the approach contributions were scaled.
    records : list[dict]
        `PinchTargets.summary` at each minimum approach temperature,
        ascending.
    optimum : dict
        The record of least TAC (the first of equal ones; nan counts as
        inf).
    T_min_app : float
        Minimum approach temperature of `optimum` [K].
    at_bound : bool
        Whether `optimum` is at the smallest or largest temperature
        evaluated: the least TAC may lie beyond the sweep.
    constant : bool
        Whether the TAC is the same at every temperature evaluated (at least
        two, each finite, to a relative 1e-12): the targets do not depend
        on the minimum approach temperature over the sweep, and `optimum`
        is merely the first record.

    Notes
    -----
    ``sweep[key]`` is the array of a key of the records over the sweep (for
    instance ``sweep['TAC']``), and `to_frame` a pandas DataFrame of them
    indexed by 'T_min_app'.

    """
    def __init__(self, targets, records, options=None,
                 scale_contributions=False):
        if not records: raise ValueError('Supertargets needs at least one record')
        self.targets = targets
        self.options = dict(options or {})
        self.scale_contributions = scale_contributions
        self.records = sorted(records, key=lambda r: r['T_min_app'])
        i = _least_TAC(self.records)
        self.optimum = self.records[i]
        self.T_min_app = self.optimum['T_min_app']
        self.at_bound = i in (0, len(self.records) - 1)
        self.constant = _constant_TAC(self.records)

    def __getitem__(self, key):
        return np.array([r[key] for r in self.records])

    def to_frame(self):
        """The records as a pandas DataFrame indexed by 'T_min_app'."""
        import pandas as pd
        return pd.DataFrame(self.records).set_index('T_min_app')

    def __repr__(self):
        n = len(self.records)
        lo, hi = self.records[0]['T_min_app'], self.records[-1]['T_min_app']
        if self.constant: bound = ', TAC constant over the sweep'
        elif _constant_TAC(self.records, where_finite=True):
            bound = ', TAC constant where finite'
        elif self.at_bound: bound = ', at the bound'
        else: bound = ''
        return (f'<{type(self).__name__}: {n} point{"s" if n != 1 else ""}, '
                f'T_min_app {lo:g}-{hi:g} K, optimum {self.T_min_app:.4g} K '
                f'(TAC {self.optimum["TAC"]:.3e} USD/yr){bound}>')


def _profile_area(x, dT, inv_U, ID, cause=''):
    """
    Area [m²] of counter-current heat transfer over the duties `x` [kJ/hr]
    (ascending) at the driving forces `dT` [K] there, linear in between (each
    stretch at its log-mean temperature difference), with an overall 1/U of
    `inv_U` [m²·K/kW]; inf with a RuntimeWarning naming the exchanger `ID`
    (followed by `cause`) at a driving force of `_DT_ZERO` or less.
    """
    if dT.min() <= _DT_ZERO:
        _warn(f'{ID} transfers heat at a driving force of {dT.min():.6g} K'
              f'{cause}')
        return math.inf
    LMTD = _lmtd(dT[:-1], dT[1:])
    return float(np.sum(np.diff(x) / LMTD)) * inv_U / 3600.

def _stage_profile(curve, H0, sign, f, Q):
    """
    ``(knots, T)`` of a stream of `curve` through an exchanger whose duty x
    runs from 0 to `Q` [kJ/hr] while the stream's enthalpy on the stage
    (fraction `f` of the whole flow) is ``H0 + sign * x``: the duties in (0,
    Q) at the breakpoints of the curve, and its temperature [K] at duties.
    """
    knots = sign * (f * curve.H - H0)
    return (knots[(knots > 0.) & (knots < Q)],
            lambda x: np.interp((H0 + sign * x) / f, curve.H, curve.T))

def _flows(stream):
    """Molar flow rate of each chemical of `stream` [kmol/hr], an array."""
    mol = stream.mol
    return mol.to_array() if hasattr(mol, 'to_array') else np.asarray(mol)

def _same_ends(curve, s_in, s_out):
    """Whether the streams `s_in` and `s_out` are at the end states of
    `curve` (on the curve's thermodynamic basis): the same enthalpies, at
    the outlet to the larger of 1e-6 of the duty and the heat of a
    micro-kelvin of the outlet stream (a flash reaches the outlet only to
    its tolerance, which is more than 1e-6 of the duty of a stream that
    changes temperature by micro-kelvins); and the same pressure and flows
    at the inlet (an inlet at the enthalpy reference has the same enthalpy
    at any flow)."""
    span = abs(curve.H_out - curve.H_in)
    a, b = _flows(curve.stream_in), _flows(s_in)
    dH_out = abs(curve.H_out - s_out.H)
    return (abs(curve.H_in - s_in.H) <= 1e-9 * (abs(curve.H_in) + span)
            and (dH_out <= 1e-6 * span or dH_out <= 1e-6 * abs(s_out.C))
            and abs(curve.P - s_in.P) <= 1e-9 * curve.P
            and a.shape == b.shape
            and np.abs(a - b).max(initial=0.) <= 1e-9 * np.abs(a).max(initial=0.))

def _check_targets(HXN, targets):
    """Raise a ValueError unless `targets` are those of the process
    streams of `HXN`'s network, in its stream order and current states."""
    hxs = HXN.original_heat_exchangers
    units = targets.units
    if (units is None or len(units) != len(hxs)
            or any(u is not hx for u, hx in zip(units, hxs))):
        raise ValueError("targets must be of the network's process streams "
                         "(the units of HXN.original_heat_exchangers, in that "
                         'order), as HXN.targets() makes them')
    ideal = targets.force_ideal_thermo
    if ideal != bool(HXN.force_ideal_thermo):
        raise ValueError(f'targets have force_ideal_thermo={ideal}, the '
                         f'network {bool(HXN.force_ideal_thermo)}')
    for c, hx in zip(targets.curves, hxs):
        s_in, s_out = hx.ins[0], hx.outs[0]
        if ideal: # the curves' basis: copies with ideal thermodynamics
            s_in = _copy(s_in, s_in.thermo.ideal())
            s_out = _copy(s_out, s_out.thermo.ideal())
        if not _same_ends(c, s_in, s_out):
            raise ValueError(f'targets are not of the current states of '
                             f'{_unit_label(hx, targets._holders)} '
                             '(simulated since?): make '
                             'them again with HXN.targets()')

def _warn(message):
    """Warn `message` (a RuntimeWarning) at the line that called hensmith,
    the first frame outside the package, not at a line of the package."""
    warn(message, RuntimeWarning, skip_file_prefixes=_PACKAGE_FRAMES)

def _stale_labels(hxs, labels):
    """The labels (of `labels`, aligned with `hxs`) of the exchangers `hxs`
    whose first inlet and outlet differ in flow (``ins[0].mol`` and
    ``outs[0].mol``) by more than `_STALE_FLOW` of the larger: an inlet
    changed since its unit was simulated, whose stream the targets would
    take from that new inlet to the outlet of the former state."""
    stale = []
    for hx, label in zip(hxs, labels):
        a, b = hx.ins[0].mol, hx.outs[0].mol
        if a.size == b.size and (np.abs(a - b).sum()
                                 > _STALE_FLOW * max(a.sum(), b.sum())):
            stale.append(label)
    return stale

def _warn_stale(labels, of=''):
    """Warn (see `_warn`) of the streams of `labels` (see `_stale_labels`),
    if any, in the targets `of` names (e.g. ``' of HXN'``)."""
    if not labels: return
    names = ', '.join(map(repr, labels))
    if len(labels) == 1:
        _warn(f'the inlet and outlet of {names} differ in flow (its inlet '
              'changed since its unit was simulated?): the targets'
              f'{of} take its stream from its inlet now to its former '
              'outlet, a state of no process; simulate the system (or its '
              'units) first')
    else:
        _warn(f'the inlets and outlets of {names} differ in flow (their '
              'inlets changed since their units were simulated?): the '
              f'targets{of} take their streams from their inlets now to '
              'their former outlets, states of no process; simulate the '
              'system (or its units) first')

def _rebuilt_label(hxs, labels, units):
    """The label (of `labels`, aligned with `hxs`) of the first of the
    exchangers `hxs` that is not one of `units` (a facility's units, their
    nested auxiliaries and the exchangers they hold through their heat
    utilities only, see `hensmith.hxn_synthesis._with_auxiliaries`: a unit
    has rebuilt it since), or None."""
    at_hand = {id(unit) for unit in units}
    for hx, label in zip(hxs, labels):
        if id(hx) not in at_hand: return label
    return None

def _miscounted(hxs, labels, units, at_hand):
    """
    ``[(label, n, listed), ...]``, in order of first occurrence in `hxs`
    (`labels` aligned with it): the exchangers of `hxs` that are among
    `at_hand` (not rebuilt since, see `_rebuilt_label`) and give `n` of its
    streams, one per time their heat utility was listed at the synthesis,
    while the `heat_utilities` of `units` (a facility's units, from which
    its synthesis takes the heat utilities) list a heat exchanger's heat
    utility of theirs `listed` times now, with or without duty. biosteam
    lists the heat utilities of a unit in parallel once per unit
    (``Unit.parallel``, which a `biosteam.AbstractStirredTankReactor` sets
    by its volume), so their number changes with the unit's size when the
    unit is simulated alone; the streams stay one per unit at the facility's
    last simulation.
    """
    given = {}
    for hx, label in zip(hxs, labels):
        given.setdefault(id(hx), [label, 0])[1] += 1
    listed = {}
    for unit in units:
        for hu in unit.heat_utilities:
            if hu.hxn_ok: listed[id(hu.unit)] = listed.get(id(hu.unit), 0) + 1
    at_hand = {id(unit) for unit in at_hand}
    return [(label, n, listed.get(key, 0))
            for key, (label, n) in given.items()
            if key in at_hand and listed.get(key, 0) != n]

def _miscounted_names(miscounted):
    """The streams of `miscounted` (see `_miscounted`) for a message, e.g.
    ``"'R1 - heat_exchanger' 5 times, listed 2 times now"``."""
    times = lambda n: f'{n} time{"" if n == 1 else "s"}'
    return '; '.join(f'{label!r} {times(n)}, listed {times(m)} now'
                     for label, n, m in miscounted)

def _check_network(HXN, targets):
    """Raise a ValueError unless every stream of `HXN`'s network is that of
    an original exchanger still among the facility's units, their nested
    auxiliaries and the exchangers they hold through their heat utilities
    only (see `_rebuilt_label`; not one that its unit has rebuilt since, as a
    `biosteam.MultistageCompressor` simulated alone after a change of `pr`
    rebuilds its intercoolers, whose former objects keep their former,
    consistent states), every such exchanger gives as many streams as the
    facility's units list its heat utility now (see `_miscounted`: not a
    unit in parallel whose number has changed since), and every stream
    enters and leaves the network at the end states
    of its curve in `targets` (which are of the units' current states): its
    copy of the real inlet where it enters the network, and the outlet of
    its last stage (the network's copies share the curves' thermodynamic
    basis)."""
    hxs = HXN.original_heat_exchangers
    units = list(HXN._get_units())
    at_hand = HXN._candidate_units(units)
    label = _rebuilt_label(hxs, targets.labels, at_hand)
    if label is not None:
        raise ValueError(f'the network of {HXN.ID} is of {label}, which is '
                         "no longer one of the facility's units nor an "
                         'auxiliary of one (rebuilt since the '
                         "facility's last simulation?): simulate the "
                         'system')
    miscounted = _miscounted(hxs, targets.labels, units, at_hand)
    if miscounted:
        raise ValueError(f'the network of {HXN.ID} has a stream per heat '
                         'utility its units listed at its last simulation, '
                         'not as they list them now: '
                         f'{_miscounted_names(miscounted)} (Unit.parallel '
                         'changed since, its unit simulated alone?): '
                         'simulate the system')
    for life_cycle, c, label in zip(HXN.stream_life_cycles, targets.curves,
                                    targets.labels):
        entry = life_cycle.entry
        if entry is None: s_in = life_cycle.life_cycle[0].s_in
        else: s_in = entry.unit.ins[entry.index]
        if not _same_ends(c, s_in, life_cycle.life_cycle[-1].s_out):
            raise ValueError(f'the network of {HXN.ID} is not of the current '
                             f'states of {label} (changed since the '
                             "facility's last simulation?): simulate the "
                             'system')

def _network_areas(HXN, targets, h, h_utilities, agents):
    """
    ``{ID: (area, area_biosteam)}`` [m²] of the process exchangers and then
    the utility exchangers of `HXN` with duty (see `network_report`), with
    the film coefficients `h` (an (N,) array) and `h_utilities` (as
    `PinchTargets.area_target` takes them; `agents`, more agents its keys
    may name). A furnace takes no area.
    """
    curves = targets.curves
    is_hot = targets.is_hot
    stages = {} # unit -> [(stream index, life stage)]
    for lc in HXN.stream_life_cycles:
        for st in lc.life_cycle:
            stages.setdefault(st.unit, []).append((lc.index, st))
    process = [hx for hx in HXN.new_HXs if hx.total_heat_transfer > 1e-12]
    utility = [hx for hx in HXN.new_HX_utils
               if hx.heat_utilities and hx.total_heat_transfer > 1e-12]
    of = [hx.heat_utilities[0].agent for hx in utility]
    areas = {}
    for hx in process:
        (i, si), (j, sj) = stages[hx]
        if not is_hot[i]: (i, si), (j, sj) = (j, sj), (i, si)
        # x from the hot inlet (the cold outlet) end
        Q = si.H_in - si.H_out
        knots_h, T_h = _stage_profile(curves[i], si.H_in, -1., si.fraction, Q)
        knots_c, T_c = _stage_profile(curves[j], sj.H_out, -1., sj.fraction, Q)
        x = np.unique(np.concatenate([[0., Q], knots_h, knots_c]))
        areas[hx.ID] = (
            _profile_area(x, T_h(x) - T_c(x), 1. / h[i] + 1. / h[j], hx.ID),
            hx.design_results.get('Area', 0.) / _FT2_PER_M2)
    for hx, agent, h_agent in zip(utility, of,
                                  _agent_films(of, h_utilities, agents,
                                               _OPTION_AGENTS)):
        if agent.isfuel:
            areas[hx.ID] = (0., 0.)
            continue
        (i, si), = stages[hx]
        hu = hx.heat_utilities[0]
        # x from the process outlet (the utility inlet) end
        H_in, H_out = si.H_in, si.H_out
        heating = H_out > H_in
        Q = abs(H_out - H_in)
        knots, T_p = _stage_profile(curves[i], H_out, -1. if heating else 1.,
                                    si.fraction, Q)
        x = np.unique(np.concatenate([[0., Q], knots]))
        T_a, T_b = hu.inlet_utility_stream.T, hu.outlet_utility_stream.T
        T_u = T_a + (T_b - T_a) * x / Q
        dT = T_u - T_p(x) if heating else T_p(x) - T_u
        span = (f'at {T_a:.2f} K' if T_a == T_b
                else f'from {T_a:.2f} to {T_b:.2f} K')
        cause = (f': its {agent.ID} {span}, as biosteam specifies it, meets '
                 f'or crosses the profile of {targets.labels[i]}')
        areas[hx.ID] = (
            _profile_area(x, dT, 1. / h[i] + 1. / h_agent, hx.ID, cause),
            hx.design_results.get('Area', 0.) / _FT2_PER_M2)
    return areas

def network_report(HXN, targets=None, **options):
    """
    Compare the heat exchanger network that a `HeatExchangerNetwork`
    facility synthesized with the targets of its process streams: energy,
    utilities, number of units, area, capital cost and total annualized cost
    (TAC) of both, side by side.

    Parameters
    ----------
    HXN : HeatExchangerNetwork
        A simulated facility, whose network is of the units' current states
        (simulate the system again after changing a unit or a stream).
    targets : PinchTargets, optional
        Targets of the facility's process streams, as
        `HeatExchangerNetwork.targets` makes them (the same units in the
        same order, the same thermodynamic basis and the units' current
        states). Defaults to ``HXN.targets()``.
    **options
        Options of `PinchTargets.summary`. The film coefficients `h` and
        `h_utilities` and the economics `interest`, `years`,
        `operating_hours` and `tea` apply to the network and the targets
        alike (the operating hours default to the system's, if it has them
        and neither `operating_hours` nor `tea` is given); the agents, `P`,
        `heat_exchanger_type` and `material` to the targets only (the
        network keeps biosteam's agents and costs).

    Returns
    -------
    dict
        'T_min_app' and 'targets_T_min_app' [K], of the facility (its
        attribute as it is now: a reused cached network keeps the one it
        was synthesized at) and of the targets; 'status'
        (``HXN.synthesis_info['status']``); 'applied', whether the network
        has a process exchanger; 'Q_hot' and 'Q_cold',
        the heat of the network's heating and cooling utilities [kJ/hr],
        process side (`HXN.actual_heat_util_load` is utility side);
        'utilities', ``{agent ID: {'duty', 'utility_duty', 'cost'}}``
        [kJ/hr, kJ/hr, USD/hr] (duty process side, a magnitude);
        'utility_cost' [USD/hr]; 'N_units', the exchangers with duty (a
        furnace included), of which 'N_process_units' and
        'N_utility_units'; 'area', the network's area like for like with
        the area target [m²], inf if an exchanger's is (see Notes);
        'area_biosteam', the area biosteam sized on terminal temperature
        differences [m²]; 'exchanger_areas',
        ``{ID: (area, area_biosteam)}`` of each exchanger with duty [m²];
        'purchase_cost' and 'installed_cost' of all the network's
        exchangers [USD]; 'annualized_capital_cost', 'operating_cost' and
        'TAC' [USD/yr], as `PinchTargets.summary` computes them; 'targets',
        `PinchTargets.summary` of the targets.

    Raises
    ------
    RuntimeError
        If the facility was not simulated.
    ValueError
        If `targets` are not those of the facility's process streams in
        their current states (enthalpies at both ends, and pressure and
        flows at the inlet), if the network's streams do not enter and
        leave it at those states, are of exchangers no longer among the
        facility's units and their auxiliaries or are not as many as the
        units list their heat utilities now (a unit or a stream changed
        since the facility's last simulation: a
        `biosteam.MultistageCompressor` simulated alone after a change of
        `pr`, say, rebuilds its intercoolers, and a unit in parallel
        simulated alone may change its number of units, ``Unit.parallel``,
        whose heat utilities biosteam lists once per unit), or for an
        invalid option.
    TypeError
        For an unknown option.

    Notes
    -----
    The network's 'area' is sized as the area target is: each exchanger
    with the film coefficients of its two sides (1 kW/m²/K by default),
    counter-current (a correction factor of 1), and integrated against the
    process duty along the temperature profiles of its streams as their
    curves in the targets model them, like for like with the area target: a
    point-load stream at its outlet temperature and a non-equilibrium end
    state (such as a vapour entering below its dew point) as its jump,
    where the stream itself may glide a little in the exchanger. A
    utility's temperature is linear in the duty between its inlet and
    outlet, and a furnace takes no area (as fuel in the targets).
    'area_biosteam' is biosteam's own sizing: heuristic overall
    coefficients (0.5 kW/m²/K, but 1 for condensing against boiling, which
    no film coefficients reproduce), a correction factor for two shells,
    the log-mean of the terminal temperature differences (which overstates
    the mean difference along a glide) and the utility duty (the process
    duty divided by the heat transfer efficiency).

    An exchanger's 'area' is inf, and so the network's 'area', with a
    RuntimeWarning naming the exchanger, where its two temperature profiles
    meet or cross. biosteam specifies a utility from the temperatures at
    the ends of its exchanger only: it takes a sensible agent to its limit
    whenever the process inlet allows it (by ``bst.HeatUtility.dT``), as
    cooling water to 324.82 K and chilled water to 300.37 K, so against a
    stream that condenses a little below that limit (a vacuum condenser of
    steam condensing at 320 K, say) the utility's line crosses the
    condensing stretch inside the exchanger: as specified, the exchanger
    cannot be built. Its 'area_biosteam' stays finite, since biosteam sizes
    on the terminal temperature differences, and so do the costs and the
    TAC, which are biosteam's. The targets keep a sensible agent's line
    inside the grand composite curve, so the area target of such streams
    is finite. The report keeps biosteam's specification rather than size
    the exchanger for more of the utility, leaving where its line stays
    inside the stream's profile: that would be another network at another
    utility cost. 'exchanger_areas' gives the areas of the other
    exchangers.

    The capital and utility costs are those of the whole network
    (`HXN.new_HXs` and `HXN.new_HX_utils`), not the facility's costs,
    which are the differences from the original exchangers (and zero for
    a network without process exchangers: the facility then keeps the
    original exchangers, to which the utility exchangers of the network,
    reported here, are equivalent).

    Only the energy targets ('Q_hot' and 'Q_cold') bound the network; the
    others are estimates, and the network may beat some of them:

    - the network serves each stream with one agent at its outlet (biosteam's
      rule: the first agent of the list that can), while the targets place
      the agents on the grand composite curve in list order (fuel last), so
      the network may use other agents than the targets, costlier or
      cheaper ones: with Linnhoff & Hindmarsh's problem at 10 K, medium
      pressure steam and chilled water where the targets need only low
      pressure steam and cooling water, but a stream heated above the
      hottest steam takes fuel over its whole utility stretch, and with
      biosteam's default prices natural gas costs less per kJ than the
      steam the targets place below the fuel (a stream of 1 kW/K heated
      from 330 to 520 K against one cooled from 400 to 300 K, at 10 K: 3.14
      USD/hr at MER against a target of 4.56 USD/hr). The utility cost
      target is the cost of biosteam's placement of its agents, not a
      minimum: neither it nor the network's utility cost bounds the other;
    - the area target is a minimum only for equal film coefficients and the
      same utilities: that problem's network takes 54.06 m² against a target
      of 59.68 m², with hotter steam and colder coolant;
    - the units target counts S - 1 per region between zeros of the
      balanced grand composite curve, as if each region's matches formed
      no loop: a network needs fewer units where a subset of streams
      balances (6 against 7 in that problem) or where one agent serves a
      stream the targets split between levels, and one more for each loop
      of its matches, such as a repeated match between two streams (two
      branches of a split stream meeting the same stream make one too) or
      two streams linked both through process matches and through a shared
      utility.

    Examples
    --------
    The liquids of the second example of `HeatExchangerNetwork` at 10 K,
    with stream splitting, so that the network reaches the energy targets:

    >>> import biosteam as bst
    >>> from hensmith import network_report
    >>> bst.main_flowsheet.set_flowsheet('network_report_example')
    >>> bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    >>> def process_stream(ID, T_in, T_out, P, **kmol_hr):
    ...     inlet = bst.Stream(ID + '_in', T=T_in, P=P, phase='l',
    ...                        units='kmol/hr', **kmol_hr)
    ...     return bst.HXutility(ID, ins=inlet, T=T_out, rigorous=False)
    >>> units = [process_stream('H1', 400., 320., 5e5, Water=60.),
    ...          process_stream('H2', 390., 330., 5e5, Ethanol=30.),
    ...          process_stream('C1', 310., 410., 5e5, Water=150.),
    ...          process_stream('H3', 360., 300., 101325., Water=200.)]
    >>> HXN = bst.HeatExchangerNetwork('HXN', T_min_app=10.,
    ...                                stream_splitting=True)
    >>> sys = bst.System.from_units('sys', units=[*units, HXN])
    >>> sys.simulate()
    >>> report = network_report(HXN)
    >>> targets = report['targets']
    >>> report['status'], report['applied']
    ('mer', True)

    The network reaches the hot utility target [kW], but serves each stream
    with one agent at its outlet: medium pressure steam and chilled water,
    where the targets need mostly low pressure steam and cooling water
    [USD/hr]:

    >>> round(report['Q_hot'] / 3600, 3), round(targets['Q_hot'] / 3600, 3)
    (102.821, 102.821)
    >>> sorted(report['utilities'])
    ['chilled_water', 'cooling_water', 'medium_pressure_steam']
    >>> round(report['utility_cost'], 3), round(targets['utility_cost'], 3)
    (5.496, 3.58)

    With hotter steam and colder coolant, its area [m²] is below the area
    target (biosteam's own sizing in between), and it needs fewer units than
    the estimate:

    >>> round(report['area'], 2), round(report['area_biosteam'], 2), round(targets['area'], 2)
    (50.62, 66.33, 70.3)
    >>> report['N_units'], targets['N_units']
    (7, 9)

    Its total annualized cost [USD/yr] is below the target's, which puts the
    area target in five shell and tubes, where the network has one:

    >>> round(report['TAC']), round(targets['TAC'])
    (62041, 82120)
    >>> targets['N_area_shells'], targets['shell_type']
    (5, 'Floating head')

    """
    if not hasattr(HXN, 'new_HXs_hot_side'):
        raise RuntimeError('simulate the heat exchanger network before '
                           'reporting on it')
    # (without the warnings of stages rebuilt since and of inlets changed
    # in flow since: _check_network raises)
    if targets is None: targets = HXN._pinch_targets()[0]
    else: _check_targets(HXN, targets)
    _check_network(HXN, targets)
    options = dict(options)
    if options.get('operating_hours') is None and options.get('tea') is None:
        hours = getattr(getattr(HXN, 'system', None), 'operating_hours', None)
        if hours is not None: options['operating_hours'] = hours
    options = targets._check_options(options)
    get = options.get
    crf, hours = _economics(get('interest'), get('years'),
                            get('operating_hours'), get('tea'))
    areas = _network_areas(
        HXN, targets, targets._film_coefficients(get('h')),
        get('h_utilities'),
        [*(get('heating_agents') or ()), *(get('cooling_agents') or ())])
    process = [hx for hx in HXN.new_HXs if hx.ID in areas]
    utility = [hx for hx in HXN.new_HX_utils if hx.ID in areas]
    hus = [hu for hx in utility for hu in hx.heat_utilities if hu.agent]
    utilities = {}
    for ID, group in bst.HeatUtility.heat_utilities_by_agent(hus).items():
        utilities[ID] = {'duty': abs(sum([hu.unit_duty for hu in group])),
                         'utility_duty': abs(sum([hu.duty for hu in group])),
                         'cost': sum([hu.cost for hu in group])}
    utility_cost = sum([hu.cost for hu in hus], 0.)
    exchangers = HXN.new_HXs + HXN.new_HX_utils
    installed = sum([hx.installed_cost for hx in exchangers], 0.)
    annualized = crf * installed
    operating = hours * utility_cost
    return {
        'T_min_app': HXN.T_min_app, 'targets_T_min_app': targets.T_min_app,
        'status': HXN.synthesis_info['status'], 'applied': bool(HXN.new_HXs),
        'Q_hot': sum([hu.unit_duty for hu in hus if hu.unit_duty > 0.], 0.),
        'Q_cold': -sum([hu.unit_duty for hu in hus if hu.unit_duty < 0.], 0.),
        'utilities': utilities, 'utility_cost': utility_cost,
        'N_units': len(process) + len(utility),
        'N_process_units': len(process), 'N_utility_units': len(utility),
        'area': sum([a for a, _ in areas.values()], 0.),
        'area_biosteam': sum([a for _, a in areas.values()], 0.),
        'exchanger_areas': areas,
        'purchase_cost': sum([hx.purchase_cost for hx in exchangers], 0.),
        'installed_cost': installed, 'annualized_capital_cost': annualized,
        'operating_cost': operating, 'TAC': annualized + operating,
        'targets': targets.summary(**options),
    }


def _format_duty(H):
    """A duty `H` [kJ/hr] for a plot label: '0', or as `_format_H` (three
    significant digits, e.g. '1.18E5')."""
    return '0' if H == 0. else _format_H(H)

def _legend(label, H):
    """`label` for the legend if a curve of heats `H` has points, else none
    (a label starting with '_' stays out of the legend)."""
    return label if H.size else '_' + label

def _sci_ticks(axis):
    """Scientific tick labels from 10⁴ on the matplotlib `axis` (six-digit
    heats run together on a narrow figure); left alone unless it has a
    plain scalar formatter."""
    from matplotlib.ticker import ScalarFormatter
    formatter = axis.get_major_formatter()
    if type(formatter) is ScalarFormatter: formatter.set_powerlimits((-3, 4))

def _reach(curves, a, b, top):
    """The top of the polylines `curves` (each an (n, 2) array of display
    points) over the stretch `a` <= x <= `b` if `top`, else their bottom;
    None if none of them runs over it."""
    ys = []
    for xy in curves:
        if len(xy) < 2: continue
        (x0, y0), (x1, y1) = xy[:-1].T, xy[1:].T
        left, right = np.minimum(x0, x1), np.maximum(x0, x1)
        keep = (right >= a) & (left <= b)
        if not keep.any(): continue
        x0, y0, x1, y1, left, right = (v[keep] for v in (x0, y0, x1, y1, left, right))
        dx = x1 - x0
        steep = dx == 0. # vertical: both ends
        slope = (y1 - y0) / np.where(steep, 1., dx)
        ys += [np.where(steep, y0, y0 + slope * (np.clip(a, left, right) - x0)),
               np.where(steep, y1, y0 + slope * (np.clip(b, left, right) - x0))]
    if not ys: return None
    ys = np.concatenate(ys)
    return ys.max() if top else ys.min()

class _Labels:
    """
    Where the annotations of `ax` added to it go, decided when drawn: at
    the final layout of the figure, which places the axes. All of them are
    placed in one pass (`_place`), kept for the renderer, the box and the
    view limits of the axes it was made for, so that drawing or measuring
    a label (as constrained layout, a legend's 'best' location and a tight
    bounding box each do) places none of them again. Each label's
    `textcoords` is an `_At` of it: a class rather than a closure, as is
    the `_Leader` of one, so that the figure can still be pickled.

    The labels keep off the polylines `curves` ((x, y) in data, their
    points that are not finite left out).
    """
    def __init__(self, ax, curves=()):
        self.ax, self.labels, self.options = ax, [], []
        self.curves = []
        for x, y in curves:
            xy = np.column_stack((x, y)).astype(float).reshape(-1, 2)
            self.curves.append(xy[np.isfinite(xy).all(axis=1)])
        self._key = self._spots = None

    def __getstate__(self):
        return {**self.__dict__, '_key': None, '_spots': None}

    def add(self, label, **options):
        """Place the annotation `label` (after those added before it)."""
        label.set_anncoords(_At(self, len(self.labels)))
        self.labels.append(label)
        self.options.append(options)

    def spots(self, renderer):
        """The display (x, y) of the text of each label, and the ends of
        its leader (or None)."""
        ax = self.ax
        key = (id(renderer), tuple(ax.viewLim.bounds), tuple(ax.bbox.bounds),
               tuple([(label.get_text(), label.get_fontsize())
                      for label in self.labels]))
        if key != self._key:
            self._spots, self._key = self._place(renderer), key
        return self._spots

    def _sizes(self, renderer):
        """The display (width, height) of the text of each label (not
        where it is, which is not placed yet)."""
        from matplotlib.text import Text
        boxes = [Text.get_window_extent(label, renderer) for label in self.labels]
        return [(box.width, box.height) for box in boxes]

    def _curves(self):
        """The `curves` in display."""
        return [self.ax.transData.transform(xy) for xy in self.curves]

class _At:
    """The `textcoords` of the label `index` of the `_Labels` `labels`."""
    def __init__(self, labels, index):
        self.labels, self.index = labels, index

    def __call__(self, renderer):
        from matplotlib.transforms import Affine2D
        return Affine2D().translate(*self.labels.spots(renderer)[self.index][0])

class _Under:
    """
    The y `xycoords` of a note under the axes `ax`: the display box of the
    axes and of the visible decorations of its x axis (tick labels, label
    and offset text) as drawn, so that the note keeps below them at any
    size (a class, so that the figure can still be pickled).
    """
    def __init__(self, ax):
        self.ax = ax

    def __call__(self, renderer):
        from matplotlib.transforms import Bbox
        ax = self.ax
        box = ax.xaxis.get_tightbbox(renderer)
        return ax.bbox if box is None else Bbox.union([ax.bbox, box])

class _Leader(Patch):
    """
    A thin line from the anchor of the label `index` of the `_Labels`
    `labels` to its text, drawn where the label sits away from its anchor
    (its path is that of the last draw, empty without a line).
    """
    def __init__(self, labels, index, **kwargs):
        from matplotlib.transforms import IdentityTransform
        super().__init__(fill=False, transform=IdentityTransform(), **kwargs)
        self.set_in_layout(False)
        self.labels, self.index = labels, index
        self._path = Path(np.zeros((0, 2)))

    def get_path(self):
        return self._path

    def path(self, renderer):
        """The path of the leader where the labels are placed for the
        `renderer` (in display; empty without a leader)."""
        ends = self.labels.spots(renderer)[self.index][1]
        return Path(np.zeros((0, 2)) if ends is None else np.array(ends))

    def draw(self, renderer):
        if not self.get_visible(): return
        self._path = self.path(renderer)
        if len(self._path.vertices): super().draw(renderer)

class _Stacked(_Labels):
    """
    Labels of the balanced composite curves (the `curves`), added by
    `add(label, offset)` in turn: each heating agent's by ascending level,
    then the fuel's, then each cooling agent's by descending level.

    A label sits `dx`, `dy` points (the `offset`) off its anchor: above
    left of it (right- and bottom-aligned) if `dy` > 0, a heating one, else
    below right of it (left- and top-aligned), a cooling one. One that
    would run past the left or right edge of the axes is moved to keep 2
    points inside them, and each is kept `abs(dy)` points above (below)
    the curves under its span. The labels of a side are then stacked in a
    column: each at least as high (low) as the one before it, which keeps
    the order of the levels, and moved on, up (down), past each label
    placed before it whose box it would overlap or come within 2 points
    of. Where that would run past the top (bottom) of the axes, they are
    set in rows instead, from the other end: each where it sits, moved
    sideways, left (right), past the labels of the levels beyond its own
    (whose rows are beyond its own, so that its leader crosses none of
    them), if that keeps the order and all of them inside the axes; else
    in the column, moved sideways past the label they would run into at
    the edge. One left more than 1.5 times its height from its anchor gets
    a leader to it.
    """
    def add(self, label, offset, leader=None):
        """Place the annotation `label` `offset` points off its anchor,
        with a leader (a `_Leader` added to the axes, with the gid
        `leader`) if it is away from it."""
        super().add(label, offset=offset)
        if leader is not None:
            self.ax.add_patch(_Leader(self, len(self.labels) - 1,
                                      color=label.get_color(), lw=0.6,
                                      gid=leader))

    def make_room(self, x=0.):
        """Extend the data limits of the axes in y (at `x`), so that the
        labels fit in their columns at the layout of the figure now
        (where a renderer is at hand)."""
        ax = self.ax
        renderer = getattr(ax.figure.canvas, 'get_renderer', lambda: None)()
        if renderer is None or not np.isfinite(ax.dataLim.intervaly).all(): return
        for _ in range(4): # the scale changes with the limits
            y0, y1 = ax.get_ylim() # autoscaled first
            below, above = self._place(renderer, bound=False)
            if below <= 0. and above <= 0.: break
            scale = (y1 - y0) / ax.bbox.height # data per pixel
            lo, hi = ax.dataLim.intervaly
            ax.update_datalim([(x, lo - 1.05 * below * scale),
                               (x, hi + 1.05 * above * scale)], updatex=False)
            ax.autoscale_view(scalex=False)

    def _place(self, renderer, bound=True):
        """The `spots`, or how far (in pixels) the columns of labels run
        past the bottom and the top of the axes if not `bound` by them."""
        ax = self.ax
        pt = ax.figure.dpi / 72. # pixels per point
        gap = 2. * pt
        frame = ax.bbox
        lo, hi = frame.x0 + gap, frame.x1 - gap
        bottom, top = frame.y0 + gap, frame.y1 - gap
        if not bound: bottom, top = -math.inf, math.inf
        curves = self._curves()
        sizes = self._sizes(renderer)
        anchors = [ax.transData.transform(label.xy) for label in self.labels]
        offsets = [options['offset'] for options in self.options]
        def clear(i, u, v):
            """`v` kept abs(dy) points off the curves under the span of
            label `i` from `u`."""
            (w, h), dy = sizes[i], offsets[i][1]
            reach = _reach(curves, u, u + w, dy > 0)
            if reach is None: return v
            if dy > 0: return max(v, reach + dy * pt)
            return min(v, reach + dy * pt - h)
        def leader(i, u, v):
            """The ends of a leader from the anchor of label `i` to the
            nearest point of its box from `u`, `v`, if that is more than
            1.5 times its height away, else None."""
            (w, h), (x, y) = sizes[i], anchors[i]
            near = (min(max(x, u), u + w), min(max(y, v), v + h))
            span = math.hypot(near[0] - x, near[1] - y)
            if span <= 1.5 * h: return None
            trim = 1.5 * pt / span
            return [(x + trim * (near[0] - x), y + trim * (near[1] - y)),
                    (near[0] - trim * (near[0] - x), near[1] - trim * (near[1] - y))]
        def stack(order, up, column, placed):
            """The lower left corners of the labels `order` of a side (up
            for heating), placed past those `placed` and each other (and,
            in rows, the leaders before them): in a column, else in rows;
            and whether that kept them inside the axes and in order,
            without moving any the other way."""
            placed = list(placed)
            corners = {}
            last = None
            clean = True
            for i in order:
                (w, h), (dx, dy) = sizes[i], offsets[i]
                x, y = anchors[i] + (dx * pt, dy * pt)
                u, v = (x - w, y) if up else (x, y - h)
                u = max(min(u, hi - w), lo) # inside the axes, left to right
                v = clear(i, u, v)
                if column and last is not None: # in order
                    v = max(v, last[0]) if up else min(v, last[1] - h)
                for _ in range(2 * len(placed) + 1):
                    hits = [box for box in placed
                            if box[0] < u + w + gap and u < box[2] + gap
                            and box[1] < v + h + gap and v < box[3] + gap]
                    if not hits: break
                    # past the nearest one, on (up or down) or sideways
                    # (left or right)
                    if up:
                        on = min(hits, key=lambda box: box[1])[3] + gap
                        side = max(hits, key=lambda box: box[0])[0] - gap - w
                        moves = [('v', on, on + h <= top), ('u', side, side >= lo)]
                    else:
                        on = max(hits, key=lambda box: box[3])[1] - gap - h
                        side = min(hits, key=lambda box: box[2])[2] + gap
                        moves = [('v', on, on >= bottom), ('u', side, side + w <= hi)]
                    if not column: moves.reverse()
                    for n, (move, value, fits) in enumerate(moves):
                        if fits: break
                    else:
                        clean = False
                        break
                    if n: clean = False
                    if move == 'v': v = value
                    else: u, v = value, clear(i, value, v)
                if not bottom <= v <= top - h:
                    clean = False
                    v = max(min(v, top - h), bottom) # inside the axes
                if not column and last is not None and (
                        v > last[0] + 0.5 * h if up else v < last[0] - 0.5 * h):
                    clean = False # out of order
                corners[i] = (u, v)
                placed.append((u, v, u + w, v + h))
                ends = None if column else leader(i, u, v)
                if ends is not None:
                    (x0, y0), (x1, y1) = ends
                    placed.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
                last = (v, v + h)
            return corners, clean
        corners = {}
        placed = []
        for up in (True, False):
            order = [i for i, (dx, dy) in enumerate(offsets) if (dy > 0) == up]
            if not order: continue
            side, clean = stack(order, up, True, placed)
            if not clean and bound:
                rows, clean = stack(order[::-1], up, False, placed)
                if clean: side = rows
            corners.update(side)
            placed += [(u, v, u + sizes[i][0], v + sizes[i][1])
                       for i, (u, v) in side.items()]
        if not bound:
            return (max([frame.y0 + gap - v for u, v in corners.values()], default=0.),
                    max([v + sizes[i][1] - frame.y1 + gap
                         for i, (u, v) in corners.items()], default=0.))
        spots = []
        for i, (w, h) in enumerate(sizes):
            u, v = corners[i]
            spots.append(((u + w, v) if offsets[i][1] > 0 else (u, v + h),
                          leader(i, u, v)))
        return spots

class _Beside(_Labels):
    """
    Labels of the steps of the grand composite curve, added by
    `add(label, heating)` in turn, each anchored at the end of its step.

    A label sits beside the end, 3 points right or left of it and 2 points
    above or below its level: on the side with more room in the axes,
    first, toward the pinch (below a heating level, above a cooling one)
    right of the end and beyond the level left of it, then the other way;
    then likewise on the other side. It takes the first of these whose box
    is inside the axes and crosses none of the `curves` (the grand
    composite curve, the first, and the steps) nor a label placed before
    it; else the nearest such box on along its level, away from the end
    (by steps of its height); else the first of the four inside the axes
    that keeps off the grand composite curve, else the first inside the
    axes, else the first.
    """
    def add(self, label, heating):
        """Place the annotation `label` (left- and bottom-aligned) of a
        `heating` step or not."""
        super().add(label, heating=heating)

    def _place(self, renderer):
        from matplotlib.transforms import Bbox
        ax = self.ax
        pt = ax.figure.dpi / 72.
        frame = ax.bbox
        gcc, *steps = [Path(xy.reshape(-1, 2)) for xy in self._curves()]
        placed = []
        spots = []
        for label, options, (w, h) in zip(self.labels, self.options,
                                          self._sizes(renderer)):
            x, y = ax.transData.transform(label.xy)
            toward = -1. if options['heating'] else 1. # the pinch
            right = frame.x1 - x >= x - frame.x0
            sides = [(side, vertical) for side in ((1., -1.) if right else (-1., 1.))
                     for vertical in ((toward, -toward) if side > 0 else (-toward, toward))]
            def score(side, vertical, slide=0.):
                """Whether the box `slide` pixels on from its place beside
                the end is outside the axes, on the curve, and on the
                steps or a label placed before it; and the box."""
                u = x + 3. * pt + slide if side > 0 else x - 3. * pt - w - slide
                v = y + 2. * pt if vertical > 0 else y - 2. * pt - h
                box = Bbox([[u, v], [u + w, v + h]])
                inside = (frame.x0 <= u and u + w <= frame.x1
                          and frame.y0 <= v and v + h <= frame.y1)
                return ((not inside, gcc.intersects_bbox(box, filled=False),
                         any(p.intersects_bbox(box, filled=False) for p in steps)
                         or any(box.overlaps(other) for other in placed)), box)
            candidates = [score(*side) for side in sides]
            if all(any(flags) for flags, box in candidates):
                # none clear beside the end: the nearest clear one on along
                # the level, away from the end
                slides = [score(*side, slide=k * h) for k in range(1, 41)
                          for side in sides]
                clear = [item for item in slides if not any(item[0])]
                candidates = clear[:1] + candidates
            box = min(candidates, key=lambda item: item[0])[1]
            placed.append(box)
            spots.append(((box.x0, box.y0), None))
        return spots

class _ClearLegend(Legend):
    """
    A legend (of loc 'best') at matplotlib's 'best' location in its axes
    among the spots that cover none of their texts (the labels), decided
    when drawn, as the labels are placed: of the spots matplotlib tries in
    the axes, the first, in its order, of those clear of the texts that
    covers the least of the rest of the axes, by matplotlib's count (of
    the points of their lines, patches and collections in it, of their
    crossings of it, and of the rectangles, such as a shaded band, that it
    overlaps; the leaders where they are drawn), which is matplotlib's
    location wherever that is clear of the texts. Else, where each of them
    covers a text, it goes above the axes (and any title of them),
    left-aligned with them, where a constrained layout or a tight bounding
    box makes room for it.
    """
    # the anchors of the locations 1 to 10 ('upper right' to 'center')
    _anchors = ('NE', 'NW', 'SW', 'SE', 'E', 'W', 'E', 'S', 'N', 'C')

    def _find_best_position(self, width, height, renderer):
        # matplotlib's hook of the 'best' location (which the legend would
        # fall back to without it): the display (x, y) of the lower left
        # corner of the legend of `width` and `height`
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Rectangle
        from matplotlib.text import Text
        from matplotlib.transforms import Bbox
        ax = self.parent
        texts = [text.get_window_extent(renderer) for text in ax.texts
                 if text.get_visible() and text.get_text()]
        paths = [line.get_transform().transform_path(line.get_path())
                 for line in ax.lines]
        boxes, points = [], []
        for patch in ax.patches:
            if isinstance(patch, _Leader):
                paths.append(patch.path(renderer))
            elif isinstance(patch, Rectangle):
                boxes.append(patch.get_bbox().transformed(patch.get_data_transform()))
            else:
                paths.append(patch.get_transform().transform_path(patch.get_path()))
        for collection in ax.collections:
            if isinstance(collection, PolyCollection):
                transform = collection.get_transform()
                paths += [transform.transform_path(path)
                          for path in collection.get_paths()]
            elif len(collection.get_offsets()):
                points.append(collection.get_offset_transform().transform(
                    collection.get_offsets()))
        points = np.concatenate(points) if points else np.zeros((0, 2))
        pad = self.borderaxespad * renderer.points_to_pixels(
            self.prop.get_size_in_points())
        frame = self.get_bbox_to_anchor()
        size = Bbox.from_bounds(0., 0., width, height)
        best = None
        for anchor in self._anchors:
            box = size.anchored(anchor, container=frame.padded(-pad))
            if any(box.overlaps(text) for text in texts): continue
            badness = (sum(box.count_contains(path.vertices)
                           + path.intersects_bbox(box, filled=False) for path in paths)
                       + box.count_contains(points) + box.count_overlaps(boxes))
            if best is None or badness < best[0]: best = (badness, box.p0)
            if not badness: break
        if best is not None: return best[1]
        # the titles are the texts of the axes besides `texts`
        titles = [title.get_window_extent(renderer).y1 for title in ax.get_children()
                  if isinstance(title, Text) and title not in ax.texts
                  and title.get_visible() and title.get_text()]
        return frame.x0, max([frame.y1, *titles]) + pad

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
        Utilities of the balanced curves, as
        `PinchTargets.balanced_composite_curves` takes them (those of
        `targets`), with kind 'balanced' only. Defaults to
        `PinchTargets.utility_targets` with the default agents.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure (with a constrained layout, which
        keeps the axis labels and a note under the x axis in it) is created
        if not given.
    file : str, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution used when saving. Defaults to 300.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        For another `kind`, for `utilities` with kind 'real' or 'shifted'
        (whose curves take none), or for ``utilities=False`` (the balanced
        curves are those of their utilities: kind 'real' draws the
        composites without them).

    Notes
    -----
    Temperatures are shown in degC. Artists carry stable gids: 'Hot
    composite' and 'Cold composite' (or 'Hot balanced composite' and 'Cold
    balanced composite'), 'Heat recovered', 'Hot utility', 'Cold utility',
    'Pinch', 'Fuel', 'Utility:<agent ID>' and 'Leader:<agent ID>'. The
    pinch is marked where the shifted composites touch: not for a threshold
    problem whose composites end apart, nor without streams on one side
    (whose empty composite stays out of the legend). Fuel, at infinite
    temperature on the balanced curves, is labeled rather than drawn,
    above the hot end of the cold balanced composite (at the top of both),
    whose top its heat closes, and so is an unserved remainder, in a note
    under the x axis ('Utility:unserved', clear of the curves, of their
    labels and of the legend): the cold balanced composite starts at the
    unserved cooling, and the hot one ends short of the cold one by the
    unserved heating. A label of a latent agent sits by
    its flat, above left of the warm end of a heating one and below right
    of the cold end of a cooling one, and a sensible agent's off the middle
    of its line, above left for heating and below right for cooling. When
    drawn (at the final layout of the figure, all of them in one pass), a
    label that would run past the left or right edge of the axes is moved
    just inside them, and off the curves. The labels of each side keep the order of the levels: each one
    that would run into the label of a lower heating (higher cooling) agent
    is moved on past it, up for heating and down for cooling (the fuel's
    past those of the heating agents), in a column for which the limits of
    the y axis leave room (as the axes are laid out when drawn on). Where
    the column would still run past the top (bottom) of the axes, the
    labels go in rows instead, each beside its flat and moved sideways past
    the labels of the levels beyond its own; or, if those would not keep
    the order inside the axes, those that would run past the edge are moved
    sideways past the label in their way. Labels can then still overlap,
    on axes too small for them all (many agents on small axes without a
    layout engine, say, or on short axes that a legend above them leaves
    shorter): a larger figure, or one with ``layout='constrained'``, gives
    them room. A label left more than 1.5 times its height away from its
    anchor gets a thin leader from it ('Leader:<agent ID>', an empty path
    otherwise), and each label a halo, so that a leader passing behind it
    strikes none of its letters. The legend, of any kind, covers none of
    the labels: when drawn, it goes to the spot of matplotlib's 'best'
    location in the axes that covers the least of the curves, the leaders
    and the band among those clear of the labels (matplotlib's own spot
    wherever that is clear of them: no fixed corner is clear of the
    composites of every problem); or, where each of those spots would cover
    a label (on small axes, say), above the axes and any title of them,
    left-aligned with them, where a constrained layout or a tight bounding
    box makes room for it.

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
    if utilities is not None and kind != 'balanced':
        raise ValueError(f'the {kind} composite curves take no utilities '
                         "(those of kind='balanced')")
    if utilities is False:
        raise ValueError('the balanced composite curves are those of their '
                         "utilities, not False (kind='real' draws the "
                         'composites without them)')
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), layout='constrained')
    else:
        fig = ax.figure
    hot_color, cold_color = '#d62728', '#2e6db4'
    text_kwargs = dict(fontsize=8)
    if kind == 'balanced':
        if utilities is None: utilities = targets.utility_targets()
        hot_T, hot_H, cold_T, cold_H = targets.balanced_composite_curves(utilities)
        finite = np.isfinite(hot_T)
        hot_T, hot_H = hot_T[finite], hot_H[finite]
        # a halo, so that a leader passing behind a label strikes none of
        # its letters
        from matplotlib.patheffects import withStroke
        label_kwargs = dict(text_kwargs, path_effects=[
            withStroke(linewidth=2.5, foreground=ax.get_facecolor())])
        ax.plot(hot_H, hot_T - 273.15, color=hot_color, lw=2,
                label=_legend('hot balanced composite', hot_H),
                gid='Hot balanced composite')
        ax.plot(cold_H, cold_T - 273.15, color=cold_color, lw=2,
                label=_legend('cold balanced composite', cold_H),
                gid='Cold balanced composite')
        labels = [] # (annotation, offset, ID) of each agent's but fuel's
        fuels = []
        unserved = []
        for u in utilities:
            if u.agent is None:
                unserved.append(f"unserved {'heating' if u.heating else 'cooling'} "
                                f'{_format_duty(u.duty)} kJ/hr (not drawn)')
                continue
            if u.agent.isfuel:
                fuels.append(u)
                continue
            label = f'{u.ID} {_format_duty(u.duty)} kJ/hr'
            T, H = (hot_T, hot_H) if u.heating else (cold_T, cold_H)
            if u.agent.T_limit is None:
                # above a heating flat, ending at its warm end (left of it the
                # hot curve is below the flat), or below a cooling flat from
                # its cold end (right of it both curves are above the flat)
                T_u = u.agent.T
                if u.heating:
                    i = np.searchsorted(T, T_u + 1e-6) - 1
                    offset, ha, va = (-2, 2), 'right', 'bottom'
                else:
                    i = np.searchsorted(T, T_u - 1e-6)
                    offset, ha, va = (2, -2), 'left', 'top'
                x = H[min(max(i, 0), H.size - 1)]
            else:
                # at the middle of a sensible agent's line, off the rising
                # composite: above left of it for heating, below right of it
                # for cooling (the other composite is on the far side)
                T_u = 0.5 * (u.T_utility_in + u.T_utility_out)
                x = np.interp(T_u, T, H)
                if u.heating:
                    offset, ha, va = (-6, 4), 'right', 'bottom'
                else:
                    offset, ha, va = (6, -4), 'left', 'top'
            annotation = ax.annotate(
                label, xy=(x, T_u - 273.15), xytext=(0, 0), ha=ha, va=va,
                color=hot_color if u.heating else cold_color,
                gid='Utility:' + u.ID, **label_kwargs)
            labels.append((annotation, offset, u.ID))
        # each moved inside the axes if it would run out of them, off the
        # curves, in the order of the levels, and on past the labels placed
        # before it that it would run into (up for heating, down for
        # cooling, or sideways at the edge of the axes); with a leader to
        # its anchor if that leaves it away from it
        placement = _Stacked(ax, [(hot_H, hot_T - 273.15), (cold_H, cold_T - 273.15)])
        heating = sorted([item for item in labels if item[1][1] > 0],
                         key=lambda item: item[0].xy[1])
        cooling = sorted([item for item in labels if item[1][1] < 0],
                         key=lambda item: -item[0].xy[1])
        for annotation, offset, ID in heating:
            placement.add(annotation, offset, leader='Leader:' + ID)
        # the fuel's heat closes the cold curve: above its hot end (at the
        # top of both curves), over the labels of the heating agents it
        # would overlap there, and of any fuel before it
        curves_T = np.concatenate((hot_T, cold_T))
        if curves_T.size:
            T_top, span = curves_T.max(), np.ptp(curves_T)
            x = max(H[-1] for H in (hot_H, cold_H) if H.size)
        else:
            T_top, span, x = targets.grid['Ts'][0], 0., 0.
        # headroom for two labels where a heating agent's is near the top
        stacked = bool(fuels) and any(T_top - 273.15 - a.xy[1] <= 0.15 * span
                                      for a, *_ in heating)
        for u in fuels:
            annotation = ax.annotate(
                f'{u.ID} (furnace) {_format_duty(u.duty)} kJ/hr',
                xy=(x, T_top - 273.15), xytext=(0, 0), ha='right', va='bottom',
                color=hot_color, gid='Fuel', **label_kwargs)
            placement.add(annotation, (-2, 4), leader='Leader:' + u.ID)
        for annotation, offset, ID in cooling:
            placement.add(annotation, offset, leader='Leader:' + ID)
        if unserved:
            # the curves end apart: say why, in a note under the x axis,
            # clear of the curves, of their labels and of the legend
            ax.annotate('\n'.join(unserved), xy=(0., 0.),
                        xycoords=('axes fraction', _Under(ax)), xytext=(0, -4),
                        textcoords='offset points', ha='left', va='top',
                        color='0.45', gid='Utility:unserved', **text_kwargs)
        # headroom for the labels above the top of the curves, and more
        # where they need it to stack in columns
        ax.margins(y=0.16 if stacked else 0.1)
        placement.make_room(x)
        ax.set_ylabel('T [°C]')
    else:
        shifted = kind == 'shifted'
        hot_T, hot_H, cold_T, cold_H = targets.composite_curves(shifted)
        if cold_H.size and hot_H.size and cold_H[0] < hot_H[-1]:
            ax.axvspan(cold_H[0], hot_H[-1], color='0.92', lw=0, zorder=0,
                       label='heat recovered', gid='Heat recovered')
        ax.plot(hot_H, hot_T - 273.15, color=hot_color, lw=2,
                label=_legend('hot composite', hot_H), gid='Hot composite')
        ax.plot(cold_H, cold_T - 273.15, color=cold_color, lw=2,
                label=_legend('cold composite', cold_H), gid='Cold composite')
        T_all = np.concatenate([hot_T, cold_T]) - 273.15
        H_hot_end = hot_H[-1] if hot_H.size else 0.
        H_cold_end = cold_H[-1] if cold_H.size else targets.Q_cold
        ax.margins(y=0.12) # headroom for the utility labels
        # above the top of the curves, ending at the right one of their hot
        # ends, and below their bottom from the middle of the cold utility:
        # inside the axes, clear of the curves by a few points
        ax.annotate(f'hot utility {_format_duty(targets.Q_hot)} kJ/hr',
                    xy=(max(H_hot_end, H_cold_end), T_all.max()), xytext=(-2, 4),
                    textcoords='offset points', ha='right', va='bottom',
                    gid='Hot utility', **text_kwargs)
        ax.annotate(f'cold utility {_format_duty(targets.Q_cold)} kJ/hr',
                    xy=(0.5 * targets.Q_cold, T_all.min()), xytext=(0, -4),
                    textcoords='offset points', ha='left', va='top',
                    gid='Cold utility', **text_kwargs)
        pinch_T = targets.pinch_T
        tol = 1e-9 * pinch_T
        if shifted and all(T.size and T.min() - tol <= pinch_T <= T.max() + tol
                           for T in (hot_T, cold_T)):
            # where the composites touch: the cold utility plus the cold
            # streams' heat below the pinch, if both composites reach the
            # pinch temperature (not for streams of one side, nor for a
            # threshold problem whose composites end apart)
            below = targets._side_heat()[1]
            x = targets.Q_cold + below[~targets.is_hot].sum()
            ax.plot([x], [pinch_T - 273.15], 'o', mfc='w', mec='k',
                    ms=7, zorder=4, label='pinch', gid='Pinch')
        ax.set_ylabel('shifted T [°C]' if shifted else 'T [°C]')
    _sci_ticks(ax.xaxis)
    ax.set_xlabel('H [kJ·h$^{-1}$]')
    # at matplotlib's 'best' location among the spots clear of the labels,
    # or above the axes if none is (added as `Axes.legend` adds one)
    ax.legend_ = _ClearLegend(ax, *ax.get_legend_handles_labels(), loc='best',
                              fontsize=8)
    ax.legend_._remove_method = ax._remove_legend
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
        Axes to draw on; a new figure (with a constrained layout, which
        keeps the axis labels in it) is created if not given.
    file : str, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution used when saving. Defaults to 300.

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
    of the grid and an unserved remainder in gray. Each label sits beside
    the end of its segment, placed when drawn: on the side with more room
    in the axes, right of the end toward the pinch (below the level for
    heating, above it for cooling) or left of it beyond the level (above it
    for heating, below it for cooling, where the heat cascaded is at least
    the end of the step), if that keeps it inside the axes and off the
    curve, the steps and the labels placed before it; else at the first
    other side of the end and of the level that does (a pocket of the
    curve, or a curve that leaves the end at a shallow angle, can run
    toward the pinch); else on along its level, away from the end, where
    it first does. Artists carry stable gids: 'GCC',
    'Pinch', 'Utility:<agent ID>' (the segments, 'Utility:unserved' for a
    remainder) and 'Label:<agent ID>'.

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
        fig, ax = plt.subplots(figsize=(5, 4.2), layout='constrained')
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
        ends = [] # (ID, x, y, heating, color) at the end of each step
        segments = [] # (x, y) of the steps and the dotted lines between
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
                    segments.append(([x, x], [previous, y]))
                ax.plot([x, x + u.duty], [y, y], lw=1.5, gid='Utility:' + ID,
                        **style)
                segments.append(([x, x + u.duty], [y, y]))
                x += u.duty
                previous = y
                ends.append((ID, x, y, u.heating, style['color']))
        # each label beside the end of its step, placed when drawn: on the
        # side with more room in the axes, toward the pinch right of the
        # end, beyond the level left of it (where the curve never is: the
        # heat cascaded there is at least the end of the step), or where
        # else it keeps inside the axes and off the curve, the steps and
        # the labels before it (a pocket of the curve, or a curve that
        # leaves the end of a step at a shallow angle, can run toward the
        # pinch)
        placement = _Beside(ax, [(H, T - 273.15), *segments])
        for ID, x, y, is_heating, color in ends:
            label = ax.annotate(ID, xy=(x, y), xytext=(0, 0), ha='left',
                                va='bottom', fontsize=7, color=color,
                                gid='Label:' + ID)
            placement.add(label, is_heating)
        ax.margins(y=0.08) # headroom for a label beyond the outermost level
    _sci_ticks(ax.xaxis)
    ax.set_xlabel('heat cascaded [kJ·h$^{-1}$]')
    ax.set_ylabel('shifted T [°C]')
    ax.grid(alpha=0.3)
    if file: fig.savefig(file, dpi=dpi, bbox_inches='tight')
    return fig, ax

def plot_supertargets(sweep, ax=None, file=None, dpi=300):
    """
    Draw the total annualized cost (TAC) of a sweep of the minimum approach
    temperature, its annualized capital and operating parts, and the least
    TAC.

    Parameters
    ----------
    sweep : Supertargets
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure (with a constrained layout, which
        keeps the axis labels in it) is created if not given.
    file : str, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution used when saving. Defaults to 300.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Artists carry stable gids: 'TAC', 'Capital' (the annualized capital
    cost), 'Operating', 'Optimum' and 'Nonfinite'. The optimum is labeled
    '(TAC constant)' if the TAC is the same over the sweep
    (`Supertargets.constant`), '(TAC constant where finite)' if it is the
    same at each temperature with a finite TAC (at least two), else '(at
    the bound)' if it is at an end of the sweep (`Supertargets.at_bound`),
    and is not marked without a finite TAC. The lines leave out the
    temperatures without a finite TAC (an unserved utility or an infinite
    area): each stretch of them is shaded ('Nonfinite', halfway to its
    finite neighbors), so the axis keeps the whole sweep.

    Examples
    --------
    The streams of `PinchTargets`:

    >>> import biosteam as bst, thermosteam as tmo
    >>> from hensmith import PinchTargets, supertarget, plot_supertargets
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
    >>> sweep = supertarget(targets)
    >>> sweep
    <Supertargets: 64 points, T_min_app 1-40 K, optimum 7.816 K (TAC 2.490e+04 USD/yr)>
    >>> fig, ax = plot_supertargets(sweep)
    >>> [line.get_gid() for line in ax.get_lines() if line.get_gid()]
    ['TAC', 'Capital', 'Operating', 'Optimum']
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    else:
        fig = ax.figure
    T = sweep['T_min_app']
    TAC = sweep['TAC']
    ax.plot(T, TAC, color='k', lw=2, label='total annualized cost',
            gid='TAC')
    ax.plot(T, sweep['annualized_capital_cost'], color='#2e6db4', lw=1.5,
            label='annualized capital cost', gid='Capital')
    ax.plot(T, sweep['operating_cost'], color='#d62728', lw=1.5,
            label='operating cost', gid='Operating')
    optimum = sweep.optimum
    if math.isfinite(optimum['TAC']):
        if sweep.constant: bound = ' (TAC constant)'
        elif _constant_TAC(sweep.records, where_finite=True):
            bound = ' (TAC constant where finite)'
        elif sweep.at_bound: bound = ' (at the bound)'
        else: bound = ''
        ax.axvline(optimum['T_min_app'], color='k', ls='--', lw=0.8)
        ax.plot([optimum['T_min_app']], [optimum['TAC']], 'o', mfc='w',
                mec='k', ms=7, zorder=4,
                label=f"optimum {optimum['T_min_app']:.3g} K{bound}",
                gid='Optimum')
    bad = ~np.isfinite(TAC)
    if bad.any():
        # the stretches without a finite TAC, which the lines leave out:
        # shaded behind them, halfway to their finite neighbors
        n = T.size
        starts = np.flatnonzero(bad & ~np.r_[False, bad[:-1]])
        ends = np.flatnonzero(bad & ~np.r_[bad[1:], False])
        for k, (a, b) in enumerate(zip(starts, ends)):
            left = T[a] if a == 0 else (T[a - 1] + T[a]) / 2
            right = T[b] if b == n - 1 else (T[b] + T[b + 1]) / 2
            ax.axvspan(left, right, color='0.85', lw=0, zorder=0,
                       label='no finite TAC' if k == 0 else '_nolegend_',
                       gid='Nonfinite')
    ax.set_xlabel('$T_{min,app}$ [K]')
    ax.set_ylabel('cost [USD·yr$^{-1}$]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    if file: fig.savefig(file, dpi=dpi, bbox_inches='tight')
    return fig, ax
