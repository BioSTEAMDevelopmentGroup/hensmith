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
(`PinchTargets.utility_targets`), the balanced composite curves, and the
area, units, capital cost and total annualized cost targets
(`PinchTargets.summary`); `supertarget` sweeps the minimum approach
temperature for the least total annualized cost, and `network_report`
compares a synthesized network with its targets. `plot_composite_curves`,
`plot_grand_composite_curve` and `plot_supertargets` draw them.
"""
import math
from collections import namedtuple
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral, Real
from warnings import warn
import numpy as np
import biosteam as bst
from ._curves import _T_EQ, _copy
from ._planner import _approach_contributions, _given_contributions
from .hxn_synthesis import (_problem_table, _pinch_cut, _format_H,
    _arrange_heat_utilities, _process_streams, _explicit_contributions,
    _contribution_key_level, _names_unit, _owner_chains, _unit_name)

__all__ = ('PinchTargets', 'supertarget', 'Supertargets', 'network_report',
           'plot_composite_curves', 'plot_grand_composite_curve',
           'plot_supertargets')

#: Kinds of the constraint positions of the grand composite curve (see
#: `_positions`): the heat arriving at a grid temperature (before its point
#: loads), the heat leaving it (after them), and the heat just above and
#: just below a utility level strictly between two grid temperatures.
_ARRIVING, _LEAVING, _ABOVE, _BELOW = 0, 1, 2, 3

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
    Process temperatures of the stretch it serves [K]: the real temperatures
    of the streams it serves there, the coldest and the hottest (heating) or
    the hottest and the coldest (cooling). biosteam prices the utility from
    them, a sensible agent from the inlet that keeps its line ``dT`` from
    the process (see Notes of `PinchTargets`).
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

def _utility_films(utilities, h_utilities, agents=()):
    """
    The film coefficient [kW/m²/K] of each of `utilities` from
    `h_utilities`: None (`_H_DEFAULT`), a number, or a mapping from agent
    IDs or agents to numbers (`_H_DEFAULT` for an agent it leaves out). A
    key must name an agent of `utilities`, of `agents` or of biosteam's
    default lists.
    """
    return _agent_films([u.agent for u in utilities], h_utilities, agents)

def _agent_films(of, h_utilities, agents=()):
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
        raise ValueError('h_utilities keys that name no agent: '
                         f"{', '.join(map(repr, unknown))}")
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
        warn('vertical heat transfer at a non-positive driving force between '
             f'H = {x[k]:.6g} and {y[k]:.6g} kJ/hr', RuntimeWarning)
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
    pinch, or a utility pinch). Its `T_in` and `T_out` are the real
    temperatures of the streams it serves there (the cold streams with heat
    above the pinch, or the hot streams with heat below it): the coldest and
    the hottest for heating, the hottest and the coldest for cooling. They
    are the stretch's ends without approach contributions (where a stream
    reaches them), and `T_out` is within the agent's limit. biosteam prices
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
    temperature. `balanced_grand_composite_curve` is the cascade net of the
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
    a bound: it assumes one component per region, and a balanced subset of
    streams needs fewer units.

    *Capital and total annualized cost.* `capital_cost_target` divides the
    area target into as many shells of equal area as the units target (less
    one furnace for all fuel) [Linnhoff90]_ and costs each with biosteam's
    correlations at one pressure (biosteam takes each exchanger's highest
    inlet pressure: its pressure factor is up to about 12 % higher for
    high pressure steam heaters): a double pipe below 150 ft² (at
    biosteam's default CE of 567.5 and 1 atm, its base purchase cost of
    about 3,211 USD just below 150 ft² jumps to 20,516 USD for a floating
    head at 150 ft², and the installed cost from about 8,055 to 64,658 USD;
    the floating head correlation, not monotone below about 126 ft², is not
    used there), a shell and tube of any area above. The furnace is costed
    on the fuel's utility duty. `summary` adds the total annualized cost,
    the installed (bare module) cost times the capital recovery factor plus
    the utility cost times the operating hours: no Lang factor and no
    working capital.

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
    and the capital cost target, five double pipes of equal area [USD]:

    >>> round(targets.area_target(utilities), 2)
    24.14
    >>> targets.units_target(utilities)
    {'above': 2, 'below': 3, 'total': 5}
    >>> capital = targets.capital_cost_target(utilities)
    >>> capital['N_shells'], capital['type'], round(capital['installed_cost'])
    (5, 'Double pipe', 33991)

    All targets, with the total annualized cost [USD/yr] at 10 % interest
    over 10 years and 8000 operating hours a year:

    >>> {key: round(value, 2) for key, value in targets.summary().items()}
    {'T_min_app': 10.0, 'Q_hot': 180000.0, 'Q_cold': 144000.0,
     'utility_cost': 1.66, 'area': 24.14, 'N_units': 5, 'N_shells': 5,
     'purchase_cost': 23150.38, 'installed_cost': 33991.4,
     'annualized_capital_cost': 5531.94, 'operating_cost': 13247.77,
     'TAC': 18779.71}

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

    def _served(self, heating):
        """``(top, bottom, shift)`` of the streams a heating agent can serve
        (the cold streams with heat above the pinch) or a cooling one (the
        hot streams with heat below it): shifted temperatures [K] and shifts
        [K]."""
        grid = self.grid
        above, below = self._side_heat()
        if heating: j = np.flatnonzero(~self.is_hot & (above > self._tol))
        else: j = np.flatnonzero(self.is_hot & (below > self._tol))
        Ts = grid['Ts']
        return Ts[grid['k_hi'][j]], Ts[grid['k_lo'][j]], self.shift[j]

    def _ends(self, served, lo, hi, heating):
        """
        ``((T_in, T_out), overlap)`` of a utility serving shifted [lo, hi]:
        the real temperatures [K] of the `served` streams overlapping the
        stretch (`overlap`, a mask), the coldest and the hottest (heating;
        the hottest and the coldest for cooling); the stretch's ends plus 0
        (heating) or `T_min_app` (cooling) if none overlaps it.
        """
        top, bottom, s = served
        overlap = (top > lo) & (bottom < hi)
        if not overlap.any():
            if heating: return (lo, hi), overlap
            return (hi + self.T_min_app, lo + self.T_min_app), overlap
        s = s[overlap]
        cold = float(np.min(np.maximum(bottom[overlap], lo) + s))
        hot = float(np.max(np.minimum(top[overlap], hi) + s))
        return ((cold, hot) if heating else (hot, cold)), overlap

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
        B = _balance(positions, [(l, q) for l, q, _ in heat],
                     [(l, q) for l, q, _ in cool])
        zero = B <= tol
        T = positions[0]
        dT = bst.HeatUtility.dT
        utilities = []
        with _isolated_pricing([a for *_, a in heat + cool if a is not None]):
            served = self._served(True)
            for level, q, agent in heat:
                g = _gap(positions, level)
                z = np.flatnonzero(zero[g:])
                end = g + z[0] + 1 if z.size else T.size
                lo = float(T[end - 1]) if z.size else self.pinch_T
                hi = min(level, float(Ts[0]))
                (T_in, T_out), overlap = self._ends(served, lo, hi, True)
                T_priced = T_in
                if _sensible(agent):
                    limit = self._line_limit(agent, True, q, T[g:end], B[g:end],
                                             [x[overlap] for x in served])
                    if limit is not None: T_priced = max(T_in, limit - dT)
                utilities.append(self._utility(agent, True, level, q, T_in,
                                               T_out, T_priced))
            served = self._served(False)
            for level, q, agent in cool:
                g = _gap(positions, level)
                z = np.flatnonzero(zero[:g])
                start = int(z[-1]) if z.size else 0
                hi = float(T[start]) if z.size else self.pinch_T
                lo = max(level, float(Ts[-1]))
                (T_in, T_out), overlap = self._ends(served, lo, hi, False)
                T_priced = T_in
                if _sensible(agent):
                    limit = self._line_limit(agent, False, q, T[start:g],
                                             B[start:g],
                                             [x[overlap] for x in served])
                    if limit is not None: T_priced = min(T_in, limit + dT)
                utilities.append(self._utility(agent, False, level, q, T_in,
                                               T_out, T_priced))
        return utilities

    @staticmethod
    def _utility(agent, heating, level, q, T_in, T_out, T_priced):
        """The `UtilityTarget` of `agent` serving `q` from `T_in` to `T_out`
        (process temperatures), priced as one biosteam heat utility from
        `T_priced` to `T_out` (inside `_isolated_pricing`)."""
        if agent is None:
            nan = math.nan
            return UtilityTarget(None, None, heating, level, q, nan, nan, nan,
                                 T_in, T_out, nan, nan)
        hu = bst.HeatUtility()
        hu(q if heating else -q, T_priced, T_out, agent=agent)
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
        it the same value.
        """
        N = self.is_hot.size
        units = self.units or ()
        universe = _owner_chains(units)
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
            elif not any(_names_unit(key, u) for u in universe):
                unknown.append(key)
            items.append((key, _film_value(value, f'h of key {key!r}')))
        if unknown:
            raise ValueError('h keys that name no stream: '
                             f"{', '.join(map(repr, unknown))} (a key is a "
                             'stream index or, for targets of heat utilities, '
                             'a unit, the ID of a unit without an owner, or '
                             '(owner.ID, unit.ID))')
        unit_items = [(k, v) for k, v in items if not isinstance(k, Integral)]
        for unit in universe:
            first = {} # level -> (key, value)
            for key, value in unit_items:
                level = _contribution_key_level(key, unit)
                if level is None: continue
                other, other_value = first.setdefault(level, (key, value))
                if other_value != value:
                    raise ValueError(
                        f'h keys {other!r} and {key!r} give unit '
                        f'{_unit_name(unit)!r} different values at the same '
                        f'level: {other_value!r} and {value!r}')
        values = np.full(N, _H_DEFAULT)
        for j in range(N):
            best = None # (level, value) of the nearest key
            for key, value in items:
                if isinstance(key, Integral):
                    level = -1 if key == j else None
                else:
                    level = _contribution_key_level(key, units[j])
                if level is not None and (best is None or level < best[0]):
                    best = (level, value)
            if best is not None: values[j] = best[1]
        return values

    def _area(self, utilities, h, h_utilities):
        """`area_target` with resolved film coefficients (`h`, an (N,)
        array; `h_utilities`, one per utility)."""
        if any(u.agent is None for u in utilities): return math.nan
        hot, cold = self._balanced_segments(
            utilities, 1. / h, [1. / x for x in h_utilities])
        return _bath(hot, cold)

    def area_target(self, utilities=None, h=None, h_utilities=None):
        """
        The area target [m²] of vertical heat transfer between the balanced
        composite curves: Bath's formula (see Notes of the class).

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            Defaults to `utility_targets` with the default agents.
        h : float, Sequence[float] or Mapping, optional
            Film coefficient of the streams [kW/m²/K]: one for all, one per
            stream in stream order, or a mapping from stream indices (int)
            or, for targets of heat utilities, keys that name units as
            approach contribution keys do (a unit, the ID of a unit without
            an owner, or ``(owner.ID, unit.ID)``; an auxiliary unit takes the
            value of its owner's key unless a key names it). The nearest key
            wins, a stream index first; streams left out take 1. Labels are
            not keys. Defaults to 1 for every stream.
        h_utilities : float or Mapping, optional
            Film coefficient of the utilities [kW/m²/K]: one for all, or a
            mapping from agent IDs or agents (the others take 1). Defaults
            to 1.

        Returns
        -------
        float
            The area [m²]: inf (with a RuntimeWarning) at a zero or
            negative driving force, nan if part of a utility target is
            unserved.

        """
        if utilities is None: utilities = self.utility_targets()
        return self._area(utilities, self._film_coefficients(h),
                          _utility_films(utilities, h_utilities))

    def units_target(self, utilities=None):
        """
        An estimate of the number of units: Euler's S - 1 per region of the
        balanced grand composite curve between its zeros (see Notes of the
        class).

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            Defaults to `utility_targets` with the default agents.

        Returns
        -------
        dict
            ``{'above': int, 'below': int, 'total': int}``: the units of the
            regions above the process pinch, below it, and both.

        """
        if utilities is None: utilities = self.utility_targets()
        positions, B = self._balanced_positions(utilities)
        T, _, kind, k = positions
        P = T.size
        tol = self._tol
        Ts, Hl, Hr = self.grid['Ts'], self.grid['Hl'], self.grid['Hr']
        # each stream's cumulative enthalpy at every position, between the
        # virtual top and bottom ones
        H = np.where(kind == _ARRIVING, Hr[:, k], Hl[:, k])
        inner = np.flatnonzero((kind == _ABOVE) | (kind == _BELOW))
        if inner.size:
            ki = k[inner]
            f = (T[inner] - Ts[ki + 1]) / (Ts[ki] - Ts[ki + 1])
            H[:, inner] = Hr[:, ki + 1] + (Hl[:, ki] - Hr[:, ki + 1]) * f
        H = np.column_stack([Hr[:, 0], H, Hl[:, -1]])
        cuts = [-1, *np.flatnonzero(B <= tol).tolist(), P]
        gaps = [_gap(positions, u.level) for u in utilities if u.duty > tol]
        pinch = int(np.flatnonzero(
            (T == Ts[self._k_pinch])
            & (kind == (_ARRIVING if self.cut == 'below' else _LEAVING)))[0])
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
            shell_area = A / N_shells
            purchase, installed, kind = _exchanger_costs(
                shell_area * _FT2_PER_M2, P, heat_exchanger_type, material)
            purchase *= N_shells
            installed *= N_shells
        else: # no area, or nan
            shell_area = purchase = installed = A
            kind = None
        furnace_duty = sum((u.utility_duty for u in fuel), 0.)
        if fuel:
            furnace_purchase, furnace_installed = _furnace_costs(furnace_duty)
            purchase += furnace_purchase
            installed += furnace_installed
        return {'area': A, 'N_shells': N_shells, 'shell_area': shell_area,
                'type': kind, 'furnace_duty': furnace_duty,
                'purchase_cost': purchase, 'installed_cost': installed}, N

    def capital_cost_target(self, utilities=None, h=None, h_utilities=None,
                            P=101325., heat_exchanger_type='Floating head',
                            material='Carbon steel/carbon steel'):
        """
        The capital cost target: the area target in as many shells of equal
        area as the units target (less a furnace), costed with biosteam's
        correlations, plus a furnace for the fuel (see Notes of the class).

        Parameters
        ----------
        utilities : list[UtilityTarget], optional
            Defaults to `utility_targets` with the default agents.
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

        Returns
        -------
        dict
            'area' [m²]; 'N_shells', the number of shells; 'shell_area', the
            area of each [m²]; 'type', biosteam's type of each ('Double pipe'
            below 150 ft², None without shells or with an infinite area);
            'furnace_duty', the utility duty of the fuel [kJ/hr];
            'purchase_cost' and 'installed_cost' (bare module) [USD]. inf
            with an infinite area, nan with an unserved utility.

        """
        if utilities is None: utilities = self.utility_targets()
        return self._capital(utilities, self._film_coefficients(h),
                             _utility_films(utilities, h_utilities), P,
                             heat_exchanger_type, material)[0]

    def _check_options(self, options):
        """The options of `summary` (agents as lists), after raising for
        any that `summary` would reject, before pricing anything."""
        unknown = [i for i in options if i not in _SUMMARY_OPTIONS]
        if unknown:
            raise TypeError(f"unexpected options: {', '.join(map(repr, unknown))} "
                            f'(the options are {", ".join(_SUMMARY_OPTIONS)})')
        options = dict(options)
        for key in ('heating_agents', 'cooling_agents'):
            if options.get(key) is not None: options[key] = list(options[key])
        get = options.get
        _economics(get('interest'), get('years'), get('operating_hours'),
                   get('tea'))
        _check_exchanger(get('heat_exchanger_type', 'Floating head'),
                         get('material', 'Carbon steel/carbon steel'))
        _pressure(get('P', 101325.))
        self._film_coefficients(get('h'))
        _utility_films((), get('h_utilities'),
                       [*(get('heating_agents') or ()),
                        *(get('cooling_agents') or ())])
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
            unless None).

        Returns
        -------
        dict
            'T_min_app' [K]; 'Q_hot' and 'Q_cold' [kJ/hr], process side;
            'utility_cost' [USD/hr]; 'area' [m²]; 'N_units', the units
            target; 'N_shells'; 'purchase_cost' and 'installed_cost' [USD];
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
            [*(heating_agents or ()), *(cooling_agents or ())])
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
                'purchase_cost': capital['purchase_cost'],
                'installed_cost': capital['installed_cost'],
                'annualized_capital_cost': annualized,
                'operating_cost': operating, 'TAC': annualized + operating}


def _least_TAC(records):
    """Index of the record of least TAC (nan counts as inf)."""
    TAC = np.array([r['TAC'] for r in records], dtype=float)
    return int(np.argmin(np.where(np.isnan(TAC), math.inf, TAC)))

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
        Evaluations per round of refinement. Defaults to 8.
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
        If no minimum approach temperature of the sweep has a finite TAC
        (each has an unserved utility or an infinite area): the optimum is
        then the smallest one, and meaningless.

    Notes
    -----
    Each minimum approach temperature is evaluated with
    ``targets.at(T_min_app, scale_contributions).summary(**options)``. Only
    the streams without an approach contribution follow the minimum approach
    temperature (``T_min_app / 2`` each), unless `scale_contributions`
    scales the specified ones by ``T_min_app / targets.T_min_app`` too, the
    contribution scaling of [Ahmad90]_; with every stream specified, the
    sweep needs it.

    The TAC is not smooth (the units target jumps, so do biosteam's
    correlations at 150 ft², and the pinch can move), so the sweep is a
    grid search: each round of refinement evaluates `n_refine` evenly spaced
    temperatures between the neighbors of the least TAC so far, so after r
    rounds the spacing is the grid's times (2/9)**r for the defaults. A
    least TAC at the smallest or largest temperature evaluated is flagged
    (`Supertargets.at_bound`).

    Examples
    --------
    One hot stream cooled 400 -> 300 K and one cold stream heated 290 ->
    390 K, both 1 kW/K. Up to 10 K they balance with no utility in a single
    exchanger of 20 m² (215 ft², a shell and tube); above it the utilities
    add units, and three double pipes of less than 150 ft² each cost less,
    so on a coarse grid the least total annualized cost (TAC) is at 15 K:

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

    Refining around it finds the least TAC just above 10 K, with almost no
    utility and the area already in three double pipes:

    >>> sweep = supertarget(targets, grid)
    >>> sweep
    <Supertargets: 30 points, T_min_app 5-30 K, optimum 10.05 K (TAC 3.510e+03 USD/yr)>
    >>> sweep.optimum['N_shells'], round(sweep.optimum['area'], 2)
    (3, 19.89)

    References
    ----------
    .. [Ahmad90] Ahmad, S., Linnhoff, B., & Smith, R. (1990). Cost optimum
        heat exchanger networks - 2. Targets and design for detailed capital
        cost models. Computers & Chemical Engineering, 14(7), 751-767.

    """
    if T_min_apps is None: T_min_apps = _T_MIN_APPS
    T = np.asarray(T_min_apps, dtype=float)
    if T.ndim != 1 or not T.size:
        raise ValueError('T_min_apps must be a non-empty sequence of '
                         'temperatures [K]')
    if not np.all(np.isfinite(T)):
        raise ValueError(f'T_min_apps must be finite, not {T_min_apps!r}')
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
        return targets.at(T_min_app, scale_contributions).summary(**options)
    records = [evaluate(float(x)) for x in np.unique(T)]
    for _ in range(refine):
        records.sort(key=lambda r: r['T_min_app'])
        i = _least_TAC(records)
        lo = records[max(i - 1, 0)]['T_min_app']
        hi = records[min(i + 1, len(records) - 1)]['T_min_app']
        if hi - lo <= 1e-6: break
        evaluated = np.array([r['T_min_app'] for r in records])
        for x in np.linspace(lo, hi, n_refine + 2)[1:-1]:
            if np.abs(evaluated - x).min() > 1e-12:
                records.append(evaluate(float(x)))
    sweep = Supertargets(targets, records, options, scale_contributions)
    if not np.isfinite(sweep['TAC']).any():
        warn('no minimum approach temperature of the sweep has a finite TAC '
             '(each has an unserved utility or an infinite area), so its '
             'optimum, the smallest, is meaningless', RuntimeWarning)
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

    def __getitem__(self, key):
        return np.array([r[key] for r in self.records])

    def to_frame(self):
        """The records as a pandas DataFrame indexed by 'T_min_app'."""
        import pandas as pd
        return pd.DataFrame(self.records).set_index('T_min_app')

    def __repr__(self):
        n = len(self.records)
        lo, hi = self.records[0]['T_min_app'], self.records[-1]['T_min_app']
        bound = ', at the bound' if self.at_bound else ''
        return (f'<{type(self).__name__}: {n} point{"s" if n != 1 else ""}, '
                f'T_min_app {lo:g}-{hi:g} K, optimum {self.T_min_app:.4g} K '
                f'(TAC {self.optimum["TAC"]:.3e} USD/yr){bound}>')


def _profile_area(x, dT, inv_U, ID):
    """
    Area [m²] of counter-current heat transfer over the duties `x` [kJ/hr]
    (ascending) at the driving forces `dT` [K] there, linear in between (each
    stretch at its log-mean temperature difference), with an overall 1/U of
    `inv_U` [m²·K/kW]; inf with a RuntimeWarning naming the exchanger `ID`
    at a driving force of `_DT_ZERO` or less.
    """
    if dT.min() <= _DT_ZERO:
        warn(f'{ID} transfers heat at a driving force of {dT.min():.6g} K',
             RuntimeWarning)
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
    `curve` (on the curve's thermodynamic basis): the same enthalpies (to
    1e-6 of its duty at the outlet, which flashes reach), and the same
    pressure and flows at the inlet (an inlet at the enthalpy reference
    has the same enthalpy at any flow)."""
    span = abs(curve.H_out - curve.H_in)
    a, b = _flows(curve.stream_in), _flows(s_in)
    return (abs(curve.H_in - s_in.H) <= 1e-9 * (abs(curve.H_in) + span)
            and abs(curve.H_out - s_out.H) <= 1e-6 * span
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
                             f'{_unit_label(hx)} (simulated since?): make '
                             'them again with HXN.targets()')

def _check_network(HXN, targets):
    """Raise a ValueError unless every stream of `HXN`'s network enters and
    leaves it at the end states of its curve in `targets` (which are of the
    units' current states): its copy of the real inlet where it enters the
    network, and the outlet of its last stage (the network's copies share
    the curves' thermodynamic basis)."""
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
                                  _agent_films(of, h_utilities, agents)):
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
        areas[hx.ID] = (
            _profile_area(x, dT, 1. / h[i] + 1. / h_agent, hx.ID),
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
        was synthesized at) and of the targets; 'status' (``HXN.synthesis_info['status']``); 'applied',
        whether the network has a process exchanger; 'Q_hot' and 'Q_cold',
        the heat of the network's heating and cooling utilities [kJ/hr],
        process side (`HXN.actual_heat_util_load` is utility side);
        'utilities', ``{agent ID: {'duty', 'utility_duty', 'cost'}}``
        [kJ/hr, kJ/hr, USD/hr] (duty process side, a magnitude);
        'utility_cost' [USD/hr]; 'N_units', the exchangers with duty (a
        furnace included), of which 'N_process_units' and
        'N_utility_units'; 'area', the network's area like for like with
        the area target (see Notes) [m²]; 'area_biosteam', the area biosteam
        sized [m²]; 'exchanger_areas', ``{ID: (area, area_biosteam)}`` of
        each exchanger with duty [m²]; 'purchase_cost' and 'installed_cost'
        of all the network's exchangers [USD]; 'annualized_capital_cost',
        'operating_cost' and 'TAC' [USD/yr], as `PinchTargets.summary`
        computes them; 'targets', `PinchTargets.summary` of the targets.

    Raises
    ------
    RuntimeError
        If the facility was not simulated.
    ValueError
        If `targets` are not those of the facility's process streams in
        their current states (enthalpies at both ends, and pressure and
        flows at the inlet), if the network's streams do not enter and
        leave it at those states (a unit or a stream changed since the
        facility's last simulation), or for an invalid option.
    TypeError
        For an unknown option.

    Notes
    -----
    The network's 'area' is sized as the area target is: each exchanger
    with the film coefficients of its two sides (1 kW/m²/K by default),
    counter-current (a correction factor of 1), and integrated along the
    exact temperature profiles of its streams (their curves in the targets)
    against the process duty; a utility's temperature is linear in the duty
    between its inlet and outlet, and a furnace takes no area (as fuel in
    the targets). 'area_biosteam' is biosteam's own sizing: heuristic
    overall coefficients (0.5 kW/m²/K, but 1 for condensing against
    boiling, which no film coefficients reproduce), a correction factor for
    two shells, the log-mean of the terminal temperature differences
    (which overstates the mean difference along a glide) and the utility
    duty (the process duty divided by the heat transfer efficiency).

    The capital and utility costs are those of the whole network
    (`HXN.new_HXs` and `HXN.new_HX_utils`), not the facility's costs,
    which are the differences from the original exchangers (and zero for
    a network without process exchangers: the facility then keeps the
    original exchangers, to which the utility exchangers of the network,
    reported here, are equivalent).

    The targets are estimates, and the network may beat some of them:

    - the network serves each stream with one agent at its outlet (biosteam's
      rule: the first agent of the list that can), while the targets place
      the agents on the grand composite curve, so the network may use other
      agents than the targets (with Linnhoff & Hindmarsh's problem at
      10 K, medium pressure steam and chilled water where the targets need
      only low pressure steam and cooling water);
    - the area target is a minimum only for equal film coefficients and the
      same utilities: that problem's network takes 54.06 m² against a target
      of 59.68 m², with hotter steam and colder coolant;
    - the units target counts S - 1 per region between zeros of the
      balanced grand composite curve: a network needs fewer units where a
      subset of streams balances (6 against 7 in that problem) or where one
      agent serves a stream the targets split between levels, and more for
      repeated matches and splits.

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
    >>> round(report['TAC']), round(targets['TAC'])  # USD/yr
    (62041, 39396)

    """
    if not hasattr(HXN, 'new_HXs_hot_side'):
        raise RuntimeError('simulate the heat exchanger network before '
                           'reporting on it')
    if targets is None: targets = HXN.targets()
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
    'Pinch', 'Fuel' and 'Utility:<agent ID>'. The pinch is marked where the
    shifted composites touch: not for a threshold problem whose composites
    end apart, nor without streams on one side (whose empty composite stays
    out of the legend). Fuel, at infinite temperature on the balanced
    curves, is labeled rather than drawn, and so is an unserved remainder
    ('Utility:unserved', in the upper left corner), without which the
    balanced curves end at different heats.

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
                label=_legend('hot balanced composite', hot_H),
                gid='Hot balanced composite')
        ax.plot(cold_H, cold_T - 273.15, color=cold_color, lw=2,
                label=_legend('cold balanced composite', cold_H),
                gid='Cold balanced composite')
        above = [] # anchors of the labels above a heating flat
        fuels = []
        unserved = []
        for u in utilities:
            if u.agent is None:
                unserved.append(f"unserved {'heating' if u.heating else 'cooling'} "
                                f'{_format_H(u.duty)} kJ/hr (not drawn)')
                continue
            if u.agent.isfuel:
                fuels.append(u)
                continue
            label = f'{u.ID} {_format_H(u.duty)} kJ/hr'
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
                if u.heating: above.append((x, T_u))
            else: # beside the middle of a sensible agent's line
                T_u = 0.5 * (u.T_utility_in + u.T_utility_out)
                x = np.interp(T_u, T, H)
                offset = (-6, 0) if u.heating else (6, 0)
                ha, va = 'right' if u.heating else 'left', 'center'
            ax.annotate(label, xy=(x, T_u - 273.15), xytext=offset,
                        textcoords='offset points', ha=ha, va=va,
                        color=hot_color if u.heating else cold_color,
                        gid='Utility:' + u.ID, **text_kwargs)
        stacked = False
        for u in fuels: # at the end of the finite hot curve
            x = hot_H[-1] if hot_H.size else 0.
            y = hot_T[-1] if hot_T.size else targets.grid['Ts'][0]
            # above the label of the top flat if that ends there too
            stacked = any(abs(x - a) <= 1e-9 * (abs(x) + abs(a)) + 1e-12
                          and abs(y - b) <= 1e-6 for a, b in above)
            ax.annotate(f'{u.ID} (furnace) {_format_H(u.duty)} kJ/hr',
                        xy=(x, y - 273.15), xytext=(-2, 14) if stacked else (0, 4),
                        textcoords='offset points', ha='right', va='bottom',
                        color=hot_color, gid='Fuel', **text_kwargs)
        if unserved: # the curves end apart: say why
            ax.text(0.02, 0.97, '\n'.join(unserved), transform=ax.transAxes,
                    ha='left', va='top', color='0.45', gid='Utility:unserved',
                    **text_kwargs)
        # headroom for a label (or two) above the top flat
        ax.margins(y=0.16 if stacked else 0.1)
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
        ax.text(0.5 * (H_hot_end + H_cold_end), T_all.max() + 1.,
                f'hot utility {_format_H(targets.Q_hot)} kJ/hr', ha='center',
                va='bottom', gid='Hot utility', **text_kwargs)
        ax.text(0.5 * targets.Q_cold, T_all.min() - 1.,
                f'cold utility {_format_H(targets.Q_cold)} kJ/hr', ha='left',
                va='top', gid='Cold utility', **text_kwargs)
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
    of the grid and an unserved remainder in gray. Each label sits at the
    end of its segment, below it for heating and above it for cooling, off
    the curve where a level touches it. Artists carry stable gids: 'GCC',
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
        # each label beside the end of its step, where a level touches the
        # curve: below it for heating (the curve rises to the right above
        # the pinch) and above it for cooling (it falls to the right below)
        for steps, color, end, (dy, va) in (
                (heating, hot_color, Ts[0], (-2, 'top')),
                (cooling, cold_color, Ts[-1], (2, 'bottom'))):
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
                ax.annotate(ID, xy=(x + u.duty, y), xytext=(3, dy),
                            textcoords='offset points', ha='left', va=va,
                            fontsize=7, color=style['color'], gid='Label:' + ID)
                x += u.duty
                previous = y
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
    Artists carry stable gids: 'TAC', 'Capital' (the annualized capital
    cost), 'Operating' and 'Optimum'.

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
    <Supertargets: 63 points, T_min_app 1-40 K, optimum 3.905 K (TAC 1.818e+04 USD/yr)>
    >>> fig, ax = plot_supertargets(sweep)
    >>> [line.get_gid() for line in ax.get_lines() if line.get_gid()]
    ['TAC', 'Capital', 'Operating', 'Optimum']
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    T = sweep['T_min_app']
    ax.plot(T, sweep['TAC'], color='k', lw=2, label='total annualized cost',
            gid='TAC')
    ax.plot(T, sweep['annualized_capital_cost'], color='#2e6db4', lw=1.5,
            label='annualized capital cost', gid='Capital')
    ax.plot(T, sweep['operating_cost'], color='#d62728', lw=1.5,
            label='operating cost', gid='Operating')
    optimum = sweep.optimum
    ax.axvline(optimum['T_min_app'], color='k', ls='--', lw=0.8)
    ax.plot([optimum['T_min_app']], [optimum['TAC']], 'o', mfc='w', mec='k',
            ms=7, zorder=4, label=f"optimum {optimum['T_min_app']:.3g} K",
            gid='Optimum')
    ax.set_xlabel('$T_{min,app}$ [K]')
    ax.set_ylabel('cost [USD·yr$^{-1}$]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    if file: fig.savefig(file, dpi=dpi, bbox_inches='tight')
    return fig, ax
