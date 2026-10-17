# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Tests of `hensmith.targeting`.

A. Composite and grand composite curves: closed-form knots of Linnhoff &
   Hindmarsh's problem, consistency with the problem table on real-thermo
   cases, and approach contributions against an independent cascade.
B. Utility targets: the multiple-utility allocation on the grand composite
   curve with biosteam's agents (levels recomputed from the agents'
   attributes, list order, biosteam's dT, unserved remainders, served
   stretches), priced against an independent isolated pricing, and the
   balanced curves.
C. Supertargets: the area target against closed forms and an independent
   integration, the units target on references with process and utility
   pinches, the capital costs against biosteam's own costing, the
   economics, and the sweep against a closed-form optimum.
D. The facility and the report: the facility's targets equal the
   synthesis's bit for bit (before and after a simulation, with approach
   contributions and with replaced heat utilities), the report's areas
   against terminal log-mean temperature differences and against an
   independent integration on the network's own streams (branches of a
   split, phase changes), the report at the units' current states only
   (a cached network re-simulated, stale states detected, also with ideal
   thermodynamics), and no side effects on the flowsheet, the agents'
   stream dumps or a later synthesis.
E. Plots: stable gids whose data equal the curves, and readable labels
   and markers.

Reference A: H1 400 -> 300 K and C1 330 -> 440 K, both 1 kW/K, at
T_min_app = 10 K: 50 kW hot and 40 kW cold utility, shifted pinch 390 K.
"""
import inspect
import math
import time
import types
import warnings
import numpy as np
import pytest
import biosteam as bst
from numpy.testing import assert_allclose
from hensmith import (HeatExchangerNetwork, PinchTargets, Supertargets, supertarget,
                      network_report, plot_composite_curves, plot_grand_composite_curve,
                      plot_supertargets)
from hensmith._curves import _copy
from hensmith.hxn_synthesis import _problem_table, temperature_interval_pinch_analysis
from hensmith.targeting import _exchanger_costs, _economics, _lmtd, _profile_area
from test_hxn import (build_system, cp_units, KEMP, simulate_HXN, simulate_strictly,
                      _gid_artists)
from test_hxn_contributions import classic_cascade
import test_hxn_targets

K = 273.15
kW = 3600.  # kJ/hr
#: Reference A (temperatures in K, CP in kW/K).
REF_A = [('H1', 400., 300., 1.), ('C1', 330., 440., 1.)]
#: A balanced pair: no utility up to T_min_app = 10 K, 10 K apart.
PAIR = [('H1', 400., 300., 1.), ('C1', 290., 390., 1.)]
#: Reference A with C1 heated to 520 K: natural gas above high pressure steam.
FURNACE = [('C1', 330., 520., 1.), ('H1', 400., 300., 1.)]
#: Contributions of the KEMP reference with approach contributions [K].
KEMP_C = {'C1': 1., 'C3': 2., 'H2': 6., 'H4': 8.}
#: Film coefficients of the KEMP reference of the area target [kW/m2/K].
KEMP_H = {'C1': 1., 'H2': 2., 'C3': .5, 'H4': 4.}

def targets_of(units, T_min_app, **kwargs):
    """`PinchTargets` of the units' heat utilities, sorted by duty as the
    facility sorts them."""
    hus = sorted([u.heat_utilities[0] for u in units], key=lambda hu: hu.duty)
    return PinchTargets.from_heat_utilities(hus, T_min_app, **kwargs)

def targets(name, rows, T_min_app, **kwargs):
    """`targets_of` constant-CP streams given in K."""
    return targets_of(cp_units(name, rows), T_min_app, **kwargs)

def agent(ID):
    return bst.HeatUtility.get_agent(ID)

def fresh_price(q, T_in, T_out, agent):
    """``(cost, utility duty, utility outlet T)`` of a heat utility priced
    with its own empty stream dump in its own temporary flowsheet (`q` > 0
    heats the process): an independent copy of the isolated pricing."""
    dump = agent.utility_stream_dump
    agent.utility_stream_dump = []
    try:
        with bst.Flowsheet('test_targeting_price'):
            hu = bst.HeatUtility()
            hu(q, T_in, T_out, agent=agent)
            return hu.cost, abs(hu.duty), hu.outlet_utility_stream.T
    finally:
        agent.utility_stream_dump = dump

def min_vertical_gap(hot_T, hot_H, cold_T, cold_H):
    """Smallest T_hot(H) - T_cold(H) over the H range both composite curves
    span (from docs/_demo_src/_common.py). Curves are piecewise linear in H,
    so the minimum is at a vertex; at a vertical (equal-H) run the hot curve
    takes its lower T and the cold curve its higher T (conservative)."""
    hot_H, hot_T = np.asarray(hot_H), np.asarray(hot_T)
    cold_H, cold_T = np.asarray(cold_H), np.asarray(cold_T)
    _, i = np.unique(hot_H, return_index=True)               # first = lower T
    hH, hT = hot_H[i], hot_T[i]
    _, j = np.unique(cold_H[::-1], return_index=True)        # last = higher T
    j = cold_H.size - 1 - j
    cH, cT = cold_H[j], cold_T[j]
    lo, hi = max(hH[0], cH[0]), min(hH[-1], cH[-1])
    Hs = np.concatenate([hH, cH])
    Hs = Hs[(Hs >= lo) & (Hs <= hi)]
    return float(np.min(np.interp(Hs, hH, hT) - np.interp(Hs, cH, cT)))

def min_driving_force(hot_T, hot_H, cold_T, cold_H):
    """Smallest T_hot - T_cold over every stretch of positive heat between
    the breakpoints of two piecewise-linear curves (a vertical run carries
    no heat: each stretch is evaluated inside the segments it lies in)."""
    end = min(hot_H[-1], cold_H[-1])
    X = np.unique(np.concatenate([hot_H, cold_H]))
    X = X[(X >= 0) & (X <= end)]
    def T_at(T, H, i, x):
        return T[i] + (T[i + 1] - T[i]) * (x - H[i]) / (H[i + 1] - H[i])
    gaps = []
    for x, y in zip(X[:-1], X[1:]):
        m = 0.5 * (x + y)
        i = np.searchsorted(hot_H, m, 'right') - 1
        j = np.searchsorted(cold_H, m, 'right') - 1
        for z in (x, y):
            gaps.append(T_at(hot_T, hot_H, i, z) - T_at(cold_T, cold_H, j, z))
    return min(gaps)

def points(T, H):
    """The polyline's points with repeated ones dropped."""
    keep = np.ones(T.size, dtype=bool)
    keep[1:] = (np.diff(T) != 0) | (np.diff(H) != 0)
    return np.column_stack([T[keep], H[keep]])

def by_ID(utilities):
    return {u.ID: u for u in utilities}

def member(T_a, T_b, Q, h=1.):
    """A member of `integrated_area`."""
    return (min(T_a, T_b), max(T_a, T_b), Q, h)

def utility_members(utilities, h=None):
    """The `integrated_area` members of the served utilities ``(hot,
    cold)``: latent at the agent's temperature (no `T_limit`), else linear
    between the utility's inlet and outlet temperatures."""
    h = h or {}
    hot, cold = [], []
    for u in utilities:
        if u.agent.T_limit is None: T_a = T_b = u.agent.T
        else: T_a, T_b = u.T_utility_in, u.T_utility_out
        (hot if u.heating else cold).append(member(T_a, T_b, u.duty, h.get(u.ID, 1.)))
    return hot, cold

def integrated_area(hot, cold, n=1_500_000):
    """
    Area [m2] by midpoint integration of ``sum_j (1/h_j) (dq_j/dH) / dT``
    over the heat of two composite curves, each built here from members
    ``(T_a, T_b, Q [kJ/hr], h)``: linear from T_a to T_b, or latent at T_a
    == T_b. At each H the curve's temperature is found by inverting the
    cumulative heat, and the heat is shared by the members present there in
    proportion to their heat capacity flow rates (on a latent flat, by the
    latent members at that temperature).
    """
    def side(members):
        knots = np.unique([T for m in members for T in m[:2]])
        def H_at(T, right):
            total = np.zeros(T.size)
            for T_a, T_b, Q, _ in members:
                if T_a == T_b: total += Q * ((T > T_a) | (right & (T == T_a)))
                else: total += Q * np.clip((T - T_a) / (T_b - T_a), 0., 1.)
            return total
        T = np.repeat(knots, 2)
        H = np.column_stack([H_at(knots, False), H_at(knots, True)]).ravel()
        return T, H
    hot_T, hot_H = side(hot)
    cold_T, cold_H = side(cold)
    end = min(hot_H[-1], cold_H[-1])
    dH = end / n
    H = (np.arange(n) + 0.5) * dH
    def weights(members, T):
        latent = [m for m in members if m[0] == m[1]]
        flat = np.zeros(T.size, dtype=bool)
        for T_a, _, _, _ in latent: flat |= T == T_a
        CP = np.zeros(T.size)
        w = np.zeros(T.size)
        Q_flat = np.zeros(T.size)
        w_flat = np.zeros(T.size)
        for T_a, T_b, Q, h in members:
            if T_a == T_b:
                at = T == T_a
                Q_flat += Q * at
                w_flat += Q * at / h
            else:
                present = (T > T_a) & (T < T_b)
                CP += Q / (T_b - T_a) * present
                w += Q / (T_b - T_a) * present / h
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(flat, w_flat / Q_flat, w / CP)
    Th = np.interp(H, hot_H, hot_T)
    Tc = np.interp(H, cold_H, cold_T)
    return float(np.sum((weights(hot, Th) + weights(cold, Tc)) / (Th - Tc)) * dH / kW)

def rows_members(rows, offset=0., h=None):
    """``(hot, cold)`` `integrated_area` members of constant-CP rows."""
    h = h or {}
    hot, cold = [], []
    for ID, T_in, T_out, CP in rows:
        m = member(T_in + offset, T_out + offset, CP * abs(T_in - T_out) * kW,
                   h.get(ID, 1.))
        (hot if T_in > T_out else cold).append(m)
    return hot, cold

@pytest.fixture(scope='module')
def kemp_hxn():
    """KEMP synthesized by the facility at 10 K."""
    units = cp_units('test_targeting_kemp_hxn', KEMP, K)
    sys, HXN = simulate_HXN(units, 10.)
    return units, HXN

@pytest.fixture(scope='module')
def kemp_hxn_replaced():
    """KEMP synthesized by the facility at 10 K, replacing the units' heat
    utilities."""
    units = cp_units('test_targeting_kemp_hxn_replaced', KEMP, K)
    sys, HXN = simulate_HXN(units, 10., replace_unit_heat_utilities=True)
    return units, HXN

@pytest.fixture(scope='module')
def quickstart_hxn():
    """The documentation's quickstart system simulated with its facility at
    5 K (two auxiliary exchangers of the column D0, a boiling and a
    condensing stream): ``(sys, HXN)``."""
    sys, HXN, _ = build_system()
    sys.simulate()
    return sys, HXN

@pytest.fixture(scope='module')
def quickstart(quickstart_hxn):
    """The targets of the documentation's quickstart system at 5 K."""
    hus = sorted(quickstart_hxn[1].original_heat_utils, key=lambda hu: hu.duty)
    return PinchTargets.from_heat_utilities(hus, 5.)

def liquids(name, **kwargs):
    """The liquids of the second example of `HeatExchangerNetwork` (and of
    `network_report`) simulated with a facility at 10 K with the options
    `kwargs`: ``(sys, HXN, units)``."""
    bst.main_flowsheet.set_flowsheet(name)
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    def process_stream(ID, T_in, T_out, P, **kmol_hr):
        inlet = bst.Stream(ID + '_in', T=T_in, P=P, phase='l',
                           units='kmol/hr', **kmol_hr)
        return bst.HXutility(ID, ins=inlet, T=T_out, rigorous=False)
    units = [process_stream('H1', 400., 320., 5e5, Water=60.),
             process_stream('H2', 390., 330., 5e5, Ethanol=30.),
             process_stream('C1', 310., 410., 5e5, Water=150.),
             process_stream('H3', 360., 300., 101325., Water=200.)]
    sys, HXN = simulate_HXN(units, 10., **kwargs)
    return sys, HXN, units

# %% Composites and grand composite curve

def test_composites_closed_form_kemp():
    t = targets_of(cp_units('test_targeting_kemp', KEMP, K), 10.)
    hot_T, hot_H, cold_T, cold_H = t.composite_curves()
    hot = [(30, 0), (60, 45), (90, 180), (145, 427.5), (150, 450), (170, 510)]
    cold = [(20, 60), (50, 120), (80, 180), (135, 510), (140, 530)]
    to_K = lambda knots: np.array([(T + K, H * kW) for T, H in knots])
    assert_allclose(points(hot_T, hot_H), to_K(hot), rtol=1e-12)
    assert_allclose(points(cold_T, cold_H), to_K(cold), rtol=1e-12)
    shifted = t.composite_curves(shifted=True)
    assert_allclose(points(*shifted[:2]), to_K(hot) - [10., 0.], rtol=1e-12)
    assert_allclose(points(*shifted[2:]), to_K(cold), rtol=1e-12)
    T, H = t.grand_composite_curve()
    assert T[0] == t.table.Ts[0] and H[0] == pytest.approx(20 * kW, rel=1e-12)
    assert H.min() == pytest.approx(0., abs=1e-9 * kW)
    assert T[np.argmin(H)] == t.pinch_T == 80. + K

@pytest.mark.parametrize('name', ['linnhoff', 'two_stream', 'condenser_at_pinch',
                                  'binary_glide'])
def test_composites_consistent_with_table(name):
    units, T_min_app = test_hxn_targets.CASES[name]()
    t = targets_of(units, T_min_app)
    hus = sorted([u.heat_utilities[0] for u in units], key=lambda hu: hu.duty)
    # bit for bit the synthesis's own pinch analysis
    analysis = temperature_interval_pinch_analysis(hus, T_min_app)
    assert t.Q_hot.hex() == analysis[1].hex()
    assert t.Q_cold.hex() == analysis[2].hex()
    # and the problem table of the same streams given directly
    direct = PinchTargets(t.streams_inlet, t.streams_quenched, t.is_hot, T_min_app)
    assert direct.table.residual.tobytes() == t.table.residual.tobytes()
    assert direct.labels == [s.ID for s in t.streams_inlet]
    assert direct.units is direct.heat_utilities is None
    assert t.units == [hu.unit for hu in hus if hu.duty > 0] + [
        hu.unit for hu in hus if hu.duty < 0]
    hot_T, hot_H, cold_T, cold_H = t.composite_curves()
    assert_allclose(cold_H[-1] - hot_H[-1], t.Q_hot, rtol=1e-9)
    assert_allclose(cold_H[0], t.Q_cold, rtol=1e-9)
    assert min_vertical_gap(hot_T, hot_H, cold_T, cold_H) >= T_min_app - 1e-6
    if t.Q_hot > 0:
        shifted_gap = min_vertical_gap(*t.composite_curves(shifted=True))
        assert abs(shifted_gap) <= 1e-6
    T, H = t.grand_composite_curve()
    assert np.all(np.diff(T) <= 0)
    assert abs(H.min()) <= t._tol
    assert_allclose([H[0], H[-1]], [t.Q_hot, t.Q_cold], rtol=1e-12, atol=t._tol)
    # the utility targets add up to the MER targets, the balanced grand
    # composite curve is non-negative and zero at the pinch, the balanced
    # composite curves end at the same heat, and the area is finite
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        utilities = t.utility_targets()
        area = t.area_target(utilities)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert_allclose([sum(u.duty for u in utilities if u.heating),
                     sum(u.duty for u in utilities if not u.heating)],
                    [t.Q_hot, t.Q_cold], rtol=1e-9, atol=t._tol)
    T, B = t.balanced_grand_composite_curve(utilities)
    assert B.min() >= -t._tol and B[T == t.pinch_T].min() <= t._tol
    hot_T, hot_H, cold_T, cold_H = t.balanced_composite_curves(utilities)
    assert_allclose(hot_H[-1], cold_H[-1], rtol=1e-9)
    assert 0. < area < math.inf

def test_contributions():
    assert 'approach_contributions' in inspect.signature(_problem_table).parameters
    units = cp_units('test_targeting_contributions', KEMP, K)
    t = targets_of(units, 10.)
    N = len(units)
    # all unspecified: the default, bit for bit
    none = PinchTargets(t.streams_inlet, t.streams_quenched, t.is_hot, 10.,
                        [None] * N, curves=t.curves)
    assert none.approach_contributions is None
    assert none.table.residual.tobytes() == t.table.residual.tobytes()
    assert none.Q_hot.hex() == t.Q_hot.hex()
    # ... at any T_min_app (only a given value needs T_min_app > 0), with
    # the length checked all the same
    t0 = targets_of(units, 0.)
    none0 = PinchTargets(t.streams_inlet, t.streams_quenched, t.is_hot, 0.,
                         [float('nan')] * N, curves=t.curves)
    assert none0.approach_contributions is None
    assert none0.Q_hot.hex() == t0.Q_hot.hex()
    with pytest.raises(ValueError, match=f'{N + 1} approach contributions '
                                         f'given for {N} streams'):
        PinchTargets(t.streams_inlet, t.streams_quenched, t.is_hot, 10.,
                     [None] * (N + 1), curves=t.curves)
    assert targets_of(units, 10., approach_contributions={}).approach_contributions is None
    # KEMP with contributions against the classic cascade (hot T - c, cold T + c)
    tc = targets_of(units, 10., approach_contributions=KEMP_C)
    labels = tc.labels
    assert labels == ['C1', 'C3', 'H2', 'H4']
    rows = {ID: (T_in + K, T_out + K, CP) for ID, T_in, T_out, CP in KEMP}
    streams = [dict(T_in=rows[ID][0], T_out=rows[ID][1], CP=rows[ID][2],
                    kind='hot' if rows[ID][0] > rows[ID][1] else 'cold')
               for ID in labels]
    c = [KEMP_C[ID] for ID in labels]
    Qh, Qc, zeros = classic_cascade(streams, c)
    assert_allclose([tc.Q_hot / kW, tc.Q_cold / kW], [Qh, Qc], rtol=1e-9)
    assert_allclose([Qh, Qc], [12., 52.], rtol=1e-12)
    assert tc.approach_contributions == tuple(c)
    assert abs(min_vertical_gap(*tc.composite_curves(shifted=True))) <= 1e-6
    real_gap = min_vertical_gap(*tc.composite_curves())
    assert real_gap >= min(c[2:]) + min(c[:2]) - 1e-9
    assert real_gap == pytest.approx(8.222222, abs=1e-6)
    # the levels of section 3.4, recomputed from shifts computed here
    hot = np.array([s['kind'] == 'hot' for s in streams])
    shift = np.where(hot, 5. + np.array(c), 5. - np.array(c))
    assert_allclose(tc.shift, shift, rtol=0, atol=0)
    pinch = zeros[0] - 5.
    assert tc.pinch_T == pytest.approx(pinch, abs=1e-9)
    top = np.array([max(s['T_in'], s['T_out']) for s in streams]) - shift
    bottom = np.array([min(s['T_in'], s['T_out']) for s in streams]) - shift
    dT = bst.HeatUtility.dT
    LP, CW = agent('low_pressure_steam'), agent('cooling_water')
    R = LP.T - LP.dT - dT
    above = [j for j in range(N) if not hot[j] and top[j] > pinch]
    reached = [R - shift[j] for j in above if top[j] > R - shift[j]]
    L_LP = min(reached) if reached else R - min(shift[j] for j in above)
    R = CW.T + CW.dT + dT
    below = [j for j in range(N) if hot[j] and bottom[j] < pinch]
    reached = [R - shift[j] for j in below if bottom[j] < R - shift[j]]
    L_CW = max(reached) if reached else R - max(shift[j] for j in below)
    utilities = by_ID(tc.utility_targets())
    assert set(utilities) == {'low_pressure_steam', 'cooling_water'}
    assert utilities['low_pressure_steam'].level == L_LP
    assert utilities['cooling_water'].level == L_CW
    assert_allclose([L_LP, L_CW], [403.189, 299.372], rtol=1e-12)
    # every form of synthesize_network: a mapping keyed by unit ID, a
    # sequence aligned with the heat utilities as given (a zero-duty one
    # included and dropped) and a callable; the per-stream sequence of
    # __init__ (stream order)
    zero = cp_units('test_targeting_contributions_zero', [('Z', 350., 350., 1.)])
    assert zero[0].heat_utilities[0].duty == 0.
    given = sorted(units, key=lambda u: u.heat_utilities[0].duty)
    given.insert(2, zero[0])
    hus = [u.heat_utilities[0] for u in given]
    forms = [KEMP_C, [KEMP_C.get(u.ID, 5.) for u in given],
             lambda hu: KEMP_C.get(hu.unit.ID)]
    for form in forms:
        other = PinchTargets.from_heat_utilities(hus, 10., form)
        assert other.labels == labels
        assert other.approach_contributions == tc.approach_contributions
        assert other.Q_hot.hex() == tc.Q_hot.hex()
    direct = PinchTargets(tc.streams_inlet, tc.streams_quenched, tc.is_hot, 10.,
                          c, curves=tc.curves)
    assert direct.Q_hot.hex() == tc.Q_hot.hex()
    with pytest.raises(ValueError, match='5 approach contributions given for 4'):
        PinchTargets.from_heat_utilities(hus[:4], 10., forms[1])
    with pytest.raises(TypeError, match='a heat utility cannot be'):
        PinchTargets.from_heat_utilities(hus, 10., {hus[0]: 1.})
    with pytest.raises(TypeError, match='approach_contributions must be a sequence'):
        PinchTargets(tc.streams_inlet, tc.streams_quenched, tc.is_hot, 10.,
                     KEMP_C, curves=tc.curves)
    with pytest.raises(ValueError, match='approach contribution of stream 1'):
        PinchTargets(tc.streams_inlet, tc.streams_quenched, tc.is_hot, 10.,
                     [1., 0., None, None], curves=tc.curves)
    # at: unspecified streams follow T_min_app, specified ones stay or scale
    with pytest.raises(ValueError, match='T_min_app must be finite and > 0'):
        tc.at(0.)
    with pytest.raises(ValueError, match='T_min_app must be finite and > 0'):
        tc.at(0., scale_contributions=True)
    partial = targets_of(units, 10., approach_contributions={'C1': 2.})
    at20 = partial.at(20.)
    assert at20.approach_contributions == (2., None, None, None)
    assert_allclose(at20.shift, [8., 0., 20., 20.], rtol=0, atol=0)
    assert partial.at(20., scale_contributions=True).approach_contributions == (
        4., None, None, None)
    assert at20.curves is partial.curves and at20.units is partial.units
    assert at20.labels == partial.labels
    assert t.at(0.).Q_hot < t.Q_hot
    assert t.at(10.).Q_hot.hex() == t.Q_hot.hex()

# %% Utilities

def test_two_steam_two_coolant_allocation():
    t = targets('test_targeting_ref_A', REF_A, 10.)
    assert (t.Q_hot / kW, t.Q_cold / kW, t.pinch_T, t.cut) == (
        pytest.approx(50.), pytest.approx(40.), 390., 'below')
    dT = bst.HeatUtility.dT
    LP, MP = agent('low_pressure_steam'), agent('medium_pressure_steam')
    CW, ChW = agent('cooling_water'), agent('chilled_water')
    q_LP = LP.T - LP.dT - dT - 390.
    q_CW = 330. - (CW.T + CW.dT + dT - 10.)
    utilities = t.utility_targets()
    assert [u.ID for u in utilities] == [
        'low_pressure_steam', 'medium_pressure_steam', 'cooling_water', 'chilled_water']
    assert [u.heating for u in utilities] == [True, True, False, False]
    assert_allclose([u.duty / kW for u in utilities],
                    [q_LP, 50. - q_LP, q_CW, 40. - q_CW], rtol=1e-9)
    served = [(390., 407.189), (407.189, 440.), (340., 312.372), (312.372, 300.)]
    assert_allclose([(u.T_in, u.T_out) for u in utilities], served, rtol=1e-12)
    for u in utilities:
        cost, utility_duty, T_out = fresh_price(
            u.duty if u.heating else -u.duty, u.T_in, u.T_out, u.agent)
        assert u.cost == pytest.approx(cost, rel=1e-12)
        assert u.utility_duty == pytest.approx(utility_duty, rel=1e-12)
        assert u.utility_duty == pytest.approx(
            u.duty / u.agent.heat_transfer_efficiency, rel=1e-12)
        assert u.T_utility_out == T_out
        assert u.flow > 0.
    assert by_ID(utilities)['cooling_water'].T_utility_out == CW.T_limit
    assert by_ID(utilities)['medium_pressure_steam'].level > t.table.Ts[0]

def test_list_order_rule():
    t = targets('test_targeting_list_order', REF_A, 10.)
    LP, MP = agent('low_pressure_steam'), agent('medium_pressure_steam')
    CW, ChW = agent('cooling_water'), agent('chilled_water')
    utilities = t.utility_targets([MP, LP], [ChW, CW])
    assert [(u.ID, round(u.duty / kW, 9)) for u in utilities] == [
        ('medium_pressure_steam', 50.), ('chilled_water', 40.)]
    assert [(u.T_in, u.T_out) for u in utilities] == [(390., 440.), (340., 300.)]

def test_dT_follows_biosteam(monkeypatch):
    t = targets('test_targeting_dT', REF_A, 10.)
    before = by_ID(t.utility_targets())
    monkeypatch.setattr(bst.HeatUtility, 'dT', 10.)
    after = by_ID(t.utility_targets())
    LP, CW = 'low_pressure_steam', 'cooling_water'
    assert after[LP].duty == pytest.approx(before[LP].duty - 5 * kW, rel=1e-9)
    assert after[CW].duty == pytest.approx(before[CW].duty - 5 * kW, rel=1e-9)
    assert after[CW].T_out == pytest.approx(before[CW].T_out + 5., rel=1e-12)
    for u in after.values():
        cost, _, T_out = fresh_price(u.duty if u.heating else -u.duty,
                                     u.T_in, u.T_out, u.agent)
        assert u.cost == pytest.approx(cost, rel=1e-12)
        assert u.T_utility_out == T_out

def test_unserved_warns():
    t = targets('test_targeting_unserved', REF_A, 10.)
    LP, CW = agent('low_pressure_steam'), agent('cooling_water')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        utilities = t.utility_targets([LP], [CW])
    messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
    assert len(messages) == 2
    assert 'heating agent' in messages[0] and 'kJ/hr' in messages[0]
    assert 'cooling agent' in messages[1]
    unserved = [u for u in utilities if u.ID is None]
    assert [(u.heating, u.level) for u in unserved] == [(True, math.inf),
                                                        (False, -math.inf)]
    assert_allclose([u.duty / kW for u in unserved], [32.811, 12.372], rtol=1e-9)
    for u in unserved:
        assert u.agent is None
        assert all(math.isnan(x) for x in (u.cost, u.flow, u.utility_duty,
                                           u.T_utility_in, u.T_utility_out))
    # the unserved heat is missing from the balanced composites
    hot_T, hot_H, cold_T, cold_H = t.balanced_composite_curves(utilities)
    assert_allclose(cold_H[-1] - hot_H[-1], (32.811 - 12.372) * kW, rtol=1e-9)
    # so the area, the capital cost and the total annualized cost are nan
    assert math.isnan(t.area_target(utilities))
    assert math.isnan(t.capital_cost_target(utilities)['installed_cost'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        summary = t.summary(heating_agents=[LP], cooling_agents=[CW])
    for key in ('utility_cost', 'area', 'installed_cost', 'TAC'):
        assert math.isnan(summary[key]), key
    assert summary['N_units'] == t.units_target(utilities)['total']

def test_kemp_single_LP(kemp_hxn):
    units, HXN = kemp_hxn
    t = targets_of(units, 10.)
    utilities = t.utility_targets()
    assert [(u.ID, round(u.duty / kW, 9)) for u in utilities] == [
        ('low_pressure_steam', 20.), ('cooling_water', 60.)]
    # the network serves C1 with medium pressure steam (its outlet, 135 degC,
    # is above low pressure steam's limit) and H4 with chilled water
    agents = {}
    for hx in HXN.new_HX_utils:
        for hu in hx.heat_utilities:
            if hu.agent: agents[hu.unit.ID] = hu.agent.ID
    index = {hx.ID: i for i, hx in enumerate(HXN.original_heat_exchangers)}
    assert agents == {f'Util_{index["C1"]}_hs': 'medium_pressure_steam',
                      f'Util_{index["H4"]}_cs': 'chilled_water'}
    network_cost = sum(hu.cost for hx in HXN.new_HX_utils for hu in hx.heat_utilities)
    assert network_cost > sum(u.cost for u in utilities)

def test_served_interval_and_balanced():
    t = targets('test_targeting_balanced', REF_A, 10.)
    utilities = t.utility_targets()
    hot_T, hot_H, cold_T, cold_H = t.balanced_composite_curves(utilities)
    hot = [(300, 0), (340, 40), (400, 100), (412.189, 100), (412.189, 117.189),
           (454.77, 117.189), (454.77, 150)]
    cold = [(280.372, 0), (300.372, 12.372), (305.372, 12.372), (324.817, 40),
            (330, 40), (390, 100), (440, 150)]
    assert_allclose(points(hot_T, hot_H / kW), hot, rtol=1e-9)
    assert_allclose(points(cold_T, cold_H / kW), cold, rtol=1e-9)
    assert_allclose([hot_H[-1], cold_H[-1]], 150 * kW, rtol=1e-9)
    T, B = t.balanced_grand_composite_curve(utilities)
    assert np.all(np.diff(T) <= 0) and B.min() >= -1e-9
    for level in (390., 407.189):
        assert B[T == level].min() == pytest.approx(0., abs=1e-9)
    assert B[0] == pytest.approx(by_ID(utilities)['medium_pressure_steam'].duty, rel=1e-12)
    assert B[-1] == pytest.approx(by_ID(utilities)['chilled_water'].duty, rel=1e-12)
    # cooling water serves the stretch it reaches, priced as biosteam prices
    # that stream (outlet 5 K below the process inlet), and the balanced
    # composites keep a positive driving force
    rows = [('H2', 400., 350., 1.), ('C2', 340., 390., 1.), ('H1', 322., 315., 1.)]
    t = targets('test_targeting_served', rows, 10.)
    CW = by_ID(t.utility_targets())['cooling_water']
    assert (CW.T_in, CW.T_out) == (322., 315.)
    assert CW.T_utility_out == pytest.approx(317., abs=1e-9)
    for T_min_app in (1., 5., 10.):
        tt = t.at(T_min_app)
        curves = tt.balanced_composite_curves()
        assert_allclose(curves[1][-1], curves[3][-1], rtol=1e-9)
        assert min_driving_force(*curves) > 0.
        assert 0. < tt.area_target() < math.inf

def test_pricing_is_isolated():
    # a fuel case: C1 330 -> 520 K needs natural gas above high pressure steam
    t = targets('test_targeting_isolated', [('C1', 330., 520., 1.),
                                            ('H1', 400., 300., 1.)], 10.)
    agents = bst.HeatUtility.heating_agents + bst.HeatUtility.cooling_agents
    dumps = [(a.utility_stream_dump, list(a.utility_stream_dump)) for a in agents]
    registered = lambda: {ID: id(s) for ID, s in bst.main_flowsheet.stream.data.items()}
    registry = registered()
    gas = agent('natural_gas')
    before = fresh_price(10 * kW, 450., 500., gas)[0]
    first = t.utility_targets()
    second = t.utility_targets()
    assert [u.cost for u in first] == [u.cost for u in second]
    assert [u.ID for u in first] == ['low_pressure_steam', 'medium_pressure_steam',
                                    'high_pressure_steam', 'natural_gas',
                                    'cooling_water', 'chilled_water']
    assert_allclose([u.duty / kW for u in first[:4]], [17.189, 42.581, 54.221, 16.009],
                    atol=1e-9)
    fuel = first[3]
    assert fuel.T_utility_out == pytest.approx(508.991, abs=1e-9)
    assert fuel.cost == pytest.approx(fresh_price(fuel.duty, fuel.T_in, fuel.T_out, gas)[0],
                                      rel=1e-12)
    assert registered() == registry
    assert '_hensmith_targets' not in bst.Flowsheet.flowsheet.__dict__
    for a, (dump, items) in zip(agents, dumps):
        assert a.utility_stream_dump is dump and a.utility_stream_dump == items
    assert fresh_price(10 * kW, 450., 500., gas)[0] == before

def vacuum_condenser(name, T_sat):
    """A vacuum condenser and a liquid cooler of water: 40 kmol/hr of vapour
    condensed at `T_sat` [K] (from 0.5 K above it) and 20 kmol/hr of liquid
    cooled from 345 K to `T_sat`."""
    test_hxn_targets.setup(name, ['Water'])
    P = bst.settings.chemicals.Water.Psat(T_sat)
    vapour = bst.Stream(name + '_vapour', Water=40., T=T_sat + .5, P=P, phase='g')
    condenser = bst.HXutility(name + '_condenser', ins=vapour, V=0., rigorous=True)
    liquid = bst.Stream(name + '_liquid', Water=20., T=345., P=3e5)
    cooler = bst.HXutility(name + '_cooler', ins=liquid, T=T_sat, rigorous=False)
    condenser.simulate()
    cooler.simulate()
    return [condenser, cooler]

@pytest.mark.parametrize('T_sat', [316., 330.])
def test_sensible_line_condenser(T_sat):
    # cooling water serves a condenser at the cold end of its stretch:
    # priced from the warm end alone (biosteam's rule) it would leave at its
    # limit, 324.817 K, passing the condensing temperature with most of its
    # duty taken (the balanced composites would cross)
    t = targets_of(vacuum_condenser(f'test_targeting_vacuum_{T_sat:.0f}', T_sat), 10.)
    dT, CW = bst.HeatUtility.dT, agent('cooling_water')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        utilities = t.utility_targets()
        area = t.area_target(utilities)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    (u,) = utilities
    assert u.ID == 'cooling_water' and u.duty == pytest.approx(t.Q_cold, rel=1e-12)
    # the highest outlet of a line that has taken H when it passes the knot
    # (T, H) of the hot composite at T - dT or below: the series minimum flow
    hot_T, hot_H, _, _ = t.composite_curves()
    knot = hot_H > 0.
    bound = CW.T + np.min((hot_T[knot] - dT - CW.T) * hot_H[-1] / hot_H[knot])
    T_out = min(CW.T_limit, bound)
    assert u.T_utility_out == pytest.approx(T_out, rel=1e-9)
    priced = fresh_price(-u.duty, min(u.T_in, bound + dT), u.T_out, CW)[0]
    assert u.cost == pytest.approx(priced, rel=1e-9)
    warm_end = fresh_price(-u.duty, u.T_in, u.T_out, CW)[0]
    if T_sat == 316.:
        assert T_out < T_sat - dT + 1. and u.cost > 3 * warm_end
    else:
        assert u.T_utility_out == CW.T_limit and u.cost == warm_end
    # the line keeps dT below the hot composite, so the area is finite, at
    # any T_min_app
    curves = t.balanced_composite_curves(utilities)
    assert min_vertical_gap(*curves) >= dT - 1e-6
    assert 0. < area < math.inf
    for T_min_app in (2., 20.):
        assert 0. < t.at(T_min_app).area_target() < math.inf

def test_sensible_line_constant_cp():
    dT = bst.HeatUtility.dT
    # cooling water: 40 of its 64 kW are below 316 K
    t = targets('test_targeting_line_water', [('H1', 340., 312.372, 1.),
                                              ('H2', 316., 312.372, 10.)], 10.)
    CW = agent('cooling_water')
    (u,) = t.utility_targets()
    q = 340. - 312.372 + 10. * (316. - 312.372)
    assert u.duty / kW == pytest.approx(q, rel=1e-9)
    T_out = CW.T + (316. - dT - CW.T) * q / (q - (340. - 316.))
    assert u.T_utility_out == pytest.approx(T_out, rel=1e-9) and T_out < 315.
    assert u.cost == pytest.approx(fresh_price(-u.duty, T_out + dT, u.T_out, CW)[0],
                                   rel=1e-9)
    curves = t.balanced_composite_curves([u])
    assert min_vertical_gap(*curves) == pytest.approx(dT, abs=1e-6)
    assert 0. < t.area_target([u]) < math.inf
    # a sensible heating agent against a concave cold composite: 110 of the
    # 180 kW of a hot oil (480 -> 420 K at most) are above 460 K
    t = targets('test_targeting_line_oil', [('C1', 390., 470., 1.),
                                            ('C2', 460., 470., 10.)], 10.)
    oil = bst.UtilityAgent('test_targeting_hot_oil', Water=1, T=480., P=5e6,
                           phase='l', thermo=bst.HeatUtility.thermo_water,
                           T_limit=420., heat_transfer_price=1e-6)
    (u,) = t.utility_targets([oil], [])
    assert (u.ID, u.T_in, u.T_out) == ('test_targeting_hot_oil', 390., 470.)
    T_out = oil.T - (oil.T - dT - 460.) * 180. / 110.
    assert u.T_utility_out == pytest.approx(T_out, rel=1e-9) and T_out > 455.
    cost, utility_duty, priced = fresh_price(u.duty, T_out - dT, u.T_out, oil)
    assert (u.cost, u.utility_duty) == (pytest.approx(cost, rel=1e-12),
                                        pytest.approx(utility_duty, rel=1e-12))
    assert priced == pytest.approx(T_out, rel=1e-12)
    curves = t.balanced_composite_curves([u])
    assert min_vertical_gap(*curves) == pytest.approx(dT, abs=1e-6)
    assert 0. < t.area_target([u]) < math.inf

def sensible_line_sweep(t, agents, T_min_apps):
    """``{T_min_app: utility targets}`` over a sweep of `t`, asserting that
    the balanced composite curves keep min(T_min_app, dT) apart (no warning,
    a finite area that falls with T_min_app)."""
    dT, sweep, areas = bst.HeatUtility.dT, {}, []
    for T_min_app in T_min_apps:
        at = t.at(T_min_app)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            utilities = at.utility_targets(*agents)
            areas.append(at.area_target(utilities))
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
        curves = at.balanced_composite_curves(utilities)
        assert min_driving_force(*curves) >= min(T_min_app, dT) - 1e-9
        sweep[T_min_app] = utilities
    assert np.all(np.isfinite(areas)) and np.all(np.diff(areas) < 0.)
    return sweep

def test_sensible_line_process_in_stretch():
    # a process stream of the other kind inside a sensible agent's stretch:
    # the line must keep dT from the process where the agent has taken little
    # of its duty, not only at the stretch's warm end (cooling) or cold end
    # (heating), which crossed the curves (area inf at any T_min_app)
    dT, T_min_apps = bst.HeatUtility.dT, (1., 2., 3., 5., 8., 10.)
    # chilled water serves H1 from 312.372 K (the utility pinch of cooling
    # water) to 287.372 K, but C1, from 297 K, takes most of H1's heat above
    # 297 K + T_min_app: the bound is there
    t = targets('test_targeting_line_cold_in', [('H1', 518.9, 272.3, 4.99),
                                                ('C1', 297., 450.3, 4.85)], 10.)
    CHW = agent('chilled_water')
    for T_min_app, utilities in sensible_line_sweep(t, (), T_min_apps).items():
        assert [u.ID for u in utilities] == ['cooling_water', 'chilled_water',
                                             'chilled_brine']
        u = utilities[1]
        T_C1 = 297. + T_min_app # H1 where C1 starts
        below = 4.99 * kW * (T_C1 - u.T_out) # H1's heat below it
        T_out = min(CHW.T_limit, CHW.T + (T_C1 - dT - CHW.T) * u.duty / below)
        assert u.T_utility_out == pytest.approx(T_out, rel=1e-12)
        priced = fresh_price(-u.duty, min(u.T_in, T_out + dT), u.T_out, CHW)[0]
        assert u.cost == pytest.approx(priced, rel=1e-12)
        if T_min_app <= 5.: assert T_out < CHW.T_limit - 2.
    # its mirror: a hot oil (500 K, down to 300 K at most) serves C1 from 290
    # K, but H1, up to 450 K, gives most of C1's heat below 450 K - T_min_app
    t = targets('test_targeting_line_hot_in', [('C1', 290., 470., 5.),
                                               ('H1', 450., 320., 4.85)], 10.)
    oil = bst.UtilityAgent('test_targeting_hot_oil_in', Water=1, T=500., P=5e6,
                           phase='l', thermo=bst.HeatUtility.thermo_water,
                           T_limit=300., heat_transfer_price=1e-6)
    q = (5. * 180. - 4.85 * 130.) * kW
    for T_min_app, utilities in sensible_line_sweep(t, ([oil], []), T_min_apps).items():
        (u,) = utilities
        assert (u.T_in, u.T_out) == (pytest.approx(290., abs=1e-9),
                                     pytest.approx(470., abs=1e-9))
        assert u.duty == pytest.approx(q, rel=1e-12)
        T_H1 = 450. - T_min_app # C1 where H1 starts
        above = 5. * kW * (470. - T_H1) # C1's heat above it
        T_out = oil.T - (oil.T - dT - T_H1) * q / above
        assert u.T_utility_out == pytest.approx(T_out, rel=1e-12)
        assert u.cost == pytest.approx(fresh_price(u.duty, T_out - dT, u.T_out, oil)[0],
                                       rel=1e-12)

def test_pricing_temperatures():
    dT = bst.HeatUtility.dT
    # the hot stream's top, 445 K shifted, is above every cold stream: medium
    # pressure steam serves C1 up to its outlet, 440 K
    t = targets('test_targeting_ends', [('H1', 455., 300., 1.),
                                        ('C1', 330., 440., 3.)], 10.)
    MP = by_ID(t.utility_targets())['medium_pressure_steam']
    assert (MP.T_in, MP.T_out) == (pytest.approx(407.189, abs=1e-9), 440.)
    # with approach contributions, the temperatures of the streams served:
    # the outlet within the agent's limit, as biosteam requires
    rng = np.random.default_rng(11)
    for n in range(24):
        rows = []
        for i in range(int(rng.integers(2, 6))):
            a, b = rng.uniform(290., 520., 2).round(1)
            if abs(a - b) < 2.: b = a + 5.
            rows.append((('H' if a > b else 'C') + str(i), float(a), float(b),
                         round(float(rng.uniform(.1, 5.)), 2)))
        contributions = {ID: round(float(rng.uniform(.5, 15.)), 1)
                         for ID, *_ in rows if rng.random() < .6}
        t = targets(f'test_targeting_ends_{n}', rows,
                    float(rng.choice([1., 5., 10., 20.])),
                    approach_contributions=contributions or None)
        for u in t.utility_targets():
            a = u.agent
            if a is None or a.isfuel: continue
            if u.heating:
                assert u.T_in <= u.T_out and u.T_out + dT <= a.T - a.dT + 1e-6
                if a.T_limit: assert u.T_utility_out >= u.T_in + dT - 1e-6
            else:
                assert u.T_in >= u.T_out and u.T_out - dT >= a.T + a.dT - 1e-6
                if a.T_limit: assert u.T_utility_out <= u.T_in - dT + 1e-6

# %% Area, units, capital and total annualized cost

def test_bath_balanced_pair(quickstart):
    t = targets('test_targeting_pair', PAIR, 10.)
    utilities = t.utility_targets()
    assert (t.Q_hot, t.Q_cold, utilities) == (0., 0., [])
    assert t.labels == ['C1', 'H1']
    # 100 kW, 10 K apart: U = 0.5 kW/m2/K by default, 1 / (1/2 + 1/0.5) else
    assert t.area_target(utilities) == pytest.approx(20., rel=1e-12)
    forms = [{0: .5, 1: 2.}, {'H1': 2., 'C1': .5}, {('H1', 'H1'): 2., ('C1', 'C1'): .5},
             {t.units[1]: 2., t.units[0]: .5}, [.5, 2.], np.array([.5, 2.]),
             {'H1': 1., 1: 2., 'C1': .5}] # the stream index is nearer
    for h in forms:
        assert t.area_target(utilities, h=h) == pytest.approx(25., rel=1e-12), h
    assert t.area_target(h=2.) == pytest.approx(10., rel=1e-12)
    assert t.area_target(h={'H1': 2.}) == pytest.approx(15., rel=1e-12)
    for h in ({'H9': 1.}, {2: 1.}, {-1: 1.}, {('D0', 'H1'): 1.}):
        with pytest.raises(ValueError, match='h keys that name no stream'):
            t.area_target(h=h)
    for h, message in (({0: True}, 'h of key 0 must be'), (0., 'h must be'),
                       (math.inf, 'h must be'), ([1., -1.], r'h\[1\] must be'),
                       ({'C1': math.nan}, "h of key 'C1' must be"),
                       ([1.], '1 film coefficients h given for 2')):
        with pytest.raises(ValueError, match=message):
            t.area_target(h=h)
    with pytest.raises(ValueError, match='different values at the same level'):
        t.area_target(h={'H1': 2., ('H1', 'H1'): 3.})
    with pytest.raises(TypeError, match='a heat utility cannot be a key of h'):
        t.area_target(h={t.heat_utilities[0]: 2.})
    for key in (True, 1.5, ('H1',)):
        with pytest.raises(TypeError, match='h keys must be'):
            t.area_target(h={key: 2.})
    # labels are not keys of targets built from streams
    direct = PinchTargets(t.streams_inlet, t.streams_quenched, t.is_hot, 10.,
                          labels=['C1', 'H1'])
    assert direct.area_target(h={0: .5, 1: 2.}) == pytest.approx(25., rel=1e-12)
    with pytest.raises(ValueError, match='name no stream'):
        direct.area_target(h={'H1': 2.})
    # utilities: reference A, keyed by agent ID or agent
    a = targets('test_targeting_pair_utilities', REF_A, 10.)
    utilities = a.utility_targets()
    LP = agent('low_pressure_steam')
    by_ID_ = a.area_target(utilities, h_utilities={'low_pressure_steam': 2.})
    assert by_ID_ < a.area_target(utilities)
    assert a.area_target(utilities, h_utilities={LP: 2.}) == by_ID_
    assert a.area_target(utilities, h_utilities=2.) < by_ID_
    with pytest.raises(ValueError, match='h_utilities keys that name no agent'):
        a.area_target(utilities, h_utilities={'steam': 2.})
    with pytest.raises(ValueError, match='different values'):
        a.area_target(utilities, h_utilities={LP: 2., 'low_pressure_steam': 3.})
    with pytest.raises(TypeError, match='h_utilities keys must be'):
        a.area_target(utilities, h_utilities={1: 2.})
    with pytest.raises(ValueError, match='h_utilities must be'):
        a.area_target(utilities, h_utilities=-1.)
    # auxiliary units: the owner's key reaches both, a key naming the unit
    # (owner.ID, unit.ID) is nearer
    labels = quickstart.labels
    reboiler, condenser = labels.index('D0 - reboiler'), labels.index('D0 - condenser')
    utilities = quickstart.utility_targets()
    both = quickstart.area_target(utilities, h={'D0': 2.})
    assert both == quickstart.area_target(utilities, h={reboiler: 2., condenser: 2.})
    assert both != quickstart.area_target(utilities)
    assert quickstart.area_target(utilities, h={'D0': 2., ('D0', 'reboiler'): 1.}) == (
        quickstart.area_target(utilities, h={condenser: 2.}))
    with pytest.raises(ValueError, match='name no stream'):
        quickstart.area_target(utilities, h={'reboiler': 2.})

def test_bath_independent_integration(quickstart):
    # reference A: the streams plus the four utilities
    t = targets('test_targeting_integration_A', REF_A, 10.)
    utilities = t.utility_targets()
    hot, cold = rows_members(REF_A)
    u_hot, u_cold = utility_members(utilities)
    expected = integrated_area(hot + u_hot, cold + u_cold)
    A = t.area_target(utilities)
    assert A == pytest.approx(expected, rel=1e-6)
    assert A == pytest.approx(24.144381021937, rel=1e-9)
    # KEMP with film coefficients per stream
    units = cp_units('test_targeting_integration_kemp', KEMP, K)
    t = targets_of(units, 10.)
    utilities = t.utility_targets()
    hot, cold = rows_members(KEMP, K, KEMP_H)
    u_hot, u_cold = utility_members(utilities)
    expected = integrated_area(hot + u_hot, cold + u_cold)
    A = t.area_target(utilities, h=KEMP_H)
    assert A == pytest.approx(expected, rel=1e-5)
    assert A == pytest.approx(56.139, rel=1e-5)
    assert t.area_target(utilities) == pytest.approx(59.6813, rel=1e-6)
    # KEMP with approach contributions: Bath on real temperatures with the
    # utilities allocated on the shifted ones
    t = targets_of(units, 10., approach_contributions=KEMP_C)
    utilities = t.utility_targets()
    hot, cold = rows_members(KEMP, K)
    u_hot, u_cold = utility_members(utilities)
    expected = integrated_area(hot + u_hot, cold + u_cold)
    A = t.area_target(utilities)
    assert A == pytest.approx(expected, rel=1e-6)
    assert A == pytest.approx(64.2819, rel=1e-6)
    # the quickstart at 5 K
    assert quickstart.area_target() == pytest.approx(5946.1, rel=1e-5)

def test_units_targets(quickstart):
    cases = [
        ('kemp', [(ID, T_in + K, T_out + K, CP) for ID, T_in, T_out, CP in KEMP], (4, 3)),
        ('ref_A', REF_A, (2, 3)),
        ('pair', PAIR, (0, 1)),
        ('threshold', [('H1', 400., 320., 1.), ('C1', 310., 350., 1.)], (0, 2)),
        ('cold_only', [('C1', 330., 440., 1.), ('C2', 300., 350., 2.)], (3, 0)),
        # low pressure steam is pinched at 407.189 K and touches both cold
        # streams on either side
        ('utility_pinch', [('H1', 400., 300., 2.), ('C1', 330., 440., 1.),
                           ('C2', 330., 440., 1.)], (4, 4)),
        # H2 and C2 balance: the network needs fewer units than the estimate
        ('balanced_subset', [('H2', 400., 350., 1.), ('C2', 340., 390., 1.),
                             ('H1', 325., 315., 1.)], (0, 2)),
    ]
    for name, rows, (above, below) in cases:
        t = targets('test_targeting_units_' + name, rows, 10.)
        assert t.units_target() == {'above': above, 'below': below,
                                    'total': above + below}, name
    t = targets('test_targeting_units_pinched', cases[5][1], 10.)
    LP = by_ID(t.utility_targets())['low_pressure_steam']
    assert (LP.duty / kW, LP.level) == (pytest.approx(34.378, abs=1e-9), 407.189)
    assert quickstart.units_target() == {'above': 5, 'below': 1, 'total': 6}
    # where the balanced curve stays zero, a stream across several pieces
    # counts once: a zero-duty stream given directly (a grid temperature in
    # the zero stretch) counts nowhere
    zero = cp_units('test_targeting_units_zero', [('Z', 350., 350., 1.)])[0]
    for name, rows in (('ref_A', REF_A), ('pair', PAIR)):
        t = targets('test_targeting_units_zero_' + name, rows, 10.)
        direct = PinchTargets(t.streams_inlet + zero.ins, t.streams_quenched + zero.outs,
                              [*t.is_hot, True], 10.)
        assert direct.table.Ts.size == t.table.Ts.size + 1
        assert direct.units_target() == t.units_target(), name
    # and two balanced pairs in one zero stretch are two units
    rows = [('H1', 400., 350., 1.), ('C1', 340., 390., 1.),
            ('H2', 340., 300., 1.), ('C2', 290., 330., 1.)]
    t = targets('test_targeting_units_two_pairs', rows, 10.)
    assert (t.Q_hot, t.Q_cold) == (0., 0.)
    assert t.units_target() == {'above': 0, 'below': 2, 'total': 2}

def test_exchanger_costs_match_biosteam(kemp_hxn):
    units, HXN = kemp_hxn
    seen = set()
    for hx in HXN.new_HXs + HXN.new_HX_utils:
        design = hx.design_results
        if 'Area' not in design: continue
        purchase, installed, kind = _exchanger_costs(
            design['Area'], design['Operating pressure'] * 101325 / 14.7)
        assert list(hx.purchase_costs) == [kind]
        assert purchase == pytest.approx(hx.purchase_cost, rel=1e-12)
        assert installed == pytest.approx(hx.installed_cost, rel=1e-12)
        seen.add(kind)
    assert seen == {'Double pipe', 'Floating head'}
    assert _exchanger_costs(math.inf)[:2] == (math.inf, math.inf)
    assert _exchanger_costs(1e300) == (math.inf, math.inf, None) # overflow
    assert all(math.isnan(x) for x in _exchanger_costs(math.nan)[:2])
    # a type and a material of biosteam's lists
    p, i, kind = _exchanger_costs(500., heat_exchanger_type='U tube',
                                  material='Stainless steel/stainless steel')
    assert kind == 'U tube' and p < _exchanger_costs(500., material='Stainless steel/stainless steel')[0]
    with pytest.raises(ValueError, match='heat_exchanger_type must be one of'):
        _exchanger_costs(500., heat_exchanger_type='Furnace')
    with pytest.raises(ValueError, match='material must be one of'):
        _exchanger_costs(500., material='steel')

@pytest.mark.parametrize('A, P_psi, kind, material', [
    (1., 14.7, 'Floating head', 'Carbon steel/carbon steel'), # F_l < 1
    (2.1718, 150., 'Fixed head', 'Carbon steel/carbon steel'),
    (149.999, 600., 'Floating head', 'Stainless steel/stainless steel'),
    (150., 600., 'U tube', 'Stainless steel/stainless steel'),
    (150., 14.7, 'Floating head', 'Carbon steel/carbon steel'),
    (2000., 150., 'Fixed head', 'Carbon steel/brass'),
    (5000., 600., 'Kettle vaporizer', 'Monel/Monel'),
    (1e4, 14.7, 'Double pipe', 'Titanium/titanium')])
def test_exchanger_costs_match_biosteam_design(A, P_psi, kind, material):
    # biosteam's own costing of an exchanger with these design results
    with bst.Flowsheet('test_targeting_costs'):
        hx = bst.HXprocess('test_targeting_costs_hx', thermo=bst.HeatUtility.thermo_water,
                           heat_exchanger_type=kind, material=material)
        hx.isfurnace = False
        hx.design_results.update({'Area': A, 'Total tube length': 20,
                                  'Operating pressure': P_psi})
        hx._cost()
        hx._load_costs()
    purchase, installed, kind = _exchanger_costs(A, P_psi * 101325 / 14.7, kind, material)
    assert list(hx.purchase_costs) == [kind]
    assert purchase == pytest.approx(hx.purchase_cost, rel=1e-12)
    assert installed == pytest.approx(hx.installed_cost, rel=1e-12)

def test_lmtd_nearly_equal():
    from decimal import Decimal, localcontext
    d2 = np.array([10., 10., 10., 10., 10., 10., 10., 1e-3, 7.])
    d1 = d2 * np.array([1., 1. + 3e-12, 1. + 1e-10, 1. - 1e-7, 1. + 1e-3, 1.4, .6,
                        1e6, 1e-6])
    with localcontext() as context:
        context.prec = 60
        exact = [float(a) if a == b else float((a - b) / (a / b).ln())
                 for a, b in zip(map(Decimal, d1), map(Decimal, d2))]
    assert_allclose(_lmtd(d1, d2), exact, rtol=4e-16, atol=0)
    assert_allclose(_lmtd(d2, d1), exact, rtol=4e-16, atol=0)

def test_capital_target_formula():
    from biosteam.units.heat_exchange import compute_furnace_purchase_cost
    t = targets('test_targeting_capital', REF_A, 10.)
    utilities = t.utility_targets()
    capital = t.capital_cost_target(utilities)
    N = t.units_target(utilities)['total']
    A = t.area_target(utilities)
    assert capital['area'] == A and capital['N_shells'] == N == 5
    assert capital['shell_area'] == A / N
    p, i, kind = _exchanger_costs(A * 10.763 / N)
    assert (capital['purchase_cost'], capital['installed_cost'], capital['type']) == (
        N * p, N * i, kind)
    assert capital['furnace_duty'] == 0.
    # a higher pressure costs more
    assert t.capital_cost_target(utilities, P=20e5)['installed_cost'] > capital['installed_cost']
    assert t.capital_cost_target(utilities, P=np.float64(101325))['installed_cost'] == (
        capital['installed_cost'])
    finite = 'P must be a finite number'
    for P, message in ((math.nan, finite), (math.inf, finite), (True, finite),
                       ('2 bar', finite), (0., 'P must be > 0'), (-1e6, 'P must be > 0')):
        with pytest.raises(ValueError, match=message):
            t.capital_cost_target(utilities, P=P)
        with pytest.raises(ValueError, match=message):
            t.summary(P=P)
        with pytest.raises(ValueError, match=message):
            _exchanger_costs(500., P)
    # the furnace: one for all fuel, costed on the utility duty
    t = targets('test_targeting_capital_furnace', FURNACE, 10.)
    utilities = t.utility_targets()
    assert_allclose([u.duty / kW for u in utilities[:4]], [17.189, 42.581, 54.221, 16.009],
                    atol=1e-9)
    fuel = utilities[3]
    assert fuel.ID == 'natural_gas' and fuel.T_utility_out == pytest.approx(508.991, abs=1e-9)
    assert fuel.cost == pytest.approx(fresh_price(fuel.duty, fuel.T_in, fuel.T_out, fuel.agent)[0],
                                      rel=1e-12)
    capital = t.capital_cost_target(utilities)
    N = t.units_target(utilities)['total']
    assert capital['furnace_duty'] == fuel.utility_duty
    assert capital['N_shells'] == N - 1
    A = t.area_target(utilities)
    p, i, _ = _exchanger_costs(A * 10.763 / (N - 1))
    C_b = compute_furnace_purchase_cost(fuel.utility_duty * 0.947817, bst.CE)
    assert capital['installed_cost'] == pytest.approx((N - 1) * i + C_b * 2.19, rel=1e-12)
    assert capital['purchase_cost'] == pytest.approx((N - 1) * p + C_b, rel=1e-12)
    summary = t.summary()
    assert summary['utility_cost'] == pytest.approx(sum(u.cost for u in utilities), rel=1e-12)
    assert summary['installed_cost'] == capital['installed_cost']
    # a zero minimum approach temperature: no driving force at the pinch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        zero = t.at(0.).summary()
    assert any('non-positive driving force' in str(w.message) for w in caught)
    assert zero['area'] == zero['installed_cost'] == zero['TAC'] == math.inf
    assert zero['Q_hot'] < t.Q_hot
    with pytest.raises(ValueError, match='heat_exchanger_type must be one of'):
        t.capital_cost_target(utilities, heat_exchanger_type='shell')

def test_crf_and_tea():
    from biosteam._tea import solve_payment
    assert _economics(.1, 10)[0] == solve_payment(1., .1, 10)
    assert _economics() == (pytest.approx(0.16275, abs=1e-5), 8000.)
    assert _economics(0., 10) == (0.1, 8000.)
    tea = types.SimpleNamespace(IRR=.08, duration=(2020, 2040), operating_hours=7920)
    assert _economics(tea=tea) == (solve_payment(1., .08, 20), 7920.)
    assert _economics(.05, 5, 100., tea) == (solve_payment(1., .05, 5), 100.)
    tea.operating_hours = None
    assert _economics(tea=tea)[1] == 8000.
    for kwargs, message in ((dict(years=0), 'years must be > 0'),
                            (dict(operating_hours=-1.), 'operating_hours must be >= 0'),
                            (dict(interest=-1.), 'interest must be > -1'),
                            (dict(interest=math.nan), 'interest must be a finite number'),
                            (dict(years=True), 'years must be a finite number')):
        with pytest.raises(ValueError, match=message):
            _economics(**kwargs)
    # the total annualized cost of the summary
    t = targets('test_targeting_tea', REF_A, 10.)
    tea.operating_hours = 7920
    summary = t.summary(tea=tea, interest=.12)
    crf = solve_payment(1., .12, 20)
    assert summary['annualized_capital_cost'] == crf * summary['installed_cost']
    assert summary['operating_cost'] == 7920 * summary['utility_cost']
    assert summary['TAC'] == summary['annualized_capital_cost'] + summary['operating_cost']
    with pytest.raises(TypeError, match="unexpected options: 'dT'"):
        t.summary(dT=5.)
    with pytest.raises(ValueError, match='years must be > 0'):
        t.summary(years=-1)

def test_optimum_T_min_app_closed_form():
    from biosteam.units.heat_exchange import (compute_double_pipe_purchase_cost,
                                              compute_floating_head_purchase_cost, p2)
    from biosteam.units.design_tools.specification_factors import (
        compute_shell_and_tube_material_factor, shell_and_tube_material_factor_coefficients)
    CP = 100.
    rows = [('H1', 400., 320., CP), ('C1', 310., 390., CP)]
    t = targets('test_targeting_closed_form', rows, 20.)
    thermo = agent('low_pressure_steam').thermo
    with bst.Flowsheet('test_targeting_closed_form_agents'):
        steam = bst.UtilityAgent('test_steam', Water=1, T=450, P=1e6, phase='g',
                                 heat_transfer_price=4e-7, thermo=thermo)
        refrigerant = bst.UtilityAgent('test_refrigerant', Water=1, T=280, P=1e3,
                                       phase='l', heat_transfer_price=2e-7, thermo=thermo)
    options = dict(heating_agents=[steam], cooling_agents=[refrigerant])
    def LMTD(a, b): return a if a == b else (a - b) / math.log(a / b)
    def installed(A): # [ft2], biosteam's correlations at 1 atm
        if A < 150:
            F_l = min(A / 2.1718, 1.)
            C_b = compute_double_pipe_purchase_cost(max(A, 2.1718), bst.CE)
            p = 14.7 / 600
            return C_b * (1.8 + F_l * (0.8510 + 0.1292 * p + 0.0198 * p**2) * 2 - 1.)
        p = 14.7 / 100
        F_m = compute_shell_and_tube_material_factor(
            A, *shell_and_tube_material_factor_coefficients['Carbon steel/carbon steel'])
        F = np.polyval(p2, 20) * (0.9803 + 0.018 * p + 0.0017 * p**2) * F_m
        return compute_floating_head_purchase_cost(A, bst.CE) * (3.17 + F - 1.)
    def TAC(T):
        Q = CP * (T - 10.) # each utility [kW]: C1 is above the shifted top of H1
        A = 2 * (Q / LMTD(40., 40. + Q / CP) + (80 * CP - Q) / T
                 + Q / LMTD(60. + Q / CP, 60.))
        crf = 0.1 * 1.1**10 / (1.1**10 - 1.)
        return crf * 3 * installed(A * 10.763 / 3) + 8000 * 3600 * (4e-7 * Q + 2e-7 * Q)
    for T in (12., 20., 35.):
        summary = t.at(T).summary(**options)
        assert summary['N_units'] == 3
        assert summary['TAC'] == pytest.approx(TAC(T), rel=1e-9)
    fine = np.arange(11., 60., 0.001)
    values = np.array([TAC(T) for T in fine])
    T_opt = fine[np.argmin(values)]
    assert T_opt == pytest.approx(21.595, abs=1e-6)
    assert np.all(np.diff(values[fine < T_opt]) < 0) and np.all(np.diff(values[fine > T_opt]) > 0)
    sweep = supertarget(t, np.arange(11., 61.), **options)
    assert abs(sweep.T_min_app - T_opt) <= 0.02
    assert sweep.optimum['TAC'] == pytest.approx(TAC(sweep.T_min_app), rel=1e-9)
    assert not sweep.at_bound
    assert sweep.options == options
    assert repr(sweep) == ('<Supertargets: 74 points, T_min_app 11-60 K, optimum '
                           '21.59 K (TAC 9.164e+04 USD/yr)>')

def test_sweep(monkeypatch):
    import hensmith.hxn_synthesis
    t = targets('test_targeting_sweep', REF_A, 10.)
    def flash(*args, **kwargs): raise AssertionError('the curves are rebuilt')
    monkeypatch.setattr(hensmith.hxn_synthesis, 'stream_curves', flash)
    sweep = supertarget(t, [20., 5., 10., 5.], refine=1, n_refine=3)
    assert isinstance(sweep, Supertargets) and sweep.targets is t
    T = sweep['T_min_app']
    assert np.all(np.diff(T) > 0) and T.size == 3 + 3
    assert_allclose(sweep['TAC'], [r['TAC'] for r in sweep.records], rtol=0, atol=0)
    frame = sweep.to_frame()
    assert frame.index.name == 'T_min_app' and list(frame.index) == list(T)
    assert list(frame.columns) == ['Q_hot', 'Q_cold', 'utility_cost', 'area', 'N_units',
                                   'N_shells', 'purchase_cost', 'installed_cost',
                                   'annualized_capital_cost', 'operating_cost', 'TAC']
    assert sweep.optimum['TAC'] == frame['TAC'].min()
    assert sweep.T_min_app == sweep.optimum['T_min_app']
    assert sweep.at_bound == (sweep.T_min_app in (T[0], T[-1]))
    single = supertarget(t, [10.], refine=2)
    assert len(single.records) == 1 and single.at_bound and 'at the bound' in repr(single)
    assert single.optimum == t.summary()
    # the options reach every evaluation
    hours = supertarget(t, [5., 10.], refine=0, operating_hours=0.)
    assert np.all(hours['operating_cost'] == 0.)
    LP, MP = agent('low_pressure_steam'), agent('medium_pressure_steam')
    listed = supertarget(t, [5., 10.], refine=0, heating_agents=[MP, LP])
    assert listed.records == supertarget(t, [5., 10.], refine=0,
                                         heating_agents=iter([MP, LP])).records
    assert listed.records != supertarget(t, [5., 10.], refine=0).records
    # validation, before sweeping
    for kwargs, error, message in (
            (dict(T_min_apps=[]), ValueError, 'non-empty'),
            (dict(T_min_apps=[[5.]]), ValueError, 'non-empty'),
            (dict(T_min_apps=[5., math.nan]), ValueError, 'finite'),
            (dict(T_min_apps=[-1., 5.]), ValueError, '>= 0 K'),
            (dict(refine=-1), ValueError, 'refine must be an int >= 0'),
            (dict(refine=1.), ValueError, 'refine must be an int >= 0'),
            (dict(n_refine=0), ValueError, 'n_refine must be an int >= 1'),
            (dict(n_refine=True), ValueError, 'n_refine must be an int >= 1'),
            (dict(dT=5.), TypeError, 'unexpected options'),
            (dict(h={'H9': 1.}), ValueError, 'name no stream'),
            (dict(h_utilities={'steam': 1.}), ValueError, 'name no agent'),
            (dict(material='steel'), ValueError, 'material must be'),
            (dict(P=math.nan), ValueError, 'P must be a finite number'),
            (dict(P='2 bar'), ValueError, 'P must be a finite number'),
            (dict(P=True), ValueError, 'P must be a finite number'),
            (dict(P=0.), ValueError, 'P must be > 0'),
            (dict(P=-1e6), ValueError, 'P must be > 0')):
        with pytest.raises(error, match=message):
            supertarget(t, **kwargs)
    # no finite TAC anywhere (the cooling is unserved): the optimum is
    # meaningless, and the sweep says so
    with pytest.warns(RuntimeWarning, match='no minimum approach temperature of the '
                      'sweep has a finite TAC'):
        unserved = supertarget(t, [5., 10.], refine=1, cooling_agents=[])
    assert np.all(np.isnan(unserved['TAC'])) and unserved.T_min_app == 5.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        supertarget(t, [5., 10.], refine=0)
    assert not [w for w in caught if 'finite TAC' in str(w.message)]
    # every stream with an approach contribution: T_min_app changes nothing
    # unless the contributions scale with it
    monkeypatch.undo()
    units = cp_units('test_targeting_sweep_kemp', KEMP, K)
    tc = targets_of(units, 10., approach_contributions=KEMP_C)
    with pytest.raises(ValueError, match='scale_contributions=True'):
        supertarget(tc)
    start = time.perf_counter()
    sweep = supertarget(tc, scale_contributions=True)
    assert time.perf_counter() - start < 20.
    assert len(sweep.records) > 60 and sweep.scale_contributions
    assert np.ptp(sweep['TAC']) > 0. and np.ptp(sweep['Q_hot']) > 0.
    record = sweep.records[0]
    assert record == tc.at(record['T_min_app'], scale_contributions=True).summary()
    partial = targets_of(units, 10., approach_contributions={'C1': 1.})
    with pytest.raises(ValueError, match='> 0 K with approach contributions'):
        supertarget(partial, [0., 5.])
    assert np.ptp(supertarget(partial, [5., 20.], refine=0)['Q_hot']) > 0.

# %% Facility and report

def terminal_area(hx):
    """Area [m2] of an exchanger of constant heat capacity streams from its
    terminal temperatures, with U = 0.5 kW/m2/K and the process duty."""
    if isinstance(hx, bst.HXprocess):
        hot = 0 if hx.ins[0].H > hx.outs[0].H else 1
        hi, ho = hx.ins[hot], hx.outs[hot]
        ci, co = hx.ins[1 - hot], hx.outs[1 - hot]
        Q = hi.H - ho.H
        d1, d2 = hi.T - co.T, ho.T - ci.T
    else:
        hu = hx.heat_utilities[0]
        pi, po = hx.ins[0], hx.outs[0]
        ui, uo = hu.inlet_utility_stream.T, hu.outlet_utility_stream.T
        Q = abs(po.H - pi.H)
        if po.H > pi.H: d1, d2 = ui - po.T, uo - pi.T
        else: d1, d2 = po.T - ui, pi.T - uo
    LMTD = d1 if abs(d1 - d2) < 1e-9 else (d1 - d2) / math.log(d1 / d2)
    return Q / kW / (0.5 * LMTD)

def test_facility_targets_bitwise(kemp_hxn, kemp_hxn_replaced):
    def assert_synthesis_targets(t, HXN):
        info = HXN.synthesis_info
        assert (t.Q_hot, t.Q_cold) == (info['Q_hot_target'], info['Q_cold_target'])
    for k, contributions in enumerate((None, {'C1': 2.})):
        units = cp_units(f'test_targeting_facility_{k}', KEMP, K)
        HXN = HeatExchangerNetwork('HXN', T_min_app=10., units=units,
                                   approach_contributions=contributions)
        before = HXN.targets() # from the units' heat utilities
        sys = bst.System.from_units('sys', units=[*units, HXN])
        simulate_strictly(sys)
        after = HXN.targets() # from the network's process streams
        for t in (before, after):
            assert_synthesis_targets(t, HXN)
            assert t.labels == ['C1', 'C3', 'H2', 'H4']
            assert t.T_min_app == 10. and t.force_ideal_thermo is False
            assert t.approach_contributions == (None if contributions is None
                                                else (2., None, None, None))
        assert all(u is hx for u, hx in zip(after.units, HXN.original_heat_exchangers))
        assert all(a is b for a, b in zip(after.heat_utilities, HXN.original_heat_utils))
        # another minimum approach temperature (the facility's is unchanged)
        assert HXN.targets(20.).Q_hot == after.at(20.).Q_hot != after.Q_hot
        assert HXN.T_min_app == 10.
    assert_synthesis_targets(kemp_hxn[1].targets(), kemp_hxn[1])
    # the units' heat utilities now carry the network's (H2 and C3 are served
    # by process exchangers alone), but the targets are those of the
    # original streams
    units, HXN = kemp_hxn_replaced
    assert sorted(u.ID for u in units if not u.heat_utilities[0].duty) == ['C3', 'H2']
    t = HXN.targets()
    assert_synthesis_targets(t, HXN)
    assert t.labels == ['C1', 'C3', 'H2', 'H4']
    with pytest.raises(RuntimeError, match='before targeting'):
        HeatExchangerNetwork('test_targeting_alone').targets()

def test_facility_targets_replaced_callable():
    # replace_unit_heat_utilities overwrites the heat utilities a callable
    # reads (the network's utility, or none for H2 and C3, which the process
    # exchangers serve): after a simulation the targets take the values that
    # simulation resolved, without calling it again
    # (H4 ends on chilled water)
    cases = [(lambda hu: hu.duty < 0, (None, None, 2., 2.)),
             (lambda hu: hu.agent.ID == 'cooling_water', (None, None, 2., None))]
    for k, (read, expected) in enumerate(cases):
        calls = []
        def f(hu):
            calls.append(hu)
            return 2. if read(hu) else None
        units = cp_units(f'test_targeting_replaced_callable_{k}', KEMP, K)
        sys, HXN = simulate_HXN(units, 10., replace_unit_heat_utilities=True,
                                approach_contributions=f)
        assert sorted(u.ID for u in units if not u.heat_utilities[0].duty) == ['C3', 'H2']
        info = HXN.synthesis_info
        assert info['status'] == 'mer'
        N_calls = len(calls)
        t = HXN.targets()
        assert len(calls) == N_calls
        assert t.approach_contributions == HXN._resolved_contributions == expected
        assert (t.Q_hot, t.Q_cold) == (info['Q_hot_target'], info['Q_cold_target'])
        assert HXN.targets(20.).Q_hot == t.at(20.).Q_hot
        report = network_report(HXN)
        assert report['targets']['Q_hot'] == report['Q_hot'] == info['Q_hot_target']
    # a callable set since applies from the next simulation; a mapping
    # (resolved by unit) at once
    HXN.approach_contributions = lambda hu: 3.
    assert HXN.targets().approach_contributions == (None, None, 2., None)
    HXN.approach_contributions = {'H2': 3.}
    assert HXN.targets().approach_contributions == (None, None, 3., None)

#: Keys of `network_report`.
REPORT_KEYS = ['T_min_app', 'targets_T_min_app', 'status', 'applied', 'Q_hot', 'Q_cold',
               'utilities', 'utility_cost', 'N_units', 'N_process_units', 'N_utility_units',
               'area', 'area_biosteam', 'exchanger_areas', 'purchase_cost', 'installed_cost',
               'annualized_capital_cost', 'operating_cost', 'TAC', 'targets']

def test_network_report(kemp_hxn):
    units, HXN = kemp_hxn
    report = network_report(HXN)
    assert list(report) == REPORT_KEYS
    assert report['applied'] is True and report['status'] == 'mer'
    assert report['T_min_app'] == report['targets_T_min_app'] == 10.
    exchangers = HXN.new_HXs + HXN.new_HX_utils
    assert report['installed_cost'] == pytest.approx(
        sum(hx.installed_cost for hx in exchangers), rel=1e-12)
    assert report['purchase_cost'] == pytest.approx(
        sum(hx.purchase_cost for hx in exchangers), rel=1e-12)
    sized = [hx for hx in exchangers if 'Area' in hx.design_results]
    assert report['area_biosteam'] == pytest.approx(
        sum(hx.design_results['Area'] for hx in sized) / 10.763, rel=1e-12)
    # like-for-like areas: the medium pressure steam heater (efficiency 0.9)
    # is sized on the process duty
    areas = report['exchanger_areas']
    assert sorted(areas) == sorted(hx.ID for hx in sized)
    for hx in sized:
        assert areas[hx.ID][0] == pytest.approx(terminal_area(hx), rel=1e-9), hx.ID
    MP = [hx for hx in HXN.new_HX_utils if hx.ID in areas
          and hx.heat_utilities[0].agent.ID == 'medium_pressure_steam']
    assert len(MP) == 1 and MP[0].heat_utilities[0].duty == pytest.approx(72000. / 0.9)
    assert round(report['area'], 2) == 54.06
    assert report['area'] < report['targets']['area']
    # the network has fewer units than the estimate (H2 and C3 balance)
    assert (report['N_units'], report['N_process_units'], report['N_utility_units']) == (6, 4, 2)
    assert report['targets']['N_units'] == 7
    assert report['Q_hot'] >= report['targets']['Q_hot'] * (1 - 1e-9)
    assert report['Q_hot'] == pytest.approx(20 * kW, rel=1e-9)
    assert report['Q_cold'] == pytest.approx(60 * kW, rel=1e-9)
    assert sorted(report['utilities']) == ['chilled_water', 'medium_pressure_steam']
    steam = report['utilities']['medium_pressure_steam']
    assert steam['duty'] == pytest.approx(20 * kW) and steam['utility_duty'] == pytest.approx(20 * kW / 0.9)
    assert report['utility_cost'] == pytest.approx(
        sum(u['cost'] for u in report['utilities'].values()), rel=1e-12)
    assert report['utility_cost'] > report['targets']['utility_cost']
    crf = _economics()[0]
    assert report['annualized_capital_cost'] == crf * report['installed_cost']
    assert report['operating_cost'] == 8000. * report['utility_cost']
    assert report['TAC'] == report['annualized_capital_cost'] + report['operating_cost']
    assert report['targets'] == HXN.targets().summary()
    # options: film coefficients and the system's operating hours
    halved = network_report(HXN, h=2., h_utilities=2.)
    assert halved['area'] == pytest.approx(report['area'] / 2, rel=1e-12)
    assert halved['targets']['area'] == pytest.approx(report['targets']['area'] / 2, rel=1e-12)
    HXN.system.operating_hours = 4000.
    try:
        hours = network_report(HXN)
        assert hours['operating_cost'] == 4000. * report['utility_cost']
        assert hours['targets']['operating_cost'] == 4000. * report['targets']['utility_cost']
        assert network_report(HXN, operating_hours=100.)['operating_cost'] == (
            100. * report['utility_cost'])
    finally:
        HXN.system.operating_hours = None
    with pytest.raises(TypeError, match='unexpected options'):
        network_report(HXN, hours=1.)
    with pytest.raises(RuntimeError, match='simulate the heat exchanger network'):
        network_report(HeatExchangerNetwork('test_targeting_report_alone'))
    # targets of the same streams in the facility's stream order (KEMP's own
    # order arranges like the facility's), and of other ones
    hus = [u.heat_utilities[0] for u in units]
    same = PinchTargets.from_heat_utilities(hus, 10.)
    assert network_report(HXN, same)['targets'] == report['targets']
    with pytest.raises(ValueError, match="of the network's process streams"):
        network_report(HXN, PinchTargets.from_heat_utilities(hus[::-1], 10.))
    with pytest.raises(ValueError, match='force_ideal_thermo'):
        network_report(HXN, PinchTargets.from_heat_utilities(hus, 10., force_ideal_thermo=True))
    inlet = HXN.original_heat_exchangers[0].ins[0]
    T = inlet.T
    try: # a stale state
        inlet.T = T + 1.
        with pytest.raises(ValueError, match='current states'):
            network_report(HXN, same)
    finally:
        inlet.T = T
    assert network_report(HXN, same)['targets'] == report['targets']
    # no process match: the report describes the utility exchangers
    rows = [('H1', 320., 300., 1.), ('C1', 330., 340., 1.)]
    _, alone = simulate_HXN(cp_units('test_targeting_report_utilities', rows), 10.)
    report = network_report(alone)
    assert report['applied'] is False and alone.installed_costs['Heat exchangers'] == 0.
    assert (report['N_units'], report['N_process_units']) == (2, 0)
    for hx in alone.new_HX_utils:
        assert report['exchanger_areas'][hx.ID][0] == pytest.approx(terminal_area(hx), rel=1e-9)
    assert report['installed_cost'] == sum(hx.installed_cost for hx in alone.new_HX_utils) > 0.
    # a zero (or crossed) driving force
    with pytest.warns(RuntimeWarning, match='HX transfers heat at a driving force of 0 K'):
        assert _profile_area(np.array([0., 1.]), np.array([5., 0.]), 2., 'HX') == math.inf
    assert _profile_area(np.array([0., kW]), np.array([5., 5.]), 2., 'HX') == 2. / 5.

def flashed_T(stream, H):
    """Temperature [K] of a copy of `stream` at equilibrium at the enthalpy
    `H` [kJ/hr] and its own pressure."""
    s = _copy(stream)
    s.vle(H=H, P=s.P)
    return s.T

def independent_area(hx, n=41):
    """
    Area [m2] of an exchanger of a network with films of 1 kW/m2/K on both
    sides, integrated independently of the targets: the process duty in n -
    1 equal parts, the temperatures of its streams at each boundary from
    flashes of copies of its own outlet and inlet streams (on a branch,
    the branch's flow), log-mean in between; a utility's temperature linear
    in the duty from its inlet (at the process outlet) to its outlet.
    """
    if isinstance(hx, bst.HXprocess):
        hot = 0 if hx.ins[0].H > hx.outs[0].H else 1
        hot_in, cold_out = hx.ins[hot], hx.outs[1 - hot]
        x = np.linspace(0., hot_in.H - hx.outs[hot].H, n)
        dT = np.array([flashed_T(hot_in, hot_in.H - q) - flashed_T(cold_out, cold_out.H - q)
                       for q in x])
    else:
        hu = hx.heat_utilities[0]
        s_in, s_out = hx.ins[0], hx.outs[0]
        heating = s_out.H > s_in.H
        x = np.linspace(0., abs(s_out.H - s_in.H), n)
        T = np.array([flashed_T(s_out, s_out.H - q if heating else s_out.H + q) for q in x])
        T_a, T_b = hu.inlet_utility_stream.T, hu.outlet_utility_stream.T
        T_u = T_a + (T_b - T_a) * x / x[-1]
        dT = T_u - T if heating else T - T_u
    d1, d2 = dT[:-1], dT[1:]
    LMTD = np.where(np.abs(d1 - d2) <= 1e-9 * d1, d1, (d1 - d2) / np.log(d1 / d2))
    return float(np.sum(np.diff(x) / LMTD)) * 2. / kW

def changes_phase(hx):
    """Whether a stream of `hx` leaves it at another vapor fraction."""
    return any(abs(s_in.vapor_fraction - s_out.vapor_fraction) > 1e-6
               for s_in, s_out in zip(hx.ins, hx.outs))

def test_network_report_independent_areas(quickstart_hxn):
    # The report integrates each exchanger along the curves of its streams
    # in the targets; an independent integration on the network's own
    # streams agrees, across the breakpoints of a phase change inside an
    # exchanger and on the branches of a split (a branch stage takes its
    # fraction of the whole stream's curve). The quickstart network boils
    # and condenses streams in process and utility exchangers alike.
    _, split, _ = liquids('test_targeting_report_split', stream_splitting=True)
    branches = {st.unit.ID: st.fraction for lc in split.stream_life_cycles
                for st in lc.life_cycle if st.fraction != 1.}
    assert sorted(branches) == ['HX_0_2_hs', 'HX_0_3_hs']
    assert_allclose(sorted(branches.values()), [0.4799, 0.5201], atol=1e-4)
    for HXN, rtol in ((split, 1e-4), (quickstart_hxn[1], 1e-3)):
        report = network_report(HXN)
        exchangers = {hx.ID: hx for hx in HXN.new_HXs + HXN.new_HX_utils}
        areas = report['exchanger_areas']
        assert len(areas) == report['N_units']
        for ID, (area, _) in areas.items():
            assert area == pytest.approx(independent_area(exchangers[ID]), rel=rtol), ID
        if HXN is not split:
            boiling = [ID for ID in areas if changes_phase(exchangers[ID])]
            assert any(ID.startswith('HX_') for ID in boiling)
            assert any(ID.startswith('Util_') for ID in boiling)

def test_network_report_current_states():
    # With cache_network, the network re-simulated at changed states is
    # reported at those states
    units = cp_units('test_targeting_report_cache', KEMP, K)
    sys, HXN = simulate_HXN(units, 10., cache_network=True)
    network = HXN.HXN_sys
    C1 = units[0]
    inlet = C1.ins[0]
    inlet.F_mass *= 1.05
    simulate_strictly(sys)
    assert HXN.HXN_sys is network
    report = network_report(HXN)
    assert report['targets'] == HXN.targets().summary()
    # (the reused structure no longer reaches the energy targets)
    assert report['Q_hot'] > report['targets']['Q_hot'] * (1 + 1e-3)
    exchangers = {hx.ID: hx for hx in HXN.new_HXs + HXN.new_HX_utils}
    for ID, (area, _) in report['exchanger_areas'].items():
        assert area == pytest.approx(terminal_area(exchangers[ID]), rel=1e-9), ID
    # a stream changed since the facility's last simulation (or its unit
    # simulated alone): the network is not of the current states
    T, F = inlet.T, inlet.F_mass
    for change in ('T', 'F', 'unit'):
        if change == 'F': inlet.F_mass = 1.5 * F
        else: inlet.T = T + 15.
        if change == 'unit': C1.simulate()
        with pytest.raises(ValueError, match='not of the current states of C1.*simulate the system'):
            network_report(HXN)
        inlet.T, inlet.F_mass = T, F
        C1.simulate()
        assert network_report(HXN) == report
    # an inlet at the enthalpy reference (298.15 K) has no enthalpy at any
    # flow: the flows are compared too
    rows = [('H1', 400., 320., 1.), ('C1', 298.15, 370., 1.)]
    sys, HXN = simulate_HXN(cp_units('test_targeting_report_reference', rows), 10.)
    inlet = HXN.original_heat_exchangers[0].ins[0]
    assert inlet.ID == 'C1_in' and inlet.H == 0.
    targets = HXN.targets()
    inlet.F_mass *= 2.
    assert inlet.H == 0.
    with pytest.raises(ValueError, match='targets are not of the current states of C1'):
        network_report(HXN, targets)
    with pytest.raises(ValueError, match='simulate the system'):
        network_report(HXN)
    # with ideal thermodynamics, on the curves' basis: stale targets, and a
    # network older than the targets
    sys, HXN, units = liquids('test_targeting_report_ideal', force_ideal_thermo=True)
    stale = HXN.targets()
    inlet = units[0].ins[0]
    inlet.T += 10.
    simulate_strictly(sys)
    with pytest.raises(ValueError, match='targets are not of the current states of H1'):
        network_report(HXN, stale)
    targets = HXN.targets()
    assert network_report(HXN, targets)['targets'] == network_report(HXN)['targets']
    inlet.T -= 10.
    units[0].simulate()
    with pytest.raises(ValueError, match='simulate the system'):
        network_report(HXN)
    with pytest.raises(ValueError, match='targets are not of the current states of H1'):
        network_report(HXN, targets)

def test_no_side_effects(kemp_hxn):
    units, HXN = kemp_hxn
    # the furnace case of test_capital_target_formula: the network heats C1
    # in a furnace
    _, furnace_HXN = simulate_HXN(cp_units('test_targeting_side_effects', FURNACE), 10.)
    report = network_report(furnace_HXN)
    assert 'natural_gas' in report['utilities']
    furnace = [hx.ID for hx in furnace_HXN.new_HX_utils
               if hx.heat_utilities and hx.heat_utilities[0].agent
               and hx.heat_utilities[0].agent.isfuel]
    assert len(furnace) == 1 and report['exchanger_areas'][furnace[0]] == (0., 0.)
    gas = agent('natural_gas')
    def furnace_cost():
        # priced with the agent's own stream dump (on which it leaves its
        # streams), in a temporary flowsheet
        with bst.Flowsheet('test_targeting_furnace_cost'):
            hu = bst.HeatUtility()
            hu(10 * kW, 450., 500., agent=gas)
            cost = hu.cost
            hu.empty()
        return cost
    furnace_cost()
    before = furnace_cost()
    facilities = (HXN, furnace_HXN)
    Q = [[float(hx.Q).hex() for hx in facility.new_HXs] for facility in facilities]
    agents = bst.HeatUtility.heating_agents + bst.HeatUtility.cooling_agents
    dumps = [(a.utility_stream_dump, list(a.utility_stream_dump)) for a in agents]
    registered = lambda: {ID: id(s) for ID, s in bst.main_flowsheet.stream.data.items()}
    registry = registered()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for facility in facilities:
            facility.targets().summary()
            network_report(facility)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert registered() == registry
    for a, (dump, items) in zip(agents, dumps):
        assert a.utility_stream_dump is dump and a.utility_stream_dump == items
    assert furnace_cost() == before
    for facility in facilities: simulate_strictly(facility.system)
    assert [[float(hx.Q).hex() for hx in facility.new_HXs] for facility in facilities] == Q

# %% Plots

def test_plot_composite_curves_gids(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = targets('test_targeting_plots', REF_A, 10.)
    for kind in ('real', 'shifted', 'balanced'):
        fig, ax = plot_composite_curves(t, kind=kind)
        try:
            lines = {a.get_gid(): a for a in ax.get_lines()}
            if kind == 'balanced':
                hot_T, hot_H, cold_T, cold_H = t.balanced_composite_curves()
                hot_line = lines['Hot balanced composite']
                cold_line = lines['Cold balanced composite']
                labels = _gid_artists(ax, 'Utility:')
                assert len(labels) == 4
                assert not _gid_artists(ax, 'Fuel')
            else:
                hot_T, hot_H, cold_T, cold_H = t.composite_curves(kind == 'shifted')
                hot_line = lines['Hot composite']
                cold_line = lines['Cold composite']
                assert len(_gid_artists(ax, 'Heat recovered')) == 1
                assert len(_gid_artists(ax, 'Hot utility')) == 1
                assert len(_gid_artists(ax, 'Cold utility')) == 1
            assert_allclose(hot_line.get_xdata(), hot_H, rtol=0, atol=0)
            assert_allclose(hot_line.get_ydata(), hot_T - 273.15, rtol=0, atol=0)
            assert_allclose(cold_line.get_xdata(), cold_H, rtol=0, atol=0)
            assert_allclose(cold_line.get_ydata(), cold_T - 273.15, rtol=0, atol=0)
            pinch = _gid_artists(ax, 'Pinch')
            assert len(pinch) == (kind == 'shifted')
            if pinch:
                assert pinch[0].get_xdata()[0] == pytest.approx(100 * kW)
                assert pinch[0].get_ydata()[0] == pytest.approx(390. - 273.15)
        finally:
            plt.close(fig)
    with pytest.raises(ValueError, match="kind must be"):
        plot_composite_curves(t, kind='hot')
    # fuel is labeled, not drawn
    furnace = targets('test_targeting_plots_fuel', [('C1', 330., 520., 1.),
                                                    ('H1', 400., 300., 1.)], 10.)
    file = tmp_path / 'balanced.png'
    fig, ax = plot_composite_curves(furnace, kind='balanced', file=str(file))
    try:
        assert file.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        fuel = _gid_artists(ax, 'Fuel')
        assert len(fuel) == 1 and 'natural_gas (furnace)' in fuel[0].get_text()
        hot = [a for a in ax.get_lines() if a.get_gid() == 'Hot balanced composite'][0]
        assert np.all(np.isfinite(hot.get_ydata()))
    finally:
        plt.close(fig)

def test_plot_gcc_gids():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = targets('test_targeting_plot_gcc', REF_A, 10.)
    utilities = t.utility_targets()
    fig, ax = plot_grand_composite_curve(t, utilities)
    try:
        T, H = t.grand_composite_curve()
        gcc, = _gid_artists(ax, 'GCC')
        assert_allclose(gcc.get_xdata(), H, rtol=0, atol=0)
        assert_allclose(gcc.get_ydata(), T - 273.15, rtol=0, atol=0)
        pinch, = _gid_artists(ax, 'Pinch')
        assert (pinch.get_xdata()[0], pinch.get_ydata()[0]) == (0., 390. - 273.15)
        segments = {a.get_gid(): a for a in _gid_artists(ax, 'Utility:')}
        assert len(segments) == 4
        for u in utilities:
            x, y = segments['Utility:' + u.ID].get_data()
            assert_allclose(y, u.level - 273.15, rtol=1e-12)
            assert x[1] - x[0] == pytest.approx(u.duty, rel=1e-12)
        assert len(_gid_artists(ax, 'Label:')) == 4
    finally:
        plt.close(fig)
    fig, ax = plot_grand_composite_curve(t, utilities=False)
    try:
        assert not _gid_artists(ax, 'Utility:') and not _gid_artists(ax, 'Label:')
    finally:
        plt.close(fig)

def test_plot_readability():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def boxes(fig, artists):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        return [a.get_window_extent(renderer) for a in artists]
    # fuel above high pressure steam: the label of the furnace sits above
    # that of the top flat, which ends where the finite hot curve does
    furnace = targets('test_targeting_plot_readable_fuel', FURNACE, 10.)
    fig, ax = plot_composite_curves(furnace, kind='balanced')
    try:
        fuel, = _gid_artists(ax, 'Fuel')
        steam, = _gid_artists(ax, 'Utility:high_pressure_steam')
        assert fuel.xy == pytest.approx(steam.xy, rel=1e-9)
        fuel, steam, frame = boxes(fig, [fuel, steam, ax])
        assert not fuel.overlaps(steam) and fuel.y0 >= steam.y1 - 1.
        assert frame.y1 >= fuel.y1
    finally:
        plt.close(fig)
    # the pinch is marked only where the shifted composites touch: not for a
    # threshold problem whose composites end apart, nor for hot streams
    # only (whose empty cold composite stays out of the legend); a balanced
    # pair touches at the top without any utility
    threshold = targets('test_targeting_plot_threshold', [('H1', 400., 300., 2.),
                                                          ('C1', 330., 380., 1.)], 10.)
    hot = targets('test_targeting_plot_hot', [('H1', 400., 300., 1.),
                                              ('H2', 380., 320., 1.)], 10.)
    pair = targets('test_targeting_plot_pair', PAIR, 10.)
    assert threshold.Q_hot == hot.Q_hot == pair.Q_hot == 0.
    for t, pinch, legend in ((threshold, False, ['heat recovered', 'hot composite', 'cold composite']),
                             (hot, False, ['hot composite']),
                             (pair, True, ['heat recovered', 'hot composite', 'cold composite', 'pinch'])):
        fig, ax = plot_composite_curves(t, kind='shifted')
        try:
            assert len(_gid_artists(ax, 'Pinch')) == pinch
            assert [x.get_text() for x in ax.get_legend().get_texts()] == legend
        finally:
            plt.close(fig)
    # the unserved remainders, without which the balanced curves end apart
    t = targets('test_targeting_plot_unserved', REF_A, 10.)
    with pytest.warns(RuntimeWarning, match='left unserved'):
        utilities = t.utility_targets([agent('low_pressure_steam')], [agent('cooling_water')])
    fig, ax = plot_composite_curves(t, kind='balanced', utilities=utilities)
    try:
        label, = _gid_artists(ax, 'Utility:unserved')
        assert label.get_text() == ('unserved heating 1.18E5 kJ/hr (not drawn)\n'
                                    'unserved cooling 4.45E4 kJ/hr (not drawn)')
    finally:
        plt.close(fig)
    # the grand composite curve at its default size: heats of six digits
    # [kJ/hr] in scientific notation, whose tick labels do not run together,
    # and every label off the curve (which meets the end of a step)
    t = targets('test_targeting_plot_readable_gcc', REF_A, 10.)
    fig, ax = plot_grand_composite_curve(t)
    try:
        lo, hi = ax.get_xlim()
        ticks = [x for x in ax.get_xticklabels() if lo <= x.get_position()[0] <= hi]
        assert ax.get_xticks()[-1] > 1e5
        labels = _gid_artists(ax, 'Label:')
        assert len(labels) == 4
        boxed = boxes(fig, ticks + labels)
        tick_boxes, label_boxes = boxed[:len(ticks)], boxed[len(ticks):]
        assert ax.xaxis.get_offset_text().get_text() == '1e5'
        assert all(not a.overlaps(b) for a, b in zip(tick_boxes, tick_boxes[1:]))
        gcc, = _gid_artists(ax, 'GCC')
        xy = ax.transData.transform(np.column_stack(gcc.get_data()))
        curve = np.concatenate([np.linspace(a, b, 500) for a, b in zip(xy, xy[1:])])
        for box, label in zip(label_boxes, labels):
            inside = ((curve[:, 0] >= box.x0) & (curve[:, 0] <= box.x1)
                      & (curve[:, 1] >= box.y0) & (curve[:, 1] <= box.y1))
            assert not inside.any(), label.get_text()
    finally:
        plt.close(fig)

def test_plot_supertargets_gids(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = targets('test_targeting_plot_sweep', REF_A, 10.)
    sweep = supertarget(t, [0., 5., 10., 20.], refine=1, n_refine=2)
    assert sweep['TAC'][0] == math.inf
    file = tmp_path / 'sweep.png'
    fig, ax = plot_supertargets(sweep, file=str(file))
    try:
        assert file.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        T = sweep['T_min_app']
        for gid, key in (('TAC', 'TAC'), ('Capital', 'annualized_capital_cost'),
                         ('Operating', 'operating_cost')):
            line, = _gid_artists(ax, gid)
            assert_allclose(line.get_xdata(), T, rtol=0, atol=0)
            assert_allclose(line.get_ydata(), sweep[key], rtol=0, atol=0)
        optimum, = _gid_artists(ax, 'Optimum')
        assert (optimum.get_xdata()[0], optimum.get_ydata()[0]) == (
            sweep.T_min_app, sweep.optimum['TAC'])
    finally:
        plt.close(fig)
