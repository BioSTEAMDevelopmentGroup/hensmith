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
C. Plots: stable gids whose data equal the curves.

Reference A: H1 400 -> 300 K and C1 330 -> 440 K, both 1 kW/K, at
T_min_app = 10 K: 50 kW hot and 40 kW cold utility, shifted pinch 390 K.
"""
import inspect
import math
import warnings
import numpy as np
import pytest
import biosteam as bst
from numpy.testing import assert_allclose
from hensmith import PinchTargets, plot_composite_curves, plot_grand_composite_curve
from hensmith.hxn_synthesis import _problem_table, temperature_interval_pinch_analysis
from test_hxn import cp_units, KEMP, simulate_HXN, _gid_artists
from test_hxn_contributions import classic_cascade
import test_hxn_targets

K = 273.15
kW = 3600.  # kJ/hr
#: Reference A (temperatures in K, CP in kW/K).
REF_A = [('H1', 400., 300., 1.), ('C1', 330., 440., 1.)]
#: Contributions of the KEMP reference with approach contributions [K].
KEMP_C = {'C1': 1., 'C3': 2., 'H2': 6., 'H4': 8.}

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

@pytest.fixture(scope='module')
def kemp_hxn():
    """KEMP synthesized by the facility at 10 K."""
    units = cp_units('test_targeting_kemp_hxn', KEMP, K)
    sys, HXN = simulate_HXN(units, 10.)
    return units, HXN

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
    for a, (dump, items) in zip(agents, dumps):
        assert a.utility_stream_dump is dump and a.utility_stream_dump == items
    assert fresh_price(10 * kW, 450., 500., gas)[0] == before

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
