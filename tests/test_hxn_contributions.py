# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Tests of per-stream approach temperature contributions
(``approach_contributions``): a match between hot stream h and cold stream c
must keep ``c_h + c_c`` instead of ``T_min_app``.

A. Default unchanged: unspecified contributions (None, nan, an empty
   mapping, a callable returning None, or every stream at
   ``T_min_app / 2``) give the default plans, problem tables and facility
   networks bit for bit, compared in-process with ``float.hex`` and
   ``tobytes``.
B. Uniform contributions: every stream at ``x`` plans the network of
   ``T_min_app = 2x`` on a shifted scale offset by a constant, and on
   real-thermo cases all hot streams at ``a`` and all cold ones at ``b``
   give the default targets at ``T_min_app = a + b`` (references that do
   not use the contribution code); mixed contributions give targets between
   those of the smallest and of the largest pair requirement. A hot
   stream's contribution that reproduces the shifts of a smaller
   ``T_min_app`` gives that default network bit for bit.
C. Non-uniform contributions against INDEPENDENT references: a closed-form
   constant-CP cascade with the classic symmetric shifts (hot streams down
   by ``c_j``, cold streams up by ``c_j``; hensmith's shifted scale is that
   scale less ``T_min_app / 2``), and a walk of every planned match on
   recomputed constant-CP temperatures that checks each pair's requirement
   at both ends (the minimum for linear profiles). The exactness check, the
   repair and the realization enforce each pair's own requirement. The
   facility (split and unsplit, constant CP and real thermodynamics)
   reaches the contribution-aware problem table's targets, keeps every
   pair's requirement on the exact states of its streams (independent
   recomputation) and gives each exchanger its pair's ``dT`` guard; each
   stream's pinch temperature is the shifted pinch plus its own shift.
D. Keys and validation: the value domain (None or nan, else a finite
   number > 0); the mapping, sequence and callable forms; keys naming
   units, auxiliary units through their owners (the nearest key wins),
   auxiliaries of one ID under different owners and stale auxiliaries, and
   the units option; unknown keys, conflicting keys and bad values,
   checked before any stream is flashed and, in the facility, even for
   idle units.
E. The ``cache_network`` rule: a cached network is reused while the
   explicit contributions resolve to the same values per stream.
"""
import math
import random
import numpy as np
import pytest
import biosteam as bst
from numpy.testing import assert_allclose
from hensmith import hxn_synthesis, HeatExchangerNetwork, PinchTargets
from hensmith import _planner as P
from hensmith.hxn_synthesis import problem_table, synthesize_network, _pinch_cut
from test_hxn import (cp_units, KEMP, R002 as R002_UNITS, planned_split,
                      simulate_HXN, simulate_strictly)
from test_hxn_regression import (internal_approach, actual_loads, CASES,
                                 STATUS_RTOL, CASCADE_RTOL)
from test_hxn_mer import _build, _case, _kelvin, APPROACH_TOL
from test_hxn_planner import (streams_from, check_network, duty_scale,
                              random_problem, R002, LINNHOFF4, HARD,
                              REPEATED_PAIR, SPLIT as PLANNER_SPLIT)
from hxn_mer_cases import NO_SPLIT, Q_UNITS, T_UNITS

NAN = float('nan')

# ---------------------------------------------------------------------------
# Independent helpers (no hensmith code)
# ---------------------------------------------------------------------------

#: contribution factors of the corpus streams, times dTmin / 2
FACTORS = (0.4, 1.0, 1.7)

def corpus_contributions(case):
    """Contributions of a corpus case's streams [source degrees]."""
    return [FACTORS[k % 3] * case['dTmin'] / 2
            for k in range(len(case['streams']))]

def corpus_streams(case):
    """A constant-CP corpus case as `_plan_numeric` streams (source units)."""
    return [dict(name=str(n), kind=k, T_in=float(a), T_out=float(b),
                 CP=float(cp)) for n, k, a, b, cp in case['streams']]

def classic_cascade(streams, c):
    """
    Closed-form constant-CP problem table with the classic symmetric shifts
    (hot streams at ``T - c_j``, cold streams at ``T + c_j``; `c` aligned
    with `streams`). Returns ``(Qh, Qc, zeros)``: the targets and the
    interval boundaries where the cascaded heat flow (``Qh`` at the top) is
    at most 1e-9 of the total duty (the top when ``Qh = 0``, the bottom
    when ``Qc = 0``). hensmith's shifted pinch is ``z - T_min_app / 2`` for
    some ``z`` in `zeros`.
    """
    spans = [(s['T_out'] - cj, s['T_in'] - cj, s['CP']) if s['kind'] == 'hot'
             else (s['T_in'] + cj, s['T_out'] + cj, -s['CP'])
             for s, cj in zip(streams, c)]
    Ts = sorted({T for lo, hi, _ in spans for T in (lo, hi)}, reverse=True)
    heat = [0.]
    for upper, lower in zip(Ts, Ts[1:]):
        heat.append(heat[-1] + sum(CP * (upper - lower) for lo, hi, CP in spans
                                   if lo <= lower and hi >= upper))
    Qh = max(0., -min(heat))
    tol = 1e-9 * duty_scale(streams)
    zeros = [T for T, h in zip(Ts, heat) if h + Qh <= tol]
    return Qh, heat[-1] + Qh, zeros

def pair_ends_ok(streams, cmap, net):
    """Every match keeps ``c_h + c_c`` at both ends (constant CP: the ends
    are the minimum); `cmap` maps stream names to contributions."""
    for m in net['matches']:
        approach = min(m['T_hot_in'] - m['T_cold_out'],
                       m['T_hot_out'] - m['T_cold_in'])
        assert approach >= cmap[m['hot']] + cmap[m['cold']] - 1e-6, m
    return True

def assert_targets(net, Qh, Qc, streams):
    tol = 1e-9 * duty_scale(streams)
    assert abs(net['targets']['Qh'] - Qh) <= tol
    assert abs(net['targets']['Qc'] - Qc) <= tol

def plan_record(plan):
    """Every number of a plan that the default path must keep, as hex."""
    hx = lambda x: float(x).hex()
    record = [(e.side, e.hot, e.cold, hx(e.Q), hx(e.H_hot_in),
               hx(e.H_cold_in), hx(e.hot_frac), hx(e.cold_frac))
              for e in plan.exchangers]
    record.append((hx(plan.pinch_T), hx(plan.Q_hot), hx(plan.Q_cold),
                   [hx(u) for u in plan.utility],
                   hx(plan.info['min_approach'])))
    return record

def table_record(table):
    """Every field of a `ProblemTable` as bytes or hex."""
    return [x.tobytes() + str(x.shape).encode() if isinstance(x, np.ndarray)
            else float(x).hex() for x in table]

def water_example():
    """The two-stream water example of `problem_table`'s docstring."""
    bst.settings.set_thermo(['Water'], cache=True)
    bst.main_flowsheet.set_flowsheet('contributions_water')
    hot_in = bst.Stream(Water=1000., T=400., P=5e5, phase='l', units='kmol/hr')
    hot_out = hot_in.copy(); hot_out.vle(T=300., P=5e5)
    cold_in = bst.Stream(Water=900., T=300., P=5e5, phase='l', units='kmol/hr')
    cold_out = cold_in.copy(); cold_out.vle(T=390., P=5e5)
    return [hot_in, cold_in], [hot_out, cold_out], [True, False], 5.

def unit_streams(units):
    """Inlets, re-flashed outlets and is_hot of the units' process streams
    (in the given order), as the facility prepares them."""
    hus = [hx.heat_utilities[0] for hx in units]
    inlets = [hu.unit.ins[0].copy() for hu in hus]
    outlets = [hu.unit.outs[0].copy() for hu in hus]
    for s in outlets: s.vle(H=s.H, P=s.P)
    return inlets, outlets, [hu.duty < 0 for hu in hus]

def by_duty(units):
    return sorted((hx.heat_utilities[0] for hx in units),
                  key=lambda hu: hu.duty)

def required(HXN, hx, of_unit):
    """The required approach of a synthesized process exchanger: the sum of
    its two streams' contributions, `of_unit` of their original units."""
    return sum(of_unit(HXN.original_heat_exchangers[i])
               for i in hxn_synthesis._stream_ports(hx))

def sampled_approach(hx, n=21):
    """The smallest hot-minus-cold temperature difference of a process
    exchanger at `n` uniform duty points, from PH flashes of copies of its
    streams (any mixture; at least the exact minimum)."""
    def T(stream, H):
        s = stream.copy()
        s.vle(H=H, P=stream.P)
        return s.T
    dH = [s_in.H - s_out.H for s_in, s_out in zip(hx.ins, hx.outs)]
    h = 0 if dH[0] >= dH[1] else 1
    hot_in, hot_out = hx.ins[h], hx.outs[h]
    cold_in, cold_out = hx.ins[1 - h], hx.outs[1 - h]
    return min(T(hot_in, hot_in.H - q) - T(cold_in, max(cold_out.H - q,
                                                        cold_in.H))
               for q in np.linspace(0., hot_in.H - hot_out.H, n))

def network_record(HXN):
    """Every number of a facility's network that the default path must
    keep, as hex or bytes."""
    hx = lambda x: float(x).hex()
    info = HXN.synthesis_info
    return ([(u.ID, hx(u.Q), hx(u.dT)) for u in HXN.new_HXs],
            [hx(u.Q) for u in HXN.new_HX_utils], # an int 0 if served
            HXN.pinch_Ts.tobytes(), hx(info['min_approach']), info['status'],
            [u.ID for u in HXN.new_splitters])

def contribution_table(units, T, of_unit):
    """The problem table of the units' process streams with their
    contributions (`of_unit` of each unit), prepared as the facility
    prepares them, and the total duty."""
    hus = by_duty(units)
    inlets, outlets, is_hot = unit_streams([hu.unit for hu in hus])
    table = problem_table(inlets, outlets, is_hot, T, approach_contributions=[
        of_unit(hu.unit) for hu in hus])
    return table, sum(abs(hu.duty) for hu in hus)

def quickstart(name):
    """The system of `HeatExchangerNetwork`'s first example: column D1
    (condenser and reboiler), its product coolers D1_H1 and D1_H2, and
    flash F1 (its heater)."""
    bst.main_flowsheet.set_flowsheet(name)
    bst.settings.set_thermo(['Water', 'Methanol', 'Glycerol'], cache=True)
    feed1 = bst.Stream('feed1', flow=(8000, 100, 25))
    feed2 = bst.Stream('feed2', flow=(10000, 1000, 10))
    D1 = bst.ShortcutColumn('D1', ins=feed1,
                            outs=('distillate', 'bottoms_product'),
                            LHK=('Methanol', 'Water'), y_top=0.99,
                            x_bot=0.01, k=2, is_divided=True)
    D1_H1 = bst.HXutility('D1_H1', ins=D1.outs[1], T=300)
    D1_H2 = bst.HXutility('D1_H2', ins=D1.outs[0], T=300)
    F1 = bst.Flash('F1', ins=feed2, outs=('vapor', 'liquid'), V=0.9,
                   P=101325)
    return D1, D1_H1, D1_H2, F1

# ---------------------------------------------------------------------------
# A. Default unchanged
# ---------------------------------------------------------------------------

PLANNER_DEFAULT = {
    'R002': (10., R002, {}),
    'LINNHOFF4': (10., LINNHOFF4, {}),
    's13_111_3x9': (*HARD['s13_111_3x9'], {}),
    'smith2005_exr18_4': (*PLANNER_SPLIT['smith2005_exr18_4'],
                          dict(stream_splitting=True)),
}

@pytest.mark.parametrize('name', list(PLANNER_DEFAULT))
def test_default_plans_bit_for_bit(name):
    # unspecified contributions plan the default network to the bit
    dT, rows, kw = PLANNER_DEFAULT[name]
    streams = streams_from(rows)
    N = len(streams)
    default = P._plan_numeric(streams, dT, **kw)['plan']
    reference = plan_record(default)
    assert 'shift' not in default.info
    assert 'min_approach_margin' not in default.info
    margins = set()
    for contributions in ([None] * N, [NAN] * N, [dT / 2] * N):
        plan = P._plan_numeric(streams, dT, approach_contributions=contributions,
                               **kw)['plan']
        assert plan_record(plan) == reference
        assert plan.status == default.status
        assert plan.info['shift'].tobytes() == np.where(
            [s['kind'] == 'hot' for s in streams], dT, 0.).tobytes()
        margins.add(float(plan.info['min_approach_margin']).hex())
        assert abs(plan.info['min_approach_margin']
                   - (default.info['min_approach'] - dT)) <= 1e-12
    assert len(margins) == 1

@pytest.mark.parametrize('name', ['water', 'case_05_boiling_cold'])
def test_default_problem_table_bit_for_bit(name):
    if name == 'water':
        inlets, outlets, is_hot, T = water_example()
    else:
        units, T = CASES[name][0]()
        inlets, outlets, is_hot = unit_streams(units)
    N = len(inlets)
    table, curves, grid = hxn_synthesis._problem_table(inlets, outlets, is_hot,
                                                       T)
    reference = table_record(table)
    assert grid['shift'].tobytes() == np.where(is_hot, T, 0.).tobytes()
    assert (grid['contributions'] == T / 2).all()
    assert table_record(problem_table(inlets, outlets, is_hot, T)) == reference
    for contributions in ([None] * N, [NAN] * N, [T / 2] * N):
        other, _, other_grid = hxn_synthesis._problem_table(
            inlets, outlets, is_hot, T, curves, contributions)
        assert table_record(other) == reference
        for key in ('Ts', 'shift', 'Hl', 'Hr', 'k_hi', 'k_lo',
                    'contributions'):
            assert other_grid[key].tobytes() == grid[key].tobytes(), key
        assert table_record(problem_table(
            inlets, outlets, is_hot, T, curves,
            approach_contributions=contributions)) == reference

@pytest.mark.parametrize('T', [0., math.ulp(0.)])
def test_unspecified_contributions_need_no_positive_T_min_app(T):
    # every value unspecified is the default path, which accepts any finite
    # T_min_app (T_min_app = 0 included): same plan and table to the bit
    streams = streams_from(LINNHOFF4)
    N = len(streams)
    is_hot_plan = [s['kind'] == 'hot' for s in streams]
    default = P._plan_numeric(streams, T)['plan']
    reference = plan_record(default)
    for contributions in ([None] * N, [NAN] * N, np.full(N, NAN)):
        plan = P._plan_numeric(streams, T,
                               approach_contributions=contributions)['plan']
        assert plan_record(plan) == reference
        assert plan.status == default.status
        assert plan.info['shift'].tobytes() == np.where(
            is_hot_plan, T, 0.).tobytes()
    inlets, outlets, is_hot, _ = water_example()
    table, curves, grid = hxn_synthesis._problem_table(inlets, outlets, is_hot,
                                                       T)
    reference = table_record(table)
    for contributions in ([None, None], [NAN, NAN], np.full(2, NAN)):
        other, _, other_grid = hxn_synthesis._problem_table(
            inlets, outlets, is_hot, T, curves, contributions)
        assert table_record(other) == reference
        for key in ('Ts', 'shift', 'Hl', 'Hr', 'k_hi', 'k_lo',
                    'contributions'):
            assert other_grid[key].tobytes() == grid[key].tobytes(), key
        assert table_record(problem_table(
            inlets, outlets, is_hot, T, curves,
            approach_contributions=contributions)) == reference

def default_network_units(name):
    """(units, T_min_app, stream_splitting) of a facility test case."""
    if name == 'r002':
        return cp_units('contributions_default_r002', R002_UNITS, 300.), 10., False
    if name == 'case_05_boiling_cold':
        return (*CASES[name][0](), False)
    return (*_build(_case(name)), True)

@pytest.mark.parametrize('name', ['r002', 'case_05_boiling_cold',
                                  'smith2005_ex18_2_split'])
def test_default_networks_bit_for_bit(name):
    # the facility with unspecified contributions in every form, or every
    # stream at T_min_app / 2, synthesizes the default network to the bit
    units, T, splitting = default_network_units(name)
    sys, HXN = simulate_HXN(units, T, f'contributions_default_{name}',
                            stream_splitting=splitting)
    reference = network_record(HXN)
    assert HXN._resolved_contributions is None
    for contributions in ({}, {u: NAN for u in units}, lambda hu: None,
                          {u: T / 2 for u in units}, None):
        HXN.approach_contributions = contributions
        simulate_strictly(sys)
        assert network_record(HXN) == reference
        info = HXN.synthesis_info
        assert (info['approach_contributions'] == T / 2).all()
        assert (float(info['min_approach_margin']).hex()
                == float(info['min_approach'] - T).hex())
    if splitting: assert HXN.new_splitters

# ---------------------------------------------------------------------------
# B. Uniform contributions
# ---------------------------------------------------------------------------

UNIFORM = {
    'LINNHOFF4': (10., LINNHOFF4, {}),
    'R002': (10., R002, {}),
    '4sp2_dt20F': (*REPEATED_PAIR['4sp2_dt20F'], {}),
    'smith2005_exr18_4': (*PLANNER_SPLIT['smith2005_exr18_4'],
                          dict(stream_splitting=True)),
}

@pytest.mark.parametrize('name', list(UNIFORM))
def test_uniform_contributions_equal_doubled_T_min_app(name):
    # every stream at x (T_min_app = 4): the network of T_min_app = 2x, on a
    # shifted scale x - 2 K above it
    dT, rows, kw = UNIFORM[name]
    streams = streams_from(rows)
    N = len(streams)
    x = dT / 2.
    scale = duty_scale(streams)
    ref = P._plan_numeric(streams, 2. * x, **kw)
    net = P._plan_numeric(streams, 4., approach_contributions=[x] * N, **kw)
    assert net['status'] == ref['status']
    for key in ('Qh', 'Qc'):
        assert abs(net['targets'][key] - ref['targets'][key]) <= 1e-9 * scale
    plan, plan_ref = net['plan'], ref['plan']
    assert ([(e.side, e.hot, e.cold) for e in plan.exchangers]
            == [(e.side, e.hot, e.cold) for e in plan_ref.exchangers])
    assert_allclose([e.Q for e in plan.exchangers],
                    [e.Q for e in plan_ref.exchangers], rtol=0,
                    atol=1e-9 * scale)
    assert abs(plan.pinch_T - (plan_ref.pinch_T + x - 2.)) <= 1e-9
    assert abs(plan.info['min_approach']
               - plan_ref.info['min_approach']) <= 1e-9
    assert abs(plan.info['min_approach_margin']
               - (plan_ref.info['min_approach'] - 2. * x)) <= 1e-9

def test_uniform_contributions_in_the_facility():
    # KEMP: every unit at 5 K with T_min_app = 4 synthesizes the network of
    # T_min_app = 10
    units = cp_units('contributions_uniform', KEMP, 300.)
    sys, HXN = simulate_HXN(units, 10., 'contributions_uniform')
    ref = [(hx.ID, hx.Q) for hx in HXN.new_HXs]
    HXN.T_min_app = 4.
    HXN.approach_contributions = {u: 5. for u in units}
    simulate_strictly(sys)
    assert [hx.ID for hx in HXN.new_HXs] == [ID for ID, _ in ref]
    assert_allclose([hx.Q for hx in HXN.new_HXs], [Q for _, Q in ref],
                    rtol=1e-9)
    assert all(hx.dT == 10. - 1e-6 for hx in HXN.new_HXs)
    assert (HXN.synthesis_info['approach_contributions'] == 5.).all()

def test_contributions_equal_to_a_shift_are_exact():
    # r002 at T_min_app = 10 with H1 at 2 K has the shifts of the default
    # run at 7 K (hot 5 + 2, cold 5 - 5), so its network is that one to the
    # bit
    units = cp_units('contributions_shift_7', R002_UNITS, 300.)
    sys, HXN = simulate_HXN(units, 7., 'contributions_shift_7')
    reference = network_record(HXN)
    min_approach = HXN.synthesis_info['min_approach']
    units = cp_units('contributions_shift_10', R002_UNITS, 300.)
    sys, HXN = simulate_HXN(units, 10., 'contributions_shift_10',
                            approach_contributions={units[3]: 2.})
    info = HXN.synthesis_info
    assert network_record(HXN) == reference
    assert info['status'] == 'mer'
    assert (float(info['min_approach_margin']).hex()
            == float(min_approach - 7.).hex())
    assert info['approach_contributions'].tolist() == [5., 5., 5., 2.]
    assert all(hx.dT == 7. - 1e-6 for hx in HXN.new_HXs)
    assert round(info['Q_hot_target'] / 3600., 6) == 74.
    assert round(info['Q_cold_target'] / 3600., 6) == 444.

@pytest.mark.parametrize('name', ['case_05_boiling_cold',
                                  'case_08_two_condensers'])
def test_real_thermo_targets_against_uniform_runs(name):
    units, T = CASES[name][0]()
    hus = by_duty(units)
    hot = [hu.duty < 0 for hu in hus]
    total = sum(abs(hu.duty) for hu in hus)

    def table(T_min_app, contributions=None):
        r = hxn_synthesis._pinch_analysis(hus, T_min_app, False, False,
                                          contributions)
        return r[12], r[14], r[10] # table, grid, heat utilities by stream

    # all hot streams at a, all cold ones at b: the default at a + b, on a
    # grid offset by b - T/2
    for a, b in ((1., 4.), (3., 0.5)):
        t, grid, streams = table(T, [a if h else b for h in hot])
        ref = table(a + b)[0]
        assert_allclose([t.hot_util_load, t.cold_util_load],
                        [ref.hot_util_load, ref.cold_util_load],
                        rtol=1e-12, atol=1e-12 * total)
        assert t.Ts.shape == ref.Ts.shape
        assert_allclose(t.Ts, ref.Ts + (b - T / 2), rtol=0, atol=1e-9)
        assert grid['contributions'].tolist() == [
            a if hu.duty < 0 else b for hu in streams] # stream order
    # mixed contributions: between the smallest and the largest requirement
    C = [(1., 4., 2., 6., 3., 0.5)[k % 6] for k in range(len(hus))]
    t = table(T, C)[0]
    c_hot = [c for c, h in zip(C, hot) if h]
    c_cold = [c for c, h in zip(C, hot) if not h]
    lo = table(min(c_hot) + min(c_cold))[0]
    hi = table(max(c_hot) + max(c_cold))[0]
    tol = 1e-9 * total
    for load in ('hot_util_load', 'cold_util_load'):
        assert (getattr(lo, load) - tol <= getattr(t, load)
                <= getattr(hi, load) + tol), load

# ---------------------------------------------------------------------------
# C. Non-uniform contributions, independent references
# ---------------------------------------------------------------------------

def test_planner_targets_match_classic_cascade():
    rng = random.Random(21)
    for _ in range(40):
        rows, dT = random_problem(rng, 2, 10)
        streams = streams_from(rows)
        c = [rng.choice((0.5, 1., 2.5, 4., 7.5)) for _ in streams]
        net = P._plan_numeric(streams, dT, approach_contributions=c,
                              work_scale=0.05)
        Qh, Qc, zeros = classic_cascade(streams, c)
        assert_targets(net, Qh, Qc, streams)
        assert min(abs(net['plan'].pinch_T - (z - dT / 2)) for z in zeros) <= 1e-9
        pair_ends_ok(streams, {s['name']: cj for s, cj in zip(streams, c)},
                     net)

@pytest.mark.parametrize('name', ['kemp', 'smith2005_exr17_2_dT10'])
def test_problem_table_matches_classic_cascade(name):
    if name == 'kemp':
        T = 10.
        units = cp_units('contributions_kemp_table', KEMP, 300.)
        streams = [dict(name=ID, kind='hot' if T_in > T_out else 'cold',
                        T_in=T_in + 300., T_out=T_out + 300., CP=CP)
                   for ID, T_in, T_out, CP in KEMP]
        given = [2., 6., 1., None]
    else:
        case = _case(name)
        units, T = _build(case)
        scale = T_UNITS[case['T_unit']][0]
        kW = Q_UNITS[case['Q_unit']]
        streams = [dict(name=n, kind=k, T_in=_kelvin(case, a),
                        T_out=_kelvin(case, b), CP=cp * kW / scale)
                   for n, k, a, b, cp in case['streams']]
        given = [x * scale for x in corpus_contributions(case)]
    c = [T / 2 if x is None else x for x in given]
    inlets, outlets, is_hot = unit_streams(units)
    assert is_hot == [s['kind'] == 'hot' for s in streams]
    table, curves, grid = hxn_synthesis._problem_table(
        inlets, outlets, is_hot, T, None, given)
    Qh, Qc, zeros = classic_cascade(streams, c)
    if name == 'kemp': # closed form: 7 kW hot, 47 kW cold, pinch 81 degC
        assert_allclose((Qh, Qc), (7., 47.), rtol=0, atol=1e-9)
        assert 381. in zeros
    total = 3600. * sum(abs(s['CP'] * (s['T_out'] - s['T_in'])) for s in streams)
    assert abs(table.hot_util_load - 3600. * Qh) <= 1e-9 * total
    assert abs(table.cold_util_load - 3600. * Qc) <= 1e-9 * total
    assert min(abs(table.pinch_T - (z - T / 2)) for z in zeros) <= 1e-9
    assert grid['shift'].tobytes() == np.where(
        is_hot, T / 2 + np.array(c), T / 2 - np.array(c)).tobytes()
    assert grid['contributions'].tolist() == c
    for (Tk, _), s in zip(hxn_synthesis._grid_knots(curves, grid), streams):
        assert_allclose([Tk[0], Tk[-1]], sorted((s['T_in'], s['T_out'])),
                        rtol=0, atol=1e-9)

NO_SPLIT_CP = [case for case in NO_SPLIT if case['kind'] == 'constant_cp']

def test_no_split_corpus_with_contributions():
    # no status assertion: an unsplit MER network is known for the default
    # requirement, not for these contributions (all 24 reached 'mer' when
    # this test was written)
    assert len(NO_SPLIT_CP) == 24
    for case in NO_SPLIT_CP:
        streams = corpus_streams(case)
        c = corpus_contributions(case)
        net = P._plan_numeric(streams, case['dTmin'], approach_contributions=c)
        Qh, Qc, _ = classic_cascade(streams, c)
        assert_targets(net, Qh, Qc, streams)
        hot, cold = check_network(streams, -math.inf, net)
        pair_ends_ok(streams, {s['name']: cj for s, cj in zip(streams, c)},
                     net)
        info = net['plan'].info
        assert net['matches'], case['name']
        assert info['min_approach_margin'] >= -1e-9, case['name']
        walked = min(min(m['T_hot_in'] - m['T_cold_out'],
                         m['T_hot_out'] - m['T_cold_in'])
                     for m in net['matches'])
        assert abs(info['min_approach'] - walked) <= 1e-9, case['name']
        if net['status'] == 'mer':
            tol = 1e-6 * duty_scale(streams)
            assert abs(hot - Qh) <= tol and abs(cold - Qc) <= tol

SPLIT_NAMES = ['smith2005_ex18_2_split', 'smith2005_exr18_4',
               '7sp4_dolan1990_dt10K', 'fs_15sp_tkm',
               'cornell_processdesign_four_stream_split']

@pytest.mark.parametrize('name', SPLIT_NAMES)
def test_split_cases_with_contributions_reach_mer(name):
    # Theorem M holds on the shared shifted scale: MER with splits
    case = _case(name)
    streams = corpus_streams(case)
    c = corpus_contributions(case)
    net = P._plan_numeric(streams, case['dTmin'], approach_contributions=c,
                          stream_splitting=True)
    assert net['status'] == 'mer'
    Qh, Qc, _ = classic_cascade(streams, c)
    assert_targets(net, Qh, Qc, streams)
    # branch temperatures: the parent's at the parent-equivalent enthalpies
    cmap = {s['name']: cj for s, cj in zip(streams, c)}
    pair_ends_ok(streams, cmap, net)
    assert net['plan'].splits
    info = net['plan'].info
    assert info['min_approach_margin'] >= -1e-9
    # split-mode reporting against the walked ends (linear profiles: the
    # ends are each match's minimum)
    ends = [min(m['T_hot_in'] - m['T_cold_out'],
                m['T_hot_out'] - m['T_cold_in']) for m in net['matches']]
    assert abs(info['min_approach'] - min(ends)) <= 1e-9
    margin = min(a - (cmap[m['hot']] + cmap[m['cold']])
                 for a, m in zip(ends, net['matches']))
    assert abs(info['min_approach_margin'] - margin) <= 1e-9

def test_exact_check_and_repair_per_pair():
    # hot stream 2 (on branches above the pinch) needs 5 K more against
    # every partner: only its exchangers are flagged and shrunk, and only
    # their exchangers get the larger dT guard
    plan, curves, knots, is_hot, streams_inlet, dT = planned_split(
        'smith2005_ex18_2_split')
    j = 2
    assert is_hot[j]
    shift = np.where(is_hot, dT, 0.)
    shift[j] += 5.
    duties = {n: e.Q for n, e in enumerate(plan.exchangers)}
    ends = hxn_synthesis._walk(plan, duties, knots, is_hot)[0]
    bad = hxn_synthesis._exact_approach(plan, duties, ends, curves, knots,
                                        shift)[2]
    assert bad and all(plan.exchangers[n].hot == j for n in bad)
    assert any(plan.exchangers[n].hot_frac != 1. for n in bad)
    others = [n for n in duties if n not in bad]
    assert others
    new, changes = hxn_synthesis._repair(plan, duties, knots, is_hot, curves,
                                         shift)
    assert {n for n, _, _ in changes} == set(bad)
    assert all(0. <= Q_after < Q_before for _, Q_before, Q_after in changes)
    assert all(new[n] == duties[n] for n in others)
    ends = hxn_synthesis._walk(plan, new, knots, is_hot)[0]
    assert not hxn_synthesis._exact_approach(plan, new, ends, curves, knots,
                                             shift)[2]
    bst.main_flowsheet.set_flowsheet('contributions_repair')
    units = hxn_synthesis._realize(plan, new, curves, knots, streams_inlet,
                                   is_hot, shift)[0]
    try:
        for n, hx in units.items():
            # the pair sets the requirement: every exchanger of stream j
            required = dT + 5. if plan.exchangers[n].hot == j else dT
            assert hx.dT == required - 1e-6, hx.ID
            assert internal_approach(hx) >= required - APPROACH_TOL, hx.ID
    finally:
        hxn_synthesis._discard(units.values())
    # one shift per stream, or a scalar
    with pytest.raises(ValueError, match='3 shifts given for 4 streams'):
        hxn_synthesis._exact_approach(plan, duties, ends, curves, knots,
                                      shift[:3])
    with pytest.raises(ValueError, match='3 shifts given for 4 streams'):
        hxn_synthesis._realize(plan, new, curves, knots, streams_inlet,
                               is_hot, shift[:3])

def facility_case(name):
    """(units, T_min_app, stream_splitting, contributions by unit) of a
    facility case with contributions: the corpus factors on constant-CP
    split cases, and hand values in the case's own unit order on
    real-thermo ones."""
    if name in CASES:
        units, T = CASES[name][0]()
        values, splitting = {
            'case_04_report_case': ([1., 4., 2., 6.], True),
            'case_08_two_condensers': ([1., 4., 2., 6., 3., 0.5], False),
        }[name]
        return units, T, splitting, dict(zip(units, values))
    case = _case(name)
    units, T = _build(case)
    scale = T_UNITS[case['T_unit']][0]
    return units, T, True, dict(zip(units, [
        scale * c for c in corpus_contributions(case)]))

@pytest.mark.parametrize('name', ['smith2005_ex18_2_split',
                                  '7sp4_dolan1990_dt10K',
                                  'case_04_report_case',
                                  'case_08_two_condensers'])
def test_facility_contributions_split_and_unsplit(name):
    # the facility reaches the contribution-aware problem table's targets
    # and keeps every pair's own requirement on the exact stream states
    units, T, splitting, C = facility_case(name)
    sys, HXN = simulate_HXN(units, T, f'contributions_{name}',
                            stream_splitting=splitting,
                            approach_contributions=C)
    info = HXN.synthesis_info
    # Theorem M with splitting; observed for case_08 (unsplit)
    assert info['status'] == 'mer'
    table, total = contribution_table(units, T, C.get)
    heat, cool = actual_loads(HXN)
    assert abs(heat - table.hot_util_load) <= STATUS_RTOL * total
    assert abs(cool - table.cold_util_load) <= STATUS_RTOL * total
    assert abs(HXN.energy_balance_percent_error) < 1e-6
    for hx in HXN.new_HXs:
        r = required(HXN, hx, C.get)
        assert internal_approach(hx) >= r - APPROACH_TOL, hx.ID
        assert abs(hx.dT - (r - 1e-6)) <= 1e-12, hx.ID
    assert info['min_approach_margin'] >= -APPROACH_TOL
    assert info['approach_contributions'].tolist() == [
        C[hx] for hx in HXN.original_heat_exchangers]
    planned = info['plan_targets']
    assert abs(planned['Q_hot'] - table.hot_util_load) <= CASCADE_RTOL * total
    assert abs(planned['Q_cold'] - table.cold_util_load) <= CASCADE_RTOL * total
    assert abs(planned['pinch_T'] - table.pinch_T) <= 1e-9
    assert planned['cut'] == _pinch_cut(table)
    if splitting: assert HXN.new_splitters and info['splits']

@pytest.mark.parametrize('sort_hus_by_T', [False, True])
def test_pinch_Ts_per_stream(sort_hus_by_T):
    # KEMP (300 K above the classic temperatures) at 10 K with C1 at 2, H2
    # at 6 and C3 at 1 K (H4 at the default 5 K): 7 kW of heating (closed
    # form, see test_problem_table_matches_classic_cascade); every stream's
    # pinch temperature is the shifted pinch plus its own shift
    units = cp_units('contributions_pinch_Ts', KEMP, 300.)
    C1, H2, C3, H4 = units
    hand = {C1: 2., H2: 6., C3: 1., H4: 5.}
    sys, HXN = simulate_HXN(units, 10., 'contributions_pinch_Ts',
                            sort_hus_by_T=sort_hus_by_T,
                            approach_contributions={C1: 2., H2: 6., 'C3': 1.})
    info = HXN.synthesis_info
    assert info['status'] == 'mer' # observed
    assert round(info['Q_hot_target'] / 3600., 6) == 7.
    hxs = HXN.original_heat_exchangers
    assert info['approach_contributions'].tolist() == [hand[hx] for hx in hxs]
    pinch_T = info['plan_targets']['pinch_T']
    expected = {'C1': 379., 'C3': 380., 'H2': 387., 'H4': 386.}
    for i, hx in enumerate(hxs):
        hot = i not in HXN.cold_indices
        T_p = pinch_T + 5. + (hand[hx] if hot else -hand[hx])
        T_in, T_out = HXN.inlet_Ts[i], HXN.outlet_Ts[i]
        # every stream reaches the pinch (C3 starts there)
        assert min(T_in, T_out) <= T_p <= max(T_in, T_out), hx.ID
        assert abs(HXN.pinch_Ts[i] - T_p) <= 1e-9, hx.ID
        assert abs(T_p - expected[hx.ID]) <= 1e-9, hx.ID

# ---------------------------------------------------------------------------
# D. Keys and validation
# ---------------------------------------------------------------------------

def test_contribution_keys_on_auxiliary_units():
    # column D1 holds its condenser and reboiler as auxiliary units, flash F1
    # its heater; D1_H1 and D1_H2 have no owner
    D1, D1_H1, D1_H2, F1 = quickstart('contributions_keys')
    units = [D1, D1_H1, D1_H2, F1]
    bst.System.from_units('contributions_keys', units=units).simulate()
    hus = bst.process_tools.heat_exchanger_utilities_from_units(units)
    IDs = [hu.unit.ID for hu in hus]
    assert sorted(IDs) == ['D1_H1', 'D1_H2', 'condenser', 'heat_exchanger',
                           'reboiler']
    none = dict.fromkeys(IDs)

    def resolve(x):
        return dict(zip(IDs, hxn_synthesis._stream_contributions(hus, x)))

    # an owner's key covers its auxiliary units; the nearest key wins
    assert resolve({'D1': 3.}) == none | {'condenser': 3., 'reboiler': 3.}
    assert resolve({'D1': 3., ('D1', 'condenser'): 1.}) == none | {
        'condenser': 1., 'reboiler': 3.}
    assert resolve({D1: 3., D1.condenser: 1.}) == none | {
        'condenser': 1., 'reboiler': 3.}
    assert resolve({F1: 2., D1_H1: 4., 'D1_H2': 1.,
                    ('D1_H1', 'D1_H1'): 4.}) == none | {
        'heat_exchanger': 2., 'D1_H1': 4., 'D1_H2': 1.}
    # None and nan are an explicit default at their level
    for unspecified in (None, NAN):
        assert resolve({'D1': 3., ('D1', 'condenser'): unspecified}) == (
            none | {'reboiler': 3.})
    assert resolve({D1: None, 'D1': NAN}) == none
    # same-level keys must agree; a bare auxiliary ID names nothing
    assert resolve({D1: 3., 'D1': 3., ('D1', 'D1'): 3.})['reboiler'] == 3.
    for x in ({D1_H1: 4., 'D1_H1': 3.}, {D1: 3., 'D1': 2.},
              {D1: None, 'D1': 3.}, {'D1': 3., D1.reboiler: 1.,
                                     ('D1', 'reboiler'): 2.}):
        with pytest.raises(ValueError, match='different values at the same '
                                             'level'):
            resolve(x)
    with pytest.raises(ValueError, match="name no unit: 'condenser', "
                                         r"\('D1', 'nope'\)"):
        resolve({'condenser': 1., ('D1', 'nope'): 1., 'D1': 2.})
    # biosteam gives a single auxiliary its bare name, so an indexed ID of
    # one names nothing
    for key in (('D1', 'condenser[0]'), ('D1', 'reboiler[7]'),
                ('D1', 'pump[0]'), ('F1', 'heat_exchanger[3]')):
        with pytest.raises(ValueError, match='name no unit'):
            resolve({key: 1.})
    # sequence (aligned with hus) and callable (once per heat utility, in
    # order) forms; heat utility and other keys are rejected
    assert hxn_synthesis._stream_contributions(
        hus, [1., None, NAN, 2., 3.]) == [1., None, None, 2., 3.]
    calls = []
    def f(hu):
        calls.append(hu)
        return 2. if hu.unit.owner is D1 else None
    assert resolve(f) == none | {'condenser': 2., 'reboiler': 2.}
    assert [id(hu) for hu in calls] == [id(hu) for hu in hus]
    with pytest.raises(TypeError, match='a heat utility cannot be an '
                                        r'approach_contributions key.*hu\.unit'):
        resolve({hus[0]: 1.})
    for key in (1, ('D1',), ('D1', 'condenser', 'x'), ('D1', 1), D1.ins[0]):
        with pytest.raises(TypeError, match='keys must be units'):
            resolve({key: 1.})
    # stale auxiliary units: biosteam.HXutilities rebuilds its exchangers at
    # every simulation, and a key holding an old one names the new one
    bst.main_flowsheet.set_flowsheet('contributions_stale')
    bst.settings.set_thermo(['Water'], cache=True)
    water = bst.Stream('water', Water=100., T=300., P=5e6, units='kmol/hr')
    U = bst.HXutilities('U', ins=water, T=500., rigorous=False)
    U.simulate()
    old = U.heat_exchangers[0]
    U.simulate()
    stale_hus = bst.process_tools.heat_exchanger_utilities_from_units([U])
    assert len(stale_hus) > 1
    assert all(hu.unit is not old for hu in stale_hus)
    assert hxn_synthesis._stream_contributions(stale_hus, {old: 2., 'U': 3.}) \
        == [2.] + [3.] * (len(stale_hus) - 1)

class TwoFlash(bst.Unit):
    """Two auxiliary flash drums, each with its own auxiliary heater: both
    heaters have the tuple key ``(ID, 'heat_exchanger')``."""
    _N_ins = 2
    _N_outs = 4
    auxiliary_unit_names = ('flash_a', 'flash_b')

    def _init(self):
        self.auxiliary('flash_a', bst.Flash, ins=self.ins[0],
                       outs=self.outs[:2], V=0.5, P=101325)
        self.auxiliary('flash_b', bst.Flash, ins=self.ins[1],
                       outs=self.outs[2:], V=0.3, P=101325)

    def _run(self):
        self.flash_a.simulate()
        self.flash_b.simulate()

    def _design(self): pass
    def _cost(self): pass

class NestedHeaters(bst.Unit):
    """A `biosteam.HXutilities` as an auxiliary unit: its exchangers, rebuilt
    at every simulation, are auxiliaries of an auxiliary."""
    _N_ins = 1
    _N_outs = 1
    auxiliary_unit_names = ('hxs',)

    def _init(self):
        self.auxiliary('hxs', bst.HXutilities, ins=self.ins[0],
                       outs=self.outs[0], T=500., rigorous=False)

    def _run(self): self.hxs.simulate()
    def _design(self): pass
    def _cost(self): pass

def test_contribution_keys_on_nested_auxiliary_units():
    # a unit key names that very unit, not a namesake under another owner;
    # a tuple key names every auxiliary of that ID under the top owner
    bst.main_flowsheet.set_flowsheet('contributions_nested')
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    a = bst.Stream('a', Water=100., Ethanol=20., units='kmol/hr', T=300.)
    b = bst.Stream('b', Water=150., Ethanol=10., units='kmol/hr', T=300.)
    U = TwoFlash('U', ins=(a, b))
    hot = bst.Stream('hot', Water=400., units='kmol/hr', T=430., P=6e5)
    C1 = bst.HXutility('C1', ins=hot, T=320., rigorous=False)
    bst.System.from_units('contributions_nested', units=[U, C1]).simulate()
    ha, hb = U.flash_a.heat_exchanger, U.flash_b.heat_exchanger
    hus = bst.process_tools.heat_exchanger_utilities_from_units([U, C1])
    assert [hu.unit for hu in hus] == [ha, hb, C1]
    assert ha.ID == hb.ID == 'heat_exchanger'

    def resolve(x):
        return hxn_synthesis._stream_contributions(hus, x)

    assert resolve({ha: 2.}) == [2., None, None]
    assert resolve({ha: 2., hb: 6.}) == [2., 6., None]
    assert resolve({('U', 'flash_b'): 6., ha: 2.}) == [2., 6., None]
    assert resolve({('U', 'flash_a'): 2., ('U', 'flash_b'): 6.}) == [
        2., 6., None]
    assert resolve({('U', 'heat_exchanger'): 2.}) == [2., 2., None]
    assert resolve({'U': 3., ha: 2.}) == [2., 3., None]
    # a conflict names the unit with its owners below the top one
    with pytest.raises(ValueError, match=r"give unit \('U', 'heat_exchanger'"
                                         r"\) \(of 'flash_b'\) different"):
        resolve({('U', 'heat_exchanger'): 2., hb: 6.5, ha: 2.})
    with pytest.raises(ValueError, match=r"unit \('U', 'heat_exchanger'\) "
                                         r"\(of 'flash_a'\) must be"):
        resolve(lambda hu: 0. if hu.unit is ha else None)
    # film coefficient keys of targets resolve the same way
    t = PinchTargets.from_heat_utilities(hus, 10.)
    j = next(j for j, unit in enumerate(t.units) if unit is ha)
    area = t.area_target(h={ha: 2.})
    assert area == t.area_target(h={j: 2.})
    assert area != t.area_target(h={('U', 'heat_exchanger'): 2.})
    # in the facility
    HXN = HeatExchangerNetwork('HXN', T_min_app=10.)
    sys = bst.System.from_units('contributions_nested_facility',
                                units=[U, C1, HXN])
    for x, given in [({ha: 2.}, {ha: 2.}), ({ha: 2., hb: 6.}, {ha: 2., hb: 6.})]:
        HXN.approach_contributions = x
        sys.simulate()
        assert [given.get(hx, 5.) for hx in HXN.original_heat_exchangers] == (
            HXN.synthesis_info['approach_contributions'].tolist())
    # a stale auxiliary of an auxiliary names the one that took its place,
    # and only that one
    bst.settings.set_thermo(['Water'], cache=True)
    water = bst.Stream('water', Water=100., T=300., P=5e6, units='kmol/hr')
    V = NestedHeaters('V', ins=water)
    V.simulate()
    old = V.hxs.heat_exchangers[0]
    V.simulate()
    stale_hus = bst.process_tools.heat_exchanger_utilities_from_units([V])
    assert len(stale_hus) > 1
    assert all(hu.unit is not old for hu in stale_hus)
    assert stale_hus[0].unit.ID == old.ID
    assert hxn_synthesis._stream_contributions(
        stale_hus, {old: 2., 'V': 3.}) == [2.] + [3.] * (len(stale_hus) - 1)

def test_contribution_keys_in_the_facility():
    # the facility, simulated twice: every key resolves to the hand values
    D1, D1_H1, D1_H2, F1 = quickstart('contributions_keys_facility')
    HXN = HeatExchangerNetwork('HXN', T_min_app=5., approach_contributions={
        'D1': 1., ('D1', 'condenser'): 1.5, F1: 1., 'D1_H1': 4.,
        D1.reboiler: 2.})
    sys = bst.System.from_units('contributions_keys_facility',
                                units=[D1, D1_H1, D1_H2, F1, HXN])
    hand = {'condenser': 1.5, 'reboiler': 2., 'heat_exchanger': 1.,
            'D1_H1': 4., 'D1_H2': 2.5}
    of_unit = lambda unit: hand[unit.ID]
    for _ in range(2):
        simulate_strictly(sys)
        info = HXN.synthesis_info
        assert info['status'] == 'mer' # observed
        assert info['approach_contributions'].tolist() == [
            hand[hx.ID] for hx in HXN.original_heat_exchangers]
        assert info['min_approach_margin'] >= -APPROACH_TOL
        assert len({round(hx.dT, 6) for hx in HXN.new_HXs}) > 1
        for hx in HXN.new_HXs:
            r = required(HXN, hx, of_unit)
            assert abs(hx.dT - (r - 1e-6)) <= 1e-12, hx.ID
            # ternary streams (internal_approach takes binaries at most)
            assert sampled_approach(hx) >= r - APPROACH_TOL, hx.ID

def test_invalid_contributions(monkeypatch):
    streams = streams_from(LINNHOFF4)
    N = len(streams)
    inlets, outlets, is_hot, T = water_example()
    for n in (N - 1, N + 1):
        with pytest.raises(ValueError, match=f'{n} approach contributions '
                                             f'given for {N} streams'):
            P._plan_numeric(streams, 10., approach_contributions=[1.] * n)
    with pytest.raises(ValueError, match='3 approach contributions given '
                                         'for 2 streams'):
        problem_table(inlets, outlets, is_hot, T,
                      approach_contributions=[1., 1., 1.])
    for value in (0., -1., math.inf, -math.inf, True, np.bool_(True), '3'):
        with pytest.raises(ValueError, match='approach contribution of '
                                             'stream 1 must be'):
            P._plan_numeric(streams, 10.,
                            approach_contributions=[None, value, None, None])
        with pytest.raises(ValueError, match='approach contribution of '
                                             'stream 1 must be'):
            problem_table(inlets, outlets, is_hot, T,
                          approach_contributions=[None, value])
    # a contribution needs a positive T_min_app (it sets the default T/2)
    with pytest.raises(ValueError, match='T_min_app must be finite and > 0'):
        P._plan_numeric(streams, 0., approach_contributions=[1.] * N)
    with pytest.raises(ValueError, match='T_min_app must be finite and > 0'):
        problem_table(inlets, outlets, is_hot, 0.,
                      approach_contributions=[1., 1.])
    # the length is checked even when every value is unspecified
    with pytest.raises(ValueError, match=f'{N + 1} approach contributions '
                                         f'given for {N} streams'):
        P._plan_numeric(streams, 0., approach_contributions=[None] * (N + 1))
    with pytest.raises(ValueError, match='3 approach contributions given '
                                         'for 2 streams'):
        problem_table(inlets, outlets, is_hot, 0.,
                      approach_contributions=[None, NAN, None])
    # nan is unspecified; numpy numbers are accepted
    default = plan_record(P._plan_numeric(streams, 10.)['plan'])
    for contributions in ([NAN] * N, np.full(N, NAN), np.full(N, 5.),
                          [np.float64(5.), 5, None, NAN]):
        plan = P._plan_numeric(streams, 10.,
                               approach_contributions=contributions)['plan']
        assert plan_record(plan) == default
    # synthesize_network: one value per heat utility, checked before any
    # stream is copied or flashed
    units = cp_units('contributions_invalid', R002_UNITS, 300.)
    hus = [hx.heat_utilities[0] for hx in units]
    def no_copy(*args, **kwargs):
        raise AssertionError('copied')
    monkeypatch.setattr(hxn_synthesis, '_copy', no_copy)
    with pytest.raises(ValueError, match='3 approach contributions given '
                                         'for 4 heat utilities'):
        synthesize_network(hus, 10., approach_contributions=[1.] * 3)
    for x in ({1., 2., 3., 4.}, '1234', b'1234', 3.):
        with pytest.raises(TypeError, match='approach_contributions must be'):
            synthesize_network(hus, 10., approach_contributions=x)
    with pytest.raises(ValueError, match='approach contribution of heat '
                                         'utility 1 must be'):
        synthesize_network(hus, 10., approach_contributions=[1., 0., 1., 1.])
    with pytest.raises(ValueError, match='T_min_app must be finite and > 0'):
        synthesize_network(hus, 0., approach_contributions=[1., 1., 1., 1.])
    with pytest.raises(AssertionError, match='copied'): # valid: flashes next
        synthesize_network(hus, 10., approach_contributions=[1., 2., None, 4.])
    # mappings and callables: keys and values are checked before any flash
    for x, error, match in [
            ({'nope': 1., 'C1': 2.}, ValueError, "name no unit: 'nope'"),
            ({hus[0]: 1.}, TypeError, 'a heat utility cannot be'),
            ({'C1': 2., ('H1', 'H1'): -1.}, ValueError,
             r"approach contribution of key \('H1', 'H1'\) must be"),
            (lambda hu: 0. if hu.unit.ID == 'C2' else None, ValueError,
             r"approach contribution of unit \('C2', 'C2'\) must be"),
            ({'C1': 2., units[0]: 3.}, ValueError, 'different values')]:
        with pytest.raises(error, match=match):
            synthesize_network(hus, 10., approach_contributions=x)
    with pytest.raises(ValueError, match='T_min_app must be finite and > 0'):
        synthesize_network(hus, 0., approach_contributions={'C1': 2.})
    # a zero-duty heat utility of hus may be keyed (it is then dropped)
    idle_in = bst.Stream('idle_in', Fluid=1000., T=400., units='kmol/hr')
    idle = bst.HXutility('idle', ins=idle_in, T=400., rigorous=False)
    idle.simulate()
    assert idle.heat_utilities[0].duty == 0.
    for x in ({idle: 3., 'C1': 2.}, lambda hu: 3.):
        with pytest.raises(AssertionError, match='copied'):
            synthesize_network(hus + idle.heat_utilities, 10.,
                               approach_contributions=x)

def test_invalid_contributions_in_the_facility():
    # bad keys and values raise at simulate, even when their unit is idle;
    # valid keys of idle or ignored units have no effect
    units = cp_units('contributions_invalid_facility', R002_UNITS, 300.)
    idle_in = bst.Stream('idle_in', Fluid=1000., T=400., units='kmol/hr')
    idle = bst.HXutility('idle', ins=idle_in, T=400., rigorous=False)
    HXN = HeatExchangerNetwork('HXN', T_min_app=10., ignored=[units[0]])
    sys = bst.System.from_units('contributions_invalid_facility',
                                units=[*units, idle, HXN])
    simulate_strictly(sys)
    reference = network_record(HXN)
    for x, error, match in [
            ({'nope': 1.}, ValueError, 'name no unit'),
            ([1.] * 3, TypeError, 'must be a mapping or a callable'),
            (np.ones(3), TypeError, 'must be a mapping or a callable'),
            ({units[1].heat_utilities[0]: 1.}, TypeError,
             'a heat utility cannot be'),
            ({idle: -1.}, ValueError, 'approach contribution of key'),
            ({idle: 2., 'idle': 3.}, ValueError, 'different values'),
            ({units[0]: 2., ('C1', 'C1'): None}, ValueError,
             'different values')]:
        HXN.approach_contributions = x
        with pytest.raises(error, match=match):
            sys.simulate()
    for x in ({units[0]: 3., idle: 2.}, {idle: 2., 'idle': 2., 'C2': NAN}):
        HXN.approach_contributions = x
        simulate_strictly(sys)
        assert network_record(HXN) == reference
        assert HXN._resolved_contributions is None

def test_keys_of_auxiliaries_a_unit_does_not_build():
    # biosteam.HXutilities builds one exchanger per steam level it needs, so
    # a lower outlet temperature drops 'heat_exchangers[2]' (and [1]): a key
    # of it, the tuple or the old object, is valid but has no effect while
    # it is missing, and names the exchanger again once it is built; keys
    # of IDs the unit never gives an auxiliary still raise (the bare name of
    # a list of auxiliaries among them)
    bst.main_flowsheet.set_flowsheet('contributions_unbuilt')
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    feed = bst.Stream('feed', Water=500., units='kmol/hr', T=300., P=20e5)
    X1 = bst.HXutilities('X1', ins=feed, T=480., rigorous=False)
    hot = bst.Stream('hot', Water=800., Ethanol=50., units='kmol/hr', T=460.,
                     P=20e5)
    X2 = bst.HXutility('X2', ins=hot, T=320., rigorous=False)
    HXN = HeatExchangerNetwork('HXN', T_min_app=10.)
    sys = bst.System.from_units('contributions_unbuilt', units=[X1, X2, HXN])

    def run(T, x):
        X1.T = T
        HXN.approach_contributions = x
        simulate_strictly(sys)
        return network_record(HXN)

    tuple_key = {('X1', 'heat_exchangers[2]'): 3.}
    run(480., tuple_key)
    old = X1.heat_exchangers[2]
    assert [hx.ID for hx in X1.heat_exchangers] == [
        'heat_exchangers[0]', 'heat_exchangers[1]', 'heat_exchangers[2]']
    assert HXN.synthesis_info['approach_contributions'].tolist() == [
        3. if hx is old else 5. for hx in HXN.original_heat_exchangers]
    for T, N in ((440., 2), (400., 1)):
        reference = run(T, None)
        assert len(X1.heat_exchangers) == N
        for x in (tuple_key, {old: 3.}):
            assert run(T, x) == reference
            assert HXN._resolved_contributions is None
        # film coefficient keys of the facility's targets alike
        targets = HXN.targets()
        area = targets.area_target()
        assert targets.area_target(h={old: 2.}) == area
        assert targets.area_target(h={('X1', 'heat_exchangers[2]'): 2.}) == area
    # built again: the old object names the exchanger that took its place
    run(480., {old: 3.})
    new = X1.heat_exchangers[2]
    assert new is not old
    assert HXN.synthesis_info['approach_contributions'].tolist() == [
        3. if hx is new else 5. for hx in HXN.original_heat_exchangers]
    run(400., None)
    targets = HXN.targets()
    for key in (('X1', 'heat_exchanger[2]'), ('X1', 'heat_exchangers[x]'),
                ('X1', 'heat_exchangers[2'), ('X2', 'heat_exchangers[2]'),
                ('X9', 'heat_exchangers[2]'), ('X1', 'heat_exchangers')):
        with pytest.raises(ValueError, match='h keys that name no stream'):
            targets.area_target(h={key: 2.})
        HXN.approach_contributions = {key: 3.}
        with pytest.raises(ValueError, match='name no unit'):
            sys.simulate()
    # outside the facility, the universe is that of the heat utilities
    hus = bst.process_tools.heat_exchanger_utilities_from_units([X1, X2])
    assert hxn_synthesis._stream_contributions(hus, tuple_key) == [None, None]
    assert hxn_synthesis._stream_contributions(hus, {old: 3.}) == [None, None]

def test_contribution_keys_with_the_units_option():
    # keys may name the units of the units option and their auxiliary units,
    # idle or not, but no other unit; a callable option is called once per
    # simulation
    D1, D1_H1, D1_H2, F1 = quickstart('contributions_units_option')
    calls = []
    def listed():
        calls.append(None)
        return [D1, D1_H1, F1]
    x = {'D1': 1., F1: 1., 'D1_H1': 4.}
    hand = {'condenser': 1., 'reboiler': 1., 'heat_exchanger': 1., 'D1_H1': 4.}
    for k, option in enumerate([[D1, D1_H1, F1], listed]):
        HXN = HeatExchangerNetwork(f'HXN_{k}', T_min_app=5., units=option,
                                   approach_contributions=x | {D1.pump: 2.})
        sys = bst.System.from_units(f'contributions_units_option_{k}',
                                    units=[D1, D1_H1, D1_H2, F1, HXN])
        simulate_strictly(sys)
        assert HXN.synthesis_info['status'] == 'mer' # observed
        assert HXN.synthesis_info['approach_contributions'].tolist() == [
            hand[hx.ID] for hx in HXN.original_heat_exchangers]
        if isinstance(option, list):
            HXN.approach_contributions = x | {'D1_H2': 2.}
            with pytest.raises(ValueError, match=r"name no unit: 'D1_H2'"):
                sys.simulate()
    assert len(calls) == 1

# ---------------------------------------------------------------------------
# E. Cache
# ---------------------------------------------------------------------------

def test_cache_reuses_unless_contributions_change():
    # a cached network is reused while the explicit contributions resolve
    # to the same values per stream; T_min_app is not part of the key
    units = cp_units('contributions_cache', KEMP, 300.)
    C = {units[0]: 2., 'H2': 6.}
    sys, HXN = simulate_HXN(units, 10., 'contributions_cache',
                            cache_network=True, approach_contributions=C)
    by_ID = lambda: dict(zip([hx.ID for hx in HXN.original_heat_exchangers],
                             HXN._resolved_contributions))
    assert by_ID() == {'C1': 2., 'H2': 6., 'C3': None, 'H4': None}
    first = HXN.HXN_sys

    def simulate(reused):
        network = HXN.HXN_sys
        simulate_strictly(sys)
        assert (HXN.HXN_sys is network) == reused
        assert HXN._synthesis_options == (False,)
        return HXN.HXN_sys

    simulate(True)
    C['H2'] = 3.
    second = simulate(False)
    assert second is not first
    assert by_ID() == {'C1': 2., 'H2': 3., 'C3': None, 'H4': None}
    hand = {'C1': 2., 'H2': 3., 'C3': 5., 'H4': 5.}
    for hx in HXN.new_HXs:
        r = required(HXN, hx, lambda unit: hand[unit.ID])
        assert abs(hx.dT - (r - 1e-6)) <= 1e-12, hx.ID
    # observed: C1-H2 5 K, C1-H4 7 K, C3-H2 8 K
    assert {round(hx.dT, 6) for hx in HXN.new_HXs} == {4.999999, 6.999999,
                                                       7.999999}
    HXN.approach_contributions = lambda hu: {'C1': 2., 'H2': 3.}.get(hu.unit.ID)
    simulate(True)
    HXN.T_min_app = 8.
    simulate(True)
    HXN.approach_contributions = None
    simulate(False)
    assert HXN._resolved_contributions is None
    HXN.approach_contributions = {}
    simulate(True)
    HXN.T_min_app = 9.
    simulate(True)
