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

A. Default unchanged: unspecified contributions (None, nan, or every stream
   at ``T_min_app / 2``) give the default plans and problem tables bit for
   bit, compared in-process with ``float.hex`` and ``tobytes``.
B. Uniform contributions: every stream at ``x`` plans the network of
   ``T_min_app = 2x`` on a shifted scale offset by a constant, and on
   real-thermo cases all hot streams at ``a`` and all cold ones at ``b``
   give the default targets at ``T_min_app = a + b`` (references that do
   not use the contribution code); mixed contributions give targets between
   those of the smallest and of the largest pair requirement.
C. Non-uniform contributions against INDEPENDENT references: a closed-form
   constant-CP cascade with the classic symmetric shifts (hot streams down
   by ``c_j``, cold streams up by ``c_j``; hensmith's shifted scale is that
   scale less ``T_min_app / 2``), and a walk of every planned match on
   recomputed constant-CP temperatures that checks each pair's requirement
   at both ends (the minimum for linear profiles). The exactness check, the
   repair and the realization enforce each pair's own requirement.
D. Validation of the value domain (None or nan, else a finite number > 0)
   and of the sequence form.
"""
import math
import random
import numpy as np
import pytest
import biosteam as bst
from numpy.testing import assert_allclose
from hensmith import hxn_synthesis
from hensmith import _planner as P
from hensmith.hxn_synthesis import problem_table, synthesize_network
from test_hxn import cp_units, KEMP, R002 as R002_UNITS, planned_split
from test_hxn_regression import internal_approach, CASES
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
    pair_ends_ok(streams, {s['name']: cj for s, cj in zip(streams, c)}, net)
    assert net['plan'].splits
    assert net['plan'].info['min_approach_margin'] >= -1e-9

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
            required = dT + 5. if n in bad else dT
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

# ---------------------------------------------------------------------------
# D. Validation
# ---------------------------------------------------------------------------

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
