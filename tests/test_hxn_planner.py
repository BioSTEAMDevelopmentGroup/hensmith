# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Tests for the MER network planner (`hensmith._planner`) on numbers only:
constant-CP problems through `_plan_numeric` and piecewise-linear curves
(flats, collapsed vertical stretches) through `plan_network`. Everything is
checked against helpers written here: a constant-CP problem table, the pinch
design rules, a network walk, residual feasibility and match feasibility by
direct evaluation.
"""
import math
import random
from collections import Counter

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hensmith import _planner as P
from hensmith import _splitting as SP
from hxn_mer_cases import SPLIT as CORPUS_SPLIT, Q_UNITS, T_UNITS


# %% Independent helpers

def streams_from(rows):
    return [dict(name=str(n), kind='hot' if k == 'h' else 'cold',
                 T_in=float(a), T_out=float(b), CP=float(cp))
            for n, k, a, b, cp in rows]


def duty_scale(streams):
    return max(1., sum(abs(s['CP'] * (s['T_out'] - s['T_in']))
                       for s in streams))


def cascade(streams, dT):
    """Constant-CP problem table: (Qh, Qc, shifted temperatures where the
    feasible cascade is zero). Hot streams are shifted down by dT."""
    spans = [(s['T_out'] - dT, s['T_in'] - dT, s['CP']) if s['kind'] == 'hot'
             else (s['T_in'], s['T_out'], -s['CP']) for s in streams]
    Ts = sorted({t for lo, hi, _ in spans for t in (lo, hi)}, reverse=True)
    r = [0.]
    for hi, lo in zip(Ts, Ts[1:]):
        r.append(r[-1] + sum(cp * (hi - lo) for a, b, cp in spans
                             if a <= lo and b >= hi))
    Qh = max(0., -min(r))
    tol = 1e-9 * duty_scale(streams)
    return Qh, r[-1] + Qh, [T for T, v in zip(Ts, r) if v + Qh <= tol]


def _matching(left, right, ok):
    owner = {}

    def augment(i, seen):
        for j in range(len(right)):
            if j not in seen and ok(left[i], right[j]):
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False
    return all(augment(i, set()) for i in range(len(left)))


def needs_split(streams, dT):
    """Pinch design method: at every zero of the cascade each hot stream at
    the pinch (above) needs its own cold stream at the pinch with CP_hot <=
    CP_cold, and each cold stream at the pinch (below) its own hot stream
    with CP_hot >= CP_cold. A violation proves that MER needs splits."""
    for pc in cascade(streams, dT)[2]:
        ph = pc + dT
        hu = [s['CP'] for s in streams if s['kind'] == 'hot'
              and s['T_out'] <= ph < s['T_in']]
        cu = [s['CP'] for s in streams if s['kind'] == 'cold'
              and s['T_in'] <= pc < s['T_out']]
        cd = [s['CP'] for s in streams if s['kind'] == 'cold'
              and s['T_in'] < pc <= s['T_out']]
        hd = [s['CP'] for s in streams if s['kind'] == 'hot'
              and s['T_out'] < ph <= s['T_in']]
        if not _matching(hu, cu, lambda h, c: h <= c * (1. + 1e-12)):
            return True
        if not _matching(cd, hd, lambda c, h: h >= c * (1. - 1e-12)):
            return True
    return False


def check_network(streams, dT, net, tol=1e-6):
    """Walk every stream through its matches in flow order (hot_seq /
    cold_seq), recompute temperatures from Q / CP and assert: both ends of
    every match keep dT (constant CP: linear profiles), positive duties,
    utilities at the far end close every stream, global balance. Returns the
    hot and cold utility."""
    S = {s['name']: s for s in streams}
    qtol = tol * duty_scale(streams)
    temps = {}
    for name, s in S.items():
        hot = s['kind'] == 'hot'
        key = 'hot_seq' if hot else 'cold_seq'
        own = sorted((m for m in net['matches']
                      if m['hot' if hot else 'cold'] == name),
                     key=lambda m: m[key])
        assert [m[key] for m in own] == list(range(1, len(own) + 1))
        T = s['T_in']
        for m in own:
            assert m['Q'] > 0.
            T2 = T - m['Q'] / s['CP'] if hot else T + m['Q'] / s['CP']
            recorded = ((m['T_hot_in'], m['T_hot_out']) if hot
                        else (m['T_cold_in'], m['T_cold_out']))
            assert_allclose(recorded, (T, T2), rtol=1e-9, atol=tol)
            temps.setdefault(id(m), {})[hot] = (T, T2)
            T = T2
        need = s['CP'] * ((T - s['T_out']) if hot else (s['T_out'] - T))
        u = (net['cold_utility'] if hot else net['hot_utility']).get(name, 0.)
        assert need >= -qtol
        assert abs(u - need) <= qtol
    for m in net['matches']:
        (hi, ho), (ci, co) = temps[id(m)][True], temps[id(m)][False]
        assert min(hi - co, ho - ci) >= dT - tol
    Qh = sum(net['hot_utility'].values())
    Qc = sum(net['cold_utility'].values())
    dh = sum(s['CP'] * (s['T_in'] - s['T_out']) for s in streams
             if s['kind'] == 'hot')
    dc = sum(s['CP'] * (s['T_out'] - s['T_in']) for s in streams
             if s['kind'] == 'cold')
    assert abs(dh + Qh - dc - Qc) <= 10. * qtol
    return Qh, Qc


def assert_mer(rows, dT, **kw):
    streams = streams_from(rows)
    net = P._plan_numeric(streams, dT, **kw)
    Qh, Qc = check_network(streams, dT, net)
    Qh_t, Qc_t, _ = cascade(streams, dT)
    tol = 1e-6 * duty_scale(streams)
    assert net['status'] == 'mer'
    assert abs(Qh - Qh_t) <= tol and abs(Qc - Qc_t) <= tol
    return net


def root_proof(plan):
    return any(s['proof'] is not None
               and s['proof'].get('rule') in ('outward', 'inward')
               for s in plan.info['sides'].values())


def random_problem(rng, n_lo=2, n_hi=6):
    """Random constant-CP problem on a 5 K grid (hot and cold streams)."""
    n = rng.randint(n_lo, n_hi)
    nh = rng.randint(1, n - 1)
    rows = []
    for k in range(n):
        hot = k < nh
        a, b = sorted(rng.sample(range(40 if hot else 20, 260 if hot else 240,
                                       5), 2))
        cp = rng.choice([.5, 1., 1.5, 2., 3., 4.])
        rows.append((f'H{k}' if hot else f'C{k}', 'h' if hot else 'c',
                     b if hot else a, a if hot else b, cp))
    return rows, float(rng.choice([5, 10, 20]))


def pinch_problem(rng):
    """Random constant-CP problem built around a pinch at P (hot streams
    ending or crossing at P + dT, cold streams starting or crossing at P),
    where the pinch design rules often fail."""
    dT = 10.
    P0 = float(rng.choice(range(100, 200, 10)))
    cps = [1., 1.5, 2., 3., 4.]
    rows = []
    for k in range(rng.randint(1, 4)):
        T_out = P0 + dT - rng.choice([0, 0, 10, 20, 40])
        rows.append((f'H{k}', 'h', P0 + dT + rng.choice(range(10, 100, 5)),
                     T_out, rng.choice(cps)))
    for k in range(rng.randint(1, 4)):
        T_in = P0 - rng.choice([0, 0, 10, 20, 40])
        rows.append((f'C{k}', 'c', T_in, P0 + rng.choice(range(10, 100, 5)),
                     rng.choice(cps)))
    return rows, dT


def random_level_curve(rng, flats=True, pieces=(1, 4), grid=False):
    """Random level curve; on an integer `grid` knots and levels coincide
    between curves (flats at the other curve's levels, shared knots)."""
    n = rng.randint(*pieces)
    if grid:
        q, y = [0.], [float(rng.randint(0, 6))]
        for _ in range(n):
            q.append(q[-1] + rng.randint(1, 5))
            y.append(y[-1] + (0. if flats and rng.random() < 0.35
                              else rng.randint(1, 4)))
        return P._LevelCurve(q, y)
    q, y = [0.], [rng.uniform(0., 20.)]
    for _ in range(n):
        q.append(q[-1] + rng.uniform(0.5, 10.))
        y.append(y[-1] + (0. if flats and rng.random() < 0.3
                          else rng.uniform(0.5, 15.)))
    return P._LevelCurve(q, y)


def random_curve_problem(rng, n_hi=6, flat_p=0.3):
    """Random piecewise-linear streams with flats on a 5 K grid (knots,
    is_hot, T_min_app)."""
    kn, hot = [], []
    for k in range(rng.randint(2, n_hi)):
        T, H = [float(rng.choice(range(20, 200, 5)))], [0.]
        for _ in range(rng.randint(1, 3)):
            flat = rng.random() < flat_p
            T.append(T[-1] + (0. if flat else rng.choice(range(5, 60, 5))))
            H.append(H[-1] + rng.choice(range(5, 100, 5)))
        kn.append((T, H))
        hot.append(k == 0 or (k > 1 and rng.random() < 0.5))
    return kn, hot, float(rng.choice([5, 10]))


def match_ok(cm, a, cf, b, x, tol=1e-9):
    """Direct check of phi_m(a + t) >= phi_f(b + t) on [0, x] at every knot
    of both curves inside the window (exact for piecewise-linear curves)."""
    ts = np.concatenate(([0., x], cm.qa - a, cf.qa - b))
    ts = ts[(ts >= 0.) & (ts <= x)]
    g = (np.interp(a + ts, cm.qa, cm.ya) - np.interp(b + ts, cf.qa, cf.ya))
    return bool(g.min() >= -tol)


def residual_slack(side, a, b):
    """min over levels of S - D (inclusive and exclusive) by direct
    evaluation of every curve at every knot and frontier level."""
    levels = set()
    for curves, front in ((side.musts, a), (side.flexes, b)):
        for c, f in zip(curves, front):
            levels.add(c.at(f))
            levels.update(y for x, y in zip(c.q, c.y) if x > f)
    worst = math.inf
    for L in levels:
        for le in (True, False):
            S = sum(max(0., (c.x_le(L) if le else c.x_lt(L)) - f)
                    for c, f in zip(side.flexes, b))
            D = sum(max(0., (c.x_le(L) if le else c.x_lt(L)) - f)
                    for c, f in zip(side.musts, a))
            worst = min(worst, S - D)
    return worst


def bisect(ok, lo, hi, largest=True, it=80):
    """Largest (or smallest) x in [lo, hi] with ok(x), ok monotone."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if ok(mid) == largest:
            lo = mid
        else:
            hi = mid
    return lo if largest else hi


# %% Stream data (constant CP): (name, 'h' | 'c', T_in, T_out, CP)

R002 = [('H1', 'h', 220, 50, 4), ('C1', 'c', 140, 170, 1),
        ('C2', 'c', 100, 120, 1), ('C3', 'c', 120, 250, 2)]
LINNHOFF4 = [('C1', 'c', 20, 135, 2), ('H2', 'h', 170, 60, 3),
             ('C3', 'c', 80, 140, 4), ('H4', 'h', 150, 30, 1.5)]
REPEATED_PAIR = {  # literature cases that need a repeated pair
    '4sp2_dt20F': (20, [
        ('H1', 'h', 500, 110, 20000), ('H2', 'h', 430, 230, 50000),
        ('H3', 'h', 400, 110, 30000), ('C1', 'c', 25, 420, 70000)]),
    'smith2005_exr18_9_threshold': (20, [
        ('1', 'h', 500, 100, 4.0), ('2', 'c', 50, 450, 1.0),
        ('3', 'c', 60, 400, 1.0), ('4', 'c', 40, 420, 0.75)]),
    'nptel_t5_5_threshold_dT5': (5, [
        ('Hot-1', 'h', 180, 60, 3.0), ('Hot-2', 'h', 140, 30, 1.5),
        ('Cold-1', 'c', 20, 135, 2.0), ('Cold-2', 'c', 80, 140, 4.0)]),
}
HARD = {  # benchmark and fresh held-out problems that defeated a prototype
    'rnd_nested_n2-6_s183': (5, [
        ('H1', 'h', 240, 115, 6.5), ('C1', 'c', 165, 270, 2.0),
        ('C2', 'c', 190, 220, 3.2), ('C3', 'c', 205, 220, 1.2),
        ('C4', 'c', 190, 220, 2.2)]),
    'rnd_nested_n2-6_s237': (5, [
        ('H1', 'h', 195, 50, 4.4), ('C1', 'c', 130, 200, 1.9),
        ('C2', 'c', 140, 150, 3.4), ('C3', 'c', 130, 160, 1.3),
        ('C4', 'c', 150, 160, 4.0), ('C5', 'c', 140, 150, 2.1)]),
    's13_111_3x9': (10, [
        ('H1', 'h', 145, 75, 1.5), ('H2', 'h', 205, 75, 1),
        ('H3', 'h', 115, 55, 2), ('C1', 'c', 90, 155, 2),
        ('C2', 'c', 45, 120, 3), ('C3', 'c', 105, 235, 3),
        ('C4', 'c', 175, 220, 3), ('C5', 'c', 90, 195, 2),
        ('C6', 'c', 155, 165, 4), ('C7', 'c', 55, 220, 0.5),
        ('C8', 'c', 75, 90, 0.5), ('C9', 'c', 140, 205, 4)]),
    'rnd_pinch_design_n7-12_s54': (5, [
        ('H1', 'h', 190, 80, 15.2), ('H2', 'h', 230, 160, 9.0),
        ('C1', 'c', 155, 240, 30.4), ('C2', 'c', 110, 250, 7.6),
        ('C3', 'c', 155, 200, 0.84), ('H3', 'h', 160, 140, 6.7),
        ('C4', 'c', 155, 210, 9.0), ('C5', 'c', 200, 240, 3.5),
        ('H4', 'h', 210, 160, 0.7), ('C6', 'c', 130, 250, 6.7)]),
    'hold_threshold_n13-20_s100208': (5, [
        ('C1', 'c', 110, 225, 6.0), ('C2', 'c', 115, 200, 7.5),
        ('C3', 'c', 175, 180, 2.5), ('C4', 'c', 90, 175, 30.0),
        ('C5', 'c', 20, 215, 1.5), ('C6', 'c', 85, 165, 5.0),
        ('C7', 'c', 155, 240, 10.0), ('C8', 'c', 105, 190, 6.0),
        ('H1', 'h', 205, 170, 1.0), ('H2', 'h', 305, 125, 4.0),
        ('H3', 'h', 255, 120, 0.5), ('H4', 'h', 210, 120, 10.0),
        ('H5', 'h', 250, 185, 2.0), ('H6', 'h', 280, 180, 3.0),
        ('H7', 'h', 315, 60, 8.0), ('H8', 'h', 275, 65, 1.0),
        ('H9', 'h', 280, 205, 3.0), ('H10', 'h', 225, 150, 4.0),
        ('H11', 'h', 290, 125, 1.0), ('H12', 'h', 320, 210, 10.0)]),
    'hold_pinch_design_n13-20_s100271': (20, [
        ('H1', 'h', 166, 78, 4.9), ('H2', 'h', 317, 197, 3.9),
        ('C1', 'c', 79, 204, 0.65), ('H3', 'h', 239, 151, 7.9),
        ('C2', 'c', 209, 277, 1.2), ('C3', 'c', 110, 157, 1.4),
        ('H4', 'h', 192, 162, 18.96), ('H5', 'h', 242, 170, 1.9),
        ('C4', 'c', 44, 145, 3.1), ('C5', 'c', 172, 225, 2.28),
        ('C6', 'c', 80, 109, 9.4), ('H6', 'h', 215, 119, 1.4),
        ('C7', 'c', 124, 258, 15.8), ('C8', 'c', 103, 165, 0.99),
        ('C9', 'c', 172, 246, 2.8), ('C10', 'c', 119, 148, 0.5)]),
    'hold_nested_n21-40_s100233': (5, [
        ('C1', 'c', 135, 262, 3.8), ('C2', 'c', 135, 296, 4.3),
        ('H1', 'h', 229, 140, 1.1), ('H2', 'h', 210, 133, 2.1),
        ('H3', 'h', 225, 207, 2.3), ('H4', 'h', 222, 187, 3.0),
        ('H5', 'h', 222, 213, 0.6), ('H6', 'h', 209, 195, 2.1),
        ('H7', 'h', 216, 177, 2.5), ('H8', 'h', 185, 178, 2.7),
        ('H9', 'h', 235, 207, 0.3), ('C3', 'c', 148, 186, 2.9),
        ('H10', 'h', 379, 265, 9.7), ('H11', 'h', 379, 268, 3.3),
        ('C4', 'c', 256, 366, 8.4), ('C5', 'c', 33, 89, 7.2),
        ('H12', 'h', 380, 337, 6.3), ('H13', 'h', 274, 249, 2.3),
        ('C6', 'c', 72, 231, 5.2), ('H14', 'h', 244, 93, 4.9),
        ('C7', 'c', 20, 127, 8.1), ('C8', 'c', 35, 55, 2.7),
        ('C9', 'c', 310, 354, 8.8), ('C10', 'c', 371, 376, 0.8),
        ('C11', 'c', 42, 59, 8.0), ('H15', 'h', 124, 63, 6.7),
        ('H16', 'h', 310, 277, 5.0), ('C12', 'c', 262, 318, 3.4),
        ('H17', 'h', 93, 25, 6.1), ('H18', 'h', 138, 121, 1.3),
        ('H19', 'h', 207, 106, 9.3)]),
}
SPLIT = {  # the pinch design rules prove that MER needs stream splits
    'smith2005_exr18_4': (10, [
        ('1', 'c', 18, 123, 0.0933), ('2', 'c', 118, 193, 0.1961),
        ('3', 'c', 189, 286, 0.1796), ('4', 'h', 159, 77, 0.2285),
        ('5', 'h', 267, 80, 0.0204), ('6', 'h', 343, 90, 0.0538)]),
    'smith2005_ex16_5_five_stream': (20, [
        ('1', 'h', 450, 50, 0.25), ('2', 'h', 50, 40, 1.5),
        ('3', 'c', 30, 400, 0.22), ('4', 'c', 30, 400, 0.05),
        ('5', 'c', 120, 121, 22.0)]),
    'rnd_uniform_n2-6_s327': (5, [
        ('C1', 'c', 111, 159, 2.4), ('C2', 'c', 91, 171, 3.3),
        ('H1', 'h', 162, 46, 9.8)]),
}


# %% 1. Level curves

def test_level_curve_flats_and_verticals():
    # a vertical stretch at q = 5 (levels 12 -> 13) collapses to the
    # conservative level: the lowest for a must, the highest for a flex
    c = P._LevelCurve([0, 2, 5, 5, 8], [10, 12, 12, 13, 16], role='must')
    assert c.q == [0., 2., 5., 8.] and c.y == [10., 12., 12., 16.]
    f = P._LevelCurve([0, 2, 5, 5, 8], [10, 12, 12, 13, 16], role='flex')
    assert f.q == [0., 2., 5., 8.] and f.y == [10., 12., 13., 16.]
    assert c.flats == (12.,) and c.Q == 8.
    assert c.at(1.) == 11. and c.at(3.) == 12. and c.at(-1.) == 10.
    assert c.at(99.) == 16.
    assert c.x_le(12.) == 5. and c.x_lt(12.) == 2.   # both ends of the flat
    assert c.x_le(9.) == 0. and c.x_lt(10.) == 0. and c.x_le(20.) == 8.
    assert c.x_lt(20.) == 8.
    assert c.slope_right(2.) == 0. and c.slope_left(2.) == 1.
    assert c.slope_right(5.) == pytest.approx(4. / 3.)
    assert c.slope_left(5.) == 0.
    # round trips off the flat, and the vectorised versions
    for q in (0.5, 1.7, 5.5, 7.9):
        assert c.x_le(c.at(q)) == pytest.approx(q)
        assert c.x_lt(c.at(q)) == pytest.approx(q)
    L = np.linspace(8., 18., 41)
    assert_allclose(c.x_le_many(L), [c.x_le(v) for v in L])
    assert_allclose(c.x_lt_many(L), [c.x_lt(v) for v in L])
    assert_allclose(c.at_many(np.linspace(0., 8., 17)),
                    [c.at(v) for v in np.linspace(0., 8., 17)])


def test_stream_knots_simplified():
    # collinear interior knots are dropped (constant CP -> one segment);
    # a flat is kept; a vertical end stretch is collapsed conservatively
    T = np.linspace(300., 400., 65)
    curves = P._stream_curves(
        [(T, 2. * (T - 300.)),
         ([300., 350., 350., 400.], [0., 50., 80., 130.]),
         ([300., 310., 400.], [0., 0., 90.])], [False, False, True], 10.)
    assert curves[0].T.tolist() == [300., 400.]
    assert curves[0].H.tolist() == [0., 200.]
    assert curves[1].T.tolist() == [300., 350., 350., 400.]
    # hot stream: lowest temperature at the vertical stretch (shifted by 10)
    assert curves[2].T.tolist() == [290., 390.]
    assert curves[2].H.tolist() == [0., 90.]
    # round-off below the tolerances is absorbed ...
    curves = P._stream_curves(
        [([300., 350., 350. - 1e-10, 400.], [0., 50., 60., 100.])], [False],
        10.)
    assert curves[0].T.tolist() == [300., 350., 350., 400.]
    # ... but knots that really decrease are rejected, not silently
    # flattened into a different stream (here a point load at 400 K)
    for kn in (([400., 300.], [0., 100.]), ([300., 400.], [100., 0.]),
               ([300., 350., 340., 400.], [0., 10., 20., 30.])):
        with pytest.raises(ValueError, match='non-decreasing'):
            P.plan_network([kn, ([280., 380.], [0., 100.])], [True, False],
                           10.)
    with pytest.raises(ValueError, match='finite'):
        P.plan_network([([300., 400.], [0., 100.]),
                        ([280., 380.], [0., 100.])], [True, False], math.nan)


# %% 2. Internal pinch

def test_max_duty_stops_at_internal_crossing():
    # a condensing must (flat at 105 on the shifted scale) against a cold
    # flex: both terminals are feasible, the knot at q = 80 is not
    cm = P._LevelCurve([0, 20, 80, 110], [95, 105, 105, 135])
    cf = P._LevelCurve([0, 110], [90, 125])
    assert cm.at(0.) - cf.at(0.) >= 0. and cm.at(110.) - cf.at(110.) >= 0.
    x = P._max_duty(cm, 0., cf, 0., 110., 1e-12)
    assert x == pytest.approx(15. * 110. / 35.)
    assert match_ok(cm, 0., cf, 0., x) and not match_ok(cm, 0., cf, 0., 80.)


# %% 3. x_res

def test_xres_closed_form_matches_bisection():
    rng = random.Random(3)
    checked = 0
    for _ in range(400):
        M, F = rng.randint(1, 3), rng.randint(1, 3)
        side = P._Side('above', [random_level_curve(rng) for _ in range(M)],
                       [random_level_curve(rng) for _ in range(F)],
                       1e-11 * 100., 1e-11 * 100.)
        a = [rng.choice((0., rng.uniform(0., 0.6 * q))) for q in side.Qm]
        b = [rng.choice((0., rng.uniform(0., 0.6 * q))) for q in side.Qf]
        if residual_slack(side, a, b) < 0.:
            continue
        d = side.analyse(a, b)
        assert d.slack == pytest.approx(residual_slack(side, a, b), abs=1e-9)
        for i in range(M):
            xres = side.xres_for(i, d)
            for j in range(F):
                lim = min(side.Qm[i] - a[i], side.Qf[j] - b[j])

                def ok(x):
                    a2, b2 = list(a), list(b)
                    a2[i] += x
                    b2[j] += x
                    return residual_slack(side, a2, b2) >= -1e-9
                expected = lim if ok(lim) else bisect(ok, 0., lim)
                assert min(xres[j], lim) == pytest.approx(expected, abs=1e-6)
                checked += 1
        if checked >= 200:
            break
    assert checked >= 200


# %% 4. Closed-form events

@pytest.mark.parametrize('flats', [False, True])
def test_events_match_bisection(flats):
    rng = random.Random(4 + flats)
    n = {'e1b': 0, 'e2b': 0, 'e3': 0}
    for _ in range(300):
        pieces = (1, 1) if rng.random() < 0.3 else (1, 4)
        cm = random_level_curve(rng, flats, pieces)
        cf = random_level_curve(rng, flats, pieces)
        a = rng.choice((0., rng.uniform(0., 0.5 * cm.Q)))
        b = rng.choice((0., rng.uniform(0., 0.5 * cf.Q)))
        rem, cap = cm.Q - a, cf.Q - b
        # E1b: largest flex advance after which the flex still finishes
        # the must in one match (feasibility falls with the advance)
        x = P._flex_stop_for(cm, a, rem, cf, b, cap)
        if cap < rem:
            assert x is None
        else:
            def ok(x):
                return match_ok(cm, a, cf, b + x, rem)
            if not ok(0.):
                assert x < 1e-7
            else:
                hi = cap - rem
                expected = hi if ok(hi) else bisect(ok, 0., hi)
                assert x == pytest.approx(expected, abs=1e-6)
                n['e1b'] += 1
        # E2b: smallest must advance after which the flex finishes it
        # (feasibility rises with the advance)
        x = P._must_stop_for(cm, a, rem, cf, b, cap, rem + 1.)

        def ok2(x):
            return rem - x <= cap + 1e-12 and match_ok(cm, a + x, cf, b,
                                                       rem - x)
        if not ok2(rem):   # the flex starts above the must's last level
            assert x is None or x >= rem - 1e-7
        else:
            expected = 0. if ok2(0.) else bisect(ok2, 0., rem, largest=False)
            assert x == pytest.approx(expected, abs=1e-6)
            n['e2b'] += 1
        # E3: the first return of the flex to the must's level (within a
        # window where neither curve runs out)
        R = rng.uniform(0., 0.5 * cap)
        xt = min(rem, cap - R)
        x = P._return_duty(cm, a, cf, b, R, xt)
        ts = np.linspace(0., xt, 4001)
        h = np.interp(a + ts, cm.qa, cm.ya) - np.interp(b + R + ts, cf.qa,
                                                         cf.ya)
        if b + R >= cf.Q or h[0] >= 0.:
            assert x is None
            continue
        up = np.flatnonzero(h >= 0.)
        if up.size == 0:
            assert x is None or x >= xt - 1e-6
            continue
        k = int(up[0])
        expected = bisect(lambda t: np.interp(a + t, cm.qa, cm.ya)
                          - np.interp(b + R + t, cf.qa, cf.ya) < 0.,
                          ts[k - 1], ts[k])
        assert x == pytest.approx(expected, abs=1e-6)
        n['e3'] += 1
    assert min(n.values()) >= 30


def test_events_match_bisection_on_coincident_levels():
    # integer knots and levels: flats of one curve sit exactly at the
    # levels of the other, frontiers sit on knots (exactness of E1b/E2b at
    # a flat's left limit, E3 on touching knots)
    rng = random.Random(44)
    n = {'e1b': 0, 'e2b': 0, 'e3': 0}
    for _ in range(1500):
        cm = random_level_curve(rng, True, (1, 5), grid=True)
        cf = random_level_curve(rng, True, (1, 5), grid=True)
        a = min(float(rng.choice((0., rng.randint(0, int(cm.Q) - 1),
                                  rng.uniform(0., 0.7 * cm.Q)))), cm.Q - .5)
        b = min(float(rng.choice((0., rng.randint(0, int(cf.Q) - 1),
                                  rng.uniform(0., 0.7 * cf.Q)))), cf.Q - .5)
        rem, cap = cm.Q - a, cf.Q - b
        x = P._flex_stop_for(cm, a, rem, cf, b, cap)
        if cap >= rem and match_ok(cm, a, cf, b, rem):
            def ok(x):
                return match_ok(cm, a, cf, b + x, rem)
            hi = cap - rem
            expected = hi if ok(hi) else bisect(ok, 0., hi)
            assert x == pytest.approx(expected, abs=1e-7)
            n['e1b'] += 1
        x = P._must_stop_for(cm, a, rem, cf, b, cap, rem + 1.)

        def ok2(x):
            return rem - x <= cap + 1e-12 and match_ok(cm, a + x, cf, b,
                                                       rem - x)
        if ok2(rem):
            expected = 0. if ok2(0.) else bisect(ok2, 0., rem, largest=False)
            if expected >= rem - 1e-6:   # only the empty match finishes it
                assert x is None or x >= rem - 1e-6
            else:
                assert x == pytest.approx(expected, abs=1e-7)
                n['e2b'] += 1
        R = float(rng.choice((0., 1., rng.uniform(0., 0.6 * cap))))
        xt = min(rem, cap - R)
        if xt <= 0. or b + R >= cf.Q:
            continue
        x = P._return_duty(cm, a, cf, b, R, xt)
        ts = np.unique(np.concatenate(([0., xt], cm.qa - a,
                                       cf.qa - b - R)))
        ts = ts[(ts >= 0.) & (ts <= xt)]
        h = (np.interp(a + ts, cm.qa, cm.ya)
             - np.interp(b + R + ts, cf.qa, cf.ya))
        if h[0] >= 0.:
            assert x is None
            continue
        up = np.flatnonzero(h >= 0.)   # h is linear between these points
        if up.size == 0:
            assert x is None or x >= xt - 1e-9
            continue
        k = int(up[0])
        expected = ts[k - 1] + (ts[k] - ts[k - 1]) * (-h[k - 1]) / (
            h[k] - h[k - 1])
        assert x == pytest.approx(expected, abs=1e-9)
        n['e3'] += 1
    assert min(n.values()) >= 100


def test_pruning_is_sound_on_curves_with_flats(monkeypatch):
    # every pruning test (pinch rules at tight levels, dead must) must be a
    # necessary condition: with both switched off the search may only get
    # slower. Pruning removes nodes from the same depth-first order, so if
    # the relaxed search finds an MER plan the pruned one must find it too;
    # a root proof it contradicts would be a false split proof.
    rng = random.Random(77)
    problems = [random_curve_problem(rng) for _ in range(400)]
    problems = [p for p in problems if any(p[1]) and not all(p[1])]
    normal = [P.plan_network(*p) for p in problems]

    def no_dead_must(self, a, b, d, open_m, open_f):
        s = self.side
        lam = {i: s.musts[i].at(a[i]) for i in open_m}
        mu = {j: s.flexes[j].at(b[j]) for j in open_f}
        rem_m = {i: s.Qm[i] - a[i] for i in open_m}
        rem_f = {j: s.Qf[j] - b[j] for j in open_f}
        return self._moves(a, b, d, open_m, open_f, lam, mu, rem_m, rem_f)
    monkeypatch.setattr(P._Side, 'rules_violation', lambda *args: None)
    monkeypatch.setattr(P._Search, '_candidates', no_dead_must)
    n_mer = 0
    for p, plan in zip(problems, normal):
        assert plan.info['dropped'] == []
        relaxed = P.plan_network(*p)
        assert relaxed.info['dropped'] == []
        if relaxed.status == 'mer':
            assert plan.status == 'mer'
            n_mer += 1
    assert n_mer >= 300


def test_vectorised_events_match_per_pair_events():
    # constant CP sides use array versions of E1, E1b, E2 and E2b
    rng = random.Random(5)
    for _ in range(60):
        side = P._Side('above', [random_level_curve(rng, False, (1, 1))
                                 for _ in range(rng.randint(2, 4))],
                       [random_level_curve(rng, False, (1, 1))
                        for _ in range(rng.randint(2, 4))], 1e-9, 1e-9)
        assert side.linear
        a = [rng.uniform(0., 0.5 * q) for q in side.Qm]
        b = [rng.uniform(0., 0.5 * q) for q in side.Qf]
        lam = {i: c.at(a[i]) for i, c in enumerate(side.musts)}
        mu = {j: c.at(b[j]) for j, c in enumerate(side.flexes)}
        rem_m = {i: side.Qm[i] - a[i] for i in lam}
        rem_f = {j: side.Qf[j] - b[j] for j in mu}
        srch = P._Search(side, 'full', None, True, 1e9)
        args = (a, b, lam, mu, rem_m, rem_f, list(lam), list(mu))
        for i in lam:
            for j in mu:
                xt = min(rem_m[i], rem_f[j])
                vec = srch._duties(i, j, xt, *args)
                side.linear = False
                ref = srch._duties(i, j, xt, *args)
                side.linear = True
                assert_allclose(vec, ref, rtol=1e-9, atol=1e-9)


# %% 5. Flat-aware pinch rules

def flat_case_plan(case, dT):
    names = list(case)
    plan = P.plan_network([(case[n][1], case[n][2]) for n in names],
                          [case[n][0] for n in names], dT)
    pairs = sorted((e.side, names[e.hot], names[e.cold], round(e.Q, 9))
                   for e in plan.exchangers)
    return plan, pairs


def test_flat_rules_boiler_and_condenser_at_the_pinch():
    # judge 1, case A: a cold stream boiling at the pinch serves two hot
    # musts in series above it; case B: a hot stream condensing at the
    # pinch serves two cold musts below it (name: (hot, T, H) knots)
    A = dict(H1=(True, [150, 200], [0, 50]), H2=(True, [150, 200], [0, 50]),
             H3=(True, [50, 150], [0, 100]), Cb=(False, [140, 140], [0, 100]),
             C2=(False, [150, 250], [0, 100]))
    plan, pairs = flat_case_plan(A, 10.)
    assert plan.status == 'mer' and plan.cut == 'above'
    assert pairs == [('above', 'H1', 'Cb', 50.), ('above', 'H2', 'Cb', 50.)]
    assert plan.info['sides']['above']['proof'] is None
    assert (plan.Q_hot, plan.Q_cold) == (100., 100.)
    B = dict(Hc=(True, [150, 150], [0, 100]), C1=(False, [100, 140], [0, 40]),
             C2=(False, [100, 140], [0, 60]), C3=(False, [140, 200], [0, 60]))
    plan, pairs = flat_case_plan(B, 10.)
    assert plan.status == 'mer' and plan.cut == 'below'
    assert pairs == [('below', 'Hc', 'C1', 40.), ('below', 'Hc', 'C2', 60.)]
    assert (plan.Q_hot, plan.Q_cold) == (60., 0.)
    # d1 pw_demo: a flex boiling exactly at the pinch level
    PW = dict(H1=(True, [105, 130], [0, 50]), H2=(True, [105, 120], [0, 30]),
              Cb=(False, [100, 100, 110], [0, 60, 70]),
              C2=(False, [100, 140], [0, 40]))
    plan, pairs = flat_case_plan(PW, 5.)
    assert plan.status == 'mer' and (plan.Q_hot, plan.Q_cold) == (30., 0.)
    assert ('above', 'H1', 'Cb', 50.) in pairs
    assert plan.info['min_approach'] >= 5.


def test_number_rule_without_flats_is_a_root_proof():
    # two hot streams at the pinch above it but only one cold stream
    streams = streams_from(SPLIT['smith2005_exr18_4'][1])
    net = P._plan_numeric(streams, 10.)
    proof = net['plan'].info['sides']['above']['proof']
    assert net['status'] == 'best_effort'
    assert proof['rule'] == 'outward' and sorted(proof['musts']) == [4, 5]
    assert proof['flexes'] == [1]


# %% 6. Root proofs = pinch design rules (constant CP)

def test_root_proof_agrees_with_pinch_rules():
    rng = random.Random(6)
    n_split = 0
    for k in range(200):
        rows, dT = random_problem(rng) if k % 2 else pinch_problem(rng)
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT, work_scale=0.2)
        assert root_proof(net['plan']) == needs_split(streams, dT)
        n_split += needs_split(streams, dT)
    for dT, rows in list(SPLIT.values()) + list(REPEATED_PAIR.values()):
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT, work_scale=0.2)
        assert root_proof(net['plan']) == needs_split(streams, dT)
    assert n_split >= 10


# %% 7-9. Unique, repeated-pair and hard networks

def test_r002_unique_network_with_repeated_pair():
    net = assert_mer(R002, 10.)
    got = sorted((m['side'], m['hot'], m['cold'], round(m['Q'], 9),
                  m['pair_index']) for m in net['matches'])
    assert got == [('below', 'H1', 'C1', 30., 1),
                   ('below', 'H1', 'C2', 20., 1),
                   ('below', 'H1', 'C3', 20., 2),
                   ('below', 'H1', 'C3', 160., 1)]
    assert net['targets'] == dict(Qh=80., Qc=450.)


@pytest.mark.parametrize('name', sorted(REPEATED_PAIR))
def test_repeated_pair_literature_cases(name):
    dT, rows = REPEATED_PAIR[name]
    net = assert_mer(rows, dT)
    pairs = [(m['side'], m['hot'], m['cold']) for m in net['matches']]
    assert len(set(pairs)) < len(pairs)


@pytest.mark.parametrize('name', sorted(HARD))
def test_hard_cases_reach_mer(name):
    dT, rows = HARD[name]
    assert_mer(rows, dT)


# %% 10-12. Determinism, budgets and best effort

def fingerprint(net):
    return ([(m['side'], m['hot'], m['cold'], m['Q'], m['hot_seq'],
              m['cold_seq']) for m in net['matches']],
            sorted(net['hot_utility'].items()),
            sorted(net['cold_utility'].items()))


def test_deterministic_and_budget_scale():
    cases = [(10., R002), (10., LINNHOFF4), HARD['rnd_nested_n2-6_s183'],
             HARD['hold_threshold_n13-20_s100208'],
             SPLIT['rnd_uniform_n2-6_s327']]
    for dT, rows in cases:
        streams = streams_from(rows)
        one = P._plan_numeric(streams, dT)
        two = P._plan_numeric(streams, dT)
        assert fingerprint(one) == fingerprint(two)   # bitwise identical
        ten = P._plan_numeric(streams, dT, work_scale=10.)
        assert ten['status'] == one['status']
        if one['status'] == 'mer':
            check_network(streams, dT, ten)


def test_budget_exhaustion_gives_feasible_best_effort(monkeypatch):
    monkeypatch.setattr(P, '_SCHEDULE', tuple(p[:3] + (1.,)
                                              for p in P._SCHEDULE))
    status = {}
    for name, (dT, rows) in [('r002', (10., R002)),
                             ('linnhoff4', (10., LINNHOFF4)),
                             ('s237', HARD['rnd_nested_n2-6_s237'])]:
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT)
        Qh, Qc = check_network(streams, dT, net)
        Qh_t, Qc_t, _ = cascade(streams, dT)
        tol = 1e-9 * duty_scale(streams)
        sides = net['plan'].info['sides'].values()
        assert all(s['status'] in ('best_effort', 'trivial') for s in sides)
        assert Qh >= Qh_t - tol and Qc >= Qc_t - tol
        assert Qh - Qh_t == pytest.approx(Qc - Qc_t, abs=tol)
        assert net['penalty'] == pytest.approx(Qh - Qh_t, abs=tol)
        # the status comes from the achieved utilities, not from the sides
        assert net['status'] == ('mer' if Qh - Qh_t <= tol else 'best_effort')
        status[name] = net['status']
    # a greedy dive reaches the targets of r002 although the MER search ran
    # out of budget: that network is MER and must be reported as such (the
    # side flags alone would call it best effort)
    assert status['r002'] == 'mer' and 'best_effort' in status.values()


@pytest.mark.parametrize('name', sorted(SPLIT))
def test_best_effort_on_split_cases(name):
    dT, rows = SPLIT[name]
    streams = streams_from(rows)
    net = P._plan_numeric(streams, dT)
    plan = net['plan']
    Qh, Qc = check_network(streams, dT, net)
    Qh_t, Qc_t, _ = cascade(streams, dT)
    tol = 1e-9 * duty_scale(streams)
    assert net['status'] == 'best_effort' and root_proof(plan)
    assert Qh - Qh_t > tol                       # never beats MER
    assert Qh - Qh_t == pytest.approx(Qc - Qc_t, abs=tol)
    # every best-effort side stays within the units cap and is no worse
    # than the best greedy dive within the cap
    curves = P._stream_curves(
        [([min(s['T_in'], s['T_out']), max(s['T_in'], s['T_out'])],
          [0., s['CP'] * abs(s['T_in'] - s['T_out'])]) for s in streams],
        [s['kind'] == 'hot' for s in streams], dT)
    scale = sum(c.duty for c in curves)
    span = max(c.T[-1] for c in curves) - min(c.T[0] for c in curves)
    sides = P._sides(curves, P._cascade(curves, scale),
                     P._REL_Q * max(scale, 1.), P._REL_T * max(span, 1.))
    for sname, info in plan.info['sides'].items():
        if info['status'] != 'best_effort':
            continue
        side = sides[sname]
        cap = math.ceil(P._BE_UNITS_CAP * (side.M + side.F))
        assert info['units'] <= cap
        diver = P._Diver(side, False, frozenset())
        best = sum(side.Qm)
        for use_rules, rest in ((True, False), (False, False), (True, True)):
            for order in P._BE_ORDERS:
                pieces, gaps = diver.dive(order, use_rules, rest)
                if P._count_units(pieces) <= cap:
                    best = min(best, sum(gaps))
        assert sum(info['gaps'].values()) <= best + tol


# %% 13. Mini-fuzz with an independent checker

def test_fuzz_small_problems(monkeypatch):
    monkeypatch.setattr(P, '_BE_WORK', 3000.)   # keep best effort short
    rng = random.Random(13)
    n_mer = 0
    for _ in range(200):
        rows, dT = random_problem(rng)
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT)
        Qh, Qc = check_network(streams, dT, net)
        Qh_t, Qc_t, _ = cascade(streams, dT)
        tol = 1e-9 * duty_scale(streams)
        assert Qh >= Qh_t - tol and Qc >= Qc_t - tol
        assert net['plan'].info['dropped'] == []
        if needs_split(streams, dT):
            assert net['status'] == 'best_effort'
        else:
            assert net['status'] == 'mer'
            assert Qh == pytest.approx(Qh_t, abs=tol)
            n_mer += 1
    assert n_mer >= 100


# %% 14. The planner's cascade

def knot_cascade(knots, is_hot, dT):
    """Problem table of piecewise-linear curves by direct evaluation:
    (Qh, Qc) from the heat flow arriving at and leaving every level."""
    curves = []
    for (T, H), hot in zip(knots, is_hot):
        curves.append(([t - (dT if hot else 0.) for t in T], list(H), hot))

    def H_le(T, H, L, strict):
        # sup of the enthalpies at which T*(H) <= L (< L if strict)
        below = (lambda t: t < L) if strict else (lambda t: t <= L)
        if not below(T[0]):
            return H[0]
        for k in range(len(T) - 1):
            if not below(T[k + 1]):   # T[k] below L, T[k + 1] not: cross
                return H[k] + (H[k + 1] - H[k]) * (L - T[k]) / (
                    T[k + 1] - T[k])
        return H[-1]
    levels = sorted({t for T, _, _ in curves for t in T}, reverse=True)
    flows = []
    for L in levels:
        for strict in (False, True):   # arriving at L, then leaving it
            flows.append(sum((1. if hot else -1.) * (H[-1] - H_le(T, H, L,
                                                                 strict))
                             for T, H, hot in curves))
    Qh = max(0., -min(flows))
    return Qh, flows[-1] + Qh


def test_planner_cascade_equals_problem_table():
    rng = random.Random(14)
    problems = [random_problem(rng, 2, 12) for _ in range(60)]
    problems += [(rows, dT) for dT, rows in (list(HARD.values())
                                             + list(SPLIT.values())
                                             + list(REPEATED_PAIR.values()))]
    for rows, dT in problems:
        streams = streams_from(rows)
        kn = [([min(s['T_in'], s['T_out']), max(s['T_in'], s['T_out'])],
               [0., s['CP'] * abs(s['T_in'] - s['T_out'])]) for s in streams]
        hot = [s['kind'] == 'hot' for s in streams]
        plan = P.plan_network(kn, hot, dT, work_scale=0.05)
        Qh, Qc, zeros = cascade(streams, dT)
        scale = duty_scale(streams)
        assert plan.Q_hot_target == pytest.approx(Qh, abs=1e-9 * scale)
        assert plan.Q_cold_target == pytest.approx(Qc, abs=1e-9 * scale)
        assert min(abs(plan.pinch_T - z) for z in zeros) <= 1e-9
    # piecewise-linear curves with flats and point loads
    for _ in range(60):
        kn, hot = [], []
        for _ in range(rng.randint(2, 6)):
            T = [rng.choice(range(20, 300, 5))]
            H = [0.]
            for _ in range(rng.randint(1, 3)):
                flat = rng.random() < 0.3
                T.append(T[-1] + (0. if flat else rng.choice(range(5, 60, 5))))
                H.append(H[-1] + rng.choice(range(5, 100, 5)))
            kn.append((T, H))
            hot.append(rng.random() < 0.5)
        if all(hot) or not any(hot):
            continue
        dT = float(rng.choice([5, 10]))
        plan = P.plan_network(kn, hot, dT, work_scale=0.05)
        Qh, Qc = knot_cascade(kn, hot, dT)
        scale = sum(H[-1] for _, H in kn)
        assert plan.Q_hot_target == pytest.approx(Qh, abs=1e-9 * scale)
        assert plan.Q_cold_target == pytest.approx(Qc, abs=1e-9 * scale)
        # and the plan itself is feasible on the curves
        assert plan.info['dropped'] == []
        assert plan.Q_hot >= Qh - 1e-9 * scale


# %% avoid_recycle and Qmin

def test_avoid_recycle_never_repeats_a_pair():
    rng = random.Random(25)
    cases = [(R002, 10.), (LINNHOFF4, 10.)] + [random_problem(rng)
                                              for _ in range(40)]
    for rows, dT in cases:
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT, avoid_recycle=True)
        Qh, Qc = check_network(streams, dT, net)
        pairs = [(m['hot'], m['cold']) for m in net['matches']]
        assert len(pairs) == len(set(pairs))   # not even across the pinch
        Qh_t = cascade(streams, dT)[0]
        assert Qh >= Qh_t - 1e-9 * duty_scale(streams)
    # r002 needs its repeated pair: best effort, still >= MER
    net = P._plan_numeric(streams_from(R002), 10., avoid_recycle=True)
    assert net['status'] == 'best_effort' and net['penalty'] > 0.


@pytest.mark.parametrize('Qmin', [25., 100., 1e9])
def test_qmin_drops_small_exchangers(Qmin):
    for rows in (R002, LINNHOFF4):
        streams = streams_from(rows)
        net = P._plan_numeric(streams, 10., Qmin=Qmin)
        check_network(streams, 10., net)
        assert all(m['Q'] >= Qmin for m in net['matches'])
        dropped = net['plan'].info['qmin_dropped']
        assert all(q < Qmin for *_, q in dropped)
        full = P._plan_numeric(streams, 10.)
        kept = sum(m['Q'] for m in net['matches'])
        assert kept + sum(q for *_, q in dropped) == pytest.approx(
            sum(m['Q'] for m in full['matches']))
        if dropped:
            assert net['status'] == 'best_effort'


# %% 15. Stream-splitting primitives (hensmith._splitting)

MERGED_BREAKPOINT = [  # flex B starts 4e-7 K above the pinch: 1.2e-6 of A's
    # heat (< tolQ = 2e-6) but far more than tolP = 2.9e-9 K of level
    ('H1', 'h', 300, 50, 1.), ('H2', 'h', 300, 50, 1.),
    ('A', 'c', 100, 250, 3.), ('B', 'c', 100 + 4e-7, 200, 1.),
    ('H3', 'h', 70, 60, 1e4), ('C3', 'c', 0, 10, 1e4)]


def sides_from(rows, dT):
    """The planner's sides of a constant-CP problem, built as in
    `plan_network`."""
    streams = streams_from(rows)
    return sides_from_knots(
        [([min(s['T_in'], s['T_out']), max(s['T_in'], s['T_out'])],
          [0., s['CP'] * abs(s['T_in'] - s['T_out'])]) for s in streams],
        [s['kind'] == 'hot' for s in streams], dT)


def sides_from_knots(knots, is_hot, dT):
    """The planner's sides of a problem given by stream knots, built as in
    `plan_network`."""
    curves = P._stream_curves(knots, is_hot, dT)
    act = [c for c in curves if c is not None]
    scale = sum(c.duty for c in act)
    span = max(c.T[-1] for c in act) - min(c.T[0] for c in act)
    return P._sides(curves, P._cascade(curves, scale),
                    P._REL_Q * max(scale, 1.), P._REL_T * max(span, 1.))


def split_sides():
    """The sides (with musts and flexes) of the SPLIT cases and of the
    merged-breakpoint case."""
    out = []
    for dT, rows in [*SPLIT.values(), (10., MERGED_BREAKPOINT)]:
        out += [s for s in sides_from(rows, dT).values() if s.M and s.F]
    return out


def random_side(rng, tol=1e-9):
    M, F = rng.randint(1, 3), rng.randint(1, 3)
    return P._Side('above', [random_level_curve(rng) for _ in range(M)],
                   [random_level_curve(rng) for _ in range(F)], tol, tol)


def random_fractions(rng):
    """One to three branch fractions (each > 0.05) summing to 1."""
    w = [rng.uniform(1., 5.) for _ in range(rng.randint(1, 3))]
    f = [x / sum(w) for x in w[:-1]]
    return f + [1. - sum(f)]


def is_forest(cells):
    """True if the bipartite graph of the (must, flex) cells has no
    cycle."""
    parent = {}

    def find(u):
        while parent.get(u, u) != u:
            u = parent[u]
        return u
    for i, j in cells:
        ru, rv = find(('m', i)), find(('f', j))
        if ru == rv:
            return False
        parent[ru] = rv
    return True


def test_branch_curve_is_the_scaled_parent():
    rng = random.Random(51)
    curves = [random_level_curve(rng) for _ in range(30)]
    curves += [random_level_curve(rng, grid=True) for _ in range(10)]
    for side in split_sides():
        curves += side.musts + side.flexes
    for c in curves:
        for f in (1., .5, 1e-3, rng.uniform(1e-3, 1.)):
            for role in ('must', 'flex'):
                bc = SP._branch_curve(c, f, role)
                assert bc.y == c.y and bc.flats == c.flats    # levels
                assert bc.q == [f * q for q in c.q] and bc.Q == f * c.Q
                assert (bc.stream, bc.H0, bc.sgn) == (c.stream, c.H0, c.sgn)
                for k in range(c.n - 1):                      # slopes / f
                    s = (c.y[k + 1] - c.y[k]) / (c.q[k + 1] - c.q[k])
                    assert bc.slope_right(bc.q[k]) == pytest.approx(
                        s / f, rel=1e-12)
                # at branch heat tau the branch is the parent at tau / f
                xs = np.linspace(0., bc.Q, 41)
                assert_allclose(bc.at_many(xs), c.at_many(xs / f), rtol=0.,
                                atol=1e-12 * (1. + abs(c.y[-1])))


def test_splitting_preserves_the_residual_cascade():
    # Lemma R: branches with fractions summing to 1 leave every composite,
    # and so the slack of (R), unchanged
    rng = random.Random(52)
    for side in split_sides() + [random_side(rng) for _ in range(40)]:
        scale = max(side.duty, 1.)
        for trial in range(4):
            if trial:
                a = [rng.uniform(0., .6 * q) for q in side.Qm]
                b = [rng.uniform(0., .6 * q) for q in side.Qf]
            else:
                a, b = [0.] * side.M, [0.] * side.F
            musts, am, flexes, bf = [], [], [], []
            for c, x in zip(side.musts, a):
                for f in random_fractions(rng):
                    musts.append(SP._branch_curve(c, f, 'must'))
                    am.append(f * x)
            for c, x in zip(side.flexes, b):
                for g in random_fractions(rng):
                    flexes.append(SP._branch_curve(c, g, 'flex'))
                    bf.append(g * x)
            bside = P._Side(side.name, musts, flexes, side.tolQ, side.tolP)
            d, bd = side.analyse(a, b), bside.analyse(am, bf)
            assert bd.slack == pytest.approx(d.slack, abs=1e-12 * scale)
            if not trial:   # the root: the same levels, the same composites
                np.testing.assert_array_equal(bd.levels, d.levels)
                for k in ('Si', 'Se', 'Di', 'De'):
                    assert_allclose(getattr(bd, k), getattr(d, k), rtol=0.,
                                    atol=1e-12 * scale)


def test_cell_margin_matches_scaled_max_duty():
    # Lemma 1: the margin at the knots of both branch curves decides (C)
    # exactly as _max_duty does on the branch-scaled curves
    rng = random.Random(53)
    tol = 1e-9
    count = [0, 0]
    for _ in range(500):
        cm, cf = random_level_curve(rng), random_level_curve(rng)
        side = P._Side('above', [cm], [cf], tol, tol)
        f = rng.choice((1., rng.uniform(.05, 1.)))
        g = rng.choice((1., rng.uniform(.05, 1.)))
        a = rng.choice((0., rng.choice(cm.q[:-1]), rng.uniform(0., cm.Q)))
        b = rng.choice((0., rng.choice(cf.q[:-1]), rng.uniform(0., cf.Q)))
        xmax = min(f * (cm.Q - a), g * (cf.Q - b))
        x = rng.choice((xmax, rng.uniform(0., xmax)))
        margin, touch = SP._cell_margin(side, SP._Cell(0, 0, x, a, b, f, g))
        bm, bf = SP._branch_curve(cm, f, 'must'), SP._branch_curve(cf, g,
                                                                   'flex')
        xd = P._max_duty(bm, f * a, bf, g * b, x, tol)
        assert (margin >= -tol) == (xd >= x - tol)
        ts = np.concatenate(([0., x], bm.qa - f * a, bf.qa - g * b))
        ts = ts[(ts >= 0.) & (ts <= x)]
        direct = (bm.at_many(f * a + ts) - bf.at_many(g * b + ts)).min()
        assert margin == pytest.approx(direct, abs=1e-9)
        count[margin >= -tol] += 1
    assert min(count) >= 100


def test_cell_margin_touch():
    cm = P._LevelCurve([0., 20.], [0., 10.])
    side = P._Side('above', [cm], [P._LevelCurve([0., 20.], [0., 10.])],
                   1e-9, 1e-9)
    # equal branch fractions: parallel at zero approach along the cell
    assert SP._cell_margin(side, SP._Cell(0, 0, 10., 0., 0., .5, .5)) == (
        0., True)
    # the flex trunk rises half as fast: zero approach at the pinch only
    assert SP._cell_margin(side, SP._Cell(0, 0, 10., 0., 0., .5, 1.)) == (
        0., False)
    # the flex branch overtakes the must trunk
    assert SP._cell_margin(side, SP._Cell(0, 0, 10., 0., 0., 1., .5))[0] == (
        pytest.approx(-5.))
    # beyond the end of the flex: infeasible whatever the levels
    assert SP._cell_margin(side, SP._Cell(0, 0, 11., 0., 0., 1., .5)) == (
        -math.inf, False)
    # a must flat on a flex flat at the same level
    side = P._Side('above', [P._LevelCurve([0., 10., 20.], [4., 4., 8.])],
                   [P._LevelCurve([0., 10., 20.], [0., 4., 4.])], 1e-9, 1e-9)
    assert SP._cell_margin(side, SP._Cell(0, 0, 10., 0., 10.)) == (0., True)
    assert SP._cell_margin(side, SP._Cell(0, 0, 10., 0., 0.)) == (0., False)


def test_vertical_coupling_positions():
    rng = random.Random(54)
    nodes = [(s, [0.] * s.M, [0.] * s.F) for s in split_sides()]
    for _ in range(400):
        side = random_side(rng)
        a = [rng.choice((0., rng.uniform(0., .6 * q))) for q in side.Qm]
        b = [rng.choice((0., rng.uniform(0., .6 * q))) for q in side.Qf]
        if residual_slack(side, a, b) >= 0.:
            nodes.append((side, a, b))
    assert len(nodes) >= 60
    for side, a, b in nodes:
        d = side.analyse(a, b)
        cp = SP._Coupling(side, a, b, d)
        X = math.fsum(q - x for q, x in zip(side.Qm, a))
        scale = max(side.duty, 1.)
        t = cp.t
        assert cp.X == X and cp.ulp == SP._SPLIT_ULP * max(X, 1.)
        assert t[0] == 0. and t[-1] == X and cp.K == t.size - 1 >= 1
        assert (np.diff(t) > cp.ulp).all()
        # no breakpoint is merged beyond round-off (and none folded here)
        raw = np.clip(np.concatenate((d.De, d.Di, d.Se, d.Si)), 0., X)
        assert cp.folded == 0
        assert np.abs(raw[:, None] - t[None, :]).min(1).max() <= cp.ulp
        # positions: monotone, from the node, exact at X, heat-consistent
        Pm, Pf = cp.Pm, cp.Pf
        assert Pm.shape == (t.size, side.M) and Pf.shape == (t.size, side.F)
        assert (np.diff(Pm, axis=0) >= 0.).all()
        assert (np.diff(Pf, axis=0) >= 0.).all()
        assert Pm[0].tolist() == a and Pf[0].tolist() == b
        assert Pm[-1].tolist() == side.Qm
        assert (Pf <= np.array(side.Qf)).all()
        assert_allclose((Pm - a).sum(1), t, rtol=0., atol=1e-12 * scale)
        assert_allclose((Pf - b).sum(1), t, rtol=0., atol=1e-12 * scale)
        # must_at and flex_at: the same closed form anywhere in [0, X]
        for k in range(t.size):
            assert_allclose(cp.must_at(t[k]), Pm[k], rtol=0., atol=cp.ulp)
            assert_allclose(cp.flex_at(t[k]), Pf[k], rtol=0., atol=cp.ulp)
        assert cp.must_at(X) == side.Qm and cp.must_at(0.) == a
        for tm in [rng.uniform(0., X) for _ in range(5)]:
            assert math.fsum(cp.must_at(tm)) - math.fsum(a) == pytest.approx(
                tm, abs=1e-12 * scale)
    # merged breakpoint: B's supply is 1.2e-6 of A's heat (< tolQ) above
    # the pinch; that breakpoint must stay (tolQ merging hides B's knot)
    side = sides_from(MERGED_BREAKPOINT, 10.)['above']
    streams = [c.stream for c in side.flexes]
    jA, jB = streams.index(2), streams.index(3)
    cp = SP._Coupling(side, [0.] * side.M, [0.] * side.F)
    tB = 3. * ((100 + 4e-7) - 100)
    assert cp.ulp < tB < side.tolQ
    k = int(np.abs(cp.t - tB).argmin())
    assert cp.t[k] == pytest.approx(tB, rel=1e-6)
    assert cp.Pf[k, jB] == 0. and cp.Pf[k, jA] == pytest.approx(tB, rel=1e-6)


def test_transport_forest():
    h, g = {0: 1., 1: 1.}, {0: 1., 1: 1.}
    full = {(i, j) for i in h for j in g}
    # north-west, with continuations first; forbidden pairs never used
    assert SP._transport(h, g, full) == {(0, 0): 1., (1, 1): 1.}
    assert SP._transport(h, g, full, [(0, 1)]) == {(0, 1): 1., (1, 0): 1.}
    assert SP._transport(h, g, full, [(0, 0)], {(0, 0)}) == {
        (0, 1): 1., (1, 0): 1.}
    # the greedy is stuck, Edmonds-Karp is not
    assert SP._transport(h, g, {(0, 0), (0, 1), (1, 0)}) == {
        (0, 1): 1., (1, 0): 1.}
    # no transport exists
    assert SP._transport({0: 2., 1: 1.}, {0: 1., 1: 2.},
                         {(0, 0), (1, 1)}) is None
    assert SP._transport(h, g, {(0, 0), (1, 0)}) is None
    # a row of round-off heat on an exhausted column still gets its cell
    assert SP._transport({0: 1., 1: 1e-12}, {0: 1.}, {(0, 0), (1, 0)},
                         tol=1e-9) == {(0, 0): 1., (1, 0): 1e-12}
    rng = random.Random(55)
    eps = np.finfo(float).eps
    for _ in range(300):
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        comp = {(i, j) for i in range(m) for j in range(n)
                if rng.random() < .6}
        q0 = {c: rng.uniform(.1, 10.) for c in sorted(comp)
              if rng.random() < .7}
        h = {i: math.fsum(x for (k, _), x in q0.items() if k == i)
             for i in range(m)}
        g = {j: (math.fsum(x for (_, k), x in q0.items() if k == j)
                 + rng.choice((0., rng.uniform(0., 5.)))) * (1. - 1e-15)
             for j in range(n)}   # flex surplus, and round-off imbalance
        forbid = {c for c in comp if c not in q0 and rng.random() < .5}
        prefer = rng.sample(sorted(comp), min(len(comp), 2))
        q = SP._transport(h, g, comp, prefer, forbid, tol=1e-9)
        assert q is not None and is_forest(q)
        assert all(x > 0. and c in comp and c not in forbid
                   for c, x in q.items())
        for i, v in h.items():   # exact row sums
            assert abs(sum(x for (k, _), x in q.items() if k == i)
                       - v) <= 4 * eps * v
        for j, v in g.items():
            assert math.fsum(x for (_, k), x in q.items() if k == j) <= (
                v + 1e-9)
        lines = {('m', i) for i, _ in q} | {('f', j) for _, j in q}
        assert len(q) <= max(len(lines) - 1, 0)
    for _ in range(100):   # a live continuation is always kept
        h = {i: rng.uniform(1., 5.) for i in range(rng.randint(1, 4))}
        g = {j: rng.uniform(1., 5.) for j in range(rng.randint(1, 4))}
        g[0] += sum(h.values())
        c = (rng.choice(list(h)), rng.choice(list(g)))
        q = SP._transport(h, g, {(i, j) for i in h for j in g}, [c])
        assert q[c] == pytest.approx(min(h[c[0]], g[c[1]]), rel=4 * eps)


# %% 16. Vertical core (hensmith._splitting)

NEAR_THRESHOLD = (10., [  # _cascade sets Qh = 0: below root slack -23.7 tolQ
    ('H1', 'h', 200, 100, 2.), ('C1', 'c', 100, 190, 1.),
    ('C2', 'c', 100, 190, 1. + 1e-9)])
NEAR_DOUBLE_PINCH = (10., [  # nptel_t5_3, CP2 = 6 - 1.58e-11: -0.05 tolQ
    ('1', 'h', 200, 65, 3.), ('2', 'h', 90, 30, 6. - 1.58e-11),
    ('3', 'c', 30, 142, 3.5), ('4', 'c', 25, 130, 4.)])
FAR_PAIR = [  # a balanced large-CP pair far below every other stream
    ('HF', 'h', 0, -10, 1e4), ('CF', 'c', -50, -40, 1e4)]


def jittered_problem(seed):
    """A random problem whose stream temperatures are offset by 3e-8..1e-6 K
    (more than tolP, less than tolQ/CP: FAR_PAIR sets the scale)."""
    rng = random.Random(seed)
    rows, dT = pinch_problem(rng) if seed % 2 == 0 else random_problem(rng)
    jit = random.Random(10_000 + seed)
    rows = [(n, k, Ti + jit.choice([0., 0., 3e-8, 1e-7, 4e-7, 1e-6]),
             To + jit.choice([0., 0., 3e-8, 1e-7, 4e-7]), cp)
            for n, k, Ti, To, cp in rows]
    return rows + FAR_PAIR, dT


def _verify_side_cells(side, cells, a0=None, leak=None):
    """Independent check of the cells of a split side: every cell by
    `_max_duty` and by direct evaluation on the branch-scaled curves; the
    branches of every stage with one fraction each, summing to 1, contiguous
    from the stage's split end (a must's far end, a flex's start); every
    must served exactly from `a0` (plus its leak) and every flex used as a
    prefix."""
    M, F = side.M, side.F
    a0 = [0.] * M if a0 is None else list(a0)
    leak = [0.] * M if leak is None else list(leak)
    tolQ, tolP = side.tolQ, side.tolP
    scale = max(side.duty, 1.)
    for c in cells:
        assert c.x > 0. and 0. < c.f <= 1. and 0. < c.g <= 1.
        cm, cf = side.musts[c.i], side.flexes[c.j]
        assert c.a + c.x / c.f <= cm.Q + tolQ
        assert c.b + c.x / c.g <= cf.Q + tolQ
        bm = SP._branch_curve(cm, c.f, 'must')
        bf = SP._branch_curve(cf, c.g, 'flex')
        am, bb = c.f * c.a, c.g * c.b
        # (C) by _max_duty against the flex lowered by tolP: its linear
        # case stops at the zero crossing of the approach, which is
        # ill-conditioned when a near-parallel cell closes to 0 (a relative
        # slope difference just above _SLOPE_EQ moves the crossing far more
        # than tolQ per ulp of level)
        low = P._LevelCurve(bf.q, [y - tolP for y in bf.y], role='flex')
        assert P._max_duty(bm, am, low, bb, c.x, 0.) >= c.x - tolQ
        ts = np.concatenate(([0., c.x], bm.qa - am, bf.qa - bb))
        ts = ts[(ts >= 0.) & (ts <= c.x)]
        assert (bm.at_many(am + ts) - bf.at_many(bb + ts)).min() >= -tolP
    for role, n, Q in (('m', M, side.Qm), ('f', F, side.Qf)):
        must = role == 'm'
        for s in range(n):
            items = {}
            for c in cells:
                if (c.i if must else c.j) == s:
                    key = c.km if must else c.kf
                    items.setdefault(('trunk', id(c)) if key is None
                                     else key[:-1], []).append(c)
            spans = []
            for key, cs in items.items():
                pos = [(c.a, c.a_end, c.f) if must else (c.b, c.b_end, c.g)
                       for c in cs]
                duty = math.fsum(c.x for c in cs)
                if key[0] == 'trunk':
                    spans.append((pos[0][0], duty))
                    continue
                # a must splits at its far end and mixes toward the pinch,
                # a flex splits at its start and mixes at its far end (as
                # in `_remix`): the branches share the split end
                split = (max(p[1] for p in pos) if must
                         else min(p[0] for p in pos))
                branches = {}
                for c, p in zip(cs, pos):
                    branches.setdefault((c.km if must else c.kf)[-1],
                                        []).append(p)
                fr = [ps[0][2] for ps in branches.values()]
                assert abs(math.fsum(fr) - 1.) <= 1e-12
                for ps in branches.values():
                    ps.sort()
                    assert all(p[2] == ps[0][2] for p in ps)
                    assert abs((ps[-1][1] if must else ps[0][0])
                               - split) <= tolQ
                    for p, p2 in zip(ps, ps[1:]):
                        assert abs(p2[0] - p[1]) <= tolQ
                spans.append((split - duty if must else split, duty))
            spans.sort()
            end = a0[s] if must else 0.
            for start, duty in spans:   # no overlap, no gap beyond leaks
                assert -tolQ <= start - end <= tolQ + sum(leak)
                end = start + duty
            if must:
                served = a0[s] + math.fsum(d for _, d in spans) + leak[s]
                assert abs(served - Q[s]) <= 1e-12 * scale
            else:
                assert end <= Q[s] + tolQ


def verify_core(side, cand, a0):
    """A core candidate: exact cells, (R) at every node, no leak, no missed
    knot, every must served."""
    assert cand.blocks and cand.blocks[0].start[0] == list(a0)
    for blk in cand.blocks:
        assert residual_slack(side, *blk.end) >= -SP._SPLIT_R_TOL * side.tolQ
    assert cand.blocks[-1].end[0] == side.Qm
    assert cand.leak_by_must == [0.] * side.M
    assert sum(blk.knots for blk in cand.blocks) == 0
    _verify_side_cells(side, cand.cells, a0)


def test_verify_side_cells_anchors_stages_like_remix():
    # a must splits at its far end and mixes toward the pinch (`_remix`):
    # its branches share their far end (10), not their start
    L, B = P._LevelCurve, SP._Cell
    side = P._Side('above', [L([0., 10.], [50., 60.])],
                   [L([0., 20.], [0., 10.])] * 2, 1e-9, 1e-9)
    far = [B(0, 0, 3., 4., 0., .5, 1., ('B', 0, 0, 0)),
           B(0, 1, 5., 0., 0., .5, 1., ('B', 0, 0, 1)),
           B(0, 0, 2., 0., 3.)]
    assert not SP._remix('m', SP._stages(far)['m', 0, ('B', 0, 0)], 1e-9)[2]
    _verify_side_cells(side, far)
    near = [B(0, 0, 3., 0., 0., .5, 1., ('B', 0, 0, 0)),
            B(0, 1, 5., 0., 0., .5, 1., ('B', 0, 0, 1)),
            B(0, 0, 2., 8., 3.)]
    with pytest.raises(AssertionError):
        _verify_side_cells(side, near)


def core_sides(rng):
    """Sides (with musts and flexes) of 200 constant-CP problems, 50 with
    point loads and 50 with temperatures closer than tolQ/CP, and of the
    merged-breakpoint and near-tie cases."""
    problems = []
    for k in range(200):
        rows, dT = pinch_problem(rng) if k % 2 else random_problem(rng)
        problems.append(sides_from(rows, dT))
    for _ in range(50):
        kn, hot, dT = random_curve_problem(rng, flat_p=0.5)
        problems.append(sides_from_knots(kn, hot, dT))
    for seed in range(50):
        problems.append(sides_from(*jittered_problem(seed)))
    problems.append(sides_from(MERGED_BREAKPOINT, 10.))
    for dT, rows in (NEAR_THRESHOLD, NEAR_DOUBLE_PINCH):
        problems.append(sides_from(rows, dT))
    return [s for sides in problems for s in sides.values() if s.M and s.F]


def test_theorem_v_random(monkeypatch):
    # Theorem V': from a node satisfying (R) (the root, pre-leaked when the
    # cascade's tolerances left it slightly negative), the chain of
    # elementary vertical blocks is feasible and every node satisfies (R)
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', False)
    rng = random.Random(61)
    checked = preleaked = skipped = 0
    for side in core_sides(rng):
        delta, a0 = SP._preleak_root(side)
        if delta > SP._preleak_max(side):   # not a tolerance-level deficit
            skipped += 1
            continue
        preleaked += delta > 0.
        assert math.fsum(a0) == pytest.approx(delta, abs=1e-12 * side.duty)
        verify_core(side, SP._drive(side, a0, 'V'), a0)
        checked += 1
    assert checked >= 360 and preleaked >= 2 and skipped == 0


@pytest.mark.parametrize('coarsen', [True, False])
def test_vertical_core_on_split_sides(monkeypatch, coarsen):
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', coarsen)
    for side in split_sides():
        delta, a0 = SP._preleak_root(side)
        assert delta == 0.
        cand = SP._drive(side, a0, 'V')
        verify_core(side, cand, a0)
        assert cand.name == 'V' and cand.meta['leak'] == 0.
        for blk in cand.blocks:   # a coarsened block keeps usable fractions
            assert blk.span == 1 or (coarsen and all(
                min(c.f, c.g) >= SP._SPLIT_MIN_FRACTION for c in blk.cells))


def test_vertical_core_from_the_preleaked_root():
    # _cascade's own tolerances leave these roots slightly negative: V from
    # the root fails its node check; V from P(delta) is exact (Lemma P)
    for dT, rows in (NEAR_THRESHOLD, NEAR_DOUBLE_PINCH):
        sides = [s for s in sides_from(rows, dT).values() if s.M and s.F]
        neg = [s for s in sides if SP._preleak_root(s)[0] > 0.]
        assert neg
        for side in neg:
            delta, a0 = SP._preleak_root(side)
            assert SP._SPLIT_R_TOL * side.tolQ < delta <= SP._preleak_max(side)
            assert delta == -side.analyse([0.] * side.M, [0.] * side.F).slack
            with pytest.raises(SP._SplitInvariantError):
                SP._drive(side, [0.] * side.M, 'V')
            assert residual_slack(side, a0, [0.] * side.F) >= (
                -SP._SPLIT_R_TOL * side.tolQ)
            verify_core(side, SP._drive(side, a0, 'V'), a0)


def test_coarsened_blocks_end_at_vertical_nodes(monkeypatch):
    # Corollary C: a coarsened block ends where the elementary chain it
    # replaces ends, so (R) there is inherited
    rng = random.Random(62)
    sides = split_sides() + [s for s in core_sides(rng)[::7]]
    n_coarse = 0
    for side in sides:
        delta, a0 = SP._preleak_root(side)
        if delta > SP._preleak_max(side):
            continue
        monkeypatch.setattr(SP, '_SPLIT_COARSEN', False)
        fine = SP._drive(side, a0, 'V')
        monkeypatch.setattr(SP, '_SPLIT_COARSEN', True)
        coarse = SP._drive(side, a0, 'V')
        verify_core(side, coarse, a0)
        assert len(coarse.blocks) <= len(fine.blocks)
        nodes = [np.array(blk.end[0] + blk.end[1]) for blk in fine.blocks]
        z_f = [0.] * side.F
        cp = SP._Coupling(side, a0, z_f, SP._exact(side).analyse(a0, z_f))
        for blk in coarse.blocks:
            end = np.array(blk.end[0] + blk.end[1])
            assert min(np.abs(end - n).max() for n in nodes) <= side.tolQ
            t = math.fsum(blk.end[0]) - math.fsum(a0)
            assert_allclose(blk.end[0], cp.must_at(t), rtol=0.,
                            atol=side.tolQ)
            assert_allclose(blk.end[1], cp.flex_at(t), rtol=0.,
                            atol=side.tolQ)
        n_coarse += len(coarse.blocks) < len(fine.blocks)
    assert n_coarse >= 10


def test_core_node_check_is_exact(monkeypatch):
    # flex 0 ends 5e-4 (< tolQ) above flex 1's start and the top level is
    # tight: after the block that stops at flex 1's start, flex 0 keeps a
    # sliver below tolQ, which the search's analysis omits (slack -tolQ/2);
    # the core analyses its nodes exactly
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', False)
    L = P._LevelCurve
    side = P._Side('above', [L([0., 10.], [0., 10.])],
                   [L([0., 5.], [0., 5.]), L([0., 5.], [4.9995, 9.9995])],
                   1e-3, 1e-9)
    cand = SP._drive(side, [0.], 'V')
    verify_core(side, cand, [0.])
    lim = -SP._SPLIT_R_TOL * side.tolQ
    assert min(side.analyse(*blk.end).slack for blk in cand.blocks) < lim
    assert min(SP._exact(side).analyse(*blk.end).slack
               for blk in cand.blocks) >= lim


NEAR_FLAT = (10., [  # jittered: near-flat pieces (glides of 3e-8..1e-6 K)
    ([110.000001, 114.0000004, 210.00000003], [0., 4., 100.]),
    ([100.00000003, 134., 134., 165.000001], [0., 68., 108., 139.]),
    ([110.0000004, 133.00000003, 133.0000001, 140.0000004],
     [0., 46., 51., 58.]),
    ([90.0000004, 110.00000003, 139., 155.000001], [0., 60., 147., 179.]),
    ([-10., 0.], [0., 1e5]), ([-50., -40.], [0., 1e5])],
    [True, True, False, False, True, False])


def test_exact_residual_arrays_on_near_flat_pieces():
    # a piece of 1e-7 K over 10 of heat has a slope of 1e8: the core's
    # residual arrays add every piece's clipped share directly, so the
    # residual never exceeds the stream's heat (a cumulative sum of slopes
    # and offsets cancels catastrophically and drifts with the level)
    L = P._LevelCurve
    c = L([0., 10., 110., 115.],
          [-110.0000001, -110., -60.0000004, -60.0000004])
    d = SP._exact(P._Side('below', [], [c], 0., 1e-9)).analyse([], [0.])
    assert d.Rfi[0].tolist() == [0., 10., 115.]
    assert d.Rfe[0].tolist() == [0., 10., 110.]
    # several pieces (flats, near-flats) per stream, frontiers inside them:
    # the arrays match the residual heat evaluated in exact arithmetic on
    # the curve from its frontier ``(f, at(f))`` (inverting the curve,
    # ``x_le``, is ill-conditioned on a near-flat piece)
    from fractions import Fraction as Fr
    rng = random.Random(5)

    def curve():
        c = random_level_curve(rng)
        y = [v + 1e-7 * k * (rng.random() < 0.3) for k, v in enumerate(c.y)]
        return L(c.q, sorted(y))

    def residual(c, f, level, inclusive):
        pts = [(f, c.at(f))] + [(x, y) for x, y in zip(c.q, c.y) if x > f]
        R, L = Fr(0), Fr(level)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            ln = Fr(x1) - Fr(x0)
            if y1 > y0:
                R += ln * min(max((L - Fr(y0)) / (Fr(y1) - Fr(y0)), 0), 1)
            elif L > y0 or inclusive and L == y0:
                R += ln
        return float(R)
    for _ in range(40):
        side = P._Side('above', [curve() for _ in range(rng.randint(1, 3))],
                       [curve() for _ in range(rng.randint(1, 3))], 0., 1e-9)
        a = [rng.uniform(0., c.Q) * rng.randint(0, 1) for c in side.musts]
        b = [rng.uniform(0., c.Q) * rng.randint(0, 1) for c in side.flexes]
        d = SP._exact(side).analyse(a, b)
        for curves, front, Ri, Re in ((side.musts, a, d.Rmi, d.Rme),
                                      (side.flexes, b, d.Rfi, d.Rfe)):
            for k, (c, f) in enumerate(zip(curves, front)):
                tol = 1e-14 * c.Q
                assert_allclose(Ri[k], [residual(c, f, y, True)
                                        for y in d.levels], rtol=0, atol=tol)
                assert_allclose(Re[k], [residual(c, f, y, False)
                                        for y in d.levels], rtol=0, atol=tol)


def test_core_on_near_flat_pieces_reaches_mer():
    # the core analyses its nodes exactly: on near-flat pieces every
    # strategy reaches MER from the pre-leaked root (the drift of the
    # residual arrays failed every core strategy at 'core node (R)')
    dT, kn, hot = NEAR_FLAT
    side = sides_from_knots(kn, hot, dT)['below']
    delta, a0 = SP._preleak_root(side)
    assert 0. < delta <= SP._preleak_max(side)
    for strategy in SP._CORE_STRATEGIES:
        verify_core(side, SP._drive(side, a0, strategy), a0)
    plan = P.plan_network(kn, hot, dT, stream_splitting=True)
    assert plan.status == 'mer'
    for s in plan.info['sides'].values():
        assert s['split'] is None or s['split']['errors'] == []


NEAR_PARALLEL = (10., [  # jittered: S:demand pairs a branch with a flex
    # whose slope differs by a relative 1e-9 over 25 K of level
    ([140.0000001, 140.0000001, 158., 160.0000004], [0., 40., 58., 62.]),
    ([100.000001, 100.00000003, 205.0000001], [0., 20., 230.]),
    ([130.000001, 130.00000003, 141.00000003, 142., 142.0000001,
      160.00000003, 160.], [0., 40., 51., 52., 62., 80., 90.]),
    ([100.0000001, 118.0000001, 230.], [0., 18., 130.]),
    ([120.0000004, 120.0000004, 170.], [0., 10., 110.]),
    ([90.0000001, 154., 215.00000003, 215.00000003], [0., 128., 250., 270.]),
    ([90.0000004, 90., 175.00000003], [0., 20., 105.]),
    ([-10., 0.], [0., 1e5]), ([-50., -40.], [0., 1e5])],
    [True, True, True, True, False, False, False, True, False])


def test_split_searches_keep_near_parallel_pairs_feasible():
    # the planner's `_max_duty` takes slopes within a relative `_SLOPE_EQ`
    # as parallel, so a converging pair may close by `_SLOPE_EQ` times its
    # level span, beyond tolP; the split path's searches use a max duty
    # whose parallel shortcut holds only where the closure stays within
    # tolP, so every piece they place satisfies (C) (`_cell_margin`)
    L = P._LevelCurve
    tolP = 1e-9
    cm = L([0., 25.], [100., 125.])
    for rate, g0, strict in ((5e-10, 0., 0.), (5e-10, 5e-9, 10.),
                             (5e-10, -5e-10, 0.), (4e-16, 0., 25.),
                             (0., -0.5 * tolP, 25.), (-1e-3, 0., 25.)):
        cf = L([0., 25.], [100. - g0, 125. - g0 + 25. * rate])
        side = P._Side('above', [cm], [cf], 1e-6, tolP)
        x = SP._max_duty_strict(cm, 0., cf, 0., 25., tolP)
        assert x == pytest.approx(strict, rel=1e-4)   # levels round off
        assert SP._cell_margin(side, SP._Cell(0, 0, x, 0., 0.))[0] >= -tolP
        lenient = P._max_duty(cm, 0., cf, 0., 25., tolP)
        assert lenient == 25. and (lenient == x or SP._cell_margin(
            side, SP._Cell(0, 0, lenient, 0., 0.))[0] < -tolP)
    # the split path searches strictly; the unsplit planner does not
    assert P._Side.max_duty is P._max_duty
    assert SP._strict(side).max_duty is SP._max_duty_strict
    # end to end: S:demand's DFS met such a pair and its plan failed (C)
    dT, kn, hot = NEAR_PARALLEL
    plan = P.plan_network(kn, hot, dT, stream_splitting=True)
    assert plan.status == 'mer'
    for s in plan.info['sides'].values():
        assert s['split'] is None or s['split']['errors'] == []
    assert plan.info['sides']['above']['split']['candidates'][
        'S:demand'] != 'error'


def one_to_one_side():
    """One must over one flex, far apart in level: one vertical block."""
    return P._Side('above', [P._LevelCurve([0., 10.], [50., 60.])],
                   [P._LevelCurve([0., 20.], [0., 10.])], 1e-9, 1e-9)


def carrier(cp, k, cells=()):
    """A finished vertical block ending at breakpoint `k` of `cp`."""
    end = (cp.Pm[k].tolist(), cp.Pf[k].tolist())
    return SP._Block('vertical', end, end, list(cells), cp=cp, k=k)


def test_vertical_block_recovery(monkeypatch):
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', False)
    # 1. missed knot: without the breakpoints of flex B's supply level (the
    # musts reach it at t1, flex A at t2), elementary block 0 puts B 4e-7 K
    # above the pinch against the musts at the pinch; the first hidden
    # composite breakpoint is inserted and the block ends there
    side = sides_from(MERGED_BREAKPOINT, 10.)['above']
    z_m, z_f = [0.] * side.M, [0.] * side.F
    cp = SP._Coupling(side, z_m, z_f)
    t1, t2 = cp.t[1], cp.t[2]
    assert t1 == pytest.approx(2. * 4e-7, rel=1e-6)
    assert t2 == pytest.approx(3. * 4e-7, rel=1e-6)
    cp.t, cp.K = np.delete(cp.t, [1, 2]), cp.K - 2
    cp.Pm, cp.Pf = np.delete(cp.Pm, [1, 2], 0), np.delete(cp.Pf, [1, 2], 0)
    blk = SP._vertical_block(side, z_m, z_f, [carrier(cp, 0)])
    assert blk.knots == 1 and cp.t[blk.k] == t1 and blk.leak == {}
    assert blk.cells and all(SP._cell_margin(side, c)[0] >= -side.tolP
                             for c in blk.cells)
    # 2. a cell failing (C) by position round-off is leaked when its heat is
    # below _SPLIT_R_TOL tolQ, attached to the must's series cell in the
    # previous block when that re-verifies, and raises otherwise
    side = one_to_one_side()
    tiny = 0.1 * SP._SPLIT_R_TOL * side.tolQ
    margin = SP._cell_margin
    monkeypatch.setattr(SP, '_cell_margin', lambda s, c: (
        (-1., False) if c.x <= tiny else margin(s, c)))
    cp = SP._Coupling(side, [0.], [0.])
    cp.insert(tiny)
    blk = SP._vertical_block(side, [0.], [0.], [carrier(cp, 0)])
    assert blk.cells == [] and blk.leak == {0: tiny} and blk.k == 1
    cp = SP._Coupling(side, [0.], [0.])
    k = cp.insert(5.)
    cp.insert(5. + tiny)
    series = SP._Cell(0, 0, 5., 0., 0.)
    blk = SP._vertical_block(side, [5.], [5.], [carrier(cp, k, [series])])
    assert blk.cells == [] and blk.leak == {}
    assert series.a_end == cp.Pm[k + 1, 0] and series.b_end == cp.Pf[k + 1, 0]
    monkeypatch.setattr(SP, '_cell_margin', lambda s, c: (-1., False))
    with pytest.raises(SP._SplitInvariantError):
        SP._vertical_block(side, [0.], [0.])


def test_vertical_block_forbid_and_used(monkeypatch):
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', False)
    side = P._Side('above', [P._LevelCurve([0., 10.], [50., 60.])] * 2,
                   [P._LevelCurve([0., 20.], [0., 10.])] * 2, 1e-9, 1e-9)
    z = [0., 0.]
    blk = SP._vertical_block(side, z, z, (), frozenset({(0, 0)}))
    assert (0, 0) not in blk.pairs and blk.pairs
    assert SP._vertical_block(side, z, z, (),
                              frozenset({(0, 0), (0, 1)})) is None
    # avoid_recycle: a used pair returns only as a series continuation
    # (coarse blocks: elementary ones repeat a pair on every split side)
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', True)
    n = 0
    for s in split_sides():
        cand = SP._drive(s, [0.] * s.M, 'V', cap1=True)
        if cand is not None:
            n += 1
            pairs = [(c.i, c.j) for c in SP._merge_cells(cand.cells, s.tolQ)]
            assert len(pairs) == len(set(pairs))
            _verify_side_cells(s, cand.cells)
    assert n >= 1


def test_vertical_block_on_a_round_off_sliver(monkeypatch):
    # must 1 starts o ulp above must 0, so the first interval holds o ulp
    # of must heat, shared by two flexes with at most one ulp each: the
    # flex heat is round-off, yet it still needs a column
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', False)
    L = P._LevelCurve
    u = 20. * SP._SPLIT_ULP
    for o in (1.2, 1.5, 1.9, 2.5, 3.5):
        side = P._Side('above', [L([0., 10.], [0., 10.]),
                                 L([0., 10. - o * u], [o * u, 10.])],
                       [L([0., 10.], [0., 10.])] * 2, 1e-9, 1e-9)
        cp = SP._Coupling(side, [0., 0.], [0., 0.])
        assert cp.Pm[1, 0] > cp.ulp and cp.Pm[1, 1] == 0.
        assert (o >= 2.) == (cp.Pf[1] > cp.ulp).all()
        verify_core(side, SP._drive(side, [0., 0.], 'V'), [0., 0.])
    # a vertical block with no transport is None only when forbidden or
    # used pairs removed the transport; otherwise it is a failed invariant
    side = one_to_one_side()
    monkeypatch.setattr(SP, '_transport', lambda *args, **kw: None)
    with pytest.raises(SP._SplitInvariantError, match='no transport'):
        SP._vertical_block(side, [0.], [0.])


def test_coarsened_block_keeps_the_minimum_fraction(monkeypatch):
    # flex 1 carries 0.005 of heat parallel to flex 0 up to level 10, so
    # every block over that range sends must 0 to flex 1 with a fraction
    # near 5e-4 < _SPLIT_MIN_FRACTION. The block to X verifies cell by cell
    # but is rejected for that fraction: elementary blocks cover the range
    # and one coarsened block the rest
    monkeypatch.setattr(SP, '_SPLIT_COARSEN', True)
    L = P._LevelCurve
    lv = [float(x) for x in range(0, 21, 2)]
    side = P._Side('above', [L([0., 20.005], [1., 21.])],
                   [L(lv, lv), L([0., .005], [0., 10.])], 1e-9, 1e-9)
    cp = SP._Coupling(side, [0.], [0., 0.])
    tiny = lambda blk: min(min(c.f, c.g) for c in blk.cells)
    cand = SP._drive(side, [0.], 'V')
    verify_core(side, cand, [0.])
    *low, top = cand.blocks
    assert len(low) == int(np.searchsorted(cp.t, 10.005 - 1e-9))
    for blk in low:
        assert blk.span == 1 and {c.j for c in blk.cells} == {0, 1}
        assert tiny(blk) < SP._SPLIT_MIN_FRACTION
    assert top.span > 1 and top.end[0] == side.Qm
    assert {c.j for c in top.cells} == {0} and tiny(top) == 1.
    # without the minimum fraction, the block to X is taken
    monkeypatch.setattr(SP, '_SPLIT_MIN_FRACTION', 0.)
    cand = SP._drive(side, [0.], 'V')
    verify_core(side, cand, [0.])
    assert [blk.span for blk in cand.blocks] == [cp.K]
    assert tiny(cand.blocks[0]) < 1e-3


def key_side(curved=False):
    musts = [P._LevelCurve([0., 10.], [50., 60.]) for _ in range(2)]
    flexes = [P._LevelCurve([0., 20.], [0., 10.]) for _ in range(2)]
    if curved:
        flexes.append(P._LevelCurve([0., 5., 20.], [0., 1., 10.]))
    for k, c in enumerate(musts + flexes):
        c.stream = k
    return P._Side('above', musts, flexes, 1e-9, 1e-9)


def test_candidate_key_and_signature():
    side = key_side()
    B = SP._Cell

    def cand(cells, name='V', order=(1, 0), **kw):
        return SP._Candidate(name, order, side, cells, **kw)
    # must 0 in two branches (a V block), must 1 on a trunk to flex 1
    split = [B(0, 0, 4., 0., 0., .4, .5, ('B', 0, 0, 0), ('B', 0, 0, 0)),
             B(0, 1, 6., 0., 0., .6, 1., ('B', 0, 0, 1), None),
             B(1, 0, 4., 0., 0., 1., .5, None, ('B', 0, 0, 1)),
             B(1, 1, 6., 4., 6.)]
    c = cand(split)
    assert (c.units, c.stages, c.branches, c.mixers) == (4, 2, 4, 1)
    assert c.key() == (0, 0, 0, 8, 1, (1, 0))
    assert c.meta == dict(candidate='V', stages=2, branches=4, leak=0.,
                          small=[])
    # the same network under another name and branch numbering
    swap = [B(0, 0, 4., 0., 0., .4, .5, ('B', 7, 0, 1), ('B', 7, 0, 1)),
            B(0, 1, 6., 0., 0., .6, 1., ('B', 7, 0, 0), None),
            B(1, 0, 4., 0., 0., 1., .5, None, ('B', 7, 0, 0)),
            B(1, 1, 6., 4., 6.)]
    assert cand(swap, 'LVT', (1, 3)).signature == c.signature
    other = [B(0, 0, 3., 0., 0., .3, 3. / 7., ('B', 0, 0, 0), ('B', 0, 0, 0)),
             B(0, 1, 7., 0., 0., .7, 1., ('B', 0, 0, 1), None),
             B(1, 0, 4., 0., 0., 1., 4. / 7., None, ('B', 0, 0, 1)),
             B(1, 1, 6., 4., 7.)]
    assert cand(other).signature != c.signature
    # series continuations merge into one unit; Qmin and tiny fractions
    series = [B(1, 1, 3., 0., 0.), B(1, 1, 7., 3., 3.)]
    c = cand(series, Qmin=8.)
    assert c.units == 1 and c.small == 0 and c.meta['small'] == []
    c = cand(series, Qmin=11.)
    assert c.units == 1 and c.small == 1 and c.meta['small'] == [(1, 3, 10.)]
    tiny = [B(0, 0, 1e-4, 0., 0., 1e-5, 1., ('B', 0, 0, 0), None),
            B(0, 1, 10. - 1e-4, 0., 0., 1. - 1e-5, 1., ('B', 0, 0, 1), None)]
    assert cand(tiny).small == 1
    # a non-isothermal flex remix followed by an exchanger on that flex
    remix = [B(0, 0, 2., 0., 0., 1., .25, None, ('B', 0, 0, 0)),
             B(1, 0, 2., 0., 0., 1., .75, None, ('B', 0, 0, 1))]
    assert cand(remix).mixbad == 0   # feeds only the flex's utility
    assert cand(remix + [B(0, 0, 1., 2., 5.)]).mixbad == 1
    iso = [B(0, 0, 2., 0., 0., 1., .5, None, ('B', 0, 0, 0)),
           B(1, 0, 2., 0., 0., 1., .5, None, ('B', 0, 0, 1)),
           B(0, 0, 1., 2., 4.)]
    assert cand(iso).mixbad == 0
    # a must splits at its far end (10) and mixes toward the pinch (2):
    # bad only if an exchanger of that must follows the mix
    remix = [B(0, 0, 3., 4., 0., .5, 1., ('B', 0, 0, 0), None),
             B(0, 1, 5., 0., 0., .5, 1., ('B', 0, 0, 1), None)]
    assert cand(remix).mixbad == 0   # feeds only the must's utility
    assert cand(remix + [B(0, 0, 2., 0., 3.)]).mixbad == 1
    # a curved stream (more than two knots) with more than _SPLIT_MIX_CAP
    # split stages (mixers) on the side counts as a bad remix, even when
    # every remix is isothermal; a straight stream does not
    side = key_side(curved=True)

    def stages(j, n):   # n isothermal stages of flex j: two branches each
        return [B(i, j, 1., k, 2. * k, 1., .5, None, ('B', k, j, i))
                for k in range(n) for i in (0, 1)]
    assert SP._SPLIT_MIX_CAP == 2
    assert side.flexes[2].n > 2 and side.flexes[0].n == 2
    for j, n, bad in ((2, 3, 1), (2, 2, 0), (0, 3, 0)):
        c = cand(stages(j, n))
        assert (c.stages, c.mixers, c.mixbad) == (n, n, bad)
    # touch counts only on a side with a curved stream
    par = [B(0, 0, 5., 0., 0., .5, .5)]   # parallel at the minimum approach
    side = P._Side('above', [P._LevelCurve([0., 20.], [0., 10.])],
                   [P._LevelCurve([0., 20.], [0., 10.])], 1e-9, 1e-9)
    assert SP._Candidate('V', (1, 0), side, par).touch == 0
    side.flexes.append(P._LevelCurve([0., 5., 20.], [-9., -8., -1.]))
    side = P._Side('above', side.musts, side.flexes, 1e-9, 1e-9)
    assert SP._Candidate('V', (1, 0), side, par).touch == 1
    # a cell that fails (C) is never accepted
    with pytest.raises(SP._SplitInvariantError):
        SP._Candidate('V', (1, 0), side, [B(0, 0, 5., 0., 1.)])


# %% 17. Split plans end to end (records and the switch)

MATCH_KEYS = {'side', 'hot', 'cold', 'Q', 'T_hot_in', 'T_hot_out',
              'T_cold_in', 'T_cold_out', 'hot_seq', 'cold_seq', 'pair_index'}
SPLIT_KEYS = {'hot_frac', 'cold_frac', 'hot_branch', 'cold_branch'}
SPLIT_ON = dict(stream_splitting=True)


def split_infos(plan):
    """``{side: info['sides'][side]['split']}`` of the split sides (a side
    whose split attempt found no candidate records one with candidate
    None)."""
    return {name: s['split'] for name, s in plan.info['sides'].items()
            if s.get('split') and s['split']['candidate'] is not None}


def check_split_network(streams, dT, net, tol=1e-6, exact=True, mer=True):
    """
    Walk every stream along ``plan.paths``: trunk exchangers in series and
    the branches of every split in parallel from the split state, each at
    ``Q / (f CP)``, then the mix at ``Q / CP`` of the total. Recompute the
    temperatures and assert: the recorded temperatures, fractions and
    branches; both ends of every exchanger keep dT (constant CP: linear
    profiles); fractions in (0, 1] summing to 1; ``H_split`` is the state
    reached and ``H_mix = H_split -/+ sum Q``; the flattened paths are
    ``plan.stages`` with consecutive seqs; utilities >= 0 close every
    stream and equal the targets up to the pre-leak; every must of a split
    side is served exactly up to its gap. With `exact`, the penalty
    is at most the pre-leak (+ 1e-12 of the scale); without it, only
    'mer'-close: temperatures offset by less than tolQ/CP let the unsplit
    planner's own tolerances reach the utilities (`_sides` omits a stream
    part of at most tolQ at the pinch; `_cascade` thresholds the target of
    a side with no must). With ``mer=False`` (a best-effort network, e.g.
    avoid_recycle), the utilities are only at least the targets. Returns
    the hot and cold utility.
    """
    plan = net['plan']
    matches = net['matches']
    scale = duty_scale(streams)
    qtol = 1e-9 * scale
    preleak = math.fsum(s['preleak'] for s in split_infos(plan).values())
    index = {id(s): k for k, s in enumerate(plan.splits)}
    assert len(index) == len(plan.splits)
    temps = [{} for _ in matches]
    assert all(u >= 0. for u in plan.utility)
    for j, s in enumerate(streams):
        hot = s['kind'] == 'hot'
        role = 'hot' if hot else 'cold'
        sg, CP = (-1. if hot else 1.), s['CP']
        lo = min(s['T_in'], s['T_out'])
        flat = []

        def run(ns, T, f, branch):
            for n in ns:
                m = matches[n]
                assert m[role] == s['name'] and m['Q'] > 0.
                if plan.splits:
                    assert m[role + '_frac'] == f
                    assert m[role + '_branch'] == branch
                T2 = T + sg * m['Q'] / (f * CP)
                assert_allclose((m[f'T_{role}_in'], m[f'T_{role}_out']),
                                (T, T2), rtol=1e-9, atol=tol)
                temps[n][hot] = (T, T2)
                flat.append(n)
                T = T2
            return T
        T = s['T_in']
        for item in plan.paths[j]:
            if not isinstance(item, SP.Split):
                T = run([item], T, 1., None)
                continue
            k = index[id(item)]
            fr = item.fractions
            assert item.stream == j and len(fr) == len(item.branches) >= 2
            assert all(0. < f <= 1. for f in fr)
            assert abs(math.fsum(fr) - 1.) <= 1e-12
            assert item.H_split == pytest.approx(CP * (T - lo), abs=qtol)
            D, ends = 0., []
            for b, (f, ns) in enumerate(zip(fr, item.branches)):
                assert ns
                ends.append(run(ns, T, f, (k, b)))
                D += math.fsum(matches[n]['Q'] for n in ns)
            assert item.H_mix == pytest.approx(item.H_split + sg * D,
                                               abs=1e-12 * scale)
            # isothermal: every branch ends (parent-equivalent) within
            # _ISO_TOL tolQ of the mix; a non-isothermal remix is the
            # stream's last item, only its utility follows
            assert item.isothermal == all(
                abs(CP * (e - lo) - item.H_mix)
                <= SP._ISO_TOL * plan.info['tolQ'] for e in ends)
            assert item.isothermal or item is plan.paths[j][-1]
            T += sg * D / CP
        assert flat == plan.stages[j]
        assert ([matches[n][role + '_seq'] for n in flat]
                == list(range(1, len(flat) + 1)))
        need = CP * ((T - s['T_out']) if hot else (s['T_out'] - T))
        u = (net['cold_utility'] if hot else net['hot_utility']).get(
            s['name'], 0.)
        assert need >= -qtol and abs(u - need) <= qtol
    assert all(len(t) == 2 for t in temps)   # every exchanger walked twice
    for t in temps:
        (hi, ho), (ci, co) = t[True], t[False]
        assert min(hi - co, ho - ci) >= dT - tol
    Qh = sum(net['hot_utility'].values())
    Qc = sum(net['cold_utility'].values())
    Qh_t, Qc_t, _ = cascade(streams, dT)
    if mer:
        assert abs(Qh - Qh_t) <= qtol + preleak
        assert abs(Qc - Qc_t) <= qtol + preleak
    else:   # never below the targets
        assert Qh >= Qh_t - qtol and Qc >= Qc_t - qtol
    # the split machinery adds no heat error: every must is served exactly
    sides = sides_from_knots(
        [([min(s['T_in'], s['T_out']), max(s['T_in'], s['T_out'])],
          [0., s['CP'] * abs(s['T_in'] - s['T_out'])]) for s in streams],
        [s['kind'] == 'hot' for s in streams], dT)
    exact_tol = 1e-12 * plan.info['scale']
    for name in split_infos(plan):
        role = 'hot' if name == 'above' else 'cold'
        gaps = plan.info['sides'][name]['gaps']
        for c in sides[name].musts:
            served = math.fsum(m['Q'] for m in matches if m['side'] == name
                               and m[role] == streams[c.stream]['name'])
            assert served + gaps.get(c.stream, 0.) == pytest.approx(
                c.Q, abs=exact_tol)
    if exact and mer:
        assert plan.penalty <= preleak + exact_tol
    return Qh, Qc


S_NAMES = tuple('S:' + rule for rule in SP._SPLIT_RULES)
SPLIT_SIZE = {  # units + extra branches + split stages of the split side,
    # at most (as recorded when Stage S was added)
    'smith2005_exr18_4': 5, 'smith2005_ex16_5_five_stream': 5,
    'rnd_uniform_n2-6_s327': 4}


def split_size(side_info):
    """Key element 4 of a split side: units + extra branches + split
    stages (every splitter and mixer pair counts like a unit)."""
    # units + extra branches (branches - stages) + split stages
    return side_info['units'] + side_info['split']['branches']


def repeated_pairs(net):
    """(hot, cold) pairs of more than one exchanger, across the pinch."""
    pairs = Counter((m['hot'], m['cold']) for m in net['matches'])
    return [p for p, n in pairs.items() if n > 1]


@pytest.mark.parametrize('name', sorted(SPLIT))
def test_split_cases_reach_mer(name):
    dT, rows = SPLIT[name]
    streams = streams_from(rows)
    net = P._plan_numeric(streams, dT, **SPLIT_ON)
    plan = net['plan']
    assert net['status'] == 'mer' and plan.splits
    assert plan.info['dropped'] == [] and plan.info['qmin_dropped'] == []
    check_split_network(streams, dT, net)
    assert set(net['matches'][0]) == MATCH_KEYS | SPLIT_KEYS
    for sname, s in plan.info['sides'].items():
        sp = s['split']
        if sp is None:   # a side the unsplit search serves
            assert s['status'] in ('mer', 'trivial')
            continue
        assert s['status'] == 'mer' and s['method'] == 'split-' + sp[
            'candidate']
        assert s['proof']['rule'] in ('outward', 'inward')
        assert sp['preleak'] == sp['leak'] == 0.
        assert sp['errors'] == [] and sp['small'] == []
        # the whole portfolio ran: Stage S (a rule reaches MER) and the
        # core backstop, and the smallest key won
        cands = sp['candidates']
        assert set(cands) == set(S_NAMES) | set(SP._CORE_STRATEGIES)
        keys = {n: k for n, k in cands.items() if isinstance(k, tuple)}
        assert 'V' in keys and any(n in keys for n in S_NAMES)
        assert sp['candidate'] == min(keys, key=keys.get)
        assert keys[sp['candidate']][1] == 0   # mixbad: no bad remix
        assert split_size(s) <= SPLIT_SIZE[name]
        assert s['units'] == len([e for e in plan.exchangers
                                  if e.side == sname])


def test_split_plan_heat_closes_exactly():
    # a leak event (_SPLIT_R_TOL tolQ, 1e-14 of the scale) is 100 times
    # inside the planner's 1e-12 heat closure
    assert SP._SPLIT_R_TOL * P._REL_Q == pytest.approx(1e-14)
    n = 0
    for dT, rows in [*SPLIT.values(), NEAR_THRESHOLD, NEAR_DOUBLE_PINCH]:
        streams = streams_from(rows)
        plan = P._plan_numeric(streams, dT, **SPLIT_ON)['plan']
        scale = plan.info['scale']
        infos = split_infos(plan)
        assert infos
        preleak = math.fsum(sp['preleak'] for sp in infos.values())
        assert all(sp['leak'] == 0. for sp in infos.values())
        assert plan.penalty <= preleak + 1e-12 * scale
        for name, sp in infos.items():   # truthful gaps
            gaps = plan.info['sides'][name]['gaps']
            assert math.fsum(gaps.values()) == pytest.approx(
                sp['preleak'] + sp['leak'], abs=1e-12 * scale)
            n += 1
    assert n >= 5


def test_splitting_off_adds_nothing():
    # every fixture but the slowest (seconds of best effort; a bitwise dump
    # of plan_network with the flag off, over the whole corpus, proves the
    # identity outside the test suite)
    slow = ('hold_nested_n21-40_s100233',)
    cases = [(10., R002), (10., LINNHOFF4), *REPEATED_PAIR.values(),
             *(v for k, v in HARD.items() if k not in slow),
             *SPLIT.values(), NEAR_THRESHOLD, (10., MERGED_BREAKPOINT)]
    info_keys = {'sides', 'qmin_dropped', 'dropped', 'min_approach', 'work',
                 'scale', 'tolQ', 'cascade'}
    side_keys = {'status', 'method', 'work', 'proof', 'units', 'M', 'F',
                 'gaps'}
    for dT, rows in cases:
        net = P._plan_numeric(streams_from(rows), dT, stream_splitting=False)
        plan = net['plan']
        assert set(plan.info) == info_keys
        assert all(set(s) == side_keys for s in plan.info['sides'].values())
        assert plan.splits == []
        assert plan.paths == {j: list(s) for j, s in plan.stages.items()}
        assert all(set(m) == MATCH_KEYS for m in net['matches'])
        assert all((e.hot_frac, e.cold_frac, e.hot_branch, e.cold_branch)
                   == (1., 1., None, None) for e in plan.exchangers)


def test_threshold_and_near_tie_roots_preleak():
    # _cascade's own tolerances leave these roots slightly negative (a
    # threshold deficit of 23.7 tolQ = 9e-8; near-equal minima, 0.05 tolQ):
    # the core starts from the pre-leaked root and the deficit is booked
    for (dT, rows), rel in ((NEAR_THRESHOLD, 23.7), (NEAR_DOUBLE_PINCH, .05)):
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT, **SPLIT_ON)
        plan = net['plan']
        assert net['status'] == 'mer'
        check_split_network(streams, dT, net)
        sides = sides_from(rows, dT)
        infos = split_infos(plan)
        leaked = {n: sp for n, sp in infos.items() if sp['preleak'] > 0.}
        assert len(leaked) == 1
        for name, sp in leaked.items():
            side = sides[name]
            slack = side.analyse([0.] * side.M, [0.] * side.F).slack
            assert sp['preleak'] == -slack
            assert sp['preleak'] == pytest.approx(rel * side.tolQ, rel=.01)
            assert sp['preleak'] <= SP._preleak_max(side)
    # a 'cascade' root with a deficit beyond the cascade's tolerances: the
    # targets cannot be met, so today's best effort (no split) applies
    L = P._LevelCurve
    side = P._Side('above', [L([0., 10.], [50., 60.])],
                   [L([0., 5.], [0., 10.])], 1e-9, 1e-9)
    z = ([0.], [0.])
    d = side.analyse(*z)
    assert side.rules_violation(*z, d) is None
    assert -d.slack > SP._preleak_max(side)
    off = P._plan_side(side)
    on = P._plan_side(side, split=dict(Qmin=0., exclude={}, prefer={}))
    assert off.proof['rule'] == on.proof['rule'] == 'cascade'
    assert on.status == 'best_effort' and on.cells == [] and on.split is None
    assert (on.pieces, on.gaps, on.method) == (off.pieces, off.gaps,
                                               off.method)


def records_case():
    """A hand-built split side: H1 (200 kW, far above both colds, which
    need 100 kW of heating: all above the pinch) splits at its far end into
    two halves that re-join non-isothermally at 100, then serves C2 and C1
    on its trunk."""
    rows = [('H1', 'h', 400, 200, 1.), ('C1', 'c', 20, 170, 1.),
            ('C2', 'c', 20, 170, 1.)]
    streams = streams_from(rows)
    knots = [([min(s['T_in'], s['T_out']), max(s['T_in'], s['T_out'])],
              [0., s['CP'] * abs(s['T_in'] - s['T_out'])]) for s in streams]
    hot = [s['kind'] == 'hot' for s in streams]
    curves = P._stream_curves(knots, hot, 10.)
    sides = sides_from_knots(knots, hot, 10.)
    side = sides['above']
    assert (side.M, side.F, sides['below'].M + sides['below'].F) == (1, 2, 0)
    B = SP._Cell
    cells = [B(0, 0, 60., 80., 0., .5, 1., ('S', 0, 0)),    # [80, 200]
             B(0, 1, 40., 120., 0., .5, 1., ('S', 0, 1)),   # [120, 200]
             B(0, 1, 60., 40., 40.), B(0, 0, 40., 0., 60.)]
    _verify_side_cells(side, cells)
    plans = {'above': P._SidePlan([], [0.], 'mer', 'split-X', cells=cells,
                                  units=4)}
    return sides, plans, curves


def test_split_records_walk_a_non_isothermal_remix(monkeypatch):
    sides, plans, curves = records_case()
    tolQ = sides['above'].tolQ
    (recs, qmin_dropped, dropped, stages, utility, min_dT, splits,
     paths) = SP._split_records(sides, plans, curves, 3, 0., tolQ)
    assert qmin_dropped == dropped == [] and min_dT >= 0.
    b0, b1, t1, t0 = range(4)   # records in cell order
    s, = splits
    # the must flows from its far end: the split, then its trunk
    assert paths == {0: [s, t1, t0], 1: [b0, t0], 2: [b1, t1]}
    assert stages == {0: [b0, b1, t1, t0], 1: [b0, t0], 2: [b1, t1]}
    assert (s.stream, s.side, s.key, s.fractions) == (0, 'above', ('S', 0),
                                                      (.5, .5))
    assert s.branches == [[b0], [b1]] and not s.isothermal
    assert (s.H_split, s.H_mix) == (200., 100.)
    H = [(e.H_hot_in, e.H_hot_out, e.hot_seq, e.hot_branch) for e in recs]
    assert H == [(200., 80., 1, (0, 0)), (200., 120., 2, (0, 1)),
                 (100., 40., 3, None), (40., 0., 4, None)]
    C = [(e.H_cold_in, e.H_cold_out, e.cold_seq, e.cold_frac) for e in recs]
    assert C == [(0., 60., 1, 1.), (0., 40., 1, 1.), (40., 100., 2, 1.),
                 (60., 100., 2, 1.)]
    assert utility == [0., 50., 50.]
    # the safety net (a bug if it fires) keeps a dropped branch, empty
    real = SP._approach_violation_split
    monkeypatch.setattr(SP, '_approach_violation_split', lambda ch, cc, e: (
        -1. if e.Q == 40. and e.hot_frac == .5 else real(ch, cc, e)))
    sides, plans, curves = records_case()
    recs, _, dropped, stages, utility, _, splits, paths = (
        SP._split_records(sides, plans, curves, 3, 0., tolQ))
    s, = splits
    assert dropped == [('above', 0, 2, 40., -1.)]
    assert s.branches == [[0], []] and s.fractions == (.5, .5)
    assert paths[0] == [s, 1, 2] and paths[2] == [1]
    assert s.H_mix == 140. and utility == [40., 50., 90.]


def test_split_side_candidates_and_the_hook():
    # the hook: a side the unsplit search serves returns exactly as today;
    # a root proof goes to the split path, which reports its candidates
    dT, rows = SPLIT['smith2005_exr18_4']
    sides = sides_from(rows, dT)
    split = dict(Qmin=0., exclude={}, prefer={})
    for name, side in sides.items():
        off = P._plan_side(side)
        on = P._plan_side(side, split=split)
        if off.status != 'best_effort':   # 'mer' or 'trivial'
            assert (on.pieces, on.gaps, on.method, on.work, on.units) == (
                off.pieces, off.gaps, off.method, off.work, off.units)
            assert on.cells == [] and on.split is None
            continue
        assert on.status == 'mer' and on.pieces == [] and on.cells
        assert on.proof == off.proof and on.gaps == [0.] * side.M
        assert on.units == len(SP._merge_cells(on.cells, side.tolQ))
        assert set(on.split) == {'candidate', 'signature', 'candidates',
                                 'stages', 'branches', 'preleak', 'leak',
                                 'small', 'errors'}
        _verify_side_cells(side, on.cells)


def test_failed_split_keeps_its_diagnostics(monkeypatch):
    # every generator fails: the side falls back to best effort, and its
    # split info keeps the attempt (no candidate, the reasons and errors),
    # unlike a side that never tried to split (None)
    def fail(side, a0, strategy, *args, **kw):
        raise SP._SplitInvariantError('injected')
    monkeypatch.setattr(SP, '_drive', fail)
    monkeypatch.setattr(SP, '_SPLIT_RULES', ())
    dT, rows = SPLIT['smith2005_exr18_4']
    streams = streams_from(rows)
    off = P._plan_numeric(streams, dT)['plan'].info['sides']
    plan = P._plan_numeric(streams, dT, **SPLIT_ON)['plan']
    assert plan.status == 'best_effort' and plan.splits == []
    assert split_infos(plan) == {}
    tried = 0
    for name, s in plan.info['sides'].items():
        o = off[name]
        assert (s['status'], s['method'], s['gaps'], s['work']) == (
            o['status'], o['method'], o['gaps'], o['work'])
        if o['status'] != 'best_effort':
            assert s['split'] is None
            continue
        tried += 1
        sp = s['split']
        assert set(sp) == {'candidate', 'signature', 'candidates', 'stages',
                           'branches', 'preleak', 'leak', 'small', 'errors'}
        assert sp['candidate'] is None and sp['signature'] is None
        assert sp['candidates'] == dict.fromkeys(SP._CORE_STRATEGIES, 'error')
        assert sp['errors'] == [f'{k}: injected' for k in SP._CORE_STRATEGIES]
        assert 0. <= sp['preleak'] <= SP._preleak_max(
            sides_from(rows, dT)[name])
    assert tried


def test_exhausted_schedule_splits(monkeypatch):
    # no root proof, but the unsplit search runs out of budget and best
    # effort leaves a penalty (> 1e-3 tolQ): the split path serves the side
    monkeypatch.setattr(P, '_SCHEDULE', tuple(p[:3] + (1.,)
                                              for p in P._SCHEDULE))
    monkeypatch.setattr(P, '_BE_WORK', 3000.)   # keep best effort short
    dT, rows = HARD['rnd_nested_n2-6_s237']
    streams = streams_from(rows)
    off = P._plan_numeric(streams, dT)
    assert off['status'] == 'best_effort' and not root_proof(off['plan'])
    # spies: the work the split path is handed, and every core candidate's
    handed, drawn = {}, {}
    split_side, drive = SP._split_side, SP._drive

    def spy_split(side, proof, cap1, forbid, work_scale, work, split):
        handed[side.name] = work
        return split_side(side, proof, cap1, forbid, work_scale, work, split)

    def spy_drive(side, *args, **kw):
        c = drive(side, *args, **kw)
        drawn.setdefault(side.name, []).append(0. if c is None else c.work)
        return c
    monkeypatch.setattr(SP, '_split_side', spy_split)
    monkeypatch.setattr(SP, '_drive', spy_drive)
    net = P._plan_numeric(streams, dT, **SPLIT_ON)
    assert net['status'] == 'mer'
    check_split_network(streams, dT, net)
    infos = split_infos(net['plan'])
    assert infos
    for name, sp in infos.items():
        s = net['plan'].info['sides'][name]
        assert s['proof'] is None and s['method'] in {
            'split-' + n for n in SP._CORE_STRATEGIES}
        # the best effort's work is counted once, with the candidates'
        assert handed[name] == off['plan'].info['sides'][name]['work']
        assert s['work'] == math.fsum([handed[name], *drawn[name]])


def two_core_generators(monkeypatch):
    """A second core generator for the selection tests: 'V0' is the chain
    of elementary vertical blocks (a different network from V)."""
    drive = SP._drive

    def fake(side, a0, strategy, *args, **kw):
        if strategy != 'V0':
            return drive(side, a0, strategy, *args, **kw)
        with monkeypatch.context() as m:
            m.setattr(SP, '_SPLIT_COARSEN', False)
            return drive(side, a0, strategy, *args, **kw)
    monkeypatch.setattr(SP, '_CORE_ORDER', SP._CORE_ORDER + ('V0',))
    monkeypatch.setattr(SP, '_CORE_STRATEGIES', ('V', 'V0'))
    monkeypatch.setattr(SP, '_drive', fake)


def test_split_exclusion_by_signature(monkeypatch):
    dT, rows = SPLIT['smith2005_ex16_5_five_stream']
    streams = streams_from(rows)

    def plan(**kw):
        net = P._plan_numeric(streams, dT, **SPLIT_ON, **kw)
        check_split_network(streams, dT, net)
        (name, sp), = split_infos(net['plan']).items()
        return name, sp
    # the whole portfolio (Stage S and the core): excluding the chosen
    # network gives a different one; excluding every network in turn
    # still returns a candidate (never None)
    name, sp = plan()
    assert sp['candidate'] in S_NAMES
    ex = {sp['signature']}
    _, sp2 = plan(_split_exclude={name: set(ex)})
    assert sp2['signature'] not in ex
    assert sp2['candidates'][sp['candidate']] == 'excluded'
    while True:
        _, last = plan(_split_exclude={name: set(ex)})
        if last['signature'] in ex:
            break
        ex.add(last['signature'])
    assert len(ex) >= 3 and not any(
        isinstance(k, tuple) for k in last['candidates'].values())
    # stickiness: a live preferred Stage S candidate that plans the same
    # network as before (its signature) is taken first, alone
    _, pr = plan(_split_prefer={name: (sp2['candidate'], sp2['signature'])})
    assert (pr['candidate'], pr['signature']) == (sp2['candidate'],
                                                  sp2['signature'])
    assert set(pr['candidates']) == {sp2['candidate']}
    # one whose network changed (another signature) is not: the whole
    # portfolio runs (the preferred one once) and the best key wins
    _, pr = plan(_split_prefer={name: (sp2['candidate'], sp['signature'])})
    assert (pr['candidate'], pr['signature']) == (sp['candidate'],
                                                  sp['signature'])
    assert set(pr['candidates']) == set(sp['candidates'])
    # an excluded preferred one: the rest of the portfolio, not it again
    _, pr = plan(_split_prefer={name: (sp['candidate'], sp['signature'])},
                 _split_exclude={name: {sp['signature']}})
    assert pr['signature'] == sp2['signature']
    assert pr['candidates'][sp['candidate']] == 'excluded'
    assert set(pr['candidates']) == set(sp['candidates'])
    # the core alone (Stage S off), its four strategies: LV and LVT build
    # one network here, so excluding the pick excludes both
    monkeypatch.setattr(SP, '_SPLIT_RULES', ())
    name, sp = plan()
    assert set(sp['candidates']) == set(SP._CORE_STRATEGIES)
    assert sp['candidate'] == 'LV'
    _, sp2 = plan(_split_exclude={name: {sp['signature']}})
    assert sp2['candidates']['LV'] == sp2['candidates']['LVT'] == 'excluded'
    assert sp2['candidate'] == 'VT' and sp2['signature'] != sp['signature']
    # one core candidate: excluding its signature still returns it (a
    # side's last candidate is never excluded)
    monkeypatch.setattr(SP, '_CORE_STRATEGIES', ('V',))
    name, sp = plan()
    sig = sp['signature']
    name2, sp2 = plan(_split_exclude={name: {sig}})
    assert (name2, sp2['candidate'], sp2['signature']) == (name, 'V', sig)
    assert sp2['candidates'] == {'V': 'excluded'}
    # two core candidates with different networks
    two_core_generators(monkeypatch)
    _, sp = plan()
    assert set(sp['candidates']) == {'V', 'V0'}
    assert sp['candidate'] == 'V'   # fewer units + branches + stages
    assert sp['candidates']['V'] < sp['candidates']['V0']
    _, ex = plan(_split_exclude={name: {sig}})
    assert ex['candidate'] == 'V0' and ex['signature'] != sig
    assert ex['candidates']['V'] == 'excluded'
    both = {sig, ex['signature']}
    _, last = plan(_split_exclude={name: both})
    assert last['candidate'] == 'V' and last['signature'] == sig
    # stickiness: a live preferred candidate is taken as is, first, while
    # its network is unchanged
    _, pr = plan(_split_prefer={name: ('V0', ex['signature'])})
    assert pr['candidate'] == 'V0' and set(pr['candidates']) == {'V0'}
    _, pr = plan(_split_prefer={name: ('V0', ex['signature'])},
                 _split_exclude={name: {ex['signature']}})
    assert pr['candidate'] == 'V' and set(pr['candidates']) == {'V', 'V0'}
    _, pr = plan(_split_prefer={name: ('V0', sig)})
    assert pr['candidate'] == 'V' and set(pr['candidates']) == {'V', 'V0'}
    # the exclusion is by network, not by name: another side's is ignored
    _, other = plan(_split_exclude={'none': {sig}})
    assert other['candidate'] == 'V'


def drifted(signature, d):
    """`signature` with every split fraction moved by `d` (alternating in
    sign, as a refine round moves the CP ratios of the knots)."""
    cells, fracs = signature
    return cells, tuple((role, stream, k, tuple(
        f + (d if b % 2 else -d) for b, f in enumerate(fs)))
        for role, stream, k, fs in fracs)


def test_split_identity_tolerates_refined_fractions():
    # a split's fractions are CP ratios on the knots, so a refine round
    # moves them (by ~1e-6 on real thermo) while the network stays the
    # same: the preferred and the excluded networks are matched by
    # `_same_network` (the structure exactly, the fractions within
    # `_SPLIT_SAME_FRACTION`), not by an exact signature
    dT, rows = SPLIT['smith2005_ex16_5_five_stream']
    streams = streams_from(rows)

    def plan(**kw):
        net = P._plan_numeric(streams, dT, **SPLIT_ON, **kw)
        (name, sp), = split_infos(net['plan']).items()
        return name, sp
    name, sp = plan()
    sig = sp['signature']
    same, tol = SP._same_network, SP._SPLIT_SAME_FRACTION
    assert sig[1] and 1e-6 < tol <= 1e-3
    assert same(sig, sig) and same(sig, drifted(sig, 1e-6))
    assert same(sig, drifted(sig, 0.5 * tol))
    assert not same(sig, drifted(sig, 2. * tol))
    assert not same(sig, (sig[0][1:], sig[1]))
    assert not same(sig, (sig[0], ()))
    # the preferred network, drifted, is still taken first and alone
    _, pr = plan(_split_prefer={name: (sp['candidate'],
                                       drifted(sig, 1e-6))})
    assert pr['candidate'] == sp['candidate']
    assert set(pr['candidates']) == {sp['candidate']}
    # an excluded network, drifted, is still excluded
    _, ex = plan(_split_exclude={name: {drifted(sig, 1e-6)}})
    assert not same(ex['signature'], sig)
    assert ex['candidates'][sp['candidate']] == 'excluded'


def test_split_signature_is_knot_independent():
    # the refine loop re-plans on refined knots and excludes (or prefers)
    # by network (`_same_network`); extra collinear knots move no fraction,
    # so they give the same pick with the same signature
    cases = [*SPLIT.values(), *(corpus_problem(name) for name in (
        'smith2005_exr18_5_nine_stream', 'fs_22sp_ph', 'cgm_unbalanced10'))]
    n = 0
    for dT, rows in cases:
        streams = streams_from(rows)
        picks = []
        for knots in (1, 3):
            plan = P._plan_numeric(streams, dT, knots, **SPLIT_ON)['plan']
            picks.append({name: (sp['candidate'], sp['signature'])
                          for name, sp in split_infos(plan).items()})
        assert picks[0] == picks[1]
        n += len(picks[0])
    assert n >= len(SPLIT)


def check_split_curves(kn, is_hot, dT, plan, tol=1e-6):
    """
    Independent knot walk of a plan on piecewise-linear streams (knots
    ``(T, H)``): T(H) by interpolation on the knots; every stream along
    ``plan.paths`` from its inlet, trunk records in series, the branches
    of a split in parallel from ``H_split`` by ``Q / f`` each, the mix at
    ``H_split -/+ sum Q``; the isothermal flag and a non-isothermal remix
    last; the utility closes every stream; the approach is at least dT at
    both ends and at every knot of every exchanger; the utilities equal
    the targets of `knot_cascade` up to the pre-leak.
    """
    scale = sum(H[-1] - H[0] for _, H in kn)
    qtol = 1e-9 * scale
    tolQ = plan.info['tolQ']
    index = {id(s): k for k, s in enumerate(plan.splits)}
    seen = Counter()
    for j, ((T, H), hot) in enumerate(zip(kn, is_hot)):
        sg, role = (-1., 'hot') if hot else (1., 'cold')
        io = ('H_hot_in', 'H_hot_out') if hot else ('H_cold_in', 'H_cold_out')
        Hc = H[-1] if hot else H[0]

        def run(ns, Hb, f, branch):
            for n in ns:
                e = plan.exchangers[n]
                seen[n] += 1
                assert getattr(e, role) == j and e.Q > 0.
                assert getattr(e, role + '_frac') == f
                assert getattr(e, role + '_branch') == branch
                H2 = Hb + sg * e.Q / f
                assert_allclose([getattr(e, io[0]), getattr(e, io[1])],
                                [Hb, H2], rtol=0, atol=qtol)
                Hb = H2
            return Hb
        for item in plan.paths[j]:
            if not isinstance(item, SP.Split):
                Hc = run([item], Hc, 1., None)
                continue
            k = index[id(item)]
            assert item.stream == j and len(item.branches) >= 2
            assert abs(math.fsum(item.fractions) - 1.) <= 1e-12
            assert item.H_split == pytest.approx(Hc, abs=qtol)
            ends = [run(ns, Hc, f, (k, b)) for b, (f, ns) in enumerate(
                zip(item.fractions, item.branches))]
            D = math.fsum(plan.exchangers[n].Q for ns in item.branches
                          for n in ns)
            Hc += sg * D
            assert item.H_mix == pytest.approx(Hc, abs=qtol)
            assert item.isothermal == all(
                abs(x - item.H_mix) <= SP._ISO_TOL * tolQ for x in ends)
            assert item.isothermal or item is plan.paths[j][-1]
        need = Hc - H[0] if hot else H[-1] - Hc
        assert need >= -qtol
        assert plan.utility[j] == pytest.approx(need, abs=qtol)
    assert all(seen[n] == 2 for n in range(len(plan.exchangers)))
    for e in plan.exchangers:   # counter-current: tau is the duty done
        (Th, Hh), (Tc, Hc) = kn[e.hot], kn[e.cold]
        taus = {0., e.Q}
        taus.update((e.H_hot_in - h) * e.hot_frac for h in Hh
                    if e.H_hot_out < h < e.H_hot_in)
        taus.update((e.H_cold_out - h) * e.cold_frac for h in Hc
                    if e.H_cold_in < h < e.H_cold_out)
        for tau in taus:
            th = np.interp(e.H_hot_in - tau / e.hot_frac, Hh, Th)
            tc = np.interp(e.H_cold_out - tau / e.cold_frac, Hc, Tc)
            assert th - tc >= dT - tol
    Qh_t, Qc_t = knot_cascade(kn, is_hot, dT)
    preleak = math.fsum(sp['preleak'] for sp in split_infos(plan).values())
    assert abs(plan.Q_hot - Qh_t) <= qtol + preleak
    assert abs(plan.Q_cold - Qc_t) <= qtol + preleak


def pinch_curve_problem(rng):
    """Random piecewise-linear problem built around a pinch at P (as
    `pinch_problem`), with CP changes and point loads, some at the
    pinch."""
    dT, P0 = 10., float(rng.choice(range(100, 200, 10)))

    def stream(lo, hi):
        Ts = sorted({lo, hi, *rng.sample(range(int(lo) + 1, int(hi)),
                                         rng.randint(0, 2))})
        T, H = [float(Ts[0])], [0.]
        for t in Ts[1:]:
            if rng.random() < 0.35:   # a point load at the current level
                T.append(T[-1])
                H.append(H[-1] + rng.choice([5., 10., 20.]))
            T.append(float(t))
            H.append(H[-1] + rng.choice([0.5, 1., 2., 3.]) * (t - T[-2]))
        return T, H
    kn, hot = [], []
    for is_hot, base in ((True, P0 + dT), (False, P0)):
        for _ in range(rng.randint(1, 3)):
            lo = base - rng.choice([0, 0, 10, 20])
            kn.append(stream(lo, base + rng.choice(range(10, 60, 5))))
            hot.append(is_hot)
    return kn, hot, dT


def curved_split_problems():
    """Point-load and CP-change variants of the number rule and the CP rule
    (each must split), then random pinch-centred curve problems."""
    yield [([100., 200.], [0., 100.]), ([100., 200.], [0., 100.]),
           ([90., 190., 190.], [0., 300., 310.])], [True, True, False], 10.
    yield [([100., 150., 200.], [0., 40., 100.]), ([100., 200.], [0., 100.]),
           ([90., 140., 190.], [0., 160., 300.])], [True, True, False], 10.
    yield [([100., 150., 150., 200.], [0., 150., 170., 320.]),
           ([90., 140., 190.], [0., 80., 200.]), ([90., 190.], [0., 200.])], [
        True, False, False], 10.
    rng = random.Random(83)
    for _ in range(60):
        yield pinch_curve_problem(rng)


def test_curved_split_plans_walk_on_the_knots():
    # plan_network with splitting on piecewise-linear streams (CP changes,
    # point loads): the records, fractions, mixes and approaches hold on
    # an independent walk of the knots, and the plan is MER
    n_split = n_flat = 0
    for k, (kn, hot, dT) in enumerate(curved_split_problems()):
        plan = P.plan_network(kn, hot, dT, stream_splitting=True)
        assert plan.status == 'mer'
        for s in plan.info['sides'].values():
            assert s['split'] is None or s['split']['errors'] == []
        assert k >= 3 or plan.splits   # the constructed cases split
        check_split_curves(kn, hot, dT, plan)
        n_split += bool(plan.splits)
        n_flat += bool(plan.splits) and any(
            len(set(T)) < len(T) for T, _ in kn)
    assert n_split >= 10 and n_flat >= 5


def test_fuzz_splitting_always_reaches_mer(monkeypatch):
    monkeypatch.setattr(P, '_BE_WORK', 3000.)   # keep best effort short
    monkeypatch.setattr(SP, '_SPLIT_S_WORK', 3000.)
    rules = SP._SPLIT_RULES
    rng = random.Random(29)
    n_split = n_same = n_s = 0
    for seed in range(300):
        # half the seeds with the core alone (Stage S off)
        monkeypatch.setattr(SP, '_SPLIT_RULES', () if seed % 2 else rules)
        jitter = seed % 3 == 2   # temperatures closer than tolQ/CP
        if jitter:
            rows, dT = jittered_problem(seed)
        elif seed % 3:   # built around a pinch, violating the pinch rules
            rows, dT = pinch_problem(rng)
            while not needs_split(streams_from(rows), dT):
                rows, dT = pinch_problem(rng)
        else:
            rows, dT = random_problem(rng)
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT, **SPLIT_ON)
        plan = net['plan']
        assert net['status'] == 'mer', (seed, rows, dT)
        assert plan.info['dropped'] == []
        check_split_network(streams, dT, net, exact=not jitter)
        infos = split_infos(plan)
        assert all(sp['leak'] == 0. and sp['errors'] == []
                   for sp in infos.values())
        if seed % 2:
            assert all(sp['candidate'] in SP._CORE_STRATEGIES
                       for sp in infos.values())
        n_s += sum(sp['candidate'] in S_NAMES for sp in infos.values())
        n_split += bool(infos)
        if not needs_split(streams, dT):
            off = P._plan_numeric(streams, dT)
            assert fingerprint(off) == fingerprint(net) and not plan.splits
            assert [(e.H_hot_in, e.H_cold_in) for e in plan.exchangers] == [
                (e.H_hot_in, e.H_cold_in) for e in off['plan'].exchangers]
            n_same += 1
    assert n_split >= 110 and n_same >= 150 and n_s >= 40


# %% 18. Stage S cut transports (hensmith._splitting)

CUT_RULES = ('partner', 'demand', 'mincell', 'nw-rho-desc', 'nw-rho-asc',
             'nw-exact-desc', 'nw-exact-asc')
TWO_HOT_ONE_COLD = (10., [  # the number rule: two hot streams, one cold
    ('H1', 'h', 200, 100, 1.), ('H2', 'h', 200, 100, 1.),
    ('C1', 'c', 90, 190, 3.)])
ONE_HOT_TWO_COLD = (10., [  # the CP rule: no cold stream is large enough
    ('H1', 'h', 200, 100, 3.), ('C1', 'c', 90, 190, 2.),
    ('C2', 'c', 90, 190, 2.)])
TINY_BRANCH = (10., [  # the cold branch for H2 would carry 1e-4 of it
    ('H1', 'h', 200, 100, 1.), ('H2', 'h', 200, 100, 1e-4),
    ('C1', 'c', 90, 190, 3.)])


def corpus_problem(name):
    """(dT, rows) of a constant-CP corpus case (`hxn_mer_cases`), in K and
    kW."""
    case = next(c for c in CORPUS_SPLIT if c['name'] == name)
    sc, off = T_UNITS[case['T_unit']]
    kw = Q_UNITS[case['Q_unit']]
    return case['dTmin'] * sc, [
        (n, kind[0], (Ti + off) * sc, (To + off) * sc, cp * kw / sc)
        for n, kind, Ti, To, cp in case['streams']]


def root_of(dT, rows, name='above'):
    """A side of a constant-CP problem, its pre-leaked root and its root
    proof."""
    side = sides_from(rows, dT)[name]
    z_m, z_f = [0.] * side.M, [0.] * side.F
    proof = side.rules_violation(z_m, z_f, side.analyse(z_m, z_f))
    return side, SP._preleak_root(side)[1], proof


def cut_sets(side, a0, proof):
    return SP._cut_sets(side, a0, [0.] * side.F, proof['level'],
                        proof['cut'], proof['rule'])


def cps(curves):
    """CP of every constant-CP curve."""
    return [1. / P._slope1(c) for c in curves]


def check_cut_transport(dem, cap, rule, cells):
    """Independent check of the cells of a cut transport: every demand
    served in full, no capacity loaded beyond its CP, fractions in (0, 1]
    summing to 1 per item, branch CP ratio ``g c / (f m) >= 1`` in every
    cell (uniform, ``rho = sum c / sum m``, for the rho rules), a forest,
    and each rule's shape."""
    m, c = dict(dem), dict(cap)
    fr = SP._cut_fractions(cells)
    assert len(fr) == len(cells)
    for d in m:
        loads = [x for k, _, x in cells if k == d]
        assert loads and math.fsum(loads) == pytest.approx(m[d], rel=1e-12)
    for k in c:
        assert math.fsum(x for _, j, x in cells if j == k) <= c[k] * (
            1. + 1e-12)
    ratio = []
    for (d, k, x), (f, g) in zip(cells, fr):
        assert x > 0. and 0. < f <= 1. and 0. < g <= 1.
        ratio.append(g * c[k] / (f * m[d]))
    assert min(ratio) >= 1. - 1e-12
    for role in (0, 1):
        for item in {cell[role] for cell in cells}:
            assert abs(math.fsum(f[role] for cell, f in zip(cells, fr)
                                 if cell[role] == item) - 1.) <= 1e-12
    assert is_forest([(d, k) for d, k, _ in cells])
    if rule.startswith('nw-rho'):
        rho = math.fsum(c.values()) / math.fsum(m.values())
        assert_allclose(ratio, rho, rtol=1e-12)
    if rule == 'partner':   # the demands stay whole
        assert len(cells) == len(m)
    if rule == 'demand':    # the capacities stay whole
        assert len({k for _, k, _ in cells}) == len(cells)


def check_items(side, a0, items, extra):
    """The branched side of `items` keeps the cascade, satisfies the pinch
    rules at its pre-leaked root, and its branches are well formed."""
    bside, a = SP._branched_side(side, items, a0)
    roles = [it[0] for it in items]
    assert roles == sorted(roles, key=lambda r: r != 'must')
    for curve, (role, p, f, _) in zip(bside.musts + bside.flexes, items):
        parent = (side.musts if role == 'must' else side.flexes)[p]
        assert curve.y == parent.y and curve.q == [f * q for q in parent.q]
    assert a == [f * a0[p] for role, p, f, _ in items if role == 'must']
    z = [0.] * bside.F
    d = bside.analyse(a, z)
    assert bside.rules_violation(a, z, d) is None
    assert abs(d.slack - side.analyse(a0, [0.] * side.F).slack) <= (
        10. * side.tolQ)
    parents = {}
    for role, p, f, key in items:
        parents.setdefault((role, p), []).append((f, key))
    assert sorted(parents) == sorted(
        [('must', i) for i in range(side.M)]
        + [('flex', j) for j in range(side.F)])
    for (role, p), br in parents.items():
        if len(br) == 1:
            assert br == [(1., None)]
            continue
        assert abs(math.fsum(f for f, _ in br) - 1.) <= 1e-12
        assert all(f >= SP._SPLIT_MIN_FRACTION for f, _ in br)
        assert [k for _, k in br] == [('S', p, n) for n in range(len(br))]
    assert extra == sum(len(br) - 1 for br in parents.values())
    return bside


@pytest.mark.parametrize('rule', CUT_RULES)
def test_cut_transport_rules(rule):
    assert SP._SPLIT_RULES == CUT_RULES
    # the number rule (2h1c): the cold stream hosts both hot streams, +1
    side, a0, proof = root_of(*TWO_HOT_ONE_COLD)
    assert proof['rule'] == 'outward'
    dem, cap = cut_sets(side, a0, proof)
    assert [d for d, _ in dem] == [0, 1] and [k for k, _ in cap] == [0]
    assert_allclose([m for _, m in dem] + [c for _, c in cap], [1., 1., 3.],
                    rtol=1e-12)
    out = SP._pinch_split(side, a0, proof, rule)
    if rule == 'demand':   # the cold stream is matched: none is left
        assert out is None and SP._cut_transport(dem, cap, rule) is None
    else:
        items, extra = out
        check_items(side, a0, items, extra)
        assert extra == 1
        assert [it[:2] for it in items] == [
            ('must', 0), ('must', 1), ('flex', 0), ('flex', 0)]
        assert_allclose([it[2] for it in items[2:]], [.5, .5], rtol=1e-12)
    # the CP rule (1h2c): the hot stream splits over both cold streams, +1
    side, a0, proof = root_of(*ONE_HOT_TWO_COLD)
    out = SP._pinch_split(side, a0, proof, rule)
    if rule == 'partner':   # no cold stream has room for the hot one
        assert out is None
    else:
        items, extra = out
        check_items(side, a0, items, extra)
        assert extra == 1 and [it[0] for it in items] == [
            'must', 'must', 'flex', 'flex']
        assert_allclose([it[2] for it in items[:2]],
                        [2 / 3, 1 / 3] if rule.startswith('nw-exact')
                        else [.5, .5], rtol=1e-12)
    # zero CP slack (smith2005_ex18_4 below: the cold stream's CP 300 on
    # the hot CPs 200 and 100): every branch has its partner's CP
    side, a0, proof = root_of(*corpus_problem('smith2005_ex18_4_split'),
                              name='below')
    out = SP._pinch_split(side, a0, proof, rule)
    if rule == 'partner':
        assert out is None
    else:
        bside = check_items(side, a0, *out)
        assert out[1] == 1
        assert_allclose(sorted(cps(bside.flexes)), sorted(cps(bside.musts)),
                        rtol=1e-12)
    # cornell above: CP 40 and 30 on CP 60 and 20; no whole demand or
    # capacity rule, and the minimum is 3 cells (k = 1: one super-bin)
    side, a0, proof = root_of(
        *corpus_problem('cornell_processdesign_four_stream_split'))
    dem, cap = cut_sets(side, a0, proof)
    assert_allclose([m for _, m in dem], [40., 30.], rtol=1e-12)
    assert_allclose([c for _, c in cap], [60., 20.], rtol=1e-12)
    cells = SP._cut_transport(dem, cap, rule)
    if rule in ('partner', 'demand'):
        assert cells is None
    else:
        check_cut_transport(dem, cap, rule, cells)
        assert len(cells) == 3
        if rule == 'mincell':
            assert cells == SP._cut_transport(dem, cap, 'nw-rho-desc')
    # random cut sets, on scales from 1e-3 to 1e3; half of them tight (the
    # capacities scaled to the demands' CP sum, or up to 30 % above it)
    rng = random.Random(CUT_RULES.index(rule))
    grid = [.5, 1., 1.5, 2., 3., 4., 7.]

    def cp(hi):
        return rng.choice(grid) if rng.random() < .7 else rng.uniform(.1, hi)
    n_ok = n_whole = 0
    for _ in range(300):
        s = rng.choice([1e-3, 1., 1e3])
        dem = [(n, s * cp(5.)) for n in range(rng.randint(1, 4))]
        cap = [(n, s * cp(8.)) for n in range(rng.randint(1, 4))]
        if rng.random() < .5:
            k = (math.fsum(m for _, m in dem) / math.fsum(c for _, c in cap)
                 * rng.choice([1., rng.uniform(1., 1.3)]))
            cap = [(n, c * k) for n, c in cap]
        cells = SP._cut_transport(dem, cap, rule)
        if math.fsum(c for _, c in cap) < math.fsum(m for _, m in dem) * (
                1. - 1e-12):
            assert cells is None
            continue
        if rule not in ('partner', 'demand'):   # these never fail then
            assert cells is not None
        if cells is not None:
            check_cut_transport(dem, cap, rule, cells)
            n_ok += 1
        # the matching is maximum: where the rule holds whole, the number
        # and CP rules and mincell split nothing
        if rule in ('partner', 'demand', 'mincell') and _matching(
                [m for _, m in dem], [c for _, c in cap],
                lambda m, c: c >= m * (1. - 1e-12)):
            assert len(cells) == len(dem) == len({k for _, k, _ in cells})
            n_whole += 1
    assert n_ok >= 100 and (n_whole >= 40 or rule.startswith('nw-'))


def test_mincell_prefers_fewer_cells():
    # k = 0 packs the demands whole: CP 2 and 1 into CP 3 and 1, 2 cells
    dem, cap = [(0, 2.), (1, 1.)], [(0, 3.), (1, 1.)]
    cells = SP._cut_transport(dem, cap, 'mincell')
    check_cut_transport(dem, cap, 'mincell', cells)
    assert sorted((d, k) for d, k, _ in cells) == [(0, 0), (1, 1)]
    assert len(SP._cut_transport(dem, cap, 'nw-rho-desc')) == 3
    # CP 1.2 bins hold one demand each: only k = 4 packs all six, so the
    # rule falls back to one group
    dem = [(n, 1.) for n in range(6)]
    cap = [(n, 1.2) for n in range(5)]
    assert SP._cut_transport(dem, cap, 'mincell') == SP._cut_transport(
        dem, cap, 'nw-rho-desc')


def test_pinch_split_screens_the_branched_root(monkeypatch):
    # a branch below _SPLIT_MIN_FRACTION merges into its sibling, which
    # undoes the split: the rule fails
    side, a0, proof = root_of(*TINY_BRANCH)
    assert SP._pinch_split(side, a0, proof, 'partner') is None
    with monkeypatch.context() as mp:
        mp.setattr(SP, '_SPLIT_MIN_FRACTION', 0.)
        items, extra = SP._pinch_split(side, a0, proof, 'partner')
        assert extra == 1 and min(it[2] for it in items) < 1e-3
    # a cut with a flat demand: splitting cannot serve it
    flat = P._LevelCurve([0., 5., 10.], [100., 100., 120.], 0)
    slope = P._LevelCurve([0., 10.], [100., 110.], 1)
    fs = P._Side('above', [flat, slope], [slope], 1e-9, 1e-9)
    assert SP._cut_sets(fs, [0., 0.], [0.], 100., '-', 'outward') is None
    # Lemma R broken (a branch that is not the scaled parent): raises
    side, a0, proof = root_of(*TWO_HOT_ONE_COLD)
    branch = SP._branch_curve
    monkeypatch.setattr(SP, '_branch_curve', lambda c, f, role='must': (
        branch(c, .5 * f, role)))
    with pytest.raises(SP._SplitInvariantError):
        SP._pinch_split(side, a0, proof, 'partner')


def test_pinch_split_on_the_corpus(monkeypatch):
    """Every constant-CP SPLIT side with a rules proof: the branched side of
    every rule that succeeds keeps the cascade and satisfies the pinch rules
    at its pre-leaked root, and some rule succeeds. On a double pinch a
    split can move the violation to the other tight cut."""
    n_sides = 0
    for case in CORPUS_SPLIT:
        if case['kind'] != 'constant_cp':
            continue
        dT, rows = corpus_problem(case['name'])
        for name, side in sides_from(rows, dT).items():
            z_m, z_f = [0.] * side.M, [0.] * side.F
            proof = side.rules_violation(z_m, z_f, side.analyse(z_m, z_f))
            if proof is None:
                continue
            a0 = SP._preleak_root(side)[1]
            ok = []
            for rule in CUT_RULES:
                out = SP._pinch_split(side, a0, proof, rule)
                if out is not None:
                    check_items(side, a0, *out)
                    ok.append(rule)
            assert ok, (case['name'], name)
            n_sides += 1
    assert n_sides == 32
    # nptel_t5_3 below: the outward split at the pinch breaks the inward
    # rule at the second tight cut, and the rho rules and mincell never
    # satisfy both; nw-exact-desc does at once, nw-exact-asc in three cuts
    side, a0, proof = root_of(
        *corpus_problem('nptel_t5_3_four_stream_split'), name='below')
    assert [rule for rule in CUT_RULES if SP._pinch_split(
        side, a0, proof, rule) is not None] == [
            'nw-exact-desc', 'nw-exact-asc']
    # the same with CP2 = 6 - 1.58e-11: the pre-leaked root is branched too
    side, a0, proof = root_of(*NEAR_DOUBLE_PINCH, name='below')
    assert max(a0) > 0. and proof['rule'] == 'outward'
    check_items(side, a0, *SP._pinch_split(side, a0, proof, 'nw-exact-asc'))
    monkeypatch.setattr(SP, '_SPLIT_CUTS', 2)
    assert SP._pinch_split(side, a0, proof, 'nw-exact-asc') is None
    assert SP._pinch_split(side, a0, proof, 'nw-exact-desc')[1] == 2


# %% 19. Stage S: pinch splits planned by the unchanged DFS

def stage_s(side, a0, proof, cap1=False, forbid=frozenset(), Qmin=0., **kw):
    """`_stage_s` with fresh errors and reasons."""
    errors, reasons = [], {}
    split = dict(Qmin=Qmin, exclude={}, prefer={})
    cands, _ = SP._stage_s(side, a0, proof, cap1, forbid, 1., split,
                           errors, reasons, **kw)
    return cands, errors, reasons


def test_stage_s_cells_fold_and_sweep():
    L = P._LevelCurve
    # fold (Lemma F): a CP-3 flex in three branches serves two CP-1 musts
    # from the pinch; the 0.1 branch is unused, so its fraction goes to
    # the others in proportion and their parent positions shrink
    musts = [L([0., 100.], [95., 195.], 0), L([0., 100.], [95., 195.], 1)]
    flex = L([0., 300.], [95., 195.], 2, role='flex')
    side = P._Side('above', musts, [flex], 1e-9, 1e-9)
    items = [('must', 0, 1., None), ('must', 1, 1., None),
             ('flex', 0, .5, ('S', 0, 0)), ('flex', 0, .4, ('S', 0, 1)),
             ('flex', 0, .1, ('S', 0, 2))]
    pieces = [(0, 0, 0., 0., 60.), (0, 0, 60., 60., 40.),
              (1, 1, 0., 0., 100.)]
    cells, why = SP._s_cells(side, items, [0., 0.], pieces)
    assert why is None
    assert [(c.i, c.j, c.x, c.a, c.b, c.f, c.km) for c in cells] == [
        (0, 0, 100., 0., 0., 1., None), (1, 0, 100., 0., 0., 1., None)]
    assert [(c.g, c.kf) for c in cells] == [(.5 / .9, ('S', 0, 0)),
                                            (.4 / .9, ('S', 0, 1))]
    _verify_side_cells(side, cells)
    # one used branch: a trunk again
    one = P._Side('above', musts[:1], [flex], 1e-9, 1e-9)
    cells, why = SP._s_cells(one, items[:1] + items[2:], [0.], pieces[:2])
    assert [(c.x, c.b, c.g, c.kf) for c in cells] == [(100., 0., 1., None)]
    # the residual sweep: must A ends r short (r <= tolQ, the DFS's own
    # tolerance); r goes to A's far-end cell, and the later cell of the
    # same flex (B's) moves with it, so the flex stays a prefix
    musts = [L([0., 50.], [95., 145.], 0), L([0., 100.], [95., 195.], 1)]
    flexes = [L([0., 150.], [95., 195.], 2, role='flex'),
              L([0., 150.], [95., 195.], 3, role='flex')]
    side = P._Side('above', musts, flexes, 1e-9, 1e-9)
    items = [('must', 0, 1., None), ('must', 1, 1., None),
             ('flex', 0, 1., None), ('flex', 1, 1., None)]
    for r in (.5e-9, -.5e-9):
        pieces = [(0, 0, 0., 0., 50. - r), (1, 1, 0., 0., 60.),
                  (1, 0, 60., 50. - r, 40.)]
        cells, why = SP._s_cells(side, items, [0., 0.], pieces)
        assert why is None
        assert abs(cells[0].x - 50.) <= 1e-13
        assert cells[2].b == cells[0].b_end and cells[2].x == 40.
        _verify_side_cells(side, cells)
    # beyond tolQ, or a must with no cell: rejected
    pieces = [(0, 0, 0., 0., 50. - 2e-9), (1, 1, 0., 0., 60.),
              (1, 0, 60., 50. - 2e-9, 40.)]
    assert SP._s_cells(side, items, [0., 0.], pieces) == (None, 'sweep')
    assert SP._s_cells(side, items, [0., 0.], pieces[:1]) == (None,
                                                              'sweep')


def test_stage_s_candidates_are_verified():
    """Every Stage S candidate on the TP split sides, the number and CP
    rule cases and the nptel_t5_3 double pinch: exact cells from the
    pre-leaked root, isothermal must remixes, no bad remix, the rule's
    order; the reasons name every rule."""
    problems = [SPLIT[n] for n in sorted(SPLIT)] + [
        TWO_HOT_ONE_COLD, ONE_HOT_TWO_COLD, NEAR_DOUBLE_PINCH,
        corpus_problem('nptel_t5_3_four_stream_split')]
    n = 0
    for dT, rows in problems:
        for name, side in sides_from(rows, dT).items():
            z_m, z_f = [0.] * side.M, [0.] * side.F
            proof = side.rules_violation(z_m, z_f, side.analyse(z_m, z_f))
            if proof is None:
                continue
            a0 = SP._preleak_root(side)[1]
            cands, errors, reasons = stage_s(side, a0, proof)
            assert errors == [] and cands and set(reasons) == set(S_NAMES)
            for c in cands:
                assert c.order == (0, S_NAMES.index(c.name))
                assert reasons[c.name] == c.key() and not c.excluded
                assert c.leak_by_must == [0.] * side.M and c.mixbad == 0
                _verify_side_cells(side, c.cells, a0)
                merged = SP._merge_cells(c.cells, side.tolQ)
                for (role, _, _), br in SP._stages(merged).items():
                    assert role == 'f' or SP._remix(role, br, side.tolQ)[2]
                n += 1
    assert n >= 10


def test_stage_s_budget(monkeypatch):
    # the rules share one budget: a rule not reached is 'budget', and the
    # core still serves the side
    monkeypatch.setattr(SP, '_SPLIT_S_WORK', 1.)
    dT, rows = SPLIT['smith2005_exr18_4']
    streams = streams_from(rows)
    net = P._plan_numeric(streams, dT, **SPLIT_ON)
    assert net['status'] == 'mer'
    check_split_network(streams, dT, net)
    (_, sp), = split_infos(net['plan']).items()
    assert sp['candidate'] in SP._CORE_STRATEGIES
    reasons = [sp['candidates'][n] for n in S_NAMES]
    assert 'budget' in reasons
    assert set(reasons) <= {'budget', 'no split', 'same split'}


def test_split_first_wins_is_deterministic(monkeypatch):
    monkeypatch.setattr(SP, '_SPLIT_FIRST_WINS', True)
    for dT, rows in [SPLIT[n] for n in sorted(SPLIT)] + [NEAR_DOUBLE_PINCH]:
        streams = streams_from(rows)
        one = P._plan_numeric(streams, dT, **SPLIT_ON)
        two = P._plan_numeric(streams, dT, **SPLIT_ON)
        assert one['status'] == 'mer'
        check_split_network(streams, dT, one)
        assert fingerprint(one) == fingerprint(two)
        infos = split_infos(one['plan'])
        assert infos and infos == split_infos(two['plan'])
        for sp in infos.values():   # the first live candidate wins
            live = [n for n, k in sp['candidates'].items()
                    if isinstance(k, tuple)]
            assert live == [sp['candidate']]


def test_splitting_with_avoid_recycle_never_repeats_a_pair(monkeypatch):
    monkeypatch.setattr(P, '_BE_WORK', 300.)   # keep best effort short
    monkeypatch.setattr(SP, '_SPLIT_S_WORK', 3000.)
    rng = random.Random(31)
    cases = [SPLIT[n] for n in sorted(SPLIT)]
    while len(cases) < 20:
        rows, dT = pinch_problem(rng)
        if needs_split(streams_from(rows), dT):
            cases.append((dT, rows))
    n_mer = n_split = n_rep = 0
    for dT, rows in cases:
        streams = streams_from(rows)
        net = P._plan_numeric(streams, dT, avoid_recycle=True, **SPLIT_ON)
        plan = net['plan']
        assert repeated_pairs(net) == []   # not even across the pinch
        assert net['status'] in ('mer', 'best_effort')
        check_split_network(streams, dT, net, mer=net['status'] == 'mer')
        n_mer += net['status'] == 'mer'
        n_split += bool(plan.splits)
        for sp in split_infos(plan).values():
            assert sp['errors'] == []
            n_rep += 'repeated pair' in sp['candidates'].values()
    # candidates repeating a pair were generated and rejected
    assert n_mer >= 6 and n_split >= 10 and n_rep >= 2


def test_split_side_keeps_cells_below_Qmin():
    # Qmin never drops a split cell: a split side's exchangers below Qmin count in the key but are
    # kept and reported (Qmin never costs MER); an unsplit side drops them
    # as today
    dT, rows = SPLIT['smith2005_exr18_4']
    streams = streams_from(rows)
    Qmin = 3.5   # above two split cells (2.2, 2.4) and an unsplit one (3.3)
    net = P._plan_numeric(streams, dT, Qmin=Qmin, **SPLIT_ON)
    plan = net['plan']
    above, below = plan.info['sides']['above'], plan.info['sides']['below']
    sp = above['split']
    assert above['status'] == below['status'] == 'mer'
    assert sp['small'] and all(q < Qmin for *_, q in sp['small'])
    # every exchanger of the split side is in the records, small ones too
    recs = [e for e in plan.exchangers if e.side == 'above']
    assert len(recs) == above['units']
    assert sorted((h, c, q) for h, c, q in sp['small']) == sorted(
        (e.hot, e.cold, e.Q) for e in recs if e.Q < Qmin)
    # the unsplit side's small exchanger is dropped, as without splitting
    (side, h, c, q), = plan.info['qmin_dropped']
    off = P._plan_numeric(streams, dT, Qmin=Qmin)['plan'].info
    assert side == 'below' and [d for d in off['qmin_dropped']
                                if d[0] == side] == [(side, h, c, q)]
    assert q == pytest.approx(3.304, abs=1e-3)
    assert net['status'] == 'best_effort'   # the drop raises the utility
    check_split_network(streams, dT, net, mer=False)


def test_stage_s_unit_bound_and_improvement(monkeypatch):
    def best(case, name):
        side, a0, proof = root_of(*corpus_problem(case), name=name)
        cands, errors, reasons = stage_s(side, a0, proof)
        assert errors == []
        return min(cands, key=SP._Candidate.key), reasons
    # the unit bound prunes the rules that cannot beat the incumbent
    # ('bound') and changes nothing else
    b, reasons = best('nptel_t4_4_four_stream_dT20', 'below')
    assert 'bound' in reasons.values()
    search = SP._s_search
    with monkeypatch.context() as m:
        m.setattr(SP, '_s_search', lambda *a: search(*a[:5], math.inf, a[6]))
        b0, r0 = best('nptel_t4_4_four_stream_dT20', 'below')
    assert 'bound' not in r0.values()
    assert (b0.name, b0.key(), b0.signature) == (b.name, b.key(), b.signature)
    # the winner's unit improvement: 7sp3 above, 8 -> 7 (units + extra
    # branches + split stages)
    b, _ = best('7sp3_dt20F', 'above')
    with monkeypatch.context() as m:
        m.setattr(SP, '_improve_units', lambda side, pieces, *a, **k: (
            pieces, 0.))
        b0, _ = best('7sp3_dt20F', 'above')
    assert (b0.key()[3], b.key()[3]) == (8, 7) and b.name == b0.name


# %% 20. Core pinch blocks and DFS tails (hensmith._splitting)

def test_search_b0_default_is_identical(monkeypatch):
    # _Search gains the flex frontiers b0 (and _improve_units/_units_guard
    # pass a0 and b0 through): at the default every search is today's,
    # and from a mid-plan node (a, b) the search plans the rest from there
    rng = random.Random(71)
    n_mid = 0
    spy_case = None
    for k in range(40):
        rows, dT = pinch_problem(rng) if k % 2 else random_problem(rng)
        for side in sides_from(rows, dT).values():
            if not (side.M and side.F):
                continue
            z_f = [0.] * side.F
            runs = [P._Search(side, 'restricted', 3, False, 3000., **kw).run()
                    for kw in ({}, dict(b0=None), dict(b0=z_f))]
            assert runs[0] == runs[1] == runs[2] and z_f == [0.] * side.F
            pieces = runs[0]
            if not pieces:
                continue
            if spy_case is None:
                spy_case = side, pieces
            args = (side, pieces, 100., False, frozenset(), 1.)
            assert (P._improve_units(*args) == P._improve_units(
                *args, a0=None, b0=None))
            # the node after the first piece
            i, j, _, _, x = pieces[0]
            a, b = [0.] * side.M, list(z_f)
            a[i], b[j] = min(x, side.Qm[i]), min(x, side.Qf[j])
            rest = P._Search(side, 'full', None, True, 3000., a0=a,
                             b0=b).run()
            if rest is None:
                continue
            n_mid += 1
            front = list(b)
            for i2, j2, a2, b2, x2 in rest:   # every flex a prefix from b
                assert b2 == front[j2] and x2 > 0.
                front[j2] = min(b2 + x2, side.Qf[j2])
            for i2 in range(side.M):   # every must served from a
                duty = math.fsum(p[4] for p in rest if p[0] == i2)
                assert abs(a[i2] + duty - side.Qm[i2]) <= side.tolQ
    assert n_mid >= 20
    # the unit searches start from (a0, b0)
    seen = []

    class Spy(P._Search):
        def __init__(self, *args, **kw):
            seen.append((kw.get('a0'), kw.get('b0')))
            super().__init__(*args, **kw)
    monkeypatch.setattr(P, '_Search', Spy)
    monkeypatch.setattr(P, '_GUARD_FACTOR', 0)   # run the guard
    side, pieces = spy_case   # frontiers that differ from each other
    a, b = [0.25 * q for q in side.Qm], [0.1 * q for q in side.Qf]
    P._improve_units(side, pieces, 100., False, frozenset(), 1., a0=a, b0=b)
    P._units_guard(side, pieces, False, frozenset(), 1., a0=a, b0=b)
    assert seen and all(s == (a, b) for s in seen)


def pinch_side(m2, cp_flex, Q_flex):
    """A side 'above' (tolerances 1e-9): musts m0 and m1 (CP 1, levels 0 to
    10) at the tight root level 0, a third must `m2` = (heat, lowest level,
    highest level) above it, and one flex over levels 0 to `Q_flex/cp_flex`.
    The rules fail outward at level 0 (two musts, one flex)."""
    L = P._LevelCurve
    Q2, lo, hi = m2
    return P._Side('above', [L([0., 10.], [0., 10.], 0),
                             L([0., 10.], [0., 10.], 1),
                             L([0., Q2], [lo, hi], 2)],
                   [L([0., Q_flex], [0., Q_flex / cp_flex], 3)], 1e-9, 1e-9)


def check_pinch_block(side, blk, a, b):
    """A pinch block from (a, b): exact cells, isothermal must branches,
    CP-based flex fractions (each branch of CP at least its must branch's),
    and (R) at its end node within `_SPLIT_R_TOL tolQ`."""
    lim = -SP._SPLIT_R_TOL * side.tolQ
    assert blk.kind == 'pinch' and blk.start == (a, b)
    v = side.rules_violation(a, b, side.analyse(a, b))
    assert v['rule'] == 'outward'
    dem, cap = map(dict, SP._cut_sets(side, a, b, v['level'], v['cut'],
                                      'outward'))
    a2, b2 = blk.end
    assert SP._exact(side).analyse(a2, b2).slack >= lim
    assert residual_slack(side, a2, b2) >= lim
    # tick-off: a must ends exactly at its end or more than tolQ short
    assert all(x == q or x < q - side.tolQ for x, q in zip(a2, side.Qm))
    fg = Counter()
    for c in blk.cells:
        assert c.a == a[c.i] and c.b == b[c.j] and c.x > 0.
        assert SP._cell_margin(side, c)[0] >= -side.tolP
        assert abs(c.a_end - a2[c.i]) <= side.tolQ   # isothermal
        assert c.g * cap[c.j] >= c.f * dem[c.i] * (1. - 1e-12)
        fg['m', c.i] += c.f
        fg['f', c.j] += c.g
    assert all(abs(s - 1.) <= 1e-12 for s in fg.values())
    for j in range(side.F):
        duty = math.fsum(c.x for c in blk.cells if c.j == j)
        assert abs(b[j] + duty - b2[j]) <= side.tolQ


TICK_OFF = (10., [  # H1 splits 0.7/0.3 and (f 400)/f rounds to 400 - 6e-14:
    # without the tick-off, H1 would stop 6e-14 short of its end
    ('H1', 'h', 200, 100, 4.), ('C1', 'c', 90, 190, 3.5),
    ('C2', 'c', 90, 190, 1.5)])


def test_pinch_block_acceptance():
    lim = -SP._SPLIT_R_TOL * 1e-9
    # the number rule (two hot, one cold) and the CP rule (one hot, two
    # cold): lambda = 1 and every must ticks off, exactly
    for dT, rows in (TWO_HOT_ONE_COLD, ONE_HOT_TWO_COLD, TICK_OFF):
        side = sides_from(rows, dT)['above']
        z_m, z_f = [0.] * side.M, [0.] * side.F
        blk = SP._pinch_block(side, z_m, z_f)
        check_pinch_block(side, blk, z_m, z_f)
        assert blk.end[0] == side.Qm   # tick-off: exactly the must ends
        assert len(blk.cells) == 2 and blk.leak == {}
        assert SP._pinch_block(side, z_m, z_f, forbid={(0, 0)}) is None
        assert SP._pinch_block(side, z_m, z_f, used={(0, 0)}) is None
        # no pinch block where the rules do not fail outward
        assert SP._pinch_block(side, z_m, z_f, viol=dict(
            rule='inward', level=0., cut='+')) is None
    # lambda < 1: at lambda = 1 the flex heat left above level 40/7 misses
    # m2 (levels 5 to 7), so (R) fails; the largest lambda is 0.875
    side = pinch_side((5., 5., 7.), 3.5, 70.)
    z_m, z_f = [0.] * 3, [0.]
    assert SP._exact(side).analyse([10., 10., 0.], [20.]).slack < lim
    blk = SP._pinch_block(side, z_m, z_f)
    check_pinch_block(side, blk, z_m, z_f)
    lam = blk.end[0][0] / 10.
    assert blk.end[0][1] == blk.end[0][0] and blk.end[0][2] == 0.
    assert abs(lam - 0.875) <= 1e-8
    assert [(c.f, c.g) for c in blk.cells] == [(1., .5), (1., .5)]
    # below _SPLIT_LAMBDA_MIN: m2 starts 1e-6 above the cut, so any lambda
    # above 2.5e-7 strands it; no block, and LV falls back to V there
    side = pinch_side((5., 1e-6, 2.), 5., 100.)
    assert side.rules_violation(z_m, z_f, side.analyse(z_m, z_f))[
        'rule'] == 'outward'
    assert SP._pinch_block(side, z_m, z_f) is None
    cand = SP._drive(side, z_m, 'LV')
    verify_core(side, cand, z_m)
    assert {blk.kind for blk in cand.blocks} == {'vertical'}


def test_tail_sweeps_the_residual():
    # the DFS closes must 0 on flex 0, which is 5e-10 (tolQ/2) short: the
    # tail sweeps the residual into the cell, so the must is served exactly
    L = P._LevelCurve
    side = P._Side('above', [L([0., 10.], [0., 10.], 0)],
                   [L([0., 10. - 5e-10], [0., 5.], 1),
                    L([0., 10.], [5., 15.], 2)], 1e-9, 1e-9)
    blk, work = SP._tail(side, [0.], [0., 0.])
    assert work > 0. and blk.kind == 'tail' and blk.start == ([0.], [0., 0.])
    (c,) = blk.cells
    assert (c.i, c.j, c.x, c.a, c.b, c.f, c.g) == (0, 0, 10., 0., 0., 1., 1.)
    assert SP._cell_margin(side, c)[0] >= 0.
    assert blk.end == ([10.], [10. - 5e-10, 0.])
    # a tail from a node where the search fails returns None
    blk, work = SP._tail(side, [0.], [10. - 5e-10, 0.])
    assert blk is None and work > 0.


CLEAN_PRELEAK = (10., [  # C1 ends 5e-8 K above H1 (12.5 tolQ): _cascade sets
    # Qh = 0, the rules fail at the root and hold at the pre-leaked root
    ('H1', 'h', 200, 100, 1.), ('H2', 'h', 150, 100, 2.),
    ('C1', 'c', 90, 190 + 5e-8, 1.), ('C2', 'c', 90, 140, 2.)])


def core_problem_sides():
    """Split sides of the SPLIT dict, the merged-breakpoint case, five
    constant-CP corpus cases and the near-threshold, near-double-pinch and
    clean pre-leaked cases (pre-leaked roots)."""
    sides = split_sides()
    for name in ('8sp1_dt20F', 'smith2005_exr18_5_nine_stream',
                 'crude_fractionation_ph11c2', 'fs_22sp_ph',
                 'cgm_unbalanced10'):
        dT, rows = corpus_problem(name)
        sides += [s for s in sides_from(rows, dT).values() if s.M and s.F]
    for dT, rows in (NEAR_THRESHOLD, NEAR_DOUBLE_PINCH, CLEAN_PRELEAK):
        sides += [s for s in sides_from(rows, dT).values() if s.M and s.F]
    return sides


def test_core_strategies_reach_mer():
    # every core strategy is MER and exact from the pre-leaked root: pinch
    # blocks (L) only where the rules fail outward, at most M + F of them;
    # a DFS tail (T) only at a rule-clean node other than the unleaked
    # root, and always last
    rng = random.Random(72)
    sides = core_problem_sides() + core_sides(rng)[::6]
    size = Counter()
    n_pinch = n_tail = n_part = n_root_tail = 0
    for side in sides:
        delta, a0 = SP._preleak_root(side)
        if delta > SP._preleak_max(side):
            continue
        cands = {}
        for strategy in SP._CORE_STRATEGIES:
            cand = cands[strategy] = SP._drive(side, a0, strategy)
            verify_core(side, cand, a0)
            assert cand.name == strategy and cand.meta['leak'] == 0.
            kinds = [blk.kind for blk in cand.blocks]
            assert 'pinch' not in kinds or 'L' in strategy
            assert 'tail' not in kinds[:-1]
            assert 'tail' not in kinds or 'T' in strategy
            assert kinds.count('pinch') <= side.M + side.F
            for blk in cand.blocks:
                a, b = blk.start
                v = side.rules_violation(a, b, side.analyse(a, b))
                if blk.kind == 'pinch':
                    check_pinch_block(side, blk, a, b)
                    # a must that stops short of its end
                    n_part += any(0. < x - a0x < q - a0x - side.tolQ
                                 for x, a0x, q in zip(blk.end[0], a,
                                                      side.Qm))
                elif blk.kind == 'tail':
                    assert v is None and (blk is not cand.blocks[0]
                                          or any(a0))
                    assert all(c.f == c.g == 1. and c.km is c.kf is None
                               for c in blk.cells)
                    n_root_tail += blk is cand.blocks[0]
            n_pinch += 'pinch' in kinds
            n_tail += 'tail' in kinds
        for s, c in cands.items():
            size[s] += c.key()[3]
        # where no strategy builds a pinch block or a tail, all are V
        if all(blk.kind == 'vertical' for c in cands.values()
               for blk in c.blocks):
            assert len({c.signature for c in cands.values()}) == 1
    assert n_pinch >= 20 and n_tail >= 20 and n_part >= 10
    assert n_root_tail >= 1
    # pinch blocks and tails shrink the networks (units + extra branches
    # + split stages, summed over the sides)
    assert size['LVT'] < size['LV'] < size['V'] and size['VT'] < size['V']
