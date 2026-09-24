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

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hensmith import _planner as P
from hensmith import _splitting as SP


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
    from the stage start; every must served exactly from `a0` (plus its
    leak) and every flex used as a prefix."""
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
        assert P._max_duty(bm, am, bf, bb, c.x, tolP) >= c.x - tolQ
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
                start = min(p[0] for p in pos)
                if key[0] != 'trunk':
                    branches = {}
                    for c, p in zip(cs, pos):
                        branches.setdefault((c.km if must else c.kf)[-1],
                                            []).append(p)
                    fr = [ps[0][2] for ps in branches.values()]
                    assert abs(math.fsum(fr) - 1.) <= 1e-12
                    for ps in branches.values():
                        ps.sort()
                        assert all(p[2] == ps[0][2] for p in ps)
                        assert abs(ps[0][0] - start) <= tolQ
                        for p, p2 in zip(ps, ps[1:]):
                            assert abs(p2[0] - p[1]) <= tolQ
                spans.append((start, math.fsum(c.x for c in cs)))
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
    for s in split_sides():
        cand = SP._drive(s, [0.] * s.M, 'V', cap1=True)
        if cand is not None:
            pairs = [(c.i, c.j) for c in SP._merge_cells(cand.cells, s.tolQ)]
            assert len(pairs) == len(set(pairs))
            _verify_side_cells(s, cand.cells)


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
        return SP._Candidate(name, order, side, [0., 0.], cells, **kw)
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
    # touch counts only on a side with a curved stream
    par = [B(0, 0, 5., 0., 0., .5, .5)]   # parallel at the minimum approach
    side = P._Side('above', [P._LevelCurve([0., 20.], [0., 10.])],
                   [P._LevelCurve([0., 20.], [0., 10.])], 1e-9, 1e-9)
    assert SP._Candidate('V', (1, 0), side, [0.], par).touch == 0
    side.flexes.append(P._LevelCurve([0., 5., 20.], [-9., -8., -1.]))
    side = P._Side('above', side.musts, side.flexes, 1e-9, 1e-9)
    assert SP._Candidate('V', (1, 0), side, [0.], par).touch == 1
    # a cell that fails (C) is never accepted
    with pytest.raises(SP._SplitInvariantError):
        SP._Candidate('V', (1, 0), side, [0.], [B(0, 0, 5., 0., 1.)])
