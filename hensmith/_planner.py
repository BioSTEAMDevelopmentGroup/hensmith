# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Planner for heat exchanger networks at minimum energy requirement (MER):
without stream splits by default, and with them, where a side of the pinch
needs them, if `stream_splitting` (see "Stream splitting (optional)"
below).

The planner works on numbers only (numpy; no BioSTEAM objects). The caller,
`hensmith.hxn_synthesis.synthesize_network`, describes each process stream
by a piecewise-linear temperature-enthalpy curve. :func:`plan_network`
returns an ordered list of process-to-process matches, the enthalpy at which
each stream enters and leaves each match, and the utility duty left on each
stream. :func:`_plan_numeric` is a front end for constant heat capacity
problems given as ``{name, kind, T_in, T_out, CP}`` dicts. It returns the
network in the certificate format of the scratch oracle used in the tests.

The model
---------
**Stream curves.** Stream ``j`` is a list of knots ``(T, H)``. Both
coordinates are non-decreasing, and ``H[0]`` and ``H[-1]`` are the ends. A
hot stream runs from ``H[-1]`` down to ``H[0]``; a cold stream runs from
``H[0]`` up to ``H[-1]``. Two consecutive knots with the same temperature and
different enthalpies make a *flat*: isothermal latent heat, a point load, or
a clipped end jump. Two consecutive knots with the same enthalpy and
different temperatures (a vertical stretch at a clipped non-equilibrium end)
are collapsed to the conservative temperature: the lowest for a hot stream,
the highest for a cold one. Interior knots within ``_KNOT_TOL`` K of the
chord through their neighbours are dropped, so a constant-CP stream becomes
a single segment and the fast paths apply. On the *shifted* scale ``T*``,
hot streams are moved down by ``T_min_app`` and cold streams are not moved.
Shifted temperatures closer than ``_LEVEL_EQ`` K are merged, exactly as the
problem table merges its grid.

**Targets.** The planner builds its own problem table [2]_ from these curves.
For every shifted level ``L`` it computes the heat flow arriving at ``L``
(before the point loads at ``L``) and the heat flow leaving it (after
them). The pinch is the first (hottest) minimum of these flows; when no hot
utility is needed it is the top of the grid. ``cut`` records which side of
the pinch the point loads *at* the pinch temperature belong to. When the
knots come from the problem-table grid, this cascade equals the problem
table.

**Sides; must and flex streams.** The network is cut at the pinch. On each
side, heat is measured from each stream's pinch end, ``q >= 0``, and every
stream gets a non-decreasing *level curve* ``phi(q)``:

====== ====== ===== =======================
side   stream role  level ``y``
====== ====== ===== =======================
above  hot    must  ``T - T_min_app``
above  cold   flex  ``T``
below  cold   must  ``-T``
below  hot    flex  ``-(T - T_min_app)``
====== ====== ===== =======================

A *must* stream has to be served completely by process matches on its side:
no cold utility above the pinch, no hot utility below it. A *flex* stream
puts whatever is left over into one utility at its far end. That placement
loses nothing, by lemma L1 below. A counter-current match of duty ``x``
between must ``i`` from frontier ``a`` and flex ``j`` from frontier ``b`` is
feasible if and only if ``phi_i(a + t) >= phi_j(b + t)`` for all ``t`` in
``[0, x]``. The planner checks this at every knot, so it also catches
internal pinches, such as a condensing vapour against a boiling mixture.
Both sides are therefore the same problem, and one search solves both.

**Lemma L1 (monotonicity).** Suppose a stream's later stages move toward its
inlet: they become hotter on a hot stream or colder on a cold one. Then no
match on that stream loses approach temperature. Every level curve is
non-decreasing, so shifting a must's matches away from the pinch raises its
levels, and shifting a flex's matches toward the pinch lowers its levels.
Four consequences follow:

(a) Utilities can sit at the far end of each stream without loss of
generality.
(b) Heat a must leaves uncovered (a *gap*) can be moved to its pinch end.
There it crosses the pinch and costs an equal extra amount of hot and of
cold utility, the *penalty*.
(c) Deleting a match keeps every other match feasible. This is what the
``Qmin`` filter and the final safety net rely on.
(d) Feasibility is monotone in the gaps, so they can be bisected.

**Residual condition (R).** At a search node with frontiers ``(a, b)``, take
any level ``L``. Let ``S(L)`` be the flex heat beyond the frontiers at
levels ``<= L`` (inclusive) or ``< L`` (exclusive), and ``D(L)`` the same
for must heat. Condition (R) is ``s(L) = S(L) - D(L) >= 0`` everywhere. It
is the problem table of the remaining problem [3]_ with splits allowed, so
any completion needs it. Advance the pair ``(i, j)`` by ``x``, and write
``Rf_j`` and ``Rm_i`` for the residuals of flex ``j`` and must ``i``. Then
``s`` becomes ``s - min(x, Rf_j) + min(x, Rm_i)``. Along a feasible match
``Rf_j >= Rm_i``, so this value never increases with ``x``. The largest
duty that keeps (R) therefore has a closed form::

    x_res(i, j) = min{ s(L) + Rm_i(L) : s(L) - Rf_j(L) + Rm_i(L) < 0 }

It is evaluated at every level point, inclusive and exclusive, and at the
crossings inside linear pieces.

**Pinch rules at tight levels.** A level where ``s = 0`` is a pinch of the
remaining problem. Two rules must then hold, each tested as a bipartite
matching that saturates the streams that need a partner:

- *Outward:* every must that reaches the cut from below and still has heat
  above it needs its own flex at the cut whose level rises no faster.
  Level slope is ``1/CP``, so this is ``CP_must <= CP_flex``.
- *Inward:* every flex whose residual reaches the cut from below needs its
  own must reaching the cut from below.

A flat piece *at* the cut has unlimited series capacity, so it may serve
several partners. A flat piece that needs a partner accepts only a flat
one. At the root, these rules are the pinch design method's [1]_ proof
that MER needs stream splitting. For constant CP they reduce to the number and
CP rules at every cascade zero.

**Search.** A depth-first search on an explicit stack builds each side
from the pinch outward. A node is the vector of frontiers. A move advances
one must and one flex by the same duty. The candidate duties are:

- ``x_top = min(x_dT, x_res, rem_i, cap_j)``, where ``x_dT`` is the
  approach limit;
- the *events* listed in :meth:`_Search._duties`.

Each event is a vertex type of the duty polytope for a fixed match order:
a stream is ticked off, a dT limit is reached, a level becomes R-tight, or
a later match along one link, one block of tied musts, or one return must
be able to start. Chains of partial services ("coupled vertices") are not
enumerated. Repeated (must, flex) pairs are allowed: series alternation
emulates a split (the unsplit search's only way to approach one; with
`stream_splitting`, a side that needs a split gets real ones instead).
Early passes cap the number of new exchangers per pair.

Budgets are counted in deterministic *work units*: one per node plus a
share proportional to the residual arrays. No wall clock is used, so the
result does not depend on machine speed. Nodes that fail are memoised.
The memo key holds the quantised frontiers, the pairs at their cap, and
the open continuations. A failure caused by the depth cap is never
memoised.

**Units.** After the first MER plan is found, a branch and bound on the
number of exchangers (the same search, with a unit bound) looks for a
smaller network. Continuing the current exchanger costs no extra unit. If a
side still has more than ``3 (M + F)`` units, a coarse-to-fine rerun with a
minimum piece size keeps the MER plan with the fewest units.

**Best effort.** This runs on a side that has a root proof or whose search
ran out of budget (with `stream_splitting`, only if the side's split
attempt finds no candidate, which Theorem M of `hensmith._splitting` rules
out without `avoid_recycle`):

1. Twelve greedy dives give an incumbent. Each dive leaks must heat when no
   partner fits.
2. Gap bisection follows, using the MER search as a monotone oracle
   (consequence (d) of L1).
3. The plan with the lowest penalty is chosen among those within a units
   cap.

**Realisation.** Each stream is walked in flow order from its inlet:

- *Hot stream:* above-pinch matches far end first, then below-pinch matches
  from the pinch, then the cooler.
- *Cold stream:* below-pinch matches far end first, then above-pinch
  matches, then the heater.

Gaps and dropped matches shift the later stages. By L1 that is feasible.
Each exchanger's approach is then checked on the curves at every knot. Any
match that violates it would be a bug; such a match is dropped and
reported. The plan says ``'mer'`` only if the utilities it achieves equal
the targets.

Guarantees and limits
---------------------
- The utilities are never below the targets. Every planned exchanger keeps
  ``T_min_app`` on the knot curves at its ends and at every knot inside it.
  The heat balance closes on every stream, and ``'mer'`` is never claimed
  falsely. Results are deterministic.
- Every pruning test is a necessary condition. A missed MER network can
  therefore only come from the finite set of candidate duties, the pair
  caps, or the budgets. In that sense completeness is empirical, not
  proven. It was validated on a certified benchmark and on fresh problems.
- Networks whose match order is cyclic cannot be represented.
- Some unsplit MER networks need arbitrarily many exchangers. Series
  alternation approaches a split only in the limit (unless
  `stream_splitting`: see below).
- A side that needs splits but has no rules proof spends its whole MER
  budget before best effort (or, with `stream_splitting`, the split
  attempt) starts.
- With `stream_splitting` and without `avoid_recycle`, every side that no
  unsplit plan serves gets a split plan at MER on the knots (Theorem M of
  `hensmith._splitting`), so the plan's status is 'mer'.

Stream splitting (optional)
---------------------------
With ``stream_splitting=True``, :func:`plan_network` hands every side that
has a root proof, or whose best effort leaves a penalty, to
`hensmith._splitting._split_side`. Sides the unsplit search serves are
planned exactly as without the option, so a problem that needs no split
gets the same plan. A split side is planned as verified *cells* in parent
coordinates (a branch of flow fraction ``f`` is the parent's curve with
every heat times ``f``), by a portfolio of generators: Stage S (splits at
the pinch, one candidate per transport rule, completed by the unsplit
search) and the core strategies V, LV, VT and LVT (vertical blocks, pinch
blocks and tails), of which V always yields an MER candidate (without
`avoid_recycle`). The candidate with the smallest key (feasibility
markers, then units plus extra branches plus split stages) wins, and
`_split_records` turns its cells into the plan's records: every split
stage becomes a `hensmith._splitting.Split` in ``Plan.splits``, a branch
exchanger carries its fractions (``Exchanger.hot_frac``/``cold_frac``)
and parent-equivalent enthalpies, and ``Plan.paths`` lists each stream's
flow order with a split as one item (``Plan.stages`` flattened). The side's
``info['sides'][side]['split']`` reports the candidates and the pick. The
module docstring of `hensmith._splitting` holds the theory: the lemmas,
Theorems V' and M, the generators, the key and the signature.

References
----------
.. [1] Linnhoff, B.; Hindmarsh, E. The Pinch Design Method for Heat
   Exchanger Networks. Chem. Eng. Sci. 1983, 38 (5), 745-763.
.. [2] Kemp, I. C. Pinch Analysis and Process Integration, 2nd ed.;
   Butterworth-Heinemann, 2007.
.. [3] Smith, R. Chemical Process Design and Integration; Wiley, 2005.

"""
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np

__all__ = ()

# %% Tolerances and constants

_REL_Q = 1e-11          # heat tolerance, relative to the total stream duty
_REL_T = 1e-11          # level tolerance, relative to the temperature span
_LEVEL_EQ = 1e-9        # K; shifted temperatures closer than this are merged
_KNOT_TOL = 1e-9        # K; collinear interior knots are dropped (< 0: keep)
_MER_TOL = 1e-9         # status 'mer' tolerance, relative to the total duty
_THRESHOLD_TOL = 1e-9   # Qh below this (relative) is a threshold problem
_APPROACH_TOL = 1e-6    # K; safety-net approach tolerance
_SLOPE_EQ = 1e-9        # relative; slopes this close count as equal
_W0 = 2000.             # residual-array size worth one work unit
_WORK_XRES = 0.2        # work of x_res and x_dT for one must (+ array share)
_WORK_EVENT = 0.05      # work of one event duty (piecewise curves)
_WORK_EVENTS_VEC = 0.5  # work of all event duties of a pair (constant CP)

#: MER passes: (mode, pair cap, extra events E5/E6, work units).
_SCHEDULE = (
    ('restricted', 2, False, 3000.),
    ('restricted', None, False, 300.),
    ('restricted', 3, False, 3000.),
    ('full', 3, False, 10000.),
    ('restricted', None, True, 5000.),
    ('full', None, True, 30000.),
)
_DEPTH_FACTOR = 8       # depth cap = _DEPTH_FACTOR * (M + F) + _DEPTH_EXTRA
_DEPTH_EXTRA = 50
_UNITS_WORK_MIN = 300.  # unit branch and bound: work per step ...
_UNITS_WORK_MAX = 3000.
_UNITS_WORK_TOTAL = 10000.  # ... and in total
_GUARD_FACTOR = 3       # units guard when a side has > 3 (M + F) units
_GUARD_ZENOS = (1e-2, 1e-3)
_GUARD_WORK = 3000.
_BE_ORDERS = ('tick', 'bestfit', 'maxduty', 'lowfit')
_BE_DIVE_ZENO = 1e-2    # dives: no non-tick-off piece below this * side duty
_BE_MIN_DUTY = 1e-7     # dives: no piece below this * side duty
_BE_REPEAT_CAP = 2      # gap oracle: pair cap
_BE_MIN_PIECE_FRAC = 0.01   # gap oracle: min piece / smaller stream's duty
_BE_ORACLE_WORK = 1500.
#: gap-oracle passes (mode, pair cap, share of the call's work): passes 1, 3
#: and 2 of `_SCHEDULE` (the unit bound already limits alternation); a
#: capped pass that runs out of work must not starve the next one
_BE_ORACLE_PASSES = (('restricted', _BE_REPEAT_CAP, 0.25),
                     ('restricted', _BE_REPEAT_CAP + 1, 0.25),
                     ('restricted', None, 0.5))
_BE_SINGLE = True       # also bisect single-must leaks (d2's leak search)
_BE_ROUNDS = 14
_BE_WORK = 30000.
_BE_UNITS_CAP = 1.5     # selection: units cap = ceil(1.5 (M + F))

_DONE, _FAIL, _DEPTH = 'done', 'fail', 'depth'


# %% Vectorised curve helpers

def _x_le_many(y, q, L):
    """``sup{q : y(q) <= L}`` for every level in `L` (``q[0]`` where
    ``L < y[0]``); `y` non-decreasing, `q` strictly increasing."""
    L = np.atleast_1d(np.asarray(L, float))
    n = y.size
    i = np.searchsorted(y, L, 'right') - 1
    out = np.where(i < 0, q[0], q[-1])
    mid = (i >= 0) & (i < n - 1)
    if mid.any():
        k = i[mid]
        out[mid] = q[k] + (q[k + 1] - q[k]) * (L[mid] - y[k]) / (
            y[k + 1] - y[k])
    return out


def _x_lt_many(y, q, L):
    """``sup{q : y(q) < L}`` for every level in `L` (``q[0]`` where
    ``L <= y[0]``)."""
    L = np.atleast_1d(np.asarray(L, float))
    n = y.size
    j = np.searchsorted(y, L, 'left')
    out = np.where(j <= 0, q[0], q[-1])
    mid = (j > 0) & (j < n)
    if mid.any():
        k = j[mid]
        out[mid] = q[k - 1] + (q[k] - q[k - 1]) * (L[mid] - y[k - 1]) / (
            y[k] - y[k - 1])
    return out


def _x_le1(y, q, L):
    """Scalar ``sup{q : y(q) <= L}`` on python lists."""
    i = bisect_right(y, L) - 1
    if i < 0:
        return q[0]
    if i >= len(y) - 1:
        return q[-1]
    return q[i] + (q[i + 1] - q[i]) * (L - y[i]) / (y[i + 1] - y[i])


def _x_lt1(y, q, L):
    """Scalar ``sup{q : y(q) < L}`` on python lists."""
    j = bisect_left(y, L)
    if j <= 0:
        return q[0]
    if j >= len(y):
        return q[-1]
    return q[j - 1] + (q[j] - q[j - 1]) * (L - y[j - 1]) / (y[j] - y[j - 1])


# %% Level curves

class _LevelCurve:
    """
    Non-decreasing piecewise-linear level curve ``y = phi(q)`` on
    ``[0, Q]`` of one stream on one side of the pinch.

    Parameters
    ----------
    q : sequence[float]
        Heat from the stream's pinch end, starting at 0.
    y : sequence[float]
        Levels on the shifted scale (see the module docstring).
    stream : int, optional
        Index of the stream.
    H0 : float, optional
        Enthalpy at ``q = 0`` (the split enthalpy).
    sgn : {1, -1}, optional
        ``H = H0 + sgn * q`` (+1 above the pinch, -1 below).
    role : {'must', 'flex'}, optional
        Which way to collapse vertical stretches (equal `q`, different
        `y`): a must keeps the lowest level, a flex the highest (the
        conservative choices).

    Notes
    -----
    Zero-length pieces are removed; slopes are ``1/CP`` and exactly zero on
    flats (equal consecutive levels). Scalar methods use python lists and
    bisection, the ``*_many`` methods numpy.
    """
    __slots__ = ('q', 'y', 'qa', 'ya', 'n', 'Q', 'linear', 'stream', 'H0',
                 'sgn', 'flats')

    def __init__(self, q, y, stream=-1, H0=0., sgn=1, role='must'):
        Q2, Y2 = [], []
        for x, v in zip(q, y):
            x, v = float(x), float(v)
            if Q2 and x <= Q2[-1]:
                if role == 'flex':
                    Y2[-1] = max(Y2[-1], v)
                continue
            if Y2 and v < Y2[-1]:
                v = Y2[-1]
            Q2.append(x)
            Y2.append(v)
        if len(Q2) == 1:
            Q2.append(Q2[0])
            Y2.append(Y2[0])
        self.q, self.y = Q2, Y2
        self.qa, self.ya = np.array(Q2), np.array(Y2)
        self.n = len(Q2)
        self.Q = Q2[-1]
        self.linear = self.n == 2
        self.stream, self.H0, self.sgn = stream, H0, sgn
        self.flats = tuple(Y2[k] for k in range(self.n - 1)
                           if Y2[k + 1] == Y2[k] and Q2[k + 1] > Q2[k])

    def __repr__(self):
        return f'<_LevelCurve stream={self.stream} Q={self.Q:.6g} n={self.n}>'

    def at(self, x):
        """Level at heat position `x` (clipped to the ends)."""
        q, y = self.q, self.y
        if x <= q[0]:
            return y[0]
        if x >= self.Q:
            return y[-1]
        i = bisect_right(q, x) - 1
        return y[i] + (y[i + 1] - y[i]) * (x - q[i]) / (q[i + 1] - q[i])

    def at_many(self, xs):
        return np.interp(xs, self.qa, self.ya)

    def x_le(self, L):
        """``sup{q : phi(q) <= L}``, 0 if ``phi(0) > L``."""
        return _x_le1(self.y, self.q, L)

    def x_lt(self, L):
        """``sup{q : phi(q) < L}``, 0 if ``phi(0) >= L``."""
        return _x_lt1(self.y, self.q, L)

    def x_le_many(self, L):
        return _x_le_many(self.ya, self.qa, L)

    def x_lt_many(self, L):
        return _x_lt_many(self.ya, self.qa, L)

    def slope_right(self, x):
        """Slope of the piece starting at or containing `x` (0 on flats)."""
        q, y = self.q, self.y
        i = min(max(bisect_right(q, x) - 1, 0), self.n - 2)
        if q[i + 1] <= q[i]:
            return 0.
        return (y[i + 1] - y[i]) / (q[i + 1] - q[i])

    def slope_left(self, x):
        """Slope of the piece ending at or containing `x` (0 on flats)."""
        q, y = self.q, self.y
        i = min(max(bisect_left(q, x), 1), self.n - 1)
        if q[i] <= q[i - 1]:
            return 0.
        return (y[i] - y[i - 1]) / (q[i] - q[i - 1])

    def window(self, lo, hi):
        """Knot positions strictly inside ``(lo, hi)``."""
        q = self.q
        return q[bisect_right(q, lo):bisect_left(q, hi)]


def _slope1(c):
    return (c.y[1] - c.y[0]) / (c.q[1] - c.q[0]) if c.q[1] > c.q[0] else 0.


# %% Match geometry (all exact on piecewise-linear curves)

def _max_duty(cm, a, cf, b, limit, tolP):
    """
    x_dT: largest ``x <= limit`` with ``phi_m(a + t) - phi_f(b + t) >=
    -tolP`` on ``[0, x]``. The gap is linear between the merged knots of both
    curves, so it is checked at each of them; a violation is interpolated to
    the zero crossing (conservative).
    """
    if limit <= 0.:
        return 0.
    g0 = cm.at(a) - cf.at(b)
    if g0 < -tolP:
        return 0.
    if cm.linear and cf.linear:
        sm, sf = _slope1(cm), _slope1(cf)
        # slopes equal to 1e-9 must not be separated by rounding
        if sm >= sf * (1. - _SLOPE_EQ):
            return limit
        return min(limit, max(g0, 0.) / (sf - sm))
    wm = cm.window(a, a + limit)
    wf = cf.window(b, b + limit)
    if len(wm) + len(wf) <= 8:
        ts = sorted({limit, *(x - a for x in wm), *(x - b for x in wf)})
        gp, tp = max(g0, 0.), 0.
        for t in ts:
            if t <= 0.:
                continue
            g = cm.at(a + t) - cf.at(b + t)
            if g < -tolP:
                if gp <= 0.:
                    return tp
                return tp + (t - tp) * gp / (gp - g)
            tp, gp = t, max(g, 0.)
        return limit
    ts = np.unique(np.concatenate(([0., limit], np.asarray(wm) - a,
                                   np.asarray(wf) - b)))
    ts = ts[(ts >= 0.) & (ts <= limit)]
    g = cm.at_many(a + ts) - cf.at_many(b + ts)
    bad = np.flatnonzero(g < -tolP)
    if bad.size == 0:
        return limit
    k = int(bad[0])
    if k == 0:
        return 0.
    gp = max(float(g[k - 1]), 0.)
    if gp <= 0.:
        return float(ts[k - 1])
    return float(ts[k - 1] + (ts[k] - ts[k - 1]) * gp / (gp - g[k]))


def _return_duty(cm, a, cf, b, R, xt):
    """
    E3: the first ``x`` in ``(0, xt)`` with ``phi_m(a + x) = phi_f(b + x +
    R)``. Serving must m by x and then other musts with R of the flex's heat
    brings the flex exactly back to m's new level, so m can resume on it.
    ``h(x) = phi_m(a + x) - phi_f(b + x + R)`` is continuous and piecewise
    linear; its first sign change from negative is found by scanning the
    merged knots (None if there is none).
    """
    if b + R >= cf.Q:
        return None
    h0 = cm.at(a) - cf.at(b + R)
    if h0 >= 0.:
        return None
    if cm.linear and cf.linear:
        sm, sf = _slope1(cm), _slope1(cf)
        if sm <= sf * (1. + 1e-12):
            return None
        return -h0 / (sm - sf)
    c0 = b + R
    ts = sorted({xt, *(x - a for x in cm.window(a, a + xt)),
                 *(x - c0 for x in cf.window(c0, c0 + xt))})
    hp, tp = h0, 0.
    for t in ts:
        if t <= 0.:
            continue
        h = cm.at(a + t) - cf.at(c0 + t)
        if h >= 0.:
            return tp + (t - tp) * (-hp) / (h - hp)
        tp, hp = t, h
    return None


def _flex_stop_for(ck, ak, remk, cf, bj, capj):
    """
    E1b: the largest flex advance ``x`` after which flex ``j`` can still
    serve the whole remainder of another must ``k`` in one match.

    After the flex advanced by x the match is feasible iff ``b_j + x + t <=
    X_le_j(phi_k(a_k + t))`` for t in [0, rem_k], i.e. ``x <= psi(t) =
    X_le_j(phi_k(a_k + t)) - b_j - t``; also ``x <= cap_j - rem_k``. psi is
    piecewise linear with breakpoints at k's knots and where phi_k crosses
    j's knot levels; where phi_k rises strictly into a flat level of j the
    left limit ``X_lt_j`` applies. Returns ``min(cap_j - rem_k, min psi)``
    (None if ``cap_j < rem_k``).
    """
    if capj < remk:
        return None
    xcap = capj - remk
    y0 = ck.at(ak)
    y1 = ck.at(ak + remk)
    if ck.linear and cf.linear:
        sk, sj = _slope1(ck), _slope1(cf)
        if sj > 0.:
            ds = sj - sk
            if ds <= _SLOPE_EQ * sj:
                ds = 0.
            return min(cf.x_le(y0 - remk * ds) - bj, xcap)
    ts = {0., remk}
    ts.update(x - ak for x in ck.window(ak, ak + remk))
    exact = {}   # points where phi_k crosses a knot level of j (see E2b)
    for L in cf.y:
        if y0 <= L <= y1:
            for x in (ck.x_le(L) - ak, ck.x_lt(L) - ak):
                if 0. < x < remk:
                    ts.add(x)
                    exact[x] = L
    ts = sorted(ts)
    phi = ck.at_many(ak + np.array(ts))
    for k, x in enumerate(ts):
        if x in exact:
            phi[k] = exact[x]
    ts = np.array(ts)
    m = float((cf.x_le_many(phi) - bj - ts).min())
    for L in cf.flats:
        if y0 < L <= y1:
            tf = ck.x_lt(L) - ak
            if 0. < tf <= remk:
                m = min(m, cf.x_lt(L) - bj - tf)
    return min(xcap, m)


def _must_stop_for(ci, ai, remi, cg, bg, capg, xt):
    """
    E2b: the smallest must advance ``x`` in ``(0, xt)`` after which flex
    ``g`` can finish must ``i`` in one match.

    With ``s = x + t`` the finishing match is feasible iff ``x >= h(s) = b_g
    + s - X_le_g(phi_i(a_i + s))`` for every s in [x, rem_i] (left limits
    with ``X_lt_g`` where phi_i rises into a flat of g) and ``rem_i - x <=
    cap_g``. h is piecewise linear; scanning its pieces in ascending order
    with the suffix maximum SM of h beyond each piece, the smallest x on a
    piece [s_k, s_k+1] satisfies ``x >= SM_k+1`` and ``x (1 - sigma) >= h_k -
    sigma s_k`` (sigma = the slope of h there). O(knots); None if no such x.
    """
    lo_x = max(0., remi - capg)
    if lo_x >= xt:
        return None
    y0 = ci.at(ai)
    y1 = ci.at(ai + remi)
    if ci.linear and cg.linear:
        # the gap of the finishing match is linear: only its two ends bind,
        # y0 + si x >= mu (start) and y0 + si remi >= mu + sg (remi - x)
        si, sg = _slope1(ci), _slope1(cg)
        mu = cg.at(bg)
        D = mu - y0
        if D > 0.:
            if si <= 0.:
                return None
            lo_x = max(lo_x, ci.x_lt(mu) - ai)
        if sg > si * (1. + _SLOPE_EQ):
            lo_x = max(lo_x, remi - (y1 - mu) / sg)
        return lo_x
    ss = {0., remi}
    ss.update(x - ai for x in ci.window(ai, ai + remi))
    exact = {}   # breakpoints where phi_i crosses a knot level of g
    for L in cg.y:
        if y0 <= L <= y1:
            for x in (ci.x_le(L) - ai, ci.x_lt(L) - ai):
                if 0. < x < remi:
                    ss.add(x)
                    exact[x] = L
    s = sorted(ss)
    K = len(s)
    phi = ci.at_many(ai + np.array(s))
    # use the exact level there: an interpolated level one ulp above a flat
    # of g would lose the jump of X_le_g at the flat (its left limit)
    for k, x in enumerate(s):
        if x in exact:
            phi[k] = exact[x]
    hR = bg + np.array(s) - cg.x_le_many(phi)
    hL = hR.copy()
    for k in range(1, K):
        if ci.slope_left(ai + s[k]) > 0.:
            hL[k] = bg + s[k] - cg.x_lt(float(phi[k]))
    SM = [0.] * (K + 1)
    SM[K] = -math.inf
    for k in range(K - 1, -1, -1):
        SM[k] = max(SM[k + 1], hL[k], hR[k])
    for k in range(K - 1):
        sk, sk1 = s[k], s[k + 1]
        lo = max(sk, SM[k + 1], lo_x)
        if lo > sk1:
            continue
        sigma = (hL[k + 1] - hR[k]) / (sk1 - sk)
        d = lo - (hR[k] + sigma * (lo - sk))
        if d >= 0.:
            return lo
        if 1. - sigma > 0.:
            x = lo - d / (1. - sigma)
            if x <= sk1:
                return x
    return None


# %% Bipartite matching

def _saturating_matching(left, right, ok, flat_left, flat_right):
    """
    True iff every item of `left` gets its own item of `right` with
    ``ok(l, r)``. A flat item (isothermal piece at the cut) has unlimited
    series capacity: a flat right item can serve any number of left items,
    and a flat left item only needs one compatible partner and consumes
    none.
    """
    owner = {}

    def augment(l, seen):
        for r in right:
            if r in seen or not ok(l, r):
                continue
            if flat_right[r]:
                return True
            seen.add(r)
            if r not in owner or augment(owner[r], seen):
                owner[r] = l
                return True
        return False

    for l in left:
        if flat_left[l]:
            if not any(ok(l, r) for r in right):
                return False
            continue
        if not augment(l, set()):
            return False
    return True


# %% One side of the pinch

class _Residual:
    """Residual arrays at a node (see `_Side.analyse`)."""
    __slots__ = ('levels', 'Rmi', 'Rme', 'Rfi', 'Rfe', 'Si', 'Se', 'Di',
                 'De', 'slack', 'nL')


class _Side:
    """
    One side of the pinch in level-curve form.

    Parameters
    ----------
    name : {'above', 'below'}
    musts, flexes : list[_LevelCurve]
    tolQ, tolP : float
        Heat and level tolerances.
    """
    #: x_dT of a pair, as the search (`_Search`) computes it
    max_duty = staticmethod(_max_duty)

    def __init__(self, name, musts, flexes, tolQ, tolP):
        self.name = name
        self.musts, self.flexes = musts, flexes
        self.M, self.F = len(musts), len(flexes)
        self.Qm = [c.Q for c in musts]
        self.Qf = [c.Q for c in flexes]
        self.tolQ, self.tolP = tolQ, tolP
        self.duty = sum(self.Qm) + sum(self.Qf)
        # constant CP (one sloped segment per curve): vectorised events
        self.sm = np.array([_slope1(c) for c in musts])
        self.sf = np.array([_slope1(c) for c in flexes])
        self.linear = (all(c.linear for c in musts + flexes)
                       and bool((self.sm > 0.).all() and (self.sf > 0.).all()))

    # -- pairs in stream indices --------------------------------------------
    def hot_cold(self, i, j):
        m, f = self.musts[i].stream, self.flexes[j].stream
        return (m, f) if self.name == 'above' else (f, m)

    def local_pairs(self, pairs):
        return frozenset((i, j) for i in range(self.M) for j in range(self.F)
                         if self.hot_cold(i, j) in pairs)

    # -- residual arrays ------------------------------------------------------
    def _pieces(self, curves, front):
        tolQ = self.tolQ
        own, lo, hi, ln = [], [], [], []
        for k, c in enumerate(curves):
            f = front[k]
            if c.Q - f <= tolQ:
                continue
            if c.linear:
                own.append(k)
                lo.append(c.at(f))
                hi.append(c.y[1])
                ln.append(c.Q - f)
                continue
            q, y = c.q, c.y
            xf, yf = f, c.at(f)
            for m in range(bisect_right(q, f), c.n):
                own.append(k)
                lo.append(yf)
                hi.append(y[m])
                ln.append(q[m] - xf)
                xf, yf = q[m], y[m]
        return (np.array(own, dtype=np.intp), np.array(lo), np.array(hi),
                np.array(ln))

    @staticmethod
    def _R(pieces, n, levels):
        """Inclusive (level <= L) and exclusive (level < L) residual heat of
        every stream at every level: one scatter per quantity and a cumulative
        sum along the levels (O(pieces + n * levels))."""
        own, lo, hi, ln = pieces
        nL = levels.size
        if own.size == 0 or nL == 0:
            z = np.zeros((n, nL))
            return z, z.copy()
        if own.size == 1 or (own[1:] > own[:-1]).all():
            # at most one piece per stream (constant CP): direct broadcast
            w = hi - lo
            flat = w <= 0.
            with np.errstate(divide='ignore', invalid='ignore'):
                f = np.clip((levels - lo[:, None]) / w[:, None], 0., 1.)
            if flat.any():
                f[flat] = levels >= lo[flat, None]
                fe = f.copy()
                fe[flat] = levels > lo[flat, None]
            else:
                fe = f
            Ri = np.zeros((n, nL))
            Ri[own] = f * ln[:, None]
            Re = Ri.copy() if fe is f else np.zeros((n, nL))
            if fe is not f:
                Re[own] = fe * ln[:, None]
            return Ri, Re
        W = nL + 1
        size = n * W
        il = np.searchsorted(levels, lo)
        ih = np.searchsorted(levels, hi)
        flat = ih == il
        base = own * W

        def scan(index, weights):
            out = np.bincount(index, weights, size).reshape(n, W)
            return out[:, :nL].cumsum(1)
        Ri = scan(base + ih, ln)
        Re = scan(base + ih + flat, ln)
        nf = ~flat
        if nf.any():
            sl = ln[nf] / (hi[nf] - lo[nf])
            idx = np.concatenate((base[nf] + il[nf], base[nf] + ih[nf]))
            B = scan(idx, np.concatenate((sl, -sl)))
            D = scan(idx, np.concatenate((lo[nf], -lo[nf])))
            part = B * (levels - D)
            Ri += part
            Re += part
        return Ri, Re

    def analyse(self, a, b):
        """Residual data at frontiers ``(a, b)``: the level set (every piece
        end beyond the frontiers), per-stream residual heat, the sums S and D
        and the minimum slack of (R)."""
        pm = self._pieces(self.musts, a)
        pf = self._pieces(self.flexes, b)
        d = _Residual()
        d.levels = L = np.unique(np.concatenate((pm[1], pm[2], pf[1], pf[2])))
        d.nL = L.size
        d.Rmi, d.Rme = self._R(pm, self.M, L)
        d.Rfi, d.Rfe = self._R(pf, self.F, L)
        d.Si, d.Se = d.Rfi.sum(0), d.Rfe.sum(0)
        d.Di, d.De = d.Rmi.sum(0), d.Rme.sum(0)
        d.slack = (min(float((d.Si - d.Di).min()), float((d.Se - d.De).min()))
                   if d.nL else 0.)
        return d

    def xres_for(self, i, d):
        """x_res(i, j) for every flex j (closed form, see the module
        docstring); +inf where unconstrained."""
        tol = self.tolQ
        F = self.F
        if d.nL == 0 or F == 0:
            return np.full(F, np.inf)
        gi = (d.Si - d.Rfi) - (d.Di - d.Rmi[i])
        ge = (d.Se - d.Rfe) - (d.De - d.Rme[i])
        hi = d.Si - d.Di + d.Rmi[i]
        he = d.Se - d.De + d.Rme[i]
        inf = np.inf
        res = np.minimum(np.where(gi < -tol, hi, inf).min(1),
                         np.where(ge < -tol, he, inf).min(1))
        if d.nL > 1:
            gL, gR = gi[:, :-1], ge[:, 1:]
            cross = ((gL < -tol) & (gR >= -tol)) | ((gR < -tol) & (gL >= -tol))
            if cross.any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    th = np.clip(gL / (gL - gR), 0., 1.)
                hc = hi[:-1] + th * (he[1:] - hi[:-1])
                res = np.minimum(res, np.where(cross, hc, inf).min(1))
        return np.maximum(res, 0.)

    # -- pinch rules at tight levels -----------------------------------------
    def tight_cuts(self, d):
        """Cuts at the zeros of the residual cascade: '+' (just above L;
        flats at L lie below it) where the inclusive slack is zero, '-' (just
        below L; flats at L lie above it) where only the exclusive one is."""
        if d.nL == 0:
            return []
        tol = self.tolQ
        bi = d.Si - d.Di
        be = d.Se - d.De
        L = d.levels
        cuts = [(float(L[k]), '+') for k in np.flatnonzero(bi <= tol)]
        cuts += [(float(L[k]), '-') for k in
                 np.flatnonzero((be <= tol) & (np.abs(be - bi) > tol))]
        cuts.sort()
        return cuts

    def rules_violation(self, a, b, d):
        """
        First violation of the flat-aware pinch rules at a tight level, or
        None. See the module docstring (outward and inward rules); a flat
        piece at the cut has unlimited series capacity but accepts only flat
        partners when it needs one.
        """
        tolQ, tolP = self.tolQ, self.tolP
        musts, flexes = self.musts, self.flexes
        open_m = [i for i in range(self.M) if self.Qm[i] - a[i] > tolQ]
        open_f = [j for j in range(self.F) if self.Qf[j] - b[j] > tolQ]
        for L, cut in self.tight_cuts(d):
            above = cut == '-'
            # ---- outward
            out_m, sm, fm = [], {}, {}
            for i in open_m:
                c = musts[i]
                if c.at(a[i]) > L + tolP:
                    continue
                q = max(c.x_lt(L) if above else c.x_le(L), a[i])
                if q < c.Q - tolQ:
                    out_m.append(i)
                    sm[i] = s = c.slope_right(q)
                    fm[i] = s == 0.
            if out_m:
                out_f, sf, ff = [], {}, {}
                for j in open_f:
                    c = flexes[j]
                    if c.at(b[j]) > L + tolP:
                        continue
                    q = max(c.x_lt(L) if above else c.x_le(L), b[j])
                    if q < c.Q - tolQ:
                        out_f.append(j)
                        sf[j] = s = c.slope_right(q)
                        ff[j] = s == 0.
                if not _saturating_matching(
                        out_m, out_f,
                        lambda i, j: sm[i] >= sf[j] * (1. - 1e-12), fm, ff):
                    return dict(side=self.name, level=L, cut=cut,
                                rule='outward',
                                musts=[musts[i].stream for i in out_m],
                                flexes=[flexes[j].stream for j in out_f])
            # ---- inward
            in_f, sf, ff = [], {}, {}
            for j in open_f:
                c = flexes[j]
                yb = c.at(b[j])
                if c.y[-1] < L - tolP:
                    continue
                if above:
                    if yb >= L - tolP:
                        continue
                    q = c.x_lt(L)
                else:
                    if yb > L + tolP:
                        continue
                    q = c.x_le(L)
                if q > b[j] + tolQ:
                    in_f.append(j)
                    sf[j] = s = c.slope_left(q)
                    ff[j] = s == 0.
            if in_f:
                in_m, sm, fm = [], {}, {}
                for i in open_m:
                    c = musts[i]
                    ya = c.at(a[i])
                    if c.y[-1] < L - tolP:
                        continue
                    if above:
                        if ya >= L - tolP:
                            continue
                        q = c.x_lt(L)
                    else:
                        if ya > L + tolP:
                            continue
                        q = c.x_le(L)
                    if q > a[i] + tolQ:
                        in_m.append(i)
                        sm[i] = s = c.slope_left(q)
                        fm[i] = s == 0.
                if not _saturating_matching(
                        in_f, in_m,
                        lambda j, i: sm[i] <= sf[j] * (1. + 1e-12), ff, fm):
                    return dict(side=self.name, level=L, cut=cut,
                                rule='inward',
                                musts=[musts[i].stream for i in in_m],
                                flexes=[flexes[j].stream for j in in_f])
        return None


# %% MER search

class _Budget(Exception):
    pass


class _Frame:
    """One node on the explicit DFS stack: its frontiers, depth, memo key,
    move generator, unit count, the undo token of the child being explored
    and whether a descendant hit the depth cap."""

    def __init__(self, a, b, depth, key, gen, units):
        self.a, self.b, self.depth, self.key = a, b, depth, key
        self.gen, self.units = gen, units
        self.undo = None
        self.hit_depth = False


class _Search:
    """
    Depth-first frontier search for an unsplit MER network on one side.

    Parameters
    ----------
    side : _Side
    mode : {'restricted', 'full'}
        'restricted' branches only on the musts at the lowest frontier level
        (pinch outward), 'full' on every must (lowest level first).
    cap : int or None
        Maximum number of NEW exchangers per (must, flex) pair; continuing
        the current exchanger of both streams is free.
    extra : bool
        Also offer the preemption (E5) and lead-sharing (E6) duties.
    budget : float
        Work units.
    unit_bound : float
        Branch and bound on the number of exchangers.
    min_piece : callable(i, j) -> float, optional
        Non-tick-off duties below this are not offered.
    forbid : frozenset
        (i, j) pairs that may not be used.
    a0 : list[float], optional
        Initial must frontiers (gaps at the pinch end; best effort).
    b0 : list[float], optional
        Initial flex frontiers (default: the pinch). With `a0`, a node of
        a stream-splitting core candidate that the search completes.
    """

    def __init__(self, side, mode='restricted', cap=None, extra=False,
                 budget=1000., unit_bound=math.inf, min_piece=None,
                 forbid=frozenset(), a0=None, b0=None):
        self.side = side
        self.mode = mode
        self.cap = cap
        self.extra = extra
        self.budget = budget
        self.unit_bound = unit_bound
        self.min_piece = min_piece
        self.forbid = forbid
        self.a0 = a0
        self.b0 = b0
        self.work = 0.
        self.exhausted = False
        self.failed = {}
        self.max_depth = _DEPTH_FACTOR * (side.M + side.F) + _DEPTH_EXTRA

    def run(self):
        """Return the pieces ``(i, j, a0, b0, x)`` of an MER plan or None."""
        s = self.side
        a = list(self.a0) if self.a0 is not None else [0.] * s.M
        b = list(self.b0) if self.b0 is not None else [0.] * s.F
        self.last_m = [None] * s.M
        self.last_f = [None] * s.F
        self.pairs = defaultdict(int)
        self.units = 0
        self.serial = 0
        try:
            return self._dfs(a, b)
        except _Budget:
            self.exhausted = True
            return None

    # -- bookkeeping ----------------------------------------------------
    def _charge(self, w):
        self.work += w
        if self.work > self.budget:
            raise _Budget

    def _cont(self, i, j):
        e = self.last_m[i]
        return e is not None and e[1] == j and self.last_f[j] is e

    def _allowed(self, i, j):
        if (i, j) in self.forbid:
            return False
        if self.cap is None or self._cont(i, j):
            return True
        return self.pairs[i, j] < self.cap

    def _push(self, i, j):
        if self._cont(i, j):
            return (i, j, None)
        saved = (self.last_m[i], self.last_f[j])
        self.serial += 1
        e = (i, j, self.serial)
        self.last_m[i] = self.last_f[j] = e
        self.units += 1
        self.pairs[i, j] += 1
        return (i, j, saved)

    def _pop(self, tok):
        i, j, saved = tok
        if saved is None:
            return
        self.last_m[i], self.last_f[j] = saved
        self.units -= 1
        self.pairs[i, j] -= 1

    def _lb_units(self, a):
        """Open musts that cannot simply continue their last exchanger each
        need at least one more unit."""
        s = self.side
        lb = 0
        for i in range(s.M):
            if s.Qm[i] - a[i] > s.tolQ:
                e = self.last_m[i]
                if e is None or self.last_f[e[1]] is not e:
                    lb += 1
        return lb

    def _key(self, a, b):
        s = self.side
        r = 10. * s.tolQ
        capped = (frozenset(p for p, c in self.pairs.items() if c >= self.cap)
                  if self.cap is not None else frozenset())
        cont = frozenset(
            (i, e[1]) for i, e in enumerate(self.last_m)
            if e is not None and self.last_f[e[1]] is e
            and s.Qm[i] - a[i] > s.tolQ and s.Qf[e[1]] - b[e[1]] > s.tolQ)
        return (tuple(round(v / r) for v in a), tuple(round(v / r) for v in b),
                capped, cont)

    # -- depth-first search on an explicit stack --------------------------
    def _dfs(self, a0, b0):
        s = self.side
        r = self._expand(a0, b0, 0)
        if r is _DONE:
            return []
        if not isinstance(r, _Frame):
            return None
        stack = [r]
        path = []
        while stack:
            fr = stack[-1]
            if fr.undo is not None:
                self._pop(fr.undo)
                path.pop()
                fr.undo = None
            mv = next(fr.gen, None)
            if mv is None:
                stack.pop()
                if fr.hit_depth:
                    if stack:
                        stack[-1].hit_depth = True
                else:
                    prev = self.failed.get(fr.key)
                    self.failed[fr.key] = (fr.units if prev is None
                                           else min(prev, fr.units))
                continue
            i, j, x = mv
            a, b = fr.a, fr.b
            a2 = list(a)
            b2 = list(b)
            a2[i] = min(a[i] + x, s.Qm[i])
            b2[j] = min(b[j] + x, s.Qf[j])
            fr.undo = self._push(i, j)
            path.append((i, j, a[i], b[j], x))
            r = self._expand(a2, b2, fr.depth + 1)
            if r is _DONE:
                return list(path)
            if r is _DEPTH:
                fr.hit_depth = True
            elif isinstance(r, _Frame):
                stack.append(r)
        return None

    def _expand(self, a, b, depth):
        s = self.side
        self.work += 1.
        if self.work > self.budget:
            raise _Budget
        if self.units + self._lb_units(a) > self.unit_bound:
            return _FAIL
        open_m = [i for i in range(s.M) if s.Qm[i] - a[i] > s.tolQ]
        if not open_m:
            return _DONE
        if depth > self.max_depth:
            return _DEPTH
        key = self._key(a, b)
        f = self.failed.get(key)
        if f is not None and f <= self.units:
            return _FAIL
        d = s.analyse(a, b)
        open_f = [j for j in range(s.F) if s.Qf[j] - b[j] > s.tolQ]
        self.work += len(open_m) * len(open_f) * d.nL / _W0
        if d.slack < -10. * s.tolQ or s.rules_violation(a, b, d) is not None:
            self.failed[key] = -1
            return _FAIL
        gen = self._candidates(a, b, d, open_m, open_f)
        if gen is None:
            self.failed[key] = -1
            return _FAIL
        return _Frame(a, b, depth, key, gen, self.units)

    # -- moves ----------------------------------------------------------
    def _candidates(self, a, b, d, open_m, open_f):
        """Dead-must test, then a lazy generator of moves (i, j, x)."""
        s = self.side
        tolQ, tolP = s.tolQ, s.tolP
        lam = {i: s.musts[i].at(a[i]) for i in open_m}
        mu = {j: s.flexes[j].at(b[j]) for j in open_f}
        rem_m = {i: s.Qm[i] - a[i] for i in open_m}
        rem_f = {j: s.Qf[j] - b[j] for j in open_f}
        # a must that no flex can ever start on is dead (flex frontiers only
        # rise), unless a flex sits flat at exactly its level
        for i in open_m:
            cm = s.musts[i]
            for j in open_f:
                if mu[j] > lam[i] + tolP or not self._allowed(i, j):
                    continue
                cf = s.flexes[j]
                lim = min(rem_m[i], rem_f[j], 2. * tolQ)
                if s.max_duty(cm, a[i], cf, b[j], lim, tolP) > tolQ:
                    break
                if abs(mu[j] - lam[i]) <= tolP and cf.slope_right(b[j]) == 0.:
                    break
            else:
                return None
        return self._moves(a, b, d, open_m, open_f, lam, mu, rem_m, rem_f)

    def _moves(self, a, b, d, open_m, open_f, lam, mu, rem_m, rem_f):
        s = self.side
        tolQ, tolP = s.tolQ, s.tolP
        order = sorted(open_m, key=lambda i: (lam[i], i))
        if self.mode == 'restricted':
            p0 = lam[order[0]]
            order = [i for i in order if lam[i] <= p0 + tolP]
        for i in order:
            cm = s.musts[i]
            xres = s.xres_for(i, d)
            self._charge(_WORK_XRES * (1. + 0.1 * len(open_f))
                         + len(open_f) * d.nL / _W0)
            opts = []
            for j in open_f:
                if mu[j] > lam[i] + tolP or not self._allowed(i, j):
                    continue
                xd = s.max_duty(cm, a[i], s.flexes[j], b[j],
                                min(rem_m[i], rem_f[j]), tolP)
                if xd <= tolQ:
                    continue
                xr = float(xres[j])
                xt = min(xd, xr)
                if xt <= tolQ:
                    continue
                cont = self._cont(i, j)
                tick_m = xt >= rem_m[i] - tolQ
                tick_f = xt >= rem_f[j] - tolQ
                dt_lim = not tick_m and not tick_f and xd <= xr + tolQ
                opts.append(((not cont, not tick_m, not tick_f, dt_lim, -xt,
                              -mu[j], j), j, xt))
            opts.sort()
            for _, j, xt in opts:
                for x in self._duties(i, j, xt, a, b, lam, mu, rem_m, rem_f,
                                      open_m, open_f):
                    yield i, j, x

    def _duties(self, i, j, xt, a, b, lam, mu, rem_m, rem_f, open_m, open_f):
        """
        Candidate duties of the match (must i, flex j): ``x_top`` first, then
        the events in descending order (deduplicated within tolQ; kept only
        strictly inside ``(tolQ, x_top - tolQ)``):

        - E1 flex saving: the flex stops where it can still start another
          must k (``X_le_j(lambda_k)``);
        - E1b: the flex stops where it can still finish k in one match;
        - E1c: musts tied at one level can be served one after the other;
        - E2 vertical switch: the must stops at another flex's level;
        - E2b: the must stops where another flex can finish it in one match;
        - E3 return: the must stops where the flex, after serving the next
          musts completely, returns exactly to its level;
        - E5 preemption and E6 lead sharing (passes with ``extra`` only).
        """
        s = self.side
        tolQ, tolP = s.tolQ, s.tolP
        cm, cf = s.musts[i], s.flexes[j]
        ai, bj = a[i], b[j]
        others = [k for k in open_m if k != i]
        flexes = [g for g in open_f if g != j]
        ev = []
        if s.linear:
            # constant CP: E1, E1b, E2 and E2b for all partners at once
            # (the same closed forms as `_flex_stop_for` / `_must_stop_for`)
            self._charge(_WORK_EVENTS_VEC)
            if others:
                ko = np.array(others)
                lam_o = np.array([lam[k] for k in others])
                rem_o = np.array([rem_m[k] for k in others])
                ev.extend((cf.x_le_many(lam_o) - bj).tolist())        # E1
                sj = s.sf[j]
                ds = sj - s.sm[ko]
                ds[ds <= _SLOPE_EQ * sj] = 0.
                fits = rem_o <= rem_f[j]
                if fits.any():                                        # E1b
                    x = np.minimum(cf.x_le_many(lam_o - rem_o * ds) - bj,
                                   rem_f[j] - rem_o)
                    ev.extend(x[fits].tolist())
            if flexes:
                go = np.array(flexes)
                mu_o = np.array([mu[g] for g in flexes])
                xl = cm.x_lt_many(mu_o) - ai
                ev.extend(xl.tolist())                                # E2
                si, remi = s.sm[i], rem_m[i]                          # E2b
                capg = np.array([rem_f[g] for g in flexes])
                lo = np.maximum(0., remi - capg)
                up = mu_o > lam[i]
                lo = np.where(up, np.maximum(lo, xl), lo)
                sg = s.sf[go]
                steep = sg > si * (1. + _SLOPE_EQ)
                with np.errstate(divide='ignore', invalid='ignore'):
                    end = remi - (cm.y[-1] - mu_o) / sg
                lo = np.where(steep, np.maximum(lo, end), lo)
                ev.extend(lo[~(up & (si <= 0.))].tolist())
        else:
            self._charge(_WORK_EVENT * (len(others) + len(flexes)))
            for k in others:
                ev.append(cf.x_le(lam[k]) - bj)                       # E1
                x = _flex_stop_for(s.musts[k], a[k], rem_m[k], cf, bj,
                                   rem_f[j])                          # E1b
                if x is not None:
                    ev.append(x)
            for g in flexes:
                ev.append(cm.x_lt(mu[g]) - ai)                        # E2
                x = _must_stop_for(cm, ai, rem_m[i], s.flexes[g], b[g],
                                   rem_f[g], xt)                      # E2b
                if x is not None:
                    ev.append(x)
        groups = {}                                                   # E1c
        for k in others:
            groups.setdefault(round(lam[k] / tolP), []).append(k)
        for grp in groups.values():
            if len(grp) < 2:
                continue
            L = lam[grp[0]]
            tot = sum(rem_m[k] for k in grp)
            xL = cf.x_le(L) - bj
            for last in grp:
                ev.append(xL - (tot - rem_m[last]))
        R = 0.                                                        # E3
        for _, k in sorted((lam[k], k) for k in others):
            R += rem_m[k]
            if R >= rem_f[j]:
                break
            x = _return_duty(cm, ai, cf, bj, R, xt)
            if x is not None:
                ev.append(x)
        if self.extra:
            for k in others:                                          # E5
                if lam[k] > lam[i]:
                    ev.append(cm.x_lt(lam[k]) - ai)
            w = 1 + sum(1 for k in others if lam[k] <= lam[i] + tolP)
            for L in sorted(lam[k] for k in others)[:2]:              # E6
                ev.append((cf.x_le(L) - bj) / max(2, w))
        out = [xt]
        for x in sorted(ev, reverse=True):
            if tolQ < x < out[-1] - tolQ:
                out.append(x)
        if self.min_piece is not None:
            thr = self.min_piece(i, j)
            out = [x for x in out if x >= thr or x >= rem_m[i] - tolQ
                   or x >= rem_f[j] - tolQ]
        return out


def _count_units(pieces):
    """Exchangers after merging continuations (same pair, consecutive on
    both streams)."""
    last_m, last_f, n = {}, {}, 0
    for i, j, a0, b0, x in pieces:
        e = last_m.get(i)
        if e is not None and e[1] == j and last_f.get(j) is e:
            continue
        e = (i, j, n)
        last_m[i] = last_f[j] = e
        n += 1
    return n


def _merge(pieces):
    """Merge continuations into exchangers ``[i, j, a0, b0, Q]`` (plan
    order of their first piece)."""
    out, last_m, last_f = [], {}, {}
    for i, j, a0, b0, x in pieces:
        e = last_m.get(i)
        if e is not None and e[1] == j and last_f.get(j) is e:
            e[4] += x
            continue
        e = [i, j, a0, b0, x]
        out.append(e)
        last_m[i] = last_f[j] = e
    return out


# %% Side plans

class _SidePlan:
    """Plan of one side: pieces (i, j, a0, b0, x), per-must gaps,
    status ('trivial' | 'mer' | 'best_effort') and diagnostics. A split
    side (`hensmith._splitting`) has no pieces but `cells` (branch
    exchangers in parent coordinates), its `split` info, and `units` given
    by the caller; a split attempt without a candidate has status 'failed'
    and only its `split` info and `work`, which `_plan_side` moves to its
    best-effort plan."""
    __slots__ = ('pieces', 'gaps', 'status', 'proof', 'method', 'work',
                 'units', 'cells', 'split')

    def __init__(self, pieces, gaps, status, method, work=0., proof=None,
                 cells=None, split=None, units=None):
        self.pieces = pieces
        self.gaps = gaps
        self.status = status
        self.method = method
        self.work = work
        self.proof = proof
        self.units = _count_units(pieces) if units is None else units
        self.cells = [] if cells is None else cells
        self.split = split

    @property
    def penalty(self):
        return float(sum(self.gaps))


def _combine_cap(cap, cap1):
    if cap1:
        return 1
    return cap


def _plan_side(side, cap1=False, forbid=frozenset(), work_scale=1.,
               split=None):
    """MER search on one side, then units; best effort when there is a root
    proof or the search runs out of budget. With `split` (stream splitting:
    ``dict(Qmin, exclude, prefer)``), a side with a root proof, or whose
    best effort leaves a penalty, is planned with stream splits first
    (`hensmith._splitting._split_side`); a side the search serves returns
    exactly as without it."""
    M, F = side.M, side.F
    if M == 0:
        return _SidePlan([], [], 'trivial', 'trivial')
    zeros_m, zeros_f = [0.] * M, [0.] * F
    if F == 0:  # cannot happen at MER (a tolerance issue): everything leaks
        return _SidePlan([], list(side.Qm), 'best_effort', 'no-flex')
    d = side.analyse(zeros_m, zeros_f)
    proof = side.rules_violation(zeros_m, zeros_f, d)
    if proof is None and d.slack < -10. * side.tolQ:
        proof = dict(side=side.name, rule='cascade', slack=d.slack)
    if proof is not None:
        if split:
            from . import _splitting
            # a 'cascade' root splits only if its deficit is the cascade's
            # own tolerance (Lemma P); beyond it the targets cannot be met
            if (proof['rule'] != 'cascade'
                    or -d.slack <= _splitting._preleak_max(side)):
                sp = _splitting._split_side(side, proof, cap1, forbid,
                                            work_scale, 0., split)
                if sp.status == 'mer':
                    return sp
                # no candidate: best effort, keeping the attempt's info
                be = _best_effort(side, proof, cap1, forbid, work_scale,
                                  sp.work)
                be.split = sp.split
                return be
        return _best_effort(side, proof, cap1, forbid, work_scale, 0.)
    work = 0.
    pieces = method = None
    seen = set()
    for mode, cap, extra, budget in _SCHEDULE:
        cap = _combine_cap(cap, cap1)
        if (mode, cap, extra) in seen:
            continue
        seen.add((mode, cap, extra))
        srch = _Search(side, mode, cap, extra, budget * work_scale,
                       forbid=forbid)
        pieces = srch.run()
        work += srch.work
        if pieces is not None:
            method = f'{mode}-cap{cap}' + ('-extra' if extra else '')
            break
    if pieces is None:
        be = _best_effort(side, None, cap1, forbid, work_scale, work)
        if split:
            from . import _splitting
            # 1e-3 tolQ, far below the heat closure the plan is held to
            # (1e-12 of the scale): no best-effort penalty that closure
            # would notice is accepted without trying the split path
            if be.penalty > _splitting._SPLIT_R_TOL * side.tolQ:
                sp = _splitting._split_side(side, None, cap1, forbid,
                                            work_scale, be.work, split)
                if sp.status == 'mer':
                    return sp
                be.work, be.split = sp.work, sp.split
        return be
    first = work
    pieces, w = _improve_units(side, pieces, first, cap1, forbid, work_scale)
    work += w
    pieces, w = _units_guard(side, pieces, cap1, forbid, work_scale)
    work += w
    return _SidePlan(pieces, [0.] * M, 'mer', method, work)


def _improve_units(side, pieces, first, cap1, forbid, work_scale, a0=None,
                   b0=None):
    """Branch and bound on the number of exchangers: the same search with a
    unit bound one below the incumbent, restricted then full mode, from the
    must and flex frontiers `a0` and `b0` (default: the pinch)."""
    U = _count_units(pieces)
    step = min(_UNITS_WORK_MAX, max(_UNITS_WORK_MIN, 3. * first)) * work_scale
    total = _UNITS_WORK_TOTAL * work_scale
    used = 0.
    cap = 1 if cap1 else None
    while U > side.M and used < total:
        found = None
        for mode in ('restricted', 'full'):
            srch = _Search(side, mode, cap, False, min(step, total - used),
                           unit_bound=U - 1, forbid=forbid, a0=a0, b0=b0)
            res = srch.run()
            used += srch.work
            if res is not None:
                found = res
                break
            if used >= total:
                break
        if found is None:
            break
        pieces, U = found, _count_units(found)
    return pieces, used


def _units_guard(side, pieces, cap1, forbid, work_scale, a0=None, b0=None):
    """If a side's MER plan has more than ``3 (M + F)`` units, rerun the
    last two passes coarse to fine with a minimum piece size and keep the
    MER plan with the fewest units (MER always wins over the unit count).
    The searches start from the must and flex frontiers `a0` and `b0`
    (default: the pinch)."""
    U = _count_units(pieces)
    if U <= _GUARD_FACTOR * (side.M + side.F):
        return pieces, 0.
    used = 0.
    best, bestU = pieces, U
    passes = [p for p in _SCHEDULE if p[2]]
    for zeno in _GUARD_ZENOS:
        thr = zeno * side.duty
        for mode, cap, extra, _ in passes:
            srch = _Search(side, mode, _combine_cap(cap, cap1), extra,
                           _GUARD_WORK * work_scale,
                           min_piece=lambda i, j, thr=thr: thr, forbid=forbid,
                           a0=a0, b0=b0)
            res = srch.run()
            used += srch.work
            if res is not None:
                u = _count_units(res)
                if u < bestU:
                    best, bestU = res, u
                break
    return best, used


# %% Best effort

def _best_effort(side, proof, cap1, forbid, work_scale, work):
    """
    Best effort for a side without a (found) unsplit MER plan: greedy dives
    as incumbents, then per-must gap bisection with the MER search as a
    monotone oracle, then selection within a units cap.
    """
    budget = _BE_WORK * work_scale
    be = _Diver(side, cap1, forbid)
    cands = [([], list(side.Qm), 'trivial')]
    for use_rules, leak_rest in ((True, False), (False, False), (True, True)):
        for order in _BE_ORDERS:
            pieces, gaps = be.dive(order, use_rules, leak_rest)
            cands.append((pieces, gaps, f'dive-{order}'
                          + ('' if use_rules else '-norules')
                          + ('-rest' if leak_rest else '')))
    used = be.work
    cap = int(math.ceil(_BE_UNITS_CAP * (side.M + side.F)))
    inc = min(cands, key=lambda c: (_count_units(c[0]) > cap, sum(c[1]),
                                    _count_units(c[0])))
    res, used = _gap_search(side, inc, cap1, forbid, work_scale, used, budget,
                            cap)
    if res is not None:
        cands.append(res)
    scored = [(sum(g), _count_units(p), k, p, g, m)
              for k, (p, g, m) in enumerate(cands)]
    within = [c for c in scored if c[1] <= cap]
    if within:
        best = min(within, key=lambda c: (c[0], c[1], c[2]))
    else:
        best = min(scored, key=lambda c: (c[1], c[0], c[2]))
    _, _, _, pieces, gaps, method = best
    return _SidePlan(pieces, list(gaps), 'best_effort', method,
                     work + used, proof)


def _gap_search(side, inc, cap1, forbid, work_scale, used, budget, max_units):
    """
    Shrink the per-must gaps of the incumbent one must at a time (largest
    first): try 0, then bisect. The oracle is the MER search from frontiers
    ``a = g`` with a pair cap, a minimum piece and the unit bound
    `max_units` (a budget failure only makes it conservative). Feasibility
    within the unit bound is monotone in each gap: widening a gap only
    trims the must's pieces at its pinch end, which by L1 keeps every other
    piece feasible and never adds a unit. Returns ((pieces, gaps, method) or
    None if no gap shrank, work used).
    """
    tolQ = side.tolQ
    g = [min(max(0., x), q) for x, q in zip(inc[1], side.Qm)]
    if sum(g) <= tolQ:
        return None, used
    best = None
    frac = _BE_MIN_PIECE_FRAC
    Qm, Qf = side.Qm, side.Qf

    def min_piece(i, j):
        return frac * min(Qm[i], Qf[j])

    def oracle(trial):
        nonlocal used
        total = _BE_ORACLE_WORK * work_scale
        for mode, cap, share in _BE_ORACLE_PASSES:
            left = min(share * total, budget - used)
            if left <= 0.:
                return None
            srch = _Search(side, mode, _combine_cap(cap, cap1), False, left,
                           unit_bound=max_units, min_piece=min_piece,
                           forbid=forbid, a0=trial)
            res = srch.run()
            used += srch.work
            if res is not None:
                return res
        return None

    tol = max(tolQ, 1e-9 * side.duty)

    def shrink(g, label):
        """Per-must bisection from the feasible gaps `g` (largest first)."""
        found = None
        for i in sorted(range(side.M), key=lambda i: (-g[i], i)):
            if g[i] <= tol:
                continue
            lo, hi = 0., g[i]
            for it in range(_BE_ROUNDS + 1):
                if used >= budget or hi - lo <= tol:
                    break
                mid = 0. if it == 0 else 0.5 * (lo + hi)
                trial = list(g)
                trial[i] = mid
                res = oracle(trial)
                if res is not None:
                    hi = g[i] = mid
                    found = (res, list(g), label)
                    if mid == 0.:
                        break
                else:
                    lo = mid
            if used >= budget:
                break
        return found

    # 1. from the incumbent's gaps
    best = shrink(g, 'gap-search')
    best_pen = sum(best[1]) if best is not None else sum(g)
    # 2. from single-must leaks (d2): the whole deficit on one must, tested
    #    first at the current best penalty (skipped if it cannot beat it)
    if _BE_SINGLE:
        for m in sorted(range(side.M), key=lambda i: (-Qm[i], i)):
            if used >= budget or best_pen <= tol:
                break
            g1 = [0.] * side.M
            g1[m] = min(Qm[m], best_pen)
            res = oracle(g1)
            if res is None:
                continue
            found = shrink(g1, 'gap-search-single')
            cand = (found if found is not None
                    else (res, g1, 'gap-search-single'))
            if sum(cand[1]) < best_pen - tol:
                best, best_pen = cand, sum(cand[1])
    return best, used


class _Diver:
    """Greedy dives for best effort (never fail): the lowest must takes the
    first admissible child in the given order; when none fits it leaks heat
    (a gap), which the realisation moves to its pinch end."""

    def __init__(self, side, cap1, forbid):
        self.side = side
        self.cap = 1 if cap1 else None
        self.forbid = forbid
        self.work = 0.
        self.min_duty = _BE_MIN_DUTY * side.duty
        self.zeno = _BE_DIVE_ZENO * side.duty

    def _ok(self, a, b):
        s = self.side
        d = s.analyse(a, b)
        self.work += 1. + s.M * s.F * d.nL / _W0
        return d.slack >= -10. * s.tolQ and s.rules_violation(a, b, d) is None

    def dive(self, order, use_rules, leak_rest):
        s = self.side
        tolQ, tolP = s.tolQ, s.tolP
        M, F = s.M, s.F
        a, b = [0.] * M, [0.] * F
        pieces, gaps = [], [0.] * M
        last_m, last_f, pairs = [None] * M, [None] * F, defaultdict(int)
        serial = 0
        for _ in range(20 * (M + F) + 50):
            live = [i for i in range(M) if s.Qm[i] - a[i] > tolQ]
            if not live:
                break
            lam = {i: s.musts[i].at(a[i]) for i in live}

            def cp(c, x):
                sl = c.slope_right(x)
                return math.inf if sl == 0. else 1. / sl
            m = min(live, key=lambda i: (lam[i], -cp(s.musts[i], a[i]), i))
            cm = s.musts[m]
            rem_m = s.Qm[m] - a[m]
            open_f = [j for j in range(F) if s.Qf[j] - b[j] > tolQ]
            mu = {j: s.flexes[j].at(b[j]) for j in open_f}
            d = s.analyse(a, b)
            self.work += 1. + M * F * d.nL / _W0
            xres = s.xres_for(m, d)

            def allowed(j):
                if (m, j) in self.forbid:
                    return False
                e = last_m[m]
                if self.cap is None or (e is not None and e[1] == j
                                        and last_f[j] is e):
                    return True
                return pairs[m, j] < self.cap
            kids = []
            for j in open_f:
                if mu[j] > lam[m] + tolP or not allowed(j):
                    continue
                rem_f = s.Qf[j] - b[j]
                xd = _max_duty(cm, a[m], s.flexes[j], b[j], min(rem_m, rem_f),
                               tolP)
                if xd <= self.min_duty:
                    continue
                x = min(xd, float(xres[j]))
                if x <= self.min_duty:
                    continue
                tick = x >= rem_m - tolQ or x >= rem_f - tolQ
                if x < self.zeno and not tick:
                    continue
                gap = lam[m] - mu[j]
                kids.append((self._key(order, tick, x, gap, j), j, x))
            kids.sort()
            choice = None
            if kids:
                if use_rules:
                    for _, j, x in kids:
                        a2, b2 = list(a), list(b)
                        a2[m] += x
                        b2[j] += x
                        if self._ok(a2, b2):
                            choice = (j, x)
                            break
                if choice is None:
                    choice = (kids[0][1], kids[0][2])
            else:  # relax (R): local progress only
                best = None
                for j in open_f:
                    if mu[j] > lam[m] + tolP or not allowed(j):
                        continue
                    xd = _max_duty(cm, a[m], s.flexes[j], b[j],
                                   min(rem_m, s.Qf[j] - b[j]), tolP)
                    if xd > self.min_duty and (best is None or xd > best[1]):
                        best = (j, xd)
                choice = best
            if choice is not None:
                j, x = choice
                pieces.append((m, j, a[m], b[j], x))
                e = last_m[m]
                if not (e is not None and e[1] == j and last_f[j] is e):
                    serial += 1
                    e = (m, j, serial)
                    last_m[m] = last_f[j] = e
                    pairs[m, j] += 1
                a[m] = min(a[m] + x, s.Qm[m])
                b[j] = min(b[j] + x, s.Qf[j])
                continue
            # leak: skip the least must heat that lets some flex serve m
            best = None
            km = cp(cm, a[m])
            for j in open_f:
                cf = s.flexes[j]
                rem_f = s.Qf[j] - b[j]
                if mu[j] > lam[m] + tolP:
                    dd = cm.x_le(mu[j]) - a[m]
                    rate = 1.
                else:  # equal level and a steeper flex: leak a fraction
                    kf = cp(cf, b[j])
                    if km == math.inf and kf == math.inf:
                        frac = 0.
                    else:
                        frac = max(0., 1. - kf / km) if km > 0. else 1.
                    y = min(rem_f, rem_m * (1. - frac))
                    dd = y * frac / max(1. - frac, 1e-12)
                    rate = 1. - frac
                dd = min(max(dd, self.min_duty), rem_m)
                if best is None or (dd, -rate) < best[0]:
                    best = ((dd, -rate), dd)
            dd = rem_m if (best is None or leak_rest) else best[1]
            gaps[m] += dd
            a[m] = min(a[m] + dd, s.Qm[m])
        for i in range(M):
            left = s.Qm[i] - a[i]
            if left > 0.:
                gaps[i] += left
        return pieces, gaps

    @staticmethod
    def _key(order, tick, x, gap, j):
        if order == 'tick':
            return (not tick, -x, gap, j)
        if order == 'bestfit':
            return (gap, -x, j)
        if order == 'maxduty':
            return (-x, gap, j)
        return (-gap, -x, j)  # 'lowfit'


# %% Stream curves, targets and sides

class _StreamCurve:
    """Cleaned knots of one stream: enthalpy `H` (strictly increasing) and
    shifted temperature `T` (non-decreasing)."""
    __slots__ = ('j', 'hot', 'H', 'T', 'H_lo', 'H_hi')

    def __init__(self, j, hot, H, T):
        self.j, self.hot = j, hot
        self.H, self.T = H, T
        self.H_lo, self.H_hi = float(H[0]), float(H[-1])

    @property
    def duty(self):
        return self.H_hi - self.H_lo


def _simplify(T, H, hot):
    """
    Collapse vertical stretches and duplicates (hot: lowest T*, cold:
    highest), then drop interior knots that lie within `_KNOT_TOL` of the
    chord from the last kept knot to the next knot, for every knot dropped
    since. The chord slopes that pass within the tolerance of a knot form an
    interval, so the test keeps the running intersection of those intervals
    (a "slope cone"): O(knots).
    """
    Tk, Hk = [], []
    for t, h in zip(T.tolist(), H.tolist()):
        if Hk and h == Hk[-1]:
            if not hot:
                Tk[-1] = t
            continue
        Tk.append(t)
        Hk.append(h)
    n = len(Hk)
    if _KNOT_TOL >= 0. and n > 2:
        tol = _KNOT_TOL
        keep = [0]
        T0, H0 = Tk[0], Hk[0]
        lo, hi = -math.inf, math.inf
        for k in range(1, n - 1):
            dH = Hk[k] - H0
            lo = max(lo, (Tk[k] - tol - T0) / dH)
            hi = min(hi, (Tk[k] + tol - T0) / dH)
            s = (Tk[k + 1] - T0) / (Hk[k + 1] - H0)
            if not lo <= s <= hi:
                keep.append(k)
                T0, H0 = Tk[k], Hk[k]
                lo, hi = -math.inf, math.inf
        keep.append(n - 1)
        Tk = [Tk[k] for k in keep]
        Hk = [Hk[k] for k in keep]
    return np.array(Tk), np.array(Hk)


def _stream_curves(knots, is_hot, T_min_app):
    """Validate, shift, snap and simplify every stream's knots; None for a
    stream without duty."""
    raw = []
    for kn, hot in zip(knots, is_hot):
        T, H = kn
        T = np.asarray(T, float).ravel()
        H = np.asarray(H, float).ravel()
        if T.size != H.size or T.size == 0:
            raise ValueError('every stream needs matching, non-empty T and H '
                             'knot arrays')
        if not (np.isfinite(T).all() and np.isfinite(H).all()):
            raise ValueError('knots must be finite')
        # absorb round-off only: a real decrease would otherwise be
        # flattened into a different stream (e.g. a point load)
        if T.size > 1 and (
                float(np.diff(T).min()) < -_APPROACH_TOL
                or float(np.diff(H).min())
                < -1e-9 * max(float(np.abs(H).max()), 1.)):
            raise ValueError('knot temperatures and enthalpies must be '
                             'non-decreasing')
        T = np.maximum.accumulate(T)
        H = np.maximum.accumulate(H)
        raw.append(_simplify(T - (T_min_app if hot else 0.), H, hot))
    if not raw:
        return []
    u = np.unique(np.concatenate([r[0] for r in raw]))
    cid = np.concatenate(([0], np.cumsum(np.diff(u) > _LEVEL_EQ)))
    last = np.concatenate((np.flatnonzero(np.diff(cid)), [u.size - 1]))
    rep = u[last]  # the highest member represents each cluster
    curves = []
    for j, ((Ts, H), hot) in enumerate(zip(raw, is_hot)):
        Ts = rep[cid[np.searchsorted(u, Ts)]]
        if H.size < 2 or not H[-1] > H[0]:
            curves.append(None)
        else:
            curves.append(_StreamCurve(j, bool(hot), H, Ts))
    return curves


def _cascade(curves, scale):
    """
    The planner's problem table on the stream curves: the heat flow arriving
    at (``F_ex``) and leaving (``F_in``) every shifted level, the targets,
    the pinch (first minimum; top of the grid if Qh == 0) and the side of
    the pinch the point loads at the pinch belong to (``cut``).
    """
    act = [c for c in curves if c is not None]
    if not act:
        return dict(Qh=0., Qc=0., pinch_T=math.nan, cut='below',
                    levels=np.zeros(0), F_ex=np.zeros(0), F_in=np.zeros(0))
    levels = np.unique(np.concatenate([c.T for c in act]))[::-1]
    F_ex = np.zeros(levels.size)
    F_in = np.zeros(levels.size)
    for c in act:
        sg = 1. if c.hot else -1.
        F_ex += sg * (c.H_hi - _x_le_many(c.T, c.H, levels))
        F_in += sg * (c.H_hi - _x_lt_many(c.T, c.H, levels))
    flow = np.minimum(F_ex, F_in)
    m = float(flow.min())
    if -m <= _THRESHOLD_TOL * scale:
        Qh, k = 0., 0
    else:
        Qh = -m
        k = int(np.flatnonzero(flow <= m + 1e-12 * scale)[0])
    Qc = float(F_in[-1]) + Qh
    if Qc < 0.:
        Qh -= Qc
        Qc = 0.
    cut = 'below' if F_ex[k] <= F_in[k] else 'above'
    return dict(Qh=Qh, Qc=Qc, pinch_T=float(levels[k]), cut=cut,
                levels=levels, F_ex=F_ex, F_in=F_in)


def _sides(curves, table, tolQ, tolP):
    """Cut every stream at the pinch into its above and below level curves
    (a part not longer than tolQ is omitted) and assign must/flex roles."""
    P, cut = table['pinch_T'], table['cut']
    parts = dict(above=([], []), below=([], []))
    for c in curves:
        if c is None:
            continue
        T, H = c.T, c.H
        if cut == 'below':
            Hs = float(_x_le_many(T, H, P)[0])
        else:
            Hs = float(_x_lt_many(T, H, P)[0])
        Hs = min(max(Hs, c.H_lo), c.H_hi)
        Ts = P if T[0] <= P <= T[-1] else float(np.interp(Hs, H, T))
        inner = (H > Hs) & (H < c.H_hi)
        if c.H_hi - Hs > tolQ:
            q = np.concatenate(([0.], H[inner] - Hs, [c.H_hi - Hs]))
            y = np.concatenate(([Ts], T[inner], [T[-1]]))
            role = 'must' if c.hot else 'flex'
            lc = _LevelCurve(q, y, c.j, Hs, 1, role)
            parts['above'][0 if c.hot else 1].append(lc)
        inner = (H > c.H_lo) & (H < Hs)
        if Hs - c.H_lo > tolQ:
            q = np.concatenate(([0.], Hs - H[inner][::-1], [Hs - c.H_lo]))
            y = -np.concatenate(([Ts], T[inner][::-1], [T[0]]))
            role = 'flex' if c.hot else 'must'
            lc = _LevelCurve(q, y, c.j, Hs, -1, role)
            parts['below'][1 if c.hot else 0].append(lc)
    return {name: _Side(name, m, f, tolQ, tolP)
            for name, (m, f) in parts.items()}


def _plan_sides(sides, avoid_recycle, work_scale, split=None):
    if not avoid_recycle:
        return {name: _plan_side(side, work_scale=work_scale, split=split)
                for name, side in sides.items()}

    def run(order):
        used, plans = set(), {}
        for name in order:
            side = sides[name]
            sp = _plan_side(side, True, side.local_pairs(used), work_scale,
                            split)
            plans[name] = sp
            used.update(side.hot_cold(i, j) for i, j, *_ in sp.pieces)
            used.update(side.hot_cold(c.i, c.j) for c in sp.cells)
        return plans

    def score(plans):
        mer = all(p.status in ('mer', 'trivial') for p in plans.values())
        return (not mer, sum(p.penalty for p in plans.values()),
                sum(p.units for p in plans.values()))
    first = run(('above', 'below'))
    if not score(first)[0]:
        return first
    second = run(('below', 'above'))
    return second if score(second) < score(first) else first


# %% Plan records

class Exchanger:
    """One process-to-process exchanger of a plan (enthalpies from the
    flow-order walk; `hot` and `cold` are stream indices; `hot_seq` and
    `cold_seq` are 1-based positions in each stream's flow order;
    `pair_index` numbers repeated pairs on one side in the order the hot
    stream meets them).

    With stream splitting, `hot_frac` and `cold_frac` are the flow
    fractions of the branches the exchanger is on (1 on a trunk), and
    `hot_branch` and `cold_branch` are ``(index into Plan.splits, branch)``
    or None on a trunk. The enthalpies are *parent-equivalent* (the
    parent's state at that enthalpy is the branch's state), so
    ``H_hot_in - H_hot_out == Q / hot_frac``; on a trunk they are the
    stream's enthalpies, as without splits."""
    __slots__ = ('side', 'hot', 'cold', 'Q', 'pair_index', 'hot_seq',
                 'cold_seq', 'H_hot_in', 'H_hot_out', 'H_cold_in',
                 'H_cold_out', '_kh', '_kc', 'hot_frac', 'cold_frac',
                 'hot_branch', 'cold_branch')

    def __init__(self):
        self.hot_frac = self.cold_frac = 1.
        self.hot_branch = self.cold_branch = None

    def __repr__(self):
        return (f'<Exchanger {self.side} hot={self.hot} cold={self.cold} '
                f'Q={self.Q:.6g} pair_index={self.pair_index}>')


class Plan:
    """
    Result of :func:`plan_network`.

    Attributes
    ----------
    status : {'mer', 'best_effort'}
        'mer' iff the planned utilities equal the targets (within
        ``_MER_TOL`` of the total duty). It does not depend on how each side
        was planned: a side that fell back to best effort but whose plan
        reaches the targets gives 'mer' (its own status in ``info['sides']``
        stays 'best_effort').
    Q_hot_target, Q_cold_target : float
        Targets of the planner's cascade.
    pinch_T : float
        Pinch on the shifted scale (hot streams shifted down by
        ``T_min_app``).
    cut : {'above', 'below'}
        Side of the pinch the point loads at `pinch_T` belong to.
    exchangers : list[Exchanger]
        Process exchangers, above-pinch ones first, each side in plan order
        (pinch outward).
    stages : dict[int, list[int]]
        For every stream, its exchangers (indices into `exchangers`) in flow
        order. With splits, the flat topological order: the exchangers
        before a split, those of its branches (branch by branch, each in
        flow order), then those after it.
    splits : list[hensmith._splitting.Split]
        The stream splits (empty without stream splitting).
    paths : dict[int, list[int or Split]]
        For every stream, its flow order with each split as one item (its
        branches run in parallel); ``stages[j]`` flattened. Without splits,
        ``paths[j] == stages[j]``.
    utility : list[float]
        Duty left for each stream's utility at its outlet end (cooling for
        hot streams, heating for cold ones; >= 0).
    Q_hot, Q_cold : float
        Planned hot and cold utility.
    penalty : float
        ``Q_hot - Q_hot_target`` (>= 0).
    info : dict
        ``sides`` (per side: status, method, work, proof, gaps, units, and
        with `stream_splitting` also ``split``: None for a side that did
        not try to split, else a dict with the chosen ``candidate``, its
        network ``signature``, every ``candidates`` key or failure reason,
        ``stages``, ``branches``, ``preleak``, ``leak``, ``small`` (split
        exchangers below `Qmin`, kept) and ``errors``; a side whose
        attempt found no candidate keeps its best effort, with
        ``candidate`` and ``signature`` None),
        ``qmin_dropped`` (list of (side, hot, cold, Q)), ``dropped`` (matches
        removed by the safety net; always empty unless there is a bug),
        ``min_approach`` (on the knot curves), ``work``, ``scale``,
        ``tolQ``, ``cascade`` (the planner's problem table).
    """
    __slots__ = ('status', 'Q_hot_target', 'Q_cold_target', 'pinch_T', 'cut',
                 'exchangers', 'stages', 'utility', 'Q_hot', 'Q_cold',
                 'penalty', 'info', 'splits', 'paths')

    def __repr__(self):
        return (f'<Plan {self.status}: {len(self.exchangers)} exchangers, '
                f'Q_hot={self.Q_hot:.6g} (target {self.Q_hot_target:.6g}), '
                f'Q_cold={self.Q_cold:.6g} (target {self.Q_cold_target:.6g})>')


def _approach_violation(ch, cc, e):
    """Smallest ``T*_hot - T*_cold`` inside exchanger `e` at its ends and
    at every knot of either curve inside it (shifted scale: >= 0 is
    feasible)."""
    Q = e.Q
    hin, cout = e.H_hot_in, e.H_cold_out
    ts = [0., Q]
    Hh = ch.H[(ch.H > e.H_hot_out) & (ch.H < hin)]
    Hc = cc.H[(cc.H > e.H_cold_in) & (cc.H < cout)]
    t = np.concatenate((ts, hin - Hh, cout - Hc))
    return float((np.interp(hin - t, ch.H, ch.T)
                  - np.interp(cout - t, cc.H, cc.T)).min())


def _walk(curves, recs, N):
    """Flow-order walk: each stream from its inlet through its exchangers;
    returns (stages, utility)."""
    stages = {j: [] for j in range(N)}
    for n, e in enumerate(recs):
        stages[e.hot].append((e._kh, n))
        stages[e.cold].append((e._kc, n))
    utility = [0.] * N
    for j in range(N):
        order = [n for _, n in sorted(stages[j])]
        stages[j] = order
        c = curves[j]
        if c is None:
            continue
        if c.hot:
            H = c.H_hi
            for pos, n in enumerate(order, 1):
                e = recs[n]
                e.H_hot_in = H
                H -= e.Q
                e.H_hot_out = H
                e.hot_seq = pos
            utility[j] = H - c.H_lo
        else:
            H = c.H_lo
            for pos, n in enumerate(order, 1):
                e = recs[n]
                e.H_cold_in = H
                H += e.Q
                e.H_cold_out = H
                e.cold_seq = pos
            utility[j] = c.H_hi - H
    return stages, utility


def plan_network(knots, is_hot, T_min_app, *, avoid_recycle=False, Qmin=0.,
                 work_scale=1., stream_splitting=False, _split_exclude=None,
                 _split_prefer=None):
    """
    Plan a heat exchanger network at minimum energy requirement, unsplit
    unless `stream_splitting` is on and a side needs splits.

    Parameters
    ----------
    knots : list[tuple[array_like, array_like]]
        ``(T, H)`` knots of every stream's temperature-enthalpy curve, both
        non-decreasing (round-off is absorbed; a real decrease raises a
        ValueError), with ``H[0]`` and ``H[-1]`` the stream's end
        enthalpies (a hot stream flows from ``H[-1]`` to ``H[0]``, a cold
        stream from ``H[0]`` to ``H[-1]``). Consecutive knots with equal
        ``T`` are isothermal (flat) pieces.
    is_hot : sequence[bool]
        True for streams that are cooled.
    T_min_app : float
        Minimum approach temperature.
    avoid_recycle : bool, optional
        Never match the same (hot, cold) pair twice anywhere.
    Qmin : float, optional
        Exchangers with a smaller duty are dropped; their duty goes to the
        utilities (this can cost MER). Exchangers of a split side are never
        dropped; those below `Qmin` are reported in the side's
        ``info['sides'][side]['split']['small']``.
    work_scale : float, optional
        Multiplies every work budget.
    stream_splitting : bool, optional
        Allow a process stream to be split into parallel branches that
        re-join. A side of the pinch that no unsplit network serves at the
        minimum energy requirement (a pinch-rule proof, or an unsplit
        search that leaves a utility penalty) is planned with splits
        instead, and then reaches the MER targets exactly on the planner's
        knots (see Notes, "Stream splitting"). Sides that an unsplit
        network serves are never split, so a problem that needs no split
        gets the same network as with the default. The splits are
        returned in ``plan.splits`` (each stream's flow order in
        ``plan.paths``), every exchanger carries the fractions and branches
        of its streams, and ``plan.info['sides'][side]['split']`` records
        each side's chosen candidate; `synthesize_network` realizes them. With
        `avoid_recycle`, a split that would repeat a stream pair is not
        used, and MER is then not guaranteed. Defaults to False.
    _split_exclude : dict[str, set[tuple]], optional
        Private (the refine loop): per side, network signatures a split
        candidate may not have unless it is the side's last candidate.
    _split_prefer : dict[str, tuple[str, tuple]], optional
        Private (the refine loop): per side, the previous round's pick as
        ``(candidate name, network signature)``. That candidate is tried
        first and taken as is if it is live and plans the same network;
        otherwise the side's whole portfolio runs.

    Returns
    -------
    Plan
        See :class:`Plan`.

    Notes
    -----
    See the module docstring for the model, the lemma L1 that justifies gap
    placement and match removal, the residual condition (R), the pinch
    rules at tight levels, the events and the guarantees. The result is
    deterministic: budgets are counted in work units, never in seconds.

    **Stream splitting.** A split side is planned by `hensmith._splitting`
    as verified cells in parent coordinates: a branch of flow fraction
    ``f`` is the parent's curve with every heat times ``f``. Its records
    carry the fractions and branches; ``plan.splits`` lists the splits and
    ``plan.paths`` each stream's flow order with the splits as items. A
    root deficit that the cascade's own tolerances absorbed into the
    targets (at most about 100 ``tolQ``) is left to the musts' utilities
    (the side's ``preleak``, included in its gaps); a split side adds
    nothing else to the penalty.

    Examples
    --------
    The four-stream problem of Linnhoff and Hindmarsh (1983) with constant
    heat capacity flow rates (kW/K) and ``T_min_app = 10``:

    >>> from hensmith._planner import plan_network
    >>> def linear(T_lo, T_hi, CP):
    ...     return [T_lo, T_hi], [0., CP * (T_hi - T_lo)]
    >>> knots = [linear(20., 135., 2.), linear(60., 170., 3.),
    ...          linear(80., 140., 4.), linear(30., 150., 1.5)]
    >>> plan = plan_network(knots, [False, True, False, True], 10.)
    >>> plan.status, plan.Q_hot, plan.Q_cold
    ('mer', 20.0, 60.0)
    >>> len(plan.exchangers)
    4

    Two hot streams that reach the pinch against one cold stream break the
    number rule above it (stream 0 is cold, 1 and 2 are hot), so no unsplit
    network reaches the targets:

    >>> knots = [linear(90., 190., 2.5), linear(60., 200., 1.),
    ...          linear(100., 200., 1.)]
    >>> is_hot = [False, True, True]
    >>> plan_network(knots, is_hot, 10.).status
    'best_effort'

    With `stream_splitting`, the cold stream is split into two branches
    above the pinch, one per hot stream (a branch of flow fraction f has f
    times the stream's heat capacity flow rate), and the plan reaches the
    targets:

    >>> plan = plan_network(knots, is_hot, 10., stream_splitting=True)
    >>> plan.status, plan.Q_hot, plan.Q_cold
    ('mer', 50.0, 40.0)
    >>> split = plan.splits[0]
    >>> split.stream, split.side, [round(f, 4) for f in split.fractions]
    (0, 'above', [0.5, 0.5])
    >>> [(e.hot, e.cold, e.Q, e.cold_frac) for e in plan.exchangers]
    [(1, 0, 100.0, 0.5), (2, 0, 100.0, 0.5)]

    """
    is_hot = [bool(h) for h in is_hot]
    N = len(knots)
    if len(is_hot) != N:
        raise ValueError('knots and is_hot must have the same length')
    dT = float(T_min_app)
    if not math.isfinite(dT):
        raise ValueError('T_min_app must be finite')
    curves = _stream_curves(knots, is_hot, dT)
    act = [c for c in curves if c is not None]
    scale = float(sum(c.duty for c in act))
    tolQ = _REL_Q * max(scale, 1.)
    if act:
        tspan = (max(float(c.T[-1]) for c in act)
                 - min(float(c.T[0]) for c in act))
    else:
        tspan = 1.
    tolP = _REL_T * max(tspan, 1.)
    table = _cascade(curves, scale)
    sides = _sides(curves, table, tolQ, tolP) if act else {}
    split = (dict(Qmin=Qmin, exclude=dict(_split_exclude or {}),
                  prefer=dict(_split_prefer or {}))
             if stream_splitting else None)
    plans = _plan_sides(sides, avoid_recycle, work_scale, split) if act else {}
    split_mode = any(p.cells for p in plans.values())
    if split_mode:
        from . import _splitting
        (recs, qmin_dropped, dropped, stages, utility, min_dT, splits,
         paths) = _splitting._split_records(sides, plans, curves, N, Qmin,
                                            tolQ)
    else:
        # exchangers in plan order (above first), with flow-order keys
        recs, qmin_dropped = [], []
        for name in ('above', 'below'):
            if name not in plans:
                continue
            side = sides[name]
            for i, j, a0, b0, Q in _merge(plans[name].pieces):
                hot, cold = side.hot_cold(i, j)
                if Q < Qmin:
                    qmin_dropped.append((name, hot, cold, Q))
                    continue
                e = Exchanger()
                e.side, e.hot, e.cold, e.Q = name, hot, cold, Q
                if name == 'above':   # hot is the must (a0), cold the flex
                    e._kh, e._kc = (0, -a0), (1, b0)
                else:                 # cold is the must (a0), hot the flex
                    e._kh, e._kc = (1, b0), (0, -a0)
                recs.append(e)
        # walk; a match violating the approach (a bug) is dropped (L1)
        dropped = []
        min_dT = math.inf
        while True:
            stages, utility = _walk(curves, recs, N)
            worst = None
            min_dT = math.inf
            for n, e in enumerate(recs):
                v = _approach_violation(curves[e.hot], curves[e.cold], e)
                min_dT = min(min_dT, v)
                if v < -_APPROACH_TOL and (worst is None or v < worst[0]):
                    worst = (v, n)
            if worst is None:
                break
            e = recs.pop(worst[1])
            dropped.append((e.side, e.hot, e.cold, e.Q, worst[0]))
        splits = []
        paths = {j: list(s) for j, s in stages.items()}
    # round-off of the summed duties (pieces are exact to tolQ) is not a
    # negative utility
    utility = [0. if -10. * tolQ < u < 0. else u for u in utility]
    count = defaultdict(int)
    for j in range(N):
        c = curves[j]
        if c is None or not c.hot:
            continue
        for n in stages[j]:
            e = recs[n]
            count[e.side, e.hot, e.cold] += 1
            e.pair_index = count[e.side, e.hot, e.cold]
    Q_hot = float(sum(u for u, c in zip(utility, curves)
                      if c is not None and not c.hot))
    Q_cold = float(sum(u for u, c in zip(utility, curves)
                       if c is not None and c.hot))
    # the status comes from the utilities achieved (after the safety net and
    # Qmin), never from the side flags: a best-effort side (a greedy dive or
    # the gap search after the MER search ran out of budget) can still reach
    # the targets, and that network is MER
    penalty = Q_hot - table['Qh']
    tol_mer = _MER_TOL * max(scale, 1.)
    mer = penalty <= tol_mer and Q_cold - table['Qc'] <= tol_mer
    plan = Plan()
    plan.status = 'mer' if mer else 'best_effort'
    plan.Q_hot_target, plan.Q_cold_target = table['Qh'], table['Qc']
    plan.pinch_T, plan.cut = table['pinch_T'], table['cut']
    plan.exchangers = recs
    plan.stages = stages
    plan.utility = utility
    plan.Q_hot, plan.Q_cold = Q_hot, Q_cold
    plan.penalty = max(0., penalty)
    plan.splits = splits
    plan.paths = paths
    side_info = {}
    for name, p in plans.items():
        side = sides[name]
        side_info[name] = dict(
            status=p.status, method=p.method, work=p.work, proof=p.proof,
            units=p.units, M=side.M, F=side.F,
            gaps={side.musts[i].stream: g for i, g in enumerate(p.gaps)
                  if g > 0.})
        if stream_splitting:
            side_info[name]['split'] = p.split
    plan.info = dict(sides=side_info, qmin_dropped=qmin_dropped,
                     dropped=dropped,
                     min_approach=(dT + min_dT) if recs else None,
                     work=sum(p.work for p in plans.values()),
                     scale=scale, tolQ=tolQ, cascade=table)
    return plan


# %% Numeric front end (constant CP)

def _plan_numeric(streams, dTmin, knots=1, **kwargs):
    """
    Plan a constant-CP problem given as dicts ``{name, kind ('hot' |
    'cold'), T_in, T_out, CP}``; `knots` > 1 splits every stream into that
    many collinear pieces (discretisation tests). Keywords go to
    :func:`plan_network`.

    Returns a dict in the certificate format of the scratch oracle:
    ``matches`` (side, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in,
    T_cold_out, hot_seq, cold_seq, pair_index; with splits also hot_frac,
    cold_frac, hot_branch and cold_branch, and branch temperatures),
    ``hot_utility`` ({cold name: Q}), ``cold_utility`` ({hot name: Q}),
    ``dTmin``, ``status``, ``penalty``, ``targets`` (Qh, Qc), ``n_units``
    and ``plan``.

    Examples
    --------
    >>> from hensmith._planner import _plan_numeric
    >>> streams = [dict(name='H1', kind='hot', T_in=220., T_out=50., CP=4.),
    ...            dict(name='C1', kind='cold', T_in=140., T_out=170., CP=1.),
    ...            dict(name='C2', kind='cold', T_in=100., T_out=120., CP=1.),
    ...            dict(name='C3', kind='cold', T_in=120., T_out=250., CP=2.)]
    >>> net = _plan_numeric(streams, 10.)
    >>> net['status'], net['targets']
    ('mer', {'Qh': 80.0, 'Qc': 450.0})
    >>> [(m['hot'], m['cold'], round(m['Q'], 6), m['pair_index'])
    ...  for m in net['matches']]
    [('H1', 'C3', 160.0, 1), ('H1', 'C1', 30.0, 1), ('H1', 'C3', 20.0, 2),
     ('H1', 'C2', 20.0, 1)]

    """
    kn, hot, lo, cp, names = [], [], [], [], []
    for s in streams:
        T_in, T_out, CP = float(s['T_in']), float(s['T_out']), float(s['CP'])
        a, b = min(T_in, T_out), max(T_in, T_out)
        T = np.linspace(a, b, int(knots) + 1)
        T[0], T[-1] = a, b
        kn.append((T, CP * (T - a)))
        hot.append(s['kind'] == 'hot')
        lo.append(a)
        cp.append(CP)
        names.append(str(s['name']))
    plan = plan_network(kn, hot, float(dTmin), **kwargs)
    matches = []
    for e in plan.exchangers:
        h, c = e.hot, e.cold
        matches.append(dict(
            side=e.side, hot=names[h], cold=names[c], Q=float(e.Q),
            T_hot_in=lo[h] + e.H_hot_in / cp[h],
            T_hot_out=lo[h] + e.H_hot_out / cp[h],
            T_cold_in=lo[c] + e.H_cold_in / cp[c],
            T_cold_out=lo[c] + e.H_cold_out / cp[c],
            hot_seq=e.hot_seq, cold_seq=e.cold_seq,
            pair_index=e.pair_index))
        if plan.splits:
            matches[-1].update(hot_frac=e.hot_frac, cold_frac=e.cold_frac,
                               hot_branch=e.hot_branch,
                               cold_branch=e.cold_branch)
    hu = {names[j]: float(u) for j, u in enumerate(plan.utility)
          if not hot[j] and u > 0.}
    cu = {names[j]: float(u) for j, u in enumerate(plan.utility)
          if hot[j] and u > 0.}
    return dict(status=plan.status, matches=matches, hot_utility=hu,
                cold_utility=cu, dTmin=float(dTmin), penalty=plan.penalty,
                targets=dict(Qh=plan.Q_hot_target, Qc=plan.Q_cold_target),
                n_units=len(matches) + len(hu) + len(cu), plan=plan)
