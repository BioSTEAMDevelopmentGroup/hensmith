# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Stream splitting for the MER planner (`hensmith._planner`).

When the pinch design rules prove that a side of the pinch has no minimum
energy requirement (MER) network without stream splits, the planner can
split streams into parallel branches. This module holds the numeric
machinery. Like the planner, it works on numbers only (numpy; no BioSTEAM
objects). The planner imports it only inside the functions that run when
splitting is on, so nothing here runs, and nothing changes, when it is off.

The notation is the planner's (see its module docstring). A side has musts
``i`` with level curves ``phi_i`` on ``[0, Qm_i]`` and flexes ``j`` with
level curves ``psi_j`` on ``[0, Qf_j]``: non-decreasing, piecewise linear,
flats included, heat measured from the pinch. A node is a pair of frontier
vectors ``(a, b)``; ``side.analyse(a, b)`` gives the level set and the
inclusive and exclusive residual composites ``Di, De`` (musts) and
``Si, Se`` (flexes). Condition (R) is ``S >= D`` at every level, inclusive
and exclusive.

Cells
-----
A cell (:class:`_Cell`) is one exchanger in *parent coordinates*. It has a
duty ``x``. Must ``i``'s branch of flow fraction ``f`` covers the parent
range ``[a, a + x/f]``, and flex ``j``'s branch of fraction ``g`` covers
``[b, b + x/g]``. A trunk has fraction 1. In the counter-current coordinate
``tau`` in ``[0, x]``, the must is at ``a + tau/f`` (its pinch-side outlet
at ``tau = 0``) and the flex at ``b + tau/g``. The cell is feasible iff

    phi_i(a + tau/f) - psi_j(b + tau/g) >= -tolP  for all tau in [0, x],
    and  b + x/g <= Qf_j.                                             (C)

**Lemma B (a branch is the scaled parent).** A branch of fraction ``f``
carries the parent's material at the parent's pressure. Enthalpy is
homogeneous of degree 1 in the flows and states are intensive, so at branch
heat ``tau`` the branch is in the parent's state at parent heat ``tau/f``.
Its level curve is the parent's with every heat times ``f``
(:func:`_branch_curve`): the same levels, slopes divided by ``f``, flats
kept.

**Lemma R (splitting preserves the cascade).** Replacing a curve by
branches whose fractions sum to 1 (each starting at ``f`` times the
parent's frontier) leaves every composite ``D(L)`` and ``S(L)`` unchanged.
So the slack of (R) is unchanged: only the pinch rules (stream counts, and
branch CP = ``f`` CP) and the approach limits see a split.

**Lemma 1 (knot exactness).** (C) holds iff it holds at ``tau = 0``, at
``tau = x``, at ``f (q - a)`` for every knot ``q`` of ``phi_i`` inside the
must range and at ``g (q - b)`` for every knot ``q`` of ``psi_j`` inside the
flex range: between these points both levels are affine in ``tau``.
:func:`_cell_margin` evaluates the approach there. It is equivalent to the
planner's ``_max_duty`` on branch-scaled curves; every cell of a split plan
is verified this way before the plan is accepted.

Vertical coupling
-----------------
:class:`_Coupling` pairs the residual problem at a node level by level.
With ``X = sum_i (Qm_i - a_i)`` and a composite heat ``t`` in ``[0, X]``,
the must positions ``P_i(t)`` take the lowest ``t`` of the must composite
(musts at the same level in proportion: a flat's heat is shared) and the
flex positions ``R_j(t)`` the lowest ``t`` of the flex composite. Both are
closed-form, piecewise-linear interpolations of ``side.analyse``'s
residual arrays, with breakpoints at every composite knot (the values of
``De, Di, Se, Si`` in ``[0, X]``). Between two consecutive breakpoints every
present must follows the same normalized profile, and so does every present
flex, which is what makes a fixed-fraction block between them feasible
under (R).

Round-off, never tolerance
--------------------------
Breakpoints are merged, and positions snapped to the curve ends, only
within round-off, ``_SPLIT_ULP * max(X, 1)`` of heat, never within the heat
tolerance ``tolQ``: ``tolQ`` of heat can hide a knot ``tolQ/CP`` away in
level, far more than ``tolP``. An interval over which no must position
changes in floating point carries no must heat; it is folded into the next
one (or, at the end, into the previous one).
"""
import math
from collections import deque

import numpy as np

from ._planner import _LevelCurve

__all__ = ()

_SPLIT_ULP = 8. * np.finfo(float).eps   # round-off, relative to max(X, 1)


# %% Cells and branch curves

def _branch_curve(c, f, role='must'):
    """
    Level curve of a branch of flow fraction `f` of curve `c` (Lemma B).

    Parameters
    ----------
    c : _LevelCurve
        The parent curve.
    f : float
        Flow fraction of the branch, in ``(0, 1]``.
    role : {'must', 'flex'}, optional
        Passed to `_LevelCurve`.

    Returns
    -------
    _LevelCurve
        The parent's levels at the parent's heats times `f`.
    """
    return _LevelCurve([f * q for q in c.q], c.y, c.stream, c.H0, c.sgn,
                       role)


class _Cell:
    """
    One exchanger of a split plan, in parent coordinates.

    Parameters
    ----------
    i, j : int
        Must and flex (local indices on the side).
    x : float
        Duty.
    a, b : float
        Parent positions where the must branch and the flex branch start
        (at the pinch-side end of the exchanger).
    f, g : float, optional
        Flow fractions of the must and flex branches (1 on a trunk).
    km, kf : tuple or None, optional
        Stage keys ``(stage id, branch)`` of the must and flex branches, or
        None on a trunk.

    Notes
    -----
    The must branch covers the parent range ``[a, a + x/f]`` and the flex
    branch ``[b, b + x/g]``.
    """
    __slots__ = ('i', 'j', 'x', 'a', 'b', 'f', 'g', 'km', 'kf')

    def __init__(self, i, j, x, a, b, f=1., g=1., km=None, kf=None):
        self.i, self.j, self.x = i, j, x
        self.a, self.b, self.f, self.g = a, b, f, g
        self.km, self.kf = km, kf

    @property
    def a_end(self):
        """Parent position where the must branch leaves the cell."""
        return self.a + self.x / self.f

    @property
    def b_end(self):
        """Parent position where the flex branch leaves the cell."""
        return self.b + self.x / self.g

    def __repr__(self):
        return (f'<_Cell must={self.i} flex={self.j} x={self.x:.6g} '
                f'a={self.a:.6g} f={self.f:.6g} b={self.b:.6g} '
                f'g={self.g:.6g}>')


def _cell_margin(side, cell):
    """
    Smallest approach of a cell over its duty (Lemma 1).

    Parameters
    ----------
    side : _Side
        The (parent) side.
    cell : _Cell
        The cell.

    Returns
    -------
    margin : float
        ``min phi_i(a + tau/f) - psi_j(b + tau/g)`` over ``tau`` in
        ``[0, x]``, exact on the piecewise-linear curves; ``-inf`` if a
        fraction is not positive or a branch runs past its curve's end by
        more than ``tolQ``. The cell satisfies (C) iff ``margin >= -tolP``.
    touch : bool
        True if the approach is within ``tolP`` of zero at two consecutive
        evaluation points more than ``tolQ`` of duty apart: the cell runs
        parallel at the minimum approach along a segment (on the pinch, a
        cell without CP slack).

    Notes
    -----
    The evaluation points are ``tau = 0``, ``tau = x`` and the knots of
    both curves strictly inside their ranges. A must knot is evaluated at
    its own position (and the flex at the matching ``tau``), and likewise a
    flex knot, so no knot level is perturbed by the change of coordinates.
    """
    cm, cf = side.musts[cell.i], side.flexes[cell.j]
    x, a, b, f, g = cell.x, cell.a, cell.b, cell.f, cell.g
    tolQ = side.tolQ
    if not (f > 0. and g > 0.):
        return -math.inf, False
    a1, b1 = a + x / f, b + x / g
    if a1 > cm.Q + tolQ or b1 > cf.Q + tolQ:
        return -math.inf, False
    wm = np.asarray(cm.window(a, a1), float)
    wf = np.asarray(cf.window(b, b1), float)
    tm = (wm - a) * f
    tf = (wf - b) * g
    tau = np.concatenate(([0., x], tm, tf))
    pm = np.concatenate(([a, a1], wm, a + tf / f))
    pf = np.concatenate(([b, b1], b + tm / g, wf))
    gap = cm.at_many(pm) - cf.at_many(pf)
    margin = float(gap.min())
    order = np.argsort(tau, kind='stable')
    low = np.abs(gap[order]) <= side.tolP
    touch = bool((low[:-1] & low[1:]
                  & (np.diff(tau[order]) > tolQ)).any())
    return margin, touch


# %% Vertical coupling

class _Coupling:
    """
    Vertical coupling of the residual problem of a side at a node.

    Parameters
    ----------
    side : _Side
        The side.
    a, b : sequence[float]
        Must and flex frontiers (parent positions) of the node.
    d : _Residual, optional
        ``side.analyse(a, b)``, if already computed.

    Attributes
    ----------
    X : float
        Must heat left, ``sum_i (Qm_i - a_i)``.
    ulp : float
        Round-off of heat, ``_SPLIT_ULP * max(X, 1)``.
    t : numpy.ndarray
        Breakpoints ``t[0] = 0 < ... < t[K] = X`` (composite heat).
    Pm, Pf : numpy.ndarray
        Must and flex positions at every breakpoint, shape ``(K + 1, M)``
        and ``(K + 1, F)``: non-decreasing, ``Pm[0] = a``, ``Pf[0] = b`` and
        ``Pm[K] = Qm`` exactly.
    K : int
        Number of intervals (0 when ``X = 0``).
    slack : float
        Slack of (R) at the node.
    folded : int
        Number of breakpoints removed because no must position changes in
        floating point over their interval.

    Notes
    -----
    Let k be the last level with ``De[k] <= t``. On the flat part
    (``t < Di[k]``), ``P_i(t) = a_i + Rme[i, k] + theta (Rmi[i, k] -
    Rme[i, k])`` with ``theta = (t - De[k]) / (Di[k] - De[k])``: a flat's
    heat is shared in proportion. On the sloped part, ``P_i(t) = a_i +
    Rmi[i, k] + theta (Rme[i, k + 1] - Rmi[i, k])`` with ``theta = (t -
    Di[k]) / (De[k + 1] - Di[k])``; past the last level it is the whole
    residual. Both parts give the residual arrays exactly at their ends
    (``theta = 0``). Flexes use the same formulas with ``S, Rfi, Rfe, b``.
    The breakpoints are ``0``, ``X`` and every value of ``De, Di, Se, Si``
    clipped to ``[0, X]``; one is dropped iff it is within `ulp` of the
    previous kept one. Positions are made monotone along ``t``, clipped to
    the curve ends and snapped to them within `ulp`.
    """
    __slots__ = ('side', 'd', 'a', 'b', 'X', 'ulp', 't', 'Pm', 'Pf', 'K',
                 'slack', 'folded')

    def __init__(self, side, a, b, d=None):
        if d is None:
            d = side.analyse(a, b)
        self.side, self.d = side, d
        self.a = np.array(a, float).reshape(side.M)
        self.b = np.array(b, float).reshape(side.F)
        self.slack = d.slack
        self.X = X = math.fsum(q - x for q, x in zip(side.Qm, a))
        self.ulp = ulp = _SPLIT_ULP * max(X, 1.)
        self.folded = 0
        if not X > 0.:
            self.t = np.zeros(1)
            self.Pm, self.Pf = self.a[None, :].copy(), self.b[None, :].copy()
            self.K = 0
            return
        raw = np.unique(np.clip(np.concatenate(
            ([0., X], d.De, d.Di, d.Se, d.Si)), 0., X))
        ts = [0.]
        for v in raw.tolist():
            if v - ts[-1] > ulp:
                ts.append(v)
        if ts[-1] != X:
            if len(ts) > 1 and X - ts[-1] <= ulp:
                ts[-1] = X
            else:
                ts.append(X)
        t = np.array(ts)
        Pm = np.maximum.accumulate(self._musts(t), axis=0)
        Pf = np.maximum.accumulate(self._flexes(t), axis=0)
        Pm[-1] = side.Qm
        keep = [0]
        last = t.size - 1
        for k in range(1, last + 1):
            if (Pm[k] == Pm[keep[-1]]).all():   # no must heat in between
                if k < last:
                    continue                    # fold into the next one
                if len(keep) > 1:
                    keep[-1] = k                # fold into the previous
                    continue
            keep.append(k)
        self.folded = last + 1 - len(keep)
        self.t, self.Pm, self.Pf = t[keep], Pm[keep], Pf[keep]
        self.K = len(keep) - 1

    @staticmethod
    def _at(ts, Ci, Ce, Ri, Re, front, Q, ulp):
        """Positions at composite heats `ts` (closed form, snapped)."""
        out = np.repeat(front[None, :], ts.size, axis=0)
        nL = Ci.size
        if nL and front.size:
            k = np.clip(np.searchsorted(Ce, ts, side='right') - 1, 0, nL - 1)
            k1 = np.minimum(k + 1, nL - 1)
            flat = ts < Ci[k]
            slope = ~flat & (k < nL - 1)
            den = np.where(flat, Ci[k] - Ce[k], 1.)
            th = np.clip(np.where(flat, (ts - Ce[k]) / den, 0.), 0., 1.)
            den = np.where(slope, Ce[k1] - Ci[k], 1.)
            th2 = np.clip(np.where(slope, (ts - Ci[k]) / den, 0.), 0., 1.)
            h = np.where(flat, Re[:, k] + th * (Ri[:, k] - Re[:, k]),
                         Ri[:, k] + th2 * (Re[:, k1] - Ri[:, k]))
            out = out + h.T
        Q = np.asarray(Q, float)
        return np.where(out >= Q - ulp, Q, out)

    def _musts(self, ts):
        d = self.d
        return self._at(ts, d.Di, d.De, d.Rmi, d.Rme, self.a, self.side.Qm,
                        self.ulp)

    def _flexes(self, ts):
        d = self.d
        return self._at(ts, d.Si, d.Se, d.Rfi, d.Rfe, self.b, self.side.Qf,
                        self.ulp)

    def must_at(self, t):
        """Must positions (list) at composite heat `t` in ``[0, X]``, by the
        closed form; ``Qm`` exactly at ``t >= X``."""
        if t >= self.X:
            return [float(q) for q in self.side.Qm]
        if t <= 0.:
            return self.a.tolist()
        return self._musts(np.array([float(t)]))[0].tolist()

    def flex_at(self, t):
        """Flex positions (list) at composite heat `t` in ``[0, X]``, by the
        closed form."""
        if t <= 0.:
            return self.b.tolist()
        return self._flexes(np.array([float(min(t, self.X))]))[0].tolist()


# %% Transport

def _transport(h, g, comp, prefer=(), forbid=frozenset(), tol=0.):
    """
    Transport of the must heats `h` to the flexes `g` on compatible pairs.

    Parameters
    ----------
    h : dict[int, float]
        Must heats (row sums), in north-west order (by frontier level).
        Musts with no heat are ignored.
    g : dict[int, float]
        Flex heats (column capacities), in north-west order. Flexes with no
        heat are ignored.
    comp : collection[tuple[int, int]]
        Compatible ``(must, flex)`` pairs.
    prefer : sequence[tuple[int, int]], optional
        Pairs tried first, in order (continuations of the previous block).
    forbid : collection[tuple[int, int]], optional
        Pairs never used.
    tol : float, optional
        Must heat that may be left unplaced by the round-off imbalance of
        `h` and `g`; it is added to the row's largest cell.

    Returns
    -------
    dict[tuple[int, int], float] or None
        Positive duties on allowed pairs. Every row sums to its heat up to
        float summation, columns stay within their heats plus `tol`, and
        the cells form a forest (at most ``m + f - 1`` of them). None if no
        such transport exists.

    Notes
    -----
    A greedy pass assigns ``min`` of the remaining row and column heats to
    the pairs in order (`prefer`, then north-west). Every assignment
    exhausts its row or its column, so its cells never form a cycle. If it
    leaves more than `tol` of some row unplaced, an Edmonds-Karp maximum
    flow on the compatibility graph decides; its cycles are then cancelled.
    Finally, the largest cell of each row takes the row's heat minus the
    others.
    """
    rows = [i for i, v in h.items() if v > 0.]
    cols = [j for j, v in g.items() if v > 0.]
    rset, cset = set(rows), set(cols)
    order, seen = [], set()
    for c in (*prefer, *[(i, j) for i in rows for j in cols]):
        if (c not in seen and c[0] in rset and c[1] in cset and c in comp
                and c not in forbid):
            seen.add(c)
            order.append(c)
    rh, rg = {i: h[i] for i in rows}, {j: g[j] for j in cols}
    q = {}
    for i, j in order:
        x = min(rh[i], rg[j])
        if x > 0.:
            q[i, j] = x
            rh[i] -= x
            rg[j] -= x
    if any(v > tol for v in rh.values()):
        q = _max_flow(h, g, rows, cols, order)
        placed = dict.fromkeys(rows, 0.)
        for (i, j), x in q.items():
            placed[i] += x
        if any(h[i] - placed[i] > tol for i in rows):
            return None
        q = _forest(q, order)
    for i in rows:
        cells = [c for c in order if c[0] == i and c in q]
        if not cells:   # round-off heat only, on exhausted columns
            cells = [c for c in order if c[0] == i][:1]
            if not cells:
                return None
            q[cells[0]] = h[i]
            continue
        big = max(cells, key=q.__getitem__)
        x = h[i] - math.fsum(q[c] for c in cells if c != big)
        if not x > 0.:
            return None
        q[big] = x
    return {c: q[c] for c in order if c in q}


def _max_flow(h, g, rows, cols, order):
    """Edmonds-Karp maximum flow from the musts (capacities `h`) to the
    flexes (capacities `g`) over the pairs in `order` (unbounded); returns
    the positive pair flows. Adjacency lists keep `order`, so the result is
    deterministic."""
    m = len(rows)
    ri = {i: 1 + k for k, i in enumerate(rows)}
    cj = {j: 1 + m + k for k, j in enumerate(cols)}
    src, snk = 0, 1 + m + len(cols)
    adj = [[] for _ in range(snk + 1)]
    cap = {}

    def add(u, v, c):
        if (u, v) not in cap:
            cap[u, v] = 0.
            cap.setdefault((v, u), 0.)
            adj[u].append(v)
            adj[v].append(u)
        cap[u, v] += c
    for i in rows:
        add(src, ri[i], h[i])
    for i, j in order:
        add(ri[i], cj[j], math.inf)
    for j in cols:
        add(cj[j], snk, g[j])
    while True:
        prev = {src: None}
        queue = deque([src])
        while queue and snk not in prev:
            u = queue.popleft()
            for v in adj[u]:
                if v not in prev and cap[u, v] > 0.:
                    prev[v] = u
                    queue.append(v)
        if snk not in prev:
            break
        aug, v = math.inf, snk
        while prev[v] is not None:
            aug = min(aug, cap[prev[v], v])
            v = prev[v]
        v = snk
        while prev[v] is not None:
            u = prev[v]
            cap[u, v] -= aug   # the bottleneck edge becomes exactly 0
            cap[v, u] += aug
            v = u
    q = {}
    for i, j in order:
        x = cap[cj[j], ri[i]]
        if x > 0.:
            q[i, j] = x
    return q


def _forest(q, order):
    """Cancel the cycles of the support of `q` (bipartite: musts and
    flexes) keeping every row and column sum; returns a forest.

    Cells are added in `order`. A cell closing a cycle with the forest built
    so far gets alternating signs around the cycle and moves the smaller of
    the two minima, which empties one cell (the new one on a tie)."""
    q = {c: q[c] for c in order if c in q}
    tree = {}   # node -> {neighbour: cell}

    def path(u, v):
        prev = {u: None}
        queue = deque([u])
        while queue:
            w = queue.popleft()
            if w == v:
                break
            for z, c in tree.get(w, {}).items():
                if z not in prev:
                    prev[z] = (w, c)
                    queue.append(z)
        if v not in prev:
            return None
        cells = []
        while prev[v] is not None:
            v, c = prev[v]
            cells.append(c)
        return cells[::-1]

    def link(c, on):
        u, v = ('m', c[0]), ('f', c[1])
        if on:
            tree.setdefault(u, {})[v] = c
            tree.setdefault(v, {})[u] = c
        else:
            del tree[u][v], tree[v][u]
    for c in list(q):
        while c in q:
            cyc = path(('f', c[1]), ('m', c[0]))
            if cyc is None:
                link(c, True)
                break
            odd, even = cyc[0::2], [c, *cyc[1::2]]
            da, db = min(q[e] for e in odd), min(q[e] for e in even)
            plus, minus, dx = (odd, even, db) if db <= da else (even, odd, da)
            for e in plus:
                q[e] += dx
            for e in minus:
                q[e] -= dx
                if not q[e] > 0.:
                    del q[e]
                    if e != c:
                        link(e, False)
    return q
