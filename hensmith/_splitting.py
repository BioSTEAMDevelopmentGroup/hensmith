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

The core (the provable backstop)
--------------------------------
**Theorem V'.** Let a node satisfy (R) and let ``[t_s, t_e]`` be two
consecutive breakpoints of its coupling. Take the must heats ``h_i =
P_i(t_e) - P_i(t_s)``, the flex heats ``g_j = R_j(t_e) - R_j(t_s)`` and any
transport ``q_ij >= 0`` with these row and column sums, and give cell
``(i, j)`` the duty ``q_ij`` from ``(P_i(t_s), R_j(t_s))`` with fractions
``q_ij / h_i`` and ``q_ij / g_j``. This elementary block is feasible and
the node after it satisfies (R). *Proof.* No knot lies inside the interval,
so every present must follows the same affine normalized profile, the must
composite's quantile ``lam_M``, and every present flex follows ``lam_F``.
In a cell both branches are at the same composite coordinate, so (C) reads
``lam_M >= lam_F``: true at both ends by (R), hence throughout. The block
consumes the lowest ``t_e - t_s`` of both composites, ``D' = max(0, D -
Delta)`` and ``S' = max(0, S - Delta)``, and that map is monotone, so (R)
holds after it. Flex heat is taken as prefixes. All branches of a stream
span the same parent range, so every remix is isothermal.

**Corollary C (coarsening).** A fixed-fraction block over several
breakpoints consumes the same heats as the elementary chain it replaces,
so it ends at the same node, where (R) is automatic. A cell is feasible iff
its normalized profiles are, independently of its duty, so the block
exists iff a transport exists on that compatibility graph; it is found by
a search (feasibility is not monotone in ``t_e``) and verified cell by
cell.

**Node discipline.** A chain of vertical blocks runs on the breakpoints
of one coupling, and every node is the coupling's closed-form positions at
a breakpoint, never the start plus the duties, so round-off does not
accumulate: the column imbalance of a transport (O(eps) of its duty) is a
sub-``tolP`` level offset that the cells are verified with. Each node is
checked, ``slack >= -_SPLIT_R_TOL tolQ``, by an exact analysis (`_exact`:
the search's analysis omits a stream's last ``tolQ`` of residual heat).

**Lemma P (pre-leak).** `_cascade`'s own tolerances can leave a root slack
down to ``-(_THRESHOLD_TOL / _REL_Q + 0.1) tolQ``: a threshold deficit of
up to ``_THRESHOLD_TOL`` of the scale, and 0.1 tolQ between near-equal
minima. Removing the lowest ``delta = -slack`` of the must composite gives
``D' = max(0, D - delta)`` with ``S`` unchanged, and ``S - D >= -delta``,
so (R) holds at ``P(delta)`` (`_preleak_root`). That heat goes to the
musts' utilities, consistent with the targets `_cascade` already set.

**Recovery.** If a cell of an elementary block fails (C), a composite
breakpoint hidden inside the interval is inserted and the block rebuilt.
Otherwise the failure is position round-off (one ulp of heat moving a
tiny-CP stream by more than ``tolP``): the failing must heat is attached
to the must's series cell of the previous block if that re-verifies, else
leaked to the must's utility if at most ``_SPLIT_R_TOL tolQ``, and recorded
(a leak event, ``1e-14`` of the scale, is 100 times inside the planner's
``1e-12`` heat closure); anything larger raises `_SplitInvariantError`.

Candidates
----------
A candidate (:class:`_Candidate`) is one verified split plan of a side. Its
key, smaller is better: exchangers below ``Qmin`` or with a fraction below
`_SPLIT_MIN_FRACTION`; non-isothermal remixes that feed a process exchanger
of the same stream, and curved streams with more than `_SPLIT_MIX_CAP`
mixers; exchangers parallel at the minimum approach on sides with a curved
stream; units + extra branches + split stages; the most mixers on one
stream; the candidate order. Its signature holds no duty or position, so it
identifies the same network across knot refinements and generators.

Selection and records
---------------------
`_split_side` generates the candidates of a side from the pre-leaked root.
The previous refine round's pick is tried first and kept if its signature
is not excluded; otherwise the smallest key wins among the candidates not
excluded (a side's last candidate is never excluded). Each must's gap is
its pre-leak plus its leak, so the side's penalty is truthful.
`_split_records` turns the side plans into the plan's records: every
split stage becomes a `Split`, branch enthalpies are parent-equivalent (a
branch exchanger moves its branch by ``Q/f``), a mix is written with the
duties (``H_mix = H_split -/+ sum Q``), and a stream's flow order lists
each split as one item (``Plan.paths``) or flattened (``Plan.stages``).
"""
import math
from collections import Counter, defaultdict, deque

import numpy as np

from ._planner import (Exchanger, _APPROACH_TOL, _LevelCurve, _REL_Q,
                       _Side, _SidePlan, _THRESHOLD_TOL, _WORK_EVENT, _merge)

__all__ = ()

_SPLIT_ULP = 8. * np.finfo(float).eps   # round-off, relative to max(X, 1)
_SPLIT_R_TOL = 1e-3          # core nodes: (R) slack and leak cap, x tolQ
_SPLIT_MIN_FRACTION = 1e-3   # smallest branch fraction of a coarsened block
_SPLIT_COARSEN = True        # False: elementary vertical blocks only
_SPLIT_MIX_CAP = 2           # mixers per curved stream and side (the key)
_ISO_TOL = 10.               # isothermal remix, x tolQ
_CORE_ORDER = ('V', 'LV', 'VT', 'LVT')   # candidate order of the core
_CORE_STRATEGIES = ('V',)    # the core strategies `_drive` runs


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

    def insert(self, t):
        """Insert the breakpoint `t`, strictly between two breakpoints, with
        its closed-form positions (kept monotone); returns its index."""
        k = int(np.searchsorted(self.t, t))
        if not (0 < k <= self.K and self.t[k - 1] < t < self.t[k]):
            raise ValueError(f'{t!r} is not strictly inside an interval')
        ts = np.array([float(t)])
        pm = np.clip(self._musts(ts)[0], self.Pm[k - 1], self.Pm[k])
        pf = np.clip(self._flexes(ts)[0], self.Pf[k - 1], self.Pf[k])
        self.t = np.insert(self.t, k, float(t))
        self.Pm = np.insert(self.Pm, k, pm, axis=0)
        self.Pf = np.insert(self.Pf, k, pf, axis=0)
        self.K += 1
        return k


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


# %% Vertical core

class _SplitInvariantError(Exception):
    """
    A core invariant failed: a node violates (R) beyond round-off, or a
    vertical block cannot be completed within round-off. The strategy fails
    and the other candidates remain. It is raised, never asserted, so that
    the driver's catch sees it and ``python -O`` cannot strip it.
    """


def _exact(side):
    """
    `side` with no heat tolerance in its residual arrays.

    `_Side.analyse` omits every stream with at most ``tolQ`` of residual
    heat: the search's tolerance. At a core node that would drop a stream's
    last sliver of heat (up to ``tolQ``, far more than round-off), move the
    coupling's breakpoints and misreport the slack of (R) by that much, so
    the core analyses its nodes exactly.
    """
    return _Side(side.name, side.musts, side.flexes, 0., side.tolP)


def _preleak_max(side):
    """Largest root deficit the core absorbs by a pre-leak: the cascade's
    own tolerances (threshold ``_THRESHOLD_TOL`` of the scale, and 0.1 tolQ
    between near-equal minima) plus round-off, ``_SPLIT_R_TOL tolQ``."""
    return (_THRESHOLD_TOL / _REL_Q + 0.1 + _SPLIT_R_TOL) * side.tolQ


def _preleak_root(side, d=None):
    """
    The pre-leaked root of a side (Lemma P).

    Parameters
    ----------
    side : _Side
        The side.
    d : _Residual, optional
        The exact analysis of the root, ``_exact(side).analyse(0, 0)``.

    Returns
    -------
    delta : float
        ``max(0, -slack)`` at the root: the must heat the core leaves to
        the musts' utilities (compare with `_preleak_max`).
    a0 : list[float]
        The must positions ``P(delta)`` of the vertical coupling at the
        root: the lowest `delta` of the must composite (zeros if
        ``delta = 0``).
    """
    z_m, z_f = [0.] * side.M, [0.] * side.F
    if d is None:
        d = _exact(side).analyse(z_m, z_f)
    delta = max(0., -d.slack)
    if not delta > 0.:
        return 0., z_m
    return delta, _Coupling(side, z_m, z_f, d).must_at(delta)


class _Block:
    """
    One block of a core candidate.

    Parameters
    ----------
    kind : {'vertical', 'pinch'}
        How the block was built.
    start, end : tuple[list[float], list[float]]
        The nodes ``(a, b)`` before and after the block; `end` holds the
        coupling's closed-form positions (never start plus sums).
    cells : list[_Cell]
        The block's cells.
    cp : _Coupling, optional
        The coupling of a vertical block; the next vertical block continues
        on it from breakpoint `k`.
    k : int, optional
        Breakpoint of `cp` where the block ends.
    span : int, optional
        Number of coupling intervals the block covers (1: elementary).
    leak : dict[int, float], optional
        Must heat left to the must's utility (position round-off).
    knots : int, optional
        Hidden breakpoints inserted to complete the block.
    work : float, optional
        Work spent on the block.
    """
    __slots__ = ('kind', 'start', 'end', 'cells', 'pairs', 'cp', 'k', 'span',
                 'leak', 'knots', 'work')

    def __init__(self, kind, start, end, cells, cp=None, k=0, span=1,
                 leak=None, knots=0, work=0.):
        self.kind, self.start, self.end = kind, start, end
        self.cells = cells
        self.pairs = frozenset((c.i, c.j) for c in cells)
        self.cp, self.k, self.span = cp, k, span
        self.leak = {} if leak is None else leak
        self.knots, self.work = knots, work


class _Vertical:
    """Vertical blocks on one coupling (see `_vertical_block`)."""

    def __init__(self, side, cp, prev, forbid, used):
        self.side, self.cp, self.prev = side, cp, prev
        # continuations: the previous block's series cells, which merge
        self.conts = ([(c.i, c.j) for c in prev.cells
                       if c.f == 1. and c.g == 1.] if prev is not None else [])
        self.used = frozenset(used)
        self.forbid = frozenset(forbid) | (self.used - frozenset(self.conts))
        self.tol = max(_SPLIT_R_TOL * side.tolQ,
                       4. * (side.M + side.F) * cp.ulp)
        self.work = 0.
        self.knots = 0

    def _margin(self, cell):
        self.work += _WORK_EVENT
        return _cell_margin(self.side, cell)[0]

    def cells(self, ks, ke, coarse):
        """
        Cells of the block over breakpoints ``[ks, ke]``.

        Returns ``(cells, bad)`` (`bad`: the cells failing (C)), or None if
        forbidden or used pairs leave no transport or, for a `coarse` block,
        if no transport exists on the compatible pairs, the block does not
        verify or it has a fraction below `_SPLIT_MIN_FRACTION`. An
        elementary block with no transport on all pairs contradicts Theorem
        V' and raises `_SplitInvariantError`.
        """
        side, cp = self.side, self.cp
        tolP = side.tolP
        am, bf = cp.Pm[ks], cp.Pf[ks]
        hm, gf = cp.Pm[ke] - am, cp.Pf[ke] - bf
        musts, flexes = side.musts, side.flexes
        rows = sorted((i for i in range(side.M) if hm[i] > 0.),
                      key=lambda i: (musts[i].at(am[i]), i))
        cols = sorted((j for j in range(side.F) if gf[j] > cp.ulp),
                      key=lambda j: (flexes[j].at(bf[j]), j))
        if rows and not cols and side.F:
            # a sliver: the flex heat, equal to the must heat up to
            # round-off, is shared by flexes with at most `ulp` each. It
            # is round-off per flex, not in total, so the largest flex
            # carries it (and the balance below)
            cols = [int(np.argmax(gf))]
        h = {i: float(hm[i]) for i in rows}
        g = {j: float(gf[j]) for j in cols}
        # the must heats are exact; round-off of the flex heats is balanced
        # on the largest flex
        gap = math.fsum(h.values()) - math.fsum(g.values())
        if gap > self.tol:
            if coarse:
                return None
            raise _SplitInvariantError(
                f'vertical block: the flexes lack {gap!r} of heat')
        if gap > 0. and g:
            j = max(g, key=g.get)
            g[j] += gap
        a = {i: float(am[i]) for i in rows}
        b = {j: float(bf[j]) for j in cols}
        if coarse:   # Corollary C: normalized profiles, independent of duty
            comp = {(i, j) for i in rows for j in cols
                    if (i, j) not in self.forbid and self._margin(_Cell(
                        i, j, h[i], a[i], b[j], 1., h[i] / g[j])) >= -tolP}
        else:        # Theorem V': every pair
            comp = {(i, j) for i in rows for j in cols}
        forbid = self.forbid
        while True:
            q = _transport(h, g, comp, self.conts, forbid, self.tol)
            if q is None:
                if coarse or forbid and _transport(
                        h, g, comp, self.conts, tol=self.tol) is not None:
                    return None   # forbidden or used pairs removed it
                # Theorem V': on all pairs, a transport always exists
                raise _SplitInvariantError(
                    f'vertical block: no transport over breakpoints '
                    f'[{ks}, {ke}] (must heats {h!r}, flex heats {g!r})')
            nm = Counter(i for i, _ in q)
            nf = Counter(j for _, j in q)
            # avoid_recycle: a used pair returns only as a series cell
            again = {c for c in q if c in self.used
                     and (nm[c[0]] > 1 or nf[c[1]] > 1)}
            if not again:
                break
            forbid = forbid | again
        col = defaultdict(list)
        for (i, j), x in q.items():
            col[j].append(x)
        col = {j: math.fsum(xs) for j, xs in col.items()}
        cells = [_Cell(i, j, x, a[i], b[j],
                       1. if nm[i] == 1 else x / h[i],
                       1. if nf[j] == 1 else x / col[j])
                 for (i, j), x in q.items()]
        bad = [c for c in cells if not self._margin(c) >= -tolP]
        if coarse and (bad or any(min(c.f, c.g) < _SPLIT_MIN_FRACTION
                                  for c in cells)):
            return None
        return cells, bad

    def coarse(self, ks):
        """The longest verified block from `ks` the search finds: ``t_e =
        X`` first, then galloping (2, 4, 8, ... intervals) and bisection
        between the last success and the first failure. Returns ``(cells,
        ke)`` or None."""
        K = self.cp.K
        out = self.cells(ks, K, True)
        if out is not None:
            return out[0], K
        good, lo, hi, step = None, ks + 1, K, 2
        while ks + step < hi:
            out = self.cells(ks, ks + step, True)
            if out is None:
                hi = ks + step
                break
            good, lo = (out[0], ks + step), ks + step
            step *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            out = self.cells(ks, mid, True)
            if out is None:
                hi = mid
            else:
                good, lo = (out[0], mid), mid
        return good

    def elementary(self, ks):
        """The elementary block ``[ks, ks + 1]`` with the recovery of a
        failing cell: ``(cells, leak)``, or None if forbidden or used pairs
        leave no transport."""
        while True:
            out = self.cells(ks, ks + 1, False)
            if out is None:
                return None
            cells, bad = out
            if not bad:
                return cells, {}
            t = self._hidden(ks)
            if t is None:
                return self._round_off(ks, cells, bad)
            self.cp.insert(t)
            self.knots += 1

    def _hidden(self, ks):
        """The first composite breakpoint hidden strictly inside the
        interval ``ks`` with must heat on both sides of it, or None."""
        cp, d = self.cp, self.cp.d
        t0, t1 = cp.t[ks], cp.t[ks + 1]
        v = np.concatenate((d.De, d.Di, d.Se, d.Si))
        for t in np.unique(v[(v > t0) & (v < t1)]).tolist():
            pm = cp._musts(np.array([t]))[0]
            if (pm > cp.Pm[ks]).any() and (pm < cp.Pm[ks + 1]).any():
                return t
        return None

    def _round_off(self, ks, cells, bad):
        """Position round-off: the must heat of the failing cells goes to the
        must's series cell of the previous block (if that re-verifies), else
        to the must's utility (a leak, at most ``_SPLIT_R_TOL tolQ``);
        anything larger raises."""
        side, cp = self.side, self.cp
        drop = {id(c) for c in bad}
        keep = [c for c in cells if id(c) not in drop]
        lost = defaultdict(list)
        for c in bad:
            lost[c.i].append(c)
        leak = {}
        for i, cs in lost.items():
            dl = math.fsum(c.x for c in cs)
            if self._attach(ks, i, cs, dl, keep):
                continue
            if not dl <= _SPLIT_R_TOL * side.tolQ:
                raise _SplitInvariantError(
                    f'vertical block: must {i} fails (C) on {dl!r} of heat')
            leak[i] = dl
        rows, cols = defaultdict(list), defaultdict(list)
        for c in keep:
            rows[c.i].append(c)
            cols[c.j].append(c)
        for i, cs in rows.items():
            if i in lost:   # the kept branches stop dl short of P_i(t_s)
                dl = math.fsum(c.x for c in lost[i])
                hi = math.fsum(c.x for c in cs)
                for c in cs:
                    c.a = float(cp.Pm[ks, i]) + dl
                    c.f = 1. if len(cs) == 1 else c.x / hi
        for j, cs in cols.items():
            s = math.fsum(c.x for c in cs)
            for c in cs:
                c.g = 1. if len(cs) == 1 else c.x / s
        for c in keep:
            if not self._margin(c) >= -side.tolP:
                raise _SplitInvariantError(
                    f'vertical block: {c!r} fails (C) after a leak')
        return keep, leak

    def _attach(self, ks, i, cs, dl, keep):
        """Attach must `i`'s failing heat `dl` (all of its cells here, on one
        flex that keeps no other cell here) to its series cell of the
        previous block, contiguous on both streams, if that re-verifies."""
        p, cp = self.prev, self.cp
        js = {c.j for c in cs}
        if p is None or len(js) != 1 or any(c.i == i or c.j in js
                                             for c in keep):
            return False
        mine = [c for c in p.cells if c.i == i]
        if len(mine) != 1:
            return False
        pc = mine[0]
        if not (pc.j in js and pc.f == 1. and pc.g == 1.
                and abs(pc.a_end - cp.Pm[ks, i]) <= cp.ulp
                and abs(pc.b_end - cp.Pf[ks, pc.j]) <= self.tol):
            return False
        x = pc.x
        pc.x = x + dl
        if self._margin(pc) >= -self.side.tolP:
            return True
        pc.x = x
        return False


def _vertical_block(side, a, b, prev=(), forbid=frozenset(), used=frozenset(),
                    d=None):
    """
    The next vertical block from node ``(a, b)`` (Theorem V', Corollary C).

    Parameters
    ----------
    side : _Side
        The side.
    a, b : sequence[float]
        The node; it satisfies (R).
    prev : sequence[_Block], optional
        The previous block, if any (a list of at most one). A vertical block
        ending at ``(a, b)`` hands over its coupling, so a chain of vertical
        blocks runs on the breakpoints of one coupling; its series cells are
        preferred (continuations).
    forbid : collection[tuple[int, int]], optional
        Pairs never used.
    used : collection[tuple[int, int]], optional
        Pairs of earlier blocks (avoid_recycle): used again only as series
        continuations of the previous block.
    d : _Residual, optional
        ``_exact(side).analyse(a, b)``, if already computed.

    Returns
    -------
    _Block or None
        None only if forbidden or used pairs leave no transport.

    Raises
    ------
    _SplitInvariantError
        If the elementary block has no transport on all pairs or cannot be
        completed within round-off, or the block would make no progress.

    Notes
    -----
    With `_SPLIT_COARSEN`, the block ending at ``X`` is tried first, then
    a gallop and bisection over the breakpoints. Feasibility is not
    monotone in the end, so this is a search: every coarsened block is
    verified cell by cell and needs every fraction >= `_SPLIT_MIN_FRACTION`.
    The elementary block needs no search. If one of its cells fails (C), a
    composite breakpoint hidden inside the interval is inserted and the
    block rebuilt; otherwise the failure is position round-off, handled by
    `_Vertical._round_off`.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    p = prev[-1] if len(prev) else None
    if (p is not None and p.cp is not None and p.k < p.cp.K
            and p.end[0] == a and p.end[1] == b):
        cp, ks = p.cp, p.k
    else:
        if d is None:
            d = _exact(side).analyse(a, b)
        cp, ks = _Coupling(side, a, b, d), 0
    if ks >= cp.K:
        raise _SplitInvariantError('vertical block: no must heat at the node')
    v = _Vertical(side, cp, p, forbid, used)
    out = v.coarse(ks) if _SPLIT_COARSEN and cp.K - ks > 1 else None
    if out is not None:
        (cells, ke), leak = out, {}
    else:
        out = v.elementary(ks)
        if out is None:
            return None
        (cells, leak), ke = out, ks + 1
    if not leak and not (cp.Pm[ke] > cp.Pm[ks]).any():
        raise _SplitInvariantError('vertical block: no progress')
    return _Block('vertical', (a, b), (cp.Pm[ke].tolist(), cp.Pf[ke].tolist()),
                  cells, cp, ke, ke - ks, leak, v.knots, v.work)


def _key_block(blk, n):
    """Stage keys ``('B', n, stream, branch)`` of the streams split in core
    block `n` (a stream with one cell in the block is a trunk: None)."""
    nm = Counter(c.i for c in blk.cells)
    nf = Counter(c.j for c in blk.cells)
    bm, bf = Counter(), Counter()
    for c in blk.cells:
        if nm[c.i] > 1:
            c.km = ('B', n, c.i, bm[c.i])
            bm[c.i] += 1
        if nf[c.j] > 1:
            c.kf = ('B', n, c.j, bf[c.j])
            bf[c.j] += 1


def _drive(side, a0, strategy, cap1=False, forbid=frozenset(), work_scale=1.,
           Qmin=0.):
    """
    One core candidate from the pre-leaked root.

    Parameters
    ----------
    side : _Side
        The side.
    a0 : sequence[float]
        The pre-leaked root (`_preleak_root`); the flexes start at 0.
    strategy : str
        One of `_CORE_STRATEGIES`.
    cap1 : bool, optional
        avoid_recycle: every pair in at most one exchanger.
    forbid : collection[tuple[int, int]], optional
        Pairs never used.
    work_scale : float, optional
        Scale of the search budgets.
    Qmin : float, optional
        Exchangers below this duty count as small in the key.

    Returns
    -------
    _Candidate or None
        None only if forbidden or used pairs leave no block.

    Raises
    ------
    _SplitInvariantError
        If a node violates (R) by more than ``_SPLIT_R_TOL tolQ`` (never an
        assert) or a block cannot be completed.

    Notes
    -----
    'V' is the chain of vertical blocks. Each node is the closed-form end
    of the previous block, so (R) is inherited (Theorem V', Corollary C)
    and round-off does not accumulate; it is checked anyway, exactly.
    """
    if strategy not in _CORE_STRATEGIES:
        raise ValueError(f'core strategy {strategy!r} is not available')
    ex = _exact(side)
    lim = -_SPLIT_R_TOL * side.tolQ
    a, b = [float(x) for x in a0], [0.] * side.F
    blocks, used, leak = [], set(), [0.] * side.M
    while any(x < q for x, q in zip(a, side.Qm)):
        d = ex.analyse(a, b)
        if d.slack < lim:
            raise _SplitInvariantError(
                f'{strategy}: core node (R): slack {d.slack!r} after '
                f'{len(blocks)} blocks')
        blk = _vertical_block(side, a, b, blocks[-1:], forbid, used, d)
        if blk is None:
            return None
        _key_block(blk, len(blocks))
        blocks.append(blk)
        a, b = blk.end
        if cap1:
            used |= blk.pairs
        for i, x in blk.leak.items():
            leak[i] += x
    return _Candidate(strategy, (1, _CORE_ORDER.index(strategy)), side, a0,
                      [c for blk in blocks for c in blk.cells], leak,
                      math.fsum(blk.work for blk in blocks), Qmin, blocks)


# %% Candidates

def _merge_cells(cells, tolQ):
    """
    Exchangers of a list of cells: consecutive cells with the same must,
    flex and stage keys, contiguous on both streams (within `tolQ`), merge
    into one (the planner's continuation rule). The cells are not modified.
    """
    out, last_m, last_f = [], {}, {}
    for c in cells:
        e = last_m.get((c.i, c.km))
        if (e is not None and e is last_f.get((c.j, c.kf))
                and abs(c.a - e.a_end) <= tolQ and abs(c.b - e.b_end) <= tolQ):
            e.x += c.x
            continue
        e = _Cell(c.i, c.j, c.x, c.a, c.b, c.f, c.g, c.km, c.kf)
        out.append(e)
        last_m[c.i, c.km] = last_f[c.j, c.kf] = e
    return out


def _stages(cells):
    """Split stages: ``{(role, stream, stage): {branch: [cells]}}`` with
    role 'm' (must) or 'f' (flex)."""
    out = {}
    for c in cells:
        if c.km is not None:
            out.setdefault(('m', c.i, c.km[:-1]), {}).setdefault(
                c.km[-1], []).append(c)
        if c.kf is not None:
            out.setdefault(('f', c.j, c.kf[:-1]), {}).setdefault(
                c.kf[-1], []).append(c)
    return out


def _remix(role, branches, tolQ):
    """Split position, mix position and isothermality of a stage (parent
    positions). A must splits at its far end and mixes toward the pinch, a
    flex splits at its pinch-side start and mixes at the far end; the mix
    is isothermal if every branch ends within ``_ISO_TOL tolQ`` of it."""
    cs = [c for cells in branches.values() for c in cells]
    D = math.fsum(c.x for c in cs)
    if role == 'm':
        p0 = max(c.a_end for c in cs)
        pm = p0 - D
        ends = [p0 - math.fsum(c.x for c in cells) / cells[0].f
                for cells in branches.values()]
    else:
        p0 = min(c.b for c in cs)
        pm = p0 + D
        ends = [p0 + math.fsum(c.x for c in cells) / cells[0].g
                for cells in branches.values()]
    return p0, pm, all(abs(e - pm) <= _ISO_TOL * tolQ for e in ends)


def _sig(v):
    """`v` rounded to 9 significant digits."""
    return float(f'{v:.9g}')


def _signature(side, merged, stages):
    """Knot-independent identity of a split network (see `_Candidate`)."""
    musts, flexes = side.musts, side.flexes
    by_stream = defaultdict(list)
    for (role, s, sid), br in stages.items():
        cs = [c for cells in br.values() for c in cells]
        start = min(c.a if role == 'm' else c.b for c in cs)
        by_stream[role, s].append((start, sid))
    ordinal = {}
    for (role, s), lst in by_stream.items():
        for n, (_, sid) in enumerate(sorted(lst)):
            ordinal[role, s, sid] = n

    def canon(role, item):
        # a branch by its partners in position order and its fraction
        bk, cs = item
        if role == 'm':
            return (tuple(flexes[c.j].stream for c in
                          sorted(cs, key=lambda c: c.a)), _sig(cs[0].f), bk)
        return (tuple(musts[c.i].stream for c in
                      sorted(cs, key=lambda c: c.b)), _sig(cs[0].g), bk)
    branch, fracs = {}, []
    for (role, s, sid), br in stages.items():
        ranked = sorted(br.items(), key=lambda item: canon(role, item))
        for n, (bk, _) in enumerate(ranked):
            branch[role, s, sid, bk] = n
        stream = (musts if role == 'm' else flexes)[s].stream
        fracs.append((role, stream, ordinal[role, s, sid],
                      tuple(canon(role, item)[1] for item in ranked)))

    def path(role, s, key):
        if key is None:
            return None
        sid = key[:-1]
        return ordinal[role, s, sid], branch[role, s, sid, key[-1]]
    cells = [(musts[c.i].stream, flexes[c.j].stream, path('m', c.i, c.km),
              path('f', c.j, c.kf)) for c in merged]
    cells.sort(key=lambda r: (r[0], r[1], r[2] or (), r[3] or ()))
    return tuple(cells), tuple(sorted(fracs))


class _Candidate:
    """
    A split plan of one side: its cells, verified, with the selection key
    and the network signature.

    Parameters
    ----------
    name : str
        Generator, e.g. 'V' or 'S:partner'.
    order : tuple
        Tie-break rank: ``(0, rule index)`` for Stage S, ``(1, index in
        _CORE_ORDER)`` for the core.
    side : _Side
        The side.
    a0 : sequence[float]
        The pre-leaked root.
    cells : list[_Cell]
        The cells (stage keys set).
    leak_by_must : sequence[float], optional
        Must heat leaked by position round-off, per must.
    work : float, optional
        Work spent.
    Qmin : float, optional
        Exchangers below this duty count as small.
    blocks : sequence[_Block], optional
        The core blocks, if any.

    Attributes
    ----------
    units : int
        Exchangers (`_merge_cells`).
    stages, branches : int
        Split stages and their branches (one splitter and one mixer each).
    mixers : int
        The most split stages (mixers) on one stream.
    small : int
        Exchangers below `Qmin` or with a fraction below
        `_SPLIT_MIN_FRACTION`.
    mixbad : int
        Non-isothermal remixes followed by a process exchanger of the same
        stream (a must mixes toward the pinch, a flex away from it; a
        utility does not count), plus curved streams (more than two knots)
        with more than `_SPLIT_MIX_CAP` mixers on the side.
    touch : int
        Exchangers parallel at the minimum approach along a segment,
        counted only on sides with a curved stream.
    meta : dict
        ``candidate``, ``stages``, ``branches``, ``leak`` and ``small``
        (``[(hot, cold, Q)]`` below `Qmin`) for the side's info.
    signature : tuple
        The sorted exchangers ``(must stream, flex stream, must branch,
        flex branch)`` (a branch is ``(split ordinal on the stream, branch
        ordinal)`` or None), and each split's fractions to 9 significant
        digits. It holds no duty or position, so it identifies the same
        network across knot refinements and generators.
    excluded : bool
        Set by the caller (exclusion by signature).

    Raises
    ------
    _SplitInvariantError
        If an exchanger fails (C) (`_cell_margin`).
    """
    __slots__ = ('name', 'order', 'cells', 'a0', 'leak_by_must', 'work',
                 'blocks', 'excluded', 'units', 'stages', 'branches',
                 'mixers', 'small', 'mixbad', 'touch', 'meta', 'signature')

    def __init__(self, name, order, side, a0, cells, leak_by_must=None,
                 work=0., Qmin=0., blocks=()):
        self.name, self.order = name, order
        self.cells = list(cells)
        self.a0 = [float(x) for x in a0]
        self.leak_by_must = ([0.] * side.M if leak_by_must is None
                             else [float(x) for x in leak_by_must])
        self.work = float(work)
        self.blocks = list(blocks)
        self.excluded = False
        tolQ, tolP = side.tolQ, side.tolP
        merged = _merge_cells(self.cells, tolQ)
        curved = any(c.n > 2 for c in side.musts + side.flexes)
        small, n_small, touch = [], 0, 0
        for c in merged:
            margin, t = _cell_margin(side, c)
            if not margin >= -tolP:
                raise _SplitInvariantError(
                    f'{name}: {c!r} fails (C) by {margin!r}')
            if c.x < Qmin:
                small.append((*side.hot_cold(c.i, c.j), c.x))
            n_small += c.x < Qmin or min(c.f, c.g) < _SPLIT_MIN_FRACTION
            touch += curved and t
        stages = _stages(merged)
        mixers = Counter((role, s) for role, s, _ in stages)
        mixbad = 0
        for (role, s, _), br in stages.items():
            p0, pm, iso = _remix(role, br, tolQ)
            if iso:
                continue
            # an exchanger of the stream after the mix (a utility is not
            # one): a must mixes toward the pinch, at `pm`, so a cell of it
            # below `pm`; a flex mixes at its far end, so a cell of it
            # outside the stage beyond its split start `p0`
            inside = {id(c) for cs in br.values() for c in cs}
            mixbad += any(id(c) not in inside and (
                c.i == s and c.a_end <= pm + tolQ if role == 'm'
                else c.j == s and c.b > p0) for c in merged)
        for (role, s), n in mixers.items():
            curve = (side.musts if role == 'm' else side.flexes)[s]
            mixbad += curve.n > 2 and n > _SPLIT_MIX_CAP
        self.units = len(merged)
        self.stages = len(stages)
        self.branches = sum(len(br) for br in stages.values())
        self.mixers = max(mixers.values(), default=0)
        self.small, self.mixbad, self.touch = n_small, mixbad, touch
        self.meta = dict(candidate=name, stages=self.stages,
                         branches=self.branches,
                         leak=math.fsum(self.leak_by_must), small=small)
        self.signature = _signature(side, merged, stages)

    def key(self):
        """Selection key, smaller is better: small exchangers, bad remixes,
        touching exchangers, units + extra branches + split stages, the most
        mixers on one stream, candidate order."""
        return (self.small, self.mixbad, self.touch,
                self.units + (self.branches - self.stages) + self.stages,
                self.mixers, self.order)

    def __repr__(self):
        return f'<_Candidate {self.name} key={self.key()}>'


# %% Portfolio and selection

def _excluded(split, side_name, cand):
    """True if the network signature of `cand` is excluded on its side
    (exclusion is by network, never by generator name)."""
    return cand.signature in split['exclude'].get(side_name, ())


def _side_plan(side, best, cands, reasons, a0, delta, errors, work, proof):
    """The side plan of the chosen candidate `best`. Each must's gap is its
    pre-leak plus its leak, so the penalty is truthful."""
    gaps = [a0[i] + best.leak_by_must[i] for i in range(side.M)]
    info = dict(best.meta, signature=best.signature, candidates=reasons,
                preleak=delta, errors=errors)
    return _SidePlan([], gaps, 'mer', 'split-' + best.name,
                     work + math.fsum(c.work for c in cands), proof,
                     cells=best.cells, split=info, units=best.units)


def _split_side(side, d, proof, cap1, forbid, work_scale, work, split):
    """
    Split plan of a side that no unsplit network serves at MER.

    Parameters
    ----------
    side : _Side
        The side.
    d : _Residual
        The search's analysis of the root, which triggered the split. The
        core analyses the root again, exactly (`_preleak_root`).
    proof : dict or None
        The root proof, kept in the side plan (it says why the side
        split), or None if the unsplit search left a penalty.
    cap1 : bool
        avoid_recycle: every pair in at most one exchanger.
    forbid : collection[tuple[int, int]]
        Pairs never used (avoid_recycle: those of the other side).
    work_scale : float
        Scale of the search budgets.
    work : float
        Work already spent on the side.
    split : dict
        ``Qmin`` (exchangers below it count as small in the key; none is
        dropped), ``exclude`` (per side, excluded network signatures) and
        ``prefer`` (per side, the candidate tried first).

    Returns
    -------
    _SidePlan or None
        A side plan with status 'mer', method ``'split-<candidate>'``, the
        cells of the chosen candidate and the ``split`` info; None only if
        no candidate exists: the root deficit exceeds `_preleak_max`, or
        avoid_recycle's pairs left no block.

    Notes
    -----
    The core starts from the pre-leaked root ``a0 = P(delta)`` (Lemma P).
    The preferred candidate (the previous refine round's pick) is generated
    first and taken as is unless its signature is excluded. Otherwise every
    generator runs (the preferred one is not run again) and the smallest
    `_Candidate.key` wins among the candidates not excluded or, if all are
    excluded, among all of them: a side's last candidate is never excluded.
    A generator that raises `_SplitInvariantError` is recorded in
    ``errors`` and the others remain.
    """
    delta, a0 = _preleak_root(side)
    core = delta <= _preleak_max(side)
    prefer = split['prefer'].get(side.name)
    cands, errors, reasons = [], [], {}

    def generate(name):
        try:
            c = _drive(side, a0, name, cap1, forbid, work_scale,
                       split['Qmin'])
        except _SplitInvariantError as e:
            errors.append(f'{name}: {e}')
            reasons[name] = 'error'
            return None
        if c is None:
            reasons[name] = 'forbidden pairs'
            return None
        c.excluded = _excluded(split, side.name, c)
        reasons[name] = 'excluded' if c.excluded else c.key()
        cands.append(c)
        return c
    if core and prefer in _CORE_STRATEGIES:
        c = generate(prefer)
        if c is not None and not c.excluded:
            return _side_plan(side, c, cands, reasons, a0, delta, errors,
                              work, proof)
    if core:
        for strategy in _CORE_STRATEGIES:
            if strategy != prefer:
                generate(strategy)
    if not cands:
        return None
    live = [c for c in cands if not c.excluded] or cands
    return _side_plan(side, min(live, key=_Candidate.key), cands, reasons,
                      a0, delta, errors, work, proof)


# %% Plan records

class Split:
    """
    One split of a stream in a plan (`hensmith._planner.Plan.splits`): the
    stream divides into parallel branches that re-join in a mixer.

    Attributes
    ----------
    stream : int
        The stream (index into the plan's streams).
    side : {'above', 'below'}
        The side of the pinch.
    key : tuple
        The planner's stage id, e.g. ``('B', block, local stream)``.
    fractions : tuple[float]
        Flow fraction of every branch; they sum to 1 within round-off.
    branches : list[list[int]]
        The exchangers of every branch (indices into ``Plan.exchangers``),
        in flow order. A branch whose exchangers were all dropped by the
        safety net (a bug) is empty: its flow bypasses.
    H_split, H_mix : float
        Stream enthalpies (the scale of the exchangers' enthalpies) where
        the stream splits and where the branches re-join: ``H_mix = H_split
        -/+ sum of the branch duties`` (hot/cold), so the fractions add no
        round-off to the stream's heat balance.
    isothermal : bool
        Every branch ends within ``_ISO_TOL tolQ`` (parent-equivalent) of
        ``H_mix``: the branches re-join at one temperature.
    """
    __slots__ = ('stream', 'side', 'key', 'fractions', 'branches',
                 'H_split', 'H_mix', 'isothermal')

    def __init__(self, stream, side, key, fractions, branches):
        self.stream, self.side, self.key = stream, side, key
        self.fractions, self.branches = fractions, branches
        self.H_split = self.H_mix = math.nan
        self.isothermal = None

    def __repr__(self):
        return (f'<Split stream={self.stream} {self.side} key={self.key} '
                f'fractions=({", ".join(f"{f:.4g}" for f in self.fractions)})'
                f' isothermal={self.isothermal}>')


def _record(side, c):
    """
    The plan record of cell `c` of `side`. Its place on each stream,
    ``(rank, role, stage key, start, end)`` in `_kh` and `_kc`, holds the
    side's rank in that stream's flow (a hot stream runs above, then
    below; a cold stream below, then above), the stream's role ('m' must,
    'f' flex), its stage key (None on a trunk) and its parent range.
    """
    e = Exchanger()
    e.side = side.name
    e.hot, e.cold = side.hot_cold(c.i, c.j)
    e.Q = c.x
    must = ('m', c.km, c.a, c.a_end)
    flex = ('f', c.kf, c.b, c.b_end)
    if side.name == 'above':   # the hot stream is the must
        e._kh, e._kc = (0,) + must, (1,) + flex
        e.hot_frac, e.cold_frac = c.f, c.g
    else:                      # the cold stream is the must
        e._kh, e._kc = (1,) + flex, (0,) + must
        e.hot_frac, e.cold_frac = c.g, c.f
    return e


def _paths(recs, N, ghosts=()):
    """
    Flow order of every stream from the records' places.

    Per stream and side, the trunk records and the stages (one `Split`
    each) occupy disjoint parent ranges, so one sort by the midpoint puts
    them pinch outward; musts flow toward the pinch (the order reversed),
    flexes away from it. Within a split, branches run in stage-key order,
    each branch's records in flow order. `ghosts` (records dropped by the
    safety net) keep their branches, empty, so the fractions still sum to
    1. Sets the records' `hot_branch` and `cold_branch`.

    Returns
    -------
    stages : dict[int, list[int]]
        Flat flow order of every stream.
    paths : dict[int, list[int or Split]]
        Flow order with each split as one item.
    splits : list[Split]
        In stream order, then flow order.
    """
    trunks = defaultdict(list)   # stream -> [(order, record)]
    stages_ = defaultdict(dict)  # stream -> {(rank, role, stage id, side):
    #                               {branch: [fraction, [(lo, hi, Q, n)]]}}
    for live, lst in ((True, recs), (False, ghosts)):
        for n, e in enumerate(lst):
            e.hot_branch = e.cold_branch = None
            for j, (rank, role, key, lo, hi), f in (
                    (e.hot, e._kh, e.hot_frac), (e.cold, e._kc, e.cold_frac)):
                sg = -1. if role == 'm' else 1.
                if key is None:
                    if live:
                        trunks[j].append(((rank, sg * (lo + hi), 0, n), n))
                    continue
                br = stages_[j].setdefault((rank, role, key[:-1], e.side), {})
                br.setdefault(key[-1], [f, []])[1].append(
                    (lo, hi, e.Q, n if live else None))
    stages, paths, splits = {}, {}, []
    for j in range(N):
        items = list(trunks[j])
        for k, ((rank, role, sid, side), br) in enumerate(
                stages_[j].items()):
            # the stage's parent range: a must splits at its far end and
            # mixes toward the pinch, a flex splits at its start
            members = [m for _, ms in br.values() for m in ms]
            D = math.fsum(m[2] for m in members)
            if role == 'm':
                hi = max(m[1] for m in members)
                mid, sg = 2. * hi - D, -1.
            else:
                mid, sg = 2. * min(m[0] for m in members) + D, 1.
            branches, fr = [], []
            for b in sorted(br):
                f, ms = br[b]
                ms.sort(key=lambda m: sg * (m[0] + m[1]))
                branches.append([m[3] for m in ms if m[3] is not None])
                fr.append(f)
            items.append(((rank, sg * mid, 1, k),
                          Split(j, side, sid, tuple(fr), branches)))
        items.sort(key=lambda it: it[0])
        path, flat = [], []
        for _, item in items:
            if isinstance(item, Split):
                for b, ns in enumerate(item.branches):
                    for n in ns:
                        e = recs[n]
                        if e.hot == j:
                            e.hot_branch = (len(splits), b)
                        else:
                            e.cold_branch = (len(splits), b)
                    flat += ns
                splits.append(item)
            else:
                flat.append(item)
            path.append(item)
        stages[j], paths[j] = flat, path
    return stages, paths, splits


def _walk_paths(curves, recs, paths, N, tolQ):
    """
    Flow-order walk of a plan with splits (the split-aware `_walk`).

    A trunk record moves the stream's enthalpy by its duty. A split
    records ``H_split``, walks every branch from it by ``Q / f`` per record
    (parent-equivalent enthalpies), sets ``H_mix = H_split -/+ sum Q`` and
    whether the remix is isothermal. Fills the records' enthalpies and
    1-based positions in the flat flow order; returns the utility of every
    stream (at its outlet end).
    """
    utility = [0.] * N
    for j in range(N):
        c = curves[j]
        if c is None:
            continue
        hot = c.hot
        H = c.H_hi if hot else c.H_lo
        pos = 0
        for item in paths[j]:
            if not isinstance(item, Split):
                pos += 1
                H = _step(recs[item], hot, H, 1., pos)
                continue
            item.H_split = H
            ends = []
            for f, ns in zip(item.fractions, item.branches):
                Hb = H
                for n in ns:
                    pos += 1
                    Hb = _step(recs[n], hot, Hb, f, pos)
                ends.append(Hb)
            D = math.fsum(recs[n].Q for ns in item.branches for n in ns)
            H = H - D if hot else H + D
            item.H_mix = H
            item.isothermal = all(abs(x - H) <= _ISO_TOL * tolQ for x in ends)
        utility[j] = H - c.H_lo if hot else c.H_hi - H
    return utility


def _step(e, hot, H, f, pos):
    """Walk record `e` on its hot or cold stream from enthalpy `H` on a
    branch of fraction `f`; returns the enthalpy after it."""
    if hot:
        e.H_hot_in = H
        H = H - e.Q / f
        e.H_hot_out = H
        e.hot_seq = pos
    else:
        e.H_cold_in = H
        H = H + e.Q / f
        e.H_cold_out = H
        e.cold_seq = pos
    return H


def _approach_violation_split(ch, cc, e):
    """
    Smallest ``T*_hot - T*_cold`` inside branch exchanger `e` (shifted
    scale: >= 0 is feasible), like `_approach_violation`: duty ``t`` in
    ``[0, Q]`` is at the parent-equivalent enthalpies ``H_hot_in -
    t/hot_frac`` and ``H_cold_out - t/cold_frac``; a knot ``H`` of either
    curve inside the exchanger at ``t = (H_hot_in - H) hot_frac`` or
    ``(H_cold_out - H) cold_frac``.
    """
    Q, fh, fc = e.Q, e.hot_frac, e.cold_frac
    hin, cout = e.H_hot_in, e.H_cold_out
    Hh = ch.H[(ch.H > e.H_hot_out) & (ch.H < hin)]
    Hc = cc.H[(cc.H > e.H_cold_in) & (cc.H < cout)]
    t = np.concatenate(([0., Q], (hin - Hh) * fh, (cout - Hc) * fc))
    return float((np.interp(hin - t / fh, ch.H, ch.T)
                  - np.interp(cout - t / fc, cc.H, cc.T)).min())


def _split_records(sides, plans, curves, N, Qmin, tolQ):
    """
    Plan records of a network with split sides (`plan_network`).

    A side without cells converts its pieces as `plan_network` does,
    including the `Qmin` filter; a split side's cells merge into its
    exchangers (`_merge_cells`) and are never dropped for `Qmin`. The walk
    (`_paths`, `_walk_paths`) and the safety net are those of
    `plan_network` with fractions (`_approach_violation_split`); a record
    the safety net drops (a bug) is reported in `dropped`.

    Returns
    -------
    recs, qmin_dropped, dropped, stages, utility, min_dT, splits, paths
        As in `plan_network`, plus the splits and the paths.
    """
    recs, qmin_dropped = [], []
    for name in ('above', 'below'):
        if name not in plans:
            continue
        side, p = sides[name], plans[name]
        if p.cells:
            cells = _merge_cells(p.cells, side.tolQ)
        else:
            cells = []
            for i, j, a0, b0, Q in _merge(p.pieces):
                if Q < Qmin:
                    qmin_dropped.append((name, *side.hot_cold(i, j), Q))
                    continue
                cells.append(_Cell(i, j, Q, a0, b0))
        recs += [_record(side, c) for c in cells]
    dropped, ghosts = [], []
    while True:
        stages, paths, splits = _paths(recs, N, ghosts)
        utility = _walk_paths(curves, recs, paths, N, tolQ)
        worst = None
        min_dT = math.inf
        for n, e in enumerate(recs):
            v = _approach_violation_split(curves[e.hot], curves[e.cold], e)
            min_dT = min(min_dT, v)
            if v < -_APPROACH_TOL and (worst is None or v < worst[0]):
                worst = (v, n)
        if worst is None:
            break
        e = recs.pop(worst[1])
        ghosts.append(e)
        dropped.append((e.side, e.hot, e.cold, e.Q, worst[0]))
    return (recs, qmin_dropped, dropped, stages, utility, min_dT, splits,
            paths)
