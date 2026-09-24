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
:func:`_cell_margin` evaluates the approach there, and every cell of a split
plan is verified this way before the plan is accepted. It establishes (C)
at ``tolP``; the planner's ``_max_duty`` does not quite: it takes linear
slopes within a relative ``_SLOPE_EQ`` as parallel, so a converging pair may
close by ``_SLOPE_EQ`` times its level span. The split path's searches
therefore run on a `_StrictSide`, whose `_max_duty_strict` applies that
shortcut only where the closure stays within ``tolP``: on branch-scaled
curves, every piece they place passes `_cell_margin`.

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

Stage S (pinch splits)
----------------------
At a tight cut where the pinch rules fail (`_cut_sets`), each demand (a
must leaving the cut outward, or a flex reaching it from below) needs its
own capacity branch with ``CP_cap >= CP_dem``. `_cut_transport` splits the
demands and capacities there by one of `_SPLIT_RULES`: demands whole,
capacities whole, Linnhoff and Hindmarsh's minimum-cell structure, or the
north-west corner with the CP slack spread uniformly or not at all. Its
branches are whole-side branches, so by Lemma R the branched side keeps
the cascade, and one `rules_violation` call at its pre-leaked root screens
the split (`_pinch_split`); on a double pinch the same rule is applied at
the cut that still fails. The branched side is an ordinary side for the
planner's search: `_stage_s` plans it from its pre-leaked root with the
first passes of the planner's DFS, unchanged but for the strict max duty
of Lemma 1 (a `_StrictSide`), under a unit bound from the
best candidate so far and one work budget for all rules
(`_SPLIT_S_WORK`). Its pieces become cells in parent coordinates
(`_s_cells`); a flex branch the DFS left unused is folded into its used
siblings (**Lemma F**: a larger fraction moves every flex position toward
the pinch, where the flex levels are no higher, so every cell stays
feasible), and each must's residual of at most ``tolQ`` (the DFS's own
tolerance) is swept into its far-end cell, so every must is served
exactly; Stage S books no leak. Must branches split at the must's inlet
and remix isothermally at the pinch; flex branches split at the pinch and
remix at the side's far end, before the stream's utility only. The best
Stage S candidate gets the planner's unit improvements.

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

**Strategies.** The driver (`_drive`) builds one candidate per strategy of
`_CORE_STRATEGIES`. 'V' chains vertical blocks. 'L' adds pinch blocks
(`_pinch_block`) where the rules fail outward: Linnhoff and Hindmarsh's
pinch split of the cut (the 'mincell' CP transport). Each must advances by
one extent on all of its branches, the largest its cells allow, and the
block is scaled back by the largest lambda (bisected) whose end node
satisfies (R), by the same exact check as every node. 'T' completes the
side with a DFS tail (`_tail`) from a node where the rules hold: the
planner's own search from ``(a, b)`` at a fraction of its budgets (on a
`_StrictSide`, Lemma 1), its must residuals swept as in Stage S. A tail never yields a node, so every
node of a core candidate is the pre-leaked root, a vertical block's
closed-form end or a pinch block's accepted end.

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
its pre-leak plus its leak, so the side's penalty is truthful. A side with
no candidate falls back to the planner's best effort, which keeps the
attempt's info (candidate None, the reasons and the errors).
`_split_records` turns the side plans into the plan's records: every
split stage becomes a `Split`, branch enthalpies are parent-equivalent (a
branch exchanger moves its branch by ``Q/f``), a mix is written with the
duties (``H_mix = H_split -/+ sum Q``), and a stream's flow order lists
each split as one item (``Plan.paths``) or flattened (``Plan.stages``).
"""
import math
from collections import Counter, defaultdict, deque
from itertools import accumulate, combinations

import numpy as np

from ._planner import (Exchanger, _APPROACH_TOL, _LevelCurve, _REL_Q,
                       _SCHEDULE, _SLOPE_EQ, _Search, _Side, _SidePlan,
                       _THRESHOLD_TOL, _WORK_EVENT, _combine_cap,
                       _improve_units, _max_duty, _merge, _slope1,
                       _units_guard)

__all__ = ()

_SPLIT_ULP = 8. * np.finfo(float).eps   # round-off, relative to max(X, 1)
_SPLIT_R_TOL = 1e-3          # core nodes: (R) slack and leak cap, x tolQ
_SPLIT_MIN_FRACTION = 1e-3   # smallest branch fraction of a coarsened block
_SPLIT_COARSEN = True        # False: elementary vertical blocks only
_SPLIT_MIX_CAP = 2           # mixers per curved stream and side (the key)
_ISO_TOL = 10.               # isothermal remix, x tolQ
_CORE_ORDER = ('V', 'LV', 'VT', 'LVT')   # candidate order of the core
_CORE_STRATEGIES = _CORE_ORDER   # the core strategies `_drive` runs
_SPLIT_BISECT = 30           # pinch block: bisection steps on lambda
_SPLIT_LAMBDA_MIN = 1e-3     # pinch block: smallest lambda accepted
_SPLIT_TAIL_WORK = 0.2       # core tail: share of the search budgets
_SPLIT_TAIL_TRIES = 3        # core tails per strategy
_SPLIT_RULES = ('partner', 'demand', 'mincell', 'nw-rho-desc', 'nw-rho-asc',
                'nw-exact-desc', 'nw-exact-asc')   # Stage S, in order
_SPLIT_CUTS = 3              # cuts per Stage S rule (double pinches)
_SPLIT_PASSES = 3            # Stage S and tails: first `_SCHEDULE` passes
_SPLIT_S_WORK = 30000.       # Stage S: work of all rules on a side
_SPLIT_FIRST_WINS = False    # True: the first live candidate wins (runtime)
_MINCELL_WORK = 20000        # search nodes of one 'mincell' transport
_CP_TOL = 1e-12              # CP compatibility, as in `rules_violation`


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


def _max_duty_strict(cm, a, cf, b, limit, tolP):
    """
    The planner's `_max_duty`, with its parallel shortcut only where it
    holds: every duty satisfies (C) at `tolP`.

    `_max_duty` takes two linear curves whose slopes agree within a
    relative `_SLOPE_EQ` as parallel and returns `limit`, so that equal
    slopes are not separated by rounding. A pair that converges (flex
    slope ``sf`` above must slope ``sm``) then closes by up to ``_SLOPE_EQ
    sf limit`` of level, beyond `tolP` over a long enough span (a branch
    whose CP a cut rule set to its partner's, on data with CPs equal to
    1e-9). Here the shortcut applies only if the gap at `limit`, ``g0 -
    (sf - sm) limit``, is at least ``-tolP``; otherwise the duty is that of
    any converging pair, ``max(g0, 0)/(sf - sm)`` (the gap closes to zero).
    """
    if limit > 0. and cm.linear and cf.linear:
        sm, sf = _slope1(cm), _slope1(cf)
        if sf * (1. - _SLOPE_EQ) <= sm < sf:
            g0 = cm.at(a) - cf.at(b)
            if g0 < -tolP:
                return 0.
            if g0 - (sf - sm) * limit >= -tolP:
                return limit
            return min(limit, max(g0, 0.) / (sf - sm))
    return _max_duty(cm, a, cf, b, limit, tolP)


class _StrictSide(_Side):
    """A `_Side` whose search places only pieces that satisfy (C) at tolP
    (`_max_duty_strict`)."""
    max_duty = staticmethod(_max_duty_strict)


def _strict(side):
    """`side` for the split path's searches (`_StrictSide`)."""
    if isinstance(side, _StrictSide):
        return side
    return _StrictSide(side.name, side.musts, side.flexes, side.tolQ,
                       side.tolP)


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


class _ExactSide(_Side):
    """A `_Side` whose residual arrays are exact to round-off (`_exact`)."""

    @staticmethod
    def _R(pieces, n, levels):
        """Inclusive (level <= L) and exclusive (level < L) residual heat of
        every stream at every level, piece by piece: a sloped piece adds
        ``clip((L - lo)/(hi - lo), 0, 1) ln``, a flat one ``ln`` from its
        level on (inclusive) or above it (exclusive). Each share lies in
        ``[0, ln]``, so a stream's residual never exceeds its heat."""
        own, lo, hi, ln = pieces
        nL = levels.size
        Ri, Re = np.zeros((n, nL)), np.zeros((n, nL))
        if own.size == 0 or nL == 0:
            return Ri, Re
        w = hi - lo
        flat = w <= 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            fi = np.clip((levels - lo[:, None]) / w[:, None], 0., 1.)
        fe = fi.copy()
        fi[flat] = levels >= lo[flat, None]
        fe[flat] = levels > lo[flat, None]
        np.add.at(Ri, own, fi * ln[:, None])
        np.add.at(Re, own, fe * ln[:, None])
        return Ri, Re


def _exact(side):
    """
    `side` with no heat tolerance in its residual arrays, which are exact
    to round-off.

    `_Side.analyse` omits every stream with at most ``tolQ`` of residual
    heat: the search's tolerance. At a core node that would drop a stream's
    last sliver of heat (up to ``tolQ``, far more than round-off), move the
    coupling's breakpoints and misreport the slack of (R) by that much, so
    the core analyses its nodes exactly.

    The search's residual arrays of a stream with several pieces are
    cumulative sums of the pieces' slopes and offsets, evaluated as
    ``slope sum * (L - offset sum)``. On a near-flat piece (a glide of
    1e-7 K over a heat of 10 has a slope of 1e8) those sums cancel
    catastrophically and the residual drifts with the level, beyond the
    stream's own heat. The core's side (`_ExactSide`) adds each piece's
    clipped share directly instead. The search keeps its arrays, so the
    unsplit planner is unchanged.
    """
    return _ExactSide(side.name, side.musts, side.flexes, 0., side.tolP)


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
    kind : {'vertical', 'pinch', 'tail'}
        How the block was built (`_vertical_block`, `_pinch_block`,
        `_tail`).
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


def _pinch_block(side, a, b, forbid=frozenset(), used=frozenset(),
                 viol=None):
    """
    A pinch block from node ``(a, b)`` (strategy 'L', see 'The core' in
    the module docstring): Linnhoff and Hindmarsh's pinch split at an
    outward violation of the rules, as one block of the core.

    Parameters
    ----------
    side : _Side
        The side.
    a, b : sequence[float]
        The node; it satisfies (R).
    forbid : collection[tuple[int, int]], optional
        Pairs never used.
    used : collection[tuple[int, int]], optional
        Pairs of earlier blocks (avoid_recycle), not used again.
    viol : dict, optional
        ``side.rules_violation`` at the node (on the search's analysis), if
        already computed.

    Returns
    -------
    _Block or None
        None if the rules do not fail outward at the node, a demand at the
        cut is flat, no transport serves the cut with every pair allowed
        and every demand moving by more than ``tolQ``, or no lambda of at
        least `_SPLIT_LAMBDA_MIN` is accepted.

    Notes
    -----
    At the violating cut (`_cut_sets`) the CP transport is 'mincell', else
    'nw-rho-desc' (`_cut_transport`). A cell ``(i, j, load)`` is a branch
    of must ``i`` of fraction ``f = load / m_i`` facing a branch of flex
    ``j`` of fraction ``g = load / sum load_j`` (`_cut_fractions`), so the
    flex branch's CP is at least the must branch's. Every cell starts at
    the node. Must ``i`` advances by one extent ``h_i`` on all of its
    branches, so its remix is isothermal: the smallest over its cells of
    ``x_max / f``, with ``x_max`` the planner's `_max_duty` on the
    branch-scaled curves (Lemma B), capped by the must's residual and the
    flex branch's room ``g (Qf_j - b_j)``. A must within ``tolQ`` of its
    end ticks off: ``h_i`` is its residual exactly, its last cell takes
    the residual less its other cells, and its node is its end. The flex
    branches end at different positions: a non-isothermal remix, which
    the key counts if the flex has a later exchanger.

    The block is accepted at lambda = 1, else at the largest lambda found
    by `_SPLIT_BISECT` bisection steps, with the cells scaled to ``lambda
    f h_i``, if every cell passes (C) (`_cell_margin`) and the end node,
    in parent coordinates (a split leaves the composites unchanged, Lemma
    R), satisfies (R) within ``_SPLIT_R_TOL tolQ`` by an exact analysis
    (`_exact`, as the driver checks its nodes). Every accepted lambda is
    checked, so nothing depends on monotonicity in lambda.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if viol is None:
        viol = side.rules_violation(a, b, side.analyse(a, b))
    if viol is None or viol['rule'] != 'outward':
        return None
    sets = _cut_sets(side, a, b, viol['level'], viol['cut'], 'outward')
    if sets is None:
        return None
    tolQ, tolP = side.tolQ, side.tolP
    Qm, Qf = side.Qm, side.Qf
    ex = _exact(side)
    lim = -_SPLIT_R_TOL * tolQ
    no = frozenset(forbid) | frozenset(used)
    work = 0.

    def extents(rows):
        # h_i (tick-offs exact) or None if a demand cannot move
        nonlocal work
        h, tick = {}, {}
        for i, cs in rows.items():
            cm, rem = side.musts[i], Qm[i] - a[i]
            hi = rem
            for j, f, g in cs:
                cf = side.flexes[j]
                bm = cm if f == 1. else _branch_curve(cm, f, 'must')
                bf = cf if g == 1. else _branch_curve(cf, g, 'flex')
                work += _WORK_EVENT
                x = _max_duty_strict(bm, f * a[i], bf, g * b[j],
                                     min(f * rem, g * (Qf[j] - b[j])), tolP)
                hi = min(hi, x / f)
            if not hi > tolQ:
                return None
            tick[i] = hi >= rem - tolQ
            h[i] = rem if tick[i] else hi
        return h, tick

    def at(rows, h, tick, lam):
        # the cells and end node at `lam`, or None if not accepted
        nonlocal work
        cells, a2, b2 = [], list(a), list(b)
        for i, cs in rows.items():
            if lam == 1. and tick[i]:
                xs = [f * h[i] for _, f, _ in cs]
                xs[-1] = h[i] - math.fsum(xs[:-1])
                a2[i] = Qm[i]
            else:
                t = lam * h[i]
                xs = [f * t for _, f, _ in cs]
                a2[i] = min(a[i] + t, Qm[i])
            cells += [_Cell(i, j, x, a[i], b[j], f, g)
                      for (j, f, g), x in zip(cs, xs)]
        for j in {c.j for c in cells}:
            b2[j] = min(b[j] + math.fsum(c.x for c in cells if c.j == j),
                        Qf[j])
        work += 1. + _WORK_EVENT * len(cells)
        if not all(c.x > 0. and _cell_margin(side, c)[0] >= -tolP
                   for c in cells):
            return None
        if not ex.analyse(a2, b2).slack >= lim:
            return None
        return cells, a2, b2
    tried = []
    for rule in ('mincell', 'nw-rho-desc'):
        loads = _cut_transport(*sets, rule)
        if not loads or loads in tried:
            continue
        tried.append(loads)
        if any((i, j) in no for i, j, _ in loads):
            continue
        rows = defaultdict(list)   # must -> [(flex, f, g)], in order
        for (i, j, _), (f, g) in zip(loads, _cut_fractions(loads)):
            rows[i].append((j, f, g))
        ext = extents(rows)
        if ext is None:
            continue
        res = at(rows, *ext, 1.)
        if res is None:
            lo, hi = 0., 1.
            for _ in range(_SPLIT_BISECT):
                mid = 0.5 * (lo + hi)
                r = at(rows, *ext, mid)
                if r is None:
                    hi = mid
                else:
                    lo, res = mid, r
            if res is None or lo < _SPLIT_LAMBDA_MIN:
                continue
        cells, a2, b2 = res
        return _Block('pinch', (a, b), (a2, b2), cells, work=work)
    return None


def _tail(side, a, b, cap1=False, forbid=frozenset(), work_scale=1.):
    """
    A DFS tail from core node ``(a, b)`` (strategy 'T', see 'The core' in
    the module docstring): the planner's unsplit search completes the side
    from there, with the strict max duty of Lemma 1 (`_StrictSide`).

    Parameters
    ----------
    side : _Side
        The side.
    a, b : sequence[float]
        The node: it satisfies (R) and the pinch rules.
    cap1 : bool, optional
        avoid_recycle: every pair in at most one exchanger.
    forbid : collection[tuple[int, int]], optional
        Pairs never used (with `cap1`, also those of the earlier blocks).
    work_scale : float, optional
        Scale of the search budgets.

    Returns
    -------
    block : _Block or None
        The tail as the candidate's last block (kind 'tail', unsplit cells,
        ending with every must at its end), or None.
    work : float
        Work spent.

    Notes
    -----
    As `_plan_side`, but from ``(a, b)`` and at `_SPLIT_TAIL_WORK` of the
    budgets: the first `_SPLIT_PASSES` passes of `_SCHEDULE`, then
    `_improve_units` and `_units_guard` on the first success. The DFS
    completes a must within ``tolQ``; its residual is swept into its
    far-end cell as in Stage S (`_s_cells`, with every item a trunk), so
    every must is served exactly, or the tail is rejected. A tail returns
    a complete plan or nothing, never a node.
    """
    a, b = [float(x) for x in a], [float(x) for x in b]
    forbid = frozenset(forbid)
    scale = _SPLIT_TAIL_WORK * work_scale
    work, seen, pieces = 0., set(), None
    ss = _strict(side)
    for mode, cap, extra, budget in _SCHEDULE[:_SPLIT_PASSES]:
        cap = _combine_cap(cap, cap1)
        if (mode, cap, extra) in seen:
            continue
        seen.add((mode, cap, extra))
        srch = _Search(ss, mode, cap, extra, budget * scale, forbid=forbid,
                       a0=a, b0=b)
        pieces = srch.run()
        work += srch.work
        if pieces is not None:
            break
    if pieces is None:
        return None, work
    pieces, w = _improve_units(ss, pieces, work, cap1, forbid, scale,
                               a0=a, b0=b)
    work += w
    pieces, w = _units_guard(ss, pieces, cap1, forbid, scale, a0=a, b0=b)
    work += w
    items = ([('must', i, 1., None) for i in range(side.M)]
             + [('flex', j, 1., None) for j in range(side.F)])
    cells, _ = _s_cells(side, items, a, pieces)
    if cells is None:
        return None, work
    b2 = [min(b[j] + math.fsum(c.x for c in cells if c.j == j), side.Qf[j])
          for j in range(side.F)]
    return _Block('tail', (a, b), (list(side.Qm), b2), cells,
                  work=work), work


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

    'L' adds pinch blocks (`_pinch_block`) at nodes where the pinch rules
    fail outward, at most ``M + F`` of them; a vertical block serves every
    node without one. A pinch block ends at a node accepted by the same
    exact check. 'T' tries a DFS tail (`_tail`) at nodes where the rules
    hold, other than the unleaked root (where the unsplit search already
    failed; a pre-leaked root qualifies), at most `_SPLIT_TAIL_TRIES`
    times; a tail that succeeds completes the candidate as its last block.
    The rules are checked on the search's analysis, as the DFS sees them.
    """
    if strategy not in _CORE_STRATEGIES:
        raise ValueError(f'core strategy {strategy!r} is not available')
    ex = _exact(side)
    lim = -_SPLIT_R_TOL * side.tolQ
    a, b = [float(x) for x in a0], [0.] * side.F
    blocks, used, leak = [], set(), [0.] * side.M
    pinch, tail = 'L' in strategy, 'T' in strategy
    n_pinch = tries = 0
    lost = 0.   # work of the failed tails
    while any(x < q for x, q in zip(a, side.Qm)):
        d = ex.analyse(a, b)
        if d.slack < lim:
            raise _SplitInvariantError(
                f'core node (R): slack {d.slack!r} after '
                f'{len(blocks)} blocks')
        viol = (side.rules_violation(a, b, side.analyse(a, b))
                if pinch or tail else None)
        if (tail and viol is None and (blocks or any(a0))
                and tries < _SPLIT_TAIL_TRIES):
            tries += 1
            blk, w = _tail(side, a, b, cap1, frozenset(forbid) | used,
                           work_scale)
            if blk is not None:
                blocks.append(blk)
                break
            lost += w
        blk = None
        if (pinch and viol is not None and viol['rule'] == 'outward'
                and n_pinch < side.M + side.F):
            blk = _pinch_block(side, a, b, forbid, used, viol)
            n_pinch += blk is not None
        if blk is None:
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
    return _Candidate(strategy, (1, _CORE_ORDER.index(strategy)), side,
                      [c for blk in blocks for c in blk.cells], leak,
                      math.fsum(blk.work for blk in blocks) + lost, Qmin,
                      blocks)


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
    __slots__ = ('name', 'order', 'cells', 'leak_by_must', 'work',
                 'blocks', 'excluded', 'units', 'stages', 'branches',
                 'mixers', 'small', 'mixbad', 'touch', 'meta', 'signature')

    def __init__(self, name, order, side, cells, leak_by_must=None,
                 work=0., Qmin=0., blocks=()):
        self.name, self.order = name, order
        self.cells = list(cells)
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
                    f'{c!r} fails (C) by {margin!r}')
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
                self.units + self.branches,   # units + extra branches + stages
                self.mixers, self.order)

    def __repr__(self):
        return f'<_Candidate {self.name} key={self.key()}>'


# %% Stage S: cut transports and the branched side

def _fits(c, m):
    """True if a capacity of CP `c` can take a demand of CP `m`: the
    compatibility of `_Side.rules_violation`, with its relative tolerance."""
    return c >= m * (1. - _CP_TOL)


def _cut_sets(side, a, b, L, cut, rule):
    """
    Demands and capacities of a pinch rule at a tight level.

    It repeats `_Side.rules_violation`'s selection at the cut, on the
    side's local indices. Outward, the demands are the musts that leave
    level `L` outward and the capacities the flexes at `L`, with ``CP =
    1/slope_right``; inward, the demands are the flexes that reach `L`
    from below and the capacities the musts, with ``CP = 1/slope_left``.
    Either way the rule is that each demand needs its own capacity branch
    with ``CP_cap >= CP_dem``.

    Parameters
    ----------
    side : _Side
        The side.
    a, b : sequence[float]
        The frontiers.
    L : float
        The level of the cut.
    cut : {'+', '-'}
        As in `_Side.tight_cuts`.
    rule : {'outward', 'inward'}
        The rule.

    Returns
    -------
    dem, cap : list[tuple[int, float]]
        ``(local index, CP)`` of the demands and of the sloped capacities,
        in index order. A flat capacity is left out: its level curve is
        split-invariant and it already has unlimited series capacity (a
        cut with one violates no rule).
    None
        If a demand is flat: a flat demand accepts only a flat partner, so
        no split serves it.
    """
    tolQ, tolP = side.tolQ, side.tolP
    above = cut == '-'

    def outward(curves, front):
        out = []
        for k, c in enumerate(curves):
            if c.Q - front[k] <= tolQ or c.at(front[k]) > L + tolP:
                continue
            q = max(c.x_lt(L) if above else c.x_le(L), front[k])
            if q < c.Q - tolQ:
                out.append((k, c.slope_right(q)))
        return out

    def inward(curves, front):
        out = []
        for k, c in enumerate(curves):
            if c.Q - front[k] <= tolQ or c.y[-1] < L - tolP:
                continue
            y = c.at(front[k])
            if above:
                if y >= L - tolP:
                    continue
                q = c.x_lt(L)
            else:
                if y > L + tolP:
                    continue
                q = c.x_le(L)
            if q > front[k] + tolQ:
                out.append((k, c.slope_left(q)))
        return out
    if rule == 'outward':
        dem, cap = outward(side.musts, a), outward(side.flexes, b)
    else:
        dem, cap = inward(side.flexes, b), inward(side.musts, a)
    if any(s == 0. for _, s in dem):
        return None
    return ([(k, 1. / s) for k, s in dem],
            [(k, 1. / s) for k, s in cap if s > 0.])


def _cut_transport(dem, cap, rule):
    """
    Branches that satisfy the pinch rule at a cut (module docstring, 'Stage S').

    Parameters
    ----------
    dem, cap : sequence[tuple[object, float]]
        ``(id, CP)`` of the demands and capacities (`_cut_sets`), CP > 0.
    rule : str
        One of `_SPLIT_RULES`:

        * 'partner' (demands stay whole): a best-fit maximum matching,
          demands by CP descending each taking the smallest free capacity
          that fits (optimal, as the neighbourhoods are nested); each
          unmatched demand then joins the capacity with the most room left,
          if that is enough.
        * 'demand' (capacities stay whole): the same matching; each
          unmatched demand then takes the largest free capacities until
          their CPs add up to its own, in proportion to them.
        * 'mincell' (the minimum-cell structure, `_mincell`).
        * 'nw-rho-desc', 'nw-rho-asc': the north-west corner, CP descending
          (ascending), against the capacities scaled to ``c/rho``, ``rho =
          sum c / sum m``: every cell has branch-CP ratio `rho`, the CP
          slack spread uniformly.
        * 'nw-exact-desc', 'nw-exact-asc': the north-west corner against
          the capacities themselves; a capacity never reached stays out.

    Returns
    -------
    list[tuple] or None
        Cells ``(d, c, load)``: a branch of demand ``d`` of CP ``load``
        faces a branch of capacity ``c``; a demand's loads add up to its CP
        and a capacity's to at most its CP (`_cut_fractions` gives the
        branch fractions). None if the rule finds no such branches, and
        always if ``sum c < sum m`` (then none exist).
    """
    if not dem:
        return []
    if not _fits(math.fsum(c for _, c in cap), math.fsum(m for _, m in dem)):
        return None
    if rule in ('partner', 'demand'):
        return _cut_match(dem, cap, rule == 'partner')
    if rule == 'mincell':
        return _mincell(dem, cap)
    desc = rule.endswith('-desc')

    def order(items):
        return sorted(items, key=lambda t: -t[1] if desc else t[1])
    return _north_west(order(dem), order(cap), rule.startswith('nw-rho'))


def _cut_match(dem, cap, partner):
    """The 'partner' (`partner`) and 'demand' rules of `_cut_transport`.
    Ties go to the lower index."""
    free = sorted(cap, key=lambda t: t[1])
    cells, rest = [], []
    for d, m in sorted(dem, key=lambda t: -t[1]):
        n = next((n for n, (_, c) in enumerate(free) if _fits(c, m)), None)
        if n is None:
            rest.append((d, m))
            continue
        cells.append((d, free.pop(n)[0], m))
    if partner:
        room = dict(cap)
        for d, k, m in cells:
            room[k] -= m
        for d, m in rest:
            k = max(room, key=room.get)
            if not _fits(room[k], m):
                return None
            room[k] -= m
            cells.append((d, k, m))
        return cells
    free.sort(key=lambda t: -t[1])
    for d, m in rest:
        take, tot = [], 0.
        while free and not _fits(tot, m):
            take.append(free.pop(0))
            tot += take[-1][1]
        if not _fits(tot, m):
            return None
        cells += [(d, k, m * c / tot) for k, c in take]
    return cells


def _north_west(dem, cap, inflate):
    """
    North-west corner: the demands, in order, fill the capacities, in
    order. With `inflate`, the capacities are scaled to ``c/rho`` (``rho =
    sum c / sum m``) and end with the demands; otherwise a capacity never
    reached stays out, whole. Each cell is the overlap of a demand and a
    capacity interval on the prefix sums, so no subtraction chain carries
    round-off; breakpoints within ``_SPLIT_ULP`` of the total coincide.
    """
    D = list(accumulate(m for _, m in dem))
    C = list(accumulate(c for _, c in cap))
    total = D[-1]
    if inflate:
        s = total / C[-1]
        C = [x * s for x in C]
        C[-1] = total
    ulp = _SPLIT_ULP * total
    cells, x, n, k = [], 0., 0, 0
    while n < len(D) and k < len(C):
        e = min(D[n], C[k])
        if e - x > ulp:
            cells.append((dem[n][0], cap[k][0], e - x))
        x = max(x, e)
        done_d, done_c = D[n] <= e + ulp, C[k] <= e + ulp
        n += done_d
        k += done_c
    return cells


def _merge_sets(n, k):
    """Families of disjoint subsets of ``range(n)``, each of two or more,
    with ``sum(|S| - 1) == k``; each family once, its sets in increasing
    order."""
    def rec(rem, avail, acc):
        if rem == 0:
            yield list(acc)
            return
        for size in range(rem + 1, 1, -1):
            for S in combinations(avail, size):
                if acc and S < acc[-1]:
                    continue
                yield from rec(rem - size + 1,
                               [x for x in avail if x not in S], acc + [S])
    return rec(k, list(range(n)), [])


def _pack(ms, size, work):
    """
    Exact bin packing by branch and bound: the demands `ms` (CP
    descending) into bins of CP `size`. The score, maximized, is (bins
    used, the smallest bin CP over its load): more groups mean fewer cells,
    and a larger ratio more CP slack. Capacity pruning, identical-bin
    symmetry breaking and the bound (the bins still reachable, the current
    smallest ratio: a load only grows); at most `_MINCELL_WORK` nodes in
    all (`work`, shared). Returns ``(score, bin of every demand)`` or None.
    """
    n, nb = len(ms), len(size)
    loads = [0.] * nb
    assign = [0] * n
    best = None

    def rec(k, used):
        nonlocal best
        if work[0] >= _MINCELL_WORK:
            return
        work[0] += 1
        ratio = min((size[t] / loads[t] for t in range(nb) if loads[t] > 0.),
                    default=math.inf)
        if k == n:
            if best is None or (used, ratio) > best[0]:
                best = ((used, ratio), list(assign))
            return
        if best is not None and (min(nb, used + n - k), ratio) <= best[0]:
            return
        m = ms[k]
        seen = set()
        for t in range(nb):
            old = loads[t]
            if (old, size[t]) in seen or not _fits(size[t], old + m):
                continue
            seen.add((old, size[t]))
            loads[t] = old + m
            assign[k] = t
            rec(k + 1, used + (old == 0.))
            loads[t] = old
    rec(0, 0)
    return best


def _mincell(dem, cap):
    """
    The 'mincell' rule of `_cut_transport`: Linnhoff and Hindmarsh's
    minimum-cell pinch design. For k = 0..3, disjoint sets of capacities
    merge into super-bins with ``sum(|S| - 1) = k`` and the demands are
    packed into the bins (`_pack`); the best packing at the smallest
    feasible k gives the groups, and each group is served by the uniform-rho
    north-west corner (CP descending). k = 0 is the number-rule fix (each
    group has one capacity, the demands stay whole) and each unit of k one
    demand split (the CP-rule fix). At the smallest feasible k every merged
    set is used: a packing that left one empty would be feasible at a
    smaller k, which was searched in full. Without a structure (or when the
    work, `_MINCELL_WORK` nodes and families in all, runs out before one is
    found) all of the cut is one group.
    """
    ms = sorted(dem, key=lambda t: -t[1])
    work = [0]
    best = None
    for k in range(4):
        for sets in _merge_sets(len(cap), k):
            work[0] += 1
            merged = {n for S in sets for n in S}
            bins = sets + [(n,) for n in range(len(cap)) if n not in merged]
            size = [math.fsum(cap[n][1] for n in S) for S in bins]
            r = _pack([m for _, m in ms], size, work)
            if r is not None and (best is None or r[0] > best[0]):
                best = (r[0], bins, r[1])
            if work[0] >= _MINCELL_WORK:
                break
        if best is not None or work[0] >= _MINCELL_WORK:
            break
    if best is None:
        return _north_west(ms, sorted(cap, key=lambda t: -t[1]), True)
    _, bins, assign = best
    cells = []
    for t in dict.fromkeys(assign):   # the groups, by their largest demand
        group = [ms[n] for n in range(len(ms)) if assign[n] == t]
        caps = sorted((cap[n] for n in bins[t]), key=lambda u: -u[1])
        cells += _north_west(group, caps, True)
    return cells


def _cut_fractions(cells):
    """
    Branch fractions ``(f, g)`` of every cell of a cut transport: the
    cell's load over the sum of the loads of its demand (capacity). A
    demand's loads add up to its CP, so ``f = load / m``; a capacity is
    split whole, with no bypass, so its branch has CP ``g c >= load``. An
    item with one cell has fraction 1 (no split).
    """
    by_d, by_c = defaultdict(list), defaultdict(list)
    for d, c, x in cells:
        by_d[d].append(x)
        by_c[c].append(x)
    sd = {d: math.fsum(v) for d, v in by_d.items()}
    sc = {c: math.fsum(v) for c, v in by_c.items()}
    return [(x / sd[d], x / sc[c]) for d, c, x in cells]


def _split_items(items, M, split):
    """
    The items (`_pinch_split`) after one cut: every item in `split`
    (``(role, local index) -> branch fractions``) becomes branches of its
    fraction times those. A branch below `_SPLIT_MIN_FRACTION` of its
    parent merges into its largest sibling (a realizable split), and a parent left with
    one branch is a trunk again. Keys are ``('S', parent, n)``, numbered per
    parent in item order.
    """
    parent, frac = [], []
    for n, (role, p, f, _) in enumerate(items):
        for g in split.get((role, n if role == 'must' else n - M), (1.,)):
            parent.append((role, p))
            frac.append(f * g)
    alive = [True] * len(frac)
    siblings = defaultdict(list)
    for n, rp in enumerate(parent):
        siblings[rp].append(n)
    for sib in siblings.values():
        while len(sib) > 1:
            s = min(sib, key=frac.__getitem__)
            if frac[s] >= _SPLIT_MIN_FRACTION:
                break
            sib.remove(s)
            frac[max(sib, key=frac.__getitem__)] += frac[s]
            alive[s] = False
    count = Counter(rp for rp, live in zip(parent, alive) if live)
    seen = Counter()
    out = []
    for (role, p), f, live in zip(parent, frac, alive):
        if not live:
            continue
        if count[role, p] == 1:
            out.append((role, p, 1., None))
            continue
        out.append((role, p, f, ('S', p, seen[role, p])))
        seen[role, p] += 1
    return out


def _branched_side(side, items, a0):
    """
    The side of `items` (`_pinch_split`), and its pre-leaked root: a must
    branch of fraction ``f`` of parent ``i`` starts at branch heat ``f
    a0_i`` (Lemma B), a flex branch at 0.
    """
    musts, flexes, a = [], [], []
    for role, p, f, _ in items:
        if role == 'must':
            c = side.musts[p]
            musts.append(c if f == 1. else _branch_curve(c, f, 'must'))
            a.append(f * a0[p])
        else:
            c = side.flexes[p]
            flexes.append(c if f == 1. else _branch_curve(c, f, 'flex'))
    return _StrictSide(side.name, musts, flexes, side.tolQ, side.tolP), a


def _pinch_split(side, a0, proof, rule):
    """
    The branched side of one Stage S rule (module docstring, 'Stage S').

    The rule's transport (`_cut_transport`) at the proof's cut splits the
    demands and capacities there into whole-side branches. By Lemma R the
    branched side keeps the parent's cascade, so one exact screen at its
    pre-leaked root decides the split: its slack of (R) must be the
    parent's, and `rules_violation` there checks the pinch rules at every
    tight cut. If a cut still violates them (a double pinch), the same rule
    is applied there, to the branches as items (a branch of a branch is a
    branch of the parent with the product fraction), at most `_SPLIT_CUTS`
    cuts in all.

    Parameters
    ----------
    side : _Side
        The side.
    a0 : sequence[float]
        Its pre-leaked root (`_preleak_root`).
    proof : dict
        The root proof: an 'outward' or 'inward' violation.
    rule : str
        One of `_SPLIT_RULES`.

    Returns
    -------
    items : list[tuple]
        ``(role, parent, fraction, key)`` of every curve of the branched
        side (`_branched_side`), the musts ('must') first, then the flexes
        ('flex'). A trunk has fraction 1 and key None, a branch a fraction
        of at least `_SPLIT_MIN_FRACTION` and key ``('S', parent, n)``.
    extra : int
        The extra branches, ``sum(branches - 1)`` over the split parents.
    None
        If the rule finds no transport at a cut, the split changes nothing,
        or a violation is left after `_SPLIT_CUTS` cuts.

    Raises
    ------
    _SplitInvariantError
        If the slack of (R) at the branched root differs from the side's by
        more than ``10 tolQ`` (Lemma R broken).
    """
    M, F = side.M, side.F
    slack0 = side.analyse(list(a0), [0.] * F).slack
    items = ([('must', i, 1., None) for i in range(M)]
             + [('flex', j, 1., None) for j in range(F)])
    bside, a, v = side, list(a0), proof
    for _ in range(_SPLIT_CUTS):
        sets = _cut_sets(bside, a, [0.] * bside.F, v['level'], v['cut'],
                         v['rule'])
        cells = None if sets is None else _cut_transport(*sets, rule)
        if cells is None:
            return None
        dem, cap = ('must', 'flex') if v['rule'] == 'outward' else (
            'flex', 'must')
        split = defaultdict(list)
        for (d, c, _), (f, g) in zip(cells, _cut_fractions(cells)):
            split[dem, d].append(f)
            split[cap, c].append(g)
        new = _split_items(items, bside.M, split)
        if new == items:
            return None
        items = new
        bside, a = _branched_side(side, items, a0)
        d = bside.analyse(a, [0.] * bside.F)
        if abs(d.slack - slack0) > 10. * side.tolQ:
            raise _SplitInvariantError(
                f'the branched root has slack {d.slack!r}, the '
                f'side {slack0!r}')
        v = bside.rules_violation(a, [0.] * bside.F, d)
        if v is None:
            return items, sum(n - 1 for n in Counter(
                it[:2] for it in items).values())
    return None


# %% Stage S: the plan

def _repeats_a_pair(cells, tolQ):
    """True if two exchangers of `cells` (`_merge_cells`) have the same
    (must, flex) pair: avoid_recycle allows every pair once, and two
    branches of a stream with one partner are two exchangers."""
    pairs = Counter((c.i, c.j) for c in _merge_cells(cells, tolQ))
    return any(n > 1 for n in pairs.values())


def _fold(items, M, used):
    """
    Lemma F: the fraction and stage key of every used flex of a branched
    side, after folding its unused siblings.

    Parameters
    ----------
    items : list[tuple]
        The branched side's items (`_pinch_split`).
    M : int
        Its number of musts.
    used : collection[int]
        Local indices of the flexes with cells.

    Returns
    -------
    dict
        ``{local flex index: (g, key)}``. A parent whose branches are all
        used keeps them. Otherwise each used branch gets ``g / G``, with
        ``G`` the sum over its used siblings (the unused fractions go to
        them in proportion), and keys renumbered in item order; a parent
        with one used branch is a trunk again, ``(1., None)``.
    """
    siblings = defaultdict(list)
    for jj, (_, p, _, _) in enumerate(items[M:]):
        siblings[p].append(jj)
    out = {}
    for p, br in siblings.items():
        live = [jj for jj in br if jj in used]
        if len(live) == len(br):
            for jj in live:
                out[jj] = items[M + jj][2:]
        elif len(live) == 1:
            out[live[0]] = (1., None)
        elif live:
            G = math.fsum(items[M + jj][2] for jj in live)
            for n, jj in enumerate(live):
                out[jj] = (items[M + jj][2] / G, ('S', p, n))
    return out


def _s_cells(side, items, a0, pieces):
    """
    Cells of a Stage S plan (module docstring, 'Stage S'): the DFS `pieces` of
    the branched side, merged (`_merge`), in parent coordinates, with the
    unused flex branches folded (`_fold`) and the residual swept.

    Parameters
    ----------
    side : _Side
        The parent side.
    items : list[tuple]
        The branched side's items (`_pinch_split`).
    a0 : sequence[float]
        The parent side's pre-leaked root.
    pieces : list[tuple]
        ``(i', j', a', b', x)`` on the branched side, from its pre-leaked
        root (`_branched_side`).

    Returns
    -------
    cells : list[_Cell] or None
        The cells, or None if the sweep fails.
    reason : None or 'sweep'

    Notes
    -----
    A piece ``(i', j', a', b', x)`` becomes ``_Cell(parent(i'), parent(j'),
    x, a'/f, b'/g, f, g, key(i'), key(j'))``: a branch is the scaled parent
    (Lemma B). Folding raises a flex branch's fraction ``g``, so its
    parent positions ``b'/g`` move toward the pinch, where the flex levels
    are no higher, and every cell stays feasible (Lemma F); the must side
    is unchanged.

    The DFS completes a must within ``tolQ`` of heat. The residual of must
    item ``i'``, ``r = f (Qm_i - a0_i)`` less its duties, is swept so that
    every must is served exactly: if ``|r| <= tolQ``, `r` is added to
    the item's far-end cell, the later cells of that cell's flex item move
    with it (the flex stays a prefix), and the cells changed are verified
    (`_cell_margin`); otherwise, or if that fails, the plan is rejected.
    Stage S never books a leak.
    """
    M = sum(it[0] == 'must' for it in items)
    tolQ, tolP = side.tolQ, side.tolP
    ex = _merge(pieces)
    fold = _fold(items, M, {e[1] for e in ex})
    moved = set()
    for i2, (_, p, f, _) in enumerate(items[:M]):
        rows = [e for e in ex if e[0] == i2]
        r = f * (side.Qm[p] - a0[p]) - math.fsum(e[4] for e in rows)
        if r == 0.:
            continue
        if not rows or not abs(r) <= tolQ:
            return None, 'sweep'
        e = max(rows, key=lambda e: e[2] + e[4])
        if not e[4] + r > 0.:
            return None, 'sweep'
        for e2 in ex:
            if e2[1] == e[1] and e2[3] > e[3]:
                e2[3] += r
                moved.add(id(e2))
        e[4] += r
        moved.add(id(e))
    cells = []
    for e in ex:
        i2, j2, a, b, x = e
        _, p, f, km = items[i2]
        g, kf = fold[j2]
        c = _Cell(p, items[M + j2][1], x, a / f, b / g, f, g, km, kf)
        if id(e) in moved and not _cell_margin(side, c)[0] >= -tolP:
            return None, 'sweep'
        cells.append(c)
    return cells, None


def _s_search(bside, a0, cap1, forbid, work_scale, bound, room):
    """
    The first `_SPLIT_PASSES` passes of `_SCHEDULE` on a branched side from
    its pre-leaked root, as `_plan_side` runs them but for the unit bound
    and the shared budget `room`.

    Returns
    -------
    pieces : list[tuple] or None
        The first success.
    work : float
        Work spent.
    cut : bool
        True if the shared budget stopped a pass.
    """
    work, seen = 0., set()
    for mode, cap, extra, budget in _SCHEDULE[:_SPLIT_PASSES]:
        cap = _combine_cap(cap, cap1)
        if (mode, cap, extra) in seen:
            continue
        seen.add((mode, cap, extra))
        left = room - work
        if left <= 0.:
            return None, work, True
        budget *= work_scale
        srch = _Search(bside, mode, cap, extra, min(budget, left),
                       unit_bound=bound, forbid=forbid, a0=a0)
        pieces = srch.run()
        work += srch.work
        if pieces is not None:
            return pieces, work, False
        if srch.exhausted and budget > left:
            return None, work, True
    return None, work, False


def _s_candidate(name, n, side, items, a0, pieces, cap1, work, split,
                 errors, reasons):
    """The Stage S candidate of DFS `pieces` (converted by `_s_cells` and
    verified by `_Candidate`), or None; its key or the reason is recorded
    in `reasons`."""
    cells, why = _s_cells(side, items, a0, pieces)
    if cells is None:
        reasons[name] = why
        return None
    if cap1 and _repeats_a_pair(cells, side.tolQ):
        reasons[name] = 'repeated pair'
        return None
    try:
        c = _Candidate(name, (0, n), side, cells, None, work,
                       split['Qmin'])
    except _SplitInvariantError as e:
        errors.append(f'{name}: {e}')
        reasons[name] = 'error'
        return None
    c.excluded = _excluded(split, side.name, c)
    reasons[name] = 'excluded' if c.excluded else c.key()
    return c


def _stage_s(side, a0, proof, cap1, forbid, work_scale, split, errors,
             reasons, only=None, skip=None):
    """
    Stage S candidates of a side: pinch splits planned by the unchanged
    DFS (module docstring, 'Stage S').

    Parameters
    ----------
    side : _Side
        The side.
    a0 : sequence[float]
        Its pre-leaked root (`_preleak_root`).
    proof : dict
        The root proof, an 'outward' or 'inward' violation.
    cap1 : bool
        avoid_recycle: every pair in at most one exchanger.
    forbid : collection[tuple[int, int]]
        Pairs never used (local indices).
    work_scale : float
        Scale of the search budgets.
    split : dict
        ``Qmin`` and ``exclude`` (`_split_side`).
    errors : list[str]
        `_SplitInvariantError` messages are appended.
    reasons : dict
        Set for every rule run, ``'S:<rule>'``: the candidate's key, or
        'excluded', or why there is no candidate: 'no split'
        (`_pinch_split` found none), 'same split' (an earlier rule's
        branching), 'budget' (the shared budget ran out), 'bound' (no plan
        within the unit bound), 'no plan', 'sweep' (`_s_cells`), 'repeated
        pair' (avoid_recycle) or 'error'.
    only : str, optional
        Run this candidate name only (the preferred one).
    skip : str, optional
        Do not run this candidate name (the preferred one already ran).

    Returns
    -------
    cands : list[_Candidate]
        The candidates, excluded ones included (`_excluded`).
    work : float
        Work spent: the DFS passes of every rule (with or without a
        candidate) and the winner's unit improvements.

    Notes
    -----
    For each rule of `_SPLIT_RULES`, in order, `_pinch_split` gives the
    branched side, and the first `_SPLIT_PASSES` passes of `_SCHEDULE`
    plan it from its pre-leaked root (`_s_search`), with the pairs of
    `forbid` forbidden on every branch pair of their parents. Once a live
    candidate has no small exchanger, bad remix or touch, the DFS runs
    with the unit bound ``score - extra - stages`` (``score`` the
    incumbent's units + extra branches + split stages; this rule's extra
    branches and split stages), which keeps ties. All rules share
    ``_SPLIT_S_WORK work_scale`` of work (deterministic); rules not
    reached are 'budget'. Each plan is converted (`_s_cells`: folding,
    the residual sweep) and verified (`_Candidate`); with `cap1`, a plan
    repeating a pair is rejected. With `_SPLIT_FIRST_WINS`, the first live
    candidate ends the rules. The winner (the smallest key among the live
    candidates) then gets the planner's `_improve_units` and
    `_units_guard` on its branched side, and the result replaces it if
    it is live and its key is not worse.
    """
    budget = _SPLIT_S_WORK * work_scale
    used = 0.
    out, seen, runs = [], [], {}
    for n, rule in enumerate(_SPLIT_RULES):
        name = 'S:' + rule
        if name == skip or (only is not None and name != only):
            continue
        if _SPLIT_FIRST_WINS and any(not c.excluded for c in out):
            break
        if used >= budget:
            reasons[name] = 'budget'
            continue
        try:
            res = _pinch_split(side, a0, proof, rule)
        except _SplitInvariantError as e:
            errors.append(f'{name}: {e}')
            reasons[name] = 'error'
            continue
        if res is None:
            reasons[name] = 'no split'
            continue
        items, extra = res
        if items in seen:
            reasons[name] = 'same split'
            continue
        seen.append(items)
        bside, a0b = _branched_side(side, items, a0)
        M = bside.M
        bforbid = frozenset(
            (i2, j2) for i2, it in enumerate(items[:M])
            for j2, jt in enumerate(items[M:]) if (it[1], jt[1]) in forbid)
        stages = len({it[:2] for it in items if it[3] is not None})
        live = [c for c in out if not c.excluded]
        inc = min(live, key=_Candidate.key) if live else None
        bound = math.inf
        if inc is not None and inc.small == inc.mixbad == inc.touch == 0:
            bound = inc.key()[3] - extra - stages
        pieces, work, cut = _s_search(bside, a0b, cap1, bforbid, work_scale,
                                      bound, budget - used)
        used += work
        if pieces is None:
            reasons[name] = ('budget' if cut else 'no plan'
                             if bound == math.inf else 'bound')
            continue
        c = _s_candidate(name, n, side, items, a0, pieces, cap1, work, split,
                         errors, reasons)
        if c is not None:
            out.append(c)
            runs[id(c)] = (bside, a0b, bforbid, items, pieces)
    live = [c for c in out if not c.excluded]
    if not live:
        return out, used
    # the winner's units, as `_plan_side` improves an unsplit plan
    best = min(live, key=_Candidate.key)
    bside, a0b, bforbid, items, pieces = runs[id(best)]
    new, w1 = _improve_units(bside, pieces, best.work, cap1, bforbid,
                             work_scale, a0=a0b)
    new, w2 = _units_guard(bside, new, cap1, bforbid, work_scale, a0=a0b)
    best.work += w1 + w2
    if new is not pieces:
        c = _s_candidate(best.name, best.order[1], side, items, a0, new,
                         cap1, best.work, split, errors, reasons)
        if c is not None and not c.excluded and c.key() <= best.key():
            out[out.index(best)] = c
        else:
            reasons[best.name] = best.key()
    return out, used + w1 + w2


# %% Portfolio and selection

def _excluded(split, side_name, cand):
    """True if the network signature of `cand` is excluded on its side
    (exclusion is by network, never by generator name)."""
    return cand.signature in split['exclude'].get(side_name, ())


def _side_plan(side, best, reasons, a0, delta, errors, work, proof):
    """The side plan of the chosen candidate `best` (None: no candidate).
    Each must's gap is its pre-leak plus its leak, so the penalty is
    truthful."""
    if best is None:
        info = dict(candidate=None, signature=None, candidates=reasons,
                    stages=0, branches=0, preleak=delta, leak=0., small=[],
                    errors=errors)
        return _SidePlan([], list(side.Qm), 'failed', 'split-none', work,
                         proof, split=info)
    gaps = [a0[i] + best.leak_by_must[i] for i in range(side.M)]
    info = dict(best.meta, signature=best.signature, candidates=reasons,
                preleak=delta, errors=errors)
    return _SidePlan([], gaps, 'mer', 'split-' + best.name, work, proof,
                     cells=best.cells, split=info, units=best.units)


def _split_side(side, proof, cap1, forbid, work_scale, work, split):
    """
    Split plan of a side that no unsplit network serves at MER.

    Parameters
    ----------
    side : _Side
        The side. The core analyses its root exactly (`_preleak_root`).
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
    _SidePlan
        If a candidate exists, a side plan with status 'mer', method
        ``'split-<candidate>'``, the cells of the chosen candidate and the
        ``split`` info. Otherwise (the root deficit exceeds `_preleak_max`,
        or every generator failed, raised `_SplitInvariantError` or was
        rejected by avoid_recycle's pairs) status 'failed', method
        'split-none', no cells, and the ``split`` info of the attempt with
        ``candidate`` None: the caller keeps it on its best-effort plan.
        Either way the work is `work` plus the work spent here.

    Notes
    -----
    Every generator starts from the pre-leaked root ``a0 = P(delta)``
    (Lemma P): Stage S (`_stage_s`, one candidate 'S:<rule>' per rule of
    `_SPLIT_RULES`) if the root proof is a pinch rule ('outward' or
    'inward'), then the core (`_CORE_STRATEGIES`). The preferred candidate
    (the previous refine round's pick) is generated first and taken as is
    unless its signature is excluded. Otherwise every generator runs (the
    preferred one is not run again) and the smallest `_Candidate.key` wins
    among the candidates not excluded or, if all are excluded, among all
    of them: a side's last candidate is never excluded. With
    `_SPLIT_FIRST_WINS`, the first live candidate wins. A generator that
    raises `_SplitInvariantError` is recorded in ``errors`` and the others
    remain. With `cap1`, a candidate repeating a pair is rejected
    (`_repeats_a_pair`). The work spent is that of Stage S (every rule's
    DFS passes, `_stage_s`) and of every core candidate.
    """
    delta, a0 = _preleak_root(side)
    core = delta <= _preleak_max(side)
    s_ok = core and proof is not None and proof['rule'] in ('outward',
                                                            'inward')
    prefer = split['prefer'].get(side.name)
    cands, errors, reasons, spent = [], [], {}, [work]

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
        spent.append(c.work)
        if cap1 and _repeats_a_pair(c.cells, side.tolQ):
            reasons[name] = 'repeated pair'
            return None
        c.excluded = _excluded(split, side.name, c)
        reasons[name] = 'excluded' if c.excluded else c.key()
        cands.append(c)
        return c

    def stage_s(**kw):
        cs, w = _stage_s(side, a0, proof, cap1, forbid, work_scale, split,
                         errors, reasons, **kw)
        spent.append(w)
        cands.extend(cs)
        return cs

    def plan(best):
        return _side_plan(side, best, reasons, a0, delta, errors,
                          math.fsum(spent), proof)
    # the preferred candidate (stickiness), taken as is if it is live
    first = []
    if s_ok and prefer is not None and prefer[2:] in _SPLIT_RULES and (
            prefer.startswith('S:')):
        first = stage_s(only=prefer)
    elif core and prefer in _CORE_STRATEGIES:
        first = [generate(prefer)]
    for c in first:
        if c is not None and not c.excluded:
            return plan(c)
    # the portfolio: Stage S, then the core
    if s_ok:
        stage_s(skip=prefer)
    if core and not (_SPLIT_FIRST_WINS and any(not c.excluded
                                               for c in cands)):
        for strategy in _CORE_STRATEGIES:
            if strategy != prefer:
                c = generate(strategy)
                if _SPLIT_FIRST_WINS and c is not None and not c.excluded:
                    break
    if not cands:
        return plan(None)
    live = [c for c in cands if not c.excluded] or cands
    return plan(min(live, key=_Candidate.key))


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
