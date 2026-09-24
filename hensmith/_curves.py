# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Per-stream temperature-enthalpy curves for the problem table and the network
planner (private module).

A `StreamCurve` is a piecewise-linear temperature-enthalpy curve of one
process stream over its own enthalpy range, built once from a handful of
flashes and evaluated afterwards without flashing. `problem_table` builds its
grid from the curves' breakpoints, and the network synthesis plans on the
same curves, so both work on one model of every stream.

Why a curve rather than flashes at grid temperatures: a table that puts only
the stream end temperatures on its grid and flashes every stream at every
grid point with ``vle(T=...)`` has three defects:

* a boiling or condensing point, a bubble or dew point, or the curvature of a
  mixture glide or of a single-phase stretch with temperature-dependent heat
  capacity that falls strictly inside a grid interval is averaged over the
  interval, which can hide a pinch (targets too low, by up to 100 %);
* ``vle(T=T_sat)`` of a single chemical keeps whatever phase split the stream
  had (thermosteam leaves it unchanged when ``|P - Psat| <= 1e-3`` Pa), so
  the enthalpy at a grid point exactly on T_sat depends on history;
* thermosteam's two-phase TP/PH/PV flashes of some mixtures (water/ethanol
  with an ethanol mole fraction of 0.2-0.5) silently return non-converged
  states.

The curve removes all three: phase boundaries are breakpoints; a pure
component's latent heat is an isothermal (flat) segment between its
saturated-liquid and saturated-vapor enthalpies from V-specified flashes;
single-phase stretches are evaluated with the phases FIXED (no flash at all,
so there is no ambiguity at T_sat) and subdivided until linear
interpolation is within `GLIDE_TOL_T` kelvin; and mixture glides are sampled
adaptively to the same tolerance, binaries along their bubble-point curve
(thermosteam's BubblePoint only), other mixtures with TP flashes whose
results are sanity-checked.

Nothing here imports biosteam or thermosteam: the curves only call methods of
the streams they are given.
"""
import heapq
from functools import partial
from warnings import warn
import numpy as np

#: Segment kinds of a `StreamCurve`.
FLAT, SENSIBLE, GLIDE = 'flat', 'sensible', 'glide'

#: Default maximum linear-interpolation error of a `StreamCurve` along a
#: two-phase glide or a curved single-phase stretch [K].
GLIDE_TOL_T = 0.002
#: Default maximum number of equilibrium evaluations per glide.
GLIDE_MAX_SAMPLES = 400
#: Phase-boundary temperatures closer than this to a stream end are moved
#: onto it [K] (end states computed by a different thermosteam path than the
#: boundary agree to ~1e-10 K; a sliver interval would only add noise).
_T_SNAP = 1e-6
#: Grid temperatures closer than this are the same grid point [K].
_T_EQ = 1e-9
#: A flat at an end temperature is a non-equilibrium jump if the PH flash at
#: its mid-enthalpy lands further than this outside the stream's own range
#: [K] (a latent-heat flat snapped onto an end lands within `_T_SNAP`).
_T_JUMP = 1e-5
#: A glide root (`StreamCurve._glide_root`) that misses its enthalpy by more
#: than this fraction of the stream's duty is rejected: flash-resolution
#: jumps next to a phase boundary are ~1e-8, spurious flashes ~1e-3 or more.
_ROOT_RTOL = 1e-7
#: An equilibrium state within this of a temperature [K] is not on the wrong
#: side of it: the inlet of a point-load stream (`_point_load_inlet`), or an
#: enthalpy limit (`hensmith.hxn_synthesis._enthalpy_limit`). Flash noise of
#: an isothermal stream is ~1e-10 K; `HXprocess` rejects 0.01 K on the wrong
#: side.
_T_SIDE = 1e-6


def _copy(stream, thermo=None):
    """
    Copy of `stream`, with its ID, that is not registered in the flowsheet
    (a leading '.' names a stream without registering it). A plain
    ``stream.copy()`` takes its ID from the source line of the call
    (thermosteam's ID magic): unless that line assigns the copy to a plain
    variable, a multi-phase copy registers as '-' and replaces the previous
    one with a RuntimeWarning, and a single-phase one takes a registry
    ticket, which keeps it alive. The curves copy streams hundreds of times
    per synthesis, so every internal copy goes through here.
    """
    return stream.copy('.' + stream.ID, thermo)


def _point_load_inlet(stream_in, T_point, is_hot):
    """
    State in which a point-load stream (a non-monotone `StreamCurve`, whose
    whole duty is at its outlet temperature `T_point`) enters its first
    exchanger: a copy of `stream_in` at equilibrium at its own enthalpy and
    pressure, or of the stream as given if that flash fails or lands on the
    wrong side of `T_point` (colder than it for a cooled stream, hotter for
    a heated one; more than `_T_SIDE`).

    A stream is a point load because its real inlet temperature lies on the
    wrong side of its outlet: a cooled stream fed colder than its outlet
    (e.g. a vapor fed below its dew point, or the copy of a saturated vapor
    under ideal thermodynamics, whose ideal dew point is higher), or a
    heated one fed hotter (a liquid above its bubble point). The plan puts
    the whole duty at the outlet temperature, where the real inlet state
    offers no heat as such: `HXprocess` compares the inlet temperatures and
    caps the partner at the real inlet temperature -/+ its `dT`. But the
    inlet relaxes, adiabatically and at constant pressure, to its
    equilibrium state at the same enthalpy (as every flash in the network
    takes it), and at fixed pressure the equilibrium temperature does not
    decrease with the enthalpy: a cooled stream's equilibrium temperature
    at its inlet enthalpy is at least the one at its lower outlet
    enthalpy, `T_point` (the outlet is quenched to equilibrium), and a
    heated stream's at most. From that state the stream delivers (takes)
    its heat no colder (hotter) than `T_point`, where the plan and the
    problem table put it, so the terminal checks of `HXprocess` agree with
    the plan. The enthalpy is the same, so no balance changes.
    """
    stream = _copy(stream_in)
    try:
        stream.vle(H=stream_in.H, P=stream.P)
    except Exception:
        return _copy(stream_in)
    T = stream.T
    if T >= T_point - _T_SIDE if is_hot else T <= T_point + _T_SIDE:
        return stream
    return _copy(stream_in)


# %% Evaluators of single-phase (sensible) segments

class _FixedPhase:
    """
    Exact enthalpy of a single-phase stretch of a stream: a reference state
    whose phase distribution is the equilibrium one throughout the stretch
    (all VLE chemicals liquid below the bubble point, vapor above the dew
    point; locked chemicals where thermosteam puts them), re-evaluated at
    any temperature without a flash.
    """
    __slots__ = ('stream',)

    def __init__(self, stream):
        self.stream = _copy(stream)

    def H(self, T):
        s = self.stream
        s.T = T
        return s.H

    def state_at_T(self, T):
        s = _copy(self.stream)
        s.T = T
        return s

    def state_at_H(self, H):
        s = _copy(self.stream)
        s.H = H  # solves T with the phase distribution fixed
        return s


class _TPFlash:
    """
    Fallback evaluator: a TP flash at each temperature, warm-started along
    the way (the behavior of the problem table before stream curves). Used
    only when the phase analysis of a stream is not possible (e.g. above the
    critical pressure, or a failed bubble/dew point).
    """
    __slots__ = ('stream0', 'stream', 'label')

    def __init__(self, stream, label):
        self.stream0 = _copy(stream)
        self.stream = _copy(stream)
        self.label = label

    def H(self, T):
        s = self.stream
        try:
            s.vle(T=T, P=s.P)
            return s.H
        except Exception as error:
            self.stream = _copy(self.stream0)
            warn(f"could not solve VLE for stream {self.label!r} at "
                 f"{T:.2f} K ({error!r}); interpolating enthalpy linearly "
                 "in temperature for the problem table", RuntimeWarning)
            return None

    def state_at_T(self, T):
        s = _copy(self.stream0)
        s.vle(T=T, P=s.P)
        return s

    def state_at_H(self, H):
        s = _copy(self.stream0)
        s.vle(H=H, P=s.P)
        return s


# %% Samplers of two-phase glides (parameter t: 0 at the glide's low-T end,
# 1 at its high-T end; T and H increase with t)

class _BinaryGlide:
    """
    Two-phase states of a binary at fixed P traced along the bubble-point
    curve of the liquid: the liquid mole fraction of the first VLE chemical
    goes from the feed's (t = 0, bubble point, V = 0) to the dew-point
    liquid's (t = 1, V = 1); T and the vapor composition y are that liquid's
    bubble point, V follows from the lever rule, and H is the enthalpy of
    liquid F(1-V)x plus vapor FVy at T. Only thermosteam's BubblePoint is
    used (robust), never a TP/PH/PV flash, which fail silently for some
    water/ethanol compositions. By the phase rule (2 components, 2 phases,
    fixed P) this path is exactly the set of equilibrium states of the feed.
    """
    __slots__ = ('bp', 'IDs', 'z0', 'F', 'P', 'u0', 'u1', 'work', 'n')

    def __init__(self, sat_liquid, IDs, bp, x_dew, P):
        mol = np.asarray(sat_liquid.imol['l', IDs], dtype=float)
        self.F = F = mol.sum()
        self.z0 = mol[0] / F
        self.bp = bp
        self.IDs = IDs
        self.P = P
        self.u0 = self.z0
        self.u1 = float(x_dew[0])
        self.work = _copy(sat_liquid)
        self.n = 0

    def _set(self, s, t):
        u = self.u0 + (self.u1 - self.u0) * t
        x = np.array([u, 1. - u])
        r = self.bp(x, P=self.P)
        y = np.asarray(r.y, dtype=float)
        dy = y[0] - u
        if dy == 0.:
            V = 0. if t < 0.5 else 1.
        else:
            V = (self.z0 - u) / dy
            V = 0. if V < 0. else 1. if V > 1. else V
        F = self.F
        s.imol['l', self.IDs] = F * (1. - V) * x
        s.imol['g', self.IDs] = F * V * y
        s.T = r.T
        self.n += 1
        return r.T

    def __call__(self, t, strict=True):
        s = self.work
        T = self._set(s, t)
        return T, s.H

    def state(self, t):
        s = _copy(self.work)
        self._set(s, t)
        return s

    sample_state = state  # bubble-point tracing reproduces its samples


class _TPGlide:
    """
    Two-phase states of a general mixture by TP flashes, t linear in T over
    [Ta, Tb]. With `strict` (Ta and Tb are the true bubble and dew points:
    no always-gas or dissolved-solute chemicals), a flash whose result is
    single-phase strictly inside the glide is reported as failed (None) so
    that the sampler drops it; a call with ``strict=False`` (root solves
    between accepted samples, see `StreamCurve._glide_root`) accepts it,
    because a TP flash a hair inside a glide end legitimately comes back
    single phase (it cannot resolve the phase boundary to the last digits).
    """
    __slots__ = ('template', 'work', 'Ta', 'Tb', 'P', 'IDs', 'n', 'failed',
                 'strict', 'phase', 'states')

    def __init__(self, template, Ta, Tb, P, IDs, strict):
        self.template = template
        self.work = _copy(template)
        self.Ta = Ta
        self.Tb = Tb
        self.P = P
        self.IDs = IDs
        self.n = 0
        self.failed = 0
        self.strict = strict
        self.phase = {}  # t -> 'l', 'g' or 'lg' (where the VLE chemicals are)
        self.states = {}  # t -> copy of the flashed state

    def __call__(self, t, strict=True):
        T = self.Ta + (self.Tb - self.Ta) * t
        s = self.work
        self.n += 1
        try:
            s.vle(T=T, P=self.P)
        except Exception:
            self.failed += 1
            self.work = _copy(self.template)
            return None
        v = float(np.sum(s.imol['g', self.IDs]))
        l = float(np.sum(s.imol['l', self.IDs]))
        tiny = 1e-12 * (v + l)
        phase = 'l' if v <= tiny else 'g' if l <= tiny else 'lg'
        if not strict: return T, s.H  # a root solve: nothing to record
        self.phase[t] = phase
        self.states[t] = _copy(s)
        if self.strict and 0. < t < 1. and phase != 'lg':
            self.failed += 1
            return None
        return T, s.H

    def phase_at(self, t):
        if t not in self.phase: self(t)
        return self.phase.get(t)

    def phase_boundaries(self, ts, T_tol=1e-6):
        """
        Where the phase set of the VLE chemicals changes between consecutive
        sampled parameters `ts` (e.g. the boiling onset of a sugar solution,
        raised above the solvent's bubble point by the dissolved solute, or
        the dew point of a humid gas, neither of which thermosteam's
        bubble/dew point functions see), located by bisection to `T_tol`
        kelvin. Returns [(t_before, t_after), ...].
        """
        out = []
        dt_tol = T_tol / max(self.Tb - self.Ta, T_tol)
        for a, b in zip(ts[:-1], ts[1:]):
            pa, pb = self.phase_at(a), self.phase_at(b)
            if pa is None or pb is None or pa == pb: continue
            while b - a > dt_tol:
                m = 0.5 * (a + b)
                pm = self.phase_at(m)
                if pm is None: break
                if pm == pa: a = m
                else: b = m
            out.append((a, b))
        return out

    def state(self, t):
        s = _copy(self.template)
        s.vle(T=self.Ta + (self.Tb - self.Ta) * t, P=self.P)
        return s

    def sample_state(self, t):
        """
        The state evaluated at `t` (a copy), or a fresh flash if `t` was
        never evaluated: a fresh TP flash does not always reproduce a
        sample, because thermosteam's TP flash of some mixtures depends on
        its starting point (see `StreamCurve._glide_root`).
        """
        s = self.states.get(t)
        return self.state(t) if s is None else _copy(s)


def _illinois(f, a, b, fa, fb, xtol=1e-12, maxiter=100):
    """Root of a monotone f on the bracket [a, b] (fa, fb of opposite sign)."""
    side = 0
    x = a
    for _ in range(maxiter):
        x = (a * fb - b * fa) / (fb - fa) if fb != fa else 0.5 * (a + b)
        if not (min(a, b) <= x <= max(a, b)): x = 0.5 * (a + b)
        fx = f(x)
        if fx == 0. or abs(b - a) < xtol: return x
        if (fx > 0.) == (fb > 0.):
            b, fb = x, fx
            if side == -1: fa *= 0.5
            side = -1
        else:
            a, fa = x, fx
            if side == 1: fb *= 0.5
            side = 1
    return x


def _sample_glide(f, t0, s0, t1, s1, tol_T, max_samples, init=None):
    """
    Adaptive sampling of a glide between parameters t0 < t1 whose exact end
    states s0 = (T0, H0), s1 = (T1, H1) are given. The worst segment is split
    first (at its parameter midpoint) until every segment's midpoint lies
    within `tol_T` kelvin (horizontally, at the midpoint's enthalpy) of the
    chord, or `max_samples` evaluations were spent. Samples that fail or are
    not monotone with respect to their neighbors are dropped. A glide no
    wider than `tol_T` needs no interior sample (the chord cannot be further
    than its width from the curve); up to 3 uniform samples are taken first
    so that an S-shaped stretch cannot fool the first midpoint test.

    `init` may hold already known interior samples {t: (T, H)}, which then
    replace the uniform ones.

    Returns (t, T, H) arrays sorted by t, and the largest remaining error
    [K] (0 when the tolerance was met).
    """
    pts = {t0: s0, t1: s1}
    n = 0
    width = s1[0] - s0[0]
    if width <= tol_T:
        return np.array([t0, t1]), np.array([s0[0], s1[0]]), \
               np.array([s0[1], s1[1]]), 0.
    if init:
        pts.update({t: r for t, r in init.items() if t0 < t < t1})
        n0 = 1
    else:
        n0 = 4 if width > 1. else 2 if width > 8. * tol_T else 1

    def ok(r, a, b):
        if r is None: return False
        (Ta, Ha), (Tb, Hb) = pts[a], pts[b]
        T, H = r
        return Ta <= T <= Tb and Ha <= H <= Hb

    ts = np.linspace(t0, t1, n0 + 1)
    for t in ts[1:-1]:
        r = f(t)
        n += 1
        if r is not None: pts[t] = r
    # enforce monotonicity of the initial set (drop offenders)
    keys = sorted(pts)
    clean = [keys[0]]
    for k in keys[1:-1]:
        if pts[clean[-1]][0] <= pts[k][0] <= pts[t1][0] and \
           pts[clean[-1]][1] <= pts[k][1] <= pts[t1][1]:
            clean.append(k)
        else:
            del pts[k]
    clean.append(t1)
    heap = []

    def push(a, b):
        nonlocal n
        m = 0.5 * (a + b)
        if n >= max_samples: return False
        r = f(m)
        n += 1
        if not ok(r, a, b): return True  # failed sample: keep the chord
        pts[m] = r
        (Ta, Ha), (Tb, Hb), (Tm, Hm) = pts[a], pts[b], r
        dH = Hb - Ha
        err = abs(Ta + (Tb - Ta) * (Hm - Ha) / dH - Tm) if dH > 0. else 0.
        if err > tol_T: heapq.heappush(heap, (-err, a, m, b))
        return True

    err_left = 0.
    for a, b in zip(clean[:-1], clean[1:]):
        if not push(a, b): err_left = np.inf  # not even checked
    while heap and err_left == 0.:
        err, a, m, b = heapq.heappop(heap)
        if not push(a, m) or not push(m, b):
            err_left = -err
    if heap: err_left = max(err_left, -heap[0][0])
    keys = sorted(pts)
    T = np.array([pts[k][0] for k in keys])
    H = np.array([pts[k][1] for k in keys])
    return np.array(keys), T, H, err_left


def _glide_crossings(sampler, t, T, H, H_lo, H_hi, tol_H, tol_T, max_samples):
    """
    Glide samples (t, T, H) completed with the exact states at which the
    glide crosses the ends of the stream's enthalpy range, H_lo and H_hi,
    between two samples. That happens when an end state is not at
    equilibrium and its enthalpy falls inside the glide, e.g. a hot stream
    fed as a liquid above its bubble point: equilibrium reaches the feed's
    enthalpy inside the glide, below the feed temperature, and above that
    temperature the curve is clipped to H_hi. Without the crossing, the
    chord from the last sample inside the range to the first clipped one
    ends at the clipped sample's temperature, off by up to the sample
    spacing rather than by `tol_T`, at a breakpoint (where the curve is
    taken as exact). The crossing is solved on the glide's own parameter
    (illinois), and the part of the bracket inside the range is sampled
    again to `tol_T`; if the solve fails, the chord's crossing is used.

    Returns (t, T, H, error) like `_sample_glide`.
    """
    err = 0.
    for H_end in (H_hi, H_lo):
        k = int(np.searchsorted(H, H_end, 'right'))  # H[k-1] <= H_end < H[k]
        if k == 0 or k == H.size: continue
        if H_end - H[k - 1] <= tol_H or H[k] - H_end <= tol_H: continue
        a, b = t[k - 1], t[k]
        found = {}

        def f(x):
            r = sampler(x)
            if r is None: raise RuntimeError('glide evaluation failed')
            found[x] = r
            return r[1] - H_end

        try:
            x = _illinois(f, a, b, H[k - 1] - H_end, H[k] - H_end)
            if x not in found: f(x)
            Tx, Hx = found[x]
            ok = (T[k - 1] <= Tx <= T[k] and a < x < b and abs(Hx - H_end)
                  <= _ROOT_RTOL * (H_hi - H_lo) + tol_H)
        except Exception:
            ok = False
        if ok:
            if H_end == H_hi:  # keep the samples inside the range
                g = _sample_glide(sampler, a, (T[k - 1], H[k - 1]),
                                  x, (Tx, H_end), tol_T, max_samples)
                part = slice(1, None)
            else:
                g = _sample_glide(sampler, x, (Tx, H_end), b, (T[k], H[k]),
                                  tol_T, max_samples)
                part = slice(None, -1)
            ts, Ts, Hs = g[0][part], g[1][part], g[2][part]
            err = max(err, g[3])
        else:
            w = (H_end - H[k - 1]) / (H[k] - H[k - 1])
            ts = np.array([a + w * (b - a)])
            Ts = np.array([T[k - 1] + w * (T[k] - T[k - 1])])
            Hs = np.array([H_end])
        t = np.concatenate((t[:k], ts, t[k:]))
        T = np.concatenate((T[:k], Ts, T[k:]))
        H = np.concatenate((H[:k], Hs, H[k:]))
    return t, T, H, err


def _sensible_crossing(H_of_T, Ta, Ha, Tb, Hb, H_end):
    """
    Temperature in (Ta, Tb) at which a single-phase stretch's exact
    enthalpy `H_of_T` equals `H_end` (Ha < H_end < Hb), or None if an
    evaluation failed (see `_glide_crossings` for why the curve needs it).
    """
    def f(T):
        H = H_of_T(T)
        if H is None: raise RuntimeError('evaluation failed')
        return H - H_end
    try:
        T = _illinois(f, Ta, Tb, Ha - H_end, Hb - H_end, xtol=1e-10)
    except Exception:
        return None
    return T if Ta < T < Tb else None


def _sensible_samples(H_of_T, Ta, Ha, Tb, Hb, clip, tol_T, max_samples):
    """
    Interior breakpoints of a single-phase stretch [Ta, Tb] (enthalpies
    Ha, Hb already clipped) such that linear interpolation between them is
    within `tol_T` kelvin of the exact (clipped) H(T): bisection in T until
    every segment's midpoint lies within `tol_T` (horizontally) of its chord
    or the segment is no wider than `tol_T`. Returns [(T, H), ...] sorted.

    Why: with temperature-dependent heat capacities the cascade is NOT
    linear between grid points, so two sensible streams whose composite
    curves touch inside an interval (e.g. liquid ethanol, whose Cp rises
    ~40 % over 100 K, against liquid water) hide a pinch from a table that
    only checks the interval ends. With these breakpoints every stream is
    within `tol_T` of piecewise linear on the grid, so the grid minimum is
    the true one to within ``tol_T * sum(CP)``.
    """
    out = []
    stack = [(Ta, Ha, Tb, Hb)]
    n = 0
    while stack and n < max_samples:
        a, fa, b, fb = stack.pop()
        if b - a <= tol_T or fb <= fa: continue
        m = 0.5 * (a + b)
        fm = H_of_T(m)
        n += 1
        if fm is None: continue
        fm = clip(fm)
        if abs(fm - 0.5 * (fa + fb)) * (b - a) <= tol_T * (fb - fa): continue
        out.append((m, fm))
        stack.append((m, fm, b, fb))
        stack.append((a, fa, m, fm))
    out.sort()
    return out


def _phase_mol(s, phase, IDs):
    if len(s.phases) > 1:
        if phase not in s.phases: return np.zeros(len(IDs))
        return np.array(s.imol[phase, IDs], dtype=float)
    if s.phase != phase: return np.zeros(len(IDs))
    return np.array(s.imol[IDs], dtype=float)


def _lever(A, B, H, T):
    """
    State at temperature `T` with enthalpy `H` made of a fraction of state
    `A` and the rest of state `B` (both taken to `T` with their phases
    fixed; thermosteam enthalpies are additive in the phase flows, so H is
    linear in the fraction). This is the state on a flat (isothermal)
    segment of a `StreamCurve`: for a pure component's latent heat it equals
    the PH flash at T_sat; for the end jump of a non-equilibrium end state
    (e.g. a vapor below its dew point) it stays at the end's temperature,
    where the curve puts that heat, whereas a PH flash would move it to the
    equilibrium temperature, outside the stream's own range.
    """
    a = _copy(A); a.T = T
    b = _copy(B); b.T = T
    Ha, Hb = a.H, b.H
    if Hb == Ha: return a
    w = (H - Ha) / (Hb - Ha)
    w = 0. if w < 0. else 1. if w > 1. else w
    IDs = a.chemicals.IDs
    phases = tuple(sorted(set(a.phases) | set(b.phases)))
    s = _copy(a)
    if len(phases) > 1 and set(s.phases) != set(phases): s.phases = phases
    for p in phases:
        mol = (1. - w) * _phase_mol(a, p, IDs) + w * _phase_mol(b, p, IDs)
        if len(phases) > 1: s.imol[p, IDs] = mol
        else: s.imol[IDs] = mol
    s.T = T
    return s


def _end_state(stream_end, T_lo, T_hi):
    """
    Return a copy of `stream_end` at equilibrium at its own enthalpy, or the
    stream as given if that equilibrium state lies outside the stream's own
    temperature range [T_lo, T_hi] (e.g. a non-condensable mislabelled as a
    liquid, whose equilibrium state at the same enthalpy is a gas at an
    absurd temperature). Either way the enthalpy is exactly `stream_end.H`.
    """
    stream = _copy(stream_end)
    try:
        stream.vle(H=stream_end.H, P=stream.P)
    except Exception:
        return _copy(stream_end)
    if T_lo <= stream.T <= T_hi: return stream
    return _copy(stream_end)


# %% The curve

class StreamCurve:
    """
    Piecewise-linear temperature-enthalpy curve of one process stream over
    its own enthalpy range, built once (a handful of flashes) and evaluated
    afterwards without flashing.

    Parameters
    ----------
    stream_in, stream_out : Stream
        The stream's real end states (the outlet already re-flashed at its
        own enthalpy, as `problem_table` expects). The curve keeps private
        copies.
    is_hot : bool, optional
        True if the stream is cooled. Defaults to ``H_in > H_out``.
    tol_T : float, optional
        Maximum linear-interpolation error along a two-phase glide or a
        curved single-phase stretch [K]. Defaults to `GLIDE_TOL_T`.
    max_samples : int, optional
        Maximum number of equilibrium evaluations per glide or stretch.
    label : str, optional
        Name used in warnings (defaults to the inlet's ID).

    Attributes
    ----------
    T, H : numpy.ndarray
        Breakpoints [K, kJ/hr] sorted by enthalpy; both non-decreasing. The
        first is the low-enthalpy end (``H[0] == H_lo``), the last the
        high-enthalpy end (``H[-1] == H_hi``), exactly.
    kinds : tuple[str]
        Kind of each segment i (breakpoints i -> i+1): 'flat' (isothermal:
        a pure component's latent heat at T_sat between its
        saturated-liquid and -vapor enthalpies, the enthalpy by which a
        non-equilibrium end state departs from equilibrium at its own
        temperature, or the whole duty of a non-monotone stream); 'sensible'
        (one phase: exact H(T) from a phase-fixed reference state, clipped;
        subdivided until linear interpolation is within `tol_T` kelvin,
        because Cp varies with T); 'glide' (a mixture's two-phase region:
        linear between samples no more than `tol_T` from the true curve).
    jumps : tuple[tuple[float, float]]
        Enthalpy ranges (H_a, H_b) of the flats that come from a
        non-equilibrium end state rather than from latent heat: flats of a
        monotone curve at an end temperature whose PH flash at
        mid-enthalpy lands outside [T_lo, T_hi] (e.g. a vapor fed below its
        dew point: its excess over the equilibrium liquid at the feed
        temperature). A state strictly inside a jump exists only as a mix
        of the two end states at the jump's temperature (see `state_at_H`);
        a PH flash there leaves the stream's range. Always empty for a
        point-load (non-monotone) curve.
    monotone : bool
        False for streams whose outlet temperature does not move with their
        duty (isothermal, or e.g. a reboiler outlet at VLE colder than its
        inlet); their curve is one flat segment at the outlet temperature,
        i.e. a point load there.
    T_lo, T_hi, H_lo, H_hi : float
        The stream's own temperature and enthalpy range.
    boundaries : dict
        Phase boundaries found inside the range ('T_sat', or 'T_bubble' /
        'T_dew'), real temperatures [K].

    Notes
    -----
    Inside [T_lo, T_hi] the stream is taken at equilibrium at the real
    temperature, with the enthalpy clipped to [H_lo, H_hi] so that a
    non-equilibrium end state can never inflate the duty; at its own two end
    temperatures it has exactly its end enthalpies. Where the equilibrium
    enthalpy crosses H_lo or H_hi inside the range (e.g. a liquid fed above
    its bubble point), the exact crossing state is a breakpoint, beyond
    which the curve is vertical (clipped). If equilibrium at an end
    temperature differs from the end's own enthalpy (a superheated liquid or
    subcooled vapor feed, say), the difference is a flat segment AT that end
    temperature: the limit of sampling ever more densely, and a property of
    the stream alone.

    Linearization error: with every glide sample, and every breakpoint of
    a curved single-phase stretch, within `tol_T` kelvin of the chord, the
    linearized stream is the true one shifted by at most `tol_T` in
    temperature at every enthalpy. Moving a hot stream's heat to higher
    temperature (or a cold stream's demand to lower temperature) never
    raises the hot-utility target, so the targets computed from the curves
    are bracketed by the exact targets at T_min_app -/+ 2 tol_T. The bound
    rests on midpoint tests, so it is approximate rather than guaranteed.

    """
    __slots__ = ('T', 'H', 'kinds', 'evals', 'glides', 'T_in', 'T_out',
                 'H_in', 'H_out', 'T_lo', 'T_hi', 'H_lo', 'H_hi', 'P',
                 'monotone', 'is_hot', 'label', 'boundaries', 'n_evals',
                 'glide_error', 'stream_in', 'stream_out', 'tol_H',
                 'method', 'tol_T', 'max_samples', 'jumps')

    def __init__(self, stream_in, stream_out, is_hot=None,
                 tol_T=GLIDE_TOL_T, max_samples=GLIDE_MAX_SAMPLES,
                 label=None):
        self.label = label or stream_in.ID
        self.tol_T = tol_T
        self.max_samples = max_samples
        # private copies: later changes to the caller's streams cannot
        # change the curve's end states
        self.stream_in = stream_in = _copy(stream_in)
        self.stream_out = stream_out = _copy(stream_out)
        self.T_in = T_in = stream_in.T
        self.T_out = T_out = stream_out.T
        self.H_in = H_in = stream_in.H
        self.H_out = H_out = stream_out.H
        self.P = stream_in.P
        if is_hot is None: is_hot = H_in > H_out
        self.is_hot = is_hot = bool(is_hot)
        sign = 1. if is_hot else -1.
        self.H_lo, self.H_hi = H_lo, H_hi = sorted((H_in, H_out))
        self.tol_H = 1e-9 * (abs(H_lo) + abs(H_hi) + (H_hi - H_lo)) + 1e-12
        self.boundaries = {}
        self.n_evals = 0
        self.glide_error = 0.
        self.glides = ()
        self.monotone = monotone = sign * (T_in - T_out) > 0.
        if not monotone:
            self.T_lo = self.T_hi = T_out
            self.T = np.array([T_out, T_out])
            self.H = np.array([H_lo, H_hi])
            self.kinds = (FLAT,)
            self.evals = (None,)
            self.method = 'point load'
        else:
            self.T_lo, self.T_hi = sorted((T_in, T_out))
            try:
                pieces = self._phase_pieces(tol_T, max_samples)
            except Exception as error:
                warn(f"phase analysis failed for stream {self.label!r} "
                     f"({error!r}); falling back to TP flashes at grid points",
                     RuntimeWarning)
                pieces = [(SENSIBLE, self.T_lo, self.T_hi,
                           _TPFlash(stream_in, self.label))]
                self.method = 'legacy'
            self._assemble(pieces)
        self.jumps = self._find_jumps()

    # -- construction ------------------------------------------------------

    def _phase_pieces(self, tol_T, max_samples):
        """
        Split [T_lo, T_hi] into ascending pieces: (SENSIBLE, Ta, Tb,
        evaluator), (FLAT, T, T, (H_left, H_right)) or (GLIDE, Ta, Tb,
        (t, T, H, sampler)).
        """
        s = self.stream_in
        P = self.P
        T_lo, T_hi = self.T_lo, self.T_hi
        vle = s.vle_chemicals
        if not vle:
            self.method = 'no VLE'
            return [(SENSIBLE, T_lo, T_hi, _FixedPhase(s))]
        chemicals = s.chemicals
        mol = s.mol
        light = any(mol[chemicals.index(c.ID)] > 0.
                    for c in getattr(chemicals, 'light_chemicals', ()))
        heavy = any(mol[chemicals.index(c.ID)] * (getattr(c, 'N_solutes', 0) or 0) > 0.
                    for c in getattr(chemicals, 'heavy_chemicals', ()))
        IDs = tuple(c.ID for c in vle)
        if len(vle) == 1 and not (light or heavy):
            return self._pure_pieces(vle[0])
        # -- mixture ---------------------------------------------------
        if light:
            T_b = -np.inf
            satL = None
        else:
            satL = _copy(s)
            satL.vle(V=0, P=P)
            T_b = satL.T
            self.n_evals += 1
            if T_b >= T_hi - _T_SNAP:
                self.method = 'mixture, liquid'
                return [(SENSIBLE, T_lo, T_hi, _FixedPhase(satL))]
        binary = len(vle) == 2 and not (light or heavy)
        x_dew = None
        if heavy:
            T_d = np.inf
            satV = None
        else:
            satV = _copy(s)
            satV.vle(V=1, P=P)
            T_d = satV.T
            self.n_evals += 1
            if binary: x_dew = s.dew_point_at_P(P).x
            if T_d <= T_lo + _T_SNAP:
                self.method = 'mixture, vapor'
                return [(SENSIBLE, T_lo, T_hi, _FixedPhase(satV))]
        pieces = []
        Ta = T_lo
        if T_b > T_lo + _T_SNAP:
            pieces.append((SENSIBLE, T_lo, T_b, _FixedPhase(satL)))
            self.boundaries['T_bubble'] = T_b
            Ta = T_b
        Tb = T_hi
        top = None
        if T_d < T_hi - _T_SNAP:
            self.boundaries['T_dew'] = T_d
            Tb = T_d
            top = (SENSIBLE, T_d, T_hi, _FixedPhase(satV))
        # glide over [Ta, Tb]
        if binary:
            self.method = 'mixture, binary bubble-curve glide'
            sampler = _BinaryGlide(satL, IDs, s.get_bubble_point(), x_dew, P)
            T0, T1 = T_b, T_d
            fT = lambda t: sampler(t)[0]
            t_at = lambda T: _illinois(lambda t: fT(t) - T, 0., 1., T0 - T, T1 - T)
            # an azeotrope (T0 == T1) gives ta = 0, tb = 1: a flat, as for
            # a pure component
            ta = 0. if Ta <= T0 + _T_SNAP else t_at(Ta)
            tb = 1. if Tb >= T1 - _T_SNAP else t_at(Tb)
            sa = (T_b, satL.H) if ta == 0. else sampler(ta)
            sb = (T_d, satV.H) if tb == 1. else sampler(tb)
            if ta > 0.: sa = (Ta, sa[1])  # exact end temperature
            if tb < 1.: sb = (Tb, sb[1])
        else:
            self.method = 'mixture, TP-flash glide'
            template = satL if satL is not None else satV
            if template is None:  # light AND heavy solutes: all glide
                template = _copy(s)
                template.vle(T=T_lo, P=P)
            strict = not (light or heavy)  # [Ta, Tb] inside the true glide
            sampler = _TPGlide(template, Ta, Tb, P, IDs, strict)
            ta, tb = 0., 1.
            sa = (Ta, satL.H) if Ta == T_b else sampler(0.)
            sb = (Tb, satV.H) if Tb == T_d else sampler(1.)
            if sa is None or sb is None:
                raise RuntimeError('TP flash failed at a glide end')
            # the saturated end states are the glide's end samples
            if Ta == T_b: sampler.states[0.] = _copy(satL)
            if Tb == T_d: sampler.states[1.] = _copy(satV)
        t, T, H, err = _sample_glide(sampler, ta, sa, tb, sb, tol_T, max_samples)
        if not binary and not strict:
            # always-gas or dissolved-solute chemicals: thermosteam's bubble
            # and dew points ignore them, so the real phase boundaries (the
            # kinks of the curve) are found where the TP flashes change
            # phase, and single-phase stretches become exact sensible pieces
            cuts = sampler.phase_boundaries(t)
            if cuts:
                samples = {k: (Tk, Hk) for k, Tk, Hk in zip(t, T, H)}
                bounds = [(ta, sa)]
                for a, b in cuts:
                    ra, rb = sampler(a), sampler(b)
                    if ra is None or rb is None: continue
                    bounds.append((a, ra)); bounds.append((b, rb))
                bounds.append((tb, sb))
                err = 0.
                for (u, su), (w, sw) in zip(bounds[:-1], bounds[1:]):
                    if w <= u: continue
                    if sw[0] - su[0] <= _T_SNAP:  # the bracket of a boundary
                        pieces.append((GLIDE, su[0], sw[0],
                                       (np.array([u, w]), np.array([su[0], sw[0]]),
                                        np.array([su[1], sw[1]]), sampler)))
                        continue
                    m = 0.5 * (u + w)
                    if sampler.phase_at(m) in ('l', 'g'):
                        pieces.append((SENSIBLE, su[0], sw[0],
                                       _FixedPhase(sampler.state(m))))
                    else:
                        g = _sample_glide(sampler, u, su, w, sw, tol_T,
                                          max_samples, init=samples)
                        err = max(err, g[3])
                        pieces.append((GLIDE, su[0], sw[0], (*g[:3], sampler)))
                self.method += ' (phase boundaries located by TP flashes)'
                for i, (a, b) in enumerate(cuts):
                    self.boundaries[f'T_phase_{i}'] = float(
                        sampler.Ta + (sampler.Tb - sampler.Ta) * b)
                t = None
        self.n_evals += sampler.n
        self.glide_error = max(self.glide_error, err)
        if err > tol_T:
            warn(f"glide of stream {self.label!r} sampled to {err:.3g} K "
                 f"(tolerance {tol_T} K) within {max_samples} evaluations",
                 RuntimeWarning)
        if t is not None: pieces.append((GLIDE, Ta, Tb, (t, T, H, sampler)))
        if top is not None: pieces.append(top)
        return pieces

    def _pure_pieces(self, chemical):
        s = self.stream_in
        P = self.P
        T_lo, T_hi = self.T_lo, self.T_hi
        Pc = getattr(chemical, 'Pc', None)
        if Pc is not None and P >= Pc:
            self.method = 'pure, supercritical (TP flashes)'
            return [(SENSIBLE, T_lo, T_hi, _TPFlash(s, self.label))]
        satL = _copy(s)
        satL.vle(V=0, P=P)
        satV = _copy(s)
        satV.vle(V=1, P=P)
        self.n_evals += 2
        T_sat = satL.T
        if T_sat > T_hi + _T_SNAP:
            self.method = 'pure, liquid'
            return [(SENSIBLE, T_lo, T_hi, _FixedPhase(satL))]
        if T_sat < T_lo - _T_SNAP:
            self.method = 'pure, vapor'
            return [(SENSIBLE, T_lo, T_hi, _FixedPhase(satV))]
        self.method = 'pure, phase change'
        if T_sat - T_lo <= _T_SNAP: T_sat = T_lo
        elif T_hi - T_sat <= _T_SNAP: T_sat = T_hi
        self.boundaries['T_sat'] = T_sat
        pieces = []
        if T_sat > T_lo: pieces.append((SENSIBLE, T_lo, T_sat, _FixedPhase(satL)))
        pieces.append((FLAT, T_sat, T_sat, (satL.H, satV.H, satL, satV)))
        if T_sat < T_hi: pieces.append((SENSIBLE, T_sat, T_hi, _FixedPhase(satV)))
        return pieces

    def _assemble(self, pieces):
        H_lo, H_hi = self.H_lo, self.H_hi
        T_lo, T_hi = self.T_lo, self.T_hi
        tol_H = self.tol_H
        tol_T, max_samples = self.tol_T, self.max_samples
        clip = lambda H: H_lo if H < H_lo else H_hi if H > H_hi else H
        if self.H_in <= self.H_out:
            low_end, high_end = self.stream_in, self.stream_out
        else:
            low_end, high_end = self.stream_out, self.stream_in
        Ts = [T_lo]
        Hs = [H_lo]
        kinds = []
        evals = []  # sensible: evaluator; flat: (state_a, state_b) thunks
        # (lazy) state at the last breakpoint, for the flats' lever rule
        last = [lambda: _end_state(low_end, T_lo, T_hi)]

        def add(T, H, kind, ev=None, state=None):
            H = clip(H)
            if H < Hs[-1]: H = Hs[-1]  # keep non-decreasing
            if T < Ts[-1]: T = Ts[-1]
            elif T > T_hi: T = T_hi  # a phase boundary a hair above T_hi
            if T == Ts[-1]:
                if H - Hs[-1] <= tol_H:  # same point
                    if state is not None: last[0] = state
                    return
                kind, ev = FLAT, (last[0], state)
            elif kind == FLAT:  # cannot happen for well-formed pieces
                kind, ev = GLIDE, None  # (linear)
            Ts.append(T); Hs.append(H); kinds.append(kind); evals.append(ev)
            if state is not None: last[0] = state

        def at_T(ev, T):
            return lambda: ev.state_at_T(T)

        def at_t(sampler, t):
            return lambda: sampler.state(t)

        glides = []
        for kind, Ta, Tb, data in pieces:
            if kind == SENSIBLE:
                ev = data
                Ha = ev.H(Ta)
                # at T_lo: a jump here is the non-equilibrium excess of the end
                if Ha is not None: add(Ta, Ha, FLAT, state=at_T(ev, Ta))
                Hb = ev.H(Tb)
                if Hb is None:  # fallback flash failed: linear
                    Hb = H_hi if Tb == T_hi else Hs[-1]
                elif Ha is not None:  # curvature (temperature-dependent Cp)
                    # exact breakpoints where the stretch enters and leaves
                    # the enthalpy range (see `_glide_crossings`)
                    a, fa, b, fb = Ta, clip(Ha), Tb, clip(Hb)
                    if Ha < H_lo - tol_H and Hb > H_lo + tol_H:
                        T_c = _sensible_crossing(ev.H, Ta, Ha, Tb, Hb, H_lo)
                        if T_c is not None:
                            add(T_c, H_lo, SENSIBLE, ev, state=at_T(ev, T_c))
                            a, fa = T_c, H_lo
                    T_top = None
                    if Ha < H_hi - tol_H and Hb > H_hi + tol_H:
                        T_top = _sensible_crossing(ev.H, Ta, Ha, Tb, Hb, H_hi)
                        if T_top is not None and T_top > a: b, fb = T_top, H_hi
                        else: T_top = None
                    for Tk, Hk in _sensible_samples(ev.H, a, fa, b, fb,
                                                    clip, tol_T, max_samples):
                        add(Tk, Hk, SENSIBLE, ev, state=at_T(ev, Tk))
                    if T_top is not None:
                        add(T_top, H_hi, SENSIBLE, ev, state=at_T(ev, T_top))
                add(Tb, Hb, SENSIBLE, ev, state=at_T(ev, Tb))
            elif kind == FLAT:
                H_l, H_v, sL, sV = data
                add(Ta, H_l, FLAT, state=partial(_copy, sL))
                add(Ta, H_v, FLAT, state=partial(_copy, sV))
            else:  # GLIDE
                t, T, H, sampler = data
                t, T, H, err = _glide_crossings(sampler, t, T, H, H_lo, H_hi,
                                                tol_H, tol_T, max_samples)
                self.glide_error = max(self.glide_error, err)
                add(T[0], H[0], FLAT, state=at_t(sampler, t[0]))
                for tk, Tk, Hk in zip(t[1:], T[1:], H[1:]):
                    add(Tk, Hk, GLIDE, state=at_t(sampler, tk))
                glides.append((t, T, H, sampler))
        self.glides = tuple(glides)
        # exact high end
        if abs(Hs[-1] - H_hi) <= tol_H and Ts[-1] == T_hi:
            Hs[-1] = H_hi
        else:
            if Ts[-1] < T_hi:
                # vertical (clipped) stretch up to T_hi: H constant
                Ts.append(T_hi); Hs.append(Hs[-1])
                kinds.append(SENSIBLE); evals.append(None)
            Ts.append(T_hi); Hs.append(H_hi)
            kinds.append(FLAT)
            evals.append((last[0], lambda: _end_state(high_end, T_lo, T_hi)))
        self.T = np.array(Ts)
        self.H = np.array(Hs)
        self.kinds = tuple(kinds)
        self.evals = tuple(evals)

    def _find_jumps(self):
        """
        Enthalpy ranges of the flats that stem from a non-equilibrium end
        state (see `jumps`). Only the flats of a monotone curve at an end
        temperature can be jumps (interior flats are latent heat; the flat
        of a point-load stream is its whole duty); each candidate costs one
        PH flash.
        """
        if not self.monotone: return ()
        jumps = []
        T_lo, T_hi = self.T_lo, self.T_hi
        for j, kind in enumerate(self.kinds):
            if kind != FLAT: continue
            T = self.T[j]
            if T_lo < T < T_hi: continue
            Ha, Hb = float(self.H[j]), float(self.H[j + 1])
            if Hb <= Ha: continue
            s = _copy(self.stream_in)
            try:
                s.vle(H=0.5 * (Ha + Hb), P=self.P)
                T_eq = s.T
            except Exception:  # no equilibrium state there at all
                T_eq = np.nan
            if not (T_lo - _T_JUMP <= T_eq <= T_hi + _T_JUMP):
                jumps.append((Ha, Hb))
        return tuple(jumps)

    # -- evaluation ----------------------------------------------------------

    def _segment_H(self, j, T):
        """Enthalpy at real T strictly inside segment j (T[j] < T < T[j+1])."""
        Ta, Tb = self.T[j], self.T[j + 1]
        Ha, Hb = self.H[j], self.H[j + 1]
        kind = self.kinds[j]
        if kind == SENSIBLE and self.evals[j] is not None:
            H = self.evals[j].H(T)
            if H is not None:
                return Ha if H < Ha else Hb if H > Hb else H
        if Tb == Ta: return Hb
        return Ha + (Hb - Ha) * (T - Ta) / (Tb - Ta)

    def _segment_of_H(self, H):
        """Index j of the segment with H[j] < H < H[j+1], or None if `H` is
        a breakpoint enthalpy or outside (H_lo, H_hi)."""
        Hbp = self.H
        if not (Hbp[0] < H < Hbp[-1]): return None
        j = int(np.searchsorted(Hbp, H, 'right')) - 1
        if Hbp[j] == H: return None
        return j

    def _glide_at(self, H):
        """The glide sample arrays (t, T, H, sampler) that contain `H`."""
        for g in self.glides:
            Hg = g[2]
            if Hg[0] <= H <= Hg[-1]: return g
        return None

    def _glide_root(self, H, state=False):
        """
        ``(t, (T, H), state)`` of the glide state with enthalpy `H`, solved
        between the two glide samples that bracket it (`state`: a copy of
        that very state, whose enthalpy is the one matched, if requested;
        else None), or None if `H` is in no glide, an evaluation failed, or
        the root missed `H` by more than `_ROOT_RTOL` times the duty.

        The parameters inside the bracket are evaluated with
        ``strict=False``: next to a bubble or dew point a TP flash can
        legitimately come back single phase, which a strict `_TPGlide`
        reports as a failure; the root then sits on that jump of the flash
        results, within about 1e-8 of the duty of `H`. A larger miss means
        the flashes are not monotone inside the bracket: thermosteam's TP
        flash of some mixtures returns spurious states at isolated
        temperatures (e.g. near-total vaporization inside the glide of
        water with 1.4 % methanol and 0.3 % glycerol, the reboiler of the
        `HeatExchangerNetwork` doctest system), which the glide sampler
        drops but a root solve cannot avoid.
        """
        g = self._glide_at(H)
        if g is None: return None
        t, Tg, Hg, sampler = g
        i = min(max(int(np.searchsorted(Hg, H, 'right')) - 1, 0), Hg.size - 2)
        last = {}
        evaluated = [None]

        def f(tt):
            r = sampler(tt, strict=False)
            if r is None: raise RuntimeError('glide evaluation failed')
            last[tt] = r
            evaluated[0] = tt
            return r[1] - H

        try:
            tt = _illinois(f, t[i], t[i + 1], Hg[i] - H, Hg[i + 1] - H)
            if evaluated[0] != tt: f(tt)  # the sampler must hold state tt
        except Exception:  # a flash that raised inside a TP glide
            return None
        r = last[tt]
        if abs(r[1] - H) > _ROOT_RTOL * (self.H_hi - self.H_lo) + self.tol_H:
            return None
        return tt, r, _copy(sampler.work) if state else None

    def _glide_state(self, H):
        """
        Glide state with enthalpy `H` (the `state_at_H` of a glide), or
        None if `H` is in no glide.

        In order: the root of `_glide_root` (a residual from a phase-boundary
        jump of the flash results, at most `_ROOT_RTOL` of the duty, is
        closed with the phase split fixed, a shift of ~1e-6 K); else a PH
        flash, accepted only if its temperature lies between the two glide
        samples that bracket `H` (the curve is monotone through exact
        samples, so the true state does); else the curve's own state: the
        lever-rule mix of the two bracketing sampled states at the
        linearized temperature `T_at(H)`. The fallbacks matter only where
        thermosteam's flashes fail or are not monotone inside a glide (see
        `_glide_root`).
        """
        g = self._glide_at(H)
        if g is None: return None
        tol_H = self.tol_H
        root = self._glide_root(H, state=True)
        if root is not None:
            s = root[2]  # the state whose enthalpy the root matched
            if abs(s.H - H) > tol_H:
                try: s.H = H
                except Exception: pass
            if abs(s.H - H) <= 10. * tol_H: return s
        t, Tg, Hg, sampler = g
        i = min(max(int(np.searchsorted(Hg, H, 'right')) - 1, 0), Hg.size - 2)
        s = _copy(self.stream_in)
        try:
            s.vle(H=H, P=self.P)
            if (Tg[i] - _T_EQ <= s.T <= Tg[i + 1] + _T_EQ
                    and abs(s.H - H) <= 10. * tol_H):
                return s
        except Exception:
            pass
        tol = _ROOT_RTOL * (self.H_hi - self.H_lo) + tol_H
        try:
            A, B = sampler.sample_state(t[i]), sampler.sample_state(t[i + 1])
            if abs(A.H - Hg[i]) > tol or abs(B.H - Hg[i + 1]) > tol: return None
            s = _lever(A, B, H, self.T_at(H))
            if abs(s.H - H) > tol_H: s.H = H
        except Exception:
            return None
        return s if abs(s.H - H) <= 10. * tol_H else None

    def H_at(self, T, side='right'):
        """
        Enthalpy [kJ/hr] at real temperature `T`. At a flat segment (a point
        load) 'left' gives the low-enthalpy end and 'right' the high one.
        Below T_lo it is H_lo, above T_hi it is H_hi.
        """
        Tbp = self.T
        il = int(np.searchsorted(Tbp, T - _T_EQ, 'left'))
        ir = int(np.searchsorted(Tbp, T + _T_EQ, 'right'))
        if ir > il:  # T is (within _T_EQ) a breakpoint temperature
            return self.H[il] if side == 'left' else self.H[ir - 1]
        if il == 0: return self.H_lo
        if il >= Tbp.size: return self.H_hi
        return self._segment_H(il - 1, T)

    def grid_limits(self, Ts, shift, index):
        """
        Left and right enthalpy limits on a descending shifted grid `Ts`.
        `index[i]` is the grid position of breakpoint i (so breakpoints are
        matched by position, never by comparing shifted floats).
        """
        n = Ts.size
        Hl = np.empty(n)
        Hr = np.empty(n)
        top = index[-1]  # highest-T breakpoint -> smallest descending index
        bottom = index[0]
        Hl[:top] = Hr[:top] = self.H_hi
        Hl[bottom + 1:] = Hr[bottom + 1:] = self.H_lo
        # breakpoints: first (lowest H) is the left limit, last the right one
        H = self.H
        for i in range(index.size - 1, -1, -1):
            Hl[index[i]] = H[i]
        for i in range(index.size):
            Hr[index[i]] = H[i]
        # grid points strictly inside a segment
        kinds, evals = self.kinds, self.evals
        T = self.T
        for j in range(index.size - 1):
            k_hi, k_lo = index[j + 1], index[j]
            if k_lo - k_hi < 2: continue
            if kinds[j] == SENSIBLE and evals[j] is not None:
                for k in range(k_hi + 1, k_lo):
                    Hl[k] = Hr[k] = self._segment_H(j, Ts[k] + shift)
            else:  # linear (glide, or a clipped constant stretch)
                x = Ts[k_hi + 1:k_lo] + shift
                Hl[k_hi + 1:k_lo] = Hr[k_hi + 1:k_lo] = \
                    H[j] + (H[j + 1] - H[j]) * (x - T[j]) / (T[j + 1] - T[j])
        return Hl, Hr

    def T_at(self, H, side='low'):
        """
        Real temperature [K] at enthalpy `H`: exact in single-phase segments,
        linear inside glides, T_sat on a flat segment. Where the enthalpy is
        constant over a temperature range (a clipped non-equilibrium
        stretch), 'low' gives its lowest temperature and 'high' its highest.
        """
        Hbp = self.H
        if H <= Hbp[0]:
            if side == 'low': return self.T[0]
            return self.T[int(np.searchsorted(Hbp, Hbp[0], 'right')) - 1]
        if H >= Hbp[-1]:
            if side == 'high': return self.T[-1]
            return self.T[int(np.searchsorted(Hbp, Hbp[-1], 'left'))]
        j = int(np.searchsorted(Hbp, H, 'right')) - 1
        if Hbp[j] == H:  # exactly at a breakpoint enthalpy
            if side == 'low':
                return self.T[int(np.searchsorted(Hbp, H, 'left'))]
            return self.T[int(np.searchsorted(Hbp, H, 'right')) - 1]
        Ta, Tb = self.T[j], self.T[j + 1]
        Ha, Hb = Hbp[j], Hbp[j + 1]
        if Ta == Tb: return Ta
        if self.kinds[j] == SENSIBLE and self.evals[j] is not None:
            try:
                T = self.evals[j].state_at_H(H).T
                return Ta if T < Ta else Tb if T > Tb else T
            except Exception:
                pass
        return Ta + (Tb - Ta) * (H - Ha) / (Hb - Ha)

    def T_exact(self, H, side='low'):
        """
        Real temperature [K] of the stream's equilibrium state with enthalpy
        `H`, without the linearization of `T_at`: inside a glide the
        temperature of the glide state with that enthalpy (a binary's
        bubble-curve root, or the TP-flash root of other mixtures); in a
        single-phase segment the exact phase-fixed solution; T_sat on a
        flat; and `T_at(H, side)` at breakpoints and on a vertical (clipped)
        stretch. Costs a root solve (about ten equilibrium evaluations)
        only inside glides; elsewhere it equals `T_at`. Inside a glide it is
        the temperature of ``state_at_H(H)`` (see `_glide_state` for where
        thermosteam's flashes fail), or `T_at` if no state was found.
        """
        j = self._segment_of_H(H)
        if j is None or self.kinds[j] != GLIDE:
            return self.T_at(H, side)
        s = self._glide_state(H)
        if s is None: return self.T_at(H, side)
        T = s.T
        Ta, Tb = self.T[j], self.T[j + 1]
        return Ta if T < Ta else Tb if T > Tb else T

    def state_at_H(self, H):
        """
        Stream state with enthalpy `H` (clipped to the range): the
        equilibrium end state at either end (the stream as given if that
        equilibrium lies outside the range; at the inlet of a point-load
        stream, `_point_load_inlet`); single phase: the phase-fixed
        reference at the temperature that gives H (exact); flat (a pure
        component's T_sat or an end jump): the lever-rule mix of the flat's
        two end states at its temperature; binary glide: the bubble-curve
        state with that enthalpy (robust); other glides: a TP flash at the
        temperature (found by bracketing) whose enthalpy is H, with
        fallbacks where thermosteam's flashes fail (see `_glide_state`).
        """
        H_lo, H_hi = self.H_lo, self.H_hi
        end = None
        if H <= H_lo + self.tol_H:
            end = self.stream_in if self.H_in <= self.H_out else self.stream_out
        elif H >= H_hi - self.tol_H:
            end = self.stream_in if self.H_in > self.H_out else self.stream_out
        if end is not None:
            if end is self.stream_in and not self.monotone:
                return _point_load_inlet(end, self.T_out, self.is_hot)
            return _end_state(end, self.T_lo, self.T_hi)
        Hbp = self.H
        j = int(np.searchsorted(Hbp, H, 'right')) - 1
        j = min(j, len(self.kinds) - 1)
        kind = self.kinds[j]
        if kind == SENSIBLE and self.evals[j] is not None:
            return self.evals[j].state_at_H(H)
        if kind == GLIDE:
            s = self._glide_state(H)
            if s is not None: return s
        # flat (T_sat of a pure chemical, or an end jump): lever rule
        # between its two end states at the flat's temperature
        if kind == FLAT and self.monotone and isinstance(self.evals[j], tuple):
            try:
                A, B = (f() for f in self.evals[j])
                s = _lever(A, B, H, self.T[j])
            except Exception:
                s = None
            if s is not None and abs(s.H - H) <= 10. * self.tol_H:
                return s
        # non-monotone stream (a point load), clipped stretch, or a glide
        # none of whose states could be found: PH flash
        s = _copy(self.stream_in)
        s.vle(H=H, P=self.P)
        return s

    def state_at_T(self, T, side='right'):
        """
        Stream state at real temperature `T`, on the `side` branch of a flat
        segment there. Single-phase: the equilibrium state at exactly T;
        otherwise the state at the curve's enthalpy `H_at(T, side)` (inside
        a glide its temperature differs from T by at most the sampling
        tolerance, but its enthalpy is the one the problem table used).
        """
        H = self.H_at(T, side)
        Tbp = self.T
        il = int(np.searchsorted(Tbp, T - _T_EQ, 'left'))
        ir = int(np.searchsorted(Tbp, T + _T_EQ, 'right'))
        if ir == il and 0 < il < Tbp.size:
            j = il - 1
            if (self.kinds[j] == SENSIBLE and self.evals[j] is not None
                    and self.H[j] < H < self.H[j + 1]):
                return self.evals[j].state_at_T(T)
        return self.state_at_H(H)

    def __repr__(self):
        return (f"<StreamCurve {self.label!r}: {'hot' if self.is_hot else 'cold'} "
                f"{self.T_lo:.2f}-{self.T_hi:.2f} K, {len(self.kinds)} segments "
                f"({self.method})>")

    def show(self):
        """Print the curve, one segment per line."""
        print(repr(self))
        for i, kind in enumerate(self.kinds):
            print(f"  {self.T[i]:10.4f} -> {self.T[i+1]:10.4f} K  "
                  f"H {self.H[i]:14.1f} -> {self.H[i+1]:14.1f}  {kind}")


def stream_curves(streams_inlet, streams_quenched, is_hot, **kwargs):
    """One `StreamCurve` per stream (see `StreamCurve` for keywords)."""
    return [StreamCurve(si, so, hot, **kwargs)
            for si, so, hot in zip(streams_inlet, streams_quenched, is_hot)]
