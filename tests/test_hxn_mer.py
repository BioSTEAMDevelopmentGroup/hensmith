# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Minimum-energy-requirement (MER) corpus tests for ``HeatExchangerNetwork``.

The problems live in ``hxn_mer_cases.py`` (data only). Every network is
synthesized through the public facility exactly as a user would (one
simulated ``HXutility`` per process stream, a ``HeatExchangerNetwork`` in the
same ``System``; see ``test_hxn_regression.synthesize``), and hensmith's
results are compared with small, independent references written in this
module (they call no hensmith code).

The two sets
------------
NO_SPLIT
    Problems for which an UNSPLIT network reaching the MER targets exists.
    The synthesized network must reach the targets (``synthesis_info
    ['status'] == 'mer'``), with every material and energy balance closed
    and every exchanger feasible. Each case carries a certificate network
    that proves the claim.
SPLIT
    Problems for which the pinch design rules PROVE that MER needs stream
    splitting. hensmith does not split streams, so the network must stay
    feasible and balanced, never beat the targets, and report
    ``'best_effort'``. A strict miss is not asserted: alternating repeated
    matches can approach MER arbitrarily closely.
Each set holds at least five problems with more than ten streams.

Provenance
----------
Constant-CP literature problems (``kind='constant_cp'``; every case carries
its full citation and source URL):

* the Furman & Sahinidis (2004) test set as digitized by Letsios,
  Kouyialis & Misener (2018) (``4sp1``, ``7sp1``, ``7sp4``, ``8sp1_sargent``,
  ``10sp1``, ``12sp1``, ``14sp1``, ``20sp1``, ``23sp1``, ``fs_*``);
* the Lee, Masso & Rudd (1970) problems as tabulated in degrees Fahrenheit
  by Muraki & Hayakawa (1982) (``4sp2``, ``5sp1``, ``7sp3``, ``8sp1_dt20F``);
* the Chen, Grossmann & Miller (2015) and Grossmann random instances from
  the same repository (``cgm_*``, ``grossmann_*``);
* industrial problems from the supplementary data of Caballero et al.
  (2021) and Yang, Zhang & Smith (2022) (``nitric_acid``, ``bandar_imam``,
  ``sorak_kravanja``, ``luo_20sp``, ``bjork_pettersson``,
  ``crude_fractionation``);
* textbook and lecture-note problems: Linnhoff & Hindmarsh (1983), Smith
  (2005) (``smith2005_*``), the NPTEL course "Process Integration"
  (``nptel_*``), Turton & Shaeiwitz, Cornell's process design wiki and
  Bagajewicz (``bagajewicz_*``).

They were collected in a certified benchmark built for hensmith's MER
synthesizer, where an independent oracle (numpy and scipy only) labelled
each problem: an unsplit MER network found by a stage-wise superstructure
MILP, re-solved as an LP and re-verified by plain arithmetic (NO_SPLIT), or
a pinch-rule proof (SPLIT). Here they are modelled with a constant-heat-
capacity pseudo-component, so their networks are exact arithmetic up to
rounding.

Real-thermodynamics problems (``kind='real_thermo'``, ``rtA*`` / ``rtB*``)
were designed for hensmith: water, ethanol, methanol and four of their
binaries at 1-10 bar, with sensible liquids and vapours, isothermal
condensers and boilers (several AT the pinch), binary glides, thresholds
and repeated pairs. Their reference targets come from an independent
dense-grid calculator (reproduced exactly on re-running it); their
certificates were designed by a pinch-outward planner, simulated with
biosteam ``HXprocess`` units and re-checked by a second independent
calculator, and three certificate duties were then lowered by at most
0.17 kJ/hr so that the exact planned states keep ``T_min_app`` (see each
case's ``note``).

What each check proves
----------------------
``test_mer_targets``
    hensmith's problem table, computed on the same streams the facility
    sees, equals an independent reference: the closed-form constant-CP
    cascade (``_cascade``) and the published values in source units, or the
    stored dense-grid targets for real thermodynamics. The pinch is a zero
    of the reference cascade (constant CP) or the reference pinch.
``test_no_split_certificate``
    An unsplit MER network exists, inside this suite: the stored certificate
    is walked stream by stream with plain arithmetic (constant CP:
    ``_verify_certificate``, a port of the oracle's checker) or with exact
    stream states (real thermodynamics: ``_verify_rt_certificate``), and it
    is feasible and meets the targets.
``test_split_proof``
    Splitting is necessary: the pinch-rule violation is re-derived here at
    the reference pinch (number rule, or no one-to-one CP-feasible pinch
    matching) and equals the stored proof. The rules are exact only for
    streams with a finite heat capacity at the pinch (an isothermal condenser
    or boiler AT the pinch can serve several partners in series), so the
    test also asserts that no stream has a point load at the pinch.
``test_network_balanced_and_feasible``
    The synthesized network is physically valid: energy balance error,
    heat - cool == net process duty, only ``HXprocess`` / ``HXutility``
    units (no splitter), every exchanger balances heat and mass, one life
    cycle per process stream whose stages follow each other, cover every
    exchanger port exactly once and never move the stream away from its
    outlet (a cold stream is never cooled), every stream enters and leaves
    the network in its original states, and the
    EXACT internal minimum approach inside every exchanger, computed from
    the streams' own states (``_Temperature``, ``_min_approach``), is at
    least ``T_min_app``. Synthesis raises no ``RuntimeWarning``.
``test_no_split_network_reaches_mer`` / ``test_split_network_never_beats_mer``
    The utilities equal both hensmith's targets and the independent reference
    (NO_SPLIT) or are never below them (SPLIT), and
    ``synthesis_info['status']`` says so.

Tolerances
----------
* Constant-CP targets: 1e-9 of the total stream duty (both sides are exact
  arithmetic on the same data; only rounding differs). Pinch temperatures:
  1e-6 degrees (grid points are stream end temperatures). Published values:
  the tolerance stored with each value (one unit in the last printed digit).
* Certificates: 1e-6 degrees and 1e-6 of the total duty, the oracle's own
  acceptance tolerance (certificates are stored to ~1e-12).
* Real-thermo targets vs the dense reference, per case
  (``_reference_tolerance``): twice the reference's own interpolation error
  at its pinch, plus 1e-9 of the total stream duty. Both targets are the
  cascade at the pinch (the pinches agree to 1e-6 K, and Q_cold - Q_hot is
  exact in both), so they differ by the sum of the streams' enthalpy
  errors there. The reference interpolates each stream linearly between
  grid nodes (uniform, ``grid_h`` apart at most, plus its phase-change
  temperatures): for a single-phase stream whose pinch temperature T lies
  in the grid interval [T_a, T_b] the error is, to leading order,
  ``(T - T_a) (T_b - T) / 2 * |dC/dT|`` (zero on a node, and for a pure
  component at its saturation temperature, which is a node). hensmith's
  curves are exact there: single-phase stretches are evaluated exactly at
  every grid temperature and latent heat is an exact flat; only binary
  glides are chords (within 0.002 K, i.e. 0.002 K times the glide's slope),
  and no glide crosses a reference pinch in this corpus (asserted). This
  estimate matches the measured differences (0 to 0.204 kJ/hr) to 0.1 %
  where one stream dominates and bounds them where errors cancel; the
  factor 2 covers the neglected higher-order terms, and 1e-9 of the total
  duty covers rounding and flash residuals (at most 5e-11 measured). The
  stored simulated certificate utilities, which are exact thermodynamics
  at the pinch, differ from the reference by the same amounts.
* Achieved utilities (NO_SPLIT) vs hensmith's targets: 1e-12 of the total
  stream duty for constant CP and 1e-10 for real thermodynamics (plus the
  reference tolerance above when compared with the reference): about 100
  times the largest deviation measured over the corpus (5e-15 and 1e-12,
  sorak_kravanja_ph13c7 and rtA10), the rounding of the constant-CP
  arithmetic and the residuals of the enthalpy flashes that realize a
  real-thermo network. SPLIT networks may not beat the targets by more
  than 1e-9 (rounding only).
* Balances: energy balance error < 1e-6 % (``test_hxn_regression``);
  heat - cool == net duty within 1e-8 of the total duty; per exchanger hot
  duty == cold duty, and per stream network end states == original ones,
  within 1e-6 of the streams' own duties (flash convergence, as above).
* Internal approach: ``T_min_app - 1e-6 K`` on exact states, the
  synthesizer's guarantee (its ``HXprocess.dT`` guard sits exactly there),
  plus 1e-9 K so that a guard-limited terminal is not decided by evaluation
  noise; temperatures come from single-phase Newton solves to 1e-10 K,
  T_sat, or an exact inversion along the bubble curve.
* Split proofs (real thermo): no stream end (other than the supply that
  creates the pinch) and no phase change within 0.5 K of the pinch.

References
----------
Linnhoff, B. & Flower, J. R. (1978). Synthesis of heat exchanger networks: I.
Systematic generation of energy optimal networks. AIChE J. 24, 633-642.
Linnhoff, B. & Hindmarsh, E. (1983). The pinch design method for heat
exchanger networks. Chem. Eng. Sci. 38, 745-763.
Kemp, I. C. (2007). Pinch Analysis and Process Integration, 2nd ed.
Butterworth-Heinemann.
"""
import math
import re
import time
import warnings
import numpy as np
import pytest
import biosteam as bst
import thermosteam as tmo
from hensmith import HeatExchangerNetwork
from hensmith.hxn_synthesis import problem_table
from hxn_mer_cases import NO_SPLIT, SPLIT, Q_UNITS, T_UNITS

# ---------------------------------------------------------------------------
# Tolerances (justified in the module docstring)
# ---------------------------------------------------------------------------

TARGET_RTOL = 1e-9      # constant-CP targets, x total stream duty
PINCH_TOL = 1e-6        # pinch temperatures [source degree or K]
CERT_TOL = 1e-6         # certificate temperatures, and x total duty for heat
REF_MARGIN = 2.         # x the dense reference's interpolation error at its pinch ...
REF_FLOOR = 1e-9        # ... + this x total duty (real-thermo targets vs reference)
MER_TOL = dict(constant_cp=1e-12, real_thermo=1e-10)  # achieved - targets, x total duty
BEAT_RTOL = 1e-9        # SPLIT utilities never below targets beyond this
EB_TOL = 1e-6           # |energy balance error| [%]
NET_RTOL = 1e-8         # heat - cool vs net duty, x total duty
DUTY_RTOL = 1e-6        # exchanger / stream enthalpies, x the streams' duties
APPROACH_TOL = 1e-6 + 1e-9  # K: the synthesizer's guard + T(H) evaluation noise
PINCH_MARGIN = 0.5      # K

def _name(case): return case['name']

def _is_cp(case): return case['kind'] == 'constant_cp'

# ---------------------------------------------------------------------------
# Building the process streams
# ---------------------------------------------------------------------------

_FLUID_THERMO = []

def _fluid_thermo():
    """Thermo with one constant-heat-capacity liquid pseudo-component: 1000*CP
    kmol/hr of it has a heat capacity flow rate of exactly 3600*CP kJ/hr/K,
    i.e. CP in kW/K, and its enthalpy is linear in temperature."""
    if not _FLUID_THERMO:
        fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1., Cn=3.6,
                             default=True)
        bst.settings.set_thermo([fluid], cache=True)
        _FLUID_THERMO.append(bst.settings.thermo)
    return _FLUID_THERMO[0]

def _kelvin(case, T):
    scale, offset = T_UNITS[case['T_unit']]
    return (T + offset) * scale

def _unit_ID(name):
    """A valid biosteam ID for a literature stream name ('1', 'C-1', ...)."""
    ID = re.sub(r'\W', '_', name)
    return ID if ID[0].isalpha() else 'S' + ID

def _build(case):
    """Return (units, T_min_app [K]): one simulated HXutility per process
    stream, i.e. what a user's flowsheet hands to HeatExchangerNetwork."""
    bst.main_flowsheet.set_flowsheet('mer_' + case['name'])
    units = []
    if _is_cp(case):
        bst.settings.set_thermo(_fluid_thermo())
        scale, _ = T_UNITS[case['T_unit']]
        kW = Q_UNITS[case['Q_unit']]
        for name, kind, T_in, T_out, CP in case['streams']:
            ID = _unit_ID(name)
            inlet = bst.Stream(ID + '_in', Fluid=1000. * CP * kW / scale,
                               T=_kelvin(case, T_in), P=101325., units='kmol/hr')
            units.append(bst.HXutility(ID, ins=inlet, T=_kelvin(case, T_out),
                                       rigorous=False))
        T_min_app = case['dTmin'] * scale
    else:
        bst.settings.set_thermo(case['chemicals'], cache=True)
        for ID, flows, T_in, T_out, P, phase, rigorous in case['streams']:
            inlet = bst.Stream(ID + '_in', P=P, units='kmol/hr', **flows)
            if T_in in ('bubble', 'dew'):
                inlet.vle(V=0 if T_in == 'bubble' else 1, P=P)
            else:
                inlet.T = T_in
                inlet.phase = phase
            if T_out in ('bubble', 'dew'):
                hx = bst.HXutility(ID, ins=inlet, V=0 if T_out == 'bubble' else 1,
                                   rigorous=True)
            else:
                hx = bst.HXutility(ID, ins=inlet, T=T_out, rigorous=rigorous)
            units.append(hx)
        T_min_app = case['T_min_app']
    for hx in units: hx.simulate()
    return units, T_min_app

def _hensmith_table(units, T_min_app):
    """hensmith's problem table on the streams HeatExchangerNetwork sees
    (inlets and re-flashed outlets of the utility exchangers)."""
    hus = sorted((hx.heat_utilities[0] for hx in units), key=lambda hu: hu.duty)
    inlets = [hu.unit.ins[0].copy() for hu in hus]
    outlets = [hu.unit.outs[0].copy() for hu in hus]
    for s in outlets: s.vle(H=s.H, P=s.P)
    return problem_table(inlets, outlets, [hu.duty < 0 for hu in hus], T_min_app)

def _duties(units):
    """(total, net) process duty [kJ/hr] of the utility exchangers."""
    duties = [hx.heat_utilities[0].unit_duty for hx in units]
    return sum(map(abs, duties)), sum(duties)

# ---------------------------------------------------------------------------
# Constant-CP references (source units, no hensmith code)
# ---------------------------------------------------------------------------

def _total_duty(case):
    return sum(CP * abs(T_out - T_in) for _, _, T_in, T_out, CP in case['streams'])

def _cascade(case):
    """Closed-form constant-CP problem table [Linnhoff & Flower 1978] in
    source units, hensmith convention (hot streams shifted down by dTmin).
    Returns (Q_hot, Q_cold, zeros, Ts): Ts are the shifted grid temperatures
    (descending) and zeros those where the feasible cascade vanishes (every
    pinch point; the top of the grid when Q_hot == 0)."""
    dT = case['dTmin']
    spans = [(T_out - dT, T_in - dT, CP) if kind == 'hot' else (T_in, T_out, -CP)
             for _, kind, T_in, T_out, CP in case['streams']]
    Ts = sorted({T for lo, hi, _ in spans for T in (lo, hi)}, reverse=True)
    heat = [0.]
    for upper, lower in zip(Ts, Ts[1:]):
        heat.append(heat[-1] + sum(CP * (upper - lower) for lo, hi, CP in spans
                                   if lo <= lower and hi >= upper))
    Q_hot = max(0., -min(heat))
    flow = [h + Q_hot for h in heat]
    tol = TARGET_RTOL * _total_duty(case)
    zeros = [T for T, f in zip(Ts, flow) if f <= tol]
    return Q_hot, flow[-1], zeros, Ts

def _finite(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))

def _verify_certificate(case):
    """Plain-arithmetic check of a constant-CP certificate (port of the
    benchmark oracle's ``verify_certificate``). Every stream passes its
    matches in series (hot streams in increasing stage, cold streams in
    decreasing stage) and ends with its utility; walking from T_in with Q/CP
    must reproduce the recorded temperatures, both ends of every match must
    keep dTmin (linear profiles, so the ends bound the whole exchanger), every
    stream must end at T_out, and the utilities must equal the closed-form
    targets. Every number must be finite: NaN compares False with everything
    and would otherwise pass. Returns a list of problems (empty: valid)."""
    dT = case['dTmin']
    cert = case['certificate']
    streams = {name: (kind, T_in, T_out, CP)
               for name, kind, T_in, T_out, CP in case['streams']}
    Q_hot, Q_cold, *_ = _cascade(case)
    qtol = CERT_TOL * max(1., _total_duty(case))
    problems = []
    matches = [dict(zip(('side', 'stage', 'hot', 'cold', 'Q', 'T_hot_in',
                         'T_hot_out', 'T_cold_in', 'T_cold_out'), m))
               for m in cert['matches']]
    for m in matches:
        if streams.get(m['hot'], ('',))[0] != 'hot' or streams.get(m['cold'], ('',))[0] != 'cold':
            problems.append(f"{m['hot']}-{m['cold']}: unknown stream or wrong kind")
        bad = [k for k in ('stage', 'Q', 'T_hot_in', 'T_hot_out', 'T_cold_in', 'T_cold_out')
               if not _finite(m[k])]
        if bad:
            problems.append(f"{m['hot']}-{m['cold']}: non-finite {bad}")
        elif not m['Q'] >= -qtol:
            problems.append(f"{m['hot']}-{m['cold']}: negative duty {m['Q']}")
    for kind, utility in (('cold', cert['hot_utility']), ('hot', cert['cold_utility'])):
        for name, Q in utility.items():
            if streams.get(name, ('',))[0] != kind or not _finite(Q) or not Q >= -qtol:
                problems.append(f'utility on {name}: not a {kind} stream, non-finite or negative')
    if problems: return problems
    temperatures = {}  # id(match) -> [T_hot_in, T_hot_out, T_cold_in, T_cold_out]
    for name, (kind, T_in, T_out, CP) in streams.items():
        hot = kind == 'hot'
        own = sorted((m for m in matches if m[kind] == name),
                     key=lambda m: m['stage'] if hot else -m['stage'])
        if len({m['stage'] for m in own}) < len(own):
            problems.append(f'{name}: two matches share a stage (a split)')
        sign, i = (-1., 0) if hot else (1., 2)
        T = T_in
        for m in own:
            t = temperatures.setdefault(id(m), [None] * 4)
            t[i] = T
            T += sign * m['Q'] / CP
            t[i + 1] = T
            recorded = (m['T_hot_in'], m['T_hot_out']) if hot else (m['T_cold_in'], m['T_cold_out'])
            if not (abs(recorded[0] - t[i]) <= CERT_TOL
                    and abs(recorded[1] - t[i + 1]) <= CERT_TOL):
                problems.append(f"{m['hot']}-{m['cold']}: recorded {name} temperatures "
                                f"{recorded} but the duties give {t[i:i + 2]}")
        T += sign * (cert['cold_utility'] if hot else cert['hot_utility']).get(name, 0.) / CP
        if not abs(T - T_out) <= CERT_TOL:
            problems.append(f'{name}: ends at {T:.9g}, target {T_out:.9g}')
    for m in matches:
        Th_in, Th_out, Tc_in, Tc_out = temperatures[id(m)]
        approach = min(Th_in - Tc_out, Th_out - Tc_in)
        if not approach >= dT - CERT_TOL:
            problems.append(f"{m['hot']}-{m['cold']}: approach {approach:.9g} < {dT}")
    for kind, total, target in (('hot', sum(cert['hot_utility'].values()), Q_hot),
                                ('cold', sum(cert['cold_utility'].values()), Q_cold)):
        if not abs(total - target) <= qtol:
            problems.append(f'{kind} utility {total:.9g} != MER target {target:.9g}')
    return problems

def _max_matching(musts, partners, ok):
    """Size of a maximum one-to-one matching (Kuhn's augmenting paths)."""
    owner = {}
    def augment(i, seen):
        for j, p in enumerate(partners):
            if j not in seen and ok(musts[i], p):
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False
    return sum(augment(i, set()) for i in range(len(musts)))

def _split_proofs(case):
    """Every pinch-rule violation of a constant-CP problem, top pinch first,
    above before below [Linnhoff & Hindmarsh 1983]. ABOVE a pinch each hot
    stream at the pinch (T_out <= T_p,hot < T_in) must end its last match
    against its own cold stream at the pinch (T_in <= T_p,cold < T_out) with
    CP_hot <= CP_cold; BELOW, each cold stream at the pinch needs its own hot
    stream with CP_hot >= CP_cold. Without splits a partner serves one such
    match, so a one-to-one matching that covers the 'must' streams is
    necessary; its absence proves that MER needs splitting."""
    dT = case['dTmin']
    *_, zeros, _ = _cascade(case)
    def snap(T, P): return P if abs(T - P) <= 1e-9 * max(1., abs(P)) else T
    proofs = []
    for pc in zeros:
        ph = pc + dT
        hot = [(n, snap(a, ph), snap(b, ph), CP)
               for n, k, a, b, CP in case['streams'] if k == 'hot']
        cold = [(n, snap(a, pc), snap(b, pc), CP)
                for n, k, a, b, CP in case['streams'] if k == 'cold']
        sides = (
            ('above', [s for s in hot if s[2] <= ph < s[1]], [s for s in cold if s[1] <= pc < s[2]],
             lambda h, c: h[3] <= c[3] * (1. + 1e-12)),
            ('below', [s for s in cold if s[1] < pc <= s[2]], [s for s in hot if s[2] < ph <= s[1]],
             lambda c, h: h[3] >= c[3] * (1. - 1e-12)),
        )
        for side, musts, partners, ok in sides:
            if len(musts) > len(partners): rule = 'number'
            elif _max_matching(musts, partners, ok) < len(musts): rule = 'cp'
            else: continue
            hot_at, cold_at = (musts, partners) if side == 'above' else (partners, musts)
            proofs.append(dict(side=side, rule=rule, pinch_hot=ph, pinch_cold=pc,
                               hot_at_pinch=sorted(s[0] for s in hot_at),
                               cold_at_pinch=sorted(s[0] for s in cold_at)))
    return proofs

# ---------------------------------------------------------------------------
# Exact stream temperatures (real thermodynamics and constant CP)
# ---------------------------------------------------------------------------

class _Temperature:
    """Temperature [K] of one stream's material at its own pressure as a
    function of its enthalpy flow H [kJ/hr], from forward property calls
    only (no TP/PH flash, so nothing is shared with the flashes of the code
    under test): a Newton solve on a single-phase copy outside the
    saturation envelope, T_sat inside it for a pure component, and an exact
    inversion along the bubble-point curve of the liquid for a binary glide
    (T = T_bubble(x), vapour y = y_bubble(x), lever rule on the first
    component). ``breaks`` are the saturated-liquid / saturated-vapour
    enthalpies, where T(H) has kinks."""

    def __init__(self, stream):
        thermo, P = stream.thermo, stream.P
        mol = np.array(stream.mol, dtype=float)
        self.P = P
        def single(phase, T):
            return bst.Stream(None, flow=mol.copy(), T=T, P=P, phase=phase,
                              units='kmol/hr', thermo=thermo)
        N = len(stream.vle_chemicals)
        if N == 0:
            self.kind, self.breaks = 'single', ()
            self.liquid = single(stream.phase if stream.phase in ('l', 'g') else 'l', stream.T)
            return
        if N > 2: raise NotImplementedError('only pure components and binaries')
        ref = stream.copy()
        self.kind = 'pure' if N == 1 else 'binary'
        self.T_bubble = ref.bubble_point_at_P(P).T
        self.T_dew = self.T_bubble if N == 1 else ref.dew_point_at_P(P).T
        self.liquid, self.vapor = single('l', self.T_bubble), single('g', self.T_dew)
        self.breaks = (self.liquid.H, self.vapor.H)
        if N == 2:
            self.bp = bp = ref.get_bubble_point()
            self.IDs = bp.IDs
            self.z = float(ref.get_normalized_mol(self.IDs)[0])
            self.F = float(ref.imol[self.IDs].sum())
            self._l = bst.Stream(None, P=P, phase='l', thermo=thermo)
            self._g = bst.Stream(None, P=P, phase='g', thermo=thermo)
            x_dew = float(ref.dew_point_at_P(P).x[0])
            self.xs = np.linspace(self.z, x_dew, 33)
            self.Hs = np.array([self._state(x)[1] for x in self.xs])

    def _state(self, x1):
        """(T, H) of the two-phase state whose liquid has mole fraction x1."""
        x = np.array([x1, 1. - x1])
        r = self.bp(x, P=self.P)
        y = np.asarray(r.y, float)
        V = 1. if abs(y[0] - x1) < 1e-15 else min(max((self.z - x1) / (y[0] - x1), 0.), 1.)
        self._l.imol[self.IDs] = self.F * (1. - V) * x
        self._g.imol[self.IDs] = self.F * V * y
        self._l.T = self._g.T = r.T
        return r.T, self._l.H + self._g.H

    @staticmethod
    def _newton(s, H):
        for _ in range(100):
            step = (H - s.H) / s.C
            s.T += step
            if abs(step) < 1e-10: break
        return s.T

    def __call__(self, H):
        if self.kind == 'single': return self._newton(self.liquid, H)
        H_bubble, H_dew = self.breaks
        tol = 1e-9 * max(1., H_dew - H_bubble)
        if H <= H_bubble + tol: return self._newton(self.liquid, H)
        if H >= H_dew - tol: return self._newton(self.vapor, H)
        if self.kind == 'pure': return self.T_bubble
        # binary glide: bracket on the stored bubble-curve states, then the
        # Illinois variant of regula falsi on the liquid mole fraction
        k = min(max(int(np.searchsorted(self.Hs, H)) - 1, 0), self.xs.size - 2)
        a, b = self.xs[k], self.xs[k + 1]
        fa, fb = self.Hs[k] - H, self.Hs[k + 1] - H
        T = self.T_bubble
        for _ in range(100):
            x = b - fb * (b - a) / (fb - fa) if fb != fa else 0.5 * (a + b)
            T, Hx = self._state(x)
            fx = Hx - H
            if abs(fx) <= 1e-12 * (H_dew - H_bubble): break
            if (fx > 0) != (fb > 0): a, fa = b, fb
            else: fa *= 0.5
            b, fb = x, fx
        return T

_TEMPERATURES = {}

def _temperature(stream):
    """Cached _Temperature for a stream's material (composition and P)."""
    mol = np.array(stream.mol, dtype=float)
    key = (id(stream.thermo), round(stream.P, 6), tuple(np.round(mol, 9)))
    if key not in _TEMPERATURES: _TEMPERATURES[key] = _Temperature(stream)
    return _TEMPERATURES[key]

def _min_approach(hot, H_hot_in, H_hot_out, cold, H_cold_in, H_cold_out, n=21):
    """Minimum hot-minus-cold temperature difference inside a counter-current
    exchanger (hot enters at H_hot_in, cold leaves at H_cold_out; position =
    duty q from the hot end), on exact states: n uniform duty points plus
    every phase boundary of either stream, then a golden-section refinement
    around the smallest sample (T(H) is smooth between the kinks, which are
    sample points)."""
    Q = H_hot_in - H_hot_out
    if Q == 0.: return math.inf       # no heat transferred
    if not Q > 0.: return math.nan    # reversed or NaN duty: never feasible
    qs = set(np.linspace(0., Q, n).tolist())
    qs.update(q for q in [H_hot_in - H for H in hot.breaks]
              + [H_cold_out - H for H in cold.breaks] if 0. < q < Q)
    qs = sorted(qs)
    def dT(q): return hot(H_hot_in - q) - cold(max(H_cold_out - q, H_cold_in))
    values = [dT(q) for q in qs]
    k = int(np.argmin(values))
    best = values[k]
    a, b = qs[max(k - 1, 0)], qs[min(k + 1, len(qs) - 1)]
    g = (math.sqrt(5.) - 1.) / 2.
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = dT(c), dT(d)
    for _ in range(30):
        if fc < fd: b, d, fd = d, c, fc; c = b - g * (b - a); fc = dT(c)
        else: a, c, fc = c, d, fd; d = a + g * (b - a); fd = dT(d)
    return min(best, fc, fd)

# ---------------------------------------------------------------------------
# Real-thermodynamics references
# ---------------------------------------------------------------------------

def _stream_ends(units):
    """{ID: (inlet, re-flashed outlet, is_hot)} of the process streams."""
    ends = {}
    for hx in units:
        inlet, outlet = hx.ins[0].copy(), hx.outs[0].copy()
        outlet.vle(H=outlet.H, P=outlet.P)
        ends[hx.ID] = (inlet, outlet, outlet.H < inlet.H)
    return ends

def _grid_error(ID, inlet, outlet, T, h):
    """Leading-order error [kJ/hr] of the dense reference's enthalpy of one
    stream `ID` (end states `inlet`, re-flashed `outlet`) at real temperature `T`:
    zero outside the stream's range or on a node of its grid (uniform with
    spacing at most `h` between its end temperatures, without the nodes
    inside a binary's two-phase range, plus its phase-change temperatures),
    else ``(T - T_a) (T_b - T) / 2 * |dC/dT|`` inside the grid interval
    [T_a, T_b] of a single-phase stretch (dC/dT by a central difference on
    a phase-fixed copy). A binary glide at `T` is not covered (AssertionError)."""
    lo, hi = sorted((inlet.T, outlet.T))
    if not lo + 1e-9 < T < hi - 1e-9: return 0.
    n = max(math.ceil((hi - lo) / h), 1)
    nodes = [float(x) for x in np.linspace(lo, hi, n + 1)]
    N = len(inlet.vle_chemicals)
    if N:
        f = _temperature(inlet)
        T_bubble, T_dew = f.T_bubble, f.T_dew
        if N > 1:
            nodes = [x for x in nodes if not T_bubble - 1e-6 <= x <= T_dew + 1e-6]
        nodes += [x for x in (T_bubble, T_dew) if lo - 1e-7 <= x <= hi + 1e-7]
        if T_bubble - 1e-6 <= T <= T_dew + 1e-6:
            assert N == 1, f'{ID}: a binary glide crosses the reference pinch'
            return 0.  # a pure component's latent heat, at a node
        phase = 'l' if T < T_bubble else 'g'
    else:
        phase = inlet.phase
    a = max(x for x in nodes if x <= T)
    b = min(x for x in nodes if x >= T)
    if T - a <= 1e-9 or b - T <= 1e-9: return 0.
    s = bst.Stream(None, flow=np.array(inlet.mol, dtype=float), T=T + 0.5, P=inlet.P,
                   phase=phase, units='kmol/hr', thermo=inlet.thermo)
    C_hi = s.C
    s.T = T - 0.5
    return (T - a) * (b - T) / 2. * abs(C_hi - s.C)

_REFERENCE_TOLERANCES = {}

def _reference_tolerance(case, units):
    """Accuracy [kJ/hr] of a real-thermo case's dense-grid reference targets
    (see the module docstring): REF_MARGIN times the sum of the streams'
    `_grid_error` at the reference pinch, plus REF_FLOOR of the total duty."""
    name = case['name']
    if name not in _REFERENCE_TOLERANCES:
        ref, dT = case['reference'], case['T_min_app']
        error = 0.
        if ref['pinch_T_shifted'] is not None:
            for ID, (inlet, outlet, hot) in _stream_ends(units).items():
                T = ref['pinch_T_shifted'] + (dT if hot else 0.)
                error += _grid_error(ID, inlet, outlet, T, ref['grid_h'])
        _REFERENCE_TOLERANCES[name] = REF_MARGIN * error + REF_FLOOR * _duties(units)[0]
    return _REFERENCE_TOLERANCES[name]

def _reference_targets(case, units):
    """(Q_hot, Q_cold, tol_hot, tol_cold) [kJ/hr] of the case's independent
    reference: the closed-form cascade (constant CP) or the stored dense-grid
    targets (real thermodynamics), with the tolerance of that reference."""
    if _is_cp(case):
        Q_hot, Q_cold, *_ = _cascade(case)
        kJ_per_hr = 3600. * Q_UNITS[case['Q_unit']]
        tol = TARGET_RTOL * _total_duty(case) * kJ_per_hr
        return Q_hot * kJ_per_hr, Q_cold * kJ_per_hr, tol, tol
    ref = case['reference']
    tol = _reference_tolerance(case, units)
    return ref['Q_hot'], ref['Q_cold'], tol, tol

def _verify_rt_certificate(case, units, T_min_app):
    """Walk a real-thermo certificate: each stream passes its matches in flow
    order (``hot_seq`` / ``cold_seq`` = 1..n) from its inlet enthalpy, every
    match keeps the exact internal approach >= T_min_app, no stream is
    over-heated or over-cooled by its matches, the far-end utilities
    reproduce the certificate's simulated utilities, and those equal the
    dense reference targets (a few duties were lowered by up to 0.17 kJ/hr
    after the simulation, see the cases' notes, so the walked utilities
    are compared with the simulated ones only). Returns (problems, worst
    approach [K])."""
    ends = _stream_ends(units)
    total = sum(abs(o.H - i.H) for i, o, _ in ends.values())
    problems, planned, last = [], {}, {}
    for m in case['certificate']:
        if not (_finite(m['Q']) and m['Q'] > 0. and m['hot'] in ends and m['cold'] in ends
                and ends[m['hot']][2] and not ends[m['cold']][2]):
            problems.append(f"{m['hot']}-{m['cold']}: bad duty {m['Q']!r}, "
                            "unknown stream or wrong kind")
    if problems: return problems, math.nan
    for ID, (inlet, outlet, hot) in ends.items():
        own = sorted((m[f"{'hot' if hot else 'cold'}_seq"], k)
                     for k, m in enumerate(case['certificate'])
                     if m['hot' if hot else 'cold'] == ID)
        if [s for s, _ in own] != list(range(1, len(own) + 1)):
            problems.append(f'{ID}: flow positions {[s for s, _ in own]} are not 1..n')
        H = inlet.H
        for _, k in own:
            H_next = H + (-1. if hot else 1.) * case['certificate'][k]['Q']
            planned[k, ID] = (H, H_next)
            H = H_next
        last[ID] = H
    worst = math.inf
    for k, m in enumerate(case['certificate']):
        (Hh_in, Hh_out), (Hc_in, Hc_out) = planned[k, m['hot']], planned[k, m['cold']]
        approach = _min_approach(_temperature(ends[m['hot']][0]), Hh_in, Hh_out,
                                 _temperature(ends[m['cold']][0]), Hc_in, Hc_out)
        worst = min(worst, approach)
        if not approach >= T_min_app - APPROACH_TOL:
            problems.append(f"{m['hot']}-{m['cold']} ({m['side']}): approach {approach:.7f} K")
    heat = cool = 0.
    for ID, (inlet, outlet, hot) in ends.items():
        need = outlet.H - last[ID]   # far-end utility: > 0 heating, < 0 cooling
        if (need > 0 if hot else need < 0) and abs(need) > DUTY_RTOL * abs(outlet.H - inlet.H):
            problems.append(f'{ID}: over-{"cooled" if hot else "heated"} by {abs(need):.6g} kJ/hr')
        heat += max(need, 0.)
        cool += max(-need, 0.)
    stored = case['certificate_utilities']
    ref = case['reference']
    tol_ref = _reference_tolerance(case, units)
    for label, Q, Q_cert, Q_ref in (('heating', heat, stored['Q_hot'], ref['Q_hot']),
                                    ('cooling', cool, stored['Q_cold'], ref['Q_cold'])):
        if not abs(Q - Q_cert) <= DUTY_RTOL * total:
            problems.append(f'{label} {Q:.10g} != simulated certificate {Q_cert:.10g}')
        if not abs(Q_cert - Q_ref) <= tol_ref:
            problems.append(f'{label}: simulated certificate {Q_cert:.10g} '
                            f'!= dense reference {Q_ref:.10g}')
    return problems, worst

def _rt_split_proof(case, units):
    """Re-derive the pinch number rule of a real-thermo SPLIT case at the
    reference pinch T_p (shifted scale; hot streams shifted down by
    T_min_app) from the streams' own temperatures. Above: hot streams that
    cross T_p vs cold streams that cross it or enter at it; below: cold
    streams that cross T_p vs hot streams that cross it or enter at it (a
    stream entering exactly at T_p is the one that creates the pinch).
    Returns (violations {side: (hot IDs, cold IDs)}, problems), where
    problems lists anything that would make the finite-CP rules
    inapplicable: an outlet at T_p, any other end or an in-range phase change
    (T_sat, bubble or dew point) within PINCH_MARGIN of T_p."""
    Tp, dT, tol = case['reference']['pinch_T_shifted'], case['T_min_app'], 1e-6
    rows, problems = [], []
    for ID, (inlet, outlet, hot) in _stream_ends(units).items():
        shift = dT if hot else 0.
        T_in, T_out = inlet.T - shift, outlet.T - shift
        lo, hi = sorted((T_in, T_out))
        rows.append((ID, hot, T_in, lo + tol < Tp < hi - tol))
        if abs(T_out - Tp) <= tol:
            problems.append(f'{ID}: outlet at the pinch')
        elif abs(T_out - Tp) < PINCH_MARGIN or (tol < abs(T_in - Tp) < PINCH_MARGIN):
            problems.append(f'{ID}: end temperature within {PINCH_MARGIN} K of the pinch')
        N = len(inlet.vle_chemicals)
        changes = [] if N == 0 else [inlet.bubble_point_at_P(inlet.P).T]
        if N == 2: changes.append(inlet.dew_point_at_P(inlet.P).T)
        for T in changes:
            if lo - tol <= T - shift <= hi + tol and abs(T - shift - Tp) < PINCH_MARGIN:
                problems.append(f'{ID}: phase change at {T:.3f} K within '
                                f'{PINCH_MARGIN} K of the pinch')
    crossing = lambda hot: [ID for ID, h, _, x in rows if h == hot and x]
    entering = lambda hot: [ID for ID, h, T_in, x in rows
                            if h == hot and not x and abs(T_in - Tp) <= tol]
    above = (crossing(True), crossing(False) + entering(False))
    below = (crossing(True) + entering(True), crossing(False))
    violations = {}
    if len(above[0]) > len(above[1]): violations['above'] = above
    if len(below[1]) > len(below[0]): violations['below'] = below
    return violations, problems

# ---------------------------------------------------------------------------
# Synthesis (cached per case: three tests inspect the same network)
# ---------------------------------------------------------------------------

_NETWORKS = {}

def _network(case):
    """Synthesize the case with the public facility, like a user would."""
    name = case['name']
    if name not in _NETWORKS:
        try:
            units, T_min_app = _build(case)
            HXN = HeatExchangerNetwork('HXN', T_min_app=T_min_app)
            sys = bst.System.from_units('sys', units=[*units, HXN])
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                # biosteam's HeatUtility.load_agent names a new 'oxygen_rich_inlet'
                # stream for every fuel (furnace) utility; the network itself replaces
                # nothing in the registry (test_synthesis_registers_no_intermediate_streams)
                warnings.filterwarnings('ignore', category=RuntimeWarning,
                                        message='.*<Stream: oxygen_rich_inlet> has been replaced in registry')
                sys.simulate()
            time_s = time.perf_counter() - t0
            hus = [hu for hx in HXN.new_HX_utils for hu in hx.heat_utilities]
            total, net = _duties(units)
            _NETWORKS[name] = dict(
                units=units, HXN=HXN, T_min_app=T_min_app, time=time_s,
                table=_hensmith_table(units, T_min_app), total=total, net=net,
                heat=sum(hu.unit_duty for hu in hus if hu.unit_duty > 0),
                cool=-sum(hu.unit_duty for hu in hus if hu.unit_duty < 0),
            )
        except Exception as error:
            _NETWORKS[name] = error
    result = _NETWORKS[name]
    if isinstance(result, Exception):
        raise RuntimeError(f'{name}: synthesis failed: {result!r}') from result
    return result

def _network_problems(net):
    """Everything physically wrong with a synthesized network (see the
    module docstring, test_network_balanced_and_feasible)."""
    HXN, T_min_app = net['HXN'], net['T_min_app']
    problems = []
    if not abs(HXN.energy_balance_percent_error) < EB_TOL:
        problems.append(f'energy balance error {HXN.energy_balance_percent_error:.3g} %')
    if not abs(net['heat'] - net['cool'] - net['net']) <= NET_RTOL * net['total']:
        problems.append(f"heat - cool = {net['heat'] - net['cool']:.10g} "
                        f"!= net duty {net['net']:.10g}")
    others = [u.ID for u in HXN.HXN_sys.units if not isinstance(u, (bst.HXprocess, bst.HXutility))]
    if others: problems.append(f'units other than HXprocess/HXutility (splits?): {others}')
    originals, cycles = HXN.original_heat_exchangers, HXN.stream_life_cycles
    if not (len(originals) == len(cycles) == len(net['units'])
            and set(originals) == set(net['units'])):
        problems.append(f'{len(cycles)} life cycles for {len(originals)} original exchangers; '
                        f"expected one per process stream ({len(net['units'])})")
    def same_state(s, t, duty):
        return (np.allclose(s.mol, t.mol, rtol=1e-12, atol=1e-12)
                and abs(s.P - t.P) <= 1e-9 * t.P and abs(s.H - t.H) <= DUTY_RTOL * duty)
    stream_duty = {}  # (unit, port) -> total duty of the process stream it carries
    stages_on = {}    # (unit, port) -> number of life-cycle stages on it
    for hx, life_cycle in zip(originals, cycles):
        duty = abs(hx.outs[0].H - hx.ins[0].H)
        sign = 1. if hx.outs[0].H >= hx.ins[0].H else -1.
        name = f'stream {life_cycle.index} ({hx.ID})'
        stages = life_cycle.life_cycle
        for stage in stages:
            key = stage.unit, stage.index
            stream_duty[key] = duty
            stages_on[key] = stages_on.get(key, 0) + 1
            # every stage moves the stream towards its outlet: a cold stream
            # is never cooled, a hot stream never heated
            change = sign * (stage.s_out.H - stage.s_in.H)
            if not change >= -DUTY_RTOL * duty:
                problems.append(f'{name}: {"cooled" if sign > 0 else "heated"} by '
                                f'{-change:.10g} kJ/hr in {stage.unit.ID}')
        for a, b in zip(stages, stages[1:]):
            if not (a.s_out is b.s_in or same_state(a.s_out, b.s_in, duty)):
                problems.append(f'{name}: {b.unit.ID} inlet (H={b.s_in.H:.10g}) is not '
                                f'the {a.unit.ID} outlet (H={a.s_out.H:.10g})')
        for label, s, original in (('inlet', stages[0].s_in, hx.ins[0]),
                                   ('outlet', stages[-1].s_out, hx.outs[0])):
            if not same_state(s, original, duty):
                problems.append(f'{name}: network {label} '
                                f'(H={s.H:.10g}, P={s.P:.8g}) != original (H={original.H:.10g}, '
                                f'P={original.P:.8g})')
    for unit, ports in [(u, (0, 1)) for u in HXN.new_HXs] + [(u, (0,)) for u in HXN.new_HX_utils]:
        for port in ports:
            n = stages_on.get((unit, port), 0)
            if n != 1:
                problems.append(f'{unit.ID} port {port}: on {n} life-cycle stages, expected 1')
    for hx in HXN.new_HXs:
        dH = [s_in.H - s_out.H for s_in, s_out in zip(hx.ins, hx.outs)]
        for s_in, s_out in zip(hx.ins, hx.outs):
            if not np.allclose(s_in.mol, s_out.mol, rtol=1e-12, atol=1e-12):
                problems.append(f'{hx.ID}: mass balance')
        scale = (stream_duty.get((hx, 0), 0.) + stream_duty.get((hx, 1), 0.)
                 or sum(map(abs, dH)))
        h = int(np.argmax(dH))
        c = 1 - h
        if not abs(dH[h] + dH[c]) <= DUTY_RTOL * scale:
            problems.append(f'{hx.ID}: heat released {dH[h]:.10g} != '
                            f'heat absorbed {-dH[c]:.10g} kJ/hr')
        if dH[h] <= 1e-12 * scale: continue  # no heat transferred
        approach = _min_approach(
            _temperature(hx.ins[h]), hx.ins[h].H, hx.outs[h].H,
            _temperature(hx.ins[c]), hx.ins[c].H, hx.outs[c].H,
        )
        if not approach >= T_min_app - APPROACH_TOL:
            problems.append(f'{hx.ID}: internal approach {approach:.7f} K < {T_min_app} K')
    return problems

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_corpus_composition():
    names = [case['name'] for case in NO_SPLIT + SPLIT]
    assert len(names) == len(set(names))
    for cases in (NO_SPLIT, SPLIT):
        assert 30 <= len(cases) <= 45
        assert sum(len(case['streams']) > 10 for case in cases) >= 5
    for case in NO_SPLIT: assert 'certificate' in case and 'proof' not in case
    for case in SPLIT: assert 'proof' in case and 'certificate' not in case
    for case in NO_SPLIT + SPLIT:
        assert case['kind'] in ('constant_cp', 'real_thermo')
        IDs = [_unit_ID(s[0]) for s in case['streams']]
        assert len(IDs) == len(set(IDs)), case['name']

@pytest.mark.parametrize('case', NO_SPLIT + SPLIT, ids=_name)
def test_mer_targets(case):
    units, T_min_app = _build(case)
    table = _hensmith_table(units, T_min_app)
    if not _is_cp(case):
        ref = case['reference']
        tol = _reference_tolerance(case, units)
        for label, got, want in (('Q_hot', table.hot_util_load, ref['Q_hot']),
                                 ('Q_cold', table.cold_util_load, ref['Q_cold'])):
            assert abs(got - want) <= tol, (label, got, want, tol)
        if ref['pinch_T_shifted'] is None:  # no hot utility: pinch at the top
            assert table.pinch_T == table.Ts[0]
        else:
            assert abs(table.pinch_T - ref['pinch_T_shifted']) <= PINCH_TOL, table.pinch_T
        return
    Q_hot, Q_cold, zeros, Ts = _cascade(case)
    tol = TARGET_RTOL * _total_duty(case)
    kJ_per_hr = 3600. * Q_UNITS[case['Q_unit']]
    scale, offset = T_UNITS[case['T_unit']]
    got_hot = table.hot_util_load / kJ_per_hr
    got_cold = table.cold_util_load / kJ_per_hr
    pinch = table.pinch_T / scale - offset
    assert abs(got_hot - Q_hot) <= tol, (got_hot, Q_hot)
    assert abs(got_cold - Q_cold) <= tol, (got_cold, Q_cold)
    assert min(abs(pinch - z) for z in zeros) <= PINCH_TOL, (pinch, zeros)
    if Q_hot <= tol: assert abs(pinch - Ts[0]) <= PINCH_TOL, (pinch, Ts[0])
    # the benchmark oracle's stored targets agree with the closed form
    stored = case['targets']
    assert abs(stored['Q_hot'] - Q_hot) <= tol and abs(stored['Q_cold'] - Q_cold) <= tol
    assert stored['pinch_cold'] in zeros
    published = case['published'] or {}
    for key, got in (('Q_hot', got_hot), ('Q_cold', got_cold)):
        if published.get(key):
            value, value_tol = published[key]
            assert abs(got - value) <= value_tol + tol, (key, got, value)
    for key, shift in (('pinch_cold', 0.), ('pinch_hot', case['dTmin'])):
        if published.get(key):
            value, _ = published[key]
            assert min(abs(value - shift - z) for z in zeros) <= PINCH_TOL, (key, value)

@pytest.mark.parametrize('case', NO_SPLIT, ids=_name)
def test_no_split_certificate(case):
    if _is_cp(case):
        problems = _verify_certificate(case)
    else:
        units, T_min_app = _build(case)
        problems, worst = _verify_rt_certificate(case, units, T_min_app)
    assert not problems, '\n'.join(problems)

@pytest.mark.parametrize('case', SPLIT, ids=_name)
def test_split_proof(case):
    proof = case['proof']
    if _is_cp(case):
        # no point loads anywhere: every stream has a finite, positive CP
        assert all(CP > 0 and T_in != T_out for _, _, T_in, T_out, CP in case['streams'])
        proofs = _split_proofs(case)
        assert proofs, 'no pinch-rule violation'
        first = proofs[0]  # the stored proof is the first violation
        assert (first['side'], first['rule']) == (proof['side'], proof['rule']), first
        assert first['hot_at_pinch'] == sorted(proof['hot_at_pinch']), first
        assert first['cold_at_pinch'] == sorted(proof['cold_at_pinch']), first
        assert abs(first['pinch_cold'] - proof['pinch_cold']) <= PINCH_TOL, first
        assert abs(first['pinch_hot'] - proof['pinch_hot']) <= PINCH_TOL, first
    else:
        units, _ = _build(case)
        violations, problems = _rt_split_proof(case, units)
        assert not problems, '\n'.join(problems)
        assert proof['side'] in violations, violations
        hot, cold = violations[proof['side']]
        assert sorted(hot) == sorted(proof['hot_at_pinch'])
        assert sorted(cold) == sorted(proof['cold_at_pinch'])

@pytest.mark.parametrize('case', NO_SPLIT + SPLIT, ids=_name)
def test_network_balanced_and_feasible(case):
    problems = _network_problems(_network(case))
    assert not problems, '\n'.join(problems)

def _utilities(case, net):
    """(label, network utility, hensmith target, reference, reference
    tolerance) for heating and cooling [kJ/hr]."""
    ref_hot, ref_cold, tol_hot, tol_cold = _reference_targets(case, net['units'])
    table = net['table']
    return [('heating', net['heat'], table.hot_util_load, ref_hot, tol_hot),
            ('cooling', net['cool'], table.cold_util_load, ref_cold, tol_cold)]

@pytest.mark.parametrize('case', NO_SPLIT, ids=_name)
def test_no_split_network_reaches_mer(case):
    net = _network(case)
    atol = MER_TOL[case['kind']] * net['total']
    for label, got, target, ref, ref_tol in _utilities(case, net):
        assert abs(got - target) <= atol, (label, got, target)
        assert abs(got - ref) <= atol + ref_tol, (label, got, ref)
    assert net['HXN'].synthesis_info['status'] == 'mer'

@pytest.mark.parametrize('case', SPLIT, ids=_name)
def test_split_network_never_beats_mer(case):
    net = _network(case)
    atol = BEAT_RTOL * net['total']
    for label, got, target, ref, ref_tol in _utilities(case, net):
        assert got >= target * (1. - BEAT_RTOL) - atol, (label, got, target)
        assert got >= ref - ref_tol - atol, (label, got, ref)
    assert net['HXN'].synthesis_info['status'] == 'best_effort'
