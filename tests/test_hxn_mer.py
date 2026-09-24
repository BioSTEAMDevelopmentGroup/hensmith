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
    splitting. Without stream splitting (the default) the network must
    stay feasible and balanced, never beat the targets, and report
    ``'best_effort'``. A strict miss is not asserted: alternating repeated
    matches can approach MER arbitrarily closely. With
    ``stream_splitting=True`` it must reach the targets, with every
    material and energy balance closed and every exchanger feasible (see
    "Networks that split streams").
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

Networks that split streams
---------------------------
A network synthesized with stream splitting is checked on its actual
stream graph (object identity, each stream's sink and port there), and its
bookkeeping against that graph (``_split_network_problems``; every problem
is tagged with its check):

G0
    Only ``HXprocess``, ``HXutility``, ``Splitter`` and ``Mixer`` units, the
    same as the facility's lists of them, with unique IDs of the
    synthesizer's forms (``HX_..``, ``Util_..``, ``Split_..``, ``Mix_..``).
G1
    One life cycle per process stream; each stream's entry port holds a
    feed of the network, and the entries are exactly the network's feeds.
G2
    Walking each stream from its entry: an ``HXprocess`` passes it on at the
    port it entered; the splitters of one split (a tree) send every branch
    to one mixer whose inlets are exactly the branch ends, and the walk goes
    on from the mixer; it ends at the stream's own utility. A mixer outside
    its split, leaving the network, the utility inside a split and a cycle
    are problems.
G3
    Every inlet of every unit is on exactly one walk; the walk is the life
    cycle (its stages, its connections, its splits) and
    ``stream_HXs_dict``; every stage's inlet carries its stage's fraction of
    the flow.
G4
    Splitters balance mass; each branch is one fraction f of the split's
    feed (phase by phase), at least 1e-3, the product of the splitters'
    ratios down to it equals the split's fraction, at the feed's T and P,
    with f times its enthalpy.
G5
    Mixers balance mass, energy and pressure, and their outlet is at
    equilibrium at its enthalpy (the flash-free ``_Temperature``); an
    isothermal re-join has every inlet at the outlet's T and specific
    enthalpy, any other feeds the stream's utility.
G6
    Every stage moves the stream towards its outlet, and no branch passes
    the outlet in specific enthalpy.
G7
    Every ``HXprocess``: mass and heat balances and the exact internal
    approach on its streams' own (branch) material, as in
    ``test_network_balanced_and_feasible``.
G8
    Each stream's exchangers change it by its duty, and its utility takes
    the whole flow.
G9
    Each stream enters and leaves the network in its original states.
G10
    Energy balance error; heat - cool == net process duty.

Without splitting, the network is held to ``_linear_network_problems``
(the checks of ``test_network_balanced_and_feasible``); with it, to G0-G10,
and also to those if it has no splitter.
``test_split_checker_agrees_on_linear_networks``
    Both checks pass on every unsplit corpus network.
``test_split_checker_detects_mutations``
    On a split network wired by hand as the facility wires one
    (``_wired_split_network``), each of fourteen defects (a misnamed or
    unlisted unit, a moved entry, a mixer inlet taken from another stream's
    trunk, swapped mixer inlets, a bypassed branch exchanger, a wrong split
    ratio, a mis-scaled branch, a mixer outlet off its inlets, a branch past
    the stream's outlet, a branch exchanger past its approach, a utility
    short of flow, a shifted outlet, misreported heat) is caught by the
    check made for it.
``test_no_split_plan_unchanged_by_splitting`` / ``test_split_cases_keep_unsplit_sides``
    At the planner (on the facility's round-0 inputs, ``_corpus_knots``),
    splitting leaves an unsplit problem's plan, and a SPLIT problem's sides
    that need no split, bit for bit as they are without it.
``test_split_network_balanced_and_feasible`` / ``test_split_network_reaches_mer``
    Every constant-CP SPLIT problem (``SPLIT_CP``), synthesized with
    ``stream_splitting=True``, passes
    G0-G10 and reaches both hensmith's targets and the independent
    reference within the NO_SPLIT tolerances (``_split_mer_problems``):
    status 'mer', no side left to best effort, every exchanger at its
    planned duty (``deviations``), nothing repaired, dropped or dropped by
    ``Qmin``, every mixer outlet at its planned state, the minimum
    approach kept, and every split side free of leaks, pre-leaks, cells
    below ``Qmin`` and failed strategies.
``test_split_network_structure``
    Every realized split has two or more branches, each with a process
    exchanger and one fraction, the fractions summing to 1; each life cycle
    holds exactly its stream's splits in flow order, with every branch's
    stages; a must re-joins isothermally.
``test_split_exact_dTmin_is_enthalpy_limited``
    A branch exchanger at exactly ``T_min_app`` stops at its enthalpy limit
    (its guard ``dT`` sits 1e-6 K lower), at its planned duty.
``test_no_split_network_with_splitting``
    A NO_SPLIT problem synthesized with ``stream_splitting=True`` is the
    network synthesized without it, bit for bit.
``test_backstop_alone_reaches_mer`` / ``..._in_the_facility``
    With the vertical core alone (no Stage S rule, strategy V only), every
    split side of ``SPLIT_CP``, and of the two constructed problems whose
    roots pre-leak, is planned at MER, with cells that pass
    test_hxn_planner's independent ``_verify_side_cells``; the facility's
    backstop network (smith2005_ex18_4_split) passes the checks above.

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
* Split networks: flows and fractions within 1e-12 of the stream's flow
  (a splitter scales its feed exactly; fractions multiply exactly up to
  rounding); splitter outlets within 1e-9 K of their feed (a copy of its
  state); mixer outlets within 1e-6 K of their equilibrium temperature and,
  where the branches re-join isothermally, of every inlet (the T(H)
  accuracy above, and the synthesizer's own mixer check is 1e-7 K);
  enthalpies within 1e-6 of the stream's duty, as for exchangers.

References
----------
Linnhoff, B. & Flower, J. R. (1978). Synthesis of heat exchanger networks: I.
Systematic generation of energy optimal networks. AIChE J. 24, 633-642.
Linnhoff, B. & Hindmarsh, E. (1983). The pinch design method for heat
exchanger networks. Chem. Eng. Sci. 38, 745-763.
Kemp, I. C. (2007). Pinch Analysis and Process Integration, 2nd ed.
Butterworth-Heinemann.
"""
import contextlib
import math
import re
import time
import warnings
from types import SimpleNamespace
import numpy as np
import pytest
import biosteam as bst
import thermosteam as tmo
from hensmith import HeatExchangerNetwork, _planner, _splitting, hxn_synthesis
from hensmith._heat_exchanger_network import _network_path
from hensmith._planner import plan_network
from hensmith.hxn_synthesis import problem_table
from hxn_mer_cases import NO_SPLIT, SPLIT, Q_UNITS, T_UNITS
from test_hxn_planner import (NEAR_DOUBLE_PINCH, NEAR_THRESHOLD, _verify_side_cells,
                              sides_from_knots)

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
FLOW_RTOL = 1e-12       # split networks: flows and fractions, x the stream's flow
SPLIT_T_TOL = 1e-9      # K: splitter outlets vs the splitter feed
MIX_T_TOL = 1e-6        # K: mixer outlet at equilibrium; isothermal re-joins
MIN_FRACTION = 1e-3     # smallest branch fraction on the corpus

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

@contextlib.contextmanager
def _variant(variant):
    """Patch the planner for one synthesis: None (the default) changes
    nothing; 'V' leaves splitting with the backstop alone (no Stage S rule,
    only the vertical core strategy)."""
    if variant is None:
        yield
        return
    if variant != 'V': raise ValueError(f'unknown variant {variant!r}')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_splitting, '_SPLIT_RULES', ())
        patch.setattr(_splitting, '_CORE_STRATEGIES', ('V',))
        yield

def _network_record(units, HXN, T_min_app, stream_splitting, variant=None,
                    time_s=None):
    """What the checks read of a simulated facility `HXN` on the process
    exchangers `units` (as `_network` returns it)."""
    hus = [hu for hx in HXN.new_HX_utils for hu in hx.heat_utilities]
    total, net = _duties(units)
    return dict(
        units=units, HXN=HXN, T_min_app=T_min_app, time=time_s,
        table=_hensmith_table(units, T_min_app), total=total, net=net,
        heat=sum(hu.unit_duty for hu in hus if hu.unit_duty > 0),
        cool=-sum(hu.unit_duty for hu in hus if hu.unit_duty < 0),
        stream_splitting=stream_splitting, variant=variant,
    )

def _synthesize(case, stream_splitting, variant):
    """Simulate the case's units with the facility (`_network`)."""
    units, T_min_app = _build(case)
    HXN = HeatExchangerNetwork('HXN', T_min_app=T_min_app,
                               stream_splitting=stream_splitting)
    sys = bst.System.from_units('sys', units=[*units, HXN])
    t0 = time.perf_counter()
    with warnings.catch_warnings(), _variant(variant):
        warnings.simplefilter('error', RuntimeWarning)
        # biosteam's HeatUtility.load_agent names a new 'oxygen_rich_inlet'
        # stream for every fuel (furnace) utility; the network itself replaces
        # nothing in the registry (test_synthesis_registers_no_intermediate_streams)
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='.*<Stream: oxygen_rich_inlet> has been replaced in registry')
        sys.simulate()
    return _network_record(units, HXN, T_min_app, stream_splitting, variant,
                           time.perf_counter() - t0)

def _network(case, stream_splitting=False, variant=None, cached=True):
    """Synthesize the case with the public facility, like a user would,
    with or without `stream_splitting` (the planner patched by `_variant`
    during the synthesis only); cached per (name, stream_splitting,
    variant), or a fresh network (never cached) unless `cached`."""
    name = case['name']
    if not cached:
        return _synthesize(case, stream_splitting, variant)
    key = (name, stream_splitting, variant)
    if key not in _NETWORKS:
        try:
            _NETWORKS[key] = _synthesize(case, stream_splitting, variant)
        except Exception as error:
            _NETWORKS[key] = error
    result = _NETWORKS[key]
    if isinstance(result, Exception):
        raise RuntimeError(f'{name}: synthesis failed: {result!r}') from result
    return result

def _facility_heat_utilities(units, T_min_app):
    """The heat utilities of `units` in the order `HeatExchangerNetwork`
    hands them to `synthesize_network`: those of its system's units (the
    order of `System.from_units`, not that of `units`), by duty (a tie keeps
    that order, e.g. smith2005_ex16_1's two 2880 kW coolers)."""
    HXN = HeatExchangerNetwork('HXN', T_min_app=T_min_app)
    system = bst.System.from_units('sys', units=[*units, HXN])
    hus = bst.process_tools.heat_exchanger_utilities_from_units(system.units)
    return sorted([hu for hu in hus if hu.duty], key=lambda hu: hu.duty)

_KNOTS = {}

def _corpus_knots(case):
    """The planner's inputs for the case, as `synthesize_network` plans its
    round 0 in the facility (`_facility_heat_utilities`, `_pinch_analysis`,
    `_grid_knots`): dict(knots, is_hot, T_min_app, time), `time` being the
    seconds taken; cached."""
    name = case['name']
    if name not in _KNOTS:
        t0 = time.perf_counter()
        units, T_min_app = _build(case)
        hus = _facility_heat_utilities(units, T_min_app)
        result = hxn_synthesis._pinch_analysis(hus, T_min_app)
        hxs, hot_indices, curves, grid = result[5], result[6], result[13], result[14]
        _KNOTS[name] = dict(knots=hxn_synthesis._grid_knots(curves, grid),
                            is_hot=[i in hot_indices for i in range(len(hxs))],
                            T_min_app=T_min_app, time=time.perf_counter() - t0)
    return _KNOTS[name]

def _case(name):
    return next(case for case in NO_SPLIT + SPLIT if case['name'] == name)

def _network_problems(net):
    """Everything wrong with a synthesized network: today's checks
    (`_linear_network_problems`) without stream splitting; with it, the
    strict split-aware checks (`_split_network_problems`), plus today's
    checks when the network has no splitter."""
    if not net.get('stream_splitting', False):
        return _linear_network_problems(net)
    problems = _split_network_problems(net)
    if not any(type(u) is bst.Splitter for u in net['HXN'].HXN_sys.units):
        problems += _linear_network_problems(net)
    return problems

def _linear_network_problems(net):
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
# Strict checks of a network that may split streams (G0-G10)
# ---------------------------------------------------------------------------

#: (unit type, ID pattern, the facility's list of them) for every unit a
#: synthesized network may hold
_NETWORK_UNITS = (
    (bst.HXprocess, re.compile(r'^HX_\d+_\d+_(hs|cs)(_\d+)?$'), 'new_HXs'),
    (bst.HXutility, re.compile(r'^Util_\d+_(hs|cs)$'), 'new_HX_utils'),
    (bst.Splitter, re.compile(r'^Split_\d+_(hs|cs)(_\d+)?(_b\d+)?$'), 'new_splitters'),
    (bst.Mixer, re.compile(r'^Mix_\d+_(hs|cs)(_\d+)?$'), 'new_mixers'),
)
#: a splitter ID; group 1 names its split (the ID without the ``_b<c>`` of
#: a chain element)
_SPLITTER_ID = re.compile(r'^(Split_\d+_(?:hs|cs)(?:_\d+)?)(?:_b\d+)?$')

def _is_stream(s):
    return isinstance(s, tmo.Stream)  # not a MissingStream

def _array(x):
    """A (sparse) flow or split vector as a numpy array."""
    return x.to_array() if hasattr(x, 'to_array') else np.asarray(x, dtype=float)

def _phase_flows(s):
    """{phase: molar flows} of a stream, phase by phase for a MultiStream."""
    if isinstance(s, tmo.MultiStream):
        return {phase: _array(s.imol[phase]) for phase in s.phases}
    return {s.phase: _array(s.mol)}

def _is_fraction_of(s, f, feed, atol):
    """Whether every phase of stream `s` carries `f` times that of `feed`."""
    flows, feeds = _phase_flows(s), _phase_flows(feed)
    zero = np.zeros_like(_array(feed.mol))
    return all(np.allclose(flows.get(p, zero), f * feeds.get(p, zero),
                           rtol=FLOW_RTOL, atol=atol) for p in {*flows, *feeds})

def _inlet_port(unit, stream):
    """Index of `stream` (by identity) in ``unit.ins``, or None."""
    for port, s in enumerate(unit.ins):
        if s is stream: return port
    return None

def _walk_stream(lc, network, claims, problem, name):
    """
    Walk the stream of life cycle `lc` on the actual stream graph (object
    identity, ``s.sink`` and the port index there) from its entry port
    (G2). An HXprocess passes it on at the port it entered. A splitter tree
    (the elements of one split, ``Split_<j>_<hs|cs>[_<k>]`` and its
    ``_b<c>`` chain, feeding each other) sends every leaf branch on to one
    mixer, whose inlets must be exactly those branch ends, and the walk
    goes on from the mixer's outlet (a split inside a branch likewise). The
    walk must end at the stream's own utility, ``lc.life_cycle[-1].unit``,
    outside every split. A mixer reached outside its split, a unit outside
    the `network` (a set of units), the utility reached inside a split and
    a cycle within the stream are problems, reported by calling
    ``problem('G2', text)``. Every inlet port passed is counted in `claims`
    (a dict keyed by (unit, port)).

    Returns
    -------
    ports : list[tuple[HXprocess, int]]
        The process exchanger ports passed, in flow order: trunk, then the
        branches of a split, branch by branch.
    edges : list[tuple]
        ``(up, up_port, down, down_port)`` for every stream followed.
    splits : list[tuple]
        ``(splitters, leaves, mixer, branches)`` for every split passed:
        its splitters in the order met, its leaves ``(splitter, port,
        ratio)`` in branch order (`ratio`: the product of the splitters'
        ratios down to that outlet, per chemical), its mixer and each
        branch's process exchanger ports.
    complete : bool
        Whether the walk reached the utility.
    """
    utility = lc.life_cycle[-1].unit
    ports, edges, splits, seen = [], [], [], set()

    def enter(unit, port):
        if (unit, port) in seen:
            problem('G2', f'{name}: a cycle through {unit.ID} inlet {port}')
            return False
        seen.add((unit, port))
        claims[unit, port] = claims.get((unit, port), 0) + 1
        return True

    def follow(unit, port):
        s = unit.outs[port]
        sink = s.sink
        index = None if sink is None else _inlet_port(sink, s)
        if sink not in network or index is None:
            problem('G2', f'{name}: leaves the network at {unit.ID} outlet {port}')
            return None
        edges.append((unit, port, sink, index))
        return sink, index

    def split(root, depth):
        match = _SPLITTER_ID.match(root.ID)
        splitters, leaves = [root], []
        def expand(splitter, ratio):
            r = _array(splitter.split)
            for port, share in ((0, r), (1, 1. - r)):
                sink = splitter.outs[port].sink
                other = (_SPLITTER_ID.match(sink.ID) if type(sink) is bst.Splitter
                         and sink in network else None)
                if match and other and other.group(1) == match.group(1):
                    nxt = follow(splitter, port)
                    if nxt is None or not enter(*nxt): return False
                    splitters.append(sink)
                    if not expand(sink, ratio * share): return False
                else:
                    leaves.append((splitter, port, ratio * share))
            return True
        if not expand(root, 1.): return None
        ends, branches = [], []
        for splitter, port, _ in leaves:
            first = len(ports)
            nxt = follow(splitter, port)
            end = None if nxt is None else segment(*nxt, depth + 1)
            if end is None: return None
            ends.append(end)
            branches.append(ports[first:])
        mixer = ends[0][0]
        if (any(unit is not mixer for unit, _ in ends)
                or sorted(port for _, port in ends) != list(range(len(mixer.ins)))):
            problem('G2', f'{name}: the branches of {root.ID} end at '
                          f'{[(unit.ID, port) for unit, port in ends]}, not at '
                          f'every inlet of one mixer')
            return None
        splits.append((splitters, leaves, mixer, branches))
        return follow(mixer, 0)

    def segment(unit, port, depth):
        # from inlet (unit, port) to the utility (depth 0) or to a mixer
        # inlet (depth > 0), returned as (unit, port)
        while True:
            if not enter(unit, port): return None
            kind = type(unit)
            if kind is bst.HXprocess:
                ports.append((unit, port))
                nxt = follow(unit, port)
            elif kind is bst.Splitter:
                nxt = split(unit, depth)
            elif kind is bst.Mixer and depth:
                return unit, port
            elif kind is bst.HXutility and unit is utility and not depth:
                return unit, port
            else:
                where = ('outside a split' if kind is bst.Mixer
                         else 'inside a split' if unit is utility
                         else 'not its utility')
                problem('G2', f'{name}: reaches {unit.ID} inlet {port} ({where})')
                return None
            if nxt is None: return None
            unit, port = nxt

    unit, port = lc.entry
    complete = unit in network and segment(unit, port, 0) is not None
    return ports, edges, splits, complete

def _split_network_problems(net):
    """
    Everything wrong with a synthesized network that may split streams:
    the checks G0-G10 of the module docstring, made on the actual stream
    graph (`_walk_stream`), against which the network's bookkeeping (the
    facility's unit lists, the life cycles with their splits and
    connections, ``stream_HXs_dict``) is checked. Every problem starts with
    its check's tag ('G0 ...').
    """
    HXN, T_min_app = net['HXN'], net['T_min_app']
    system = HXN.HXN_sys
    units = list(system.units)
    network = set(units)
    problems = []
    def problem(tag, text): problems.append(f'{tag} {text}')
    def IDs(items): return [u.ID for u in items]
    # G0 unit types, the facility's lists of them, IDs
    kinds = [kind for kind, _, _ in _NETWORK_UNITS]
    for u in units:
        if type(u) not in kinds: problem('G0', f'{u.ID}: a {type(u).__name__}')
    for kind, pattern, attr in _NETWORK_UNITS:
        listed = list(getattr(HXN, attr))
        present = [u for u in units if type(u) is kind]
        if not (len(set(listed)) == len(listed) and set(listed) == set(present)):
            problem('G0', f'{attr} {IDs(listed)} != the {kind.__name__} units '
                          f'{IDs(present)}')
        for u in present:
            if not pattern.match(u.ID): problem('G0', f'{u.ID}: not a {kind.__name__} ID')
    if len(set(IDs(units))) != len(units):
        problem('G0', f'duplicate IDs in {sorted(IDs(units))}')
    originals, cycles = HXN.original_heat_exchangers, HXN.stream_life_cycles
    if not (len(originals) == len(cycles) == len(net['units'])
            and set(originals) == set(net['units'])):
        problem('G1', f'{len(cycles)} life cycles for {len(originals)} original exchangers; '
                      f"expected one per process stream ({len(net['units'])})")
    def same_state(s, t, duty):
        return (_is_stream(s) and np.allclose(s.mol, t.mol, rtol=1e-12, atol=1e-12)
                and abs(s.P - t.P) <= 1e-9 * t.P and abs(s.H - t.H) <= DUTY_RTOL * duty)
    def specific(s): return s.H / s.F_mol if s.F_mol else math.nan
    claims = {}       # (unit, inlet port) -> number of stream walks through it
    stream_duty = {}  # (HXprocess, port) -> total duty of the stream it carries
    entries = []
    for hx, lc in zip(originals, cycles):
        inlet, outlet = hx.ins[0], hx.outs[0]
        duty = abs(outlet.H - inlet.H)
        sign = 1. if outlet.H >= inlet.H else -1.
        F = inlet.F_mol
        atol = FLOW_RTOL * F
        name = f'stream {lc.index} ({hx.ID})'
        stages = lc.life_cycle
        utility = stages[-1].unit
        for stage in stages: stream_duty[stage.unit, stage.index] = duty
        # G1 the entry is a feed of the network
        unit, port = lc.entry
        entry = unit.ins[port]
        entries.append(entry)
        if unit not in network or entry.source in network:
            problem('G1', f'{name}: its entry, {unit.ID} inlet {port}, is not a feed '
                          f'of the network')
        if type(utility) is not bst.HXutility:
            problem('G2', f'{name}: its life cycle ends at {utility.ID}, not at a utility')
        # G2 the series-parallel walk
        ports, edges, walked, complete = _walk_stream(lc, network, claims, problem, name)
        # G3 the bookkeeping is the walk
        path = ports + [(utility, 0)] if complete else ports
        planned = [(stage.unit, stage.index) for stage in stages]
        if path != planned:
            problem('G3', f'{name}: walked {[(u.ID, p) for u, p in path]} != its life '
                          f'cycle {[(u.ID, p) for u, p in planned]}')
        connections = list(lc.connections())
        if len(set(edges)) != len(edges) or set(edges) != set(connections):
            extra = [(a.ID, i, b.ID, j) for a, i, b, j in set(edges) - set(connections)]
            missing = [(a.ID, i, b.ID, j) for a, i, b, j in set(connections) - set(edges)]
            problem('G3', f'{name}: walked {extra} beyond its life-cycle connections, '
                          f'which also list {missing}')
        for stage in stages:
            s = stage.s_in
            if not (_is_stream(s) and abs(stage.fraction * F - s.F_mol) <= FLOW_RTOL * F):
                problem('G3', f'{name}: {stage.unit.ID} port {stage.index} is a stage with '
                              f'fraction {stage.fraction:.15g} of the flow, its inlet '
                              f'carries {s.F_mol / F if _is_stream(s) else 0.:.15g}')
        HXs = HXN.stream_HXs_dict[lc.index]
        if [id(u) for u in HXs] != [id(u) for u, _ in planned]:
            problem('G3', f'{name}: stream_HXs_dict {IDs(HXs)} != its life cycle')
        own = list(lc.splits)
        for splitters, leaves, mixer, branches in walked:
            root = splitters[0]
            split = next((sp for sp in own if sp.splitters[0] is root), None)
            if split is None:
                problem('G3', f'{name}: the split at {root.ID} is not one of its splits')
            else:
                own.remove(split)
                if not ([id(u) for u in split.splitters] == [id(u) for u in splitters]
                        and split.mixer is mixer
                        and [[id(u) for u in hxs] for hxs in split.branches]
                        == [[id(u) for u, _ in b] for b in branches]):
                    problem('G3', f'{name}: {split!r} is not the split walked from {root.ID}')
            # G4 splitters: a splitter tree is one split
            feed = root.ins[0]
            for u in splitters:
                total = _array(u.outs[0].mol) + _array(u.outs[1].mol)
                if not np.allclose(total, _array(u.ins[0].mol), rtol=FLOW_RTOL, atol=atol):
                    problem('G4', f'{u.ID}: its outlets carry {total.sum():.15g} kmol/hr, '
                                  f'its feed {u.ins[0].F_mol:.15g}')
            for b, (u, port, ratio) in enumerate(leaves):
                s = u.outs[port]
                f = s.F_mol / feed.F_mol if feed.F_mol else math.nan
                label = f'{u.ID} outlet {port} (branch {b} of {name})'
                if not MIN_FRACTION <= f < 1.:
                    problem('G4', f'{label}: fraction {f:.15g}')
                if not _is_fraction_of(s, f, feed, atol):
                    problem('G4', f'{label}: its flows are not {f:.15g} x those of the feed')
                if split is not None and not (
                        b < len(split.fractions)
                        and np.all(np.abs(ratio - split.fractions[b]) <= FLOW_RTOL)):
                    problem('G4', f'{label}: the split ratios give {np.unique(ratio)}, '
                                  f'the split fractions are {split.fractions}')
                if not (abs(s.T - feed.T) <= SPLIT_T_TOL and s.P == feed.P):
                    problem('G4', f'{label}: at {s.T:.12g} K, {s.P:.12g} Pa; the feed at '
                                  f'{feed.T:.12g} K, {feed.P:.12g} Pa')
                if not abs(s.H - f * feed.H) <= DUTY_RTOL * duty:
                    problem('G4', f'{label}: H = {s.H:.10g} != {f:.15g} x the feed H '
                                  f'{feed.H:.10g} kJ/hr')
            # G5 mixers
            out = mixer.outs[0]
            H_in = math.fsum([s.H for s in mixer.ins])
            mol_in = sum([_array(s.mol) for s in mixer.ins])
            if not np.allclose(_array(out.mol), mol_in, rtol=FLOW_RTOL, atol=atol):
                problem('G5', f'{mixer.ID}: mass balance')
            if not abs(out.H - H_in) <= DUTY_RTOL * duty:
                problem('G5', f'{mixer.ID}: outlet H = {out.H:.10g} != that of its inlets, '
                              f'{H_in:.10g} kJ/hr')
            if any(s.P != out.P for s in mixer.ins):
                problem('G5', f'{mixer.ID}: inlets at {[s.P for s in mixer.ins]} Pa, '
                              f'outlet at {out.P} Pa')
            T_H = _temperature(out)(out.H) if out.F_mol else math.nan
            if not abs(out.T - T_H) <= MIX_T_TOL:
                problem('G5', f'{mixer.ID}: outlet at {out.T:.9f} K, not at equilibrium at '
                              f'its enthalpy ({T_H:.9f} K)')
            if split is None: continue
            if split.isothermal:
                h = specific(out)
                for b, s in enumerate(mixer.ins):
                    if not (abs(specific(s) - h) <= DUTY_RTOL * duty / F
                            and abs(s.T - out.T) <= MIX_T_TOL):
                        problem('G5', f'{mixer.ID} inlet {b}: at {s.T:.9f} K, '
                                      f'{specific(s):.10g} kJ/kmol; the isothermal '
                                      f're-join at {out.T:.9f} K, {h:.10g} kJ/kmol')
            elif out.sink is not utility:
                problem('G5', f'{mixer.ID}: a non-isothermal re-join feeding '
                              f'{getattr(out.sink, "ID", None)}, not the utility')
        if own:
            problem('G3', f'{name}: splits never walked: {own}')
        # G6 every stage moves the stream towards its outlet, and no branch
        # passes the outlet (in specific enthalpy)
        h_out = specific(outlet)
        for u, p in ports + ([(utility, 0)] if complete else []):
            s_in, s_out = u.ins[p], u.outs[p]
            change = sign * (s_out.H - s_in.H)
            if not change >= -DUTY_RTOL * duty:
                problem('G6', f'{name}: {"cooled" if sign > 0 else "heated"} by '
                              f'{-change:.10g} kJ/hr in {u.ID}')
            if not sign * (h_out - specific(s_out)) >= -DUTY_RTOL * duty / F:
                problem('G6', f'{name}: passes its outlet in {u.ID} ({specific(s_out):.10g} '
                              f'kJ/kmol; outlet {h_out:.10g} kJ/kmol)')
        # G8 the stream's exchangers change it by its duty, and its utility
        # takes the whole flow
        if complete:
            change = math.fsum([u.outs[p].H - u.ins[p].H for u, p in ports]
                               + [utility.outs[0].H - utility.ins[0].H])
            if not abs(change - (outlet.H - inlet.H)) <= DUTY_RTOL * duty:
                problem('G8', f'{name}: its exchangers change it by {change:.10g} kJ/hr, '
                              f'not {outlet.H - inlet.H:.10g}')
            if not np.allclose(_array(utility.ins[0].mol), _array(inlet.mol),
                               rtol=FLOW_RTOL, atol=atol):
                problem('G8', f'{name}: {utility.ID} takes {utility.ins[0].F_mol:.15g} '
                              f'kmol/hr of {F:.15g}')
        # G9 end states
        for label, s, original in (('inlet', entry, inlet),
                                   ('outlet', utility.outs[0], outlet)):
            if not same_state(s, original, duty):
                problem('G9', f'{name}: network {label} != original (H={original.H:.10g} '
                              f'kJ/hr, P={original.P:.8g} Pa)')
    # G1 the entries are the network's feeds
    feeds = [s for u in units for s in u.ins if s.source not in network]
    entry_IDs = {id(s) for s in entries}
    if not (len(entry_IDs) == len(entries) == len(feeds)
            and entry_IDs == {id(s) for s in feeds} == {id(s) for s in system.feeds}):
        problem('G1', f'entry streams {sorted(map(str, entries))} != the feeds of the '
                      f'network {sorted(map(str, feeds))} (of HXN_sys: '
                      f'{sorted(map(str, system.feeds))})')
    # G3 every inlet of every unit is on exactly one stream walk
    for u in units:
        for port in range(len(u.ins)):
            n = claims.get((u, port), 0)
            if n != 1: problem('G3', f'{u.ID} inlet {port}: on {n} stream walks, expected 1')
    # G7 every process exchanger: mass, heat and the exact internal approach
    # on its streams' own material (C11-C13)
    for hx in [u for u in units if type(u) is bst.HXprocess]:
        if not all(map(_is_stream, [*hx.ins, *hx.outs])):
            problem('G7', f'{hx.ID}: a port without a stream')
            continue
        dH = [s_in.H - s_out.H for s_in, s_out in zip(hx.ins, hx.outs)]
        for s_in, s_out in zip(hx.ins, hx.outs):
            if not np.allclose(s_in.mol, s_out.mol, rtol=1e-12, atol=1e-12):
                problem('G7', f'{hx.ID}: mass balance')
        scale = (stream_duty.get((hx, 0), 0.) + stream_duty.get((hx, 1), 0.)
                 or sum(map(abs, dH)))
        h = int(np.argmax(dH))
        c = 1 - h
        if not abs(dH[h] + dH[c]) <= DUTY_RTOL * scale:
            problem('G7', f'{hx.ID}: heat released {dH[h]:.10g} != heat absorbed '
                          f'{-dH[c]:.10g} kJ/hr')
        if dH[h] <= 1e-12 * scale: continue  # no heat transferred
        if not (hx.ins[h].F_mol and hx.ins[c].F_mol):
            problem('G7', f'{hx.ID}: an empty inlet')
            continue
        approach = _min_approach(
            _temperature(hx.ins[h]), hx.ins[h].H, hx.outs[h].H,
            _temperature(hx.ins[c]), hx.ins[c].H, hx.outs[c].H,
        )
        if not approach >= T_min_app - APPROACH_TOL:
            problem('G7', f'{hx.ID}: internal approach {approach:.7f} K < {T_min_app} K')
    # G10 energy balance error; heat - cool == net duty (C1, C2)
    if not abs(HXN.energy_balance_percent_error) < EB_TOL:
        problem('G10', f'energy balance error {HXN.energy_balance_percent_error:.3g} %')
    if not abs(net['heat'] - net['cool'] - net['net']) <= NET_RTOL * net['total']:
        problem('G10', f"heat - cool = {net['heat'] - net['cool']:.10g} "
                       f"!= net duty {net['net']:.10g}")
    return problems

# ---------------------------------------------------------------------------
# A split network wired by hand, as the facility wires one
# ---------------------------------------------------------------------------

def _converge(system):
    """Converge a network's system and design and cost its units, as
    ``HeatExchangerNetwork._cost`` does (without its fallback: a failure
    raises)."""
    system._setup()
    system.converge()
    for unit in system.units: unit._summary()

_WIRED = {}

def _wired_split_network(case, cached=False):
    """
    The case's network with stream splitting, wired by hand the way
    `HeatExchangerNetwork` wires its own (the stream-splitting design,
    5.2): `synthesize_network` with `stream_splitting`, one
    `StreamLifeCycle` per stream given the splits, a copy of the stream's
    real inlet in its entry port, every connection of every life cycle,
    and one converged `System`. Returned as `_network` returns a network,
    its 'HXN' a namespace with the facility's attributes; a fresh network
    unless `cached`.
    """
    name = case['name']
    if cached and name in _WIRED: return _WIRED[name]
    units, T_min_app = _build(case)
    hus = _facility_heat_utilities(units, T_min_app)
    flowsheet = bst.Flowsheet('mer_wired_HXN')
    for registry in flowsheet.registries: registry.clear()
    info = {}
    with flowsheet.temporary(), bst.IgnoreDockingWarnings():
        (hot_side, cold_side, utils, hxs, _, T_out, _, _, _, _, stream_HXs, _,
         cold_indices) = hxn_synthesis.synthesize_network(
            hus, T_min_app, info=info, stream_splitting=True)
        new_HXs = hot_side + cold_side
        cycles = []
        for i in range(len(hxs)):
            lc = hxn_synthesis.StreamLifeCycle(i, i in cold_indices)
            lc.get_life_cycle(new_HXs, utils, splits=info['splits'])
            cycles.append(lc)
        for i, lc in enumerate(cycles):
            unit, port = lc.entry
            inlet = unit.ins[port]
            inlet.copy_like(hxs[i].ins[0])
            if isinstance(unit, (bst.HXprocess, bst.Splitter)):
                hxn_synthesis._first_inlet(inlet, i in info['point_loads'], T_out[i],
                                           i not in cold_indices)
        for lc in cycles:
            for up, up_port, down, down_port in lc.connections():
                down.ins[down_port] = up.outs[up_port]
        splitters = [u for split in info['splits'] for u in split.splitters]
        mixers = [split.mixer for split in info['splits']]
        path, recycles = _network_path(new_HXs + utils + splitters + mixers, cycles)
        system = bst.System('mer_wired', path, recycle=recycles or None)
        system.set_tolerance(method='fixedpoint', subsystems=True, mol=1e-9, rmol=1e-12,
                             T=1e-8, rT=1e-12, maxiter=200)
        _converge(system)
    new_hus = [hu for hx in utils for hu in hx.heat_utilities]
    def load(hus): return sum([abs(hu.duty * hu.agent.heat_transfer_efficiency)
                               for hu in bst.HeatUtility.sum_by_agent(hus)])
    Q_bal = (2. * sum([abs(hx.Q) for hx in new_HXs]) + load(new_hus)) / load(hus)
    HXN = SimpleNamespace(
        HXN_sys=system, new_HXs=new_HXs, new_HX_utils=utils, new_splitters=splitters,
        new_mixers=mixers, original_heat_exchangers=hxs, stream_life_cycles=cycles,
        stream_HXs_dict=stream_HXs, synthesis_info=info,
        energy_balance_percent_error=100. * (Q_bal - 1.),
    )
    total, net = _duties(units)
    result = dict(
        units=units, HXN=HXN, T_min_app=T_min_app, total=total, net=net,
        heat=sum(hu.unit_duty for hu in new_hus if hu.unit_duty > 0),
        cool=-sum(hu.unit_duty for hu in new_hus if hu.unit_duty < 0),
        stream_splitting=True, variant=None,
    )
    if cached: _WIRED[name] = result
    return result

# ---------------------------------------------------------------------------
# Defects of a split network, one per check (each returns the check's tag)
# ---------------------------------------------------------------------------

MUTATION_CASE = 'smith2005_ex18_2_split'

def _split_stage(net, b=0):
    """The network's first split, the index of its stream's life cycle, and
    the life-cycle stage of the first exchanger of its branch `b`."""
    HXN = net['HXN']
    split = HXN.synthesis_info['splits'][0]
    n = next(n for n, lc in enumerate(HXN.stream_life_cycles) if lc.index == split.stream)
    stage = next(s for s in HXN.stream_life_cycles[n].life_cycle
                 if s.unit is split.branches[b][0])
    return split, n, stage

def _rewire(unit, port, stream):
    with bst.IgnoreDockingWarnings(): unit.ins[port] = stream

def _misnamed_mixer(net):
    split, *_ = _split_stage(net)
    split.mixer.ID = 'Mixer_' + split.mixer.ID
    return 'G0'

def _unlisted_splitter(net):
    net['HXN'].new_splitters.remove(_split_stage(net)[0].splitters[0])
    return 'G0'

def _entry_moved(net):
    # the stream's entry moved into its first branch
    split, n, stage = _split_stage(net)
    net['HXN'].stream_life_cycles[n].entry = hxn_synthesis._Port(stage.unit, stage.index)
    return 'G1'

def _mixer_inlet_from_a_trunk(net):
    # a mixer inlet re-pointed to another stream's trunk
    split, *_ = _split_stage(net)
    other = next(lc for lc in net['HXN'].stream_life_cycles
                 if lc.index != split.stream and not lc.splits and len(lc.life_cycle) > 1)
    _rewire(split.mixer, 1, other.life_cycle[0].s_out)
    return 'G2'

def _mixer_inlets_swapped(net):
    # the branches re-join at each other's mixer inlets
    mixer = _split_stage(net)[0].mixer
    with bst.IgnoreDockingWarnings(): mixer.ins[:] = [mixer.ins[1], mixer.ins[0], *mixer.ins[2:]]
    return 'G3'

def _branch_exchanger_bypassed(net):
    # the exchanger's inlet stream wired to its outlet's sink
    stage = _split_stage(net)[2]
    s_out = stage.s_out
    _rewire(s_out.sink, _inlet_port(s_out.sink, s_out), stage.s_in)
    return 'G3'

def _wrong_split_ratio(net):
    split, *_ = _split_stage(net)
    split.splitters[0].split = 0.9 * split.fractions[0]
    _converge(net['HXN'].HXN_sys)
    return 'G4'

def _branch_inlet_scaled(net):
    # a branch inlet carries the wrong fraction
    _split_stage(net)[2].s_in.scale(1.1)
    return 'G4'

def _mixer_outlet_shifted(net):
    mixer = _split_stage(net)[0].mixer
    mixer.outs[0].T += 0.01
    return 'G5'

def _branch_passes_the_outlet(net):
    split, n, stage = _split_stage(net)
    outlet = net['HXN'].original_heat_exchangers[n].outs[0]
    sign = 1. if net['HXN'].stream_life_cycles[n].cold else -1.
    stage.s_out.T = outlet.T + sign
    return 'G6'

def _branch_H_lim_shifted(net):
    # a branch exchanger's enthalpy limits moved on by 5 % of its duty,
    # past its approach (its dT guard lowered too: at T_min_app - 1e-6 K it
    # would cap the duty at the terminals). Branch 1: on MUTATION_CASE its
    # partner comes from outside the split stream's loop (branch 0's
    # partner is heated by the split stream below the pinch, and the
    # converged loop lowers its inlet, keeping the approach)
    split, n, stage = _split_stage(net, 1)
    hx = stage.unit
    hx.dT = 0.5 * net['T_min_app']
    for k in (0, 1):
        H_lim = getattr(hx, f'H_lim{k}')
        if H_lim is not None:
            setattr(hx, f'H_lim{k}', H_lim + 0.05 * (H_lim - hx.ins[k].H))
    _converge(net['HXN'].HXN_sys)
    return 'G7'

def _utility_inlet_short(net):
    split, n, stage = _split_stage(net)
    net['HXN'].stream_life_cycles[n].life_cycle[-1].s_in.scale(1. - 1e-9)
    return 'G8'

def _utility_outlet_shifted(net):
    split, n, stage = _split_stage(net)
    net['HXN'].stream_life_cycles[n].life_cycle[-1].s_out.T += 0.01
    return 'G9'

def _heat_misreported(net):
    net['heat'] += 1e-6 * net['total']
    return 'G10'

SPLIT_MUTATIONS = {f.__name__.strip('_'): f for f in (
    _misnamed_mixer, _unlisted_splitter, _entry_moved, _mixer_inlet_from_a_trunk,
    _mixer_inlets_swapped, _branch_exchanger_bypassed, _wrong_split_ratio,
    _branch_inlet_scaled, _mixer_outlet_shifted, _branch_passes_the_outlet,
    _branch_H_lim_shifted, _utility_inlet_short, _utility_outlet_shifted,
    _heat_misreported,
)}

# ---------------------------------------------------------------------------
# Plans (bitwise comparison)
# ---------------------------------------------------------------------------

def _same(a, b):
    """Whether `a` and `b` are equal, floats bit for bit (nested dicts,
    lists, tuples and arrays)."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(_same(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(map(_same, a, b)))
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and a.tobytes() == b.tobytes())
    if isinstance(a, (float, np.floating)):
        return (isinstance(b, (float, np.floating))
                and float(a).hex() == float(b).hex())
    return type(a) is type(b) and a == b

def _plan_exchangers(plan):
    return [(e.side, e.hot, e.cold, e.Q, e.H_hot_in, e.H_hot_out, e.H_cold_in,
             e.H_cold_out, e.pair_index, e.hot_seq, e.cold_seq, e.hot_frac,
             e.cold_frac, e.hot_branch, e.cold_branch) for e in plan.exchangers]

def _side_matches(plan, side):
    """{stream: [(side, hot, cold, Q) of its exchangers on `side`, in flow
    order]}."""
    return {j: [(e.side, e.hot, e.cold, e.Q) for e in (plan.exchangers[n] for n in stages)
                if e.side == side] for j, stages in plan.stages.items()}

def _plan_without_best_effort(knots, is_hot, T_min_app):
    """`plan_network` without stream splitting, except that a side it
    cannot serve at MER is left unplanned instead of planned for best
    effort (`_planner._best_effort` is patched to return every must's
    whole duty as a gap). Without `avoid_recycle`, `plan_network` plans
    each side on its own (`_planner._plan_sides`), and only a side whose
    status is then 'best_effort' reaches `_best_effort`, so every 'mer'
    and 'trivial' side is exactly as in the real plan; the best-effort
    searches of the SPLIT problems (94 s in all) are skipped."""
    def unplanned(side, proof, cap1, forbid, work_scale, work):
        return _planner._SidePlan([], list(side.Qm), 'best_effort', 'unplanned',
                                  work, proof)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_planner, '_best_effort', unplanned)
        return plan_network(knots, is_hot, T_min_app)

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

# ---------------------------------------------------------------------------
# Stream splitting: the strict checker and the planner at the corpus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', NO_SPLIT + SPLIT, ids=_name)
def test_split_checker_agrees_on_linear_networks(case):
    # on the unsplit networks, the strict split-aware checks (G0-G10) and
    # today's checks agree: neither finds a problem
    net = _network(case)
    problems = _split_network_problems(net)
    assert not problems, '\n'.join(problems)
    problems = _linear_network_problems(net)
    assert not problems, '\n'.join(problems)

@pytest.mark.parametrize('source', ['wired', 'facility'])
@pytest.mark.parametrize('mutation', list(SPLIT_MUTATIONS))
def test_split_checker_detects_mutations(mutation, source):
    # a split network, wired by hand (as the facility wires it) or by the
    # facility itself, passes the strict checks; each defect, applied to a
    # fresh copy, is caught by the check made for it (and possibly others)
    case = _case(MUTATION_CASE)
    if source == 'wired':
        net = _wired_split_network(case, cached=True)
    else:
        net = _network(case, True)
    problems = _network_problems(net)
    assert not problems, '\n'.join(problems)
    assert net['HXN'].synthesis_info['splits']
    if source == 'wired':
        net = _wired_split_network(case)
    else:
        net = _network(case, True, cached=False)
    tag = SPLIT_MUTATIONS[mutation](net)
    problems = _network_problems(net)
    assert any(p.startswith(tag + ' ') for p in problems), (tag, problems)

@pytest.mark.parametrize('case', NO_SPLIT, ids=_name)
def test_no_split_plan_unchanged_by_splitting(case):
    # a problem that needs no split plans bit for bit as without splitting
    knots = _corpus_knots(case)
    args = knots['knots'], knots['is_hot'], knots['T_min_app']
    off = plan_network(*args)
    on = plan_network(*args, stream_splitting=True)
    assert on.status == off.status == 'mer'
    assert on.splits == []
    assert all(side['split'] is None for side in on.info['sides'].values())
    info = dict(on.info, sides={name: {k: v for k, v in side.items() if k != 'split'}
                                for name, side in on.info['sides'].items()})
    assert _same(info, off.info)
    assert _same(_plan_exchangers(on), _plan_exchangers(off))
    for attr in ('Q_hot_target', 'Q_cold_target', 'pinch_T', 'cut', 'stages',
                 'utility', 'Q_hot', 'Q_cold', 'penalty', 'paths'):
        assert _same(getattr(on, attr), getattr(off, attr)), attr

#: SPLIT problems with a side that an unsplit plan serves
UNSPLIT_SIDES = {'smith2005_ex18_2_split': 'below', 'rtB05_above_2h1c': 'below',
                 'luo_20sp_ph10c10': 'above'}

@pytest.mark.parametrize('case', SPLIT, ids=_name)
def test_split_cases_keep_unsplit_sides(case):
    # the side of a SPLIT problem that an unsplit plan serves (or that is
    # trivial) is planned exactly as without splitting: the same matches in
    # the same order along every stream, and the same side info. Enthalpies
    # are not compared: a stream crossing into the split side enters the
    # unsplit side at another enthalpy once the split side reaches MER.
    knots = _corpus_knots(case)
    args = knots['knots'], knots['is_hot'], knots['T_min_app']
    on = plan_network(*args, stream_splitting=True)
    assert on.status == 'mer'
    off = _plan_without_best_effort(*args)
    kept = [name for name, side in off.info['sides'].items()
            if side['status'] in ('mer', 'trivial')]
    assert all(side['method'] == 'unplanned' for name, side in off.info['sides'].items()
               if name not in kept)
    if case['name'] in UNSPLIT_SIDES: assert UNSPLIT_SIDES[case['name']] in kept
    for name in kept:
        side_on, side_off = on.info['sides'][name], off.info['sides'][name]
        assert side_on['split'] is None, name
        assert _same({k: side_on[k] for k in side_off}, side_off), name
        assert _same(_side_matches(on, name), _side_matches(off, name)), name

def test_network_variant_is_patched_for_the_synthesis_only():
    # the backstop variant ('V') patches the splitting portfolio inside its
    # context only, so a cached variant network never leaks the patch
    rules, strategies = _splitting._SPLIT_RULES, _splitting._CORE_STRATEGIES
    with _variant('V'):
        assert _splitting._SPLIT_RULES == ()
        assert _splitting._CORE_STRATEGIES == ('V',)
    assert _splitting._SPLIT_RULES is rules
    assert _splitting._CORE_STRATEGIES is strategies
    with _variant(None):
        assert _splitting._SPLIT_RULES is rules
    with pytest.raises(ValueError):
        with _variant('LV'): pass

# ---------------------------------------------------------------------------
# Stream splitting: the facility at the corpus
# ---------------------------------------------------------------------------

#: the constant-CP SPLIT problems
SPLIT_CP = [case for case in SPLIT if _is_cp(case)]

#: NO_SPLIT problems synthesized end to end with stream splitting (at the
#: planner, test_no_split_plan_unchanged_by_splitting covers them all)
NO_SPLIT_WITH_SPLITTING = ['4sp1_lee1970_dt10F', 'linnhoff_4stream']

#: problems (dT, rows) of test_hxn_planner whose roots the planner's own
#: cascade tolerances leave slightly negative, so that splitting starts
#: from the pre-leaked root: a threshold deficit of 23.7 tolQ, and
#: nptel_t5_3 with CP2 = 6 - 1.58e-11 (near-equal minima, 0.05 tolQ)
PRELEAK_PROBLEMS = {'near_threshold': NEAR_THRESHOLD,
                    'near_double_pinch': NEAR_DOUBLE_PINCH}

def _numeric_knots(dT, rows):
    """Planner inputs (knots, is_hot, T_min_app) of a constant-CP problem
    given as test_hxn_planner rows, as `_planner._plan_numeric` builds
    them."""
    knots, is_hot = [], []
    for _, kind, T_in, T_out, CP in rows:
        a, b = float(min(T_in, T_out)), float(max(T_in, T_out))
        T = np.array([a, b])
        knots.append((T, float(CP) * (T - a)))
        is_hot.append(kind == 'h')
    return knots, is_hot, float(dT)

def _curved(case):
    """{(stream, side): whether the planner's curve of the stream on that
    side of the pinch has a kink (more than two knots), on the facility's
    round-0 knots}."""
    knots = _corpus_knots(case)
    sides = sides_from_knots(knots['knots'], knots['is_hot'], knots['T_min_app'])
    return {(c.stream, name): c.n > 2 for name, side in sides.items()
            for c in side.musts + side.flexes}

def _split_mer_problems(case, net, candidate=None):
    """Everything that keeps the split network `net` of `case` from being
    the MER network the synthesizer promises (test_split_network_reaches_mer);
    `candidate`, if given, is the name every split side must have picked."""
    HXN, T_min_app = net['HXN'], net['T_min_app']
    info = HXN.synthesis_info
    problems = []
    if info['status'] != 'mer': problems.append(f"status {info['status']!r}")
    atol = MER_TOL[case['kind']] * net['total']
    for label, got, target, ref, ref_tol in _utilities(case, net):
        if not abs(got - target) <= atol:
            problems.append(f'{label} {got!r} != target {target!r}')
        if not abs(got - ref) <= atol + ref_tol:
            problems.append(f'{label} {got!r} != reference {ref!r}')
    if not (HXN.new_splitters and HXN.new_mixers and info['splits']):
        problems.append('no split')
    for key in ('repaired', 'dropped', 'qmin_dropped', 'split_deviations', 'deviations'):
        if info[key] != []: problems.append(f'{key}: {info[key]}')
    if not info['min_approach'] >= T_min_app - APPROACH_TOL:
        problems.append(f"min_approach {info['min_approach']!r} K")
    split_sides = 0
    for name, side in info['sides'].items():
        if side['status'] == 'best_effort': problems.append(f'{name}: best effort')
        split = side['split']
        if split is None: continue
        split_sides += 1
        if split['candidate'] is None or side['method'] != f"split-{split['candidate']}":
            problems.append(f"{name}: method {side['method']!r}, candidate "
                            f"{split['candidate']!r}")
        if candidate is not None and split['candidate'] != candidate:
            problems.append(f"{name}: picked {split['candidate']!r}")
        for key, empty in (('leak', 0.), ('preleak', 0.), ('small', []), ('errors', [])):
            if split[key] != empty: problems.append(f'{name}: {key} {split[key]!r}')
    if not split_sides: problems.append('no side split')
    curved = _curved(case)
    mixers = {}
    for split in info['splits']:
        key = split.stream, split.side
        mixers[key] = mixers.get(key, 0) + 1
    for key, n in mixers.items():
        if curved[key] and n > _splitting._SPLIT_MIX_CAP:
            problems.append(f'stream {key[0]} {key[1]}: {n} mixers on a curved stream')
    return problems

@pytest.mark.parametrize('case', SPLIT_CP, ids=_name)
def test_split_network_balanced_and_feasible(case):
    # the strict split-aware checks (G0-G10) on the network synthesized
    # with stream splitting
    problems = _network_problems(_network(case, True))
    assert not problems, '\n'.join(problems)

@pytest.mark.parametrize('case', SPLIT_CP, ids=_name)
def test_split_network_reaches_mer(case):
    # with stream splitting, a problem whose MER needs splits reaches both
    # hensmith's targets and the independent reference, with every
    # exchanger at its planned duty (no deviation), nothing repaired or
    # dropped, and splits free of leaks and small cells
    problems = _split_mer_problems(case, _network(case, True))
    assert not problems, '\n'.join(problems)

@pytest.mark.parametrize('case', SPLIT_CP, ids=_name)
def test_split_network_structure(case):
    # every realized split: at least two branches with one fraction each,
    # summing to 1, each with a process exchanger; the life cycles hold
    # exactly the stream's splits, in flow order, with the branches' stages;
    # a must (a hot stream above the pinch, a cold one below) re-joins at
    # one temperature
    HXN = _network(case, True)['HXN']
    splits = HXN.synthesis_info['splits']
    assert splits
    cycles = {lc.index: lc for lc in HXN.stream_life_cycles}
    new_HXs = {id(hx) for hx in HXN.new_HXs}
    for split in splits:
        assert len(split.fractions) == len(split.branches) >= 2
        assert abs(math.fsum(split.fractions) - 1.) <= 1e-12
        assert all(split.branches), split
        assert all(type(hx) is bst.HXprocess and id(hx) in new_HXs
                   for branch in split.branches for hx in branch), split
        assert (any(split.mixer is u for u in HXN.new_mixers)
                and all(any(s is u for u in HXN.new_splitters) for s in split.splitters))
        must = (split.side == 'above') != cycles[split.stream].cold
        if must: assert split.isothermal, split
    for lc in HXN.stream_life_cycles:
        own = [split for split in splits if split.stream == lc.index]
        first = 'below' if lc.cold else 'above'
        own.sort(key=lambda split: (split.side != first, split.index))
        assert [id(split) for split in lc.splits] == [id(split) for split in own]
        for k, split in enumerate(lc.splits):
            for b, branch in enumerate(split.branches):
                stages = [stage for stage in lc.life_cycle if stage.branch == (k, b)]
                assert [id(stage.unit) for stage in stages] == [id(hx) for hx in branch]
                assert all(stage.fraction == split.fractions[b] for stage in stages)
        assert ([id(u) for u in HXN.stream_HXs_dict[lc.index]]
                == [id(stage.unit) for stage in lc.life_cycle])

def _planned_duties(case):
    """{exchanger ID: planned duty [kJ/hr]} of the facility's round-0 plan
    of `case` with stream splitting (its IDs as `hxn_synthesis._realize`
    names the exchangers)."""
    knots = _corpus_knots(case)
    plan = plan_network(knots['knots'], knots['is_hot'], knots['T_min_app'],
                        Qmin=1e-3, stream_splitting=True)
    duties = {}
    for e in plan.exchangers:
        suffix = '' if e.pair_index == 1 else f'_{e.pair_index}'
        ID = (f'HX_{e.cold}_{e.hot}_hs{suffix}' if e.side == 'above'
              else f'HX_{e.hot}_{e.cold}_cs{suffix}')
        duties[ID] = e.Q
    return duties

@pytest.mark.parametrize('case', [_case('smith2005_ex18_4_split')], ids=_name)
def test_split_exact_dTmin_is_enthalpy_limited(case):
    # a branch exchanger at exactly the minimum approach (at the pinch) runs
    # with its guard dT = T_min_app - 1e-6 K, so its duty ends where an
    # enthalpy limit (f times the parent's) binds: the planned duty
    net = _network(case, True)
    HXN, T_min_app = net['HXN'], net['T_min_app']
    info = HXN.synthesis_info
    assert info['refine_rounds'] == 0  # round 0 is the realized plan
    planned = _planned_duties(case)
    duty = {lc.index: abs(hx.outs[0].H - hx.ins[0].H)
            for hx, lc in zip(HXN.original_heat_exchangers, HXN.stream_life_cycles)}
    at_pinch = 0
    for split in info['splits']:
        for hx in (hx for branch in split.branches for hx in branch):
            assert hx.dT == T_min_app - 1e-6
            streams = [int(n) for n in hx.ID.split('_')[1:3]]  # at ports 0, 1
            limited = [k for k, H_lim in enumerate((hx.H_lim0, hx.H_lim1))
                       if H_lim is not None
                       and abs(hx.outs[k].H - H_lim) <= DUTY_RTOL * duty[streams[k]]]
            assert limited, hx.ID
            scale = duty[streams[0]] + duty[streams[1]]
            assert abs(hx.Q - planned[hx.ID]) <= hxn_synthesis._DUTY_TOL * scale, hx.ID
            h = 0 if hx.ins[0].T > hx.ins[1].T else 1
            c = 1 - h
            terminal = min(hx.ins[h].T - hx.outs[c].T, hx.outs[h].T - hx.ins[c].T)
            at_pinch += abs(terminal - T_min_app) <= 1e-9
    assert at_pinch

@pytest.mark.parametrize('name', NO_SPLIT_WITH_SPLITTING)
def test_no_split_network_with_splitting(name):
    # a problem that needs no split gives, end to end with stream splitting,
    # exactly the network it gives without (bit for bit)
    case = _case(name)
    on, off = _network(case, True), _network(case)
    info_on, info_off = on['HXN'].synthesis_info, off['HXN'].synthesis_info
    assert info_on['status'] == info_off['status'] == 'mer'
    assert info_on['splits'] == on['HXN'].new_splitters == on['HXN'].new_mixers == []
    assert not any(type(u) in (bst.Splitter, bst.Mixer) for u in on['HXN'].HXN_sys.units)
    assert ([(hx.ID, float(hx.Q).hex()) for hx in on['HXN'].new_HXs]
            == [(hx.ID, float(hx.Q).hex()) for hx in off['HXN'].new_HXs])
    for key in ('refine_rounds', 'repaired'):
        assert info_on[key] == info_off[key], key
    problems = _network_problems(on)
    assert not problems, '\n'.join(problems)

@pytest.mark.parametrize('name', [case['name'] for case in SPLIT_CP]
                                 + list(PRELEAK_PROBLEMS))
def test_backstop_alone_reaches_mer(name, monkeypatch):
    # the vertical core alone (no Stage S rule, strategy V only) plans every
    # split side at MER from the pre-leaked root, with cells that pass the
    # independent check (test_hxn_planner); only the constructed problems
    # pre-leak
    if name in PRELEAK_PROBLEMS:
        knots, is_hot, T_min_app = _numeric_knots(*PRELEAK_PROBLEMS[name])
    else:
        k = _corpus_knots(_case(name))
        knots, is_hot, T_min_app = k['knots'], k['is_hot'], k['T_min_app']
    monkeypatch.setattr(_splitting, '_SPLIT_RULES', ())
    monkeypatch.setattr(_splitting, '_CORE_STRATEGIES', ('V',))
    planned = {}
    plan_side = _planner._plan_side
    def spy(side, *args, **kwargs):
        result = plan_side(side, *args, **kwargs)
        planned[side.name] = side, result
        return result
    monkeypatch.setattr(_planner, '_plan_side', spy)
    plan = plan_network(knots, is_hot, T_min_app, stream_splitting=True)
    assert plan.status == 'mer'
    splits = {n: side['split'] for n, side in plan.info['sides'].items()
              if side['split'] is not None}
    assert splits and all(split['candidate'] == 'V' for split in splits.values())
    assert all(split['leak'] == 0. for split in splits.values())
    preleak = math.fsum(split['preleak'] for split in splits.values())
    assert (preleak > 0.) == (name in PRELEAK_PROBLEMS)
    assert plan.penalty <= preleak + 1e-12 * plan.info['scale']
    for n in splits:
        side, result = planned[n]
        assert result.status == 'mer'
        _verify_side_cells(side, result.cells, _splitting._preleak_root(side)[1])

@pytest.mark.parametrize('case', [_case('smith2005_ex18_4_split')], ids=_name)
def test_backstop_alone_reaches_mer_in_the_facility(case):
    # the facility with the vertical core alone: MER, and a strictly
    # balanced and feasible network
    net = _network(case, True, variant='V')
    problems = _network_problems(net) + _split_mer_problems(case, net, candidate='V')
    assert not problems, '\n'.join(problems)
