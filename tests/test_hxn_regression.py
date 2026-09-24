# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Regression tests for heat exchanger network synthesis on synthetic systems.

Ten synthetic systems of increasing complexity (all with phase-changing
streams from case 3 on). For each, the synthesized network must

(i)   close its energy balance (|error| < 1e-6 %) without RuntimeWarnings,
(ii)  never beat the minimum-energy-requirement (MER) targets of the problem
      table computed on the same streams, and report status 'mer' exactly
      when it reaches them,
(iii) keep the minimum approach temperature INSIDE every process exchanger,
      on the exact states of its streams (``T_min_app - 1e-6`` K, the
      synthesizer's guarantee; the check of ``test_hxn_mer``, which
      evaluates temperatures without the flashes of the code under test),
(iv)  plan on the problem table's own cascade (the planner's targets, pinch
      and pinch cut equal the table's), and
(v)   recover at least as much heat as documented in ``CASES`` below, so that
      no future change to ``hensmith`` silently makes the synthesizer perform worse.

The documented utility loads were recorded by running this file directly
(``python tests/test_hxn_regression.py`` prints them) with the pinch-outward
MER planner (``hensmith._planner``) and the curve-based problem table
(``hensmith._curves``), which replaced the four-pass heuristic synthesizer
and the grid of stream end temperatures. Cases 1-3, 6, 7 and 10 reach their
MER targets; cases 4, 8 and 9 need stream splits for MER (the pinch design
rules fail at the pinch: case 4 above it, where the superheat of the
ethanol vapor and the hot water both reach the pinch against a single cold
stream; case 8 above and below it; case 9 above it), so their loads are
best-effort networks 0.12 %, 0.55 % and 0.06 % above the hot-utility
target. Relative to the previous baselines (recorded with the heuristic
synthesizer at commits ``1ab689ff`` and later on branches
``hxn-regression-tests`` and ``fix-hxn-hot-side-offset-break``):

* cases 4 and 10 were RAISED (2.37319e6 -> 2.40548e6 and 1.40742e7 ->
  1.44713e7 kJ/hr heating): the old loads lay below the corrected MER
  targets (2.40262e6 and 1.44713e7) and were reachable only because
  exchangers of the old networks crossed internally (exact internal
  approaches of 4.63 K and 2.07 K, below T_min_app = 5 K), which the
  terminal checks of HXprocess do not see; both new networks keep 5 K
  everywhere;
* cases 5, 8 and 9 were LOWERED (case 5 from 3.2224e6 to 0 kJ/hr heating:
  the new network reaches its threshold target);
* cases 1-3, 6 and 7 are unchanged.

A documented load of zero is met to within 1e-9 of the total stream duty
(the problem table's own threshold tolerance): the utilities come from
enthalpy flashes, which leave residuals of ~1e-12 of the duty.
Improvements leave slack; a maintainer lowers the numbers deliberately when
a better network is intended. Never raise them to make a failing test pass
(cases 4 and 10 were raised because the old networks were infeasible).

``test_hxn_regression_with_splitting`` synthesizes every case again with
``stream_splitting=True`` and holds it to the same checks (i)-(v). Cases
4, 8 and 9 (``SPLIT_CASES``) must then reach their MER targets with split
streams; every other case must give, bit for bit, the network it gives
without the option (the same exchangers and duties, refine rounds and
repairs).
"""
import warnings
import pytest
import biosteam as bst
from numpy.testing import assert_allclose
from hensmith import HeatExchangerNetwork
from hensmith.hxn_synthesis import problem_table, _pinch_cut
# exact stream temperatures from forward property calls only (no flashes)
from test_hxn_mer import _temperature, _min_approach, APPROACH_TOL

EB_TOLERANCE = 1e-6  # percent; converged networks close to ~1e-10 %
MER_RTOL = 1e-9      # network may not beat the MER target by more than this x total duty
DOC_RTOL = 1e-3      # network may not be worse than documented by more than this
ZERO_RTOL = 1e-9     # a documented zero load, x total stream duty (see above)
STATUS_RTOL = 1e-6   # 'mer' iff the loads equal the targets, x total duty
CASCADE_RTOL = 1e-9  # planner targets vs problem table, x total duty

def utility_hx(ID, T, P, phase, T_out, rigorous=None, **flow):
    """A simulated HXutility acting as one process stream (kmol/hr flows)."""
    s = bst.Stream(ID + '_in', T=T, P=P, phase=phase, units='kmol/hr', **flow)
    if rigorous is None: rigorous = phase == 'g'
    hx = bst.HXutility(ID, ins=s, T=T_out, rigorous=rigorous)
    hx.simulate()
    return hx

def boiling_hx(ID, T, P, T_out, **flow):
    """A cold liquid stream heated past its bubble point (rigorous VLE)."""
    return utility_hx(ID, T, P, 'l', T_out, rigorous=True, **flow)

def setup(name, chemicals=('Water', 'Ethanol')):
    bst.settings.set_thermo(list(chemicals), cache=True)
    bst.main_flowsheet.set_flowsheet('test_hxn_regression_' + name)

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_01_two_liquids():
    """Hot liquid, cold liquid; trivial counter-current match."""
    setup('01')
    return [utility_hx('H1', 400., 5e5, 'l', 320., Water=1000.),
            utility_hx('C1', 300., 101325., 'l', 360., Water=1000.)], 5.

def case_02_pinch_limited():
    """Cold target above the hot inlet: part of the heating must be utility."""
    setup('02')
    return [utility_hx('H1', 360., 5e5, 'l', 320., Water=1000.),
            utility_hx('C1', 300., 5e5, 'l', 380., Water=800.)], 5.

def case_03_condenser_two_colds():
    """Condensing ethanol vapor against two cold liquids."""
    setup('03')
    return [utility_hx('H1', 355., 101325., 'g', 340., Ethanol=400.),
            utility_hx('C1', 300., 101325., 'l', 345., Water=1500.),
            utility_hx('C2', 310., 101325., 'l', 340., Water=300., Ethanol=300.)], 5.

def case_04_report_case():
    """The 4-stream report case: 2 colds, condensing hot, hot liquid."""
    setup('04')
    return [utility_hx('C1', 300., 101325., 'l', 390., Water=2000.),
            utility_hx('C2', 310., 101325., 'l', 345., Water=500., Ethanol=500.),
            utility_hx('H1', 352., 101325., 'g', 340., Ethanol=300.),
            utility_hx('H2', 420., 5e5, 'l', 320., Water=800.)], 5.

def case_05_boiling_cold():
    """A cold stream that boils (water -> steam) and a condensing hot stream."""
    setup('05')
    return [boiling_hx('C1', 330., 101325., 380., Water=300.),
            utility_hx('C2', 300., 101325., 'l', 350., Water=1000.),
            utility_hx('H1', 420., 5e5, 'g', 330., Water=250.),
            utility_hx('H2', 400., 5e5, 'l', 310., Water=1500.)], 5.

def case_06_mixed_pressures():
    """5-bar condensing hot against 1-atm boiling cold; T_min_app = 10."""
    setup('06')
    return [utility_hx('H1', 430., 5e5, 'g', 400., Water=200.),
            utility_hx('H2', 380., 5e5, 'l', 320., Water=1200.),
            boiling_hx('C1', 340., 101325., 375., Water=150.),
            utility_hx('C2', 300., 101325., 'l', 360., Ethanol=800.),
            utility_hx('C3', 320., 101325., 'l', 390., Water=600.)], 10.

def case_07_threshold():
    """Threshold problem: heating dominates so the cold target is ~0;
    one hot stream condenses and subcools."""
    setup('07')
    return [utility_hx('H1', 375., 101325., 'g', 310., Water=80.),
            utility_hx('H2', 360., 101325., 'l', 330., Water=200.),
            utility_hx('C1', 300., 101325., 'l', 370., Water=1500.),
            utility_hx('C2', 305., 101325., 'l', 350., Ethanol=800.),
            boiling_hx('C3', 340., 101325., 355., Ethanol=200.),
            utility_hx('C4', 320., 101325., 'l', 360., Water=700.)], 5.

def case_08_two_condensers():
    """Two condensers at different temperatures (ethanol 1 atm, water 2 bar)
    against three colds, one of which boils."""
    setup('08')
    return [utility_hx('H1', 355., 101325., 'g', 335., Ethanol=300.),
            utility_hx('H2', 400., 2e5, 'g', 360., Water=150.),
            utility_hx('H3', 390., 5e5, 'l', 330., Water=900.),
            utility_hx('C1', 300., 101325., 'l', 345., Water=1200.),
            boiling_hx('C2', 330., 101325., 370., Ethanol=250.),
            utility_hx('C3', 310., 101325., 'l', 380., Water=700.)], 5.

def case_09_near_degenerate():
    """Eight streams with two near-equal pinch candidates, a partially
    condensing hot stream (wet outlet) and a partially boiling cold stream."""
    setup('09')
    return [utility_hx('H1', 380., 101325., 'g', 372., Water=120.),   # partial condensation
            utility_hx('H2', 365., 101325., 'g', 330., Ethanol=250.),
            utility_hx('H3', 410., 5e5, 'l', 340., Water=700.),
            utility_hx('H4', 345., 101325., 'l', 305., Ethanol=900.),
            boiling_hx('C1', 350., 101325., 373.5, Water=200.),        # partial boiling
            utility_hx('C2', 300., 101325., 'l', 340., Water=1500.),
            boiling_hx('C3', 320., 101325., 352., Ethanol=300.),
            utility_hx('C4', 335., 101325., 'l', 395., Water=500.)], 5.

def case_10_ten_streams():
    """Ten streams mixing liquids, condensers, boilers, and pressures."""
    setup('10')
    return [utility_hx('H1', 355., 101325., 'g', 320., Ethanol=300.),
            utility_hx('H2', 420., 5e5, 'g', 340., Water=150.),
            utility_hx('H3', 395., 5e5, 'l', 330., Water=1000.),
            utility_hx('H4', 370., 101325., 'l', 310., Ethanol=700.),
            utility_hx('H5', 380., 101325., 'g', 372.5, Water=100.),   # partial condensation
            utility_hx('C1', 300., 101325., 'l', 360., Water=2000.),
            boiling_hx('C2', 330., 101325., 380., Water=200.),
            boiling_hx('C3', 320., 101325., 352., Ethanol=400.),
            utility_hx('C4', 310., 101325., 'l', 345., Water=400., Ethanol=400.),
            utility_hx('C5', 340., 101325., 'l', 390., Water=600.)], 5.

# name -> (builder, documented hot utility load [kJ/hr], documented cold utility load [kJ/hr])
# Documented values: see module docstring for provenance.
CASES = {
    'case_01_two_liquids':         (case_01_two_liquids,         0, 1.53912e+06),
    'case_02_pinch_limited':       (case_02_pinch_limited,       1.81522e+06, 0),
    'case_03_condenser_two_colds': (case_03_condenser_two_colds, 0, 9.49905e+06),
    'case_04_report_case':         (case_04_report_case,         2.40548e+06, 3.601e+06),
    'case_05_boiling_cold':        (case_05_boiling_cold,        0, 4.59225e+06),
    'case_06_mixed_pressures':     (case_06_mixed_pressures,     2.12463e+06, 0),
    'case_07_threshold':           (case_07_threshold,           1.85977e+07, 0),
    'case_08_two_condensers':      (case_08_two_condensers,      3.01517e+06, 7.35711e+06),
    'case_09_near_degenerate':     (case_09_near_degenerate,     1.40002e+07, 9.56801e+06),
    'case_10_ten_streams':         (case_10_ten_streams,         1.44713e+07, 8.46199e+06),
}

#: the cases whose MER needs stream splits (see the module docstring)
SPLIT_CASES = ('case_04_report_case', 'case_08_two_condensers', 'case_09_near_degenerate')

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def synthesize(builder, stream_splitting=False):
    units, T_min_app = builder()
    HXN = HeatExchangerNetwork('HXN', T_min_app=T_min_app,
                               stream_splitting=stream_splitting)
    sys = bst.System.from_units('sys', units=[*units, HXN])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # biosteam's HeatUtility.load_agent names a new 'oxygen_rich_inlet'
        # stream for every fuel (furnace) utility; the network itself replaces
        # nothing in the registry (test_synthesis_registers_no_intermediate_streams)
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='.*<Stream: oxygen_rich_inlet> has been replaced in registry')
        sys.simulate()
    return units, HXN, T_min_app

def HXN_table(units, T_min_app):
    """The problem table of the process streams, prepared as the facility
    prepares them."""
    hus = [hx.heat_utilities[0] for hx in units]
    hus.sort(key=lambda hu: hu.duty)
    streams_inlet = [hu.unit.ins[0].copy() for hu in hus]
    streams_quenched = [hu.unit.outs[0].copy() for hu in hus]
    for s in streams_quenched: s.vle(H=s.H, P=s.P)
    is_hot = [hu.duty < 0 for hu in hus]
    return problem_table(streams_inlet, streams_quenched, is_hot, T_min_app)

def mer_targets(units, T_min_app):
    table = HXN_table(units, T_min_app)
    return table.hot_util_load, table.cold_util_load

def actual_loads(HXN):
    hus = [hu for hx in HXN.new_HX_utils for hu in hx.heat_utilities]
    heat = sum(hu.unit_duty for hu in hus if hu.unit_duty > 0)
    cool = -sum(hu.unit_duty for hu in hus if hu.unit_duty < 0)
    return heat, cool

def internal_approach(hx):
    """Exact minimum approach [K] inside a process exchanger, on the states of
    its own streams (inf if it transfers no heat)."""
    dH = [s_in.H - s_out.H for s_in, s_out in zip(hx.ins, hx.outs)]
    h = 0 if dH[0] >= dH[1] else 1
    c = 1 - h
    if dH[h] <= 0.: return float('inf')
    return _min_approach(_temperature(hx.ins[h]), hx.ins[h].H, hx.outs[h].H,
                         _temperature(hx.ins[c]), hx.ins[c].H, hx.outs[c].H)

def check_network(name, units, HXN, T_min_app):
    """Checks (i)-(v) of the module docstring on the simulated network."""
    _, doc_heat, doc_cool = CASES[name]
    total = sum(abs(hx.heat_utilities[0].unit_duty) for hx in units)
    # (i) energy balance
    assert abs(HXN.energy_balance_percent_error) < EB_TOLERANCE, name
    # (ii) MER targets are a lower bound, reached iff the status says so;
    # energy identity holds
    heat, cool = actual_loads(HXN)
    hot_target, cold_target = mer_targets(units, T_min_app)
    net_duty = sum(hx.heat_utilities[0].unit_duty for hx in units)
    assert heat >= hot_target - MER_RTOL * total, (name, heat, hot_target)
    assert cool >= cold_target - MER_RTOL * total, (name, cool, cold_target)
    assert_allclose(heat - cool, net_duty, rtol=1e-8, err_msg=name)
    info = HXN.synthesis_info
    at_mer = (abs(heat - hot_target) <= STATUS_RTOL * total
              and abs(cool - cold_target) <= STATUS_RTOL * total)
    assert info['status'] == ('mer' if at_mer else 'best_effort'), (name, info['status'])
    # (iii) exact internal approach inside every process exchanger
    for hx in HXN.new_HXs:
        approach = internal_approach(hx)
        assert approach >= T_min_app - APPROACH_TOL, (name, hx.ID, approach)
    assert info['min_approach'] >= T_min_app - APPROACH_TOL, (name, info['min_approach'])
    # (iv) the planner's cascade is the problem table's
    table = HXN_table(units, T_min_app)
    planned = info['plan_targets']
    assert abs(planned['Q_hot'] - table.hot_util_load) <= CASCADE_RTOL * total, name
    assert abs(planned['Q_cold'] - table.cold_util_load) <= CASCADE_RTOL * total, name
    assert abs(planned['pinch_T'] - table.pinch_T) <= 1e-9, name
    assert planned['cut'] == _pinch_cut(table), name
    # (v) never worse than documented
    assert doc_heat is not None and doc_cool is not None, f'{name}: baseline not recorded'
    atol = ZERO_RTOL * total
    assert heat <= doc_heat * (1 + DOC_RTOL) + atol, (name, heat, doc_heat)
    assert cool <= doc_cool * (1 + DOC_RTOL) + atol, (name, cool, doc_cool)

@pytest.mark.parametrize('name', list(CASES))
def test_hxn_regression(name):
    builder = CASES[name][0]
    check_network(name, *synthesize(builder))

@pytest.mark.parametrize('name', list(CASES))
def test_hxn_regression_with_splitting(name):
    # with stream splitting, every case passes the same checks; the cases
    # whose MER needs splits reach it with them, and every other case gives
    # the network synthesized without the option, bit for bit
    builder = CASES[name][0]
    units, HXN, T_min_app = synthesize(builder, stream_splitting=True)
    check_network(name, units, HXN, T_min_app)
    info = HXN.synthesis_info
    if name in SPLIT_CASES:
        assert info['status'] == 'mer', name
        assert info['splits'] and HXN.new_splitters and HXN.new_mixers, name
        return
    default = synthesize(builder)[1]
    assert info['status'] == default.synthesis_info['status'] == 'mer', name
    assert info['splits'] == HXN.new_splitters == HXN.new_mixers == [], name
    assert ([(hx.ID, float(hx.Q).hex()) for hx in HXN.new_HXs]
            == [(hx.ID, float(hx.Q).hex()) for hx in default.new_HXs]), name
    for key in ('refine_rounds', 'repaired'):
        assert info[key] == default.synthesis_info[key], (name, key)

if __name__ == '__main__':
    for name, (builder, *_) in CASES.items():
        units, HXN, T_min_app = synthesize(builder)
        heat, cool = actual_loads(HXN)
        hot_target, cold_target = mer_targets(units, T_min_app)
        approach = min(map(internal_approach, HXN.new_HXs), default=float('inf'))
        print(f"{name}: heat={heat:.6g} cool={cool:.6g} "
              f"(MER hot={hot_target:.6g} cold={cold_target:.6g}; "
              f"status={HXN.synthesis_info['status']}; "
              f"min internal approach={approach:.6f} K; "
              f"EB error={HXN.energy_balance_percent_error:.2e}%)")
