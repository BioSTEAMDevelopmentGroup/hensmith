# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Tests for the heat exchanger network facility.
"""
import warnings
import pytest
import biosteam as bst
import numpy as np
from numpy.testing import assert_allclose
from hensmith import HeatExchangerNetwork

def build_system(N_columns=1):
    """Doctest system of HeatExchangerNetwork; `N_columns > 1` adds more
    ShortcutColumns so auxiliary heat exchangers have duplicate IDs."""
    bst.settings.set_thermo(['Water', 'Methanol', 'Glycerol'], cache=True)
    bst.main_flowsheet.set_flowsheet('test_hxn')
    units = []
    feeds = []
    for i in range(N_columns):
        feed = bst.Stream(f'feed{i}', flow=(8000, 100 * (i + 1), 25))
        D = bst.ShortcutColumn(f'D{i}', ins=feed, LHK=('Methanol', 'Water'),
                               y_top=0.99, x_bot=0.01, k=2, is_divided=True)
        H1 = bst.HXutility(f'D{i}_H1', ins=D.outs[1], T=300)
        H2 = bst.HXutility(f'D{i}_H2', ins=D.outs[0], T=300)
        units.extend([D, H1, H2])
        feeds.append(feed)
    feed2 = bst.Stream('feed_flash', flow=(10000, 1000, 10))
    F1 = bst.Flash('F1', ins=feed2, V=0.9, P=101325)
    HXN = HeatExchangerNetwork('HXN', T_min_app=5.)
    sys = bst.System.from_units('sys', units=[*units, F1, HXN])
    return sys, HXN, feeds[0]

def network_results(HXN):
    return dict(
        heat=HXN.actual_heat_util_load,
        cool=HXN.actual_cool_util_load,
        Q=np.array([hx.Q for hx in HXN.new_HXs]),
        installed=HXN.installed_costs['Heat exchangers'],
    )

def assert_same_results(a, b, rtol=1e-6):
    for key in a:
        assert_allclose(a[key], b[key], rtol=rtol, err_msg=key)

def simulate_cached(sys, HXN):
    HXN.cache_network = True
    HXN_sys = HXN.HXN_sys
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        sys.simulate()
    assert HXN.HXN_sys is HXN_sys, 'cached network was not used'

def test_cache_network_matches_fresh_synthesis():
    sys, HXN, feed = build_system()
    sys.simulate()
    fresh = network_results(HXN)
    assert HXN.actual_heat_util_load < 0.9 * HXN.original_heat_util_load
    simulate_cached(sys, HXN)
    assert_same_results(network_results(HXN), fresh)

def test_cache_network_perturbed_feed():
    sys, HXN, feed = build_system()
    sys.simulate()
    feed.F_mass *= 1.000001
    simulate_cached(sys, HXN)
    cached = network_results(HXN)
    HXN.cache_network = False
    sys.simulate()
    assert_same_results(cached, network_results(HXN))

def test_cache_network_duplicate_IDs():
    sys, HXN, feed = build_system(N_columns=2)
    sys.simulate()
    IDs = [hx.ID for hx in HXN.original_heat_exchangers]
    assert len(IDs) != len(set(IDs)), 'test needs duplicate auxiliary IDs'
    fresh = network_results(HXN)
    simulate_cached(sys, HXN)
    assert_same_results(network_results(HXN), fresh)

def assert_no_phantom_utility_exchangers(HXN):
    """Every utility exchanger either has exactly no duty, and so no design
    and no cost, or a real duty, and a cost; a duty at the level of flash
    noise would be costed as a minimum-size exchanger."""
    for hx, life_cycle in zip(HXN.original_heat_exchangers, HXN.stream_life_cycles):
        util = life_cycle.life_cycle[-1].unit
        assert isinstance(util, bst.HXutility)
        duty = abs(hx.outs[0].H - hx.ins[0].H)
        if util.Hnet == 0.:
            assert not util.design_results and util.installed_cost == 0., util.ID
        else:
            assert abs(util.Hnet) > 1e-6 * duty, (util.ID, util.Hnet)
            assert util.installed_cost > 0., util.ID

def test_served_streams_leave_their_utility_exchangers_unchanged():
    # streams 3 and 4 of the doctest system are served completely by process
    # exchangers; the cached network used to re-flash stream 4 at its outlet
    # enthalpy in its utility exchanger, which moved its phase split in the
    # last digits and left a net duty of -9.3e-10 kJ/hr that was costed as a
    # minimum-size exchanger (1,304 USD installed) absent from the fresh one
    sys, HXN, feed = build_system()
    sys.simulate()
    served = [util for util in HXN.new_HX_utils if util.Hnet == 0.]
    assert len(served) == 2
    assert_no_phantom_utility_exchangers(HXN)
    simulate_cached(sys, HXN)
    assert [util for util in HXN.new_HX_utils if util.Hnet == 0.] == served
    for util in served: # the stream leaves in the state it enters
        inlet, product = util.ins[0], util.outs[0]
        assert product.T == inlet.T and product.H == inlet.H
        assert (product.mol == inlet.mol).all()
    assert_no_phantom_utility_exchangers(HXN)

def test_utility_exchangers_below_Qmin_are_costed():
    # Qmin drops small process matches only: every utility exchanger with a
    # duty is designed and costed, however small the duty (here H1's 10 kW
    # cooler, below Qmin = 100,000 kJ/hr)
    units = cp_units('Qmin_cost', [('C1', 300., 400., 1.), ('H1', 450., 350., 1.1)])
    Qmin = 1e5
    sys, HXN = simulate_HXN(units, 10., Qmin=Qmin)
    hx, = HXN.new_HXs
    assert hx.Q > Qmin
    cooler, = [u for u in HXN.new_HX_utils if u.Hnet]
    assert_allclose(-cooler.Hnet, 10. * kW, rtol=1e-9)
    assert -cooler.Hnet < Qmin and cooler.purchase_cost > 0.
    utils = HXN.new_HX_utils
    assert HXN.new_purchase_costs_HXu == [u.purchase_cost for u in utils]
    assert_allclose(HXN.purchase_costs['Heat exchangers'], max(0., (
        hx.purchase_cost + sum(u.purchase_cost for u in utils)
        - sum(u.purchase_cost for u in units)
    )), rtol=1e-12)

def test_energy_balance_error_contributions_ignored_none():
    sys, HXN, feed = build_system()
    sys.simulate()
    N = len(HXN.original_heat_utils)
    errors = HXN._energy_balance_error_contributions()
    assert len(errors) == N
    assert HXN.ignored is None

def strict_simulate(item):
    """Simulate a system or unit; a RuntimeWarning is an error (except
    biosteam's furnace-air registry bookkeeping, see `simulate_HXN`)."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='.*<Stream: oxygen_rich_inlet> has been replaced in registry')
        item.simulate()

def network_loads(HXN):
    return [HXN.original_heat_util_load, HXN.original_cool_util_load,
            HXN.actual_heat_util_load, HXN.actual_cool_util_load]

def assert_units_carry_the_network_utilities(HXN, units):
    """Each original heat utility equals the heat utility of its OWN
    stream's utility exchanger, every unit's utility cost is that of its
    heat utilities, and the facility carries none."""
    for i, hu in enumerate(HXN.original_heat_utils):
        assert hu.unit is HXN.original_heat_exchangers[i]
        util = HXN.stream_life_cycles[i].life_cycle[-1].unit
        assert _stream_ports(util) == (i,)
        new = util.heat_utilities[0]
        assert hu.agent is new.agent, hu.unit.ID
        assert (hu.duty, hu.flow, hu.cost) == (new.duty, new.flow, new.cost)
    for unit in units:
        costs = sum(hu.cost for hu in unit.heat_utilities) + unit.power_utility.cost
        assert_allclose(unit.utility_cost, costs, rtol=1e-12, atol=0.)
    assert HXN.heat_utilities == [] and HXN.utility_cost == 0.

@pytest.mark.parametrize('case', ['kemp', 'doctest'])
def test_replace_unit_heat_utilities(case):
    # replace_unit_heat_utilities raised an AttributeError (biosteam's
    # Unit._load_utility_cost no longer exists) and zipped the original
    # heat utilities, in stream order (cold streams first), with
    # new_HX_utils, listed hot streams first (Kemp: C1's heater would take
    # H2's cooler duty). Each original heat utility now takes its own
    # stream's utility, so the units report the total utility cost that
    # the facility's net utilities report without the option.
    if case == 'kemp':
        units = cp_units('replace_kemp', KEMP, 273.15)
        HXN = HeatExchangerNetwork('HXN', T_min_app=10.)
        sys = bst.System.from_units('sys_replace', units=[*units, HXN])
    else: # auxiliary exchangers: a column's condenser and boiler, a flash's
        sys, HXN, _ = build_system()
        units = [u for u in sys.units if u is not HXN]
    strict_simulate(sys)
    N = len(HXN.original_heat_utils)
    loads = network_loads(HXN)
    unit_costs = [u.utility_cost for u in units]
    total = sum(unit_costs) + HXN.utility_cost
    assert HXN.utility_cost < 0.
    served = [i for i, lc in enumerate(HXN.stream_life_cycles)
              if not lc.life_cycle[-1].unit.heat_utilities[0].duty]
    assert served
    HXN.replace_unit_heat_utilities = True
    strict_simulate(sys) # the units, then the network
    assert_units_carry_the_network_utilities(HXN, units)
    assert_allclose(network_loads(HXN), loads, rtol=1e-9)
    assert_allclose(sum(u.utility_cost for u in units), total, rtol=1e-9)
    assert HXN.synthesis_info['status'] == 'mer'
    # the network alone again: the units still hold the replaced utilities,
    # in which the served streams have none; the network restores the
    # original ones first, so no stream drops out and nothing changes
    strict_simulate(HXN)
    assert len(HXN.original_heat_utils) == N
    assert_units_carry_the_network_utilities(HXN, units)
    assert_allclose(network_loads(HXN), loads, rtol=1e-9)
    assert_allclose(sum(u.utility_cost for u in units), total, rtol=1e-9)
    # without the option, the units get their own utilities back
    HXN.replace_unit_heat_utilities = False
    strict_simulate(HXN)
    assert_allclose([u.utility_cost for u in units], unit_costs, rtol=1e-12)
    assert_allclose(sum(unit_costs) + HXN.utility_cost, total, rtol=1e-9)
    assert_allclose(network_loads(HXN), loads, rtol=1e-9)

def test_HXN_flowsheet_is_the_network_flowsheet():
    """HXN_flowsheet must be the '<sys>_HXN' Flowsheet that holds the network,
    not the main-flowsheet proxy (which reads back as whatever is active)."""
    sys, HXN, _ = build_system()
    sys.simulate()
    fs = HXN.HXN_flowsheet
    assert isinstance(fs, bst.Flowsheet)
    assert fs is not bst.main_flowsheet
    assert fs.ID == sys.ID + '_HXN'
    assert fs.ID != bst.main_flowsheet.ID
    assert fs is bst.main_flowsheet.flowsheet[sys.ID + '_HXN']
    # Every unit of the synthesized network is registered in that flowsheet
    # (Registry iterates over its registered objects), and none of them leaked
    # into the user's main flowsheet.
    network_units = HXN.HXN_sys.units
    assert network_units
    assert all(unit in fs.unit for unit in network_units)
    assert not any(unit in bst.main_flowsheet.unit for unit in network_units)

def test_HXN_sys_is_named_and_registered_in_the_network_flowsheet():
    """HXN_sys carries the '<sys>_HXN' ID and is registered in the network
    flowsheet's system registry, so it is addressable there like the network
    units are, and print() shows the name rather than '-'."""
    sys, HXN, _ = build_system()
    sys.simulate()
    ID = sys.ID + '_HXN'
    HXN_sys = HXN.HXN_sys
    assert HXN_sys.ID == ID
    assert str(HXN_sys) == ID
    network_flowsheet = bst.main_flowsheet.flowsheet[sys.ID + '_HXN']
    assert network_flowsheet.system[ID] is HXN_sys
    assert bst.main_flowsheet.system.search(ID) is None
    # A fresh synthesis builds a new network flowsheet with empty registries,
    # so re-registering the same ID cannot collide; the cached path keeps the
    # named object.
    sys.simulate()
    assert HXN.HXN_sys is not HXN_sys
    assert HXN.HXN_sys.ID == ID
    network_flowsheet = bst.main_flowsheet.flowsheet[sys.ID + '_HXN']
    assert network_flowsheet.system[ID] is HXN.HXN_sys
    HXN.cache_network = True
    sys.simulate()
    assert HXN.HXN_sys.ID == ID

def test_streams_inlet_holds_the_inlet_states():
    """`streams_inlet` is a clean list of inlet copies, one per exchanger in
    `original_heat_exchangers` order, not the cold-side working list whose
    entries end up as pinch states or exchanger outlets."""
    sys, HXN, _ = build_system()
    sys.simulate()
    assert len(HXN.streams_inlet) == len(HXN.original_heat_exchangers)
    network_streams = {id(s) for hx in HXN.new_HXs + HXN.new_HX_utils
                       for s in (*hx.ins, *hx.outs)}
    for s, hx in zip(HXN.streams_inlet, HXN.original_heat_exchangers):
        inlet = hx.ins[0]
        assert_allclose(s.T, inlet.T)
        assert_allclose(s.H, inlet.H)
        assert_allclose(s.mol, inlet.mol)
        assert id(s) not in network_streams

def test_life_stage_reports_enthalpy_flow_units():
    """LifeStage enthalpies are Stream.H values, i.e. flows in kJ/hr, and its
    repr and show() label them as such."""
    sys, HXN, _ = build_system()
    sys.simulate()
    stage = HXN.stream_life_cycles[0].life_cycle[0]
    assert stage.H_in == stage.s_in.H
    assert ' kJ/hr,' in repr(stage) and repr(stage).endswith(' kJ/hr>')
    assert stage._info().count(' kJ/hr') == 2
    assert repr(HXN.stream_life_cycles[0]).count(' kJ/hr') == 2 * len(HXN.stream_life_cycles[0].life_cycle)

# ---------------------------------------------------------------------------
# Problem-table (pinch) analysis
# ---------------------------------------------------------------------------

from hensmith.hxn_synthesis import (
    temperature_interval_pinch_analysis, problem_table, load_duties, pinch_state,
)

def utility_hx(ID, T, P, phase, T_out, **flow):
    """A simulated HXutility acting as one process stream (kmol/hr flows)."""
    s = bst.Stream(ID + '_in', T=T, P=P, phase=phase, units='kmol/hr', **flow)
    hx = bst.HXutility(ID, ins=s, T=T_out,
                       rigorous=(phase == 'g'))  # liquid streams stay liquid (report's case)
    hx.simulate()
    return hx

def synthetic_units():
    """Report's 4-stream case: two cold, one condensing hot, one hot liquid."""
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('test_hxn_synthetic')
    return [utility_hx('C1', 300., 101325., 'l', 390., Water=2000.),
            utility_hx('C2', 310., 101325., 'l', 345., Water=500., Ethanol=500.),
            utility_hx('H1', 352., 101325., 'g', 340., Ethanol=300.),
            utility_hx('H2', 420., 5e5, 'l', 320., Water=800.)]

def heat_utilities(units):
    hus = [hx.heat_utilities[0] for hx in units]
    hus.sort(key=lambda hu: hu.duty)
    return hus

def pinch_streams(hus):
    """Inlet/quenched-outlet stream copies exactly as the synthesizer prepares them."""
    streams_inlet = [hu.unit.ins[0].copy() for hu in hus]
    streams_quenched = [hu.unit.outs[0].copy() for hu in hus]
    for s in streams_quenched: s.vle(H=s.H, P=s.P)
    is_hot = [hu.duty < 0 for hu in hus]
    return streams_inlet, streams_quenched, is_hot

def assert_energy_consistent(hus, T_min_app):
    """Invariants of a correct problem table, independent of the synthesizer."""
    unit_duties = np.array([hu.unit_duty for hu in hus])
    table = problem_table(*pinch_streams(hus), T_min_app)
    # (a) each stream's grid contributions telescope to its real duty
    #     (hot: +|dH|, cold: -dH)
    per_stream = table.interval_H.sum(axis=1) + table.point_H.sum(axis=1)
    assert_allclose(per_stream, -unit_duties, rtol=1e-9)
    # (b) targets are non-negative and hot - cold is the net demand
    assert table.hot_util_load >= 0. and table.cold_util_load >= 0.
    assert_allclose(table.hot_util_load - table.cold_util_load,
                    unit_duties.sum(), rtol=1e-9)
    # (c) a target can never exceed the un-integrated load
    assert table.hot_util_load <= unit_duties[unit_duties > 0].sum() * (1 + 1e-9)
    assert table.cold_util_load <= -unit_duties[unit_duties < 0].sum() * (1 + 1e-9)
    # (d) the public wrapper reports the same targets
    pinch_T_arr, hot, cold, *_ = temperature_interval_pinch_analysis(hus, T_min_app)
    assert_allclose([hot, cold], [table.hot_util_load, table.cold_util_load], rtol=1e-12)
    return table

def test_problem_table_energy_consistency_doctest_system():
    sys, HXN, feed = build_system()
    sys.simulate()
    hus = HXN._get_original_heat_utilties()
    hus.sort(key=lambda hu: hu.duty)
    assert_energy_consistent(hus, 5.)

@pytest.mark.parametrize('T_min_app', [5., 10., 20.])
def test_problem_table_energy_consistency_synthetic(T_min_app):
    hus = heat_utilities(synthetic_units())
    table = assert_energy_consistent(hus, T_min_app)
    # the condensing ethanol stream (352 K in, ~351.4 K dew point) must keep
    # its latent heat: hot target well below the un-integrated heating load
    heating = sum(hu.unit_duty for hu in hus if hu.unit_duty > 0)
    assert table.hot_util_load < 0.5 * heating

def test_problem_table_two_streams_closed_form():
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('test_hxn_two_streams')
    hot = utility_hx('Hw', 400., 5e5, 'l', 300., Water=1000.)
    # Case 1 (threshold problem): the hot stream covers every interval of the
    # smaller cold stream -> no hot utility, cold utility = net surplus.
    cold = utility_hx('Cs', 300., 5e5, 'l', 390., Water=900.)
    hus = heat_utilities([hot, cold])
    table = problem_table(*pinch_streams(hus), 5.)
    Q_hot = -hot.heat_utilities[0].unit_duty
    Q_cold = cold.heat_utilities[0].unit_duty
    assert table.hot_util_load == 0.
    assert_allclose(table.cold_util_load, Q_hot - Q_cold, rtol=1e-9)
    assert table.pinch_T == table.Ts[0]   # no pinch: everything sits below it
    # Case 2: the larger cold stream is short of heat everywhere; the cascade
    # minimum is at the cold inlet (300 K on the shifted scale) and the hot
    # utility is the cold duty minus what the hot stream gives down to 305 K.
    cold = utility_hx('Cb', 300., 5e5, 'l', 390., Water=1200.)
    hus = heat_utilities([hot, cold])
    table = problem_table(*pinch_streams(hus), 5.)
    s = hot.ins[0].copy(); s.vle(T=305., P=s.P)
    Q_hot_above_pinch = hot.ins[0].H - s.H
    Q_cold = cold.heat_utilities[0].unit_duty
    assert_allclose(table.hot_util_load, Q_cold - Q_hot_above_pinch, rtol=1e-9)
    assert table.pinch_T == 300.
    # remaining hot-stream heat below the pinch leaves as cold utility
    assert_allclose(table.cold_util_load, s.H - hot.outs[0].H, rtol=1e-9)

def test_problem_table_non_monotone_stream_is_point_load():
    # A heated stream whose outlet is colder than its inlet (e.g. a column
    # reboiler outlet at VLE): treated as an isothermal load at T_out.
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    a = bst.Stream('a', Water=100., T=372., P=101325., phase='l', units='kmol/hr')
    b = bst.Stream('b', Water=100., T=371., P=101325., phase='g', units='kmol/hr')
    dH = b.H - a.H
    assert dH > 0
    table = problem_table([a], [b], [False], 5.)
    assert_allclose(table.point_H, [[-dH]])
    assert table.interval_H.shape == (1, 0)
    assert_allclose([table.hot_util_load, table.cold_util_load, table.pinch_T],
                    [dH, 0., 371.])
    table = problem_table([b], [a], [True], 5.)
    assert_allclose([table.hot_util_load, table.cold_util_load, table.pinch_T],
                    [0., dH, 372. - 5.])  # outlet T 372 K, shifted by T_min_app

def test_problem_table_point_load_cannot_heat_above_itself():
    # A source at shifted temperature T cannot serve sinks above T: an
    # isothermal condensing hot stream at 400 K (shifted to 395 K) must not
    # cover the 395-398 K segment of a cold stream that runs 392 -> 398 K.
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    hot_in = bst.Stream('hv', Water=100., T=400., P=101325., phase='g',
                        units='kmol/hr')
    hot_out = bst.Stream('hl', Water=100., T=400., P=101325., phase='l',
                         units='kmol/hr')
    P_hot = hot_in.H - hot_out.H
    assert P_hot > 0
    cold_in = bst.Stream('cl', Water=1000., T=392., P=5e5, phase='l',
                         units='kmol/hr')
    cold_out = cold_in.copy('cl_out'); cold_out.vle(T=398., P=5e5)
    s = cold_in.copy(); s.vle(T=395., P=5e5)
    H_392, H_395, H_398 = cold_in.H, s.H, cold_out.H
    assert P_hot > H_398 - H_392   # more than enough heat overall
    table = problem_table([hot_in, cold_in], [hot_out, cold_out],
                          [True, False], 5.)
    assert_allclose(table.Ts, [398., 395., 392.])
    # heat arriving at 395 K, before the point load there, is short by the
    # 395-398 K segment of the cold stream
    assert_allclose(table.hot_util_load, H_398 - H_395, rtol=1e-9)
    assert table.pinch_T == 395.
    assert_allclose(table.cold_util_load, P_hot - (H_395 - H_392), rtol=1e-9)

def test_load_duties_non_equilibrium_inlet_conserves_energy():
    # A non-rigorous HXutility can carry a superheated liquid (ethanol at
    # 370 K, 1 atm; bp 351.4 K). Flashing it at the 355 K pinch gives vapor
    # with far more enthalpy than the inlet has; the pinch split must clip
    # to the stream's real enthalpy range so that the hot-side and cold-side
    # loads sum to the stream's duty (and are not phantom latent heat).
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    s_in = bst.Stream('ne_in', Ethanol=700., T=370., P=101325., phase='l',
                      units='kmol/hr')
    s_out = bst.Stream('ne_out', Ethanol=700., T=310., P=101325., phase='l',
                       units='kmol/hr')
    duty = s_in.H - s_out.H
    flashed = s_in.copy(); flashed.vle(T=355., P=101325.)
    assert flashed.H > s_in.H   # the trap: equilibrium enthalpy above H_in
    Q_hot_side, Q_cold_side = {}, {}
    load_duties([s_in], [s_out], np.array([355.]), np.array([310.]), [0],
                lambda i: False, Q_hot_side, Q_cold_side)
    assert Q_hot_side[0][0] == Q_cold_side[0][0] == 'cool'
    assert_allclose(Q_hot_side[0][1] + Q_cold_side[0][1], duty, rtol=1e-9)
    # the stream cannot give any heat above the pinch: its whole enthalpy
    # range lies below the equilibrium state at 355 K
    assert_allclose(Q_hot_side[0][1], 0., atol=1e-6)
    assert_allclose(Q_cold_side[0][1], duty, rtol=1e-9)
    # and the transient state used for matching carries exactly H_in, at
    # equilibrium (two-phase at the bubble point), not at the fictitious
    # 370 K of the superheated liquid, so matching is consistent with the
    # problem table
    s = pinch_state(s_in, s_out, 355.)
    assert_allclose(s.H, s_in.H, rtol=1e-9)
    assert s.T < 355. and 'g' in s.phase and 'l' in s.phase

def test_pinch_state_at_endpoints_uses_real_states():
    # pinch_T == T_in must put the whole duty on one side even when the
    # inlet is not at equilibrium: flashing a superheated liquid at its own
    # T does not reproduce H_in, and the resulting enthalpy can lie inside
    # the stream's range so that clipping alone would not catch it.
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    s_in = bst.Stream('sh_in', Water=100., T=380., P=101325., phase='l',
                      units='kmol/hr')
    s_out = bst.Stream('sh_out', Water=100., T=400., P=101325., phase='g',
                       units='kmol/hr')
    duty = s_out.H - s_in.H
    flashed = s_in.copy(); flashed.vle(T=380., P=101325.)
    assert s_in.H < flashed.H < s_out.H   # the trap: in range, but not H_in
    Q_hot_side, Q_cold_side = {}, {}
    load_duties([s_in], [s_out], np.array([380.]), np.array([400.]), [0],
                lambda i: True, Q_hot_side, Q_cold_side)
    assert_allclose(Q_hot_side[0][1], duty, rtol=1e-9)
    assert_allclose(Q_cold_side[0][1], 0., atol=1e-6)
    assert_allclose(pinch_state(s_in, s_out, 380.).H, s_in.H, rtol=1e-9)
    assert_allclose(pinch_state(s_in, s_out, 400.).H, s_out.H, rtol=1e-9)
    # a phase-mislabelled non-condensable: clipping must return the real
    # end state, never an equilibrium re-flash (N2 "liquid" at 400 K would
    # otherwise come back as gas at thousands of K)
    bst.settings.set_thermo(['Water', 'N2'], cache=True)
    h_in = bst.Stream('n2_in', N2=100., T=400., P=101325., phase='l',
                      units='kmol/hr')
    h_out = bst.Stream('n2_out', N2=100., T=320., P=101325., phase='l',
                       units='kmol/hr')
    s = pinch_state(h_in, h_out, 355.)
    assert h_out.H <= s.H <= h_in.H
    assert 320. <= s.T <= 400.

def assert_path_follows_streams(HXN):
    """Every stage of every stream is simulated after the stage that feeds
    it, except across a declared recycle (tear) stream."""
    path = list(HXN.HXN_sys.path)
    assert set(path) == set(HXN.new_HXs + HXN.new_HX_utils)
    recycles = HXN.HXN_sys.recycle or []
    if not isinstance(recycles, list): recycles = [recycles]
    position = {unit: i for i, unit in enumerate(path)}
    for life_cycle in HXN.stream_life_cycles:
        stages = life_cycle.life_cycle
        for a, b in zip(stages, stages[1:]):
            s = a.unit.outs[a.index]
            assert b.unit.ins[b.index] is s
            assert position[b.unit] > position[a.unit] or s in recycles, (a.unit, b.unit)
    return recycles

def test_network_path_follows_the_streams():
    # hensmith orders the network itself (a topological order of the stream
    # stages, tearing loops), so the path never depends on a general network
    # sort that need not settle on intertwined loops
    units = synthetic_units()
    sys, HXN = simulate_HXN(units, 5., 'sys_path')
    assert_path_follows_streams(HXN)
    assert abs(HXN.energy_balance_percent_error) < 1e-6

def test_synthetic_network_needs_a_split():
    # the report case (regression case 04): above the pinch, the superheat of
    # the ethanol vapor and the hot water both reach the pinch against a
    # single cold stream, so MER needs a stream split (pinch number rule);
    # the unsplit network is a best-effort one that never beats MER
    units = synthetic_units()
    sys, HXN = simulate_HXN(units, 5., 'sys_synthetic')
    hus = heat_utilities(units)
    table = problem_table(*pinch_streams(hus), 5.)
    actual_heat = sum(hu.unit_duty for hx in HXN.new_HX_utils
                      for hu in hx.heat_utilities if hu.unit_duty > 0)
    assert table.hot_util_load * (1 - 1e-9) <= actual_heat <= table.hot_util_load * 1.002
    info = HXN.synthesis_info
    assert info['status'] == 'best_effort'
    proof = info['sides']['above']['proof']
    assert proof['rule'] == 'outward' and len(proof['musts']) == 2
    assert_feasible(HXN, 5.)

# ---------------------------------------------------------------------------
# Synthesis on the MER planner: realization, life cycles, facility
# ---------------------------------------------------------------------------

import thermosteam as tmo
from hensmith.hxn_synthesis import (
    synthesize_network, StreamLifeCycle, _pinch_cut, _stream_ports,
)
import hensmith.hxn_synthesis as hxn_synthesis
from test_hxn_regression import (
    internal_approach, actual_loads, mer_targets, case_05_boiling_cold,
    case_10_ten_streams,
)
from test_hxn_mer import APPROACH_TOL
import test_hxn_targets

kW = 3600.  # kJ/hr
_FLUID = []

def cp_units(name, rows, offset=0.):
    """One simulated HXutility per constant heat capacity stream ``(ID,
    T_in, T_out, CP [kW/K])``, temperatures + `offset` [K]; 1000*CP kmol/hr
    of the `Fluid` pseudo-component has a heat capacity flow rate of CP
    kW/K."""
    if not _FLUID:
        fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1.,
                             Cn=3.6, default=True)
        bst.settings.set_thermo([fluid], cache=True)
        _FLUID.append(bst.settings.thermo)
    bst.settings.set_thermo(_FLUID[0])
    bst.main_flowsheet.set_flowsheet(name)
    units = []
    for ID, T_in, T_out, CP in rows:
        inlet = bst.Stream(ID + '_in', Fluid=1000. * CP, T=T_in + offset,
                           units='kmol/hr')
        hx = bst.HXutility(ID, ins=inlet, T=T_out + offset, rigorous=False)
        hx.simulate()
        units.append(hx)
    return units

def simulate_HXN(units, T_min_app, name='sys', **kwargs):
    """Simulate the units with a HeatExchangerNetwork; a RuntimeWarning
    (including a stream replaced in the registry, except biosteam's own
    furnace-air stream) is an error."""
    HXN = HeatExchangerNetwork('HXN', T_min_app=T_min_app, **kwargs)
    sys = bst.System.from_units(name, units=[*units, HXN])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # biosteam's HeatUtility.load_agent names a new 'oxygen_rich_inlet'
        # stream for every fuel (furnace) utility; the network itself replaces
        # nothing in the registry (test_synthesis_registers_no_intermediate_streams)
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='.*<Stream: oxygen_rich_inlet> has been replaced in registry')
        sys.simulate()
    return sys, HXN

def assert_feasible(HXN, T_min_app, EB=1e-6):
    """Balanced network whose every process exchanger keeps T_min_app inside
    it on the exact states of its streams."""
    assert abs(HXN.energy_balance_percent_error) < EB
    for hx in HXN.new_HXs:
        assert internal_approach(hx) >= T_min_app - APPROACH_TOL, hx.ID
    assert HXN.synthesis_info['min_approach'] is None or (
        HXN.synthesis_info['min_approach'] >= T_min_app - APPROACH_TOL)

def total_duty(units):
    return sum(abs(hx.heat_utilities[0].unit_duty) for hx in units)

#: Problem r002: its only unsplit MER network matches H1 twice with C3
#: (160 / 30 / 20 / 20 kW below the pinch); 80 kW hot and 450 kW cold
#: utility at T_min_app = 10 K (temperatures + 300 K).
R002 = [('C1', 140., 170., 1.), ('C2', 100., 120., 1.),
        ('C3', 120., 250., 2.), ('H1', 220., 50., 4.)]
#: Linnhoff & Hindmarsh (1983): 20 kW hot, 60 kW cold at 10 K (degC).
KEMP = [('C1', 20., 135., 2.), ('H2', 170., 60., 3.),
        ('C3', 80., 140., 4.), ('H4', 150., 30., 1.5)]

def r002(name='r002', fillers=0):
    """r002 plus `fillers` utility-only cold streams far above it (300 kW
    each, so that they sort after C3 and push H1's index up)."""
    rows = R002[:3] + [(f'F{i}', 500., 600., 3.) for i in range(fillers)] + R002[3:]
    return cp_units(name, rows, 300.)

def test_repeated_pair_IDs_and_life_cycles():
    # nine filler cold streams make H1 stream 12 and C1, C3 streams 1, 2
    # (heating duties ascending: C2 0, C1 1, C3 2, fillers 3-11), so a
    # substring match on '_1_', '_2_' or 's_2_' would confuse them
    units = r002('r002_12', fillers=9)
    sys, HXN = simulate_HXN(units, 10.)
    IDs = [hx.ID for hx in HXN.new_HXs + HXN.new_HX_utils]
    assert len(IDs) == len(set(IDs))
    assert sorted(hx.ID for hx in HXN.new_HXs) == [
        'HX_12_0_cs', 'HX_12_1_cs', 'HX_12_2_cs', 'HX_12_2_cs_2']
    assert all(_stream_ports(hx) is not None for hx in HXN.new_HXs + HXN.new_HX_utils)
    cycles = {lc.index: lc for lc in HXN.stream_life_cycles}
    stages = lambda i: [(s.unit.ID, s.index) for s in cycles[i].life_cycle]
    # flow order: H1 passes C3's first match at its hot end, C3's second
    # match after C1's; C3 meets its second match first (cold end)
    assert stages(12) == [('HX_12_2_cs', 0), ('HX_12_1_cs', 0), ('HX_12_2_cs_2', 0),
                          ('HX_12_0_cs', 0), ('Util_12_cs', 0)]
    assert stages(2) == [('HX_12_2_cs_2', 1), ('HX_12_2_cs', 1), ('Util_2_hs', 0)]
    assert stages(1) == [('HX_12_1_cs', 1), ('Util_1_hs', 0)]
    assert stages(5) == [('Util_5_hs', 0)]
    assert_path_follows_streams(HXN)
    # recomputed after the rewiring of `_cost` (inlets are now the previous
    # stages' outlets, e.g. 'HX_12_2_cs_2__s_12'), the life cycles agree
    for i, lc in cycles.items():
        again = StreamLifeCycle(i, lc.cold).get_life_cycle(HXN.new_HXs, HXN.new_HX_utils)
        assert [(s.unit, s.index) for s in again] == [(s.unit, s.index) for s in lc.life_cycle]
        assert [u for u in HXN.stream_HXs_dict[i]] == [s.unit for s in lc.life_cycle]
    heat, cool = actual_loads(HXN)
    assert_allclose([heat, cool], [80. * kW + 9 * 300. * kW, 450. * kW], rtol=1e-9)
    assert HXN.synthesis_info['status'] == 'mer'
    assert_feasible(HXN, 10.)

def test_process_exchangers_reproduce_the_plan():
    units = r002()
    hus = sorted([hx.heat_utilities[0] for hx in units], key=lambda hu: hu.duty)
    info = {}
    hs, cs, utils, *_ = synthesize_network(hus, 10., info=info)
    assert not hs and [hx.ID for hx in cs] == ['HX_3_2_cs', 'HX_3_1_cs', 'HX_3_2_cs_2', 'HX_3_0_cs']
    assert_allclose([hx.Q / kW for hx in cs], [160., 30., 20., 20.], rtol=1e-9)
    for hx in cs:
        # both enthalpy limits are the planned outlets; dT only guards
        assert_allclose([hx.H_lim0, hx.H_lim1], [hx.outs[0].H, hx.outs[1].H], rtol=1e-12)
        assert hx.dT == 10. - 1e-6
    assert not info['deviations'] and not info['dropped'] and not info['repaired']
    assert info['refine_rounds'] == 0
    assert_allclose([info['Q_hot'], info['Q_cold']], [80. * kW, 450. * kW], rtol=1e-12)

def test_internal_pinch_of_condenser_against_boiling_mixture():
    # a mixture boiling at 353-357 K must not take the latent heat of
    # ethanol condensing at 351.57 K (the old synthesizer's -3.68 K cross);
    # every exchanger keeps 5 K on the exact states, inside it too
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('internal_pinch')
    ATM = 101325.
    def rigorous(ID, T, P, phase, T_out, **flow):
        s = bst.Stream(ID + '_in', T=T, P=P, phase=phase, units='kmol/hr', **flow)
        hx = bst.HXutility(ID, ins=s, T=T_out, rigorous=True)
        hx.simulate()
        return hx
    units = [rigorous('C1', 330., ATM, 'l', 365., Water=150., Ethanol=150.),
             rigorous('C2', 340., ATM, 'l', 365., Water=360., Ethanol=40.),
             utility_hx('C3', 300., ATM, 'l', 340., Water=600.),
             utility_hx('H1', 420., 5e5, 'l', 330., Water=1200.),
             utility_hx('H2', 390., ATM, 'g', 340., Ethanol=250.)]
    sys, HXN = simulate_HXN(units, 5.)
    assert_feasible(HXN, 5.)
    heat, cool = actual_loads(HXN)
    hot_target, cold_target = mer_targets(units, 5.)
    assert heat >= hot_target - 1e-9 * total_duty(units)
    status = HXN.synthesis_info['status']
    assert status == ('mer' if heat - hot_target <= 1e-6 * total_duty(units) else 'best_effort')

def test_two_stream_boiler_reaches_its_target():
    # cold water boiled to saturated vapor at 1 atm (100 kmol/hr) against
    # hot water 400 -> 330 K at 5 bar (1000 kmol/hr): the pinch is the
    # boiler's T_sat; the old network crossed by 26.6 K inside an exchanger
    units, T_min_app = test_hxn_targets.case_two_stream()
    sys, HXN = simulate_HXN(units, T_min_app)
    heat, cool = actual_loads(HXN)
    assert_allclose(heat, 2395102., rtol=1e-5)
    assert HXN.synthesis_info['status'] == 'mer'
    assert_feasible(HXN, T_min_app)

@pytest.mark.parametrize('builder', [case_05_boiling_cold, case_10_ten_streams])
def test_non_equilibrium_inlets(builder):
    # case 05 H1: vapor at 420 K and 5 bar, below its dew point; case 10 H4:
    # liquid ethanol at 370 K, above its boiling point
    units, T_min_app = builder()
    sys, HXN = simulate_HXN(units, T_min_app)
    assert_feasible(HXN, T_min_app)
    assert HXN.synthesis_info['status'] == 'mer'

def test_outlet_inside_a_non_equilibrium_jump():
    # the vapor fed below its dew point (420 K, 5 bar) is a flat at 420 K on
    # its curve; the small cold stream takes part of it, so the hot stream
    # leaves its first exchanger inside that jump: its enthalpy limit is left
    # out (a flash there would leave the stream's range) and the cold
    # stream's limit sets the duty
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('jump')
    units = [utility_hx('H1', 420., 5e5, 'g', 330., Water=250.),
             utility_hx('C1', 300., 101325., 'l', 350., Water=100.)]
    sys, HXN = simulate_HXN(units, 5.)
    hx, = HXN.new_HXs
    assert hx.ID == 'HX_1_0_cs' and hx.H_lim0 is None and hx.H_lim1 is not None
    assert_allclose(hx.Q, units[1].outs[0].H - units[1].ins[0].H, rtol=1e-9)
    assert HXN.synthesis_info['status'] == 'mer'
    assert_feasible(HXN, 5.)

def test_exactness_refinement(monkeypatch):
    # with chords 0.5 K off the exact curves, a pinch match inside the
    # curved (ethanol) stream would cross by 0.007 K: the exact check finds
    # it, the knots are refined and the network re-planned
    curves = hxn_synthesis.stream_curves
    monkeypatch.setattr(hxn_synthesis, 'stream_curves',
                        lambda *args, **kwargs: curves(*args, tol_T=0.5, **kwargs))
    units, T_min_app = test_hxn_targets.case_curvature()
    sys, HXN = simulate_HXN(units, T_min_app)
    info = HXN.synthesis_info
    assert info['refine_rounds'] >= 1
    assert_feasible(HXN, T_min_app)
    heat, cool = actual_loads(HXN)
    assert_allclose([heat, cool], [info['Q_hot_plan'], info['Q_cold_plan']],
                    rtol=1e-9, atol=1e-9 * total_duty(units))

def test_loop_converges_exactly():
    # H4-C1 above and below the pinch form a recycle loop
    units = cp_units('kemp', KEMP, 273.15)
    sys, HXN = simulate_HXN(units, 10.)
    assert assert_path_follows_streams(HXN)  # a torn loop
    assert_feasible(HXN, 10., EB=1e-8)
    assert_allclose(actual_loads(HXN), [20. * kW, 60. * kW], rtol=1e-9)
    assert HXN.synthesis_info['status'] == 'mer'

def test_cached_network_with_repeated_pairs():
    units = r002('r002_cache')
    sys, HXN = simulate_HXN(units, 10.)
    fresh = [hx.Q for hx in HXN.new_HXs], actual_loads(HXN)
    # (a) feeds unchanged: the cached network reproduces the fresh one
    HXN.cache_network = True
    HXN_sys = HXN.HXN_sys
    sys.simulate()
    assert HXN.HXN_sys is HXN_sys
    assert_allclose([hx.Q for hx in HXN.new_HXs], fresh[0], rtol=1e-9)
    assert_allclose(actual_loads(HXN), fresh[1], rtol=1e-9)
    # (b) H1 5 % larger: below the pinch the cold streams are served as
    # planned (each keeps its share) and H1 cools the rest; the network
    # stays balanced and feasible
    units[-1].ins[0].F_mass *= 1.05
    sys.simulate()
    assert HXN.HXN_sys is HXN_sys
    assert_allclose([hx.Q for hx in HXN.new_HXs], fresh[0], rtol=1e-9)
    H1_duty = 4. * 170. * kW
    assert_allclose(actual_loads(HXN), [fresh[1][0], fresh[1][1] + 0.05 * H1_duty], rtol=1e-9)
    assert_feasible(HXN, 10.)

@pytest.mark.parametrize('index, factor', [(0, 1.05), (1, 0.95)])
def test_cached_network_caps_the_partner_at_its_outlet(index, factor):
    # C1 5 % larger (H2 5 % smaller): below the pinch C1's share of HX_2_0_cs
    # grows beyond what H2 has left for it; with no limit on H2 the match
    # cooled H2 past its outlet and its "cooler" then heated it with steam
    units = cp_units(f'kemp_cap_{index}', KEMP, 300.)
    sys, HXN = simulate_HXN(units, 10.)
    HXN_sys = HXN.HXN_sys
    HXN.cache_network = True
    units[index].ins[0].F_mol *= factor
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # biosteam's HeatUtility.load_agent names a new 'oxygen_rich_inlet'
        # stream for every fuel (furnace) utility; the network itself replaces
        # nothing in the registry (test_synthesis_registers_no_intermediate_streams)
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='.*<Stream: oxygen_rich_inlet> has been replaced in registry')
        sys.simulate()
    assert HXN.HXN_sys is HXN_sys, 'cached network was not used'
    assert abs(HXN.energy_balance_percent_error) < 1e-6
    tol = 1e-9 * total_duty(units)
    for lc in HXN.stream_life_cycles:
        hx = HXN.original_heat_exchangers[lc.index]
        H_in, H_out = hx.ins[0].H, hx.outs[0].H
        sign = 1. if H_out > H_in else -1.
        for stage in lc.life_cycle:  # every stage moves toward the outlet
            assert sign * (stage.H_out - stage.H_in) >= -tol, stage.unit.ID
            assert sign * (H_out - stage.H_out) >= -tol, stage.unit.ID
    heat, cool = actual_loads(HXN)
    net = sum(hx.heat_utilities[0].unit_duty for hx in units)
    assert_allclose(heat - cool, net, rtol=1e-9)
    for hx in HXN.new_HXs:
        assert internal_approach(hx) >= 10. - APPROACH_TOL, hx.ID

def test_C_flow_from_process_enthalpies():
    # a stream heated with high-pressure steam (heat transfer efficiency
    # 0.85): its heat capacity flow rate is that of the process stream
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('C_flow')
    cold = utility_hx('C1', 400., 20e5, 'l', 480., Water=100.)
    hot = utility_hx('H1', 450., 20e5, 'l', 350., Water=100.)
    hu = cold.heat_utilities[0]
    assert hu.agent.heat_transfer_efficiency < 1.
    result = synthesize_network([hu, hot.heat_utilities[0]], 5.)
    C_flow = result[7]
    dH = cold.outs[0].H - cold.ins[0].H
    assert_allclose(C_flow[0], dH / 80., rtol=1e-12)
    assert not np.isclose(C_flow[0], abs(hu.duty) / 80., rtol=1e-3)

def test_synthesize_network_returns():
    units = r002('r002_returns')
    hus = [hx.heat_utilities[0] for hx in units]  # given order: C1, C2, C3, H1
    result = synthesize_network(hus, 10.)
    assert len(result) == 13
    (hs, cs, utils, hxs, T_in, T_out, pinch_T, C_flow, hus_rearranged,
     streams_inlet, stream_HXs, hot_indices, cold_indices) = result
    assert all(isinstance(hx, bst.HXprocess) for hx in hs + cs)
    assert all(isinstance(hx, bst.HXutility) for hx in utils)
    assert [u.ID for u in utils] == ['Util_3_cs', 'Util_0_hs', 'Util_1_hs', 'Util_2_hs']
    assert hus_rearranged == hus and hxs == units
    for array in (T_in, T_out, pinch_T, C_flow):
        assert isinstance(array, np.ndarray) and array.shape == (4,)
    assert hot_indices == [3] and cold_indices == [0, 1, 2]
    assert sorted(stream_HXs) == [0, 1, 2, 3]
    for i, stages in stream_HXs.items():
        assert isinstance(stages[-1], bst.HXutility) and stages[-1].ID.startswith(f'Util_{i}_')
        assert all(isinstance(u, bst.HXprocess) for u in stages[:-1])
    assert [u.ID for u in stream_HXs[2]] == ['HX_3_2_cs_2', 'HX_3_2_cs', 'Util_2_hs']

def test_avoid_recycle_never_repeats_a_pair():
    units = r002('r002_avoid')
    sys, HXN = simulate_HXN(units, 10., avoid_recycle=True)
    pairs = [frozenset(_stream_ports(hx)) for hx in HXN.new_HXs]
    assert len(pairs) == len(set(pairs))
    heat, cool = actual_loads(HXN)
    assert heat >= 80. * kW * (1 - 1e-9)
    assert HXN.synthesis_info['status'] == 'best_effort'  # r002 needs the repeat
    assert_feasible(HXN, 10.)

def test_Qmin_drops_small_exchangers():
    units = r002('r002_Qmin')
    Qmin = 25. * kW
    sys, HXN = simulate_HXN(units, 10., Qmin=Qmin)
    assert HXN.new_HXs and all(hx.Q >= Qmin for hx in HXN.new_HXs)
    assert HXN.synthesis_info['qmin_dropped']
    assert HXN.synthesis_info['status'] == 'best_effort'
    assert_feasible(HXN, 10.)

def latent_heat_ranges(HXN, index):
    """Stages of a stream on each side of the pinch: (cold-side, hot-side)."""
    stages = HXN.stream_life_cycles[index].life_cycle
    side = lambda s, name: isinstance(s.unit, bst.HXprocess) and s.unit.ID.split('_')[3] == name
    return ([s for s in stages if side(s, 'cs')], [s for s in stages if side(s, 'hs')])

def test_point_loads_at_the_pinch_follow_the_cut():
    ATM = 101325.
    # a boiler at the pinch (cut 'above'): its latent heat is heated above
    # the pinch only
    units, T_min_app = test_hxn_targets.case_two_stream()
    sys, HXN = simulate_HXN(units, T_min_app)
    table = mer_table(units, T_min_app)
    assert _pinch_cut(table) == 'above'
    i = HXN.original_heat_exchangers.index(units[1])
    sat_liquid = units[1].ins[0].copy(); sat_liquid.vle(V=0, P=ATM)
    below, above = latent_heat_ranges(HXN, i)
    tol = 1e-9 * abs(sat_liquid.H)
    assert all(s.s_out.H <= sat_liquid.H + tol for s in below)
    assert above and all(s.s_in.H >= sat_liquid.H - tol for s in above)
    # a condenser at the pinch (cut 'below'): its latent heat is released
    # below the pinch only
    units, T_min_app = test_hxn_targets.case_condenser_at_pinch()
    sys, HXN = simulate_HXN(units, T_min_app)
    assert _pinch_cut(mer_table(units, T_min_app)) == 'below'
    i = HXN.original_heat_exchangers.index(units[0])
    sat_vapor = units[0].ins[0].copy(); sat_vapor.vle(V=1, P=ATM)
    below, above = latent_heat_ranges(HXN, i)
    tol = 1e-9 * abs(sat_vapor.H)
    assert all(s.s_out.H >= sat_vapor.H - tol for s in above)
    assert below and all(s.s_in.H <= sat_vapor.H + tol for s in below)
    assert HXN.synthesis_info['status'] == 'mer'
    assert_feasible(HXN, T_min_app)

def test_non_monotone_stream_is_a_point_load():
    # a superheated water/ethanol liquid (365 K) taken to its dew point
    # (357.4 K) exits colder than it entered: a point load at its outlet
    # temperature; its match takes it at equilibrium at its feed enthalpy
    # (353.07 K, colder than its outlet), from which its planned outlet is
    # a valid enthalpy limit (a flash to it from the 365 K feed would cool
    # a heated stream), and stays feasible
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('non_monotone')
    s = bst.Stream('nm_in', Water=150., Ethanol=150., T=365., P=101325.,
                   phase='l', units='kmol/hr')
    reboiler = bst.HXutility('NM', ins=s, V=1, rigorous=True)
    reboiler.simulate()
    assert reboiler.outs[0].T < reboiler.ins[0].T
    units = [reboiler, utility_hx('H1', 420., 5e5, 'l', 330., Water=1000.)]
    sys, HXN = simulate_HXN(units, 5.)
    hx, = HXN.new_HXs
    assert hx.ins[0].T < reboiler.outs[0].T
    assert hx.H_lim0 is not None and hx.H_lim1 is not None
    assert not HXN.synthesis_info['dropped']
    heat, cool = actual_loads(HXN)
    # the network reaches MER here, so the bound holds with equality and the
    # two sums (utility exchangers vs. problem table) differ by round-off
    assert heat >= mer_targets(units, 5.)[0] - 1e-9 * total_duty(units)
    assert_feasible(HXN, 5.)

def point_load_units(kind):
    """A point-load stream (port 1 of its only match) whose whole duty a
    process stream can serve: 'cold', the superheated liquid of
    test_non_monotone_stream_is_a_point_load boiled to its dew point, colder
    than its feed; 'hot', water vapor fed below its dew point (420 K, 5 bar)
    and cooled 1 K, whose quenched outlet (424.98 K) is hotter than its
    feed."""
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('point_load_' + kind)
    if kind == 'cold':
        s = bst.Stream('nm_in', Water=150., Ethanol=150., T=365., P=101325.,
                       phase='l', units='kmol/hr')
        point = bst.HXutility('NM', ins=s, V=1, rigorous=True)
        point.simulate()
        return [point, utility_hx('H1', 420., 5e5, 'l', 330., Water=5000.)]
    s = bst.Stream('nm_in', Water=100., T=420., P=5e5, phase='g', units='kmol/hr')
    point = bst.HXutility('NM', ins=s, T=419., rigorous=False)
    point.simulate()
    quenched = point.outs[0].copy()
    quenched.vle(H=quenched.H, P=quenched.P)
    assert quenched.T > point.ins[0].T
    return [point, utility_hx('C1', 300., 101325., 'l', 350., Water=1000.)]

@pytest.mark.parametrize('kind', ['cold', 'hot'])
def test_point_load_whole_duty_is_matched(kind):
    # the match takes the point load's whole duty (port 1), with both
    # limits: the point load enters at equilibrium at its feed enthalpy, on
    # the plan's side of its outlet temperature, where its planned outlet is
    # a valid HXprocess limit ('cold': 353.07 K, colder than its outlet,
    # not the 365 K feed, from which the flash to its outlet failed and the
    # match was dropped, losing MER; 'hot': at its outlet temperature,
    # 424.98 K, not 420 K)
    units = point_load_units(kind)
    sys, HXN = simulate_HXN(units, 5.)
    info = HXN.synthesis_info
    hx, = HXN.new_HXs
    point = units[0]
    duty = abs(point.outs[0].H - point.ins[0].H)
    assert hx.H_lim0 is not None and hx.H_lim1 is not None
    assert info['point_loads'] == [HXN.original_heat_exchangers.index(point)]
    if kind == 'cold':
        assert hx.ins[1].T < point.outs[0].T < point.ins[0].T
    else:
        assert hx.ins[1].T > point.ins[0].T + 4.
    assert abs(hx.ins[1].H - point.ins[0].H) <= 1e-9 * duty
    assert_allclose(hx.Q, abs(point.outs[0].H - point.ins[0].H), rtol=1e-9)
    assert not info['dropped'] and not info['deviations']
    assert info['status'] == 'mer'
    total = total_duty(units)
    assert_allclose(actual_loads(HXN), mer_targets(units, 5.), atol=1e-6 * total)
    assert_feasible(HXN, 5.)

def capped_point_load_units(kind):
    """A mixture point-load stream (index 0 of the returned units) whose
    real feed temperature, on the wrong side of its outlet temperature,
    would cap its partner in HXprocess short of the plan (by feed -/+ dT),
    and whose equilibrium state at its feed enthalpy is not at its outlet
    temperature, plus that partner. 'ideal': 91 wt% ethanol vapor at its
    dew point (364.02 K, 1.62 bar; the condenser of the oilcane
    biorefinery's D302), 10 % condensed; under `force_ideal_thermo` its
    ideal dew point is higher, so it is fed as a vapor below its dew point
    and its quenched outlet (369.84 K) and equilibrium feed (370.26 K) are
    hotter than its feed. 'superheated': the water/ethanol liquid at 365 K
    of test_non_monotone_stream_is_a_point_load boiled to its dew point
    (357.44 K; equilibrium feed 353.07 K)."""
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('capped_point_load_' + kind)
    if kind == 'ideal':
        P = 162120.
        s = bst.Stream('D_in', Ethanol=19.69, Water=5.16, P=P, units='kmol/hr')
        s.vle(V=1, P=P)
        point = bst.HXutility('D', ins=s, V=0.9, rigorous=True)
        point.simulate()
        T = s.T
        partner = utility_hx('C1', T - 8., 101325., 'l', T - 1.5, Water=100.)
    else:
        s = bst.Stream('nm_in', Water=150., Ethanol=150., T=365., P=101325.,
                       phase='l', units='kmol/hr')
        point = bst.HXutility('NM', ins=s, V=1, rigorous=True)
        point.simulate()
        partner = utility_hx('H1', 380., 5e5, 'l', 363., Water=300.)
    return [point, partner]

@pytest.mark.parametrize('kind', ['ideal', 'superheated'])
def test_point_load_enters_at_equilibrium_on_the_plan_side(kind):
    # the plan puts a point load's whole duty at its outlet temperature,
    # beyond its real feed temperature; the match kept the real feed unless
    # its equilibrium state at the feed enthalpy lay exactly at the outlet
    # temperature (a pure component), so HXprocess capped the partner at
    # the feed temperature -/+ dT: 'ideal' delivered 22.7 of 49.2 MJ/hr
    # (the oilcane O1 HX_8_5_cs under force_ideal_thermo: 1.38 of 3.38
    # MW), 'superheated' 228.0 of 387.2 MJ/hr. The point load now enters at
    # equilibrium at its feed enthalpy, on the plan's side of its outlet
    # temperature, and the network reaches MER as planned.
    units = capped_point_load_units(kind)
    ideal = kind == 'ideal'
    sys, HXN = simulate_HXN(units, 5., force_ideal_thermo=ideal)
    info = HXN.synthesis_info
    point = units[0]
    index = HXN.original_heat_exchangers.index(point)
    assert info['point_loads'] == [index]
    hx, = HXN.new_HXs
    s_in = hx.ins[_stream_ports(hx).index(index)]
    T_out = HXN.outlet_Ts[index]  # the (ideal) quenched outlet
    if ideal:  # cooled, fed colder than its outlet: enters hotter than it
        assert point.ins[0].T < T_out - 5. and s_in.T > T_out
    else:  # heated, fed hotter than its outlet: enters colder than it
        assert point.ins[0].T > T_out + 7. and s_in.T < T_out
    feed = point.ins[0].copy(thermo=s_in.thermo)
    assert_allclose(s_in.H, feed.H, rtol=1e-12)
    assert hx.H_lim0 is not None and hx.H_lim1 is not None
    assert not info['deviations'] and not info['dropped'] and not info['repaired']
    assert info['status'] == 'mer'
    targets = [info['Q_hot_target'], info['Q_cold_target']]
    total = total_duty(units)
    assert_allclose([info['Q_hot'], info['Q_cold']], targets, atol=1e-6 * total)
    assert_allclose(actual_loads(HXN), targets, atol=1e-6 * total)
    assert_feasible(HXN, 5.)

def test_synthesis_registers_no_intermediate_streams():
    # the curves and the synthesis copy streams hundreds of times; a plain
    # ``stream.copy()`` takes its ID from the source line (thermosteam's ID
    # magic), so a multi-phase copy on a line that assigns no plain
    # variable registered as '-', replacing the previous one with a
    # RuntimeWarning (128 per synthesis of this system, 282 on the oilcane
    # biorefinery O1, against 2 before the curves), and a single-phase copy
    # took a registry ticket. Every internal copy is now unregistered.
    sys, HXN, feed = build_system()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sys.simulate()
    replaced = [str(w.message) for w in caught if 'replaced in registry' in str(w.message)]
    assert not replaced, replaced[:3]
    assert HXN.synthesis_info['status'] == 'mer'
    # the problem table (curves, flashes) registers nothing at all
    streams = pinch_streams(HXN.original_heat_utils)
    registered = lambda: {ID: id(s) for ID, s in bst.main_flowsheet.stream.data.items()}
    before = registered()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        problem_table(*streams, 5.)
    assert not [w for w in caught if 'registry' in str(w.message)]
    assert registered() == before

@pytest.mark.parametrize('T_min_app', [5., 30.])
def test_reboiler_fed_above_saturation_against_steam(T_min_app):
    # a column reboiler: water fed as a liquid at 396 K and half boiled at
    # 1 atm leaves at 373.12 K, a point load colder than its feed, served by
    # condensing saturated steam at 3 bar (406.67 K, also a point load). The
    # match was lost, and MER with it, by an HXprocess limit at the
    # reboiler's planned outlet, on the wrong side of its feed temperature,
    # and, for T_min_app above 10.67 K, by HXprocess comparing the 396 K
    # feed with the steam: the reboiler now enters the exchanger at
    # equilibrium at its feed enthalpy (373.12 K), as the plan sees it.
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet(f'reboiler_{T_min_app:g}')
    feed = bst.Stream('R_in', Water=100., T=396., P=101325., phase='l',
                      units='kmol/hr')
    reboiler = bst.HXutility('R', ins=feed, V=0.5, rigorous=True)
    steam = bst.Stream('S_in', Water=150., P=3e5, units='kmol/hr')
    steam.vle(V=1, P=3e5)
    condenser = bst.HXutility('S', ins=steam, V=0, rigorous=True)
    for unit in (reboiler, condenser): unit.simulate()
    assert reboiler.outs[0].T < reboiler.ins[0].T - 20.
    units = [reboiler, condenser]
    sys, HXN = simulate_HXN(units, T_min_app)
    info = HXN.synthesis_info
    assert sorted(info['point_loads']) == [0, 1]
    hx, = HXN.new_HXs
    cold_in = min(hx.ins, key=lambda s: s.T)
    assert abs(cold_in.T - reboiler.outs[0].T) < 1e-6
    assert_allclose(cold_in.H, feed.H, rtol=1e-12)
    duty = reboiler.outs[0].H - reboiler.ins[0].H
    assert_allclose(hx.Q, duty, rtol=1e-9)  # the reboiler's whole duty
    assert not info['dropped'] and not info['deviations']
    assert info['status'] == 'mer'
    assert_allclose(actual_loads(HXN), mer_targets(units, T_min_app),
                    atol=1e-9 * total_duty(units))
    assert_feasible(HXN, T_min_app)

@pytest.mark.parametrize('T_cold_out', [354.87, 354.9, 354.93])
def test_liquid_fed_above_its_bubble_point_keeps_the_approach(T_cold_out):
    # the hot liquid (368 K) is at equilibrium at 359.84 K at its feed
    # enthalpy; its curve used to reach it at 359.95 K, so the network
    # crossed by up to 0.09 K at the hot end, undetected, on too low targets
    bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    bst.main_flowsheet.set_flowsheet('clipped_glide')
    s = bst.Stream('H1_in', Water=900., Ethanol=100., T=368., P=101325.,
                   phase='l', units='kmol/hr')
    hot = bst.HXutility('H1', ins=s, T=320., rigorous=True)
    hot.simulate()
    units = [hot, utility_hx('C1', 300., 5e5, 'l', T_cold_out, Water=400.)]
    sys, HXN = simulate_HXN(units, 5.)
    assert_feasible(HXN, 5.)
    heat, cool = actual_loads(HXN)
    hot_target, cold_target = mer_targets(units, 5.)
    # one match: C1 is heated to the hot stream's equilibrium feed
    # temperature less T_min_app, the rest by utility
    feed = s.copy(); feed.vle(H=s.H, P=s.P)
    C1_in = units[1].ins[0]
    top = C1_in.copy(); top.T = feed.T - 5.
    assert_allclose(hot_target, units[1].outs[0].H - top.H, rtol=1e-6)
    assert_allclose([heat, cool], [hot_target, cold_target], atol=1e-6 * total_duty(units))
    assert HXN.synthesis_info['status'] == 'mer'

def test_ideal_thermo_doctest_system_reaches_mer():
    # with ideal thermodynamics, hot stream 4 (338.00 K fed, quenched outlet
    # 338.32 K) is a point load that one match serves completely
    sys, HXN, feed = build_system()
    HXN.force_ideal_thermo = True
    sys.simulate()
    info = HXN.synthesis_info
    assert not info['dropped']
    assert abs(info['penalty']) <= 1e-9 * info['Q_hot_target']
    assert info['status'] == 'mer'

def mer_table(units, T_min_app):
    hus = heat_utilities(units)
    return problem_table(*pinch_streams(hus), T_min_app)

@pytest.mark.parametrize('rows, dT, targets', [
    # threshold, no hot utility
    ([('H1', 200., 40., 2.), ('C1', 50., 150., 1.)], 10., (0., 220.)),
    # threshold, no cold utility
    ([('H1', 200., 100., 1.), ('C1', 50., 250., 1.)], 10., (100., 0.)),
    # a pinch at 205 and zeros of the cascade at 200, 120, 100 and 20
    ([('C1', 20., 100., 1.), ('H1', 110., 20., 1.), ('C2', 120., 200., 1.),
      ('H2', 210., 130., 1.), ('C3', 205., 215., 1.)], 10., (10., 10.)),
])
def test_thresholds_and_multiple_pinches(rows, dT, targets):
    units = cp_units('threshold', rows, 273.15)
    table = mer_table(units, dT)
    assert_allclose([table.hot_util_load / kW, table.cold_util_load / kW], targets,
                    atol=1e-9 * total_duty(units) / kW)
    sys, HXN = simulate_HXN(units, dT)
    assert_allclose(actual_loads(HXN), [Q * kW for Q in targets],
                    atol=1e-9 * total_duty(units))
    assert HXN.synthesis_info['status'] == 'mer'
    assert_feasible(HXN, dT)

def test_synthesis_info():
    units = r002('r002_info')
    sys, HXN = simulate_HXN(units, 10.)
    info = HXN.synthesis_info
    assert info['status'] == 'mer' and info['penalty'] <= 1e-9 * total_duty(units)
    assert info['sides']['below']['status'] == 'mer' and info['sides']['below']['proof'] is None
    assert_allclose([info['Q_hot_target'], info['Q_cold_target']], [80. * kW, 450. * kW])
    # two hot streams reach the pinch against one cold stream above it: the
    # number rule proves that MER needs a split
    units = cp_units('split', [('H1', 150., 100., 1.), ('H2', 150., 100., 1.),
                               ('C1', 90., 140., 3.)], 273.15)
    sys, HXN = simulate_HXN(units, 10.)
    info = HXN.synthesis_info
    assert info['status'] == 'best_effort' and info['penalty'] > 0.
    proof = info['sides']['above']['proof']
    assert proof['side'] == 'above' and len(proof['musts']) == 2 and len(proof['flexes']) == 1
    heat, cool = actual_loads(HXN)
    assert_allclose([heat - cool], [info['Q_hot_target'] - info['Q_cold_target']], rtol=1e-9)
    assert heat >= info['Q_hot_target'] and heat == pytest.approx(info['Q_hot'], rel=1e-9)
    assert_feasible(HXN, 10.)

def test_docstring_references_are_cited():
    # the documentation build fails on a reference ('.. [label]') that its
    # docstring never cites ('[label]_')
    import inspect, re
    import hensmith._curves, hensmith._planner
    modules = (hensmith._heat_exchanger_network, hensmith.hxn_synthesis,
               hensmith._planner, hensmith._curves)
    def docstrings(obj, seen):
        if id(obj) in seen: return
        seen.add(id(obj))
        if obj.__doc__: yield obj
        for name, member in vars(obj).items():
            if isinstance(member, (staticmethod, classmethod)): member = member.__func__
            if isinstance(member, property): member = member.fget
            if ((inspect.isfunction(member) or inspect.isclass(member))
                    and member.__module__ == owner.__name__):
                yield from docstrings(member, seen)
    checked = 0
    for owner in modules:
        for obj in docstrings(owner, set()):
            doc = obj.__doc__
            for label in re.findall(r'^\s*\.\. \[([^\]]+)\]', doc, re.M):
                checked += 1
                assert f'[{label}]_' in doc, f'{obj.__name__}: [{label}] not cited'
    assert checked >= 9

# --- pinch diagram -----------------------------------------------------------

class _FakeStage:
    def __init__(self, unit): self.unit = unit

class _FakeLifeCycle:
    def __init__(self, units, cold=True):
        self.cold = cold
        self.life_cycle = [_FakeStage(u) for u in units]

def test_pinch_diagram_column_order_follows_stream_direction():
    from hensmith.hxn_synthesis import _order_exchanger_columns
    h1, h2, h3 = 'h1', 'h2', 'h3'
    # stream A visits h2 then h1; stream B visits h1 then h3 -> h2, h1, h3
    cycles = [_FakeLifeCycle([h2, h1]), _FakeLifeCycle([h1, h3])]
    assert _order_exchanger_columns([h1, h2, h3], cycles) == [h2, h1, h3]
    # exchangers not in the requested subset are ignored, order is stable
    assert _order_exchanger_columns([h3, h1], cycles) == [h1, h3]
    # a hot stream flows right to left, so its stage order is reversed:
    # hot stream visits h1 then h3 -> h3 left of h1
    cycles = [_FakeLifeCycle([h2, h1]), _FakeLifeCycle([h1, h3], cold=False)]
    assert _order_exchanger_columns([h1, h2, h3], cycles) == [h2, h3, h1]
    # contradictory constraints (a cycle) fall back to the given order
    cycles = [_FakeLifeCycle([h1, h2]), _FakeLifeCycle([h2, h1])]
    assert _order_exchanger_columns([h2, h1], cycles) == [h2, h1]

def _gid_artists(ax, prefix):
    return [a for a in ax.findobj() if (a.get_gid() or '').startswith(prefix)]

def test_pinch_diagram_doctest_system():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys, HXN, feed = build_system()
    sys.simulate()
    assert HXN.new_HXs_hot_side + HXN.new_HXs_cold_side == HXN.new_HXs
    fig, ax = HXN.plot_pinch_diagram()
    try:
        # one connector per process exchanger
        connectors = _gid_artists(ax, 'HX:')
        assert {a.get_gid() for a in connectors} == {'HX:' + hx.ID for hx in HXN.new_HXs}
        # one utility marker per utility exchanger with a duty above Qmin
        utils = _gid_artists(ax, 'Util:')
        expected = {'Util:' + hx.ID for hx in HXN.new_HX_utils
                    if abs(hx.outs[0].H - hx.ins[0].H) > HXN.Qmin}
        assert {a.get_gid() for a in utils} == expected
        # one row per stream, with inlet temperatures in degC
        texts = {t.get_text() for t in ax.texts}
        for T in HXN.inlet_Ts: assert f'{T - 273.15:.1f}' in texts
        for T in HXN.outlet_Ts: assert f'{T - 273.15:.1f}' in texts
        for i in range(len(HXN.inlet_Ts)): assert str(i) in texts
    finally:
        plt.close(fig)

def test_pinch_diagram_stream_labels():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from hensmith.hxn_synthesis import _auxiliary_name, _stream_label
    sys, HXN, feed = build_system()
    sys.simulate()
    D1 = bst.main_flowsheet.unit.D0
    D1_H1 = bst.main_flowsheet.unit.D0_H1
    assert _auxiliary_name(D1.condenser) == 'condenser'
    assert _auxiliary_name(D1_H1) is None
    # label composition
    assert _stream_label(D1.condenser, True, True, False) == 'D0 - condenser'
    assert _stream_label(D1.condenser, True, False, False) == 'D0'
    assert _stream_label(D1.condenser, False, True, False) == 'condenser'
    assert _stream_label(D1_H1, True, True, True) == 'D0_H1 (' + D1_H1.ins[0].ID + ')'
    assert _stream_label(D1_H1, False, False, True) == D1_H1.ins[0].ID
    assert _stream_label(D1_H1, False, False, False) == ''
    # an unnamed inlet adds nothing
    assert _stream_label(D1.reboiler, True, True, True) == 'D0 - reboiler'
    # every stream gets a label on the figure; toggles remove them
    fig, ax = HXN.plot_pinch_diagram()
    try:
        labels = {a.get_gid(): a.get_text() for a in _gid_artists(ax, 'Label:')}
        hxs = HXN.original_heat_exchangers
        assert labels == {
            f'Label:{i}': _stream_label(hx, True, True, True)
            for i, hx in enumerate(hxs)
        }
        assert any(text.startswith('D0 - condenser') for text in labels.values())
        assert any(text == 'F1 - heat_exchanger (feed_flash)' for text in labels.values())
    finally:
        plt.close(fig)
    fig, ax = HXN.plot_pinch_diagram(show_units=False, show_auxiliary_units=False,
                                     show_stream_IDs=False)
    try:
        assert not _gid_artists(ax, 'Label:')
    finally:
        plt.close(fig)

def test_pinch_diagram_requires_simulation():
    sys, HXN, feed = build_system()
    with pytest.raises(RuntimeError, match='simulate'):
        HXN.plot_pinch_diagram()

def test_pinch_diagram_legend():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys, HXN, feed = build_system()
    sys.simulate()
    fig, ax = HXN.plot_pinch_diagram()
    try:
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels == ['Cold stream', 'Hot stream', 'Process heat exchange',
                          'Hot utility', 'Cold utility', 'Pinch']
        # utility markers are colored by utility type, consistent with the legend
        handles = dict(zip(labels, legend.legend_handles))
        hot_util_color = handles['Hot utility'].get_markeredgecolor()
        cold_util_color = handles['Cold utility'].get_markeredgecolor()
        assert hot_util_color != cold_util_color
        heaters = {hx.ID for hx in HXN.new_HX_utils if hx.outs[0].H > hx.ins[0].H}
        for artist in _gid_artists(ax, 'Util:'):
            heater = artist.get_gid()[len('Util:'):] in heaters
            expected = hot_util_color if heater else cold_util_color
            assert artist.get_markeredgecolor() == expected, artist.get_gid()
    finally:
        plt.close(fig)
    fig, ax = HXN.plot_pinch_diagram(show_legend=False)
    try:
        assert ax.get_legend() is None
    finally:
        plt.close(fig)

if __name__ == '__main__':
    pytest.main([__file__, '-q', '-p', 'no:cacheprovider'])
