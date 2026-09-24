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

# ---------------------------------------------------------------------------
# Stream splitting: exact checks and exchangers on branches (flow fractions)
# ---------------------------------------------------------------------------

from hensmith._curves import StreamCurve, GLIDE, _copy
from hensmith._planner import Exchanger, Plan, plan_network
from hensmith import _splitting
from test_hxn_planner import records_case
from hxn_mer_cases import SPLIT as MER_SPLIT
import test_hxn_mer

#: Branch fractions of the hot and the cold stream in the branch tests (not
#: dyadic, so that a lost or a doubled fraction shows).
FH, FC = 0.37, 0.61

def branch_streams(kind):
    """Inlets and quenched outlets ``[h_in, h_out, c_in, c_out]`` of a hot
    and a cold stream: constant CP (400 -> 300 K, 2 kW/K against 330 ->
    390 K, 3 kW/K), or hot water (rtB07's H1, 400 -> 300 K at 5 bar)
    against rtB07's water/methanol boiler C3 (330 -> 372 K at 1 atm),
    whose curve has a glide."""
    if kind == 'constant_cp':
        units = cp_units('branch_cp', [('H1', 400., 300., 2.),
                                       ('C1', 330., 390., 3.)])
    else:
        bst.settings.set_thermo(['Water', 'Methanol'], cache=True)
        bst.main_flowsheet.set_flowsheet('branch_glide')
        units = []
        for ID, T, P, T_out, rigorous, flow in (
                ('H1', 400., 5e5, 300., False, dict(Water=30.)),
                ('C3', 330., 101325., 372., True,
                 dict(Water=12., Methanol=3.))):
            s = bst.Stream(ID + '_in', T=T, P=P, phase='l', units='kmol/hr',
                           **flow)
            hx = bst.HXutility(ID, ins=s, T=T_out, rigorous=rigorous)
            hx.simulate()
            units.append(hx)
    streams = []
    for hx in units:
        s_in, s_out = _copy(hx.ins[0]), _copy(hx.outs[0])
        s_out.vle(H=s_out.H, P=s_out.P)
        streams += [s_in, s_out]
    return streams

def branch_curves(kind, fh=1., fc=1.):
    """Curves and knots (the curves' breakpoints) of `branch_streams`, the
    hot stream at `fh` of its flow and the cold one at `fc` of its flow."""
    h_in, h_out, c_in, c_out = branch_streams(kind)
    for s in (h_in, h_out): s.scale(fh)
    for s in (c_in, c_out): s.scale(fc)
    curves = [StreamCurve(h_in, h_out, True), StreamCurve(c_in, c_out, False)]
    return curves, [(c.T, c.H - c.H_lo) for c in curves]

def branch_exchanger(curves):
    """A branch exchanger on the full-flow `curves` (parent-equivalent
    enthalpies): the hot branch (FH) enters at the hot stream's inlet, the
    cold branch (FC) at 10 % of the cold stream's duty, and the duty is 90 %
    of what the tighter branch has left."""
    ch, cc = curves
    H_hot_in = ch.H_hi - ch.H_lo
    H_cold_in = 0.1 * (cc.H_hi - cc.H_lo)
    Q = 0.9 * min(FH * H_hot_in, FC * (cc.H_hi - cc.H_lo - H_cold_in))
    return H_hot_in, H_cold_in, Q

@pytest.mark.parametrize('kind', ['constant_cp', 'glide'])
def test_branch_exchanger_approach_matches_scaled_curve(kind):
    # a branch at fraction f runs its parent's curve with f times the
    # enthalpy: the check on the parent curves with the fractions equals a
    # direct computation on the curves of the scaled streams, and its
    # violating states are parent-equivalent (what `_refine_knots` takes)
    curves, knots = branch_curves(kind)
    scaled, scaled_knots = branch_curves(kind, FH, FC)
    ch, cc = curves
    chf, ccf = scaled
    if kind == 'glide': assert GLIDE in cc.kinds
    H_hot_in, H_cold_in, Q = branch_exchanger(curves)
    approach = hxn_synthesis._exchanger_approach
    inf = float('inf')
    worst, points = approach(curves, knots, 0, 1, H_hot_in, H_cold_in, Q,
                             inf, FH, FC)
    worst_f, _ = approach(scaled, scaled_knots, 0, 1, chf.H_hi - chf.H_lo,
                          FC * H_cold_in, Q, inf)
    assert abs(worst - worst_f) <= 1e-9
    assert len(points) > 2
    assert worst == min(T_hot - T_cold for T_hot, _, T_cold, _ in points)
    H_cold_out = H_cold_in + Q / FC
    for T_hot, H_hot, T_cold, H_cold in points:
        q = (H_hot_in - H_hot) * FH  # the duty done, the same on both
        assert -1e-12 * Q <= q <= Q * (1. + 1e-12)
        assert abs((H_cold_out - H_cold) * FC - q) <= 1e-12 * Q
        assert abs(chf.T_exact(chf.H_lo + FH * H_hot, 'low') - T_hot) <= 1e-9
        assert abs(ccf.T_exact(ccf.H_lo + FC * H_cold, 'high')
                   - T_cold) <= 1e-9
    # every knot inside the branches is a position of the check (between
    # positions both knot curves are linear)
    H_hot_out = H_hot_in - Q / FH
    for (_, H), at, lo, hi in (
            (knots[0], [p[1] for p in points], H_hot_out, H_hot_in),
            (knots[1], [p[3] for p in points], H_cold_in, H_cold_out)):
        for x in H[(H > lo) & (H < hi)]:
            assert np.abs(np.array(at) - x).min() <= 1e-12 * H[-1]
    # a violation of 0.5 K is found where it is (the knots screen the rest)
    assert approach(curves, knots, 0, 1, H_hot_in, H_cold_in, Q,
                    worst + 0.5, FH, FC)[0] == worst
    # `_exact_approach` checks a plan's exchangers on their fractions
    e = Exchanger()
    e.hot, e.cold, e.hot_frac, e.cold_frac = 0, 1, FH, FC
    plan = Plan()
    plan.exchangers = [e]
    ends = {(0, 0): (H_hot_in, H_hot_out), (0, 1): (H_cold_in, H_cold_out)}
    assert hxn_synthesis._exact_approach(plan, {0: Q}, ends, curves, knots,
                                         inf) == (worst, {
        0: [(T_hot, H_hot) for T_hot, H_hot, _, _ in points],
        1: [(T_cold, H_cold) for _, _, T_cold, H_cold in points]}, [0])
    # fractions of 1 are the unsplit check, bit for bit
    for T_min_app in (inf, worst + 0.5):
        assert (approach(curves, knots, 0, 1, H_hot_in, H_cold_in, Q,
                         T_min_app, 1., 1.)
                == approach(curves, knots, 0, 1, H_hot_in, H_cold_in, Q,
                            T_min_app))

@pytest.mark.parametrize('kind', ['constant_cp', 'glide'])
def test_shrink_with_fractions_is_monotone(kind):
    # with both branch inlets fixed, a smaller duty moves the cold branch's
    # outlet toward its inlet: the exact approach never falls as the duty
    # falls, the feasible duties are [0, Q'] and `_shrink` finds Q' (as on
    # the curves of the scaled streams)
    curves, knots = branch_curves(kind)
    scaled, scaled_knots = branch_curves(kind, FH, FC)
    H_hot_in, H_cold_in, Q = branch_exchanger(curves)
    approach, shrink = hxn_synthesis._exchanger_approach, hxn_synthesis._shrink
    inf = float('inf')
    xs = np.linspace(0.05, 1., 20) * Q
    dTs = [approach(curves, knots, 0, 1, H_hot_in, H_cold_in, x, inf,
                    FH, FC)[0] for x in xs]
    assert all(b <= a + 1e-9 for a, b in zip(dTs, dTs[1:]))
    T_min_app = 0.5 * (dTs[0] + dTs[-1])  # Q itself violates
    Q_ok = shrink(curves, knots, 0, 1, H_hot_in, H_cold_in, Q, T_min_app,
                  FH, FC)
    assert 0. < Q_ok < Q
    for x in xs:
        violates = bool(approach(curves, knots, 0, 1, H_hot_in, H_cold_in,
                                 x, T_min_app, FH, FC)[1])
        assert violates == (x > Q_ok), x
    limit = T_min_app - hxn_synthesis._APPROACH_TOL
    at = approach(curves, knots, 0, 1, H_hot_in, H_cold_in, Q_ok, inf,
                  FH, FC)[0]
    assert limit <= at <= limit + 1e-6
    chf = scaled[0]
    Q_f = shrink(scaled, scaled_knots, 0, 1, chf.H_hi - chf.H_lo,
                 FC * H_cold_in, Q, T_min_app)
    assert abs(Q_ok - Q_f) <= 1e-8 * Q
    if kind == 'constant_cp':
        # closed form: the approach is smallest at the cold end, where the
        # hot branch leaves 400 K - Q / (FH 2 kW/K) against the cold
        # branch's inlet, 330 K + 10 % of 60 K
        assert_allclose(Q_ok, FH * 2. * kW * (400. - 336. - limit),
                        rtol=1e-8)
    assert (shrink(curves, knots, 0, 1, H_hot_in, H_cold_in, Q, T_min_app,
                   1., 1.)
            == shrink(curves, knots, 0, 1, H_hot_in, H_cold_in, Q,
                      T_min_app))

def records_plan():
    """The hand-built split plan of `test_hxn_planner.records_case` (H1,
    400 -> 200 with CP 1, splits into halves that serve C1 and C2, re-joins
    at 100 and serves C2 and C1 on its trunk), with its knot enthalpies."""
    sides, plans, curves = records_case()
    (recs, _, dropped, stages, _, _, splits,
     paths) = _splitting._split_records(sides, plans, curves, 3, 0.,
                                         sides['above'].tolQ)
    assert not dropped
    plan = Plan()
    plan.exchangers, plan.stages, plan.paths, plan.splits = (
        recs, stages, paths, splits)
    knots = [(np.array([200., 400.]), np.array([0., 200.])),
             (np.array([20., 170.]), np.array([0., 150.])),
             (np.array([20., 170.]), np.array([0., 150.]))]
    return plan, knots, [True, False, False]

def test_walk_split_paths():
    plan, knots, is_hot = records_plan()
    walk, nodes = hxn_synthesis._walk, hxn_synthesis._split_nodes
    b0, b1, t1, t0 = range(4)
    duties = {n: e.Q for n, e in enumerate(plan.exchangers)}
    ends, last, pair_index = walk(plan, duties, knots, is_hot)
    # the planner's own walk: parent-equivalent enthalpies on the branches
    for n, e in enumerate(plan.exchangers):
        assert ends[n, e.hot] == (e.H_hot_in, e.H_hot_out)
        assert ends[n, e.cold] == (e.H_cold_in, e.H_cold_out)
    assert ends[b0, 0] == (200., 80.) and ends[b1, 0] == (200., 120.)
    assert ends[t1, 0] == (100., 40.) and ends[t0, 0] == (40., 0.)
    assert last == [0., 100., 100.]
    assert pair_index == {b0: 1, t0: 2, b1: 1, t1: 2}
    assert nodes(plan, duties, ends) == ({0: (b0, b1), 1: (), 2: ()},
                                         [(200., [80., 120.], 100.)])
    # a dropped branch exchanger: its branch bypasses, the stream re-joins
    # at H_split - 60 and C2 starts on its trunk exchanger
    del duties[b1]
    ends, last, pair_index = walk(plan, duties, knots, is_hot)
    assert ends == {(b0, 0): (200., 80.), (b0, 1): (0., 60.),
                    (t1, 0): (140., 80.), (t1, 2): (0., 60.),
                    (t0, 0): (80., 40.), (t0, 1): (60., 100.)}
    assert last == [40., 100., 60.]
    assert pair_index == {b0: 1, t0: 2, t1: 1}
    assert nodes(plan, duties, ends) == ({0: (b0,), 1: (), 2: ()},
                                         [(200., [80., 200.], 140.)])
    # a shrunk branch exchanger: its branch ends earlier (by Q / f), the
    # mix and the later stages move toward the inlet by the duty
    duties = {n: e.Q for n, e in enumerate(plan.exchangers)}
    duties[b0] = 30.
    ends, last, _ = walk(plan, duties, knots, is_hot)
    assert ends[b0, 0] == (200., 140.) and ends[b0, 1] == (0., 30.)
    assert ends[t1, 0] == (130., 70.) and ends[t0, 0] == (70., 30.)
    assert last == [30., 70., 100.]
    assert nodes(plan, duties, ends)[1] == [(200., [140., 120.], 130.)]
    # a split whose branches were all dropped is passed as a trunk (it is
    # not realized): no first nodes, no split ends
    duties = {t1: 60., t0: 40.}
    ends, last, _ = walk(plan, duties, knots, is_hot)
    assert ends[t1, 0] == (200., 140.) and ends[t0, 0] == (140., 100.)
    assert last == [100., 40., 60.]
    assert nodes(plan, duties, ends) == ({0: (), 1: (), 2: ()}, [None])
    # without splits, the walk is the unsplit one on `stages`
    plan.splits, plan.paths = [], None
    assert walk(plan, duties, knots, is_hot)[:2] == (ends, last)

def planned_split(name):
    """Split plan of corpus case `name` on the synthesizer's round-0 grid
    knots, with what the synthesis hands to its exact checks."""
    case = next(c for c in MER_SPLIT if c['name'] == name)
    units, dT = test_hxn_mer._build(case)
    hus = sorted([hx.heat_utilities[0] for hx in units],
                 key=lambda hu: hu.duty)
    r = hxn_synthesis._pinch_analysis(hus, dT)
    hxs, hot_indices, streams_inlet, curves, grid = (r[5], r[6], r[9],
                                                     r[13], r[14])
    is_hot = [i in hot_indices for i in range(len(hxs))]
    knots = hxn_synthesis._grid_knots(curves, grid)
    plan = plan_network(knots, is_hot, dT, stream_splitting=True)
    assert plan.status == 'mer' and plan.splits
    return plan, curves, knots, is_hot, streams_inlet, dT

@pytest.mark.parametrize('name', ['smith2005_ex18_2_split', 'rtB05_above_2h1c'])
def test_realize_split_branch_ports(name):
    # every branch exchanger is an HXprocess on f of its stream's flow: its
    # inlet is the parent's state at the branch's inlet enthalpy (the real
    # inlet where the stream splits at its inlet) scaled by f, and its
    # enthalpy limit is f times the parent's at the planned outlet
    plan, curves, knots, is_hot, streams_inlet, dT = planned_split(name)
    span = [c.H_hi - c.H_lo for c in curves]
    duties = {n: e.Q for n, e in enumerate(plan.exchangers)}
    ends, last, _ = hxn_synthesis._walk(plan, duties, knots, is_hot)
    for n, e in enumerate(plan.exchangers):
        assert_allclose(ends[n, e.hot], (e.H_hot_in, e.H_hot_out),
                        rtol=0, atol=1e-12 * span[e.hot])
        assert_allclose(ends[n, e.cold], (e.H_cold_in, e.H_cold_out),
                        rtol=0, atol=1e-12 * span[e.cold])
    assert_allclose(last, [u if hot else D - u for u, D, hot
                           in zip(plan.utility, span, is_hot)],
                    rtol=0, atol=1e-12 * sum(span))
    first_nodes, split_ends = hxn_synthesis._split_nodes(plan, duties, ends)
    assert len(split_ends) == len(plan.splits)
    for s, (H_split, H_ends, H_mix) in zip(plan.splits, split_ends):
        assert_allclose([H_split, H_mix], [s.H_split, s.H_mix], rtol=0,
                        atol=1e-12 * span[s.stream])
        assert len(H_ends) == len(s.branches)
    # smith2005_ex18_2 splits a cold stream at its inlet; rtB05 splits its
    # cold stream after its trunk exchanger below the pinch
    assert any(first_nodes.values()) == name.startswith('smith')
    min_approach, violations, bad = hxn_synthesis._exact_approach(
        plan, duties, ends, curves, knots, dT)
    if name.startswith('smith'):   # constant CP: the knots are exact
        assert not bad and min_approach >= dT - 1e-9
    bst.main_flowsheet.set_flowsheet('realize_' + name)
    units, first, _ = hxn_synthesis._realize(
        plan, duties, curves, knots, streams_inlet, is_hot, dT)
    try:
        n_branch = 0
        for n, hx in units.items():
            e = plan.exchangers[n]
            above = e.side == 'above'
            ports = (e.cold, e.hot) if above else (e.hot, e.cold)
            fracs = ((e.cold_frac, e.hot_frac) if above
                     else (e.hot_frac, e.cold_frac))
            duty = span[e.hot] + span[e.cold]
            assert abs(hx.Q - duties[n]) <= 1e-9 * duty, hx.ID
            for k, (j, f) in enumerate(zip(ports, fracs)):
                n_branch += f != 1.
                curve = curves[j]
                H_in, H_out = ends[n, j]
                s_in, s_out = hx.ins[k], hx.outs[k]
                assert_allclose(s_in.F_mol, f * streams_inlet[j].F_mol,
                                rtol=1e-14)
                assert abs(s_in.H - f * (curve.H_lo + H_in)) <= 1e-9 * duty
                assert abs(s_out.H - f * (curve.H_lo + H_out)) <= 1e-9 * duty
                H_lim = getattr(hx, f'H_lim{k}')
                assert H_lim is not None
                assert_allclose(H_lim, f * (curve.H_lo + H_out), rtol=1e-14)
                if n == first[j] or n in first_nodes[j]:
                    assert s_in.T == streams_inlet[j].T   # the real inlet
            if name.startswith('smith'):
                assert internal_approach(hx) >= dT - APPROACH_TOL, hx.ID
        assert n_branch
    finally:
        hxn_synthesis._discard(units.values())

def test_repair_shrinks_branches_with_fractions(monkeypatch):
    # 5 K more than planned: `_repair` shrinks every violating exchanger,
    # branches with their fractions, in one pass, to duties that keep the
    # new approach on the exact states (and on the realized exchangers)
    plan, curves, knots, is_hot, streams_inlet, dT = planned_split(
        'smith2005_ex18_2_split')
    T_min_app = dT + 5.
    duties = {n: e.Q for n, e in enumerate(plan.exchangers)}
    ends = hxn_synthesis._walk(plan, duties, knots, is_hot)[0]
    bad = hxn_synthesis._exact_approach(plan, duties, ends, curves, knots,
                                        T_min_app)[2]
    fractions = [(e.hot_frac, e.cold_frac) for e in plan.exchangers]
    assert any(fractions[n] != (1., 1.) for n in bad)
    shrink, calls = hxn_synthesis._shrink, []
    def spy(*args):
        calls.append(args[-2:])
        return shrink(*args)
    monkeypatch.setattr(hxn_synthesis, '_shrink', spy)
    new, changes = hxn_synthesis._repair(plan, duties, knots, is_hot, curves,
                                         T_min_app)
    assert [n for n, _, _ in changes] == bad   # one pass
    assert calls == [fractions[n] for n in bad]
    assert all(0. <= Q_after < Q_before for _, Q_before, Q_after in changes)
    ends = hxn_synthesis._walk(plan, new, knots, is_hot)[0]
    worst, _, bad = hxn_synthesis._exact_approach(plan, new, ends, curves,
                                                  knots, T_min_app)
    assert not bad and worst >= T_min_app - hxn_synthesis._APPROACH_TOL
    bst.main_flowsheet.set_flowsheet('repair_split')
    units = hxn_synthesis._realize(plan, new, curves, knots, streams_inlet,
                                   is_hot, T_min_app)[0]
    try:
        for hx in units.values():
            assert internal_approach(hx) >= T_min_app - APPROACH_TOL, hx.ID
    finally:
        hxn_synthesis._discard(units.values())

# ---------------------------------------------------------------------------
# Stream splitting: splitters, mixers and the refine loop (synthesize_network)
# ---------------------------------------------------------------------------

import math

SPLIT_CASES = ['smith2005_ex18_2_split', 'rtB05_above_2h1c']

def split_units(name, flowsheet=None):
    """Heat utilities of corpus case `name` in the order the facility gives
    them (by duty) and its T_min_app, in flowsheet `flowsheet` (if given)."""
    case = next(c for c in MER_SPLIT if c['name'] == name)
    units, dT = test_hxn_mer._build(case)
    hus = sorted([hx.heat_utilities[0] for hx in units],
                 key=lambda hu: hu.duty)
    if flowsheet: bst.main_flowsheet.set_flowsheet(flowsheet)
    return hus, dT

def split_synthesis(name, flowsheet=None, **kwargs):
    """`synthesize_network` with stream splitting on corpus case `name`:
    its result, its info, the streams' curves and T_min_app."""
    hus, dT = split_units(name, flowsheet)
    curves = hxn_synthesis._pinch_analysis(hus, dT)[13]
    info = {}
    result = synthesize_network(hus, dT, info=info, stream_splitting=True,
                                **kwargs)
    return result, info, curves, dT

def spy_plans(monkeypatch):
    """Record every plan (and the keywords it was planned with) that
    `synthesize_network` makes."""
    plans, plan_network = [], hxn_synthesis.plan_network
    def spy(*args, **kwargs):
        plan = plan_network(*args, **kwargs)
        plans.append((plan, kwargs))
        return plan
    monkeypatch.setattr(hxn_synthesis, 'plan_network', spy)
    return plans

def stream_port(hx, j):
    """Port of stream `j` in synthesized process exchanger `hx`."""
    return _stream_ports(hx).index(j)

def test_synthesize_network_splitting_requires_info():
    units = r002('r002_split_info')
    hus = [hx.heat_utilities[0] for hx in units]
    with pytest.raises(ValueError, match='info'):
        synthesize_network(hus, 10., stream_splitting=True)

@pytest.mark.parametrize('name', [*SPLIT_CASES, 'crude_fractionation_ph11c2',
                                  'rtB03_above_hot_vapors'])
def test_synthesize_network_with_splitting(name):
    # smith2005_ex18_2 splits a cold stream at its inlet above the pinch,
    # isothermally; rtB05 (real thermo) splits its cold stream as its last
    # node, non-isothermally, into its heater; the crude unit splits into
    # three branches or more (splitter chains of several elements); rtB03
    # splits a vapor at its inlet, whose real inlet is not bit for bit its
    # state at the inlet enthalpy: all reach MER through splitter chains,
    # branch exchangers and rigorous mixers
    result, info, curves, dT = split_synthesis(name, 'synth_' + name)
    assert len(result) == 13
    (hs, cs, utils, hxs, T_in, T_out, pinch_T, C_flow, hus_rearranged,
     streams_inlet, stream_HXs, hot_indices, cold_indices) = result
    assert info['status'] == 'mer' and info['stream_splitting'] is True
    assert info['dropped'] == info['repaired'] == []
    assert info['split_deviations'] == [] and info['deviations'] == []
    scale = sum(abs(c.H_out - c.H_in) for c in curves)
    assert_allclose([info['Q_hot'], info['Q_cold']],
                    [info['Q_hot_target'], info['Q_cold_target']],
                    rtol=0, atol=1e-6 * scale)
    process = hs + cs
    assert all(isinstance(hx, bst.HXprocess) for hx in process)
    for i, stages in stream_HXs.items():
        assert isinstance(stages[-1], bst.HXutility)
        assert all(isinstance(u, bst.HXprocess) for u in stages[:-1])
    # every process exchanger is on both its streams, once
    assert sorted(hx.ID for stages in stream_HXs.values()
                  for hx in stages[:-1]) == sorted(
        [hx.ID for hx in process] * 2)
    splits = info['splits']
    assert splits and all(isinstance(sp, hxn_synthesis.StreamSplit)
                          for sp in splits)
    branch_HXs = {hx for sp in splits for b in sp.branches for hx in b}
    ordinal = {}
    for sp in splits:
        j, n = sp.stream, len(sp.fractions)
        hot = j in hot_indices
        curve = curves[j]
        duty = abs(curve.H_out - curve.H_in)
        tag = 'hs' if sp.side == 'above' else 'cs'
        ordinal[j, tag] = ordinal.get((j, tag), 0) + 1
        assert sp.index == ordinal[j, tag]
        base = f'Split_{j}_{tag}' + ('' if sp.index == 1 else f'_{sp.index}')
        assert [u.ID for u in sp.splitters] == [base] + [
            f'{base}_b{c}' for c in range(2, n)]
        assert all(type(u) is bst.Splitter for u in sp.splitters)
        assert type(sp.mixer) is bst.Mixer and sp.mixer.rigorous
        assert sp.mixer.ID == 'Mix' + base[len('Split'):]
        assert len(sp.branches) == n == len(sp.mixer.ins) >= 2
        assert abs(math.fsum(sp.fractions) - 1.) <= 1e-12
        # the chain: element c sends f_c / (f_c + ... + f_n) of its feed to
        # its first outlet; its second outlet feeds element c + 1
        F = streams_inlet[j].F_mol
        feed = sp.splitters[0].ins[0]
        assert_allclose(feed.F_mol, F, rtol=1e-14)
        assert abs(feed.H - sp.H_split) <= 1e-9 * duty
        for c, u in enumerate(sp.splitters):
            assert_allclose(u.split, sp.fractions[c]
                            / math.fsum(sp.fractions[c:]), rtol=1e-15)
            if c: assert u.ins[0] is sp.splitters[c - 1].outs[1]
        for b, f in enumerate(sp.fractions):
            unit, port = sp.outlet(b)
            assert unit in sp.splitters
            assert_allclose(unit.outs[port].F_mol, f * F, rtol=1e-12)
            for hx in sp.branches[b]:
                assert hx in process
                assert_allclose(hx.ins[stream_port(hx, j)].F_mol, f * F,
                                rtol=1e-12)
        # the stream re-joins at its split enthalpy -/+ the branch duties
        Q = math.fsum(hx.Q for b in sp.branches for hx in b)
        assert abs(sp.H_mix - (sp.H_split - Q if hot else sp.H_split + Q)
                   ) <= 1e-9 * duty
        assert abs(sp.mixer.outs[0].H - sp.H_mix) <= 1e-9 * duty
        # the stream's stages: its trunk exchangers before the split, the
        # branches (branch by branch, each in flow order), then the rest
        stages = stream_HXs[j][:-1]
        branches = [hx for b in sp.branches for hx in b]
        k = stages.index(branches[0])
        assert stages[k:k + len(branches)] == branches
        assert len([hx for hx in stages[:k]
                    if hx not in branch_HXs]) == sp.position
        if k == 0:   # a split at the stream's inlet takes the real inlet
            real = _copy(streams_inlet[j])
            hxn_synthesis._first_inlet(real, not curve.monotone,
                                       curve.T_out, hot)
            assert feed.T == real.T
    # smith2005_ex18_2 splits at the inlet and re-joins at the pinch (one
    # temperature); rtB05 re-joins non-isothermally into its heater
    if name.startswith('crude'):
        assert max(len(sp.fractions) for sp in splits) >= 3
    elif name.startswith('smith'):
        assert [(sp.position, sp.isothermal) for sp in splits] == [(0, True)]
    elif name.startswith('rtB05'):
        assert [(sp.position, sp.isothermal) for sp in splits] == [(1, False)]
        (sp,) = splits
        util = stream_HXs[sp.stream][-1]
        assert abs(util.ins[0].H - sp.H_mix) <= 1e-9 * abs(
            curves[sp.stream].H_out - curves[sp.stream].H_in)

@pytest.mark.parametrize('name', SPLIT_CASES)
def test_split_mixer_outlet_is_the_planned_state(name, monkeypatch):
    # a rigorous mixer seeded at the planned state lands on it: T within
    # _MIX_T_TOL of the planned mix state and H equal to its inlets' (to
    # round-off for constant CP; a real-thermo outlet's H is the H of the
    # converged T); isothermal inlets share the mix state, others each
    # carry the end state of their branch
    result, info, curves, dT = split_synthesis(name, 'mix_' + name)
    assert hxn_synthesis._MIX_T_TOL == 1e-7
    smith = name.startswith('smith')
    for sp in info['splits']:
        j = sp.stream
        curve = curves[j]
        duty = abs(curve.H_out - curve.H_in)
        out = sp.mixer.outs[0]
        planned = curve.state_at_H(sp.H_mix)
        assert abs(out.T - planned.T) <= hxn_synthesis._MIX_T_TOL
        H_in = math.fsum(s.H for s in sp.mixer.ins)
        assert abs(out.H - H_in) <= (1e-12 * abs(H_in) if smith
                                     else 1e-9 * duty)
        assert_allclose(out.F_mol, result[9][j].F_mol, rtol=1e-12)
        assert [s.ID for s in sp.mixer.ins] == [
            f's_{j}_{b}__{sp.mixer.ID}' for b in range(len(sp.fractions))]
        assert out.ID == f'{sp.mixer.ID}__s_{j}'
        for f, s, branch in zip(sp.fractions, sp.mixer.ins, sp.branches):
            assert_allclose(s.F_mol, f * out.F_mol, rtol=1e-12)
            if sp.isothermal:
                assert abs(s.T - planned.T) <= 1e-9
            else:
                hx = branch[-1]
                assert abs(s.H - hx.outs[stream_port(hx, j)].H
                           ) <= 1e-9 * duty
    # one rule for every mixer: outside the tolerance, it is reported
    monkeypatch.setattr(hxn_synthesis, '_MIX_T_TOL', -1.)
    _, info, _, _ = split_synthesis(name, 'mix_dev_' + name)
    assert [d['ID'] for d in info['split_deviations']] == [
        sp.mixer.ID for sp in info['splits']]
    for d, sp in zip(info['split_deviations'], info['splits']):
        assert set(d) == {'ID', 'T_plan', 'T', 'H_plan', 'H'}
        assert d['T'] == sp.mixer.outs[0].T and d['H'] == sp.mixer.outs[0].H

def test_unsplit_refine_loop_identical_with_splitting(monkeypatch):
    # with chords 0.5 K off the exact curves and no refine round, round 0
    # violates and is repaired; no side splits, so the flag changes nothing
    # (the refine loop with splitting is the default loop, R3)
    curves = hxn_synthesis.stream_curves
    monkeypatch.setattr(hxn_synthesis, 'stream_curves',
                        lambda *args, **kwargs: curves(*args, tol_T=0.5, **kwargs))
    monkeypatch.setattr(hxn_synthesis, '_MAX_REFINE', 0)
    units, T_min_app = test_hxn_targets.case_curvature()
    hus = [hx.heat_utilities[0] for hx in units]
    plans = spy_plans(monkeypatch)
    runs = []
    for flag in (False, True):
        bst.main_flowsheet.set_flowsheet(f'curvature_split_{flag}')
        info = {}
        result = synthesize_network(hus, T_min_app, info=info,
                                    stream_splitting=flag)
        runs.append((info, [(hx.ID, hx.Q) for hx in result[0] + result[1]]))
    (off, net_off), (on, net_on) = runs
    assert len(plans) == 2   # one round each
    assert on['refine_rounds'] == off['refine_rounds'] == 0
    assert on['repaired'] and on['repaired'] == off['repaired']
    assert net_on == net_off
    assert 'splits' not in off and 'stream_splitting' not in off
    assert on['splits'] == on['split_deviations'] == []
    assert all(s['split'] is None for s in on['sides'].values())

def test_split_retry_changes_the_candidate(monkeypatch):
    # a violation on a split side after the last refine round (injected in
    # round 0, with no refine rounds): the side's network is excluded by its
    # signature, and the retry round plans a different one there
    monkeypatch.setattr(hxn_synthesis, '_MAX_REFINE', 0)
    plans = spy_plans(monkeypatch)
    exact, injected = hxn_synthesis._exact_approach, []
    def inject(plan, duties, ends, curves, knots, T_min_app):
        worst, violations, bad = exact(plan, duties, ends, curves, knots,
                                       T_min_app)
        if len(plans) == 1 and not injected:
            assert not bad
            n = next(n for n, e in enumerate(plan.exchangers)
                     if (e.hot_frac, e.cold_frac) != (1., 1.))
            e = plan.exchangers[n]
            curve, H = curves[e.hot], ends[n, e.hot][0]  # its hot end
            violations = {e.hot: [(curve.T_exact(curve.H_lo + H), H)]}
            bad = [n]
            injected.append(e.side)
        return worst, violations, bad
    monkeypatch.setattr(hxn_synthesis, '_exact_approach', inject)
    result, info, curves, dT = split_synthesis('smith2005_ex18_2_split',
                                               'split_retry')
    (s,) = injected
    assert len(plans) == 2 and info['refine_rounds'] == 1
    sp0 = plans[0][0].info['sides'][s]['split']
    sp1 = plans[1][0].info['sides'][s]['split']
    assert sp1['signature'] != sp0['signature']
    assert plans[1][1]['_split_exclude'] == {s: {sp0['signature']}}
    assert plans[1][1]['_split_prefer'] == {
        s: (sp0['candidate'], sp0['signature'])}
    assert sp1['candidates'][sp0['candidate']] == 'excluded'
    # the retry's plan is the one realized (no restore)
    assert info['sides'] is plans[1][0].info['sides']
    assert info['status'] == 'mer' and info['split_deviations'] == []

def test_split_retry_restores_the_best_plan(monkeypatch):
    # a retry that does worse (injected: one violating exchanger on the
    # split side in round 0, all of that side's in the retry): the round-0
    # plan, with the fewest violating exchangers, is restored and realized
    monkeypatch.setattr(hxn_synthesis, '_MAX_REFINE', 0)
    monkeypatch.setattr(hxn_synthesis, '_MAX_SPLIT_RETRY', 1)
    plans = spy_plans(monkeypatch)
    exact, injected = hxn_synthesis._exact_approach, []
    def inject(plan, duties, ends, curves, knots, T_min_app):
        worst, violations, bad = exact(plan, duties, ends, curves, knots,
                                       T_min_app)
        if len(injected) < len(plans) <= 2:   # once in each round
            assert not bad
            s = next(e.side for e in plan.exchangers
                     if (e.hot_frac, e.cold_frac) != (1., 1.))
            bad = [n for n, e in enumerate(plan.exchangers) if e.side == s]
            bad = bad[:1] if len(plans) == 1 else bad
            violations = {}
            for n in bad:
                e = plan.exchangers[n]
                curve, H = curves[e.hot], ends[n, e.hot][0]
                violations.setdefault(e.hot, []).append(
                    (curve.T_exact(curve.H_lo + H), H))
            injected.append((s, bad))
        return worst, violations, bad
    monkeypatch.setattr(hxn_synthesis, '_exact_approach', inject)
    result, info, curves, dT = split_synthesis('smith2005_ex18_2_split',
                                               'split_restore')
    (s, bad0), (s1, bad1) = injected
    assert s1 == s and len(bad0) == 1 < len(bad1)
    assert len(plans) == 2 and info['refine_rounds'] == 1
    sp0 = plans[0][0].info['sides'][s]['split']
    assert plans[1][0].info['sides'][s]['split']['signature'] != (
        sp0['signature'])
    assert info['sides'] is plans[0][0].info['sides']   # restored
    assert info['status'] == 'mer' and info['repaired'] == []
    (sp,) = info['splits']
    assert sp.fractions == plans[0][0].splits[0].fractions

def test_split_realization_failure_merges_the_split(monkeypatch):
    # a mixer that cannot be simulated: its split's branch exchangers are
    # dropped (their duty goes to the utilities), no splitter or mixer is
    # left, the stream passes as a trunk and every balance still closes
    plans = spy_plans(monkeypatch)
    def fail(self): raise RuntimeError('mixer failed')
    monkeypatch.setattr(bst.Mixer, '_run', fail)
    result, info, curves, dT = split_synthesis('smith2005_ex18_2_split',
                                               'split_failure')
    hs, cs, utils = result[:3]
    streams_inlet, stream_HXs = result[9], result[10]
    (plan, _), = plans
    (split,) = plan.splits
    branch = [n for ns in split.branches for n in ns]
    assert len(info['dropped']) == len(branch) >= 2
    assert all('mixer failed' in d['error'] for d in info['dropped'])
    assert_allclose(sorted(d['Q'] for d in info['dropped']),
                    sorted(plan.exchangers[n].Q for n in branch), rtol=1e-15)
    assert info['splits'] == [] and info['status'] == 'best_effort'
    assert not [ID for ID in bst.main_flowsheet.unit.data
                if ID.startswith(('Split_', 'Mix_'))]
    scale = sum(abs(c.H_out - c.H_in) for c in curves)
    assert_allclose([info['Q_hot'], info['Q_cold']],
                    [info['Q_hot_plan'], info['Q_cold_plan']],
                    rtol=0, atol=1e-6 * scale)
    Q_lost = math.fsum(d['Q'] for d in info['dropped'])
    assert_allclose(info['Q_hot_plan'], info['Q_hot_target'] + Q_lost,
                    rtol=1e-9)
    # every remaining exchanger runs whole streams
    for hx in hs + cs:
        for k, j in enumerate(_stream_ports(hx)):
            assert_allclose(hx.ins[k].F_mol, streams_inlet[j].F_mol,
                            rtol=1e-14)
    assert len(hs + cs) == len(plan.exchangers) - len(branch)

if __name__ == '__main__':
    pytest.main([__file__, '-q', '-p', 'no:cacheprovider'])
