# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Tutorial chapter 05 (docs/source/tutorial/05_targeting.rst): targeting and
supertargeting with ``hensmith.targeting`` -- the facility's targets
(``HXN.targets()``), the multiple-utility targets priced by biosteam, the
composite, balanced composite and grand composite curves drawn by the
library, the area, units, capital and total annualized cost targets, the
``T_min_app`` sweep for the least total annualized cost (on the quickstart
system and on a two-stream problem with an interior optimum), and the report
that compares the synthesized network with its targets. Regions between
``# [start:x]`` / ``# [end:x]`` are literalinclude'd by the page; everything
else is plumbing.

    python docs/_demo_src/examples/ch05_targeting.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _common  # noqa: E402  (sets NUMBA_DISABLE_JIT / DISABLE_PREFERENCES, Agg)
from _common import capturing, save


# [start:imports]
import numpy as np
import matplotlib.pyplot as plt
from hensmith import (supertarget, network_report, plot_composite_curves,
                      plot_grand_composite_curve, plot_supertargets)
# [end:imports]


def knots(T, H):
    """Knots [K, kJ/hr] of a polyline without its zero-length steps and
    without the points inside a straight run, so that two polylines tracing
    the same curve on different grids compare knot by knot."""
    T, H = np.asarray(T, float), np.asarray(H, float)
    step = np.r_[True, (np.abs(np.diff(T)) > 1e-9) | (np.abs(np.diff(H)) > 1e-6)]
    T, H = T[step], H[step]
    if T.size < 3: return T, H
    a, b, c = slice(None, -2), slice(1, -1), slice(2, None)
    cross = (T[b] - T[a]) * (H[c] - H[a]) - (H[b] - H[a]) * (T[c] - T[a])
    bend = np.abs(cross) > 1e-9 * np.abs(T[c] - T[a]) * np.abs(H[c] - H[a])
    keep = np.r_[True, bend, True]
    return T[keep], H[keep]


def main():
    # [start:system]
    import biosteam as bst
    from hensmith import HeatExchangerNetwork

    bst.settings.set_thermo(['Water', 'Methanol', 'Glycerol'], cache=True)
    bst.main_flowsheet.set_flowsheet('quickstart')
    feed1 = bst.Stream('feed1', flow=(8000, 100, 25))
    feed2 = bst.Stream('feed2', flow=(10000, 1000, 10))
    D1 = bst.ShortcutColumn('D1', ins=feed1,
                            outs=('distillate', 'bottoms_product'),
                            LHK=('Methanol', 'Water'),
                            y_top=0.99, x_bot=0.01, k=2,
                            is_divided=True)
    D1_H1 = bst.HXutility('D1_H1', ins=D1.outs[1], T=300)
    D1_H2 = bst.HXutility('D1_H2', ins=D1.outs[0], T=300)
    F1 = bst.Flash('F1', ins=feed2, outs=('vapor', 'liquid'), V=0.9, P=101325)
    HXN = HeatExchangerNetwork('HXN', T_min_app=5.)
    sys = bst.System.from_units('sys', units=[D1, D1_H1, D1_H2, F1, HXN])
    sys.simulate()
    # [end:system]
    with capturing('ch05_targets'):
        # [start:targets]
        targets = HXN.targets()        # the facility's streams at its T_min_app
        print(targets)
        print('streams:', targets.labels)
        print(f'shifted pinch: {targets.pinch_T:.2f} K')
        info = HXN.synthesis_info
        print('equal to the synthesis targets:',
              targets.Q_hot == info['Q_hot_target'], targets.Q_cold == info['Q_cold_target'])
        print(f'heating utility of the network: {HXN.actual_heat_util_load:.4g} kJ/hr')
        # [end:targets]
    with capturing('ch05_utilities'):
        # [start:utilities]
        utilities = targets.utility_targets()     # biosteam's default agents
        print('agent               duty        utility duty  level   process T [K]     cost')
        print('                    [kJ/hr]     [kJ/hr]       [K]     in -> out         [USD/hr]')
        for u in utilities:
            print(f'{u.ID:<18}  {u.duty:<10.4g}  {u.utility_duty:<12.4g}  {u.level:<6.2f}  '
                  f'{u.T_in:.2f} -> {u.T_out:.2f}  {u.cost:.2f}')
        print(f'utility cost target: {sum(u.cost for u in utilities):.2f} USD/hr')
        for u in utilities:
            print(f'{u.ID}: {u.flow:.4g} kmol/hr, from {u.T_utility_in:.2f} to {u.T_utility_out:.2f} K')
        # [end:utilities]
    # [start:composites]
    fig, ax = plot_composite_curves(targets)
    fig_balanced, ax_balanced = plot_composite_curves(targets, kind='balanced',
                                                      utilities=utilities)
    # [end:composites]
    save(fig, 'tutorial_05_composite_curves.png')
    save(fig_balanced, 'tutorial_05_balanced_composites.png')
    plt.close(fig); plt.close(fig_balanced)
    # [start:gcc]
    fig, ax = plot_grand_composite_curve(targets, utilities=utilities)
    # [end:gcc]
    save(fig, 'tutorial_05_grand_composite.png')
    plt.close(fig)
    with capturing('ch05_summary'):
        # [start:supertargets]
        print('units:', targets.units_target(utilities))
        capital = targets.capital_cost_target(utilities)
        print(f"{capital['N_shells']} shells of {capital['shell_area']:.4g} m2, {capital['type']}")
        summary = targets.summary()     # 10 % over 10 years; 8000 h/yr (no system hours)
        for key, value in summary.items():
            print(f'{key:<24} {value:.6g}')
        # [end:supertargets]
    with capturing('ch05_sweep'):
        # [start:sweep]
        sweep = supertarget(targets, np.arange(1., 31.))
        print(sweep)
        print('T_min_app  Q_hot      Q_cold     area    units  installed  capital    operating  TAC')
        print('[K]        [kJ/hr]    [kJ/hr]    [m2]           [USD]      [USD/yr]   [USD/yr]   [USD/yr]')
        optimum = sweep.optimum
        after = sweep.records[sweep.records.index(optimum) + 1]     # the next point evaluated
        for r in sweep.records:
            if r['T_min_app'] in (1., 2., 5., 10., 15., 20., 25., 30.) or r is optimum or r is after:
                print(f"{r['T_min_app']:<9.4g}  {r['Q_hot']:<9.4g}  {r['Q_cold']:<9.4g}  "
                      f"{r['area']:<6.0f}  {r['N_units']:<5d}  {r['installed_cost']:<9.4g}  "
                      f"{r['annualized_capital_cost']:<9.4g}  {r['operating_cost']:<9.4g}  {r['TAC']:.0f}")
        print(f'optimum {sweep.T_min_app:.4g} K, at the bound of the sweep: {sweep.at_bound}')
        fig, ax = plot_supertargets(sweep)
        # [end:sweep]
    save(fig, 'tutorial_05_supertargets.png')
    plt.close(fig)
    with capturing('ch05_trade_off'):
        # [start:trade_off]
        import thermosteam as tmo
        from hensmith import PinchTargets
        fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1.,
                             Cn=3.6, default=True)     # 1000 kmol/hr: 1 kW/K
        bst.settings.set_thermo([fluid])
        bst.main_flowsheet.set_flowsheet('trade_off')
        H1 = bst.HXutility('H1', ins=bst.Stream('H1_in', Fluid=1e5, T=400.),
                           T=320., rigorous=False)       # 100 kW/K, 400 -> 320 K
        C1 = bst.HXutility('C1', ins=bst.Stream('C1_in', Fluid=1e5, T=310.),
                           T=390., rigorous=False)       # 100 kW/K, 310 -> 390 K
        H1.simulate(); C1.simulate()
        pair = PinchTargets.from_heat_utilities([H1.heat_utilities[0], C1.heat_utilities[0]])
        water = bst.HeatUtility.get_heating_agent('low_pressure_steam').thermo
        steam = bst.UtilityAgent('steam_450K', Water=1, T=450., P=1e6, phase='g',
                                 heat_transfer_price=4e-7, thermo=water)
        refrigerant = bst.UtilityAgent('water_280K', Water=1, T=280., P=1e3, phase='l',
                                       heat_transfer_price=2e-7, thermo=water)
        pair_sweep = supertarget(pair, np.arange(11., 61.), heating_agents=[steam],
                                 cooling_agents=[refrigerant])
        print(pair_sweep)
        for key in ('Q_hot', 'area', 'N_units', 'installed_cost', 'annualized_capital_cost',
                    'operating_cost', 'TAC'):
            print(f'{key:<24} {pair_sweep.optimum[key]:.6g}')
        fig, ax = plot_supertargets(pair_sweep)
        # [end:trade_off]
    save(fig, 'tutorial_05_trade_off.png')
    plt.close(fig)
    with capturing('ch05_report'):
        # [start:report]
        report = network_report(HXN)
        target = report['targets']      # targets.summary() with the same options
        print('                        network     target')
        print(f"utility cost [USD/hr]   {report['utility_cost']:<10.2f}  {target['utility_cost']:.2f}")
        print(f"units                   {report['N_units']:<10d}  {target['N_units']}")
        print(f"area [m2]               {report['area']:<10.0f}  {target['area']:.0f}")
        print(f"area, biosteam [m2]     {report['area_biosteam']:.0f}")
        print(f"installed cost [USD]    {report['installed_cost']:<10.4g}  {target['installed_cost']:.4g}")
        print(f"TAC [USD/yr]            {report['TAC']:<10.0f}  {target['TAC']:.0f}")
        print('exchanger    area   biosteam [m2]')
        for ID, (area, area_biosteam) in report['exchanger_areas'].items():
            print(f'{ID:<11}  {area:<5.0f}  {area_biosteam:.0f}')
        # [end:report]
    # plumbing checks: the library's curves are chapter 2's
    from examples.ch02_pinch_analysis import composite_curves, grand_composite
    ours = targets.composite_curves()
    theirs = composite_curves(targets.table, T_min_app=5.)
    for (T, H), (T2, H2) in zip((ours[:2], ours[2:]), (theirs[:2], theirs[2:])):
        (T, H), (T2, H2) = knots(T, H), knots(T2, H2)
        assert T.size == T2.size, (T.size, T2.size)
        assert np.allclose(T, T2, rtol=0, atol=1e-6) and np.allclose(H, H2, rtol=1e-9, atol=1e-3)
    T, H = targets.grand_composite_curve()
    H2, T2 = grand_composite(targets.table)
    assert np.array_equal(T, T2) and np.array_equal(H, H2)
    assert targets.Q_hot == info['Q_hot_target'] and targets.Q_cold == info['Q_cold_target']
    assert report['Q_hot'] >= targets.Q_hot * (1 - 1e-9) and report['status'] == 'mer'
    assert abs(report['utility_cost'] - target['utility_cost']) < 1e-6 * target['utility_cost']
    assert abs(pair_sweep.T_min_app - 21.595) <= 0.02 and not pair_sweep.at_bound


if __name__ == '__main__':
    main()
