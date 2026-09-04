# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Tutorial chapter 04 (docs/source/tutorial/04_configuring.rst): configuring a
HeatExchangerNetwork -- sweeping the minimum approach temperature to expose the
utility/capital trade-off, scoping the analysis with ``ignored=``, and
synthesizing a larger ten-stream system (the ten-stream case of the regression
suite, inlined here rather than imported from ``tests``). Regions between
``# [start:x]`` / ``# [end:x]`` are literalinclude'd by the page; everything
else is plumbing.

    python docs/_demo_src/examples/ch04_configuring.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _common  # noqa: E402  (sets NUMBA_DISABLE_JIT / DISABLE_PREFERENCES, Agg)
from _common import capturing, save


# [start:imports]
import matplotlib.pyplot as plt
import biosteam as bst
from hensmith import HeatExchangerNetwork
from hensmith.hxn_synthesis import problem_table
# [end:imports]


# [start:helpers]
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
# [end:helpers]


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
    with capturing('ch04_sweep'):
        # [start:sweep]
        rows = []
        for T_min_app in (2., 5., 10., 15., 20., 30.):
            HXN.T_min_app = T_min_app
            sys.simulate()
            rows.append((T_min_app, HXN.actual_heat_util_load, HXN.actual_cool_util_load,
                         HXN.installed_costs['Heat exchangers']))
        print('T_min_app [K]   heating [kJ/hr]   cooling [kJ/hr]   added installed cost [USD]')
        for T, heat, cool, cost in rows:
            print(f'{T:13.0f}   {heat:15.4g}   {cool:15.4g}   {cost:26.4g}')
        # [end:sweep]
    # [start:sweep_plot]
    T = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(T, [r[1] / 1e6 for r in rows], 'o-', color='#d62728', label='heating utility')
    ax1.plot(T, [r[2] / 1e6 for r in rows], 's-', color='#2e6db4', label='cooling utility')
    ax1.set_xlabel('T_min_app [K]'); ax1.set_ylabel('utility load [GJ/hr]')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(T, [r[3] / 1e6 for r in rows], 'o-', color='k')
    ax2.set_xlabel('T_min_app [K]'); ax2.set_ylabel('added installed cost [MUSD]'); ax2.grid(alpha=0.3)
    fig.tight_layout()
    # [end:sweep_plot]
    save(fig, 'tutorial_04_T_min_app_sweep.png')
    plt.close(fig)
    with capturing('ch04_ignored'):
        # [start:ignored]
        HXN.T_min_app = 5.
        HXN.ignored = [D1_H1]        # leave the bottoms cooler out of the analysis
        sys.simulate()
        print(f'streams in the network: {len(HXN.stream_life_cycles)}')
        print(f'heating utility: {HXN.original_heat_util_load:.4g} -> {HXN.actual_heat_util_load:.4g} kJ/hr')
        print(f'cooling utility: {HXN.original_cool_util_load:.4g} -> {HXN.actual_cool_util_load:.4g} kJ/hr')
        HXN.ignored = None
        # [end:ignored]
    with capturing('ch04_ten_streams'):
        # [start:ten_streams]
        bst.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        bst.main_flowsheet.set_flowsheet('ten_streams')
        units = [utility_hx('H1', 355., 101325., 'g', 320., Ethanol=300.),
                 utility_hx('H2', 420., 5e5, 'g', 340., Water=150.),
                 utility_hx('H3', 395., 5e5, 'l', 330., Water=1000.),
                 utility_hx('H4', 370., 101325., 'l', 310., Ethanol=700.),
                 utility_hx('H5', 380., 101325., 'g', 372.5, Water=100.),   # partial condensation
                 utility_hx('C1', 300., 101325., 'l', 360., Water=2000.),
                 boiling_hx('C2', 330., 101325., 380., Water=200.),
                 boiling_hx('C3', 320., 101325., 352., Ethanol=400.),
                 utility_hx('C4', 310., 101325., 'l', 345., Water=400., Ethanol=400.),
                 utility_hx('C5', 340., 101325., 'l', 390., Water=600.)]
        HXN10 = HeatExchangerNetwork('HXN', T_min_app=5.)
        sys10 = bst.System.from_units('sys10', units=[*units, HXN10])
        sys10.simulate()
        hus = HXN10.original_heat_utils
        streams_inlet = [hu.unit.ins[0].copy() for hu in hus]
        streams_quenched = [hu.unit.outs[0].copy() for hu in hus]
        for s in streams_quenched: s.vle(H=s.H, P=s.P)
        table = problem_table(streams_inlet, streams_quenched, [hu.duty < 0 for hu in hus], 5.)
        print(f'process exchangers: {len(HXN10.new_HXs)}')
        print(f'hot utility:  target {table.hot_util_load:.4g}, network {HXN10.actual_heat_util_load:.4g} kJ/hr')
        print(f'cold utility: target {table.cold_util_load:.4g}, network {HXN10.actual_cool_util_load:.4g} kJ/hr')
        print(f'energy balance error: {HXN10.energy_balance_percent_error:.2g} %')
        fig, ax = HXN10.plot_pinch_diagram()
        # [end:ten_streams]
    save(fig, 'tutorial_04_ten_streams_pinch_diagram.png')
    plt.close(fig)
    assert len(HXN10.new_HXs) == 15, len(HXN10.new_HXs)
    assert HXN10.actual_heat_util_load >= table.hot_util_load * (1 - 1e-3)


if __name__ == '__main__':
    main()
