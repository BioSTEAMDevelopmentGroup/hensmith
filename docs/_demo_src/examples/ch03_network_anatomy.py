# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Tutorial chapter 03 (docs/source/tutorial/03_network_anatomy.rst): what a
simulated HeatExchangerNetwork exposes -- its own flowsheet and system, the
per-stream life cycles and life stages, the pinch temperatures, a minimal
pinch diagram, and the cost/utility accounting. Regions between
``# [start:x]`` / ``# [end:x]`` are literalinclude'd by the page; everything
else is plumbing.

    python docs/_demo_src/examples/ch03_network_anatomy.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _common  # noqa: E402  (sets NUMBA_DISABLE_JIT / DISABLE_PREFERENCES, Agg)
from _common import capturing, save, save_diagram
import matplotlib.pyplot as plt


def main():
    # [start:imports]
    import matplotlib.pyplot as plt
    # [end:imports]
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
    for theme in ('light', 'dark'):
        save_diagram(HXN.HXN_sys, f'tutorial_03_hxn_flowsheet_{theme}.png', theme)
    with capturing('ch03_flowsheet'):
        # [start:flowsheet]
        print(HXN.HXN_flowsheet)
        print(HXN.HXN_sys)
        print([unit.ID for unit in HXN.HXN_sys.units])
        # [end:flowsheet]
    with capturing('ch03_life_cycles'):
        # [start:life_cycles]
        for life_cycle in HXN.stream_life_cycles:
            life_cycle.show()
        # [end:life_cycles]
    with capturing('ch03_stage'):
        # [start:stage]
        stage = HXN.stream_life_cycles[1].life_cycle[0]
        print(stage.unit, '| stream position', stage.index)
        print(stage.s_in.ID, '->', stage.s_out.ID)
        print(f'{stage.H_in:.4g} -> {stage.H_out:.4g} kJ/hr')
        # [end:stage]
    with capturing('ch03_pinch_Ts'):
        # [start:pinch_Ts]
        for i, life_cycle in enumerate(HXN.stream_life_cycles):
            kind = 'cold' if life_cycle.cold else 'hot '
            print(f'stream {i} ({kind}): in {HXN.inlet_Ts[i] - 273.15:5.1f} °C, '
                  f'pinch {HXN.pinch_Ts[i] - 273.15:5.1f} °C, out {HXN.outlet_Ts[i] - 273.15:5.1f} °C')
        # [end:pinch_Ts]
    # [start:minimal]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    HXN.plot_pinch_diagram(show_units=False, show_auxiliary_units=False,
                           show_stream_IDs=False, show_legend=False, ax=ax)
    # [end:minimal]
    save(fig, 'tutorial_03_pinch_diagram_minimal.png')
    plt.close(fig)
    with capturing('ch03_accounting'):
        # [start:accounting]
        print(f'energy balance error: {HXN.energy_balance_percent_error:.2g} % '
              f'(warns above {100 * HXN.acceptable_energy_balance_error:.0f} %)')
        print(f'original exchangers, purchase cost: {sum(HXN.original_purchase_costs):.4g} USD')
        print(f'new process exchangers, purchase:   {sum(HXN.new_purchase_costs_HXp):.4g} USD')
        print(f'new utility exchangers, purchase:   {sum(HXN.new_purchase_costs_HXu):.4g} USD')
        print(f'facility purchase cost (added):     {HXN.purchase_costs["Heat exchangers"]:.4g} USD')
        print(f'facility installed cost (added):    {HXN.installed_costs["Heat exchangers"]:.4g} USD')
        print('facility heat utilities (new minus original):')
        for hu in HXN.heat_utilities:
            print(f'  {hu.ID:<20} duty {hu.duty:>11.4g} kJ/hr   cost {hu.cost:>8.4g} USD/hr')
        # [end:accounting]
    assert len(HXN.new_HXs) == 4


if __name__ == '__main__':
    main()
