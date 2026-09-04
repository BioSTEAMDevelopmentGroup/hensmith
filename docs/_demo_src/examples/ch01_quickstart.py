# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Tutorial chapter 01 (docs/source/tutorial/01_quickstart.rst): the canonical
column + flash system, end to end. Regions between ``# [start:x]`` /
``# [end:x]`` are literalinclude'd by the page; everything else is plumbing.

    python docs/_demo_src/examples/ch01_quickstart.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _common  # noqa: E402  (sets NUMBA_DISABLE_JIT / DISABLE_PREFERENCES, Agg)
from _common import capturing, save, save_diagram, write_summary
import matplotlib.pyplot as plt


def main():
    # [start:build]
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
    # [end:build]
    # [start:network]
    HXN = HeatExchangerNetwork('HXN', T_min_app=5.)
    sys = bst.System.from_units('sys', units=[D1, D1_H1, D1_H2, F1, HXN])
    # [end:network]
    for theme in ('light', 'dark'):
        save_diagram(sys, f'tutorial_01_quickstart_flowsheet_{theme}.png', theme)
    # [start:simulate]
    sys.simulate()
    # [end:simulate]
    with capturing('ch01_results'):
        # [start:results]
        print(HXN.results())
        # [end:results]
    with capturing('ch01_loads'):
        # [start:loads]
        print(f'heating utility: {HXN.original_heat_util_load:.4g} -> {HXN.actual_heat_util_load:.4g} kJ/hr')
        print(f'cooling utility: {HXN.original_cool_util_load:.4g} -> {HXN.actual_cool_util_load:.4g} kJ/hr')
        print(f'energy balance error: {HXN.energy_balance_percent_error:.2g} %')
        print(f'added installed cost: {HXN.installed_costs["Heat exchangers"]:.3g} USD')
        print(f'process exchangers: {[hx.ID for hx in HXN.new_HXs]}')
        # [end:loads]
    with capturing('ch01_life_cycles'):
        # [start:life_cycles]
        print(HXN.stream_life_cycles)
        # [end:life_cycles]
    # [start:pinch]
    fig, ax = HXN.plot_pinch_diagram()
    # [end:pinch]
    save(fig, 'tutorial_01_quickstart_pinch_diagram.png')
    plt.close(fig)
    heat_red = 100 * (1 - HXN.actual_heat_util_load / HXN.original_heat_util_load)
    cool_red = 100 * (1 - HXN.actual_cool_util_load / HXN.original_cool_util_load)
    stream_1 = HXN.stream_life_cycles[1].life_cycle
    write_summary('ch01_summary', {
        'n_streams': len(HXN.stream_life_cycles),
        'n_auxiliary': sum(1 for hx in HXN.original_heat_exchangers if hx.owner is not hx),
        'n_process_hxs': len(HXN.new_HXs),
        'heating_before': f'{HXN.original_heat_util_load:.4g}',
        'heating_after': f'{HXN.actual_heat_util_load:.4g}',
        'heating_reduction_percent': f'{heat_red:.1f}',
        'cooling_before': f'{HXN.original_cool_util_load:.4g}',
        'cooling_after': f'{HXN.actual_cool_util_load:.4g}',
        'cooling_reduction_percent': f'{cool_red:.1f}',
        'heat_ratio': f'{HXN.actual_heat_util_load / HXN.original_heat_util_load:.2f}',
        'utility_cost_usd_per_hr': f'{HXN.utility_cost:.0f}',
        'installed_cost_usd': f'{HXN.installed_costs["Heat exchangers"]:.3g}',
        'stream_1_process_hxs': sum(1 for st in stream_1 if isinstance(st.unit, bst.HXprocess)),
        'energy_balance_percent_error': f'{HXN.energy_balance_percent_error:.2g}',
        'energy_balance_percent_error_abs': f'{abs(HXN.energy_balance_percent_error):.0e}',
    })
    assert len(HXN.new_HXs) == 4 and len(HXN.stream_life_cycles) == 5, 'quickstart network changed'
    assert abs(HXN.energy_balance_percent_error) < 1e-6


if __name__ == '__main__':
    main()
