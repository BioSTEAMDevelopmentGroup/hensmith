# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Tutorial chapter 02 (docs/source/tutorial/02_pinch_analysis.rst): the problem
table, composite curves and grand composite curve of the quickstart system.
Regions are literalinclude'd by the page; `composite_curves` and
`grand_composite` are also imported by make_hero_gif.py.

    python docs/_demo_src/examples/ch02_pinch_analysis.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _common  # noqa: E402  (sets NUMBA_DISABLE_JIT / DISABLE_PREFERENCES, Agg)
from _common import capturing, save, min_vertical_gap
import numpy as np


# [start:composite_curves]
def composite_curves(table, T_min_app):
    """Hot and cold composite curves (real T [K] vs H [kJ/hr]) of a
    ProblemTable. Walking the shifted grid from the coldest boundary up,
    each interval adds the heat of the streams of that kind in it (a
    diagonal segment) and each point load adds heat at constant temperature
    (a horizontal step). Hot streams were shifted down by T_min_app, so
    their real temperature is Ts + T_min_app. The cold curve starts at
    H = cold utility so the curves overlap by exactly the recovered heat
    and the right-hand overhang is the hot utility."""
    Ts = table.Ts
    n = Ts.size
    is_hot = (table.interval_H.sum(axis=1) + table.point_H.sum(axis=1)) > 0

    def build(rows, T_offset, H0):
        interval = np.abs(table.interval_H[rows].sum(axis=0))   # n - 1 intervals
        point = np.abs(table.point_H[rows].sum(axis=0))         # n boundaries
        T, H = [Ts[n - 1] + T_offset], [H0]
        for k in range(n - 1, -1, -1):
            if point[k] > 0:
                T.append(Ts[k] + T_offset); H.append(H[-1] + point[k])
            if k > 0:
                T.append(Ts[k - 1] + T_offset); H.append(H[-1] + interval[k - 1])
        T, H = np.array(T), np.array(H)
        nz = np.flatnonzero(np.diff(H) > 0)   # drop end segments with no stream of this kind
        return T[nz[0]:nz[-1] + 2], H[nz[0]:nz[-1] + 2]

    hot_T, hot_H = build(is_hot, T_min_app, 0.)
    cold_T, cold_H = build(~is_hot, 0., table.cold_util_load)
    return hot_T, hot_H, cold_T, cold_H
# [end:composite_curves]


# [start:grand_composite]
def grand_composite(table):
    """Heat cascaded through each shifted grid temperature when the minimum
    hot utility is supplied: the value arriving at each boundary and the
    value leaving it after its point loads (a horizontal step where a
    stream is isothermal). It touches zero at the pinch."""
    leaving = table.residual + table.hot_util_load
    arriving = leaving - table.point_H.sum(axis=0)
    H = np.column_stack([arriving, leaving]).ravel()
    T = np.repeat(table.Ts, 2)
    return H, T
# [end:grand_composite]


def main():
    # [start:imports]
    import numpy as np
    import matplotlib.pyplot as plt
    import biosteam as bst
    from hensmith import HeatExchangerNetwork
    from hensmith.hxn_synthesis import problem_table
    # [end:imports]
    with capturing('ch02_threshold'):
        # [start:threshold]
        bst.settings.set_thermo(['Water'])
        hot_in = bst.Stream(Water=1000., T=400., P=5e5, phase='l', units='kmol/hr')
        hot_out = hot_in.copy(); hot_out.vle(T=300., P=5e5)
        cold_in = bst.Stream(Water=900., T=300., P=5e5, phase='l', units='kmol/hr')
        cold_out = cold_in.copy(); cold_out.vle(T=390., P=5e5)
        table = problem_table([hot_in, cold_in], [hot_out, cold_out], [True, False], 5.)
        print('shifted grid Ts [K]:', table.Ts)
        print('hot utility target [kJ/hr]: ', round(table.hot_util_load, 3))
        print('cold utility target [kJ/hr]:', round(table.cold_util_load, -1))
        print('pinch (shifted) [K]:', table.pinch_T)
        # [end:threshold]
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
    with capturing('ch02_table'):
        # [start:table]
        hus = HXN.original_heat_utils
        streams_inlet = [hu.unit.ins[0].copy() for hu in hus]
        streams_quenched = [hu.unit.outs[0].copy() for hu in hus]
        for s in streams_quenched: s.vle(H=s.H, P=s.P)
        is_hot = [hu.duty < 0 for hu in hus]
        table = problem_table(streams_inlet, streams_quenched, is_hot, T_min_app=5.)
        print('shifted grid Ts [K]:', table.Ts.round(2))
        print(f'hot utility target:  {table.hot_util_load:.4g} kJ/hr')
        print(f'cold utility target: {table.cold_util_load:.4g} kJ/hr')
        print(f'pinch (shifted):     {table.pinch_T:.2f} K')
        # [end:table]
    # [start:curves]
    hot_T, hot_H, cold_T, cold_H = composite_curves(table, T_min_app=5.)
    GJ = 1e6  # kJ/hr -> GJ/hr
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvspan(cold_H[0] / GJ, hot_H[-1] / GJ, color='0.92', lw=0, zorder=0, label='heat recovered')
    ax.plot(hot_H / GJ, hot_T - 273.15, color='#d62728', lw=2, label='hot composite')
    ax.plot(cold_H / GJ, cold_T - 273.15, color='#2e6db4', lw=2, label='cold composite')
    y_lo, y_hi = hot_T[0] - 273.15, cold_T[-1] - 273.15
    ax.margins(y=0.12)   # headroom so the utility labels sit inside the frame
    # the cold-utility span is ~6 px wide on this axis, too narrow for an arrow:
    # bound it with two ticks and label it with a leader
    ax.vlines([0, cold_H[0] / GJ], y_lo - 1.5, y_lo + 1.5, color='k', lw=0.8, zorder=3)
    ax.annotate(f'cold utility {table.cold_util_load:.3g} kJ/hr', xy=(cold_H[0] / 2 / GJ, y_lo),
                xytext=(30, -18), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='-', lw=0.6, shrinkB=4))
    ax.annotate('', xy=(hot_H[-1] / GJ, y_hi), xytext=(cold_H[-1] / GJ, y_hi), arrowprops=dict(arrowstyle='<->'))
    ax.text((hot_H[-1] + cold_H[-1]) / 2 / GJ, y_hi + 2, f'hot utility {table.hot_util_load:.3g} kJ/hr',
            ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('H [GJ/hr]'); ax.set_ylabel('T [°C]')
    ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=0.3)
    # [end:curves]
    save(fig, 'tutorial_02_composite_curves.png')
    plt.close(fig)
    # [start:gcc]
    H_cascade, T_cascade = grand_composite(table)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(H_cascade / GJ, T_cascade - 273.15, color='k', lw=2)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.plot([0], [table.pinch_T - 273.15], 'o', mfc='w', mec='k', ms=8)
    ax.annotate(f'pinch  {table.pinch_T - 273.15:.1f} °C (shifted)', xy=(0, table.pinch_T - 273.15),
                xytext=(12, -14), textcoords='offset points', fontsize=8)
    ax.set_xlabel('heat cascaded [GJ/hr]'); ax.set_ylabel('shifted T [°C]'); ax.grid(alpha=0.3)
    # [end:gcc]
    save(fig, 'tutorial_02_grand_composite.png')
    plt.close(fig)
    with capturing('ch02_compare'):
        # [start:compare]
        print(f'hot utility:  target {table.hot_util_load:.4g}, network {HXN.actual_heat_util_load:.4g} kJ/hr')
        print(f'cold utility: target {table.cold_util_load:.4g}, network {HXN.actual_cool_util_load:.4g} kJ/hr')
        # Those loads are utility-side (duty = unit_duty / heat transfer
        # efficiency); the targets are process-side enthalpy differences, so
        # sum the process-side duties of the network's utility exchangers too.
        new_hus = [hu for hx in HXN.new_HX_utils for hu in hx.heat_utilities]
        heat = sum(hu.unit_duty for hu in new_hus if hu.unit_duty > 0)
        cool = -sum(hu.unit_duty for hu in new_hus if hu.unit_duty < 0)
        print(f'hot utility,  process side: target {table.hot_util_load:.4g}, network {heat:.4g} kJ/hr')
        print(f'cold utility, process side: target {table.cold_util_load:.4g}, network {cool:.4g} kJ/hr')
        # [end:compare]
    # plumbing checks: the drawn curves are consistent with the table
    gap = min_vertical_gap(hot_T, hot_H, cold_T, cold_H)
    print(f'minimum vertical approach of the composite curves: {gap:.6f} K')
    assert gap >= 5. - 1e-6, gap
    assert np.isclose(cold_H[-1] - hot_H[-1], table.hot_util_load, rtol=1e-9)
    assert np.isclose(cold_H[0], table.cold_util_load, rtol=1e-9)
    assert np.isclose(H_cascade.min(), 0., atol=1e-6 * abs(H_cascade).max())


if __name__ == '__main__':
    main()
