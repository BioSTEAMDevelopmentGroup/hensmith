# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Render the landing-page hero: the quickstart system's real composite curves
sliding together until they touch at the 5 K pinch and apart again, as a
seamless 8 s loop (20 fps, 2000 x 720 px), light and dark, plus the frame-0
(converged) stills served under prefers-reduced-motion.

    python docs/_demo_src/make_hero_gif.py
    -> docs/source/_static/images/demo/hero_{light,dark}.gif
       docs/source/_static/images/demo/hero_{light,dark}_still.png
"""
import sys
import _common
from _common import (IMAGES, THEMES, DPI, build_quickstart_system,
                     problem_table_from_network, min_vertical_gap, report)
sys.path.insert(0, str(_common.HERE))
from examples.ch02_pinch_analysis import composite_curves  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

OUT = IMAGES / 'demo'
T_MIN_APP = 5.
DUR, FPS = 8.0, 20
N = int(DUR * FPS)
FIGSIZE = (10.0, 3.6)                 # x DPI 200 = 2000 x 720
FIGSIZE_SMALL = (8.0, 2.88)           # fallback if over the size budget
MAX_BYTES = 2.5 * 1024 * 1024
GJ = 1e6


def smoothstep(u):
    u = min(max(u, 0.), 1.)
    return u * u * (3 - 2 * u)


def separation(frame):
    """0 = converged, 1 = fully apart; frame 0 is converged (the still)."""
    u = frame / N
    if u < 0.25: return 0.
    if u < 0.5: return smoothstep((u - 0.25) / 0.25)
    if u < 0.75: return 1.
    return 1. - smoothstep((u - 0.75) / 0.25)


def render(theme, curves, table, a, figsize):
    t = THEMES[theme]
    hot_T, hot_H, cold_T, cold_H = curves
    Q_hot = hot_H[-1]
    offset = table.cold_util_load + a * (Q_hot - table.cold_util_load)
    cH = cold_H - cold_H[0] + offset
    recovered, hot_util, cold_util = max(Q_hot - offset, 0.), cH[-1] - Q_hot, offset
    H_max = (Q_hot + (cold_H[-1] - cold_H[0])) / GJ        # extent in the apart state
    fig = plt.figure(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(t['bg'])
    ax = fig.add_axes([0.07, 0.17, 0.91, 0.72])
    ax.set_facecolor(t['bg'])
    for s in ax.spines.values(): s.set_color(t['grid'])
    ax.tick_params(colors=t['muted'], labelsize=10)
    ax.grid(color=t['grid'], lw=0.6, alpha=0.7)
    ax.set_xlim(-0.03 * H_max, 1.03 * H_max)
    ax.set_ylim(2, 114)      # room below the curves for the two label rows (utility, recovered)
    ax.set_xlabel('H [GJ/hr]', color=t['muted'], fontsize=11)
    ax.set_ylabel('T [°C]', color=t['muted'], fontsize=11)
    # Extent brackets: '|-|' end bars stay legible at any span (the converged
    # cold-utility span is ~8 px wide; '<->' heads would merge into a blob).
    bracket = dict(arrowstyle='|-|, widthA=0.5, widthB=0.5', mutation_scale=8, lw=1.2)
    if recovered > 0:
        ax.axvspan(offset / GJ, Q_hot / GJ, color=t['muted'], alpha=0.14, lw=0, zorder=0)
        ax.text((offset + Q_hot) / 2 / GJ, 10.5, f'heat recovered  {recovered / GJ:.1f} GJ/hr',
                ha='center', va='top', color=t['ink'], fontsize=10)
    ax.plot(hot_H / GJ, hot_T - 273.15, color=t['hot'], lw=2.4, label='hot composite')
    ax.plot(cH / GJ, cold_T - 273.15, color=t['cold'], lw=2.4, label='cold composite')
    y_top = cold_T[-1] - 273.15
    ax.annotate('', xy=(Q_hot / GJ, y_top + 3), xytext=(cH[-1] / GJ, y_top + 3),
                arrowprops=dict(color=t['hot'], **bracket))
    ax.text((Q_hot + cH[-1]) / 2 / GJ, y_top + 5, f'hot utility  {hot_util / GJ:.1f} GJ/hr',
            ha='center', va='bottom', color=t['hot'], fontsize=10)
    y_bot = hot_T[0] - 273.15
    # 4 K below the hot curve's foot so the bar clears the pinch bracket,
    # whose lower end is the cold curve's start (1.9 K below the foot).
    ax.annotate('', xy=(0, y_bot - 4), xytext=(offset / GJ, y_bot - 4),
                arrowprops=dict(color=t['cold'], **bracket))
    # Left-anchored at H = 0: centred on the span it would spill over the y
    # tick labels when the span is the 1.94 GJ/hr converged overhang.
    ax.text(0, y_bot - 6.5, f'cold utility  {cold_util / GJ:.2f} GJ/hr',
            ha='left', va='top', color=t['cold'], fontsize=10)
    if a < 0.15:                                     # the pinch bracket fades in as the curves meet
        alpha = 1. - a / 0.15
        k = int(np.argmin(np.abs(cold_T - table.pinch_T)))
        x = cH[k] / GJ
        y0, y1 = table.pinch_T - 273.15, table.pinch_T + T_MIN_APP - 273.15
        ax.annotate('', xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(color=t['ink'], alpha=alpha, **bracket))
        ax.text(x + 0.012 * H_max, (y0 + y1) / 2, f'pinch   ΔT_min = {T_MIN_APP:.0f} K',
                ha='left', va='center', color=t['ink'], fontsize=10, alpha=alpha)
    ax.text(0.0, 1.06, 'hensmith  ·  composite curves of the quickstart system  '
            '(column + flash, 5 streams)', transform=ax.transAxes, color=t['ink'], fontsize=12.5)
    # Lower right is empty in every state (both curves end near 100 °C on
    # the right); upper left sits on the shaded band and the pinch corner.
    ax.legend(loc='lower right', fontsize=10, frameon=False, labelcolor=t['ink'])
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def encode(frames, path):
    sample = np.concatenate([frames[0], frames[N // 2]], axis=0)
    palette = Image.fromarray(sample).quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                               dither=Image.Dither.NONE)
    q = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / FPS),
              loop=0, optimize=False)
    return path.stat().st_size


def main():
    sys_, HXN, _ = build_quickstart_system('hero')
    sys_.simulate()
    table = problem_table_from_network(HXN, T_MIN_APP)
    curves = composite_curves(table, T_MIN_APP)
    gap = min_vertical_gap(*curves)
    assert gap >= T_MIN_APP - 1e-6, gap
    print(f'composite curves: hot 0 -> {curves[1][-1]:.4g}, cold {curves[3][0]:.4g} -> {curves[3][-1]:.4g} kJ/hr; '
          f'min approach {gap:.4f} K')
    for theme in ('light', 'dark'):
        for figsize in (FIGSIZE, FIGSIZE_SMALL):
            frames = [render(theme, curves, table, separation(i), figsize) for i in range(N)]
            gif = OUT / f'hero_{theme}.gif'
            size = encode(frames, gif)
            if size <= MAX_BYTES: break
            print(f'{gif.name}: {size / 1e6:.2f} MB > budget, re-rendering at {figsize}')
        assert size <= MAX_BYTES, size
        d_adjacent = max(np.abs(frames[i].astype(int) - frames[i + 1]).mean() for i in range(N - 1))
        d_wrap = np.abs(frames[0].astype(int) - frames[-1]).mean()
        assert d_wrap <= d_adjacent + 1e-9, (d_wrap, d_adjacent)     # seamless loop
        still = OUT / f'hero_{theme}_still.png'
        Image.fromarray(frames[0]).save(still)
        report(gif); report(still)
        print(f'  frames {N} at {frames[0].shape[1]}x{frames[0].shape[0]} px')


if __name__ == '__main__':
    main()
