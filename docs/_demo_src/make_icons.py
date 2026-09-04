# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Render the four landing-page card icons, light and dark (512 x 512, transparent).

    python docs/_demo_src/make_icons.py
    -> docs/source/_static/images/icons/<name>_{light,dark}.png
       name in: getting-started, concepts, api, contributing
"""
import _common
from _common import IMAGES, THEMES, report
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from PIL import Image

INK = {'light': '#1f2a2e', 'dark': '#e8eef0'}
LW = 7
OUT = IMAGES / 'icons'


def exchanger(ax, theme, x0, y0, w, h, lw=LW):
    """Cold arrow (top, ->) over hot arrow (bottom, <-) joined by a connector.

    The arrowheads are sized with ``mutation_scale`` (about 2.2 x lw long, as
    in make_logo.py): matplotlib's default 10 pt scale makes them narrower
    than the shaft. The node rings are filled with the theme background so
    the connector does not show through them at 100 px."""
    t = THEMES[theme]
    head = 'head_length=0.6,head_width=0.3'
    ax.annotate('', xy=(x0 + w, y0 + h), xytext=(x0, y0 + h),
                arrowprops=dict(arrowstyle='-|>,' + head, color=t['cold'], lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=3.7 * lw))
    ax.annotate('', xy=(x0, y0), xytext=(x0 + w, y0),
                arrowprops=dict(arrowstyle='-|>,' + head, color=t['hot'], lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=3.7 * lw))
    ax.plot([x0 + w / 2] * 2, [y0, y0 + h], '-o', color=INK[theme], lw=lw * 0.8,
            mfc=t['bg'], mec=INK[theme], mew=lw * 0.5, ms=16, zorder=5)


def getting_started(ax, theme):
    exchanger(ax, theme, 1.3, 3.0, 6.0, 4.0)
    ax.add_patch(Polygon([[7.9, 4.2], [7.9, 5.8], [9.2, 5.0]], closed=True,
                         facecolor=INK[theme], edgecolor='none'))


def concepts(ax, theme):
    t = THEMES[theme]
    ax.plot([1.2, 3.8, 6.0, 8.6], [3.6, 6.0, 6.5, 8.6], '-', color=t['hot'], lw=LW, solid_capstyle='round')
    ax.plot([2.6, 5.0, 7.4, 9.0], [1.4, 4.2, 5.2, 7.2], '-', color=t['cold'], lw=LW, solid_capstyle='round')
    # dT_min bracket at the closest approach
    ax.annotate('', xy=(5.0, 6.25), xytext=(5.0, 4.2),
                arrowprops=dict(arrowstyle='<->,head_length=0.35,head_width=0.2', color=INK[theme], lw=LW * 0.6,
                                mutation_scale=24))


def api(ax, theme):
    ax.text(0.9, 5.0, '{', fontsize=88, family='DejaVu Sans', color=INK[theme], ha='left', va='center')
    ax.text(9.1, 5.0, '}', fontsize=88, family='DejaVu Sans', color=INK[theme], ha='right', va='center')
    exchanger(ax, theme, 3.1, 3.7, 3.8, 2.6, lw=LW * 0.8)


def contributing(ax, theme):
    t = THEMES[theme]
    ax.plot([3.5, 3.5], [1.2, 8.8], '-', color=INK[theme], lw=LW, solid_capstyle='round')
    branch = Path([(3.5, 2.6), (6.6, 3.4), (6.6, 5.0), (6.6, 6.6), (3.5, 7.4)],
                  [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(branch, facecolor='none', edgecolor=t['cold'], lw=LW, capstyle='round'))
    for x, y, c in ((3.5, 2.6, INK[theme]), (3.5, 7.4, INK[theme]), (6.6, 5.0, t['hot'])):
        ax.plot([x], [y], 'o', mfc=THEMES[theme]['bg'], mec=c, mew=LW * 0.55, ms=20, zorder=5)


ICONS = {'getting-started': getting_started, 'concepts': concepts, 'api': api, 'contributing': contributing}


def main():
    for name, draw in ICONS.items():
        for theme in ('light', 'dark'):
            fig = plt.figure(figsize=(2.56, 2.56), dpi=200)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal'); ax.axis('off')
            draw(ax, theme)
            path = OUT / f'{name}_{theme}.png'
            fig.savefig(path, transparent=True, dpi=200)
            plt.close(fig)
            assert Image.open(path).size == (512, 512), path
            report(path)


if __name__ == '__main__':
    main()
