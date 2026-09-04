# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Render the hensmith logo (mark + wordmark) and the mark alone, light and dark.

    python docs/_demo_src/make_logo.py
    -> docs/source/_static/images/logo/logo_hensmith_{light,dark}.png   (2000 px wide)
       docs/source/_static/images/logo/mark_hensmith_{light,dark}.png   (600 x 600)
"""
import io
import _common
from _common import IMAGES, THEMES, report
import matplotlib.pyplot as plt
from PIL import Image

INK = {'light': '#1f2a2e', 'dark': '#e8eef0'}
OUT = IMAGES / 'logo'


def draw_mark(ax, theme, x0=0.0, y0=0.0, s=1.0):
    """The exchanger glyph in a 2.4 x 1.8 box at (x0, y0), scaled by s."""
    t = THEMES[theme]
    lw = 14 * s
    ax.annotate('', xy=(x0 + 2.4 * s, y0 + 1.5 * s), xytext=(x0, y0 + 1.5 * s),
                arrowprops=dict(arrowstyle='-|>,head_length=0.9,head_width=0.45',
                                color=t['cold'], lw=lw, shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(x0, y0 + 0.3 * s), xytext=(x0 + 2.4 * s, y0 + 0.3 * s),
                arrowprops=dict(arrowstyle='-|>,head_length=0.9,head_width=0.45',
                                color=t['hot'], lw=lw, shrinkA=0, shrinkB=0))
    ax.plot([x0 + 1.2 * s] * 2, [y0 + 0.3 * s, y0 + 1.5 * s], '-o', color=INK[theme],
            lw=lw * 0.8, mfc=t['bg'], mec=INK[theme], mew=lw * 0.55, ms=52 * s, zorder=5,
            solid_capstyle='round')


def render(theme, with_text):
    fig = plt.figure(figsize=(12, 3), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.set_aspect('equal'); ax.axis('off')
    draw_mark(ax, theme, x0=0.4, y0=0.6)
    if with_text:
        ax.text(3.35, 1.45, 'hensmith', fontsize=118, weight='bold',
                family='DejaVu Sans', color=INK[theme], va='center', ha='left')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, dpi=200)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert('RGBA')
    return im.crop(im.getbbox())


def main():
    for theme in ('light', 'dark'):
        logo = render(theme, True)
        logo = logo.resize((2000, round(2000 * logo.height / logo.width)), Image.LANCZOS)
        assert logo.width == 2000 and 0.12 < logo.height / logo.width < 0.40, logo.size
        p = OUT / f'logo_hensmith_{theme}.png'; logo.save(p); report(p)
        mark = render(theme, False)
        side = 600
        mark = mark.resize((520, round(520 * mark.height / mark.width)), Image.LANCZOS)
        canvas = Image.new('RGBA', (side, side), (0, 0, 0, 0))
        canvas.paste(mark, ((side - mark.width) // 2, (side - mark.height) // 2), mark)
        assert canvas.size == (side, side)
        p = OUT / f'mark_hensmith_{theme}.png'; canvas.save(p); report(p)


if __name__ == '__main__':
    main()
