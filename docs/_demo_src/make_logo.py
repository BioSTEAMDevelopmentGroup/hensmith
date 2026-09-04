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
PT_PER_UNIT = 72   # render() maps 12 data units onto a 12-inch figure: 1 unit = 1 in = 72 pt


def ring_radius(ms):
    """Radius of a hollow 'o' marker's stroke centreline, in data units (as
    in make_icons.py): a butt-capped stroke narrower than ``ms`` that ends
    here lies entirely under the ring's edge stroke, so it meets the ring
    with no gap and never shows inside the hollow."""
    return ms / 2 / PT_PER_UNIT


def draw_mark(ax, theme, x0=0.0, y0=0.0, s=1.0):
    """The exchanger glyph at (x0, y0), scaled by s: the shafts span x0..x0+2.4
    and each arrowhead overshoots its end by 0.2, so the glyph occupies a
    2.8 x 1.8 box starting at x0 - 0.2. The tips are set 0.2 past the shaft
    ends (and the head kept at mutation_scale=34) so the stroked head base
    clears the circle instead of butting against it.

    The node rings are hollow and every stroke stops at the ring it meets
    (the construction of make_icons.exchanger), so the mark is the same
    open-circle glyph on any ground. The PNGs are transparent and sit on the
    navbar, the demo chrome and the poster, none of which is the page
    background the rings used to be filled with: filled with THEMES['dark']
    ['bg'] the dark variant showed dark discs where the light one showed
    open rings."""
    t = THEMES[theme]
    ink = INK[theme]
    lw = 14 * s
    ms, mew = 52 * s, lw * 0.55
    r = ring_radius(ms)
    cx, y_cold, y_hot = x0 + 1.2 * s, y0 + 1.5 * s, y0 + 0.3 * s
    head = 'head_length=0.9,head_width=0.45'

    def arrow(x_from, x_to, y, color):
        # butt-capped: FancyArrowPatch's default round cap would poke lw/2 into the hollow
        ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                    arrowprops=dict(arrowstyle='-|>,' + head, color=color, lw=lw,
                                    shrinkA=0, shrinkB=0, mutation_scale=34, capstyle='butt'))

    def tail(x_end, x_ring, y, color):
        # butt-capped so it stops at the ring; a dot restores the round outer tip
        ax.plot([x_end, x_ring], [y, y], '-', color=color, lw=lw, solid_capstyle='butt')
        ax.plot([x_end], [y], 'o', mfc=color, mec='none', ms=lw)

    tail(x0, cx - r, y_cold, t['cold'])
    arrow(cx + r, x0 + 2.6 * s, y_cold, t['cold'])
    tail(x0 + 2.4 * s, cx + r, y_hot, t['hot'])
    arrow(cx - r, x0 - 0.2 * s, y_hot, t['hot'])
    ax.plot([cx, cx], [y_hot + r, y_cold - r], '-', color=ink, lw=lw * 0.8, solid_capstyle='butt')
    ax.plot([cx, cx], [y_hot, y_cold], 'o', mfc='none', mec=ink, mew=mew, ms=ms, zorder=5)


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
