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

The icons are transparent line art: the node rings are hollow (``mfc='none'``)
and every stroke stops at the ring it meets, so the icons do not depend on the
colour behind them (pydata-sphinx-theme's card body is not the page
background). Only the ink and palette colours differ between the themes.
"""
import math
import _common
from _common import IMAGES, THEMES, report
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from PIL import Image

INK = {'light': '#1f2a2e', 'dark': '#e8eef0'}
LW = 7
OUT = IMAGES / 'icons'
FIG_IN = 2.56                          # figure edge in inches: 512 px at 200 dpi
UNITS = 10                             # axis span in data units
PT_PER_UNIT = FIG_IN * 72 / UNITS      # 18.432 pt: marker sizes are in points, geometry in units


def ring_radius(ms):
    """Radius of a hollow 'o' marker's stroke centreline, in data units.

    A butt-capped stroke of width < ``ms`` that ends here lies entirely under
    the ring's edge stroke, so it meets the ring with no gap and never enters
    the hollow (a stroke drawn through a hollow ring collapses to a blob at
    100 px)."""
    return ms / 2 / PT_PER_UNIT


def ring(ax, x, y, color, ms, mew):
    ax.plot([x], [y], 'o', mfc='none', mec=color, mew=mew, ms=ms, zorder=5)


def exchanger(ax, theme, x0, y0, w, h, lw=LW):
    """Cold arrow (top, ->) over hot arrow (bottom, <-) joined by a connector,
    with a hollow node ring where the connector meets each arrow.

    The arrowheads are sized with ``mutation_scale`` (about 2.2 x lw long, as
    in make_logo.py): matplotlib's default 10 pt scale makes them narrower
    than the shaft. Each arrow is a tail segment plus an arrowhead segment
    that stop at the ring, and the connector runs from ring edge to ring
    edge (see ``ring_radius``). All of them are butt-capped: unlike other
    patches, FancyArrowPatch defaults to round caps, which would poke lw/2
    past the segment's start into the hollow."""
    t = THEMES[theme]
    head = 'head_length=0.6,head_width=0.3'
    ms, mew = 16, lw * 0.5
    r = ring_radius(ms)
    cx, y_cold, y_hot = x0 + w / 2, y0 + h, y0

    def arrow(x_from, x_to, y, color):
        ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                    arrowprops=dict(arrowstyle='-|>,' + head, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=3.7 * lw, capstyle='butt'))

    ax.plot([x0, cx - r], [y_cold] * 2, '-', color=t['cold'], lw=lw, solid_capstyle='butt')
    arrow(cx + r, x0 + w, y_cold, t['cold'])
    ax.plot([x0 + w, cx + r], [y_hot] * 2, '-', color=t['hot'], lw=lw, solid_capstyle='butt')
    arrow(cx - r, x0, y_hot, t['hot'])
    ax.plot([cx] * 2, [y_hot + r, y_cold - r], '-', color=INK[theme], lw=lw * 0.8, solid_capstyle='butt')
    for y in (y_hot, y_cold):
        ring(ax, cx, y, INK[theme], ms, mew)


def getting_started(ax, theme):
    exchanger(ax, theme, 1.3, 3.0, 6.0, 4.0)
    ax.add_patch(Polygon([[7.9, 4.2], [7.9, 5.8], [9.2, 5.0]], closed=True,
                         facecolor=INK[theme], edgecolor='none'))


def concepts(ax, theme):
    t = THEMES[theme]
    ax.plot([1.2, 3.8, 6.0, 8.6], [3.6, 6.0, 6.5, 8.6], '-', color=t['hot'], lw=LW, solid_capstyle='round')
    ax.plot([2.6, 5.0, 7.4, 9.0], [1.4, 4.2, 5.2, 7.2], '-', color=t['cold'], lw=LW, solid_capstyle='round')
    # dT_min bracket between the curves
    ax.annotate('', xy=(5.0, 6.25), xytext=(5.0, 4.2),
                arrowprops=dict(arrowstyle='<->,head_length=0.35,head_width=0.2', color=INK[theme], lw=LW * 0.6,
                                mutation_scale=24))


def api(ax, theme):
    ax.text(0.9, 5.0, '{', fontsize=88, family='DejaVu Sans', color=INK[theme], ha='left', va='center')
    ax.text(9.1, 5.0, '}', fontsize=88, family='DejaVu Sans', color=INK[theme], ha='right', va='center')
    exchanger(ax, theme, 3.1, 3.7, 3.8, 2.6, lw=LW * 0.8)


def _bisect(f, lo, hi, n=60):
    """Root of a monotone ``f`` on [lo, hi] (``f(lo)`` and ``f(hi)`` differ in sign)."""
    sign_lo = f(lo) > 0
    for _ in range(n):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == sign_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def trim_quad(P0, P1, P2, r0, r2):
    """Control points of the quadratic Bezier ``P0 P1 P2`` shortened so it
    starts at distance ``r0`` from ``P0`` and ends at distance ``r2`` from
    ``P2`` (the rings at its ends, see ``ring_radius``).

    The parameters ``a`` and ``b`` of those two points are found by bisection
    (the distance from an endpoint is monotone over that endpoint's half of
    these gentle curves); the sub-curve on [a, b] is then exact by de
    Casteljau: its end control points are the curve at ``a`` and ``b`` and
    its middle control point is the curve's polar form at (a, b)."""
    def B(t):
        return tuple((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
                     for p0, p1, p2 in zip(P0, P1, P2))
    a = _bisect(lambda t: math.dist(B(t), P0) - r0, 0.0, 0.5)
    b = _bisect(lambda t: r2 - math.dist(B(t), P2), 0.5, 1.0)
    Q1 = tuple((1 - a) * (1 - b) * p0 + ((1 - a) * b + a * (1 - b)) * p1 + a * b * p2
               for p0, p1, p2 in zip(P0, P1, P2))
    return B(a), Q1, B(b)


def contributing(ax, theme):
    t = THEMES[theme]
    ink = INK[theme]
    ms, mew = 20, LW * 0.55
    r = ring_radius(ms)
    x, y_lo, y_hi = 3.5, 2.6, 7.4
    fork = (6.6, 5.0)
    # trunk: three segments that stop at the two ink nodes, round tips at the ends
    for ya, yb in ((1.2, y_lo - r), (y_lo + r, y_hi - r), (y_hi + r, 8.8)):
        ax.plot([x, x], [ya, yb], '-', color=ink, lw=LW, solid_capstyle='butt')
    ax.plot([x, x], [1.2, 8.8], 'o', mfc=ink, mec='none', ms=LW)
    # branch: two quadratic Beziers through the hot node, each trimmed to its rings
    for P0, P1, P2 in (((x, y_lo), (6.6, 3.4), fork), (fork, (6.6, 6.6), (x, y_hi))):
        Q0, Q1, Q2 = trim_quad(P0, P1, P2, r, r)
        path = Path([Q0, Q1, Q2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor=t['cold'], lw=LW, capstyle='butt'))
    for (px, py), c in (((x, y_lo), ink), ((x, y_hi), ink), (fork, t['hot'])):
        ring(ax, px, py, c, ms, mew)


ICONS = {'getting-started': getting_started, 'concepts': concepts, 'api': api, 'contributing': contributing}


def main():
    for name, draw in ICONS.items():
        for theme in ('light', 'dark'):
            fig = plt.figure(figsize=(FIG_IN, FIG_IN), dpi=200)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, UNITS); ax.set_ylim(0, UNITS); ax.set_aspect('equal'); ax.axis('off')
            draw(ax, theme)
            path = OUT / f'{name}_{theme}.png'
            fig.savefig(path, transparent=True, dpi=200)
            plt.close(fig)
            assert Image.open(path).size == (512, 512), path
            report(path)


if __name__ == '__main__':
    main()
