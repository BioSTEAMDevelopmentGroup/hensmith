# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Shared helpers for the docs asset scripts in ``docs/_demo_src``.

Never imported by the library and never executed on Read the Docs: the
scripts that import this module generate the figures, GIFs and captured
text outputs that are committed under ``docs/source``. Import it *before*
biosteam so the environment variables below take effect.
"""
import os
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')     # long Windows paths break numba's cache; docs never need JIT
os.environ.setdefault('DISABLE_PREFERENCES', '1')   # ignore the user's saved thermosteam preferences: deterministic diagrams
import io
import sys
import contextlib
import pathlib
import matplotlib
matplotlib.use('Agg')
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent            # docs/_demo_src
ROOT = HERE.parents[1]                                     # repo root
SOURCE = ROOT / 'docs' / 'source'
STATIC = SOURCE / '_static'
IMAGES = STATIC / 'images'
GENERATED = SOURCE / '_generated'
for _d in (IMAGES / 'logo', IMAGES / 'icons', IMAGES / 'demo',
           IMAGES / 'examples', GENERATED):
    _d.mkdir(parents=True, exist_ok=True)

DPI = 200
# The library's own pinch-diagram colours (hxn_synthesis.plot_pinch_diagram).
PALETTE = dict(cold='#2e6db4', hot='#d62728', cold_bg='#e6f0fa', hot_bg='#fbe9e7', ink='#000000')
# 'bg' must match pydata-sphinx-theme's page background (GIFs are opaque).
THEMES = {
    'light': dict(bg='#ffffff', ink='#1f2a2e', cold='#2e6db4', hot='#d62728',
                  cold_bg='#e6f0fa', hot_bg='#fbe9e7', grid='#d9dee3', muted='#6b7680'),
    'dark':  dict(bg='#14181e', ink='#d7dce8', cold='#5b9be0', hot='#f0665f',
                  cold_bg='#1b2733', hot_bg='#2e1f22', grid='#2a323c', muted='#98a3b3'),
}


def build_quickstart_system(flowsheet, T_min_app=5.):
    """The canonical HeatExchangerNetwork doctest system, with the same unit
    definitions as the ``build``/``network`` regions of
    ``examples/ch01_quickstart.py`` (the tutorial shows that code; this copy
    is for the scripts that do not, e.g. make_hero_gif.py)."""
    import biosteam as bst
    from hensmith import HeatExchangerNetwork
    bst.settings.set_thermo(['Water', 'Methanol', 'Glycerol'], cache=True)
    bst.main_flowsheet.set_flowsheet(flowsheet)
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
    HXN = HeatExchangerNetwork('HXN', T_min_app=T_min_app)
    sys = bst.System.from_units('sys', units=[D1, D1_H1, D1_H2, F1, HXN])
    return sys, HXN, dict(D1=D1, D1_H1=D1_H1, D1_H2=D1_H2, F1=F1, feed1=feed1, feed2=feed2)


def report(path):
    print(f'wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)')


def capture(name, text):
    """Write ``_generated/<name>.txt`` with '\\n' line endings."""
    path = GENERATED / f'{name}.txt'
    text = str(text).replace('\r\n', '\n').rstrip('\n') + '\n'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    report(path)
    return path


@contextlib.contextmanager
def capturing(name):
    """Capture everything printed inside the block into ``_generated/<name>.txt``
    (and echo it to the real stdout)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield
    text = buffer.getvalue()
    sys.stdout.write(text)
    capture(name, text)


def write_summary(name, mapping):
    """Write ``key = value`` lines (strings) for build_demo.py and grep checks."""
    capture(name, '\n'.join(f'{k} = {v}' for k, v in mapping.items()))


def save(fig, name, subdir='examples', **kwargs):
    kwargs.setdefault('dpi', DPI)
    kwargs.setdefault('bbox_inches', 'tight')
    path = IMAGES / subdir / name
    fig.savefig(path, **kwargs)
    report(path)
    return path


def save_diagram(system, name, theme):
    """Render ``system.diagram()`` to ``_static/images/examples/<name>`` in the
    given theme. Uses display=False (returns the graphviz Digraph) and pipes
    the PNG bytes ourselves: biosteam's save_digraph rejects any file path
    containing a '.' when a format is given."""
    import biosteam as bst
    if theme == 'light':
        bst.preferences.light_mode()
    else:
        bst.preferences.dark_mode(bg=THEMES['dark']['bg'])
    digraph = system.diagram(format='png', display=False)
    path = IMAGES / 'examples' / name
    path.write_bytes(digraph.pipe(format='png'))
    report(path)
    return path


def problem_table_from_network(HXN, T_min_app):
    """The problem table of a simulated network's original streams, built
    exactly as tests/test_hxn_regression.py::mer_targets does."""
    from hensmith.hxn_synthesis import problem_table
    hus = HXN.original_heat_utils
    streams_inlet = [hu.unit.ins[0].copy() for hu in hus]
    streams_quenched = [hu.unit.outs[0].copy() for hu in hus]
    for s in streams_quenched: s.vle(H=s.H, P=s.P)
    is_hot = [hu.duty < 0 for hu in hus]
    return problem_table(streams_inlet, streams_quenched, is_hot, T_min_app)


def min_vertical_gap(hot_T, hot_H, cold_T, cold_H):
    """Smallest T_hot(H) - T_cold(H) over the H range both composite curves
    span. Curves are piecewise linear in H, so the minimum is at a vertex;
    at a vertical (equal-H) run the hot curve takes its lower T and the cold
    curve its higher T (conservative)."""
    hot_H, hot_T = np.asarray(hot_H), np.asarray(hot_T)
    cold_H, cold_T = np.asarray(cold_H), np.asarray(cold_T)
    _, i = np.unique(hot_H, return_index=True)               # first = lower T
    hH, hT = hot_H[i], hot_T[i]
    _, j = np.unique(cold_H[::-1], return_index=True)        # last = higher T
    j = cold_H.size - 1 - j
    cH, cT = cold_H[j], cold_T[j]
    lo, hi = max(hH[0], cH[0]), min(hH[-1], cH[-1])
    Hs = np.concatenate([hH, cH])
    Hs = Hs[(Hs >= lo) & (Hs <= hi)]
    return float(np.min(np.interp(Hs, hH, hT) - np.interp(Hs, cH, cT)))
