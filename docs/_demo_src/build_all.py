# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Regenerate every docs asset, in order, one Python process at a time.

    python docs/_demo_src/build_all.py

Each script runs as a sequential subprocess of this interpreter (importing
biosteam writes numba's on-disk cache; two processes at once corrupt it), so
never run two of these scripts concurrently by hand either. Stops at the
first failure. Needs graphviz ``dot`` on PATH for the flowsheet figures.
"""
import os
import sys
import time
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = [
    'make_logo.py',
    'make_icons.py',
    'examples/ch01_quickstart.py',
    'examples/ch02_pinch_analysis.py',
    'examples/ch03_network_anatomy.py',
    'examples/ch04_configuring.py',
    'make_hero_gif.py',
    'build_demo.py',      # consumes the chapter-01 captures and figures, and the logo
    'make_poster.py',     # consumes the chapter-01 pinch diagram and the logo
]


def main():
    env = {**os.environ, 'NUMBA_DISABLE_JIT': '1', 'DISABLE_PREFERENCES': '1'}
    t0 = time.time()
    for rel in SCRIPTS:
        t = time.time()
        print(f'=== {rel}', flush=True)
        subprocess.run([sys.executable, str(HERE / rel)], check=True, cwd=str(ROOT), env=env)
        print(f'=== {rel} done in {time.time() - t:.0f} s', flush=True)
    print(f'all docs assets regenerated in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
