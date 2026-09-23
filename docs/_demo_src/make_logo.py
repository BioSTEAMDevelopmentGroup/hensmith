# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Derive the hensmith logo (mark + wordmark) and the mark alone from the
committed master artwork ``hensmith_logo_source.png``.

    python docs/_demo_src/make_logo.py
    -> docs/source/_static/images/logo/logo_hensmith.png   (2000 px wide)
       docs/source/_static/images/logo/mark_hensmith.png   (600 x 600)

One logo serves both themes: the artwork is coloured (a white-faced hen whose
comb is a pinch-diagram trace, and a two-tone wordmark), so there is no
light/dark variant. The logo is the master trimmed to its opaque extent; the
mark is everything left of the first fully transparent column (the gap
between the hen's beak and the wordmark), trimmed and centred on a
transparent square.
"""
import numpy as np
import _common
from _common import HERE, IMAGES, report
from PIL import Image

SOURCE = HERE / 'hensmith_logo_source.png'
OUT = IMAGES / 'logo'


def trim(im):
    return im.crop(im.getbbox())


def main():
    master = trim(Image.open(SOURCE).convert('RGBA'))
    logo = master.resize((2000, round(2000 * master.height / master.width)), Image.LANCZOS)
    assert logo.width == 2000 and 0.12 < logo.height / logo.width < 0.40, logo.size
    p = OUT / 'logo_hensmith.png'; logo.save(p); report(p)

    opaque = (np.asarray(master)[:, :, 3] > 0).any(axis=0)
    gap = int(np.argmin(opaque))            # first fully transparent column
    assert 0 < gap < master.width // 2, gap
    mark = trim(master.crop((0, 0, gap, master.height)))
    side, fit = 600, 560
    scale = fit / max(mark.size)
    mark = mark.resize((round(mark.width * scale), round(mark.height * scale)), Image.LANCZOS)
    canvas = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    canvas.paste(mark, ((side - mark.width) // 2, (side - mark.height) // 2), mark)
    p = OUT / 'mark_hensmith.png'; canvas.save(p); report(p)


if __name__ == '__main__':
    main()
