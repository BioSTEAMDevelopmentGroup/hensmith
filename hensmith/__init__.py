# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
hensmith: automated heat exchanger network synthesis, modeling, integration,
thermodynamics, and heuristics for BioSTEAM systems.
"""
__version__ = '0.1.2'

# IMPORTANT: never star-import biosteam at module scope anywhere in this
# package (or in any module it imports during initialization). biosteam
# lists 'HeatExchangerNetwork' in its __all__, and the name is only bound
# into biosteam at the end of this module (see below), so a
# `from biosteam import *` executed while hensmith is still initializing
# raises AttributeError. Plain `import biosteam as bst` is safe (and is what
# the modules below use); biosteam's tests/test_hensmith_integration.py
# guards this invariant.

from ._heat_exchanger_network import *
from .hxn_synthesis import *

from . import _heat_exchanger_network
from . import hxn_synthesis

__all__ = (
    *_heat_exchanger_network.__all__,
    *hxn_synthesis.__all__,
)

# %% Register with biosteam
#
# biosteam imports hensmith at the very end of its own __init__ so that
# bst.HeatExchangerNetwork works out of the box, and relies on this block to
# bind the name: in the hensmith-first import order biosteam's __init__ runs
# while this module is still initializing (HeatExchangerNetwork is not defined
# yet), so binding here, last, is what works for either order. The name must
# stay out of biosteam.facilities.__all__, which biosteam star-imports before
# this block can run.
import biosteam as _bst
_bst.HeatExchangerNetwork = _bst.facilities.HeatExchangerNetwork = HeatExchangerNetwork
del _bst
