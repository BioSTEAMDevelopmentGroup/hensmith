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
__version__ = '0.1.1'

# IMPORTANT: never star-import biosteam at module scope anywhere in this
# package (or in any module it imports during initialization). biosteam
# lists 'HeatExchangerNetwork' in its __all__ as a lazy (PEP 562) re-export
# from hensmith, so a `from biosteam import *` executed while hensmith is
# still initializing would resolve the name against the partially
# initialized hensmith module and raise ImportError. Plain
# `import biosteam as bst` is safe (and is what the modules below use);
# biosteam's tests/test_hensmith_integration.py guards this invariant.

from ._heat_exchanger_network import *
from .hxn_synthesis import *

from . import _heat_exchanger_network
from . import hxn_synthesis

__all__ = (
    *_heat_exchanger_network.__all__,
    *hxn_synthesis.__all__,
)
