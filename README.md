# hensmith
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://hensmith.readthedocs.io/en/latest/)

**H**eat **E**xchanger **N**etwork **S**ynthesis, **M**odeling, **I**ntegration, **T**hermodynamics, and **H**euristics.

`hensmith` provides the automated heat exchanger network (HXN) synthesis
facility previously distributed as `biosteam.facilities.hxn`, including
`HeatExchangerNetwork`, pinch/problem-table analysis, and pinch diagram
plotting for [BioSTEAM](https://github.com/BioSTEAMDevelopmentGroup/biosteam)
systems.

```python
import biosteam as bst  # hensmith units plug into BioSTEAM systems
from hensmith import HeatExchangerNetwork
```

## Documentation

Full documentation is at [hensmith.readthedocs.io](https://hensmith.readthedocs.io/en/latest/):
the [quickstart](https://hensmith.readthedocs.io/en/latest/tutorial/01_quickstart.html),
the [tutorial](https://hensmith.readthedocs.io/en/latest/tutorial/index.html), and the
[API reference](https://hensmith.readthedocs.io/en/latest/API/api.html).

[![Watch the quickstart demo](https://raw.githubusercontent.com/BioSTEAMDevelopmentGroup/hensmith/master/docs/source/_static/images/examples/quickstart_demo_poster.png)](https://hensmith.readthedocs.io/en/latest/_static/quickstart_demo.html)

## Citation

If you use hensmith in your work, please cite it as:

> Bhagwat, S. S., & Cortés-Peña, Y. R. (2026). HENSMITH: Heat Exchanger Network Synthesis, Modeling, Integration, Thermodynamics, and Heuristics (Version v0.1.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22416033

## License

UIUC/NCSA open-source license — see `LICENSE.txt`.
