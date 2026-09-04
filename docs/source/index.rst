Heat Exchanger Network Synthesis
================================

.. currentmodule:: hensmith

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorial/index
   concepts
   API/api
   contributing/index

.. raw:: html

   <div style="text-align:center;max-width:760px;margin:0 auto 1.25rem;">
     <picture class="only-dark">
       <source media="(prefers-reduced-motion: reduce)"
               srcset="_static/images/demo/hero_dark_still.png">
       <img class="only-dark" src="_static/images/demo/hero_dark.gif"
            width="1000" height="360" style="width:100%;height:auto;"
            alt="Hot and cold composite curves of a five-stream distillation and flash system converging to a 5 K pinch; the overlap is recovered heat, the overhangs are the remaining hot and cold utility.">
     </picture>
     <picture class="only-light">
       <source media="(prefers-reduced-motion: reduce)"
               srcset="_static/images/demo/hero_light_still.png">
       <img class="only-light" src="_static/images/demo/hero_light.gif"
            width="1000" height="360" style="width:100%;height:auto;"
            alt="Hot and cold composite curves of a five-stream distillation and flash system converging to a 5 K pinch; the overlap is recovered heat, the overhangs are the remaining hot and cold utility.">
     </picture>
   </div>

.. rst-class:: landing-lede

hensmith (**H**\ eat **E**\ xchanger **N**\ etwork **S**\ ynthesis,
**M**\ odeling, **I**\ ntegration, **T**\ hermodynamics, and **H**\ euristics)
is the automated heat exchanger network synthesis facility for BioSTEAM
systems: :class:`HeatExchangerNetwork` is a BioSTEAM ``Facility`` that
performs a pinch analysis on every heating and cooling utility in a system,
synthesizes a network of process exchangers, and reports the utility savings
and added capital cost as part of the system's techno-economic analysis.
Watch it run in the `Quickstart`_ demo below.

Quickstart
----------

The canonical example is a small methanol/water system: a shortcut
distillation column whose condenser and reboiler are auxiliary exchangers, a
cooler on each of the column's two products, and a flash whose feed is heated
by its own auxiliary exchanger. Adding a :class:`HeatExchangerNetwork` to that
system cuts its heating utility by 17.5 % and its cooling utility by 96.8 %
with 4 process exchangers, and the network's installed cost joins the
system's techno-economic analysis like any other unit's. The interactive demo
below runs it end to end: build the flowsheet, add the network, simulate, and
inspect the synthesized exchangers and the pinch diagram.

.. raw:: html

   <div style="margin:1.5rem auto;max-width:1120px;">
     <iframe src="_static/quickstart_demo.html"
             title="hensmith quickstart — interactive demo"
             loading="lazy"
             style="width:100%;height:840px;border:1px solid rgba(128,128,128,0.25);border-radius:14px;display:block;"></iframe>
     <p style="text-align:center;font-size:0.85rem;margin-top:0.4rem;opacity:0.8;">
       Interactive quickstart demo &mdash;
       <a href="_static/quickstart_demo.html" target="_blank" rel="noopener">open in a new tab</a>.
     </p>
   </div>

The :doc:`quickstart chapter <tutorial/01_quickstart>` walks through the same
example line by line, and the full :doc:`tutorial <tutorial/index>` continues
with the pinch analysis behind it, the anatomy of the synthesized network, and
how to configure the network for a larger system.


.. grid:: 1 2 3 4
    :class-row: sd-align-major-center


    .. grid-item-card:: Getting Started
       :text-align: center
       :link: tutorial/index
       :link-type: doc
       :padding: 1

       .. image:: _static/images/icons/getting-started_dark.png
          :height: 100
          :class: only-dark
          :align: center

       .. image:: _static/images/icons/getting-started_light.png
          :height: 100
          :class: only-light
          :align: center

       Tutorials on hensmith


    .. grid-item-card:: Key Concepts
       :text-align: center
       :link: concepts
       :link-type: doc
       :padding: 1

       .. image:: _static/images/icons/concepts_dark.png
          :height: 100
          :class: only-dark
          :align: center

       .. image:: _static/images/icons/concepts_light.png
          :height: 100
          :class: only-light
          :align: center

       Pinch analysis and the synthesis heuristics


    .. grid-item-card:: API Reference
       :text-align: center
       :link: API/api
       :link-type: doc
       :padding: 1

       .. image:: _static/images/icons/api_dark.png
          :height: 100
          :class: only-dark
          :align: center

       .. image:: _static/images/icons/api_light.png
          :height: 100
          :class: only-light
          :align: center

       Every public class and function


    .. grid-item-card:: Contributing
       :text-align: center
       :link: contributing/index
       :link-type: doc
       :padding: 1

       .. image:: _static/images/icons/contributing_dark.png
          :height: 100
          :class: only-dark
          :align: center

       .. image:: _static/images/icons/contributing_light.png
          :height: 100
          :class: only-light
          :align: center

       Development, tests, and authors


Installation
------------

Get the latest version of hensmith from
`PyPI <https://pypi.org/project/hensmith/>`__. If you have an installation of
Python with pip, simply install it with:

.. code-block:: bash

   $ pip install hensmith

.. note::

   hensmith requires ``biosteam>=2.54.0``. Until that biosteam release is on
   PyPI, install the BioSTEAM stack from GitHub HEAD first — the same commands
   the hensmith continuous-integration workflow uses — and then install
   hensmith without letting pip resolve its dependencies:

   .. code-block:: bash

      $ pip install --no-cache-dir git+https://github.com/CalebBell/fluids.git
      $ pip install --no-cache-dir git+https://github.com/CalebBell/chemicals.git
      $ pip install --no-cache-dir git+https://github.com/CalebBell/thermo.git
      $ pip install --no-cache-dir git+https://github.com/BioSTEAMDevelopmentGroup/thermosteam.git
      $ pip install --no-cache-dir git+https://github.com/BioSTEAMDevelopmentGroup/biosteam.git
      $ pip install --no-deps hensmith

To get the git version, use:

.. code-block:: bash

   $ git clone https://github.com/BioSTEAMDevelopmentGroup/hensmith

Or download directly from the
`GitHub page <https://github.com/BioSTEAMDevelopmentGroup/hensmith>`__.

Importing hensmith binds the facility into biosteam as
``bst.HeatExchangerNetwork``, so ``import hensmith`` and
``import biosteam as bst`` can be run in either order and
``bst.HeatExchangerNetwork`` is :class:`hensmith.HeatExchangerNetwork` in both.

Citation
--------

hensmith builds on BioSTEAM. If you use it in your work, please cite the
BioSTEAM paper:

    Cortes-Peña, Y., Kumar, D., Singh, V., & Guest, J. S. (2020). BioSTEAM: A
    Fast and Flexible Platform for the Design, Simulation, and Techno-Economic
    Analysis of Biorefineries under Uncertainty. *ACS Sustainable Chemistry &
    Engineering*, 8(8), 3302–3310. https://doi.org/10.1021/acssuschemeng.9b07040

The pinch-analysis and network-synthesis methodology that hensmith implements
follows chapter 9 of:

    Seider, W. D., Lewin, D. R., Seader, J. D., Widagdo, S., Gani, R., & Ng,
    M. K. (2017). *Product and Process Design Principles*. Wiley. Heat
    Exchanger Networks (Chapter 9).
