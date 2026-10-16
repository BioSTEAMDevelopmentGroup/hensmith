API Reference
=============

.. currentmodule:: hensmith

hensmith exports thirteen names. :class:`HeatExchangerNetwork` is the
BioSTEAM facility most users need; the functions and classes of
``hensmith.hxn_synthesis`` are the pinch analysis and synthesis machinery
behind it, usable on their own; and those of ``hensmith.targeting`` target
utilities, area, units, capital and total annualized cost, find the
``T_min_app`` of the least targeted total annualized cost, and compare a
network with its targets.

.. autosummary::

   HeatExchangerNetwork
   problem_table
   ProblemTable
   synthesize_network
   StreamLifeCycle
   plot_pinch_diagram
   PinchTargets
   supertarget
   Supertargets
   network_report
   plot_composite_curves
   plot_grand_composite_curve
   plot_supertargets

.. toctree::
   :maxdepth: 2

   heat_exchanger_network
   hxn_synthesis
   targeting

.. note::

   Importing hensmith binds the facility into biosteam as
   ``bst.HeatExchangerNetwork`` (and ``bst.facilities.HeatExchangerNetwork``),
   whichever package is imported first; it is the same class object.
