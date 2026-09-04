API Reference
=============

.. currentmodule:: hensmith

hensmith exports six names. :class:`HeatExchangerNetwork` is the BioSTEAM
facility most users need; the functions and classes of
``hensmith.hxn_synthesis`` are the pinch analysis and synthesis machinery
behind it, usable on their own.

.. autosummary::

   HeatExchangerNetwork
   problem_table
   ProblemTable
   synthesize_network
   StreamLifeCycle
   plot_pinch_diagram

.. toctree::
   :maxdepth: 2

   heat_exchanger_network
   hxn_synthesis

.. note::

   Importing hensmith binds the facility into biosteam as
   ``bst.HeatExchangerNetwork`` (and ``bst.facilities.HeatExchangerNetwork``),
   whichever package is imported first; it is the same class object.
