Targeting and supertargeting (hensmith.targeting)
=================================================

.. currentmodule:: hensmith

``hensmith.targeting`` says what any heat exchanger network of a set of
process streams can achieve, before (or instead of) synthesizing one.
:class:`PinchTargets` holds the problem table of the streams and derives from
it the composite and grand composite curves, the targets of several utilities
priced by biosteam (:class:`~hensmith.targeting.UtilityTarget`), the balanced
composite curves, and the targets of area, units, capital cost and total
annualized cost. :func:`supertarget` sweeps the minimum approach temperature
for the least total annualized cost and returns :class:`Supertargets`;
:func:`network_report` compares a synthesized network with its targets; and
:func:`plot_composite_curves`, :func:`plot_grand_composite_curve` and
:func:`plot_supertargets` draw them. A facility's targets come from
:meth:`HeatExchangerNetwork.targets`. Nothing here changes the synthesis.
:doc:`../tutorial/05_targeting` shows every function on the quickstart system,
and :doc:`../concepts` explains the methods and their limits.

.. autoclass:: PinchTargets
   :no-members:

.. automethod:: PinchTargets.from_heat_utilities

.. automethod:: PinchTargets.at

.. automethod:: PinchTargets.composite_curves

.. automethod:: PinchTargets.grand_composite_curve

.. automethod:: PinchTargets.utility_targets

.. automethod:: PinchTargets.balanced_composite_curves

.. automethod:: PinchTargets.balanced_grand_composite_curve

.. automethod:: PinchTargets.area_target

.. automethod:: PinchTargets.units_target

.. automethod:: PinchTargets.capital_cost_target

.. automethod:: PinchTargets.summary

.. autoclass:: hensmith.targeting.UtilityTarget
   :no-members:

.. autofunction:: supertarget

.. autoclass:: Supertargets
   :no-members:

.. automethod:: Supertargets.to_frame

.. autofunction:: network_report

.. autofunction:: plot_composite_curves

.. autofunction:: plot_grand_composite_curve

.. autofunction:: plot_supertargets

.. note::

   :class:`~hensmith.targeting.UtilityTarget` is not exported by
   ``hensmith``; it is documented because
   :meth:`PinchTargets.utility_targets` returns it. The private helpers of
   ``hensmith.targeting`` (those whose names start with an underscore) are not
   part of the supported API.
