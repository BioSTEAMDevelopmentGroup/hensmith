Pinch analysis and synthesis (hensmith.hxn_synthesis)
=====================================================

.. currentmodule:: hensmith

``hensmith.hxn_synthesis`` holds the machinery behind
:class:`HeatExchangerNetwork`: :func:`problem_table` builds the
temperature-interval heat cascade of a set of process streams and locates
the pinch, :func:`synthesize_network` adds the sequential, heuristic
matching of hot and cold streams on each side of that pinch,
:class:`StreamLifeCycle` records the exchangers each stream ends up passing
through, and :func:`plot_pinch_diagram` draws the result. All four are
usable on their own, without a :class:`HeatExchangerNetwork` instance.

.. autofunction:: problem_table

.. autoclass:: ProblemTable
   :no-members:

.. autofunction:: synthesize_network

.. autoclass:: StreamLifeCycle
   :members:

.. autoclass:: hensmith.hxn_synthesis.LifeStage
   :no-members:

.. autofunction:: plot_pinch_diagram

.. note::

   **Internals.** ``hensmith.hxn_synthesis.temperature_interval_pinch_analysis``,
   ``hensmith.hxn_synthesis.pinch_state`` and
   ``hensmith.hxn_synthesis.load_duties`` are public in name only: they are
   steps of :func:`synthesize_network`, are not exported by ``hensmith``, and
   are not part of the supported API. Their signatures and behavior may change
   without notice.
