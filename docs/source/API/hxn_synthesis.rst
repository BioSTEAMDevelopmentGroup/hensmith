Pinch analysis and synthesis (hensmith.hxn_synthesis)
=====================================================

.. currentmodule:: hensmith

``hensmith.hxn_synthesis`` holds the machinery behind
:class:`HeatExchangerNetwork`: :func:`problem_table` builds the
temperature-interval heat cascade of a set of process streams on their
temperature-enthalpy curves and locates the pinch, :func:`synthesize_network`
plans an unsplit network from the pinch outward on the same curves -- one that
reaches the minimum energy requirement (MER) targets whenever its search finds
one, or, with ``stream_splitting=True``, a network with stream splits where a
side of the pinch needs them -- and realizes it as BioSTEAM exchangers (and
the splits as :class:`~hensmith.hxn_synthesis.StreamSplit` splitter chains and
mixers), :class:`StreamLifeCycle`
records the exchangers each stream ends up passing through, and
:func:`plot_pinch_diagram` draws the result. All four are usable on their own,
without a :class:`HeatExchangerNetwork` instance; :doc:`../concepts` explains
the method.

.. autofunction:: problem_table

.. autoclass:: ProblemTable
   :no-members:

.. autofunction:: synthesize_network

.. autoclass:: StreamLifeCycle
   :members:

.. autoclass:: hensmith.hxn_synthesis.LifeStage
   :no-members:

.. autoclass:: hensmith.hxn_synthesis.StreamSplit
   :no-members:

.. autofunction:: plot_pinch_diagram

.. note::

   **Internals.** ``hensmith.hxn_synthesis.temperature_interval_pinch_analysis``
   (the first step of :func:`synthesize_network`: preparing the process streams
   and running the problem table on them), ``hensmith.hxn_synthesis.pinch_state``
   and ``hensmith.hxn_synthesis.load_duties`` (standalone helpers that split a
   stream at a pinch temperature; the synthesis itself plans on the stream
   curves and does not use them) are public in name only: they are not exported
   by ``hensmith``, and are not part of the supported API. Neither are the
   private modules ``hensmith._curves`` (the stream temperature-enthalpy curves),
   ``hensmith._planner`` (the MER planner) and ``hensmith._splitting`` (stream
   splitting: the split candidates and their theory, in its module docstring).
   Their signatures and behavior may change without notice.
   :class:`~hensmith.hxn_synthesis.StreamSplit` and
   :class:`~hensmith.hxn_synthesis.LifeStage` are not exported by ``hensmith``
   either; they are documented because synthesis results and life cycles hold
   them.
