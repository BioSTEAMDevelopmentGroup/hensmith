Anatomy of a synthesized network
================================

:doc:`01_quickstart` reported what the network saved and :doc:`02_pinch_analysis`
where those targets come from. Neither looked at the network itself. This
chapter opens it: the flowsheet and ``System`` the facility builds for its own
exchangers, the naming that makes them readable, the per-stream life cycles
that record which exchangers each stream passes through and in what order, the
per-stream pinch temperatures that split the design into its two sides, the
options of the pinch diagram, and the cost and utility accounting the facility
reports.

Every number and figure below is output of the code shown on this page. The
system built here is chapter 1's build repeated verbatim, so this page runs on
its own.

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:imports]
   :end-before: # [end:imports]
   :dedent:

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:system]
   :end-before: # [end:system]
   :dedent:

The network flowsheet
---------------------

By default, the facility never touches the streams or exchangers of the system
it integrates. Everything it synthesizes -- stream copies, process exchangers,
utility exchangers -- is created in a flowsheet of its own, and gathered into a
``System`` of its own.

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:flowsheet]
   :end-before: # [end:flowsheet]
   :dedent:

.. literalinclude:: /_generated/ch03_flowsheet.txt
   :language: text

``HXN.HXN_flowsheet`` is that flowsheet, a ``bst.Flowsheet`` named after the
system it belongs to, ``sys.ID + '_HXN'`` -- here ``sys_HXN``. It is created
each time the facility synthesizes a network, and its registries are cleared
before synthesis, so the network's IDs neither collide with the original
flowsheet's nor accumulate across repeated simulations.

``HXN.HXN_sys`` is the ``bst.System`` built from the synthesized units. It is
constructed from a network of those units and named after the flowsheet,
``sys_HXN``, in whose system registry it is registered -- so
``HXN.HXN_flowsheet.system.sys_HXN`` resolves to it, just as the exchangers
resolve through ``HXN.HXN_flowsheet.unit``. It is an ordinary ``System``
holding the nine units listed on the third line. They are listed in the order the system
simulates them, which is derived from the rewired stream connections rather
than from the order synthesis created them: a hot-side exchanger is synthesized
before the cold-side exchangers that feed it, so synthesis order would leave it
with stale inlets. Recycle loops in the network are converged by the system's
own fixed-point solver.

The IDs carry the whole topology. A process exchanger is an ``HXprocess`` named
``HX_<cold>_<hot>_hs`` when the match was made in the hot-side pass and
``HX_<hot>_<cold>_cs`` when it was made in the cold-side pass -- note that the
two orders differ: a cold-side exchanger names its hot stream first, a hot-side
one its cold stream first. A utility
exchanger is an ``HXutility`` named ``Util_<index>_hs`` for a cold stream,
which is finished by a hot utility above the pinch, and ``Util_<index>_cs`` for
a hot stream, which is finished by a cold utility below it. The indices are
stream indices: positions in the rearranged utility list of
:func:`~hensmith.synthesize_network`, cold streams first and then hot ones, as
described in :doc:`02_pinch_analysis`. The stream copies are named after the
exchanger they touch, ``s_<index>__<exchanger>`` on the way in and
``<exchanger>__s_<index>`` on the way out.

All four process exchangers of this network end in ``_hs``: every match was
made in the hot-side pass, which is the same fact as the pinch diagram of
:doc:`01_quickstart` showing all four connectors to the right of the pinch
line.

.. code-block:: python

   HXN.HXN_sys.diagram()

.. figure:: /_static/images/examples/tutorial_03_hxn_flowsheet_light.png
   :figclass: only-light
   :width: 720
   :alt: Flowsheet of the synthesized quickstart network: nine units, the four process heat exchangers HX_1_4_hs, HX_0_2_hs, HX_1_2_hs and HX_1_3_hs drawn as two-inlet nodes feeding the utility exchangers Util_0_hs and Util_1_hs (heating), Util_2_cs (cooling), and the grey zero-duty nodes Util_3_cs and Util_4_cs.

   The synthesized network as its own flowsheet, ``sys_HXN``. The four
   two-inlet nodes are the process exchangers; each takes one cold and one hot
   stream copy and passes both on. The five single-inlet nodes are the utility
   exchangers that finish each stream: ``Util_0_hs`` and ``Util_1_hs`` are
   heating, ``Util_2_cs`` is cooling, and ``Util_3_cs`` and ``Util_4_cs`` are
   drawn grey because they carry no utility at all -- their inlet and outlet
   enthalpies are equal, 2.47e+06 and 7.18e+05 kJ/hr, so streams 3 and 4 are
   brought to their outlet states by process heat exchange alone. The stream
   names show the wiring: ``s_1__HX_1_4_hs`` enters ``HX_1_4_hs`` carrying
   stream 1, and ``HX_1_4_hs__s_1`` leaves it and enters ``HX_1_2_hs``.

.. figure:: /_static/images/examples/tutorial_03_hxn_flowsheet_dark.png
   :figclass: only-dark
   :width: 720
   :alt: Flowsheet of the synthesized quickstart network: nine units, the four process heat exchangers HX_1_4_hs, HX_0_2_hs, HX_1_2_hs and HX_1_3_hs drawn as two-inlet nodes feeding the utility exchangers Util_0_hs and Util_1_hs (heating), Util_2_cs (cooling), and the grey zero-duty nodes Util_3_cs and Util_4_cs.

   The synthesized network as its own flowsheet, ``sys_HXN``. The four
   two-inlet nodes are the process exchangers; each takes one cold and one hot
   stream copy and passes both on. The five single-inlet nodes are the utility
   exchangers that finish each stream: ``Util_0_hs`` and ``Util_1_hs`` are
   heating, ``Util_2_cs`` is cooling, and ``Util_3_cs`` and ``Util_4_cs`` are
   drawn grey because they carry no utility at all -- their inlet and outlet
   enthalpies are equal, 2.47e+06 and 7.18e+05 kJ/hr, so streams 3 and 4 are
   brought to their outlet states by process heat exchange alone. The stream
   names show the wiring: ``s_1__HX_1_4_hs`` enters ``HX_1_4_hs`` carrying
   stream 1, and ``HX_1_4_hs__s_1`` leaves it and enters ``HX_1_2_hs``.

Stream life cycles
------------------

A flowsheet of nine units says what exists; it does not say, for one stream,
what happens to it. That is what a :class:`~hensmith.StreamLifeCycle` records.
The facility builds one per stream after synthesis, aligned with
``HXN.original_heat_exchangers``, and stores them in
``HXN.stream_life_cycles``.

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:life_cycles]
   :end-before: # [end:life_cycles]
   :dedent:

.. literalinclude:: /_generated/ch03_life_cycles.txt
   :language: text

A life cycle has the attributes ``index``, the stream's index; ``name``,
``s_<index>``; ``cold``, ``True`` for a heated stream and ``False`` for a cooled
one; and ``life_cycle``, the list of stages. It is recovered from IDs alone --
the exchangers whose ID contains ``_<index>_``, keeping those whose inlet at the
matching position has an ID containing ``s_<index>_``. The
stages are then sorted by inlet enthalpy, ascending for a cold stream and
descending for a hot one, which is flow direction in both cases since a cold
stream gains enthalpy as it goes and a hot stream loses it.

Read stream 1, the longest life cycle here: it passes ``HX_1_4_hs``,
``HX_1_2_hs`` and ``HX_1_3_hs`` and then its utility exchanger ``Util_1_hs``,
its enthalpy rising 0, 3.34e+04, 5.06e+06, 2.3e+07 and finally 2.79e+08 kJ/hr.
Stream 2 runs the other way, 4.52e+07 to 8.12e+06 to 3.1e+06 kJ/hr through two
process exchangers and then to 1.14e+06 kJ/hr through ``Util_2_cs``. Each
stage's outlet enthalpy is the next stage's inlet enthalpy because the facility
rewires the units after synthesis, making each stage's outlet stream the inlet
of the following stage. Streams 3 and 4 end on a stage whose inlet and outlet
enthalpies are equal, 2.47e+06 and 7.18e+05 kJ/hr: their utility exchangers
have nothing left to do.

One stage on its own:

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:stage]
   :end-before: # [end:stage]
   :dedent:

.. literalinclude:: /_generated/ch03_stage.txt
   :language: text

A ``LifeStage`` holds only two things, ``unit`` and ``index``; everything else
is a property read from the unit when accessed. ``index`` is the position of
this stream in the exchanger's ``ins`` and ``outs`` -- 0 or 1 for an
``HXprocess``, always 0 for an ``HXutility``. Here it is 0, because a hot-side
process exchanger is constructed with its cold stream first. ``s_in`` and
``s_out`` are ``unit.ins[index]`` and ``unit.outs[index]``, and ``H_in`` and
``H_out`` are their enthalpies, so a life cycle always reflects the current
state of the network rather than a snapshot taken at synthesis. This stage
takes stream 1 from 0 to 3.338e+04 kJ/hr.

Per-stream pinch temperatures
-----------------------------

The pinch analysis produces three arrays indexed like the life cycles, which
the facility stores as ``HXN.inlet_Ts``, ``HXN.outlet_Ts`` and
``HXN.pinch_Ts``. The first two are each stream's inlet temperature and its
quenched outlet temperature (:doc:`02_pinch_analysis`). The third is the
temperature at which a stream is handed from the cold-side design to the
hot-side design -- the point at which the synthesizer splits it in two.

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:pinch_Ts]
   :end-before: # [end:pinch_Ts]
   :dedent:

.. literalinclude:: /_generated/ch03_pinch_Ts.txt
   :language: text

The process pinch of this system is a single shifted temperature, 298.15 K
(:doc:`02_pinch_analysis`), which stands for two real ones: 25.0 °C for cold
streams and, ``T_min_app`` higher, 30.0 °C for hot streams. Each stream is then
classified against the pinch temperature of its own kind.

- A stream that reaches the pinch is split there, and its ``pinch_T`` is the
  pinch temperature of its kind. Stream 2 crosses it, 98.2 to 26.8 °C, and
  stream 1 enters exactly at it, 25.0 °C; they are split at 30.0 and 25.0 °C
  respectively.
- A stream whose outlet stops short of the pinch never reaches it, and its
  ``pinch_T`` is its own *outlet* temperature: it lies wholly on one side, and
  the split is a formality at its far end. Streams 3 and 4 are hot streams that
  cool only to 64.9 and 64.8 °C, far above the 30.0 °C hot-stream pinch, and
  those outlet temperatures are exactly what ``pinch_Ts`` reports for them.
- A stream whose *inlet* is already past the pinch is likewise not split, and
  its ``pinch_T`` is its inlet temperature. Stream 0 is a cold stream entering
  at 33.2 °C, above the 25.0 °C cold-stream pinch, so its ``pinch_T`` is
  33.2 °C.

That last clause also catches isothermal and non-monotone streams -- a stream
whose outlet lies on the wrong side of its inlet for the sign of its duty, such
as a cold stream whose equilibrium outlet ends up cooler than it entered.
Rather than spread a point load across the cascade, these get
``pinch_T = T_in`` too, which assigns the whole of their duty to a single side
of the design: the hot side for a cold stream, the cold side for a hot one.

Reading the pinch diagram
-------------------------

:func:`~hensmith.plot_pinch_diagram` draws the life cycles above. Called as
``HXN.plot_pinch_diagram``, the facility supplies the life cycles, the inlet
and outlet temperatures, the hot-side and cold-side exchanger lists, its
``Qmin`` and the original exchangers, and forwards ``file`` and every other
keyword argument. Those remaining arguments are:

``show_units``, ``show_auxiliary_units`` and ``show_stream_IDs``
    The three parts of each stream's label, built from its *original* heat
    exchanger and joined as ``<unit> - <auxiliary> (<inlet stream ID>)``:
    respectively the ID of the unit owning that exchanger, the name of the
    exchanger within its owner when it is an auxiliary one (``condenser`` and
    the like), and the ID of that exchanger's inlet stream. All three default
    to ``True``; with all three off no label is drawn and the original
    exchangers are not needed, while turning any of them on without those
    exchangers raises ``ValueError``.

``show_legend``
    Adds a legend of the six symbols -- cold stream, hot stream, process heat
    exchange, hot utility, cold utility, pinch -- below the axes.

``ax``
    Draws into a ``matplotlib`` axes provided by the caller instead of creating
    a figure; the figure returned is then the one that axes belongs to.

``file`` and ``dpi``
    Save the figure to a path, at the given resolution.

Turning off the labels and the legend leaves the quantitative skeleton of the
diagram:

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:minimal]
   :end-before: # [end:minimal]
   :dedent:

.. figure:: /_static/images/examples/tutorial_03_pinch_diagram_minimal.png
   :class: white-bg
   :width: 100%
   :alt: Minimal pinch diagram of the synthesized quickstart network, drawn into a caller-provided axes with the stream labels and the legend suppressed: two blue cold streams indexed 0 and 1 running left to right above three red hot streams indexed 2, 3 and 4 running right to left, T and H columns on both sides reading 33.2 °C and 5.38E6 kJ/hr to 99.5 °C and 6.92E7 kJ/hr for stream 0, four vertical process-exchanger connectors with boxed duties between the streams, a dashed pinch line with all four connectors on its hot side, and the Cold side and Hot side captions along the bottom.

   The same network as the pinch diagram of :doc:`01_quickstart`, with
   ``show_stream_IDs=False`` and ``show_legend=False`` -- and the two unit-label
   options off as well -- drawn into an axes created by the caller. Only the
   stream annotations and the legend are gone: the ``T`` and ``H`` columns on
   both sides remain, and so do the boxed exchanger duties in the ``ΔH`` row,
   the bold stream index at the inlet of each stream, the dashed pinch line and
   the ``Cold side`` and ``Hot side`` captions. The columns read off the life
   cycles above: stream 1 enters at 25.0 °C with 0.00E0 kJ/hr and leaves at
   95.9 °C with 2.79E8 kJ/hr, and the first connector it meets carries the
   3.34E4 kJ/hr of its first stage. The four duties are the same four as in
   :doc:`01_quickstart`.

Exchanger columns are ordered independently on each side of the pinch, by
``_order_exchanger_columns``. Every stream's stage order is a chain of
precedence constraints between the exchangers it meets -- reversed for hot
streams, which are drawn right to left -- and a topological sort of that graph
(Kahn's algorithm, ties broken by the order the exchangers were synthesized in)
lays them out so that every stream meets its exchangers in flow direction. That
is why stream 1 reads its three connectors left to right in exactly the order
of its life cycle. Constraints that contradict each other, which would require
some stream to flow backwards, cannot be satisfied by any ordering; the
synthesis order is then used unchanged.

Energy balance and cost accounting
----------------------------------

What the facility reports about itself is one consistency check and a set of
differences.

.. literalinclude:: /../_demo_src/examples/ch03_network_anatomy.py
   :language: python
   :start-after: # [start:accounting]
   :end-before: # [end:accounting]
   :dedent:

.. literalinclude:: /_generated/ch03_accounting.txt
   :language: text

``energy_balance_percent_error`` compares the heat the synthesized network
moves with the heat the original exchangers moved. The numerator is twice the
absolute duty of every new process exchanger -- twice, because each process
match takes that heat off one stream and puts it onto another -- plus the
process-side duty of the new utility exchangers, each agent's ``duty``
multiplied by its ``heat_transfer_efficiency``. The denominator is the same
product summed over the original utilities. The ratio less one, times 100, is
the reported percentage: -1.8e-11 % here, which is round-off.

The tolerance it is checked against, ``acceptable_energy_balance_error``, is a
*fraction*, not a percentage: the class attribute is ``0.02``, that is 2 %, and
the constructor argument of the same name defaults to ``None``, meaning that
the class value is kept. The check is on the absolute error. Exceeding it warns
with a ``RuntimeWarning`` naming the error and the tolerance, or raises a
``RuntimeError`` instead if the class attribute ``raise_energy_balance_error``
is set to ``True``.

The costs are differences, clipped at zero. ``original_purchase_costs`` is the
purchase cost of each *original* exchanger, one entry per stream, 3.365e+05 USD
in total here; ``new_purchase_costs_HXp`` and ``new_purchase_costs_HXu`` are
the same for the synthesized process and utility exchangers, 4.73e+05 and
2.096e+05 USD. The facility's own ``purchase_costs['Heat exchangers']`` -- and
its identical ``baseline_purchase_costs`` entry -- is ``max(0, new - original)``
over those three sums, 4.73e+05 + 2.096e+05 - 3.365e+05 = 3.461e+05 USD. Its
``installed_costs['Heat exchangers']`` is formed exactly the same way from the
installed costs of the same exchangers rather than their purchase costs, and
is the larger figure here, 1.114e+06 USD. Clipping at zero means a network
whose exchangers happen to be cheaper than the ones they replace is reported as
adding nothing rather than as a capital credit; and if the synthesis produced
no process exchangers at all, both entries are set to zero and the facility
reports no utilities either.

The utilities are differences too. ``original_utility_costs`` holds the
original heat utilities summed by agent -- reversed in sign, since they are the
very objects that were negated to form the difference -- and
``new_utility_costs`` holds the new utility exchangers' utilities summed by
agent. ``HXN.heat_utilities`` is the sum of the two, that is new - original,
which is why every cost printed above is negative: -388.6 USD/hr of low
pressure steam, -210.6 USD/hr of chilled water and -5.976 USD/hr of cooling
water are savings. The duties carry the sign convention of their agent, so the
steam duty is negative, -6.321e+07 kJ/hr, while the chilled and cooling water
duties are positive, 4.212e+07 and 1.794e+07 kJ/hr, because cooling duties are
negative to begin with and a positive difference again means less of them.
Setting ``replace_unit_heat_utilities=True`` moves this reporting onto the
process units instead, as :doc:`04_configuring` describes.

Where to next
-------------

- :doc:`04_configuring` -- the constructor options of the facility, what
  ``T_min_app`` and ``ignored`` change, and a ten-stream network.
- :doc:`../concepts` -- the pinch concepts, the synthesis heuristics, and what
  a synthesized network is and is not guaranteed to be.
- :doc:`../API/api` -- the full API reference.
