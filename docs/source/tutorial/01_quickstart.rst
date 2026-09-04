Quickstart
==========

The canonical hensmith example is a small methanol/water system: a shortcut
distillation column (divided into a separately-costed rectifier and stripper)
whose condenser and reboiler are auxiliary exchangers, a cooler on each of the
column's two products, and a flash whose feed is heated by its own auxiliary
exchanger. A
:class:`~hensmith.HeatExchangerNetwork` facility added to that system
integrates every heating and cooling utility in it. Once biosteam is imported
the simulation takes about a second, and every number, table and figure below
is output of the code shown on this page.

Build the flowsheet
-------------------

.. literalinclude:: /../_demo_src/examples/ch01_quickstart.py
   :language: python
   :start-after: # [start:build]
   :end-before: # [end:build]
   :dedent:

``bst.settings.set_thermo`` fixes the chemicals and the property package used
for every stream that follows; ``cache=True`` reuses cached chemical objects
rather than recreating them. The two feeds are given as molar flow rates in
kmol/hr, in the order the chemicals were declared: ``feed1`` is 8000 water,
100 methanol and 25 glycerol, and ``feed2`` is 10000 water, 1000 methanol and
10 glycerol.

``D1`` separates methanol (light key) from water (heavy key) with a shortcut
method. Its condenser and reboiler are always auxiliary ``HXutility`` units of
the column, each with its own heat utility, and the network integrates each of
them like any other exchanger. ``is_divided=True`` costs the rectifier and
stripper as two separate vessels rather than one, and matches the
:class:`~hensmith.HeatExchangerNetwork` docstring example. ``D1_H1`` and
``D1_H2`` cool the bottoms product and the distillate to 300 K, and each is a
stand-alone exchanger in its own right. ``F1`` flashes ``feed2`` to 90 % vapor
at 101325 Pa; the heating its feed needs is carried by an auxiliary exchanger
of the flash, created when the flash is constructed and simulated as part of
the flash's own run.

Add the network
---------------

.. literalinclude:: /../_demo_src/examples/ch01_quickstart.py
   :language: python
   :start-after: # [start:network]
   :end-before: # [end:network]
   :dedent:

``T_min_app=5.`` is the minimum approach temperature in K: no synthesized
process exchanger is allowed to transfer heat across a temperature difference
smaller than this. It is the one thermodynamic knob that sets how much
integration is feasible, and it is revisited in :doc:`02_pinch_analysis`.
``bst.System.from_units`` builds the system from the units given, sorting the
process units into a simulation order and setting the facility aside; the
network is listed among the units but is not connected to any of them.

.. code-block:: python

   sys.diagram()

.. figure:: /_static/images/examples/tutorial_01_quickstart_flowsheet_light.png
   :figclass: only-light
   :width: 720
   :alt: Flowsheet of the quickstart system: feed1 enters the divided distillation column D1, whose distillate and bottoms product go to the heat exchangers D1_H2 and D1_H1; feed2 enters the flash F1, which produces vapor and liquid; the HXN heat exchanger network sits apart as an unconnected block.

   The quickstart system. ``HXN`` appears as a facility with no material
   streams: it neither receives nor produces process material, and only reads
   the heat utilities of the other units. Besides the two visible exchangers
   ``D1_H1`` and ``D1_H2``, the column's condenser and reboiler and the flash's
   feed exchanger are auxiliary exchangers inside their owner units, and the
   network integrates those as well -- 5 streams in total, of which 3 are
   auxiliaries.

.. figure:: /_static/images/examples/tutorial_01_quickstart_flowsheet_dark.png
   :figclass: only-dark
   :width: 720
   :alt: Flowsheet of the quickstart system: feed1 enters the divided distillation column D1, whose distillate and bottoms product go to the heat exchangers D1_H2 and D1_H1; feed2 enters the flash F1, which produces vapor and liquid; the HXN heat exchanger network sits apart as an unconnected block.

   The quickstart system. ``HXN`` appears as a facility with no material
   streams: it neither receives nor produces process material, and only reads
   the heat utilities of the other units. Besides the two visible exchangers
   ``D1_H1`` and ``D1_H2``, the column's condenser and reboiler and the flash's
   feed exchanger are auxiliary exchangers inside their owner units, and the
   network integrates those as well -- 5 streams in total, of which 3 are
   auxiliaries.

Because :class:`~hensmith.HeatExchangerNetwork` is a BioSTEAM ``Facility``, it
is simulated only after every process unit in the system has converged, so it
always works with final duties. Its ``network_priority = -2`` is the lowest of
the standard facilities, which places it first among them: the chilled water
package, cooling tower and boiler that follow are then sized on the loads the
network has already reduced.

Simulate
--------

.. literalinclude:: /../_demo_src/examples/ch01_quickstart.py
   :language: python
   :start-after: # [start:simulate]
   :end-before: # [end:simulate]
   :dedent:

Converging the process units comes first; the network's pinch analysis,
synthesis and costing all run afterwards, in its costing step. The original
streams and heat exchangers of the system are left untouched -- the stream
copies and the new exchangers the network creates live in a separate flowsheet
named ``sys_HXN``, after the ID of the system.

Read the savings
----------------

.. literalinclude:: /../_demo_src/examples/ch01_quickstart.py
   :language: python
   :start-after: # [start:results]
   :end-before: # [end:results]
   :dedent:

.. literalinclude:: /_generated/ch01_results.txt
   :language: text

Every utility row of this table is a *difference*, not a requirement: the
facility's ``heat_utilities`` are the utilities of the new network summed with
the reversed original ones, that is new - original. A negative cost is
therefore a saving, and all three agents show one here: -389 USD/hr of low
pressure steam, -211 USD/hr of chilled water and -5.98 USD/hr of cooling water,
for a total utility cost of -605 USD/hr. The flow rows are negative for all
three agents as well, so less of each utility is consumed. The duty rows carry
the sign convention of the agent they belong to: the steam duty is
-6.32e+07 kJ/hr, exactly the reduction in heating load reported below, while
the chilled and cooling water duties are positive (4.21e+07 and
1.79e+07 kJ/hr) because cooling duties are negative to begin with, so a
positive difference again means less cooling.

The capital rows are differences too, but clipped at zero. Both the network's
``installed_costs['Heat exchangers']`` and ``purchase_costs['Heat exchangers']``
are ``max(0, new - original)``, the *added* exchanger cost -- 1.11e+06 USD
here. Clipping at zero means a network
whose exchangers happen to be cheaper than the ones it replaces is reported as
adding nothing, rather than as a capital credit.

.. literalinclude:: /../_demo_src/examples/ch01_quickstart.py
   :language: python
   :start-after: # [start:loads]
   :end-before: # [end:loads]
   :dedent:

.. literalinclude:: /_generated/ch01_loads.txt
   :language: text

The heating load falls from 3.609e+08 to 2.977e+08 kJ/hr, a reduction of
17.5 %; the ratio of the two, 0.82, is the value checked by the
:class:`~hensmith.HeatExchangerNetwork` docstring example. The cooling load
falls from 6.201e+07 to 1.96e+06 kJ/hr, a reduction of 96.8 %: nearly all of
the cooling duty of this system can be recovered into a stream that needed
heating. Four process exchangers do that work, and they are the ``new_HXs`` of
the network. The energy balance error, -1.8e-11 %, checks that the synthesized
network moves exactly as much heat as the original one; it is computed on
every synthesis and compared against ``acceptable_energy_balance_error``.

Draw the pinch diagram
----------------------

.. literalinclude:: /../_demo_src/examples/ch01_quickstart.py
   :language: python
   :start-after: # [start:pinch]
   :end-before: # [end:pinch]
   :dedent:

.. figure:: /_static/images/examples/tutorial_01_quickstart_pinch_diagram.png
   :class: white-bg
   :width: 100%
   :alt: Pinch diagram of the synthesized quickstart network: two blue cold streams above three red hot streams, joined by four vertical process-exchanger connectors labelled 3.34E4, 5.03E6, 3.71E7 and 1.79E7 kJ/hr, all to the right of the dashed pinch line, with hot utility circles at the outlet of both cold streams and one cold utility circle on the hot stream D1_H1 (bottoms_product).

   The synthesized network, read as a pinch diagram. The two cold streams
   (blue, drawn left to right) are ``0`` ``D1 - reboiler`` and ``1``
   ``F1 - heat_exchanger (feed2)``; the three hot streams (red, drawn right to
   left) are ``2`` ``D1_H1 (bottoms_product)``, ``3``
   ``D1 - condenser (vapor)`` and ``4`` ``D1_H2 (distillate)``. Each stream is
   annotated with its inlet and outlet temperature and enthalpy flow. The four
   vertical connectors are the process exchangers, each labelled with its duty
   in kJ/hr: 3.34E4 between streams 1 and 4, 5.03E6 between 1 and 2, 3.71E7
   between 0 and 2, and 1.79E7 between 1 and 3. Columns are ordered so that a
   stream meets its exchangers in flow direction, which is why stream 1 reads
   3.34E4, 5.03E6, 1.79E7 from left to right. The dashed line is the pinch,
   separating the cold-side design on its left from the hot-side design on its
   right; all four exchangers of this network lie on the hot side. The open
   circles are the utility exchangers that finish each stream: a hot utility
   (red) at the outlet of both cold streams, and a single cold utility (blue)
   on stream 2. Streams 3 and 4 carry no circle because their remaining
   cooling duty is zero -- process heat exchange alone brings them to their
   outlet temperature.

Where to next
-------------

- :doc:`02_pinch_analysis` -- where the utility targets above come from, and
  what ``T_min_app`` changes.
- :doc:`03_network_anatomy` -- the exchangers and stream life cycles behind
  the diagram, unit by unit.
- :doc:`04_configuring` -- the constructor options of the facility, and
  applying it to a larger system.
- :doc:`../concepts` -- the pinch concepts and terminology used throughout.
- :doc:`../API/api` -- the full API reference.
