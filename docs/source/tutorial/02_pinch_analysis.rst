Pinch analysis and targets
==========================

The utility targets behind :doc:`01_quickstart` are not fitted numbers: they
come from a temperature-interval heat cascade -- the problem table -- built on
exactly the streams the network is synthesized from. This chapter follows that
computation end to end: how hensmith turns the heat utilities of a simulated
system into hot and cold process streams, what :func:`~hensmith.problem_table`
does with them, how the resulting :class:`~hensmith.ProblemTable` can be
redrawn as composite curves and as a grand composite curve, and how the network
synthesized in chapter 1 compares with the targets those curves define.

Every number and figure below is output of the code shown on this page. The
system built here is chapter 1's build repeated verbatim, so this page runs on
its own.

Streams as heat utilities
-------------------------

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:imports]
   :end-before: # [end:imports]
   :dedent:

A pinch analysis needs process streams, but what a simulated biosteam system
carries is heat utilities: one ``HeatUtility`` per heat exchanger, each with a
duty and a reference to the exchanger that owns it. hensmith collects them with
``bst.process_tools.heat_exchanger_utilities_from_units`` over the units in
scope and keeps the ones with a nonzero duty, minus any belonging to units
passed as ``ignored``. Every heating or cooling requirement in the system --
including the auxiliary exchangers inside columns and flashes -- therefore
becomes exactly one process stream, and the list of those utilities is kept on
the facility as ``HXN.original_heat_utils``. That attribute holds the list in
the order the synthesizer rearranges it into -- every heating utility first,
then every cooling one -- so a utility's position in it is the stream index
used throughout the network and its stream life cycle.

Each utility becomes a pair of end states -- an inlet and a quenched outlet --
of one process stream. The inlet is the exchanger's own inlet,
``hu.unit.ins[0]``; the outlet is the exchanger's outlet quenched to
equilibrium at its own enthalpy, ``s.vle(H=s.H, P=s.P)``, so that the end state
the analysis works from is an equilibrium state and the enthalpy path between
the two end temperatures is thermodynamically consistent -- which matters for
the phase-changing streams that dominate this system. The sign of the duty says
which kind of stream it is: ``duty > 0`` means the exchanger heats its stream,
which is a *cold* stream in pinch terms, and ``duty < 0`` means it cools it, a
*hot* stream. That distinction is the ``is_hot`` argument of
:func:`~hensmith.problem_table`, and this is the same construction the facility
performs internally before synthesis, and that ``tests/test_hxn_regression.py``
uses to compute its reference targets.

The problem table
-----------------

The smallest instructive case is a two-stream threshold problem, the example
in the :func:`~hensmith.problem_table` docstring: a stream of water is cooled
while a slightly smaller stream of water is heated over an overlapping
temperature range.

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:threshold]
   :end-before: # [end:threshold]
   :dedent:

.. literalinclude:: /_generated/ch02_threshold.txt
   :language: text

Hot streams are shifted *down* by ``T_min_app`` and cold streams are left
alone. On that shifted scale, two streams at equal temperature are in reality
exactly ``T_min_app`` apart, so heat may be cascaded from any shifted
temperature to any lower one without ever violating the minimum approach. In
the grid above, the hot stream's two end temperatures appear shifted down by
the 5 K approach as 395 and 295 K, while the cold stream's appear unshifted as
390 and 300 K; the grid ``Ts`` is the union of all such end temperatures,
sorted descending.

Between consecutive grid temperatures, each monotone stream contributes the
enthalpy it releases or absorbs over that interval, evaluated at its *real*
temperature and clipped to its own enthalpy range, with a positive sign for hot
streams and a negative one for cold. Because the grid always contains a
stream's own end temperatures, those contributions telescope exactly to the
stream's duty: no heat is created or lost by the discretization. Streams with
no temperature span of their own -- an isothermal condenser, or a stream whose
outlet moves against its duty, such as a reboiler outlet at equilibrium -- are
not spread over intervals at all; they enter as *point loads* at their shifted
outlet temperature.

Cascading those contributions down the grid, with no hot utility supplied,
gives the heat *leaving* each boundary, the ``residual`` field. Feasibility
must hold for the heat *arriving* at each boundary too -- the residual before
that boundary's point loads are applied -- because a point source sitting at a
grid temperature cannot serve a sink above it. The worst deficit over both
flows is the minimum hot utility, and its location is the pinch. Here the
cascade never goes negative, which is what a threshold problem means:
``hot_util_load`` is 0.0, all of the surplus leaves as 1445550.0 kJ/hr of cold
utility, and ``pinch_T`` reports the top of the grid, 395.0 K.

The quickstart system is the same computation on five streams:

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:system]
   :end-before: # [end:system]
   :dedent:

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:table]
   :end-before: # [end:table]
   :dedent:

.. literalinclude:: /_generated/ch02_table.txt
   :language: text

The five streams produce a grid of ten shifted temperatures, from 372.6 K down
to 295 K. Ten boundaries out of five streams is itself a statement about the
streams: a monotone stream contributes both of its end temperatures and a
point load contributes only one, so every stream here is monotone. Exactly
equal boundaries would be merged into one grid entry; the two entries that
print as 333 are distinct values that differ by less than the 0.01 K shown --
the condenser's shifted outlet and the distillate cooler's shifted inlet. Four
of those boundaries sit within about half a Kelvin of each other around 333 K
-- 333.53, 333 twice, and 332.98. The column's condenser spans the upper two,
333.53 down to 333, which is 65.4 down to 64.9 °C on the real scale; the
distillate cooler ``D1_H2`` takes the stream from there, so its own upper
boundary is that second 333, and the outlet the analysis works with lies only
0.02 K below it, at 332.98, because the cooler removes just 3.34e+04 kJ/hr.
Neither is a point load: both are spread over intervals like any other stream,
only very narrow ones, and the cooler's load is too small to see on the curves
below. The targets are 2.828e+08 kJ/hr of hot utility and 1.936e+06 kJ/hr of
cold utility, and the pinch is at 298.15 K on the shifted scale. Since hot
streams were shifted down by the 5 K approach, that one shifted temperature
stands for two real ones: 25 °C for the cold streams and 30 °C for the hot
ones. It is the temperature that splits the synthesized network into its
hot-side and cold-side designs, and the dashed line drawn on the pinch diagram
of chapter 1.

Composite curves
----------------

hensmith itself ships one plot, the pinch diagram of a synthesized network. The
two curves below are not library functions: they are computed in this
tutorial's script from the fields of the :class:`~hensmith.ProblemTable`, and
they are shown here because they are the standard way to read what the table
says before any network exists.

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:composite_curves]
   :end-before: # [end:composite_curves]
   :dedent:

Walking the shifted grid upwards from its coldest boundary, each interval adds
the heat of the streams of one kind in it as a diagonal segment, and each point
load adds heat at constant temperature as a horizontal step. The hot curve is
drawn back on the real scale by adding ``T_min_app`` to the shifted grid, so
that the vertical distance between the two curves is a real temperature
difference and is nowhere smaller than the approach. The hot curve starts at
``H = 0`` and the cold curve starts at ``H`` equal to the cold utility target,
which places the two so that their horizontal overlap is exactly the heat that
can be recovered and each overhang is exactly one utility target. Ends of the
grid where no stream of that kind exists carry no load, and are trimmed off
rather than drawn as vertical segments.

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:curves]
   :end-before: # [end:curves]
   :dedent:

.. figure:: /_static/images/examples/tutorial_02_composite_curves.png
   :class: white-bg
   :width: 720
   :alt: Composite curves of the quickstart system: a red hot composite rising from H = 0 at about 27 °C to about 98 °C at roughly 62 GJ/hr with a horizontal step near 65 °C, and a blue cold composite starting at the cold utility target of 1.936e+06 kJ/hr and rising to about 99.5 °C at roughly 345 GJ/hr; the shaded band between them is the recovered heat, the left overhang is the cold utility and the right overhang the hot utility target of 2.828e+08 kJ/hr.

   Composite curves of the quickstart system. The hot composite (red) begins at
   ``H = 0`` at its cold end and ends where the last hot stream is exhausted;
   its near-horizontal step is the column condenser condensing over about half
   a Kelvin, from 65.4 to 64.9 °C. The cold composite (blue) begins at the cold
   utility target, 1.936e+06 kJ/hr, and ends at that offset plus the total
   heating demand of the system. The shaded band where the two overlap
   horizontally is the heat that process-to-process exchange can recover; the
   overhang to the left of it is the cold utility target, 1.936e+06 kJ/hr, and
   the overhang to the right is the hot utility target, 2.828e+08 kJ/hr, both
   annotated on the figure to three significant figures. The
   cold composite extends far to the right of the hot one because this system
   needs a great deal more heating than it has cooling available.

The same cascade can also be plotted directly, as a grand composite curve:

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:grand_composite]
   :end-before: # [end:grand_composite]
   :dedent:

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:gcc]
   :end-before: # [end:gcc]
   :dedent:

.. figure:: /_static/images/examples/tutorial_02_grand_composite.png
   :class: white-bg
   :width: 720
   :alt: Grand composite curve of the quickstart system: heat cascaded in GJ/hr against shifted temperature in °C, running from the top of the grid down through an open circle where the curve touches zero at the pinch, 25.0 °C on the shifted scale (298.15 K), and on below the pinch to the bottom of the grid at 295 K, with a near-horizontal step near 60 °C shifted where the column condenser condenses over a span of about half a Kelvin, between the grid boundaries 333.53 and 333 K.

   The grand composite curve: the heat cascaded through each shifted grid
   temperature once the minimum hot utility is supplied, plotted against that
   shifted temperature. Each boundary contributes two values, the heat arriving
   at it and the heat leaving it after its point loads, so a point load would
   appear as an exactly horizontal step. This system has none: the
   near-horizontal step near 60 °C shifted is the column condenser, spread over
   the two grid boundaries about half a Kelvin apart, 333.53 and 333 K
   on the shifted scale -- the same load that steps the hot composite curve at
   the corresponding real temperature. The curve touches zero exactly at the
   pinch, 298.15 K on the shifted scale, marked with an open circle, and
   continues below it to the bottom of the grid, 295 K. The value at the
   top of the curve is the hot utility supplied, 2.828e+08 kJ/hr, and the value
   at the bottom is the cold utility rejected, 1.936e+06 kJ/hr, small enough to
   be indistinguishable from zero on this axis. Touching zero is what makes
   further recovery impossible: no heat crosses the pinch.

Targets versus the synthesized network
--------------------------------------

The targets are a property of the streams alone. What the synthesized network
of chapter 1 actually achieves is reported by the facility:

.. literalinclude:: /../_demo_src/examples/ch02_pinch_analysis.py
   :language: python
   :start-after: # [start:compare]
   :end-before: # [end:compare]
   :dedent:

.. literalinclude:: /_generated/ch02_compare.txt
   :language: text

The four lines are two different comparisons, and the difference between them
is not a property of the network at all. The first pair uses
``HXN.actual_heat_util_load`` and ``HXN.actual_cool_util_load``, which sum the
``duty`` of each new utility exchanger's ``HeatUtility``. That is the
utility-side duty: biosteam defines ``duty`` as the exchanger's process-side
duty divided by the utility agent's heat-transfer efficiency, so it includes
the heat the agent loses on the way in. The second pair sums ``unit_duty``
instead -- the process-side duty of the same exchangers -- which is the
quantity the problem table computes, an enthalpy difference of the process
streams themselves.

Compared like with like, on the process side, the network reaches the hot
utility target exactly: 2.828e+08 kJ/hr against a target of 2.828e+08 kJ/hr.
The utility-side figure, 2.977e+08 kJ/hr, is that same target divided by the
heat-transfer efficiency of biosteam's low-pressure steam agent, which is
below one; it is the steam the plant must raise, not heat the network failed
to recover. The cold utility needs no such correction, because the cooling
water agent's efficiency is one and both lines therefore read 1.96e+06 kJ/hr
against a target of 1.936e+06 kJ/hr. That small excess is a genuine shortfall
of the network: the targets are a bound the synthesizer works towards, not a
guarantee it attains, because a network has to be built from real exchangers
between real streams, one side of the pinch at a time.

Both directions of that statement are checked by the test suite, and checked on
the process side: ``tests/test_hxn_regression.py`` compares with its
``actual_loads`` helper, which sums ``unit_duty`` exactly as the second pair of
lines above does. It synthesizes ten synthetic systems of
increasing complexity and requires of each synthesized network that it close
its energy balance, that it "never beat the minimum-energy-requirement (MER)
targets of the problem table computed on the same streams", and that it
"recover at least as much heat as documented in ``CASES``". A network that beat
its target would be reporting an infeasible design; a network that fell short
of a recorded result would be a silent regression in the synthesizer.

Where to next
-------------

- :doc:`03_network_anatomy` -- the exchangers, stream life cycles and pinch
  temperatures behind the diagram, unit by unit.
- :doc:`04_configuring` -- what changing ``T_min_app`` does to the targets and
  to the cost of reaching them.
- :doc:`../concepts` -- the pinch concepts and terminology used throughout.
