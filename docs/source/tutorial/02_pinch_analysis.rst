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
temperature to any lower one without ever violating the minimum approach. The
grid above runs from 395 K, the hot stream's inlet shifted down by the 5 K
approach, to 295 K, its shifted outlet; the cold stream's ends, unshifted at
390 and 300 K, lie in between. The grid holds 25 points rather than those four
because it is the union, sorted descending, of the breakpoints of every
stream's temperature-enthalpy curve. hensmith describes each stream by such a
curve, built once from a handful of flashes: its breakpoints are the stream's
end temperatures, every phase boundary inside its range, and -- since the heat
capacity of liquid water varies with temperature -- interior points that keep
a straight line between neighbours within 0.002 K of the true curve. Every
temperature at which a curve bends is therefore a grid point, and nothing that
happens inside an interval can hide a pinch.

Between consecutive grid temperatures, each stream contributes the enthalpy
its curve releases or absorbs over that interval, evaluated at its *real*
temperature and clipped to its own enthalpy range, with a positive sign for hot
streams and a negative one for cold. Because the grid always contains every
breakpoint of a stream's curve, its own end temperatures among them, those
contributions telescope exactly to the stream's duty: no heat is created or
lost by the discretization. Heat that a curve gives up or takes at a single
temperature is not spread over an interval at all but enters as a *point load*
at that grid temperature: the latent heat of a pure component boiling or
condensing at its saturation temperature, and the whole duty of a stream with
no temperature span of its own -- an isothermal condenser, or a stream whose
outlet moves against its duty, such as a reboiler outlet at equilibrium --
which sits at its shifted outlet temperature.

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

The five streams produce a grid of 175 shifted temperatures, from 372.60 K
down to 295.00 K. Their shifted end temperatures are among them; most of the
rest trace two-phase glides. Every stream here is a mixture of water, methanol
and glycerol, so none boils or condenses at a single temperature: the
column's reboiler and the flash's feed heater heat a liquid past its bubble
point and on along a glide, the condenser and the distillate cooler ``D1_H2``
glide from end to end, and the bottoms cooler ``D1_H1`` cools a liquid whose
heat capacity varies with temperature. Each glide and each curved stretch is
sampled until a straight line between neighbouring points is within 0.002 K
of the true curve. With no flat segment in any curve, and every outlet moving
in the direction of its stream's duty, the table has no point loads at all --
the second line. The condenser is still the most conspicuous stream: it gives
up its latent heat over about half a Kelvin, from 65.4 to 64.9 °C on the real
scale (:doc:`03_network_anatomy` lists every stream's end temperatures), and
the distillate cooler takes the stream on from there. The
targets are 2.828e+08 kJ/hr of hot utility and 1.936e+06 kJ/hr of cold utility,
and the pinch is at 298.15 K on the shifted scale. Since hot streams were
shifted down by the 5 K approach, that one shifted temperature stands for two
real ones: 25 °C for the cold streams and 30 °C for the hot ones. It is the
temperature that splits the synthesized network into its hot-side and cold-side
designs, and the dashed line drawn on the pinch diagram of chapter 1.

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
   :alt: Grand composite curve of the quickstart system: heat cascaded in GJ/hr against shifted temperature in °C, running from the top of the grid down through an open circle where the curve touches zero at the pinch, 25.0 °C on the shifted scale (298.15 K), and on below the pinch to the bottom of the grid at 295 K, with a near-horizontal step near 60 °C shifted where the column condenser condenses over a span of about half a Kelvin.

   The grand composite curve: the heat cascaded through each shifted grid
   temperature once the minimum hot utility is supplied, plotted against that
   shifted temperature. Each boundary contributes two values, the heat arriving
   at it and the heat leaving it after its point loads, so a point load would
   appear as an exactly horizontal step. This system has none: the
   near-horizontal step near 60 °C shifted is the column condenser, whose
   glide spans about half a Kelvin, 65.4 to 64.9 °C on the real scale and
   5 K lower on the shifted one -- the same load that steps the hot composite
   curve at the corresponding real temperature. The curve touches zero exactly at the
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

The first four lines are two different comparisons, and the difference between them
is not a property of the network at all. The first pair uses
``HXN.actual_heat_util_load`` and ``HXN.actual_cool_util_load``, which sum the
``duty`` of each new utility exchanger's ``HeatUtility``. That is the
utility-side duty: biosteam defines ``duty`` as the exchanger's process-side
duty divided by the utility agent's heat-transfer efficiency, so it includes
the heat the agent loses on the way in. The second pair sums ``unit_duty``
instead -- the process-side duty of the same exchangers -- which is the
quantity the problem table computes, an enthalpy difference of the process
streams themselves.

Compared like with like, on the process side, the network reaches both
targets exactly: 2.828e+08 kJ/hr of hot utility and 1.936e+06 kJ/hr of cold
utility, against targets of 2.828e+08 and 1.936e+06 kJ/hr. The utility-side
heating figure, 2.977e+08 kJ/hr, is that same target divided by the
heat-transfer efficiency of biosteam's low-pressure steam agent, which is
below one; it is the steam the plant must raise, not heat the network failed
to recover. The cold utility needs no such correction, because the cooling
agents used here (chilled and cooling water) have an efficiency of one, and
both of its lines read 1.936e+06 kJ/hr.

The last line says the same thing in one word: the synthesis reports
``HXN.synthesis_info['status']`` as ``mer`` because the utilities of the network
it realized equal these targets. That is by construction rather than by luck.
The synthesizer plans each side of the pinch from the pinch outward on the
same stream curves this table was built from, so its own cascade *is* this
table, and it reaches the targets whenever its search finds a network without
stream splits that does (:doc:`../concepts` describes the planner). Where the
pinch design rules prove that the targets need a stream split, which hensmith
does not make by default, the status is ``best_effort`` and the network lies
slightly above the targets instead; with ``stream_splitting=True`` hensmith
splits streams there and reaches the targets. :doc:`04_configuring` shows all
three outcomes on this system.

Both directions of that statement are checked by the test suite, on the
process side. ``tests/test_hxn_mer.py`` synthesizes 40 problems for which an
unsplit MER network is known to exist and requires every one of them to reach
its targets and report ``mer``, and 38 problems that provably need splits,
which by default must never beat their targets and must report
``best_effort``, and with ``stream_splitting=True`` must reach them and report
``mer``.
``tests/test_hxn_regression.py`` compares with its ``actual_loads`` helper,
which sums ``unit_duty`` exactly as the second pair of lines above does. It
synthesizes ten synthetic systems of increasing complexity and requires of
each synthesized network that it close its energy balance, that it never beat
the MER targets of the problem table computed on the same streams (and report
``mer`` exactly when it reaches them), that it keep ``T_min_app`` inside every
exchanger, and that it recover at least as much heat as recorded in the test
file. A network that beat its target would be reporting an infeasible design;
a network that fell short of a recorded result would be a silent regression
in the synthesizer.

Where to next
-------------

- :doc:`03_network_anatomy` -- the exchangers, stream life cycles and pinch
  temperatures behind the diagram, unit by unit.
- :doc:`04_configuring` -- what changing ``T_min_app`` does to the targets and
  to the cost of reaching them.
- :doc:`../concepts` -- the pinch concepts and terminology used throughout.
