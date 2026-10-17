Targeting and supertargeting
============================

The chapters before this one judged a network by one pair of numbers, the
minimum energy requirement (MER) targets of its problem table. Pinch analysis
can say much more before a single exchanger exists: how the heating and
cooling targets split between several utilities and what they cost, how much
exchanger area and how many units any network of these streams needs, what
that area costs, and therefore which ``T_min_app`` minimizes the total
annualized cost. ``hensmith.targeting`` computes all of these
(:class:`~hensmith.PinchTargets`), sweeps ``T_min_app`` for the least total
annualized cost (:func:`~hensmith.supertarget`), draws the composite curves,
the grand composite curve and the sweep, and compares a synthesized network
with its targets (:func:`~hensmith.network_report`).

Every number and figure below is output of the code shown on this page. The
quickstart system built here is chapter 1's build repeated verbatim, so this
page runs on its own.

Imports and the system
----------------------

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:imports]
   :end-before: # [end:imports]
   :dedent:

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:system]
   :end-before: # [end:system]
   :dedent:

The facility's targets
----------------------

:meth:`HXN.targets() <hensmith.HeatExchangerNetwork.targets>` returns the
targets of the streams the facility synthesized from, in its stream order and
at its ``T_min_app`` (or at another one, ``HXN.targets(10.)``, without
synthesizing anything).

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:targets]
   :end-before: # [end:targets]
   :dedent:

.. literalinclude:: /_generated/ch05_targets.txt
   :language: text

The streams are labeled by their units, an auxiliary exchanger by its owner
and its own ID. The targets are the synthesis's own, bit for bit: the
facility prepares the streams for targeting exactly as the synthesis prepares
them. They are **process side**: 2.828e+08 kJ/hr is the heat the cold
streams need from a hot utility. The network's heating load, 2.977e+08 kJ/hr
here and in :doc:`04_configuring`, is **utility side**: biosteam divides each
duty by its agent's heat transfer efficiency, 0.95 for steam, and 2.828e+08 /
0.95 is 2.977e+08.

Nothing is stored on the facility, and nothing is re-synthesized: the targets
are those of the units' current states, which, with ``cache_network=True``,
can differ from the cached network's. The same object can be built without a
facility, from the heat utilities of heat exchangers with
:meth:`PinchTargets.from_heat_utilities <hensmith.PinchTargets.from_heat_utilities>`
(as the trade-off example below does) or from streams as
:func:`~hensmith.problem_table` takes them. The facility's selection,
``bst.process_tools.heat_exchanger_utilities_from_units(sys.units)``, is the
one to give it: a heat utility that a unit adds for itself (a reactor's or a
heated mixer's, ``hu.hxn_ok`` False) raises a ``ValueError``, as its unit's
first inlet and outlet need not be the stream it heats or cools. After a
simulation with ``replace_unit_heat_utilities=True``, though, the units' heat
utilities hold the network's utilities, not the process duties, and those of
the streams that the process exchangers serve completely are empty: the
selection then gives the wrong streams (with a ``RuntimeWarning``), and
``HXN.targets()`` is the one to take.

Several utilities on the grand composite curve
----------------------------------------------

The MER targets say how much heating and cooling the streams need, not which
utilities supply them. :meth:`~hensmith.PinchTargets.utility_targets` places
biosteam's heating and cooling agents on the grand composite curve,
following biosteam's own rules for which agent may serve which temperature,
and prices each with biosteam.

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:utilities]
   :end-before: # [end:utilities]
   :dedent:

.. literalinclude:: /_generated/ch05_utilities.txt
   :language: text

Each agent has a **level** on the shifted scale: the warmest temperature it
can serve, or the coldest, for a coolant. A heating agent's level is its
temperature less biosteam's approach, the agent's own ``dT`` plus
``HeatUtility.dT``: 412.19 - 0 - 5 = 407.19 K for low pressure steam. A
coolant's level is its temperature plus that approach, moved onto the shifted
scale by subtracting ``T_min_app``, the hot streams' shift:
280.37 + 2 + 5 - 5 = 282.37 K for chilled water. Low pressure steam, first in
biosteam's list of heating agents, is above the top of the cold streams, so it
takes the whole hot utility target; a hotter steam would only be used for what
it cannot serve.
Every cold stream above the pinch is heated by it, from 298.15 K (the pinch)
to 372.60 K, and biosteam prices that duty at 1829.93 USD/hr, with the same
division by the heat transfer efficiency as in the network: the utility duty
is 2.977e+08 kJ/hr. On the cold side, cooling water, the first coolant in the
list, gets nothing: all the cold utility is needed below 303.15 K (30 °C, the
pinch on the hot streams' scale), colder than cooling water can serve. Chilled
water serves it. A sensible agent, it enters at 280.37 K and leaves at 298.15
K, 5 K below the warmest process temperature it serves, 303.15 K: biosteam
limits a sensible agent's outlet at that end, and hensmith also keeps its
line 5 K from the process across the whole stretch, which here allows the
same outlet (a condenser at the cold end of the stretch would hold it lower).
The utility cost target, 1839.61 USD/hr, is the cost of the MER utilities
with biosteam's agents placed this way.

The grand composite curve shows the allocation: each agent is a horizontal
segment at its level, as wide as its duty, the heating ones from the lowest
level up, each starting where the one below ends, and the cooling ones
mirrored below the pinch.

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:gcc]
   :end-before: # [end:gcc]
   :dedent:

.. figure:: /_static/images/examples/tutorial_05_grand_composite.png
   :class: white-bg
   :width: 560
   :alt: Grand composite curve of the quickstart system: a black curve of shifted temperature in degrees Celsius against the heat cascaded in kJ/hr, starting near zero at the pinch, marked by an open circle at 25 degrees Celsius, and rising to the hot utility target at about 99 degrees Celsius, with a red horizontal segment labelled low_pressure_steam at about 134 degrees Celsius spanning the whole hot utility target, and a very short blue segment labelled chilled_water near 9 degrees Celsius.

   The grand composite curve of the quickstart system at ``T_min_app = 5`` K
   (the same curve as chapter 2's), with the utility targets as a staircase:
   low pressure steam at its level, 407.19 K (134 °C on the shifted scale),
   serves all 2.828e+08 kJ/hr of the hot utility target, above every cold
   stream; chilled water at 282.37 K serves the 1.936e+06 kJ/hr of cold
   utility below the pinch. The open circle is the pinch, 298.15 K shifted.

Composite curves, plain and balanced
------------------------------------

Chapter 2 drew the composite curves from the fields of a
:class:`~hensmith.ProblemTable` in its own script;
:func:`~hensmith.plot_composite_curves` draws them from the targets, and
:meth:`~hensmith.PinchTargets.composite_curves` returns their points. With
``kind='balanced'`` the utilities join the curves: the heating utilities the
hot composite and the cooling ones the cold composite, so that both curves end
at the same heat. These **balanced composite curves** are what the area
target below is computed on.

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:composites]
   :end-before: # [end:composites]
   :dedent:

.. figure:: /_static/images/examples/tutorial_05_composite_curves.png
   :class: white-bg
   :width: 720
   :alt: Composite curves of the quickstart system drawn by hensmith: a red hot composite curve above a blue cold composite curve on axes of temperature in degrees Celsius against enthalpy in kJ/hr, a shaded band marking the heat recovered, a cold utility label of 1.94E6 kJ/hr at the cold end and a hot utility label of 2.83E8 kJ/hr at the warm end.

   The composite curves of chapter 2, drawn by the library: the shaded band is
   the heat recovered, the cold composite starts at the cold utility target,
   1.936e+06 kJ/hr, and ends the hot utility target, 2.828e+08 kJ/hr, beyond
   the end of the hot one. The script checks that their knots are chapter 2's.

.. figure:: /_static/images/examples/tutorial_05_balanced_composites.png
   :class: white-bg
   :width: 720
   :alt: Balanced composite curves of the quickstart system: the red hot balanced composite follows the hot composite and then rises vertically to a long flat at about 139 degrees Celsius labelled low_pressure_steam 2.83E8 kJ/hr, which runs to the end of the blue cold balanced composite; the blue curve starts with a short steep line from about 7 to 25 degrees Celsius labelled chilled_water 1.94E6 kJ/hr and then follows the cold composite.

   The balanced composite curves. The steam condenses at 412.19 K, so it is a
   flat at that temperature (139 °C), 2.828e+08 kJ/hr long, at the warm end of
   the hot curve; chilled water warms from 280.37 to 298.15 K, a line at the
   cold end of the cold curve, 1.936e+06 kJ/hr long. Both curves now span the
   same heat; the area target assumes that every bit of it is transferred
   vertically, from the hot curve to the cold curve below it.

Area, units, capital and total cost
-----------------------------------

On the balanced composite curves, :meth:`~hensmith.PinchTargets.summary`
targets the rest of the cost of a network. The methods that take
``utilities`` need those of the same targets: a list from targets at another
``T_min_app`` (``targets.at(20.)``, say) or a part of one would not balance
the curves, so its duties are checked against ``Q_hot`` and ``Q_cold`` and a
mismatch raises a ``ValueError``.

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:supertargets]
   :end-before: # [end:supertargets]
   :dedent:

.. literalinclude:: /_generated/ch05_summary.txt
   :language: text

- **Area**, 5946.13 m²: Bath's formula, the area of an ideal network that
  transfers heat vertically between the balanced composite curves, each
  stretch between their breakpoints at its log-mean temperature difference
  and each stream's share of it over its own film coefficient. By default
  every stream and utility has a film coefficient of 1 kW/m²/K, so any pair
  has biosteam's heuristic overall coefficient of 0.5 kW/m²/K; ``h=`` and
  ``h_utilities=`` give streams and agents their own.
- **Units**, 6 (5 above the pinch and 1 below): Euler's rule, the number of
  streams and utilities less one, in each region between the zeros of the
  balanced grand composite curve. It is an estimate, not a bound: a subset of
  streams that balances on its own needs fewer units.
- **Capital**: the area divided into as many shells of equal area as there
  are units, here 6 floating head shells of 991 m², each costed with
  biosteam's own correlations: 2.28172e+06 USD installed. When the equal
  shells would fall below 150 ft² (14 m²), where biosteam's double pipe costs
  a fraction of a shell and tube, the target keeps the shell and tubes the
  area fills if they cost more (see :doc:`../concepts`);
  ``'N_area_shells'``, ``'shell_area'`` and ``'shell_type'`` say which shells
  carry the area.
- **Total annualized cost** (TAC), 1.50882e+07 USD/yr: the installed cost
  times the capital recovery factor (371339 USD/yr: 10 % interest over 10
  years by default) plus the utility cost times the operating hours
  (1.47169e+07 USD/yr: 8000 h/yr by default). It counts the bare module cost
  of the exchangers only, no Lang factor and no working capital.
  ``interest=``, ``years=`` and ``operating_hours=`` change the economics, or
  ``tea=`` takes them from a biosteam TEA. The facility's targets know their
  system: if it has operating hours (a TEA sets them), the summary and the
  sweep below use them instead of 8000 h/yr. The quickstart system has none.

Supertargeting: the least total annualized cost
-----------------------------------------------

Chapter 4 swept ``T_min_app`` by synthesizing a network at each value.
:func:`~hensmith.supertarget` sweeps the targets instead: it reuses the
streams' curves, so no stream is flashed again, and evaluates the summary at
each ``T_min_app``, then refines the grid around the least TAC. Each point
still builds its own problem table, which reads every stream's enthalpy on the
grid; on a system of many streams with rigorous thermodynamics, a sweep of the
default 40 points and three refinements can take longer than a synthesis, and
a coarser ``T_min_apps`` or a smaller ``refine`` keeps it short.

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:sweep]
   :end-before: # [end:sweep]
   :dedent:

.. literalinclude:: /_generated/ch05_sweep.txt
   :language: text

The quickstart system is dominated by steam: its operating cost, 1.454e+07 to
1.654e+07 USD/yr over the sweep, is about forty times its annualized capital
cost, so the TAC follows the utility cost and falls as ``T_min_app`` does. Below
about 1.85 K the system is a threshold problem: it needs no cold utility at
all, so the hot utility, its steam and the balanced composite curves no longer
change with ``T_min_app``, and neither does any target. The TACs at 1 and at
1.848 K differ by less than 7 USD/yr, 4e-7 of the TAC, which is no physical
effect: the problem table's grid moves with ``T_min_app`` over the curved
(vapor-liquid) profiles of the streams, and the area follows it, too little
to show in the table (6836 m² at both). Every ``T_min_app`` of that flat
stretch is an optimum but for these few USD/yr. The area, and with it the TAC,
falls steadily across the stretch as the grid moves, so the least TAC is always
the warmest point evaluated below the threshold, here 1.848 K (inside the
sweep, not at its bound), 0.005 K short of the step. That step comes at the
next point, 1.853 K: the cold utility appears and needs a unit of its own (6
units), which adds about 12000 USD/yr to the TAC. Between 10 and 15 K, where
chapter 4's sweep saw the pinch move to the column condenser's vapor inlet,
the units target jumps to 9.

.. figure:: /_static/images/examples/tutorial_05_supertargets.png
   :class: white-bg
   :width: 640
   :alt: Total annualized cost against T_min_app from 1 to 30 K for the quickstart system: the black total annualized cost and the red operating cost both rise almost linearly from about 1.5e7 USD per year to about 1.7e7 USD per year, the blue annualized capital cost stays near 4e5 USD per year along the bottom, and an open circle with a dashed vertical line marks the optimum at 1.85 K.

   The sweep of the quickstart system. The total annualized cost (black) is
   the operating cost (red) plus the annualized capital cost (blue); the
   circle and the dashed line mark the least TAC, 1.494e+07 USD/yr at 1.848 K.
   Capital is a small fraction of the TAC, so the optimum has the least
   utility: it is on the flat stretch below 1.85 K, where no cold utility (and
   no unit for it) is needed, and any point of that stretch would do.

A trade-off between capital and energy needs a system whose capital
matters. The two streams below, a hot and a cold one of 100 kW/K each, are
served by steam at 450 K and by water evaporating at 280 K, two latent agents
made for the example with prices of 4e-7 and 2e-7 USD/kJ. Their targets come
straight from their heat utilities with
:meth:`PinchTargets.from_heat_utilities <hensmith.PinchTargets.from_heat_utilities>`,
and ``heating_agents=`` and ``cooling_agents=`` are passed on to every summary
of the sweep.

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:trade_off]
   :end-before: # [end:trade_off]
   :dedent:

.. literalinclude:: /_generated/ch05_trade_off.txt
   :language: text

Here the annualized capital cost (71605.6 USD/yr at the optimum) falls as
``T_min_app`` grows and the operating cost (20032 USD/yr) rises with it, and
the least TAC, 91637.6 USD/yr, is at 21.59 K, well inside the sweep: the
classic supertargeting curve.

.. figure:: /_static/images/examples/tutorial_05_trade_off.png
   :class: white-bg
   :width: 640
   :alt: Total annualized cost against T_min_app from 11 to 60 K for a two-stream problem: the blue annualized capital cost falls steeply from about 110000 USD per year and flattens near 53000, the red operating cost rises linearly from near zero to about 87000 USD per year, and the black total annualized cost has a U shape with its minimum, marked by an open circle and a dashed vertical line, at 21.6 K.

   The sweep of the two-stream problem: the capital cost (blue) falls and the
   operating cost (red) rises with ``T_min_app``, and their sum (black) has
   its minimum, 9.164e+04 USD/yr, at 21.59 K.

The shells of this pair are large, so the capital cost falls smoothly. With
small streams it falls in steps: about one shell and tube each time the
shrinking area fills one fewer of 150 ft², or when the last one goes and
double pipes take the area. The least TAC then often lies just past such a
step, at the first ``T_min_app`` evaluated beyond it, and says more about
biosteam's 150 ft² than about a trade-off of energy and capital; the sweep's
``'N_area_shells'`` and ``'shell_type'`` (``sweep.to_frame()``) show where the
steps are. A network has such steps too, but where its own exchangers cross
150 ft².

The network against its targets
-------------------------------

:func:`~hensmith.network_report` compares the network the facility
synthesized with the targets of its streams, on the same options.

.. literalinclude:: /../_demo_src/examples/ch05_targeting.py
   :language: python
   :start-after: # [start:report]
   :end-before: # [end:report]
   :dedent:

.. literalinclude:: /_generated/ch05_report.txt
   :language: text

The network reaches the utility cost target, 1839.61 USD/hr: it is at MER,
with the steam and chilled water that the targets allocate. The comparison of
everything else needs care, because a target is an estimate of an ideal
network:

- **Area.** The report's ``'area'``, 7462 m², re-sizes every exchanger of the
  network like the target sizes it: the same film coefficients, a correction
  factor of 1 and the integrated temperature difference along the exact
  stream profiles. On that basis the network needs more area than the 5946 m²
  target, as it should with equal film coefficients and the same utilities.
  biosteam's own sizing, 5536 m² (``'area_biosteam'``), is lower still,
  mostly because it takes an overall coefficient of 1.0 kW/m²/K for
  condensing steam against a boiling stream (the steam heaters ``Util_0_hs``
  and ``Util_1_hs``: 177 and 1200 m² instead of 356 and 2983 m²), and because
  its terminal log-mean temperature difference overstates the mean driving
  force of a glide; it also applies a correction factor for its shells and
  sizes utility exchangers on the utility side duty.
- **Units.** The network has 7 exchangers against an estimate of 6 because
  it has a loop. Above the pinch, the bottoms cooler ``D1_H1`` heats both the
  reboiler (``HX_0_2_hs``) and the flash's feed heater (``HX_1_2_hs``), and
  each of these also has its own steam heater (``Util_0_hs`` and
  ``Util_1_hs``), so the four exchangers close a loop: reboiler, ``D1_H1``,
  feed heater, steam and back to the reboiler. Euler's S - 1 counts the steam
  as one stream however many streams it heats, and assumes the matches of a
  region form no loop. The estimate can be wrong both ways: a balanced subset
  of streams needs fewer units, and each loop needs one more, whether a
  repeated match between two streams (two branches of a split stream meeting
  the same stream make one too) or, as here, two streams linked both through
  process matches and through a shared utility.
- **Cost.** The installed cost and the TAC of the network (2.184e+06 USD and
  15072295 USD/yr) are biosteam's, on biosteam's areas, and here they are
  below the targets' (2.282e+06 USD and 15088200 USD/yr), which divide a
  larger area into equal shells. Neither is a bound on the other.

The report's ``'targets'`` is :meth:`~hensmith.PinchTargets.summary` with the
same options, and the network and its targets are priced over the same
operating hours (the system's, if it has them, else 8000 h/yr);
``exchanger_areas`` gives both areas of each exchanger, which is
where a gap between the network and its target can be traced. The
synthesis itself does not optimize area, capital or total cost: it reaches
MER first and then keeps the number of exchangers low. Supertargeting is how
to choose the ``T_min_app`` it synthesizes at.

Where to next
-------------

- :doc:`../concepts` -- the multiple-utility rule, the area, units and cost
  targets, the sweep, and the limits of each.
- :doc:`../API/targeting` -- every function, class and option of
  ``hensmith.targeting``.
