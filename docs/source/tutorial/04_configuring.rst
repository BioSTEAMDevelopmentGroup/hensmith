Configuring the network and a larger system
===========================================

The three chapters before this one held everything fixed: a minimum approach
temperature of 5 K, every unit of the system in scope, and one small
five-stream flowsheet. This chapter varies all three. It sweeps ``T_min_app``
over the quickstart system to expose the trade-off between recovered heat and
added area, narrows the analysis with ``ignored=``, goes through the remaining
constructor options of :class:`~hensmith.HeatExchangerNetwork` one by one, and
finishes by synthesizing a ten-stream network with ten process exchangers.

Every number and figure below is output of the code shown on this page. The
quickstart system built here is chapter 1's build repeated verbatim, so this
page runs on its own.

Imports and the system
----------------------

.. literalinclude:: /../_demo_src/examples/ch04_configuring.py
   :language: python
   :start-after: # [start:imports]
   :end-before: # [end:imports]
   :dedent:

Two module-level helpers are defined as well. ``utility_hx`` builds one
``bst.Stream`` at a given temperature, pressure and phase, puts a
``bst.HXutility`` on it that brings it to ``T_out``, and simulates that
exchanger; the result is a single unit carrying exactly one heat utility,
which is to say exactly one process stream as far as the network is concerned.
``boiling_hx`` is the same helper specialized to a liquid stream heated with
rigorous VLE, so that it may cross its bubble point. Both are used only by the
ten-stream system at the end of this page.

.. literalinclude:: /../_demo_src/examples/ch04_configuring.py
   :language: python
   :start-after: # [start:helpers]
   :end-before: # [end:helpers]
   :dedent:

The system itself is the one from :doc:`01_quickstart`: a divided shortcut
column with its auxiliary condenser and reboiler, a cooler on each of its two
products, and a flash whose feed is heated by an auxiliary exchanger --
five heating and cooling requirements in total.

.. literalinclude:: /../_demo_src/examples/ch04_configuring.py
   :language: python
   :start-after: # [start:system]
   :end-before: # [end:system]
   :dedent:

The minimum approach temperature
--------------------------------

``T_min_app`` is a plain attribute of the facility, and it is live: assigning
to it and simulating again is all that is needed to see a different network.
Nothing is memoized by default -- ``cache_network`` is ``False``, so each
simulation runs the pinch analysis and the synthesis again from the current
duties -- which makes a sweep a loop over six assignments.

.. literalinclude:: /../_demo_src/examples/ch04_configuring.py
   :language: python
   :start-after: # [start:sweep]
   :end-before: # [end:sweep]
   :dedent:

.. literalinclude:: /_generated/ch04_sweep.txt
   :language: text

Both utility loads rise monotonically with the approach temperature: heating
goes from 2.957e+08 kJ/hr at 2 K to 3.289e+08 kJ/hr at 30 K, and cooling from
9.219e+04 to 3.157e+07 kJ/hr. The added installed cost mostly falls, from
2.02e+06 USD at 2 K to 2.496e+05 USD at 30 K. That is the classic pinch
trade-off. ``T_min_app`` enters the calculation in two places, and both push
the same way. In the problem table it is the amount by which hot streams are
shifted down before the cascade, so a larger value moves the hot streams
further from the cold ones and raises both utility targets. In the synthesis
it is the approach every match must keep along its whole length -- checked at
every breakpoint of both streams' temperature-enthalpy curves, and verified on
their exact states. A smaller approach therefore admits more matches and lets
each one transfer more heat, but the exchangers that do so work across a
smaller temperature difference and need more area for the same duty -- which
is what the right-hand panel below prices.

The cost does not fall monotonically, though: it rises from 6.004e+05 USD at
10 K to 6.684e+05 USD at 15 K, and the last three columns say why. Up to 10 K
the pinch stays at 298.15 K on the shifted scale, and the network reaches the
minimum energy requirement (MER) targets -- status ``mer`` -- with three or
four exchangers. From 15 K on, the pinch moves to the column condenser's
vapor inlet, 65.4 °C on the real scale (323.53, 318.53 and 308.53 K shifted:
the same real temperature less 15, 20 and 30 K). Below that pinch the pinch
design rules fail: the cold streams that reach it cannot each be paired with
a hot stream whose heat capacity flow rate is at least as large, which proves
that the targets need a stream split. hensmith does not split streams, so the
network is a best-effort one, and the status column shows how close it comes:
6.63e+03, 4.7e+03 and 837 kJ/hr of process-side heating above the target, at
most a few thousandths of a percent of the heating load. At 15 K it takes eight
exchangers to get that close, among them repeated matches between the same
two streams, which emulate the missing split -- and which cost more than the
four exchangers at 10 K. ``HXN.synthesis_info`` records each outcome,
including the pinch-rule proof, under ``'sides'``.

.. literalinclude:: /../_demo_src/examples/ch04_configuring.py
   :language: python
   :start-after: # [start:sweep_plot]
   :end-before: # [end:sweep_plot]
   :dedent:

.. figure:: /_static/images/examples/tutorial_04_T_min_app_sweep.png
   :class: white-bg
   :width: 720
   :alt: Two-panel line plot of a minimum approach temperature sweep on the quickstart system. Left panel: utility load in GJ/hr against T_min_app in K, with the heating utility (red circles) rising gently across the top of the panel and the cooling utility (blue squares) rising from near zero along the bottom as T_min_app goes from 2 to 30 K. Right panel: added installed cost in MUSD against T_min_app, falling steeply from 2.02e+06 USD at 2 K to 6.004e+05 USD at 10 K, rising slightly to 6.684e+05 USD at 15 K, and falling again to 2.496e+05 USD at 30 K.

   The ``T_min_app`` trade-off on the quickstart system. Left: the heating
   utility load rises from 2.957e+08 kJ/hr at 2 K to 3.289e+08 kJ/hr at 30 K
   and the cooling utility load from 9.219e+04 to 3.157e+07 kJ/hr, so less heat
   is recovered as the approach widens. Right: the added installed cost of the
   network falls over the same range, from 2.02e+06 USD at 2 K to 2.496e+05
   USD at 30 K, most of the drop happening between 2 and 10 K; the bump at
   15 K is the best-effort network that needs more exchangers once the pinch
   has moved. The default 5 K used throughout this tutorial sits on the steep
   part of the cost curve: 2.977e+08 kJ/hr of heating, 1.936e+06 kJ/hr of
   cooling and 1.112e+06 USD of added installed cost. Choosing ``T_min_app`` is
   choosing a point on these two curves; the economically sensible one depends
   on utility prices and on the cost of exchanger area, neither of which the
   network optimizes for you.

Scoping the analysis
--------------------

By default the network analyzes every unit of the system it belongs to. Two
constructor options change that. ``units`` restricts the analysis to a given
set of unit operations -- either an iterable of units, or a callable returning
one, which is evaluated at simulation time so that the set may depend on the
converged flowsheet; leaving it as ``None`` (the default) means all units of
the system. ``ignored`` works the other way round: it names units whose heat
utilities are dropped from the analysis, again as an iterable or a callable.
Whatever survives both filters is collected with
``bst.process_tools.heat_exchanger_utilities_from_units``, and utilities with a
zero duty are discarded, leaving one process stream per remaining utility.

Excluding a unit is not only an exclusion from the *synthesis*: it is an
exclusion from the accounting as well. ``original_heat_util_load`` and
``original_cool_util_load`` are sums over the utilities that were analyzed, so
the "before" figures move too. Ignoring ``D1_H1``, the cooler on the column's
bottoms product, makes the point:

.. literalinclude:: /../_demo_src/examples/ch04_configuring.py
   :language: python
   :start-after: # [start:ignored]
   :end-before: # [end:ignored]
   :dedent:

.. literalinclude:: /_generated/ch04_ignored.txt
   :language: text

Four streams are analyzed instead of the five of :doc:`01_quickstart`, and the
one that left carried most of the recoverable heat of this system. On the
cooling side the pool itself shrinks: chapter 1 reported ``6.201e+07 ->
1.936e+06 kJ/hr``, and here the same two numbers read ``1.797e+07 ->
0 kJ/hr``. The remaining cooling demand is recovered completely -- the process
exchangers bring every remaining hot stream to its outlet, and a stream served
that way passes through its utility exchanger untouched, so no cooling
utility is left at all -- but it is a much smaller demand, and the cooling
actually saved falls with it. The
heating side shows the loss directly, since its pool is unchanged: the same
3.609e+08 kJ/hr of heating is only reduced to 3.42e+08 kJ/hr, against
2.977e+08 kJ/hr when ``D1_H1`` was in scope. The bottoms cooler is a large,
high-temperature hot stream, and without it there is far less heat available
to serve the column's reboiler. Setting ``ignored`` back to ``None`` restores
the full analysis on the next simulation.

Other options
-------------

The remaining keyword arguments of :class:`~hensmith.HeatExchangerNetwork` are
listed in the :doc:`../API/api` reference; what each of them does to the
synthesis is described below. Four of them -- ``Qmin``, ``force_ideal_thermo``,
``avoid_recycle`` and ``sort_hus_by_T`` -- are passed straight through to
:func:`~hensmith.synthesize_network`, which can also be called directly on a
list of heat utilities.

``Qmin`` is a duty floor, in kJ/hr, defaulting to 1e-3. A planned exchanger
whose duty is below it (``Q < Qmin``) is dropped from the plan and its duty
left to the utilities, which keeps numerically negligible matches out of the
network; the streams simply carry that heat on to their next match or to their
utility exchangers. Dropping a match never makes another one infeasible, but a
large ``Qmin`` can cost MER, and ``HXN.synthesis_info['qmin_dropped']`` lists
what it removed. The same value is passed to the pinch diagram, where a utility
exchanger whose enthalpy change is at or below it (``<=``, rather than the
strict ``<`` of the synthesis) is not marked with a circle -- the reason a
stream that finishes on process heat alone carries no utility symbol.

``cache_network`` (default ``False``) reuses a synthesized topology across
simulations, which is worth doing when the same system is simulated many times
with slightly different inputs, as in a Monte Carlo or a sensitivity analysis.
When it is on and a network has already been synthesized, the units owning the
current heat utilities are compared with the exchangers behind the cached one;
if the sets are identical the cached network is kept and only the stream states
and the exchanger specifications are updated. Each life cycle's first inlet is
copied from the current stream. Each process exchanger keeps, as the enthalpy
limit of the stream its side of the pinch must serve completely (the hot
stream above the pinch, the cold stream below it), the same *fraction* of that
stream's duty as at synthesis, so a changed feed rescales every stage instead
of letting the first one take the whole duty; its partner transfers what that
limit sets, but never past its own outlet, and takes the rest to its utility
exchanger, which brings every stream to its new outlet enthalpy. A reused
network is not planned again, so it need not be at MER for the new duties,
and ``synthesis_info`` still describes the synthesis that produced it. The
reused network is then re-converged and re-checked stream by stream: the
outlet of each life cycle must reproduce the original exchanger's outlet in
composition, pressure and enthalpy, and each stream's original exchanger must
have a finite installed cost. If any of those checks fails, hensmith warns with
a ``RuntimeWarning`` saying that the cache algorithm failed and the cached
network was ignored, discards the cache and synthesizes a fresh network. The
energy balance is treated the same way but silently: a cached network whose
energy balance error exceeds the tolerance is discarded and re-synthesized
without a warning, and a warning is issued only if the freshly synthesized
network also fails.

``replace_unit_heat_utilities`` (default ``False``) changes where the savings
are reported. By default the facility carries the *difference* between the new
and the original utilities as its own ``heat_utilities``, which is why the
utility rows of its results table are negative (:doc:`01_quickstart`), and the
original units keep the utilities they had before integration. With the option
on, each synthesized utility exchanger's heat utility is copied onto the
corresponding original heat utility instead and the owner's utility cost is
reloaded, so the process units themselves report the reduced duties, and the
facility is left with no ``heat_utilities`` of its own. The option applies only
when at least one process exchanger was synthesized; if the synthesis produced
no matches, the facility reports no utilities and no added cost either way.

``avoid_recycle`` (default ``False``) forbids matching the same hot/cold stream
pair more than once anywhere -- neither twice on one side of the pinch nor once
on each side. Without it, the planner may match a pair repeatedly: alternating
two partners in series emulates a stream split, and some unsplit MER networks
need it (the repeated exchangers carry a suffix, ``HX_3_2_cs_2`` for the second
one, say). Two exchangers between the same two streams can close a loop in the
network, which its ``System`` tears at a recycle stream and converges by
fixed-point iteration. Turning the option on trades those networks -- and
possibly MER with them -- for a network in which no two exchangers connect the
same pair of streams.

``force_ideal_thermo`` (default ``False``) runs the analysis on copies of the
streams made with ideal thermodynamics (``i.thermo.ideal()``), and the
synthesized exchangers inherit that property package. It is an escape hatch for
systems whose rigorous VLE is expensive or fragile; the network it produces is
only as accurate as that assumption.

``sort_hus_by_T`` (default ``False``) reorders the streams before the analysis:
heating utilities are sorted by inlet temperature in descending order and
cooling utilities in ascending order. Heating utilities always precede cooling
ones in the rearranged list regardless, and it is that list that fixes the
stream indices used by every per-stream array, by the stream life cycles and by
the pinch diagram. The order does not change the targets; it breaks ties in
the planner's search, so sorting can change the network that comes out. The
default order is by signed duty, which is how the
facility hands the utilities to the synthesis: cold streams from the smallest
heating duty up, hot streams from the largest cooling duty down.

``acceptable_energy_balance_error`` overrides, for one instance, the class
attribute of the same name. It is a *fraction*, not a percentage: the default
``0.02`` means 2 %. After convergence the network's heat flows are compared
with the original ones, and the absolute deviation from unity of that ratio is
tested against this tolerance; ``energy_balance_percent_error`` reports the
same quantity multiplied by 100. Exceeding it warns with a ``RuntimeWarning``,
or raises a ``RuntimeError`` if the class attribute
``raise_energy_balance_error`` is set to ``True``.

A larger system
---------------

The quickstart system has five streams and produces four process exchangers.
The system below has ten streams and produces ten, on both sides of the pinch,
which is enough for the structure of a synthesized network to be visible. It is the ten-stream case of
the regression suite, ``tests/test_hxn_regression.py::case_10_ten_streams``,
inlined here rather than imported from ``tests``: five hot streams -- two
condensers (``H1``, ``H2``), one partial condensation (``H5``) and two liquid
coolers, at two different pressures -- and five cold streams, two of which are
taken past their bubble point with rigorous VLE (``C2``, ``C3``).

.. literalinclude:: /../_demo_src/examples/ch04_configuring.py
   :language: python
   :start-after: # [start:ten_streams]
   :end-before: # [end:ten_streams]
   :dedent:

.. literalinclude:: /_generated/ch04_ten_streams.txt
   :language: text

The targets come from :func:`~hensmith.problem_table` applied to the same
streams the facility synthesized from, exactly as in :doc:`02_pinch_analysis`.
The comparison is printed twice because there are two conventions in play. The
facility's ``actual_heat_util_load`` and ``actual_cool_util_load`` are
utility-side loads: the duty an agent must supply, which includes that agent's
heat transfer efficiency. The problem table knows nothing about agents -- its
targets are process-side enthalpy differences. Summing the ``unit_duty`` of the
new utility exchangers instead, as ``tests/test_hxn_regression.py`` does, gives
the process-side loads that are comparable with the targets. On the hot side
the two differ: 1.523e+07 kJ/hr utility-side against 1.447e+07 kJ/hr
process-side, for a target of 1.447e+07 kJ/hr. On the cold side the cooling
agents used here need no such correction and both read 8.462e+06 kJ/hr, against
a target of 8.462e+06 kJ/hr. The network reaches both targets -- the synthesis
reports ``mer`` -- and those process-side numbers, 1.447e+07 and
8.462e+06 kJ/hr, are the values the regression suite records for this case.
Its energy balance error is 2e-12 %.

The phase changes of these streams are what make this case demanding. Two
condensers, a partial condensation and two streams boiled past their bubble
points put flat and steep stretches into the stream curves, and a network
checked only at the terminals of its exchangers could let two streams cross
*inside* one of them. hensmith keeps ``T_min_app`` everywhere inside every
exchanger, on the exact stream states, so its targets and its network hold for
the real streams.

.. figure:: /_static/images/examples/tutorial_04_ten_streams_pinch_diagram.png
   :class: white-bg
   :width: 100%
   :alt: Pinch diagram of the synthesized ten-stream network: five blue cold streams labelled C5, C4, C2, C1 and C3 drawn left to right above five red hot streams labelled H1, H2, H4, H3 and H5 drawn right to left, joined by ten vertical process-exchanger connectors, five of them to the left of the dashed pinch line and five to the right, with hot utility circles at the outlets of three cold streams and cold utility circles at the outlets of four hot streams.

   The synthesized ten-stream network. The five cold streams (blue, indices 0
   to 4) run left to right and the five hot streams (red, indices 5 to 9) right
   to left, each annotated with its inlet and outlet temperature and enthalpy
   flow, the outlets being the quenched end states the analysis works from. The
   ten vertical connectors are the process exchangers, each labelled with its
   duty in kJ/hr; five of them lie to the left of the dashed pinch line, in the
   cold-side design, and five to the right, in the hot-side design. Streams 0
   (``C5``) and 5 (``H1``) are matched with each other on both sides of the
   pinch, which closes a loop in the network that its ``System`` tears and
   converges. Three cold streams still need a hot utility at their outlet (open
   red circles on the right) and four hot streams a cold utility (open blue
   circles on the left); the others are brought to their outlets by process
   heat exchange alone. The shaded background marks the two sides of the pinch.

Where to next
-------------

- :doc:`../concepts` -- the pinch concepts, the MER planner and the
  validation behind everything shown in this tutorial.
- :doc:`../API/api` -- the full API reference, including a table of every
  constructor option and every attribute the facility sets.
