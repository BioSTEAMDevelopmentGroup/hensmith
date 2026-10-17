Key Concepts
============

.. currentmodule:: hensmith

This page collects the ideas hensmith implements: what a minimum approach
temperature is, how the problem table turns a set of process streams into
utility targets and a pinch, how those targets extend to several utilities,
area, units and cost, how :func:`~hensmith.synthesize_network` turns the
targets into a network of exchangers, and what the result is and is not
guaranteed to be. Everything below describes the behavior of the code in
``hensmith/hxn_synthesis.py``, ``hensmith/_heat_exchanger_network.py`` and
``hensmith/targeting.py``, and of two private helpers,
``hensmith/_curves.py`` (the stream curves) and ``hensmith/_planner.py`` (the
network planner); the :doc:`tutorial/index` shows the same concepts on a
running system.

Heat integration and the minimum approach temperature
-----------------------------------------------------

Every heating and cooling duty in a flowsheet is paid for with a utility:
steam, cooling water, chilled water, refrigerant. Heat integration replaces
part of that spending with process-to-process exchange -- heat taken from a
stream that must be cooled and given to a stream that must be heated. What is
left after the process exchangers have done as much as they can is the
irreducible utility demand.

Heat only flows down a temperature gradient, and a finite exchanger can only
transfer heat across a finite temperature difference: the closer the two
streams are in temperature, the more area is needed for the same duty, since
the area of a counter-current exchanger scales as
:math:`Q / (U \Delta T_{lm})`. hensmith expresses that limit as a single
number, the minimum approach temperature ``T_min_app`` (in K, default ``5.``),
used to shift hot streams in the problem table and required between the
streams everywhere inside every synthesized exchanger:

- in :func:`~hensmith.problem_table`, where every hot stream's temperature is
  shifted *down* by ``T_min_app`` (with approach contributions, every stream
  by its own shift; see below) before the streams are compared, so that two
  streams which meet on the shifted scale are really ``T_min_app`` apart;
- in the planner of :func:`~hensmith.synthesize_network`, which admits a
  match only if the two streams stay at least ``T_min_app`` apart at every
  breakpoint of their temperature-enthalpy curves along the whole exchanger,
  not only at its two ends; and
- on every synthesized process exchanger, whose approach is verified on the
  exact states of its two streams, again along its whole length. The
  exchanger itself is a ``biosteam.HXprocess`` constructed with ``dT`` equal
  to its pair's required approach minus 1e-6, a guard against rounding only.

Lowering ``T_min_app`` lowers the utility targets and raises exchanger area;
raising it does the reverse. It is the parameter in hensmith that sets the
trade between utilities and capital, and because
:class:`~hensmith.HeatExchangerNetwork` reports its capital as *added* cost
and its utilities as *differences* (see below), both sides of that trade are
visible in the facility's results.

Shifted temperatures and the problem table
------------------------------------------

:func:`~hensmith.problem_table` builds the temperature-interval heat cascade
of a set of streams, given each stream's inlet, its outlet quenched to
equilibrium at its own enthalpy, a flag saying whether it is cooled, and
``T_min_app``. Its result is a :class:`~hensmith.ProblemTable`.

**Stream curves.** Each stream is first described by a piecewise-linear
temperature-enthalpy curve over its own enthalpy range, built once from a
handful of flashes and evaluated afterwards without flashing. Its breakpoints
are the stream's two end temperatures and every phase boundary inside its
range (a pure component's saturation temperature, a mixture's bubble and dew
points); a pure component's latent heat is a flat (isothermal) segment
between its saturated-liquid and saturated-vapor enthalpies; a mixture's
two-phase glide is sampled, and so is every curved single-phase stretch
(temperature-dependent heat capacity), densely enough that linear
interpolation stays within 0.002 K of the true curve. Inside its range the
stream is taken at equilibrium with its enthalpy clipped to
:math:`[H_{lo}, H_{hi}]`, so a stream copy at an interior temperature can
never carry more enthalpy than the real stream ever has (a non-equilibrium
outlet, for instance); what a non-equilibrium end state departs from
equilibrium at its own end temperature becomes a flat there.

**The shifted grid.** Hot streams are shifted down by ``T_min_app``; cold
streams are not (with approach contributions, every stream by its own shift;
see below). The grid ``Ts`` is the sorted (descending) union of the
shifted breakpoints of all curves, temperatures closer than 1e-9 K being one
grid point. Every temperature at which any stream's curve bends or jumps is
therefore a grid point, and between two consecutive grid points every stream
is linear to within 0.002 K. A grid of the stream end temperatures alone
would average a boiling point, a dew or bubble point, or the curvature of a
glide that falls strictly inside an interval over that interval, which can
hide a pinch and make the hot utility target too low.

**Per-stream contributions.** Write :math:`H_j^-(T)` and :math:`H_j^+(T)` for
the low- and high-enthalpy limits of stream :math:`j`'s curve at the *real*
temperature :math:`T + \mathrm{shift}_j`; they differ only where the curve has
a flat at that temperature, and they are :math:`H_{hi}` above and
:math:`H_{lo}` below the stream's range. The heat a stream contributes to the
open interval between grid temperatures :math:`T_k` and :math:`T_{k+1}`, and
at the grid temperature :math:`T_k` itself, are

.. math::

   \mathrm{interval\_H}[j,k] = s_j \left( H_j^-(T_k) - H_j^+(T_{k+1}) \right),
   \qquad
   \mathrm{point\_H}[j,k] = s_j \left( H_j^+(T_k) - H_j^-(T_k) \right),

with :math:`s_j = +1` for a hot stream and :math:`-1` for a cold one. The
stream's own end points are grid points selected by position rather than by
a float comparison, so the contributions telescope exactly:

.. math::

   \sum_k \mathrm{interval\_H}[j,k] + \sum_k \mathrm{point\_H}[j,k]
       = s_j \left| H_{out} - H_{in} \right|,

that is, to the stream's duty.

**Point loads.** ``point_H`` is nonzero only where a curve is flat: at a pure
component's (shifted) saturation temperature, which receives its latent heat;
at an end temperature where a non-equilibrium end state departs from
equilibrium; and at the shifted outlet temperature of a *point-load stream*
-- an isothermal stream, or one whose outlet temperature moves *against* its
duty (a heated stream that leaves cooler than it entered, such as a reboiler
outlet quenched to equilibrium) -- which has no interval to occupy and puts
its whole duty :math:`s_j |H_{out} - H_{in}|` there.

**The cascade.** Starting from zero hot utility, the heat leaving grid
boundary :math:`T_k` (after that boundary's point loads) is

.. math::

   \mathrm{residual}[k] = \sum_j \sum_{i \le k} \mathrm{point\_H}[j,i]
                        + \sum_j \sum_{i < k} \mathrm{interval\_H}[j,i].

Feasibility must hold for the heat *arriving* at each boundary as well, before
its point loads are applied,

.. math::

   \mathrm{arriving}[k] = \mathrm{residual}[k]
                        - \sum_j \mathrm{point\_H}[j,k],

because a point source located at :math:`T_k` cannot serve a sink above
:math:`T_k`. hensmith therefore takes the elementwise minimum of the two
flows; the most negative value is the deficit hot utility must make up,

.. math::

   \mathrm{hot\_util\_load} = -\min_k \min(\mathrm{residual}[k],
                                           \mathrm{arriving}[k]),

its first location is the pinch, and the heat left at the bottom of the
cascade is the cold utility,
:math:`\mathrm{cold\_util\_load} = \mathrm{residual}[-1] +
\mathrm{hot\_util\_load}`. Whether that minimum is the arriving or the leaving
flow at the pinch fixes the side of the pinch that the point loads *at* the
pinch temperature belong to, the pinch *cut*.

Because every stream is linear to within 0.002 K between grid points, the
grid minimum of the cascade is the true one to within 0.002 K times the sum of
the heat capacity flow rates; equivalently, the targets lie between the exact
targets at ``T_min_app`` minus and plus 0.004 K (approximately: the 0.002 K
tolerance is established by midpoint tests). For streams of constant heat
capacity the curves are exact.

**Threshold problems.** When that minimum is not negative -- or negative by no
more than 1e-9 of the total stream duty -- no hot utility is needed at all.
The table then reports zero hot utility and places the pinch at the top of the
grid, ``Ts[0]``. A cold utility that comes out slightly negative through
rounding is absorbed back into the hot utility so that the identity

.. math::

   \mathrm{hot\_util\_load} - \mathrm{cold\_util\_load}
       = \sum_j \text{(stream duty)}

stays exact: the difference between the two targets is always the net heating
demand of the whole set of streams. That identity is the table's own
consistency check -- whatever the algorithm does with the cascade, it can
neither invent nor destroy energy.

Approach temperature contributions
----------------------------------

A single ``T_min_app`` holds every match to the same driving-force floor,
whether its streams condense and boil with high film coefficients or are
gases with low ones. Each stream :math:`j` may therefore carry its own
*approach temperature contribution* :math:`c_j`, in K, as described by Kemp
(2007): a process match between hot stream :math:`h` and cold stream :math:`c`
must keep :math:`c_h + c_c` everywhere inside it, on the exact states, and a
stream without a contribution contributes ``T_min_app / 2``, so that without
contributions every pair requires ``T_min_app`` exactly as before. A small
contribution suits a condensing or boiling stream, a large one a gas.
:func:`~hensmith.problem_table` takes the contributions as a sequence aligned
with its streams, :func:`~hensmith.synthesize_network` as a sequence aligned
with its heat utilities, a mapping or a callable, and
:class:`~hensmith.HeatExchangerNetwork` as a mapping or a callable (see
below).

**The shift.** Writing :math:`\Delta T_{min}` for ``T_min_app``, every stream
is shifted by its own amount, :math:`T^* = T - \mathrm{shift}_j`, with

.. math::

   \mathrm{shift}_h = \tfrac{1}{2} \Delta T_{min} + c_h \quad \text{(hot)},
   \qquad
   \mathrm{shift}_c = \tfrac{1}{2} \Delta T_{min} - c_c \quad \text{(cold)},

so that

.. math::

   T^*_h \ge T^*_c \iff T_h - T_c \ge \mathrm{shift}_h - \mathrm{shift}_c
                     = c_h + c_c.

Two streams that meet on the shifted scale are therefore exactly their pair's
required approach apart, and the problem table above is built on that scale
with no other change. A contribution is None or nan (unspecified) or a finite
number greater than zero. Zero is excluded so that every pair keeps a
positive driving force: a match allowed to touch would have none at its pinch
and need an infinite area. (The ``HXprocess`` guard is the requirement less
1e-6 K, so a pair whose contributions sum to 1e-6 K or less still gets a
guard of zero or less, as ``T_min_app <= 1e-6`` K does by default.) Whenever
a contribution is given, ``T_min_app`` must be finite and positive as well,
since it sets the default ``T_min_app / 2``; without contributions, or with
every one unspecified, ``T_min_app = 0`` is accepted as before.

**The default is unchanged, bit for bit.** An unspecified stream gets
exactly the default shifts, ``T_min_app`` if hot and 0 if cold
(:math:`x/2 + x/2 = x` and :math:`x/2 - x/2 = 0` hold exactly in floating
point), and with nothing specified the code takes the old expression
literally. A run without contributions, with every contribution unspecified
or with every one equal to ``T_min_app / 2`` gives the same targets and the
same network, number for number.

**One shifted scale for the planner.** The planner's search sees one curve
per stream on the shifted scale, not the shifts themselves, and on that scale
a match is feasible exactly when its real streams keep their pair's
requirement. The MER targets, the pinch design rules, the remaining problem
analysis and the guarantee of `Stream splitting`_ below therefore carry over
unchanged; a branch is its parent's curve with every heat scaled by its
fraction, so it keeps its parent's shift.

**Per pair on the real scale.** What works on real temperatures uses the
pair's own value. A pair's required approach is
:math:`\mathrm{shift}_h - \mathrm{shift}_c` -- equal to :math:`c_h + c_c` up
to round-off, and to ``T_min_app`` exactly without contributions. The exact
approach verification and the repair described below enforce it, each
``HXprocess`` gets it less 1e-6 K as its ``dT``, and
``synthesis_info['min_approach_margin']`` reports the smallest margin of any
exchanger's exact approach over its own pair's requirement (no less than
-1e-6 K, the check's rounding guard), next to ``'min_approach'``, the
smallest approach itself. Wherever this page says that a match keeps
``T_min_app``, read its pair's required approach.

**Per-stream pinch temperatures.** The pinch of the problem table, and the
``'pinch_T'`` of ``synthesis_info['plan_targets']``, stay on the shifted
scale. Stream :math:`j`'s own pinch temperature is that pinch plus
:math:`\mathrm{shift}_j` -- ``T_min_app`` above it for a hot stream and at
it for a cold one by default -- and that is what the facility's ``pinch_Ts``
reports. The real composite curves no longer keep one fixed distance: they
can come as close as the smallest hot plus the smallest cold contribution,
and only the shifted curves carry the pinch.

**Utilities are not affected.** The contributions apply to the process
matches only: each stream's utility exchanger gets its agent from BioSTEAM's
own rule (``HeatUtility.dT`` plus the agent's ``dT``), whatever the stream's
contribution.

**Keys in the facility.** :class:`~hensmith.HeatExchangerNetwork` takes a
mapping or a callable, since the order of its streams is internal. A mapping's
keys name units: a unit object (matched by identity), the ID of a unit
without an owner (``'D1'``), or ``(owner.ID, unit.ID)`` for an auxiliary
unit (``('D1', 'condenser')``; a bare auxiliary ID such as ``'condenser'``
names nothing). The ``owner`` of a pair is the top owner, so a pair names
every auxiliary of that ID under it, at any depth: if a unit holds two
auxiliary flash drums, each with its own ``heat_exchanger``, the pair
``('U', 'heat_exchanger')`` names both heaters, and the heater objects
or the drums (``('U', 'flash_a')``) tell them apart. A heat utility takes
the value of the key nearest to its unit along the unit's chain of owners,
so ``'D1'`` covers a column's condenser and reboiler and
``('D1', 'condenser')`` overrides it for the condenser. A None or nan value
is an explicit default that wins at its level like any value, and two keys
of the same level for a unit must agree. ID keys are the robust form: a
unit that rebuilds its auxiliary units at every simulation
(``biosteam.HXutilities``, for one) leaves a key holding an old auxiliary
object, which then names the auxiliary that took its place: the one of the
same ID under owners of the same IDs. Heat utilities cannot be keys,
because BioSTEAM replaces them at every simulation; a value per heat
utility is given by a callable
``f(hu)`` instead, which is called once per heat utility at every simulation
and should therefore be pure (after a simulation, ``HXN.targets()`` takes the
values that simulation resolved instead of calling it again, since
``replace_unit_heat_utilities`` overwrites the heat utilities by then). The
keys and values are checked against every unit available to the network
and its auxiliary units before any heat utility is filtered out: a key that
names none of them, a bad value or two
conflicting keys raise a ``ValueError`` even while their unit is idle, but a
valid key whose unit holds no heat utility in a simulation (no duty,
``ignored``, ``hxn_ok=False``) has no effect, so that a Monte Carlo sample in
which a unit falls idle does not fail.

**The cache.** With ``cache_network=True``, a cached network is reused only
while the explicit contributions resolve to the same value for every stream
(see *Reusing a network* below). ``T_min_app`` is never part of the reuse
test, with or without explicit contributions, so changing it alone keeps a
cached network, and a reused network keeps the streams without a
contribution at the ``T_min_app / 2`` it was synthesized with (as does
``synthesis_info['approach_contributions']``).

**Additive requirements only.** A requirement that belongs to a pair but is
not the sum of two stream terms cannot be written as one shift per stream,
so it has no problem table and is not supported.

Targets: MER, pinch, composite and grand composite curves
---------------------------------------------------------

The two loads returned by the table are the **minimum energy requirement**
(MER) targets: the least hot and cold utility any network operating with this
``T_min_app`` can use. They are available before a single exchanger has been
placed, which is what makes pinch analysis a *targeting* method -- the target
comes first, and the network is judged against it.

The same information can be read graphically. The **composite curves** plot
cumulative enthalpy against temperature for all hot streams together and all
cold streams together, on the real temperature scale. Where the two curves
overlap horizontally, heat can pass from hot to cold: that overlap is the heat
recovered by process exchange. The overhang of the cold curve at its warm end
is the hot utility, and the overhang of the hot curve at its cold end is the
cold utility. The place where the curves come closest vertically is the pinch,
and by construction they approach no closer than ``T_min_app`` (with approach
contributions, no closer than the smallest hot plus the smallest cold
contribution; the shifted curves carry the pinch).

.. figure:: /_static/images/examples/tutorial_02_composite_curves.png
   :class: white-bg
   :width: 720
   :alt: Composite curves of the quickstart system: a red hot composite curve above a blue cold composite curve, on axes of temperature in degrees Celsius against enthalpy in GJ/hr, with a shaded band marking the recovered heat, a cold utility bracket labelled 1.94e+06 kJ/hr at the cold end and a hot utility arrow labelled 2.83e+08 kJ/hr at the warm end.

   Composite curves of the quickstart system at ``T_min_app = 5`` K. The
   shaded band is the heat the two curves can exchange with each other; the
   arrows are the two MER targets, a hot utility of 2.828e+08 kJ/hr and a cold
   utility of 1.936e+06 kJ/hr. The curves come closest at the cold end, where
   the pinch of this system lies -- 298.15 K on the shifted scale.
   :doc:`tutorial/02_pinch_analysis` builds this figure from a
   :class:`~hensmith.ProblemTable`; :doc:`tutorial/05_targeting` draws it
   with the library.

The **grand composite curve** plots the same cascade differently: the heat
carried through each shifted grid temperature once the minimum hot utility is
supplied, against that temperature. It touches zero exactly at the pinch for a
pinched problem; for a threshold problem, where no hot utility is needed, the
curve's minimum may be strictly positive and the pinch is placed at the top of
the grid, ``Ts[0]``, by convention. Its shape shows where in the temperature
range utility has to be added or removed.

Both curves are library functions: :class:`~hensmith.PinchTargets` returns
them (:meth:`~hensmith.PinchTargets.composite_curves`, on the real or the
shifted scale, and :meth:`~hensmith.PinchTargets.grand_composite_curve`), on
the knots of the table's own grid, so they reproduce the table to rounding;
:func:`~hensmith.plot_composite_curves` and
:func:`~hensmith.plot_grand_composite_curve` draw them.

Both pictures express the three rules of pinch design: **no heat may cross the
pinch**, **no cold utility may be used above it**, and **no hot utility below
it**. Violating any one of them makes the network use more of both utilities
than the targets require, by the amount transferred across the pinch.

Multiple utilities on the grand composite curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MER targets say how much heat the utilities must supply and remove, not
which utilities. :meth:`~hensmith.PinchTargets.utility_targets` places
BioSTEAM's heating and cooling agents on the grand composite curve, heating
and cooling separately, and prices each one with BioSTEAM.

**Levels.** BioSTEAM lets a heating agent serve a stream only below
``agent.T - agent.dT - dT``, and a cooling agent only above ``agent.T +
agent.dT + dT``, where ``dT`` is ``bst.HeatUtility.dT``, read at call time.
On the shifted scale that limit is the agent's **level**: the limit itself for
a heating agent and the limit less ``T_min_app`` for a cooling agent, which is
BioSTEAM's rule exactly. With approach contributions the streams have
different shifts, and an agent's level is the highest single shifted
temperature (the lowest, for cooling), up to the highest of the streams' own
limits, at which it serves no stream beyond that stream's own real limit. A
level serves only what lies below it (above it, for cooling), so a stream
lying wholly beyond its limit bounds the level only at the stream's own end;
streams with slack are then served conservatively.

**List order and fuel.** Non-fuel agents are placed in the order of
BioSTEAM's lists, BioSTEAM's own "first feasible agent" rule: for the default
lists, the lowest pressure steam first, and the warmest coolant first. Each
takes as much of the remaining duty as the curve above its level (below it,
for cooling) allows, given the agents already placed. What remains of the
heating goes to the first fuel agent, a furnace at the top of the curve; if
there is none, or if some cooling remains, it is left unserved, with a
``RuntimeWarning`` and a nan cost. A level equal to a grid temperature
serves that temperature's point loads; BioSTEAM's own rule is strict, and the
two differ only at exact equality. The exception is a sensible agent whose
line would have no driving force left there: with ``agent.dT`` of 0
(``UtilityAgent``'s default), it could serve a point load at its limit only
at its own inlet temperature, at an unbounded flow, so its level stays just
short of that point load, which the next agent serves, as in BioSTEAM.

**Served intervals and pricing.** Each agent is priced as one BioSTEAM heat
utility over the stretch it serves: from its level, or from the end of the
grid if that is nearer, to the next zero of the **balanced grand composite
curve** (the cascade net of the utilities), which is the pinch or a *utility
pinch* where a lower agent is used to its limit.
A target's ``T_in`` and ``T_out`` are the real temperatures of the streams it
serves in that stretch: the coldest and the hottest for heating, the hottest
and the coldest for cooling. A point load at an end of the stretch counts
where the stretch holds it: a boiler at the pinch, whose latent heat the cut
puts above it, is served at its boiling temperature. Without approach contributions
``T_in`` and ``T_out`` are the stretch's ends where streams reach them, and in
every case ``T_out`` is within the agent's limit. BioSTEAM prices a latent
agent and fuel from them.

A **sensible** agent, such as cooling water, warms (or cools) along a line as
it takes its duty, and BioSTEAM checks that line only at the process inlet: a
coolant leaves at ``T_in - dT`` or at its limit, whichever is lower. Inside
the stretch the line must keep ``dT`` from the process too. Cooling water that
serves a condenser at the cold end of its stretch, say, has taken most of its
duty by the time it passes the condensing temperature, so it must leave well
below ``T_in - dT``. At each point of the stretch the agent has taken at least
the duty less the balanced heat flowing there (on the grand composite curve:
its line stays inside the curve, as a non-isothermal utility's does in pinch
analysis), and that bounds its outlet. hensmith prices the agent at the
highest outlet the bound allows (the lowest, for heating), by giving BioSTEAM
the matching inlet. Its flow, and therefore its cost, is then the least a
single stream of the agent needs. With approach contributions the bound takes,
at each point, the lowest real temperature of the hot streams there (the
highest of the cold ones), which is conservative.

BioSTEAM divides each duty by the agent's heat transfer efficiency, so a
target's ``utility_duty`` is utility side while its ``duty``, like the MER
targets, is process side.
Each pricing uses fresh utility streams in a temporary flowsheet: the costs
are deterministic, and the user's flowsheet, the agents and their later costs
are untouched.

Supertargeting: area, units, capital and total cost
---------------------------------------------------

Energy is one part of the cost of a network; its exchangers are the other.
Supertargeting targets the exchangers too -- their area, their number and
their capital cost -- before design, like the MER targets, and with them the
total annualized cost, which a sweep of ``T_min_app`` then minimizes
(Linnhoff & Ahmad, 1990; Ahmad, Linnhoff & Smith, 1990).
:meth:`~hensmith.PinchTargets.summary` returns every target at one
``T_min_app`` and :func:`~hensmith.supertarget` sweeps it.

**Balanced composite curves.** Adding the utility targets to the composite
curves -- the heating utilities to the hot curve, the cooling ones to the cold
curve -- gives two curves that span the same heat
(:meth:`~hensmith.PinchTargets.balanced_composite_curves`): a latent agent is a
flat at its temperature, a sensible one a line between its inlet and outlet
temperatures, and fuel a furnace that takes no area. If some utility is
unserved, the curves cannot balance, and the area, capital and total cost
targets are nan; the unserved cooling, the coldest part of the cold curve, is
left out as a gap at its start, which keeps the two curves aligned.

**Area.** The area target is Bath's formula (Townsend & Linnhoff, 1984) on
the balanced composite curves: heat is transferred vertically, from the hot
curve to the cold curve below it, in counter-current (a correction factor of
1). Split the heat axis at every breakpoint of either curve into intervals
:math:`i`; in each, stream or utility :math:`j` carries heat :math:`q_{ij}`,
and

.. math::

   A = \sum_i \frac{1}{\Delta T_{LM,i}} \sum_j \frac{q_{ij}}{h_j},

where :math:`\Delta T_{LM,i}` is the log-mean temperature difference of
interval :math:`i` on the real scale and :math:`h_j` the film coefficient of
:math:`j`. Every curve and every stream's share of it is linear in the heat
inside an interval, so the sum is exact. hensmith takes :math:`h_j` = 1
kW/m²/K for every stream and utility by default, which gives any pair
BioSTEAM's heuristic overall coefficient of 0.5 kW/m²/K (condensing against
boiling, where BioSTEAM takes 1.0, is not reproduced); ``h=`` and
``h_utilities=`` set others. A zero or negative driving force anywhere
makes the area infinite, with a ``RuntimeWarning``, as at ``T_min_app = 0``.

**Units.** A network of :math:`S` streams and utilities that forms one
connected component needs at least :math:`S - 1` units (Euler's rule). The
units target applies it to each region between the zeros of the balanced
grand composite curve -- the process pinch and every utility pinch --,
:math:`S` counting the streams with heat in the region and the utilities
acting in it. It is an estimate, **S - 1 per region, assuming one component
and no loop**: a subset of streams that balances on its own needs fewer
units, and each loop of matches one more.

**Capital.** The area is put in as many shells as the units target,
:math:`N`, less one furnace for all the fuel, and every shell is costed with
BioSTEAM's own correlations; the furnace is costed on the fuel's utility duty.
Linnhoff & Ahmad (1990) divide the area equally, :math:`N \cdot C(A / N)`:
with economies of scale (:math:`C = a + b A^c`) that is the costliest way to
divide it, and every unit added adds cost. BioSTEAM's correlations are not of
that form: a double pipe exchanger below 150 ft² costs at most about 8,100
USD installed, a shell and tube at 150 ft² about 64,700 USD. An equal division
can then put every shell below 150 ft², so that a utility of a few watts that
adds units would cut the capital target by most of it (the four-stream problem
of Linnhoff & Hindmarsh at 5.56 K: seven double pipes instead of four floating
heads, a fifth of the cost), although a network just adds a small exchanger
and keeps its large ones. So when the
equal division puts the shells below 150 ft² and the area fills
:math:`k = \lfloor A / 150\ \mathrm{ft^2} \rfloor \geq 1` shells of at least
150 ft², the target is the costlier of the :math:`N` double pipes and of
:math:`k` equal shell and tubes with :math:`N - k` shells of no area (each
costed as BioSTEAM costs a vanishing double pipe, about 1,300 USD installed).
This is the least change to the equal division that never falls when a unit
is added; where it departs from it, it tends to be high (networks keep only
part of their area in shell and tubes) as the equal division tends to be low.
The summary's ``'N_area_shells'``, ``'shell_area'`` and ``'shell_type'`` say
which shells carry the area. BioSTEAM's other specifics carry over: the
floating head correlation is not monotone below about 126 ft², which the
double pipe hides, and there is no largest shell area. All shells are costed
at one pressure, ``P``, whereas BioSTEAM takes each exchanger's highest inlet
pressure; for high pressure steam heaters its pressure factor is up to about
12 % higher.

**Total annualized cost.** With a capital recovery factor over :math:`n`
years at interest :math:`i`,

.. math::

   \mathrm{TAC} = \mathrm{CRF} \, C_{installed} + t_{op} \, C_{utility},
   \qquad \mathrm{CRF} = \frac{i (1 + i)^n}{(1 + i)^n - 1},

with :math:`i` = 10 %, :math:`n` = 10 years and :math:`t_{op}` = 8000 h/yr
by default, or taken from a BioSTEAM TEA (``tea=``). The targets of a facility
(``HXN.targets()``) take :math:`t_{op}` from its system when the system has
operating hours (a TEA sets them), as the network report does. It counts the
bare module cost of the exchangers only: no Lang factor and no working capital.

**The sweep.** :func:`~hensmith.supertarget` evaluates the targets over a grid
of ``T_min_app`` values, reusing the streams' curves (nothing is flashed
again), and then refines the grid around the least total annualized cost. The
cost is not smooth -- the units target jumps, the double pipe switches to a
shell and tube, the pinch moves -- so the refinement only narrows a bracket,
and ``at_bound`` flags an optimum at an end of the sweep. The capital target
drops by about a shell and tube wherever the shrinking area fills one fewer of
150 ft², so the least cost often lies just past such a step rather than at a
smooth trade-off of energy and capital: compare the ``'N_area_shells'`` and
``'shell_type'`` of its neighbors (``sweep.to_frame()``). A network has such
steps too, but where its own exchangers cross 150 ft², which the target's
equal shells place only roughly. Where the targets do
not depend on ``T_min_app`` -- below the threshold of a threshold problem, the
utilities and the balanced composite curves stay the same -- the costs differ
only by the discretization of the curves, by a tiny relative amount (4e-7
of the total cost for the quickstart system), so where the optimum lies in
that flat stretch means nothing physically; where the discretization trend is
steady, it lies at an end of the stretch (for the quickstart system, the
warmest point evaluated, just short of the threshold). A sweep with no finite
total cost (an unserved utility everywhere, say) is not refined and warns that
its optimum is meaningless. With approach
contributions, only the unspecified streams follow ``T_min_app``;
``scale_contributions=True`` scales the specified ones with it, the
contribution scaling of Ahmad, Linnhoff & Smith (1990).

**A network against its targets.** :func:`~hensmith.network_report` compares a
synthesized network with its targets: utility cost, units, area, capital and
total annualized cost. Its ``'area'`` re-sizes the network's exchangers like
the target sizes them (the same film coefficients, a correction factor of 1,
the integrated driving force along the exact stream profiles and the process
side duty), so it is comparable with the target; ``'area_biosteam'`` is
BioSTEAM's own sizing, which includes its correction factor, the terminal
log-mean temperature difference and the utility side duty. The report checks
that its targets and the network's streams are both of the units' current
states: after a stream or a unit changes, it raises a ``ValueError`` until the
system is simulated again. The comparison has limits:

- the Bath target bounds the area only with equal film coefficients and the
  same utilities;
- the network gives each stream one agent at its outlet, BioSTEAM's list
  rule, while the target allocates the agents on the grand composite curve,
  so the network can use costlier agents than the target (on the four-stream
  problem of Linnhoff & Hindmarsh at 10 K, the target is low pressure steam
  and cooling water, the network medium pressure steam and chilled water);
- the units target is an estimate in both directions, since Euler's S - 1
  assumes one component and no loop per region: a balanced subset of
  streams, or one agent serving a stream whose heat the target splits
  between two utility levels, needs fewer units; each loop of matches needs
  one more, whether a repeated match between two streams (two branches of a
  split stream meeting the same stream make one too) or two streams linked
  both through process matches and through a shared utility (for such a
  loop, the quickstart network at 5 K has 7 units against 6, see
  :doc:`tutorial/05_targeting`).

From targets to a network: the pinch-outward MER planner
--------------------------------------------------------

:func:`~hensmith.synthesize_network` takes the heat utilities of the process,
runs the problem table above, plans a network *without stream splits* that
reaches the MER targets whenever its search finds one (by default; with
``stream_splitting=True`` it splits streams where MER needs them, see `Stream
splitting`_ below), and realizes the plan
as BioSTEAM exchangers. Streams are numbered in a rearranged order -- heated
streams first, then cooled streams -- and every array, exchanger ID and life
cycle uses that index. The order only breaks ties in the planner's search: the
facility hands the utilities over sorted by signed duty, and
``sort_hus_by_T`` sorts them by inlet temperature instead.

**The planner's model.** Each stream enters the planner as its knots on the
problem-table grid: its enthalpy at every grid temperature inside its range,
with two knots where its curve has a flat. Every breakpoint of every curve is
a grid point and the table is linear between grid points, so the planner's own
cascade reproduces the table exactly -- the same targets, the same pinch and
the same pinch cut. Each stream is cut at the pinch into an above-pinch part
and a below-pinch part (a stream lying wholly on one side has an empty part on
the other), and the two sides are planned as two independent problems.

**Must and flex streams.** Above the pinch no cold utility may be used, so
every hot stream there is a **must**: process matches have to cool it
completely. The cold streams above the pinch are **flex** streams: whatever
their matches leave over is supplied by one hot utility at their far (hot)
end. Below the pinch the roles swap: no hot utility may be used, so every cold
stream is a must, and the hot streams are flex streams finished by one cooler
at their cold end. Putting a flex stream's utility at its far end loses
nothing: moving a stream's later matches toward its inlet never reduces the
approach of any match on it, because every stream's curve is monotone.

**Building each side from the pinch outward.** A depth-first search builds
each side one match at a time, starting at the pinch, where the driving forces
are smallest, and moving outward. A match of duty :math:`x` between a must and
a flex stream is feasible only if the two streams keep ``T_min_app`` at every
knot along it, so internal pinches -- a condensing vapor against a boiling
mixture, say -- are respected, not just the exchanger's terminals. Every step
also keeps the problem table of the *remaining* problem feasible (remaining
problem analysis): the largest duty that does so has a closed form, so no step
can make MER unreachable by its own table. Wherever that remaining table is
tight -- a pinch of the remaining problem -- the pinch design rules of Linnhoff
and Hindmarsh must hold:

- above the pinch, every hot stream at the pinch needs its own cold stream at
  the pinch, with :math:`C_{hot} \le C_{cold}` (the number rule
  :math:`N_{hot} \le N_{cold}` and the heat-capacity-flow rule);
- below the pinch, every cold stream at the pinch needs its own hot stream at
  the pinch, with :math:`C_{hot} \ge C_{cold}`.

Both rules are generalized to isothermal segments: a flat -- a pure component
boiling or condensing exactly at the pinch -- has unlimited series capacity
and can serve several partners in turn. At the process pinch itself a
violation of these rules *proves* that MER needs stream splitting.

The candidate duties of a match are the largest feasible one and a finite set
of events: a stream is ticked off, a partner is saved for another stream, a
stream switches partner or returns to an earlier one. The search is budgeted
in deterministic work units rather than seconds, so its result does not depend
on the speed of the machine. Once a MER plan is found, a branch and bound on
the number of exchangers looks for a smaller one; consecutive pieces of the
same match are merged into one exchanger.

**Repeated pairs.** The same hot and cold stream may be matched more than
once on the same side: alternating two partners in series emulates a split,
and some unsplit MER networks need it (a real split is optional, see `Stream
splitting`_). Process exchangers are named
``HX_<cold>_<hot>_hs`` above the pinch (the hot-side design) and
``HX_<hot>_<cold>_cs`` below it (the cold-side design), the first number being
the stream at port 0; the *n*-th exchanger of the same pair on the same side,
counted in the order the hot stream meets them, gets the suffix ``_<n>`` for
:math:`n \ge 2` (for example ``HX_3_2_cs_2``). Utility exchangers are
``Util_<index>_hs`` for a cold stream and ``Util_<index>_cs`` for a hot one.
``avoid_recycle=True`` forbids matching any pair twice anywhere -- on one side
or across the two -- so that no two exchangers connect the same pair of
streams, at the cost of the MER networks that need a repeated pair.

**Best effort when splitting is needed.** By default, a side whose pinch rules
prove that MER needs a split, or whose search runs out of budget, gets a
best-effort plan instead (with ``stream_splitting=True`` it is planned with
splits, see `Stream splitting`_). Heat that a must stream cannot place (a *gap*) is moved to the
stream's pinch end, where it crosses the pinch at the cost of an equal amount
of extra hot and cold utility, the *penalty*; greedy dives and a bisection of
the gaps keep that penalty small, though not minimal in general. Such a
network reports ``'best_effort'``.

**Small matches.** A planned exchanger with a duty below ``Qmin`` (default
``1e-3`` kJ/hr) is dropped and its duty left to the utilities. Removing a
match never reduces the approach of another, so the rest of the plan stays
feasible; a large ``Qmin`` can, however, cost MER.

**Realization.** Each stream is walked in flow order from its inlet: a hot
stream through its above-pinch matches (from its inlet end), then its
below-pinch matches, then its cooler; a cold stream through its below-pinch
matches, then its above-pinch matches, then its heater. Each match becomes one
plain ``HXprocess`` whose two enthalpy limits, ``H_lim0`` and ``H_lim1``, are
the planned outlet enthalpies of its two streams, so that its duty reproduces
the plan. A planned outlet whose equilibrium state is not past the stream's
state at the exchanger inlet cannot be a limit (``HXprocess`` rejects it): one
strictly inside a non-equilibrium end jump, or a state of a point-load stream
on the wrong side of its inlet. That stream's limit is left out, and the other
stream's sets the duty. A stream's first exchanger receives its real inlet --
except a point-load stream, which enters at equilibrium at its inlet enthalpy,
on the side of its outlet temperature where the plan put its duty -- and every
later exchanger receives the stream's exact state at the planned enthalpy.
Each exchanger is simulated once; one whose duty differs from the plan is
reported, and one that cannot be simulated at all is dropped, its duty going
to the utilities. Finally one rigorous ``HXutility`` per stream takes it to its outlet
enthalpy, and an ``AssertionError`` is raised if that does not reproduce the
quenched outlet enthalpy and temperature, so a network that would not deliver
the specified outlets fails loudly rather than silently.

**Exact approach verification.** The knots are exact at grid points but are
chords in between, up to 0.002 K off inside glides and curved single-phase
stretches. Every planned exchanger is therefore checked on the exact states of
its two streams wherever its planned approach is within that margin of
``T_min_app``: at its ends, at every breakpoint inside it and, where the exact
approach is not linear between two positions, by a search for a dip in
between. ``HXprocess`` itself checks only its two terminals, which would miss
an internal pinch at a phase change. Where a MER plan falls short by more than
1e-6 K, the exact states are inserted as knots and the network is planned
again, for at most three rounds. A best-effort plan, or a MER plan still short
after the last round, instead has each violating match shrunk to the largest
duty that keeps the approach, the rest going to the utilities: with its inlets
fixed, a smaller duty can only raise a match's approach, and the later stages
of both streams move toward their inlets, which never reduces another match's
approach. Streams of constant heat capacity never need either step.

**The report.** ``synthesize_network(..., info={})`` fills the dictionary it is
given, and :class:`~hensmith.HeatExchangerNetwork` keeps it as
``synthesis_info``. ``'status'`` is ``'mer'`` only if the utilities of the
*realized* network, computed from the simulated exchanger duties, equal the
targets (to 1e-6 of the total stream duty), and ``'best_effort'`` otherwise.
Next to it are the targets (``'Q_hot_target'``, ``'Q_cold_target'``), the
planned and the realized utilities, the ``'penalty'``, per side of the pinch
(``'sides'``) the search method, its work, any proof that a split is needed
and the gaps, the planner's own targets and pinch, the number of refinement
rounds, the smallest approach inside any process exchanger, the matches that
were shrunk (``'repaired'``), dropped (``'qmin_dropped'``, ``'dropped'``) or
deviated from their plan (``'deviations'``), and the point-load streams. With
``stream_splitting=True`` it also holds the realized splits (``'splits'``), any
mixer whose outlet is off its planned state (``'split_deviations'``) and, per
side, the split candidate chosen (``'split'``). The full list is under the
``info`` keyword of :func:`~hensmith.synthesize_network`.

Stream splitting
----------------

By default hensmith does not split streams, so a side of the pinch whose
pinch design rules prove that MER needs a split gets a best-effort network.
``HeatExchangerNetwork(..., stream_splitting=True)`` (or
``synthesize_network(..., stream_splitting=True, info={})``) lets the planner
split a process stream into parallel branches that re-join. Only a side that
no unsplit network serves at MER -- one with a pinch-rule proof, or whose
unsplit search leaves a utility penalty -- is planned with splits; a side that
an unsplit network serves is planned exactly as without the option, so a
problem that needs no split gets the same network either way.

**Branches.** A branch of flow fraction :math:`f` carries the parent stream's
material at the parent's pressure, so at branch heat :math:`q` it is in the
parent's state at heat :math:`q/f`: its temperature-enthalpy curve is the
parent's with every heat times :math:`f`, and its heat capacity flow rate is
:math:`f` times the parent's. Splitting a stream into branches whose fractions
sum to one leaves the problem table unchanged, so a split never changes the
targets; it changes only what the pinch design rules count (one stream per
branch, each with :math:`f\,C`) and the approach inside each exchanger. The
planner builds a split side from a portfolio of candidates -- splits at the
pinch that repair the number and heat-capacity-flow rules, completed by the
ordinary search, and a *vertical* construction that matches every stream's
branches at the same position of the composite curves -- verifies every
exchanger at every knot of both branch curves, and keeps the best candidate
by a key that first avoids tiny exchangers or branches and re-joins at
different temperatures ahead of another exchanger, then counts exchangers,
extra branches and split stages. The theory (the lemmas, the constructions
and their proofs) is in the module docstring of ``hensmith._splitting``.

**Realization.** Each split becomes a chain of BioSTEAM ``Splitter`` units,
one fewer than its branches (IDs ``Split_<stream>_<hs|cs>``, with ``_<n>`` for
a stream's *n*-th split on that side and ``_b<c>`` for the chain's element
:math:`c \ge 2`), and a rigorous, adiabatic ``Mixer`` (``Mix_<stream>_<hs|cs>``)
where the branches re-join; neither is costed. A branch exchanger runs its
fraction of the flow in the parent's states, with an enthalpy limit
:math:`f` times the parent's, and every mixer starts at its planned state. The
facility lists them in ``new_splitters`` and ``new_mixers`` and simulates them
in ``HXN_sys`` with the exchangers; ``synthesis_info['splits']`` describes
every split (:class:`~hensmith.hxn_synthesis.StreamSplit`), and a split
stream's life cycle marks each branch stage with its branch and fraction.

**The guarantee and its limits.** Without ``avoid_recycle``, every side that
needs a split gets a split plan at MER on the planner's knots: the vertical
construction always succeeds (up to a deficit of about 1e-9 of the total duty
that the problem table's own tolerances already absorbed into the targets).
With constant heat capacities the knots are exact and the realized network
reaches the targets to round-off. With real thermodynamics the knots are
chords of the exact curves, so a split exchanger can fall short of
``T_min_app`` on the exact states by up to the chords' tolerance; the same
refinement rounds as for unsplit networks close that, plus up to two rounds
that try another candidate, and only an exchanger still short after them is
shrunk (``'repaired'``, status ``'best_effort'``). With ``avoid_recycle``, a
split that would repeat a stream pair is not used, and MER is then not
guaranteed. The number of exchangers is minimized only among the candidates a
side generates, not globally. The test suite reaches ``'mer'`` with splits on
all 38 split-needing problems of its corpus and on the three regression
systems that need splits, each checked on its actual stream graph: splitters,
mixers at equilibrium, balances and the exact approach inside every exchanger.
:doc:`tutorial/04_configuring` shows an example.

Rigor and phase change
----------------------

Process streams in a biorefinery boil, condense and change composition, so
hensmith never assumes a constant heat capacity. Every enthalpy it uses comes
from thermosteam:

- **Quenched outlets.** Before any analysis, each stream's outlet copy is
  re-flashed at its own enthalpy (``s.vle(H=s.H, P=s.P)``). An upstream
  ``HXutility`` that was not solved rigorously can leave an outlet in a
  non-equilibrium state; quenching puts that heat at the temperature the
  equilibrium model says it is available at.
- **Curves instead of flashes on the grid.** Flashing every stream at every
  grid temperature has three defects that the stream curves avoid. A phase
  boundary or a curvature inside a grid interval is averaged away (see *The
  shifted grid* above). A ``vle(T=T_sat)`` of a single chemical keeps
  whatever phase split the stream had, so the enthalpy exactly at a
  saturation temperature depends on history; the curve takes a pure
  component's latent heat as a flat between V-specified saturated states and
  evaluates single-phase stretches with their phases fixed, so a grid point
  on a saturation temperature is never ambiguous. And thermosteam's
  two-phase flashes of some mixtures (water and ethanol with 20-50 % ethanol,
  for instance) silently return non-converged states; the curve traces a
  binary glide along its bubble-point curve and sanity-checks the flashes of
  other mixtures.
- **Point loads.** A pure component's latent heat contributes at its
  saturation temperature as a point load, and so does the whole duty of an
  isothermal or non-monotone stream at its outlet temperature, instead of
  being smeared over an interval; that is what keeps the cascade -- and the
  pinch it locates -- correct for latent duties.
- **Pinch states.** ``pinch_state`` returns the state of a stream at a pinch
  temperature from its curve, deterministically: a pinch on the stream's own
  saturation temperature is resolved by the side of the pinch cut, never by
  whatever phase split a previous flash left, and the enthalpy is clipped to
  the stream's real range, so the state never carries heat the real stream
  does not have. It is a standalone analysis helper; the synthesis plans on
  the curves themselves.
- **Rigorous exchangers.** Every synthesized process exchanger is an
  ``HXprocess`` simulated from exact inlet states with both enthalpy limits
  at its planned outlets and its approach verified on exact states, and every
  synthesized utility exchanger is an ``HXutility`` with ``rigorous=True``,
  specified by enthalpy rather than by temperature.

The network as a BioSTEAM system
--------------------------------

:class:`~hensmith.HeatExchangerNetwork` is a BioSTEAM ``Facility``. Its
``_run`` and ``_design`` do nothing; all of the work happens in ``_cost``,
which runs only after every process unit of the system has converged, so the
duties it integrates are final ones.

**A separate flowsheet.** The synthesis happens inside a temporary flowsheet
named ``<sys>_HXN``, where ``<sys>`` is the system's ID. The original streams,
exchangers and units are never modified (unless ``replace_unit_heat_utilities``
is set, described below): the network is built from copies, and
those copies and the new exchangers are what you inspect afterwards through
``new_HXs``, ``new_HX_utils`` and the ``stream_life_cycles``
(:class:`~hensmith.StreamLifeCycle`) that
:func:`~hensmith.plot_pinch_diagram` draws.

**Which streams take part.** The facility collects the heat utilities of its
``units`` (all units of the system unless a list or a callable is given),
drops any utility flagged as not usable for HXN integration
(``hxn_ok=False``, set by some unit operations) or with non-positive flow,
anything listed in ``ignored``, and anything with zero duty, and sorts what is
left by duty. Auxiliary exchangers -- a column's condenser and reboiler, a
flash's feed heater -- are included like any other.

**Convergence.** After synthesis each stream's stages are rewired in series
(the branches of a split in parallel, from its splitter chain to its mixer),
and the new units are assembled into a ``System``, ``HXN_sys``, whose path
follows the streams: every stage links to the next stage of the same stream,
and the path is a topological order of that graph (Kahn's algorithm,
ties broken by the order of the exchangers). Where the graph has a cycle --
a pair of streams matched both above and below the pinch, or repeated matches
in alternating order -- the unit with the fewest unplaced predecessors comes
next and its inlets from later units become recycle streams. Every exchanger
starts at its planned state, so the loops are at their fixed point after one
pass; the system is converged by fixed-point iteration to tight tolerances
(a temperature change of 1e-8 K), which closes the energy balance to about
1e-10 %. Should convergence raise, every unit is run once and a
``RuntimeWarning`` is issued.

**Streams served by process exchange alone.** A stream that its process
exchangers bring to its outlet (to within 1e-9 of its duty, the residual of
the enthalpy flashes) leaves its utility exchanger in exactly the state it
enters it, so that exchanger has no duty and no cost -- rather than a spurious
duty of a few 1e-9 kJ/hr from re-flashing the stream, which biosteam would
design and cost as a minimum-size exchanger.

**Where it sits among the facilities.** ``network_priority = -2`` is lower
than that of any other standard BioSTEAM facility, and facilities are
simulated in increasing order of priority, so the network is integrated before
the chilled water package, cooling tower and boiler are sized: they see the
loads it has already reduced.

**Costs and utilities are differences.** The facility's capital is the *added*
exchanger cost, ``max(0, new - original)`` for both installed and purchase
costs, so a network whose exchangers happen to be cheaper than the ones it
replaces is reported as adding nothing rather than as a credit. Its heat
utilities are the new utilities summed by agent with the *reversed* original
ones -- new minus original -- so a negative utility cost on the facility is a
saving. Setting ``replace_unit_heat_utilities=True`` instead overwrites each
original unit's heat utility with that of its own stream's utility exchanger,
reloads the utility costs of the unit and of its owner, and leaves the
facility itself carrying none; the original data are given back before the
network is costed again, so the next network is synthesized from the units'
own utilities. If no process match was made at all, the facility reports zero
capital and no utilities.

**Reusing a network.** With ``cache_network=True`` a later simulation checks
whether the set of units behind the heat utilities, and the explicit approach
contributions they resolve to, are unchanged; if they are, the
same network configuration is reused instead of being synthesized again. Each
life cycle's first inlet is copied from the current stream, and each process
exchanger keeps, as the enthalpy limit of its *must* stream (the hot stream
above the pinch, the cold stream below it), the fraction of that stream's
duty at which its limit sat at synthesis, so a changed feed rescales every
stage instead of letting the first one take the whole duty. Its partner, the
flex stream, transfers what that sets, but never past its own outlet, and
takes the rest to its utility exchanger; a port that had no limit at
synthesis gets none. The utility exchangers bring every stream to its new
outlet enthalpy. A reused network is not planned again, so it need not be at
MER for the new duties, and ``synthesis_info`` still describes the synthesis
that produced it. If the reused network then fails its outlet checks,
hensmith warns, discards the cache and re-synthesizes from scratch. If it fails
its energy balance instead, the cache is discarded and the network
re-synthesized silently; a warning is issued only if the freshly synthesized
network fails the same check.

Validation
----------

hensmith checks every network it synthesizes, and its test suite checks the
synthesizer against independent references.

**The energy balance.** After convergence, every process exchanger duty is
counted twice -- it satisfies a cooling demand and a heating demand at once --
and added to the new utility duties, and the total is compared with the
original utility duties, each utility weighted by its agent's heat transfer
efficiency:

.. math::

   Q_{bal} = \frac{2 \sum_p |Q_p| + \sum_u |q_u \eta_u|}
                  {\sum_o |q_o \eta_o|}.

``energy_balance_percent_error`` is :math:`100 (Q_{bal} - 1)`, reported for
diagnostics. The check itself compares the underlying fraction: when
:math:`|Q_{bal} - 1|` exceeds ``acceptable_energy_balance_error`` (a class
attribute, ``0.02``, i.e. 2%, overridable per instance), hensmith warns -- or
raises, if ``raise_energy_balance_error`` is set. A cached network that fails
this check is instead discarded and re-synthesized, as described above. A
converged network closes this balance to numerical precision.

**Outlet reproduction.** Each stream's life cycle must end in the same state
as the original exchanger's outlet: composition, pressure and enthalpy are
asserted stream by stream. The utility exchangers created during synthesis are
checked the same way, against the quenched outlet enthalpy and temperature.

**The approach inside every exchanger.** The synthesis verifies every process
exchanger on the exact states of its streams (see *Exact approach
verification* above) and records the smallest approach it found in
``synthesis_info['min_approach']``, and the smallest margin of an
exchanger's approach over its pair's required approach in
``synthesis_info['min_approach_margin']``; the tests re-check both with
temperatures computed independently of the code under test.

**MER identification and achievement.** ``tests/test_hxn_mer.py`` synthesizes
a corpus of 78 problems through the public facility. For 40 of them -- 24
constant heat capacity problems from the literature and 16
real-thermodynamics problems with condensers and boilers at the pinch,
desuperheating and subcooling, and binary glides; 11 with more than ten
streams -- an unsplit MER network exists, proven inside the suite by a
certificate network that is re-checked there (by plain arithmetic for
constant heat capacity); the synthesized network must reach the targets and
report ``'mer'``. For the other 38 (25 from the literature and 13 with real
thermodynamics; 13 with more than ten streams) the pinch design rules prove
that MER needs stream splitting, a proof re-derived in the test module; by
default the network must never beat the targets and must report
``'best_effort'``, and with ``stream_splitting=True`` it must reach them and
report ``'mer'``, checked on its actual stream graph (splitters, mixers at
equilibrium, every stream's closure). The 40 problems of the first set must
also plan exactly as without the option (18 of them are synthesized both
ways, and must give the same network). In both sets the
targets must equal an independent
reference -- a closed-form constant heat capacity cascade and the published
values, or a dense-grid calculator for real thermodynamics -- and every
material and energy balance and the exact internal approach of every
exchanger are checked.

**Regression cases.** ``tests/test_hxn_regression.py`` builds ten synthetic
systems of increasing complexity -- all with phase-changing streams from the
third on -- and for each one requires that the synthesized network (i) closes
its energy balance without raising ``RuntimeWarning``, (ii) never uses less
hot or cold utility than the MER targets computed on the same streams, and
reports ``'mer'`` exactly when it reaches them, (iii) keeps each pair's
required approach inside every process exchanger on exact states, (iv) is
planned on the problem table's own cascade, and (v) recovers at least as
much heat as a load recorded in the test file. A network that improves leaves slack in (v); those
recorded numbers are lowered deliberately by a maintainer, never raised to
make a failing test pass. Every case is synthesized again with
``stream_splitting=True`` and held to the same checks, and the three cases
that need stream splits for MER must then reach their targets.

**Approach contributions.** ``tests/test_hxn_contributions.py`` checks the
contributions against references that do not use them. Without
contributions -- and with every contribution unspecified or equal to
``T_min_app / 2`` -- plans, problem tables and facility networks must equal
the default ones bit for bit, compared in-process. Every stream at
:math:`x` must plan the network of ``T_min_app`` :math:`= 2x`, and on
real-thermodynamics cases all hot streams at :math:`a` and all cold ones at
:math:`b` must give the default targets at ``T_min_app`` :math:`= a + b`.
With mixed contributions, the targets of 40 random problems, of the 24
constant heat capacity no-split problems of the MER corpus and of five split
ones (which must reach MER with ``stream_splitting=True``) must equal a
closed-form constant heat capacity cascade with the classic symmetric shifts,
and every planned match must keep its pair's requirement on recomputed
temperatures. The facility, with and without splits, on constant heat
capacity and real-thermodynamics cases, must reach the contribution-aware
targets, keep each pair's requirement on exact states recomputed
independently and give each exchanger its pair's ``dT``; the exact check and
the repair must act on each pair's own requirement. The key rules, the
validation of keys and values (for idle units too, and before any stream is
flashed) and the ``cache_network`` rule are tested as well.

**Targeting.** ``tests/test_targeting.py`` checks ``hensmith.targeting``
against references computed independently of it: the composite curves of
the four-stream problem of Linnhoff & Hindmarsh in closed form, and the
composite and grand composite curves of constant heat capacity and
real-thermodynamics cases against their problem tables; the multiple-utility
allocations and served intervals against duties computed from the agents'
own attributes, each priced again in isolation; the area target against a
midpoint integration of hand-built composite curves (with approach
contributions and per-stream film coefficients too) and against closed
forms; the units target on eight reference problems, utility pinches and
balanced subsets among them; the exchanger costs against BioSTEAM's own; the
capital recovery factor against BioSTEAM's; and the swept optimum against
the closed-form total annualized cost of a two-stream problem. The
facility's targets must equal the synthesis's bit for bit, with and without
approach contributions; the network report's areas must equal independent
log-mean areas integrated along the network's own streams where these are at
equilibrium (the report follows each stream as the targets' curves model it,
like for like with the area target, so it may differ slightly for a
point-load stream, held at its outlet temperature, or a non-equilibrium end
state, such as a vapour entering below its dew point); and targeting must
leave the flowsheet's streams, the agents, later utility costs and a later
synthesis unchanged.

**Doctests.** The examples in the docstrings are executed as part of the test
suite, so the numbers printed in the API reference are numbers the code
currently produces.

Guarantees and limitations
--------------------------

What the synthesized network is guaranteed to be:

- **Never better than MER.** Its utilities are never below the targets of the
  problem table, and ``'mer'`` is reported only if the realized network
  reaches them.
- **Feasible everywhere inside.** Every process exchanger keeps its pair's
  required approach (``T_min_app``, or ``c_h + c_c``) minus 1e-6 K on the
  exact stream states at its ends and at every checked position inside it,
  and the heat balance closes on every stream.
- **Deterministic.** The search is budgeted in work units, not seconds, so
  the same streams give the same network however fast the machine is.

What it is not:

- **MER is reached whenever the planner finds an unsplit network -- which is
  an empirical, not a proven, property.** Every pruning test of the search is
  a necessary condition, so a missed MER network can only come from the
  finite set of candidate duties, the caps on repeated pairs or the work
  budgets. The planner reached MER on every problem of a certified benchmark
  of about 1,700 problems with 2 to 40 streams for which an unsplit MER
  network exists, and it does so on all 40 no-split problems of the test
  suite, but no proof covers every problem.
- **By default, streams are not split.** Every stream stays a single branch
  through the network. Where the pinch design rules prove that MER needs a
  split, the network is a best-effort one whose penalty is small but not
  minimal in general; repeated matches between the same two streams,
  alternating in series, can approach a split only in the limit. With
  ``stream_splitting=True`` such a side is split and reaches MER, within the
  limits given under `Stream splitting`_: exact on constant heat capacity
  streams, closed by refinement on real thermodynamics, not guaranteed with
  ``avoid_recycle``, and with the number of exchangers minimized only among
  the candidates the planner generates.
- **Energy first, then units; no cost optimization.** MER always takes
  precedence over the number of exchangers, and an unsplit MER network can
  need many of them. The branch and bound reduces the number of exchangers
  among MER plans, but the synthesis does not optimize area, capital or total
  cost; ``hensmith.targeting`` targets them, finds the ``T_min_app`` of the
  least targeted total annualized cost (see `Supertargeting: area, units,
  capital and total cost`_), and :func:`~hensmith.network_report` compares a
  network with the targets.
- **Some networks cannot be represented.** Networks whose match order is
  cyclic are outside the planner's model. A side that needs a split without a
  pinch-rule proof spends its whole MER search budget before the best-effort
  step (or the split attempt), which costs time rather than quality.
- **Flash failures inside some glides.** Thermosteam's TP flashes fail
  silently inside the glides of some mixtures (water and ethanol with 20-50 %
  ethanol, for instance); an exchanger simulated there can deviate from its
  plan, and is then reported in ``synthesis_info['deviations']`` (a split's
  mixer, in ``synthesis_info['split_deviations']``).
- **Only streams behind existing utility exchangers are integrated.** The
  facility sees a process stream only through a heat utility attached to a
  unit of the system. A duty carried some other way is invisible to it;
  utilities flagged as not usable for HXN integration (``hxn_ok=False``) or
  with non-positive flow, and streams excluded with ``ignored``, are removed
  from the analysis entirely.
- **Additive approach temperatures only.** Per-stream approach temperature
  contributions are supported, so a match's requirement is ``T_min_app`` or
  the sum of its two streams' contributions; a requirement that belongs to a
  pair and is not such a sum is not (see
  `Approach temperature contributions`_). The utility exchangers use
  BioSTEAM's approach rule whatever a stream's contribution.
- **Targets are estimates.** Beyond the MER targets, the targets of
  ``hensmith.targeting`` are estimates: the area target bounds a network's
  area only with equal film coefficients and the same utilities, the units
  target can be wrong in both directions, the capital target assumes shells
  of equal area at one pressure (keeping the shell and tubes the area fills
  when an equal division would drop below 150 ft²), and with approach
  contributions the utility levels are conservative and their pricing
  temperatures approximate. The optimum ``T_min_app`` is the optimum of these
  estimates, and often sits just past a step of the capital target where the
  area fills one shell and tube fewer: a step a network shares, though not
  at the same ``T_min_app``.

References
----------

- Linnhoff, B., & Hindmarsh, E. (1983). The pinch design method for heat
  exchanger networks. *Chemical Engineering Science*, 38(5), 745-763.
- Smith, R. (2005). *Chemical Process Design and Integration*. Wiley.
- Kemp, I. C. (2007). *Pinch Analysis and Process Integration: A User Guide on
  Process Integration for the Efficient Use of Energy* (2nd ed.).
  Butterworth-Heinemann.
- Seider, W. D., Lewin, D. R., Seader, J. D., Widagdo, S., Gani, R., & Ng,
  M. K. (2017). *Product and Process Design Principles*. Wiley. Heat Exchanger
  Networks (Chapter 9).
- Townsend, D. W., & Linnhoff, B. (1984). Surface area targets for heat
  exchanger networks. *IChemE 11th Annual Research Meeting*, Bath, UK.
- Linnhoff, B., & Ahmad, S. (1990). Cost optimum heat exchanger networks - 1.
  Minimum energy and capital using simple models for capital cost.
  *Computers & Chemical Engineering*, 14(7), 729-750.
- Ahmad, S., Linnhoff, B., & Smith, R. (1990). Cost optimum heat exchanger
  networks - 2. Targets and design for detailed capital cost models.
  *Computers & Chemical Engineering*, 14(7), 751-767.
- Cortes-Pena, Y., Kumar, D., Singh, V., & Guest, J. S. (2020). BioSTEAM: A
  fast and flexible platform for the design, simulation, and techno-economic
  analysis of biorefineries under uncertainty. *ACS Sustainable Chemistry &
  Engineering*, 8(8), 3302-3310. https://doi.org/10.1021/acssuschemeng.9b07040

.. seealso::

   :doc:`tutorial/02_pinch_analysis` applies the problem table to a real
   system; :doc:`tutorial/03_network_anatomy` walks through a synthesized
   network exchanger by exchanger; :doc:`API/api` documents every public name.
