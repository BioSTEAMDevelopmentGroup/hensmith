Key Concepts
============

.. currentmodule:: hensmith

This page collects the ideas hensmith implements: what a minimum approach
temperature is, how the problem table turns a set of process streams into
utility targets and a pinch, how :func:`~hensmith.synthesize_network` turns
those targets into a network of exchangers, and what the result is and is not
guaranteed to be. Everything below describes the behavior of the code in
``hensmith/hxn_synthesis.py`` and ``hensmith/_heat_exchanger_network.py``; the
:doc:`tutorial/index` shows the same concepts on a running system.

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
required between the streams of every candidate match, enforced on every
synthesized exchanger, and used to shift hot streams in the problem table:

- in :func:`~hensmith.problem_table`, where every hot stream's temperature is
  shifted *down* by ``T_min_app`` before the streams are compared, so that two
  streams which meet on the shifted scale are really ``T_min_app`` apart;
- in every matching pass of :func:`~hensmith.synthesize_network`, where it
  sets which candidates are eligible (a hot stream is only paired with a cold
  stream more than ``T_min_app`` below it) and enters the driving-force
  ranking that orders them; and
- on every synthesized process exchanger, which is a ``biosteam.HXprocess``
  constructed with ``dT=T_min_app`` and therefore stops transferring heat when
  its outlet temperatures come that close.

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

**The shifted grid.** Hot streams are shifted down by ``T_min_app``; cold
streams are not. The grid ``Ts`` is the sorted (descending) set of shifted end
temperatures of all streams, so the intervals between consecutive grid
temperatures are exactly the intervals over which the population of streams
does not change.

**Per-stream contributions.** For a monotone stream -- one whose outlet moves
in the direction its duty implies -- the heat contributed to the interval
between grid temperatures :math:`T_k` and :math:`T_{k+1}` is

.. math::

   \mathrm{interval\_H}[j,k] = s_j \left( H_j(T_k) - H_j(T_{k+1}) \right),
   \qquad s_j = +1 \; \text{(hot)}, \; -1 \; \text{(cold)},

where :math:`H_j` is obtained by flashing a copy of the stream at the *real*
temperature :math:`T + \mathrm{shift}` and clipping the result to
:math:`[H_{in}, H_{out}]`. The stream's own two end points are assigned
:math:`H_{in}` and :math:`H_{out}` by position rather than by a float
comparison, so the sum over intervals telescopes exactly:

.. math::

   \sum_k \mathrm{interval\_H}[j,k] = s_j \left| H_{out} - H_{in} \right|,

that is, to the stream's duty. The clipping matters: a stream copy flashed at
an interior temperature may carry more enthalpy than the real stream ever has
(a non-equilibrium outlet, for instance), and without it that stream would
inflate an interval and break the identity above.

**Point loads.** Isothermal streams, and streams whose outlet temperature
moves *against* their duty -- a heated stream that leaves cooler than it
entered, such as a reboiler outlet quenched to equilibrium -- have no interval
to occupy. They enter the table as a point load
:math:`s_j |H_{out} - H_{in}|` at their shifted outlet temperature, in
``point_H``.

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

its location is the pinch, and the heat left at the bottom of the cascade is
the cold utility,
:math:`\mathrm{cold\_util\_load} = \mathrm{residual}[-1] +
\mathrm{hot\_util\_load}`.

**Threshold problems.** When that minimum is not negative -- or negative by no
more than a tiny fraction of the total stream duty -- no hot utility is needed
at all. The table then reports zero hot
utility and places the pinch at the top of the grid, ``Ts[0]``. A cold utility
that comes out slightly negative through rounding is absorbed back into the
hot utility so that the identity

.. math::

   \mathrm{hot\_util\_load} - \mathrm{cold\_util\_load}
       = \sum_j \text{(stream duty)}

stays exact: the difference between the two targets is always the net heating
demand of the whole set of streams. That identity is the table's own
consistency check -- whatever the algorithm does with the cascade, it can
neither invent nor destroy energy.

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
and by construction they approach no closer than ``T_min_app``.

.. figure:: /_static/images/examples/tutorial_02_composite_curves.png
   :class: white-bg
   :width: 720
   :alt: Composite curves of the quickstart system: a red hot composite curve above a blue cold composite curve, on axes of temperature in degrees Celsius against enthalpy in GJ/hr, with a shaded band marking the recovered heat, a cold utility arrow of 1.936e+06 kJ/hr at the cold end and a hot utility arrow of 2.828e+08 kJ/hr at the warm end.

   Composite curves of the quickstart system at ``T_min_app = 5`` K. The
   shaded band is the heat the two curves can exchange with each other; the
   arrows are the two MER targets, a hot utility of 2.828e+08 kJ/hr and a cold
   utility of 1.936e+06 kJ/hr. The curves come closest at the cold end, where
   the pinch of this system lies -- 298.15 K on the shifted scale.
   :doc:`tutorial/02_pinch_analysis` builds this figure from a
   :class:`~hensmith.ProblemTable`.

The **grand composite curve** plots the same cascade differently: the heat
carried through each shifted grid temperature once the minimum hot utility is
supplied, against that temperature. It touches zero exactly at the pinch for a
pinched problem; for a threshold problem, where no hot utility is needed, the
curve's minimum may be strictly positive and the pinch is placed at the top of
the grid, ``Ts[0]``, by convention. Its shape shows where in the temperature
range utility has to be added or removed.

Both pictures express the three rules of pinch design: **no heat may cross the
pinch**, **no cold utility may be used above it**, and **no hot utility below
it**. Violating any one of them makes the network use more of both utilities
than the targets require, by the amount transferred across the pinch.

From targets to a network: hensmith's synthesis heuristics
----------------------------------------------------------

:func:`~hensmith.synthesize_network` takes the heat utilities of the process,
runs the pinch analysis above, and then places exchangers. Streams are
numbered in a rearranged order -- heated streams first, then cooled streams --
and every array, exchanger ID and life cycle uses that index.

Each stream gets a **pinch temperature** of its own: the process pinch on the
stream's own scale (the table's ``pinch_T`` for a cold stream, that plus
``T_min_app`` for a hot one) when the stream crosses it; its inlet temperature
when the stream already starts past the pinch, or is isothermal or
non-monotone; its outlet temperature when the stream ends before reaching the
pinch. That temperature splits the stream's duty into a hot-side (above-pinch)
part and a cold-side (below-pinch) part, and the state of the stream there --
computed by ``pinch_state``, with enthalpy clipped to the stream's real range
-- is the state in which it enters the design on the far side of the pinch.
Each design works with its own **transient stream** per stream index, advanced
every time a match is made, so a candidate is always evaluated at the state
the stream has actually reached rather than at its original inlet. The passes
walk the streams by index, so the stream order is the matching priority. The
facility hands the utilities over sorted by signed duty, so by default the cold
stream with the smallest heating duty and the hot stream with the largest
cooling duty are tried first; ``sort_hus_by_T`` replaces that with inlet
temperature.

Matching then proceeds in four passes, each creating ``HXprocess`` units that
exchange as much heat as the approach temperature (``dT``), the outlet
enthalpy of one stream (``H_lim0``) and a temperature limit on the other
(``T_lim1``) allow:

1. **Cold-side design.** For each hot stream, the eligible cold streams are
   those with a heat-capacity flow rate no greater than the hot stream's
   (:math:`C_{hot} \ge C_{cold}`) and a current temperature more than
   ``T_min_app`` below it. They are tried in decreasing order of

   .. math::

      \min(C_{hot}, C_{cold}) \cdot (T_{hot} - T_{cold} - T_{min,app}),

   a rough measure of how much heat the match can move, with ``H_lim0`` the
   hot stream's outlet enthalpy and ``T_lim1`` the cold stream's pinch
   temperature; the loop ends as soon as the hot stream reaches its outlet
   enthalpy. Streams lying entirely above the pinch are skipped, and so are
   isothermal and non-monotone streams, whose pinch temperature equals their
   inlet temperature and therefore marks them unavailable on both sides.

2. **Hot-side design.** The mirror image, run per cold stream, with the
   heat-capacity inequality reversed (:math:`C_{cold} \ge C_{hot}`), the same
   ranking, ``H_lim0`` the cold stream's outlet enthalpy and ``T_lim1`` the
   hot stream's pinch temperature; streams lying entirely below the pinch are
   skipped.

   The two inequalities are the feasibility criteria of the pinch design
   method: immediately below the pinch a match can only keep the approach
   temperature over its whole length if the hot stream's heat-capacity flow
   rate is at least the cold stream's, and immediately above it the reverse.

3. **Offset passes.** Two clean-up passes, one for the heating still owed on
   the cold side and one for the cooling still owed on the hot side, walk the
   streams in index order and match any pair that still has opposite demands
   on that side and is at least ``T_min_app`` apart. These passes drop the
   heat-capacity inequality and use the limited stream's own *outlet*
   temperature instead of its pinch temperature as ``T_lim1``; in the
   hot-side pass the cold stream is taken in whichever of its two transient
   states carries the most enthalpy.

4. **Utility exchangers.** One rigorous ``HXutility`` per stream finishes the
   job, taking the stream from its furthest transient state to its required
   outlet enthalpy. The result is asserted against the stream's quenched
   outlet enthalpy and temperature, so a network that would not actually
   deliver the specified outlets fails loudly rather than silently.

Three settings guard the passes. ``Qmin`` (default ``1e-3`` kJ/hr) discards
any candidate exchanger whose duty comes out below it. A match whose
``HXprocess`` cannot be simulated is discarded too, and because each candidate
is keyed by its exchanger ID -- ``HX_<hot>_<cold>_cs`` on the cold side,
``HX_<cold>_<hot>_hs`` on the hot side -- **a given ordered pair is attempted
at most once per side**, across the design pass and the offset pass that share
that ID namespace. Finally, ``avoid_recycle=True`` refuses any pair already
matched anywhere in the four passes, so that no two exchangers connect the
same pair of streams -- a second exchanger between the same two streams can
form a recycle loop in the network.

The passes are what make the result a *heuristic* network: every match is
committed as soon as it is made, and no pass revisits an earlier decision.

Rigor and phase change
----------------------

Process streams in a biorefinery boil, condense and change composition, so
hensmith never assumes a constant heat capacity. Every enthalpy it uses comes
from a thermosteam flash:

- **Quenched outlets.** Before any analysis, each stream's outlet copy is
  re-flashed at its own enthalpy (``s.vle(H=s.H, P=s.P)``). An upstream
  ``HXutility`` that was not solved rigorously can leave an outlet in a
  non-equilibrium state; quenching puts that heat at the temperature the
  equilibrium model says it is available at.
- **Interval enthalpies.** Within the problem table a single stream copy is
  walked down the grid, so each flash is warm-started from the previous
  boundary, and every result is clipped to the stream's own enthalpy range. If
  a flash fails, hensmith warns and interpolates that boundary's enthalpy
  linearly in temperature rather than abandoning the table.
- **Pinch states.** ``pinch_state`` flashes the inlet copy at the stream's
  pinch temperature and accepts the result only when its enthalpy lies inside
  the stream's real range; otherwise the stream never passes through that
  equilibrium state, and the equilibrium state at the nearer *end* enthalpy is
  used instead. Either way the hot-side and cold-side loads split
  :math:`|H_{in} - H_{out}|` exactly, and the transient stream used for
  matching never carries heat the real stream does not have.
- **Point loads.** A condenser or reboiler stream that changes phase at one
  temperature contributes its whole duty at that temperature instead of being
  smeared over an interval, which is what keeps the cascade -- and the pinch
  it locates -- correct for latent duties.
- **Rigorous exchangers.** Every synthesized process exchanger is an
  ``HXprocess`` solved rigorously against its ``dT``, ``H_lim0`` and
  ``T_lim1`` limits, and every synthesized utility exchanger is an
  ``HXutility`` with ``rigorous=True``, specified by enthalpy rather than by
  temperature.

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

**Convergence.** The new exchangers are assembled into a network with
``Network.from_units(..., interaction=False)``, whose path follows the rewired
stream connections rather than the order in which the exchangers were
synthesized. That ordering matters because a hot-side exchanger is created
before the cold-side ones that feed it, and a single pass in synthesis order
would run it on stale inlets; if the path cannot be fully ordered, hensmith
warns and leaves convergence to settle the loop. The resulting ``HXN_sys`` has
its tolerance set with ``method='fixedpoint'`` on itself and its subsystems
and is converged with ``System.converge()``; should convergence raise, every
unit is run once and a ``RuntimeWarning`` is issued.

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
original unit's heat utility with the new one and leaves the facility itself
carrying none. If no process match was made at all, the facility reports zero
capital and no utilities.

**Reusing a network.** With ``cache_network=True`` a later simulation checks
whether the set of units behind the heat utilities is unchanged; if it is, the
same network configuration is reused with updated inlet states and enthalpy
limits instead of being synthesized again. If the reused network then fails
its outlet checks, hensmith warns, discards the cache and re-synthesizes from
scratch. If it fails its energy balance instead, the cache is discarded and
the network re-synthesized silently; a warning is issued only if the freshly
synthesized network fails the same check.

Validation
----------

Nothing about a heuristic network is self-evidently right, so hensmith checks
it in four ways.

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

**The MER bound.** The targets of the problem table are a lower bound the
network cannot beat. ``tests/test_hxn_regression.py`` builds ten synthetic
systems of increasing complexity -- all with phase-changing streams from the
third on -- and for each one requires that the synthesized network (i) closes
its energy balance without raising ``RuntimeWarning``, (ii) never uses less
hot or cold utility than the MER targets computed on the same streams, and
(iii) recovers at least as much heat as a load recorded in the test file. A
network that improves leaves slack in (iii); those recorded numbers are
lowered deliberately by a maintainer, never raised to make a failing test
pass.

**Doctests.** The examples in the docstrings are executed as part of the test
suite, so the numbers printed in the API reference are numbers the code
currently produces.

Limitations
-----------

hensmith implements the classical pinch design method with a specific set of
heuristics. Four consequences are worth stating plainly.

- **The network is heuristic, not optimal.** Matches are committed in sequence
  and never revisited; there is no search over the space of networks and no
  optimization of area, capital or number of units. The result depends on the
  order of the streams and on the driving-force ranking, and it is not
  guaranteed to reach the MER targets -- only never to beat them.
- **Streams are not split.** Every stream stays a single branch through the
  network. Where the pinch design method would split a stream to satisfy the
  heat-capacity flow rate inequality, hensmith simply does not make that match
  in the design pass; the duty is picked up later by an offset pass or by a
  utility exchanger.
- **Only streams behind existing utility exchangers are integrated.** The
  facility sees a process stream only through a heat utility attached to a
  unit of the system. A duty carried some other way is invisible to it;
  utilities flagged as not usable for HXN integration (``hxn_ok=False``) or
  with non-positive flow, and streams excluded with ``ignored``, are removed
  from the analysis entirely.
- **One approach temperature for everything.** A single ``T_min_app`` shifts
  the problem table and constrains every synthesized exchanger. There is no
  per-stream or per-match approach temperature, so a match whose exchange is
  cheap and one whose exchange is expensive are held to the same driving-force
  floor.

References
----------

- Linnhoff, B., & Hindmarsh, E. (1983). The pinch design method for heat
  exchanger networks. *Chemical Engineering Science*, 38(5), 745-763.
- Seider, W. D., Lewin, D. R., Seader, J. D., Widagdo, S., Gani, R., & Ng,
  M. K. (2017). *Product and Process Design Principles*. Wiley. Heat Exchanger
  Networks (Chapter 9).
- Kemp, I. C. (2007). *Pinch Analysis and Process Integration: A User Guide on
  Process Integration for the Efficient Use of Energy* (2nd ed.).
  Butterworth-Heinemann.
- Cortes-Pena, Y., Kumar, D., Singh, V., & Guest, J. S. (2020). BioSTEAM: A
  fast and flexible platform for the design, simulation, and techno-economic
  analysis of biorefineries under uncertainty. *ACS Sustainable Chemistry &
  Engineering*, 8(8), 3302-3310. https://doi.org/10.1021/acssuschemeng.9b07040

.. seealso::

   :doc:`tutorial/02_pinch_analysis` applies the problem table to a real
   system; :doc:`tutorial/03_network_anatomy` walks through a synthesized
   network exchanger by exchanger; :doc:`API/api` documents every public name.
