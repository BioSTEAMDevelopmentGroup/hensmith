HeatExchangerNetwork
====================

.. currentmodule:: hensmith

:class:`HeatExchangerNetwork` is a BioSTEAM facility that runs a pinch
analysis over the heating and cooling utilities of a whole system,
synthesizes a network of process heat exchangers that meets part of those
duties by stream-to-stream exchange -- at the minimum energy requirement
(MER) targets whenever it finds such a network without stream splits, or,
with ``stream_splitting=True``, with stream splits where MER needs them -- and
reports the utility loads and capital cost that result. The original units,
streams and heat exchangers are left untouched: the stream copies and
synthesized exchangers live in a separate flowsheet named ``<sys>_HXN``. See
:doc:`../tutorial/index` for a worked example and :doc:`../concepts` for the
method.

.. autoclass:: HeatExchangerNetwork
   :no-members:
   :show-inheritance:

.. automethod:: HeatExchangerNetwork.plot_pinch_diagram

Constructor options
-------------------

In addition to ``ID``, ``T_min_app`` and ``units`` documented above, the
constructor accepts the following options; :doc:`../tutorial/04_configuring`
shows what each of them changes.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Type
     - Meaning
   * - ``ignored``
     - Iterable[Unit] or callable
     - Units whose heat utilities are excluded from the analysis; a callable is evaluated at simulation time. Defaults to None.
   * - ``Qmin``
     - float, kJ/hr
     - Planned exchangers with a duty below this are dropped and their duty left to the utilities (a large value can cost MER), and utility exchangers at or below it are not marked on the pinch diagram. Defaults to 1e-3.
   * - ``force_ideal_thermo``
     - bool
     - Run the analysis on stream copies with ideal thermodynamics; the synthesized exchangers inherit that thermo. Defaults to False.
   * - ``cache_network``
     - bool
     - Reuse the network configuration of the previous simulation when the set of units contributing heat utilities and ``stream_splitting`` are unchanged, updating only stream states and exchanger specifications: each process exchanger keeps the fraction of its stream's duty at which its enthalpy limit sat at synthesis (the splitters keep their branch fractions), and the utility exchangers bring every stream to its new outlet. The reused network is not planned again, so it need not be at MER for the new duties. Defaults to False.
   * - ``avoid_recycle``
     - bool
     - Never match the same hot/cold stream pair twice anywhere (on one side of the pinch or across the two), so that no two exchangers connect the same pair and form a recycle loop; this forbids the repeated matches some unsplit MER networks need. Defaults to False.
   * - ``acceptable_energy_balance_error``
     - float
     - When given, sets an instance attribute that overrides the class default of 0.02 (see below). Defaults to None, i.e. the class value is used.
   * - ``replace_unit_heat_utilities``
     - bool
     - Copy each synthesized utility exchanger's heat utility onto the corresponding original heat utility and reload that unit's utility cost, instead of reporting the net utilities on the facility itself. Applies only when at least one process exchanger was synthesized. Defaults to False.
   * - ``sort_hus_by_T``
     - bool
     - Sort the heating utilities by inlet temperature descending and the cooling utilities ascending before the analysis, so that inlet temperature rather than signed duty (the default: smallest heating duty first, largest cooling duty first) sets the stream indices, which break ties in the planner's search. Defaults to False.
   * - ``stream_splitting``
     - bool
     - Allow a process stream to be split into parallel branches that re-join. A side of the pinch that no unsplit network serves at MER is planned with splits and reaches the targets exactly on the planner's knots; sides that an unsplit network serves are never split, so a problem that needs no split gets the same network as with the default. Each split is a chain of ``Splitter`` units and a rigorous ``Mixer`` (adiabatic, no cost), listed in ``new_splitters`` and ``new_mixers``. With ``avoid_recycle``, a split that would repeat a stream pair is not used, and MER is then not guaranteed. Stored as an attribute of the same name, and part of the ``cache_network`` key. Defaults to False.

Class attributes
----------------

Defaults shared by every instance; assigning to an instance overrides the
value for that instance only.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Meaning
   * - ``ticket_name = 'HXN'``
     - str
     - Default ID of an unnamed instance (``'HXN'``; facilities are not auto-numbered).
   * - ``acceptable_energy_balance_error = 0.02``
     - float (fraction)
     - Fraction (0.02 = 2 %) absolute energy balance error above which the simulation warns (or raises); a cached network exceeding it is discarded and resynthesized.
   * - ``raise_energy_balance_error = False``
     - bool
     - Raise a ``RuntimeError`` instead of warning when the energy balance error exceeds the tolerance above.
   * - ``network_priority = -2``
     - int
     - Facility ordering key; facilities are simulated in ascending order of this value, so the network runs before the utility facilities.

Attributes set by simulation
----------------------------

The following attributes are set at the end of every simulation. They carry
no docstrings of their own, so autodoc cannot list them; each per-stream list
or array is indexed by the stream index of the rearranged utility list (see
:func:`synthesize_network`). With ``cache_network=True`` the attributes that
describe the synthesized topology are kept from the synthesis that produced
the cached network.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Meaning
   * - ``original_heat_util_load``
     - float, kJ/hr
     - Total duty of the original heating utilities, before integration.
   * - ``actual_heat_util_load``
     - float, kJ/hr
     - Total heating duty of the synthesized utility exchangers, after integration.
   * - ``original_cool_util_load``
     - float, kJ/hr
     - Total magnitude of the duty of the original cooling utilities, before integration.
   * - ``actual_cool_util_load``
     - float, kJ/hr
     - Total magnitude of the cooling duty of the synthesized utility exchangers, after integration.
   * - ``energy_balance_percent_error``
     - float, %
     - Percent deviation from one of the ratio (twice the duty of each process exchanger, plus the new utility duties weighted by their agents' heat-transfer efficiency) / (the original utility duties weighted the same way), as computed in ``_cost``.
   * - ``synthesis_info``
     - dict
     - The synthesis report (see the ``info`` keyword of :func:`synthesize_network`): ``'status'`` is ``'mer'`` when the network's utilities equal the MER targets and ``'best_effort'`` otherwise; next to it the targets, the planned and realized utilities, the penalty, per side of the pinch any proof that a split is needed, and the smallest approach inside any process exchanger. With ``stream_splitting``, also ``'stream_splitting'``, ``'splits'`` (the realized splits, :class:`~hensmith.hxn_synthesis.StreamSplit`), ``'split_deviations'`` and, per side, the split candidate chosen (``'split'``). Kept from the synthesis that produced a cached network.
   * - ``stream_life_cycles``
     - list[StreamLifeCycle]
     - Ordered sequence of exchangers each stream passes through, aligned with ``original_heat_exchangers``; a split stream's life cycle also lists its splits and marks each branch stage with its branch and flow fraction.
   * - ``new_HXs``
     - list[HXprocess]
     - All synthesized process exchangers, the hot-side ones followed by the cold-side ones.
   * - ``new_HXs_hot_side``
     - list[HXprocess]
     - Process exchangers of the hot-side (above-pinch) design, in plan order (from the pinch outward), IDs ``HX_<cold>_<hot>_hs``.
   * - ``new_HXs_cold_side``
     - list[HXprocess]
     - Process exchangers of the cold-side (below-pinch) design, in plan order, IDs ``HX_<hot>_<cold>_cs``; on either side the *n*-th exchanger of a repeated pair gets the suffix ``_<n>``.
   * - ``new_HX_utils``
     - list[HXutility]
     - One rigorous utility exchanger per stream, bringing it from its last process exchanger (or its inlet, if it was not matched) to its outlet enthalpy.
   * - ``new_splitters``
     - list[Splitter]
     - The splitters of the network's stream splits, every split's chain in order (IDs ``Split_<stream>_<hs|cs>``, with ``_<n>`` for a stream's *n*-th split on that side and ``_b<c>`` for the chain's element *c* >= 2); empty without a split, and always without ``stream_splitting``.
   * - ``new_mixers``
     - list[Mixer]
     - One rigorous, adiabatic mixer per stream split, where its branches re-join (IDs ``Mix_<stream>_<hs|cs>``, with ``_<n>`` as for the splitters); empty without a split.
   * - ``original_heat_exchangers``
     - list[Unit]
     - The original heat exchangers behind the analyzed heat utilities, in stream order.
   * - ``original_heat_utils``
     - list[HeatUtility]
     - The original heat utilities rearranged into stream order, so that they align with ``stream_life_cycles``.
   * - ``HXN_sys``
     - System
     - The system built from the synthesized exchangers (and the splitters and mixers of any stream split), named ``<sys>_HXN`` and registered in ``HXN_flowsheet``; converged and summarized during costing.
   * - ``HXN_flowsheet``
     - Flowsheet
     - The flowsheet ``<sys>_HXN`` holding the network's stream copies and exchangers.
   * - ``pinch_Ts``
     - ndarray, K
     - Per-stream pinch temperature (informational): the process pinch on the stream's own scale when the stream crosses it, else its inlet temperature (inlet already past the pinch, or an isothermal or non-monotone stream) or its outlet temperature (stream ending before the pinch).
   * - ``inlet_Ts``
     - ndarray, K
     - Inlet temperature of each stream.
   * - ``outlet_Ts``
     - ndarray, K
     - Outlet temperature of each stream, after quenching the outlet to equilibrium at its own enthalpy.
   * - ``streams_inlet``
     - list[Stream]
     - One copy of each stream's inlet, in stream order, as prepared for the analysis; the synthesis works on further copies, so these keep their inlet state.
   * - ``stream_HXs_dict``
     - dict[int, list[Unit]]
     - Exchangers that each stream index passes through: its process exchangers in flow order, then its utility exchanger. Where the stream splits, the order is topological: the exchangers before the split, those of its branches (branch by branch, each in flow order), then those after it.
   * - ``cold_indices``
     - list[int]
     - Stream indices of the heated (cold) streams.
   * - ``original_purchase_costs``
     - list[float], USD
     - Purchase cost of each original heat exchanger.
   * - ``new_purchase_costs_HXp``
     - list[float], USD
     - Purchase cost of each synthesized process exchanger, indexed like ``new_HXs``.
   * - ``new_purchase_costs_HXu``
     - list[float], USD
     - Purchase cost of each synthesized utility exchanger, indexed like ``new_HX_utils``.
   * - ``original_utility_costs``
     - list[HeatUtility]
     - The original heat utilities summed by agent, with their duties reversed in sign so that they net against the new ones.
   * - ``new_utility_costs``
     - list[HeatUtility]
     - The heat utilities of the synthesized utility exchangers, summed by agent.
