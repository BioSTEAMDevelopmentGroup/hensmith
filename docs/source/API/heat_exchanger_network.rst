HeatExchangerNetwork
====================

.. currentmodule:: hensmith

:class:`HeatExchangerNetwork` is a BioSTEAM facility that runs a pinch
analysis over the heating and cooling utilities of a whole system,
synthesizes a network of process heat exchangers that meets part of those
duties by stream-to-stream exchange, and reports the utility loads and
capital cost that result. The original units, streams and heat exchangers
are left untouched: the stream copies and synthesized exchangers live in a
separate flowsheet named ``<sys>_HXN``. See :doc:`../tutorial/index` for a
worked example.

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
     - Candidate exchangers with a duty below this are discarded during synthesis, and utility exchangers at or below it are not marked on the pinch diagram. Defaults to 1e-3.
   * - ``force_ideal_thermo``
     - bool
     - Run the analysis on stream copies with ideal thermodynamics; the synthesized exchangers inherit that thermo. Defaults to False.
   * - ``cache_network``
     - bool
     - Reuse the network configuration of the previous simulation when the set of units contributing heat utilities is unchanged, updating only stream states and exchanger specifications. Defaults to False.
   * - ``avoid_recycle``
     - bool
     - Never match the same hot/cold stream pair twice, so that no two exchangers connect the same pair and form a recycle loop. Defaults to False.
   * - ``acceptable_energy_balance_error``
     - float
     - When given, sets an instance attribute that overrides the class default of 0.02 (see below). Defaults to None, i.e. the class value is used.
   * - ``replace_unit_heat_utilities``
     - bool
     - Copy each synthesized utility exchanger's heat utility onto the corresponding original heat utility and reload that unit's utility cost, instead of reporting the net utilities on the facility itself. Applies only when at least one process exchanger was synthesized. Defaults to False.
   * - ``sort_hus_by_T``
     - bool
     - Sort the heating utilities by inlet temperature descending and the cooling utilities ascending before the analysis, so that inlet temperature rather than signed duty (the default: smallest heating duty first, largest cooling duty first) sets the matching priority. Defaults to False.

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
   * - ``stream_life_cycles``
     - list[StreamLifeCycle]
     - Ordered sequence of exchangers each stream passes through, aligned with ``original_heat_exchangers``.
   * - ``new_HXs``
     - list[HXprocess]
     - All synthesized process exchangers, the hot-side ones followed by the cold-side ones.
   * - ``new_HXs_hot_side``
     - list[HXprocess]
     - Process exchangers of the hot-side (above-pinch) design.
   * - ``new_HXs_cold_side``
     - list[HXprocess]
     - Process exchangers of the cold-side (below-pinch) design.
   * - ``new_HX_utils``
     - list[HXutility]
     - One rigorous utility exchanger per stream, bringing it from its last process exchanger (or its inlet, if it was not matched) to its outlet enthalpy.
   * - ``original_heat_exchangers``
     - list[Unit]
     - The original heat exchangers behind the analyzed heat utilities, in stream order.
   * - ``original_heat_utils``
     - list[HeatUtility]
     - The original heat utilities rearranged into stream order, so that they align with ``stream_life_cycles``.
   * - ``HXN_sys``
     - System
     - The system built from the synthesized exchangers, named ``<sys>_HXN`` and registered in ``HXN_flowsheet``; converged and summarized during costing.
   * - ``HXN_flowsheet``
     - Flowsheet
     - The flowsheet ``<sys>_HXN`` holding the network's stream copies and exchangers.
   * - ``pinch_Ts``
     - ndarray, K
     - Per-stream pinch temperature at which the stream's duty is split between the hot-side and cold-side designs.
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
     - Exchangers, the process ones then the utility one, that each stream index passes through, in synthesis order rather than flow order.
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
