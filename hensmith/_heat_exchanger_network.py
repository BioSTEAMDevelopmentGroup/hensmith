# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangb2@illinois.edu>, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Created on Sat Aug 22 21:58:19 2020
@author: sarangbhagwat and yoelcp
"""
import heapq
import biosteam as bst
import numpy as np
from .hxn_synthesis import (
    synthesize_network, StreamLifeCycle, plot_pinch_diagram, _first_inlet,
)
from warnings import warn

__all__ = ('HeatExchangerNetwork',)

#: A stream whose utility exchanger would transfer at most this fraction of
#: the stream's duty is already at its outlet: the plan leaves it no utility
#: duty (the planner resolves duties to 1e-9 of the total), and what is left
#: is the residual of the enthalpy flashes that realize the plan (at most
#: 1e-10 of the duty over the test suite, where the smallest utility duty a
#: network does leave is 3e-7 of the stream's duty).
_SERVED_RTOL = 1e-9

def _pass_through_served_streams(stream_life_cycles, original_units):
    """
    Let every stream that the process exchangers bring to its outlet (to
    within `_SERVED_RTOL` of its duty) leave its utility exchanger in the
    state it enters it.

    A rigorous `HXutility` re-flashes its feed at the outlet enthalpy even
    when the feed already has it; the flash converges from its own previous
    state, so the phase split can move in the last digits and, with the
    enthalpies of formation, leave a spurious net duty (~1e-9 kJ/hr) that
    biosteam designs and costs as a minimum-size exchanger, depending on
    flash noise (e.g. present in a cached network but not in the fresh one).
    Passing the feed through gives the exchanger exactly no duty, which
    biosteam neither designs nor costs; an exchanger with any real duty is
    left as it is (and costed as always). Utility outlets feed nothing else
    in the network, so the converged network needs no further pass.
    """
    for life_cycle, unit in zip(stream_life_cycles, original_units):
        util = life_cycle.life_cycle[-1].unit
        if not isinstance(util, bst.HXutility): continue
        feed, product = util.ins[0], util.outs[0]
        duty = abs(unit.outs[0].H - unit.ins[0].H)
        if abs(product.H - feed.H) <= _SERVED_RTOL * duty:
            product.copy_like(feed)

def _load_utility_costs(unit):
    """Recompute the utility cost of `unit` and of its owner (the unit whose
    heat utilities include those of `unit`, e.g. a column for its
    condenser) from their heat utilities."""
    unit._load_operation_costs()
    owner = unit.owner
    if owner is not unit: owner._load_operation_costs()

def _move_flows(unit):
    """
    Give the outlets of `unit`, a unit of a cached network, the flows that
    its inlets carry now, so that the network's convergence starts from
    the new flows (`HeatExchangerNetwork._cost` moves them in path order).
    No enthalpy flash runs here: only the convergence, which the facility
    guards, runs the rigorous units.

    - A splitter splits its feed at its fixed fractions (`Splitter._run`
      copies the feed's intensive state and flashes nothing).
    - A mixer's outlet takes the summed flows of its inlets (running the
      rigorous mixer would flash).
    - Any other unit's outlet ``k`` takes the flows of its inlet ``k``.

    A multiphase outlet keeps its temperature and pressure: its flows are
    scaled to the new total and, if that does not reproduce the inlet
    flows, replaced by them and split between the phases at its
    temperature and pressure.
    """
    if isinstance(unit, bst.Splitter):
        unit._run()
    elif isinstance(unit, bst.Mixer):
        s_out, = unit.outs
        ins = unit.ins
        mol = sum([s.mol for s in ins])
        if isinstance(s_out, bst.MultiStream):
            s_out.F_mol = sum([s.F_mol for s in ins])
            if not s_out.mol.sparse_equal(mol):
                s_out.imol.mix_from([s.imol for s in ins])
                s_out.vle(T=s_out.T, P=s_out.P)
        else:
            s_out.mol[:] = mol
    else:
        for s_in, s_out in zip(unit.ins, unit.outs):
            if isinstance(s_out, bst.MultiStream):
                s_out.F_mol = s_in.F_mol
                if not s_out.mol.sparse_equal(s_in.mol):
                    s_out.copy_flow(s_in)
                    s_out.vle(T=s_out.T, P=s_out.P)
            else:
                s_out.mol[:] = s_in.mol

def _network_path(units, stream_life_cycles):
    """
    Return the simulation path of a synthesized network and its recycle
    (tear) streams.

    A stream's stages form a series-parallel chain (in series, except that
    the branches of a split run in parallel between its splitter chain and
    its mixer), so the network is a directed graph with an edge along every
    connection of every life cycle (`StreamLifeCycle.connections`; for an
    unsplit stream, from each stage to the next one). The path is a
    topological order of that graph (Kahn's algorithm, ties broken by the
    order of `units`); where the graph has a cycle (e.g. a pair of streams
    matched both above and below the pinch, or repeated matches in
    alternating order), the unit with the fewest unplaced predecessors
    comes next and its inlets from later units become recycle streams
    (possibly a splitter outlet). Every unit then runs after all its
    feeders except across a declared recycle, and the fixed-point iteration
    of the resulting `System` converges the loops. (Deterministic, unlike a
    general network sort, which need not settle on intertwined loops.)
    """
    position = {u: i for i, u in enumerate(units)}
    successors = {u: [] for u in units}
    N_waiting = {u: 0 for u in units}
    for life_cycle in stream_life_cycles:
        for up, up_port, down, _ in life_cycle.connections():
            successors[up].append((down, up.outs[up_port]))
            N_waiting[down] += 1
    ready = [position[u] for u in units if not N_waiting[u]]
    heapq.heapify(ready)
    placed = {}
    while len(placed) < len(units):
        if ready:
            unit = units[heapq.heappop(ready)]
            if unit in placed: continue
        else: # a cycle: break it where the fewest feeders are missing
            unit = min((u for u in units if u not in placed),
                       key=lambda u: (N_waiting[u], position[u]))
        placed[unit] = len(placed)
        for other, _ in successors[unit]:
            N_waiting[other] -= 1
            if not N_waiting[other] and other not in placed:
                heapq.heappush(ready, position[other])
    path = sorted(units, key=placed.__getitem__)
    recycles = [s for unit in units for other, s in successors[unit]
                if placed[other] <= placed[unit]]
    return path, recycles


class HeatExchangerNetwork(bst.Facility):
    """
    Create a HeatExchangerNetwork object that will perform a pinch analysis
    on the entire system's heating and cooling utility objects. The heat
    exchanger network reduces the heating and cooling utility requirements 
    of the system and may add additional capital cost.
    
    Parameters
    ----------
    ID : str
        Unique name for the facility.
    T_min_app : float
        Minimum approach temperature observed during synthesis of heat exchanger network.
    units : Iterable[Unit], optional
        All unit operations available to the heat exchanger network. Defaults
        to all unit operations in the system.
    stream_splitting : bool, optional
        Allow a process stream to be split into parallel branches that
        re-join, where the minimum energy requirement needs it (see
        `synthesize_network`). The branches are realized with
        `biosteam.Splitter` chains (`new_splitters`) and rigorous
        `biosteam.Mixer` units (`new_mixers`), which are adiabatic and
        cost nothing. Defaults to False: the network has no split, and
        both lists are empty.

    Notes
    -----
    Unless `stream_splitting`, the network is synthesized without stream
    splits by :func:`~hensmith.hxn_synthesis.synthesize_network`: a problem
    table on the streams' temperature-enthalpy curves gives the minimum
    energy requirement (MER) targets, and a planner builds each side of the
    pinch from the pinch outward [1]_ [2]_, keeping `T_min_app` everywhere
    inside every exchanger on the exact stream states. It reaches the
    targets whenever its search finds an unsplit network that does; the same
    pair of streams may then be matched more than once (IDs with a suffix
    ``_<n>``, e.g. ``HX_3_2_cs_2``), since series alternation can replace a
    split. Where the pinch design rules prove that MER needs stream
    splitting, the network is a best-effort one close to the targets, or,
    with `stream_splitting`, one that splits streams to reach them. The
    outcome is recorded in `synthesis_info` (a dict; see the `info` keyword
    of `synthesize_network`): 'status' is 'mer' when the network's utilities
    equal the targets and 'best_effort' otherwise, with the targets, the
    planned utilities and, per side of the pinch, any proof that a split is
    needed.

    Original system stream and heat exchanger objects are preserved. All
    stream copies and new HX objects can be found in a newly created
    flowsheet '<sys>_HXN' where <sys> is the name of the system associated
    to the HeatExchangerNetwork object. Each stream passes its exchangers in
    series (where it splits, its branches run in parallel from a splitter
    chain to a mixer); the network is simulated as a `System` (`HXN_sys`)
    whose path follows the streams, with the loops that repeated matches can
    form torn and converged to a tight tolerance (every exchanger starts at
    its planned state, so the loops are at their fixed point after one
    pass).

    With `cache_network`, a network is reused while the set of heat
    exchangers and `stream_splitting` are the same: each process exchanger
    keeps, as its enthalpy limit, the share of the stream's duty it had at
    synthesis (on the stream that the plan serves completely on that side of
    the pinch; its partner transfers that share, but never past its own
    outlet, and takes the rest to its utility), and the utility exchangers
    bring every stream to its new outlet. The splitters keep their
    fractions, so a branch exchanger's limit is its branch's fraction of the
    whole stream's. If the cached network does not reproduce the outlets, or
    its energy balance is off, the network is synthesized again.

    Every utility exchanger is designed and costed by biosteam as usual. A
    stream that its process exchangers bring to its outlet (within 1e-9 of
    its duty: the residual of the enthalpy flashes) leaves its utility
    exchanger in the state it enters it, so that exchanger has exactly no
    duty and no cost instead of a spurious duty from re-flashing the
    stream.

    The facility's heat utilities are the new utilities less the original
    ones, summed by agent (a negative utility cost is a saving). With
    `replace_unit_heat_utilities`, each original heat utility takes the heat
    utility of its own stream's utility exchanger instead, the utility costs
    of its unit and of that unit's owner are reloaded, and the facility
    carries no heat utilities. The original data are given back before the
    network is costed again, so that the network is synthesized from the
    units' own utilities whether or not the units were simulated again.

    References
    ----------
    .. [1] Linnhoff, B., & Hindmarsh, E. (1983). The pinch design method for
        heat exchanger networks. Chemical Engineering Science, 38(5),
        745-763.
    .. [2] Seider, W. D., Lewin,  D. R., Seader, J. D., Widagdo, S., Gani, R.,
        & Ng, M. K. (2017). Product and Process Design Principles. Wiley.
        Heat Exchanger Networks (Chapter 9)
    
    Examples
    --------
    >>> import biosteam as bst
    >>> bst.settings.set_thermo(['Water', 'Methanol', 'Glycerol'])
    >>> feed1 = bst.Stream('feed1', flow=(8000, 100, 25))
    >>> feed2 = bst.Stream('feed2', flow=(10000, 1000, 10))
    >>> D1 = bst.ShortcutColumn('D1', ins=feed1,
    ...                     outs=('distillate', 'bottoms_product'),
    ...                     LHK=('Methanol', 'Water'),
    ...                     y_top=0.99, x_bot=0.01, k=2,
    ...                     is_divided=True)
    >>> D1_H1 = bst.HXutility('D1_H1', ins = D1.outs[1], T = 300)
    >>> D1_H2 = bst.HXutility('D1_H2', ins = D1.outs[0], T = 300)
    >>> F1 = bst.Flash('F1', ins=feed2,
    ...                outs=('vapor', 'liquid'), V = 0.9, P = 101325)
    >>> HXN = bst.HeatExchangerNetwork('HXN', T_min_app = 5.)
    >>> sys = bst.System.from_units('sys', units=[D1, D1_H1, D1_H2, F1, HXN])
    >>> sys.simulate()
    >>> # See all results
    >>> round(HXN.actual_heat_util_load/HXN.original_heat_util_load, 2)
    0.82
    >>> abs(HXN.energy_balance_percent_error) < 0.01
    True
    >>> HXN.synthesis_info['status']  # the utilities equal the MER targets
    'mer'
    >>> HXN.stream_life_cycles
    [<StreamLifeCycle: Stream_0, cold
    	life_cycle = [
    		<LifeStage: <HXprocess: HX_0_2_hs>, H_in = 5.38e+06 kJ/hr, H_out = 4.24e+07 kJ/hr>
    		<LifeStage: <HXutility: Util_0_hs>, H_in = 4.24e+07 kJ/hr, H_out = 6.92e+07 kJ/hr>
    	]>, <StreamLifeCycle: Stream_1, cold
    	life_cycle = [
    		<LifeStage: <HXprocess: HX_1_2_hs>, H_in = 0 kJ/hr, H_out = 5.05e+06 kJ/hr>
    		<LifeStage: <HXprocess: HX_1_4_hs>, H_in = 5.05e+06 kJ/hr, H_out = 5.08e+06 kJ/hr>
    		<LifeStage: <HXprocess: HX_1_3_hs>, H_in = 5.08e+06 kJ/hr, H_out = 2.3e+07 kJ/hr>
    		<LifeStage: <HXutility: Util_1_hs>, H_in = 2.3e+07 kJ/hr, H_out = 2.79e+08 kJ/hr>
    	]>, <StreamLifeCycle: Stream_2, hot
    	life_cycle = [
    		<LifeStage: <HXprocess: HX_0_2_hs>, H_in = 4.52e+07 kJ/hr, H_out = 8.12e+06 kJ/hr>
    		<LifeStage: <HXprocess: HX_1_2_hs>, H_in = 8.12e+06 kJ/hr, H_out = 3.07e+06 kJ/hr>
    		<LifeStage: <HXutility: Util_2_cs>, H_in = 3.07e+06 kJ/hr, H_out = 1.14e+06 kJ/hr>
    	]>, <StreamLifeCycle: Stream_3, hot
    	life_cycle = [
    		<LifeStage: <HXprocess: HX_1_3_hs>, H_in = 2.04e+07 kJ/hr, H_out = 2.47e+06 kJ/hr>
    		<LifeStage: <HXutility: Util_3_cs>, H_in = 2.47e+06 kJ/hr, H_out = 2.47e+06 kJ/hr>
    	]>, <StreamLifeCycle: Stream_4, hot
    	life_cycle = [
    		<LifeStage: <HXprocess: HX_1_4_hs>, H_in = 7.51e+05 kJ/hr, H_out = 7.18e+05 kJ/hr>
    		<LifeStage: <HXutility: Util_4_cs>, H_in = 7.18e+05 kJ/hr, H_out = 7.18e+05 kJ/hr>
    	]>]
    
    """
    ticket_name = 'HXN'
    acceptable_energy_balance_error = 0.02
    raise_energy_balance_error = False
    network_priority = -2
    _N_ins = 0
    _N_outs = 0
    _units= {'Flow rate': 'kg/hr',
             'Work': 'kW'}
    
    def __init__(self, ID='', T_min_app=5., units=None, ignored=None, Qmin=1e-3,
                 force_ideal_thermo=False, cache_network=False, avoid_recycle=False,
                 acceptable_energy_balance_error=None, replace_unit_heat_utilities=False,
                 sort_hus_by_T=False, stream_splitting=False):
        bst.Facility.__init__(self, ID, None, None)
        self.T_min_app = T_min_app
        self.units = units
        self.ignored = ignored
        self.Qmin = Qmin
        self.force_ideal_thermo = force_ideal_thermo
        self.cache_network = cache_network
        self.avoid_recycle = avoid_recycle
        self.replace_unit_heat_utilities = replace_unit_heat_utilities
        self.sort_hus_by_T = sort_hus_by_T
        self.stream_splitting = stream_splitting
        if acceptable_energy_balance_error is not None:
            self.acceptable_energy_balance_error = acceptable_energy_balance_error
        
    def _get_original_heat_utilties(self):
        sys = self.system
        if self.units:
            units = self.units
            if callable(units): units = units()
        else:
            units = sys.units
        ignored = self.ignored
        if ignored:
            if callable(ignored): ignored = ignored()
            ignored_hx_utils = sum([i.heat_utilities for i in ignored], [])
        else:
            ignored_hx_utils = ()
        hx_utils = bst.process_tools.heat_exchanger_utilities_from_units(units)
        return [i for i in hx_utils if i.duty and i not in ignored_hx_utils]
        
    def _run(self): pass

    def _enter_network(self, index, stage, stream):
        """Bring `stream` (the network copy of stream `index`'s real inlet,
        which enters the network at `stage.unit`: its life cycle's entry
        port, or its first stage) to the state in which its first process
        exchanger takes it, directly or, where the stream splits at its
        inlet, through the splitter chain (which keeps the intensive state):
        at equilibrium at its inlet enthalpy for a point-load stream (see
        `hensmith.hxn_synthesis._first_inlet`)."""
        if not isinstance(stage.unit, (bst.HXprocess, bst.Splitter)): return
        point_load = index in self.synthesis_info.get('point_loads', ())
        _first_inlet(stream, point_load, self.outlet_Ts[index],
                     index not in self.cold_indices)

    def _replace_unit_heat_utilities(self, heat_utilities, stream_life_cycles):
        """
        Overwrite each original heat utility with the heat utility of its own
        stream's utility exchanger, the last stage of the stream's life
        cycle (`heat_utilities` and `stream_life_cycles` are both in stream
        order; `new_HX_utils` is not: it lists the hot streams first), and
        reload the utility costs of its unit and of the unit's owner. The
        original data are kept for `_restore_unit_heat_utilities`.
        """
        replaced = []
        for hu, life_cycle in zip(heat_utilities, stream_life_cycles):
            new = life_cycle.life_cycle[-1].unit.heat_utilities[0]
            replaced.append((hu, hu.copy()))
            if new.agent: hu.copy_like(new)
            else: hu.empty() # a stream the process exchangers serve
            _load_utility_costs(hu.unit)
        self._replaced_heat_utilities = replaced

    def _restore_unit_heat_utilities(self):
        """
        Copy their original data back into the heat utilities that
        `_replace_unit_heat_utilities` last overwrote, and reload their
        units' utility costs, wherever the unit still holds that heat
        utility. A unit that
        is simulated again replaces its heat utilities with new ones (as
        every unit is before the facility in a system simulation), but one
        that is not (e.g. when the network alone is simulated again, or
        once per ignored utility in `_energy_balance_error_contributions`)
        would otherwise hand the network its own utilities as the process
        duties, and a stream that it serves completely, with no utility
        left, would drop out of the next network.
        """
        replaced = getattr(self, '_replaced_heat_utilities', None)
        if not replaced: return
        self._replaced_heat_utilities = None
        for hu, original in replaced:
            unit = hu.unit
            if any(i is hu for i in unit.heat_utilities):
                hu.copy_like(original)
                _load_utility_costs(unit)

    def _design(self): pass
    def _load_capital_costs(self): pass # Do not replace installed costs

    def _cost(self):
        sys = self.system
        flowsheet = bst.Flowsheet(sys.ID + '_HXN')
        with flowsheet.temporary():
            self._restore_unit_heat_utilities()
        hx_utils = self._get_original_heat_utilties()
        use_cached_network = False
        # A network synthesized with other options (stream splitting on or
        # off), or by a facility from before stream splitting (no options
        # recorded; its life cycles have no entry port), is never reused.
        if (self.cache_network and hasattr(self, 'original_heat_utils')
                and hasattr(self, '_stage_fractions')
                and getattr(self, '_synthesis_options', None)
                    == (self.stream_splitting,)):
            # Units are a stable key to compare whether system has changed configuration.
            hu_by_unit = {hu.unit: hu for hu in hx_utils}
            use_cached_network = (
                hu_by_unit.keys() == set(self.original_heat_exchangers)
            )
        with flowsheet.temporary(), bst.IgnoreDockingWarnings():
            if use_cached_network:
                # Keep the same configuration, but update stream life cycles and heat exchanger specifications.
                # Note that stream_life_cycles are aligned with original_heat_exchangers (from synthesize_network).
                # Heat utils are rearranged so that they align with stream_life_cycles too.
                hxs = self.original_heat_exchangers
                hx_heat_utils_rearranged = [hu_by_unit[hx] for hx in hxs]
                stream_life_cycles = self.stream_life_cycles
                new_HXs = self.new_HXs
                new_HX_utils = self.new_HX_utils
                stage_fractions = self._stage_fractions
                stage_scales = self._stage_scales
                for i, life_cycle in enumerate(stream_life_cycles):
                    hx = hxs[i]
                    s_util_in = hx.ins[0]
                    # at its entry port, as in a fresh synthesis
                    entry = life_cycle.entry
                    s_lc = entry.unit.ins[entry.index]
                    s_lc.copy_like(s_util_in)
                    self._enter_network(i, entry, s_lc)
                    H_in = s_util_in.H
                    H_out = hx.outs[0].H
                    for lc in life_cycle.life_cycle:
                        unit = lc.unit
                        if isinstance(unit, bst.HXutility):
                            unit.H = H_out
                            continue
                        # Each exchanger keeps the share of its "must"
                        # stream's duty (port 1: the hot stream above the
                        # pinch, the cold stream below it), which the plan
                        # serves completely; the partner (port 0), which
                        # leaves its remainder to its utility, takes what
                        # that transfers up to its own outlet (its planned
                        # share would cut the must short when only the
                        # must's duty changed; no limit at all would let a
                        # grown must pull it past its outlet, so that its
                        # utility runs backwards). A port without a limit
                        # in the synthesis (see `synthesize_network`) gets
                        # none. A branch stage (fraction f of the flow) takes
                        # f times the whole stream's limit at its share (the
                        # partner: f times the stream's outlet), since the
                        # splits keep their fractions.
                        index = lc.index
                        key = unit.ID, index
                        fraction = stage_fractions.get(key)
                        if (index == 0 and fraction is not None
                                and (unit.ID, 1) in stage_fractions):
                            fraction = 1.
                        if fraction is None:
                            H_lim = None
                        else:
                            H_lim = H_in + fraction * (H_out - H_in)
                            f = stage_scales.get(key, 1.)
                            if f != 1.: H_lim *= f
                        setattr(unit, f'H_lim{index}', H_lim)
                sys = self.HXN_sys
                for unit in sys.units: _move_flows(unit)
            else:
                # Signed-duty order is the default matching priority of the
                # synthesis passes: smallest heating duty first among the cold
                # streams, largest cooling duty first among the hot streams.
                hx_utils.sort(key = lambda x: x.duty)
                self.HXN_flowsheet = HXN_F = flowsheet
                for i in HXN_F.registries: i.clear()
                self.synthesis_info = synthesis_info = {}
                HXs_hot_side, HXs_cold_side, new_HX_utils, hxs, T_in_arr,\
                T_out_arr, pinch_T_arr, C_flow_vector, hx_heat_utils_rearranged, streams_inlet, stream_HXs_dict,\
                hot_indices, cold_indices = \
                synthesize_network(hx_utils, self.T_min_app, self.Qmin,
                                   self.force_ideal_thermo, self.avoid_recycle,
                                   self.sort_hus_by_T, info=synthesis_info,
                                   stream_splitting=self.stream_splitting)
                new_HXs = HXs_hot_side + HXs_cold_side
                # the realized splits (none without stream splitting)
                splits = synthesis_info.get('splits', ())
                self.new_splitters = new_splitters = [
                    u for split in splits for u in split.splitters
                ]
                self.new_mixers = new_mixers = [split.mixer for split in splits]
                self.new_HXs_hot_side = HXs_hot_side
                self.new_HXs_cold_side = HXs_cold_side
                self.cold_indices = cold_indices
                self.original_heat_exchangers = hxs
                self.new_HXs = new_HXs
                self.new_HX_utils = new_HX_utils
                self.streams_inlet = streams_inlet
                stream_life_cycles = self._get_stream_life_cycles()
                self.stream_HXs_dict = stream_HXs_dict
                self.pinch_Ts = pinch_T_arr
                self.inlet_Ts = T_in_arr
                self.outlet_Ts = T_out_arr
                all_units = new_HXs + new_HX_utils + new_splitters + new_mixers
                IDs = set([i.ID for i in all_units])
                assert len(all_units) == len(IDs)
                # Each stream enters at its entry port (its first stage, or
                # the splitter chain of a split at its inlet) as a copy of
                # its real inlet, and every connection of its life cycle
                # (for an unsplit stream, from each stage to the next) is
                # wired.
                for i, life_cycle in enumerate(stream_life_cycles):
                    entry = life_cycle.entry
                    s_util = hx_heat_utils_rearranged[i].unit.ins[0]
                    s_lc = entry.unit.ins[entry.index]
                    s_lc.copy_like(s_util)
                    self._enter_network(i, entry, s_lc)
                for life_cycle in stream_life_cycles:
                    for up, up_port, down, down_port in life_cycle.connections():
                        down.ins[down_port] = up.outs[up_port]
                # For the cached network: the share of its stream's duty at
                # which each process stage's enthalpy limit sits, so that a
                # changed feed rescales every stage instead of letting the
                # first one take the whole duty. A branch stage (fraction f
                # of the flow) has f times the whole stream's limit: its
                # share is that of the whole stream's limit (on the parent
                # basis), and f is kept in `_stage_scales`. The options the
                # network was synthesized with are the cache's key.
                self._stage_fractions = stage_fractions = {}
                self._stage_scales = stage_scales = {}
                self._synthesis_options = (self.stream_splitting,)
                for i, life_cycle in enumerate(stream_life_cycles):
                    hx = hx_heat_utils_rearranged[i].unit
                    H_in = hx.ins[0].H
                    span = hx.outs[0].H - H_in
                    for lc in life_cycle.life_cycle:
                        unit = lc.unit
                        if isinstance(unit, bst.HXutility): continue
                        H_lim = getattr(unit, f'H_lim{lc.index}')
                        if H_lim is None: continue
                        f = lc.fraction
                        if f != 1.:
                            H_lim /= f
                            stage_scales[unit.ID, lc.index] = f
                        stage_fractions[unit.ID, lc.index] = (
                            (H_lim - H_in) / span if span else 0.
                        )
                path, recycles = _network_path(all_units, stream_life_cycles)
                self.HXN_sys = sys = bst.System(HXN_F.ID, path,
                                                recycle=recycles or None)
                # Every stage starts at its planned state (synthesize_network
                # runs each exchanger once), so recycle loops (e.g. a pair
                # matched above and below the pinch) are at their fixed point
                # after the first pass; tight tolerances close the energy
                # balance to ~1e-10 % (molar flows never change: the
                # temperature criterion governs; 1e-8 K sits above the flash
                # noise).
                sys.set_tolerance(method='fixedpoint', subsystems=True,
                                  mol=1e-9, rmol=1e-12, T=1e-8, rT=1e-12,
                                  maxiter=200)

            original_purchase_costs = [hx.purchase_cost for hx in hxs]
            original_installed_costs = [hx.installed_cost for hx in hxs]
            sys._setup()
            try: 
                sys.converge()
            except:
                for i in sys.units: i._run()
                warn('heat exchanger network was not able to converge', RuntimeWarning)
            _pass_through_served_streams(
                stream_life_cycles, [hu.unit for hu in hx_heat_utils_rearranged]
            )
            for i in sys.units: i._summary()
            for i in range(len(stream_life_cycles)):
                hx = hx_heat_utils_rearranged[i].unit
                P = hx.ins[0].P
                s_util = hx.outs[0]
                lc = stream_life_cycles[i].life_cycle[-1]
                s_lc = lc.unit.outs[lc.index]
                IDs = tuple([i.ID for i in s_util.available_chemicals])
                if use_cached_network:
                    try:
                        assert np.isfinite(hx.installed_cost)
                        np.testing.assert_allclose(s_util.imol[IDs], s_lc.imol[IDs])
                        np.testing.assert_allclose(P, s_lc.P, rtol=1e-3, atol=0.1)
                        np.testing.assert_allclose(s_util.H, s_lc.H, rtol=1e-3, atol=1.)
                    except AssertionError as e:
                        msg = ("heat exchanger network cache algorithm failed, "
                               f"cached network ignored: {e}")
                        warn(msg, RuntimeWarning, stacklevel=2)
                        del self.original_heat_utils
                        self._cost()
                        return
                else:
                    np.testing.assert_allclose(s_util.imol[IDs], s_lc.imol[IDs], rtol=1e-3, atol=0.1)
                    np.testing.assert_allclose(P, s_lc.P, rtol=1e-3, atol=0.1)
                    np.testing.assert_allclose(s_util.H, s_lc.H, rtol=1e-3, atol=1.)
            new_purchase_costs_HXp = []
            new_purchase_costs_HXu = []
            new_installed_costs_HXp = []
            new_installed_costs_HXu = []
            new_utility_costs = []
            for hx in new_HX_utils:
                new_installed_costs_HXu.append(hx.installed_cost)
                new_purchase_costs_HXu.append(hx.purchase_cost)
                new_utility_costs.append(hx.utility_cost)
            for new_HX in new_HXs:
                new_purchase_costs_HXp.append(new_HX.purchase_cost)
                new_installed_costs_HXp.append(new_HX.installed_cost)
            hu_sums1 = bst.HeatUtility.sum_by_agent(hx_heat_utils_rearranged)
            new_heat_utils = sum([hx.heat_utilities for hx in new_HX_utils], [])
            hu_sums2 = bst.HeatUtility.sum_by_agent(new_heat_utils)
            # to change sign on duty without switching heat/cool (i.e. negative costs):
            for hu in hu_sums1: hu.reverse()
            hus_final = bst.HeatUtility.sum_by_agent(hu_sums1 + hu_sums2)
            Q_bal = (
                (2.*sum([abs(i.Q) for i in new_HXs])
                 + sum([abs(i.duty * i.agent.heat_transfer_efficiency) for i in hu_sums2]))
                / sum([abs(i.duty * i.agent.heat_transfer_efficiency) for i in hu_sums1])
            )
            energy_balance_error = Q_bal - 1
            self.energy_balance_percent_error = 100 * energy_balance_error
            
            if new_HXs:
                self.installed_costs['Heat exchangers'] = max(0, (
                        sum(new_installed_costs_HXp)
                        + sum(new_installed_costs_HXu)
                        - sum(original_installed_costs)
                ))
                self.purchase_costs['Heat exchangers'] = self.baseline_purchase_costs['Heat exchangers'] = max(0, (
                    sum(new_purchase_costs_HXp) 
                    + sum(new_purchase_costs_HXu)
                    - sum(original_purchase_costs)
                ))
                # with replace_unit_heat_utilities, the units carry the new
                # utilities (replaced below, once the network is final)
                self.heat_utilities = (
                    [] if self.replace_unit_heat_utilities else hus_final
                )
            else: # if no matches were made, retain all original HXutilities (i.e., don't add the -- relatively minor -- differences between new and original HXutilities)
                self.installed_costs['Heat exchangers'] = 0.
                self.baseline_purchase_costs['Heat exchangers'] = self.purchase_costs['Heat exchangers'] = 0.
                self.heat_utilities = []
            self.original_heat_utils = hx_heat_utils_rearranged
            self.original_purchase_costs = original_purchase_costs
            self.original_utility_costs = hu_sums1
            self.new_purchase_costs_HXp = new_purchase_costs_HXp
            self.new_purchase_costs_HXu = new_purchase_costs_HXu
            self.new_utility_costs = hu_sums2
            new_hus = bst.process_tools.heat_exchanger_utilities_from_units(new_HX_utils)
            hus_heating = [hu for hu in hx_utils if hu.duty > 0]
            hus_cooling = [hu for hu in hx_utils if hu.duty < 0]
            self.original_heat_util_load = sum([hu.duty for hu in hus_heating])
            self.original_cool_util_load = sum([abs(hu.duty) for hu in hus_cooling])
            self.actual_heat_util_load = sum([hu.duty for hu in new_hus if hu.duty>0])
            self.actual_cool_util_load = sum([abs(hu.duty) for hu in new_hus if hu.duty<0])
            if abs(energy_balance_error) > self.acceptable_energy_balance_error:
                if use_cached_network:
                    del self.original_heat_utils
                    self._cost()
                    return
                msg = ("heat exchanger network energy balance is off by "
                      f"{energy_balance_error:.2%} (an absolute error greater "
                      f"than {self.acceptable_energy_balance_error:.2%})")
                if self.raise_energy_balance_error:
                    raise RuntimeError(msg)
                else:
                    warn(msg, RuntimeWarning, stacklevel=2)
            # Last, so that nothing above (the loads, a re-synthesis) reads
            # the original heat utilities after they are overwritten
            if new_HXs and self.replace_unit_heat_utilities:
                self._replace_unit_heat_utilities(hx_heat_utils_rearranged,
                                                  stream_life_cycles)

    def _energy_balance_error_contributions(self):
        original_ignored = ignored = self.ignored
        if ignored and callable(ignored): ignored = ignored()
        energy_balance_errors = {}
        for hu in self._get_original_heat_utilties():
            self.ignored = list(ignored or ()) + [hu.unit]
            if hasattr(hu.unit, 'owner'):
                ID = hu.unit.owner.ID, hu.unit.ID
            else:
                ID = hu.unit.ID
            try:
                self.simulate()
            except: 
                energy_balance_errors[ID] = (hu, None)
            else:
                energy_balance_errors[ID] = (hu, self.energy_balance_percent_error)
        self.ignored = original_ignored
        return energy_balance_errors
            
    def _get_stream_life_cycles(self):
        cold_indices = self.cold_indices
        new_HXs = self.new_HXs
        new_HX_utils = self.new_HX_utils
        streams = self.streams_inlet
        indices = [i for i in range(len(streams))]
        # each stream's own splits (``synthesis_info['splits']``); a stream
        # without any gets its life cycle exactly as without splitting
        splits = {}
        for split in self.synthesis_info.get('splits', ()):
            splits.setdefault(split.stream, []).append(split)
        SLCs = [StreamLifeCycle(index, index in cold_indices) for index in indices]
        for SLC in SLCs:
            if SLC.index in splits:
                SLC.get_life_cycle(new_HXs, new_HX_utils,
                                   splits=splits[SLC.index])
            else:
                SLC.get_life_cycle(new_HXs, new_HX_utils)
        stream_life_cycles = SLCs
        self.stream_life_cycles = stream_life_cycles
        return stream_life_cycles
        
    def plot_pinch_diagram(self, file=None, **kwargs):
        """
        Draw the pinch diagram of the synthesized network; see
        :func:`~hensmith.hxn_synthesis.plot_pinch_diagram`
        for the keyword arguments. Returns the matplotlib figure and axes.
        """
        if not hasattr(self, 'new_HXs_hot_side'):
            raise RuntimeError('simulate the heat exchanger network before '
                               'plotting its pinch diagram')
        return plot_pinch_diagram(
            self.stream_life_cycles, self.inlet_Ts, self.outlet_Ts,
            self.new_HXs_hot_side, self.new_HXs_cold_side,
            Qmin=self.Qmin, original_hxs=self.original_heat_exchangers,
            file=file, **kwargs,
        )

    def get_original_hxs_associated_with_streams(self): # pragma: no cover
        original_units = self.system.units
        original_heat_utils = self.original_heat_utils
        original_hx_utils = [i.unit for i in original_heat_utils]
        original_hxs = {}
        stream_index = 0
        for hx in original_hx_utils:
            if '.' in hx.ID: # Names like 'U.1', i.e. non-explicitly named unit (e.g. auxillary HX)
                for unit in original_units:
                    if isinstance(unit, bst.units.MultiEffectEvaporator):
                        for key, component in unit.components.items():
                            if isinstance(component, list):
                                for subcomponent in component:
                                    if subcomponent is hx:
                                         original_hxs[stream_index] = (unit, key)
                            elif component is hx:
                                original_hxs[stream_index] = (unit, key)
                    elif isinstance(unit, bst.units.BinaryDistillation)\
                        or isinstance(unit, bst.units.ShortcutColumn):
                        if unit.boiler is hx:
                            original_hxs[stream_index] = (unit, 'boiler')
                        elif unit.condenser is hx:
                            original_hxs[stream_index] = (unit, 'condenser')
                    elif hasattr(unit, 'heat_exchanger'):
                        if unit.heat_exchanger is hx:
                            original_hxs[stream_index] = (unit, 'heat exchanger')
            else: # Explicitly named unit
                original_hxs[stream_index] = (hx, '')
            stream_index += 1
        self.original_hxs = original_hxs
        return original_hxs
    
    def save_stream_life_cycles_as_csv(self):
        if not hasattr(self, 'stream_life_cycles'):
            self.stream_life_cycles = self._get_stream_life_cycles()
        stream_life_cycles = self.stream_life_cycles
        if not hasattr(self, 'original_hxs'):
            self.original_hxs = self.get_original_hxs_associated_with_streams()
        original_hxs = self.original_hxs
        import csv
        from datetime import datetime
        dateTimeObj = datetime.now()
        filename = 'HXN-%s_%s.%s.%s.%s.%s.csv'%(self.system.ID, dateTimeObj.year,
                                                dateTimeObj.month, dateTimeObj.day,
                                                dateTimeObj.hour, dateTimeObj.minute)
        inlet_Ts = self.inlet_Ts
        outlet_Ts = self.outlet_Ts
        with open(filename, 'w', newline='') as file:
            csvWriter = csv.writer(file, delimiter=',')
            csvWriter.writerow(['Stream', 'Type', 'Original unit', 'HXN unit',
                                'H_in (kJ/hr)', 'H_out (kJ/hr)', 'T_in (C)',
                                'T_out (C)'])
            for life_cycle in stream_life_cycles:
                stream = life_cycle.index
                streamtype = 'Cold' if life_cycle.cold else 'Hot'
                original_unit = original_hxs[stream][0].ID
                if original_hxs[stream][1]:
                    original_unit += ' - ' + original_hxs[stream][1]
                # One row per stage, at its own enthalpies. A stream that
                # splits at its inlet enters its splitter chain first, which
                # gets a row at the whole stream's inlet enthalpy (its first
                # stage is then a branch: a fraction of the flow).
                rows = [(stage.unit.ID, stage.H_in, stage.H_out)
                        for stage in life_cycle.life_cycle]
                entry = life_cycle.entry.unit
                if isinstance(entry, bst.Splitter):
                    H_in = life_cycle.H_in
                    rows.insert(0, (entry.ID, H_in, H_in))
                last = len(rows) - 1
                for n, (ID, H_in, H_out) in enumerate(rows):
                    T_in = inlet_Ts[stream] - 273.15 if n == 0 else None
                    T_out = outlet_Ts[stream] - 273.15 if n == last else None
                    csvWriter.writerow([stream, streamtype, original_unit, ID,
                                        H_in, H_out, T_in, T_out])
