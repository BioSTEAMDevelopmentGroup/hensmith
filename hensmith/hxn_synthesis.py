# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Pinch analysis and heat exchanger network synthesis: the problem table
(`problem_table`), the synthesis of an unsplit network at minimum energy
requirement (`synthesize_network`, on the planner of `hensmith._planner`),
stream life cycles (`StreamLifeCycle`) and pinch diagrams
(`plot_pinch_diagram`).
"""
from collections import namedtuple
import heapq
import math
import re
import numpy as np
import biosteam as bst
from warnings import warn
from ._curves import (StreamCurve, stream_curves, _end_state, _T_EQ, _T_SIDE,
                      _copy, _point_load_inlet)
from ._planner import plan_network
from ._splitting import Split

__all__ = ('StreamLifeCycle', 'ProblemTable', 'problem_table',
           'synthesize_network', 'plot_pinch_diagram')

#: IDs of the synthesized exchangers: process exchangers above the pinch
#: ``HX_<cold>_<hot>_hs`` and below it ``HX_<hot>_<cold>_cs`` (the first
#: number is the stream at port 0), with ``_<n>`` for the n-th exchanger of a
#: repeated pair; utility exchangers ``Util_<index>_hs|cs``.
_PROCESS_ID = re.compile(r'^HX_(\d+)_(\d+)_(hs|cs)(?:_\d+)?$')
_UTILITY_ID = re.compile(r'^Util_(\d+)_(hs|cs)$')

def _stream_ports(unit):
    """Stream index at each inlet port of a synthesized exchanger, parsed
    from its ID (``(a, b)`` for ``HX_<a>_<b>_...``, ``(a,)`` for
    ``Util_<a>_...``), or None if the ID is not one of the synthesizer's."""
    ID = unit.ID
    match = _PROCESS_ID.match(ID)
    if match: return int(match.group(1)), int(match.group(2))
    match = _UTILITY_ID.match(ID)
    if match: return (int(match.group(1)),)
    return None

class LifeStage:
    """
    One stage of a stream's passage through the synthesized network: the
    heat exchanger it passes through and which of that exchanger's
    inlet/outlet pairs carries the stream.

    Parameters
    ----------
    unit : HXprocess or HXutility
        Heat exchanger of the synthesized network.
    index : int
        Position of the stream in `unit.ins` / `unit.outs` (0 or 1 for an
        `HXprocess`; always 0 for an `HXutility`).

    Attributes
    ----------
    s_in : Stream
        `unit.ins[index]`, the stream entering this stage.
    s_out : Stream
        `unit.outs[index]`, the stream leaving this stage.
    H_in : float
        Enthalpy of `s_in` [kJ/hr], read from the stream when accessed.
    H_out : float
        Enthalpy of `s_out` [kJ/hr], read from the stream when accessed.

    """
        
    def __init__(self, unit, index):
        self.unit = unit
        self.index = index
    
    @property
    def s_in(self): return self.unit.ins[self.index]
    
    @property
    def s_out(self): return self.unit.outs[self.index]
    
    @property
    def H_in(self): return self.s_in.H
    
    @property
    def H_out(self): return self.s_out.H
    
    def _info(self, N_tabs=1):
        tabs = N_tabs*'\t'
        return (f"{type(self).__name__}: {self.unit.ID}\n"
                + tabs + f"H_in = {self.H_in:.3g} kJ/hr\n"
                + tabs + f"H_out = {self.H_out:.3g} kJ/hr")

    def __repr__(self):
        return (f"<{type(self).__name__}: {repr(self.unit)}, H_in = {round(self.H_in, 4):.3g} kJ/hr, H_out = {round(self.H_out, 4):.3g} kJ/hr>")
        
    def show(self):
        print(self._info())
    _ipython_display_ = show
        
        
class StreamLifeCycle:
    """
    The ordered sequence of heat exchangers one process stream passes through
    in a synthesized heat exchanger network, from its inlet state to its
    final utility exchanger.

    Streams are numbered by their position in the rearranged utility list
    of `synthesize_network` (cold streams first, then hot streams); the
    network's stream copies and exchanger IDs embed that index
    (``s_<index>__<exchanger ID>`` for the inlet streams of the exchangers,
    ``HX_<hot>_<cold>_cs`` / ``HX_<cold>_<hot>_hs`` for process exchangers,
    with a suffix ``_<n>`` for the n-th exchanger of a repeated pair,
    ``Util_<index>_cs`` / ``Util_<index>_hs`` for utility exchangers), which
    is how the life cycle is recovered from the exchangers: the IDs are
    parsed (the first number is the stream at port 0, the second the stream
    at port 1), so the stream indices are matched exactly and never as
    substrings of other indices or of rewired stream IDs.

    Parameters
    ----------
    index : int
        Stream index in the synthesized network.
    cold : bool
        True for a heated (cold) stream, False for a cooled (hot) stream.

    Attributes
    ----------
    index : int
        Stream index in the synthesized network.
    cold : bool
        Whether the stream is a heated (cold) stream.
    name : str
        ``'s_<index>'``, the prefix of the stream's copies in the network.
    life_cycle : list[LifeStage] or None
        Stages in flow order, set by `get_life_cycle`; None until then.

    Notes
    -----
    `HeatExchangerNetwork` builds one life cycle per stream after synthesis
    and stores them in `HeatExchangerNetwork.stream_life_cycles`, aligned
    with `HeatExchangerNetwork.original_heat_exchangers`;
    `plot_pinch_diagram` draws them.

    """
    
    def __init__(self, index, cold):
        self.index = index
        self.name = 's_%s'%index
        self.cold = cold
        self.life_cycle = None
        
    def get_relevant_units(self, index, new_HXs, new_HX_utils):
        """
        Return the process and utility exchangers (two lists) that carry
        stream `index`: those whose parsed ID (``HX_<a>_<b>_<hs|cs>[_<n>]``
        or ``Util_<a>_<hs|cs>``) names it; for any other ID, those whose ID
        contains ``_<index>_``.
        """
        def relevant(hx):
            ports = _stream_ports(hx)
            if ports is None: return '_%s_'%index in hx.ID
            return index in ports
        new_HXs_relevant = [hx for hx in new_HXs if relevant(hx)]
        new_HX_utils_relevant = [hx for hx in new_HX_utils if relevant(hx)]
        return new_HXs_relevant, new_HX_utils_relevant
        
    def get_life_cycle(self, new_HXs, new_HX_utils):
        """
        Build and return the list of `LifeStage` objects for this stream.

        Parameters
        ----------
        new_HXs : list[HXprocess]
            Process exchangers of the synthesized network.
        new_HX_utils : list[HXutility]
            Utility exchangers of the synthesized network.

        Returns
        -------
        list[LifeStage]
            One stage per port that carries this stream: for an exchanger
            ID of the synthesizer (see the class notes) the port its ID
            assigns to `index` (0 or 1 for a process exchanger, 0 for a
            utility exchanger); for any other ID, every port among 0 and 1
            (0 for a utility) whose inlet ID contains ``'s_<index>_'``.
            Sorted in flow direction: by inlet enthalpy, ascending for a
            cold stream and descending for a hot one; ties (zero-duty
            stages only) put the stream's first side of the pinch first
            (cold-side stages for a cold stream, hot-side stages for a hot
            one) and the utility last. Also stored as `life_cycle`.

        """
        index = self.index
        name = self.name
        cold = self.cold
        new_HXs_relevant, new_HX_utils_relevant =\
            self.get_relevant_units(index, new_HXs, new_HX_utils)
        life_cycle = []
        for units, N_ports in ((new_HXs_relevant, 2), (new_HX_utils_relevant, 1)):
            for unit in units:
                ports = _stream_ports(unit)
                if ports is None:
                    life_cycle.extend([LifeStage(unit, k) for k in range(N_ports)
                                       if name + '_' in unit.ins[k].ID])
                else:
                    life_cycle.extend([LifeStage(unit, k) for k, i in enumerate(ports)
                                       if i == index])
        sign = 1. if cold else -1.
        first_side = '_cs' if cold else '_hs'
        def flow_order(stage):
            ID = stage.unit.ID
            if isinstance(stage.unit, bst.HXutility): rank = 2
            else: rank = 0 if first_side in ID else 1
            return (sign * stage.H_in, rank)
        life_cycle.sort(key=flow_order)
        self.life_cycle = life_cycle
        return life_cycle
        
    def __repr__(self):
        life_cycle = self.life_cycle
        cold = self.cold
        if not self.life_cycle:
            return 'Not initialized; run StreamLifeCycle.get_life_cycle or\
                  HX_Network.get_stream_life_cycles first.' 
        else:
            index = self.index
            name = 'Stream_%s'%index
            strtype = 'cold' if cold else 'hot'
            rep = ''
            for LifeStage in life_cycle:
                line = '\t\t' + repr(LifeStage) + '\n'
                rep += line
            rep = '<StreamLifeCycle: ' + name + ', ' + strtype  + '\n\tlife_cycle = [\n' +  rep[:-1] + '\n\t]>'
            return rep
        
    def show(self):
        """Print the life cycle, one stage per line."""
        info = repr(self).replace('[', '').replace(']', '').replace('life_cycle =', 'life_cycle:')
        print(info[1:-1])
        
    _ipython_display_ = show


ProblemTable = namedtuple(
    'ProblemTable',
    ['Ts', 'interval_H', 'point_H', 'residual',
     'hot_util_load', 'cold_util_load', 'pinch_T']
)

ProblemTable.__doc__ = """
Result of `problem_table`: the temperature-interval heat cascade of a set of
process streams on the *shifted* temperature scale (hot streams shifted down
by the minimum approach temperature, cold streams unshifted).

Attributes
----------
Ts : numpy.ndarray
    Shifted grid temperatures [K], descending: every shifted breakpoint of
    every stream's temperature-enthalpy curve, i.e. its end temperatures,
    the phase boundaries inside its range (a pure component's saturation
    temperature, a mixture's bubble and dew points), the samples of its
    two-phase glides and of its curved single-phase stretches
    (temperature-dependent heat capacity), and the outlet temperature of
    every point-load stream (see `point_H`). Temperatures closer than
    1e-9 K are one grid point.
interval_H : numpy.ndarray
    (N streams x n-1 intervals) heat contributed by each stream to each
    open interval (Ts[k], Ts[k+1]) [kJ/hr]: positive for hot streams (heat
    released), negative for cold streams (heat required); zero outside the
    stream's own temperature range.
point_H : numpy.ndarray
    (N x n) heat contributed *at* each grid temperature [kJ/hr], with the
    same sign convention: the jump of the stream's curve there, i.e. a pure
    component's latent heat at its (shifted) saturation temperature, the
    enthalpy by which a non-equilibrium end state departs from equilibrium
    at its own end temperature, and the whole duty of a point-load stream
    at its shifted outlet temperature (isothermal streams and streams whose
    outlet temperature moves against their duty).
residual : numpy.ndarray
    (n,) heat cascaded *leaving* each grid temperature, after its point
    loads, when no hot utility is supplied [kJ/hr]; negative where that
    cascade is infeasible.
hot_util_load : float
    Minimum hot utility target [kJ/hr] (zero for a threshold problem).
cold_util_load : float
    Minimum cold utility target [kJ/hr].
pinch_T : float
    Shifted grid temperature of the pinch [K] (``Ts[0]`` for a threshold
    problem); the hot-stream pinch temperature is `pinch_T + T_min_app`,
    the cold-stream one is `pinch_T`.

See Also
--------
problem_table : builds the table and documents the cascade.

"""

def problem_table(streams_inlet, streams_quenched, is_hot, T_min_app,
                  curves=None):
    """
    Energy-consistent problem table (temperature-interval heat cascade).

    Parameters
    ----------
    streams_inlet : list[Stream]
        Inlet stream of each utility heat exchanger.
    streams_quenched : list[Stream]
        Corresponding outlet streams, re-flashed at their enthalpy.
    is_hot : Sequence[bool]
        True where the stream is cooled.
    T_min_app : float
        Minimum approach temperature [K].
    curves : list, optional
        Prebuilt temperature-enthalpy curves of the same streams, in the
        same order (e.g. shared with the network synthesis); built here if
        not given.

    Returns
    -------
    ProblemTable
        Grid temperatures `Ts` (shifted scale, descending), per-stream
        `interval_H` (N x n-1) and `point_H` (N x n) contributions (+ for
        hot, - for cold), the cascade `residual` (n) *leaving* each
        boundary (i.e. after its point loads), `hot_util_load`,
        `cold_util_load` and the shifted-scale `pinch_T`.

    Notes
    -----
    Each stream is described by a piecewise-linear temperature-enthalpy
    curve built once from a handful of flashes: breakpoints at its end
    temperatures and at every phase boundary inside its range (a pure
    component's saturation temperature, a mixture's bubble and dew points);
    a flat (isothermal) segment for a pure component's latent heat, between
    its saturated-liquid and saturated-vapor enthalpies; samples of each
    mixture glide (binaries traced along their bubble-point curve); and
    interior breakpoints of each curved single-phase stretch
    (temperature-dependent Cp), dense enough that linear interpolation is
    within 0.002 K of the true curve. Single-phase stretches are evaluated
    with their phases fixed (no flash), so a grid point exactly at a
    saturation temperature is never ambiguous. Hot streams are shifted down
    by `T_min_app`; cold streams are not. The grid is the union of all
    shifted breakpoints, so every point at which any stream's curve bends
    or jumps is a grid point, every stream is within 0.002 K of linear
    between grid points, and the grid minimum of the cascade is the true
    one to within ``0.002 K * sum(CP)``.

    At a grid temperature a stream contributes the jump of its curve there
    as a point load (`point_H`), and between two grid temperatures the heat
    of its curve in that open interval (`interval_H`). Inside its own
    temperature range a stream is taken at equilibrium with its enthalpy
    clipped to its real range, so a non-equilibrium end state (e.g. a
    superheated liquid from a non-rigorous HXutility) can never inflate the
    duty; what it departs from equilibrium at its own end temperature is a
    point load there. Every stream's contributions therefore telescope
    exactly to ``sign * |H_out - H_in|``. Streams whose outlet temperature
    does not move with their duty (isothermal, or a heated stream that
    exits colder than it entered, e.g. a reboiler outlet at VLE) are point
    loads at their outlet temperature.

    The cascade starting from zero hot utility is residual[k] =
    sum(point_H[:, :k+1]) + sum(interval_H[:, :k]), the heat *leaving*
    boundary Ts[k]. Feasibility must also hold for the heat *arriving* at
    Ts[k] before its point loads are applied, arriving[k] = residual[k] -
    sum(point_H[:, k]), because a source at Ts[k] cannot serve a sink above
    Ts[k]. The minimum over both flows, min(residual, arriving), fixes the
    hot utility target, `residual[-1] + hot_util_load` the cold one, and its
    first location the pinch. With the per-stream identity above,
    hot_util_load - cold_util_load equals the net heating demand.

    Examples
    --------
    A threshold problem: 1000 kmol/hr of water cooled 400 -> 300 K supplies
    every interval of 900 kmol/hr of water heated 300 -> 390 K, so no hot
    utility is needed and the surplus leaves as cold utility. (The grid
    between the ends holds the breakpoints that follow the curvature of
    liquid water's enthalpy.)

    >>> import biosteam as bst
    >>> from hensmith.hxn_synthesis import problem_table
    >>> bst.settings.set_thermo(['Water'])
    >>> hot_in = bst.Stream(Water=1000., T=400., P=5e5, phase='l', units='kmol/hr')
    >>> hot_out = hot_in.copy(); hot_out.vle(T=300., P=5e5)
    >>> cold_in = bst.Stream(Water=900., T=300., P=5e5, phase='l', units='kmol/hr')
    >>> cold_out = cold_in.copy(); cold_out.vle(T=390., P=5e5)
    >>> table = problem_table([hot_in, cold_in], [hot_out, cold_out],
    ...                       [True, False], 5.)
    >>> table.Ts[[0, -1]]  # shifted grid ends (hot streams 5 K down)
    array([395., 295.])
    >>> round(table.hot_util_load, 3)
    0.0
    >>> round(table.cold_util_load, -1)
    1445550.0
    >>> table.pinch_T
    395.0
    """
    return _problem_table(streams_inlet, streams_quenched, is_hot, T_min_app,
                          curves)[0]

def _problem_table(streams_inlet, streams_quenched, is_hot, T_min_app,
                   curves=None):
    """
    Return the `ProblemTable` of `problem_table` together with the stream
    curves it was built from and its grid enthalpies, as ``(table, curves,
    grid)``.

    `grid` is a dict with

    * 'Ts': the table's shifted grid temperatures (descending, n);
    * 'shift': each stream's shift (`T_min_app` for hot streams, 0 for
      cold ones; N), so stream j's real temperature at grid index k is
      ``Ts[k] + shift[j]``;
    * 'Hl', 'Hr': (N x n) each stream's enthalpy at every grid temperature,
      left (low-enthalpy) and right (high-enthalpy) limit; they differ only
      where the stream's curve has a flat (point load) at that grid
      temperature, and are H_hi above and H_lo below the stream's range;
    * 'k_hi', 'k_lo': (N,) grid indices of each stream's own T_hi and T_lo
      breakpoints (``k_hi <= k_lo``; equal for a point-load stream), so its
      range is selected by position, not by comparing shifted floats.

    The table is exactly ``point_H = sign * (Hr - Hl)`` and ``interval_H =
    sign * (Hl[:, :-1] - Hr[:, 1:])`` (sign +1 hot, -1 cold): piecewise-
    linear curves through the knots (Ts[k] + shift[j], Hl[j, k]) and
    (Ts[k] + shift[j], Hr[j, k]) reproduce the table's cascade exactly.
    """
    N = len(streams_inlet)
    is_hot = np.asarray(is_hot, dtype=bool)
    sign = np.where(is_hot, 1., -1.)
    shift = np.where(is_hot, T_min_app, 0.)
    if curves is None:
        curves = stream_curves(streams_inlet, streams_quenched, is_hot)
    elif len(curves) != N:
        raise ValueError(f'{len(curves)} curves given for {N} streams')
    H_in = np.array([c.H_in for c in curves])
    H_out = np.array([c.H_out for c in curves])
    shifted = [c.T - shift[j] for j, c in enumerate(curves)]
    sizes = [x.size for x in shifted]
    values, inverse = np.unique(np.concatenate(shifted), return_inverse=True)
    # merge grid temperatures closer than 1e-9 K (ascending clusters)
    cluster = np.concatenate([[0], np.cumsum(np.diff(values) > _T_EQ)])
    n = int(cluster[-1]) + 1
    # the highest member of each cluster represents it (selected explicitly:
    # NumPy does not specify which value a repeated fancy index keeps)
    top = values[np.flatnonzero(np.append(np.diff(cluster) > 0, True))]
    Ts = top[::-1].copy()
    position = (n - 1) - cluster[inverse]  # descending grid index of each breakpoint
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    Hl = np.empty((N, n))
    Hr = np.empty((N, n))
    k_hi = np.empty(N, dtype=int)
    k_lo = np.empty(N, dtype=int)
    for j, c in enumerate(curves):
        index = position[offsets[j]:offsets[j + 1]]
        Hl[j], Hr[j] = c.grid_limits(Ts, shift[j], index)
        k_hi[j] = index[-1]
        k_lo[j] = index[0]
    point_H = sign[:, None] * (Hr - Hl)
    interval_H = sign[:, None] * (Hl[:, :-1] - Hr[:, 1:])
    point_total = point_H.sum(axis=0)
    residual = np.cumsum(
        point_total + np.concatenate([[0.], interval_H.sum(axis=0)])
    )
    # heat arriving at each boundary, before that boundary's point loads:
    # a point source at Ts[k] cannot serve sinks above Ts[k], so the cascade
    # must be non-negative both before and after the point loads
    arriving = residual - point_total
    flow = np.minimum(residual, arriving)
    k_pinch = int(np.argmin(flow))
    scale = np.abs(H_out - H_in).sum()
    if -flow[k_pinch] <= 1e-9 * scale:  # threshold problem: no hot utility
        hot_util_load = 0.
        k_pinch = 0
    else:
        hot_util_load = -flow[k_pinch]
    cold_util_load = residual[-1] + hot_util_load
    if cold_util_load < 0.:
        # only reachable in the threshold branch, by at most 1e-9 * scale:
        # absorb the rounding into the hot utility so that
        # hot_util_load - cold_util_load == sum(unit_duty) stays exact
        hot_util_load -= cold_util_load
        cold_util_load = 0.
    table = ProblemTable(Ts, interval_H, point_H, residual,
                         hot_util_load, cold_util_load, Ts[k_pinch])
    grid = dict(Ts=Ts, shift=shift, Hl=Hl, Hr=Hr, k_hi=k_hi, k_lo=k_lo)
    return table, curves, grid

def _pinch_cut(table):
    """
    Return which side of the pinch the point loads *at* the pinch
    temperature belong to: 'below' if the zero-heat-flow cut is the flow
    arriving at `pinch_T` (before its point loads), 'above' if it is the
    flow leaving it (after them); always 'below' for a threshold problem
    (no hot utility: everything lies below the pinch at ``Ts[0]``). Split
    every stream with ``side = 'right' if cut == 'below' else 'left'``
    (see `pinch_state`) to agree with the table: the heat above the split
    is then exactly the hot utility target and the heat below it the cold
    one. (`synthesize_network` does not split streams: the planner finds
    the same cut in its own cascade, ``plan.cut``, which reproduces the
    table's.)
    """
    if table.hot_util_load == 0.: return 'below'
    k = int(np.flatnonzero(table.Ts == table.pinch_T)[0])
    point_total = table.point_H.sum(axis=0)
    arriving = table.residual - point_total
    return 'below' if arriving[k] <= table.residual[k] else 'above'

def _pinch_analysis(hus, T_min_app=10, force_ideal_thermo=False,
                    sort_hus_by_T=False):
    """
    The first step of `synthesize_network`: prepare the process streams
    behind `hus` and run the problem table on them. Returns the 12 values
    of `temperature_interval_pinch_analysis` (which wraps this function)
    and then the table's ``table, curves, grid`` (see `_problem_table`),
    on which the network is planned, so that the curves are built only
    once.
    """
    hx_utils = hus
    hus_heating = [hu for hu in hx_utils if hu.duty > 0]
    hus_cooling = [hu for hu in hx_utils if hu.duty < 0]
    if sort_hus_by_T:
        hus_heating.sort(key=lambda i: i.unit.ins[0].T, reverse=True)
        hus_cooling.sort(key=lambda i: i.unit.ins[0].T)
    hx_utils_rearranged = hus_heating + hus_cooling
    hxs = [hu.unit for hu in hx_utils_rearranged]
    # unregistered copies (see `hensmith._curves._copy`); the inlets are
    # registered under their own IDs below
    if force_ideal_thermo:
        streams_inlet = [_copy(hx.ins[0], hx.ins[0].thermo.ideal()) for hx in hxs]
        streams_quenched = [_copy(hx.outs[0], hx.outs[0].thermo.ideal()) for hx in hxs]
    else:
        streams_inlet = [_copy(hx.ins[0]) for hx in hxs]
        streams_quenched = [_copy(hx.outs[0]) for hx in hxs]
    for i in streams_quenched: i.vle(H=i.H, P=i.P)
    for i in range(len(streams_inlet)):
        stream = streams_inlet[i]
        ID = 'Util_%s'%i
        stream.ID = 's_%s__%s'%(i,ID)
    N_heating = len(hus_heating)
    cold_indices = list(range(N_heating))
    hot_indices = list(range(N_heating, len(hxs)))
    indices = cold_indices + hot_indices
    T_in_arr = np.array([stream.T for stream in streams_inlet])
    T_out_arr = np.array([i.T for i in streams_quenched])
    is_hot = np.zeros(len(hxs), dtype=bool)
    is_hot[hot_indices] = True
    table, curves, grid = _problem_table(streams_inlet, streams_quenched,
                                         is_hot, T_min_app)
    hot_util_load = table.hot_util_load
    cold_util_load = table.cold_util_load
    pinch_cold_stream_T = table.pinch_T
    pinch_hot_stream_T = pinch_cold_stream_T + T_min_app
    # Per-stream pinch temperature, for information only (returned as
    # `HeatExchangerNetwork.pinch_Ts`; `load_duties` splits a stream there):
    # the network is planned on the curves (see `synthesize_network`), not
    # on these temperatures. A stream already entirely on one side of the
    # process pinch (T_in past pinch_cold_stream_T for a cold stream, or
    # past pinch_hot_stream_T for a hot stream) is not split; its pinch_T is
    # its own T_in. So is a non-monotone stream (T_out on the wrong side of
    # T_in for its duty, e.g. a cold stream whose VLE outlet ends up cooler
    # than it entered: a point load at T_out), whose whole duty
    # `load_duties` then puts on a single side (hot side for a cold stream,
    # cold side for a hot one).
    pinch_T_arr = []
    for i in cold_indices:
        if T_in_arr[i] > pinch_cold_stream_T or T_in_arr[i] > T_out_arr[i]:
            pinch_T_arr.append(T_in_arr[i])
        elif T_out_arr[i] < pinch_cold_stream_T:
            pinch_T_arr.append(T_out_arr[i])
        else:
            pinch_T_arr.append(pinch_cold_stream_T)
    for i in hot_indices:
        if T_in_arr[i] < pinch_hot_stream_T or T_in_arr[i] < T_out_arr[i]:
            pinch_T_arr.append(T_in_arr[i])
        elif T_out_arr[i] > pinch_hot_stream_T:
            pinch_T_arr.append(T_out_arr[i])
        else:
            pinch_T_arr.append(pinch_hot_stream_T)
    pinch_T_arr = np.array(pinch_T_arr)
    return pinch_T_arr, hot_util_load, cold_util_load, T_in_arr, T_out_arr,\
           hxs, hot_indices, cold_indices, indices, streams_inlet, hx_utils_rearranged, \
           streams_quenched, table, curves, grid

def temperature_interval_pinch_analysis(hus,
                                        T_min_app=10,
                                        force_ideal_thermo=False,
                                        sort_hus_by_T=False):
    """
    Prepare the process streams behind `hus` and run the problem table on
    them, as `synthesize_network` does first; a standalone pinch analysis
    (the network itself is planned on the table's stream curves, which this
    function does not return).

    Heating utilities (``hu.duty > 0``, cold streams) come first, then
    cooling utilities; zero-duty utilities are dropped. Each stream is a
    copy of its exchanger's inlet (renamed ``s_<index>__Util_<index>``) and
    of its outlet re-flashed at its own enthalpy.

    Returns
    -------
    pinch_T_arr : numpy.ndarray
        Per-stream pinch temperature (see `synthesize_network`).
    hot_util_load, cold_util_load : float
        MER targets of `problem_table` [kJ/hr].
    T_in_arr, T_out_arr : numpy.ndarray
        Inlet and quenched outlet temperatures [K].
    hxs : list[Unit]
        The original heat exchangers, in stream order.
    hot_indices, cold_indices, indices : list[int]
        Stream indices of the hot streams, the cold streams, and all
        (cold first).
    streams_inlet, hx_utils_rearranged, streams_quenched : list
        Inlet copies, heat utilities and quenched outlet copies, in stream
        order.

    """
    return _pinch_analysis(hus, T_min_app, force_ideal_thermo,
                           sort_hus_by_T)[:12]

def pinch_state(stream_in, stream_out, T_pinch, side=None, curve=None):
    """
    Return a copy of the stream in the state it has when it crosses the
    pinch, with enthalpy guaranteed to lie within [min(H_in, H_out),
    max(H_in, H_out)].

    Parameters
    ----------
    stream_in, stream_out : Stream
        The stream's real end states (the outlet quenched to equilibrium at
        its own enthalpy).
    T_pinch : float
        The stream's (real, unshifted) pinch temperature [K].
    side : str, optional
        Branch of a flat (isothermal) segment of the stream at `T_pinch`
        (latent heat or a non-equilibrium end jump): 'right' takes its
        high-enthalpy end (the load at the pinch goes below the pinch),
        'left' its low-enthalpy end (the load goes above). Pass ``side =
        'right' if cut == 'below' else 'left'``, with ``cut`` the pinch cut
        of the table (`_pinch_cut`), to split every stream consistently
        with the targets. If not given, an end temperature returns that
        end's state (below) and elsewhere hot streams take 'right' and cold
        streams 'left'.
    curve : StreamCurve, optional
        Prebuilt curve of the stream (e.g. from `_problem_table`); built
        here if needed and not given.

    Notes
    -----
    The state comes from the stream's temperature-enthalpy curve (see
    `problem_table`), so it is deterministic: a pinch at the stream's own
    saturation temperature is resolved by `side`, never by whatever phase
    split a previous flash left, and a pinch inside a glide gets the
    table's enthalpy. Inside the stream's temperature range the enthalpy is
    clipped to its real range, as in the table: a non-equilibrium inlet
    (e.g. a superheated liquid from a non-rigorous HXutility) has less
    enthalpy than the equilibrium fluid at the pinch, so the state returned
    is instead the equilibrium state at the nearer end enthalpy. Without
    `side`, `T_pinch` equal to an end temperature returns that end's state
    (the equilibrium state at the end enthalpy, see `_end_state`), because
    flashing a non-equilibrium inlet at its own temperature does not
    reproduce `H_in`. Using the *equilibrium* state at the end enthalpy,
    rather than the stream as given, keeps the synthesizer consistent with
    the problem table: the heat is offered at the temperature the
    equilibrium model says it is available, not at a fictitious one.

    Either way the hot-side and cold-side loads split `|H_in - H_out|`
    exactly, and the state never carries heat the real stream does not
    have.

    A standalone analysis helper (see also `load_duties`): the network
    synthesis does not split streams at a pinch temperature, it plans on
    the curves themselves (see `synthesize_network`).
    """
    if side is None:
        T_lo, T_hi = sorted((stream_in.T, stream_out.T))
        if T_pinch == stream_in.T: return _end_state(stream_in, T_lo, T_hi)
        if T_pinch == stream_out.T: return _end_state(stream_out, T_lo, T_hi)
    if curve is None: curve = StreamCurve(stream_in, stream_out)
    if side is None: side = 'right' if curve.is_hot else 'left'
    return curve.state_at_T(T_pinch, side)

def load_duties(streams, streams_quenched, pinch_T_arr, T_out_arr, indices,
                is_cold, Q_hot_side, Q_cold_side):
    """
    Fill `Q_hot_side` and `Q_cold_side` with each stream's duty above and
    below its pinch temperature, ``[kind, duty]`` with kind 'heat' (cold
    streams) or 'cool' (hot streams) and duties below 0.01 kJ/hr set to 0,
    from the stream's `pinch_state` at ``pinch_T_arr[index]`` (e.g. from
    `temperature_interval_pinch_analysis`). A standalone analysis helper,
    like `pinch_state`: `synthesize_network` does not use it.
    """
    for index in indices:
        H_in = streams[index].H
        H_out = streams_quenched[index].H
        H_pinch = pinch_state(streams[index], streams_quenched[index],
                              pinch_T_arr[index]).H
        if not is_cold(index):
            dH1 = H_in - H_pinch
            dH2 = H_pinch - H_out
            if abs(dH1)<0.01: dH1 = 0
            if abs(dH2)<0.01: dH2 = 0
            Q_hot_side[index] = ['cool', dH1]
            Q_cold_side[index] = ['cool', dH2]
        else:
            dH1 = H_out - H_pinch
            dH2 = H_pinch - H_in
            if abs(dH1)<0.01: dH1 = 0
            if abs(dH2)<0.01: dH2 = 0
            Q_hot_side[index] = ['heat', dH1]
            Q_cold_side[index] = ['heat', dH2]


# %% Network synthesis

#: Exact-state approach acceptance [K]. Synthesized process exchangers get
#: ``HXprocess(dT=T_min_app - _APPROACH_TOL)`` as a guard only: the planner
#: and the exactness check enforce `T_min_app` on the exact states.
_APPROACH_TOL = 1e-6
#: A simulated process-exchanger duty that differs from the planned one by
#: more than this times the two streams' total duties is a deviation.
_DUTY_TOL = 1e-6
#: Status 'mer' needs the utilities of the realized network (from the
#: simulated duties) within this times the total stream duty of the targets:
#: the accuracy of the enthalpy flashes that realize the plan (the plan
#: itself reaches the targets within `hensmith._planner._MER_TOL`).
_ACHIEVED_TOL = 1e-6
#: Rounds of exact-state verification and local knot refinement.
_MAX_REFINE = 3

def _grid_knots(curves, grid):
    """
    Knots ``(T, H - H_lo)`` of every stream on the problem-table grid, the
    planner's model of the stream: at every grid temperature inside the
    stream's own range, its left and right enthalpy limits there (two knots
    where the curve has a flat). Every breakpoint of every curve is a grid
    point and the table is linear between grid points, so the planner's
    cascade on these knots is exactly the table's. Enthalpies are relative
    to the stream's H_lo, so that large absolute enthalpies cost the planner
    no precision.
    """
    Ts, shift, Hl, Hr = grid['Ts'], grid['shift'], grid['Hl'], grid['Hr']
    knots = []
    for j, curve in enumerate(curves):
        k = np.arange(grid['k_lo'][j], grid['k_hi'][j] - 1, -1) # ascending T
        T = np.repeat(Ts[k] + shift[j], 2)
        H = np.column_stack((Hl[j, k], Hr[j, k])).ravel() - curve.H_lo
        keep = np.ones(T.size, dtype=bool)
        keep[1::2] = Hr[j, k] != Hl[j, k]
        knots.append((T[keep], H[keep]))
    return knots

def _knot_T(knots, H, hot):
    """
    Temperature of a knot curve at enthalpies `H`, linear between knots; a
    vertical stretch (a clipped non-equilibrium end) counts at its lowest
    temperature for a hot stream and at its highest for a cold one, as in
    the planner.
    """
    T, Hk = knots
    rises = np.diff(Hk) > 0.
    if hot: keep = np.concatenate(([True], rises))
    else: keep = np.concatenate((rises, [True]))
    return np.interp(H, Hk[keep], T[keep])

def _curve_tol_T(curve):
    """Largest distance [K] between a stream's linearized curve (and so its
    grid knots) and its exact states."""
    if not curve.monotone: return 0.
    return max(curve.tol_T, curve.glide_error)

def _interval_min(f, a, fa, b, fb):
    """
    Minimum of a smooth function `f` on [a, b], given its end values.

    A minimum inside the interval shows at an end as a slope that points
    into it (f falls from `a`, or rises into `b`); both one-sided slopes are
    taken from a step of 1e-6 of the interval. Without such a slope the
    minimum is an end (the smooth pieces of temperature-enthalpy curves
    bend one way over a knot interval). Otherwise a scan of the interval
    brackets the dip and golden-section search narrows it to 1e-7 of the
    interval. Returns the smallest value found.
    """
    eps = 1e-6 * (b - a)
    fa_ = f(a + eps)
    fb_ = f(b - eps)
    if fa_ >= fa and fb_ >= fb: return min(fa, fb)
    xs = [a, a + eps, *np.linspace(a, b, 9)[1:-1].tolist(), b - eps, b]
    fs = [fa, fa_, *[f(x) for x in xs[2:-2]], fb_, fb]
    k = int(np.argmin(fs))
    if k in (0, len(xs) - 1): return fs[k]
    lo, x, hi, fx = xs[k - 1], xs[k], xs[k + 1], fs[k]
    g = 0.5 * (3. - 5.**0.5)  # golden section
    xtol = 1e-7 * (b - a)
    while hi - lo > xtol:
        u = x + g * (hi - x) if hi - x > x - lo else x - g * (x - lo)
        fu = f(u)
        if fu < fx:
            if u > x: lo = x
            else: hi = x
            x, fx = u, fu
        elif u > x: hi = u
        else: lo = u
    return fx

def _exchanger_approach(curves, knots, h, c, H_hot_in, H_cold_in, Q,
                        T_min_app, fh=1., fc=1.):
    """
    Exact-state check of one counter-current exchanger of duty `Q` in which
    stream `h` enters hot at `H_hot_in` and stream `c` enters cold at
    `H_cold_in` (enthalpies relative to each stream's H_lo).

    With stream splitting, `fh` and `fc` are the flow fractions of the hot
    and the cold branch in the exchanger: a branch at fraction f has its
    parent's states at f times the parent's enthalpy flow, so the duty
    moves its parent-equivalent enthalpy by ``Q / f``. The duty position q
    in [0, Q] (from the hot inlet and the cold outlet end) lies at
    ``H_hot_in - q / fh`` and ``H_cold_out - q / fc``, a knot H at
    ``(H_hot_in - H) fh`` or ``(H_cold_out - H) fc``; the states returned
    are parent-equivalent (what `_refine_knots` takes). At f = 1 every
    expression is the unsplit one, bit for bit (division and
    multiplication by 1. are exact).

    The positions are the ends and every knot and curve breakpoint inside
    the exchanger. Between two consecutive positions both knot curves are
    linear, so the planned approach (on the knots) is too, and each stream's
    exact states are within `_curve_tol_T` of its knots: an interval whose
    two ends have a planned approach of at least `T_min_app` plus that
    margin is feasible on the exact states. Every other interval is
    checked with `StreamCurve.T_exact` at its ends and midpoint and, where
    the exact approach is not linear (curved single-phase stretches and
    glides), searched for a minimum inside it (`_interval_min`).

    Returns the smallest approach found [K] (exact where evaluated) and the
    exact states ``(T_hot, H_hot, T_cold, H_cold)`` wherever the exact
    approach is below ``T_min_app - _APPROACH_TOL``. With ``T_min_app =
    inf`` every interval is checked, so the approach returned is the exact
    minimum over the exchanger and the states are all those evaluated.
    This is the only exact internal-approach check of the synthesis (the
    exchangers themselves, `HXprocess`, check their two terminals only,
    which misses an internal pinch at a phase change).
    """
    hot, cold = curves[h], curves[c]
    H_hot_out, H_cold_out = H_hot_in - Q / fh, H_cold_in + Q / fc
    qs = [0., Q]
    for H in (knots[h][1], hot.H - hot.H_lo):
        qs.extend((H_hot_in - H[(H > H_hot_out) & (H < H_hot_in)]) * fh)
    for H in (knots[c][1], cold.H - cold.H_lo):
        qs.extend((H_cold_out - H[(H > H_cold_in) & (H < H_cold_out)]) * fc)
    qs = np.unique(np.clip(qs, 0., Q))
    planned = (_knot_T(knots[h], H_hot_in - qs / fh, True)
               - _knot_T(knots[c], H_cold_out - qs / fc, False))
    near = planned < T_min_app + _curve_tol_T(hot) + _curve_tol_T(cold) + 1e-9
    limit = T_min_app - _APPROACH_TOL
    points = []
    states = {}

    def exact(q):
        if q not in states:
            qh, qc = q / fh, q / fc
            T_hot = hot.T_exact(hot.H_lo + H_hot_in - qh, 'low')
            T_cold = cold.T_exact(cold.H_lo + H_cold_out - qc, 'high')
            states[q] = dT = T_hot - T_cold
            if dT < limit:
                points.append((T_hot, H_hot_in - qh, T_cold, H_cold_out - qc))
        return states[q]

    worst = float(planned[~near].min()) if not near.all() else np.inf
    if qs.size == 1:
        if near[0]: worst = min(worst, exact(qs[0]))
        return worst, points
    for k in np.flatnonzero(near[:-1] | near[1:]):
        a, b = qs[k], qs[k + 1]
        fa, fm, fb = exact(a), exact(0.5 * (a + b)), exact(b)
        worst = min(worst, fa, fm, fb)
        if abs(fa + fb - 2. * fm) > 1e-9: # not linear: look for a dip
            worst = min(worst, _interval_min(exact, a, fa, b, fb))
    return worst, points

def _exact_approach(plan, duties, ends, curves, knots, T_min_app):
    """
    `_exchanger_approach` of every exchanger in `duties` (duty by index into
    ``plan.exchangers``) at the enthalpies of the walk `ends` (see `_walk`),
    with the flow fractions of its branches (1 on a trunk).
    Returns the smallest approach [K], the violating exact states by stream,
    ``{stream: [(T_exact, H - H_lo), ...]}``, and the violating exchangers.
    """
    worst = np.inf
    violations = {}
    bad = []
    for n, Q in duties.items():
        e = plan.exchangers[n]
        h, c = e.hot, e.cold
        approach, points = _exchanger_approach(
            curves, knots, h, c, ends[n, h][0], ends[n, c][0], Q, T_min_app,
            e.hot_frac, e.cold_frac
        )
        worst = min(worst, approach)
        if points: bad.append(n)
        for T_hot, H_hot, T_cold, H_cold in points:
            violations.setdefault(h, []).append((T_hot, H_hot))
            violations.setdefault(c, []).append((T_cold, H_cold))
    return worst, violations, bad

def _shrink(curves, knots, h, c, H_hot_in, H_cold_in, Q, T_min_app,
            fh=1., fc=1.):
    """
    Largest duty ``Q' <= Q`` (to 1e-9 of Q) at which the exchanger of
    `_exchanger_approach` (branches at flow fractions `fh` and `fc`) keeps
    ``T_min_app - _APPROACH_TOL`` on the exact states. With both inlets
    fixed, a smaller duty lowers the cold stream's enthalpy (so its
    temperature) at every position and shortens the exchanger, so the
    approach can only grow: the feasible duties form an interval [0, Q']
    and bisection finds its end. On branches too: duty position q lies at
    ``H_cold_in + (Q - q) / fc`` on the cold branch, which falls with Q,
    and at ``H_hot_in - q / fh`` on the hot one, whatever Q.
    """
    def ok(x):
        return not _exchanger_approach(curves, knots, h, c, H_hot_in,
                                       H_cold_in, x, T_min_app, fh, fc)[1]
    lo, hi = 0., Q
    # a first guess from the local heat capacity flow rates saves most of
    # the bisection: the violations are within the chord error of the knots
    approach, _ = _exchanger_approach(curves, knots, h, c, H_hot_in,
                                      H_cold_in, Q, T_min_app, fh, fc)
    CP = 0. # of the branches: f times the parent's
    for j, f in ((h, fh), (c, fc)):
        T, H = knots[j]
        dT = np.diff(T)
        dH = np.diff(H)
        slopes = dH[dT > 0.] / dT[dT > 0.]
        if slopes.size: CP = max(CP, f * float(slopes.max()))
    guess = Q - 2. * (T_min_app - approach) * CP
    if 0. < guess < Q and ok(guess): lo = guess
    while hi - lo > 1e-9 * Q:
        mid = 0.5 * (lo + hi)
        if ok(mid): lo = mid
        else: hi = mid
    return lo

def _repair(plan, duties, knots, is_hot, curves, T_min_app):
    """
    Shrink every exchanger that falls short of ``T_min_app - _APPROACH_TOL``
    on the exact states to its largest feasible duty (`_shrink`); the rest of
    its duty goes to the utilities. Shrinking a match moves the later stages
    of both its streams toward their inlets, which never reduces another
    exchanger's approach (the curves are monotone), so one pass suffices;
    the loop only guards against rounding. A branch exchanger shrinks with
    its flow fractions; its later branch stages and the mix (at the split
    enthalpy -/+ the surviving branch duties, see `_walk`) move toward the
    inlet too. Returns the new duties and the changes, ``[(n, Q_before,
    Q_after)]``.
    """
    duties = dict(duties)
    changes = []
    for _ in range(len(duties) + 1):
        ends = _walk(plan, duties, knots, is_hot)[0]
        bad = _exact_approach(plan, duties, ends, curves, knots, T_min_app)[2]
        if not bad: break
        for n in bad:
            e = plan.exchangers[n]
            ends = _walk(plan, duties, knots, is_hot)[0]
            Q = _shrink(curves, knots, e.hot, e.cold, ends[n, e.hot][0],
                        ends[n, e.cold][0], duties[n], T_min_app,
                        e.hot_frac, e.cold_frac)
            changes.append((n, duties[n], Q))
            duties[n] = Q
    return duties, changes

def _refine_knots(knots, violations):
    """
    Return `knots` with the exact states of `violations` inserted (or, at an
    existing knot, i.e. a chord point inside a glide, corrected), clamped
    between the neighbouring knot temperatures so that every curve stays
    monotone.
    """
    knots = list(knots)
    for j, points in violations.items():
        T, H = knots[j]
        T, H = T.tolist(), H.tolist()
        for T_new, H_new in sorted(points, key=lambda p: p[1]):
            i = int(np.searchsorted(H, H_new))
            if i < len(H) and H[i] == H_new:
                lo = T[i - 1] if i > 0 else T_new
                hi = T[i + 1] if i + 1 < len(T) else T_new
                T[i] = min(max(T_new, lo), hi)
            else:
                lo = T[i - 1] if i > 0 else T_new
                hi = T[i] if i < len(T) else T_new
                T.insert(i, min(max(T_new, lo), hi))
                H.insert(i, H_new)
        knots[j] = (np.array(T), np.array(H))
    return knots

def _enthalpy_limit(curve, s_in, H, hot):
    """
    `H` if `HXprocess` can take it as the enthalpy limit of the stream of
    `curve` entering an exchanger in state `s_in`, else None (the other
    stream's limit then sets the duty).

    `HXprocess` flashes the stream to the limit and rejects an equilibrium
    state on the wrong side of the inlet temperature (a heated stream
    colder than its inlet). On a monotone curve that happens only strictly
    inside a non-equilibrium end jump (see `StreamCurve.jumps`). A
    point-load stream's (non-monotone curve's) equilibrium states are not
    ordered with its real inlet temperature: e.g. a liquid fed above its
    bubble point and boiled to its dew point, colder than its feed, has
    every state past its real inlet on the wrong side, its outlet included.
    It enters its first exchanger at equilibrium at its inlet enthalpy
    instead (see `_first_inlet`), from which its states are ordered, unless
    that flash failed. So the equilibrium state at `H` is compared with the
    inlet directly.
    """
    if curve.monotone:
        tol = curve.tol_H
        inside = any(H_a + tol < H < H_b - tol for H_a, H_b in curve.jumps)
        return None if inside else H
    try:
        T = curve.state_at_H(H).T
    except Exception:
        return None
    past = T <= s_in.T + _T_SIDE if hot else T >= s_in.T - _T_SIDE
    return H if past else None

def _first_inlet(stream, point_load, T_point, hot):
    """
    Bring `stream`, a copy of a process stream's real inlet, in place to
    the state in which the stream enters its first process exchanger: the
    real inlet, except that a point-load stream (non-monotone
    `StreamCurve`, whose whole duty is planned at its outlet temperature
    `T_point`; `hot` if it is cooled) enters at equilibrium at its inlet
    enthalpy and pressure, which lies on the plan's side of `T_point`
    (see `hensmith._curves._point_load_inlet`; kept as given if that flash
    fails). The enthalpy is the same either way, so no balance changes.

    `HXprocess` judges a match by the inlet temperatures (the hotter inlet
    is the hot stream, no heat moves unless they are more than `dT` apart,
    and the partner's outlet is capped at the inlet temperature -/+ `dT`),
    so a point-load stream's real inlet, on the wrong side of `T_point` by
    definition, would make it refuse or cut short a match that the plan
    keeps `T_min_app` for at `T_point`. E.g. a reboiler fed as a liquid
    above its boiling point enters as the vapor-liquid mixture it flashes
    to, and a vapor fed below its dew point (e.g. the ideal-thermo copy of
    a saturated vapor whose ideal dew point is higher) as the mixture it
    partially condenses to, hotter than its feed.
    """
    if point_load: stream.copy_like(_point_load_inlet(stream, T_point, hot))

class _RealizationError(Exception):
    """An exchanger of the plan could not be simulated."""
    def __init__(self, n, ID, error):
        super().__init__(n, ID, error)
        self.n, self.ID, self.error = n, ID, error

def _walk(plan, duties, knots, is_hot):
    """
    Enthalpies (relative to each stream's H_lo) at which every stream enters
    and leaves each of its exchangers in `duties` (duty by index into
    ``plan.exchangers``), walking it in flow order from its inlet
    (``plan.stages``), and at which it enters its utility; plus the pair
    index of every exchanger (its rank among the exchangers of the same
    (side, hot, cold) pair, in the order the hot stream meets them). A
    smaller duty (a dropped or shrunk match) shifts the later stages of both
    streams toward their inlets.

    With stream splitting (``plan.splits``), every stream is walked along
    ``plan.paths``: a split's branches start at the split enthalpy and a
    branch exchanger of flow fraction f moves the branch by its duty over
    f, so its `ends` are parent-equivalent (the enthalpies of the full flow
    in the branch's state); the stream re-joins at the split enthalpy -/+
    the sum of the surviving branch duties, which the fractions leave free
    of round-off. A dropped branch exchanger shortens its branch, a branch
    without any bypasses, and later stages move toward the inlet by the
    duty lost (`_split_nodes` gives the split enthalpies).
    """
    ends = {}
    last = []
    if not plan.splits:
        for j, hot in enumerate(is_hot):
            H = knots[j][1][-1] if hot else 0.
            for n in plan.stages[j]:
                if n not in duties: continue
                Q = duties[n]
                H_next = H - Q if hot else H + Q
                ends[n, j] = (H, H_next)
                H = H_next
            last.append(H)
    else:
        for j, hot in enumerate(is_hot):
            H = knots[j][1][-1] if hot else 0.
            for item in plan.paths[j]:
                if isinstance(item, Split):
                    H = _walk_split(item, duties, ends, H, hot)
                    continue
                if item not in duties: continue
                Q = duties[item]
                H_next = H - Q if hot else H + Q
                ends[item, j] = (H, H_next)
                H = H_next
            last.append(H)
    count = {}
    pair_index = {}
    for j, hot in enumerate(is_hot):
        if not hot: continue
        for n in plan.stages[j]:
            if n not in duties: continue
            e = plan.exchangers[n]
            key = (e.side, e.hot, e.cold)
            count[key] = pair_index[n] = count.get(key, 0) + 1
    return ends, last, pair_index

def _branch_duty(split, duties):
    """Sum (exactly rounded) of the duties of `split`'s branch exchangers
    in `duties`."""
    return math.fsum([duties[n] for ns in split.branches for n in ns
                      if n in duties])

def _walk_split(split, duties, ends, H, hot):
    """
    `_walk` over `split` (a `hensmith._splitting.Split` of a stream that is
    cooled if `hot`) from the split enthalpy `H`: fill the `ends` of its
    branch exchangers in `duties`, each branch from `H` by duty over
    fraction, and return the enthalpy where the branches re-join.
    """
    j = split.stream
    for f, ns in zip(split.fractions, split.branches):
        Hb = H
        for n in ns:
            if n not in duties: continue
            Q = duties[n] / f
            H_next = Hb - Q if hot else Hb + Q
            ends[n, j] = (Hb, H_next)
            Hb = H_next
    D = _branch_duty(split, duties)
    return H - D if hot else H + D

def _split_nodes(plan, duties, ends):
    """
    Where the streams of a plan with splits (``plan.splits``) split and
    re-join, for the exchangers in `duties` and their walk `ends` (see
    `_walk`, whose arithmetic this repeats).

    Returns
    -------
    first_nodes : dict[int, tuple[int]]
        For every stream, the first exchanger of every branch of the split
        that is its first node (the first item of ``plan.paths[j]`` with an
        exchanger in `duties`), in branch order: they all take the stream's
        real inlet (`_realize`). Empty if the first node is not a split.
    split_ends : list[tuple or None]
        For every split of ``plan.splits``, ``(H_split, [H_end, ...],
        H_mix)``: the enthalpies (relative to the stream's H_lo) where it
        splits, where each branch ends (a parent-equivalent enthalpy;
        ``H_split`` for a branch without exchangers, which bypasses) and
        where the branches re-join. None for a split whose branches have
        no exchanger left: it is not realized, the stream passes it
        unchanged.
    """
    split_ends = []
    for split in plan.splits:
        j = split.stream
        live = [[n for n in ns if n in duties] for ns in split.branches]
        n = next((ns[0] for ns in live if ns), None)
        if n is None:
            split_ends.append(None)
            continue
        H = ends[n, j][0]
        D = _branch_duty(split, duties)
        H_mix = H - D if plan.exchangers[n].hot == j else H + D
        split_ends.append((H, [ends[ns[-1], j][1] if ns else H
                               for ns in live], H_mix))
    first_nodes = {}
    for j, path in plan.paths.items():
        first_nodes[j] = ()
        for item in path:
            if not isinstance(item, Split):
                if item in duties: break
                continue
            firsts = [next((n for n in ns if n in duties), None)
                      for ns in item.branches]
            firsts = tuple(n for n in firsts if n is not None)
            if firsts:
                first_nodes[j] = firsts
                break
    return first_nodes, split_ends

def _discard(units):
    """Remove units, and the streams connected to them, from the registry of
    the active flowsheet (after a failed realization)."""
    for unit in units:
        for s in (*unit.ins, *unit.outs): bst.main_flowsheet.stream.discard(s)
        bst.main_flowsheet.unit.discard(unit)

def _realize(plan, duties, curves, knots, streams_inlet, is_hot, T_min_app):
    """
    Build and run one plain `HXprocess` per exchanger in `duties` (duty by
    index into ``plan.exchangers``), in plan order. Raise
    `_RealizationError` (after discarding the units built so far) if one of
    them cannot be simulated. Returns ``(units, first, last)``: the units by
    exchanger index, each stream's first exchanger (None if it has none)
    and the enthalpy at which it enters its utility (relative to its H_lo).

    A branch exchanger of a split stream (flow fraction f) runs f of the
    stream's flow in the states of the full stream: its inlet is the full
    stream's state at the branch's (parent-equivalent) inlet enthalpy,
    scaled by f, and its enthalpy limit f times the full stream's at the
    planned outlet (`_enthalpy_limit` judges the full-flow state). The
    first exchanger of every branch of a split at a stream's inlet takes
    the real inlet (`_split_nodes`).
    """
    ends, last, pair_index = _walk(plan, duties, knots, is_hot)
    first = [next((n for n in plan.stages[j] if n in duties), None)
             for j in range(len(is_hot))]
    first_nodes = _split_nodes(plan, duties, ends)[0] if plan.splits else {}
    units = {}
    dT = T_min_app - _APPROACH_TOL
    for n in sorted(duties):
        e = plan.exchangers[n]
        h, c = e.hot, e.cold
        suffix = '' if pair_index[n] == 1 else f'_{pair_index[n]}'
        if e.side == 'above':
            ID = f'HX_{c}_{h}_hs{suffix}'
            ports = (c, h)
            fractions = (e.cold_frac, e.hot_frac)
        else:
            ID = f'HX_{h}_{c}_cs{suffix}'
            ports = (h, c)
            fractions = (e.hot_frac, e.cold_frac)
        ins, outs, H_lims = [], [], []
        for j, f in zip(ports, fractions):
            curve = curves[j]
            H_in, H_out = ends[n, j]
            if first[j] == n or n in first_nodes.get(j, ()):
                s = _copy(streams_inlet[j]) # the real inlet state
                _first_inlet(s, not curve.monotone, curve.T_out, is_hot[j])
            else:
                s = curve.state_at_H(curve.H_lo + H_in)
            s.ID = f's_{j}__{ID}'
            # a planned outlet whose equilibrium state is not past the inlet
            # (inside a non-equilibrium end jump, or a point load's): leave
            # it to the other stream's limit
            H_lim = _enthalpy_limit(curve, s, curve.H_lo + H_out, is_hot[j])
            if f != 1.: # a branch: f of the flow, in the same states
                s.scale(f)
                if H_lim is not None: H_lim *= f
            ins.append(s)
            outs.append(s.copy(f'{ID}__s_{j}'))
            H_lims.append(H_lim)
        hx = bst.HXprocess(ID=ID, ins=ins, outs=outs, H_lim0=H_lims[0],
                           H_lim1=H_lims[1], dT=dT, thermo=ins[0].thermo)
        units[n] = hx
        try:
            hx._run()
        except Exception as error:
            _discard(units.values())
            raise _RealizationError(n, ID, error)
    return units, first, last

def synthesize_network(hus, T_min_app=5., Qmin=1e-3, force_ideal_thermo=False,
                       avoid_recycle=False, sort_hus_by_T=False, info=None):
    """
    Synthesize a heat exchanger network without stream splits for the
    process streams behind a set of utility heat exchangers: pinch analysis
    (`problem_table`), then a pinch-outward plan that reaches the minimum
    energy requirement (MER) targets whenever the search finds an unsplit
    network that does, realized with one `HXprocess` per match and one
    rigorous `HXutility` per stream.

    Parameters
    ----------
    hus : list[HeatUtility]
        One heat utility per process stream; `hu.unit` is the original heat
        exchanger (its `ins[0]`/`outs[0]` are the stream's end states) and
        the sign of `hu.duty` marks the stream: positive = heated (cold
        stream), negative = cooled (hot stream); zero-duty utilities are
        dropped. Heating utilities are placed before cooling utilities;
        within each group the given order is kept unless `sort_hus_by_T`.
        All returned per-stream arrays and lists are indexed in that
        rearranged order (the stream index), which also breaks ties in the
        planner's search.
    T_min_app : float, optional
        Minimum approach temperature [K]: kept on the exact stream states
        at both ends of and everywhere inside every process exchanger, and
        used to shift hot streams in the problem table. Defaults to 5.
    Qmin : float, optional
        Planned exchangers with a duty below this [kJ/hr] are dropped and
        their duty left to the utilities (a large value can cost MER).
        Defaults to 1e-3.
    force_ideal_thermo : bool, optional
        Analyze copies of the streams with ideal thermodynamics
        (``thermo.ideal()``); the synthesized exchangers inherit that
        thermo. Defaults to False.
    avoid_recycle : bool, optional
        Never match the same (hot, cold) pair twice anywhere, so no two
        exchangers connect the same pair of streams (a second exchanger
        between them can form a recycle loop in the network). This
        forbids the repeated matches that some unsplit MER networks need.
        Defaults to False.
    sort_hus_by_T : bool, optional
        Sort the heating utilities by inlet temperature, descending, and the
        cooling utilities ascending, before analysis. Defaults to False.
    info : dict, optional
        If given, filled with the synthesis report: 'status' ('mer' if the
        realized network's utilities equal the targets, else
        'best_effort'), 'Q_hot_target' and 'Q_cold_target' (the problem
        table's targets), 'Q_hot_plan' and 'Q_cold_plan' (the planned
        utilities), 'Q_hot' and 'Q_cold' (the utilities of the realized
        network, from the simulated exchanger duties), 'penalty'
        (``Q_hot_plan - Q_hot_target``), 'sides' (per side of the pinch:
        status, method, work, proof of a needed split, gaps, units),
        'plan_targets' (the planner's own cascade in the first round:
        Q_hot, Q_cold, pinch_T, cut), 'refine_rounds', 'min_approach' (the
        smallest approach inside any process exchanger [K]; exact on the
        stream states wherever it is within the curves' linearization
        tolerance of `T_min_app`, else from the knots),
        'deviations' (exchangers whose simulated duty differs from the
        plan), 'qmin_dropped' (matches dropped by `Qmin`), 'repaired'
        (matches shrunk to keep `T_min_app` on the exact states, see
        Notes), 'dropped' (matches that could not be simulated;
        normally empty) and 'point_loads' (the indices of the streams
        whose outlet temperature does not move with their duty, e.g. an
        isothermal condenser or a reboiler fed as a liquid above its
        boiling point, so that their whole duty is a point load at the
        outlet temperature; each enters its first process exchanger at
        equilibrium at its inlet enthalpy, see Notes).

    Returns
    -------
    HXs_hot_side : list[HXprocess]
        Process exchangers of the hot-side (above-pinch) design, in plan
        order (from the pinch outward), IDs ``HX_<cold>_<hot>_hs``;
        ``ins``/``outs`` [0] is the cold stream and [1] the hot stream.
    HXs_cold_side : list[HXprocess]
        Process exchangers of the cold-side (below-pinch) design, in plan
        order, IDs ``HX_<hot>_<cold>_cs``; ``ins``/``outs`` [0] is the hot
        stream and [1] the cold stream. The n-th exchanger (n >= 2) of the
        same pair on the same side, counted in the order the hot stream
        meets them, gets the suffix ``_<n>`` (e.g. ``HX_3_2_cs_2``).
    new_HX_utils : list[HXutility]
        One rigorous utility exchanger per stream (possibly of zero duty)
        bringing it from its last process exchanger to its outlet enthalpy,
        IDs ``Util_<index>_cs`` (hot streams) / ``Util_<index>_hs`` (cold
        streams); listed hot streams first.
    hxs : list[Unit]
        The original heat exchangers, in stream order.
    T_in_arr, T_out_arr : numpy.ndarray
        Inlet and (quenched) outlet temperatures of each stream [K].
    pinch_T_arr : numpy.ndarray
        Per-stream pinch temperature [K] (informational): the process pinch
        on the stream's own scale when the stream crosses it
        (`ProblemTable.pinch_T` for a cold stream, that plus `T_min_app` for
        a hot one); the inlet temperature of a stream whose inlet already
        lies past the pinch in its direction of flow, or that is isothermal
        or non-monotone; the outlet temperature of a stream that ends
        before reaching the pinch.
    C_flow_vector : numpy.ndarray
        Heat capacity flow rate of each process stream, ``|H_out - H_in| /
        |T_in - T_out|`` [kJ/hr/K] from its inlet and quenched outlet (the
        temperature difference is replaced by 1e-12 for an isothermal
        stream, which therefore ranks as a very large flow rate).
    hx_utils_rearranged : list[HeatUtility]
        The heat utilities of `hus` in stream order.
    streams_inlet : list[Stream]
        One copy of each stream's inlet, in stream order, as prepared for
        the analysis (ideal-thermo copies if `force_ideal_thermo`). The
        network works on further copies, so these keep their inlet state.
    stream_HXs_dict : dict[int, list[Unit]]
        For each stream index, its process exchangers in flow order, then
        its utility exchanger.
    hot_indices, cold_indices : list[int]
        Stream indices of the hot and cold streams.

    Notes
    -----
    *Curves and targets.* Every stream's outlet is quenched to equilibrium
    at its own enthalpy and described by a piecewise-linear temperature-
    enthalpy curve (see `problem_table`), built once. The problem table
    [Kemp07]_ on the union of all breakpoints gives the targets, the pinch
    and the side of the pinch that point loads at the pinch temperature
    belong to. The planner models each stream by its knots on that grid,
    which reproduces the table's cascade exactly.

    *Planner.* Each stream is cut at the pinch; above it the hot streams,
    below it the cold streams must be served completely by process matches
    ("musts"), while the partners ("flexes") leave any remainder to a
    utility at their far end. A depth-first search builds each side from
    the pinch outward, one match at a time. Every match keeps `T_min_app`
    at every knot (so internal pinches, e.g. a condensing vapor against a
    boiling mixture, are respected), and every step keeps the problem
    table of the remaining problem feasible (remaining problem analysis
    [Smith05]_, as a closed-form bound on the duty). At every level where
    that table is tight, the pinch design rules [LH83]_ (see also
    [Seider17]_, Chapter 9; number and heat-capacity-flow rules,
    generalized to isothermal segments, which can serve several partners
    in series) must hold; at the pinch itself a violation proves that MER
    needs stream splitting.
    Candidate duties are the largest feasible one and a finite set of
    events (a stream ticked off, a partner saved for another stream, a
    switch of partner, a return). The same pair may be matched repeatedly,
    which emulates a split by series alternation. Budgets are counted in
    deterministic work units, so results do not depend on machine speed.
    A branch and bound then reduces the number of exchangers. A side that
    is proven to need splits, or whose search runs out of budget, gets a
    best-effort plan: heat a must cannot place is moved to its pinch end,
    where it crosses the pinch at the cost of an equal amount of extra hot
    and cold utility (the penalty), minimized by greedy dives and a
    bisection of these gaps. See `hensmith._planner` for the details.

    *Realization.* Each stream is walked in flow order from its inlet: a
    hot stream through its hot-side matches from its inlet end, then its
    cold-side matches, then its cooler; a cold stream through its
    cold-side matches, then its hot-side matches, then its heater.
    Each match becomes a plain `HXprocess` whose two enthalpy limits
    (`H_lim0`, `H_lim1`) are the planned outlet enthalpies, so that its
    duty reproduces the plan; its `dT` is ``T_min_app - 1e-6`` K, only a
    guard against rounding (the approach is enforced by the plan and the
    check below). A planned outlet whose equilibrium state is not past the
    stream's state at the exchanger inlet cannot be a limit (`HXprocess`
    rejects it): one strictly inside a non-equilibrium end jump (see
    `StreamCurve.jumps`), or on a point-load stream colder (hotter) than
    its inlet state when heated (cooled). That stream's limit is left out
    and the other stream's sets the duty; a match in which neither stream
    can take a limit runs to its `dT` guard (a deviation if its duty
    differs). A stream's first exchanger gets its real inlet, except a
    point-load stream (whose outlet temperature does not move with its
    duty, so the plan places its whole duty there, a temperature its real
    inlet lies beyond): it enters at equilibrium at its inlet enthalpy,
    which lies on the plan's side of its outlet temperature (a reboiler
    fed as a liquid above its boiling point enters as the mixture it
    flashes to; a vapor fed below its dew point, e.g. under
    `force_ideal_thermo`, as the mixture it partially condenses to),
    because `HXprocess` compares the inlet temperatures with `dT` and
    would refuse or cut short a match planned there. Later exchangers get
    the stream's exact state at the planned enthalpy.
    Every exchanger is simulated once; one whose duty differs from the
    plan is reported in `info`. Each stream ends in one rigorous
    `HXutility` to its outlet enthalpy; an `AssertionError` is raised if
    it does not reproduce the quenched outlet within tolerance.

    *Exactness.* The knots are exact at grid points but chords in between
    (at most 0.002 K off inside glides and curved single-phase stretches).
    Every planned exchanger is therefore checked on the exact stream states
    wherever its planned approach is within that margin of `T_min_app`.
    Where a MER plan falls short by more than 1e-6 K, the exact states are
    inserted as knots and the network is planned again (at most three
    rounds). A best-effort plan (or a MER plan still short after the last
    round) instead has each violating match shrunk to the largest duty that
    keeps the approach, the rest going to the utilities: with its inlets
    fixed a smaller duty can only raise a match's approach, and the later
    stages of both streams move toward their inlets, which never reduces
    another match's approach. Constant heat capacity streams never need
    either step.

    *Guarantees and limits.* The utilities are never below the targets.
    Every process exchanger keeps ``T_min_app - 1e-6`` K on the exact
    states at its ends and at every checked position inside it, and the
    heat balance closes on every stream. 'mer' is reported only if the
    realized network reaches the targets. The result is deterministic.
    Completeness is empirical, not proven: every pruning test is a
    necessary condition, so a missed MER network can only come from the
    finite set of candidate duties, the caps on repeated pairs or the work
    budgets; the planner reached MER on every unsplit-feasible problem of a
    certified benchmark of about 1,700 problems with 2-40 streams. Networks
    whose match order is cyclic cannot be represented. An unsplit MER
    network can need many exchangers (series alternation approaches a
    split only in the limit); MER always takes precedence over the number
    of units. Problems that need stream splits get a best-effort network
    whose penalty is small but not minimal in general. A side that needs
    splits without a pinch-rule proof spends its whole MER budget before
    the best-effort step. Thermosteam's TP flashes fail silently inside
    the glides of some mixtures (e.g. water-ethanol with 20-50 % ethanol);
    an exchanger simulated there can deviate from its plan (reported in
    `info['deviations']`).

    `HeatExchangerNetwork` calls this function, rewires each stream's
    stages in series, converges the network as a `System` and costs it.

    Examples
    --------
    Problem r002: one hot stream against three cold ones (heat capacity
    flow rates in kW/K, temperatures 300 K above those of the classic
    problem, T_min_app = 10 K). Its only unsplit MER network matches the
    hot stream twice with the same cold stream. A constant heat capacity
    pseudo-component makes 1000 kmol/hr of fluid per kW/K:

    >>> import biosteam as bst, thermosteam as tmo
    >>> from hensmith.hxn_synthesis import synthesize_network
    >>> Fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1.,
    ...                      Cn=3.6, default=True)
    >>> bst.settings.set_thermo([Fluid], cache=True)
    >>> def process_stream(ID, T_in, T_out, CP):
    ...     inlet = bst.Stream(ID + '_in', Fluid=1000. * CP, T=T_in,
    ...                        units='kmol/hr')
    ...     hx = bst.HXutility(ID, ins=inlet, T=T_out, rigorous=False)
    ...     hx.simulate()
    ...     return hx
    >>> units = [process_stream('C1', 440., 470., 1.),
    ...          process_stream('C2', 400., 420., 1.),
    ...          process_stream('C3', 420., 550., 2.),
    ...          process_stream('H1', 520., 350., 4.)]
    >>> hus = [hx.heat_utilities[0] for hx in units]
    >>> info = {}
    >>> result = synthesize_network(hus, T_min_app=10., info=info)
    >>> HXs_hot_side, HXs_cold_side, new_HX_utils = result[:3]
    >>> for hx in HXs_hot_side + HXs_cold_side:
    ...     print(hx.ID, round(hx.Q / 3600., 6), 'kW')
    HX_3_2_cs 160.0 kW
    HX_3_0_cs 30.0 kW
    HX_3_2_cs_2 20.0 kW
    HX_3_1_cs 20.0 kW
    >>> info['status']
    'mer'

    The utilities equal the MER targets, 80 kW of heating and 450 kW of
    cooling:

    >>> duties = [(hx.outs[0].H - hx.ins[0].H) / 3600. for hx in new_HX_utils]
    >>> round(sum(Q for Q in duties if Q > 0), 6), round(-sum(Q for Q in duties if Q < 0), 6)
    (80.0, 450.0)
    >>> round(info['Q_hot_target'] / 3600., 6), round(info['Q_cold_target'] / 3600., 6)
    (80.0, 450.0)

    References
    ----------
    .. [LH83] Linnhoff, B., & Hindmarsh, E. (1983). The pinch design method
        for heat exchanger networks. Chemical Engineering Science, 38(5),
        745-763.
    .. [Kemp07] Kemp, I. C. (2007). Pinch Analysis and Process Integration
        (2nd ed.). Butterworth-Heinemann.
    .. [Smith05] Smith, R. (2005). Chemical Process Design and Integration.
        Wiley.
    .. [Seider17] Seider, W. D., Lewin, D. R., Seader, J. D., Widagdo, S.,
        Gani, R., & Ng, M. K. (2017). Product and Process Design Principles.
        Wiley. Heat Exchanger Networks (Chapter 9).

    """
    pinch_T_arr, hot_util_load, cold_util_load, T_in_arr, T_out_arr, \
        hxs, hot_indices, cold_indices, indices, streams_inlet, \
        hx_utils_rearranged, streams_quenched, table, curves, grid = \
        _pinch_analysis(hus, T_min_app, force_ideal_thermo, sort_hus_by_T)
    N = len(hxs)
    is_hot = [False] * N
    for i in hot_indices: is_hot[i] = True
    H_in_arr = np.array([c.H_in for c in curves])
    H_out_arr = np.array([c.H_out for c in curves])
    dTs = np.abs(T_in_arr - T_out_arr)
    C_flow_vector = np.abs(H_out_arr - H_in_arr) / np.maximum(dTs, 1e-12)
    duty = np.abs(H_out_arr - H_in_arr)
    scale = float(duty.sum())
    # Plan on the grid knots and check the plan on the exact states. Where it
    # falls short, a MER plan is planned again on knots refined with the
    # exact states (at most _MAX_REFINE rounds); a best-effort plan, or a MER
    # plan still short after the last round, has its violating matches
    # shrunk locally instead (a re-plan of a best-effort side repeats its
    # whole search to recover a duty of the order of the chord error).
    knots = _grid_knots(curves, grid)
    for refine_round in range(_MAX_REFINE + 1):
        plan = plan_network(knots, is_hot, T_min_app,
                            avoid_recycle=avoid_recycle, Qmin=Qmin)
        if not refine_round: plan_targets = plan.info['cascade']
        duties = {n: e.Q for n, e in enumerate(plan.exchangers)}
        ends = _walk(plan, duties, knots, is_hot)[0]
        min_approach, violations, bad = _exact_approach(
            plan, duties, ends, curves, knots, T_min_app
        )
        if (not violations or plan.status != 'mer'
            or refine_round == _MAX_REFINE): break
        knots = _refine_knots(knots, violations)
    repaired = []
    if bad:
        duties, changes = _repair(plan, duties, knots, is_hot, curves,
                                  T_min_app)
        for n, Q_before, Q_after in changes:
            e = plan.exchangers[n]
            repaired.append(dict(side=e.side, hot=e.hot, cold=e.cold,
                                 Q_plan=Q_before, Q=Q_after))
            if Q_after < Qmin: del duties[n]
        ends = _walk(plan, duties, knots, is_hot)[0]
        min_approach = _exact_approach(plan, duties, ends, curves, knots,
                                       T_min_app)[0]
    # Realize; a match that cannot be simulated is dropped (its duty goes to
    # the utilities: removing a match never reduces another's approach).
    dropped = []
    while True:
        try:
            units, first, last = _realize(plan, duties, curves, knots,
                                          streams_inlet, is_hot, T_min_app)
        except _RealizationError as failure:
            dropped.append(dict(ID=failure.ID, Q=duties.pop(failure.n),
                                error=repr(failure.error)))
        else:
            break
    HXs_hot_side = [units[n] for n in sorted(units)
                    if plan.exchangers[n].side == 'above']
    HXs_cold_side = [units[n] for n in sorted(units)
                     if plan.exchangers[n].side == 'below']
    deviations = []
    for n, hx in units.items():
        e = plan.exchangers[n]
        if abs(hx.Q - duties[n]) > _DUTY_TOL * (duty[e.hot] + duty[e.cold]):
            deviations.append(dict(ID=hx.ID, Q_plan=duties[n], Q=hx.Q))
    # One rigorous utility per stream, hot streams first
    stream_HXs_dict = {i: [units[n] for n in plan.stages[i] if n in units]
                       for i in indices}
    new_HX_utils = []
    for i in hot_indices + cold_indices:
        hot = is_hot[i]
        curve = curves[i]
        ID = 'Util_%s_cs'%i if hot else 'Util_%s_hs'%i
        if first[i] is None:
            s = _copy(streams_inlet[i])
        else:
            s = curve.state_at_H(curve.H_lo + last[i])
        s.ID = 's_%s__%s'%(i, ID)
        outlet = s.copy('%s__s_%s'%(ID, i))
        new_HX_util = bst.units.HXutility(ID=ID, ins=s, outs=outlet,
                                          H=H_out_arr[i], rigorous=True,
                                          thermo=s.thermo)
        new_HX_util._run()
        s_out = new_HX_util.outs[0]
        atol_T = 5. if 's' in hxs[i].outs[0].phases else 0.001
        if hot:
            np.testing.assert_allclose(s_out.H, H_out_arr[i], rtol=5e-3, atol=1.)
            np.testing.assert_allclose(s_out.T, T_out_arr[i], rtol=5e-3, atol=atol_T)
        else:
            np.testing.assert_allclose(s_out.H, H_out_arr[i], rtol=1e-2, atol=1.)
            np.testing.assert_allclose(s_out.T, T_out_arr[i], rtol=5e-2, atol=atol_T)
        new_HX_utils.append(new_HX_util)
        stream_HXs_dict[i].append(new_HX_util)
    if info is not None:
        # utilities of the plan and of the realized network: each stream's
        # duty less what its process exchangers transfer
        planned = {n: duties[n] for n in units}
        simulated = {n: hx.Q for n, hx in units.items()}
        loads = []
        for Q_of in (planned, simulated):
            Q_hot = Q_cold = 0.
            for i in indices:
                remaining = duty[i] - sum(Q_of[n] for n in plan.stages[i]
                                          if n in units)
                if is_hot[i]: Q_cold += remaining
                else: Q_hot += remaining
            loads.append((Q_hot, Q_cold))
        (Q_hot_plan, Q_cold_plan), (Q_hot, Q_cold) = loads
        tol = _ACHIEVED_TOL * scale
        mer = (plan.status == 'mer'
               and abs(Q_hot - table.hot_util_load) <= tol
               and abs(Q_cold - table.cold_util_load) <= tol)
        info.update(
            status='mer' if mer else 'best_effort',
            Q_hot_target=table.hot_util_load,
            Q_cold_target=table.cold_util_load,
            Q_hot_plan=Q_hot_plan, Q_cold_plan=Q_cold_plan,
            Q_hot=Q_hot, Q_cold=Q_cold,
            penalty=Q_hot_plan - table.hot_util_load,
            sides=plan.info['sides'],
            plan_targets=dict(Q_hot=plan_targets['Qh'],
                              Q_cold=plan_targets['Qc'],
                              pinch_T=plan_targets['pinch_T'],
                              cut=plan_targets['cut']),
            refine_rounds=refine_round,
            min_approach=min_approach if duties else None,
            deviations=deviations,
            qmin_dropped=plan.info['qmin_dropped'],
            dropped=dropped, repaired=repaired,
            point_loads=[i for i in indices if not curves[i].monotone],
        )
    return HXs_hot_side, HXs_cold_side, new_HX_utils, hxs, T_in_arr,\
           T_out_arr, pinch_T_arr, C_flow_vector, hx_utils_rearranged, streams_inlet, stream_HXs_dict,\
           hot_indices, cold_indices


# Pinch diagram

def _order_exchanger_columns(hxs, stream_life_cycles):
    """
    Order heat exchangers left to right so that every stream meets its
    exchangers in flow direction (cold streams flow left to right, hot streams
    right to left). The per-stream stage orders define a precedence graph;
    a topological sort (Kahn's algorithm, ties broken by the given order)
    yields a consistent layout. Contradictory constraints, which would need a
    stream to flow backwards, fall back to the given order.
    """
    hxs = list(hxs)
    position = {hx: i for i, hx in enumerate(hxs)}
    successors = {hx: [] for hx in hxs}
    N_predecessors = {hx: 0 for hx in hxs}
    for life_cycle in stream_life_cycles:
        stages = [i.unit for i in life_cycle.life_cycle if i.unit in position]
        if not life_cycle.cold: stages.reverse()
        for a, b in zip(stages, stages[1:]):
            if b not in successors[a]:
                successors[a].append(b)
                N_predecessors[b] += 1
    ready = [position[hx] for hx in hxs if not N_predecessors[hx]]
    heapq.heapify(ready)
    ordered = []
    while ready:
        hx = hxs[heapq.heappop(ready)]
        ordered.append(hx)
        for other in successors[hx]:
            N_predecessors[other] -= 1
            if not N_predecessors[other]: heapq.heappush(ready, position[other])
    return ordered if len(ordered) == len(hxs) else hxs

def _format_H(H):
    mantissa, exponent = f'{H:.2e}'.split('e')
    return f'{mantissa}E{int(exponent)}'

def _auxiliary_name(unit):
    """
    Return the (dotted) name of an auxiliary unit within its owner, e.g.
    'condenser' or 'evaporators[0].heat_exchanger', or None if the unit is
    not auxiliary.
    """
    owner = unit.owner
    if owner is unit: return None
    def search(parent, prefix):
        for name, aux in parent.get_auxiliary_units_with_names():
            if aux is unit: return prefix + name
            if hasattr(aux, 'get_auxiliary_units_with_names'):
                found = search(aux, prefix + name + '.')
                if found: return found
    return search(owner, '') or unit.ID.lstrip('.')

def _stream_label(hx, show_units, show_auxiliary_units, show_stream_IDs):
    """
    Label of a stream from its original heat exchanger `hx`:
    '<owner> - <auxiliary name> (<inlet stream ID>)', with each part
    optional.
    """
    parts = []
    if show_units: parts.append(hx.owner.ID)
    if show_auxiliary_units:
        auxname = _auxiliary_name(hx)
        if auxname: parts.append(auxname)
    label = ' - '.join(parts)
    if show_stream_IDs:
        ID = hx.ins[0].ID
        if ID: label = f'{label} ({ID})' if label else ID
    return label

def plot_pinch_diagram(stream_life_cycles, inlet_Ts, outlet_Ts,
                       hot_side_HXs, cold_side_HXs, Qmin=1e-3,
                       original_hxs=None, show_units=True,
                       show_auxiliary_units=True, show_stream_IDs=True,
                       show_legend=True, ax=None, file=None, dpi=300):
    """
    Draw a pinch diagram of a synthesized heat exchanger network: cold
    streams (blue, flowing left to right) above hot streams (red, flowing
    right to left), one vertical connector per process heat exchanger with
    its duty, a dashed pinch line separating the cold-side from the
    hot-side exchangers, and circles marking the utility exchangers that
    bring each stream to its outlet temperature.

    Parameters
    ----------
    stream_life_cycles : list[StreamLifeCycle]
        One per stream, as built by HeatExchangerNetwork.
    inlet_Ts, outlet_Ts : array-like
        Stream inlet and outlet temperatures [K], indexed like the life cycles.
    hot_side_HXs, cold_side_HXs : list[HXprocess]
        Process exchangers above and below the pinch.
    Qmin : float, optional
        Utility exchangers with a duty at or below this [kJ/hr] are not marked.
    original_hxs : list[Unit], optional
        The original heat exchanger of each stream (indexed like the life
        cycles). Required for the stream labels below.
    show_units : bool, optional
        Label each stream with the unit operation that owns its original heat
        exchanger (the main unit for auxiliary exchangers).
    show_auxiliary_units : bool, optional
        Label each stream with the name of its original heat exchanger within
        the main unit (e.g. 'condenser'), if it is an auxiliary unit.
    show_stream_IDs : bool, optional
        Label each stream with the ID of the original heat exchanger's inlet.
    show_legend : bool, optional
        Add a legend of the symbols below the diagram.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if not given.
    file : str, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution used when saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    Temperatures are shown in degC and heat flows in kJ/hr at the inlet and
    outlet of each stream. Exchanger columns on each side of the pinch are
    ordered so that each stream meets them in flow direction whenever the
    network allows it. Stream labels read '<unit> - <auxiliary> (<stream>)'
    next to the stream index at the inlet.

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
    >>> fig, ax = HXN.plot_pinch_diagram()
    >>> connectors = [i for i in ax.findobj() if (i.get_gid() or '').startswith('HX:')]
    >>> len(connectors) == len(HXN.new_HXs)
    True
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    import matplotlib.pyplot as plt
    # Artists carry stable gids ('HX:<ID>', 'Util:<ID>', 'Label:<index>')
    # so the drawing can be checked structurally in tests.
    show_labels = show_units or show_auxiliary_units or show_stream_IDs
    if show_labels and original_hxs is None:
        raise ValueError('original_hxs is required to label streams with '
                         'units, auxiliary units, or stream IDs')
    cold_color, hot_color = '#2e6db4', '#d62728'
    cold_bg, hot_bg = '#e6f0fa', '#fbe9e7'
    process_hxs = set(hot_side_HXs) | set(cold_side_HXs)
    # Stream index and stage of each side of every process exchanger, by identity
    hx_streams = {hx: {} for hx in process_hxs}
    for index, life_cycle in enumerate(stream_life_cycles):
        for stage in life_cycle.life_cycle:
            if stage.unit in hx_streams:
                hx_streams[stage.unit][life_cycle.cold] = (index, stage)
    cold_side_HXs = _order_exchanger_columns(cold_side_HXs, stream_life_cycles)
    hot_side_HXs = _order_exchanger_columns(hot_side_HXs, stream_life_cycles)
    columns = cold_side_HXs + hot_side_HXs
    N_cs = len(cold_side_HXs)
    N_columns = len(columns)
    # x layout: 0 stream ends | 1 cold utilities | 2..N_cs+1 cold side |
    # pinch | N_cs+2..N+1 hot side | N+2 hot utilities | N+3 stream ends
    x_start, x_cold_util = 0., 1.
    x_columns = {hx: 2. + i for i, hx in enumerate(columns)}
    x_pinch = N_cs + 1.5
    x_hot_util = N_columns + 2.
    x_end = N_columns + 3.
    # y layout: cold streams on top, hot streams below, duty labels in between
    cold_streams = [i for i, lc in enumerate(stream_life_cycles) if lc.cold]
    hot_streams = [i for i, lc in enumerate(stream_life_cycles) if not lc.cold]
    N_hot = len(hot_streams)
    N_cold = len(cold_streams)
    gap = 2.5
    y = {}
    for k, i in enumerate(hot_streams): y[i] = N_hot - k
    for k, i in enumerate(cold_streams): y[i] = N_hot + gap + N_cold - k
    y_label = N_hot + (gap + 1.) / 2.
    y_top = N_hot + gap + N_cold + 1.
    y_bottom = 0.
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6., 0.75 * (N_columns + 4) + 3.), 0.4 * y_top + 1.)
        )
    else:
        fig = ax.figure
    # Background and pinch line
    x_min, x_max = x_start - 1.8, x_end + 1.8
    ax.axvspan(x_min, x_pinch, color=cold_bg, lw=0, zorder=0)
    ax.axvspan(x_pinch, x_max, color=hot_bg, lw=0, zorder=0)
    ax.axvline(x_pinch, color='k', ls='--', lw=1, zorder=1)
    ax.text(x_min + 0.2, y_bottom + 0.1, 'Cold side', color=cold_color,
            weight='bold', ha='left', va='bottom')
    ax.text(x_max - 0.2, y_bottom + 0.1, 'Hot side', color=hot_color,
            weight='bold', ha='right', va='bottom')
    # Column headers
    header_kwargs = dict(ha='center', va='bottom', weight='bold', fontsize=8)
    for x_T, x_H in ((x_start - 1.3, x_start - 0.6), (x_end + 0.6, x_end + 1.3)):
        ax.text(x_T, y_top, 'T\n[°C]', **header_kwargs)
        ax.text(x_H, y_top, 'H\n[kJ·h$^{-1}$]', **header_kwargs)
    ax.text(x_start - 0.3, y_label, 'ΔH\n[kJ·h$^{-1}$]',
            ha='right', va='center', weight='bold', fontsize=8)
    # Streams
    value_kwargs = dict(ha='center', va='center', fontsize=8)
    for index, life_cycle in enumerate(stream_life_cycles):
        cold = life_cycle.cold
        color = cold_color if cold else hot_color
        yi = y[index]
        stages = life_cycle.life_cycle # never empty: each stream has a utility stage
        H_in = stages[0].H_in
        H_out = stages[-1].H_out
        T_in = inlet_Ts[index] - 273.15
        T_out = outlet_Ts[index] - 273.15
        # T is the outer column on the left and the inner column on the right
        T_left, H_left, T_right, H_right = (
            (T_in, H_in, T_out, H_out) if cold else (T_out, H_out, T_in, H_in)
        )
        x_in, x_out, sign = (x_start, x_end, 1) if cold else (x_end, x_start, -1)
        ax.annotate('', xy=(x_out, yi), xytext=(x_in, yi),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2,
                                    shrinkA=0, shrinkB=0), zorder=2)
        ax.text(x_start - 1.3, yi, f'{T_left:.1f}', color=color, **value_kwargs)
        ax.text(x_start - 0.6, yi, _format_H(H_left), color=color, **value_kwargs)
        ax.text(x_end + 0.6, yi, f'{T_right:.1f}', color=color, **value_kwargs)
        ax.text(x_end + 1.3, yi, _format_H(H_right), color=color, **value_kwargs)
        # Index and label share a baseline above the stream, clear of the
        # exchanger circles
        y_text = yi + 0.25
        ax.text(x_in + sign * 0.3, y_text, str(index), color=color,
                ha='center', va='baseline', weight='bold', fontsize=9)
        if show_labels:
            label = _stream_label(original_hxs[index], show_units,
                                  show_auxiliary_units, show_stream_IDs)
            # the smaller label reads as centered with the index when its
            # baseline is slightly higher
            ax.text(x_in + sign * 0.6, y_text + 0.08, label, color=color,
                    ha='left' if cold else 'right', va='baseline', fontsize=7,
                    zorder=6, gid=f'Label:{index}',
                    bbox=dict(boxstyle='square,pad=0.15', fc='w', ec='none'))
        # Utility exchangers: a cold stream ends in a hot utility (red), a
        # hot stream in a cold utility (blue)
        x_util, util_color = (x_hot_util, hot_color) if cold else (x_cold_util, cold_color)
        for stage in stages:
            unit = stage.unit
            if unit in process_hxs: continue
            if abs(stage.H_out - stage.H_in) <= Qmin: continue
            ax.plot([x_util], [yi], 'o', mfc='w', mec=util_color, mew=1.2,
                    ms=6, zorder=4, gid='Util:' + unit.ID)
    # Process exchangers
    for hx in columns:
        streams = hx_streams[hx]
        if len(streams) != 2:
            warn(f'{hx.ID} is not in exactly one hot and one cold stream '
                 'life cycle; it is not drawn', RuntimeWarning)
            continue
        (i_cold, stage_cold), (i_hot, stage_hot) = streams[True], streams[False]
        x = x_columns[hx]
        Q = abs(stage_hot.H_in - stage_hot.H_out)
        ax.plot([x, x], [y[i_hot], y[i_cold]], '-o', color='k', mfc='w',
                mew=1.2, ms=6, lw=1.2, zorder=3, gid='HX:' + hx.ID)
        ax.text(x, y_label, _format_H(Q), rotation=90, ha='center',
                va='center', fontsize=8, zorder=5,
                bbox=dict(boxstyle='square,pad=0.25', fc='w', ec='k', lw=0.8))
    if show_legend:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], color=cold_color, lw=1.2, marker='>', markevery=[-1],
                   ms=5, label='Cold stream'),
            Line2D([], [], color=hot_color, lw=1.2, marker='<', markevery=[0],
                   ms=5, label='Hot stream'),
            Line2D([], [], color='k', lw=1.2, marker='o', mfc='w', mew=1.2,
                   ms=6, label='Process heat exchange'),
            Line2D([], [], ls='', marker='o', mfc='w', mec=hot_color, mew=1.2,
                   ms=6, label='Hot utility'),
            Line2D([], [], ls='', marker='o', mfc='w', mec=cold_color, mew=1.2,
                   ms=6, label='Cold utility'),
            Line2D([], [], color='k', ls='--', lw=1, label='Pinch'),
        ]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.),
                  ncol=3, fontsize=7, frameon=False, handlelength=2.5,
                  columnspacing=1.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bottom, y_top + 1.2)
    ax.set_axis_off()
    if file: fig.savefig(file, dpi=dpi, bbox_inches='tight')
    return fig, ax
