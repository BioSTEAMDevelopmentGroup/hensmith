# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Tests of the minimum-energy-requirement (MER) targets: the per-stream
temperature-enthalpy curves (`hensmith._curves.StreamCurve`), the problem
table built on them, the pinch cut and split, `pinch_state`, and the exact
internal-approach check `_min_approach`.

Reference targets come from an INDEPENDENT dense-grid calculator that does
not use hensmith: every stream's enthalpy is sampled every 0.25 K with TP
flashes outside two-phase regions, a pure component's latent heat is a jump
at T_sat between V=0 and V=1 enthalpies, binary glides are traced along the
bubble-point curve of the liquid (thermosteam BubblePoint only, 201 + 101
states), and the cascade of the resulting piecewise-linear profiles is
evaluated with left and right limits at every node (scratchpad
``realthermo/dense_targets.py`` of the MER work, ``stream_profile(h=0.25,
nV=201)`` + ``cascade``; biosteam 7ff69657, thermosteam f768d38). The numbers
below were recomputed with it on exactly the builders in this module. The
calculator is accurate to ~0.2 kJ/hr on these cases; the curves are within
their documented linearization bound (0.002 K times the heat-capacity flow
rates at the pinch) of it.
"""
import functools
import numpy as np
import pytest
import biosteam as bst
import thermosteam as tmo
from numpy.testing import assert_allclose
from hensmith.hxn_synthesis import (
    problem_table, _problem_table, _pinch_cut, pinch_state,
)
from hensmith._curves import (
    StreamCurve, GLIDE_TOL_T, FLAT, SENSIBLE, GLIDE, _min_approach,
)

ATM = 101325.

# ---------------------------------------------------------------------------
# Stream builders (simulated HXutility units, as in test_hxn_regression.py)
# ---------------------------------------------------------------------------

def setup(name, chemicals=('Water', 'Ethanol')):
    bst.settings.set_thermo(list(chemicals), cache=True)
    bst.main_flowsheet.set_flowsheet('test_hxn_targets_' + name)

def utility_hx(ID, T, P, phase, T_out, rigorous=None, **flow):
    """A simulated HXutility acting as one process stream (kmol/hr flows)."""
    s = bst.Stream(ID + '_in', T=T, P=P, phase=phase, units='kmol/hr', **flow)
    if rigorous is None: rigorous = phase == 'g'
    hx = bst.HXutility(ID, ins=s, T=T_out, rigorous=rigorous)
    hx.simulate()
    return hx

def boiling_hx(ID, T, P, T_out, **flow):
    """A cold liquid heated past its bubble point (rigorous VLE)."""
    return utility_hx(ID, T, P, 'l', T_out, rigorous=True, **flow)

def vapor_fraction_hx(ID, T, P, V, **flow):
    """A liquid at T heated to vapor fraction V (V=0: bubble, V=1: dew point)."""
    s = bst.Stream(ID + '_in', T=T, P=P, phase='l', units='kmol/hr', **flow)
    hx = bst.HXutility(ID, ins=s, V=V, rigorous=True)
    hx.simulate()
    return hx

def xE(total, x):
    """Water/ethanol flows with ethanol mole fraction x."""
    return dict(Water=total * (1. - x), Ethanol=total * x)

def case_two_stream():
    """Cold water boiled to saturated vapor at 1 atm against hot liquid
    water: the pinch is the cold stream's T_sat (its latent heat at the
    pinch, cut 'above'). The old table had no T_sat on its grid and gave 0
    hot utility."""
    setup('two_stream')
    return [utility_hx('H1', 400., 5e5, 'l', 330., Water=1000.),
            vapor_fraction_hx('C1', 300., ATM, 1, Water=100.)], 5.

def case_curvature():
    """Liquids only: water cooled against ethanol, whose Cp rises ~40 %
    over the range; the composite curves touch INSIDE the stream range
    (terminal approaches 5 K, internal < 1 K), so the pinch is found only
    with breakpoints that follow the curvature. The old table gave 0."""
    setup('curvature')
    return [utility_hx('H1', 400., 10e5, 'l', 300., Water=1000.),
            utility_hx('C1', 295., 10e5, 'l', 395., Ethanol=569.1)], 5.

def case_condenser_at_pinch():
    """Ethanol desuperheated, condensed and subcooled (set B): the pinch is
    the hot stream's shifted T_sat (its latent heat at the pinch, cut
    'below'). The old table gave 0 hot utility."""
    setup('condenser_at_pinch')
    return [utility_hx('H1', 385., ATM, 'g', 320., Ethanol=400.),
            utility_hx('H2', 350., 5e5, 'l', 310., Water=300.),
            utility_hx('C1', 300., ATM, 'l', 358., Water=1500.),
            utility_hx('C2', 330., ATM, 'l', 345., Ethanol=200.)], 5.

def case_Tsat_coincidence():
    """Boiling water C1 whose T_sat is also C2's outlet (heated to its
    bubble point at the same pressure), so the saturation temperature is a
    shared grid point; the pinch is there (set J)."""
    setup('Tsat_coincidence')
    return [boiling_hx('C1', 340., ATM, 390., Water=200.),
            vapor_fraction_hx('C2', 300., ATM, 0, Water=300.),
            utility_hx('H1', 423., 5e5, 'l', 330., Water=1500.),
            utility_hx('H2', 385., 5e5, 'l', 340., Water=300.)], 5.

def case_binary_glide():
    """Water/ethanol (ethanol mole fraction 0.1, 1 atm; a composition whose
    flashes are reliable) boiled from subcooled to superheated through its
    glide (bubble 359.53 K, dew 370.44 K), against ethanol condensing at
    2 bar (T_sat 369.86 K) and hot water: the pinch is the condenser's
    shifted T_sat, INSIDE the cold stream's glide."""
    setup('binary_glide')
    return [boiling_hx('C1', 330., ATM, 380., **xE(200., 0.1)),
            utility_hx('H1', 385., 2e5, 'g', 355., Ethanol=150.),
            utility_hx('H2', 400., 5e5, 'l', 340., Water=300.)], 5.

K = 273.15
LINNHOFF = [('C1', 20., 135., 2.), ('H2', 170., 60., 3.),
            ('C3', 80., 140., 4.), ('H4', 150., 30., 1.5)]  # degC, degC, kW/K

def case_linnhoff():
    """Constant-Cp four-stream problem of Linnhoff & Hindmarsh (1983),
    T_min_app 10 K: 20 kW hot and 60 kW cold utility, pinch 90/80 degC. The
    `Fluid` pseudo-component has Cn = 3.6 kJ/kmol/K, so 1000*CP kmol/hr is a
    heat-capacity flow rate of CP kW/K."""
    Fluid = tmo.Chemical('Fluid', search_db=False, phase='l', MW=1., Cn=3.6,
                         default=True)
    bst.settings.set_thermo([Fluid], cache=True)
    bst.main_flowsheet.set_flowsheet('test_hxn_targets_linnhoff')
    units = []
    for name, T_in, T_out, CP in LINNHOFF:
        s = bst.Stream(name + '_in', Fluid=1000. * CP, T=T_in + K, P=101325.,
                       units='kmol/hr')
        hx = bst.HXutility(name, ins=s, T=T_out + K, rigorous=False)
        hx.simulate()
        units.append(hx)
    return units, 10.

# name -> builder
CASES = {
    'two_stream': case_two_stream,
    'curvature': case_curvature,
    'condenser_at_pinch': case_condenser_at_pinch,
    'Tsat_coincidence': case_Tsat_coincidence,
    'binary_glide': case_binary_glide,
    'linnhoff': case_linnhoff,
}

# name -> (hot utility, cold utility [kJ/hr], shifted pinch temperature [K])
# from the independent dense calculator (see module docstring). The old
# problem table (end temperatures only, TP flashes at grid points) missed
# the pinch of the first four (hot utility 0) and gave 24 % of the target on
# the binary glide; it was exact only on the constant-Cp problem.
DENSE = {
    'two_stream':         (2395101.84, 3088578.56, 373.124295848),
    # identical at 0.05 K spacing; minimum between grid nodes of hensmith
    'curvature':          (319444.38, 327619.16, 347.0),
    'condenser_at_pinch': (276488.61, 12523800.52, 346.570441659),
    'Tsat_coincidence':   (2923386.93, 4169525.67, 373.124295848),
    'binary_glide':       (5386912.87, 4109097.31, 364.8572114718511),
    'linnhoff':           (72000., 216000., 353.15),  # 20 / 60 kW, 80 degC
}

def prepare(units):
    """Stream copies exactly as the synthesizer prepares them (heat
    utilities sorted by duty; outlets quenched at their own enthalpy)."""
    hus = [hx.heat_utilities[0] for hx in units]
    hus.sort(key=lambda hu: hu.duty)
    names = [hu.unit.ID for hu in hus]
    streams_inlet = [hu.unit.ins[0].copy() for hu in hus]
    streams_quenched = [hu.unit.outs[0].copy() for hu in hus]
    for s in streams_quenched: s.vle(H=s.H, P=s.P)
    is_hot = [hu.duty < 0 for hu in hus]
    return names, streams_inlet, streams_quenched, is_hot

@functools.lru_cache(maxsize=None)
def analyzed(name):
    """(names, streams_inlet, streams_quenched, is_hot, T_min_app, table,
    curves, grid) of a case, built once per test session (read-only)."""
    units, T_min_app = CASES[name]()
    names, ins, outs, is_hot = prepare(units)
    table, curves, grid = _problem_table(ins, outs, is_hot, T_min_app)
    return names, ins, outs, is_hot, T_min_app, table, curves, grid

def heat_capacity_flows(curves):
    """Mean heat-capacity flow rate of each monotone stream [kJ/hr/K]."""
    return [(c.H_hi - c.H_lo) / (c.T_hi - c.T_lo) for c in curves if c.monotone]

def closed_form_table(streams_inlet, streams_quenched, is_hot, T_min_app):
    """The problem table as computed before stream curves: grid of the
    shifted end temperatures only, each monotone stream's enthalpy at every
    grid temperature inside its range (phases fixed, clipped), isothermal
    and non-monotone streams as point loads at their outlet. Exact for
    constant-Cp streams, for which the new table must be bit-identical."""
    N = len(streams_inlet)
    is_hot = np.asarray(is_hot, dtype=bool)
    sign = np.where(is_hot, 1., -1.)
    shift = np.where(is_hot, T_min_app, 0.)
    T_in = np.array([s.T for s in streams_inlet])
    T_out = np.array([s.T for s in streams_quenched])
    H_in = np.array([s.H for s in streams_inlet])
    H_out = np.array([s.H for s in streams_quenched])
    monotone = (sign * (T_in - T_out)) > 0.
    T_hi = np.where(monotone, np.maximum(T_in, T_out), T_out) - shift
    T_lo = np.where(monotone, np.minimum(T_in, T_out), T_out) - shift
    Ts = np.unique(np.concatenate([T_hi, T_lo]))[::-1]
    n = Ts.size
    interval_H = np.zeros((N, n - 1))
    point_H = np.zeros((N, n))
    for j in range(N):
        if monotone[j]:
            idx = np.flatnonzero((Ts <= T_hi[j]) & (Ts >= T_lo[j]))
            H_lo, H_hi = sorted((H_in[j], H_out[j]))
            Hs = np.empty(idx.size)
            Hs[0], Hs[-1] = H_hi, H_lo
            s = streams_inlet[j].copy()
            for m in range(1, idx.size - 1):
                s.T = Ts[idx[m]] + shift[j]
                Hs[m] = min(max(s.H, H_lo), H_hi)
            interval_H[j, idx[:-1]] = sign[j] * (Hs[:-1] - Hs[1:])
        else:
            k = np.searchsorted(-Ts, -T_hi[j])
            point_H[j, k] = sign[j] * abs(H_out[j] - H_in[j])
    point_total = point_H.sum(axis=0)
    residual = np.cumsum(point_total + np.concatenate([[0.], interval_H.sum(axis=0)]))
    flow = np.minimum(residual, residual - point_total)
    k_pinch = int(np.argmin(flow))
    if -flow[k_pinch] <= 1e-9 * np.abs(H_out - H_in).sum():
        hot, k_pinch = 0., 0
    else:
        hot = -flow[k_pinch]
    cold = residual[-1] + hot
    if cold < 0.: hot, cold = hot - cold, 0.
    return Ts, interval_H, point_H, residual, hot, cold, Ts[k_pinch]

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', list(CASES))
def test_targets_match_dense_reference(name):
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed(name)
    Q_hot, Q_cold, pinch_T = DENSE[name]
    if name == 'curvature':
        # linearization bound: every stream within GLIDE_TOL_T of its curve
        tol = GLIDE_TOL_T * sum(heat_capacity_flows(curves))
        assert abs(table.pinch_T - pinch_T) < 1.  # the minimum is flat there
    else:
        tol = max(2e-5 * Q_hot, 2.)
        assert_allclose(table.pinch_T, pinch_T, rtol=0, atol=1e-6)
    assert abs(table.hot_util_load - Q_hot) <= tol, (table.hot_util_load, Q_hot)
    assert abs(table.cold_util_load - Q_cold) <= tol, (table.cold_util_load, Q_cold)
    # the public function returns the same table
    public = problem_table(ins, outs, is_hot, dT, curves=curves)
    assert public.hot_util_load == table.hot_util_load
    assert np.array_equal(public.residual, table.residual)

def test_curvature_pinch_needs_interior_breakpoints():
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('curvature')
    # an end-temperature grid (4 points) would miss the pinch entirely
    assert table.Ts.size > 4
    assert table.hot_util_load > 0.9 * DENSE['curvature'][0]
    assert {c.method for c in curves} == {'pure, liquid'}

def test_constant_cp_identity():
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('linnhoff')
    reference = closed_form_table(ins, outs, is_hot, dT)
    for field, value in zip(table._fields, reference):
        assert np.array_equal(getattr(table, field), value), field
    assert_allclose([table.hot_util_load / 3600., table.cold_util_load / 3600.],
                    [20., 60.], rtol=1e-12)
    assert table.pinch_T == 80. + K
    assert all(c.method == 'no VLE' and c.kinds == (SENSIBLE,) for c in curves)

@pytest.mark.parametrize('name', list(CASES))
def test_table_invariants(name):
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed(name)
    sign = np.where(is_hot, 1., -1.)
    duty = np.array([c.H_hi - c.H_lo for c in curves])
    scale = duty.sum()
    # (a) every stream's contributions telescope to its duty
    per_stream = table.interval_H.sum(axis=1) + table.point_H.sum(axis=1)
    assert_allclose(per_stream, sign * duty, rtol=1e-12, atol=0)
    # (b) hot - cold utility is the net heating demand
    assert table.hot_util_load >= 0. and table.cold_util_load >= 0.
    assert abs((table.hot_util_load - table.cold_util_load)
               + (sign * duty).sum()) <= 1e-12 * scale
    # (c) contributions have the stream's sign
    assert (sign[:, None] * table.interval_H >= 0.).all()
    assert (sign[:, None] * table.point_H >= 0.).all()
    # (d) the grid enthalpies reproduce the table exactly
    Ts, Hl, Hr, shift = grid['Ts'], grid['Hl'], grid['Hr'], grid['shift']
    assert Ts is table.Ts and Hl.shape == Hr.shape == (len(curves), Ts.size)
    assert np.array_equal(table.point_H, sign[:, None] * (Hr - Hl))
    assert np.array_equal(table.interval_H, sign[:, None] * (Hl[:, :-1] - Hr[:, 1:]))
    assert (np.diff(Ts) < 0.).all()
    for j, c in enumerate(curves):
        k_hi, k_lo = grid['k_hi'][j], grid['k_lo'][j]
        assert_allclose([Ts[k_hi] + shift[j], Ts[k_lo] + shift[j]],
                        [c.T_hi, c.T_lo], rtol=0, atol=1e-8)
        assert (Hr[j, :k_hi + 1] == c.H_hi).all()
        assert (Hl[j, :k_hi] == c.H_hi).all()
        assert (Hl[j, k_lo:] == c.H_lo).all()
        assert (Hr[j, k_lo + 1:] == c.H_lo).all()
        assert (np.diff(Hl[j]) <= 0.).all() and (Hl[j] <= Hr[j]).all()

@pytest.mark.parametrize('name', list(CASES))
def test_pinch_split_reproduces_targets(name):
    """Splitting every stream at the pinch on the side `_pinch_cut` names
    puts exactly the hot utility target above the pinch and the cold one
    below it, whether the split comes from the grid, the curve or
    `pinch_state`."""
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed(name)
    cut = _pinch_cut(table)
    side = 'right' if cut == 'below' else 'left'
    k = int(np.flatnonzero(table.Ts == table.pinch_T)[0])
    above = below = 0.
    for j, (si, so, hot, c) in enumerate(zip(ins, outs, is_hot, curves)):
        H_split = (grid['Hr'] if side == 'right' else grid['Hl'])[j, k]
        T_p = table.pinch_T + (dT if hot else 0.)
        assert c.H_at(T_p, side) == H_split
        H_state = pinch_state(si, so, T_p, side=side, curve=c).H
        assert abs(H_state - H_split) <= 1e-9 * (c.H_hi - c.H_lo) + 1e-6
        sgn = -1. if hot else 1.
        above += sgn * (c.H_hi - H_split)
        below -= sgn * (H_split - c.H_lo)
    scale = sum(c.H_hi - c.H_lo for c in curves)
    assert abs(above - table.hot_util_load) <= 1e-12 * scale
    assert abs(below - table.cold_util_load) <= 1e-12 * scale

def test_pinch_cut_sides():
    # cold latent heat at the pinch: boiling needs heat from above
    for name in ('two_stream', 'Tsat_coincidence'):
        table = analyzed(name)[5]
        assert _pinch_cut(table) == 'above', name
    # hot latent heat at the pinch: condensing heat goes below
    table = analyzed('condenser_at_pinch')[5]
    assert _pinch_cut(table) == 'below'
    # the boiling stream's flat sits exactly at the pinch
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('two_stream')
    j = names.index('C1')
    k = int(np.flatnonzero(table.Ts == table.pinch_T)[0])
    satL = ins[j].copy(); satL.vle(V=0, P=ATM)
    assert_allclose(grid['Hr'][j, k] - grid['Hl'][j, k], outs[j].H - satL.H, rtol=1e-9)
    # a threshold problem has everything below its pinch at Ts[0]
    setup('threshold')
    units = [utility_hx('H1', 400., 5e5, 'l', 300., Water=1000.),
             utility_hx('C1', 300., 5e5, 'l', 390., Water=900.)]
    table = problem_table(*prepare(units)[1:], 5.)
    assert table.hot_util_load == 0. and table.pinch_T == table.Ts[0]
    assert _pinch_cut(table) == 'below'

def test_problem_table_accepts_prebuilt_curves():
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('condenser_at_pinch')
    fresh = problem_table(ins, outs, is_hot, dT)
    shared = problem_table(ins, outs, is_hot, dT, curves=curves)
    for a, b, c in zip(table, fresh, shared):
        assert np.array_equal(a, b) and np.array_equal(a, c)
    with pytest.raises(ValueError):
        problem_table(ins, outs, is_hot, dT, curves=curves[:-1])

# ---------------------------------------------------------------------------
# StreamCurve
# ---------------------------------------------------------------------------

def _interior_enthalpies(c, n_max=12):
    """A few enthalpies strictly inside the curve: segment midpoints (evenly
    chosen) and uniform points."""
    mids = 0.5 * (c.H[:-1] + c.H[1:])
    mids = mids[(c.H[1:] > c.H[:-1])]
    if mids.size > n_max: mids = mids[np.linspace(0, mids.size - 1, n_max).astype(int)]
    Hs = np.concatenate([mids, np.linspace(c.H_lo, c.H_hi, 7)[1:-1]])
    return [H for H in Hs if c.H_lo + c.tol_H < H < c.H_hi - c.tol_H]

def _segment(c, H):
    return min(int(np.searchsorted(c.H, H, 'right')) - 1, len(c.kinds) - 1)

@pytest.mark.parametrize('name', list(CASES))
def test_stream_curve_invariants_and_round_trips(name):
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed(name)
    for c in curves:
        T, H = c.T, c.H
        assert H[0] == c.H_lo and H[-1] == c.H_hi
        assert len(c.kinds) == T.size - 1 == H.size - 1
        assert (np.diff(T) >= 0.).all() and (np.diff(H) >= 0.).all()
        assert c.T_lo <= T.min() and T.max() <= c.T_hi
        for j, kind in enumerate(c.kinds):
            assert kind in (FLAT, SENSIBLE, GLIDE)
            if kind == FLAT: assert T[j] == T[j + 1]
        duty = c.H_hi - c.H_lo
        for H_ in _interior_enthalpies(c):
            s = c.state_at_H(H_)
            assert abs(s.H - H_) <= 1e-9 * duty, (c, H_)
            kind = c.kinds[_segment(c, H_)]
            T_lin, T_ex = c.T_at(H_), c.T_exact(H_)
            if kind == GLIDE:
                assert abs(T_ex - s.T) <= 1e-9
                assert abs(T_lin - T_ex) <= 2. * GLIDE_TOL_T
            else:
                assert T_ex == T_lin
                assert abs(s.T - T_lin) <= 1e-9
            if kind == SENSIBLE:
                assert abs(c.H_at(T_lin) - H_) <= 1e-9 * duty

def test_jumps_mark_non_equilibrium_end_states():
    setup('jumps')
    # regression case 05's H1: water vapor fed at 420 K and 5 bar, below its
    # 425 K saturation temperature; its equilibrium states at the feed
    # enthalpy lie outside the stream's own range [330, 420] K
    H1 = utility_hx('H1', 420., 5e5, 'g', 330., Water=250.)
    s_in = H1.ins[0].copy()
    s_out = H1.outs[0].copy(); s_out.vle(H=s_out.H, P=s_out.P)
    c = StreamCurve(s_in, s_out, True)
    liquid = s_in.copy(); liquid.phase = 'l'  # equilibrium at the feed T
    assert c.method == 'pure, liquid'
    assert c.kinds[-1] == FLAT and c.T[-1] == c.T[-2] == 420.
    assert len(c.jumps) == 1
    H_a, H_b = c.jumps[0]
    assert H_b == c.H_hi == s_in.H
    assert_allclose(H_a, liquid.H, rtol=1e-12)
    H_mid = 0.5 * (H_a + H_b)
    flashed = s_in.copy(); flashed.vle(H=H_mid, P=5e5)
    assert flashed.T > 424.
    # inside the jump the curve's state stays at the feed temperature
    s = c.state_at_H(H_mid)
    assert_allclose([s.T, s.H], [420., H_mid], rtol=1e-12)
    assert c.T_exact(H_mid) == c.T_at(H_mid) == 420.
    # a latent-heat flat at an end is not a jump (C1 boils to its dew point)
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('two_stream')
    boiler = curves[names.index('C1')]
    assert boiler.kinds[-1] == FLAT and boiler.jumps == ()
    assert all(c.jumps == () for c in curves)
    # a non-monotone stream (a reboiler outlet colder than its inlet) is a
    # point load: its single flat is its whole duty, not a jump
    setup('jumps_point_load')
    a = bst.Stream('nm_in', Water=100., T=372., P=ATM, phase='l', units='kmol/hr')
    b = bst.Stream('nm_out', Water=100., T=371., P=ATM, phase='g', units='kmol/hr')
    point = StreamCurve(a, b, False)
    assert not point.monotone and point.kinds == (FLAT,) and point.jumps == ()
    # a superheated-liquid feed heated to vapor: the jump is at its inlet
    # (the low end), from the liquid's enthalpy up to the vapor's at 380 K
    s_in = bst.Stream('shl_in', Water=100., T=380., P=ATM, phase='l', units='kmol/hr')
    s_out = bst.Stream('shl_out', Water=100., T=400., P=ATM, phase='g', units='kmol/hr')
    c = StreamCurve(s_in, s_out, False)
    vapor = s_in.copy(); vapor.phase = 'g'
    assert c.kinds[0] == FLAT and c.T[0] == c.T[1] == 380.
    assert len(c.jumps) == 1 and c.jumps[0][0] == s_in.H
    assert_allclose(c.jumps[0][1], vapor.H, rtol=1e-12)

def test_T_exact_on_flats_and_vertical_stretches():
    setup('T_exact')
    # superheated liquid ethanol (370 K, 1 atm; bubble point 351.4 K) cooled
    # to 310 K: its curve ends with a vertical (clipped) stretch at H_in
    s_in = bst.Stream('sh_in', Ethanol=700., T=370., P=ATM, phase='l', units='kmol/hr')
    s_out = bst.Stream('sh_out', Ethanol=700., T=310., P=ATM, phase='l', units='kmol/hr')
    c = StreamCurve(s_in, s_out, True)
    T_sat = c.boundaries['T_sat']
    assert c.T[-1] == 370. and c.T[-2] == T_sat and c.H[-1] == c.H[-2] == s_in.H
    assert c.T_exact(c.H_hi, 'low') == T_sat
    assert c.T_exact(c.H_hi, 'high') == 370.
    j = c.kinds.index(FLAT)
    H_flat = 0.5 * (c.H[j] + c.H[j + 1])
    assert c.T_exact(H_flat) == c.T_at(H_flat) == T_sat
    assert c.jumps == ()  # the flat is interior latent heat

def clipped_end_streams(kind):
    """(inlet, outlet, is_hot, end) of a stream fed off equilibrium whose
    curve reaches its feed enthalpy strictly inside a piece and is clipped
    beyond it ('hi': H_hi, 'lo': H_lo)."""
    if kind == 'glide top':  # water/ethanol liquid fed above its bubble point
        s = bst.Stream('ge_in', Water=900., Ethanol=100., T=368., P=ATM,
                       phase='l', units='kmol/hr')
        o = s.copy('ge_out'); o.T = 320.
        return s, o, True, 'hi'
    if kind == 'glide bottom':  # the same mixture fed as vapor below its dew point
        s = bst.Stream('gb_in', Water=900., Ethanol=100., T=362., P=ATM,
                       phase='g', units='kmol/hr')
        o = s.copy('gb_out'); o.T = 385.
        return s, o, False, 'lo'
    # pure water fed two-phase at 400 K, above T_sat: the equilibrium vapor
    # reaches its enthalpy at 388.4 K
    s = bst.MultiStream('sv_in', T=400., P=ATM, l=[('Water', 1.)],
                        g=[('Water', 99.)], units='kmol/hr')
    o = bst.Stream('sv_out', Water=100., T=320., P=ATM, phase='l', units='kmol/hr')
    return s, o, True, 'hi'

@pytest.mark.parametrize('kind', ['glide top', 'glide bottom', 'sensible top'])
def test_clipped_curve_ends_at_the_exact_crossing(kind):
    # the breakpoint where the curve reaches the feed enthalpy is the exact
    # equilibrium state there (a PH flash, independent of the curve's
    # samplers); it used to keep the temperature of the first clipped sample
    # (0.108 K, 0.036 K and 0.0024 K off), which the exact-state check
    # trusts at breakpoints
    setup('clipped_' + kind.replace(' ', '_'))
    s_in, s_out, is_hot, end = clipped_end_streams(kind)
    c = StreamCurve(s_in, s_out, is_hot)
    H_end, side = (c.H_hi, 'low') if end == 'hi' else (c.H_lo, 'high')
    assert H_end == s_in.H
    duty = c.H_hi - c.H_lo
    for d in (0., 1e-6, 1e-3, 1e-2, 3e-2):
        H = H_end - d * duty if end == 'hi' else H_end + d * duty
        flash = s_in.copy(); flash.vle(H=H, P=s_in.P)
        tol = 1e-6 if d == 0. else GLIDE_TOL_T
        assert abs(c.T_at(H, side) - flash.T) <= tol, (d, c.T_at(H, side), flash.T)
        assert abs(c.T_exact(H, side) - flash.T) <= 1e-6, (d, c.T_exact(H, side), flash.T)

def test_T_exact_in_binary_glide():
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('binary_glide')
    c = curves[names.index('C1')]
    assert c.method == 'mixture, binary bubble-curve glide'
    glide = [j for j, kind in enumerate(c.kinds) if kind == GLIDE]
    assert len(glide) > 10
    T_b, T_d = c.boundaries['T_bubble'], c.boundaries['T_dew']
    previous = -np.inf
    for j in glide[::max(1, len(glide) // 8)]:
        H_ = 0.5 * (c.H[j] + c.H[j + 1])
        T_ex = c.T_exact(H_)
        assert c.T[j] <= T_ex <= c.T[j + 1] and T_b <= T_ex <= T_d
        assert abs(T_ex - c.T_at(H_)) <= 2. * GLIDE_TOL_T
        assert T_ex > previous
        previous = T_ex
        # the state is on the bubble-point curve: its liquid boils at T
        s = c.state_at_H(H_)
        assert abs(s.T - T_ex) <= 1e-9
        liquid = bst.Stream(None, T=s.T, P=ATM, phase='l', units='kmol/hr',
                            thermo=s.thermo)
        liquid.imol['Water', 'Ethanol'] = s.imol['l', ('Water', 'Ethanol')]
        assert_allclose(liquid.bubble_point_at_P(ATM).T, T_ex, rtol=0, atol=1e-6)

def reboiler_curve(name):
    """Curve of the reboiler stream of the `HeatExchangerNetwork` doctest
    system (water with 1.4 % methanol and 0.3 % glycerol boiled at 1 atm):
    a ternary glide sampled with TP flashes."""
    setup(name, ('Water', 'Methanol', 'Glycerol'))
    hx = boiling_hx('C1', 306.37, ATM, 372.6, Water=8487., Methanol=118.85,
                    Glycerol=25.01)
    s_in = hx.ins[0].copy()
    s_out = hx.outs[0].copy(); s_out.vle(H=s_out.H, P=s_out.P)
    return StreamCurve(s_in, s_out, False)

def test_tp_glide_states_next_to_the_glide_ends():
    """A ternary glide is sampled with TP flashes, which cannot resolve the
    bubble point to the last digits: a flash a hair inside the glide can
    come back single phase. `state_at_H` and `T_exact` must still return
    the state with the requested enthalpy there (an exchanger stage starting
    just past a bubble point), not fail on that flash."""
    c = reboiler_curve('tp_glide_ends')
    assert c.method == 'mixture, TP-flash glide'
    glide = [j for j, kind in enumerate(c.kinds) if kind == GLIDE]
    first, last = glide[0], glide[-1]
    assert c.T[first] == c.boundaries['T_bubble']
    assert c.T[last + 1] == c.T_hi
    duty = c.H_hi - c.H_lo
    for j, f in ((first, 1e-9), (first, 1e-6), (last, 1. - 1e-6), (last, 1. - 1e-9)):
        H = c.H[j] + f * (c.H[j + 1] - c.H[j])
        s = c.state_at_H(H)
        assert abs(s.H - H) <= 1e-9 * duty
        assert c.T[j] - 1e-6 <= s.T <= c.T[j + 1] + 1e-6
        assert c.T[j] <= c.T_exact(H) <= c.T[j + 1]

def test_tp_glide_states_where_tp_flashes_are_spurious():
    """Between 372.0 and 372.5 K, thermosteam's TP flash of the reboiler
    stream returns near-total vaporization at isolated temperatures (the
    other flashes, the PH flash and the curve agree on ~20-45 % vapor). The
    glide sampler drops those flashes, but a root solve inside a segment
    cannot avoid them: its result must be rejected, not returned as a
    state whose enthalpy is off by up to the whole duty (or a crash). The
    curve is monotone through exact samples, so the true state with an
    enthalpy between two samples lies between their temperatures."""
    c = reboiler_curve('tp_glide_spurious')
    duty = c.H_hi - c.H_lo
    glide = [j for j, kind in enumerate(c.kinds)
             if kind == GLIDE and c.T[j + 1] > 372. and c.T[j] < 372.5]
    assert len(glide) >= 4
    for j in glide[1::2]:
        H = c.H[j] + 0.37 * (c.H[j + 1] - c.H[j])
        s = c.state_at_H(H)
        assert abs(s.H - H) <= 1e-9 * duty
        assert c.T[j] <= s.T <= c.T[j + 1]
        assert abs(c.T_exact(H) - s.T) <= 1e-6
        assert s.imol['g'].sum() < 0.5 * s.F_mol

def test_glide_state_falls_back_to_the_curve(monkeypatch):
    """Where neither the root solve nor a PH flash gives a state between
    the two glide samples that bracket an enthalpy, the glide state is the
    curve's own: the lever-rule mix of the two sampled states at the
    linearized temperature."""
    c = reboiler_curve('glide_fallback')
    duty = c.H_hi - c.H_lo

    class NoFlash:
        def copy(self): return self
        def vle(self, **kwargs): raise RuntimeError('flash failed')

    monkeypatch.setattr(StreamCurve, '_glide_root', lambda self, H, state=False: None)
    monkeypatch.setattr(c, 'stream_in', NoFlash())
    glide = [j for j, kind in enumerate(c.kinds) if kind == GLIDE]
    for j in (glide[1], glide[len(glide) // 2], glide[-2]):
        H = c.H[j] + 0.37 * (c.H[j + 1] - c.H[j])
        s = c._glide_state(H)
        assert abs(s.H - H) <= 1e-9 * duty
        assert abs(s.T - c.T_at(H)) <= 1e-6

# ---------------------------------------------------------------------------
# pinch_state
# ---------------------------------------------------------------------------

def test_pinch_state_deterministic_at_T_sat():
    """A pinch exactly at a pure stream's saturation temperature: the state
    is the saturated liquid ('left') or vapor ('right') by construction, not
    whatever phase split a flash at T_sat happens to keep."""
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('Tsat_coincidence')
    j = names.index('C1')
    si, so, c = ins[j], outs[j], curves[j]
    T_sat = c.boundaries['T_sat']
    assert table.pinch_T == T_sat and c.T_lo < T_sat < c.T_hi
    satL = si.copy(); satL.vle(V=0, P=si.P)
    satV = si.copy(); satV.vle(V=1, P=si.P)
    for side, sat, phase in (('left', satL, 'l'), ('right', satV, 'g')):
        states = [pinch_state(si, so, T_sat, side=side),
                  pinch_state(si, so, T_sat, side=side, curve=c),
                  pinch_state(si, so, T_sat, side=side, curve=c)]
        # one curve: bit-identical; a curve rebuilt later can differ in the
        # last digits only (thermosteam's cached bubble/dew point solvers
        # warm-start from their previous call)
        assert states[1].H == states[2].H
        assert_allclose(states[0].H, states[1].H, rtol=1e-11)
        assert_allclose(states[0].H, sat.H, rtol=1e-10)
        assert_allclose(states[0].T, T_sat, rtol=0, atol=1e-6)
        assert_allclose(states[0].imol[phase, 'Water'], 200., rtol=1e-12)
    # without a side, a cold stream's interior pinch takes the left branch
    assert (pinch_state(si, so, T_sat, curve=c).H
            == pinch_state(si, so, T_sat, side='left', curve=c).H)
    # at an end temperature without a side: that end's state (old contract)
    assert_allclose(pinch_state(si, so, si.T).H, si.H, rtol=1e-12)
    assert_allclose(pinch_state(si, so, so.T).H, so.H, rtol=1e-12)

# ---------------------------------------------------------------------------
# _min_approach
# ---------------------------------------------------------------------------

def test_min_approach_finds_internal_pinch_of_condenser():
    """Ethanol vapor desuperheated and condensed against water: 10 K and
    ~18 K at the terminals, but only ~1.8 K where condensation starts."""
    setup('min_approach')
    hot = utility_hx('H', 360., ATM, 'g', 340., Ethanol=100.)
    cold = utility_hx('C', 320., ATM, 'l', 350., Water=2000.)
    h_in, h_out = hot.ins[0].copy(), hot.outs[0].copy()
    c_in, c_out = cold.ins[0].copy(), cold.outs[0].copy()
    ch = StreamCurve(h_in, h_out, True)
    cc = StreamCurve(c_in, c_out, False)
    Q = ch.H_hi - ch.H_lo
    assert Q < cc.H_hi - cc.H_lo
    dT_min, q = _min_approach(ch, ch.H_hi, cc, cc.H_hi, Q)
    # independent: condensation starts at the saturated-vapor enthalpy
    satV = h_in.copy(); satV.vle(V=1, P=ATM)
    q_sat = h_in.H - satV.H
    water = c_out.copy(); water.H = c_out.H - q_sat  # liquid, phase fixed
    assert_allclose(q, q_sat, rtol=1e-9)
    assert_allclose(dT_min, satV.T - water.T, rtol=0, atol=1e-6)
    assert 1. < dT_min < 3.
    # the terminals alone look comfortable
    water.H = c_out.H - Q
    assert h_in.T - c_out.T == 10. and h_out.T - water.T > 15.
    assert _min_approach(ch, ch.H_hi, cc, cc.H_hi, Q, exact=False) == (dT_min, q)

def test_min_approach_curved_heat_capacity():
    """Water against ethanol liquid over their full ranges: both terminal
    approaches are ~5 K but the curves come within ~1 K inside; checked
    against a dense scan of independent phase-fixed temperature solves."""
    names, ins, outs, is_hot, dT, table, curves, grid = analyzed('curvature')
    jh, jc = names.index('H1'), names.index('C1')
    ch, cc = curves[jh], curves[jc]
    Q = min(ch.H_hi - ch.H_lo, cc.H_hi - cc.H_lo)
    dT_min, q = _min_approach(ch, ch.H_hi, cc, cc.H_hi, Q)
    hs, cs = ins[jh].copy(), outs[jc].copy()
    dTs = []
    for x in np.linspace(0., Q, 401):
        hs.H = ch.H_hi - x
        cs.H = cc.H_hi - x
        dTs.append(hs.T - cs.T)
    assert min(dTs[0], dTs[-1]) > 4.99
    assert dT_min < 1.5
    assert abs(dT_min - min(dTs)) < 2e-3
    assert 0. < q < Q

def test_min_approach_exact_inside_glides():
    """Two water/ethanol glides (ethanol mole fraction 0.1, whose flashes are
    reliable) at 2.5 bar and 1 atm: at the minimum, each stream sits at a
    breakpoint of one curve and inside a glide segment of the other, where
    the linearized temperature is off by up to the sampling tolerance. The
    exact minimum equals independent PH flashes at its position."""
    setup('min_approach_glides')
    P = 2.5e5
    hot = utility_hx('H1', 400., P, 'g', 340., rigorous=True, **xE(180., 0.1))
    cold = boiling_hx('C1', 330., ATM, 380., **xE(200., 0.1))
    h_in, h_out = hot.ins[0].copy(), hot.outs[0].copy()
    c_in, c_out = cold.ins[0].copy(), cold.outs[0].copy()
    for s in (h_out, c_out): s.vle(H=s.H, P=s.P)
    ch = StreamCurve(h_in, h_out, True)
    cc = StreamCurve(c_in, c_out, False)
    assert GLIDE in ch.kinds and GLIDE in cc.kinds
    Q = min(ch.H_hi - ch.H_lo, cc.H_hi - cc.H_lo)
    dT, q = _min_approach(ch, ch.H_hi, cc, cc.H_hi, Q)
    dT_lin, q_lin = _min_approach(ch, ch.H_hi, cc, cc.H_hi, Q, exact=False)
    h, c = h_in.copy(), c_in.copy()

    def flashed(x):
        h.vle(H=ch.H_hi - x, P=P)
        c.vle(H=cc.H_hi - x, P=ATM)
        return h.T - c.T

    assert abs(dT - flashed(q)) < 1e-6
    assert abs(dT_lin - dT) > 1e-5  # the exact re-evaluation mattered
    assert abs(dT_lin - dT) <= 2. * (ch.tol_T + cc.tol_T)
    # no position nearby is closer (dense independent scan around q)
    xs = np.linspace(max(0., q - 0.01 * Q), min(Q, q + 0.01 * Q), 41)
    dense = [flashed(x) for x in xs]
    assert min(dense) >= dT - 1e-6

# ---------------------------------------------------------------------------
# Legacy entry points on the new table
# ---------------------------------------------------------------------------

def test_pinch_analysis_builds_curves_once():
    from hensmith.hxn_synthesis import (
        _pinch_analysis, temperature_interval_pinch_analysis, load_duties,
    )
    units = case_condenser_at_pinch()[0]
    hus = [hx.heat_utilities[0] for hx in units]
    hus.sort(key=lambda hu: hu.duty)
    full = _pinch_analysis(hus, 5.)
    legacy = temperature_interval_pinch_analysis(hus, 5.)
    assert len(full) == 15 and len(legacy) == 12
    table, curves, grid = full[12:]
    assert_allclose([legacy[1], legacy[2]], [table.hot_util_load, table.cold_util_load],
                    rtol=1e-12)
    assert len(curves) == len(hus) and grid['Hl'].shape == (len(hus), table.Ts.size)
    pinch_T_arr, T_out_arr, indices = full[0], full[4], full[8]
    streams_inlet, streams_quenched = full[9], full[11]
    cold = set(full[7])
    duties = []
    for kw in ({}, dict(curves=curves)):
        Q_hot_side, Q_cold_side = {}, {}
        load_duties(streams_inlet, streams_quenched, pinch_T_arr, T_out_arr,
                    indices, lambda i: i in cold, Q_hot_side, Q_cold_side, **kw)
        duties.append((Q_hot_side, Q_cold_side))
    for without, with_curves in zip(duties[0], duties[1]):
        assert without.keys() == with_curves.keys()
        for i, (kind, Q) in without.items():
            assert with_curves[i][0] == kind
            assert_allclose(with_curves[i][1], Q, rtol=1e-10, atol=1e-6)
    for i in indices:
        total = duties[0][0][i][1] + duties[0][1][i][1]
        assert_allclose(total, abs(streams_quenched[i].H - streams_inlet[i].H),
                        rtol=0, atol=0.02)
