# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2026-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""
Data of the minimum-energy-requirement (MER) corpus used by
``test_hxn_mer.py``. Data only: every check lives in the test module.

``NO_SPLIT`` holds problems for which an *unsplit* network reaching the MER
targets is known to exist (each carries a certificate network); ``SPLIT``
holds problems for which the pinch design rules *prove* that MER needs stream
splitting (each carries the proof). Both lists mix two kinds of case:

``kind='constant_cp'`` (literature problems)
    ``streams`` are ``(name, 'hot' | 'cold', T_in, T_out, CP)`` in the
    SOURCE units: temperatures in ``T_unit`` ('C', 'F' or 'K') and heat
    capacity flow rates in ``Q_unit`` per ``T_unit`` degree (``Q_UNITS``
    converts the heat-rate unit to kW, ``T_UNITS`` the temperatures to
    kelvin). ``targets`` are the problem-table targets in source units
    (hensmith convention: hot streams shifted down by ``dTmin``;
    ``pinch_cold`` is the shifted pinch, ``pinch_hot = pinch_cold + dTmin``;
    the first cascade minimum, or the top of the grid when ``Q_hot == 0``),
    computed by the independent oracle of the benchmark (see the test
    module). ``published`` holds the values printed in the source as
    ``(value, absolute tolerance)`` in source units (``None`` where the
    source gives none): one unit in the last printed digit of a utility
    target unless ``note`` says otherwise, zero for a zero target and for a
    pinch temperature (always a stream end temperature). ``certificate``
    (NO_SPLIT) is a verified unsplit MER network:
    ``matches`` are ``(side, stage, hot, cold, Q, T_hot_in, T_hot_out,
    T_cold_in, T_cold_out)`` in source units, hot streams meet their matches
    in increasing stage and cold streams in decreasing stage, and each stream
    has at most one utility at its far end (``hot_utility`` on cold streams,
    ``cold_utility`` on hot streams). ``needs_repeated_pair`` is True when
    no unsplit MER network exists without matching some pair twice on one
    side (complete search), False when one does, None if unknown.
    ``proof`` (SPLIT) is the first pinch-rule violation (top pinch first,
    above before below): ``rule`` 'number' (more "must" streams at the pinch
    than partners) or 'cp' (no one-to-one CP-feasible pinch matching), with
    the streams at the pinch on that side.

``kind='real_thermo'`` (water / ethanol / methanol and their binaries)
    ``streams`` are ``(ID, {chemical: kmol/hr}, T_in, T_out, P, phase_in,
    rigorous)`` with T in K or 'bubble' / 'dew'; each is built as a simulated
    ``bst.HXutility(ID, ins=inlet, T=T_out, rigorous=rigorous)`` (``V=0`` /
    ``V=1`` with ``rigorous=True`` for 'bubble' / 'dew' outlets; a 'bubble' /
    'dew' inlet is the saturated state at ``P``). ``reference`` holds MER
    targets [kJ/hr] and the shifted pinch [K] (None: no hot utility) from an
    independent dense-grid calculator (``grid_h`` K single-phase grid, ``nV``
    two-phase states traced along the bubble curve; accuracy about 0.2
    kJ/hr). ``certificate`` (NO_SPLIT) lists process matches with their duty
    [kJ/hr], side of the pinch and 1-based position in FLOW order along each
    stream (``hot_seq`` / ``cold_seq``); every stream then gets at most one
    utility at its far end, and ``certificate_utilities`` are that network's
    simulated utilities. ``proof`` (SPLIT) is a pinch number-rule violation
    at the reference pinch; no stream has an end temperature or phase change
    within 0.5 K of that pinch except the supply temperature that creates it.
    ``short_of_tickoff`` (rtA02) names the streams of a proof that every
    unsplit MER network needs a pinch match that stops short of tick-off and
    a repeated pair.

Provenance: the constant-CP problems, their targets, certificates and proofs
come from a certified benchmark built for the hensmith MER synthesizer
(stream data fetched from the cited sources; labels from an independent
MILP/pinch-rule oracle; certificates re-verified by plain arithmetic); the
real-thermodynamics problems were designed for hensmith, their certificates
simulated with biosteam ``HXprocess`` units and verified at 101+ points per
exchanger (internal approach >= T_min_app - 1e-6 K), and their split proofs
re-derived independently.
"""

__all__ = ('NO_SPLIT', 'SPLIT', 'Q_UNITS', 'T_UNITS')

#: Source heat-rate unit -> kW ('unspecified': the source gives no unit; read
#: as kW). International Table Btu (1055.05585262 J).
Q_UNITS = {
    'kW': 1.,
    'MW': 1e3,
    'Btu/hr': 1.05505585262 / 3600.,
    '1e3 Btu/hr': 1.05505585262 / 3.6,
    'unspecified': 1.,
}

#: Source temperature unit -> (kelvin per degree, offset): T_K = (T + offset) * scale.
T_UNITS = {
    'K': (1., 0.),
    'C': (1., 273.15),
    'F': (5. / 9., 459.67),
}

NO_SPLIT = [
    # ----- constant-CP literature problems -----
    dict(
        name='4sp1_lee1970_dt10F',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '4sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Lee, Masso & Rudd (1970), Ind. Eng. Chem. Fundam. 9, 48 (4SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/4sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/4sp1.dat)'
        ),
        T_unit='F', Q_unit='1e3 Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 320.0, 200.0, 16.67),
            ('H2', 'hot', 480.0, 280.0, 20.0),
            ('C1', 'cold', 140.0, 320.0, 14.45),
            ('C2', 'cold', 240.0, 500.0, 11.53),
        ],
        targets=dict(Q_hot=345.9, Q_cold=747.5000000000003, pinch_hot=480.0, pinch_cold=470.0),
        published=dict(Q_hot=(345.9, 0.1), Q_cold=(747.5, 0.1), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'H2', 'C2', 2651.9, 480.0, 347.405, 240.0, 470.0),
                ('below', 2, 'H2', 'C1', 600.5999999999999,
                 347.405, 317.375, 278.4359861591696, 320.00000000000006),
                ('below', 3, 'H1', 'C1', 2000.4, 320.0, 200.0, 140.0, 278.4359861591696),
            ],
            hot_utility={'C2': 345.9},
            cold_utility={'H2': 747.5},
        ),
    ),
    dict(
        name='4sp2_dt20F',
        kind='constant_cp',
        citation=(
            'Problem 4SP2 (Lee, Masso & Rudd 1970 test-problem family) as tabulated in Muraki, M.; '
            'Hayakawa, T. (1982). Practical synthesis method for heat exchanger network. J. Chem. '
            'Eng. Japan 15(2), 136-141, Table 1 (stream data; design data: minimum allowable approach'
            ' temperature 20 F for exchangers).'
        ),
        source_url='https://www.jstage.jst.go.jp/article/jcej1968/15/2/15_2_136/_pdf/-char/ja',
        T_unit='F', Q_unit='Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 500, 110, 20000),
            ('H2', 'hot', 430, 230, 50000),
            ('H3', 'hot', 400, 110, 30000),
            ('C1', 'cold', 25, 420, 70000),
        ],
        targets=dict(Q_hot=1150000.0, Q_cold=0.0, pinch_hot=45.0, pinch_cold=25.0),
        published=None,
        needs_repeated_pair=True,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'H1', 'C1', 2140000.0000000014,
                 500.0, 392.99999999999994, 372.99999999999994, 403.57142857142856),
                ('above', 2, 'H2', 'C1', 6474999.999999996,
                 430.0, 300.5000000000001, 280.5, 372.99999999999994),
                ('above', 3, 'H1', 'C1', 1409999.9999999977,
                 392.99999999999994, 322.50000000000006, 260.3571428571429, 280.5),
                ('above', 4, 'H2', 'C1', 3525000.0000000033,
                 300.5000000000001, 230.00000000000006, 210.0, 260.3571428571429),
                ('above', 5, 'H3', 'C1', 8700000.0, 400.0, 110.0, 85.71428571428572, 210.0),
                ('above', 6, 'H1', 'C1', 4250000.000000001,
                 322.50000000000006, 110.0, 25.0, 85.71428571428572),
            ],
            hot_utility={'C1': 1150000.0000000012},
            cold_utility={},
        ),
    ),
    dict(
        name='linnhoff_4stream',
        kind='constant_cp',
        citation=(
            'Linnhoff, B. & Hindmarsh, E. (1983). The pinch design method for heat exchanger '
            'networks. Chem. Eng. Sci. 38, 745-763 (four-stream example); also Kemp, I.C. (2007). '
            'Pinch Analysis and Process Integration, 2nd ed., Butterworth-Heinemann.'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('C1', 'cold', 20, 135, 2),
            ('H2', 'hot', 170, 60, 3),
            ('C3', 'cold', 80, 140, 4),
            ('H4', 'hot', 150, 30, 1.5),
        ],
        targets=dict(Q_hot=20.0, Q_cold=60.0, pinch_hot=90.0, pinch_cold=80.0),
        published=dict(Q_hot=(20, 1.0), Q_cold=(60, 1.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'H2', 'C3', 240.0, 170.0, 90.0, 80.0, 140.0),
                ('above', 1, 'H4', 'C1', 90.0, 150.0, 90.0, 80.0, 125.0),
                ('below', 2, 'H2', 'C1', 30.0, 90.0, 80.0, 65.0, 80.0),
                ('below', 3, 'H4', 'C1', 90.0, 90.0, 30.0, 20.0, 65.0),
            ],
            hot_utility={'C1': 20.0},
            cold_utility={'H2': 60.0},
        ),
    ),
    dict(
        name='nptel_t3_10_threshold',
        kind='constant_cp',
        citation=(
            "Mohanty B (IIT Roorkee). NPTEL course 103107094 'Process Integration' lecture notes, "
            'Module 3 Lecture 8, Table 3.10 (Example 02).'
        ),
        source_url=(
            'https://archive.nptel.ac.in/content/storage2/courses/103107094/module3/lecture5/lecture5'
            '.pdf'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('Hot-1', 'hot', 350, 290, 3.5),
            ('Hot-2', 'hot', 400, 290, 1.8),
            ('Cold-1', 'cold', 150, 350, 2.0),
            ('Cold-2', 'cold', 290, 400, 2.5),
        ],
        targets=dict(Q_hot=267.0, Q_cold=0.0, pinch_hot=160.0, pinch_cold=150.0),
        published=dict(Q_hot=(267, 1.0), Q_cold=(0.0, 0.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'Hot-1', 'Cold-1', 210.0, 350.0, 290.0, 159.0, 264.0),
                ('above', 1, 'Hot-2', 'Cold-2', 180.0, 400.0, 300.0, 290.0, 362.0),
                ('above', 2, 'Hot-2', 'Cold-1', 18.0, 300.0, 290.0, 150.0, 159.0),
            ],
            hot_utility={'Cold-1': 172.0, 'Cold-2': 95.0},
            cold_utility={},
        ),
    ),
    dict(
        name='nptel_t3_9_threshold_split_shown',
        kind='constant_cp',
        citation=(
            "Mohanty B (IIT Roorkee). NPTEL course 103107094 'Process Integration' lecture notes, "
            'Module 3 Lecture 8, Table 3.9 = Module 5 Lecture 29, Table 5.6 (design Fig. 5.30).'
        ),
        source_url=(
            'https://archive.nptel.ac.in/content/storage2/courses/103107094/module5/lecture4/lecture4'
            '.pdf'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('Hot-1', 'hot', 190, 55, 3.5),
            ('Hot-2', 'hot', 155, 40, 1.8),
            ('Cold-1', 'cold', 20, 140, 2.0),
            ('Cold-2', 'cold', 70, 150, 2.5),
        ],
        targets=dict(Q_hot=0.0, Q_cold=239.5, pinch_hot=190.0, pinch_cold=180.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(239.5, 0.1), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'Hot-1', 'Cold-2', 65.0, 190.0, 171.42857142857142, 124.0, 150.0),
                ('below', 2, 'Hot-1', 'Cold-1', 240.0,
                 171.42857142857142, 102.85714285714285, 20.0, 140.0),
                ('below', 2, 'Hot-2', 'Cold-2', 135.0, 155.0, 80.0, 70.0, 124.0),
            ],
            hot_utility={},
            cold_utility={'Hot-1': 167.49999999999997, 'Hot-2': 72.0},
        ),
    ),
    dict(
        name='nptel_t5_5_threshold_dT5',
        kind='constant_cp',
        citation=(
            "Mohanty B (IIT Roorkee). NPTEL course 103107094 'Process Integration' lecture notes, "
            'Module 5 Lecture 29, Table 5.5 (designs Figs. 5.27-5.28).'
        ),
        source_url=(
            'https://archive.nptel.ac.in/content/storage2/courses/103107094/module5/lecture4/lecture4'
            '.pdf'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=5,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('Hot-1', 'hot', 180, 60, 3.0),
            ('Hot-2', 'hot', 140, 30, 1.5),
            ('Cold-1', 'cold', 20, 135, 2.0),
            ('Cold-2', 'cold', 80, 140, 4.0),
        ],
        targets=dict(Q_hot=0.0, Q_cold=55.0, pinch_hot=180.0, pinch_cold=175.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(55, 1.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=True,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'Hot-1', 'Cold-1', 45.0, 180.0, 165.0, 112.5, 135.0),
                ('below', 2, 'Hot-1', 'Cold-2', 240.0, 165.0, 85.0, 80.0, 140.0),
                ('below', 2, 'Hot-2', 'Cold-1', 135.0, 140.0, 50.0, 45.0, 112.5),
                ('below', 3, 'Hot-1', 'Cold-1', 50.0, 85.0, 68.33333333333333, 20.0, 45.0),
            ],
            hot_utility={},
            cold_utility={'Hot-1': 24.999999999999986, 'Hot-2': 30.0},
        ),
    ),
    dict(
        name='smith2005_exr18_9_threshold',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 18, Exercise 9, '
            'Tables 18.17 and 18.18.'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2018.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 500, 100, 4.0),
            ('2', 'cold', 50, 450, 1.0),
            ('3', 'cold', 60, 400, 1.0),
            ('4', 'cold', 40, 420, 0.75),
        ],
        targets=dict(Q_hot=0.0, Q_cold=575.0, pinch_hot=500.0, pinch_cold=480.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(575.0, 1.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=True,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, '1', '3', 51.25, 500.0, 487.1875, 348.75, 400.0),
                ('below', 2, '1', '2', 188.75, 487.1875, 440.0, 261.25, 450.0),
                ('below', 3, '1', '4', 285.0, 440.0, 368.75, 40.0, 420.0),
                ('below', 4, '1', '3', 288.75, 368.75, 296.5625, 60.0, 348.75),
                ('below', 5, '1', '2', 211.25, 296.5625, 243.75, 50.0, 261.25),
            ],
            hot_utility={},
            cold_utility={'1': 575.0},
        ),
    ),
    dict(
        name='turton_shaeiwitz2012_example1',
        kind='constant_cp',
        citation=(
            "Turton R, Shaeiwitz J (2012). 'Pinch Technology / Heat Exchange Networks' lecture notes "
            '(West Virginia University), first example (as reproduced on notes.edubirdie.com).'
        ),
        source_url=(
            'https://notes.edubirdie.com/docs/kentucky-state-university/che-102-general-chemistry-i/1'
            '06468-pinch-technology-heat-exchange-networks'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 200, 120, 3.0),
            ('2', 'hot', 140, 100, 5.0),
            ('3', 'cold', 100, 170, 3.0),
            ('4', 'cold', 110, 190, 2.0),
        ],
        targets=dict(Q_hot=60.0, Q_cold=130.0, pinch_hot=140.0, pinch_cold=130.0),
        published=dict(Q_hot=(60, 1.0), Q_cold=(130, 1.0), pinch_hot=(140, 0.0), pinch_cold=(130, 0.0)),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, '1', '4', 60.0, 200.0, 180.0, 130.0, 160.0),
                ('above', 2, '1', '3', 120.0, 180.0, 140.0, 130.0, 170.0),
                ('below', 3, '1', '4', 40.0, 140.0, 126.66666666666667, 110.0, 130.0),
                ('below', 3, '2', '3', 90.0, 140.0, 122.0, 100.0, 130.0),
            ],
            hot_utility={'4': 60.0},
            cold_utility={'1': 20.000000000000014, '2': 110.0},
        ),
    ),
    dict(
        name='turton_shaeiwitz2012_inclass',
        kind='constant_cp',
        citation=(
            "Turton R, Shaeiwitz J (2012). 'Pinch Technology / Heat Exchange Networks' lecture notes "
            '(West Virginia University), in-class example (as reproduced on notes.edubirdie.com).'
        ),
        source_url=(
            'https://notes.edubirdie.com/docs/kentucky-state-university/che-102-general-chemistry-i/1'
            '06468-pinch-technology-heat-exchange-networks'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 250, 100, 1.0),
            ('2', 'hot', 280, 120, 4.0),
            ('3', 'cold', 100, 200, 2.0),
            ('4', 'cold', 120, 230, 5.0),
        ],
        targets=dict(Q_hot=40.0, Q_cold=80.0, pinch_hot=140.0, pinch_cold=120.0),
        published=dict(Q_hot=(40, 1.0), Q_cold=(80, 1.0), pinch_hot=(140, 0.0), pinch_cold=(120, 0.0)),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, '2', '3', 10.0, 280.0, 277.5, 175.0, 180.0),
                ('above', 2, '1', '3', 110.0, 250.0, 140.0, 120.0, 175.0),
                ('above', 2, '2', '4', 550.0, 277.5, 140.0, 120.0, 230.0),
                ('below', 3, '2', '3', 40.0, 140.0, 130.0, 100.0, 120.0),
            ],
            hot_utility={'3': 40.0},
            cold_utility={'1': 40.0, '2': 40.0},
        ),
    ),
    dict(
        name='5sp1_dt20F',
        kind='constant_cp',
        citation=(
            'Problem 5SP1 (Lee, Masso & Rudd 1970) as tabulated in Muraki, M.; Hayakawa, T. (1982). '
            'Practical synthesis method for heat exchanger network. J. Chem. Eng. Japan 15(2), '
            '136-141, Table 1 (stream data; design data: minimum allowable approach temperature 20 F '
            "for exchangers); targets from Muraki & Hayakawa (1982) Table 3 'Problem table for 5SP1'."
        ),
        source_url='https://www.jstage.jst.go.jp/article/jcej1968/15/2/15_2_136/_pdf/-char/ja',
        T_unit='F', Q_unit='Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 480, 250, 31500),
            ('H2', 'hot', 400, 150, 25200),
            ('C1', 'cold', 200, 400, 24700),
            ('C2', 'cold', 100, 400, 21600),
            ('C3', 'cold', 150, 360, 24500),
        ],
        targets=dict(Q_hot=3020000.0, Q_cold=0.0, pinch_hot=120.0, pinch_cold=100.0),
        published=dict(Q_hot=(3020000.0, 1.0), Q_cold=(0.0, 0.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'H1', 'C3', 5145000.0, 480.0, 316.66666666666663, 150.0, 360.0),
                ('above', 1, 'H2', 'C1', 4446000.0, 400.0, 223.57142857142858, 200.0, 380.0),
                ('above', 2, 'H1', 'C2', 2100000.0,
                 316.66666666666663, 249.99999999999994, 185.83333333333331, 283.05555555555554),
                ('above', 3, 'H2', 'C2', 1854000.0,
                 223.57142857142858, 150.0, 100.0, 185.83333333333331),
            ],
            hot_utility={'C1': 494000.0, 'C2': 2526000.0000000005},
            cold_utility={},
        ),
    ),
    dict(
        name='smith2005_exr18_1',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 18, Exercise 1, '
            'Table 18.6.'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2018.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 200, 100, 0.4),
            ('2', 'hot', 200, 100, 0.2),
            ('3', 'hot', 150, 60, 1.2),
            ('4', 'cold', 50, 140, 1.1),
            ('5', 'cold', 80, 120, 2.4),
        ],
        targets=dict(Q_hot=30.0, Q_cold=3.0, pinch_hot=90.0, pinch_cold=80.0),
        published=dict(Q_hot=None, Q_cold=None, pinch_hot=(90, 0.0), pinch_cold=(80, 0.0)),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, '2', '4', 9.0, 200.0, 155.0, 126.36363636363636, 134.54545454545453),
                ('above', 1, '3', '5', 72.0, 150.0, 90.0, 80.0, 110.0),
                ('above', 2, '1', '4', 40.0, 200.0, 100.0, 90.0, 126.36363636363636),
                ('above', 3, '2', '4', 11.0, 155.0, 100.0, 80.0, 90.0),
                ('below', 4, '3', '4', 33.0, 90.0, 62.5, 50.0, 80.0),
            ],
            hot_utility={'4': 6.000000000000015, '5': 24.0},
            cold_utility={'3': 3.0},
        ),
    ),
    dict(
        name='smith2005_exr17_2_dT10',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 17, Exercise 2, '
            'Table 17.8 (dTmin = 10 C row).'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2017.pdf',
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 160, 42, 75.0),
            ('2', 'hot', 100, 90, 1600.0),
            ('3', 'hot', 75, 40, 600.0),
            ('4', 'cold', 25, 140, 35.0),
            ('5', 'cold', 25, 79, 605.0),
            ('6', 'cold', 80, 150, 165.0),
        ],
        targets=dict(Q_hot=7150.0, Q_cold=4755.0, pinch_hot=100.0, pinch_cold=90.0),
        published=dict(Q_hot=(7150.0, 1.0), Q_cold=(4755.0, 1.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, '1', '6', 4500.0, 160.0, 100.0, 90.0, 117.27272727272728),
                ('below', 2, '1', '4', 2275.0, 100.0, 69.66666666666667, 25.0, 90.0),
                ('below', 2, '2', '6', 1650.0, 100.0, 98.96875, 80.0, 90.0),
                ('below', 3, '2', '5', 11670.0, 98.96875, 91.675, 59.710743801652896, 79.0),
                ('below', 4, '3', '5', 21000.0, 75.0, 40.0, 25.0, 59.710743801652896),
            ],
            hot_utility={'4': 1750.0, '6': 5399.999999999999},
            cold_utility={'1': 2075.0000000000005, '2': 2679.9999999999955},
        ),
    ),
    dict(
        name='7sp1_masso1969_dt10F',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '7sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Masso & Rudd (1969), AIChE J. 15, 10 (7SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/7sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/7sp1.dat)'
        ),
        T_unit='F', Q_unit='1e3 Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 440.0, 150.0, 28.0),
            ('H2', 'hot', 520.0, 300.0, 23.8),
            ('H3', 'hot', 390.0, 150.0, 33.6),
            ('C1', 'cold', 100.0, 430.0, 16.0),
            ('C2', 'cold', 180.0, 350.0, 32.76),
            ('C3', 'cold', 200.0, 400.0, 26.35),
            ('C4', 'cold', 350.0, 410.0, 19.84),
        ],
        targets=dict(Q_hot=0.0, Q_cold=4110.4, pinch_hot=520.0, pinch_cold=510.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(4110.4, 0.1), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'H1', 'C3', 5270.0, 440.0, 251.78571428571428, 200.0, 400.0),
                ('below', 1, 'H2', 'C4', 1190.4, 520.0, 469.98319327731093, 350.0, 410.0),
                ('below', 1, 'H3', 'C2', 5569.2, 390.0, 224.25, 180.0, 350.0),
                ('below', 2, 'H2', 'C1', 3011.428571428571,
                 469.98319327731093, 343.452581032413, 241.7857142857143, 430.0),
                ('below', 3, 'H1', 'C1', 2268.571428571429,
                 251.78571428571428, 170.76530612244898, 100.0, 241.7857142857143),
            ],
            hot_utility={},
            cold_utility={'H1': 581.4285714285713, 'H2': 1034.171428571429, 'H3': 2494.8},
        ),
    ),
    dict(
        name='smith2005_ex16_1_low_T_distillation',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 16, Example 16.1, '
            'Table 16.4 (low-temperature distillation process).'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2016.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=5,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1_feed', 'hot', 20, 0, 0.04),
            ('2_col1_cond', 'hot', -19, -20, 1.2),
            ('3_col2_cond', 'hot', -39, -40, 0.8),
            ('4_col1_reb', 'cold', 19, 20, 1.2),
            ('5_col2_reb', 'cold', -1, 0, 0.8),
            ('6_col2_bottoms', 'cold', 0, 20, 0.01),
            ('7_col2_overheads', 'cold', -40, 20, 0.01),
        ],
        targets=dict(Q_hot=1.8399999999999999, Q_cold=1.8399999999999999, pinch_hot=-19.0, pinch_cold=-24.0),
        published=dict(Q_hot=(1.84, 0.01), Q_cold=(1.84, 0.01), pinch_hot=(-19, 0.0), pinch_cold=(-24, 0.0)),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, '1_feed', '5_col2_reb', 0.64, 20.0, 4.0, -1.0, -0.20000000000000007),
                ('above', 2, '1_feed', '7_col2_overheads', 0.16000000000000003,
                 4.0, -8.881784197001252e-16, -24.0, -7.9999999999999964),
                ('below', 3, '2_col1_cond', '7_col2_overheads', 0.16,
                 -19.0, -19.133333333333333, -40.0, -24.0),
            ],
            hot_utility={'4_col1_reb': 1.2, '5_col2_reb': 0.16000000000000006, '6_col2_bottoms': 0.2, '7_col2_overheads': 0.27999999999999997},
            cold_utility={'2_col1_cond': 1.0400000000000005, '3_col2_cond': 0.8},
        ),
    ),
    dict(
        name='10sp1_pho1973_dt10F',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '10sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Pho & Lapidus (1973), AIChE J. 19, 1182 (10SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/10sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/10sp1.dat)'
        ),
        T_unit='F', Q_unit='Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 320.0, 200.0, 16670.0),
            ('H2', 'hot', 480.0, 280.0, 20000.0),
            ('H3', 'hot', 440.0, 150.0, 28000.0),
            ('H4', 'hot', 520.0, 300.0, 23800.0),
            ('H5', 'hot', 390.0, 150.0, 33600.0),
            ('C1', 'cold', 140.0, 320.0, 14450.0),
            ('C2', 'cold', 240.0, 431.0, 11530.0),
            ('C3', 'cold', 100.0, 430.0, 16000.0),
            ('C4', 'cold', 180.0, 350.0, 32760.0),
            ('C5', 'cold', 200.0, 400.0, 26350.0),
        ],
        targets=dict(Q_hot=0.0, Q_cold=6497970.0, pinch_hot=520.0, pinch_cold=510.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(6497970.0, 1.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'H2', 'C3', 4000000.0, 480.0, 280.0, 180.0, 430.0),
                ('below', 1, 'H3', 'C5', 5270000.0, 440.0, 251.78571428571428, 200.0, 400.0),
                ('below', 1, 'H4', 'C2', 2202230.0, 520.0, 427.46932773109245, 240.0, 431.0),
                ('below', 1, 'H5', 'C4', 5569200.0, 390.0, 224.25, 180.0, 350.0),
                ('below', 2, 'H3', 'C3', 1280000.0,
                 251.78571428571428, 206.07142857142856, 100.0, 180.0),
                ('below', 2, 'H4', 'C1', 2601000.0,
                 427.46932773109245, 318.18361344537817, 140.0, 320.0),
            ],
            hot_utility={},
            cold_utility={'H1': 2000400.0, 'H3': 1569999.9999999995, 'H4': 432770.00000000047, 'H5': 2494800.0},
        ),
    ),
    dict(
        name='nitric_acid_ph6c5',
        kind='constant_cp',
        citation=(
            'Caballero, J.A., Pavao, L.V., Costa, C.B.B. & Ravagnani, M.A.S.S. (2021). A Novel '
            'Sequential Approach for the Design of Heat Exchanger Networks. Front. Chem. Eng. '
            '3:733186, Supplementary Data Sheet 1, Table S19 (Example 10, Nitric Acid Plant PH6C5); '
            'original problem: Castillo, E., Acevedo, L. & Reverberi, A.P. (1998). Cleaner production'
            ' of nitric acid by heat transfer optimization: a case study. Chem. Biochem. Eng. Q. 12, '
            '157-165.'
        ),
        source_url=(
            'https://public-pages-files-2025.frontiersin.org/articles/733186/file/Data_Sheet_1.docx/7'
            '33186_supplementary-materials_datasheets_1_docx/1'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 840.0, 40.0, 4.9894),
            ('H2', 'hot', 76.0, 45.0, 4.684),
            ('H3', 'hot', 50.0, 40.0, 0.772),
            ('H4', 'hot', 180.0, 77.0, 0.6097),
            ('H5', 'hot', 180.0, 179.0, 292.7),
            ('H6', 'hot', 90.0, 45.0, 3.066),
            ('C1', 'cold', 24.0, 25.0, 329.8),
            ('C2', 'cold', 25.0, 70.0, 0.5383),
            ('C3', 'cold', 35.0, 122.0, 3.727),
            ('C4', 'cold', 90.0, 180.0, 0.6097),
            ('C5', 'cold', 180.0, 181.0, 2581.1),
        ],
        targets=dict(Q_hot=0.0, Q_cold=1323.6676, pinch_hot=840.0, pinch_cold=830.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(1323.67, 0.01), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'H1', 'C1', 329.8, 840.0, 773.8998677195655, 24.0, 25.0),
                ('below', 1, 'H4', 'C2', 24.2235, 180.0, 140.26980482204362, 25.0, 70.0),
                ('below', 2, 'H1', 'C5', 2581.1,
                 773.8998677195655, 256.5831562913377, 180.0, 181.0),
                ('below', 3, 'H1', 'C3', 324.24899999999997,
                 256.5831562913377, 191.59558263518664, 35.0, 122.0),
                ('below', 4, 'H1', 'C4', 54.873000000000005,
                 191.59558263518664, 180.59766705415487, 90.0, 180.0),
            ],
            hot_utility={},
            cold_utility={'H1': 701.4980000000003, 'H2': 145.204, 'H3': 7.720000000000001, 'H4': 38.5756, 'H5': 292.7, 'H6': 137.97},
        ),
    ),
    dict(
        name='12sp1_sargent1978_dt10F',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '12sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Grossmann & Sargent (1978), Comput. Chem. Eng. 2, 1 (12SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/12sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/12sp1.dat)'
        ),
        T_unit='F', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 668.0, 200.0, 402.678),
            ('H2', 'hot', 596.0, 258.0, 222.318),
            ('H3', 'hot', 434.0, 240.0, 341.26),
            ('H4', 'hot', 301.0, 220.0, 553.66),
            ('H5', 'hot', 490.0, 150.0, 37.725),
            ('H6', 'hot', 460.0, 415.0, 90.645),
            ('H7', 'hot', 405.0, 150.0, 79.546),
            ('H8', 'hot', 323.0, 290.0, 92.034),
            ('H9', 'hot', 290.0, 150.0, 85.303),
            ('C1', 'cold', 40.0, 240.0, 620.198),
            ('C2', 'cold', 240.0, 695.0, 822.659),
            ('C3', 'cold', 108.0, 354.0, 138.299),
        ],
        targets=dict(Q_hot=105554.014, Q_cold=0.0, pinch_hot=50.0, pinch_cold=40.0),
        published=dict(Q_hot=(105554.014, 0.001), Q_cold=(0.0, 0.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=True,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'H1', 'C2', 72020.90585006942,
                 668.0, 489.145168471907, 479.145168471907, 566.6916559595167),
                ('above', 1, 'H6', 'C1', 4079.0249999999996,
                 460.0, 415.0, 233.42302780724864, 240.0),
                ('above', 2, 'H2', 'C2', 32552.971632738256,
                 596.0, 449.57473694105624, 439.57473694105613, 479.145168471907),
                ('above', 2, 'H8', 'C1', 3037.1220000000003,
                 323.0, 290.0, 228.52600782330802, 233.42302780724864),
                ('above', 3, 'H1', 'C2', 31211.806036767786,
                 489.145168471907, 411.6345867247845, 401.6345867247845, 439.57473694105613),
                ('above', 3, 'H4', 'C1', 44846.46,
                 301.0, 220.0, 156.21610034214876, 228.52600782330802),
                ('above', 4, 'H3', 'C2', 13042.984662183188,
                 434.0, 395.77991952709607, 385.77991952709607, 401.6345867247845),
                ('above', 4, 'H5', 'C1', 12826.5,
                 490.0, 150.0, 135.534801789106, 156.21610034214876),
                ('above', 5, 'H2', 'C2', 16388.413516000743,
                 449.57473694105624, 375.85864775349273, 365.85864775349273, 385.77991952709607),
                ('above', 6, 'H2', 'C3', 22079.134000000005,
                 375.85864775349273, 276.545348785348, 194.35217897454066, 354.0),
                ('above', 6, 'H3', 'C2', 10049.229695092426,
                 395.77991952709607, 366.33249030863385, 353.6431007345062, 365.85864775349273),
                ('above', 7, 'H1', 'C2', 37854.046422945736,
                 411.6345867247845, 317.62883914745044, 307.62883914745044, 353.6431007345062),
                ('above', 7, 'H2', 'C1', 4122.964851261007,
                 276.545348785348, 257.99999999999994, 128.88698149419858, 135.534801789106),
                ('above', 7, 'H9', 'C3', 11942.42, 290.0, 150.0, 108.0, 194.35217897454066),
                ('above', 8, 'H3', 'C2', 28402.827493985387,
                 366.33249030863385, 283.1032003420823, 273.10320034208223, 307.62883914745044),
                ('above', 8, 'H7', 'C1', 20284.230000000003,
                 405.0, 149.99999999999997, 96.18092633117, 128.88698149419858),
                ('above', 9, 'H1', 'C2', 27232.645690217043,
                 317.62883914745044, 250.0, 240.0, 273.10320034208223),
                ('above', 10, 'H1', 'C1', 20133.90000000001,
                 250.0, 199.99999999999997, 63.71726150155106, 96.18092633117),
                ('above', 11, 'H3', 'C1', 14709.398148738961,
                 283.1032003420823, 240.0000000000001, 40.0, 63.71726150155106),
            ],
            hot_utility={'C2': 105554.014},
            cold_utility={},
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='14sp1_sargent1978_dt10F',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '14sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Grossmann & Sargent (1978), Comput. Chem. Eng. 2, 1 (14SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/14sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/14sp1.dat)'
        ),
        T_unit='F', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 480.0, 300.0, 18.4),
            ('H2', 'hot', 460.0, 200.0, 24.18),
            ('H3', 'hot', 450.0, 300.0, 20.8),
            ('H4', 'hot', 400.0, 190.0, 18.2),
            ('H5', 'hot', 390.0, 250.0, 28.22),
            ('H6', 'hot', 320.0, 125.0, 19.17),
            ('H7', 'hot', 280.0, 125.0, 27.2),
            ('C1', 'cold', 150.0, 430.0, 13.3),
            ('C2', 'cold', 180.0, 400.0, 13.02),
            ('C3', 'cold', 220.0, 400.0, 27.2),
            ('C4', 'cold', 110.0, 350.0, 22.2),
            ('C5', 'cold', 140.0, 390.0, 28.16),
            ('C6', 'cold', 130.0, 280.0, 15.6),
            ('C7', 'cold', 100.0, 240.0, 13.05),
        ],
        targets=dict(Q_hot=0.0, Q_cold=426.34999999999934, pinch_hot=480.0, pinch_cold=470.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(426.35, 0.01), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'H1', 'C3', 3311.9999999999995,
                 480.0, 300.0, 278.2352941176471, 400.0),
                ('below', 1, 'H2', 'C2', 1902.3638014718056,
                 460.0, 381.32490481919746, 253.88910894993813, 400.0),
                ('below', 1, 'H3', 'C1', 531.9999999999998, 450.0, 424.4230769230769, 390.0, 430.0),
                ('below', 1, 'H5', 'C4', 1665.5934782608713,
                 390.0, 330.97826086956513, 274.9732667450058, 350.0),
                ('below', 1, 'H6', 'C6', 147.74898108684283,
                 320.0, 312.2926979088762, 270.5289114687921, 280.0),
                ('below', 1, 'H7', 'C7', 438.21623656168276,
                 280.0, 263.88910894993813, 206.42021175772544, 239.99999999999997),
                ('below', 2, 'H2', 'C7', 619.2948941803672,
                 381.32490481919746, 355.71303988204414, 158.96466431095402, 206.42021175772544),
                ('below', 2, 'H3', 'C5', 1943.652173913045,
                 424.4230769230769, 330.97826086956513, 320.97826086956513, 390.0),
                ('below', 2, 'H4', 'C1', 2414.0,
                 400.0, 267.3626373626373, 208.49624060150376, 390.0),
                ('below', 2, 'H6', 'C4', 608.9117860564119,
                 312.2926979088762, 280.52891146879216, 247.5448079136359, 274.9732667450058),
                ('below', 2, 'H7', 'C2', 962.0361985281945,
                 263.88910894993813, 228.52013106287217, 180.0, 253.88910894993813),
                ('below', 3, 'H2', 'C3', 1584.0000000000005,
                 355.71303988204414, 290.2043550185206, 220.0, 278.2352941176471),
                ('below', 3, 'H3', 'C4', 644.3478260869551,
                 330.97826086956513, 300.0, 218.52013106287217, 247.5448079136359),
                ('below', 3, 'H5', 'C5', 1507.2065217391284,
                 330.97826086956513, 277.56909992912824, 267.4553020009881, 320.97826086956513),
                ('below', 3, 'H6', 'C6', 2192.251018913157,
                 280.52891146879216, 166.17048586038544, 130.0, 270.5289114687921),
                ('below', 4, 'H2', 'C5', 2181.1413043478265,
                 290.2043550185206, 200.00000000000006, 190.0, 267.4553020009881),
                ('below', 4, 'H5', 'C1', 778.0,
                 277.56909992912824, 249.99999999999997, 150.0, 208.49624060150376),
                ('below', 4, 'H7', 'C4', 1619.908695652173,
                 228.52013106287217, 168.96466431095405, 145.55127089835986, 218.52013106287217),
                ('below', 5, 'H4', 'C5', 1407.9999999999998,
                 267.3626373626373, 189.99999999999994, 140.0, 190.0),
                ('below', 5, 'H6', 'C4', 789.2382139435887,
                 166.17048586038544, 125.00000000000001, 110.0, 145.55127089835986),
                ('below', 5, 'H7', 'C7', 769.48886925795,
                 168.96466431095405, 140.6746323529412, 100.0, 158.96466431095402),
            ],
            hot_utility={},
            cold_utility={'H7': 426.3500000000003},
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='bandar_imam_aromatics_ph6c10',
        kind='constant_cp',
        citation=(
            'Caballero, J.A., Pavao, L.V., Costa, C.B.B. & Ravagnani, M.A.S.S. (2021). A Novel '
            'Sequential Approach for the Design of Heat Exchanger Networks. Front. Chem. Eng. '
            '3:733186, Supplementary Data Sheet 1, Table S23 (Example 12, PH6C10); original problem: '
            'Khorasany, R.M. & Fesanghary, M. (2009). A novel approach for synthesis of cost-optimal '
            'heat exchanger networks. Comput. Chem. Eng. 33, 1363-1370 (Bandar Imam aromatics plant).'
        ),
        source_url=(
            'https://public-pages-files-2025.frontiersin.org/articles/733186/file/Data_Sheet_1.docx/7'
            '33186_supplementary-materials_datasheets_1_docx/1'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 385.0, 159.0, 131.51),
            ('H2', 'hot', 516.0, 43.0, 1198.96),
            ('H3', 'hot', 132.0, 82.0, 378.52),
            ('H4', 'hot', 91.0, 60.0, 589.545),
            ('H5', 'hot', 217.0, 43.0, 186.216),
            ('H6', 'hot', 649.0, 43.0, 116.0),
            ('C1', 'cold', 30.0, 385.0, 119.1),
            ('C2', 'cold', 99.0, 471.0, 191.05),
            ('C3', 'cold', 437.0, 521.0, 377.91),
            ('C4', 'cold', 78.0, 418.6, 160.43),
            ('C5', 'cold', 217.0, 234.0, 1297.7),
            ('C6', 'cold', 256.0, 266.0, 2753.0),
            ('C7', 'cold', 49.0, 149.0, 197.39),
            ('C8', 'cold', 59.0, 163.4, 123.156),
            ('C9', 'cold', 163.0, 649.0, 95.98),
            ('C10', 'cold', 219.0, 221.3, 1997.5),
        ],
        targets=dict(Q_hot=3965.790000000002, Q_cold=407528.69459999993, pinch_hot=516.0, pinch_cold=506.0),
        published=None,
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'H6', 'C9', 12765.34, 649.0, 538.9539655172414, 506.0, 639.0),
                ('above', 2, 'H6', 'C3', 2662.660000000001,
                 538.9539655172414, 516.0, 506.0, 513.0457516339869),
                ('below', 3, 'H2', 'C3', 26075.79, 516.0, 494.25132614932943, 437.0, 506.0),
                ('below', 3, 'H6', 'C9', 12573.38, 516.0, 407.6087931034483, 375.0, 506.0),
                ('below', 4, 'H1', 'C9', 20347.76, 385.0, 230.27594859706485, 163.0, 375.0),
                ('below', 4, 'H2', 'C8', 464.92639999999847,
                 494.25132614932943, 493.86355141122306, 159.62489850271203, 163.4),
                ('below', 4, 'H6', 'C1', 42280.5,
                 407.6087931034483, 43.12172413793104, 30.0, 385.0),
                ('below', 5, 'H2', 'C2', 24063.237599999968,
                 493.86355141122306, 473.7934593314206, 345.0474347029575, 471.0),
                ('below', 5, 'H5', 'C8', 9683.128000000002,
                 217.0, 165.0005584912145, 81.0, 159.62489850271203),
                ('below', 6, 'H2', 'C6', 27530.0,
                 473.7934593314206, 450.8318926402883, 256.0, 266.0),
                ('below', 6, 'H4', 'C8', 2709.4320000000002, 91.0, 86.40419815281277, 59.0, 81.0),
                ('below', 6, 'H5', 'C7', 19739.0,
                 165.0005584912145, 58.999999999999986, 49.0, 149.0),
                ('below', 7, 'H2', 'C10', 4594.250000000023,
                 450.8318926402883, 447.00003002602256, 219.0, 221.3),
                ('below', 8, 'H2', 'C5', 22060.9, 447.00003002602256, 428.6, 217.0, 234.0),
                ('below', 9, 'H2', 'C4', 54642.458000000006, 428.6, 383.0251201040902, 78.0, 418.6),
                ('below', 10, 'H2', 'C2', 47007.36240000004,
                 383.0251201040902, 343.81833889370785, 99.0, 345.0474347029575),
            ],
            hot_utility={'C3': 3005.9900000000016, 'C9': 959.8000000000001},
            cold_utility={'H1': 9373.499999999998, 'H2': 360669.1556, 'H3': 18926.0, 'H4': 15566.463000000002, 'H5': 2979.4559999999974, 'H6': 14.120000000000573},
        ),
    ),
    dict(
        name='20sp1_sargent1978_dt10F',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '20sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Grossmann & Sargent (1978), Comput. Chem. Eng. 2, 1 (20SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/20sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/20sp1.dat)'
        ),
        T_unit='F', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 550.0, 450.0, 28.4),
            ('H2', 'hot', 520.0, 480.0, 23.87),
            ('H3', 'hot', 500.0, 440.0, 24.6),
            ('H4', 'hot', 460.0, 370.0, 17.0),
            ('H5', 'hot', 415.0, 340.0, 30.69),
            ('H6', 'hot', 400.0, 300.0, 19.36),
            ('H7', 'hot', 365.0, 320.0, 25.5),
            ('H8', 'hot', 300.0, 250.0, 12.42),
            ('H9', 'hot', 240.0, 170.0, 20.72),
            ('H10', 'hot', 170.0, 140.0, 18.9),
            ('C1', 'cold', 375.0, 420.0, 22.32),
            ('C2', 'cold', 300.0, 370.0, 23.87),
            ('C3', 'cold', 320.0, 360.0, 34.96),
            ('C4', 'cold', 280.0, 340.0, 26.04),
            ('C5', 'cold', 260.0, 330.0, 13.44),
            ('C6', 'cold', 225.0, 280.0, 32.0),
            ('C7', 'cold', 240.0, 265.0, 11.1),
            ('C8', 'cold', 170.0, 220.0, 22.96),
            ('C9', 'cold', 140.0, 200.0, 14.4),
            ('C10', 'cold', 80.0, 140.0, 13.92),
        ],
        targets=dict(Q_hot=0.0, Q_cold=3362.850000000001, pinch_hot=550.0, pinch_cold=540.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(3362.85, 0.01), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'H1', 'C1', 1004.4, 550.0, 514.6338028169014, 375.0, 420.0),
                ('below', 1, 'H10', 'C10', 567.0, 170.0, 140.0, 99.26724137931035, 140.0),
                ('below', 1, 'H2', 'C9', 864.0, 520.0, 483.8039379974864, 140.0, 200.0),
                ('below', 1, 'H3', 'C3', 1398.4, 500.0, 443.1544715447154, 320.0, 360.0),
                ('below', 1, 'H4', 'C7', 277.5, 460.0, 443.6764705882353, 240.0, 265.0),
                ('below', 1, 'H5', 'C4', 1562.3999999999999,
                 415.0, 364.0909090909091, 280.0, 340.0),
                ('below', 1, 'H6', 'C6', 1760.0, 400.0, 309.0909090909091, 225.0, 280.0),
                ('below', 1, 'H9', 'C8', 1148.0, 240.0, 184.59459459459458, 170.0, 220.0),
                ('below', 2, 'H1', 'C2', 1670.9,
                 514.6338028169014, 455.7992957746479, 300.0, 370.0),
                ('below', 2, 'H4', 'C5', 940.8, 443.6764705882353, 388.3352941176471, 260.0, 330.0),
                ('below', 2, 'H9', 'C10', 190.60000000000014,
                 184.59459459459458, 175.39575289575288, 85.57471264367815, 99.26724137931035),
                ('below', 3, 'H3', 'C10', 77.59999999999991,
                 443.1544715447154, 440.0, 80.0, 85.57471264367815),
            ],
            hot_utility={},
            cold_utility={'H1': 164.69999999999962, 'H2': 90.80000000000038, 'H4': 311.7000000000006, 'H5': 739.3500000000009, 'H6': 176.0000000000006, 'H7': 1147.5, 'H8': 621.0, 'H9': 111.79999999999963},
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='sorak_kravanja_ph13c7',
        kind='constant_cp',
        citation=(
            'Caballero, J.A., Pavao, L.V., Costa, C.B.B. & Ravagnani, M.A.S.S. (2021). A Novel '
            'Sequential Approach for the Design of Heat Exchanger Networks. Front. Chem. Eng. '
            '3:733186, Supplementary Data Sheet 1, Table S25 (Example 13, PH13C7); original problem: '
            'Sorak, A. & Kravanja, Z. (2001). Simultaneous MINLP synthesis of heat exchanger networks'
            ' comprising different exchanger types. Comput.-Aided Chem. Eng. 9, 1095-1100.'
        ),
        source_url=(
            'https://public-pages-files-2025.frontiersin.org/articles/733186/file/Data_Sheet_1.docx/7'
            '33186_supplementary-materials_datasheets_1_docx/1'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 576.0, 437.0, 23.1),
            ('H2', 'hot', 599.0, 399.0, 15.22),
            ('H3', 'hot', 530.0, 382.0, 15.15),
            ('H4', 'hot', 449.0, 237.0, 14.76),
            ('H5', 'hot', 368.0, 177.0, 10.7),
            ('H6', 'hot', 121.0, 114.0, 149.6),
            ('H7', 'hot', 202.0, 185.0, 258.2),
            ('H8', 'hot', 185.0, 113.0, 8.38),
            ('H9', 'hot', 140.0, 120.0, 59.89),
            ('H10', 'hot', 69.0, 66.0, 165.79),
            ('H11', 'hot', 120.0, 68.0, 8.74),
            ('H12', 'hot', 67.0, 35.0, 7.62),
            ('H13', 'hot', 1034.5, 576.0, 21.3),
            ('C1', 'cold', 123.0, 343.0, 10.61),
            ('C2', 'cold', 20.0, 156.0, 6.65),
            ('C3', 'cold', 156.0, 157.0, 3291.0),
            ('C4', 'cold', 20.0, 182.0, 26.63),
            ('C5', 'cold', 182.0, 318.0, 31.19),
            ('C6', 'cold', 318.0, 320.0, 4011.83),
            ('C7', 'cold', 322.0, 923.78, 17.6),
        ],
        targets=dict(Q_hot=1831.0679999999998, Q_cold=0.0, pinch_hot=30.0, pinch_cold=20.0),
        published=dict(Q_hot=(1831.1, 0.1), Q_cold=(0.0, 0.0), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'H13', 'C7', 9766.050000000001,
                 1034.5, 576.0, 322.0, 876.8892045454546),
                ('above', 1, 'H2', 'C6', 3044.0,
                 599.0, 399.0, 319.0336355229409, 319.7923915021325),
                ('above', 1, 'H3', 'C5', 1725.766170027145,
                 530.0, 416.08804158236666, 262.6692475143589, 318.0),
                ('above', 1, 'H4', 'C1', 1882.83,
                 449.0, 321.4369918699187, 165.54194156456174, 343.0),
                ('above', 1, 'H7', 'C4', 1452.88,
                 202.0, 196.3730441518203, 127.4419827262486, 182.0),
                ('above', 2, 'H3', 'C6', 516.4338299728552,
                 416.08804158236666, 382.0, 318.9049077777541, 319.0336355229409),
                ('above', 2, 'H4', 'C5', 1246.29,
                 321.4369918699187, 237.0, 222.71124815559008, 262.6692475143589),
                ('above', 2, 'H8', 'C1', 377.1, 185.0, 140.0, 130.0, 165.54194156456174),
                ('above', 3, 'H5', 'C6', 419.4361700271447,
                 368.0, 328.80035794138837, 318.80035794138837, 318.9049077777541),
                ('above', 3, 'H8', 'C2', 226.26, 140.0, 113.0, 95.97593984962407, 130.0),
                ('above', 3, 'H9', 'C1', 74.27, 140.0, 138.7598931374186, 123.0, 130.0),
                ('above', 4, 'H1', 'C6', 3210.9, 576.0, 437.0, 318.0, 318.80035794138837),
                ('above', 4, 'H11', 'C2', 401.0208336462637,
                 120.0, 74.1166094226243, 35.672055090787424, 95.97593984962407),
                ('above', 4, 'H5', 'C5', 1133.3885828775797,
                 328.80035794138837, 222.8761913173155, 186.3730441518203, 222.71124815559008),
                ('above', 4, 'H9', 'C4', 1123.53,
                 138.7598931374186, 120.0, 85.25159594442358, 127.4419827262486),
                ('above', 5, 'H5', 'C3', 490.8752470952754,
                 222.8761913173155, 177.00000000000003, 156.850843133669, 157.0),
                ('above', 5, 'H6', 'C4', 1047.2,
                 121.0, 114.0, 45.927525347352606, 85.25159594442358),
                ('above', 5, 'H7', 'C5', 136.39524709527495,
                 196.3730441518203, 195.84478990280684, 182.0, 186.3730441518203),
                ('above', 6, 'H11', 'C4', 53.45916635373629,
                 74.1166094226243, 68.0, 43.92004632543235, 45.927525347352606),
                ('above', 6, 'H7', 'C3', 2800.1247529047246,
                 195.84478990280684, 185.0, 156.0, 156.850843133669),
                ('above', 7, 'H12', 'C4', 139.6208336462636,
                 67.0, 48.67705595193391, 38.67705595193391, 43.92004632543235),
                ('above', 8, 'H10', 'C4', 497.37, 69.0, 66.0, 20.0, 38.67705595193391),
                ('above', 8, 'H12', 'C2', 104.2191663537364,
                 48.67705595193391, 35.0, 20.0, 35.672055090787424),
            ],
            hot_utility={'C2': 172.9, 'C6': 832.8899999998175, 'C7': 825.2779999999983},
            cold_utility={},
        ),
    ),
    dict(
        name='23sp1_mocsny1984_dt10K',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '23sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Mocsny, D.; Govind, R. (1984). Decomposition strategy for the '
            'synthesis of minimum-unit heat exchanger networks. AIChE J. 30, 853 (23SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/23sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/23sp1.dat)'
        ),
        T_unit='K', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 433.0, 366.0, 8.79),
            ('H2', 'hot', 522.0, 411.0, 10.55),
            ('H3', 'hot', 510.0, 349.0, 14.77),
            ('H4', 'hot', 544.0, 422.0, 12.56),
            ('H5', 'hot', 472.0, 339.0, 17.73),
            ('H6', 'hot', 505.0, 339.0, 14.77),
            ('H7', 'hot', 544.0, 420.0, 12.56),
            ('H8', 'hot', 475.0, 339.0, 17.73),
            ('H9', 'hot', 583.0, 478.0, 12.53),
            ('H10', 'hot', 517.0, 366.0, 8.32),
            ('H11', 'hot', 511.0, 339.0, 6.96),
            ('C1', 'cold', 323.0, 423.0, 7.62),
            ('C2', 'cold', 389.0, 495.0, 6.08),
            ('C3', 'cold', 311.0, 494.0, 8.44),
            ('C4', 'cold', 355.0, 450.0, 17.28),
            ('C5', 'cold', 366.0, 478.0, 13.9),
            ('C6', 'cold', 311.0, 490.0, 8.44),
            ('C7', 'cold', 350.0, 450.0, 17.28),
            ('C8', 'cold', 352.0, 468.0, 13.9),
            ('C9', 'cold', 366.0, 478.0, 8.44),
            ('C10', 'cold', 311.0, 489.0, 8.44),
            ('C11', 'cold', 422.0, 478.0, 21.78),
            ('C12', 'cold', 339.0, 411.0, 13.84),
        ],
        targets=dict(Q_hot=0.0, Q_cold=2553.6700000000005, pinch_hot=583.0, pinch_cold=573.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(2553.67, 0.01), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'H10', 'C3', 119.89027127338659,
                 517.0, 502.59011162579486, 479.79499155528595, 494.0),
                ('below', 1, 'H2', 'C6', 44.93838277524462,
                 522.0, 517.7404376516356, 484.67554706454445, 489.99999999999994),
                ('below', 1, 'H3', 'C11', 457.2900000000001,
                 510.0, 479.039268788084, 457.004132231405, 478.0),
                ('below', 1, 'H4', 'C9', 464.2, 544.0, 507.04140127388536, 423.0, 478.0),
                ('below', 1, 'H5', 'C4', 1452.5454795115836,
                 472.0, 390.07414103149557, 365.9406551208574, 449.99999999999994),
                ('below', 1, 'H7', 'C5', 619.5151288693213,
                 544.0, 494.6755470645445, 433.43056626839416, 478.0),
                ('below', 1, 'H9', 'C10', 486.94539701184476,
                 583.0, 544.1376379080731, 431.3050477474118, 489.0),
                ('below', 2, 'H1', 'C9', 481.08, 433.0, 378.26962457337885, 366.0, 423.0),
                ('below', 2, 'H11', 'C3', 841.643978420791,
                 511.0, 390.0741410314955, 380.07414103149557, 479.79499155528595),
                ('below', 2, 'H2', 'C1', 762.0,
                 517.7404376516356, 445.51294949997686, 323.0, 423.0),
                ('below', 2, 'H3', 'C7', 557.3344447707283,
                 479.039268788084, 441.3050477474118, 417.7468492609532, 449.99999999999994),
                ('below', 2, 'H4', 'C12', 334.321550894797,
                 507.04140127388536, 480.4234434000958, 386.84381857696553, 411.0),
                ('below', 2, 'H7', 'C6', 937.9248711306788,
                 494.6755470645445, 420.0, 373.5470078310517, 484.67554706454445),
                ('below', 2, 'H8', 'C11', 559.7260600613718,
                 475.0, 443.43056626839416, 431.3050477474118, 457.004132231405),
                ('below', 3, 'H2', 'C6', 109.84041830404057,
                 445.51294949997686, 435.1015354427218, 360.5327402594829, 373.5470078310517),
                ('below', 3, 'H3', 'C10', 1015.3746029881552,
                 441.3050477474118, 372.5593061774622, 311.0, 431.3050477474118),
                ('below', 3, 'H4', 'C8', 315.7421213151672,
                 480.4234434000958, 455.28473947372896, 445.28473947372896, 468.0),
                ('below', 3, 'H5', 'C3', 250.58670519561386,
                 390.07414103149557, 375.94065512085746, 350.3837731173233, 380.07414103149557),
                ('below', 3, 'H6', 'C7', 909.3805362158186,
                 505.0, 443.43056626839416, 365.1206608225378, 417.7468492609532),
                ('below', 3, 'H8', 'C11', 202.66393993862823,
                 443.43056626839416, 432.0, 422.0, 431.3050477474118),
                ('below', 3, 'H9', 'C12', 490.3946029881551,
                 544.1376379080731, 505.00000000000006, 351.4106825229081, 386.84381857696553),
                ('below', 4, 'H3', 'C12', 171.76384611704793,
                 372.5593061774622, 360.93006811943593, 339.0, 351.4106825229081),
                ('below', 4, 'H4', 'C6', 418.05632779003605,
                 455.28473947372896, 422.0, 311.0, 360.5327402594829),
                ('below', 4, 'H5', 'C4', 14.538498909207476,
                 375.94065512085746, 375.1206608225378, 365.09930680435235, 365.9406551208574),
                ('below', 4, 'H6', 'C5', 937.2848711306787,
                 443.43056626839416, 379.9718749257619, 366.0, 433.43056626839416),
                ('below', 4, 'H9', 'C2', 14.652121315167165,
                 505.00000000000006, 503.83063676654695, 492.59011162579486, 495.0),
                ('below', 5, 'H10', 'C2', 629.8278786848329,
                 502.59011162579486, 426.88964543771397, 389.0, 492.59011162579486),
                ('below', 5, 'H11', 'C4', 174.51602157920905,
                 390.0741410314955, 364.99999999999994, 355.0, 365.09930680435235),
                ('below', 5, 'H5', 'C7', 261.28501901345294,
                 375.1206608225378, 360.3837731173233, 350.0, 365.1206608225378),
                ('below', 5, 'H9', 'C8', 323.65787868483284,
                 503.83063676654695, 478.0, 422.0, 445.28473947372896),
                ('below', 6, 'H5', 'C3', 332.3990451102086,
                 360.3837731173233, 341.6359420338372, 311.0, 350.3837731173233),
                ('below', 6, 'H8', 'C8', 973.0, 432.0, 377.1212633953751, 352.0, 422.0),
            ],
            hot_utility={},
            cold_utility={'H1': 107.85000000000007, 'H2': 254.27119892071497, 'H3': 176.20710612406867, 'H5': 46.73525225993324, 'H6': 605.154592653503, 'H8': 675.8900000000002, 'H10': 506.6018500417802, 'H11': 180.9599999999996},
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='fs_28sp_as1',
        kind='constant_cp',
        citation=(
            'Furman, K.C. & Sahinidis, N.V. (2004). Approximation algorithms for the minimum number '
            'of matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565, test instance '28sp-as1' (original source per the test set's references.txt: "
            "'smith:1989' (Ahmad & Smith, 1989)); digitized in Letsios, D., Kouyialis, G. & Misener, "
            'R. (2018). Heuristics with performance guarantees for the minimum number of matches '
            'problem in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (data repository '
            'github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/furman_sahinidis/dat_files/28sp-as1.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS1', 'hot', 230, 80, 30),
            ('HS2', 'hot', 200, 40, 45),
            ('HS3', 'hot', 110, 45, 0.1),
            ('HS4', 'hot', 115, 40, 0.1),
            ('HS5', 'hot', 105, 40, 0.1),
            ('HS6', 'hot', 110, 42, 0.1),
            ('HS7', 'hot', 117, 48, 0.1),
            ('HS8', 'hot', 103, 50, 0.12),
            ('HS9', 'hot', 110, 45, 0.1),
            ('HS10', 'hot', 115, 40, 0.1),
            ('HS11', 'hot', 105, 40, 0.1),
            ('HS12', 'hot', 110, 42, 0.1),
            ('HS13', 'hot', 117, 48, 0.1),
            ('HS14', 'hot', 103, 50, 0.1),
            ('HS15', 'hot', 115, 42, 0.1),
            ('HS16', 'hot', 117, 43, 0.1),
            ('CS1', 'cold', 40, 180, 40),
            ('CS2', 'cold', 140, 280, 60),
            ('CS3', 'cold', 170, 270, 0.1),
            ('CS4', 'cold', 175, 265, 0.1),
            ('CS5', 'cold', 180, 275, 0.1),
            ('CS6', 'cold', 168, 277, 0.1),
            ('CS7', 'cold', 181, 267, 0.1),
            ('CS8', 'cold', 170, 270, 0.1),
            ('CS9', 'cold', 175, 265, 0.1),
            ('CS10', 'cold', 180, 275, 0.1),
            ('CS11', 'cold', 168, 277, 0.1),
            ('CS12', 'cold', 181, 267, 0.1),
        ],
        targets=dict(Q_hot=5446.0, Q_cold=3144.7599999999993, pinch_hot=150.0, pinch_cold=140.0),
        published=dict(Q_hot=(5446.0, 1.0), Q_cold=(3144.76, 0.01), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('above', 1, 'HS1', 'CS2', 1275.0, 230.0, 187.5, 177.5, 198.75),
                ('above', 2, 'HS1', 'CS1', 1125.0, 187.5, 150.0, 140.0, 168.125),
                ('above', 2, 'HS2', 'CS2', 2250.0, 200.0, 150.0, 140.0, 177.5),
                ('below', 3, 'HS2', 'CS1', 4000.0, 150.0, 61.111111111111114, 40.0, 140.0),
            ],
            hot_utility={'CS1': 475.0, 'CS2': 4875.0, 'CS3': 10.0, 'CS4': 9.0, 'CS5': 9.5, 'CS6': 10.9, 'CS7': 8.6, 'CS8': 10.0, 'CS9': 9.0, 'CS10': 9.5, 'CS11': 10.9, 'CS12': 8.6},
            cold_utility={'HS1': 2100.0, 'HS2': 950.0000000000001, 'HS3': 6.5, 'HS4': 7.5, 'HS5': 6.5, 'HS6': 6.800000000000001, 'HS7': 6.9, 'HS8': 6.359999999999999, 'HS9': 6.5, 'HS10': 7.5, 'HS11': 6.5, 'HS12': 6.800000000000001, 'HS13': 6.9, 'HS14': 5.300000000000001, 'HS15': 7.300000000000001, 'HS16': 7.4},
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='fs_37sp_yfyv',
        kind='constant_cp',
        citation=(
            'Furman, K.C. & Sahinidis, N.V. (2004). Approximation algorithms for the minimum number '
            'of matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565, test instance '37sp-yfyv' (original source per the test set's references.txt:"
            " 'yu:2000' (Yu, Fang, Yao & Vian, 2000)); digitized in Letsios, D., Kouyialis, G. & "
            'Misener, R. (2018). Heuristics with performance guarantees for the minimum number of '
            'matches problem in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (data '
            'repository github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/furman_sahinidis/dat_files/37sp-yfyv.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS1', 'hot', 175.0, 150.0, 4923),
            ('HS2', 'hot', 168.0, 55.0, 767),
            ('HS3', 'hot', 190.0, 175.0, 15796),
            ('HS4', 'hot', 160.0, 135.0, 6402),
            ('HS5', 'hot', 102.0, 40.0, 40417),
            ('HS6', 'hot', 170.0, 40.0, 795),
            ('HS7', 'hot', 98.4, 40.0, 2216),
            ('HS8', 'hot', 97.4, 40.0, 2290),
            ('HS9', 'hot', 98.4, 40.0, 1984),
            ('HS10', 'hot', 40.0, 25.0, 652),
            ('HS11', 'hot', 25.0, 10.0, 692),
            ('HS12', 'hot', 104.7, 40.0, 573),
            ('HS13', 'hot', 80.0, 40.0, 840),
            ('HS14', 'hot', 40.0, 25.0, 339),
            ('HS15', 'hot', 25.0, 7.4, 1553),
            ('HS16', 'hot', 57.5, 40.0, 358),
            ('HS17', 'hot', 11.7, 0.0, 1819),
            ('HS18', 'hot', 99.0, 96.2, 529),
            ('HS19', 'hot', 30.8, 28.6, 6690),
            ('HS20', 'hot', 44.7, 40.0, 5307),
            ('HS21', 'hot', 976.0, 300.0, 25220),
            ('CS1', 'cold', 10.0, 160.0, 4923),
            ('CS2', 'cold', 170.0, 170.5, 15796),
            ('CS3', 'cold', 120.0, 120.5, 6402),
            ('CS4', 'cold', 170.0, 170.5, 13667),
            ('CS5', 'cold', 170.0, 222.0, 1604),
            ('CS6', 'cold', 68.0, 90.0, 778),
            ('CS7', 'cold', 14.5, 15.0, 1448),
            ('CS8', 'cold', 122.0, 122.5, 386),
            ('CS9', 'cold', 2.4, 2.6, 1625),
            ('CS10', 'cold', 68.0, 68.5, 3241),
            ('CS11', 'cold', 4.5, 5.0, 3262),
            ('CS12', 'cold', 52.0, 53.0, 4666),
            ('CS13', 'cold', 175.0, 196.0, 6300),
            ('CS14', 'cold', 40.0, 99.0, 6490),
            ('CS15', 'cold', 33.9, 79.9, 1875),
            ('CS16', 'cold', 130.0, 302.0, 12703),
        ],
        targets=dict(Q_hot=0.0, Q_cold=17180884.30000001, pinch_hot=976.0, pinch_cold=966.0),
        published=dict(Q_hot=(0.0, 0.0), Q_cold=(17180884.3, 0.1), pinch_hot=None, pinch_cold=None),
        needs_repeated_pair=False,
        certificate=dict(
            # (side, stage, hot, cold, Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out)
            matches=[
                ('below', 1, 'HS12', 'CS12', 4666.0, 104.7, 96.55689354275742, 52.0, 53.0),
                ('below', 1, 'HS20', 'CS9', 325.0000000000003, 44.7, 44.63876012813266, 2.4, 2.6),
                ('below', 1, 'HS21', 'CS16', 2184916.0, 976.0, 889.3657414750198, 130.0, 302.0),
                ('below', 1, 'HS3', 'CS3', 3201.0, 190.0, 189.79735376044567, 120.0, 120.5),
                ('below', 1, 'HS4', 'CS15', 86250.00000000001,
                 160.0, 146.52764761012185, 33.9, 79.9),
                ('below', 1, 'HS5', 'CS10', 1620.5, 102.0, 101.95990548531559, 68.0, 68.5),
                ('below', 1, 'HS6', 'CS6', 17116.0, 170.0, 148.4704402515723, 68.0, 90.0),
                ('below', 1, 'HS9', 'CS11', 1631.0, 98.4, 97.57792338709677, 4.5, 5.0),
                ('below', 2, 'HS21', 'CS14', 382910.0,
                 889.3657414750198, 874.1829500396511, 40.0, 99.0),
                ('below', 3, 'HS21', 'CS5', 83408.0,
                 874.1829500396511, 870.8757335448058, 170.0, 222.0),
                ('below', 4, 'HS21', 'CS13', 132300.0,
                 870.8757335448058, 865.6298969072166, 175.0, 196.0),
                ('below', 5, 'HS21', 'CS2', 7898.0,
                 865.6298969072166, 865.3167327517843, 170.0, 170.5),
                ('below', 6, 'HS21', 'CS4', 6833.5,
                 865.3167327517843, 865.0457771609834, 170.0, 170.5),
                ('below', 7, 'HS21', 'CS8', 193.0,
                 865.0457771609834, 865.0381245043617, 122.0, 122.5),
                ('below', 8, 'HS21', 'CS7', 724.0,
                 865.0381245043617, 865.0094171292626, 14.5, 15.0),
                ('below', 9, 'HS21', 'CS1', 738450.0,
                 865.0094171292626, 835.7290840602698, 10.0, 160.0),
            ],
            hot_utility={},
            cold_utility={'HS1': 123075.0, 'HS2': 86671.0, 'HS3': 233738.99999999985, 'HS4': 73800.00000000006, 'HS5': 2504233.5, 'HS6': 86233.99999999999, 'HS7': 129414.40000000001, 'HS8': 131446.0, 'HS9': 114234.6, 'HS10': 9780.0, 'HS11': 10380.0, 'HS12': 32407.100000000002, 'HS13': 33600.0, 'HS14': 5085.0, 'HS15': 27332.800000000003, 'HS16': 6265.0, 'HS17': 21282.3, 'HS18': 1481.1999999999985, 'HS19': 14717.999999999995, 'HS20': 24617.900000000023, 'HS21': 13511087.500000004},
        ),
        note=(
            'heat unit not stated in the source; read as kW; HS21 (976 -> 300, CP 25220) carries 99.2'
            ' % of Q_cold; stream duties span 193 to 1.7e7, so heat tolerances must be absolute or '
            'per stream'
        ),
    ),
    # ----- real-thermodynamics problems -----
    dict(
        name='rtA01_linnhoff4_water',
        kind='real_thermo',
        description=(
            'Linnhoff/Kemp 4-stream problem transposed to pressurized liquid water (10 bar; T+260 K; '
            "CP ratio 2:3:4:1.5), sensible liquids only. Pinch at C3's supply (340 K shifted). "
            'Textbook design: above H2-C3 and H4-C1 (CP rule), below H2-C1 at the pinch then H4-C1.'
        ),
        chemicals=['Water'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 95.4}, 280.0, 395.0, 1000000.0, 'l', False),
            ('H2', {'Water': 142.0}, 430.0, 320.0, 1000000.0, 'l', False),
            ('C3', {'Water': 190.8}, 340.0, 400.0, 1000000.0, 'l', False),
            ('H4', {'Water': 71.6}, 410.0, 290.0, 1000000.0, 'l', False),
        ],
        reference=dict(Q_hot=71439.59349029488, Q_cold=213628.25624684175, pinch_T_shifted=340.0, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C3', Q=868894.2475563139, side='above', hot_seq=1, cold_seq=1),
            dict(hot='H4', cold='C1', Q=327198.45001137024, side='above', hot_seq=1, cold_seq=3),
            dict(hot='H2', cold='C1', Q=321281.72114567866, side='below', hot_seq=2, cold_seq=2),
            dict(hot='H4', cold='C1', Q=110154.80542325939, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=71439.59349029558, Q_cold=213628.25624684172),
    ),
    dict(
        name='rtA02_r002_short_of_tickoff',
        kind='real_thermo',
        description=(
            "Real-thermo analog of the reviewer's r002 (T' = 280 + 0.7 (T - 50), dT 7): one hot water"
            ' liquid (5 bar) is the only hot stream below the pinch (its supply sets the pinch) and '
            'must heat three cold liquids. The pinch match H1-C3 must stop SHORT of tick-off so that '
            "H1 is still hot enough for C1 (364 + 7 K); C3's remainder needs a second H1-C3 match "
            "(repeated pair on one side). Proof data in 'short_of_tickoff'."
        ),
        chemicals=['Water', 'Methanol'],
        T_min_app=7.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 200.0}, 399.0, 280.0, 500000.0, 'l', False),
            ('C1', {'Water': 50.0}, 343.0, 364.0, 500000.0, 'l', False),
            ('C2', {'Methanol': 45.0}, 315.0, 329.0, 500000.0, 'l', False),
            ('C3', {'Water': 100.0}, 329.0, 420.0, 500000.0, 'l', False),
        ],
        reference=dict(Q_hot=215300.6069739886, Q_cold=1189188.2932688468, pinch_T_shifted=392.0, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H1', cold='C3', Q=426925.2248520291, side='below', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C1', Q=79397.10780429505, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H1', cold='C3', Q=50438.708460174385, side='below', hot_seq=3, cold_seq=1),
            dict(hot='H1', cold='C2', Q=54494.70316711828, side='below', hot_seq=4, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=215300.60697345063, Q_cold=1189188.2932688466),
        short_of_tickoff={'flex': 'H1', 'pinch_stream': 'C3', 'blocked': 'C1'},
    ),
    dict(
        name='rtA03_condenser_at_pinch',
        kind='real_thermo',
        description=(
            'Pure ethanol isothermal condenser (1 atm, 351.57 K) sets the pinch. Its latent heat '
            'belongs below the pinch and, being isothermal AT the pinch, serves both cold pinch '
            'streams (water, methanol liquids) in series. Above: the hot water liquid is matched with'
            ' both cold streams in series.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Ethanol': 60.0}, 'dew', 'bubble', 101325.0, 'l', True),
            ('H2', {'Water': 100.0}, 410.0, 320.0, 500000.0, 'l', False),
            ('C1', {'Water': 150.0}, 300.0, 365.0, 101325.0, 'l', False),
            ('C2', {'Methanol': 80.0}, 320.0, 380.0, 500000.0, 'l', False),
        ],
        reference=dict(Q_hot=120517.44856532227, Q_cold=1963607.2531972774, pinch_T_shifted=341.570441659, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C1', Q=265743.13585361623, side='above', hot_seq=2, cold_seq=2),
            dict(hot='H2', cold='C2', Q=179369.4143462823, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C1', Q=469792.899233465, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C2', Q=153139.85799572145, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=120517.60446348641, Q_cold=1963607.409095517),
    ),
    dict(
        name='rtA04_boiler_at_pinch_two_hot',
        kind='real_thermo',
        description=(
            'Pure water boiler (1 atm, 373.12 K, saturated liquid -> saturated vapor) sets the pinch '
            'and is the only cold stream above it, while TWO hot streams are at the pinch (water '
            'liquid; ethanol vapor with desuperheat+condense+subcool at 5 bar). The isothermal boiler'
            ' takes both hot pinch ends in series: the textbook number rule does not apply to a point'
            ' load at the pinch.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 60.0}, 'bubble', 'dew', 101325.0, 'l', True),
            ('C2', {'Methanol': 100.0}, 300.0, 330.0, 101325.0, 'l', False),
            ('H1', {'Water': 120.0}, 440.0, 330.0, 1000000.0, 'l', False),
            ('H2', {'Ethanol': 40.0}, 450.0, 360.0, 500000.0, 'g', True),
        ],
        reference=dict(Q_hot=207461.562170011, Q_cold=362360.7674778069, pinch_T_shifted=373.124295848, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H1', cold='C1', Q=526688.6299635024, side='above', hot_seq=1, cold_seq=1),
            dict(hot='H2', cold='C1', Q=1704906.6844570914, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C2', Q=254679.48708687196, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=207461.35793606797, Q_cold=362360.56324386346),
    ),
    dict(
        name='rtA05_steam_desup_cond_sub_pinch',
        kind='real_thermo',
        description=(
            '3 bar steam desuperheated, condensed (406.67 K) and subcooled; its condensation sets the'
            ' pinch and its isothermal latent heat (below the pinch) serves three cold streams in '
            'series. Cold methanol vapor heated above the pinch; hot water liquid.'
        ),
        chemicals=['Water', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 50.0}, 470.0, 370.0, 300000.0, 'g', True),
            ('H2', {'Water': 60.0}, 420.0, 330.0, 500000.0, 'l', False),
            ('C1', {'Water': 100.0}, 310.0, 410.0, 500000.0, 'l', False),
            ('C2', {'Water': 150.0}, 300.0, 360.0, 101325.0, 'l', False),
            ('C3', {'Methanol': 150.0}, 370.0, 430.0, 200000.0, 'g', True),
        ],
        reference=dict(Q_hot=194351.73207261035, Q_cold=915116.1093185253, pinch_T_shifted=396.67242046090075, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C1', Q=61704.57743186306, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C3', Q=109592.64957664348, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C1', Q=656201.7066676746, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H1', cold='C3', Q=201067.43893844355, side='below', hot_seq=3, cold_seq=1),
            dict(hot='H1', cold='C2', Q=678725.4578145891, side='below', hot_seq=4, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=194351.82227062166, Q_cold=915116.1995166652),
    ),
    dict(
        name='rtA06_boiler_subcooled_to_superheated',
        kind='real_thermo',
        description=(
            'Ethanol at 2 bar heated from subcooled liquid through boiling (369.86 K, the pinch) to '
            'superheated vapor; pressurized hot water, superheated methanol vapor and steam (sensible'
            ' vapors) above the pinch all end against the isothermal boiling; H1-C2 repeated below.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Ethanol': 50.0}, 320.0, 400.0, 200000.0, 'l', True),
            ('C2', {'Water': 100.0}, 300.0, 350.0, 101325.0, 'l', False),
            ('H1', {'Water': 200.0}, 450.0, 340.0, 1000000.0, 'l', False),
            ('H2', {'Methanol': 100.0}, 430.0, 380.0, 300000.0, 'g', True),
            ('H3', {'Water': 80.0}, 420.0, 380.0, 101325.0, 'g', True),
            ('H4', {'Water': 80.0}, 345.0, 305.0, 101325.0, 'l', False),
        ],
        reference=dict(Q_hot=561984.2063972547, Q_cold=138719.69179782073, pinch_T_shifted=369.8572114718511, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H1', cold='C1', Q=1085177.8094160939, side='above', hot_seq=1, cold_seq=3),
            dict(hot='H2', cold='C1', Q=259996.606789554, side='above', hot_seq=1, cold_seq=4),
            dict(hot='H3', cold='C1', Q=109645.12869567331, side='above', hot_seq=1, cold_seq=5),
            dict(hot='H1', cold='C1', Q=301383.8797179568, side='below', hot_seq=2, cold_seq=2),
            dict(hot='H1', cold='C2', Q=113221.42617213994, side='below', hot_seq=3, cold_seq=3),
            dict(hot='H1', cold='C2', Q=189056.71286161157, side='below', hot_seq=4, cold_seq=2),
            dict(hot='H4', cold='C1', Q=27853.116260636656, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H4', cold='C2', Q=74572.82531694858, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=561984.1703396922, Q_cold=138719.65574025473),
    ),
    dict(
        name='rtA07_hot_glide_below_pinch',
        kind='real_thermo',
        description=(
            'Water/ethanol xE=0.1 vapor condensed across its glide (370.4 -> 359.5 K) entirely below '
            "the pinch (set by H2's supply); water and methanol liquids."
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 54.0, 'Ethanol': 6.0}, 390.0, 340.0, 101325.0, 'g', True),
            ('C1', {'Water': 120.0}, 300.0, 355.0, 101325.0, 'l', False),
            ('C2', {'Methanol': 80.0}, 305.0, 330.0, 101325.0, 'l', False),
            ('H2', {'Water': 50.0}, 400.0, 320.0, 500000.0, 'l', False),
            ('C3', {'Water': 30.0}, 350.0, 400.0, 500000.0, 'l', False),
        ],
        reference=dict(Q_hot=22951.838720510947, Q_cold=2166901.159473365, pinch_T_shifted=390.0, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C3', Q=91090.40752076614, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C1', Q=497579.48524906003, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C2', Q=170907.40238710644, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=22951.842907392653, Q_cold=2166901.1635355684),
    ),
    dict(
        name='rtA08_cold_glide_threshold_Qc0',
        kind='real_thermo',
        description=(
            'Threshold problem with ZERO cold utility (pinch at the bottom): water/methanol xM=0.2 '
            'boiled across its glide (354.8 -> 367.8 K), 2 bar steam condenser, water and methanol '
            'liquids; H1-C1 repeated.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=5.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 40.0, 'Methanol': 10.0}, 320.0, 375.0, 101325.0, 'l', True),
            ('H1', {'Water': 100.0}, 450.0, 340.0, 1000000.0, 'l', False),
            ('H2', {'Water': 30.0}, 'dew', 'bubble', 200000.0, 'l', True),
            ('C2', {'Water': 100.0}, 300.0, 340.0, 101325.0, 'l', False),
            ('H3', {'Methanol': 60.0}, 369.0, 310.0, 500000.0, 'l', False),
            ('C3', {'Water': 50.0}, 380.0, 440.0, 1000000.0, 'l', False),
        ],
        reference=dict(Q_hot=346449.0061063774, Q_cold=0.0, pinch_T_shifted=300.0, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H3', cold='C2', Q=263629.53775957617, side='above', hot_seq=2, cold_seq=1),
            dict(hot='H1', cold='C1', Q=122060.26416768176, side='above', hot_seq=3, cold_seq=1),
            dict(hot='H1', cold='C2', Q=37715.031266609614, side='above', hot_seq=2, cold_seq=2),
            dict(hot='H3', cold='C1', Q=59702.80590285431, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C1', Q=684644.6186496097, side='above', hot_seq=1, cold_seq=3),
            dict(hot='H2', cold='C1', Q=1193994.78569886, side='above', hot_seq=1, cold_seq=4),
        ],
        certificate_utilities=dict(Q_hot=346449.00595814106, Q_cold=0.0),
    ),
    dict(
        name='rtA09_EM_glide_threshold_Qh0',
        kind='real_thermo',
        description=(
            'Threshold problem with ZERO hot utility: ethanol/methanol 50/50 vapor condensed across '
            'its 1.9 K glide against water and ethanol liquids.'
        ),
        chemicals=['Ethanol', 'Methanol', 'Water'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Ethanol': 40.0, 'Methanol': 40.0}, 370.0, 320.0, 101325.0, 'g', True),
            ('C1', {'Water': 120.0}, 300.0, 350.0, 101325.0, 'l', False),
            ('H2', {'Water': 60.0}, 390.0, 330.0, 500000.0, 'l', False),
            ('C2', {'Ethanol': 50.0}, 320.0, 345.0, 101325.0, 'l', False),
        ],
        reference=dict(Q_hot=0.0, Q_cold=2980517.6092781536, pinch_T_shifted=None, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C1', Q=159501.6510267902, side='below', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C2', Q=157702.63148631965, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C1', Q=292719.5061940499, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=0.0, Q_cold=2980517.609051818),
    ),
    dict(
        name='rtA10_near_azeotrope_boiling',
        kind='real_thermo',
        description=(
            'Water/ethanol xE=0.7 (0.5 K glide) boiled from 320 to 370 K by a 2 bar steam condenser '
            "and hot water; pinch at H2's supply; certificate with repeated pairs (H2-C1 x2, H2-C2 "
            'x3).'
        ),
        chemicals=['Water', 'Ethanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 12.0, 'Ethanol': 28.0}, 320.0, 370.0, 101325.0, 'l', True),
            ('H1', {'Water': 40.0}, 'dew', 'bubble', 200000.0, 'l', True),
            ('H2', {'Water': 80.0}, 400.0, 310.0, 500000.0, 'l', False),
            ('C2', {'Water': 60.0}, 300.0, 360.0, 101325.0, 'l', False),
            ('C3', {'Water': 40.0}, 380.0, 420.0, 500000.0, 'l', False),
        ],
        reference=dict(Q_hot=92231.05593656626, Q_cold=53141.16286954272, pinch_T_shifted=390.0, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C3', Q=30489.403854961973, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H2', cold='C1', Q=152483.90257546544, side='below', hot_seq=2, cold_seq=3),
            dict(hot='H2', cold='C2', Q=46654.429105993186, side='below', hot_seq=3, cold_seq=3),
            dict(hot='H1', cold='C1', Q=1591993.04759848, side='below', hot_seq=1, cold_seq=2),
            dict(hot='H2', cold='C2', Q=142688.73539810738, side='below', hot_seq=4, cold_seq=2),
            dict(hot='H2', cold='C1', Q=37750.67623817102, side='below', hot_seq=5, cold_seq=1),
            dict(hot='H2', cold='C2', Q=82147.0186217351, side='below', hot_seq=6, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=92231.06247271306, Q_cold=53141.16941052169),
        note=(
            'match 5 (H2-C1, below) duty lowered by 0.00432 kJ/hr from the designed '
            '37750.68055333849: on exact planned states its internal approach was 9.999999018 K (the '
            'original check used the simulated HXprocess, whose dT limit absorbed the shortfall)'
        ),
    ),
    dict(
        name='rtA11_boiler_pinch_condenser_below',
        kind='real_thermo',
        description=(
            'Water boiler at 2 bar (393.36 K) sets the pinch and is served above by superheated '
            'methanol vapor; below, a 1 atm steam condenser (+subcool) boils ethanol (isothermal) and'
            ' heats methanol to its bubble point.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 80.0}, 'dew', 330.0, 101325.0, 'l', True),
            ('C1', {'Methanol': 100.0}, 300.0, 'bubble', 101325.0, 'l', True),
            ('C2', {'Ethanol': 20.0}, 'bubble', 'dew', 101325.0, 'l', True),
            ('H2', {'Methanol': 100.0}, 440.0, 360.0, 200000.0, 'g', True),
            ('C3', {'Water': 10.0}, 'bubble', 'dew', 200000.0, 'l', True),
        ],
        reference=dict(Q_hot=202525.07677420386, Q_cold=2624473.4991559633, pinch_T_shifted=393.36009132800956, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C3', Q=195473.1894763694, side='above', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C2', Q=782805.6359628483, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C1', Q=322999.2659982466, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=202525.0724232506, Q_cold=2624473.4948050147),
    ),
    dict(
        name='rtA12_multi_phase_change',
        kind='real_thermo',
        description=(
            'Phase change at four pressures: steam condenser (5 bar), ethanol '
            'desuperheat+condense+subcool (3 bar), methanol heated and boiled to saturated vapor (5 '
            'bar; its boiling sets the pinch and takes two hot pinch ends in series), water boiler (2'
            ' bar), two cold liquids.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 30.0}, 'dew', 'bubble', 500000.0, 'l', True),
            ('H2', {'Ethanol': 40.0}, 420.0, 350.0, 300000.0, 'g', True),
            ('H3', {'Water': 70.0}, 440.0, 330.0, 1000000.0, 'l', False),
            ('C1', {'Methanol': 30.0}, 330.0, 'dew', 500000.0, 'l', True),
            ('C2', {'Water': 25.0}, 'bubble', 'dew', 200000.0, 'l', True),
            ('C3', {'Ethanol': 80.0}, 300.0, 345.0, 101325.0, 'l', False),
            ('C4', {'Water': 100.0}, 300.0, 340.0, 101325.0, 'l', False),
        ],
        reference=dict(Q_hot=500420.7481043098, Q_cold=1150417.3847702402, pinch_T_shifted=384.5157827754926, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C1', Q=83931.63674364518, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H3', cold='C1', Q=246372.09596558643, side='above', hot_seq=1, cold_seq=3),
            dict(hot='H1', cold='C2', Q=994995.65474905, side='above', hot_seq=2, cold_seq=1),
            dict(hot='H1', cold='C1', Q=158515.56603519246, side='above', hot_seq=1, cold_seq=4),
            dict(hot='H3', cold='C1', Q=157913.7405402002, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H2', cold='C3', Q=439000.3781692564, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H2', cold='C4', Q=301344.5690261858, side='below', hot_seq=3, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=500420.73265142273, Q_cold=1150417.3693173532),
    ),
    dict(
        name='rtA13_smith4_cp_assignment',
        kind='real_thermo',
        description=(
            "Smith 4-stream problem transposed (T' = 280 + 0.6 (T - 50), dT 6; hot ethanol liquid at "
            '10 bar, water otherwise): above the pinch the only CP-feasible pinch assignment is '
            'H1-C3, H2-C4 (a greedy one-at-a-time choice H1-C4 strands H2).'
        ),
        chemicals=['Water', 'Ethanol'],
        T_min_app=6.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Ethanol': 44.0}, 412.0, 346.0, 1000000.0, 'l', False),
            ('H2', {'Water': 110.0}, 382.0, 286.0, 500000.0, 'l', False),
            ('C3', {'Water': 100.0}, 280.0, 376.0, 500000.0, 'l', False),
            ('C4', {'Water': 250.0}, 346.0, 376.0, 500000.0, 'l', False),
        ],
        reference=dict(Q_hot=144708.54461331974, Q_cold=85052.11983049946, pinch_T_shifted=346.0, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H1', cold='C3', Q=227206.4282181022, side='above', hot_seq=2, cold_seq=2),
            dict(hot='H2', cold='C4', Q=250272.12817810866, side='above', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C4', Q=173035.39775382716, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H2', cold='C3', Q=497529.9264931389, side='below', hot_seq=2, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=144708.54461332085, Q_cold=85052.11983049929),
    ),
    dict(
        name='rtA14_11_streams',
        kind='real_thermo',
        description=(
            '11 streams: an ethanol condenser at the pinch serves four cold streams below in series; '
            'steam desuperheat+condense+subcool, methanol vapor, water boiler (2 bar), water/methanol'
            ' glide boiler, superheated steam heating; 13-exchanger certificate.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 30.0}, 460.0, 350.0, 300000.0, 'g', True),
            ('H2', {'Water': 60.0}, 445.0, 320.0, 1000000.0, 'l', False),
            ('H3', {'Ethanol': 40.0}, 'dew', 'bubble', 101325.0, 'l', True),
            ('H4', {'Methanol': 60.0}, 420.0, 365.0, 200000.0, 'g', True),
            ('H5', {'Water': 40.0}, 400.0, 310.0, 500000.0, 'l', False),
            ('C1', {'Water': 100.0}, 300.0, 365.0, 101325.0, 'l', False),
            ('C2', {'Ethanol': 30.0}, 330.0, 390.0, 500000.0, 'l', False),
            ('C3', {'Water': 30.0}, 'bubble', 'dew', 200000.0, 'l', True),
            ('C4', {'Methanol': 50.0}, 300.0, 330.0, 101325.0, 'l', False),
            ('C5', {'Water': 32.0, 'Methanol': 8.0}, 330.0, 375.0, 101325.0, 'l', True),
            ('C6', {'Water': 200.0}, 390.0, 450.0, 101325.0, 'g', True),
        ],
        reference=dict(Q_hot=1560258.735430469, Q_cold=1316330.2243316392, pinch_T_shifted=341.570441659, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H1', cold='C5', Q=1358205.2067580407, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H2', cold='C1', Q=114565.30614198213, side='above', hot_seq=3, cold_seq=2),
            dict(hot='H5', cold='C2', Q=53768.43887301241, side='above', hot_seq=3, cold_seq=2),
            dict(hot='H4', cold='C2', Q=153310.0690398405, side='above', hot_seq=2, cold_seq=3),
            dict(hot='H5', cold='C1', Q=62596.784427095336, side='above', hot_seq=2, cold_seq=3),
            dict(hot='H2', cold='C5', Q=121865.33679724854, side='above', hot_seq=2, cold_seq=3),
            dict(hot='H5', cold='C5', Q=30943.70051040844, side='above', hot_seq=1, cold_seq=4),
            dict(hot='H2', cold='C3', Q=194105.64009960758, side='above', hot_seq=1, cold_seq=1),
            dict(hot='H4', cold='C3', Q=14972.761368432752, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H3', cold='C1', Q=313195.2661556434, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H3', cold='C2', Q=44290.79681488048, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H3', cold='C5', Q=36258.40314162732, side='below', hot_seq=3, cold_seq=1),
            dict(hot='H3', cold='C4', Q=127339.74354343598, side='below', hot_seq=4, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=1560258.8319406067, Q_cold=1316330.320852139),
        note=(
            'match 5 (H2-C5, above) duty lowered by 0.168 kJ/hr from the designed 121865.50525344802:'
            ' on exact planned states its internal approach was 9.99996307 K (the original check used'
            ' the simulated HXprocess, whose dT limit absorbed the shortfall)'
        ),
    ),
    dict(
        name='rtA15_13_streams',
        kind='real_thermo',
        description=(
            '13 streams: a 5 bar steam condenser sets the pinch; ethanol '
            'desuperheat+condense+subcool, water/ethanol glide condenser, methanol heated+boiled (2 '
            'bar), water boiler (3 bar), ethanol/methanol liquid, superheated steam heating; '
            '13-exchanger certificate.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 20.0}, 'dew', 'bubble', 500000.0, 'l', True),
            ('H2', {'Water': 50.0}, 450.0, 330.0, 1000000.0, 'l', False),
            ('H3', {'Ethanol': 50.0}, 400.0, 330.0, 101325.0, 'g', True),
            ('H4', {'Methanol': 40.0}, 380.0, 300.0, 500000.0, 'l', False),
            ('H5', {'Water': 18.0, 'Ethanol': 2.0}, 385.0, 350.0, 101325.0, 'g', True),
            ('H6', {'Water': 60.0}, 360.0, 310.0, 101325.0, 'l', False),
            ('C1', {'Water': 150.0}, 290.0, 340.0, 101325.0, 'l', False),
            ('C2', {'Methanol': 20.0}, 300.0, 'dew', 200000.0, 'l', True),
            ('C3', {'Ethanol': 30.0}, 320.0, 390.0, 500000.0, 'l', False),
            ('C4', {'Water': 15.0}, 'bubble', 'dew', 300000.0, 'l', True),
            ('C5', {'Water': 30.0}, 330.0, 440.0, 1000000.0, 'l', False),
            ('C6', {'Ethanol': 20.0, 'Methanol': 20.0}, 300.0, 340.0, 101325.0, 'l', False),
            ('C7', {'Water': 130.0}, 400.0, 460.0, 101325.0, 'g', True),
        ],
        reference=dict(Q_hot=162883.29807923216, Q_cold=2133923.14180115, pinch_T_shifted=414.9810791030153, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H2', cold='C7', Q=98033.26346495637, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H2', cold='C5', Q=193894.0523444919, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H1', cold='C7', Q=66853.02291096421, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H1', cold='C4', Q=588529.5104252113, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H1', cold='C3', Q=90054.93925884178, side='below', hot_seq=3, cold_seq=3),
            dict(hot='H3', cold='C3', Q=83579.70977752375, side='below', hot_seq=1, cold_seq=2),
            dict(hot='H2', cold='C2', Q=32128.79617349571, side='below', hot_seq=3, cold_seq=4),
            dict(hot='H3', cold='C2', Q=50121.82774911004, side='below', hot_seq=2, cold_seq=3),
            dict(hot='H4', cold='C2', Q=56991.90964679243, side='below', hot_seq=1, cold_seq=2),
            dict(hot='H5', cold='C2', Q=648353.761543988, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H6', cold='C3', Q=114581.32092974387, side='below', hot_seq=1, cold_seq=1),
            dict(hot='H5', cold='C6', Q=165583.817636755, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H3', cold='C1', Q=565056.8769924915, side='below', hot_seq=3, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=162883.30622663576, Q_cold=2133923.149963759),
        note=(
            'match 2 (H1-C7, below) duty lowered by 0.00377 kJ/hr from the designed 66853.026678144: '
            'on exact planned states its internal approach was 9.999999158 K (the original check used'
            ' the simulated HXprocess, whose dT limit absorbed the shortfall)'
        ),
    ),
    dict(
        name='rtA16_vapor_sensible',
        kind='real_thermo',
        description=(
            'Sensible superheated steam cooling (480 -> 400 K) and superheated ethanol vapor heating '
            '(356 -> 450 K), methanol isothermal condenser (3 bar), ethanol '
            'desuperheat+condense+subcool (2 bar) whose condensation sets the pinch and serves two '
            'cold streams below in series.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 100.0}, 480.0, 400.0, 101325.0, 'g', True),
            ('H2', {'Methanol': 30.0}, 'dew', 'bubble', 300000.0, 'l', True),
            ('H3', {'Ethanol': 20.0}, 440.0, 330.0, 200000.0, 'g', True),
            ('C1', {'Ethanol': 60.0}, 356.0, 450.0, 101325.0, 'g', True),
            ('C2', {'Water': 60.0}, 300.0, 360.0, 101325.0, 'l', False),
            ('C3', {'Methanol': 50.0}, 300.0, 330.0, 101325.0, 'l', False),
        ],
        reference=dict(Q_hot=51784.17096424871, Q_cold=1466346.9426415022, pinch_T_shifted=359.8572114718511, grid_h=0.25, nV=201),
        certificate=[
            dict(hot='H3', cold='C1', Q=114966.52074040612, side='above', hot_seq=1, cold_seq=2),
            dict(hot='H1', cold='C1', Q=277033.7096281294, side='above', hot_seq=1, cold_seq=3),
            dict(hot='H3', cold='C2', Q=270841.59293253225, side='below', hot_seq=2, cold_seq=1),
            dict(hot='H3', cold='C1', Q=17266.598586789332, side='below', hot_seq=3, cold_seq=1),
            dict(hot='H2', cold='C3', Q=127339.74354343598, side='below', hot_seq=1, cold_seq=1),
        ],
        certificate_utilities=dict(Q_hot=51784.251091282465, Q_cold=1466347.0227685352),
    ),
]

SPLIT = [
    # ----- constant-CP literature problems -----
    dict(
        name='cornell_processdesign_four_stream_split',
        kind='constant_cp',
        citation=(
            "Cornell University Process Design Wiki (processdesign), 'Pinch analysis' worked example "
            '(after Towler & Sinnott 2013, Chemical Engineering Design, 2nd ed.).'
        ),
        source_url='https://design.cbe.cornell.edu/index.php?title=Pinch_analysis',
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 180, 40, 40.0),
            ('2', 'hot', 150, 60, 30.0),
            ('3', 'cold', 30, 180, 60.0),
            ('4', 'cold', 80, 160, 20.0),
        ],
        targets=dict(Q_hot=2900.0, Q_cold=600.0, pinch_hot=100.0, pinch_cold=80.0),
        published=dict(Q_hot=(2900, 1.0), Q_cold=(600, 1.0), pinch_hot=(100, 0.0), pinch_cold=(80, 0.0)),
        proof=dict(
            side='above', rule='cp', pinch_hot=100.0, pinch_cold=80.0,
            hot_at_pinch=['1', '2'],
            cold_at_pinch=['3', '4'],
        ),
    ),
    dict(
        name='nptel_t4_4_four_stream_dT20',
        kind='constant_cp',
        citation=(
            "Mohanty B (IIT Roorkee). NPTEL course 103107094 'Process Integration' lecture notes, "
            'Module 4 Lecture 13, Table 4.4 (modified problem table with multiple utilities).'
        ),
        source_url=(
            'https://archive.nptel.ac.in/content/storage2/courses/103107094/module4/lecture4/lecture4'
            '.pdf'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 150, 60, 2.5),
            ('2', 'hot', 90, 60, 8.0),
            ('3', 'cold', 20, 125, 3.0),
            ('4', 'cold', 25, 100, 3.0),
        ],
        targets=dict(Q_hot=105.0, Q_cold=30.0, pinch_hot=90.0, pinch_cold=70.0),
        published=dict(Q_hot=(105, 1.0), Q_cold=(30, 1.0), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='below', rule='cp', pinch_hot=90.0, pinch_cold=70.0,
            hot_at_pinch=['1', '2'],
            cold_at_pinch=['3', '4'],
        ),
    ),
    dict(
        name='nptel_t5_3_four_stream_split',
        kind='constant_cp',
        citation=(
            "Mohanty B (IIT Roorkee). NPTEL course 103107094 'Process Integration' lecture notes, "
            'Module 5 Lecture 28, Table 5.3 (designs Figs. 5.21-5.23).'
        ),
        source_url=(
            'https://archive.nptel.ac.in/content/storage2/courses/103107094/module5/lecture3/lecture3'
            '.pdf'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 200, 65, 3.0),
            ('2', 'hot', 90, 30, 6.0),
            ('3', 'cold', 30, 142, 3.5),
            ('4', 'cold', 25, 130, 4.0),
        ],
        targets=dict(Q_hot=87.0, Q_cold=40.0, pinch_hot=90.0, pinch_cold=80.0),
        published=dict(Q_hot=(87, 1.0), Q_cold=(40, 1.0), pinch_hot=(90, 0.0), pinch_cold=(80, 0.0)),
        proof=dict(
            side='below', rule='cp', pinch_hot=90.0, pinch_cold=80.0,
            hot_at_pinch=['1', '2'],
            cold_at_pinch=['3', '4'],
        ),
    ),
    dict(
        name='smith2005_ex18_2_split',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 18, Example 18.2, '
            'Table 18.2; MER design Fig. 18.17.'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2018.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 720, 320, 0.045),
            ('2', 'hot', 520, 220, 0.04),
            ('3', 'cold', 300, 900, 0.043),
            ('4', 'cold', 200, 550, 0.02),
        ],
        targets=dict(Q_hot=9.2, Q_cold=6.4, pinch_hot=520.0, pinch_cold=500.0),
        published=dict(Q_hot=(9.2, 0.1), Q_cold=(6.4, 0.1), pinch_hot=(520, 0.0), pinch_cold=(500, 0.0)),
        proof=dict(
            side='above', rule='cp', pinch_hot=520.0, pinch_cold=500.0,
            hot_at_pinch=['1'],
            cold_at_pinch=['3', '4'],
        ),
    ),
    dict(
        name='smith2005_ex18_4_split',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 18, Example 18.4, '
            'Table 18.5; designs Figs. 18.22-18.24 (same data as Exercise 18.6, Table 18.13).'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2018.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 150, 50, 0.2),
            ('2', 'hot', 170, 40, 0.1),
            ('3', 'cold', 50, 120, 0.3),
            ('4', 'cold', 80, 110, 0.5),
        ],
        targets=dict(Q_hot=7.0, Q_cold=4.0, pinch_hot=90.0, pinch_cold=80.0),
        published=dict(Q_hot=(7.0, 1.0), Q_cold=(4.0, 1.0), pinch_hot=(90, 0.0), pinch_cold=(80, 0.0)),
        proof=dict(
            side='below', rule='cp', pinch_hot=90.0, pinch_cold=80.0,
            hot_at_pinch=['1', '2'],
            cold_at_pinch=['3'],
        ),
    ),
    dict(
        name='smith2005_ex16_5_five_stream',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 16, Example 16.5, '
            'Table 16.7 / cascade Table 16.8.'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2016.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 450, 50, 0.25),
            ('2', 'hot', 50, 40, 1.5),
            ('3', 'cold', 30, 400, 0.22),
            ('4', 'cold', 30, 400, 0.05),
            ('5', 'cold', 120, 121, 22.0),
        ],
        targets=dict(Q_hot=21.900000000000002, Q_cold=15.0, pinch_hot=50.0, pinch_cold=30.0),
        published=dict(Q_hot=(21.9, 0.1), Q_cold=(15.0, 1.0), pinch_hot=(50, 0.0), pinch_cold=(30, 0.0)),
        proof=dict(
            side='above', rule='cp', pinch_hot=50.0, pinch_cold=30.0,
            hot_at_pinch=['1'],
            cold_at_pinch=['3', '4'],
        ),
    ),
    dict(
        name='smith2005_exr18_4',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 18, Exercise 4, '
            'Table 18.10.'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2018.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'cold', 18, 123, 0.0933),
            ('2', 'cold', 118, 193, 0.1961),
            ('3', 'cold', 189, 286, 0.1796),
            ('4', 'hot', 159, 77, 0.2285),
            ('5', 'hot', 267, 80, 0.0204),
            ('6', 'hot', 343, 90, 0.0538),
        ],
        targets=dict(Q_hot=13.9472, Q_cold=8.1852, pinch_hot=159.0, pinch_cold=149.0),
        published=dict(Q_hot=(13.95, 0.01), Q_cold=(8.18, 0.01), pinch_hot=(159, 0.0), pinch_cold=(149, 0.0)),
        proof=dict(
            side='above', rule='number', pinch_hot=159.0, pinch_cold=149.0,
            hot_at_pinch=['5', '6'],
            cold_at_pinch=['2'],
        ),
    ),
    dict(
        name='smith2005_exr18_7_six_stream',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 18, Exercise 7, '
            'Tables 18.14 and 18.15.'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2018.pdf',
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 170, 88, 23.0),
            ('2', 'hot', 278, 90, 2.0),
            ('3', 'hot', 354, 100, 5.0),
            ('4', 'cold', 30, 135, 9.0),
            ('5', 'cold', 130, 205, 20.0),
            ('6', 'cold', 200, 298, 18.0),
        ],
        targets=dict(Q_hot=1528.0, Q_cold=851.0, pinch_hot=170.0, pinch_cold=160.0),
        published=dict(Q_hot=(1528.0, 1.0), Q_cold=(851.0, 1.0), pinch_hot=(170, 0.0), pinch_cold=(160, 0.0)),
        proof=dict(
            side='above', rule='number', pinch_hot=170.0, pinch_cold=160.0,
            hot_at_pinch=['2', '3'],
            cold_at_pinch=['5'],
        ),
    ),
    dict(
        name='7sp3_dt20F',
        kind='constant_cp',
        citation=(
            'Problem 7SP3 (crude-oil preheat train; Grossmann & Sargent 1978, Comput. Chem. Eng. 2, '
            '1) as tabulated in Muraki, M.; Hayakawa, T. (1982). Practical synthesis method for heat '
            'exchanger network. J. Chem. Eng. Japan 15(2), 136-141, Table 1 (stream data; design '
            'data: minimum allowable approach temperature 20 F for exchangers); targets from Muraki &'
            " Hayakawa (1982) Table 2 'Problem table for 7SP3' and text."
        ),
        source_url='https://www.jstage.jst.go.jp/article/jcej1968/15/2/15_2_136/_pdf/-char/ja',
        T_unit='F', Q_unit='Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 675, 150, 153018),
            ('H2', 'hot', 592, 452, 111220),
            ('H3', 'hot', 543, 115, 45507),
            ('H4', 'hot', 428, 344, 598500),
            ('H5', 'hot', 398, 100, 116482),
            ('H6', 'hot', 301, 231, 1277434),
            ('C1', 'cold', 60, 710, 478351),
        ],
        targets=dict(Q_hot=85862451.0, Q_cold=64722563.0, pinch_hot=428.0, pinch_cold=408.0),
        published=dict(Q_hot=(85900000.0, 100000.0), Q_cold=(64700000.0, 100000.0), pinch_hot=(428, 0.0), pinch_cold=(408, 0.0)),
        proof=dict(
            side='above', rule='number', pinch_hot=428.0, pinch_cold=408.0,
            hot_at_pinch=['H1', 'H3'],
            cold_at_pinch=['C1'],
        ),
        note=(
            'published targets: Muraki & Hayakawa report 85.9e6 / 64.7e6 Btu/hr (three significant '
            'figures)'
        ),
    ),
    dict(
        name='7sp4_dolan1990_dt10K',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '7sp4' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Dolan, W.B.; Cummings, P.T.; LeVan, M.D. (1990). Algorithmic '
            'efficiency of simulated annealing for heat exchanger network design. Comput. Chem. Eng. '
            '14, 1039 (7SP4).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/7sp4.dat (targets: '
            'data/mip_instances/furman_sahinidis/7sp4.dat)'
        ),
        T_unit='K', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 630.555, 338.888, 7.913),
            ('H2', 'hot', 583.333, 505.555, 5.803),
            ('H3', 'hot', 555.555, 319.444, 2.374),
            ('H4', 'hot', 494.444, 447.222, 31.652),
            ('H5', 'hot', 477.777, 311.111, 6.3305),
            ('H6', 'hot', 422.222, 383.333, 65.943),
            ('C1', 'cold', 288.888, 650.0, 24.795),
        ],
        targets=dict(Q_hot=2431.4914290000006, Q_cold=1911.7607919999969, pinch_hot=494.444, pinch_cold=484.444),
        published=dict(Q_hot=(2431.491429, 1e-06), Q_cold=(1911.760792, 1e-06), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='number', pinch_hot=494.444, pinch_cold=484.444,
            hot_at_pinch=['H1', 'H3'],
            cold_at_pinch=['C1'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='8sp1_dt20F',
        kind='constant_cp',
        citation=(
            'Problem 8SP1 (Grossmann & Sargent 1978, Comput. Chem. Eng. 2, 1, per the '
            'Furman-Sahinidis reference list) as tabulated in Muraki, M.; Hayakawa, T. (1982). '
            'Practical synthesis method for heat exchanger network. J. Chem. Eng. Japan 15(2), '
            '136-141, Table 1 (stream data; design data: minimum allowable approach temperature 20 F '
            'for exchangers).'
        ),
        source_url='https://www.jstage.jst.go.jp/article/jcej1968/15/2/15_2_136/_pdf/-char/ja',
        T_unit='F', Q_unit='Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 470, 320, 22400),
            ('H2', 'hot', 450, 240, 17500),
            ('H3', 'hot', 370, 150, 28500),
            ('H4', 'hot', 310, 200, 20100),
            ('C1', 'cold', 200, 420, 16800),
            ('C2', 'cold', 150, 400, 23200),
            ('C3', 'cold', 185, 330, 35100),
            ('C4', 'cold', 140, 300, 17250),
        ],
        targets=dict(Q_hot=2227000.0, Q_cold=397500.0, pinch_hot=170.0, pinch_cold=150.0),
        published=None,
        proof=dict(
            side='above', rule='cp', pinch_hot=170.0, pinch_cold=150.0,
            hot_at_pinch=['H3'],
            cold_at_pinch=['C2', 'C4'],
        ),
    ),
    dict(
        name='8sp1_sargent1978_dt10F',
        kind='constant_cp',
        citation=(
            'Furman, K.C.; Sahinidis, N.V. (2004). Approximation algorithms for the minimum number of'
            ' matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565; instance '8sp1' as digitized in Letsios, D.; Kouyialis, G.; Misener, R. "
            '(2018). Heuristics with performance guarantees for the minimum number of matches problem'
            ' in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (GitHub data set). '
            'Original problem: Grossmann, I.E.; Sargent, R.W.H. (1978). Optimum design of heat '
            'exchanger networks. Comput. Chem. Eng. 2, 1 (8SP1).'
        ),
        source_url=(
            'https://github.com/cog-imperial/min_matches_heuristics/blob/master/data/original_instanc'
            'es/furman_sahinidis/dat_files/8sp1.dat (targets: '
            'data/mip_instances/furman_sahinidis/8sp1.dat)'
        ),
        T_unit='F', Q_unit='1e3 Btu/hr',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 470.0, 320.0, 22.4),
            ('H2', 'hot', 450.0, 240.0, 17.5),
            ('H3', 'hot', 370.0, 150.0, 28.5),
            ('H4', 'hot', 310.0, 200.0, 20.1),
            ('C1', 'cold', 200.0, 420.0, 16.8),
            ('C2', 'cold', 150.0, 400.0, 23.2),
            ('C3', 'cold', 185.0, 330.0, 35.1),
            ('C4', 'cold', 140.0, 300.0, 17.25),
        ],
        targets=dict(Q_hot=1942.0, Q_cold=112.5, pinch_hot=160.0, pinch_cold=150.0),
        published=dict(Q_hot=(1942.0, 1.0), Q_cold=(112.5, 0.1), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='cp', pinch_hot=160.0, pinch_cold=150.0,
            hot_at_pinch=['H3'],
            cold_at_pinch=['C2', 'C4'],
        ),
    ),
    dict(
        name='nptel_t5_7_eight_stream_split',
        kind='constant_cp',
        citation=(
            "Mohanty B (IIT Roorkee). NPTEL course 103107094 'Process Integration' lecture notes, "
            'Module 5 Lecture 30, Table 5.7 (MER design Figs. 5.34-5.36).'
        ),
        source_url=(
            'https://archive.nptel.ac.in/content/storage2/courses/103107094/module5/lecture5/lecture5'
            '.pdf'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H-1', 'hot', 150, 40, 0.1),
            ('H-2', 'hot', 140, 30, 0.15),
            ('H-3', 'hot', 130, 25, 0.15),
            ('H-4', 'hot', 150, 30, 0.2),
            ('C-1', 'cold', 20, 140, 0.1),
            ('C-2', 'cold', 15, 130, 0.15),
            ('C-3', 'cold', 25, 145, 0.25),
            ('C-4', 'cold', 80, 140, 0.3),
        ],
        targets=dict(Q_hot=16.25, Q_cold=6.25, pinch_hot=90.0, pinch_cold=80.0),
        published=dict(Q_hot=(16.25, 0.01), Q_cold=(6.25, 0.01), pinch_hot=(90, 0.0), pinch_cold=(80, 0.0)),
        proof=dict(
            side='below', rule='cp', pinch_hot=90.0, pinch_cold=80.0,
            hot_at_pinch=['H-1', 'H-2', 'H-3', 'H-4'],
            cold_at_pinch=['C-1', 'C-2', 'C-3'],
        ),
    ),
    dict(
        name='smith2005_exr18_5_nine_stream',
        kind='constant_cp',
        citation=(
            'Smith R (2005). Chemical Process Design and Integration. Wiley, Ch. 18, Exercise 5, '
            'Tables 18.11 and 18.12.'
        ),
        source_url='https://www.ou.edu/class/che-design/che5480-13/Smith-Chapter%2018.pdf',
        T_unit='C', Q_unit='MW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('1', 'hot', 120, 65, 0.5),
            ('2', 'hot', 80, 50, 0.3),
            ('3', 'hot', 135, 110, 0.29),
            ('4', 'hot', 220, 95, 0.02),
            ('5', 'hot', 135, 105, 0.26),
            ('6', 'cold', 65, 90, 0.15),
            ('7', 'cold', 75, 200, 0.14),
            ('8', 'cold', 30, 210, 0.1),
            ('9', 'cold', 60, 140, 0.05),
        ],
        targets=dict(Q_hot=20.95, Q_cold=31.75, pinch_hot=135.0, pinch_cold=115.0),
        published=dict(Q_hot=(20.95, 0.01), Q_cold=(31.75, 0.01), pinch_hot=(135, 0.0), pinch_cold=(115, 0.0)),
        proof=dict(
            side='below', rule='cp', pinch_hot=135.0, pinch_cold=115.0,
            hot_at_pinch=['3', '4', '5'],
            cold_at_pinch=['7', '8', '9'],
        ),
    ),
    dict(
        name='crude_fractionation_ph11c2',
        kind='constant_cp',
        citation=(
            'Yang, Z., Zhang, N. & Smith, R. (2022). Enhanced superstructure optimization for heat '
            'exchanger network synthesis using deterministic approach. Front. Sustain. 3:976717, '
            'Supplementary Table S.3 (case study 3, crude oil fractionation); original problem: Kim, '
            'S.Y., Jongsuwat, P., Suriyapraphadilok, U. & Bagajewicz, M. (2017). Global optimization '
            'of heat exchanger networks. Part 1: Stages/substages superstructure. Ind. Eng. Chem. '
            'Res. 56, 5944-5957, with heat capacities corrected by Pavao, Costa & Ravagnani (2018b) '
            'Ind. Eng. Chem. Res. 57, 2560-2573.'
        ),
        source_url=(
            'https://public-pages-files-2025.frontiersin.org/articles/976717/file/Data_Sheet_1.docx/9'
            '76717_supplementary-materials_datasheets_1_docx/1'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 140.2, 39.5, 106.5),
            ('H2', 'hot', 248.8, 110, 31.81),
            ('H3', 'hot', 170.1, 60, 33.93),
            ('H4', 'hot', 277, 121.9, 24.58),
            ('H5', 'hot', 250.6, 90, 132.2),
            ('H6', 'hot', 210, 163, 115.76),
            ('H7', 'hot', 303.6, 270.2, 234.98),
            ('H8', 'hot', 360, 290, 39.81),
            ('H9', 'hot', 178.6, 108.9, 47.85),
            ('H10', 'hot', 359.6, 280, 24.53),
            ('H11', 'hot', 290, 115, 39.81),
            ('C1', 'cold', 30, 130, 202.48),
            ('C2', 'cold', 130, 350, 289.92),
        ],
        targets=dict(Q_hot=19467.17199999999, Q_cold=7686.155999999995, pinch_hot=210.0, pinch_cold=200.0),
        published=None,
        proof=dict(
            side='above', rule='number', pinch_hot=210.0, pinch_cold=200.0,
            hot_at_pinch=['H2', 'H4', 'H5', 'H11'],
            cold_at_pinch=['C2'],
        ),
    ),
    dict(
        name='bagajewicz_ou_crude_unit_15_stream',
        kind='constant_cp',
        citation=(
            "Bagajewicz M (2013). CHE 5480 'Problems - Pinch Technology', Problem 2 (crude unit), "
            'University of Oklahoma.'
        ),
        source_url=(
            'https://www.ou.edu/class/che-design/che5480-13/Problems%20Pinch%20Technology.pdf'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=20,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('TCR', 'hot', 134.0, 108.2, 1007.8),
            ('MCR', 'hot', 227.7, 190.8, 356.19),
            ('LCR', 'hot', 268.1, 198.8, 403.31),
            ('KER', 'hot', 232.7, 40.0, 120.03),
            ('LGO', 'hot', 238.2, 45.0, 44.43),
            ('HGO', 'hot', 279.3, 68.0, 70.57),
            ('LR1', 'hot', 335.9, 262.7, 139.55),
            ('LR2', 'hot', 335.9, 186.1, 70.65),
            ('LR3', 'hot', 239.4, 186.2, 236.3),
            ('LR4', 'hot', 186.2, 90.0, 77.23),
            ('NAP', 'hot', 115.2, 56.0, 1460.04),
            ('C1', 'cold', 39.0, 153.0, 523.35),
            ('C2', 'cold', 153.0, 165.2, 619.8),
            ('C3', 'cold', 155.3, 348.0, 585.68),
            ('C4', 'cold', 150.0, 270.0, 175.8),
        ],
        targets=dict(Q_hot=56704.56799999998, Q_cold=96477.18799999998, pinch_hot=134.0, pinch_cold=114.0),
        published=None,
        proof=dict(
            side='above', rule='number', pinch_hot=134.0, pinch_cold=114.0,
            hot_at_pinch=['KER', 'LGO', 'HGO', 'LR4'],
            cold_at_pinch=['C1'],
        ),
    ),
    dict(
        name='bjork_pettersson_15sp_ph8c7',
        kind='constant_cp',
        citation=(
            'Caballero, J.A., Pavao, L.V., Costa, C.B.B. & Ravagnani, M.A.S.S. (2021). A Novel '
            'Sequential Approach for the Design of Heat Exchanger Networks. Front. Chem. Eng. '
            '3:733186, Supplementary Data Sheet 1, Table S21 (Example 11, PH8C7); original problem: '
            'Bjork, K.M. & Pettersson, F. (2003). Optimization of large-scale heat exchanger network '
            'synthesis problems. Proc. IASTED Modelling and Simulation, 313-318.'
        ),
        source_url=(
            'https://public-pages-files-2025.frontiersin.org/articles/733186/file/Data_Sheet_1.docx/7'
            '33186_supplementary-materials_datasheets_1_docx/1'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 180.0, 75.0, 30.0),
            ('H2', 'hot', 280.0, 120.0, 60.0),
            ('H3', 'hot', 180.0, 75.0, 30.0),
            ('H4', 'hot', 140.0, 40.0, 30.0),
            ('H5', 'hot', 220.0, 120.0, 50.0),
            ('H6', 'hot', 180.0, 55.0, 35.0),
            ('H7', 'hot', 200.0, 60.0, 30.0),
            ('H8', 'hot', 120.0, 40.0, 100.0),
            ('C1', 'cold', 40.0, 230.0, 20.0),
            ('C2', 'cold', 100.0, 220.0, 60.0),
            ('C3', 'cold', 40.0, 190.0, 35.0),
            ('C4', 'cold', 50.0, 190.0, 30.0),
            ('C5', 'cold', 50.0, 250.0, 60.0),
            ('C6', 'cold', 90.0, 190.0, 50.0),
            ('C7', 'cold', 160.0, 250.0, 60.0),
        ],
        targets=dict(Q_hot=8900.0, Q_cold=6525.0, pinch_hot=140.0, pinch_cold=130.0),
        published=None,
        proof=dict(
            side='above', rule='cp', pinch_hot=140.0, pinch_cold=130.0,
            hot_at_pinch=['H1', 'H2', 'H3', 'H5', 'H6', 'H7'],
            cold_at_pinch=['C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
        ),
    ),
    dict(
        name='fs_15sp_tkm',
        kind='constant_cp',
        citation=(
            'Furman, K.C. & Sahinidis, N.V. (2004). Approximation algorithms for the minimum number '
            'of matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565, test instance '15sp-tkm' (original source per the test set's references.txt: "
            "'kokossis:2000' (Tantimuratha, Kokossis & Mueller, 2000)); digitized in Letsios, D., "
            'Kouyialis, G. & Misener, R. (2018). Heuristics with performance guarantees for the '
            'minimum number of matches problem in heat recovery network design. Comput. Chem. Eng. '
            '113, 57-85 (data repository github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/furman_sahinidis/dat_files/15sp-tkm.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS1', 'hot', 137, 14, 5.3),
            ('HS2', 'hot', 135, 66, 16.8),
            ('HS3', 'hot', 111, 110, 3327),
            ('HS4', 'hot', 105, 95, 135.4),
            ('HS5', 'hot', 76, 66, 134.6),
            ('HS6', 'hot', 75, 74, 928),
            ('HS7', 'hot', 66, 65, 840),
            ('HS8', 'hot', 65, 64, 12000),
            ('HS9', 'hot', 37, 14, 324),
            ('CS1', 'cold', 4, 112, 56.6),
            ('CS2', 'cold', 76, 100, 108.1),
            ('CS3', 'cold', 70, 90, 205),
            ('CS4', 'cold', 38, 85, 18.4),
            ('CS5', 'cold', 45, 70, 280.1),
            ('CS6', 'cold', 4, 45, 314),
        ],
        targets=dict(Q_hot=5828.5, Q_cold=1338.0999999999976, pinch_hot=66.0, pinch_cold=56.0),
        published=dict(Q_hot=(5828.5, 0.1), Q_cold=(1338.1, 0.1), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='below', rule='number', pinch_hot=66.0, pinch_cold=56.0,
            hot_at_pinch=['HS1', 'HS7'],
            cold_at_pinch=['CS1', 'CS4', 'CS5'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='cgm_balanced8',
        kind='constant_cp',
        citation=(
            'Chen, Y., Grossmann, I.E. & Miller, D.C. (2015). Computational strategies for '
            'large-scale MILP transshipment models for heat exchanger network synthesis. Comput. '
            "Chem. Eng. 82, 68-83, benchmark instance 'balanced8' (minlp.org); as distributed in "
            'Letsios, D., Kouyialis, G. & Misener, R. (2018). Heuristics with performance guarantees '
            'for the minimum number of matches problem in heat recovery network design. Comput. Chem.'
            ' Eng. 113, 57-85 (data repository github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/chen_grossmann_miller/dat_files/balanced8.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS0', 'hot', 400.0, 120.0, 1.0),
            ('HS1', 'hot', 340.0, 120.0, 2.0),
            ('HS2', 'hot', 380.0, 150.0, 1.5),
            ('HS3', 'hot', 300.0, 100.0, 2.5),
            ('HS4', 'hot', 420.0, 160.0, 1.7),
            ('HS5', 'hot', 390.0, 110.0, 0.8),
            ('HS6', 'hot', 360.0, 200.0, 1.2),
            ('HS7', 'hot', 280.0, 130.0, 1.8),
            ('CS0', 'cold', 160.0, 400.0, 1.5),
            ('CS1', 'cold', 100.0, 250.0, 1.3),
            ('CS2', 'cold', 50.0, 300.0, 2.5),
            ('CS3', 'cold', 200.0, 380.0, 2.8),
            ('CS4', 'cold', 150.0, 450.0, 1.9),
            ('CS5', 'cold', 100.0, 180.0, 0.8),
            ('CS6', 'cold', 200.0, 350.0, 1.7),
            ('CS7', 'cold', 120.0, 330.0, 1.6),
        ],
        targets=dict(Q_hot=320.0, Q_cold=104.0, pinch_hot=210.0, pinch_cold=200.0),
        published=dict(Q_hot=(320, 1.0), Q_cold=(104, 1.0), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='number', pinch_hot=210.0, pinch_cold=200.0,
            hot_at_pinch=['HS0', 'HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6', 'HS7'],
            cold_at_pinch=['CS0', 'CS1', 'CS2', 'CS3', 'CS4', 'CS6', 'CS7'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='cgm_balanced10',
        kind='constant_cp',
        citation=(
            'Chen, Y., Grossmann, I.E. & Miller, D.C. (2015). Computational strategies for '
            'large-scale MILP transshipment models for heat exchanger network synthesis. Comput. '
            "Chem. Eng. 82, 68-83, benchmark instance 'balanced10' (minlp.org); as distributed in "
            'Letsios, D., Kouyialis, G. & Misener, R. (2018). Heuristics with performance guarantees '
            'for the minimum number of matches problem in heat recovery network design. Comput. Chem.'
            ' Eng. 113, 57-85 (data repository github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/chen_grossmann_miller/dat_files/balanced10.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS0', 'hot', 400.0, 120.0, 1.0),
            ('HS1', 'hot', 340.0, 120.0, 2.0),
            ('HS2', 'hot', 380.0, 150.0, 1.5),
            ('HS3', 'hot', 300.0, 100.0, 2.5),
            ('HS4', 'hot', 420.0, 160.0, 1.7),
            ('HS5', 'hot', 390.0, 110.0, 0.8),
            ('HS6', 'hot', 360.0, 200.0, 1.2),
            ('HS7', 'hot', 280.0, 130.0, 1.8),
            ('HS8', 'hot', 250.0, 80.0, 1.1),
            ('HS9', 'hot', 330.0, 170.0, 1.3),
            ('CS0', 'cold', 160.0, 400.0, 1.5),
            ('CS1', 'cold', 100.0, 250.0, 1.3),
            ('CS2', 'cold', 50.0, 300.0, 2.5),
            ('CS3', 'cold', 200.0, 380.0, 2.8),
            ('CS4', 'cold', 150.0, 450.0, 1.9),
            ('CS5', 'cold', 100.0, 180.0, 0.8),
            ('CS6', 'cold', 200.0, 350.0, 1.7),
            ('CS7', 'cold', 120.0, 330.0, 1.6),
            ('CS8', 'cold', 110.0, 220.0, 0.9),
            ('CS9', 'cold', 190.0, 360.0, 2.1),
        ],
        targets=dict(Q_hot=474.0, Q_cold=197.0, pinch_hot=210.0, pinch_cold=200.0),
        published=dict(Q_hot=(474, 1.0), Q_cold=(197, 1.0), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='number', pinch_hot=210.0, pinch_cold=200.0,
            hot_at_pinch=['HS0', 'HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6', 'HS7', 'HS8', 'HS9'],
            cold_at_pinch=['CS0', 'CS1', 'CS2', 'CS3', 'CS4', 'CS6', 'CS7', 'CS8', 'CS9'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='cgm_unbalanced10',
        kind='constant_cp',
        citation=(
            'Chen, Y., Grossmann, I.E. & Miller, D.C. (2015). Computational strategies for '
            'large-scale MILP transshipment models for heat exchanger network synthesis. Comput. '
            "Chem. Eng. 82, 68-83, benchmark instance 'unbalanced10' (minlp.org); as distributed in "
            'Letsios, D., Kouyialis, G. & Misener, R. (2018). Heuristics with performance guarantees '
            'for the minimum number of matches problem in heat recovery network design. Comput. Chem.'
            ' Eng. 113, 57-85 (data repository github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/chen_grossmann_miller/dat_files/unbalanced10.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS0', 'hot', 400.0, 120.0, 6.0),
            ('HS1', 'hot', 340.0, 120.0, 2.0),
            ('HS2', 'hot', 380.0, 150.0, 0.5),
            ('HS3', 'hot', 300.0, 100.0, 8.0),
            ('HS4', 'hot', 420.0, 160.0, 3.0),
            ('HS5', 'hot', 390.0, 110.0, 4.0),
            ('HS6', 'hot', 360.0, 200.0, 0.2),
            ('HS7', 'hot', 280.0, 130.0, 0.6),
            ('HS8', 'hot', 250.0, 80.0, 1.5),
            ('HS9', 'hot', 330.0, 170.0, 4.0),
            ('CS0', 'cold', 160.0, 400.0, 14.0),
            ('CS1', 'cold', 100.0, 250.0, 3.0),
            ('CS2', 'cold', 50.0, 300.0, 0.4),
            ('CS3', 'cold', 200.0, 380.0, 2.5),
            ('CS4', 'cold', 150.0, 450.0, 2.0),
            ('CS5', 'cold', 100.0, 180.0, 6.0),
            ('CS6', 'cold', 200.0, 350.0, 1.5),
            ('CS7', 'cold', 120.0, 330.0, 0.2),
            ('CS8', 'cold', 110.0, 220.0, 5.5),
            ('CS9', 'cold', 190.0, 360.0, 3.0),
        ],
        targets=dict(Q_hot=825.0, Q_cold=755.0, pinch_hot=300.0, pinch_cold=290.0),
        published=dict(Q_hot=(825, 1.0), Q_cold=(755, 1.0), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='cp', pinch_hot=300.0, pinch_cold=290.0,
            hot_at_pinch=['HS0', 'HS1', 'HS2', 'HS4', 'HS5', 'HS6', 'HS9'],
            cold_at_pinch=['CS0', 'CS2', 'CS3', 'CS4', 'CS6', 'CS7', 'CS9'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='luo_20sp_ph10c10',
        kind='constant_cp',
        citation=(
            'Caballero, J.A., Pavao, L.V., Costa, C.B.B. & Ravagnani, M.A.S.S. (2021). A Novel '
            'Sequential Approach for the Design of Heat Exchanger Networks. Front. Chem. Eng. '
            '3:733186, Supplementary Data Sheet 1, Table S27 (Example 14, PH10C10); original problem:'
            ' Luo, X., Wen, Q.Y. & Fieg, G. (2009). A hybrid genetic algorithm for synthesis of heat '
            'exchanger networks. Comput. Chem. Eng. 33, 1169-1181.'
        ),
        source_url=(
            'https://public-pages-files-2025.frontiersin.org/articles/733186/file/Data_Sheet_1.docx/7'
            '33186_supplementary-materials_datasheets_1_docx/1'
        ),
        T_unit='C', Q_unit='kW',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('H1', 'hot', 180.0, 75.0, 30.0),
            ('H2', 'hot', 280.0, 120.0, 15.0),
            ('H3', 'hot', 180.0, 75.0, 30.0),
            ('H4', 'hot', 140.0, 45.0, 30.0),
            ('H5', 'hot', 220.0, 120.0, 25.0),
            ('H6', 'hot', 180.0, 55.0, 10.0),
            ('H7', 'hot', 170.0, 45.0, 30.0),
            ('H8', 'hot', 180.0, 50.0, 30.0),
            ('H9', 'hot', 280.0, 90.0, 15.0),
            ('H10', 'hot', 180.0, 60.0, 30.0),
            ('C1', 'cold', 40.0, 230.0, 20.0),
            ('C2', 'cold', 120.0, 260.0, 35.0),
            ('C3', 'cold', 40.0, 190.0, 35.0),
            ('C4', 'cold', 50.0, 190.0, 30.0),
            ('C5', 'cold', 50.0, 250.0, 20.0),
            ('C6', 'cold', 40.0, 150.0, 10.0),
            ('C7', 'cold', 40.0, 150.0, 20.0),
            ('C8', 'cold', 120.0, 210.0, 35.0),
            ('C9', 'cold', 40.0, 130.0, 35.0),
            ('C10', 'cold', 60.0, 120.0, 30.0),
        ],
        targets=dict(Q_hot=4650.0, Q_cold=500.0, pinch_hot=180.0, pinch_cold=170.0),
        published=None,
        proof=dict(
            side='below', rule='cp', pinch_hot=180.0, pinch_cold=170.0,
            hot_at_pinch=['H1', 'H2', 'H3', 'H5', 'H6', 'H8', 'H9', 'H10'],
            cold_at_pinch=['C1', 'C2', 'C3', 'C4', 'C5', 'C8'],
        ),
    ),
    dict(
        name='fs_22sp1',
        kind='constant_cp',
        citation=(
            'Furman, K.C. & Sahinidis, N.V. (2004). Approximation algorithms for the minimum number '
            'of matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565, test instance '22sp1' (original source per the test set's references.txt: "
            "'miguel:1998'); digitized in Letsios, D., Kouyialis, G. & Misener, R. (2018). Heuristics"
            ' with performance guarantees for the minimum number of matches problem in heat recovery '
            'network design. Comput. Chem. Eng. 113, 57-85 (data repository '
            'github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/furman_sahinidis/dat_files/22sp1.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS1', 'hot', 137.8, 51.7, 5.28),
            ('HS2', 'hot', 160, 51.7, 10.12),
            ('HS3', 'hot', 232.2, 160, 12.93),
            ('HS4', 'hot', 187.8, 87.8, 9.6),
            ('HS5', 'hot', 232.2, 148.9, 10.98),
            ('HS6', 'hot', 160, 93.3, 21.1),
            ('HS7', 'hot', 248.9, 187.8, 7.92),
            ('HS8', 'hot', 154.4, 93.3, 10.47),
            ('HS9', 'hot', 160, 65.6, 15.04),
            ('HS10', 'hot', 232.2, 115.6, 13.19),
            ('HS11', 'hot', 243.3, 160, 4.414),
            ('CS1', 'cold', 37.8, 115.6, 6.89),
            ('CS2', 'cold', 54.4, 137.8, 8.23),
            ('CS3', 'cold', 65.6, 148.9, 14.86),
            ('CS4', 'cold', 43.3, 173.9, 9.23),
            ('CS5', 'cold', 173.9, 215.6, 18.47),
            ('CS6', 'cold', 82.2, 204.4, 6.87),
            ('CS7', 'cold', 173.9, 260, 22.56),
            ('CS8', 'cold', 60, 148.9, 9.1),
            ('CS9', 'cold', 85, 173.9, 6.33),
            ('CS10', 'cold', 65.6, 204.4, 12.23),
            ('CS11', 'cold', 176.7, 260, 19.81),
        ],
        targets=dict(Q_hot=2369.864400000001, Q_cold=647.8105999999993, pinch_hot=183.9, pinch_cold=173.9),
        published=dict(Q_hot=(2369.8644, 0.0001), Q_cold=(647.8106, 0.0001), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='number', pinch_hot=183.9, pinch_cold=173.9,
            hot_at_pinch=['HS3', 'HS4', 'HS5', 'HS10', 'HS11'],
            cold_at_pinch=['CS5', 'CS6', 'CS7', 'CS10'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='fs_22sp_ph',
        kind='constant_cp',
        citation=(
            'Furman, K.C. & Sahinidis, N.V. (2004). Approximation algorithms for the minimum number '
            'of matches problem in heat exchanger network synthesis. Ind. Eng. Chem. Res. 43(14), '
            "3554-3565, test instance '22sp-ph' (original source per the test set's references.txt: "
            "'polley:1999' (Polley & Heggs, 1999)); digitized in Letsios, D., Kouyialis, G. & "
            'Misener, R. (2018). Heuristics with performance guarantees for the minimum number of '
            'matches problem in heat recovery network design. Comput. Chem. Eng. 113, 57-85 (data '
            'repository github.com/cog-imperial/min_matches_heuristics).'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/furman_sahinidis/dat_files/22sp-ph.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS1', 'hot', 66, 65, 12.0),
            ('HS2', 'hot', 140, 30, 14),
            ('HS3', 'hot', 65, 38, 46),
            ('HS4', 'hot', 56, 55, 5.9),
            ('HS5', 'hot', 80, 30, 46),
            ('HS6', 'hot', 59, 58, 3.5),
            ('HS7', 'hot', 121, 38, 18),
            ('HS8', 'hot', 59, 58, 9.5),
            ('HS9', 'hot', 188, 30, 52.8),
            ('HS10', 'hot', 48, 47, 2.0),
            ('HS11', 'hot', 50, 49, 3.46),
            ('CS1', 'cold', 20, 80, 8),
            ('CS2', 'cold', 38, 80, 72),
            ('CS3', 'cold', 38, 49, 30),
            ('CS4', 'cold', 140, 141, 12.0),
            ('CS5', 'cold', 79, 80, 7.6),
            ('CS6', 'cold', 38, 80, 16),
            ('CS7', 'cold', 120, 121, 4.1),
            ('CS8', 'cold', 110, 111, 8.0),
            ('CS9', 'cold', 88, 204, 75.2),
            ('CS10', 'cold', 66, 67, 2.2),
            ('CS11', 'cold', 114, 115, 3.8),
        ],
        targets=dict(Q_hot=3209.8999999999996, Q_cold=4897.759999999999, pinch_hot=121.0, pinch_cold=111.0),
        published=dict(Q_hot=(3209.9, 0.1), Q_cold=(4897.76, 0.01), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='number', pinch_hot=121.0, pinch_cold=111.0,
            hot_at_pinch=['HS2', 'HS9'],
            cold_at_pinch=['CS9'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    dict(
        name='grossmann_balanced12_r0',
        kind='constant_cp',
        citation=(
            'Letsios, D., Kouyialis, G. & Misener, R. (2018). Heuristics with performance guarantees '
            'for the minimum number of matches problem in heat recovery network design. Comput. Chem.'
            " Eng. 113, 57-85, instance 'balanced12_random0' ('Randomly generated by random procedure"
            " and seed provided as a personal communication from Ignacio Grossmann to Ruth Misener', "
            'per the instance file header); data repository '
            'github.com/cog-imperial/min_matches_heuristics.'
        ),
        source_url=(
            'https://raw.githubusercontent.com/cog-imperial/min_matches_heuristics/master/data/origin'
            'al_instances/grossmann_random/dat_files/balanced12_random0.dat'
        ),
        T_unit='C', Q_unit='unspecified',  # CP in Q_unit per T_unit degree
        dTmin=10.0,
        streams=[  # (name, kind, T_in, T_out, CP) in source units
            ('HS0', 'hot', 400.0, 120.0, 1.0),
            ('HS1', 'hot', 340.0, 120.0, 1.8),
            ('HS2', 'hot', 380.0, 150.0, 1.6),
            ('HS3', 'hot', 300.0, 100.0, 2.5),
            ('HS4', 'hot', 420.0, 160.0, 1.8),
            ('HS5', 'hot', 390.0, 110.0, 0.8),
            ('HS6', 'hot', 360.0, 200.0, 1.3),
            ('HS7', 'hot', 280.0, 130.0, 1.7),
            ('HS8', 'hot', 250.0, 80.0, 1.2),
            ('HS9', 'hot', 330.0, 170.0, 1.2),
            ('HS10', 'hot', 430.0, 300.0, 2.2),
            ('HS11', 'hot', 200.0, 100.0, 2.4),
            ('CS0', 'cold', 160.0, 400.0, 1.6),
            ('CS1', 'cold', 100.0, 250.0, 1.3),
            ('CS2', 'cold', 50.0, 300.0, 2.3),
            ('CS3', 'cold', 200.0, 380.0, 2.6),
            ('CS4', 'cold', 150.0, 450.0, 2.0),
            ('CS5', 'cold', 100.0, 180.0, 0.8),
            ('CS6', 'cold', 200.0, 350.0, 1.7),
            ('CS7', 'cold', 120.0, 330.0, 1.7),
            ('CS8', 'cold', 110.0, 220.0, 0.9),
            ('CS9', 'cold', 190.0, 360.0, 2.3),
            ('CS10', 'cold', 260.0, 420.0, 1.7),
            ('CS11', 'cold', 80.0, 180.0, 1.2),
        ],
        targets=dict(Q_hot=482.0, Q_cold=323.0, pinch_hot=210.0, pinch_cold=200.0),
        published=dict(Q_hot=(482, 1.0), Q_cold=(323, 1.0), pinch_hot=None, pinch_cold=None),
        proof=dict(
            side='above', rule='number', pinch_hot=210.0, pinch_cold=200.0,
            hot_at_pinch=['HS0', 'HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6', 'HS7', 'HS8', 'HS9'],
            cold_at_pinch=['CS0', 'CS1', 'CS2', 'CS3', 'CS4', 'CS6', 'CS7', 'CS8', 'CS9'],
        ),
        note='heat unit not stated in the source; read as kW',
    ),
    # ----- real-thermodynamics problems -----
    dict(
        name='rtB01_above_3h2c_liquids',
        kind='real_thermo',
        description=(
            "Above the pinch (set by H4's supply, 360 K) three hot liquids but only two cold liquids."
        ),
        chemicals=['Water', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 50.0}, 420.0, 330.0, 500000.0, 'l', False),
            ('H2', {'Methanol': 40.0}, 400.0, 320.0, 1000000.0, 'l', False),
            ('H3', {'Water': 45.0}, 430.0, 340.0, 1000000.0, 'l', False),
            ('H4', {'Water': 200.0}, 360.0, 300.0, 500000.0, 'l', False),
            ('C1', {'Water': 100.0}, 320.0, 410.0, 500000.0, 'l', False),
            ('C2', {'Water': 100.0}, 330.0, 400.0, 500000.0, 'l', False),
        ],
        reference=dict(Q_hot=199865.70406719862, Q_cold=855202.7793932713, pinch_T_shifted=350.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2', 'H3'], cold_at_pinch=['C1', 'C2']),
    ),
    dict(
        name='rtB02_below_3c2h_liquids',
        kind='real_thermo',
        description=(
            "Below the pinch (set by C4's supply, 350 K) three cold liquids but only two hot liquids."
        ),
        chemicals=['Water', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 40.0}, 300.0, 380.0, 500000.0, 'l', False),
            ('C2', {'Methanol': 35.0}, 310.0, 370.0, 500000.0, 'l', False),
            ('C3', {'Water': 40.0}, 320.0, 365.0, 101325.0, 'l', False),
            ('H1', {'Water': 80.0}, 420.0, 320.0, 1000000.0, 'l', False),
            ('H2', {'Water': 70.0}, 410.0, 330.0, 1000000.0, 'l', False),
            ('C4', {'Water': 150.0}, 350.0, 415.0, 500000.0, 'l', False),
        ],
        reference=dict(Q_hot=313914.13748605247, Q_cold=34769.5691902393, pinch_T_shifted=350.0, grid_h=0.25, nV=201),
        proof=dict(side='below', rule='number', hot_at_pinch=['H1', 'H2'], cold_at_pinch=['C1', 'C2', 'C3']),
    ),
    dict(
        name='rtB03_above_hot_vapors',
        kind='real_thermo',
        description=(
            'Three superheated vapors (steam, methanol, ethanol; all 15+ K above their dew points) '
            "cross the pinch (set by hot water's supply, 405 K) against two cold streams (water "
            'liquid, ethanol vapor).'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 150.0}, 450.0, 390.0, 101325.0, 'g', True),
            ('H2', {'Methanol': 100.0}, 430.0, 360.0, 101325.0, 'g', True),
            ('H3', {'Ethanol': 60.0}, 440.0, 385.0, 200000.0, 'g', True),
            ('H4', {'Water': 200.0}, 405.0, 330.0, 1000000.0, 'l', False),
            ('C1', {'Water': 110.0}, 380.0, 440.0, 1000000.0, 'l', False),
            ('C2', {'Ethanol': 100.0}, 370.0, 430.0, 101325.0, 'g', True),
        ],
        reference=dict(Q_hot=130823.82884698862, Q_cold=1216207.413343743, pinch_T_shifted=395.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2', 'H3'], cold_at_pinch=['C1', 'C2']),
    ),
    dict(
        name='rtB04_below_cold_boilers',
        kind='real_thermo',
        description=(
            'Three cold streams heated to saturated vapor (ethanol, water at 1 atm, methanol at 2 '
            'bar) cross the pinch as liquids (boiling 11-33 K above it) against two hot streams '
            '(water liquid; 5 bar steam desuperheat+condense+subcool).'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Ethanol': 20.0}, 300.0, 'dew', 101325.0, 'l', True),
            ('C2', {'Water': 10.0}, 310.0, 'dew', 101325.0, 'l', True),
            ('C3', {'Methanol': 25.0}, 300.0, 'dew', 200000.0, 'l', True),
            ('H1', {'Water': 80.0}, 440.0, 320.0, 1000000.0, 'l', False),
            ('H2', {'Water': 20.0}, 470.0, 330.0, 500000.0, 'g', True),
            ('C4', {'Water': 200.0}, 340.0, 420.0, 1000000.0, 'l', False),
        ],
        reference=dict(Q_hot=1895887.4494512205, Q_cold=5772.621524257353, pinch_T_shifted=340.0, grid_h=0.25, nV=201),
        proof=dict(side='below', rule='number', hot_at_pinch=['H1', 'H2'], cold_at_pinch=['C1', 'C2', 'C3']),
    ),
    dict(
        name='rtB05_above_2h1c',
        kind='real_thermo',
        description='Minimal: two hot liquids (water, ethanol) vs one cold liquid above the pinch.',
        chemicals=['Water', 'Ethanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 60.0}, 400.0, 320.0, 500000.0, 'l', False),
            ('H2', {'Ethanol': 30.0}, 390.0, 330.0, 500000.0, 'l', False),
            ('C1', {'Water': 150.0}, 310.0, 410.0, 500000.0, 'l', False),
            ('H3', {'Water': 200.0}, 360.0, 300.0, 101325.0, 'l', False),
        ],
        reference=dict(Q_hot=370155.00754718046, Q_cold=752527.3075279158, pinch_T_shifted=350.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2'], cold_at_pinch=['C1']),
    ),
    dict(
        name='rtB06_below_2c1h',
        kind='real_thermo',
        description=(
            'Minimal: two cold liquids (water, methanol) vs one hot liquid below the pinch.'
        ),
        chemicals=['Water', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 50.0}, 300.0, 360.0, 101325.0, 'l', False),
            ('C2', {'Methanol': 40.0}, 305.0, 360.0, 300000.0, 'l', False),
            ('H1', {'Water': 120.0}, 430.0, 310.0, 1000000.0, 'l', False),
            ('C3', {'Water': 150.0}, 340.0, 420.0, 1000000.0, 'l', False),
        ],
        reference=dict(Q_hot=331162.5377505963, Q_cold=89804.33369865306, pinch_T_shifted=340.0, grid_h=0.25, nV=201),
        proof=dict(side='below', rule='number', hot_at_pinch=['H1'], cold_at_pinch=['C1', 'C2']),
    ),
    dict(
        name='rtB07_above_with_cold_glide',
        kind='real_thermo',
        description=(
            "Three hot liquids vs two cold liquids above the pinch (set by H4's supply, 330 K), with "
            'a water/methanol xM=0.2 glide boiler above the pinch.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 30.0}, 400.0, 300.0, 500000.0, 'l', False),
            ('H2', {'Methanol': 25.0}, 370.0, 305.0, 500000.0, 'l', False),
            ('H3', {'Ethanol': 15.0}, 380.0, 310.0, 500000.0, 'l', False),
            ('C1', {'Water': 60.0}, 300.0, 360.0, 101325.0, 'l', False),
            ('C2', {'Ethanol': 25.0}, 305.0, 345.0, 101325.0, 'l', False),
            ('H4', {'Water': 100.0}, 330.0, 300.0, 101325.0, 'l', False),
            ('C3', {'Water': 12.0, 'Methanol': 3.0}, 330.0, 372.0, 101325.0, 'l', True),
        ],
        reference=dict(Q_hot=543031.7081302783, Q_cold=248869.0700432139, pinch_T_shifted=320.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2', 'H3'], cold_at_pinch=['C1', 'C2']),
    ),
    dict(
        name='rtB08_above_steam_creator',
        kind='real_thermo',
        description=(
            'Three hot streams (water liquid, methanol vapor, ethanol liquid at 10 bar) vs two cold '
            '(water liquid, methanol vapor) above a pinch set by the supply of 1 atm superheated '
            'steam that condenses 27 K below the pinch.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 40.0}, 440.0, 360.0, 1000000.0, 'l', False),
            ('H2', {'Methanol': 50.0}, 440.0, 380.0, 200000.0, 'g', True),
            ('H3', {'Ethanol': 20.0}, 420.0, 350.0, 1000000.0, 'l', False),
            ('H4', {'Water': 150.0}, 400.0, 330.0, 101325.0, 'g', True),
            ('C1', {'Water': 100.0}, 350.0, 445.0, 1000000.0, 'l', False),
            ('C2', {'Methanol': 50.0}, 360.0, 430.0, 101325.0, 'g', True),
        ],
        reference=dict(Q_hot=232877.7794391609, Q_cold=6666506.837679262, pinch_T_shifted=390.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2', 'H3'], cold_at_pinch=['C1', 'C2']),
    ),
    dict(
        name='rtB09_below_with_cold_glide',
        kind='real_thermo',
        description=(
            "Two cold liquids vs one hot liquid below the pinch (set by C4's supply, 340 K); "
            'water/methanol xM=0.2 glide boiler (bubble -> dew) above the pinch.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 40.0}, 300.0, 365.0, 101325.0, 'l', False),
            ('C2', {'Methanol': 35.0}, 305.0, 350.0, 200000.0, 'l', False),
            ('H1', {'Water': 100.0}, 420.0, 300.0, 1000000.0, 'l', False),
            ('C4', {'Water': 120.0}, 340.0, 410.0, 500000.0, 'l', False),
            ('C5', {'Water': 12.0, 'Methanol': 3.0}, 'bubble', 'dew', 101325.0, 'l', True),
        ],
        reference=dict(Q_hot=820543.2968615409, Q_cold=150099.186024755, pinch_T_shifted=340.0, grid_h=0.25, nV=201),
        proof=dict(side='below', rule='number', hot_at_pinch=['H1'], cold_at_pinch=['C1', 'C2']),
    ),
    dict(
        name='rtB10_11_streams_above',
        kind='real_thermo',
        description=(
            '11 streams: three hot (water, methanol liquids, ethanol vapor) vs two cold liquids at '
            "the pinch above (set by H6's supply, 360 K); steam condenser, water boiler, ethanol and "
            'water/methanol liquids elsewhere.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 30.0}, 440.0, 330.0, 1000000.0, 'l', False),
            ('H2', {'Methanol': 25.0}, 400.0, 320.0, 1000000.0, 'l', False),
            ('H3', {'Ethanol': 15.0}, 420.0, 330.0, 101325.0, 'g', True),
            ('H4', {'Water': 10.0}, 'dew', 'bubble', 300000.0, 'l', True),
            ('H5', {'Water': 40.0}, 350.0, 300.0, 101325.0, 'l', False),
            ('H6', {'Water': 150.0}, 360.0, 310.0, 101325.0, 'l', False),
            ('C1', {'Water': 60.0}, 320.0, 410.0, 500000.0, 'l', False),
            ('C2', {'Methanol': 50.0}, 330.0, 375.0, 500000.0, 'l', False),
            ('C3', {'Water': 10.0}, 'bubble', 'dew', 101325.0, 'l', True),
            ('C4', {'Ethanol': 30.0}, 300.0, 340.0, 101325.0, 'l', False),
            ('C5', {'Water': 24.0, 'Methanol': 6.0}, 300.0, 345.0, 101325.0, 'l', False),
        ],
        reference=dict(Q_hot=50692.99592223286, Q_cold=1037531.089196143, pinch_T_shifted=350.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2', 'H3'], cold_at_pinch=['C1', 'C2']),
    ),
    dict(
        name='rtB11_12_streams_below',
        kind='real_thermo',
        description=(
            '12 streams: three cold liquids vs two hot (water liquid, methanol vapor) at the pinch '
            "below (set by C4's supply, 360 K); ethanol and methanol condensers, EM glide condenser, "
            'water boiler, water/methanol liquid elsewhere.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('C1', {'Water': 30.0}, 300.0, 368.0, 101325.0, 'l', False),
            ('C2', {'Methanol': 25.0}, 310.0, 365.0, 300000.0, 'l', False),
            ('C3', {'Ethanol': 15.0}, 305.0, 375.0, 300000.0, 'l', False),
            ('H1', {'Water': 90.0}, 440.0, 320.0, 1000000.0, 'l', False),
            ('H2', {'Methanol': 40.0}, 420.0, 365.0, 200000.0, 'g', True),
            ('C4', {'Water': 150.0}, 360.0, 440.0, 1000000.0, 'l', False),
            ('H3', {'Ethanol': 20.0}, 'dew', 'bubble', 101325.0, 'l', True),
            ('H4', {'Water': 40.0}, 345.0, 300.0, 101325.0, 'l', False),
            ('H5', {'Ethanol': 10.0, 'Methanol': 10.0}, 360.0, 320.0, 101325.0, 'g', True),
            ('H6', {'Methanol': 10.0}, 'dew', 'bubble', 101325.0, 'l', True),
            ('C5', {'Water': 16.0, 'Methanol': 4.0}, 300.0, 340.0, 101325.0, 'l', False),
            ('C6', {'Water': 10.0}, 'bubble', 'dew', 200000.0, 'l', True),
        ],
        reference=dict(Q_hot=795155.8893279578, Q_cold=2023387.438575476, pinch_T_shifted=360.0, grid_h=0.25, nV=201),
        proof=dict(side='below', rule='number', hot_at_pinch=['H1', 'H2'], cold_at_pinch=['C1', 'C2', 'C3']),
    ),
    dict(
        name='rtB12_above_2h1c_cond_boil_below',
        kind='real_thermo',
        description=(
            "Two hot (water liquid, methanol vapor) vs one cold liquid above the pinch (set by H4's "
            'supply, 380 K); methanol condenser and methanol boiler (1 atm) far below the pinch.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 40.0}, 440.0, 340.0, 1000000.0, 'l', False),
            ('H2', {'Methanol': 60.0}, 430.0, 365.0, 200000.0, 'g', True),
            ('H3', {'Methanol': 15.0}, 'dew', 'bubble', 101325.0, 'l', True),
            ('H4', {'Water': 150.0}, 380.0, 320.0, 500000.0, 'l', False),
            ('C1', {'Water': 100.0}, 350.0, 430.0, 1000000.0, 'l', False),
            ('C2', {'Methanol': 6.0}, 'bubble', 'dew', 101325.0, 'l', True),
        ],
        reference=dict(Q_hot=119397.5250737188, Q_cold=1012224.379402471, pinch_T_shifted=370.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2'], cold_at_pinch=['C1']),
    ),
    dict(
        name='rtB13_above_3h2c_glide_below',
        kind='real_thermo',
        description=(
            'Three hot (water, ethanol liquids at 10 bar; methanol vapor) vs two cold water liquids '
            "above the pinch (set by H4's supply, 390 K); water/ethanol xE=0.1 glide condenser and "
            'ethanol boiler below the pinch.'
        ),
        chemicals=['Water', 'Ethanol', 'Methanol'],
        T_min_app=10.0,
        streams=[  # (ID, {chemical: kmol/hr}, T_in, T_out, P [Pa], phase_in, rigorous)
            ('H1', {'Water': 40.0}, 450.0, 350.0, 1000000.0, 'l', False),
            ('H2', {'Ethanol': 20.0}, 415.0, 340.0, 1000000.0, 'l', False),
            ('H3', {'Methanol': 50.0}, 440.0, 385.0, 300000.0, 'g', True),
            ('H4', {'Water': 150.0}, 390.0, 330.0, 500000.0, 'l', False),
            ('C1', {'Water': 90.0}, 360.0, 445.0, 1000000.0, 'l', False),
            ('C2', {'Water': 50.0}, 370.0, 420.0, 500000.0, 'l', False),
            ('H5', {'Water': 18.0, 'Ethanol': 2.0}, 385.0, 345.0, 101325.0, 'g', True),
            ('C3', {'Ethanol': 10.0}, 'bubble', 'dew', 101325.0, 'l', True),
            ('C4', {'Water': 60.0}, 300.0, 350.0, 101325.0, 'l', False),
        ],
        reference=dict(Q_hot=205293.29827915056, Q_cold=1028410.1302859504, pinch_T_shifted=380.0, grid_h=0.25, nV=201),
        proof=dict(side='above', rule='number', hot_at_pinch=['H1', 'H2', 'H3'], cold_at_pinch=['C1', 'C2']),
    ),
]
