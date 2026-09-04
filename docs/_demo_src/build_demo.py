# -*- coding: utf-8 -*-
# hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
# Thermodynamics, and Heuristics
# Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
# for license details.
"""Build the interactive quickstart demo from its template.

Fills the image placeholders with paths relative to ``_static``, injects the
terminal panes from the chapter-01 captures, generates the callouts from
``ch01_summary.txt`` and asserts that every callout number appears in the
captures, that every code line shown is a whitespace-normalised substring of
the chapter-01 script (the one allowed difference is the omitted
``cache=True``), and that no non-ASCII character or placeholder survives.
Fails loudly if the demo would drift from the tutorial.

    python docs/_demo_src/build_demo.py
    -> docs/source/_static/quickstart_demo.html
"""
import html
import json
import re
import _common
from _common import STATIC, GENERATED, HERE, report

TEMPLATE = HERE / 'quickstart_demo_template.html'
OUT = STATIC / 'quickstart_demo.html'
CH01 = HERE / 'examples' / 'ch01_quickstart.py'

IMAGES = {
    '__LOGO_LIGHT__': 'images/logo/logo_hensmith_light.png',
    '__LOGO_DARK__': 'images/logo/logo_hensmith_dark.png',
    '__MARK_LIGHT__': 'images/logo/mark_hensmith_light.png',
    '__MARK_DARK__': 'images/logo/mark_hensmith_dark.png',
    '__FIG_FLOWSHEET_LIGHT__': 'images/examples/tutorial_01_quickstart_flowsheet_light.png',
    '__FIG_FLOWSHEET_DARK__': 'images/examples/tutorial_01_quickstart_flowsheet_dark.png',
    '__FIG_PINCH__': 'images/examples/tutorial_01_quickstart_pinch_diagram.png',
}
ENTITIES = {'—': '&mdash;', '–': '&ndash;', '→': '&rarr;', '←': '&larr;', '·': '&middot;',
            '≈': '&asymp;', '×': '&times;', '−': '&minus;', '’': '&rsquo;', '“': '&ldquo;',
            '”': '&rdquo;', '°': '&deg;', 'Δ': '&Delta;', '…': '&hellip;'}


def read(name):
    return (GENERATED / f'{name}.txt').read_text(encoding='utf-8')


def summary():
    return dict(line.split(' = ', 1) for line in read('ch01_summary').splitlines() if ' = ' in line)


def terminal(text):
    return html.escape(text.replace('\t', '    ').rstrip('\n'))


def callouts(S):
    neg = lambda s: s.replace('-', '−')
    return {
        '__CALLOUTS_01__': [(S['n_streams'], 'streams to integrate'),
                            (S['n_auxiliary'], 'auxiliary exchangers')],
        '__CALLOUTS_02__': [(neg('-' + S['heating_reduction_percent']) + ' %', 'heating utility'),
                            (neg('-' + S['cooling_reduction_percent']) + ' %', 'cooling utility'),
                            (neg(S['utility_cost_usd_per_hr']) + ' USD/hr', 'utility cost (savings)')],
        '__CALLOUTS_03__': [(S['n_process_hxs'], 'process exchangers'),
                            ('pinch', 'cold side | hot side'),
                            (S['installed_cost_usd'] + ' USD', 'added installed cost')],
        '__CALLOUTS_04__': [(S['stream_1_process_hxs'], 'exchangers on Stream_1 before its utility'),
                            (S['energy_balance_percent_error_abs'] + ' %', 'energy balance error')],
    }


def check_callouts(cal, S):
    haystack = ' '.join([read('ch01_summary'), read('ch01_results'), read('ch01_loads')])
    haystack = re.sub(r'\s+', ' ', haystack)
    for token, items in cal.items():
        for v, _ in items:
            for num in re.findall(r'\d[\d.]*(?:e[+-]?\d+)?', v):
                assert num in haystack, f'{token}: {v!r} ({num}) not found in the chapter-01 captures'


def check_code_lines(tpl):
    """Assert every code line of the demo's scene 01 is real chapter-01 code.

    The pane wraps statements differently from the script, so lines are not
    compared one to one; instead the *whole* script is normalised to a single
    space-separated string and each normalised demo line must be a substring
    of it. That is strictly stronger than a per-line match: a re-wrapped
    statement still matches (whitespace, including newlines, is collapsed),
    but a line whose remainder drifts from the script no longer passes just
    because it happens to contain one long script line.
    """
    script = CH01.read_text(encoding='utf-8')
    normalized = re.sub(r'\s+', ' ', script)
    normalized_no_cache = normalized.replace(', cache=True', '')
    # the array ends at the "]," that precedes the caption (code strings contain "]" too)
    m = re.search(r'label:"Build the flowsheet".*?code:\[(.*?)\],\s*caption:', tpl, re.S)
    assert m, 'scene 01 code block not found'
    lines = json.loads('[' + m.group(1).strip().rstrip(',') + ']')
    for line in lines:
        key = re.sub(r'\s+', ' ', line.strip())
        if not key or key == 'sys.diagram()': continue
        assert key in normalized or key in normalized_no_cache, \
            f'demo code line is not in ch01_quickstart.py: {line!r}'
    return lines


def main():
    tpl = TEMPLATE.read_text(encoding='utf-8')
    S = summary()
    for token, rel in IMAGES.items():
        assert (STATIC / rel).exists(), rel
        assert token in tpl, token
        tpl = tpl.replace(token, rel)
    tpl = tpl.replace('__TERM_RESULTS__', terminal(read('ch01_results')))
    tpl = tpl.replace('__TERM_LIFECYCLES__', terminal(read('ch01_life_cycles')))
    cal = callouts(S)
    check_callouts(cal, S)
    for token, items in cal.items():
        assert token in tpl, token
        tpl = tpl.replace(token, json.dumps([{'v': v, 'l': l} for v, l in items], ensure_ascii=False))
    check_code_lines(tpl)
    for ch, ent in ENTITIES.items():
        tpl = tpl.replace(ch, ent)
    leftover = sorted({c for c in tpl if ord(c) > 127})
    assert not leftover, f'un-encoded non-ASCII characters remain: {leftover}'
    assert not re.search(r'__[A-Z0-9_]+__', tpl), re.findall(r'__[A-Z0-9_]+__', tpl)
    OUT.write_text(tpl, encoding='utf-8', newline='\n')
    for rel in re.findall(r'"(images/[^"]+\.png)"', tpl):
        assert (STATIC / rel).exists(), rel
    report(OUT)


if __name__ == '__main__':
    main()
