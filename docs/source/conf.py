# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder (hensmith).
import os
import re
import sys
import pathlib
import importlib.util

os.environ.setdefault('NUMBA_DISABLE_JIT', '1')      # docs never need JIT; avoids numba cache writes
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                         # document this checkout even if not pip-installed

# Fallback only: mock modules that are genuinely missing. Inert when the
# biosteam stack is installed (locally via PYTHONPATH; on RTD via post_install).
autodoc_mock_imports = [
    m for m in ('biosteam', 'thermosteam', 'numba')
    if importlib.util.find_spec(m) is None
]

import hensmith  # noqa: E402
print(f'[conf.py] documenting hensmith {hensmith.__version__} from {hensmith.__file__}')

project = 'hensmith'
author = 'Sarang S. Bhagwat'
copyright = '2020-2026, BioSTEAM Development Group'
version = release = hensmith.__version__

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinx_copybutton',
]

autosummary_generate = False                          # API pages are hand-written
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'biosteam': ('https://biosteam.readthedocs.io/en/latest/', None),
    'thermosteam': ('https://thermosteam.readthedocs.io/en/latest/', None),
}
intersphinx_disabled_domains = ['std']
intersphinx_timeout = 30
if os.environ.get('SPHINX_OFFLINE'):                  # local builds without network: -W would fail on inventories
    intersphinx_mapping = {}

templates_path = []
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'
html_title = 'hensmith'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_favicon = '_static/images/logo/mark_hensmith_light.png'
html_theme_options = {
    'logo': {
        'image_light': '_static/images/logo/logo_hensmith_light.png',
        'image_dark': '_static/images/logo/logo_hensmith_dark.png',
    },
    'show_toc_level': 2,
    'icon_links': [
        {'name': 'GitHub', 'url': 'https://github.com/BioSTEAMDevelopmentGroup/hensmith',
         'icon': 'fa-brands fa-github', 'type': 'fontawesome'},
        {'name': 'PyPI', 'url': 'https://pypi.org/project/hensmith/',
         'icon': 'fa-brands fa-python', 'type': 'fontawesome'},
    ],
}

copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

linkcheck_ignore = [r'https://doi\.org/.*']           # DOIs 403 on bots; checked manually

# --- Loud guard: every asset referenced by the sources must be committed ---
# RTD never runs docs/_demo_src (no graphviz there, and the outputs are the
# reviewed source of truth), so a figure/capture/GIF that was not regenerated
# and committed must fail the build with a clear list, not render as an empty
# include or a broken image.
SRC = pathlib.Path(__file__).parent
_missing = []
for _rst in sorted(SRC.rglob('*.rst')):
    _text = _rst.read_text(encoding='utf-8')
    _refs = [(d, t) for d, t in re.findall(
        r'^\s*\.\. (literalinclude|figure|image):: (\S+)', _text, re.M)]
    _refs += [('html', t) for t in re.findall(r'(?:src|srcset)="(_static/[^"\s]+)"', _text)]
    for _directive, _target in _refs:
        _path = (SRC / _target.lstrip('/')) if _target.startswith('/') else (_rst.parent / _target)
        if not _path.resolve().exists():
            _missing.append(f'{_rst.relative_to(SRC)}: {_directive}:: {_target}')
if _missing:
    raise FileNotFoundError(
        'missing committed asset(s); run docs/_demo_src/build_all.py locally and commit:\n  '
        + '\n  '.join(_missing))
