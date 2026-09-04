Contributing to hensmith
========================

Bug reports, regression cases, and pull requests are welcome. This page
describes how the repository is laid out, how to set up an environment that
matches continuous integration, how the tests work, and the conventions a
change has to respect.

Where the code lives
--------------------

The library is two modules under ``hensmith/``:

``hensmith/_heat_exchanger_network.py``
    The ``HeatExchangerNetwork`` facility unit: its ``_run``, ``_design`` and
    ``_cost`` methods, and its integration with a BioSTEAM system's heat
    utilities.

``hensmith/hxn_synthesis.py``
    The algorithms: the problem table and pinch analysis (``problem_table``,
    ``ProblemTable``), network synthesis (``synthesize_network``), the
    per-stream bookkeeping of ``StreamLifeCycle``, and ``plot_pinch_diagram``.

``hensmith/__init__.py`` re-exports the ``__all__`` of both modules and holds
the biosteam registration block described in `The import contract`_ below.
The public API of both modules is documented under :doc:`../API/api`.

Tests live in ``tests/``:

``tests/test_hxn.py``
    Behavior of the ``HeatExchangerNetwork`` unit on small, hand-built
    systems.

``tests/test_hxn_regression.py``
    Ten synthetic systems of increasing complexity. For each, the synthesized
    network must close its energy balance, must not beat the minimum-energy
    requirement targets of the problem table computed on the same streams,
    and must recover at least as much heat as the utility loads documented in
    the file.

Development environment
-----------------------

hensmith requires Python 3.12 or newer; continuous integration runs 3.12 and
3.13. Its only runtime dependency is BioSTEAM.

Because hensmith tracks the development version of the BioSTEAM stack,
install that stack from GitHub HEAD exactly as CI does, from a checkout of
this repository:

.. code-block:: bash

   pip install --upgrade pip
   pip install wheel pytest
   pip install --no-cache-dir git+https://github.com/CalebBell/fluids.git
   pip install --no-cache-dir git+https://github.com/CalebBell/chemicals.git
   pip install --no-cache-dir git+https://github.com/CalebBell/thermo.git
   pip install --no-cache-dir git+https://github.com/BioSTEAMDevelopmentGroup/thermosteam.git
   pip install --no-cache-dir git+https://github.com/BioSTEAMDevelopmentGroup/biosteam.git
   pip uninstall hensmith -y
   pip install --no-deps -e .

Two steps in that sequence are deliberate.

``pip uninstall hensmith -y``
    Installing biosteam pulls in a released hensmith as a dependency. It is
    removed before the checkout is installed in editable mode, so that
    ``import hensmith`` resolves to the working tree and not to a stale copy
    in ``site-packages``.

``pip install --no-deps -e .``
    ``setup.py`` pins ``biosteam>=2.54.0``: 2.54.0 is the first biosteam
    release without the bundled ``biosteam.facilities.hxn`` copy of this
    package. An older biosteam would coexist with hensmith as two distinct
    ``HeatExchangerNetwork`` classes, with the bundled copy winning
    ``bst.HeatExchangerNetwork`` and silently breaking ``isinstance`` checks
    downstream. The pin is circular by design -- biosteam depends on hensmith
    and hensmith pins biosteam -- so letting pip resolve it here would fail
    whenever this repository's floor is ahead of the biosteam release on
    PyPI. ``--no-deps`` skips that resolution; the stack installed above
    already satisfies the requirement.

Running the tests
-----------------

Run the suite from the repository root:

.. code-block:: bash

   pytest . --disable-numba=1

CI runs ``pytest . -v --disable-numba=1 -p no:cacheprovider``. The suite is
expected to be fully green: exit status 0, no failures and no errors.

``--disable-numba`` is defined in ``conftest.py`` and defaults to ``1``; it
sets ``NUMBA_DISABLE_JIT``, so the tests run interpreted instead of paying
for (and caching) numba compilation. ``conftest.py`` also sets
``DISABLE_PREFERENCES`` (so that a run does not depend on locally saved
BioSTEAM preferences), ``FILTER_WARNINGS`` and ``PY_IGNORE_IMPORTMISMATCH``.

**Doctests are tests.** ``pytest.ini`` adds ``--doctest-modules``, so every
``Examples`` block in every module under ``hensmith/`` is executed and its
output compared. The option flags are ``NORMALIZE_WHITESPACE``,
``IGNORE_EXCEPTION_DETAIL``, ``NUMBER`` and ``ELLIPSIS``; ``NUMBER`` compares
floating-point results only to the precision written in the expected output,
so a docstring that prints fewer digits tolerates more drift.

A change to a synthesis or design algorithm legitimately changes the numbers
a doctest prints. **Regenerate those expected outputs by running the example
and pasting its actual output** -- never hand-tune digits until a comparison
passes, which turns a real regression into a green test. Say in the commit
body which behavior change moved the numbers.

The regression suite states the same rule in stronger form. The documented
utility loads in ``tests/test_hxn_regression.py`` were recorded by running
that file directly (it prints them), and an improvement to the synthesizer
simply leaves slack against them. A maintainer lowers a documented load
deliberately when a better network is intended; the module docstring is
explicit about the other direction:

    Never raise them to make a failing test pass.

Raising a documented load converts a synthesizer regression into a passing
test, which is exactly what the case list exists to prevent.

The import contract
-------------------

biosteam and hensmith refer to each other on purpose: ``bst.HeatExchangerNetwork``
works out of the box, and hensmith is still a separately released package.
The cycle is import-safe in both import orders only because of the following
invariants. Breaking one makes the import fail in a way that depends on which
package the user imported first, so treat them as part of the code:

* **Never star-import biosteam** (``from biosteam import *``) at module
  scope, anywhere in this package or in any module imported while hensmith is
  initializing. biosteam lists ``HeatExchangerNetwork`` in its ``__all__`` as
  a re-export *from hensmith*, and the name is bound into biosteam only at
  the very end of ``hensmith/__init__.py``. A star-import running before that
  point resolves the name against a half-initialized module and fails. Plain
  ``import biosteam as bst`` is safe, and is what the modules use.

* **The registration block stays last.** biosteam imports hensmith at the
  very end of its own ``__init__``, and hensmith's ``__init__`` sets
  ``bst.HeatExchangerNetwork`` and ``bst.facilities.HeatExchangerNetwork`` as
  its final statement. Binding last is what makes either order work: when
  hensmith is imported first, biosteam's ``__init__`` runs while hensmith is
  still initializing and the class does not exist yet, so the name can only
  be bound once hensmith's own definitions are complete.

* **The name stays out of** ``biosteam.facilities.__all__``. biosteam
  star-imports its ``facilities`` subpackage before hensmith's registration
  block can run; listing the name there would resolve it eagerly and
  re-create the cycle.

biosteam's own test suite guards the star-import invariant, so breaking it
also fails CI in that repository.

Conventions
-----------

**Branches.** One branch per fix or feature, branched from ``master`` and
named in lowercase kebab-case (``fix-<what>``, ``<feature>-<noun>``). Do not
commit to ``master`` directly, and start a new branch for an unrelated fix
rather than piling it onto the current one.

**Commit messages.** A lowercase, imperative, one-line summary, then a body
explaining the *underlying cause*, the reasoning, and how the change was
validated -- not a restatement of the diff.

**Copyright header.** Every module starts with the hensmith header:

.. code-block:: python

   # -*- coding: utf-8 -*-
   # hensmith: Heat Exchanger Network Synthesis, Modeling, Integration,
   # Thermodynamics, and Heuristics
   # Copyright (C) 2020-, Sarang Bhagwat <sarangbhagwat.developer@gmail.com>
   #
   # This module is under the UIUC open-source license. See
   # github.com/BioSTEAMDevelopmentGroup/hensmith/blob/master/LICENSE.txt
   # for license details.

New files use it with the current year and the contributing author. Do not
reorder the authors already listed in a file; append a new one at the end.

**Docstrings.** NumPy style, with the usual ``Parameters``, ``Returns``,
``Notes``, ``Examples`` and ``References`` sections, and a runnable doctest
under ``Examples`` for anything public. Cite the pinch-analysis or network
synthesis methodology under ``References``.

**Exports.** Public names are ``__all__``-driven: add a new public class or
function to its module's ``__all__``, which ``hensmith/__init__.py`` unions.

**Public API and defaults.** The signature of ``HeatExchangerNetwork`` and
the public functions of ``hxn_synthesis`` are used by downstream biorefinery
models, so prefer additive, backward-compatible changes. Default design and
synthesis parameters -- the minimum approach temperature, the cost and area
correlations, the constructor defaults -- are part of that contract:
changing one shifts every downstream techno-economic analysis, so propose
such a change on its own rather than folding it into an unrelated fix.

**Style.** Match the surrounding module: four-space indentation, no trailing
whitespace, private helpers prefixed with ``_``, and line lengths in keeping
with the file being edited. Because the oldest Python CI runs is 3.12, avoid syntax and
standard-library features newer than 3.12.

**Fix causes, not symptoms.** A clamp, a broad ``except``, or a retry loop
that makes a traceback disappear is a fix only when the analysis shows it is
the correct behavior. Pinch analysis has enough structure -- monotone
cascades, piecewise-linear composite curves, energy balances -- that a
closed-form or bracketed-monotone solution often exists where an iterative
one is in place; it is worth looking for one before patching the iteration,
and worth writing the derivation down when it replaces numerics.

Documentation
-------------

The documentation is Sphinx sources under ``docs/source``. Build it from the
repository root:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -W -b html docs/source docs/build/html

``-W`` turns warnings into errors, matching the Read the Docs configuration:
a page that builds cleanly locally builds cleanly there.

Figures, animations, and captured program output are not written by hand.
They are produced by the authoring scripts in ``docs/_demo_src`` and
regenerated with a single command:

.. code-block:: bash

   python docs/_demo_src/build_all.py

which runs each script in turn, one process at a time. It needs Graphviz's
``dot`` on the local ``PATH`` for the flowsheet diagrams, so it is a local
step only: Read the Docs never runs it, and the generated assets are
committed to the repository.

To add or change a tutorial figure or an example output, edit the
corresponding chapter script under ``docs/_demo_src/examples/``, run
``build_all.py``, and commit the regenerated files together with the page
that uses them. The :doc:`tutorial <../tutorial/index>` pages include code
from those scripts region by region instead of pasting it, so the code on the
page is code that was actually executed. Likewise, every number quoted in
prose or in a caption is copied from a capture file under
``docs/source/_generated`` -- never typed from memory or carried over from an
earlier run.

Reporting issues
----------------

Report bugs and feature requests as GitHub issues on the repository:
https://github.com/BioSTEAMDevelopmentGroup/hensmith

The most useful report contains a **minimal system that reproduces the
problem** -- the smallest set of streams and units that still shows it --
together with the full traceback, or the incorrect numbers alongside the
expected ones, and the versions of hensmith, biosteam and thermosteam in
use. A reproducer of that size can usually be turned into a test in
``tests/`` directly, which is the fastest route from a report to a fix.
