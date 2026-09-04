# `docs/_demo_src` — the source of truth for every docs asset

Every image, GIF, animation still and captured text block under
`docs/source/_static/` and `docs/source/_generated/` is *generated* by a script
in this directory. Nothing here is part of the `hensmith` library — these
modules are never imported by `hensmith/**`, never imported by Sphinx's
`conf.py`, and **never run on Read the Docs**. RTD only ever sees the committed
outputs, so the assets must be regenerated locally and committed whenever the
library's behavior, numbers or figures change.

The scripts are the reason the tutorial cannot quietly drift: the pages
`literalinclude` *regions of these scripts* and `literalinclude` the *captured
output of running them*, so the code a reader copies and the numbers printed
next to it come from the same executed program.

## What each script produces

Output paths are relative to `docs/source/`; the scripts create any missing
directories themselves.

| Script | Outputs |
| --- | --- |
| `make_logo.py` | `_static/images/logo/logo_hensmith_light.png`, `_static/images/logo/logo_hensmith_dark.png` (2000 px wide), `_static/images/logo/mark_hensmith_light.png`, `_static/images/logo/mark_hensmith_dark.png` (600 × 600) |
| `make_icons.py` | `_static/images/icons/{getting-started,concepts,api,contributing}_{light,dark}.png` — eight 512 × 512 transparent card icons |
| `examples/ch01_quickstart.py` | `_static/images/examples/tutorial_01_quickstart_flowsheet_light.png`, `…_flowsheet_dark.png`, `_static/images/examples/tutorial_01_quickstart_pinch_diagram.png`; `_generated/ch01_results.txt`, `ch01_loads.txt`, `ch01_life_cycles.txt`, `ch01_summary.txt` |
| `examples/ch02_pinch_analysis.py` | `_static/images/examples/tutorial_02_composite_curves.png`, `tutorial_02_grand_composite.png`; `_generated/ch02_threshold.txt`, `ch02_table.txt`, `ch02_compare.txt` |
| `examples/ch03_network_anatomy.py` | `_static/images/examples/tutorial_03_hxn_flowsheet_light.png`, `…_hxn_flowsheet_dark.png`, `tutorial_03_pinch_diagram_minimal.png`; `_generated/ch03_flowsheet.txt`, `ch03_life_cycles.txt`, `ch03_stage.txt`, `ch03_pinch_Ts.txt`, `ch03_accounting.txt` |
| `examples/ch04_configuring.py` | `_static/images/examples/tutorial_04_T_min_app_sweep.png`, `tutorial_04_ten_streams_pinch_diagram.png`; `_generated/ch04_sweep.txt`, `ch04_ignored.txt`, `ch04_ten_streams.txt` |
| `make_hero_gif.py` | `_static/images/demo/hero_light.gif`, `hero_dark.gif` (8 s loop, 20 fps, 2000 × 720), `hero_light_still.png`, `hero_dark_still.png` (the frame-0 stills served under `prefers-reduced-motion`) |
| `build_demo.py` | `_static/quickstart_demo.html` — the interactive quickstart demo, filled in from `quickstart_demo_template.html` |
| `make_poster.py` | `_static/images/examples/quickstart_demo_poster.png` — the README poster that links to the demo (2400 × 1260) |

Two files here are not scripts: `_common.py` (shared helpers: paths, themes,
the capture context manager, figure saving, the quickstart system builder) and
`quickstart_demo_template.html` (the demo's hand-written template, with
`__TOKEN__` placeholders that `build_demo.py` fills). The only committed asset
under `docs/source/_static/` that is *not* generated here is
`_static/css/custom.css`, which is hand-written.

Order matters: `make_hero_gif.py` imports `composite_curves` from
`examples/ch02_pinch_analysis.py`; `build_demo.py` consumes the chapter-01
captures and figures; `make_poster.py` consumes the chapter-01 pinch diagram
and the dark logo. `build_all.py` lists the scripts in a valid order.

## Regenerate

From the **repo root**, with the `IBO_2026` interpreter:

```powershell
python docs\_demo_src\build_all.py
```

`build_all.py` runs each script as a **sequential subprocess** and stops at the
first failure. That sequencing is not incidental: importing biosteam writes
numba's shared on-disk cache, and two Python processes writing it at once
corrupt it — so never run two of these scripts (or the test suite alongside
one) concurrently by hand either. A full run takes a minute or two; the hero
GIFs dominate it.

Graphviz's `dot` must be on `PATH` — the flowsheet figures in chapters 01 and
03 are rendered by biosteam's `system.diagram()`.

Every script is also runnable on its own, from the repo root, when you only
need to refresh one asset:

```powershell
python docs\_demo_src\examples\ch02_pinch_analysis.py
```

After regenerating, `git status` should be clean apart from assets you *meant*
to change. Check content drift with `git diff --stat docs/source/_generated`,
not with `git status`: the scripts write `\n` line endings while the checkout
is CRLF, so a capture can be listed as modified by `git status` and still be
byte-identical in git's normalized view.

## How the tutorial stays truthful

Three mechanisms, all enforced at regeneration time:

- **Regions.** Each chapter script marks the code the page shows with
  `# [start:<region>]` / `# [end:<region>]` comments, and the `.rst` page pulls
  it in with `literalinclude` + `:start-after:` / `:end-before:`. The reader
  therefore sees code that *was executed*, not a retyped copy. Chapters 02–04
  each build the quickstart system in a `system` region whose body is exactly
  chapter 01's `build` + `network` + `simulate` bodies, so all four chapters
  provably describe one system.
- **Captures.** `_common.capturing(name)` redirects everything printed inside a
  block into `_generated/<name>.txt` (and echoes it), and
  `_common.write_summary` writes `key = value` lines. The pages
  `literalinclude` those files, so every number in the prose comes from a real
  run rather than from memory.
- **`build_demo.py` assertions.** The interactive demo is the easiest thing to
  let drift, so its build asserts, and fails loudly, that: every image
  placeholder resolves to a file that exists; every callout number appears
  somewhere in the chapter-01 captures; every code line shown in the demo's
  scene 01 is (after collapsing whitespace) a substring of `ch01_quickstart.py`
  itself, the one allowed difference being the omitted `, cache=True`; and no
  non-ASCII character or unfilled `__TOKEN__` survives into the output HTML.

## Conventions

- `_common.py` sets `NUMBA_DISABLE_JIT=1` and `DISABLE_PREFERENCES=1` (via
  `os.environ.setdefault`, so an outer setting wins) **before** biosteam is
  imported: long Windows paths break numba's cache, and the docs must not
  depend on the user's saved thermosteam preferences. `build_all.py` sets both
  in the subprocess environment too. Import `_common` before biosteam in any
  new script.
- `_common.py` calls `matplotlib.use('Agg')` before `pyplot` is imported;
  everything renders headless, and `savefig` never blocks.
- Captures are written with `newline='\n'` and a single trailing newline, so
  the committed text is stable across platforms.
- Chapter figures go through `_common.save` (dpi 200, `bbox_inches='tight'`) and
  flowsheets through `_common.save_diagram`, which pipes graphviz's PNG bytes
  itself because biosteam's `save_digraph` rejects file paths containing a `.`
  when a format is given.
- Outputs **are committed**: `.gitignore` does not ignore images, GIFs or the
  generated text. Regenerate, inspect the diff, and commit the assets with the
  change that caused them to move.
- Never create a directory named `build` here or anywhere else under `docs/`
  except the gitignored `docs/build/` Sphinx output — `.gitignore` has a bare
  `build/` rule that would silently swallow it.
- New Python modules in this directory start with the hensmith copyright
  header, and stay within Python 3.12 syntax (CI runs 3.12/3.13).
