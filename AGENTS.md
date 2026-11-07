# AGENTS.md: working on the calphy code base

Guidance for coding agents (and humans) changing calphy itself. For *using* calphy to
compute free energies, see the user skill
[`skills/calphy/SKILL.md`](skills/calphy/SKILL.md) and the documentation at
https://calphy.org.

calphy computes absolute free energies by non-equilibrium thermodynamic integration,
driving LAMMPS. Users publish numbers from it. A change that alters a free energy by
0.1 meV/atom without anyone noticing is a serious bug, even if every test passes. Treat
physics-affecting changes accordingly.

## Licence (read before writing code)

calphy is under the **Academic Software Licence v1.0** (ASL). See `LICENSE` and
`CONTRIBUTING.md`.

- Contributions are inbound = outbound ASL.
- **Never copy code from GPL/LGPL/AGPL projects, including LAMMPS.** Permissive code
  (BSD/MIT/Apache-2.0) is fine, with its notice and source.
- New modules in `calphy/` carry the standard licence header. Copy it from an existing
  module.
- Do not add copyleft dependencies.

## Setup and tests

```bash
pip install -e ".[test]"                 # add [library] for the pylammpsmpi backend
pytest tests -m "not lammps" -q          # what CI runs (ubuntu + macos, py3.11/3.12); no LAMMPS needed
```

LAMMPS-backed tests are marked `lammps` (and the heavy ones `slow`). Run them before a
release, or after any change to emitted commands or physics:

```bash
conda env create -f environment.yml && conda activate calphy && pip install -e ".[test]"
pytest tests -m lammps -v
```

- A test that needs a real `lmp` binary must carry `@pytest.mark.lammps`, so CI stays
  binary-free.
- `.gitignore` ignores `*.dat`, `*.data`, `*log*`, `traj.*` and the run-folder patterns
  (`fe-*`, `ts-*`, …). New test fixtures with those names, or a module whose name
  contains "log", need `git add -f`. Check `git status` before committing.
- Never commit run output. Running calphy or `read_inputfile` writes `*.data` files and
  run folders into the CWD. Run experiments outside the repo, or clean up after them.

## Package map

| module | role |
|---|---|
| `input.py` | pydantic input models (`Calculation` + one model per block), all validation, `read_inputfile`, folder naming |
| `queuekernel.py` | `calphy_kernel`: runs ONE calculation in the foreground; `setup_calculation` / `run_calculation` |
| `kernel.py` | `calphy`: writes a script per calculation and submits it through `scheduler.py` (`local`, `slurm`, `sge`) |
| `routines.py` | mode drivers (`routine_fe`, `_ts`, `_tscale`, `_pscale`, `_alchemy`, `_composition_scaling`) and `MeltingTemp` |
| `phase.py` | `Phase` base class: equilibration, pressure convergence, reversible-scaling / temperature / pressure sweeps, reports |
| `solid.py`, `liquid.py`, `alchemy.py`, `dcci.py` | reference-specific stages (Einstein crystal, UFM + ideal gas, alchemical switching, coexistence lines) |
| `integrators.py` | analytic reference free energies and work integration (`find_w`, `integrate_rs`, `integrate_ps`, …) |
| `runner.py` | `BaseRunner`, `ExecutableRunner` (segmented `lmp` subprocess, `SessionState` replay), preflight style check |
| `library_runner.py` | `LibraryRunner` (pylammpsmpi live session; optional dependency, lazily imported) |
| `helpers.py` | `create_object` (runner factory), pair-style plumbing, solid-fraction analysis (pyscal3) |
| `postprocessing.py` | `read_report`, `gather_results`, composition-scaling corrections, coexistence-line readers |
| `structures.py` | built-in lattices via ASE |
| `phase_diagram.py`, `clitools.py` | phase-diagram construction and the `calphy_phase_diagram` CLI |
| `range_scan.py` | the optional pre-flight phase-transition scan |

`skills/calphy/` is the agent skill for *users* (`SKILL.md` + `references/`). It names
input keys, defaults and fixed-in-version markers, so update it in the same change as
any of those.

## Invariants: break these and results go wrong silently

**LAMMPS is only commands in, files out.** Drivers send command strings through the
runner and read results back from files (`avg.dat`, `msd.dat`, `forward_N.dat`,
`ts.forward_N.dat`, dumps, `conf.*.data`). Consequences:

- Before Python reads a file produced by the current session, call `lmp.sync()`. In
  executable mode, that is what flushes and runs the segment.
- A continuous path (a switching leg, a reversible-scaling / temperature / pressure sweep)
  must contain **no reads**, so it is never split across segments.
- Every command's first token must be in `runner.KNOWN_TOKENS`. Unknown commands make the
  runner raise by design. Add a new token deliberately, with its INIT / STICKY / ONE_SHOT
  / REMOVAL class, and a test.
- Dumps must not stay live across a segment boundary. Use the existing dump helpers.

**Every LAMMPS session is closed, on success and on error.** Stages open their session
with `self._open_lammps()`, and the routines run each stage as `with job: job.stage(...)`.
New stages and routines must follow the same pattern. In library mode a session left open
after an exception keeps executorlib's non-daemon threads alive: the process prints its
traceback and then never exits, holding its allocation until walltime. Close in the
error path, and never swallow the original exception.

**Barostat and thermostat targets in sweeps are physics.** In reversible scaling the
potential is λU, so the barostat target must be λp. It was once computed but used only
in a log message (fixed in 2.3.0). Any change to `_install_sweep_barostat`,
`_reversible_scaling_*`, `temperature_scaling` or `pressure_scaling` must keep
`tests/test_sweep_barostat_targets.py` green and extend it if the schedule changes.

**Analytic references and integrands.** Changes to `integrators.py` (Einstein crystal,
UFM, ideal gas, COM correction, mass correction, `integrate_rs`/`integrate_ps`) need a
test against an independently computed value, not a re-run of the same code.
`integrate_rs` defaults (`scale_energy=False`, `nsims=5`) differ from what `routine_ts`
passes. Callers must be explicit.

**Phase checks.** `tolerance.solid_fraction = 0` / `liquid_fraction = 1` mean "disabled"
because the measured fraction is in [0, 1]. Keep that meaning. Modes that rely on
detection (`melting_temperature`, `dcci`) enable it via
`routines.enable_phase_detection`, and log that they did. A disabled check should not
run the (expensive, occasionally crashing) structure analysis.

## Input changes

- All input lives in the models in `input.py`. Models are strict (`_StrictInput`), so an
  unknown key is an error with a hint. Keep it that way.
- To add a key, add it to the right block with type, default and `description`. Document
  it in `docs/source/inputfile.md` **in the same change**. Add validation tests to
  `tests/test_input_validation.py` or a sibling.
- To remove or rename a key, add it to `_REMOVED_KEYS` with a migration message. Never
  ignore it silently.
- Validate enumerations. A free-form string that is compared against literals downstream
  silently falls through to a default branch on a typo. For example,
  `equilibration_control` (only `nose-hoover` vs everything else) currently has this flaw.
- Mode-specific constraints belong in validation, not in a comment or a docstring.
  Examples: `pscale` needs a 2-component pressure, and `fe` with a 2-list temperature uses
  only the first value.
- Defaults decide users' results. Changing a default changes published numbers on rerun.
  Say so in the changelog / PR description, and log a warning where the old behaviour
  would have differed.

## Golden command streams and baselines

- `tests/test_golden_streams.py` records the exact LAMMPS command stream of each stage
  with a recording runner and compares it to `tests/golden/<name>.txt`.
  `assert_golden` **writes the golden when the file is missing** and passes. A deleted or
  misnamed golden therefore always passes once. Check that new goldens are committed and
  reviewed.
- A golden mismatch after your change means the emitted commands changed. Assume your
  code is wrong until you can explain every diff line. If the change is intended:
  1. regenerate the affected golden (delete, rerun),
  2. read the full diff,
  3. state in the commit message why each changed command is correct.
- `tests/baselines/` holds frozen real-LAMMPS results (see `BASELINE_INFO.md`). They are a
  contract. Never edit them to make code pass.

## Physics changes: what "done" means

1. A unit or golden test that pins the emitted LAMMPS commands or the formula.
2. A real-LAMMPS check on a small system (e.g. the Cu EAM baselines), comparing against
   an **independent** route. Sweep vs `fe` spot checks at points inside the sweep, `pscale`
   vs `fe` at the end pressure, and composition scaling vs two direct `fe` runs all work.
   Reporting dissipation alone is not validation: it is blind to errors common to the
   forward and backward paths.
3. Validate where the defect you fix is *small* as well as where it is large. A fix that
   overshoots, or gets the sign of λ wrong, passes the large-error case.
4. Record numbers, not adjectives, in the PR: before and after, with error bars.

## Outputs are an API

Users and downstream tools parse these by name and column:

- `report.yaml` keys (`results.free_energy`, `error`, `work`, `pv`, `dissipation`,
  `ts_dissipation`, `ts_dissipation_high`, `reference_system`, …)
- `temperature_sweep.dat`, `pressure_sweep.dat`
- the column order of `ts.*.dat` / `ps.*.dat` (`dU, press, vol, lambda`)
- `input_file.yaml` (the resolved input, including the backfilled seed)
- the folder-name scheme
- the `STATE:` lines in logs

Do not rename or reorder these without a deprecation path. Add new keys rather than
changing the meaning of old ones. Units: eV/atom, K, bar, Å³ (whole cell in sweep files).

## Versions and releases

- The version string is in `pyproject.toml`, `calphy/__init__.py` and `calphy/input.py`.
  Bump all three with `bumpversion` (`.bumpversion.cfg`).
- Releases publish to PyPI from tags (`.github/workflows/deploy.yml`).
- `pyscal3 >= 4` is required, and enforced at import in `helpers.py`.
- `pylammpsmpi > 0.4.1` is the `library` extra. Keep it optional: importing calphy must
  never require it.

## Working style

- Keep PRs focused, one concern each. Use `%`-formatting for LAMMPS command strings, as
  the existing code does, and the existing logger patterns. Write user-visible warnings
  into `calphy.log` with what to do about them.
- For multi-part features, the repository's practice is a plan document (see
  `EXECUTABLE_RUNNER_PLAN.md`, `DCCI_PLAN.md`) with ordered parts and acceptance criteria.
  Work part by part and keep each part green.
- If LAMMPS semantics are in doubt (restart contents, fix state across `run`s, what a
  barostat ramps), stop and check the LAMMPS manual, or ask. Do not guess.
- Keep `docs/source/*.md` and `examples/` in step with the code. Several docs and examples
  have drifted before.
