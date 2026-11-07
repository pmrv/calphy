---
name: calphy
description: >-
  Use calphy (github.com/ICAMS/calphy) to compute ABSOLUTE free energies of solids and
  liquids from LAMMPS molecular dynamics: Frenkel-Ladd / Einstein-crystal and
  Uhlenbeck-Ford liquid references, reversible-scaling F(T) and G(T) sweeps, pressure
  scaling G(p), melting temperatures, coexistence lines (dynamic Clausius-Clapeyron),
  and alchemical / composition-scaling free-energy differences. Read this BEFORE writing a
  calphy input YAML, launching calphy (CLI, Python or batch job), or interpreting
  report.yaml / temperature_sweep.dat. Several shipped defaults silently give wrong or
  wasted results (phase checks disabled, liquid melt temperature too high, unsorted
  anchor temperature). Triggers: "calphy", "free energy", "Gibbs free energy",
  "thermodynamic integration", "Frenkel-Ladd", "Einstein crystal", "ti/spring",
  "Uhlenbeck-Ford", "ufm", "reversible scaling", "F(T) curve", "melting temperature",
  "melting point from MD", "phase stability", "solid-liquid coexistence", "dcci",
  "Clausius-Clapeyron", "pscale", "tscale", "alchemy", "composition_scaling",
  "dissipation", "report.yaml", "temperature_sweep.dat", "calphy_kernel".
---

# calphy: absolute free energies from MD

calphy (Menon et al., *Phys. Rev. Materials* **5**, 103801, 2021) computes the absolute
free energy of one phase at one composition. It switches the real potential to an
analytic reference: an Einstein crystal (`fix ti/spring`) for solids, or the Uhlenbeck-Ford
model (`pair_style ufm`) plus ideal gas for liquids. It then carries that value along T,
p or composition by non-equilibrium thermodynamic integration. It drives LAMMPS, so it
works with any LAMMPS pair style.

What calphy does *not* do for you: decide which phase is stable. Comparing curves
(melting points, phase boundaries, phase diagrams) is a separate step. Use a
phase-diagram tool, or calphy's own `calphy.phase_diagram` / `calphy_phase_diagram`.

**Licence.** From 2.1.0, calphy is under the Academic Software Licence: academic,
non-commercial use only. Releases up to 2.0.1 were ASL/BSD-3. Check that your use fits.

## 0. Know which calphy you have, before trusting anything below

```bash
calphy -v                                   # or: python -c "import calphy; print(calphy.__version__)"
python -c "import calphy, os; print(os.path.dirname(calphy.__file__))"   # which install
```

Behaviour changed a lot between versions. The landmarks:

| version | what matters |
|---|---|
| 1.x | legacy input forms, `script_mode`, staged CLI; `cumtrapz` import breaks on scipy ≥ 1.14 |
| 2.0.x | `calculations:` list required, strict input validation, `md.seed`; default `execution_mode: executable`. **`ts`/`tscale` at p ≠ 0 are wrong in every release before 2.3.0**: the sweep barostat holds the full p instead of λ·p |
| 2.1.x | one scaled copy of the potential in sweeps (≈2× cheaper per sweep step); warm-started, chained `fe` iterations; barostat bug still present |
| 2.3.0 (2026-10) | λ·p barostat fix; `ts_dissipation` reports the *worst* iteration; `lambda_schedule` removed (hard error); built-in lattices from ASE; needs `pyscal3 >= 4` |
| `main` after 2.3.0 | LAMMPS session closed when a stage raises (no more processes that survive their own traceback in library mode); spring constants from the unrounded MSD |

If you are on anything before 2.3.0, do not run `ts` at non-zero pressure. Use `pscale` (§1, `references/pressure-and-tension.md`) or upgrade.
Anything in this skill marked *(main)* is not in a release yet.

## 1. Which mode answers which question

| `mode` | computes | `temperature` | `pressure` |
|---|---|---|---|
| `fe` | F (or G) at one state point | scalar | scalar, `[[p,p,p]]`, or `None` (fixed box) |
| `ts` | G(T) over a range: an `fe` anchor at `temperature[0]`, then reversible scaling | `[T_anchor, T_end]` | scalar (iso) or `[[p,p,p]]` |
| `tscale` | like `ts`, but scales the thermostat temperature directly instead of the potential | `[T0, T1]` | scalar |
| `pscale` | G(p) at fixed T: an `fe` anchor at `pressure[0]`, then ∫V dp | scalar | **must be** `[p0, p1]` |
| `melting_temperature` | Tm, by bracketing solid and liquid `ts` sweeps around a guess | scalar guess | scalar |
| `dcci` | a solid-liquid coexistence line from one known point | `T_coex` at `pressure[0]` | `[p0, p1]` |
| `alchemy` | ΔF between two potentials (`pair_style`/`pair_coeff` lists of 2) | scalar | scalar |
| `composition_scaling` | ΔF between compositions (types swapped along the path) | scalar | scalar |
| `fe-qtb` | `fe` for a solid with a quantum thermal bath and a quantum Einstein reference | scalar | scalar |

`ts` is the workhorse. A few `fe` runs at points inside the sweep are the independent check
that the sweep is right (§6). `melting_temperature` re-reads its input file. Drive it from
a YAML file, not from a constructed object.

## 2. A minimal input that avoids the silent traps

One YAML holds a list under `calculations:`. Every key is validated. An unknown key is a
hard error with a "did you mean" hint, so build inputs as dicts and let calphy validate
them. Full key table: `references/input-keys.md`.

```yaml
calculations:
- element: [Cu]
  mass: [63.546]
  lattice: fcc                 # built-in: fcc bcc hcp diamond simple_cubic a15; or a LAMMPS data file
  lattice_constant: 3.615      # near the T/p equilibrium value, or pressure convergence crawls
  repeat: [8, 8, 8]            # 2048 atoms
  reference_phase: solid       # solid -> Einstein crystal; liquid -> UFM + ideal gas
  mode: ts
  temperature: [300, 1200]     # [0] IS the anchor; calphy does not sort. Solid: cold end first
  pressure: 0.0                # bar
  pair_style: eam/alloy
  pair_coeff: "* * /abs/path/Cu.eam.alloy Cu"   # absolute path; trailing labels in `element` order
  n_equilibration_steps: 25000
  n_switching_steps: [50000, 200000]  # [reference switching, sweep]; a scalar applies to both
  n_iterations: 3              # >= 2, or the reported error is exactly 0
  md: {timestep: 0.001, seed: 12345}  # seed optional: the used seed is written to input_file.yaml anyway
  tolerance:
    solid_fraction: 0.7        # solid runs. The default 0 disables melt detection entirely
    liquid_fraction: 1.0       # 1 = solidification check off (correct for a solid)
  queue: {scheduler: local, cores: 8}   # cores = MPI ranks for LAMMPS
```

For a **liquid**, change these and keep the rest:

```yaml
  reference_phase: liquid
  temperature: [1600, 1100]    # anchor HOT, sweep down. A cold anchor in a sluggish liquid dissipates
  temperature_high: 2000       # the melt temperature. Default is 2x temperature[1]: can boil the liquid
  tolerance: {solid_fraction: 0.0, liquid_fraction: 0.05}
```

Other non-obvious input facts, all checked against source:

- `pressure: [0, 0, 0]` (flat, shape (3,)) is **rejected** ("Unknown format for
  pressure"), even though the docs list it. Anisotropic is `[[p, p, p]]`, and the three
  values must be equal. `pressure_coupling: aniso | tri` overrides the inferred coupling.
  Use `tri` for triclinic cells. A scalar pressure gives `iso`, which freezes c/a of a
  non-cubic crystal.
- `pressure: None` (or `null`) fixes the box (NVT). The run then measures the pressure and
  still adds `p·V`. `results.free_energy` is therefore F + p_measured·V. Subtract
  `results.pv` if you want Helmholtz F(V, T).
- `equilibration_control` accepts exactly `nose-hoover` or `berendsen`. Anything else is not
  validated and silently means Berendsen, including `nose_hoover` with an underscore,
  which appears in upstream examples. Omit it to get Nose-Hoover.
- For `fe`, a 2-list `temperature` runs only the first value. Write one calculation per
  state point.
- `lattice` as a data file must be `atom_style atomic`. Types are mapped to `element` in
  order. For multi-element `pair_coeff`, calphy checks that element order matches.
- Relative `pair_coeff` paths are resolved against the CWD at parse time, and only if
  the file exists. Pass absolute paths. `$VARS` and `~` are expanded.
- Folder name: `[<folder_prefix>-]<mode>-<lattice>-<phase>-<int(T)>-<int(p)>`, created in
  the CWD, or wherever an absolute `folder_prefix` points. If it already exists the run
  stops at once: `Simulation folder … exists. Please remove and run again!` Two
  calculations that differ only below 1 K or 1 bar collide.

## 3. Running it: CLI, Python, batch systems

| command | what it does |
|---|---|
| `calphy_kernel -i input.yaml -k N` | runs calculation N **in the foreground** and returns when it is done. Use this |
| `calphy -i input.yaml` | writes a `<id>.sub` script per calculation and *submits* it via `queue.scheduler` (`local`, `slurm`, `sge`). With `local` it starts every calculation in the background and returns at once |
| `calphy_phase_diagram` | the phase-diagram helper |

So inside any batch job, call `calphy_kernel` directly, and propagate its exit status:

```bash
cd "$RUNDIR" || exit 1          # calphy writes into the CWD; fail loudly if cd fails
calphy_kernel -i "input_${TAG}.yaml" -k 0
rc=$?; echo "calphy exit $rc"; exit $rc   # a trailing echo alone would report success
```

Python (equivalent to `calphy_kernel`, runs in-process, writes into the CWD):

```python
from calphy.input import read_inputfile
import calphy.queuekernel as cq
calc = read_inputfile("input.yaml")[0]      # validates; also writes a <id>.data structure file in CWD
job = cq.run_calculation(cq.setup_calculation(calc))
job.report["results"]["free_energy"]
```

Patterns that held up across many campaigns:

- **One input file per calculation** (`input_<tag>.yaml`, always `-k 0`). A shared file plus
  per-job indices breaks the moment anything rewrites the file.
- **Run from a dedicated run directory.** Even parsing writes `<id>.<k>.data` files into
  the CWD.
- **Requested MPI slots must equal `queue.cores`.** If the launcher cannot place that many
  ranks, it can print one error into stdout and then hang rather than exit.

**LAMMPS must have the reference styles.** Check before the first real run. Details and
the library-mode check are in `references/running-and-lammps.md`.

| needed for | style | LAMMPS package |
|---|---|---|
| solid `fe`/`ts`/`tscale` | `fix ti/spring` | EXTRA-FIX |
| liquid, `melting_temperature`, `dcci` | `pair_style ufm` | EXTRA-PAIR |
| `ts`, `tscale`, `dcci`, liquids, overlay potentials | `pair_style hybrid/scaled` | EXTRA-PAIR (core in recent versions) |
| `monte_carlo.n_swaps > 0` | `fix atom/swap` | MC |
| `fe-qtb` | `fix qtb` | QTB |
| your potential | e.g. `eam/alloy`, `snap`, `pace` | MANYBODY, ML-SNAP, ML-PACE, … |

```bash
lmp -h | grep -E -w 'ti/spring|ufm|hybrid/scaled|atom/swap'     # executable mode
```

`execution_mode: executable` (default) runs `lmp` as a subprocess in restart-continued
segments. It needs `lmp` (`lammps_executable` → `$CALPHY_LAMMPS_EXECUTABLE` → PATH) and,
for `cores > 1`, an `mpirun -np`-style launcher. It runs a preflight style check (skip with
`CALPHY_SKIP_PREFLIGHT=1`). `execution_mode: library` keeps one live session through
pylammpsmpi (`pip install calphy[library]`, needs the `lammps` Python module). Library mode
has no per-segment restarts, so it suits potentials with an expensive start-up, but it gets
**no preflight**. `md.cmdargs` reaches LAMMPS in both modes, for example
`-k on g 1 -sf kk` for KOKKOS.

## 4. Reading results

Per calculation folder:

| file | content |
|---|---|
| `report.yaml` | `results.free_energy` (eV/atom), `error`, `reference_system`, `einstein_crystal`, `com_correction`, `work`, `dissipation`, `pv`; for `ts`/`tscale` also `ts_dissipation`, `ts_dissipation_high`; `average.vol_atom`, `spring_constant` or `density`; `input.pressure` |
| `temperature_sweep.dat` | `ts`/`tscale`: T [K], G [eV/atom], error. Written only after **all** iterations finish |
| `pressure_sweep.dat` | `pscale`: p [bar], G [eV/atom], error |
| `forward_N.dat`, `backward_N.dat` | reference switching, iteration N |
| `ts.forward_N.dat`, `ts.backward_N.dat` | sweep: `dU, press, vol, lambda`, one row per step |
| `input_file.yaml` | the fully resolved input, including the seed actually used |
| `calphy.log`, `*.log.lammps` | the log, and LAMMPS logs per stage |

Meaning of the numbers:

- `free_energy` is **per atom and includes p·V**: it is G at p. Solid:
  `reference_system + work + pv`. Liquid: `f_ideal + reference_system − work + pv`.
  `f_ideal` is not in the report.
- It is **classical**, and includes the momentum (mass) term. Compare with other classical
  results for the same masses; use `fe-qtb` or a harmonic correction for quantum effects
  at low T.
- A liquid's F includes ideal mixing entropy, through the ideal-gas reference. A solid
  sampled on one fixed decoration does not include configurational entropy.
- `alchemy` and `composition_scaling` report a **ΔF** (the work), not an absolute F. Add
  the absolute F of the starting state. For `composition_scaling`, mixing entropy is not
  included; the report gives `entropy_contribution` and `mass_correction` separately
  (`references/composition-and-alchemy.md`).
- Progress in `calphy.log`: `Integration cycle k finished` belongs to the reference stage,
  `TS integration cycle k finished` to the sweep. A `ts` run is **2 × n_iterations**
  switching cycles in total, not n_iterations.
- Aggregate many runs with `calphy.postprocessing.gather_results(folder)` (pandas).
  `read_report(folder)` lives in `calphy.postprocessing`, not `calphy.input`.

## 5. Pitfalls (from real campaigns; mechanisms in `references/pitfalls.md`)

1. **Phase checks are off by default.** `solid_fraction: 0` / `liquid_fraction: 1`
   can never fire, so calphy integrates through melting or freezing and reports F for a
   phase that no longer exists. Only `melting_temperature` and `dcci` switch them on.
   Set them yourself.
2. **The detector only knows bcc/fcc/hcp/sc/diamond**, via Steinhardt parameters in
   pyscal3. Compounds and complex cells can read as "melted" while intact, and strained
   cells as partly liquid. For those, disable the check (0.0) and watch dissipation and
   volume instead.
3. **`temperature[0]` is the anchor.** Anchor solids cold, liquids hot. A
   deeply supercooled liquid anchor gave 9 meV/atom dissipation, and more switching steps
   do not fix a misplaced anchor.
4. **Liquid `temperature_high` defaults to 2 × T_end.** That can exceed the boiling
   point: the box expands without bound until OOM or timeout. If melting fails, the melt
   cycle retries at up to 1.9 × `temperature_high`. Set it just above Tm and check
   `calphy.log` for repeated `Starting melting cycle`.
5. **Do not sweep a solid far above Tm.** A superheated solid survives for a while, then
   degrades quietly: high dissipation, or a biased curve with *low* dissipation. Keep
   the solid's top temperature ≲ 1.05–1.1 Tm(p), and note that Tm moves with p.
6. **Budget sweeps per Δλ, not per kelvin.** Reversible scaling integrates in λ = T0/T.
   100→800 K spans Δλ = 0.875; 1900→2050 K spans 0.073. Split wide ranges into
   overlapping chunks, each with its own anchor. Keep T_end/T_anchor ≲ 2 at p ≠ 0.
7. **`ts` at p ≠ 0 before calphy 2.3.0 is wrong** (barostat not λ-scaled). Errors are
   ~10–30 meV/atom at 5 GPa, and grow as λ falls. Dissipation does not show it. Use ≥ 2.3.0, or
   `pscale` from a validated p = 0 curve.
8. **`pscale` with a scalar `pressure` is a silent no-op**: ∫V dp over zero width. It also
   reports no dissipation, and column 4 of `ps.*.dat` is not pressure. Use column 2.
9. **Low-T spring constants** (≤ 2.3.0): the MSD is rounded to 0.01 Å² before
   k = 3k_BT/MSD, giving tens of % error below ~300 K. An MSD that rounds to 0 becomes
   1 Å², about 100× too soft. The result stays formally exact but switching gets much
   harder. Set `spring_constants` explicitly for cold anchors. Fixed in *(main)*.
10. **Small cells and pressure convergence.** The default `tolerance.pressure: 10` bar
    is below the pressure noise of a few-hundred-atom cell. Expect up to `md.n_cycles`
    × `n_small_steps` steps, then `Pressure did not converge`.
11. **A dead LAMMPS worker can leave the job alive** (library mode, ≤ 2.3.0): the
    traceback goes to stderr, `calphy.log` just stops, the scheduler says RUNNING. Watch
    file-write staleness in the run folder, not job state.
12. **A walltime kill is not a total loss.** Finished sweep iterations are on disk, and
    `calphy.integrators.integrate_rs` can rebuild the curve from them. Never resubmit over
    a folder with recoverable iterations (`references/verification-and-recovery.md`).
13. **Executable mode memory:** one `mpirun` + `lmp` per segment. One campaign ran out of
    memory with 16 ranks; library mode avoided it.
14. **Hysteresis cancels systematic errors.** A sweep with `ts_dissipation_high: false`
    was still several meV/atom off in absolute terms. Dissipation is necessary, not
    sufficient.

## 6. Verify before you report a number

1. `report.yaml` exists and `calphy.log` has no `WARNING` about disabled detection, a high
   dissipation, or unconverged spring constants.
2. Dissipation: ~1e-5 to 1e-4 eV/atom is clean. ~1e-3 is the default tolerance: look at it.
   ≥ 1e-2 means the structure changed partway through the path.
3. `n_iterations ≥ 2`, so `error` is a real standard error.
4. **Spot checks:** 2–3 independent `fe` runs inside each `ts` chunk must agree with the
   sweep within its error. This is the only absolute check.
5. **Chunk joins:** overlapping chunks must agree in G (offset) *and* in slope
   (entropy). Equal slopes with an offset point to an anchor problem; different slopes
   point to a sweep problem. Never shift chunks onto each other to hide a join.
6. **Volume sanity:** `ts.*.dat` volume vs T must expand on heating at constant p. A
   contracting solid means the sweep did not hold its isobar.
7. Physical sanity: compare the 0 K limit with a relaxed static energy, using
   `G + 3 k_B T ln T`, which is ~linear for a classical harmonic solid. Take C_p from a
   smooth fit, never from finite differences of the samples.

Detailed checks and recovery code are in `references/verification-and-recovery.md`.
Pressure, tension and pscale are in `references/pressure-and-tension.md`.
