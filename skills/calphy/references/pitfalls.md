# calphy pitfalls: mechanisms and evidence

Each item is something that went wrong in real campaigns: Fe SNAP solids and liquids
under ±5 GPa, Au-Pt ACE alchemical diagrams, LJ reference curves, GRACE/ACE multi-element
phase diagrams. All are rephrased generically, and the source mechanism was checked in
calphy's code. Version notes say where upstream changed behaviour.

## Phase identity

**1. The structural checks are off by default.** `tolerance.solid_fraction = 0` and
`liquid_fraction = 1`. The measured solid fraction is in [0, 1], so neither comparison can
trigger: "nothing melted" and "detection disabled" give identical output. calphy logs a
warning (2.x) but continues. `melting_temperature` and `dcci` enable 0.7 / 0.05 for their
sub-runs and say so in the log. Every other mode needs them set explicitly. An entire
liquid branch was once corrupted this way: it froze and integrated on.

**2. The detector is a Steinhardt classifier for simple lattices.** pyscal3 recognises
bcc/fcc/hcp/sc/diamond. Consequences:
- Intermetallics and other complex cells can read 0.0 solid fraction while intact. Runs
  were killed as "melted" in minutes.
- Strained or premelting cells read partly liquid: a held crystal read 0.63. A cell
  caught mid-melt read 0.88. Treat 0.7 as a last-resort catch, not a margin.
- A cavitated liquid is still "liquid", so the freeze check cannot see voids.
- Every release so far (up to 2.3.0) runs the structure analysis even when the check is
  disabled. With pyscal3 3.x (calphy before 2.3.0) it segfaulted (exit 139) on some
  triclinic dumps at the start of a backward sweep. pyscal3 ≥ 4 fixes the triclinic
  neighbour bug, and a calphy PR to skip disabled checks was open at the time of writing.

For compounds, set `solid_fraction: 0.0`. Rely on dissipation, the volume trace and an
independent displacement check (with minimum-image convention).

## Anchors, ranges, references

**3. `temperature[0]` is the anchor and is not sorted.** For a liquid, anchor at the hot
end and sweep down. A UFM anchor in a supercooled, sluggish liquid produced 9.2 meV/atom
of dissipation, concentrated where the liquid vitrified. A liquid anchored too cold was
also the suspected cause of a melting point ~550 K too low in another campaign. For a
solid, anchor cold enough to be safely crystalline but not so cold that the spring
constant breaks (item 6).

**4. Do not sweep a solid far into superheating.** Measured on bcc Fe: a hold test
survived 1.135 Tm and melted at 1.16 Tm. Sweeps whose top was at 1.135 Tm "completed"
but were biased (−11.9 meV/atom), with dissipation anomalously high for their tiny Δλ.
Sweeps above it raised `MeltedError` or crashed. Keep the top ≲ 1.05–1.08 Tm(p). In
another case a sweep 230 K above Tm biased only the backward branch and moved the solid
crossing by +100 K. Splitting it into two chunks fixed it. "It completed" is not evidence
the window was safe.

**5. Budget switching per Δλ.** Reversible scaling integrates in λ = T0/T. Steps
allocated ∝ ΔT starve the cold chunks: dissipation correlated +0.77 with steps⁻¹ per Δλ.
≥ 250k steps per unit Δλ held dissipation below 0.1 meV/atom on a 5.5k-atom SNAP cell.
Split wide ranges into overlapping chunks (~50 K overlap), each with its own anchor, so
errors do not compound and the overlaps become a free check.

**6. Spring constants at low T (≤ 2.3.0).** k = 3 k_B T / ⟨MSD⟩, with the MSD rounded to
0.01 Å². Below ~300 K this is tens of percent off (−34 % at 100 K for bcc Fe). Runs at the
same anchor T then report bit-identical k across pressures; that is the tell. If the MSD
rounds to 0.00, calphy used 1.00 Å² (a spring ~100× too soft, logged as a warning). The
free energy stays formally exact, because the reference uses the same k. But a mismatched
reference costs switching work and dissipation, and confounds Δλ analyses. Fix: give
`spring_constants` (from a warmer run, or k = 3k_BT/MSD from your own NVT). `main` after
2.3.0 uses the unrounded MSD and converges k on its relative change.

**7. The liquid melt temperature.** `temperature_high` defaults to 2 × `temperature[1]`.
For one potential that was 4400 K, above boiling. The NPT melt then expanded the box
without bound: volume 4e10 Å³, `PotEng ≈ 0`, TIMEOUT and OOM. Setting it just above Tm
is not the whole fix. If the melt check fails (with `liquid_fraction` < 1), calphy
retries at `temperature_high × 1.0, 1.1, …, 1.9`, under NPT at the full target pressure.
Grep `calphy.log` for `Starting melting cycle`: more than one line means the escalation
rescued you. For low-boiling elements no single ceiling is safe. Hand calphy a pre-melted
liquid configuration as `lattice` instead.

**8. `melting_cycle: false` may not melt anything.** It rattles the lattice and cools it
under NVT, with no barostat. That cannot cavitate, which is useful under tension, but for
some potentials the result recrystallises: `SolidifiedError: System solidified, increase
temperature`. The reliable recipe is a configuration you melted and checked yourself.

## Pressure

**9. `ts`/`tscale` at p ≠ 0 before 2.3.0.** The sweep barostat was given `iso p p` while
the potential was scaled to λU. The correct target `λ·p` was computed, but only printed in
the log message. At +5 GPa this produced −9.8 meV/atom in G and −6 % in volume, and a
100→800 K solid that *contracted* on heating. Chunk joins were off by up to 34 meV/atom.
Under tension, cold chunks crashed (`Domain too large`). Dissipation stayed small
throughout. Fixed in 2.3.0 (`iso p λp` forward, `λp λp` at T_end, `λp p` backward). On
an older build, compare `calphy.log` (`P a → b bar`) with the `fix npt` line in the stage
LAMMPS log. If both ends are the same number at p ≠ 0, the build has the bug. Any
reversible-scaling results at p ≠ 0 from releases before 2.3.0 need re-checking against `fe`
spot checks.

**10. Pressure input shape.**
- `[p, p, p]` is rejected; use `[[p, p, p]]`.
- A scalar gives an isotropic barostat. It froze c/a for a tetragonal crystal (shape
  error 0.7–3.3 meV/atom), and one early "this phase does not melt" result came from it.
- Use `[[p,p,p]]` or `pressure_coupling: aniso` for non-cubic cells, and `tri` for
  triclinic data files.
- With `aniso`, give the averaging enough steps (`md.n_small_steps`) to relax the shape.
- Values must be plain floats. `yaml.safe_dump` refuses numpy scalars.

**11. `pressure: None` is a fixed box, but the reported number is still G-like.** calphy
measures the mean pressure of the fixed box and adds `p_measured · V`. For Helmholtz F at
that volume, subtract `results.pv`.

**12. Pressure convergence in small cells.** The convergence test is
|⟨p⟩ − p_target| < `tolerance.pressure`, which defaults to 10 bar. For ~250 atoms the
statistical noise is far larger, so it loops to `md.n_cycles` and raises `Pressure did not
converge`. Loosen it (hundreds of bar), use a bigger cell, or start from a good
`lattice_constant`.

## Running and bookkeeping

**13. `calphy -i` is a submitter, not a runner.** It writes a script per calculation and
starts or submits it. With `scheduler: local` it returns immediately while the
calculations run in the background. Inside a batch job the job then ends and kills them.
Use `calphy_kernel -i file -k N`.

**14. Existing folder = immediate failure.** The folder name is a function of
(prefix, mode, lattice basename, phase, int T, int p). Re-running, or two calculations that
collide, fail in seconds with `Simulation folder … exists`. To redo a run, move the old
folder into a subdirectory. A sibling rename like `tag-….old` still matches a `tag-*` glob.

**15. Shared input file + per-job index.** Writing several calculations into one
`input.yaml` and launching `-k 0`, `-k 1`, … breaks as soon as a later submission
rewrites the file: IndexError, or a different calculation than intended. It also destroys
provenance. Use one file per calculation, always `-k 0`.

**16. Relative paths.** `pair_coeff` file paths are resolved against the CWD at parse
time, and only if the file exists there. A job script that `cd`s elsewhere first gets
the literal relative path, and paths double up (`runs/x/runs/x/…`). Resolve everything
to absolute in the code that writes the inputs. In shell scripts, write
`cd "$DIR" || exit 1`.

**17. Exit status.** A job script ending in `echo done` reports success for a run that
died on its first line. Propagate calphy's exit code.

**18. Output volume.** `ts.*.dat` has one row per MD step, so sweeps of 10⁵–10⁶ steps
give large files. `n_print_steps > 0` dumps whole configurations and once filled a disk
quota, after which runs hung in I/O. Leave it at 0 unless debugging.

**19. Wrong calphy imported.** A `PYTHONPATH` entry or an editable checkout can shadow
the installed package, depending on the CWD. Assert `calphy.__file__` / `__version__` (or a
feature probe) inside the job.

## Interpretation

**20. Dissipation is necessary, not sufficient.** It measures forward/backward hysteresis.
An error common to both directions cancels: the pre-2.3.0 pressure bug, a superheated
chunk, a biased anchor. A sweep reporting 0.06 meV/atom and `ts_dissipation_high: false`
was 3.4 meV/atom off against an `fe` spot check. Use spot checks and joins
(verification-and-recovery.md).

**21. `ts_dissipation` before 2.3.0 reported the best iteration**, which could hide one
sweep that crossed a transition. From 2.3.0 it reports the worst.

**22. `error` is a standard error across iterations.** With `n_iterations: 1` it is
exactly 0. It also says nothing about systematic error.

**23. Tiny smoke tests are not regression tests.** 54 atoms × 500 steps scatter by
hundreds of meV/atom between identical runs. An old claim that "unseeded calphy is
irreproducible" came from such a test with a wrong lattice constant. Since 2.0 the seed is
recorded in `input_file.yaml`, so any run can be reproduced by rerunning that file.

**24. `melting_temperature` mode needs good inputs to converge.** Each attempt runs a full
solid and liquid `ts` over [T−step, T+step]. A poor guess, or noisy sweeps, made the
bracket oscillate, at hours per attempt. Running your own solid and liquid `ts` chunks and
locating the crossing yourself was cheaper and more controllable. If you use the mode,
read `STATE:` lines in its log, including `STATE: Tm unreliable, sweep dissipation too
high`.
