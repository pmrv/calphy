# Running calphy: LAMMPS builds, execution modes, MPI, batch systems, monitoring

## Install

- `pip install calphy` installs the Python package only. Add `calphy[library]` for
  pylammpsmpi.
- `conda install -c conda-forge calphy` pulls in a LAMMPS too.
- Check the version you actually got (`calphy -v`). Older conda-forge builds have broken:
  1.2.x imports `scipy.integrate.cumtrapz`, removed in scipy 1.14, so it installs and then
  fails on import.
- calphy ≥ 2.3.0 needs `pyscal3 >= 4`; earlier 2.x releases need `pyscal3 < 4`. The wrong pair fails at
  import.
- Pin a version or commit for a campaign. Never run a long campaign against a moving
  branch or path checkout. Inside the job, assert you import the calphy you think you do:
  `python -c "import calphy; print(calphy.__file__, calphy.__version__)"`.

## LAMMPS requirements, and how to check them

calphy needs these styles in addition to your potential's own package:

| style | package | needed for |
|---|---|---|
| `fix ti/spring` | EXTRA-FIX | solid `fe`, `ts`, `tscale` (Einstein reference) |
| `pair_style ufm` | EXTRA-PAIR | liquids, `melting_temperature`, `dcci` |
| `pair_style hybrid/scaled` | EXTRA-PAIR in older LAMMPS, core in newer | sweeps, liquids, overlay potentials |
| `fix atom/swap` | MC | `monte_carlo.n_swaps > 0`, `composition_scaling` with swaps |
| `fix qtb` | QTB | `fe-qtb` |
| `compute msd`, `compute temp/com`, `fix npt/nvt/langevin/ave/time/print/momentum` | core | everything |

Executable mode:

```bash
lmp -h | grep -E -w 'ti/spring|ufm|hybrid/scaled|atom/swap|qtb'
```

calphy also runs this check itself (the preflight) and names the missing package.
`CALPHY_SKIP_PREFLIGHT=1` disables it, for builds that crash on `-h`.

Library mode has **no preflight**. Check the module you will import, in the environment
and on the node type that will run the job:

```python
from lammps import lammps
l = lammps(cmdargs=["-log", "none", "-screen", "none"])
print(l.installed_packages)
for cat, name in [("fix", "ti/spring"), ("pair", "ufm"), ("pair", "hybrid/scaled")]:
    print(cat, name, l.has_style(cat, name))
l.close()
```

A missing style fails only when that stage starts, for example 10+ minutes into a liquid
run at `pair_style ufm`. If the process then hangs instead of exiting (below), the job
looks alive for hours. Do not probe a shared library with `strings`: its default minimum
length of 4 hides 3-letter style names such as `ufm`.

## The two execution modes

| | `executable` (default) | `library` |
|---|---|---|
| transport | `[mpirun -np N] lmp -in calphy.segK.lmp …`, one subprocess per segment; state carried by restart files plus replayed commands | one live session via pylammpsmpi/executorlib |
| needs | an `lmp` binary; for `cores > 1` an `mpirun`/`mpiexec`-style launcher that accepts `-np` | the `lammps` Python module (built with shared libs) and `pylammpsmpi > 0.4.1`; for `cores > 1` a working MPI launcher for mpi4py/executorlib |
| preflight | yes | no |
| costs | the potential is re-read in every segment; one MPI start-up per segment (observed: ~1 GB per rank, OOM at 16 ranks) | everything stays in one process tree |
| failure mode | `LammpsExecutionError` naming the segment log | ≤ 2.3.0: an exception in a stage can leave the process alive forever (see below) |

In executable mode, every command calphy sends must start with a token the runner knows.
Custom `md.init_commands` with unusual first tokens raise `RunnerStateError`.

`md.cmdargs` is passed through in both modes. Use it for accelerator packages:
`-k on g 1 -sf kk -pk kokkos newton on neigh half` for KOKKOS on one GPU, with
`queue.cores: 1`, i.e. one rank per GPU. Under KOKKOS, LAMMPS warns that
`fix ti/spring` is not compatible with device sorting and falls back to host sorting.
That is expected. GPU and CPU free energies of the same input agreed within 0.1 meV/atom
in our tests.

## MPI and batch systems (scheduler-agnostic)

- `queue.cores` is the number of LAMMPS MPI ranks calphy launches itself. In a batch job,
  request that many **tasks/slots**, not one task with many CPUs. Otherwise the launcher
  refuses ("not enough slots"), prints that once into stdout, and the run can hang
  instead of exiting.
- The launcher must belong to the MPI that LAMMPS (and mpi4py) were built against. A
  missing or mismatched launcher can hang silently rather than error, so test `cores: 2`
  interactively once per environment.
- Strong scaling saturates. Measure (e.g. 8/16/32 ranks on your cell size) before
  filling a node; the full node was slower than 32 of 36 cores in one test.
- Do not let calphy submit from inside a job. Write your own job script that runs
  `calphy_kernel -i input_<tag>.yaml -k 0` in the run directory, and exit with its status.
- Do not rebuild or update the software environment while jobs that use it are queued or
  running.

## Cost: how many steps a run takes

A `ts` run is `routine_fe` (averaging, then `n_iterations` forward+backward reference
switches), then `n_iterations` forward+backward sweeps. Rough MD step count:

```
fe     ≈ averaging + 2·n_iter·(n_eq + sw0)
ts     ≈ fe + 2·n_iter·(n_eq + sw1)        # ≥ 2.1.1: one copy of the potential per sweep step
pscale ≈ fe + 2·n_iter·(n_eq + sw1)
```

Averaging is the pressure and spring-constant convergence, each up to `md.n_cycles` ×
`md.n_small_steps`. With a good `lattice_constant` it is a few cycles. Before 2.1.1 a
sweep step evaluated the potential **twice** (`hybrid/scaled` with two copies). Reference
switching steps cost more than plain MD (the extra `fix ti/spring` or UFM pair). Run one
short scout per new potential and cell size, read `Integration cycle 1 finished in X s`
from `calphy.log`, and scale from it. Ask for about 2× the prediction. A single sweep
iteration must fit comfortably, because the iteration in flight at a kill is lost.

## Monitoring: job state is not health

- `report.yaml` is written after the reference stage, **before** the sweep. Its existence
  does not mean a `ts` run finished. The completion markers are `temperature_sweep.dat`
  (`ts`/`tscale`) or `pressure_sweep.dat` (`pscale`), and the process exit status.
- Progress lines in `calphy.log`: `Integration cycle k finished` (reference stage),
  `TS integration cycle k finished` (sweep), `Pressure scaling cycle k finished`.
- A progress monitor that counts only `ts.forward_*.dat` reads 0 during the whole
  reference stage, so a live run looks dead.
- **Dead but RUNNING**: with library mode on ≤ 2.3.0, a LAMMPS error inside a stage, e.g.
  `Domain too large for neighbor bins` from a box that blew up, raised a traceback to
  *stderr*. The process then never exited, because executorlib's worker threads were
  still waiting on a session nobody closed. `calphy.log` simply stops. Detect it by
  file-write staleness of the run folder, e.g. nothing modified for > 1 h while switching.
  Read the job's stderr after cancelling: the real error is there. Fixed on `main` after
  2.3.0 (stages closed in a context manager).
- A job that dies in seconds is almost always bookkeeping, not the machine: the folder
  already exists, the `-k` index is out of range, or a relative path was resolved
  against the wrong CWD (look for a doubled path component).
