# fcpot — force-constant potential plugin for LAMMPS

`fix fcpot` evaluates the harmonic force-constant Hamiltonian

    E = 1/2 * sum_ij u_i^T Phi_ij u_j,   u_i = x_i - X_i  (minimum image)

from a file of reference sites and sparse 3x3 force-constant blocks, for
use as the exactly-solvable reference of calphy's harmonic (phonon)
Frenkel-Ladd path (`harmonic_reference.implementation: fcpot`). The fix
applies forces scaled by an equal-style variable — `(1-lambda)` during
the switching. Its outputs follow the modern Fix energy/virial API:

| access    | value |
|-----------|-------|
| `f_ID`    | scaled energy `scale * E` — the actual Hamiltonian contribution, so `fix_modify ID energy yes` yields a conserved total energy |
| `f_ID[1]` | **unscaled** reference energy `E` — exactly the `dU_ref` column of the switching integrand |
| `f_ID[2]` | current value of `scale` |

With `fix_modify ID energy yes` / `virial yes` the fix additionally
tallies per-atom energies and per-atom + global virials via
`Fix::ev_tally`: each force-constant block `(i, j)` deposits its scaled
energy `1/2 scale u_i^T Phi_ij u_j` and virial `(X_i + u_i) ⊗ F_block`
on the owner of atom `i`, so `compute pe/atom fix` / `compute
stress/atom fix` and pressure see the reference potential. The per-atom
virial uses the reference-consistent coordinate `X_i + u_i` (smooth
across periodic boundaries); the global sum is well defined when the
blocks obey the acoustic sum rule.

Minimum image is applied to the *displacements*, so pairs across
periodic boundaries are handled exactly without any unwrapped-coordinate
bookkeeping; blocks are applied by the owner of atom *i* only, so the
fix is MPI-parallel without global communication.

## Build

    ./build.sh [LAMMPS_INCLUDE_DIR]

Without arguments the plugin is compiled against the headers shipped
with the `lammps` python wheel (`pip install lammps`). Requires a LAMMPS
binary built with the PLUGIN package (the wheel qualifies). Load with

    plugin load /path/to/fcpotplugin.so
    fix ID all fcpot harmonic.fc v_scale

The force-constant file is written by `calphy.harmonic.write_fc_file`
(spring-network blocks, or full hiphive FC2 blocks when
`fitting_backend: hiphive`).

## Related work

- [rohskopf/hessian](https://github.com/rohskopf/hessian) implements a
  similar second-order Taylor-series potential as LAMMPS pair styles
  (`pair_style hessian2`, targeting the 17 Nov 2016 LAMMPS release). It
  predates this plugin and inspired confidence that a Hessian potential
  is practical inside LAMMPS; fcpot was written independently to target
  current LAMMPS releases via the plugin interface, to avoid per-step
  global position communication, and to provide the scaling/energy
  hooks that nonequilibrium thermodynamic integration needs.
- [GPUMD](https://gpumd.org/potentials/fcp.html) supports hiphive-style
  force-constant potentials natively (hiphive ships `write_fcp_txt` for
  it); no equivalent exists in upstream LAMMPS at the time of writing.
