# calphy input keys (verified against `calphy/input.py`, 2.3.0 / `main` @ 196ace6)

Every block is a strict pydantic model. An unknown key raises, and the message suggests
the nearest key or the block it belongs in. Removed v1 keys (`savefile`, `save_job`,
`load_job`, top-level `seed`, `lambda_schedule`, `script_mode`) raise with a migration
message.

The file is `{calculations: [ {...}, {...} ]}`. Any other top-level key is an error. A file
with top-level `element:` is the legacy v1 format, and v2 refuses it.

Units are LAMMPS `metal`: K, bar, Å, ps, eV, g/mol. The `units` command can be overridden
through `md.init_commands`, but every calphy post-processing formula assumes metal units.

## Calculation block

| key | type / default | notes |
|---|---|---|
| `element` | list[str], required | order = LAMMPS atom types 1..n |
| `mass` | list[float], same length | g/mol |
| `mode` | `fe` `fe-qtb` `ts` `tscale` `pscale` `alchemy` `composition_scaling` `melting_temperature` `dcci` | unknown modes fail at dispatch, not at validation |
| `reference_phase` | `solid` / `liquid` | selects the Einstein crystal or UFM reference. `dcci` forces `solid` |
| `lattice` | str, default `""` | built-in `fcc bcc hcp diamond simple_cubic`/`sc` `a15` (single element only); a LAMMPS data file path (`atom_style atomic`); or a Materials Project id `mp-…` (needs `mp_api` and the `materials_project` block). Empty = the element's ASE ground state with `repeat` [5,5,5] |
| `lattice_constant` | float, 0 | 0 with a built-in lattice = ASE reference value. Ignored for data files |
| `c_over_a` | float or None | built-in `hcp` only, otherwise an error. None = ideal √(8/3) |
| `repeat` | [int,int,int], [1,1,1] | built-in and MP lattices only. **Not applied to data files** in 2.3.0 |
| `file_format` | `lammps-data` | the only format accepted |
| `temperature` | float or [T0, T1] | `ts`/`tscale`: T0 is the anchor (not sorted). `fe`: scalar (a 2-list silently uses only T0). `melting_temperature`: guess. `dcci`: T_coex. If 0, the mendeleev melting point is used (single element only) |
| `temperature_high` | float, 0 → 2 × last temperature | liquid melt temperature. Set it explicitly |
| `pressure` | bar, default 0 | accepted shapes: scalar; `[p]`; `[p0, p1]` (`pscale`, `dcci`); `[[p,p,p]]`; `[[p,p,p],[q,q,q]]`. The three components must be equal. **A flat `[p,p,p]` (shape 3) is rejected** ("Unknown format for pressure"), although the docs list it. `None`/`null` = fixed box |
| `pressure_coupling` | `iso` / `aniso` / `tri`, None | None → `iso` for scalar input, `aniso` for the nested 3-form. `tri` needs a data file with a triclinic box |
| `npt` | bool, True | False: `ts` sweeps and `alchemy` in NVT. With `npt: False` keep `pressure: 0` for `ts`/`tscale`, because the integrator adds p·V regardless |
| `melting_cycle` | bool, True | liquid only. False = rattle the lattice instead of melting it, which may not melt it |
| `pair_style` | str or list[str] | a list for `alchemy` (from, to) and for `pair_mode: overlay` |
| `pair_coeff` | str or list[str] | the third token (the file) is made absolute if it exists relative to the CWD; `$VAR` and `~` are expanded |
| `pair_mode` | None / `overlay` | sum of several component styles. Not allowed with `alchemy` unless `alchemy_coupling` |
| `potential_file` | deprecated | use `pair_style` / `pair_coeff` |
| `fix_potential_path` | bool, True | turn off to pass `pair_coeff` through verbatim |
| `n_equilibration_steps` | int, 25000 | equilibration before each switching leg |
| `n_switching_steps` | int or [ref, sweep], [50000, 50000] | a scalar applies to both |
| `n_iterations` | int, 1 | independent forward and backward repetitions. `error` is 0 when this is 1 |
| `n_print_steps`, `n_print_steps_equilibration` | int, 0 | > 0 dumps full configurations during switching or equilibration, which gets large fast |
| `spring_constants` | list[float] or None | one per element (eV/Å²). Skips the MSD estimate (a short NVT run still happens) |
| `equilibration_control` | None → `nose-hoover` | only the exact strings `nose-hoover` and `berendsen` are meaningful. **Anything else silently means Berendsen** (no validation) |
| `folder_prefix` | str or None | prefixed to the folder name with `-`. An absolute path relocates the folder |
| `execution_mode` | `executable` (default) / `library` | see running-and-lammps.md |
| `lammps_executable`, `mpi_executable` | str or None | executable mode only. Fallback: `$CALPHY_LAMMPS_EXECUTABLE`/`lmp` and `$CALPHY_MPI_EXECUTABLE`/`mpirun` |
| `alchemy_coupling` | bool, False | `alchemy` variant that ramps only the last component of an overlay |
| `phase_name`, `reference_composition` | str, float | metadata for `calphy.phase_diagram` / `gather_results` |
| `kernel`, `inputfile`, `n_elements` | internal | filled in by the reader; do not set them |

## `md:`

| key | default | notes |
|---|---|---|
| `timestep` | 0.001 ps | |
| `n_small_steps` | 10000 | length of one averaging or convergence cycle, and of the melt and rattle runs |
| `n_every_steps`, `n_repeat_steps` | 10, 10 | `fix ave/time` sampling |
| `n_cycles` | 100 | maximum convergence cycles (pressure, spring constant) |
| `thermostat_damping`, `barostat_damping` | 0.1, 0.1 (ps) | damping during switching. Scalars only |
| `cmdargs` | `""` | extra LAMMPS command-line arguments, e.g. `-k on g 1 -sf kk -pk kokkos newton on neigh half` |
| `init_commands` | [] | replaces calphy's `units`/`boundary`/`atom_style`/`timestep` lines by first token, or appends |
| `seed` | None | master seed. None draws one, and the seed used is written to the folder's `input_file.yaml` |

## `tolerance:`

| key | default | notes |
|---|---|---|
| `solid_fraction` | **0.0** | melt check fires when the solid fraction falls below this. 0 = never. Use ~0.7 for simple-lattice solids |
| `liquid_fraction` | **1.0** | freeze check fires when the solid fraction rises above this. 1 = never. Use ~0.05 for liquids. The same threshold decides whether the liquid melt cycle succeeded |
| `dissipation` | 0.001 eV/atom | sets `ts_dissipation_high` and the log warning. 0 disables |
| `pressure` | 10 bar | NPT convergence target. Too tight for cells of a few hundred atoms |
| `spring_constant` | 0.1 | relative change of k between cycles (in 2.3.0 this compared the rounded MSD spread) |
| `lattice_constant` | 0.0002 | |

## `queue:`

`scheduler` (`local` / `slurm` / `sge`), `cores` (**1**; LAMMPS MPI ranks), `jobname`,
`walltime`, `queuename`, `memory`, `commands` (lines prepended to the generated script),
`options` (dict → `#SBATCH --key=value`). Only `cores` matters when you run
`calphy_kernel` yourself. The rest configures the scripts `calphy -i` writes.

## `uhlenbeck_ford_model:` (liquid reference)

`p` 50, `sigma` 1.5 Å. `sigma` may also be a dict of pair values (`{A_A: …, B_B: …,
A_B: …}`). With `single_sigma` (and optionally `single_p`) it becomes a two-leg path to a
single-component UFM. The cutoff is 5 σ.

## `monte_carlo:` (swaps during switching; needs the MC package)

`n_steps` 1 (swap every n steps), `n_swaps` 0, `forward_swap_types` [], `reverse_swap_types`
[], `allow_all_swaps` True, `use_custom_lammps` False.

## `composition_scaling:`

`output_chemical_composition` (dict element → count; the total must equal the input
count), `restrictions` (list like `["Al-O"]`). The input composition is read from the
structure. **`input_chemical_composition` is no longer an input key**, although older
examples and inputs still carry it.

## `melting_temperature:`

`step` 200 K (half-width of the bracket around `temperature`), `attempts` 5. `guess` is
accepted but not used by the code in 2.3.0. The guess is `temperature`, and if that is
unset the mendeleev melting point.

## `phase_transition_detection:` (pre-scan before a `ts` sweep)

`mode` `none` (default) / `warn` / `adapt` / `stop`, `prescan_steps` 20000,
`onset_fraction` 0.85. `adapt` cuts the sweep's top temperature to the detected onset.
Treat it as a diagnostic until validated on your system.

## `dcci:`

`n_block_steps` 1000, `integrator` `euler`/`trapezoid`, `stop_at_target_pressure` True,
`n_check_blocks` 0, `hysteresis_tolerance` 5 K, `parallel_cells` False. Needs `pressure:
[p0, p1]`, an explicit scalar `temperature` (the coexistence T at p0), `npt: True`, and no
swaps.

## Thermostat blocks

`nose_hoover:` `thermostat_damping` 0.1, `barostat_damping` 0.1 (equilibration).
`berendsen:` 100, 100. `quantum_thermal_bath:` (for `fe-qtb`) `thermostat_damping` 0.1,
`barostat_damping` 0.1, `f_max` 200 THz, `n_f` 100.

## `materials_project:`

`api_key` is the **name of an environment variable** holding the key, not the key itself.
`conventional` True, `target_natoms` 1500.
