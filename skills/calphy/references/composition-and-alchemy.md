# `alchemy` and `composition_scaling`

Both modes return a **free-energy difference**, not an absolute free energy.
`results.free_energy` is the switching work, mass-corrected for `composition_scaling`. To
get an absolute F, add the absolute F of the starting state (an `fe` or `ts` result at
the same T and p).

## `alchemy`: switch between two potentials

- `pair_style: [A, B]`, `pair_coeff: [A_coeff, B_coeff]`. The work is ∫⟨U_B − U_A⟩_λ dλ
  along a linear mix.
- Uses: upsampling (a cheap potential's absolute F plus Δ to an expensive one), potential
  comparisons, and switching one species to another when the potential models it as a
  type change.
- `npt: False` runs the switch at the volume equilibrated with the first potential.
- `pair_mode: overlay` is not allowed here unless `alchemy_coupling: True`, which ramps
  only the last overlay component from 0 to 1. To change the composition of an overlay
  potential, use `composition_scaling`.

## `composition_scaling`: switch composition at fixed structure

- Input: a structure (usually a LAMMPS data file) with the *input* composition, plus
  `composition_scaling.output_chemical_composition` with the target counts. The total
  number of atoms must be conserved. Older inputs also list `input_chemical_composition`;
  in 2.3.0 that is read from the structure and is **not** a valid key any more.
- Every `pair_coeff` must address all types (`* * …`), because calphy renumbers types.
  Element order in `element`, `mass` and the `pair_coeff` labels must match.
- During switching all masses are set to one reference mass. The mass difference is then
  added analytically: `results.mass_correction`, already applied to `free_energy`.
- `results.entropy_contribution` is the ideal-mixing entropy change between input and
  output composition, in eV/K/atom (multiply by −T for a free energy). It is **not**
  folded into `free_energy`. Whether you need it depends on your path:
  - Switching on a fixed decoration does not sample configurational entropy, so a random
    solid solution needs it added.
  - With MC swaps the situation differs.
  - Decide explicitly, and check against a direct `fe` calculation of the end
    composition. This is how the upstream example validates the mode.
- Intermediate λ are not physical intermediate compositions. Only the end points are
  correct free energies, because the mixing contribution is missing in between.
- One kind of transformation per run (`More than one kind of transformation found!`). More
  than two elements is untested upstream (logged as a warning).
- Cost: with MC swaps the per-step cost was ~3× plain MD in our runs.

## Lessons from real use

- **Write down which field you consume.** One project read `results.work` in its README
  and `results.free_energy` in its analysis notebook. They differ by the mass correction,
  which was ~2e-4 eV/atom there. That is small, but it is an inconsistency nobody chose.
- Alchemical results converged quickly in switching steps: 10k vs 25k steps agreed to
  1e-4 eV/atom for a Pt-Au switch. Measure this for your system before scaling up.
- 1.x releases needed patches for `composition_scaling` in some execution paths. Use ≥ 2.x
  and check the mode on a small cell against a direct `fe` difference first.
- Build a manifest of which runs are the current terminal (pure-element) references, and
  resolve them by name. A diagram built from stale terminals sitting next to newer runs
  gives no error, just a wrong diagram.
