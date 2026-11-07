# Pressure dependence: `ts` at pressure, `pscale`, and negative pressure

## Three routes to G(T, p)

1. **`ts` at each pressure** (calphy ≥ 2.3.0 only). Chunk with T_end / T_anchor ≲ 2 and
   check every chunk with `fe` spot checks. Even with the λp fix, wide-λ chunks at high
   |p| deserve the extra checks.
2. **`ts` at p = 0, then `pscale` at each T of interest.** Safe on every version: at
   p = 0 there is no barostat target to mis-scale, and `pscale` never scales the potential.
   One `pscale` run covers a whole pressure range, so the cost scales with the number of
   temperatures, not of pressures.
3. **`fe` grid** over (T, p). Simple and robust, but no curve between points.

## `pscale`

- Computes G(T, p) = G(T, p0) + ∫ V dp at fixed T. `routine_pscale` runs a full `fe`
  anchor at `pressure[0]` first, then `n_iterations` forward and backward pressure ramps.
- `pressure` must be `[p0, p1]` (or `[[p0,p0,p0],[p1,p1,p1]]`). A scalar is accepted and
  gives a zero-width ramp: `pressure_sweep.dat` comes out flat at the anchor value, after
  you have paid for the whole run. Assert the 2-list in your input builder.
- `temperature` is a scalar. `n_switching_steps: [anchor, ramp]`, as for `ts`.
- Output: `pressure_sweep.dat` (p [bar], G [eV/atom], error) and `ps.forward_N.dat` /
  `ps.backward_N.dat`. In `ps.*.dat`, **column 2 is pressure and column 3 is volume**.
  Column 4 is labelled lambda but is a ramp from 1 to p_stop (in bar). It only looks like
  pressure when p0 = 0, and `integrate_ps` ignores it.
- No dissipation is reported. Compute it from `ps.forward_N.dat` / `ps.backward_N.dat`
  yourself (|W_f − W_b|/2 along the ramp).
- The ∫V dp part does not depend on the anchor. If you already have a validated
  G(T, p0), you can take the work from the ramp files and add your own constant.
- Check: ∂G/∂p at fixed T must reproduce the sampled volumes per atom. Across a set of
  runs this costs nothing.

## Melting moves with pressure

The superheating limit of a solid follows Tm(p). Reusing one set of chunk boundaries
across a pressure series pushes the top chunk over the limit wherever Tm is lower. In
practice that means under tension (Tm fell ~40 K/GPa in one case). For an isothermal
`pscale` ramp toward tension, the binding melting point is the one at the *far* end of
the ramp. Compression is the safe direction for a solid. For a liquid it is the reverse:
compression approaches freezing. Linear Clausius-Clapeyron did not hold over ±5 GPa in our
data, so measure one real crossing per pressure before extrapolating Tm(p).

## Negative pressure (tension)

- Both phases are metastable against cavitation. The solid check notices a disordered
  solid, but a cavitated liquid still looks liquid (Steinhardt). Watch volume per atom:
  a runaway V is the only reliable signal.
- Locate the spinodal first. An NPT ramp from 0 to strongly negative pressure over a few
  tens of ps, watching V/atom, costs about as much as one yes/no hold test. In one case the
  liquid spinodal (−5.5 GPa) was much shallower than the solid's (−7.4 GPa), and the liquid
  margin was the binding one. Then hold at the target pressure for several times the
  sweep's equilibration. Cavitation is nucleation, so a short pass is necessary, not
  sufficient.
- The liquid melt cycle barostats to the **full target pressure** at `temperature_high`,
  and escalates on failure. Under tension that walks toward the spinodal.
  `melting_cycle: false` avoids the barostat but may not melt the crystal. What worked:
  melt at p = 0 yourself, then give calphy the liquid configuration. calphy equilibrates at
  zero pressure first and then converges onto the target.
- `Domain too large for neighbor bins` is the LAMMPS symptom of a box that ran away
  (cavitation or vaporisation). Before calphy 2.3.0 it was also caused by the RS barostat
  bug at the cold end of wide chunks.
- A run that exploded is *finished*. A resubmission guard that counts rows will treat it
  as incomplete and relaunch it forever. Treat "volume left the building" as a terminal
  state.
