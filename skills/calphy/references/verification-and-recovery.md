# Verifying results, recovering killed runs, using F(T) curves

## The checks, and what each can and cannot see

| check | cost | catches | blind to |
|---|---|---|---|
| `dissipation` / `ts_dissipation` | free | switching too fast; a phase change partway through a path | errors common to forward and backward (barostat target, superheated chunk, biased anchor) |
| `error` | free (needs `n_iterations ≥ 2`) | sampling noise | everything systematic |
| chunk join (offset in the overlap) | free with overlapping chunks | an absolute disagreement between two chunks | which chunk is wrong |
| overlap **slope** test (compare S = −dG/dT in the overlap) | free | separates an anchor offset (same slope, constant shift) from a sweep-shape error (different slopes) | — |
| volume trace from `ts.*.dat` (column 3 / natoms vs T = T0/λ) | free | a sweep that did not hold its isobar (V falling on heating), runaway boxes | — |
| independent `mode: fe` spot checks inside a chunk | one reference run each | absolute error of the sweep at that T | — |
| ∂G/∂p = V across a pressure series | free | inconsistency between pressures | — |

Rules that fell out of real campaigns:

- Run 2–3 `fe` spot checks per chunk, spaced along it, so a λ-dependent error shows up.
  This is the only absolute arbiter.
- **Never align chunks by shifting one onto the other.** Where slopes differ, alignment
  hides a curve of the wrong shape, and every derived quantity (S, C_p, dTm/dp) stays wrong
  while the plot looks continuous.
- To validate a fix or a new protocol, test in the regime where the expected defect is
  *small*. A fix that overshoots passes the large-error case and fails the small one.
- Report joins and spot-check residuals next to dissipation whenever a curve is stitched.

## Dissipation bands (eV/atom)

| value | meaning |
|---|---|
| 1e-5 – 1e-4 | clean |
| ~1e-3 | the default `tolerance.dissipation`; slow down or narrow the chunk |
| 1e-2 – 1e-1 | the structure changed partway (melted, froze, vitrified, transformed). More iterations do not help |

## Recovering a run killed before all sweep iterations finished

`temperature_sweep.dat` is written only after all `n_iterations` sweeps. Each finished
iteration's `ts.forward_N.dat` / `ts.backward_N.dat` is complete on disk, and the
iteration that was running is truncated.

```python
import os, tempfile, numpy as np, yaml
from calphy.integrators import integrate_rs

def recover_sweep(folder, natoms):
    rep = yaml.safe_load(open(os.path.join(folder, "report.yaml")))
    inp = yaml.safe_load(open(os.path.join(folder, "input_file.yaml")))["calculations"][0]
    T0 = float(np.atleast_1d(inp["temperature"])[0])
    p = float(rep["input"]["pressure"])               # bar
    f0 = rep["results"]["free_energy"]                # the anchor G at T0
    # count only a leading run of complete forward/backward pairs
    n, ref = 0, None
    while True:
        f = os.path.join(folder, f"ts.forward_{n+1}.dat")
        b = os.path.join(folder, f"ts.backward_{n+1}.dat")
        if not (os.path.exists(f) and os.path.exists(b)):
            break
        lf, lb = (sum(1 for _ in open(x)) for x in (f, b))
        ref = ref or lf
        if min(lf, lb) < 0.999 * ref:                 # truncated: the iteration in flight
            break
        n += 1
    if n == 0:
        return None
    # integrate_rs ALWAYS writes temperature_sweep.dat into its folder:
    # run it on a temp dir of symlinks so the run tree is never modified
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, n + 1):
            for d in ("forward", "backward"):
                os.symlink(os.path.join(os.path.abspath(folder), f"ts.{d}_{i}.dat"),
                           os.path.join(tmp, f"ts.{d}_{i}.dat"))
        (T, F, err), ediss = integrate_rs(tmp, f0, T0, natoms, p=p, nsims=n,
                                          scale_energy=True,     # True for ts, False for tscale
                                          return_values=True)
    return T, F, err, ediss, n
```

Notes:

- The `integrate_rs` defaults (`nsims=5`, `scale_energy=False`) are **not** the `ts` values.
  Pass both explicitly.
- `natoms` is the atom count of the simulated cell (`input_configuration.data` header).
- With `return_values=True` it returns `((T, F, err), ediss)` (checked for 2.0.1 and
  2.3.0). The first element is three arrays, i.e. (3, N) column-wise. Iterating it yields
  whole columns, not points.
- Before trusting the recovery, check that it reproduces `temperature_sweep.dat` exactly
  on a run that *did* finish.
- Truncation error falls roughly as 1/n: 0.17, 0.04 and 0.02 meV/atom for 1, 2 and 3 of
  4 iterations in one case.
- Never relaunch over a folder with recoverable iterations. A guard that only looked for
  `temperature_sweep.dat` threw away thousands of core-hours of finished sweeps.

## Using a finished F(T) curve

- `temperature_sweep.dat` has one row per sweep step (10⁴–10⁶ rows). Downsample to a few
  hundred points before fitting. Some interpolators are O(N²).
- **C_p from a smooth fit only** (e.g. a CALPHAD-type a + bT + cT ln T + dT² form).
  Finite differences of the samples amplify chunk joins: one join of 0.1 meV/atom
  produced C_p = 7 k_B where a fit gave 3.9.
- **T → 0 extrapolation:** a classical harmonic solid has F = E0 − 3k_BT ln T + const·T,
  whose slope diverges at 0, so a polynomial misses by tens of meV/atom. G + 3k_BT ln T is
  linear in T, with intercept ≈ the relaxed static energy E0. Use that to check the curve
  against a static calculation, or to pin F(0) = E0 in a fit.
- **Phase crossings:** a crossing is only data where *both* curves have samples. Run solid
  sweeps a little into superheating (with the melt check on) and liquid sweeps a little
  into undercooling, so the crossing is interpolated, not extrapolated.
- **Mixing entropy:** calphy liquid F already contains ideal mixing entropy, through the
  ideal-gas reference. Solid F from a fixed decoration contains none. Tell your
  phase-diagram tool which convention each input follows. Double counting the entropy is
  the classic error.
- **Precision consistency:** compare phases computed with the same potential build and
  numerical precision. Reduced-precision GPU inference, e.g. TF32 in ML potentials,
  shifted energies by 0.5–3.5 meV/atom, by different amounts per phase, which is
  10–50 K on transition temperatures.
