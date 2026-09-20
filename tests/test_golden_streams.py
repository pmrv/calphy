"""Golden command-stream tests (Part 2).

Each test drives one calphy method with a RecordingRunner (no LAMMPS binary) and
snapshots the emitted command stream against tests/golden/<name>.txt.

Loops that converge on data (pressure, spring constant) are forced to converge on
the first cycle by loosening the convergence tolerances -- the tolerances are a
Python-side check and never appear in the command stream, so this does not change
the recorded commands, only the number of cycles (the multi-cycle repetition is
exercised with real LAMMPS in Part 6).  The pre-set box/spring values used for the
integration / sweep methods stand in for what run_averaging would have measured.
"""
import os

import pytest

from calphy.solid import Solid
from calphy.liquid import Liquid
from calphy.alchemy import Alchemy

from conftest import assert_golden

# Loosen convergence so the static avg.dat/msd.dat fixtures converge on cycle 1.
LOOSE_TOL = {"tolerance": {"pressure": 1e12, "spring_constant": 1e12}}

# Deterministic stand-ins for values run_averaging would set on the job.
BOX = 18.075
KSPRING = [2.0]


def _set_state(job, box=BOX, k=None):
    job.lx = job.ly = job.lz = box
    if k is not None:
        job.k = k


# --------------------------------------------------------------------------- #
# Solid fe: averaging (three equilibration paths) + integration
# --------------------------------------------------------------------------- #
def test_solid_fe_averaging(make_calc, recorded_job):
    calc = make_calc("B1", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc, fixtures=["avg.dat", "msd.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "solid_fe_averaging")


def test_solid_fe_averaging_finite_p(make_calc, recorded_job):
    calc = make_calc("B2", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc, fixtures=["avg.dat", "msd.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "solid_fe_averaging_finite_p")


def test_solid_fe_averaging_fixlattice(make_calc, recorded_job):
    calc = make_calc("B3", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc, fixtures=["avg.dat", "msd.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "solid_fe_averaging_fixlattice")


def test_solid_fe_integration(make_calc, recorded_job):
    calc = make_calc("B1", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job, k=KSPRING)
    job.run_integration(iteration=1)
    assert_golden(rec.commands, "solid_fe_integration")


# --------------------------------------------------------------------------- #
# Liquid fe: averaging (melt cycle / rattle) + integration
# --------------------------------------------------------------------------- #
def test_liquid_fe_averaging_meltcycle(make_calc, recorded_job):
    calc = make_calc("B4", **LOOSE_TOL)
    # solid-fraction sequence forces the melt loop to iterate 3 times (2 "still
    # solid" then a melted reading that breaks the loop).
    n = calc._natoms
    job, rec = recorded_job(
        Liquid, calc, fixtures=["avg.dat"], solid_fraction_seq=[n, n, 0]
    )
    job.run_averaging()
    assert_golden(rec.commands, "liquid_fe_averaging_meltcycle")


def test_liquid_fe_averaging_rattle(make_calc, recorded_job):
    calc = make_calc("B5", **LOOSE_TOL)
    job, rec = recorded_job(Liquid, calc, fixtures=["avg.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "liquid_fe_averaging_rattle")


def test_liquid_fe_integration(make_calc, recorded_job):
    calc = make_calc("B4", **LOOSE_TOL)
    job, rec = recorded_job(Liquid, calc)
    _set_state(job)
    job.run_integration(iteration=1)
    assert_golden(rec.commands, "liquid_fe_integration")


# --------------------------------------------------------------------------- #
# ts (reversible scaling): forward / backward / uniform-temperature schedule
# --------------------------------------------------------------------------- #
def test_ts_forward_solid(make_calc, recorded_job):
    calc = make_calc("B6", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job._reversible_scaling_forward(iteration=1)
    assert_golden(rec.commands, "ts_forward_solid")


def test_ts_backward_solid(make_calc, recorded_job):
    calc = make_calc("B6", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job._reversible_scaling_backward(iteration=1)
    assert_golden(rec.commands, "ts_backward_solid")


def test_ts_uniform_temperature(make_calc, recorded_job):
    calc = make_calc("B6", lambda_schedule="uniform_temperature", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job._reversible_scaling_forward(iteration=1)
    assert_golden(rec.commands, "ts_uniform_temperature")


# --------------------------------------------------------------------------- #
# tscale / pscale
# --------------------------------------------------------------------------- #
def test_tscale(make_calc, recorded_job):
    calc = make_calc("B8", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job.temperature_scaling(iteration=1)
    assert_golden(rec.commands, "tscale")


def test_pscale(make_calc, recorded_job):
    calc = make_calc("B9", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job.pressure_scaling(iteration=1)
    assert_golden(rec.commands, "pscale")


# --------------------------------------------------------------------------- #
# alchemy
# --------------------------------------------------------------------------- #
def test_alchemy_averaging(make_calc, recorded_job):
    calc = make_calc("B10", **LOOSE_TOL)
    job, rec = recorded_job(Alchemy, calc, fixtures=["avg.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "alchemy_averaging")


def test_alchemy_integration(make_calc, recorded_job):
    calc = make_calc("B10", **LOOSE_TOL)
    job, rec = recorded_job(Alchemy, calc)
    _set_state(job)
    job.run_integration(iteration=1)
    assert_golden(rec.commands, "alchemy_integration")


# --------------------------------------------------------------------------- #
# QTB (fe-qtb): averaging + integration.  Command recording needs no QTB binary.
# --------------------------------------------------------------------------- #
def test_qtb_averaging(make_calc, recorded_job):
    calc = make_calc("B11", **LOOSE_TOL)
    assert calc._qtb is True and calc.mode == "fe"
    job, rec = recorded_job(Solid, calc, fixtures=["avg.dat", "msd.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "qtb_averaging")


def test_qtb_integration(make_calc, recorded_job):
    calc = make_calc("B11", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job, k=KSPRING)
    job.run_integration(iteration=1)
    assert_golden(rec.commands, "qtb_integration")


# --------------------------------------------------------------------------- #
# Overlay potential (hybrid/overlay of two component pair styles)
# --------------------------------------------------------------------------- #
OVERLAY = {
    "pair_mode": "overlay",
    "pair_style": ["eam/alloy", "eam/alloy"],
    "pair_coeff": [
        "* * tests/Cu01.eam.alloy Cu",
        "* * tests/Cu01.eam.alloy Cu",
    ],
}


def test_overlay_potential(make_calc, recorded_job):
    calc = make_calc("B1", **OVERLAY, **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc, fixtures=["avg.dat", "msd.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "overlay_potential")


# --------------------------------------------------------------------------- #
# ts with Monte Carlo swaps (forward integration, two swap types)
# --------------------------------------------------------------------------- #
def test_ts_mc_swaps(make_calc, recorded_job):
    calc = make_calc(
        "B6",
        monte_carlo={
            "n_swaps": 5,
            "n_steps": 100,
            "forward_swap_types": [1, 2],
            "reverse_swap_types": [1, 2],
        },
        **LOOSE_TOL,
    )
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job._reversible_scaling_forward(iteration=1)
    assert_golden(rec.commands, "ts_mc_swaps")


# --------------------------------------------------------------------------- #
# Prescan (phase_transition_detection).  Records the diagnostic-ramp command
# stream; the RangeScan analysis (calphy.range_scan) runs against the recorded
# prescan.forward.dat when present and is skipped gracefully in dry-run.
# --------------------------------------------------------------------------- #
def test_prescan(make_calc, recorded_job):
    calc = make_calc("B12")
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job.scan_temperature_range()
    assert_golden(rec.commands, "prescan")


# --------------------------------------------------------------------------- #
# fe integration: chained iterations
# --------------------------------------------------------------------------- #
def test_solid_fe_integration_chained(make_calc, recorded_job, tmp_path):
    """Iteration 2 starts from the configuration the previous backward leg
    wrote (conf.fe.backward_1.data) rather than from conf.equilibration.data."""
    calc = make_calc("B1", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job, k=KSPRING)
    open(os.path.join(str(tmp_path), "conf.fe.backward_1.data"), "w").close()
    job.run_integration(iteration=2)
    assert_golden(rec.commands, "solid_fe_integration_chained")


def test_integration_start_configuration_fallback(make_calc, recorded_job, tmp_path):
    """Without the chained file a later iteration falls back to the
    equilibration configuration; iteration 1 never looks for one."""
    calc = make_calc("B1", **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    equil = os.path.join(str(tmp_path), "conf.equilibration.data")
    assert job._integration_start_configuration(1) == equil
    assert job._integration_start_configuration(3) == equil
    chained = os.path.join(str(tmp_path), "conf.fe.backward_2.data")
    open(chained, "w").close()
    assert job._integration_start_configuration(3) == chained
    assert job._integration_start_configuration(2) == equil   # backward_1 missing


def _seeds(commands):
    """(velocity seed, thermostat seed) drawn by one integration iteration."""
    vel = [c for c in commands if c.startswith("velocity all create")]
    therm = [c for c in commands if c.startswith("fix") and " langevin " in c]
    assert vel and therm
    return int(vel[-1].split()[4]), int(therm[-1].split()[7])


@pytest.mark.parametrize("JobClass, scenario, kwargs",
                         [(Solid, "B1", {"k": KSPRING}), (Liquid, "B4", {})])
def test_iterations_draw_fresh_seeds(make_calc, recorded_job, tmp_path, JobClass, scenario, kwargs):
    """Chained iterations reuse positions, so their independence rests on every
    iteration drawing new velocity and thermostat seeds from the job's stream."""
    calc = make_calc(scenario, **LOOSE_TOL)
    job, rec = recorded_job(JobClass, calc)
    _set_state(job, **kwargs)
    seen = []
    for it in (1, 2, 3):
        open(os.path.join(str(tmp_path), "conf.fe.backward_%d.data" % (it - 1)), "w").close()
        job.run_integration(iteration=it)
        seen.append(_seeds(rec.commands))
    vel, therm = zip(*seen)
    assert len(set(vel)) == 3, "velocity seeds repeat across iterations: %s" % (vel,)
    assert len(set(therm)) == 3, "thermostat seeds repeat across iterations: %s" % (therm,)


# --------------------------------------------------------------------------- #
# composition_scaling: the potential is rewritten into an initial and a final
# composition; with pair_mode overlay every component is carried along.
# --------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZRCU = os.path.join(REPO, "examples", "example_10", "ZrCu.data")


def _composition_scaling_job(make_calc, recorded_job, fixtures=(), **overrides):
    from calphy.composition_transformation import CompositionTransformation
    from calphy.routines import set_composition_scaling_potential

    # the transformation reads the real structure, so pin the sentinel afterwards
    calc = make_calc("B13", lattice_sentinel=None, lattice=ZRCU, **LOOSE_TOL, **overrides)
    comp = CompositionTransformation(calc)
    set_composition_scaling_potential(calc, comp)
    calc.element = comp.pair_list_old
    calc.lattice = "structure.data"
    return recorded_job(Alchemy, calc, fixtures=fixtures)


def test_composition_scaling_overlay_averaging(make_calc, recorded_job):
    job, rec = _composition_scaling_job(make_calc, recorded_job, fixtures=["avg.dat"])
    job.run_averaging()
    assert_golden(rec.commands, "composition_scaling_overlay_averaging")


def test_composition_scaling_overlay_integration(make_calc, recorded_job):
    job, rec = _composition_scaling_job(make_calc, recorded_job)
    _set_state(job)
    job.run_integration(iteration=1)
    assert_golden(rec.commands, "composition_scaling_overlay_integration")


def test_composition_scaling_integration(make_calc, recorded_job):
    """Plain (single component) potential: the pre-overlay command stream."""
    job, rec = _composition_scaling_job(
        make_calc, recorded_job,
        pair_mode=None,
        pair_style="eam/fs",
        pair_coeff="* * examples/potentials/ZrCu.eam.fs Zr Cu",
    )
    _set_state(job)
    job.run_integration(iteration=1)
    assert_golden(rec.commands, "composition_scaling_integration")


# --------------------------------------------------------------------------- #
# Barostat targets at finite pressure (issue #3).  At p = 0 the goldens above
# cannot tell a ramped barostat from a pinned one, so pin the numbers here:
# the scaled-potential sweeps must run at λp, the real-thermostat ramps at p.
# --------------------------------------------------------------------------- #
P_BAR = 50000.0
LF = 400.0 / 700.0           # B6/B8 sweep 400 -> 700 K
P_LF = LF * P_BAR


def _npt_fixes(commands, fix_id):
    return [c for c in commands if c.startswith("fix %s all npt" % fix_id)]


def _iso(cmd):
    """(Pstart, Pstop) of an ``fix ... npt ... iso Pstart Pstop Pdamp`` line."""
    tok = cmd.split()
    i = tok.index("iso")
    return float(tok[i + 1]), float(tok[i + 2])


def _fix_before_sweep(commands, fix_id, sweep_file):
    """The last ``fix <fix_id> all npt`` issued before the sweep's print fix."""
    idx = next(
        i for i, c in enumerate(commands)
        if c.startswith("fix f3 all print") and c.endswith(sweep_file)
    )
    return [c for c in commands[:idx] if c.startswith("fix %s all npt" % fix_id)][-1]


def test_ts_forward_barostat_ramps_to_lambda_p(make_calc, recorded_job):
    calc = make_calc("B6", pressure=P_BAR, **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job._reversible_scaling_forward(iteration=1)

    fixes = _npt_fixes(rec.commands, "f1")
    # warm start and COM-constrained equilibration hold the full pressure
    assert _iso(fixes[0]) == (P_BAR, P_BAR)
    assert _iso(fixes[1]) == (P_BAR, P_BAR)
    # the sweep itself ramps p -> lf*p, with the COM temperature re-attached
    sweep_fix = _fix_before_sweep(rec.commands, "f1", "ts.forward_1.dat")
    assert _iso(sweep_fix) == pytest.approx((P_BAR, P_LF))
    assert "fixedpoint ${xcm} ${ycm} ${zcm}" in sweep_fix
    assert rec.commands[rec.commands.index(sweep_fix) + 1] == "fix_modify f1 temp tcm"


def test_ts_backward_barostat_ramps_from_lambda_p(make_calc, recorded_job):
    calc = make_calc("B6", pressure=P_BAR, **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job._reversible_scaling_backward(iteration=1)

    fixes = _npt_fixes(rec.commands, "f1")
    # middle equilibration runs the lf-scaled potential, so it sits at lf*p
    assert _iso(fixes[0]) == pytest.approx((P_LF, P_LF))
    # the backward sweep ramps lf*p -> p
    sweep_fix = _fix_before_sweep(rec.commands, "f1", "ts.backward_1.dat")
    assert _iso(sweep_fix) == pytest.approx((P_LF, P_BAR))
    assert rec.commands[rec.commands.index(sweep_fix) + 1] == "fix_modify f1 temp tcm"


def test_ts_uniform_temperature_refused_at_finite_p(make_calc):
    with pytest.raises(ValueError, match="uniform_temperature"):
        make_calc(
            "B6", pressure=P_BAR, lambda_schedule="uniform_temperature",
            **LOOSE_TOL
        )
    # p = 0 and NVT stay allowed
    make_calc("B6", pressure=0.0, lambda_schedule="uniform_temperature", **LOOSE_TOL)
    make_calc(
        "B6", pressure=P_BAR, npt=False, lambda_schedule="uniform_temperature",
        **LOOSE_TOL
    )


def test_tscale_holds_pressure_and_ramps_back(make_calc, recorded_job):
    calc = make_calc("B8", pressure=P_BAR, **LOOSE_TOL)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job.temperature_scaling(iteration=1)

    # real thermostat + unscaled potential: every block sits on the isobar
    for cmd in _npt_fixes(rec.commands, "1") + _npt_fixes(rec.commands, "f2"):
        assert _iso(cmd) == (P_BAR, P_BAR), cmd
    fwd = _fix_before_sweep(rec.commands, "f2", "ts.forward_1.dat").split()
    bwd = _fix_before_sweep(rec.commands, "f2", "ts.backward_1.dat").split()
    assert (fwd[5], fwd[6]) == ("400.000000", "700.000000")
    assert (bwd[5], bwd[6]) == ("700.000000", "400.000000")


def test_prescan_holds_pressure(make_calc, recorded_job):
    calc = make_calc("B12", pressure=P_BAR)
    job, rec = recorded_job(Solid, calc)
    _set_state(job)
    job.scan_temperature_range()
    for cmd in _npt_fixes(rec.commands, "1") + _npt_fixes(rec.commands, "f2"):
        assert _iso(cmd) == (P_BAR, P_BAR), cmd
