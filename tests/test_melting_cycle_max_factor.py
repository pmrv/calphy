"""melting_cycle_max_factor caps the melting cycle's temperature escalation.

melt_structure retried at temperature_high * thmult for a hard-coded
thmult = 1.0 .. 1.9, which can put the thermostat above the boiling point.
The default keeps that ladder; 1.0 is a single attempt at temperature_high.
"""
from types import SimpleNamespace
import logging

import numpy as np
import pytest
from pydantic import ValidationError

import calphy.liquid as liquid_mod
from calphy.errors import SolidifiedError
from calphy.input import Calculation
from calphy.liquid import Liquid

BASE = dict(
    element=["Cu"], mass=[63.5], mode="fe", temperature=1500, pressure=0,
    lattice="FCC", lattice_constant=3.6, reference_phase="liquid",
    pair_style="eam/alloy", pair_coeff="* * Cu.eam.alloy Cu", temperature_high=1800,
    tolerance={"liquid_fraction": 0.05},
)


def melt_temperatures(tmp_path, monkeypatch, **extra):
    """Thermostat temperatures of every melting attempt for a cell that never melts."""
    monkeypatch.setattr(liquid_mod.ph, "find_solid_fraction", lambda f: 90)
    calc = Calculation(**{**BASE, **extra})
    temps = []
    lmp = SimpleNamespace(command=lambda s: None, rotate_logs=lambda s: None)
    job = SimpleNamespace(
        calc=calc, natoms=100, simfolder=str(tmp_path),
        logger=logging.getLogger("test_melting_cycle_max_factor"),
        fix_nose_hoover=lambda lmp, temp_start_factor, temp_end_factor:
            temps.append(temp_start_factor * calc._temperature),
        unfix_nose_hoover=lambda lmp: None,
        dump_current_snapshot=lambda lmp, f: None,
        lammps_close=lambda lmp: None,
    )
    with pytest.raises(SolidifiedError):
        Liquid.melt_structure(job, lmp)
    return temps


def test_default_is_the_historical_ladder(tmp_path, monkeypatch):
    assert melt_temperatures(tmp_path, monkeypatch) == pytest.approx(
        [1800 * f for f in np.arange(1.0, 2.0, 0.1)])


def test_one_is_a_single_attempt_at_temperature_high(tmp_path, monkeypatch):
    assert melt_temperatures(tmp_path, monkeypatch, melting_cycle_max_factor=1.0) == [1800.0]


def test_cap_bounds_the_escalation(tmp_path, monkeypatch):
    assert melt_temperatures(tmp_path, monkeypatch, melting_cycle_max_factor=1.2) == pytest.approx(
        [1800.0, 1980.0, 2160.0])


def test_below_one_is_rejected():
    with pytest.raises(ValidationError):
        Calculation(**{**BASE, "melting_cycle_max_factor": 0.9})
