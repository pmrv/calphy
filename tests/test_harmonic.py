"""
Tests for the harmonic (phonon) spring-network reference
(calphy/harmonic.py) and its input-schema surface.

These tests are pure Python: the spring-network model is exercised with
synthetic data, so no LAMMPS executable is required.
"""

import os

import numpy as np
import pytest
import scipy.constants as const
import yaml

from calphy.harmonic import (
    HarmonicModel,
    find_neighbor_pairs,
    group_pairs,
    minimum_image,
    read_lammps_dump,
    tdep_reference,
)
from calphy.input import read_inputfile


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def make_fcc(a=4.05, n=3):
    """fcc supercell positions and box"""
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    pos = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pos.extend(base + np.array([i, j, k]) * a)
    return np.array(pos), np.array([n * a] * 3)


@pytest.fixture
def fcc_model():
    pos, box = make_fcc()
    types = np.ones(len(pos), dtype=int)
    return HarmonicModel(pos, box, types, [26.98], cutoff=4.2)


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------


def test_neighbor_pairs_fcc_shells():
    pos, box = make_fcc()
    pairs, distances = find_neighbor_pairs(pos, box, 4.2)
    n = len(pos)
    # 12 first neighbours (a/sqrt(2)) and 6 second neighbours (a) per atom
    assert len(pairs) == (12 * n + 6 * n) // 2
    types = np.ones(n, dtype=int)
    group_index, groups = group_pairs(pairs, distances, types)
    assert len(groups) == 2
    assert groups[0]["count"] == 12 * n // 2
    assert groups[1]["count"] == 6 * n // 2
    assert np.isclose(groups[0]["distance"], 4.05 / np.sqrt(2), atol=1e-8)
    assert np.isclose(groups[1]["distance"], 4.05, atol=1e-8)


def test_cutoff_larger_than_half_box_raises():
    pos, box = make_fcc(n=2)
    with pytest.raises(ValueError, match="half"):
        find_neighbor_pairs(pos, box, cutoff=0.51 * box[0])


def test_minimum_image():
    box = np.array([10.0, 10.0, 10.0])
    v = np.array([[9.0, -9.0, 0.2]])
    out = minimum_image(v, box)
    assert np.allclose(out, [[-1.0, 1.0, 0.2]])


# ----------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------


def test_fit_recovers_known_spring_constants(fcc_model):
    pos = fcc_model.reference_positions
    k_true = np.array([1.2, 0.4])
    fcc_model.set_spring_constants(k_true)

    rng = np.random.default_rng(42)
    snaps, forces = [], []
    for _ in range(8):
        p = pos + rng.uniform(-0.05, 0.05, size=pos.shape)
        snaps.append(p)
        forces.append(fcc_model.forces(p))

    fit_model = HarmonicModel(
        pos, fcc_model.box, fcc_model.types, [26.98], cutoff=4.2
    )
    rmse = fit_model.fit(snaps, forces)
    assert rmse < 1e-10
    assert np.allclose(fit_model.k_groups, k_true, atol=1e-10)


def test_fit_subtracts_base_forces(fcc_model):
    pos = fcc_model.reference_positions
    k_true = np.array([1.0, 0.3])
    fcc_model.set_spring_constants(k_true)

    rng = np.random.default_rng(7)
    offset = rng.normal(0, 0.01, size=pos.shape)  # spurious residual force
    snaps, forces = [], []
    for _ in range(8):
        p = pos + rng.uniform(-0.04, 0.04, size=pos.shape)
        snaps.append(p)
        forces.append(fcc_model.forces(p) + offset)

    fit_model = HarmonicModel(
        pos, fcc_model.box, fcc_model.types, [26.98], cutoff=4.2
    )
    fit_model.fit(snaps, forces, base_forces=offset)
    assert np.allclose(fit_model.k_groups, k_true, atol=1e-10)


# ----------------------------------------------------------------------
# Hessian and phonons
# ----------------------------------------------------------------------


def test_hessian_matches_finite_differences(fcc_model):
    fcc_model.set_spring_constants([1.2, 0.4])
    pos = fcc_model.reference_positions
    H = fcc_model.hessian()
    eps = 1e-5
    for atom, comp in [(0, 0), (5, 1), (17, 2)]:
        p1 = pos.copy()
        p1[atom, comp] += eps
        p2 = pos.copy()
        p2[atom, comp] -= eps
        # gradient of the energy is -force
        col_fd = (-fcc_model.forces(p1) + fcc_model.forces(p2)).reshape(-1) / (
            2 * eps
        )
        assert np.allclose(col_fd, H[:, 3 * atom + comp], atol=1e-8)


def test_frequencies_zero_modes_and_count(fcc_model):
    fcc_model.set_spring_constants([1.2, 0.4])
    omega = fcc_model.frequencies()
    n = fcc_model.natoms
    assert len(omega) == 3 * n - 3
    assert np.all(omega > 0)


def test_unstable_network_raises(fcc_model):
    # strongly negative springs make the network unstable
    fcc_model.set_spring_constants([-1.0, -1.0])
    with pytest.raises(ValueError, match="unstable"):
        fcc_model.frequencies()


# ----------------------------------------------------------------------
# free energy
# ----------------------------------------------------------------------


def test_quantum_reduces_to_classical_at_high_temperature(fcc_model):
    fcc_model.set_spring_constants([1.2, 0.4])
    omega = fcc_model.frequencies()
    T = 20000.0
    f_cl = fcc_model.free_energy(T, quantum=False, omega=omega)["f_modes"]
    f_q = fcc_model.free_energy(T, quantum=True, omega=omega)["f_modes"]
    # leading Wigner correction: sum (hbar w)^2 / (24 kT) per atom
    corr = np.sum((const.hbar * omega) ** 2 / (24 * const.k * T)) / (
        const.e * fcc_model.natoms
    )
    assert f_q > f_cl
    assert np.isclose(f_q - f_cl, corr, rtol=1e-3)


def test_quantum_free_energy_approaches_zero_point_energy(fcc_model):
    fcc_model.set_spring_constants([1.2, 0.4])
    omega = fcc_model.frequencies()
    T = 0.1  # K; Bose term negligible
    f_q = fcc_model.free_energy(T, quantum=True, omega=omega)["f_modes"]
    zpe = np.sum(0.5 * const.hbar * omega) / (const.e * fcc_model.natoms)
    assert np.isclose(f_q, zpe, rtol=1e-6)


def test_classical_free_energy_scaling_with_springs(fcc_model):
    """F_cl shifts by (3N-3)/N * kT/2 * ln(4) when all k are scaled by 4."""
    T = 900.0
    kb_eV = const.physical_constants["Boltzmann constant in eV/K"][0]
    fcc_model.set_spring_constants([1.2, 0.4])
    f1 = fcc_model.free_energy(T, quantum=False)["f_modes"]
    fcc_model.set_spring_constants([4 * 1.2, 4 * 0.4])
    f2 = fcc_model.free_energy(T, quantum=False)["f_modes"]
    n = fcc_model.natoms
    expected = (3 * n - 3) / n * 0.5 * kb_eV * T * np.log(4)
    assert np.isclose(f2 - f1, expected, rtol=1e-10)


def test_com_term_is_small_and_negative(fcc_model):
    fcc_model.set_spring_constants([1.2, 0.4])
    res = fcc_model.free_energy(500.0)
    assert res["f_com"] < 0
    assert abs(res["f_com"]) < 0.05  # eV/atom, finite-size term


# ----------------------------------------------------------------------
# site tether
# ----------------------------------------------------------------------


def test_tether_eigenvalue_shift(fcc_model):
    """For equal masses, the tether shifts every eigenvalue of the
    mass-weighted Hessian by exactly 2 k_t / m."""
    from calphy.harmonic import OMEGA2_TO_SI

    fcc_model.set_spring_constants([1.2, 0.4])
    omega_free = fcc_model.frequencies()  # 3N-3 finite modes

    k_t = 0.3
    m = 26.98
    fcc_model.set_tether(k_t)
    omega_teth = fcc_model.frequencies()  # all 3N modes
    n = fcc_model.natoms
    assert len(omega_teth) == 3 * n

    shift = 2 * k_t / m * OMEGA2_TO_SI
    expected = np.sort(
        np.concatenate((np.zeros(3), omega_free**2)) + shift
    )
    assert np.allclose(omega_teth**2, expected, rtol=1e-8)


def test_tether_stabilises_unstable_network(fcc_model):
    """A network with negative springs that is unstable on its own can be
    made positive definite by a sufficiently stiff tether."""
    fcc_model.set_spring_constants([-0.05, -0.05])
    with pytest.raises(ValueError):
        fcc_model.frequencies()
    fcc_model.set_tether(5.0)
    omega = fcc_model.frequencies()
    assert np.all(omega > 0)


def test_tether_com_correction_einstein_form(fcc_model):
    fcc_model.set_spring_constants([1.2, 0.4])
    fcc_model.set_tether(0.3)
    res = fcc_model.free_energy(600.0)
    # Frenkel-Smit CM correction: small negative eV/atom value at this size
    assert res["f_com"] < 0
    assert abs(res["f_com"]) < 0.05
    assert res["f_total"] == pytest.approx(res["f_modes"] + res["f_com"])


def test_save_load_roundtrip_with_tether(fcc_model, tmp_path):
    fcc_model.set_spring_constants([1.2, 0.4])
    fcc_model.set_tether(0.25)
    fn = str(tmp_path / "model.npz")
    fcc_model.save(fn)
    from calphy.harmonic import HarmonicModel

    loaded = HarmonicModel.load(fn)
    assert loaded.tether_k == pytest.approx(0.25)
    assert np.allclose(loaded.frequencies(), fcc_model.frequencies(), rtol=1e-10)


# ----------------------------------------------------------------------
# LAMMPS file IO
# ----------------------------------------------------------------------


def test_write_bond_data_and_coeffs(fcc_model, tmp_path):
    fcc_model.set_spring_constants([1.2, 0.4])
    fcc_model.set_tether(0.3)
    n = fcc_model.natoms
    k_ein = fcc_model.einstein_anchor_constants(600.0)

    fn = tmp_path / "conf.harmonic.data"
    thermal = fcc_model.reference_positions + 0.01
    fcc_model.write_bond_data(str(fn), thermal)
    text = fn.read_text()
    assert "%d atoms" % (2 * n) in text
    assert "2 atom types" in text
    # G network types + tether + one Einstein type per element
    assert "%d bond types" % (2 + 1 + 1) in text
    assert "%d bonds" % (len(fcc_model.pairs) + 2 * n) in text

    # bonds section sanity: tether and einstein bonds connect id i to
    # ghost id offset+i
    lines = text.splitlines()
    start = lines.index("Bonds") + 2
    bonds = [tuple(int(x) for x in l.split()) for l in lines[start:] if l.strip()]
    assert len(bonds) == len(fcc_model.pairs) + 2 * n
    offset = fcc_model.ghost_id_offset
    tether_bonds = [b for b in bonds if b[1] == 3]
    ein_bonds = [b for b in bonds if b[1] == 4]
    assert len(tether_bonds) == n
    assert len(ein_bonds) == n
    for b in tether_bonds + ein_bonds:
        assert b[3] == b[2] + offset

    cmds = fcc_model.bond_coeff_commands(k_ein)
    assert len(cmds) == 4
    assert "harmonic 1.2" in cmds[0].replace("000000000", "")
    assert cmds[2].split()[2] == "harmonic"  # tether
    assert float(cmds[2].split()[3]) == pytest.approx(0.3)
    assert float(cmds[2].split()[4]) == 0.0
    assert cmds[3].split()[2] == "class2"
    assert float(cmds[3].split()[3]) == 0.0  # r0
    assert float(cmds[3].split()[4]) == pytest.approx(k_ein[0])
    assert float(cmds[3].split()[5]) == 0.0  # K3
    assert float(cmds[3].split()[6]) == 0.0  # K4


def test_einstein_anchor_constants_tether_only_limit(fcc_model):
    """With zero network springs, H = 2 k_t I, so <u^2> = 3kT/(2 k_t)
    and the amplitude-matched anchor constant is exactly k_t."""
    fcc_model.set_spring_constants([0.0, 0.0])
    fcc_model.set_tether(0.7)
    K_E = fcc_model.einstein_anchor_constants(600.0)
    assert K_E[0] == pytest.approx(0.7, rel=1e-10)


def test_reference_energies(fcc_model):
    fcc_model.set_spring_constants([1.2, 0.4])
    fcc_model.set_tether(0.3)
    k_ein = np.array([2.0])
    rng = np.random.default_rng(5)
    disp = rng.uniform(-0.1, 0.1, size=(fcc_model.natoms, 3))
    pos = fcc_model.reference_positions + disp
    e_harm, e_ein = fcc_model.reference_energies(pos, k_ein)
    u2 = np.sum(disp**2)
    assert e_ein == pytest.approx(2.0 * u2, rel=1e-10)
    assert e_harm == pytest.approx(fcc_model.energy(pos) + 0.3 * u2, rel=1e-10)
    # at the reference both vanish
    e_harm0, e_ein0 = fcc_model.reference_energies(
        fcc_model.reference_positions, k_ein
    )
    assert abs(e_harm0) < 1e-10 and abs(e_ein0) < 1e-10


def test_read_lammps_data(tmp_path):
    content = """conf (written by test)

4 atoms
1 atom types
0.0 10.0 xlo xhi
0.0 11.0 ylo yhi
0.0 12.0 zlo zhi

Masses

1 63.546

Atoms # atomic

2 1 1.0 2.0 3.0 0 0 0
1 1 4.0 5.0 6.0 0 1 0
4 1 0.5 0.5 0.5 0 0 0
3 1 9.0 9.0 9.0 0 0 0

Velocities

1 0.1 0.1 0.1
2 0.2 0.2 0.2
3 0.3 0.3 0.3
4 0.4 0.4 0.4
"""
    from calphy.harmonic import read_lammps_data

    fn = tmp_path / "conf.data"
    fn.write_text(content)
    data = read_lammps_data(str(fn))
    assert np.allclose(data["box"], [10, 11, 12])
    assert list(data["ids"]) == [1, 2, 3, 4]
    assert np.allclose(data["positions"][0], [4.0, 5.0, 6.0])
    assert np.allclose(data["positions"][1], [1.0, 2.0, 3.0])


def test_model_save_load_roundtrip(fcc_model, tmp_path):
    fcc_model.set_spring_constants([1.2, 0.4])
    fn = str(tmp_path / "model.npz")
    fcc_model.save(fn)
    loaded = HarmonicModel.load(fn)
    assert np.allclose(loaded.k_groups, fcc_model.k_groups)
    assert np.allclose(loaded.reference_positions, fcc_model.reference_positions)
    assert np.allclose(
        loaded.frequencies(), fcc_model.frequencies(), rtol=1e-10
    )


def test_read_lammps_dump(tmp_path):
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 11.0
0.0 12.0
ITEM: ATOMS id type x y z fx fy fz
2 1 1.0 2.0 3.0 0.1 0.2 0.3
1 2 4.0 5.0 6.0 -0.1 -0.2 -0.3
"""
    fn = tmp_path / "test.dump"
    fn.write_text(content)
    data = read_lammps_dump(str(fn))
    assert np.allclose(data["box"], [10, 11, 12])
    # sorted by id
    assert list(data["ids"]) == [1, 2]
    assert list(data["types"]) == [2, 1]
    assert np.allclose(data["positions"][0], [4.0, 5.0, 6.0])
    assert np.allclose(data["forces"][1], [0.1, 0.2, 0.3])


def test_read_lammps_dump_rejects_triclinic(tmp_path):
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 10.0 0.5
0.0 11.0 0.0
0.0 12.0 0.0
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
"""
    fn = tmp_path / "tri.dump"
    fn.write_text(content)
    with pytest.raises(ValueError, match="[Tt]riclinic"):
        read_lammps_dump(str(fn))


# ----------------------------------------------------------------------
# force-constant blocks (fcpot implementation)
# ----------------------------------------------------------------------


def test_fc_blocks_match_hessian(fcc_model):
    from calphy.harmonic import hessian_from_blocks

    fcc_model.set_spring_constants([1.2, 0.4])
    H_blocks = hessian_from_blocks(
        fcc_model.natoms, fcc_model.fc_blocks(include_tether=False)
    )
    assert np.allclose(H_blocks, fcc_model.hessian(), atol=1e-12)

    fcc_model.set_tether(0.3)
    H_blocks_t = hessian_from_blocks(
        fcc_model.natoms, fcc_model.fc_blocks(include_tether=True)
    )
    assert np.allclose(H_blocks_t, fcc_model.hessian(), atol=1e-12)


def test_frequencies_from_hessian_matches_model(fcc_model):
    from calphy.harmonic import frequencies_from_hessian

    fcc_model.set_spring_constants([1.2, 0.4])
    om1 = frequencies_from_hessian(
        fcc_model.hessian(), fcc_model.masses, tethered=False
    )
    assert np.allclose(om1, fcc_model.frequencies(), rtol=1e-12)

    fcc_model.set_tether(0.3)
    om2 = frequencies_from_hessian(
        fcc_model.hessian(), fcc_model.masses, tethered=True
    )
    assert np.allclose(om2, fcc_model.frequencies(), rtol=1e-12)


def test_acoustic_sum_rule():
    from calphy.harmonic import apply_acoustic_sum_rule, hessian_from_blocks

    rng = np.random.default_rng(3)
    n = 5
    blocks = []
    for i in range(n):
        blocks.append((i, i, rng.normal(size=(3, 3))))  # bogus diagonals
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 4)]:
        B = rng.normal(size=(3, 3))
        blocks.append((i, j, B))
    fixed = apply_acoustic_sum_rule(n, blocks)
    H = hessian_from_blocks(n, fixed)
    # translation vectors are exact null vectors
    for ax in range(3):
        t = np.zeros(3 * n)
        t[ax::3] = 1.0
        assert np.allclose(H @ t, 0.0, atol=1e-12)


def test_write_fc_file_roundtrip(fcc_model, tmp_path):
    from calphy.harmonic import write_fc_file

    fcc_model.set_spring_constants([1.2, 0.4])
    blocks = fcc_model.fc_blocks(include_tether=False)
    fn = tmp_path / "test.fc"
    write_fc_file(str(fn), fcc_model.ids, fcc_model.reference_positions, blocks)

    lines = [
        l for l in fn.read_text().splitlines() if l.strip() and not l.startswith("#")
    ]
    n = fcc_model.natoms
    assert int(lines[0]) == n
    sites = np.array([[float(x) for x in l.split()[1:]] for l in lines[1 : n + 1]])
    assert np.allclose(sites, fcc_model.reference_positions)
    nb = int(lines[n + 1])
    assert nb == len(blocks)
    for l in lines[n + 2 :]:
        tok = l.split()
        assert int(tok[0]) <= int(tok[1])
        assert len(tok) == 11


# ----------------------------------------------------------------------
# multi-component grouping
# ----------------------------------------------------------------------


def test_binary_lattice_groups_by_type_pair():
    # B2 (CsCl-like) lattice: type 1 at corners, type 2 at centres
    a = 3.0
    n = 3
    pos, types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pos.append(np.array([i, j, k]) * a)
                types.append(1)
                pos.append((np.array([i, j, k]) + 0.5) * a)
                types.append(2)
    pos = np.array(pos)
    types = np.array(types)
    box = np.array([n * a] * 3)

    model = HarmonicModel(pos, box, types, [50.0, 100.0], cutoff=3.2)
    # shells: 1-2 at a*sqrt(3)/2 = 2.598, 1-1 and 2-2 at a = 3.0
    labels = {(g["type_i"], g["type_j"]) for g in model.groups}
    assert (1, 2) in labels
    assert (1, 1) in labels
    assert (2, 2) in labels
    # masses assigned per type
    assert model.masses[types == 1][0] == 50.0
    assert model.masses[types == 2][0] == 100.0


# ----------------------------------------------------------------------
# input schema
# ----------------------------------------------------------------------


def _base_calc():
    return {
        "element": "Cu",
        "lattice": "FCC",
        "lattice_constant": 3.615,
        "mass": 63.546,
        "mode": "fe",
        "n_equilibration_steps": 1000,
        "n_iterations": 1,
        "n_switching_steps": 1500,
        "pair_coeff": "* * tests/Cu01.eam.alloy Cu",
        "pair_style": "eam/alloy",
        "pressure": 0.0,
        "reference_phase": "solid",
        "repeat": [4, 4, 4],
        "temperature": 300.0,
    }


def _write_yaml(tmp_path, payload):
    fn = tmp_path / "input.yaml"
    fn.write_text(yaml.safe_dump(payload))
    return str(fn)


def test_input_defaults(tmp_path):
    calc = _base_calc()
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)
    hr = opts.harmonic_reference
    assert hr.enabled is False
    assert hr.cutoff == pytest.approx(5.0)
    assert hr.n_snapshots == 25
    assert hr.quantum is False
    assert hr.fitting_backend == "leastsq"


def test_input_enable_and_options(tmp_path):
    calc = _base_calc()
    calc["harmonic_reference"] = {
        "enabled": True,
        "cutoff": 4.2,
        "n_snapshots": 10,
        "displacement": 0.03,
        "quantum": True,
    }
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)
    hr = opts.harmonic_reference
    assert hr.enabled is True
    assert hr.cutoff == pytest.approx(4.2)
    assert hr.n_snapshots == 10
    assert hr.displacement == pytest.approx(0.03)
    assert hr.quantum is True


def test_input_rejects_liquid_reference(tmp_path):
    calc = _base_calc()
    calc["reference_phase"] = "liquid"
    calc["harmonic_reference"] = {"enabled": True}
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    with pytest.raises(Exception, match="liquid"):
        read_inputfile(fn)


def test_input_rejects_script_mode(tmp_path):
    calc = _base_calc()
    calc["script_mode"] = True
    calc["lammps_executable"] = "lmp"
    calc["mpi_executable"] = "mpirun"
    calc["harmonic_reference"] = {"enabled": True}
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    with pytest.raises(Exception, match="script_mode"):
        read_inputfile(fn)


def test_input_rejects_alchemy(tmp_path):
    calc = _base_calc()
    calc["mode"] = "alchemy"
    calc["harmonic_reference"] = {"enabled": True}
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    with pytest.raises(Exception, match="alchemy|only supported"):
        read_inputfile(fn)


def test_input_fcpot_requires_plugin_path(tmp_path):
    calc = _base_calc()
    calc["harmonic_reference"] = {"enabled": True, "implementation": "fcpot"}
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    with pytest.raises(Exception, match="plugin_path"):
        read_inputfile(fn)


def test_fcpot_integration_command_stream(tmp_path, monkeypatch):
    from calphy.solid import Solid
    import calphy.helpers as ph

    calc = _base_calc()
    calc["lattice"] = os.path.join(os.path.dirname(__file__), "conf1.data")
    calc["harmonic_reference"] = {
        "enabled": True,
        "cutoff": 4.2,
        "implementation": "fcpot",
        "plugin_path": "/opt/fcpotplugin.so",
    }
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)

    sim = tmp_path / "sim"
    sim.mkdir()
    job = Solid(calculation=opts, simfolder=str(sim))

    pos, box = make_fcc(a=3.615)
    model = HarmonicModel(
        pos, box, np.ones(len(pos), dtype=int), [63.546], cutoff=4.2
    )
    model.set_spring_constants([1.0, 0.5])
    job.harmonic_model = model
    job.harmonic_fc_blocks = model.fc_blocks(include_tether=False)
    job.lx = job.ly = job.lz = box[0]
    job.vol = float(np.prod(box))
    _write_atomic_data(str(sim / "conf.equilibration.data"), pos + 0.02, box)

    rec = _RecordingRunner(str(sim))
    monkeypatch.setattr(ph, "create_object", lambda calc, directory: rec)
    job.run_fcpot_integration(iteration=1)
    script = "\n".join(rec.commands)

    assert "plugin load      /opt/fcpotplugin.so" in script
    assert "atom_modify      map array" in script
    # the real potential IS scaled by lambda (regression: it must not be
    # a plain unscaled pair style)
    assert "pair_style       hybrid/scaled v_flambda eam/alloy" in script
    assert "pair_style eam/alloy\n" not in script
    assert "pair_coeff       * * eam/alloy" in script
    # reference via the fix, scaled by (1-lambda); the unscaled reference
    # energy is the fix vector element 1 (the scalar is the scaled energy)
    assert "fix              ffc all fcpot" in script
    assert "v_blambda" in script
    assert "variable         dU2 equal f_ffc[1]/atoms" in script
    # single leg
    assert "forward_1.dat" in script
    assert "backward_1.dat" in script
    assert "forward_leg1_1.dat" not in script
    assert "variable         flambda equal ramp(${li},${lf})" in script
    assert "variable         flambda equal ramp(${lf},${li})" in script
    assert "zero yes" in script


def test_qtb_forces_quantum_reference(tmp_path):
    calc = _base_calc()
    calc["mode"] = "fe-qtb"
    calc["harmonic_reference"] = {"enabled": True}
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)
    assert opts._qtb is True
    assert opts.harmonic_reference.quantum is True


# ----------------------------------------------------------------------
# solid-phase integration surface (command generation, no LAMMPS run)
# ----------------------------------------------------------------------


def _write_atomic_data(filename, positions, box, mass=63.546):
    with open(filename, "w") as f:
        f.write("conf\n\n%d atoms\n1 atom types\n" % len(positions))
        for ax, name in enumerate("xyz"):
            f.write("0.0 %.10f %slo %shi\n" % (box[ax], name, name))
        f.write("\nMasses\n\n1 %f\n\nAtoms # atomic\n\n" % mass)
        for i, p in enumerate(positions):
            f.write("%d 1 %.10f %.10f %.10f\n" % (i + 1, p[0], p[1], p[2]))


class _RecordingRunner:
    """Minimal stand-in for the LAMMPS runner returned by
    ``helpers.create_object``: records every ``command(str)`` and treats any
    other lifecycle call (``close``/``rotate_logs``/``sync`` ...) as a no-op,
    so the integration command stream can be inspected without a LAMMPS run.
    """

    def __init__(self, directory="."):
        self.directory = directory
        self.commands = []

    def command(self, s):
        self.commands.append(s)

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_harmonic_integration_command_stream(tmp_path, monkeypatch):
    from calphy.solid import Solid
    import calphy.helpers as ph

    calc = _base_calc()
    calc["lattice"] = os.path.join(
        os.path.dirname(__file__), "conf1.data"
    )
    calc["harmonic_reference"] = {"enabled": True, "cutoff": 4.2}
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)

    sim = tmp_path / "sim"
    sim.mkdir()
    job = Solid(calculation=opts, simfolder=str(sim))

    pos, box = make_fcc(a=3.615)
    model = HarmonicModel(
        pos, box, np.ones(len(pos), dtype=int), [63.546], cutoff=4.2
    )
    model.set_spring_constants([1.0, 0.5])
    model.set_tether(0.25)
    job.harmonic_model = model
    job.harmonic_einstein_k = model.einstein_anchor_constants(300.0)
    job.lx = job.ly = job.lz = box[0]
    job.vol = float(np.prod(box))

    # thermal configuration file expected by the integration stage
    _write_atomic_data(
        str(sim / "conf.equilibration.data"), pos + 0.02, box
    )

    rec = _RecordingRunner(str(sim))
    monkeypatch.setattr(ph, "create_object", lambda calc, directory: rec)
    job.run_harmonic_integration(iteration=1)
    script = "\n".join(rec.commands)

    # topology data file written with ghosts and bonds
    assert os.path.exists(str(sim / "conf.harmonic.data"))
    # bond-based reference, both sub-styles, adapt scaling
    assert "atom_style       bond" in script
    assert "special_bonds    lj/coul 1.0 1.0 1.0" in script
    assert "bond_style       hybrid harmonic class2" in script
    assert "fix              fad all adapt 1 bond harmonic k 1*3 v_refscale" in script
    assert "bond class2 k2 4*4 v_einscale scale yes reset yes" in script
    assert "compute          cb all bond" in script
    # real potential scaled via hybrid/scaled with a zero substyle for
    # the ghost type coverage
    assert "hybrid/scaled v_flambda eam/alloy 1.0 zero" in script
    assert "Cu NULL" in script
    assert "pair_coeff       * * zero" in script
    # two-leg output files
    assert "forward_leg1_1.dat" in script
    assert "backward_leg1_1.dat" in script
    assert "forward_leg2_1.dat" in script
    assert "backward_leg2_1.dat" in script
    # stage variables
    assert "variable         flambda equal ramp(${li},${lf})" in script
    assert "variable         mu equal ramp(${li},${lf})" in script
    assert "variable         refscale equal v_mu" in script
    assert "variable         einscale equal 1.0-v_mu" in script
    # per-real-atom normalisation, exact unscaled recovery, COM thermostat
    assert "group            real type 1" in script
    assert "c_cb[1]/(v_refscale+1.0e-12)/v_nreal" in script
    assert "c_cb[2]/(v_einscale+1.0e-12)/v_nreal" in script
    assert "zero yes" in script


def test_build_harmonic_model_from_dumps(tmp_path):
    """End-to-end fitting from LAMMPS-format dump files."""
    from calphy.solid import Solid

    calc = _base_calc()
    calc["lattice"] = os.path.join(os.path.dirname(__file__), "conf1.data")
    calc["mass"] = 26.98
    calc["element"] = "Al"
    calc["pair_coeff"] = "* * tests/Cu01.eam.alloy Cu"
    calc["harmonic_reference"] = {"enabled": True, "cutoff": 4.2, "n_snapshots": 6}
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)

    sim = tmp_path / "sim"
    sim.mkdir()
    job = Solid(calculation=opts, simfolder=str(sim))
    job.vol = None

    # write synthetic dump files from a known spring model
    pos, box = make_fcc()
    n = len(pos)
    types = np.ones(n, dtype=int)
    truth = HarmonicModel(pos, box, types, [26.98], cutoff=4.2)
    truth.set_spring_constants([1.2, 0.4])

    def write_dump(filename, positions, forces):
        with open(filename, "w") as f:
            f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n" % n)
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for b in box:
                f.write("0.0 %f\n" % b)
            f.write("ITEM: ATOMS id type x y z fx fy fz\n")
            for i in range(n):
                f.write(
                    "%d 1 %f %f %f %f %f %f\n"
                    % (i + 1, *positions[i], *forces[i])
                )

    write_dump(str(sim / "harmonic.reference.dump"), pos, np.zeros_like(pos))
    rng = np.random.default_rng(3)
    for s in range(6):
        p = pos + rng.uniform(-0.05, 0.05, size=pos.shape)
        write_dump(str(sim / ("harmonic.snapshot_%d.dump" % s)), p, truth.forces(p))

    job.build_harmonic_model()

    assert np.allclose(job.harmonic_model.k_groups, [1.2, 0.4], atol=1e-8)
    assert os.path.exists(str(sim / "harmonic.model.npz"))
    assert os.path.exists(str(sim / "harmonic.frequencies.dat"))
    assert job.harmonic_model.tether_k == pytest.approx(0.5 * 1.2)
    assert np.all(job.harmonic_einstein_k > 0)
    assert (
        job.harmonic_fe_quantum["f_modes"] > job.harmonic_fe_classical["f_modes"]
    )

    # thermodynamic assembly: fabricate two-leg switching output
    job.lx = job.ly = job.lz = box[0]
    job.vol = float(np.prod(box))
    lam = np.linspace(1, 0, 100)
    for prefix, val in [("leg1", 0.5), ("leg2", 0.2)]:
        du = np.full_like(lam, val)  # constant integrand
        np.savetxt(
            str(sim / ("forward_%s_1.dat" % prefix)),
            np.column_stack((du, np.zeros_like(du), lam)),
        )
        np.savetxt(
            str(sim / ("backward_%s_1.dat" % prefix)),
            np.column_stack((du, np.zeros_like(du), lam[::-1])),
        )
    job.thermodynamic_integration()
    # per leg: w = 0.5*(fw-bw) = -val -> total w = -0.7
    assert np.isclose(job.w, -0.7, atol=1e-8)
    assert np.isclose(job.w_leg1, -0.5, atol=1e-8)
    assert np.isclose(job.w_leg2, -0.2, atol=1e-8)

    # the analytic anchor is the Einstein crystal at the matched
    # constants (calphy curvature convention = 2 K_E)
    from calphy.integrators import get_einstein_crystal_fe

    fe_e, fcm_e = get_einstein_crystal_fe(
        job.calc,
        job.vol,
        [2.0 * k for k in job.harmonic_einstein_k],
        return_contributions=True,
    )
    assert job.fref == pytest.approx(fe_e + fcm_e)
    assert np.isclose(job.fe, job.fref + 0.7, atol=1e-8)


# ----------------------------------------------------------------------
# TDEP effective force constants (from equilibrium MD)
# ----------------------------------------------------------------------


def test_tdep_reference_mean_and_base_force():
    """Mean positions average out thermal noise and a rigid offset; the
    mean residual force is the frame-averaged force."""
    pos, box = make_fcc(a=3.615)
    rng = np.random.default_rng(3)
    offset = np.array([0.13, -0.07, 0.05])  # e.g. thermal drift of the sites
    frames, fframes = [], []
    for _ in range(200):
        p = pos + offset + rng.normal(0, 0.05, size=pos.shape)
        frames.append(np.mod(p, box))  # wrap into the box (PBC)
        fframes.append(rng.normal(0, 0.01, size=pos.shape))
    mean_pos, mean_force = tdep_reference(frames, fframes, box)
    # mean recovered up to a whole-box image
    d = minimum_image(mean_pos - (pos + offset), box)
    assert np.abs(d).max() < 0.02
    assert np.allclose(mean_force, np.mean(fframes, axis=0))


def test_tdep_reference_requires_frames():
    with pytest.raises(ValueError, match="at least one frame"):
        tdep_reference([], [], np.array([1.0, 1.0, 1.0]))


def test_tdep_recovers_force_constants_from_thermal_forces():
    """The TDEP path recovers the force constants from thermal-force data
    when fitted, symmetry-aware, on the relaxed (symmetric) sites -- which
    for a perfect crystal are the thermal mean -- with the mean MD force
    removed as the base force. Building the fit on the symmetric reference
    keeps hiphive symmetry-reduced (a few DOF); the noisy sample mean would
    instead collapse it to a full unsymmetrised P1 fit."""
    hiphive = pytest.importorskip("hiphive")
    from calphy.harmonic import fit_with_hiphive

    sites, box = make_fcc(a=3.615)  # perfect fcc: symmetric sites
    types = np.ones(len(sites), dtype=int)
    truth = HarmonicModel(sites, box, types, [63.546], cutoff=4.2)
    truth.set_spring_constants([1.3, 0.45])

    # thermal frames about the symmetric sites (the thermal mean sits on
    # the symmetric sites by symmetry); symmetry reduces the fit to a few
    # DOF, so a modest number of frames over-determines it
    rng = np.random.default_rng(11)
    frames, forces = [], []
    for _ in range(20):
        p = sites + rng.normal(0, 0.04, size=sites.shape)
        frames.append(np.mod(p, box))
        forces.append(truth.forces(p))

    mean_pos, _ = tdep_reference(frames, forces, box)
    # the thermal mean stays on the symmetric sites (drift is sampling noise)
    assert np.abs(minimum_image(mean_pos - sites, box)).max() < 0.05

    # fit on the SYMMETRIC reference (not the noisy mean); the forces are
    # referenced to those same symmetric sites, so no base force is needed
    fit = HarmonicModel(sites, box, types, [63.546], cutoff=4.2)
    fit_with_hiphive(fit, frames, forces)
    assert np.allclose(fit.k_groups, [1.3, 0.45], atol=0.02)
    assert hasattr(fit, "full_fc_blocks") and len(fit.full_fc_blocks) > 0


def test_tdep_input_schema(tmp_path):
    calc = _base_calc()
    calc["harmonic_reference"] = {
        "enabled": True,
        "fitting_backend": "tdep",
        "sampling_interval": 50,
        "n_snapshots": 30,
        "plugin_path": "/opt/fcpotplugin.so",
    }
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)
    hr = opts.harmonic_reference
    assert hr.fitting_backend == "tdep"
    assert hr.sampling_interval == 50


def test_tdep_sampling_command_stream(tmp_path):
    """The TDEP sampling stage continues the equilibrium MD and dumps
    positions+forces every sampling_interval steps -- no minimisation and
    no imposed displacement."""
    from calphy.solid import Solid

    calc = _base_calc()
    calc["lattice"] = os.path.join(os.path.dirname(__file__), "conf1.data")
    calc["harmonic_reference"] = {
        "enabled": True,
        "fitting_backend": "tdep",
        "n_snapshots": 5,
        "sampling_interval": 40,
        "plugin_path": "/opt/fcpotplugin.so",
    }
    fn = _write_yaml(tmp_path, {"calculations": [calc]})
    [opts] = read_inputfile(fn)

    sim = tmp_path / "sim"
    sim.mkdir()
    job = Solid(calculation=opts, simfolder=str(sim))
    rec = _RecordingRunner(str(sim))
    job._sample_tdep_snapshots(rec)
    script = "\n".join(rec.commands)

    assert "fix              3 all nvt temp" in script
    assert "run               40" in script
    assert "harmonic.snapshot_0.dump" in script
    assert "harmonic.snapshot_4.dump" in script
    # TDEP does not minimise or impose displacements
    assert "minimize" not in script
    assert "displace_atoms" not in script


def test_idealize_orthogonal_box():
    from calphy.harmonic import idealize_orthogonal_box

    pos = np.array([[5.0, 5.005, 4.995], [2.5, 2.5, 2.5]])
    # near-cubic -> all three snapped to the common mean, frac coords kept
    box = np.array([10.0, 10.01, 9.99])
    nb, npos = idealize_orthogonal_box(box, pos)
    assert np.allclose(nb, 10.0)
    assert np.allclose(npos / nb, pos / box)
    # near-tetragonal -> the two close axes snap, the distinct one is kept
    box2 = np.array([10.0, 10.005, 12.0])
    nb2, _ = idealize_orthogonal_box(box2, pos)
    assert np.isclose(nb2[0], nb2[1]) and np.isclose(nb2[2], 12.0)
    # genuine orthorhombic -> untouched (no false symmetry imposed)
    box3 = np.array([10.0, 11.0, 12.0])
    nb3, _ = idealize_orthogonal_box(box3, pos)
    assert np.allclose(nb3, box3)


def test_hiphive_robust_to_box_anisotropy():
    """A perfect fcc crystal in a slightly anisotropic box (as left by
    per-axis pressure convergence) must still fit -- the raw hiphive
    ClusterSpace trips on the near-degenerate metric, the idealisation
    fixes it."""
    pytest.importorskip("hiphive")
    from calphy.harmonic import fit_with_hiphive

    lattice, box = make_fcc(a=3.615)
    types = np.ones(len(lattice), dtype=int)
    box_p = box * np.array([1.0015, 0.9989, 1.0007])  # ~0.1% anisotropy
    ref = lattice * (box_p / box)  # exact fcc fractional coords
    truth = HarmonicModel(ref, box_p, types, [63.546], cutoff=4.2)
    truth.set_spring_constants([1.3, 0.45])
    rng = np.random.default_rng(0)
    frames = [ref + rng.normal(0, 0.05, ref.shape) for _ in range(20)]
    forces = [truth.forces(p) for p in frames]

    fit = HarmonicModel(ref, box_p, types, [63.546], cutoff=4.2)
    fit_with_hiphive(fit, frames, forces)  # must not raise
    assert np.allclose(fit.k_groups, [1.3, 0.45], atol=0.03)
    assert hasattr(fit, "full_fc_blocks") and len(fit.full_fc_blocks) > 0
