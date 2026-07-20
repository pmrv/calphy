"""
calphy: a Python library and command line interface for automated free
energy calculations.

Copyright 2021  (c) Sarath Menon^1, Yury Lysogorskiy^2, Ralf Drautz^2
^1: Max Planck Institut für Eisenforschung, Dusseldorf, Germany
^2: Ruhr-University Bochum, Bochum, Germany

calphy is published and distributed under the Academic Software License v1.0 (ASL).
calphy is distributed in the hope that it will be useful for non-commercial academic research,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
calphy API is published and distributed under the BSD 3-Clause "New" or "Revised" License
See the LICENSE FILE for more details.

Harmonic (phonon) crystal reference for solid Frenkel-Ladd calculations
-----------------------------------------------------------------------

This module implements a Born-von Karman *spring network* reference system:
pairwise harmonic springs between neighbouring lattice sites, with spring
constants fitted to displacement-force data of the real interatomic
potential (the same random-displacement fitting strategy used by force
constant packages such as hiphive).

Unlike the Einstein crystal (independent oscillators tethered to lattice
sites), the spring network is a *coupled* harmonic crystal with a genuine
phonon dispersion, so its energy landscape is much closer to that of the
real potential. This reduces the dissipation during nonequilibrium
Hamiltonian interpolation and shortens the switching times needed for a
given accuracy.

Its free energy is known exactly: the reference is a quadratic
Hamiltonian, so diagonalising its (mass-weighted) Hessian yields the
normal-mode frequencies and thereby the classical *or quantum* harmonic
free energy.

The reference used for switching is the exactly-quadratic force-constant
Hamiltonian ``E = 1/2 u^T Phi u`` in the atomic displacements from the
reference sites, evaluated in LAMMPS by the compiled ``fix fcpot`` plugin
(plugins/fcpot). Being globally quadratic it cannot fold or exchange
sites, so it needs no site tether and a single switching leg. The lambda
switching to the real potential reuses the same ``pair_style
hybrid/scaled`` machinery calphy already employs for the Uhlenbeck-Ford
liquid reference.

Free energy (per system), with k running over the 3N-3 finite modes:

classical:  F = kT * sum_k ln(hbar*omega_k / kT)  +  F_com
quantum:    F = sum_k [ hbar*omega_k / 2 + kT ln(1 - exp(-hbar*omega_k/kT)) ] + F_com

For the (untethered) force-constant reference the three translational
zero modes are dropped and F_com is the free-particle COM term
-kT ln[V (2 pi M kT/h^2)^(3/2)].

A weak site tether (``set_tether``) is still supported at the class level
for analysis: it anchors each atom to its reference site,

    U_ref = 1/2 u^T Phi u  +  k_t * sum_i |x_i - X_i|^2,

removing the zero modes so k runs over all 3N modes, with the
Frenkel-Smit COM correction

    F_com = -kT ln[ V (beta / (2 pi sum_i mu_i^2 / k_t))^(3/2) ]

(mu_i = m_i / M). It is not used by the fcpot switching path.
"""

import os

import numpy as np
import scipy.constants as const
from scipy.spatial import cKDTree

import yaml

# constants (units consistent with calphy.integrators)
kb = const.physical_constants["Boltzmann constant in eV/K"][0]  # eV/K
kbJ = const.physical_constants["Boltzmann constant"][0]  # J/K
hJ = const.physical_constants["Planck constant"][0]  # J s
hbarJ = hJ / (2 * np.pi)
Na = const.physical_constants["Avogadro constant"][0]
eV2J = const.eV
AMU = 1e-3 / Na  # kg

# conversion factor: omega^2 in eV/(A^2 amu) -> rad^2/s^2
OMEGA2_TO_SI = eV2J / (1e-20 * AMU)


def read_lammps_dump(filename):
    """
    Read a LAMMPS text dump written as
    ``dump ... custom ... id type x y z <extra fields>``.

    Parameters
    ----------
    filename : str
        path to the dump file (single snapshot; the first snapshot is read)

    Returns
    -------
    data : dict
        with keys ``ids`` (N,), ``types`` (N,), ``box`` (3,) box lengths,
        ``positions`` (N,3), and one (N,3) array per extra vector field
        found among [fx fy fz] -> ``forces`` and [vx vy vz] ->
        ``velocities``. Atoms are sorted by id.
    """
    with open(filename, "r") as fin:
        lines = fin.readlines()

    natoms = None
    box = []
    fields = None
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(lines[i + 1].split()[0])
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            if "xy" in line:
                raise ValueError(
                    "Triclinic boxes are not supported by the harmonic "
                    "(phonon) reference; use an orthogonal cell."
                )
            for j in range(3):
                lo, hi = [float(x) for x in lines[i + 1 + j].split()[:2]]
                box.append(hi - lo)
            i += 4
        elif line.startswith("ITEM: ATOMS"):
            fields = line.split()[2:]
            for j in range(natoms):
                rows.append([float(x) for x in lines[i + 1 + j].split()])
            i += 1 + natoms
        else:
            i += 1

    if natoms is None or fields is None:
        raise ValueError("Could not parse LAMMPS dump file %s" % filename)

    rows = np.array(rows)
    idx = {f: c for c, f in enumerate(fields)}
    for f in ["id", "type", "x", "y", "z"]:
        if f not in idx:
            raise ValueError("Field %s missing in dump file %s" % (f, filename))

    order = np.argsort(rows[:, idx["id"]].astype(int), kind="stable")
    rows = rows[order]

    data = {
        "ids": rows[:, idx["id"]].astype(int),
        "types": rows[:, idx["type"]].astype(int),
        "box": np.array(box),
        "positions": rows[:, [idx["x"], idx["y"], idx["z"]]],
    }
    if all(f in idx for f in ["fx", "fy", "fz"]):
        data["forces"] = rows[:, [idx["fx"], idx["fy"], idx["fz"]]]
    if all(f in idx for f in ["vx", "vy", "vz"]):
        data["velocities"] = rows[:, [idx["vx"], idx["vy"], idx["vz"]]]
    return data


def read_lammps_data(filename):
    """
    Minimal parser for an orthogonal-box LAMMPS data file with
    atom_style atomic (as written by write_data), returning atoms sorted
    by id.

    Returns
    -------
    data : dict with keys ``ids``, ``types``, ``box``, ``positions``
    """
    with open(filename, "r") as fin:
        lines = fin.readlines()

    natoms = None
    box = np.zeros(3)
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i].split("#")[0].strip()
        if line.endswith(" atoms"):
            natoms = int(line.split()[0])
        elif line.endswith("xlo xhi") or line.endswith("ylo yhi") or line.endswith("zlo zhi"):
            lo, hi = float(line.split()[0]), float(line.split()[1])
            ax = {"x": 0, "y": 1, "z": 2}[line.split()[2][0]]
            if abs(lo) > 1e-9:
                raise ValueError(
                    "Data file %s has a non-zero box origin; the harmonic "
                    "reference expects boxes remapped to the origin." % filename
                )
            box[ax] = hi - lo
        elif line.endswith("xy xz yz"):
            if any(abs(float(x)) > 1e-9 for x in line.split()[:3]):
                raise ValueError(
                    "Triclinic boxes are not supported by the harmonic reference."
                )
        elif line.startswith("Atoms"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and lines[j].strip():
                tok = lines[j].split("#")[0].split()
                if len(tok) < 5:
                    break
                rows.append([float(x) for x in tok[:5]])
                j += 1
            i = j
            continue
        i += 1

    if natoms is None or len(rows) != natoms:
        raise ValueError(
            "Could not parse data file %s (%s atoms found, %s declared)"
            % (filename, len(rows), natoms)
        )
    rows = np.array(rows)
    order = np.argsort(rows[:, 0].astype(int), kind="stable")
    rows = rows[order]
    return {
        "ids": rows[:, 0].astype(int),
        "types": rows[:, 1].astype(int),
        "box": box,
        "positions": rows[:, 2:5],
    }


def minimum_image(vectors, box):
    """
    Apply the minimum image convention to displacement vectors in an
    orthogonal periodic box.
    """
    return vectors - box * np.round(vectors / box)


def tdep_reference(positions, forces, box):
    """
    Mean atomic positions and mean force from a set of MD frames, for the
    temperature-dependent effective potential (TDEP) reference.

    The reference sites of the effective harmonic model are the thermal
    *mean* positions (not the 0 K minimum), so that the fitted force
    constants are the anharmonically renormalised effective ones at the
    sampled state point. Frames are aligned to the first frame with the
    minimum-image convention before averaging, so atoms that wrap across a
    periodic boundary between frames average correctly (valid as long as
    per-frame displacements stay below half the box, i.e. a solid with no
    diffusion).

    Parameters
    ----------
    positions : sequence of (N, 3) arrays
        atomic positions of each frame (sorted by id, consistent order)
    forces : sequence of (N, 3) arrays
        atomic forces of each frame, same order
    box : (3,) array
        orthogonal box lengths

    Returns
    -------
    mean_positions : (N, 3) array
        thermal mean positions, in the image of the first frame
    mean_force : (N, 3) array
        mean residual force (subtracted as the base force in the fit; it
        vanishes by symmetry for a perfect crystal but is removed
        explicitly so any residual does not bias the force constants)
    """
    positions = [np.asarray(p, dtype=float) for p in positions]
    forces = [np.asarray(f, dtype=float) for f in forces]
    if len(positions) == 0:
        raise ValueError("tdep_reference needs at least one frame")
    box = np.asarray(box, dtype=float)
    anchor = positions[0]
    disp_sum = np.zeros_like(anchor)
    for p in positions:
        disp_sum += minimum_image(p - anchor, box)
    mean_positions = anchor + disp_sum / len(positions)
    mean_force = np.mean(forces, axis=0)
    return mean_positions, mean_force


def find_neighbor_pairs(positions, box, cutoff):
    """
    Find all unique atom pairs within ``cutoff`` under periodic boundary
    conditions (orthogonal box).

    Returns
    -------
    pairs : (P, 2) int array of 0-based indices, i < j
    distances : (P,) float array
    """
    if np.any(2 * cutoff >= box):
        raise ValueError(
            "harmonic_reference.cutoff (%.2f A) must be smaller than half "
            "the smallest box length (%.2f A); use a larger supercell or a "
            "smaller cutoff." % (cutoff, np.min(box) / 2)
        )
    wrapped = np.mod(positions, box)
    # guard against numerical edge case where mod returns exactly box
    wrapped = np.where(wrapped >= box, wrapped - box, wrapped)
    tree = cKDTree(wrapped, boxsize=box)
    pairs = np.array(sorted(tree.query_pairs(r=cutoff)), dtype=int)
    if len(pairs) == 0:
        raise ValueError(
            "No neighbor pairs found within cutoff %.2f A; increase "
            "harmonic_reference.cutoff." % cutoff
        )
    dr = minimum_image(positions[pairs[:, 1]] - positions[pairs[:, 0]], box)
    distances = np.linalg.norm(dr, axis=1)
    return pairs, distances


def group_pairs(pairs, distances, types, tolerance=0.05):
    """
    Group pairs into "bond types" by (sorted element-type pair, neighbour
    shell distance). Distances within ``tolerance`` (A) of each other are
    merged into the same shell.

    Returns
    -------
    group_index : (P,) int array assigning each pair to a group
    groups : list of dict with keys type_i, type_j, distance (mean), count
    """
    ti = types[pairs[:, 0]]
    tj = types[pairs[:, 1]]
    tmin = np.minimum(ti, tj)
    tmax = np.maximum(ti, tj)

    group_index = np.full(len(pairs), -1, dtype=int)
    groups = []
    for a, b in sorted(set(zip(tmin, tmax))):
        mask = (tmin == a) & (tmax == b)
        sel = np.where(mask)[0]
        d = distances[sel]
        order = np.argsort(d)
        # cluster sorted distances by gaps larger than tolerance
        start = 0
        d_sorted = d[order]
        for k in range(1, len(d_sorted) + 1):
            if k == len(d_sorted) or (d_sorted[k] - d_sorted[k - 1]) > tolerance:
                members = sel[order[start:k]]
                group_index[members] = len(groups)
                groups.append(
                    {
                        "type_i": int(a),
                        "type_j": int(b),
                        "distance": float(np.mean(distances[members])),
                        "count": int(len(members)),
                    }
                )
                start = k
    return group_index, groups


def frequencies_from_hessian(H, masses, tethered, zero_mode_tolerance=1e-6):
    """
    Normal-mode angular frequencies (rad/s, ascending) of a Hessian H
    (eV/A^2) with per-atom masses (g/mol). For a translation-invariant
    Hamiltonian (``tethered=False``) exactly three zero modes are
    expected and dropped; a tethered one must be strictly positive
    definite.
    """
    masses = np.asarray(masses, dtype=float)
    invsqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    D = H * invsqrt_m[:, None] * invsqrt_m[None, :]
    evals = np.linalg.eigvalsh(D)

    scale = np.max(np.abs(evals))
    tol = zero_mode_tolerance * scale
    n_zero = int(np.sum(np.abs(evals) < tol))
    n_neg = int(np.sum(evals < -tol))

    if tethered:
        if n_neg > 0 or n_zero > 0:
            raise ValueError(
                "Tethered reference Hamiltonian has %d unstable and %d zero "
                "modes; it must be strictly positive definite. Increase the "
                "tether constant (set_tether) or the neighbour cutoff."
                % (n_neg, n_zero)
            )
        return np.sqrt(np.sort(evals) * OMEGA2_TO_SI)

    if n_neg > 0:
        raise ValueError(
            "Reference Hamiltonian has %d unstable (imaginary frequency) "
            "modes; it is not a valid crystal Hamiltonian. Try increasing "
            "harmonic_reference.cutoff or n_snapshots/displacement." % n_neg
        )
    if n_zero != 3:
        raise ValueError(
            "Reference Hamiltonian has %d zero modes; exactly 3 (rigid "
            "translations) are expected. Try increasing "
            "harmonic_reference.cutoff." % n_zero
        )
    return np.sqrt(np.sort(evals)[3:] * OMEGA2_TO_SI)


def hessian_from_blocks(natoms, blocks):
    """
    Dense (3N, 3N) Hessian from a list of (i, j, 3x3 block) entries with
    0-based indices and i <= j; the (j, i) block is the transpose.
    """
    H = np.zeros((3 * natoms, 3 * natoms))
    for i, j, B in blocks:
        B = np.asarray(B, dtype=float)
        H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += B
        if i != j:
            H[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] += B.T
    return H


def apply_acoustic_sum_rule(natoms, blocks):
    """
    Enforce the acoustic sum rule by resetting each diagonal block to
    minus the sum of the off-diagonal blocks of its row,
    Phi_ii = -sum_{j != i} Phi_ij. This guarantees exact translation
    invariance (three exact zero modes) of the written reference.
    Off-diagonal blocks are also symmetrised through their transpose
    counterparts already implied by the i <= j storage.

    Returns a new blocks list (i <= j, 0-based).
    """
    rowsum = [np.zeros((3, 3)) for _ in range(natoms)]
    offdiag = []
    for i, j, B in blocks:
        B = np.asarray(B, dtype=float)
        if i == j:
            continue
        offdiag.append((i, j, B))
        rowsum[i] += B
        rowsum[j] += B.T
    out = [(i, i, -rowsum[i]) for i in range(natoms)]
    out.extend(offdiag)
    return out


def write_fc_file(filename, ids, positions, blocks, comment=""):
    """
    Write a force-constant file for LAMMPS ``fix fcpot``:

        natoms
        tag x y z                       (reference sites)
        nblocks
        tag_i tag_j  p11 ... p33        (3x3 blocks, tag_i <= tag_j)

    ``blocks`` holds (i, j, 3x3) entries with 0-based indices into
    ``ids``/``positions`` and i <= j.
    """
    ids = np.asarray(ids)
    positions = np.asarray(positions, dtype=float)
    with open(filename, "w") as f:
        f.write("# calphy harmonic (phonon) reference force constants %s\n" % comment)
        f.write("%d\n" % len(ids))
        for p in range(len(ids)):
            f.write(
                "%d %.10f %.10f %.10f\n"
                % (ids[p], positions[p, 0], positions[p, 1], positions[p, 2])
            )
        f.write("%d\n" % len(blocks))
        for i, j, B in blocks:
            ti, tj = int(ids[i]), int(ids[j])
            if ti > tj:
                ti, tj = tj, ti
                B = np.asarray(B).T
            f.write(
                "%d %d %s\n"
                % (ti, tj, " ".join("%.12e" % v for v in np.asarray(B).flat))
            )


class HarmonicModel:
    """
    Pairwise harmonic spring-network model

        E = sum_p K_{g(p)} (r_p - r0_p)^2

    where each pair p belongs to a group g (bond type) sharing one spring
    constant, and r0_p is the exact pair distance in the reference
    (energy-minimised) configuration. The convention E = K (r-r0)^2 (the
    1/2 absorbed into K) matches LAMMPS ``pair_style list``/``bond_style
    harmonic``.
    """

    def __init__(
        self,
        reference_positions,
        box,
        types,
        masses,
        ids=None,
        cutoff=5.0,
        distance_tolerance=0.05,
    ):
        """
        Parameters
        ----------
        reference_positions : (N, 3) array
            equilibrium positions (A)
        box : (3,) array
            orthogonal box lengths (A)
        types : (N,) int array
            LAMMPS atom types (1-based)
        masses : list/array
            mass per type in g/mol; masses[t-1] is the mass of type t
        ids : (N,) int array, optional
            LAMMPS atom ids; defaults to 1..N
        cutoff : float
            neighbour cutoff for springs (A)
        distance_tolerance : float
            shell-grouping distance tolerance (A)
        """
        self.reference_positions = np.asarray(reference_positions, dtype=float)
        self.box = np.asarray(box, dtype=float)
        self.types = np.asarray(types, dtype=int)
        self.natoms = len(self.reference_positions)
        self.ids = (
            np.arange(1, self.natoms + 1) if ids is None else np.asarray(ids, dtype=int)
        )
        self.masses_per_type = np.asarray(masses, dtype=float)
        self.masses = self.masses_per_type[self.types - 1]
        self.cutoff = float(cutoff)
        self.distance_tolerance = float(distance_tolerance)

        self.pairs, self.r0 = find_neighbor_pairs(
            self.reference_positions, self.box, self.cutoff
        )
        self.group_index, self.groups = group_pairs(
            self.pairs, self.r0, self.types, tolerance=self.distance_tolerance
        )
        self.n_groups = len(self.groups)
        # quantise r0 per group: LAMMPS bond types carry one r0 per type,
        # so the model uses the group value consistently everywhere. In a
        # crystal the within-shell spread is numerical noise.
        group_r0 = np.array([g["distance"] for g in self.groups])
        self.r0 = group_r0[self.group_index]
        self.k_groups = None
        # site-tether constant (eV/A^2); set via set_tether(). None = pure
        # network (translation invariant, 3 zero modes).
        self.tether_k = None

        # unit bond vectors in the reference configuration (normalised
        # with the actual distances, not the group-quantised r0)
        dr = minimum_image(
            self.reference_positions[self.pairs[:, 1]]
            - self.reference_positions[self.pairs[:, 0]],
            self.box,
        )
        self._ref_unit = dr / np.linalg.norm(dr, axis=1)[:, None]

    # ------------------------------------------------------------------
    # model evaluation (used for fitting and for tests)
    # ------------------------------------------------------------------

    @property
    def k_pairs(self):
        """Spring constant per pair (eV/A^2)."""
        if self.k_groups is None:
            raise RuntimeError("Model has not been fitted yet")
        return self.k_groups[self.group_index]

    def energy(self, positions):
        """Model energy (eV) for a configuration."""
        dr = minimum_image(
            positions[self.pairs[:, 1]] - positions[self.pairs[:, 0]], self.box
        )
        r = np.linalg.norm(dr, axis=1)
        return float(np.sum(self.k_pairs * (r - self.r0) ** 2))

    def forces_per_group(self, positions):
        """
        Design tensor A with shape (N, 3, G) such that the model forces
        are ``A @ k_groups`` — forces are linear in the spring constants.

        For pair p = (i, j) with E_p = K (r - r0)^2:
        F_i = +2K (r - r0) * unit_ij on atom i, and -that on atom j,
        where unit_ij points from i to j.
        """
        dr = minimum_image(
            positions[self.pairs[:, 1]] - positions[self.pairs[:, 0]], self.box
        )
        r = np.linalg.norm(dr, axis=1)
        unit = dr / r[:, None]
        contrib = (2.0 * (r - self.r0))[:, None] * unit  # (P, 3)

        A = np.zeros((self.natoms, 3, self.n_groups))
        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        for g in range(self.n_groups):
            mask = self.group_index == g
            np.add.at(A[:, :, g], i[mask], contrib[mask])
            np.add.at(A[:, :, g], j[mask], -contrib[mask])
        return A

    def forces(self, positions):
        """Model forces (eV/A) for a configuration."""
        if self.k_groups is None:
            raise RuntimeError("Model has not been fitted yet")
        return self.forces_per_group(np.asarray(positions, dtype=float)) @ self.k_groups

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, displaced_positions, forces, base_forces=None):
        """
        Least-squares fit of the group spring constants to
        displacement-force data of the real potential.

        Parameters
        ----------
        displaced_positions : list of (N, 3) arrays
            atomic positions of the displaced snapshots
        forces : list of (N, 3) arrays
            real-potential forces for each snapshot (eV/A)
        base_forces : (N, 3) array, optional
            residual forces of the reference configuration (subtracted
            from each snapshot to correct for imperfect minimisation)

        Returns
        -------
        rmse : float
            root mean square force residual of the fit (eV/A)
        """
        A_rows = []
        b_rows = []
        for pos, f in zip(displaced_positions, forces):
            A = self.forces_per_group(np.asarray(pos, dtype=float))
            b = np.asarray(f, dtype=float)
            if base_forces is not None:
                b = b - base_forces
            A_rows.append(A.reshape(-1, self.n_groups))
            b_rows.append(b.reshape(-1))
        A = np.concatenate(A_rows, axis=0)
        b = np.concatenate(b_rows, axis=0)

        k, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.k_groups = k
        residual = A @ k - b
        rmse = float(np.sqrt(np.mean(residual**2)))
        self.fit_rmse = rmse
        self.fit_force_norm = float(np.sqrt(np.mean(b**2)))
        return rmse

    def set_spring_constants(self, k_groups):
        """Directly set group spring constants (eV/A^2)."""
        k_groups = np.asarray(k_groups, dtype=float)
        if not len(k_groups) == self.n_groups:
            raise ValueError(
                "expected %d spring constants, got %d"
                % (self.n_groups, len(k_groups))
            )
        self.k_groups = k_groups

    def set_tether(self, k_t):
        """
        Set the site-tether constant k_t (eV/A^2): every atom is anchored
        to its reference site by E = k_t |x - X|^2 as part of the
        reference Hamiltonian.
        """
        if k_t is not None and k_t <= 0:
            raise ValueError("tether constant must be positive")
        self.tether_k = None if k_t is None else float(k_t)

    # ------------------------------------------------------------------
    # phonons and free energy
    # ------------------------------------------------------------------

    def hessian(self):
        """
        Analytic Hessian of the spring network at the reference
        configuration, shape (3N, 3N), units eV/A^2.

        For an unstretched spring (r = r0) the pair Hessian block is
        purely longitudinal: d2E/du_i du_j = -2K rhat rhat^T. The site
        tether k_t |x - X|^2 adds 2 k_t to every diagonal element.
        """
        n = self.natoms
        H = np.zeros((3 * n, 3 * n))
        k = self.k_pairs
        for p in range(len(self.pairs)):
            i, j = self.pairs[p]
            block = 2.0 * k[p] * np.outer(self._ref_unit[p], self._ref_unit[p])
            H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= block
            H[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] -= block
            H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += block
            H[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += block
        if self.tether_k is not None:
            H[np.arange(3 * n), np.arange(3 * n)] += 2.0 * self.tether_k
        return H

    def frequencies(self, zero_mode_tolerance=1e-6):
        """
        Normal-mode angular frequencies of the network.

        Returns
        -------
        omega : array
            angular frequencies in rad/s, ascending. All 3N modes for a
            tethered model; the 3N-3 finite modes for an untethered one
            (the three rigid translations are dropped).

        Raises
        ------
        ValueError
            if the reference Hamiltonian is not positive (semi-)definite
            as required (unstable modes, or extra zero modes in the
            untethered case).
        """
        H = self.hessian()
        invsqrt_m = np.repeat(1.0 / np.sqrt(self.masses), 3)
        D = H * invsqrt_m[:, None] * invsqrt_m[None, :]
        evals = np.linalg.eigvalsh(D)  # eV/(A^2 amu)

        # tolerance relative to the largest eigenvalue
        scale = np.max(np.abs(evals))
        tol = zero_mode_tolerance * scale
        n_zero = int(np.sum(np.abs(evals) < tol))
        n_neg = int(np.sum(evals < -tol))

        if self.tether_k is not None:
            if n_neg > 0 or n_zero > 0:
                raise ValueError(
                    "Tethered spring network has %d unstable and %d zero "
                    "modes; the reference must be strictly positive "
                    "definite. Increase the tether constant (set_tether) "
                    "or the neighbour cutoff." % (n_neg, n_zero)
                )
            return np.sqrt(np.sort(evals) * OMEGA2_TO_SI)

        if n_neg > 0:
            raise ValueError(
                "Fitted spring network has %d unstable (imaginary frequency) "
                "modes. The harmonic reference is not a valid crystal "
                "Hamiltonian; try increasing harmonic_reference.cutoff to "
                "include more neighbour shells, or increase n_snapshots / "
                "displacement." % n_neg
            )
        if n_zero != 3:
            raise ValueError(
                "Fitted spring network has %d zero modes; exactly 3 (rigid "
                "translations) are expected. Try increasing "
                "harmonic_reference.cutoff so more neighbour shells "
                "stabilise the lattice." % n_zero
            )
        finite = np.sort(evals)[3:]
        return np.sqrt(finite * OMEGA2_TO_SI)

    def free_energy(self, temperature, volume=None, quantum=False, omega=None):
        """
        Harmonic free energy of the spring network, per atom, in eV.

        Parameters
        ----------
        temperature : float
            temperature in K
        volume : float, optional
            box volume in A^3 for the centre-of-mass (zero mode) term;
            defaults to the model box volume
        quantum : bool
            if True, use the quantum harmonic-oscillator free energy
            (zero-point energy + Bose-Einstein term); otherwise classical
        omega : array, optional
            precomputed angular frequencies (rad/s); computed if None

        Returns
        -------
        result : dict
            keys ``f_modes`` (vibrational part), ``f_com`` (zero-mode /
            centre-of-mass part), ``f_total``, and ``omega`` (rad/s)
        """
        if omega is None:
            omega = self.frequencies()
        if volume is None:
            volume = float(np.prod(self.box))

        beta = 1.0 / (kbJ * temperature)
        x = hbarJ * omega * beta

        if quantum:
            f_modes_J = np.sum(0.5 * hbarJ * omega + kbJ * temperature * np.log1p(-np.exp(-x)))
        else:
            f_modes_J = kbJ * temperature * np.sum(np.log(x))
        f_modes = f_modes_J / eV2J / self.natoms

        V = volume * 1e-30  # m^3
        if self.tether_k is not None:
            # COM-constrained switching correction, Frenkel-Smit form as
            # in the Einstein path (https://doi.org/10.1063/5.0044833).
            # The COM marginal of the tethered network is identical to
            # that of an Einstein crystal with per-atom constant k_t,
            # because the network part is translation invariant.
            mass_kg = self.masses * AMU
            mu = mass_kg / np.sum(mass_kg)
            # the Frenkel-Smit formula is written for E = (k/2) r^2, so k
            # is the curvature; our tether E = k_t r^2 has curvature 2 k_t
            k_t = 2.0 * self.tether_k * (eV2J / 1e-20)  # J/m^2
            mu2_over_k_sum = np.sum(mu**2) / k_t
            F_cm_J = kbJ * temperature * np.log(
                V * (beta / (2 * np.pi * mu2_over_k_sum)) ** 1.5
            )
            f_com = -F_cm_J / eV2J / self.natoms
        else:
            # centre of mass of the translation-invariant network:
            # free particle of total mass M in volume V
            M = np.sum(self.masses) * AMU  # kg
            f_com_J = -kbJ * temperature * np.log(
                V * (2 * np.pi * M * kbJ * temperature / hJ**2) ** 1.5
            )
            f_com = f_com_J / eV2J / self.natoms

        return {
            "f_modes": float(f_modes),
            "f_com": float(f_com),
            "f_total": float(f_modes + f_com),
            "omega": omega,
        }

    # ------------------------------------------------------------------
    # LAMMPS output
    # ------------------------------------------------------------------

    def fc_blocks(self, include_tether=False):
        """
        Sparse force-constant blocks of the network Hamiltonian's
        harmonic expansion, as (i, j, 3x3) with 0-based indices and
        i <= j, matching :meth:`hessian`. Used for the exactly-quadratic
        ``fix fcpot`` reference, which by default carries no tether (a
        globally quadratic Hamiltonian cannot fold).
        """
        k = self.k_pairs
        diag = [np.zeros((3, 3)) for _ in range(self.natoms)]
        blocks = []
        for p in range(len(self.pairs)):
            i, j = int(self.pairs[p, 0]), int(self.pairs[p, 1])
            B = 2.0 * k[p] * np.outer(self._ref_unit[p], self._ref_unit[p])
            blocks.append((min(i, j), max(i, j), -B))
            diag[i] += B
            diag[j] += B
        if include_tether and self.tether_k is not None:
            for i in range(self.natoms):
                diag[i] += 2.0 * self.tether_k * np.eye(3)
        return [(i, i, diag[i]) for i in range(self.natoms)] + blocks

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def save(self, filename):
        np.savez_compressed(
            filename,
            reference_positions=self.reference_positions,
            box=self.box,
            types=self.types,
            ids=self.ids,
            masses_per_type=self.masses_per_type,
            cutoff=self.cutoff,
            distance_tolerance=self.distance_tolerance,
            k_groups=self.k_groups if self.k_groups is not None else np.array([]),
            tether_k=np.nan if self.tether_k is None else self.tether_k,
        )

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        model = cls(
            reference_positions=data["reference_positions"],
            box=data["box"],
            types=data["types"],
            masses=data["masses_per_type"],
            ids=data["ids"],
            cutoff=float(data["cutoff"]),
            distance_tolerance=float(data["distance_tolerance"]),
        )
        if len(data["k_groups"]) > 0:
            model.set_spring_constants(data["k_groups"])
        if "tether_k" in data and np.isfinite(float(data["tether_k"])):
            model.set_tether(float(data["tether_k"]))
        return model


# ----------------------------------------------------------------------
# optional hiphive backend
# ----------------------------------------------------------------------


def fit_with_hiphive(model, displaced_positions, forces, base_forces=None, logger=None):
    """
    Fit the spring network via hiphive: fit full second-order force
    constants to the displacement-force data, then project each pair
    force-constant block onto the bond direction,

        k_pair = -rhat^T Phi_ij rhat / 2

    (the factor 1/2 converts to the E = K (r-r0)^2 convention), and
    average within each bond-type group.

    The resulting model is still the exactly-solvable spring network:
    hiphive only provides a regularised estimate of the spring constants.

    Requires the optional packages ``hiphive`` (and its fitting backend
    ``trainstation`` for recent versions).
    """
    try:
        from ase import Atoms
        from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
    except ImportError as e:
        raise ImportError(
            "harmonic_reference.fitting_backend=hiphive requires the "
            "'hiphive' package (pip install hiphive). Original error: %s" % e
        )
    try:
        from hiphive import Optimizer
    except ImportError:
        try:
            from trainstation import Optimizer
        except ImportError as e:
            raise ImportError(
                "hiphive fitting requires the 'trainstation' package "
                "(pip install trainstation). Original error: %s" % e
            )

    cell = np.diag(model.box)
    ref = Atoms(
        numbers=model.types,  # chemical identity only needs to be distinct per type
        positions=model.reference_positions,
        cell=cell,
        pbc=True,
    )

    cs = ClusterSpace(ref, [model.cutoff])
    sc = StructureContainer(cs)
    for pos, f in zip(displaced_positions, forces):
        disp = minimum_image(
            np.asarray(pos, dtype=float) - model.reference_positions, model.box
        )
        atoms = ref.copy()
        atoms.new_array("displacements", disp)
        fval = np.asarray(f, dtype=float)
        if base_forces is not None:
            fval = fval - base_forces
        atoms.new_array("forces", fval)
        sc.add_structure(atoms)

    opt = Optimizer(sc.get_fit_data(), train_size=1.0)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    fcs = fcp.get_force_constants(ref)

    # per-pair second-order force constant blocks; hiphive's ForceConstants
    # supports cluster indexing, with a dense-array fallback for API drift
    fc_array = None
    try:
        np.array(fcs[(int(model.pairs[0][0]), int(model.pairs[0][1]))])
    except Exception:
        fc_array = fcs.get_fc_array(order=2)

    k_pairs = np.zeros(len(model.pairs))
    for p, (i, j) in enumerate(model.pairs):
        if fc_array is None:
            phi = np.array(fcs[(int(i), int(j))])
        else:
            phi = fc_array[i, j]
        rhat = model._ref_unit[p]
        k_pairs[p] = -0.5 * rhat @ phi @ rhat

    # reduce to group values
    k_groups = np.zeros(model.n_groups)
    for g in range(model.n_groups):
        mask = model.group_index == g
        k_groups[g] = np.mean(k_pairs[mask])
    model.set_spring_constants(k_groups)

    if logger is not None:
        logger.info("hiphive FC2 fit summary: %s" % str(opt))

    # keep the full second-order blocks for the fcpot (force-constant
    # potential) implementation: (i, j, 3x3) with i <= j over the model's
    # neighbour pairs, ASR-consistent diagonals applied by the caller
    full = []
    for p, (i, j) in enumerate(model.pairs):
        if fc_array is None:
            phi = np.array(fcs[(int(i), int(j))])
        else:
            phi = fc_array[i, j]
        full.append((int(min(i, j)), int(max(i, j)), np.asarray(phi, dtype=float)))
    model.full_fc_blocks = apply_acoustic_sum_rule(model.natoms, full)
    return model
