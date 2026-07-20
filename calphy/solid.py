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

More information about the program can be found in:
Menon, Sarath, Yury Lysogorskiy, Jutta Rogal, and Ralf Drautz.
“Automated Free Energy Calculation from Atomistic Simulations.” Physical Review Materials 5(10), 2021
DOI: 10.1103/PhysRevMaterials.5.103801

For more information contact:
sarath.menon@ruhr-uni-bochum.de/yury.lysogorskiy@icams.rub.de
"""

import numpy as np
import yaml
import copy
import sys
import os

from calphy.integrators import *
import calphy.helpers as ph
import calphy.phase as cph
from calphy.errors import *
from calphy.harmonic import (
    HarmonicModel,
    read_lammps_dump,
    read_lammps_data,
    fit_with_hiphive,
)


class Solid(cph.Phase):
    """
    Class for free energy calculation with solid as the reference state

    Parameters
    ----------
    options : dict
        dict of input options

    kernel : int
        the index of the calculation that should be run from
        the list of calculations in the input file

    simfolder : string
        base folder for running calculations

    """

    def __init__(self, calculation=None, simfolder=None, log_to_screen=False):

        # call base class
        super().__init__(
            calculation=calculation,
            simfolder=simfolder,
            log_to_screen=log_to_screen,
        )

    @property
    def _use_harmonic_reference(self):
        hr = self.calc.harmonic_reference
        return hr is not None and hr.enabled

    def run_spring_constant_convergence(self, lmp):
        """ """
        if self.calc._qtb:
            qtb = self.calc.quantum_thermal_bath
            lmp.command("fix              3 all nve")
            lmp.command(
                "fix              3q all qtb temp %f damp %f seed %d f_max %f N_f %d"
                % (
                    self.calc._temperature,
                    qtb.thermostat_damping,
                    np.random.randint(1, 10**8),
                    qtb.f_max,
                    qtb.n_f,
                )
            )
        else:
            lmp.command(
                "fix              3 all nvt temp %f %f %f"
                % (
                    self.calc._temperature,
                    self.calc._temperature,
                    self.calc.md.thermostat_damping[1],
                )
            )

        # apply fix
        lmp = ph.compute_msd(lmp, self.calc)

        if ph.check_if_any_is_none(self.calc.spring_constants):
            # similar averaging routine
            laststd = 0.00
            for i in range(self.calc.md.n_cycles):
                lmp.command("run              %d" % int(self.calc.md.n_small_steps))
                lmp.sync()  # flush before analyse_spring_constants reads msd.dat
                k_mean, k_std = self.analyse_spring_constants(lmp)
                self.logger.info(
                    "At count %d mean k is %f std is %f" % (i + 1, k_mean[0], k_std[0])
                )
                if np.abs(laststd - k_std[0]) < self.calc.tolerance.spring_constant:
                    # now reevaluate spring constants
                    self.assign_spring_constants(k_mean)
                    break
                laststd = k_std[0]

        else:
            if not (len(self.calc.spring_constants) == self.calc.n_elements):
                raise ValueError(
                    "Spring constant input length should be same as number of elements, spring constant length %d, # elements %d"
                    % (len(self.calc.spring_constants), self.calc.n_elements)
                )

            # still run a small NVT cycle
            lmp.command("run              %d" % int(self.calc.md.n_small_steps))
            self.k = self.calc.spring_constants
            self.logger.info("Used user input sprint constants")
            self.logger.info(self.k)

        if self.calc._qtb:
            lmp.command("unfix         3q")
        lmp.command("unfix         3")

    def analyse_spring_constants(self, lmp):
        """
        Analyse spring constant routine
        """
        ncount = int(self.calc.n_equilibration_steps) // int(
            self.calc.md.n_every_steps * self.calc.md.n_repeat_steps
        )

        # now we can check if it converted; read cumulative msd.dat across segments
        msd = lmp.read_timeseries("msd.dat")
        k_mean = []
        k_std = []
        for i in range(self.calc.n_elements):
            quant = msd[:, i + 1][-ncount + 1 :]
            mean_quant = np.round(np.mean(quant), decimals=2)
            std_quant = np.round(np.std(quant), decimals=2)
            if mean_quant == 0:
                self.logger.warning(
                    "MSD for element index %d averaged to ~0; spring-constant "
                    "estimation is unreliable. Consider increasing equilibration "
                    "time or providing spring_constants explicitly." % i
                )
                mean_quant = 1.00
            mean_quant = 3 * kb * self.calc._temperature / mean_quant
            k_mean.append(mean_quant)
            k_std.append(std_quant)
        return k_mean, k_std

    def assign_spring_constants(self, k):
        """
        Here the spring constants are finalised, add added to the class
        """
        # first replace any provided values with user values
        if ph.check_if_any_is_not_none(self.calc.spring_constants):
            spring_constants = copy.copy(self.calc.spring_constants)
            k = ph.replace_nones(spring_constants, k, logger=self.logger)

        # add sanity checks
        k = ph.validate_spring_constants(k, logger=self.logger)

        # now save
        self.k = k

        self.logger.info("finalized sprint constants")
        self.logger.info(self.k)

    def run_averaging(self):
        """
        Run averaging routine

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Run averaging routine using LAMMPS. Starting from the initial lattice two different routines can
        be followed:
        If pressure is specified, MD simulations are run until the pressure converges within the given
        threshold value.
        If `fix_lattice` option is True, then the input structure is used as it is and the corresponding pressure
        is calculated.
        At the end of the run, the averaged box dimensions are calculated.
        """

        lmp = ph.create_object(self.calc, self.simfolder)

        # set up potential
        lmp = ph.set_pair_style(lmp, self.calc)

        # set up structure
        lmp = ph.create_structure(lmp, self.calc)

        lmp = ph.set_pair_coeff(lmp, self.calc)
        lmp = ph.set_mass(lmp, self.calc)

        # add some computes
        lmp.command("variable         mvol equal vol")
        lmp.command("variable         mlx equal lx")
        lmp.command("variable         mly equal ly")
        lmp.command("variable         mlz equal lz")
        lmp.command("variable         mpress equal press")
        lmp.command("variable         mpe equal pe/atoms")
        lmp.command("variable         metotal equal etotal/atoms")
        lmp.command("variable         mtemp equal temp")

        # Run if a constrained lattice is not needed
        if not self.calc._fix_lattice:
            if self.calc._pressure == 0:
                self.run_zero_pressure_equilibration(lmp)
            else:
                self.run_finite_pressure_equilibration(lmp)

            # this is when the averaging routine starts
            self.run_pressure_convergence(lmp)

            # dump snapshot and check if melted
            self.dump_current_snapshot(lmp, "traj.equilibration_stage1.dat")
            self.check_if_melted(lmp, "traj.equilibration_stage1.dat")

        # run if a constrained lattice is used
        else:
            # routine in which lattice constant will not varied, but is set to a given fixed value
            self.run_constrained_pressure_convergence(lmp)

        if self._use_harmonic_reference:
            # construct and fit the harmonic (phonon) spring-network
            # reference instead of the Einstein-crystal MSD routine
            self.run_harmonic_reference_construction(lmp)
        else:
            # start MSD calculation routine
            # there two possibilities here - if spring constants are provided, use it. If not, calculate it
            self.run_spring_constant_convergence(lmp)

        # check for melting
        self.dump_current_snapshot(lmp, "traj.equilibration_stage2.dat")
        self.check_if_melted(lmp, "traj.equilibration_stage2.dat")
        lmp = ph.write_data(lmp, "conf.equilibration.data")
        # close object and process traj
        self.lammps_close(lmp=lmp)
        lmp.rotate_logs("averaging")

    # ------------------------------------------------------------------
    # harmonic (phonon) spring-network reference
    # ------------------------------------------------------------------

    def run_harmonic_reference_construction(self, lmp):
        """
        Build the harmonic (phonon) reference during the averaging stage.

        The box is first remapped to the converged average dimensions so
        that the reference geometry matches the integration stage exactly.
        The structure is then energy-minimised at fixed box to obtain the
        reference sites, a set of random-displacement snapshots is
        generated and their real-potential forces are recorded, and
        finally the thermal state is restored and briefly re-equilibrated
        so that ``conf.equilibration.data`` contains a thermalised
        configuration as usual. The spring constants are fitted in
        :meth:`build_harmonic_model`.
        """
        hr = self.calc.harmonic_reference
        self.logger.info("Constructing harmonic (phonon) spring-network reference")
        self.logger.info(
            "spring cutoff %f A, %d snapshots, displacement amplitude %f A, backend %s"
            % (hr.cutoff, hr.n_snapshots, hr.displacement, hr.fitting_backend)
        )

        # remap to the converged average box
        lmp = ph.remap_box(lmp, self.lx, self.ly, self.lz)

        # save the thermalised state to restore after fitting
        lmp.command("reset_timestep    0")
        lmp.command(
            "write_dump        all custom traj.harmonic_thermal.dump "
            "id type x y z vx vy vz modify sort id"
        )

        # relax atomic positions at fixed box: these are the SYMMETRIC
        # reference sites. Using the relaxed lattice (rather than the noisy
        # sample mean) as the reference is what lets the symmetry-aware fit
        # exploit the space group of a perfect crystal -- a mean over a
        # finite trajectory breaks the symmetry and collapses hiphive to a
        # full unsymmetrised (P1) fit. For a perfect crystal the relaxed
        # lattice in the thermally-expanded box IS the thermal mean.
        lmp.command("min_style         cg")
        lmp.command("minimize          0.0 1.0e-8 10000 100000")
        lmp.command("reset_timestep    0")
        lmp.command(
            "write_dump        all custom harmonic.reference.dump "
            "id type x y z fx fy fz modify sort id"
        )

        # restore the thermal state (positions + velocities)
        lmp.command(
            "read_dump         traj.harmonic_thermal.dump 0 x y z vx vy vz "
            "box no replace yes"
        )

        if hr.fitting_backend == "tdep":
            # temperature-dependent effective potential: sample the
            # equilibrium MD trajectory for the fit forces (the reference
            # sites stay the symmetric relaxed lattice written above)
            self._sample_tdep_snapshots(lmp)
        else:
            # random-displacement snapshots with real-potential forces
            for s in range(hr.n_snapshots):
                lmp.command(
                    "read_dump         harmonic.reference.dump 0 x y z "
                    "box no replace yes"
                )
                lmp.command(
                    "displace_atoms    all random %f %f %f %d units box"
                    % (
                        hr.displacement,
                        hr.displacement,
                        hr.displacement,
                        np.random.randint(1, 10**6),
                    )
                )
                lmp.command("run               0")
                lmp.command(
                    "write_dump        all custom harmonic.snapshot_%d.dump "
                    "id type x y z fx fy fz modify sort id" % s
                )
            # restore the thermal state and decorrelate briefly
            lmp.command(
                "read_dump         traj.harmonic_thermal.dump 0 x y z vx vy vz "
                "box no replace yes"
            )
            if self.calc._qtb:
                qtb = self.calc.quantum_thermal_bath
                lmp.command("fix              3 all nve")
                lmp.command(
                    "fix              3q all qtb temp %f damp %f seed %d f_max %f N_f %d"
                    % (
                        self.calc._temperature,
                        qtb.thermostat_damping,
                        np.random.randint(1, 10**8),
                        qtb.f_max,
                        qtb.n_f,
                    )
                )
            else:
                lmp.command(
                    "fix              3 all nvt temp %f %f %f"
                    % (
                        self.calc._temperature,
                        self.calc._temperature,
                        self.calc.md.thermostat_damping[1],
                    )
                )
            lmp.command("run               %d" % int(self.calc.md.n_small_steps))
            if self.calc._qtb:
                lmp.command("unfix            3q")
            lmp.command("unfix            3")

        # fit the spring network and evaluate its free energy
        self.build_harmonic_model()

    def _sample_tdep_snapshots(self, lmp):
        """
        Collect thermal snapshots (positions + forces) from the
        equilibrated MD trajectory for the TDEP effective-FC fit.

        The cell is already equilibrated at the target (T, p) when this
        runs, so we simply continue the dynamics and dump ``n_snapshots``
        frames spaced ``sampling_interval`` steps apart. No minimisation
        and no displacement is imposed; the reference sites are the mean
        of these frames (computed in :meth:`build_harmonic_model`). The
        final frame is left in place as the thermalised configuration
        written to ``conf.equilibration.data``.
        """
        hr = self.calc.harmonic_reference
        self.logger.info(
            "TDEP sampling: %d frames every %d steps, cutoff %f A (hiphive "
            "symmetry-aware FC2 from equilibrium MD)"
            % (hr.n_snapshots, hr.sampling_interval, hr.cutoff)
        )
        if self.calc._qtb:
            qtb = self.calc.quantum_thermal_bath
            lmp.command("fix              3 all nve")
            lmp.command(
                "fix              3q all qtb temp %f damp %f seed %d f_max %f N_f %d"
                % (
                    self.calc._temperature,
                    qtb.thermostat_damping,
                    np.random.randint(1, 10**8),
                    qtb.f_max,
                    qtb.n_f,
                )
            )
        else:
            lmp.command(
                "fix              3 all nvt temp %f %f %f"
                % (
                    self.calc._temperature,
                    self.calc._temperature,
                    self.calc.md.thermostat_damping[1],
                )
            )
        for s in range(hr.n_snapshots):
            lmp.command("run               %d" % int(hr.sampling_interval))
            lmp.command(
                "write_dump        all custom harmonic.snapshot_%d.dump "
                "id type x y z fx fy fz modify sort id" % s
            )
        if self.calc._qtb:
            lmp.command("unfix            3q")
        lmp.command("unfix            3")

    def build_harmonic_model(self):
        """
        Fit the spring network to the recorded displacement-force data,
        verify its stability, evaluate its exact (classical and quantum)
        harmonic free energy, and write the LAMMPS ``pair_style list``
        spring file used in the switching stage.
        """
        hr = self.calc.harmonic_reference

        # fitting snapshots (positions + forces)
        positions = []
        forces = []
        for s in range(hr.n_snapshots):
            snap = read_lammps_dump(
                os.path.join(self.simfolder, "harmonic.snapshot_%d.dump" % s)
            )
            positions.append(snap["positions"])
            forces.append(snap["forces"])

        # reference sites: the SYMMETRIC relaxed lattice (both backends) --
        # building the fit's symmetry from the space group of a perfect
        # crystal needs a symmetric reference, not a noisy sample mean
        ref = read_lammps_dump(
            os.path.join(self.simfolder, "harmonic.reference.dump")
        )
        ref_positions = ref["positions"]
        ref_box, ref_types, ref_ids = ref["box"], ref["types"], ref["ids"]
        # displacements (and forces) are referenced to the symmetric relaxed
        # sites; the reference's own residual force (~0 after minimisation)
        # is the base force for both fitting backends
        base_forces = ref.get("forces", None)

        if hr.fitting_backend == "tdep":
            # temperature-dependent effective potential: the fit data are
            # thermal MD frames. For a perfect crystal the thermal mean sits
            # on the symmetric relaxed sites (zero mean displacement by
            # symmetry), so referencing to them is unbiased *and* keeps
            # hiphive symmetry-reduced. Report how far the sampled mean
            # actually drifts as an anharmonicity/symmetry diagnostic.
            from calphy.harmonic import tdep_reference, minimum_image

            mean_pos, mean_force = tdep_reference(positions, forces, ref_box)
            drift = float(
                np.abs(minimum_image(mean_pos - ref_positions, ref_box)).max()
            )
            self.logger.info(
                "TDEP: fitting effective FC2 to %d MD frames; thermal-mean "
                "drift from relaxed sites %.3f A, mean force %.2e eV/A"
                % (hr.n_snapshots, drift, np.abs(mean_force).max())
            )
            if drift > 0.5 * hr.distance_tolerance:
                self.logger.warning(
                    "TDEP thermal-mean drift %.3f A exceeds half the shell "
                    "tolerance; the sites may be under-sampled or the cell "
                    "may be low-symmetry/relaxing (fit still valid, but "
                    "consider more snapshots)." % drift
                )

        model = HarmonicModel(
            reference_positions=ref_positions,
            box=ref_box,
            types=ref_types,
            masses=self.calc.mass,
            ids=ref_ids,
            cutoff=hr.cutoff,
            distance_tolerance=hr.distance_tolerance,
        )
        self.logger.info(
            "Spring network: %d springs in %d bond types (shells)"
            % (len(model.pairs), model.n_groups)
        )

        if hr.fitting_backend in ("hiphive", "tdep"):
            fit_with_hiphive(
                model, positions, forces, base_forces=base_forces, logger=self.logger
            )
            # report the quality of the projected spring model
            res = []
            for pos, f in zip(positions, forces):
                fmodel = model.forces(pos)
                ftarget = f if base_forces is None else f - base_forces
                res.append(fmodel - ftarget)
            model.fit_rmse = float(np.sqrt(np.mean(np.concatenate(res) ** 2)))
        else:
            model.fit(positions, forces, base_forces=base_forces)

        self.logger.info(
            "Spring fit force RMSE %f eV/A" % model.fit_rmse
        )

        # exactly-quadratic force-constant reference: no tether (a
        # globally quadratic Hamiltonian cannot fold), no second leg
        if hr.fitting_backend in ("hiphive", "tdep") and hasattr(
            model, "full_fc_blocks"
        ):
            blocks = model.full_fc_blocks
            self.logger.info(
                "fcpot reference uses full FC2 blocks (%d)" % len(blocks)
            )
        else:
            blocks = model.fc_blocks(include_tether=False)
            self.logger.info(
                "fcpot reference uses spring-network blocks (%d)"
                % len(blocks)
            )
        self.harmonic_fc_blocks = blocks

        for g, grp in enumerate(model.groups):
            self.logger.info(
                "bond type %d: types (%d, %d), r0 %f A, k %f eV/A^2, %d springs"
                % (
                    g + 1,
                    grp["type_i"],
                    grp["type_j"],
                    grp["distance"],
                    model.k_groups[g],
                    grp["count"],
                )
            )
            if model.k_groups[g] < 0:
                self.logger.warning(
                    "bond type %d has a negative spring constant; this is "
                    "acceptable as long as the total network is stable "
                    "(checked below)" % (g + 1)
                )

        # frequencies; raises if the network is not a stable crystal
        from calphy.harmonic import hessian_from_blocks, frequencies_from_hessian

        H_fc = hessian_from_blocks(model.natoms, self.harmonic_fc_blocks)
        omega = frequencies_from_hessian(H_fc, model.masses, tethered=False)
        thz = omega / (2 * np.pi) / 1e12
        self.logger.info(
            "Phonon spectrum of reference: %f - %f THz over %d modes"
            % (thz[0], thz[-1], len(omega))
        )

        fe_classical = model.free_energy(
            self.calc._temperature, quantum=False, omega=omega
        )
        fe_quantum = model.free_energy(
            self.calc._temperature, quantum=True, omega=omega
        )
        self.logger.info(
            "Harmonic reference free energy (classical): %f eV/atom "
            "(modes %f, com %f)"
            % (
                fe_classical["f_total"],
                fe_classical["f_modes"],
                fe_classical["f_com"],
            )
        )
        self.logger.info(
            "Harmonic reference free energy (quantum): %f eV/atom "
            "(modes %f, com %f)"
            % (fe_quantum["f_total"], fe_quantum["f_modes"], fe_quantum["f_com"])
        )

        self.harmonic_model = model
        self.harmonic_fe_classical = fe_classical
        self.harmonic_fe_quantum = fe_quantum

        # artefacts for the integration stage and for post-processing
        model.save(os.path.join(self.simfolder, "harmonic.model.npz"))
        from calphy.harmonic import write_fc_file

        write_fc_file(
            os.path.join(self.simfolder, "harmonic.fc"),
            model.ids,
            model.reference_positions,
            self.harmonic_fc_blocks,
            comment="(%s blocks)" % hr.fitting_backend,
        )
        bi = np.array([b[0] for b in self.harmonic_fc_blocks])
        bj = np.array([b[1] for b in self.harmonic_fc_blocks])
        bb = np.array(
            [np.asarray(b[2]).reshape(9) for b in self.harmonic_fc_blocks]
        )
        np.savez_compressed(
            os.path.join(self.simfolder, "harmonic.fcblocks.npz"),
            bi=bi,
            bj=bj,
            blocks=bb,
        )
        np.savetxt(
            os.path.join(self.simfolder, "harmonic.frequencies.dat"),
            np.column_stack((thz, omega * 6.582119569e-16 * 1e3)),
            header="frequency_THz hbar_omega_meV",
        )

    def run_fcpot_integration(self, iteration=1):
        """
        Single-leg nonequilibrium Hamiltonian interpolation between the
        real potential and the exactly-quadratic force-constant reference
        E = 1/2 u^T Phi u, evaluated by the compiled ``fix fcpot`` plugin
        (plugins/fcpot). The reference is exactly solvable from the
        force-constant eigenvalues, needs no site tether (a globally
        quadratic Hamiltonian cannot fold or exchange sites) and no
        second switching leg. The real potential is scaled with
        ``pair_style hybrid/scaled``; the fix applies forces scaled by
        (1-lambda) and reports the UNSCALED reference energy as
        ``f_ffc[1]``, which is exactly the dU_ref column of the
        switching integrand (its scalar ``f_ffc`` is the scaled
        Hamiltonian contribution, with per-atom energy/virial tallies
        under ``fix_modify``). Files: forward_<i>.dat /
        backward_<i>.dat (dU_real, dU_ref, lambda; per atom).
        """
        lmp = ph.create_object(self.calc, self.simfolder)

        model = self.harmonic_model
        hr = self.calc.harmonic_reference
        T = self.calc._temperature
        tdamp = self.calc.md.thermostat_damping[1]
        plugin_path = os.path.abspath(hr.plugin_path)
        fcfile = os.path.join(self.simfolder, "harmonic.fc")

        lmp.command("variable         li equal 1.0")
        lmp.command("variable         lf equal 0.0")

        # fix fcpot resolves atom ids through the atom map
        lmp.command("atom_modify      map array")
        lmp.command("plugin load      %s" % plugin_path)

        # stage-control variables: lambda scales the real potential,
        # (1-lambda) the reference; both must be defined before the
        # scaled pair style references them
        lmp.command("variable         flambda equal 1.0")
        lmp.command("variable         blambda equal 1.0-v_flambda")

        # real potential scaled by lambda via hybrid/scaled
        lmp.command(ph.scaled_pair_style_command(self.calc, ["v_flambda"]))

        conf = os.path.join(self.simfolder, "conf.equilibration.data")
        therm = read_lammps_data(conf)
        if not np.allclose(therm["box"], model.box, atol=1e-3):
            raise RuntimeError(
                "Box of %s (%s) does not match the harmonic reference box "
                "(%s)." % (conf, therm["box"], model.box)
            )
        lmp = ph.read_data(lmp, conf)

        for command in ph.hybrid_pair_coeff_commands(self.calc):
            lmp.command(command)
        lmp = ph.set_mass(lmp, self.calc)

        lmp.command("fix              ffc all fcpot %s v_blambda" % fcfile)

        compute_commands, real_energy, compute_ids = ph.real_pair_compute_commands(
            self.calc
        )
        for command in compute_commands:
            lmp.command(command)

        lmp.command("variable         step equal step")
        lmp.command("variable         dU1 equal (%s)/atoms" % real_energy)
        lmp.command("variable         dU2 equal f_ffc[1]/atoms")

        lmp.command("compute          Tcm all temp/com")
        lmp.command("thermo_style     custom step v_dU1 v_dU2 c_Tcm")
        lmp.command("thermo           1000")

        # start-up consistency check: unscaled reference energy from the
        # fix vs the python force-constant model
        lmp.command("run              0")
        checkfile = os.path.join(self.simfolder, "harmonic.check.dat")
        lmp.command('print            "${dU2}" file %s screen no' % checkfile)
        self._verify_fcpot_energy(checkfile, therm["positions"])

        lmp.command(
            "velocity         all create %f %d mom yes rot yes dist gaussian"
            % (T, np.random.randint(1, 10000))
        )
        lmp.command("fix              f1 all nve")
        if self.calc._qtb:
            qtb = self.calc.quantum_thermal_bath
            lmp.command(
                "fix              f2 all qtb temp %f damp %f seed %d f_max %f N_f %d"
                % (
                    T,
                    qtb.thermostat_damping,
                    np.random.randint(1, 10**8),
                    qtb.f_max,
                    qtb.n_f,
                )
            )
        else:
            lmp.command(
                "fix              f2 all langevin %f %f %f %d zero yes"
                % (T, T, tdamp, np.random.randint(1, 10000))
            )
            lmp.command("fix_modify       f2 temp Tcm")

        # equilibrate on the real potential (lambda = 1)
        lmp.command("run              %d" % self.calc.n_equilibration_steps)

        # FWD: real -> force-constant reference
        lmp.command("variable         flambda equal ramp(${li},${lf})")
        lmp.command(
            'fix              f3 all print 1 "${dU1} ${dU2} ${flambda}" '
            "screen no file forward_%d.dat" % iteration
        )
        lmp.command("run              %d" % self.calc._n_switching_steps)
        lmp.command("unfix            f3")

        # equilibrate on the pure reference (lambda = 0)
        lmp.command("variable         flambda equal 0.0")
        lmp.command("run              %d" % self.calc.n_equilibration_steps)

        # BKD: force-constant reference -> real
        lmp.command("variable         flambda equal ramp(${lf},${li})")
        lmp.command(
            'fix              f3 all print 1 "${dU1} ${dU2} ${flambda}" '
            "screen no file backward_%d.dat" % iteration
        )
        lmp.command("run              %d" % self.calc._n_switching_steps)
        lmp.command("unfix            f3")

        lmp.command("unfix            f1")
        lmp.command("unfix            f2")
        lmp.command("unfix            ffc")
        for compute_id in compute_ids:
            lmp.command("uncompute        %s" % compute_id)

        self.lammps_close(lmp=lmp)
        logfile = os.path.join(self.simfolder, "log.lammps")
        try:
            if os.path.exists(logfile):
                os.rename(
                    logfile, os.path.join(self.simfolder, "integration.log.lammps")
                )
        except OSError as e:
            self.logger.warning(f"Failed to rename log file: {e}")

    def _verify_fcpot_energy(self, checkfile, positions):
        """
        Compare the fix fcpot reference energy against the python
        force-constant model at the same configuration.
        """
        if not os.path.exists(checkfile):
            return
        from calphy.harmonic import hessian_from_blocks, minimum_image

        e_lmp = float(np.loadtxt(checkfile, ndmin=1)[0]) * self.natoms
        model = self.harmonic_model
        H = hessian_from_blocks(model.natoms, self.harmonic_fc_blocks)
        u = minimum_image(
            np.asarray(positions, dtype=float) - model.reference_positions,
            model.box,
        ).reshape(-1)
        e_py = 0.5 * u @ H @ u
        scale = max(abs(e_py), 1.0)
        if abs(e_lmp - e_py) > 1e-5 * scale:
            raise RuntimeError(
                "fcpot reference consistency check failed: LAMMPS %.8f eV "
                "vs python model %.8f eV. The switching would measure the "
                "wrong Hamiltonian." % (e_lmp, e_py)
            )
        self.logger.info(
            "fcpot reference consistency check passed (%.6f eV)" % e_lmp
        )

    def _restore_harmonic_model(self):
        """
        Reload the fitted spring model from disk (e.g. when
        thermodynamic_integration is called in a fresh process).
        """
        model = HarmonicModel.load(
            os.path.join(self.simfolder, "harmonic.model.npz")
        )
        self.harmonic_model = model
        from calphy.harmonic import hessian_from_blocks, frequencies_from_hessian

        data = np.load(os.path.join(self.simfolder, "harmonic.fcblocks.npz"))
        self.harmonic_fc_blocks = [
            (int(i), int(j), b.reshape(3, 3))
            for i, j, b in zip(data["bi"], data["bj"], data["blocks"])
        ]
        H_fc = hessian_from_blocks(model.natoms, self.harmonic_fc_blocks)
        omega = frequencies_from_hessian(H_fc, model.masses, tethered=False)
        self.harmonic_fe_classical = model.free_energy(
            self.calc._temperature, quantum=False, omega=omega
        )
        self.harmonic_fe_quantum = model.free_energy(
            self.calc._temperature, quantum=True, omega=omega
        )

    def run_integration(self, iteration=1):
        """
        Run integration routine

        Parameters
        ----------
        iteration : int, optional
            iteration number for running independent iterations

        Returns
        -------
        None

        Notes
        -----
        Run the integration routine where the initial and final systems are connected using
        the lambda parameter. See algorithm 4 in publication.
        """
        if self._use_harmonic_reference:
            if not hasattr(self, "harmonic_model"):
                self._restore_harmonic_model()
            return self.run_fcpot_integration(iteration=iteration)

        lmp = ph.create_object(self.calc, self.simfolder)

        # set up potential
        lmp = ph.set_pair_style(lmp, self.calc)

        # read in the conf file
        # conf = os.path.join(self.simfolder, "conf.equilibration.dump")
        conf = os.path.join(self.simfolder, "conf.equilibration.data")
        lmp = ph.read_data(lmp, conf)

        lmp = ph.set_pair_coeff(lmp, self.calc)
        lmp = ph.set_mass(lmp, self.calc)

        # remap the box to get the correct pressure
        lmp = ph.remap_box(lmp, self.lx, self.ly, self.lz)

        # create groups - each species belong to one group
        for i in range(self.calc.n_elements):
            lmp.command("group  g%d type %d" % (i + 1, i + 1))

        # get counts of each group
        for i in range(self.calc.n_elements):
            lmp.command("variable   count%d equal count(g%d)" % (i + 1, i + 1))

        # initialise everything
        lmp.command("run               0")

        # apply initial fixes
        lmp.command("fix               f1 all nve")

        # apply fix for each spring
        # TODO: Add option to select function
        for i in range(self.calc.n_elements):
            lmp.command(
                "fix               ff%d g%d ti/spring 10.0 100 100 function 2"
                % (i + 1, i + 1)
            )

        # apply temp fix
        if self.calc._qtb:
            qtb = self.calc.quantum_thermal_bath
            lmp.command(
                "fix               f3 all qtb temp %f damp %f seed %d f_max %f N_f %d"
                % (
                    self.calc._temperature,
                    qtb.thermostat_damping,
                    np.random.randint(1, 10**8),
                    qtb.f_max,
                    qtb.n_f,
                )
            )
            # QTB does not consume a base temperature compute, so the temp/com
            # group/correction trick used by langevin does not apply.
            lmp.command("compute           Tcm all temp/com")
        else:
            lmp.command(
                "fix               f3 all langevin %f %f %f %d zero yes"
                % (
                    self.calc._temperature,
                    self.calc._temperature,
                    self.calc.md.thermostat_damping[1],
                    np.random.randint(1, 10000),
                )
            )

            # compute com and apply to fix
            lmp.command("compute           Tcm all temp/com")
            lmp.command("fix_modify        f3 temp Tcm")

        lmp.command("variable          step    equal step")
        lmp.command("variable          dU1      equal pe/atoms")
        for i in range(self.calc.n_elements):
            lmp.command("variable          dU%d      equal f_ff%d" % (i + 2, i + 1))

        lmp.command("variable          lambda  equal f_ff1[1]")

        # add thermo command to force variable evaluation
        lmp.command("thermo_style      custom step pe c_Tcm")
        lmp.command("thermo            10000")

        # Create velocity
        lmp.command(
            "velocity          all create %f %d mom yes rot yes dist gaussian"
            % (self.calc._temperature, np.random.randint(1, 10000))
        )

        # reapply
        for i in range(self.calc.n_elements):
            lmp.command(
                "fix               ff%d g%d ti/spring %f %d %d function 2"
                % (
                    i + 1,
                    i + 1,
                    self.k[i],
                    self.calc._n_switching_steps,
                    self.calc.n_equilibration_steps,
                )
            )

        # Equilibriate structure
        lmp.command("run               %d" % self.calc.n_equilibration_steps)

        # write out energy
        str1 = 'fix f4 all print 1 "${dU1} '
        str2 = []
        for i in range(self.calc.n_elements):
            str2.append("${dU%d}" % (i + 2))

        str2.append('${lambda}"')
        str2 = " ".join(str2)
        title_cols = (
            ["dU_sys[eV/atom]"]
            + ["dU_ref%d[eV/atom]" % (i + 1) for i in range(self.calc.n_elements)]
            + ["lambda"]
        )
        str3 = ' title "# %s" screen no file forward_%d.dat' % (
            " ".join(title_cols),
            iteration,
        )
        command = str1 + str2 + str3
        lmp.command(command)

        if self.calc.n_print_steps > 0:
            lmp.command(
                "dump              d1 all custom %d traj.fe.forward_%d.dat id type mass x y z fx fy fz"
                % (self.calc.n_print_steps, iteration)
            )

        # turn on swap moves
        # if self.calc.monte_carlo.n_swaps > 0:
        #    self.logger.info(f'{self.calc.monte_carlo.n_swaps} swap moves are performed between 1 and 2 every {self.calc.monte_carlo.n_steps}')
        #    lmp.command("fix  swap all atom/swap %d %d %d %d ke yes types 1 2"%(self.calc.monte_carlo.n_steps,
        #                                                                        self.calc.monte_carlo.n_swaps,
        #                                                                        np.random.randint(1, 10000),
        #                                                                        self.calc._temperature))
        #
        #    lmp.command("variable a equal f_swap[1]")
        #    lmp.command("variable b equal f_swap[2]")
        #    lmp.command("fix             swap2 all print 1 \"${a} ${b}\" screen no file swap.fe.forward_%d.dat"%iteration)

        # Forward switching over ts steps
        lmp.command("run               %d" % self.calc._n_switching_steps)
        lmp.command("unfix             f4")

        if self.calc.n_print_steps > 0:
            lmp.command("undump           d1")

        # if self.calc.monte_carlo.n_swaps > 0:
        #    lmp.command("unfix swap")
        #    lmp.command("unfix swap2")

        # Equilibriate
        lmp.command("run               %d" % self.calc.n_equilibration_steps)

        # write out energy
        str1 = 'fix f4 all print 1 "${dU1} '
        str2 = []
        for i in range(self.calc.n_elements):
            str2.append("${dU%d}" % (i + 2))

        str2.append('${lambda}"')
        str2 = " ".join(str2)
        title_cols = (
            ["dU_sys[eV/atom]"]
            + ["dU_ref%d[eV/atom]" % (i + 1) for i in range(self.calc.n_elements)]
            + ["lambda"]
        )
        str3 = ' title "# %s" screen no file backward_%d.dat' % (
            " ".join(title_cols),
            iteration,
        )
        command = str1 + str2 + str3
        lmp.command(command)

        if self.calc.n_print_steps > 0:
            lmp.command(
                "dump              d1 all custom %d traj.fe.backward_%d.dat id type mass x y z fx fy fz"
                % (self.calc.n_print_steps, iteration)
            )

        # add swaps if n_swap is > 0
        # if self.calc.monte_carlo.n_swaps > 0:
        #    self.logger.info(f'{self.calc.monte_carlo.n_swaps} swap moves are performed between 1 and 2 every {self.calc.monte_carlo.n_steps}')
        #    lmp.command("fix  swap all atom/swap %d %d %d %d ke yes types 2 1"%(self.calc.monte_carlo.n_steps,
        #                                                                        self.calc.monte_carlo.n_swaps,
        #                                                                        np.random.randint(1, 10000),
        #                                                                        self.calc._temperature))
        #
        #    lmp.command("variable a equal f_swap[1]")
        #    lmp.command("variable b equal f_swap[2]")
        #    lmp.command("fix             swap2 all print 1 \"${a} ${b}\" screen no file swap.fe.backward_%d.dat"%iteration)

        # Reverse switching over ts steps
        lmp.command("run               %d" % self.calc._n_switching_steps)
        lmp.command("unfix             f4")

        if self.calc.n_print_steps > 0:
            lmp.command("undump           d1")

        # if self.calc.monte_carlo.n_swaps > 0:
        #    lmp.command("unfix swap")
        #    lmp.command("unfix swap2")

        # close object
        self.lammps_close(lmp=lmp)
        lmp.rotate_logs("integration")

    def thermodynamic_integration(self):
        """
        Calculate free energy after integration step

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Calculates the final work, energy dissipation and free energy by
        matching with Einstein crystal
        """
        if self._use_harmonic_reference:
            return self._harmonic_thermodynamic_integration()

        use_quantum_reference = self.calc._qtb
        if use_quantum_reference:
            self.logger.info(
                "Using quantum harmonic-oscillator Einstein-crystal reference "
                "(required for self-consistency with QTB sampling)."
            )
        fe, fcm = get_einstein_crystal_fe(
            self.calc, self.vol, self.k,
            return_contributions=True,
            quantum=use_quantum_reference,
        )

        w, q, qerr = find_w(self.simfolder, self.calc, full=True, solid=True)

        self.fref = fe + fcm
        self.feinstein = fe
        self.fcm = fcm
        self.w = w
        self.qdiss = q
        self.ferr = qerr

        # add pressure contribution if required
        if self.calc._pressure != 0:
            p = self.calc._pressure / EV_A3_TO_BAR
            v = self.vol / self.natoms
            self.pv = p * v
        else:
            self.pv = 0

        # calculate final free energy
        self.fe = self.fref + self.w + self.pv

    def _harmonic_thermodynamic_integration(self):
        """
        Free energy assembly for the single-leg harmonic (phonon)
        force-constant reference path:

            F_real = F_fc - w (+ pV)

        A single leg switches the real potential to the exactly-quadratic
        force-constant reference E = 1/2 u^T Phi u, whose free energy F_fc
        is exact from its eigenvalues (classical or quantum, including the
        Frenkel-Smit centre-of-mass correction). The quality of the fit
        controls only the switching dissipation, never the correctness.
        """
        if not hasattr(self, "harmonic_model"):
            self._restore_harmonic_model()

        quantum = self.calc.harmonic_reference.quantum
        if quantum:
            self.logger.info("Using the quantum reference free energy")
            if not self.calc._qtb:
                self.logger.info(
                    "Note: switching MD is classical; the result is a "
                    "one-shot quantum correction (classical anharmonicity "
                    "on top of a quantum harmonic baseline)."
                )

        # single leg against the exactly-quadratic force-constant
        # reference, whose free energy is exact from its eigenvalues
        w, q, qerr = find_w(self.simfolder, self.calc, full=True, solid=False)
        fe_dict = (
            self.harmonic_fe_quantum if quantum else self.harmonic_fe_classical
        )
        self.fref = fe_dict["f_total"]
        self.feinstein = fe_dict["f_modes"]
        self.fcm = fe_dict["f_com"]
        self.w = w
        self.ferr = qerr

        if self.calc._pressure != 0:
            p = self.calc._pressure / EV_A3_TO_BAR
            v = self.vol / self.natoms
            self.pv = p * v
        else:
            self.pv = 0

        self.fe = self.fref - self.w + self.pv

    def submit_report(self, extra_dict=None):
        """
        Add harmonic-reference details to the report when the phonon
        reference is active.
        """
        if self._use_harmonic_reference and hasattr(self, "harmonic_model"):
            model = self.harmonic_model
            omega = self.harmonic_fe_classical["omega"]
            thz = omega / (2 * np.pi) / 1e12
            hdict = {
                "harmonic_reference": {
                    "quantum": bool(self.calc.harmonic_reference.quantum),
                    "n_springs": int(len(model.pairs)),
                    "n_bond_types": int(model.n_groups),
                    "spring_constants": " ".join(
                        np.round(model.k_groups, decimals=6).astype(str)
                    ),
                    "shell_distances": " ".join(
                        np.round(
                            [g["distance"] for g in model.groups], decimals=4
                        ).astype(str)
                    ),
                    "fit_rmse": float(getattr(model, "fit_rmse", 0.0)),
                    "frequency_range_THz": "%f %f" % (thz[0], thz[-1]),
                    "network_harmonic_fe_classical": float(
                        self.harmonic_fe_classical["f_total"]
                    ),
                    "network_harmonic_fe_quantum": float(
                        self.harmonic_fe_quantum["f_total"]
                    ),
                }
            }
            if extra_dict is not None:
                hdict.update(extra_dict)
            super().submit_report(extra_dict=hdict)
        else:
            super().submit_report(extra_dict=extra_dict)
