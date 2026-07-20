/* ----------------------------------------------------------------------
   fix fcpot: harmonic force-constant potential for Frenkel-Ladd
   thermodynamic integration (calphy harmonic phonon reference).

   E(u) = (1/2) * sum_{i,j} u_i^T Phi_ij u_j,   u_i = x_i - X_i (min. image)

   where X are fixed reference sites and Phi_ij are 3x3 force-constant
   blocks read from a file. The Hamiltonian is exactly quadratic in the
   displacements, so its free energy follows exactly from the eigenvalues
   of the mass-weighted force-constant matrix.

   Usage:
     fix ID group fcpot <fcfile> <scale>

   <scale> is either a constant or an equal-style variable (v_name),
   evaluated every step; the applied forces are scaled by it (used for
   the (1-lambda) switching).

   Outputs:
     f_ID       scalar: SCALED energy scale * E, the actual Hamiltonian
                contribution of the fix (consistent with the applied
                forces, so "fix_modify ID energy yes" gives a conserved
                total energy)
     f_ID[1]    UNSCALED reference energy E, i.e. exactly the dU_ref
                column of the switching integrand
     f_ID[2]    current value of <scale>

   With "fix_modify ID energy yes" / "virial yes" the fix also tallies
   the scaled energy and virial per atom (and the virial globally) in
   the modern Fix ev_tally style: each force-constant block (i, j)
   tallies its scaled energy 0.5 * scale * u_i^T Phi_ij u_j and virial
   (X_i + u_i) (x) F_block to the owner of atom i. The per-block virial
   uses the reference-consistent coordinate X_i + u_i, which is smooth
   across periodic boundaries; the global sum is well defined when the
   blocks obey the acoustic sum rule.

   File format (text; '#' comments and blank lines ignored):
     natoms
     tag x y z                 (natoms lines: reference sites)
     nblocks
     tag_i tag_j p11 p12 p13 p21 p22 p23 p31 p32 p33   (nblocks lines)

   Blocks are stored once per (i, j) with tag_i <= tag_j; the transpose
   is applied for (j, i) automatically. Diagonal blocks (tag_i == tag_j)
   must be included explicitly.

   Notes: the fix does not integrate motion; pair it with nve + a
   thermostat at fixed box, as calphy does. Forces are applied to atoms
   of the fix group only in the sense that all atoms referenced by the
   file receive forces; the group is ignored for the energy sum (use a
   file covering the intended atoms).
------------------------------------------------------------------------- */

#include "fix_fcpot.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixFCPot::FixFCPot(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), xref(nullptr), btagi(nullptr), btagj(nullptr), bphi(nullptr),
    udisp(nullptr), scalevar(nullptr)
{
  if (narg != 5) error->all(FLERR, "Illegal fix fcpot command: fix ID group fcpot file scale");

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extscalar = 1;
  extvector = -1;
  extlist_storage[0] = 1;    // unscaled reference energy: extensive
  extlist_storage[1] = 0;    // scale: intensive
  extlist = extlist_storage;
  energy_global_flag = 1;
  energy_peratom_flag = 1;
  virial_global_flag = 1;
  virial_peratom_flag = 1;

  // scale argument: constant or equal-style variable v_name
  scalestyle = CONSTANT;
  scaleconst = 1.0;
  if (strncmp(arg[4], "v_", 2) == 0) {
    scalestyle = VARIABLE;
    scalevar = utils::strdup(arg[4] + 2);
  } else {
    scaleconst = utils::numeric(FLERR, arg[4], false, lmp);
  }

  // read the force-constant file on rank 0 and broadcast
  nref = 0;
  nblocks = 0;
  maxtag = 0;
  fccut = 0.0;

  bigint natoms_file = 0;
  bigint nblocks_file = 0;

  double *xbuf = nullptr;
  tagint *tbuf = nullptr;
  double *pbuf = nullptr;
  tagint *ibuf = nullptr;
  tagint *jbuf = nullptr;

  if (comm->me == 0) {
    FILE *fp = fopen(arg[3], "r");
    if (!fp) error->one(FLERR, "Cannot open fix fcpot file {}", arg[3]);

    char line[1024];
    auto nextline = [&]() -> char * {
      while (fgets(line, 1024, fp)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '\0' || *p == '\n' || *p == '#') continue;
        return p;
      }
      return nullptr;
    };

    char *p = nextline();
    if (!p) error->one(FLERR, "Unexpected end of fix fcpot file");
    natoms_file = utils::bnumeric(FLERR, strtok(p, " \t\n"), false, lmp);

    tbuf = new tagint[natoms_file];
    xbuf = new double[3 * natoms_file];
    for (bigint n = 0; n < natoms_file; ++n) {
      p = nextline();
      if (!p) error->one(FLERR, "Unexpected end of fix fcpot file (sites)");
      tbuf[n] = utils::tnumeric(FLERR, strtok(p, " \t\n"), false, lmp);
      for (int k = 0; k < 3; ++k)
        xbuf[3 * n + k] = utils::numeric(FLERR, strtok(nullptr, " \t\n"), false, lmp);
    }

    p = nextline();
    if (!p) error->one(FLERR, "Unexpected end of fix fcpot file (block count)");
    nblocks_file = utils::bnumeric(FLERR, strtok(p, " \t\n"), false, lmp);

    ibuf = new tagint[nblocks_file];
    jbuf = new tagint[nblocks_file];
    pbuf = new double[9 * nblocks_file];
    for (bigint n = 0; n < nblocks_file; ++n) {
      p = nextline();
      if (!p) error->one(FLERR, "Unexpected end of fix fcpot file (blocks)");
      ibuf[n] = utils::tnumeric(FLERR, strtok(p, " \t\n"), false, lmp);
      jbuf[n] = utils::tnumeric(FLERR, strtok(nullptr, " \t\n"), false, lmp);
      if (ibuf[n] > jbuf[n]) error->one(FLERR, "fix fcpot blocks must have tag_i <= tag_j");
      for (int k = 0; k < 9; ++k)
        pbuf[9 * n + k] = utils::numeric(FLERR, strtok(nullptr, " \t\n"), false, lmp);
    }
    fclose(fp);
  }

  MPI_Bcast(&natoms_file, 1, MPI_LMP_BIGINT, 0, world);
  MPI_Bcast(&nblocks_file, 1, MPI_LMP_BIGINT, 0, world);
  if (comm->me != 0) {
    tbuf = new tagint[natoms_file];
    xbuf = new double[3 * natoms_file];
    ibuf = new tagint[nblocks_file];
    jbuf = new tagint[nblocks_file];
    pbuf = new double[9 * nblocks_file];
  }
  MPI_Bcast(tbuf, natoms_file, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(xbuf, 3 * natoms_file, MPI_DOUBLE, 0, world);
  MPI_Bcast(ibuf, nblocks_file, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(jbuf, nblocks_file, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(pbuf, 9 * nblocks_file, MPI_DOUBLE, 0, world);

  nref = natoms_file;

  // reference sites indexed by tag
  for (bigint n = 0; n < nref; ++n) maxtag = MAX(maxtag, tbuf[n]);
  memory->create(xref, 3 * (maxtag + 1), "fcpot:xref");
  memory->create(hasref, (int) (maxtag + 1), "fcpot:hasref");
  for (tagint t = 0; t <= maxtag; ++t) hasref[t] = 0;
  for (bigint n = 0; n < nref; ++n) {
    tagint t = tbuf[n];
    if (t < 1) error->all(FLERR, "fix fcpot: invalid atom tag in file");
    xref[3 * t + 0] = xbuf[3 * n + 0];
    xref[3 * t + 1] = xbuf[3 * n + 1];
    xref[3 * t + 2] = xbuf[3 * n + 2];
    hasref[t] = 1;
  }

  // mirror the (i <= j) storage into both directions so each block is
  // applied by the rank owning atom i only
  nblocks = 0;
  for (bigint n = 0; n < nblocks_file; ++n) nblocks += (ibuf[n] == jbuf[n]) ? 1 : 2;
  memory->create(btagi, nblocks, "fcpot:btagi");
  memory->create(btagj, nblocks, "fcpot:btagj");
  memory->create(bphi, nblocks, 9, "fcpot:bphi");

  bigint m = 0;
  for (bigint n = 0; n < nblocks_file; ++n) {
    if (!hasref[ibuf[n]] || !hasref[jbuf[n]])
      error->all(FLERR, "fix fcpot: block references atom without a reference site");
    btagi[m] = ibuf[n];
    btagj[m] = jbuf[n];
    for (int k = 0; k < 9; ++k) bphi[m][k] = pbuf[9 * n + k];
    ++m;
    if (ibuf[n] != jbuf[n]) {
      btagi[m] = jbuf[n];
      btagj[m] = ibuf[n];
      // transpose
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) bphi[m][3 * a + b] = pbuf[9 * n + 3 * b + a];
      ++m;
    }
  }

  // longest interacting pair distance (for the ghost-atom cutoff)
  for (bigint n = 0; n < nblocks_file; ++n) {
    if (ibuf[n] == jbuf[n]) continue;
    double d[3];
    for (int k = 0; k < 3; ++k) d[k] = xref[3 * ibuf[n] + k] - xref[3 * jbuf[n] + k];
    domain->minimum_image(FLERR, d[0], d[1], d[2]);
    fccut = MAX(fccut, sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
  }

  delete[] tbuf;
  delete[] xbuf;
  delete[] ibuf;
  delete[] jbuf;
  delete[] pbuf;

  eref_all = 0.0;
  lastscale = scaleconst;
  eflag_caller = 0;
  maxatom = 0;
}

/* ---------------------------------------------------------------------- */

FixFCPot::~FixFCPot()
{
  memory->destroy(xref);
  memory->destroy(hasref);
  memory->destroy(btagi);
  memory->destroy(btagj);
  memory->destroy(bphi);
  memory->destroy(udisp);
  delete[] scalevar;
}

/* ---------------------------------------------------------------------- */

int FixFCPot::setmask()
{
  int mask = 0;
  mask |= PRE_REVERSE;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFCPot::init()
{
  if (!atom->map_style)
    error->all(FLERR, "fix fcpot requires an atom map (use atom_modify map array)");

  if (scalestyle == VARIABLE) {
    scaleindex = input->variable->find(scalevar);
    if (scaleindex < 0) error->all(FLERR, "fix fcpot: variable {} does not exist", scalevar);
    if (!input->variable->equalstyle(scaleindex))
      error->all(FLERR, "fix fcpot: variable {} is not equal-style", scalevar);
  }

  // make sure ghost atoms cover the longest force-constant pair
  comm->cutghostuser = MAX(comm->cutghostuser, fccut + 2.0);
}

/* ---------------------------------------------------------------------- */

void FixFCPot::setup(int vflag)
{
  post_force(vflag);
}

void FixFCPot::min_setup(int vflag)
{
  post_force(vflag);
}

void FixFCPot::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   store eflag, so it can be used in post_force to tally per-atom energies
------------------------------------------------------------------------- */

void FixFCPot::setup_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag, vflag);
}

void FixFCPot::pre_reverse(int eflag, int /*vflag*/)
{
  eflag_caller = eflag;
}

/* ---------------------------------------------------------------------- */

void FixFCPot::post_force(int vflag)
{
  double scale = scaleconst;
  if (scalestyle == VARIABLE) scale = input->variable->compute_equal(scaleindex);
  lastscale = scale;

  ev_init(eflag_caller, vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;

  // cache minimum-image displacements u = x - X for local + ghost atoms
  if (nall > maxatom) {
    memory->destroy(udisp);
    maxatom = nall + 256;
    memory->create(udisp, maxatom, 3, "fcpot:udisp");
  }
  for (int i = 0; i < nall; ++i) {
    tagint t = tag[i];
    if (t < 1 || t > maxtag || !hasref[t]) {
      udisp[i][0] = udisp[i][1] = udisp[i][2] = 0.0;
      continue;
    }
    double d0 = x[i][0] - xref[3 * t + 0];
    double d1 = x[i][1] - xref[3 * t + 1];
    double d2 = x[i][2] - xref[3 * t + 2];
    domain->minimum_image(FLERR, d0, d1, d2);
    udisp[i][0] = d0;
    udisp[i][1] = d1;
    udisp[i][2] = d2;
  }

  double esum = 0.0;
  for (bigint n = 0; n < nblocks; ++n) {
    const int il = atom->map(btagi[n]);
    if (il < 0 || il >= nlocal) continue;    // block applied by owner of atom i
    const int jl = atom->map(btagj[n]);
    if (jl < 0)
      error->one(FLERR, "fix fcpot: interacting atom {} not found as local or ghost atom",
                 btagj[n]);

    const double *phi = bphi[n];
    const double *uj = udisp[jl];
    const double *ui = udisp[il];

    const double g0 = phi[0] * uj[0] + phi[1] * uj[1] + phi[2] * uj[2];
    const double g1 = phi[3] * uj[0] + phi[4] * uj[1] + phi[5] * uj[2];
    const double g2 = phi[6] * uj[0] + phi[7] * uj[1] + phi[8] * uj[2];

    const double fb0 = -scale * g0;
    const double fb1 = -scale * g1;
    const double fb2 = -scale * g2;

    f[il][0] += fb0;
    f[il][1] += fb1;
    f[il][2] += fb2;

    const double eblk = 0.5 * (ui[0] * g0 + ui[1] * g1 + ui[2] * g2);
    esum += eblk;

    if (evflag) {
      // reference-consistent coordinate X_i + u_i: smooth across
      // periodic boundaries, so the global virial is well defined
      const tagint ti = btagi[n];
      const double r0 = xref[3 * ti + 0] + ui[0];
      const double r1 = xref[3 * ti + 1] + ui[1];
      const double r2 = xref[3 * ti + 2] + ui[2];
      double v6[6];
      v6[0] = r0 * fb0;
      v6[1] = r1 * fb1;
      v6[2] = r2 * fb2;
      v6[3] = r0 * fb1;
      v6[4] = r0 * fb2;
      v6[5] = r1 * fb2;
      int one = il;
      ev_tally(1, &one, 1.0, scale * eblk, v6);
    }
  }

  MPI_Allreduce(&esum, &eref_all, 1, MPI_DOUBLE, MPI_SUM, world);
}

/* ----------------------------------------------------------------------
   scalar    scaled energy scale * E (the fix's Hamiltonian contribution)
   vector[0] unscaled reference energy E = 1/2 u^T Phi u (dU_ref integrand)
   vector[1] current scale
------------------------------------------------------------------------- */

double FixFCPot::compute_scalar()
{
  return lastscale * eref_all;
}

double FixFCPot::compute_vector(int n)
{
  return (n == 0) ? eref_all : lastscale;
}
