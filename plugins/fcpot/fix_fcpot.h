/* -*- c++ -*- ----------------------------------------------------------
   fix fcpot: harmonic force-constant potential for Frenkel-Ladd
   thermodynamic integration (calphy harmonic phonon reference).
   See fix_fcpot.cpp for documentation.
------------------------------------------------------------------------- */

#ifndef LMP_FIX_FCPOT_H
#define LMP_FIX_FCPOT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixFCPot : public Fix {
 public:
  FixFCPot(class LAMMPS *, int, char **);
  ~FixFCPot() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void setup_pre_reverse(int, int) override;
  void pre_reverse(int, int) override;
  void post_force(int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  double compute_vector(int) override;

 private:
  enum { CONSTANT, VARIABLE };

  bigint nref;         // number of reference sites
  tagint maxtag;       // largest tag with a reference site
  double *xref;        // reference site coordinates, indexed by 3*tag
  int *hasref;         // 1 if a tag has a reference site
  bigint nblocks;      // mirrored force-constant blocks
  tagint *btagi;       // block atom i (owner applies the block)
  tagint *btagj;       // block atom j
  double **bphi;       // 3x3 block, row-major
  double fccut;        // longest interacting pair distance

  int scalestyle;
  double scaleconst;
  char *scalevar;
  int scaleindex;

  double **udisp;      // per-atom minimum-image displacement cache
  int maxatom;

  double eref_all;     // reduced unscaled reference energy 1/2 u^T Phi u
  double lastscale;    // scale evaluated in the last post_force

  int eflag_caller;    // energy flag captured in pre_reverse for tallies
  int extlist_storage[2];
};

}    // namespace LAMMPS_NS

#endif
