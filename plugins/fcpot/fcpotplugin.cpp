/* ----------------------------------------------------------------------
   LAMMPS plugin registration for fix fcpot.
   Build with build.sh; load in LAMMPS with:  plugin load fcpotplugin.so
------------------------------------------------------------------------- */

#include "lammpsplugin.h"
#include "version.h"

#include "fix_fcpot.h"

using namespace LAMMPS_NS;

static Fix *fcpot_creator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixFCPot(lmp, argc, argv);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "fcpot";
  plugin.info = "harmonic force-constant potential for Frenkel-Ladd TI (calphy)";
  plugin.author = "calphy harmonic reference";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &fcpot_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
