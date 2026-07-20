#!/bin/sh
# Build the fcpot LAMMPS plugin against an installed LAMMPS.
#
#   ./build.sh [LAMMPS_INCLUDE_DIR]
#
# If no include dir is given, the headers shipped with the `lammps`
# python package are used (pip install lammps). The resulting
# fcpotplugin.so is loaded in LAMMPS with:  plugin load fcpotplugin.so
# (requires a LAMMPS build with the PLUGIN package).
set -e
cd "$(dirname "$0")"

INC="$1"
if [ -z "$INC" ]; then
  INC=$(python3 -c "import lammps, os; print(os.path.join(os.path.dirname(lammps.__file__), 'include', 'lammps'))")
fi
if [ ! -f "$INC/fix.h" ]; then
  echo "LAMMPS headers not found at $INC" >&2
  exit 1
fi

# lammpsplugin.h is not shipped with all header bundles; fetch if needed
if [ ! -f "$INC/lammpsplugin.h" ] && [ ! -f lammpsplugin.h ]; then
  curl -sL https://raw.githubusercontent.com/lammps/lammps/stable/src/lammpsplugin.h -o lammpsplugin.h
fi

# version.h is not shipped either: generate the version string of the
# LAMMPS the plugin will be loaded into (must match at load time)
if [ ! -f "$INC/version.h" ]; then
  python3 - <<'EOF'
from lammps import lammps
import datetime
l = lammps(cmdargs=["-screen", "none", "-log", "none"])
v = str(l.version())  # e.g. 20250722
l.close()
d = datetime.date(int(v[:4]), int(v[4:6]), int(v[6:8]))
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
s = "%d %s %d" % (d.day, months[d.month-1], d.year)
open("version.h", "w").write('#define LAMMPS_VERSION "%s"\n' % s)
print("generated version.h:", s)
EOF
fi

MPIFLAGS=""
if command -v mpicxx >/dev/null 2>&1; then
  CXX=mpicxx
else
  CXX=c++
  for d in /usr/include/x86_64-linux-gnu/mpich /usr/include/mpich /usr/lib/x86_64-linux-gnu/mpich/include; do
    [ -f "$d/mpi.h" ] && MPIFLAGS="-I$d" && break
  done
fi

$CXX -O2 -fPIC -shared $MPIFLAGS -I"$INC" -I. -o fcpotplugin.so fcpotplugin.cpp fix_fcpot.cpp
echo "built $(pwd)/fcpotplugin.so"
