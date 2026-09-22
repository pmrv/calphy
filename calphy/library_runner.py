"""
calphy: a Python library and command line interface for automated free
energy calculations.

Copyright 2021-2026 (c) Sarath Menon, Yury Lysogorskiy, Ralf Drautz
Interdisciplinary Centre for Advanced Materials Simulation (ICAMS),
Ruhr University Bochum, 44801 Bochum, Germany

calphy is published and distributed under the Academic Software Licence v1.0 (ASL).
calphy is distributed in the hope that it will be useful for non-commercial academic
research, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the LICENSE file for details.

The ASL permits academic non-commercial use only. Contact
sarath.menon@ruhr-uni-bochum.de to enquire about commercial use rights.

More information about the program can be found in:
Menon, Sarath, Yury Lysogorskiy, Jutta Rogal, and Ralf Drautz.
"Automated Free Energy Calculation from Atomistic Simulations." Physical Review Materials 5(10), 2021
DOI: 10.1103/PhysRevMaterials.5.103801

For more information contact:
sarath.menon@ruhr-uni-bochum.de
"""
"""pylammpsmpi-backed LAMMPS runner (the optional ``library`` execution mode).

``LibraryRunner`` drives a live in-memory LAMMPS session through
:mod:`pylammpsmpi` instead of segmenting scripts for an external binary.
Commands are forwarded immediately, so :meth:`sync` is a no-op -- there is no
segmentation, no restart replay, and no ``.seg*`` files.  Data accessors
currently read the same files the executable backend does; because every read
goes through the runner they can later be swapped for direct library calls
without touching driver code.

pylammpsmpi is an optional dependency (``pip install calphy[library]``) and is
imported lazily inside :class:`LibraryRunner` -- importing this module (or any
other part of calphy) never requires it.
"""
import os
import logging
import threading

from calphy.runner import BaseRunner, _normalize_cmdargs

logger = logging.getLogger(__name__)

# ``threading._register_atexit`` is private, but it is the *only* hook that runs
# early enough to matter here.  ``atexit`` callbacks run **after**
# ``threading._shutdown`` has already joined every non-daemon thread, so an
# ``atexit`` hook that shuts an executor down never gets the chance to run: the
# interpreter is already blocked joining the very threads the hook would
# release.  ``threading._register_atexit`` callbacks run *inside*
# ``threading._shutdown``, before that join.  This is exactly why the stdlib's
# own ``concurrent.futures.thread`` uses it (``Lib/concurrent/futures/thread.py``)
# rather than ``atexit``.  Degrade gracefully if a future CPython drops it --
# the explicit ``close()`` calls on the success path are unaffected.
_register_atexit = getattr(threading, "_register_atexit", None)
if _register_atexit is None:  # pragma: no cover - depends on CPython internals
    logger.debug(
        "threading._register_atexit is unavailable; a LAMMPS session leaked by "
        "an escaping exception will not be closed at interpreter shutdown."
    )


class LibraryRunner(BaseRunner):
    """Drives a live ``pylammpsmpi.LammpsLibrary`` session.

    The raw library handle is exposed as :attr:`lmp` so future backend-specific
    accessors (direct thermo/structure reads) can build on it.
    """

    def __init__(self, *, cores, cmdargs, directory):
        super().__init__(directory)
        try:
            from pylammpsmpi import LammpsLibrary
        except ImportError as exc:
            raise ImportError(
                "execution_mode 'library' requires pylammpsmpi, which is an "
                "optional dependency. Install it with `pip install "
                "calphy[library]` (the `lammps` python module must also be "
                "importable, e.g. from the conda-forge lammps package), or "
                "remove execution_mode from the input file to use the default "
                "executable mode."
            ) from exc

        self.cores = cores
        self._log_index = 0
        self._closed = False
        cmdargs = _normalize_cmdargs(cmdargs)
        # Suppress LAMMPS stdout; Python logging handles screen output.
        if "-screen" not in cmdargs:
            cmdargs.extend(["-screen", "none"])
        # Redirect the session log to a known scratch name so rotate_logs can
        # collect per-stage logs (LAMMPS would otherwise write log.lammps).
        if "-log" not in cmdargs:
            cmdargs.extend(["-log", self._log_name(0)])
        self.cmdargs = cmdargs

        self.lmp = LammpsLibrary(
            cores=cores, working_directory=directory, cmdargs=cmdargs
        )
        self._register_shutdown_hook()
        self._activate_mliap()

    def _register_shutdown_hook(self):
        """Make sure the session is closed even if an exception escapes.

        pylammpsmpi drives LAMMPS through executorlib, whose task threads are
        *non-daemon* and block on ``queue.get()`` until the executor is shut
        down -- which only happens via :meth:`close`.  If an exception escapes
        between creating a runner and closing it, ``close()`` is never reached,
        those threads never return, and ``threading._shutdown`` blocks joining
        them forever: the process never exits and keeps its whole allocation
        until walltime (see ICAMS/calphy issue on the 48 h hang).

        The hook keeps a *strong* reference to the bound method on purpose.  A
        ``weakref`` would be collected in precisely the situation this guards
        against: once the exception has propagated out and its traceback is
        released, the driver frame holding ``lmp`` is gone and the runner
        becomes collectable while executorlib's threads are still blocked --
        the hook would then find a dead referent and the hang would come back.
        The cost of the strong reference is one *already closed* runner per
        stage (a dozen or so per job), which is negligible; :meth:`close` is
        idempotent via ``self._closed``, so the hook is a no-op for every
        runner that was closed normally.
        """
        if _register_atexit is None:
            return
        _register_atexit(self._close_at_shutdown)

    def _close_at_shutdown(self):
        """Shutdown-hook wrapper around :meth:`close`; never raises."""
        if self._closed:
            return
        logger.warning(
            "LAMMPS session in %s was not closed explicitly; closing it at "
            "interpreter shutdown so the process can exit.", self.directory,
        )
        try:
            self.close()
        except Exception:  # pragma: no cover - best effort at shutdown
            logger.exception("failed to close the LAMMPS session at shutdown")

    def _activate_mliap(self):
        """Register the mliappy coupling in the live session when available.

        Ported from upstream (ICAMS/calphy#260): python-coupled ML-IAP models
        need an explicit activation call on the library. A binary driven by
        ExecutableRunner instead loads ML-IAP models through its own embedded
        python, so this is library-mode only. The activation proxies only
        exist in pylammpsmpi > 0.4.1 (the calphy[library] pin); on older
        versions warn and skip instead of crashing the fresh session.
        """
        try:
            import lammps.mliap  # noqa: F401
        except ImportError:
            return
        name = (
            "activate_mliappy_kokkos"
            if "-k" in self.cmdargs or "-kokkos" in self.cmdargs
            else "activate_mliappy"
        )
        activate = getattr(self.lmp.lmp, name, None)
        if activate is None:
            logger.warning(
                "lammps.mliap is available but this pylammpsmpi provides no "
                "%s (needs pylammpsmpi > 0.4.1); python-coupled ML-IAP "
                "models will not work in this session.", name,
            )
            return
        activate()

    @staticmethod
    def _log_name(k):
        return "calphy.live.%d.log" % k

    # -- backend contract ----------------------------------------------------- #
    def _dispatch(self, cmd):
        """Forward the validated command to the live session immediately."""
        self.lmp.command(cmd)

    def sync(self):
        """No-op: the session is live and LAMMPS flushes fix output as it runs."""

    def close(self):
        """End the pylammpsmpi session (flushes and closes the LAMMPS log)."""
        if not self._closed:
            self.lmp.close()
            self._closed = True

    def rotate_logs(self, stage_name):
        """Move the log written since the last rotation to ``<stage_name>.log.lammps``.

        On a live session, switching the LAMMPS ``log`` target closes the current
        scratch log so it can be renamed; the drivers however always rotate right
        after :meth:`close`, where the scratch log is already complete and no
        command can (or need) be sent.  The ``log`` command is internal
        bookkeeping, not part of the logical command stream, so it bypasses
        :meth:`command`.
        """
        self._log_index += 1
        if not self._closed:
            self.lmp.command("log %s" % self._log_name(self._log_index))
        prev = os.path.join(self.directory, self._log_name(self._log_index - 1))
        out = os.path.join(self.directory, "%s.log.lammps" % stage_name)
        if os.path.exists(prev):
            os.replace(prev, out)
        else:
            open(out, "w").close()
