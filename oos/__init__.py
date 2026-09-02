"""Multi-objective on-orbit-servicing mission planning."""

import os
import tempfile

# Numba caches compiled functions next to the source by default. When the
# package sits on a network or FUSE-mounted folder - a synced drive, or a
# remote bridge - that cache thrashes: numba writes, renames and deletes cache
# files, the mount refuses the deletes, and every import recompiles from
# scratch while __pycache__ fills with .fuse_hidden stubs. Compilation goes
# from about 2 seconds to minutes.
#
# Point the cache at local disk unless the caller has already chosen somewhere.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "oos-numba")
    if os.path.isdir(os.path.expanduser("~"))
    else os.path.join(tempfile.gettempdir(), "oos-numba"),
)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
