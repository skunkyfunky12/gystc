"""One answer to "does this path stay inside that folder?".

Paths reach GYSTC from outside in several ways: a curation proposals file,
argv of the maintenance scripts, the ``--selfcheck`` report flag. Any of them
can be written by an agent rather than a person, and a single ``..`` would then
point a write or a read anywhere the process can reach. The CLI entry points
(curation apply/preview, --selfcheck, bundle_model --out, check_secrets) call
``resolve_within``. The MCP write paths (tools/store.py, tools/versioning.py,
tools/classify_tool.py) and the dashboard's /api/vault handler still carry
their own ``resolve().is_relative_to()`` checks -- same rule, older code; new
guards belong here.

The check follows the canonical form: resolve first (``..``, absolute paths and
symlinks all collapse into where the path really points), then compare against
the resolved base plus a separator, so ``vault-evil`` never passes as ``vault``.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_within(path: str | os.PathLike[str], base: str | os.PathLike[str]) -> Path:
    """Resolve *path* relative to *base*; raise ``ValueError`` if it leaves *base*.

    Returns the resolved absolute path. *base* itself counts as inside. The
    target does not need to exist yet, so this also guards files about to be
    created.
    """
    base_dir = os.path.realpath(base)
    resolved = os.path.realpath(os.path.join(base_dir, path))
    # A drive or filesystem root already ends in a separator.
    prefix = base_dir if base_dir.endswith(os.sep) else base_dir + os.sep
    if resolved != base_dir and not resolved.startswith(prefix):
        raise ValueError(f"path escapes {base_dir}: {os.fspath(path)}")
    return Path(resolved)
