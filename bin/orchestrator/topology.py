"""Thin adapter over TheRock's own ``build_topology`` module.

TheRock already ships a static, checked-in description of its build topology in
``BUILD_TOPOLOGY.toml`` plus an importable parser
(``build_tools/_therock_utils/build_topology.py``) that knows the CI/CD build
*stages*, the artifact groups and their dependencies, and the git submodules
each stage needs. This adapter loads that parser from a TheRock checkout (when
available) so the orchestrator can align ``--shard`` cut points to stage
boundaries.

Everything here degrades gracefully: if the checkout, the TOML, or a suitable
TOML parser (``tomllib`` on 3.11+, else ``tomli``) is missing, the loader
returns ``None`` and the caller falls back to a plain contiguous partition.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any


def load_build_topology(therock_dir: str) -> Any | None:
    """Return a ``BuildTopology`` for ``therock_dir`` or ``None``.

    ``therock_dir`` is the root of a TheRock checkout (``SROCK_THEROCK_DIR``).
    Failure for any reason (no checkout, no TOML, no TOML parser, an import or
    parse error) yields ``None`` so sharding can fall back to count-based.
    """
    if not therock_dir:
        return None
    toml_path = os.path.join(therock_dir, "BUILD_TOPOLOGY.toml")
    module_path = os.path.join(
        therock_dir, "build_tools", "_therock_utils", "build_topology.py"
    )
    if not (os.path.isfile(toml_path) and os.path.isfile(module_path)):
        return None
    # Make `import _therock_utils...` resolvable for build_topology's own
    # relative-free imports, then load the module by path so we do not depend on
    # TheRock being on PYTHONPATH.
    build_tools = os.path.join(therock_dir, "build_tools")
    if build_tools not in sys.path:
        sys.path.insert(0, build_tools)
    try:
        spec = importlib.util.spec_from_file_location(
            "_therock_build_topology", module_path
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BuildTopology(toml_path)
    except Exception:
        # tomllib/tomli missing, parse error, schema drift, etc. -> no topology.
        return None


def submodule_stage_rank(topo: Any) -> dict[str, int]:
    """Map each submodule name to the rank of the earliest stage that needs it.

    The rank is the index of the stage in ``get_build_order()`` (a topological
    order of stages by artifact-group dependencies). A submodule used by several
    stages is assigned its earliest (smallest-rank) stage, which is where it is
    first built.
    """
    rank: dict[str, int] = {}
    try:
        order = topo.get_build_order()
    except Exception:
        return rank
    for idx, stage_name in enumerate(order):
        try:
            submodules = topo.get_submodules_for_stage(stage_name)
        except Exception:
            continue
        for sub in submodules:
            name = getattr(sub, "name", None)
            if name and name not in rank:
                rank[name] = idx
    return rank
