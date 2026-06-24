"""Thin adapter over TheRock's own ``build_topology`` module.

TheRock already ships a static, checked-in description of its build topology in
``BUILD_TOPOLOGY.toml`` plus an importable parser
(``build_tools/_therock_utils/build_topology.py``) that knows the CI/CD build
*stages*, the artifact groups and their dependencies, and the git submodules
each stage needs. This adapter loads that parser from a TheRock checkout (when
available) so the orchestrator can drive stage-based shards: each shard is a
build stage, and import/build/export operate on its artifacts and subprojects.

Everything here degrades gracefully: if the checkout, the TOML, or a suitable
TOML parser (``tomllib`` on 3.11+, else ``tomli``) is missing, the loader
returns ``None`` and the caller reports that sharding is unavailable.
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


def stage_names(topo: Any) -> list[str]:
    """The build stages in dependency (build) order, or [] on any failure."""
    try:
        return list(topo.get_build_order())
    except Exception:
        try:
            return [s.name for s in topo.get_build_stages()]
        except Exception:
            return []


def stage_submodules(topo: Any, stage: str) -> list[str]:
    """The git submodule names a stage needs (its source sets), or []."""
    try:
        return [
            getattr(s, "name", "")
            for s in topo.get_submodules_for_stage(stage)
            if getattr(s, "name", "")
        ]
    except Exception:
        return []


def stage_artifact_groups(topo: Any, stage: str) -> list[str]:
    """The artifact-group names a stage owns, or []."""
    try:
        return list(topo.build_stages[stage].artifact_groups)
    except Exception:
        return []


def produced_artifact_names(topo: Any, stage: str) -> set[str]:
    """Artifact names produced by a stage (what `export` pushes), or set()."""
    try:
        return set(topo.get_produced_artifacts(stage))
    except Exception:
        return set()


def inbound_artifact_names(topo: Any, stage: str) -> set[str]:
    """Artifact names a stage consumes from upstream stages, or set()."""
    try:
        return set(topo.get_inbound_artifacts(stage))
    except Exception:
        return set()


def group_names(topo: Any) -> list[str]:
    """The artifact groups in dependency (build) order, or [] on any failure.

    Topologically sorts the ``artifact_groups`` over their ``artifact_group_deps``
    so a group always follows the groups it consumes. Groups are the shard unit
    in group-based sharding.
    """
    try:
        groups = topo.artifact_groups
    except Exception:
        return []
    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        group = groups.get(name)
        if group is not None:
            for dep in getattr(group, "artifact_group_deps", []):
                if dep in groups:
                    visit(dep)
        order.append(name)

    for name in groups:
        visit(name)
    return order


def group_source_sets(topo: Any, group: str) -> list[str]:
    """The source-set names a group draws from, or [].

    These feed ``fetch_sources.py --source-sets`` so a group shard only checks
    out the submodules it needs.
    """
    try:
        return list(topo.artifact_groups[group].source_sets)
    except Exception:
        return []


def group_submodules(topo: Any, group: str) -> list[str]:
    """The git submodules a group's source sets contain, or []."""
    try:
        subs: list[str] = []
        for set_name in topo.artifact_groups[group].source_sets:
            source_set = topo.source_sets.get(set_name)
            if source_set is None:
                continue
            for submodule in source_set.submodules:
                name = getattr(submodule, "name", "")
                if name and name not in subs:
                    subs.append(name)
        return subs
    except Exception:
        return []


def group_produced_names(topo: Any, group: str) -> set[str]:
    """Artifact names a group produces (what `export` pushes), or set()."""
    try:
        return {a.name for a in topo.get_artifacts_in_group(group)}
    except Exception:
        return set()


def group_inbound_names(topo: Any, group: str) -> set[str]:
    """Artifact names a group consumes from its dependency groups, or set().

    Mirrors ``BuildTopology.get_inbound_artifacts`` at single-group granularity:
    the artifacts of the group's direct ``artifact_group_deps`` plus the
    transitive ``artifact_deps`` closure, minus the group's own products.
    """
    try:
        group_obj = topo.artifact_groups[group]
    except Exception:
        return set()
    inbound: set[str] = set()
    collect = getattr(topo, "_collect_transitive_artifact_deps", None)
    for dep_group in getattr(group_obj, "artifact_group_deps", []):
        try:
            dep_artifacts = topo.get_artifacts_in_group(dep_group)
        except Exception:
            continue
        for artifact in dep_artifacts:
            inbound.add(artifact.name)
            if collect:
                collect(artifact.name, inbound)
    try:
        own = topo.get_artifacts_in_group(group)
    except Exception:
        own = []
    for artifact in own:
        for dep in getattr(artifact, "artifact_deps", []):
            inbound.add(dep)
            if collect:
                collect(dep, inbound)
    inbound -= {a.name for a in own}
    return inbound


def group_dependencies(topo: Any, group: str) -> list[str]:
    """The group's direct dependency groups, in build order, or [].

    These are the shards to import (``--import-shard``) before building this one.
    """
    try:
        deps = set(topo.artifact_groups[group].artifact_group_deps)
    except Exception:
        return []
    return [g for g in group_names(topo) if g in deps]


def stage_dependencies(topo: Any, stage: str) -> list[str]:
    """The upstream stages this stage depends on, in build order, or [].

    A stage depends on another when one of its artifact groups lists an
    ``artifact_group_deps`` entry that the other stage owns. These are exactly
    the shards whose artifacts must be imported (``--import-shard``) before this
    one can build. Mirrors the direct-dependency logic of ``get_build_order``.
    """
    try:
        dep_groups: set[str] = set()
        for group_name in topo.build_stages[stage].artifact_groups:
            group = topo.artifact_groups.get(group_name)
            if group is not None:
                dep_groups.update(group.artifact_group_deps)
    except Exception:
        return []
    deps: list[str] = []
    for other in stage_names(topo):
        if other == stage:
            continue
        try:
            owned = set(topo.build_stages[other].artifact_groups)
        except Exception:
            continue
        if owned & dep_groups:
            deps.append(other)
    return deps


def submodule_stage_rank(topo: Any) -> dict[str, int]:
    """Map each submodule name to the rank of the earliest stage that needs it.

    The rank is the index of the stage in ``get_build_order()`` (a topological
    order of stages by artifact-group dependencies). A submodule used by several
    stages is assigned its earliest (smallest-rank) stage, which is where it is
    first built.
    """
    rank: dict[str, int] = {}
    for idx, stage_name in enumerate(stage_names(topo)):
        for name in stage_submodules(topo, stage_name):
            if name not in rank:
                rank[name] = idx
    return rank


def subproject_stage(topo: Any) -> dict[str, str]:
    """Map each submodule name to the *name* of the earliest stage that needs it.

    The string-keyed counterpart of ``submodule_stage_rank``: a submodule used
    by several stages is assigned its earliest (build-order) stage, which is
    where it is first built. Used to select the orchestrator's per-subproject
    tasks that belong to a given build shard.
    """
    order = stage_names(topo)
    stage: dict[str, str] = {}
    for stage_name in order:
        for name in stage_submodules(topo, stage_name):
            if name not in stage:
                stage[name] = stage_name
    return stage
