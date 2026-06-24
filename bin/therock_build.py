#!/usr/bin/env python3
"""therock_build.py - introspectable orchestrator for TheRock's super-build.

A thin entry point that wires the TheRock backend into the shared, backend-
agnostic orchestration core (the ``orchestrator`` package). It exposes the same
workflow as ``aomp_build.py`` -- resolve a component set, topologically order
it, elaborate fine-grained tasks, list/run them by number/range/glob/continue,
write per-task logs and completion stamps, shard the work, and export/import a
git-fingerprint manifest -- but the components are TheRock subprojects taken
from the introspection map (``<build>/subproject_map.json``) and each task runs
as ``ninja -C <build> <subproject>+<action>``.

Bootstrap (clone, venv, fetch_sources, configure with -DTHEROCK_INTROSPECTION=ON)
is delegated to the existing ``srock-bin`` scripts; pass --reconfigure to drive
it. Stdlib only.
"""

from __future__ import annotations

import sys

from orchestrator import core
from orchestrator.therock_backend import DEFAULT_CHILD_PATH


def main(argv: list[str]) -> int:
    core.PROG = "therock_build"
    parser = core.build_arg_parser(
        "therock_build.py", None,
        description="Introspectable orchestrator for TheRock's CMake super-build "
                    "(srock). The build *scope* is selected with --add "
                    "(--add all | --add all-debug; minimal is the default; add "
                    "sysdeps to bundle system deps). The *source config* (which "
                    "TheRock branches to build) is selected with -c/--config "
                    "(amd-staging default; develop for native upstream; see "
                    "`list-configs`).",
        inherit_path_note=DEFAULT_CHILD_PATH,
    )
    core.add_backend_options(parser, default_backend="therock")
    args = parser.parse_args(argv)
    return core.run(args, core.make_backend(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
