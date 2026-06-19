#!/usr/bin/env python3
"""aomp_build.py - unified, introspectable AOMP build orchestrator.

This driver pulls the taskified per-component build scripts (build_<name>.sh,
each speaking the command_dispatcher interface from aomp_utils) into a single
workflow:

  * resolve a component set from a CUDF-style config (configs/aomp.cudf),
    honoring --add/--remove of components and feature aliases;
  * topologically order the components (deterministic, declaration-order
    tie-break, which reproduces the canonical build_aomp.sh ordering);
  * elaborate fine-grained tasks by querying each component's `list`,
    filtered by the selected build variant(s);
  * list those tasks, or run them individually, by number/range, from a
    point onward (`continue`), by glob, or all at once - mirroring the
    `amd-build` selector grammar;
  * write per-task numbered logs and tail the log on failure;
  * export/import a JSON version manifest (per-component git fingerprint) for
    reproducible builds.

The orchestration core is backend-agnostic (see the `orchestrator` package);
this entry point wires in the AOMP backend. Stdlib only.
"""

from __future__ import annotations

import sys

from orchestrator import core
from orchestrator.aomp_backend import DEFAULT_CHILD_PATH, DEFAULT_CONFIG, AompBackend


def main(argv: list[str]) -> int:
    core.PROG = "aomp_build"
    parser = core.build_arg_parser(
        "aomp_build.py", DEFAULT_CONFIG,
        description="Unified AOMP component build orchestrator.",
        inherit_path_note=DEFAULT_CHILD_PATH,
    )
    args = parser.parse_args(argv)
    return core.run(args, AompBackend())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
