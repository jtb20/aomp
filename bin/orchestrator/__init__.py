"""Build orchestration core shared by the AOMP and TheRock backends.

The orchestrator is split into:

  * ``model``    - backend-independent data types (Config, Package, Task).
  * ``backend``  - the abstract Backend interface the core drives.
  * ``core``     - generic orchestration: component resolution, topological
                   ordering, task elaboration, the selector grammar, the
                   execution loop (logs + stamps), and the version manifest.
  * ``aomp_backend``     - the AOMP backend (the ``build_<name>.sh`` contract).
  * ``therock_backend``  - the TheRock backend (CMake super-build introspection).

``aomp_build.py`` is a thin entry point that wires the AOMP backend into the
core; behavior is identical to the original single-file orchestrator.
"""
