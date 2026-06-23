"""The abstract Backend interface driven by the orchestration core.

A backend supplies everything that is specific to *what* is being built and
*how* a single step runs, while the core owns the generic orchestration
(component resolution, topological ordering, task elaboration, the selector
grammar, the logging/stamp execution loop, and the manifest workflow).

The two implementations are:
  * AompBackend     - per-component ``build_<name>.sh`` scripts (a CUDF config).
  * TheRockBackend  - a CMake super-build introspected to ``subproject_map.json``.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

from .model import Config, RawTask, Task


class Backend(ABC):
    #: short identifier, e.g. "aomp" / "therock"
    name: str = "backend"

    # --- configuration & environment ------------------------------------- #
    @abstractmethod
    def load_config(self, args: argparse.Namespace) -> Config:
        """Build the component graph (packages/features/request)."""

    @abstractmethod
    def config_name(self, args: argparse.Namespace) -> str:
        """A short name for the active config (used for manifest filenames)."""

    @abstractmethod
    def build_child_env(self, args: argparse.Namespace) -> dict[str, str]:
        """Construct the isolated environment for child build steps."""

    @abstractmethod
    def discover_env(self, env: dict[str, str]) -> dict[str, str]:
        """Resolve build directories. Must include "BUILD_DIR"; may include
        install-dir / symlink keys used by the -C/--clean install_clean task."""

    # --- task elaboration ------------------------------------------------- #
    @abstractmethod
    def component_configs(self, comp: str, env: dict[str, str]) -> list[str]:
        """Advertised build configs/variants for a component (e.g. ["default",
        "asan"]). An empty list means the component is config-less."""

    @abstractmethod
    def list_component_tasks(
        self, comp: str, env: dict[str, str]
    ) -> list[RawTask]:
        """The ordered raw tasks for a component: (action, cfgname, payload).
        Config-less init/fini tasks use cfgname=None."""

    def trailing_tasks(
        self, components: list[str], env: dict[str, str]
    ) -> list[Task]:
        """Whole-build pseudo-tasks appended after all per-component tasks.

        These are not tied to a single component (so they are not produced by
        list_component_tasks): e.g. TheRock's whole-tree "assemble the combined
        dist tree" and "install to the final dir" steps, which are deliberately
        not per-subproject. Returned tasks are fully formed and run via
        task_command like any other. The default is no trailing tasks."""
        return []

    # --- execution -------------------------------------------------------- #
    @abstractmethod
    def task_command(
        self, task: Task, env: dict[str, str]
    ) -> tuple[list[str], dict[str, str]]:
        """The argv and any extra environment for running one (non-builtin)
        task. The core handles logging, stamps and BUILD_TYPE injection."""

    # --- manifest / clean (optional; sensible defaults) ------------------- #
    def component_src_dir(self, comp: str, env: dict[str, str]) -> str:
        """Source directory for a component, for the git-fingerprint manifest."""
        return ""

    def external_repos(self, env: dict[str, str]) -> dict[str, str]:
        """Non-component repos to record in the manifest: name -> abs path."""
        return {}

    def floating_components(self) -> set[str]:
        """Components never rolled back on manifest import (track HEAD)."""
        return set()

    def install_clean_task(self, env_info: dict[str, str]) -> Task | None:
        """The -C/--clean pseudo-task (wipe install dir), or None if unsupported."""
        return None

    def list_features(self, env: dict[str, str]) -> list[dict] | None:
        """Rows for the `list-features` selector, or None if the backend has no
        feature concept. Each row is a dict with at least 'name' and 'enabled';
        the TheRock backend lists its THEROCK_ENABLE_* features. Default: None."""
        return None

    def built_components(self, env: dict[str, str]) -> set[str] | None:
        """Components the backend considers already built (so they can be pinned
        / shown as done), or None if the backend has no such notion.

        The TheRock backend returns the components with a valid (non-empty)
        stage dir -- i.e. exactly the ones buildctl.py would mark prebuilt.
        `list` uses this to show a [pinned] suffix and to render an already-built
        component's checkbox as done. Default: None (e.g. the AOMP backend)."""
        return None

    def provision_preconfig(self, args: argparse.Namespace) -> int | None:
        """Standalone source provisioning that must run *before* load_config
        (and therefore before any configure/source fetch). Returns None when
        there is nothing to do (the run continues normally), or an exit code to
        return immediately. Used by TheRock's --migrate-aomp, which seeds the
        submodule slots from an AOMP checkout before the configure that would
        otherwise fetch them. The default does nothing."""
        return None

    def provision_sources(
        self, args: argparse.Namespace, env: dict[str, str],
        components: list[str],
    ) -> int:
        """Pre-build source provisioning hook (--clone / --therock-symlinks /
        --migrate-aomp). Runs once after component resolution, before any task
        elaboration. `components` is the resolved, ordered component list (used
        e.g. to decide whether rocmlibs need cloning). Returns an exit code:
        non-zero aborts the run. The default does nothing."""
        return 0

    def prepare_run(
        self, selected_comps: set[str], env: dict[str, str],
        args: argparse.Namespace,
    ) -> None:
        """Hook invoked once after task selection, before any task runs.

        `selected_comps` is the set of component names appearing in the tasks
        about to run. A backend may use this to adjust build state for the
        upcoming run -- e.g. the TheRock backend marks out-of-scope components
        as prebuilt (so a focused/incremental subset build, and any later
        whole-tree install, does not rebuild dependents the user is not working
        on). Honors args.dry_run. The default does nothing."""
        return None

    # --- sharding (optional) --------------------------------------------- #
    def shard_run_lengths(
        self, tasks: list[Task], env: dict[str, str]
    ) -> list[int] | None:
        """Preferred contiguous run boundaries for --shard, or None.

        The default (None) makes --shard split the dependency-ordered task list
        into N balanced segments by task count. A backend may instead return the
        lengths of contiguous runs (summing to len(tasks)) that the shard cut
        points should snap to -- e.g. TheRock groups tasks by build *stage* so a
        shard boundary never splits a stage. Returning a grouping never reorders
        tasks; it only biases where the N cuts fall."""
        return None
