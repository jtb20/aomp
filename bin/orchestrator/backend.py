"""The abstract Backend interface driven by the orchestration core.

A backend supplies everything that is specific to *what* is being built and
*how* a single step runs, while the core owns the generic orchestration
(component resolution, topological ordering, task elaboration, the selector
grammar, the logging/stamp execution loop, and the manifest workflow).

The two implementations are:
  * AompBackend     - per-component ``build_<name>.sh`` scripts (a CUDF config).
  * TheRockBackend  - a CMake super-build introspected to ``subprojects_map.json``.
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
