"""Backend-independent data types for the build orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Package:
    name: str
    depends: list[str] = field(default_factory=list)
    # Dependencies that do not affect build *ordering* but do mean "rebuild me
    # if this changes" (e.g. a tool that links a library at install/runtime).
    # Kept separate from `depends` so topological ordering stays minimal (and
    # acyclic), while reverse-dependency closure (--rdeps) can consider both.
    runtime_depends: list[str] = field(default_factory=list)
    xdir: str = "."
    order: int = 0  # declaration order, used as topo-sort tie-break


@dataclass
class Config:
    packages: dict[str, Package] = field(default_factory=dict)
    features: dict[str, list[str]] = field(default_factory=dict)
    request: list[str] = field(default_factory=list)


@dataclass
class Task:
    comp: str
    action: str          # e.g. "cmake" (without any "task_" prefix)
    cfgname: str | None  # build config/variant, e.g. "default" / "asan"
    single_config: bool = False  # component advertises only "default"
    # Builtin (orchestrator-run) tasks have no backing backend command.
    # Currently only "install_clean": wipe the install dir. `targets` then
    # holds the paths it operates on ([install_dir, symlink]).
    builtin: str | None = None
    targets: list[str] = field(default_factory=list)
    # Backend-opaque execution payload. The backend that produced the task
    # knows how to turn this into a command via Backend.task_command(). For the
    # AOMP backend this is {"script": ..., "script_args": [...]}; for TheRock it
    # is {"bin": ..., "target": ..., "env": {...}}.
    payload: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        # "component/variant/stage" for config-bearing tasks. The variant
        # segment is dropped (-> "component/stage") for the config-less
        # init/fini tasks (precheck/patch/unpatch) and for components whose only
        # advertised config is "default".
        if self.cfgname and not self.single_config:
            return f"{self.comp}/{self.cfgname}/{self.action}"
        return f"{self.comp}/{self.action}"


# A "raw" task as produced by a backend before variant filtering / Task
# construction: (action, cfgname, payload).
RawTask = tuple[str, "str | None", dict]
