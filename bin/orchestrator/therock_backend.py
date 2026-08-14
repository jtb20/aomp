"""TheRock backend: drive TheRock's CMake super-build via introspection.

TheRock is a single CMake super-build that assembles the whole ROCm stack from
git submodules. Rather than a per-component script contract (the AOMP backend),
this backend reads the build graph from the introspection JSON emitted by
TheRock when configured with ``-DTHEROCK_INTROSPECTION=ON`` (PR #1234):

    <build>/subproject_map.json

mapping each subproject to its source/build dirs, build- and runtime-deps, and
the ninja action targets ``<subproject>+{configure,build,stage,dist,expunge}``.
Each subproject+action becomes an orchestrator task that runs as
``ninja -C <build> <subproject>+<action>``.

Bootstrap, environment, and the cmake configuration are reused from the
established ``srock-bin`` integration in this repo (``srock_common_vars`` /
``setup_srock.sh``) rather than reimplemented, so the orchestrator and the
existing two-script srock workflow stay consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess

from . import core, source_config, source_layout, topology
from .backend import Backend
from .model import Config, Package, RawTask, Task

# bin/ holds this package; srock-bin is a sibling of bin/ at the repo root.
BIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BIN_DIR)
SROCK_BIN_DIR = os.path.join(REPO_ROOT, "srock-bin")
SROCK_COMMON_VARS = os.path.join(SROCK_BIN_DIR, "srock_common_vars")
SETUP_SROCK = os.path.join(SROCK_BIN_DIR, "setup_srock.sh")
# Builds the cmake/ninja prerequisite toolchain (it self-checks and is a no-op
# when already built). Run as the leading `therock/prereq` task.
BUILD_CMAKE = os.path.join(SROCK_BIN_DIR, "build_cmake.sh")

# Self-contained helper that filters the shared artifact store to a producer
# shard's artifacts and runs buildctl.py bootstrap (used by --import-shard).
SHARD_ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "shard_artifacts.py")

# PR #1234's introspection cmake, bundled here so a stock TheRock checkout
# (which does not carry it) can still be introspected: the bootstrap copies it
# into <TheRock>/cmake/ and appends an include + invocation to CMakeLists.txt.
INTROSPECTION_CMAKE = os.path.join(
    SROCK_BIN_DIR, "therock_subproject_introspection.cmake"
)
INTROSPECTION_MARKER = "srock orchestrator: build introspection"

# The introspection JSON (singular "subproject"), written to CMAKE_BINARY_DIR by
# cmake/therock_subproject_introspection.cmake.
SUBPROJECT_MAP = "subproject_map.json"

# Artifact -> composing-subproject map, written alongside subproject_map.json by
# the same introspection module. Maps each topology artifact name to the cmake
# subprojects (SUBPROJECT_DEPS) that build it; the backend joins this with
# BUILD_TOPOLOGY.toml's artifact->group relation to map subprojects to artifact
# groups (the shard unit). Absent on checkouts whose therock_artifacts.cmake
# lacks the recorded property, in which case group-shard subproject selection is
# unavailable (the backend degrades gracefully).
ARTIFACT_MAP = "artifact_map.json"

# Marker guarding the one-line property injection into therock_artifacts.cmake
# (see _inject_artifact_deps_property).
ARTIFACT_DEPS_MARKER = "srock orchestrator: artifact subproject deps"

# The build type a component takes when --build-type does not name one. Dropping
# a previously-set --build-type therefore reverts that scope to this default
# (rather than retaining the cached value), which is treated as a configuration
# change requiring --reconfigure.
DEFAULT_BUILD_TYPE = "Release"

# Whether TheRock builds each component's test suites (THEROCK_BUILD_TESTING).
# Deliberately the opposite of TheRock's own default, which follows CTest's
# BUILD_TESTING and is therefore ON: enabling testing brings in test-only
# subprojects -- rocPRIM_tests is the notable one -- whose compile time dwarfs
# the libraries they test, which is painful against a Debug compiler. Opt back in
# with --build-tests.
DEFAULT_BUILD_TESTING = False

# Companion catalog of THEROCK_ENABLE_* features, written alongside it by the
# same introspection module. Maps each feature (group option or per-artifact
# feature) to its enabled state, description, requires list, and whether it is a
# therock_add_feature() feature (vs a coarse group option).
FEATURE_MAP = "feature_map.json"

# --add tokens that are configure toggles handled specially (not as generic
# THEROCK_ENABLE_<X> features): all/all-debug drive SROCK_CONFIG, sysdeps drives
# THEROCK_BUNDLE_SYSDEPS (see build_child_env / _srock_config / _wants_sysdeps).
_SPECIAL_TOGGLES = frozenset({"all", "all-debug", "sysdeps"})

# Default config name (mirrors srock_common_vars SROCK_CONFIG default).
DEFAULT_CONFIG = "minimal"

# Marker file in the TheRock checkout recording the *source config* its sources
# are currently checked out for (see source_config.py). The shared checkout is
# switched in place between configs; a mismatch between this marker and the
# requested config means the sources must be switched (branch checkout + submodule
# resync) on the next reconfigure (see _ensure_subproject_map). Absent on a legacy
# checkout predating source configs (treated as "unknown", no forced switch).
SOURCE_CONFIG_MARKER = ".srock-source-config"

# Forward build pipeline, in execution order. Deliberately excluded:
#   * "dist"    -- in TheRock, a subproject's "+dist" target is wired into
#                  whole-*distribution* assembly (artifacts/distributions add
#                  themselves as dependencies of it), so `ninja <sub>+dist`
#                  builds unrelated subprojects across the runtime-dependency
#                  graph (e.g. dist-ing a base sysdep drags in llvm). That is a
#                  whole-tree packaging step, not a per-component one. The
#                  per-subproject "+stage" already populates the combined local
#                  dist dir with this+transitive stage installs (a file copy, no
#                  extra builds), so an in-order build still produces a usable
#                  staged tree; assemble named distributions / install with a
#                  whole-tree step (e.g. srock-bin/build_srock.sh's `ninja
#                  install`).
#   * "expunge" -- destructive clean; would wipe a subproject mid-build. Use
#                  `ninja <subproject>+expunge` or -C/--clean (whole install).
# The remaining actions ("+configure"/"+build"/"+stage") depend only on a
# subproject's genuine build prerequisites (its own configure->build, its
# build-deps' configure, and its compiler toolchain), not the full runtime-dep
# closure -- so running them respects real ordering without doing unasked work.
FORWARD_ACTIONS = ("configure", "build", "stage")

# Full per-component action menu (lifecycle order) exposed by -a/--all. Beyond
# the default FORWARD_ACTIONS this adds the destructive clean ("expunge", first
# so a full per-component run reads as clean->configure->build->stage->dist) and
# the per-component "dist" (which triggers whole-tree distribution assembly, as
# noted above). Handy for listing the full capability set and for targeted
# selection while untangling a build; not meant for routine full builds.
ALL_ACTIONS = ("expunge", "configure", "build", "stage", "dist")

# Whole-tree pseudo-tasks appended after every per-subproject task (see
# trailing_tasks). These are the install-all counterparts to amd-build's final
# install: "dist" assembles the combined distribution tree under <build>/dist
# (target "therock-dist", the same step CI runs as
# `cmake --build build --target therock-dist`), and "install" copies that tree
# to the final install dir (CMAKE_INSTALL_PREFIX = $SROCK_INSTALL_DIR), exactly
# as srock-bin/build_srock.sh's `ninja install` does. They are whole-tree on
# purpose: unlike per-subproject "+dist", these are where the full ROCm SDK is
# assembled/installed. (comp, action, ninja target)
WHOLE_TREE_TASKS = (
    ("therock", "dist", "therock-dist"),
    ("therock", "install", "install"),
)

# Subprojects whose source is meant to track a branch tip (amd-staging) per the
# srock workflow, so they are not rolled back on manifest import. Names are the
# cmake subproject/target names; unmatched names are simply ignored.
FLOATING_COMPONENTS = {"amd-llvm", "hipify", "spirv-llvm-translator"}

# Same curated identity/locale pass-through as the AOMP backend: build from a
# clean environment so what gets built is under the orchestrator's control.
ENV_PASSTHROUGH = (
    "HOME", "USER", "LOGNAME", "SHELL", "TERM",
    "LANG", "LANGUAGE", "TZ", "TMPDIR",
    "DISPLAY", "XAUTHORITY",
    "SSH_AUTH_SOCK", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY",
)
DEFAULT_CHILD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _clean(value: str) -> str:
    """Normalize a cmake get_target_property() result: unset properties come
    through as the literal "<name>-NOTFOUND" (or "NOTFOUND"); treat as empty."""
    if not value or value == "NOTFOUND" or value.endswith("-NOTFOUND"):
        return ""
    return value


def _clean_list(values: list) -> list[str]:
    out: list[str] = []
    for value in values or []:
        cleaned = _clean(str(value))
        if cleaned:
            out.append(cleaned)
    return out


def _wants_build_tests(args: argparse.Namespace) -> bool:
    """Whether the user asked for test suites via ``--build-tests``."""
    return bool(getattr(args, "build_tests", DEFAULT_BUILD_TESTING))


def _cmake_is_true(value: str) -> bool:
    """Whether a CMakeCache value is true under cmake's ``if()`` rules, limited
    to the constants a boolean cache entry can hold."""
    return value.strip().upper() not in {
        "", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND",
    }


def _wants_sysdeps(args: argparse.Namespace) -> bool:
    """Whether the user opted into bundled system deps via ``--add sysdeps``.

    ``sysdeps`` is a configure toggle surfaced through ``--add`` rather than a
    component group, so it is detected from the raw ``--add`` selectors (which
    may be comma-joined) independent of feature/component resolution."""
    for entry in getattr(args, "add", None) or []:
        if any(tok.strip() == "sysdeps" for tok in str(entry).split(",")):
            return True
    return False


def _normalize_feature(token: str) -> str:
    """Normalize a user --add token to a TheRock feature name: uppercase with
    dashes turned into underscores, so `hipdnn`, `HIPDNN`, and `ml-libs` map to
    `HIPDNN` / `ML_LIBS` (the suffix of the THEROCK_ENABLE_<NAME> cache var)."""
    return token.strip().upper().replace("-", "_")


def _feature_aliases(name: str) -> set[str]:
    """The --add spellings accepted for a feature name (e.g. ML_LIBS ->
    {ML_LIBS, ml_libs, ml-libs}) so resolve_components recognizes the token."""
    lower = name.lower()
    return {name, lower, lower.replace("_", "-")}


def _is_vendored(name: str) -> bool:
    """Whether a subproject is a vendored third-party / system library.

    TheRock names every vendored dependency with the ``therock-`` target
    prefix (e.g. ``therock-boost``, ``therock-zlib``); the real ROCm components
    (amd-llvm, ROCR-Runtime, rocm-core, ...) never use it. Used to seed the
    default request from the real components only (their dependency closure
    then pulls back in the vendored libs that are actually needed)."""
    return name.startswith("therock-")


def _srock_config(args: argparse.Namespace) -> str:
    """The SROCK_CONFIG name to build with.

    For TheRock the config is *not* selected with -c/--config (which is
    unsupported, see _explicit_config); it is a configure toggle surfaced
    through --add: `--add all` or `--add all-debug`, with `minimal` the default.
    all-debug (full, may include failing components) wins over all (full minus
    known-failing)."""
    toks: set[str] = set()
    for entry in getattr(args, "add", None) or []:
        toks.update(t.strip() for t in str(entry).split(","))
    if "all-debug" in toks:
        return "all-debug"
    if "all" in toks:
        return "all"
    return DEFAULT_CONFIG


def _explicit_config(args: argparse.Namespace) -> str | None:
    """A *source config* name the user passed via -c/--config, or None.

    For TheRock, -c/--config selects a *source config* (which TheRock branches to
    build, see source_config.py) -- not the build *scope* (minimal/all/all-debug),
    which is chosen with --add. The aomp_build entry defaults --config to a .cudf
    *path* and the therock_build entry defaults it to None; empty/path/.cudf
    values are the inherited defaults, not an explicit source config name, so they
    return None (falling back to the default source config)."""
    config = getattr(args, "config", None)
    if not config or os.sep in config or config.endswith(".cudf"):
        return None
    return config


def _resolve_source_config(args: argparse.Namespace) -> source_config.SourceConfig:
    """The TheRock source config to build (default amd-staging).

    Resolves the -c/--config name to a SourceConfig. An unknown name warns and
    falls back to the default (so a typo never silently builds the wrong branch
    without notice, but also never hard-fails a build). The build scope (--add)
    is orthogonal and unaffected."""
    name = _explicit_config(args) or source_config.DEFAULT_SOURCE_CONFIG
    try:
        return source_config.load(name)
    except source_config.SourceConfigError as exc:
        core._warn(
            f"{exc}; falling back to default source config "
            f"'{source_config.DEFAULT_SOURCE_CONFIG}'."
        )
        return source_config.load(source_config.DEFAULT_SOURCE_CONFIG)


class TheRockBackend(Backend):
    name = "therock"

    def __init__(self) -> None:
        self._args: argparse.Namespace | None = None
        self._child_env: dict[str, str] | None = None
        self._env_info: dict[str, str] | None = None
        self._cfg: Config | None = None
        # Per-subproject metadata from the introspection JSON.
        self._meta: dict[str, dict] = {}
        # THEROCK_ENABLE_* feature catalog from feature_map.json.
        self._features_meta: dict[str, dict] = {}
        # artifact name -> composing cmake subprojects, from artifact_map.json.
        self._artifact_map: dict[str, list[str]] = {}

    # --- configuration & environment ------------------------------------- #
    def load_config(self, args: argparse.Namespace) -> Config:
        env = self.build_child_env(args)
        info = self.discover_env(env)
        # Resolve requested THEROCK_ENABLE_* features and (when --reconfigure)
        # inject their enable flags into the configure, *before* the (re)config
        # below regenerates the maps.
        self._apply_feature_flags(env, info, args)
        # Likewise translate --build-type into configure-time -D flags (idempotent
        # against the current CMakeCache; only a real change needs --reconfigure).
        self._apply_build_type_flags(env, info, args)
        # THEROCK_BUILD_TESTING is already on SROCK_CMAKE_EXTRA (build_child_env);
        # this only rejects a flip that the pending run would not actually apply.
        self._assert_build_testing_ok(info, args)
        json_path = self._ensure_subproject_map(env, info)
        # The feature catalog is regenerated by the (re)configure above.
        self._features_meta = self._read_feature_map(
            os.path.join(info["BUILD_DIR"], FEATURE_MAP)
        )
        # Companion artifact->subproject map (drives group-shard selection).
        self._artifact_map = self._read_artifact_map(
            os.path.join(info["BUILD_DIR"], ARTIFACT_MAP)
        )

        try:
            with open(json_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            core._fail(f"cannot read introspection map '{json_path}': {exc}")

        if not isinstance(data, dict) or not data:
            core._fail(f"introspection map '{json_path}' is empty or malformed")

        nameset = set(data.keys())
        cfg = Config()
        for order, (name, fields) in enumerate(data.items()):
            fields = fields or {}
            self._meta[name] = {
                "src": _clean(str(fields.get("src", ""))),
                "bin": _clean(str(fields.get("bin", ""))),
                "install_dest": _clean(str(fields.get("install_dest", ""))),
                "build_deps": _clean_list(fields.get("build_deps", [])),
                "runtime_deps": _clean_list(fields.get("runtime_deps", [])),
                "build_pool": _clean(str(fields.get("build_pool", ""))),
                "compiler_toolchain": _clean(str(fields.get("compiler_toolchain", ""))),
                "actions": _clean_list(fields.get("actions", [])),
            }
            # Build-time deps drive ordering. Drop any that are not themselves
            # introspected subprojects (e.g. helper targets) so the topo sort
            # only references real packages.
            depends = [d for d in self._meta[name]["build_deps"] if d in nameset]
            # Runtime deps don't affect ordering but do propagate rebuilds: a
            # component that consumes amd-llvm only at link/runtime (rocgdb,
            # comgr, hipcc, ...) must still rebuild when amd-llvm changes, so we
            # record them for --rdeps reverse-dependency closure.
            rdeps = [
                d for d in self._meta[name]["runtime_deps"]
                if d in nameset and d not in depends
            ]
            cfg.packages[name] = Package(
                name=name, depends=depends, runtime_depends=rdeps,
                xdir=".", order=order,
            )

        # Convenience features for --add/--remove, grouping subprojects by their
        # build pool and compiler toolchain.
        for name, meta in self._meta.items():
            pool = meta["build_pool"]
            if pool:
                cfg.features.setdefault(f"pool-{pool}", []).append(name)
            toolchain = meta["compiler_toolchain"]
            if toolchain:
                cfg.features.setdefault(f"toolchain-{toolchain}", []).append(name)

        # Convenience feature to opt every vendored third-party lib back in at
        # once (e.g. `--add thirdparty`), since they are excluded from the
        # default request below.
        vendored = sorted(n for n in nameset if _is_vendored(n))
        if vendored:
            cfg.features["thirdparty"] = vendored

        # Configure toggles surfaced through --add (see _srock_config /
        # _wants_sysdeps / build_child_env), registered as recognized (empty)
        # features so resolve_components accepts the tokens. The components they
        # bring into the build appear once TheRock is (re)configured with the
        # corresponding setting and are then pulled in by the default-request
        # dependency closure -- pair these with --reconfigure to apply them:
        #   sysdeps    -> THEROCK_BUNDLE_SYSDEPS=ON
        #   all        -> SROCK_CONFIG=all       (full build minus known-failing)
        #   all-debug  -> SROCK_CONFIG=all-debug (full build, may include failing)
        for toggle in ("sysdeps", "all", "all-debug"):
            cfg.features.setdefault(toggle, [])
        # Every introspected THEROCK_ENABLE_* feature is likewise a configure
        # toggle surfaced through --add (e.g. `--add hipdnn`); register each (and
        # its dash/case spellings) as an empty feature so resolve_components
        # accepts it. The components a feature brings in appear in the map once
        # TheRock is reconfigured with it on (see _apply_feature_flags).
        for fname in self._features_meta:
            for alias in _feature_aliases(fname):
                cfg.features.setdefault(alias, [])

        # Default request: the real ROCm components plus their full build- AND
        # runtime-dependency closure. This mirrors what a native
        # `cmake --build build` actually compiles -- the real components and the
        # vendored libs they genuinely need (e.g. therock-simde,
        # therock-msgpack-cxx) -- while leaving out vendored third-party libs
        # that no built component depends on (boost, eigen, googletest, ...).
        # Those remain *declared* (so `--add <name>` / `--add thirdparty` can
        # opt them in) and are still assembled by the whole-tree `therock/dist`
        # task whenever a distribution requires them.
        request = {n for n in nameset if not _is_vendored(n)}
        changed = True
        while changed:
            changed = False
            for name in list(request):
                meta = self._meta.get(name, {})
                for dep in (*meta["build_deps"], *meta["runtime_deps"]):
                    if dep in nameset and dep not in request:
                        request.add(dep)
                        changed = True
        cfg.request = sorted(request)
        self._cfg = cfg
        self._warn_unknown_build_type_comps(cfg, args)
        return cfg

    def _warn_unknown_build_type_comps(
        self, cfg: Config, args: argparse.Namespace,
    ) -> None:
        """Warn (don't fail) on per-component --build-type names that aren't
        known subprojects: cmake silently ignores an unused <name>_BUILD_TYPE
        cache var, so a typo would otherwise pass unnoticed."""
        _, per_comp_bt = core.parse_build_type_specs(
            getattr(args, "build_type", []) or []
        )
        for comp in sorted(per_comp_bt):
            if comp not in cfg.packages:
                core._warn(
                    f"--build-type '{comp}={per_comp_bt[comp]}': '{comp}' is "
                    f"not a known subproject; the -D{comp}_BUILD_TYPE flag will "
                    f"have no effect"
                )

    def config_name(self, args: argparse.Namespace) -> str:
        # Combine source config and build scope so distinct source selections
        # (e.g. amd-staging vs develop) get distinct export manifests and never
        # collide, mirroring how they occupy distinct CMake configurations.
        return f"{_resolve_source_config(args).name}-{_srock_config(args)}"

    def list_source_configs(self) -> list[dict] | None:
        rows: list[dict] = []
        for cfg in source_config.catalog():
            rows.append({
                "name": cfg.name,
                "description": cfg.description,
                "therock_branch": cfg.therock_branch,
                "compiler_branch": cfg.compiler_branch,
                "default": cfg.name == source_config.DEFAULT_SOURCE_CONFIG,
            })
        return rows

    # --- feature selection (THEROCK_ENABLE_*) ----------------------------- #
    def _read_feature_map(self, path: str) -> dict[str, dict]:
        """Best-effort read of feature_map.json; {} if missing/malformed."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _read_artifact_map(self, path: str) -> dict[str, list[str]]:
        """Best-effort read of artifact_map.json; {} if missing/malformed.

        Normalizes to ``{artifact: [subproject, ...]}`` (dropping any malformed
        entries). Missing on checkouts without the recorded artifact deps
        property, in which case group-shard subproject selection is unavailable.
        """
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, list[str]] = {}
        for name, deps in data.items():
            if isinstance(deps, list):
                out[name] = [str(d) for d in deps]
        return out

    def _requested_features(
        self, args: argparse.Namespace, catalog: dict[str, dict]
    ) -> set[str]:
        """The THEROCK_ENABLE_* features requested via --add.

        A token is a feature request when its normalized form is in the current
        feature catalog. On a cold start (no catalog yet -- the very first
        configure) any non-special token is taken as a feature candidate so an
        initial `--add hipdnn --reconfigure` still works; cmake harmlessly
        ignores an unknown -D define."""
        cold = not catalog
        out: set[str] = set()
        for entry in getattr(args, "add", None) or []:
            for tok in str(entry).split(","):
                tok = tok.strip()
                if not tok or tok.lower() in _SPECIAL_TOGGLES:
                    continue
                norm = _normalize_feature(tok)
                if norm in catalog or cold:
                    out.add(norm)
        return out

    def _disabled_features(
        self, args: argparse.Namespace, catalog: dict[str, dict]
    ) -> set[str]:
        """The THEROCK_ENABLE_* features the user asked to turn off via --remove.

        Symmetric with _requested_features: a --remove token is a feature
        *disable* request when its normalized form is in the current feature
        catalog (or, on a cold start with no catalog yet, any non-special token).
        For TheRock, a feature-backed subproject (e.g. amd-dbgapi / rocgdb) is
        removed from the build by switching its feature off at configure time --
        not by pruning the orchestrator task list, which the whole-tree install
        would rebuild anyway."""
        cold = not catalog
        out: set[str] = set()
        for entry in getattr(args, "remove", None) or []:
            for tok in str(entry).split(","):
                tok = tok.strip()
                if not tok or tok.lower() in _SPECIAL_TOGGLES:
                    continue
                norm = _normalize_feature(tok)
                if norm in catalog or cold:
                    out.add(norm)
        return out

    def _cascade_disables(
        self, disabled: set[str], catalog: dict[str, dict]
    ) -> set[str]:
        """Expand a disable set with every enabled feature that requires one of
        its members (transitively).

        TheRock features form a `requires` DAG (e.g. ROCGDB requires AMD_DBGAPI).
        Disabling a feature must also disable anything that depends on it, or the
        configure would be inconsistent. Walks the reverse of `requires` to a
        fixpoint, only pulling in features currently marked enabled (disabling an
        already-off feature is a no-op). Returns the full disable set."""
        full = set(disabled)
        changed = True
        while changed:
            changed = False
            for fname, meta in catalog.items():
                if fname in full:
                    continue
                if not (meta or {}).get("enabled", False):
                    continue
                requires = set((meta or {}).get("requires", []) or [])
                if requires & full:
                    full.add(fname)
                    changed = True
        return full

    def _apply_feature_flags(
        self, env: dict[str, str], info: dict[str, str],
        args: argparse.Namespace,
    ) -> None:
        """Validate requested feature enables/disables and, under --reconfigure,
        append their -DTHEROCK_ENABLE_<X>=ON/OFF flags to SROCK_CMAKE_EXTRA so the
        (re)configure picks them up. Without --reconfigure, changing a feature's
        current state is a hard error: nothing would configure the tree, so the
        request would be silently ignored."""
        catalog = self._read_feature_map(
            os.path.join(info["BUILD_DIR"], FEATURE_MAP)
        )
        requested = self._requested_features(args, catalog)
        disables = self._disabled_features(args, catalog)

        # A token cannot both enable and disable a feature.
        conflict = requested & disables
        if conflict:
            names = ", ".join(sorted(f.lower() for f in conflict))
            core._fail(
                f"feature(s) given to both --add and --remove: {names}."
            )

        # Disabling a feature drags down anything that requires it.
        cascaded = self._cascade_disables(disables, catalog) - disables
        disables |= cascaded

        if not requested and not disables:
            return

        if not bool(getattr(args, "reconfigure", False)):
            # A change relative to the current configuration: an enable of a
            # not-yet-enabled feature, or a disable of a currently-enabled one.
            pending_on = sorted(
                f for f in requested
                if not catalog.get(f, {}).get("enabled", False)
            )
            pending_off = sorted(
                f for f in disables
                if catalog.get(f, {}).get("enabled", False)
            )
            problems = []
            if pending_on:
                problems.append(
                    "not enabled: " + ", ".join(f.lower() for f in pending_on)
                )
            if pending_off:
                problems.append(
                    "still enabled: " + ", ".join(f.lower() for f in pending_off)
                )
            if problems:
                core._fail(
                    f"requested feature change(s) require a reconfigure "
                    f"({'; '.join(problems)}).\n"
                    f"  Re-run with --reconfigure to apply them (reconfigures "
                    f"TheRock and regenerates the build maps)."
                )
            return  # already in the requested state; nothing to (re)configure

        if cascaded:
            print(
                f"{core.PROG}: also disabling feature(s) that require the "
                f"removed one(s): "
                f"{', '.join(sorted(f.lower() for f in cascaded))}",
                flush=True,
            )
        flags = [f"-DTHEROCK_ENABLE_{f}=ON" for f in sorted(requested)]
        flags += [f"-DTHEROCK_ENABLE_{f}=OFF" for f in sorted(disables)]
        extra = env.get("SROCK_CMAKE_EXTRA", "")
        env["SROCK_CMAKE_EXTRA"] = f"{extra} {' '.join(flags)}".strip()
        if self._child_env is not None:
            self._child_env["SROCK_CMAKE_EXTRA"] = env["SROCK_CMAKE_EXTRA"]

    def list_features(self, env: dict[str, str]) -> list[dict] | None:
        """Rows for the `list-features` selector: every THEROCK_ENABLE_*
        feature with its enabled state, requires list, and description."""
        rows: list[dict] = []
        for name in sorted(self._features_meta):
            meta = self._features_meta[name] or {}
            rows.append({
                "name": name,
                "enabled": bool(meta.get("enabled", False)),
                "requires": list(meta.get("requires", []) or []),
                "description": str(meta.get("description", "") or ""),
            })
        return rows

    def build_child_env(self, args: argparse.Namespace) -> dict[str, str]:
        if self._child_env is not None:
            return self._child_env
        self._args = args
        src = os.environ
        env: dict[str, str] = {}

        for name in ENV_PASSTHROUGH:
            if name in src:
                env[name] = src[name]
        for name, value in src.items():
            if name.startswith("LC_"):
                env[name] = value

        # As with the AOMP backend, build a clean PATH so the build is isolated
        # from stray exports. TheRock discovers its build tools (cmake, ninja,
        # patchelf, meson, ...) from PATH via find_program; these are provided
        # by srock's supplemental dirs and the venv, which discover_env layers
        # on top of this PATH (and srock_venv_activate pip-installs patchelf and
        # meson into the venv). Use --inherit-path / --pass-env when a build
        # genuinely needs a tool or variable from the caller's environment.
        env["PATH"] = src.get("PATH", DEFAULT_CHILD_PATH) if args.inherit_path \
            else DEFAULT_CHILD_PATH

        for name in args.pass_env:
            for var in (n.strip() for n in name.split(",")):
                if var and var in src:
                    env[var] = src[var]

        def setdir(name: str, value: str | None) -> None:
            if value is not None:
                env[name] = os.path.abspath(os.path.expanduser(value))

        # Generic directory flags mapped onto the srock conventions:
        #   -s/--source  -> SROCK_REPOS   (parent of the TheRock checkout)
        #   -i/--install -> SROCK_LINK    (the install symlink; the versioned
        #                                  SROCK_INSTALL_DIR derives from it)
        #   -p/--prereq  -> SROCK_SUPP    (supplemental cmake/ninja/etc.)
        # -b/--build is not used: TheRock always builds under <TheRock>/build.
        setdir("SROCK_REPOS", args.source)
        setdir("SROCK_LINK", args.install)
        setdir("SROCK_SUPP", args.prereq)
        therock_dir = getattr(args, "therock_dir", None)
        setdir("SROCK_THEROCK_DIR", therock_dir)

        # SROCK_CONFIG (build scope) comes from --add (all / all-debug; minimal
        # default). Orthogonal to the *source config* below.
        env["SROCK_CONFIG"] = _srock_config(args)

        # -c/--config selects the *source config*: which TheRock branches the
        # srock scripts check out. Translate it to the srock branch env vars so
        # setup_srock.sh clones/switches the right sources (and applies the
        # matching compiler override + patch set). Branches only -- submodule
        # SHAs always come from whatever those branches record (see
        # source_config.py). These are deliberately not in ENV_PASSTHROUGH, so
        # the config (not a stray parent export) is authoritative.
        srcfg = _resolve_source_config(args)
        env["SROCK_THEROCK_BRANCH"] = srcfg.therock_branch
        env["SROCK_COMPILER_BRANCH"] = srcfg.compiler_branch
        if args.gfx:
            env["GFXLIST"] = args.gfx
        if args.jobs is not None:
            env["SROCK_JOB_THREADS"] = str(args.jobs)
            env["NINJA_NPROCS"] = str(args.jobs)
        if args.sudo:
            env["SUDO"] = "yes"

        # Bundled system deps (THEROCK_BUNDLE_SYSDEPS). Default OFF: system
        # libraries are resolved from the host. Opt in with `--add sysdeps`,
        # which flips this ON for a self-contained, portable install (and lets
        # the default-request closure pull the bundled libs real components
        # need). Appended to SROCK_CMAKE_EXTRA so it wins over the config block
        # (srock_common_vars adds $SROCK_CMAKE_EXTRA last), and emitted
        # explicitly in *both* directions so the toggle survives CMake's cache
        # across --reconfigure. Only affects a (re)configure -- pair
        # `--add sysdeps` with --reconfigure to apply it.
        bundle = "ON" if _wants_sysdeps(args) else "OFF"
        extra = env.get("SROCK_CMAKE_EXTRA", "")
        env["SROCK_CMAKE_EXTRA"] = (
            f"{extra} -DTHEROCK_BUNDLE_SYSDEPS={bundle}".strip()
        )

        # Component test suites (THEROCK_BUILD_TESTING). Default OFF -- see
        # DEFAULT_BUILD_TESTING -- and opt in with --build-tests. Emitted in both
        # directions like the sysdeps toggle above, which matters more here
        # because TheRock's own default is ON: passing it every time makes the
        # setting a function of the command line rather than of whatever the cmake
        # cache carries, and keeps it across the configures that do start from an
        # empty build dir (--fresh-configure, a source switch). Only a
        # (re)configure applies it, so a flip without --reconfigure is rejected
        # (_assert_build_testing_ok).
        testing = "ON" if _wants_build_tests(args) else "OFF"
        extra = env.get("SROCK_CMAKE_EXTRA", "")
        env["SROCK_CMAKE_EXTRA"] = (
            f"{extra} -DTHEROCK_BUILD_TESTING={testing}".strip()
        )

        self._child_env = env
        return env

    def _requested_build_type_vars(
        self, args: argparse.Namespace,
    ) -> dict[str, str]:
        """The cmake cache vars --build-type asks for: {CMAKE_BUILD_TYPE: T,
        <comp>_BUILD_TYPE: T, ...}. Empty if --build-type was not used."""
        global_bt, per_comp_bt = core.parse_build_type_specs(
            getattr(args, "build_type", []) or []
        )
        requested: dict[str, str] = {}
        if global_bt:
            requested["CMAKE_BUILD_TYPE"] = global_bt
        for comp, bt in per_comp_bt.items():
            requested[f"{comp}_BUILD_TYPE"] = bt
        return requested

    def _current_build_types(self, build_dir: str) -> dict[str, str]:
        """The build type cache vars currently configured in CMakeCache.txt
        (CMAKE_BUILD_TYPE and any <name>_BUILD_TYPE). {} if not yet configured."""
        path = os.path.join(build_dir, "CMakeCache.txt")
        current: dict[str, str] = {}
        try:
            with open(path, encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line or line.startswith(("#", "//")):
                        continue
                    key, sep, value = line.partition("=")
                    if not sep or ":" not in key:
                        continue
                    name = key.split(":", 1)[0]
                    if name == "CMAKE_BUILD_TYPE" or name.endswith("_BUILD_TYPE"):
                        current[name] = value
        except OSError:
            return {}
        return current

    def _desired_build_types(
        self, args: argparse.Namespace, current: dict[str, str],
    ) -> dict[str, str]:
        """The build type cache vars the command line implies, given what is
        currently configured.

        This is the explicitly requested vars PLUS reverts for anything the cache
        still carries that the command line no longer asks for: a cached
        <comp>_BUILD_TYPE not named this time reverts to the requested global (or
        DEFAULT_BUILD_TYPE), and a cached global reverts to DEFAULT_BUILD_TYPE.
        That way dropping a --build-type returns the scope to the default instead
        of retaining the stale cached value. Scopes that are neither requested
        nor already cached are left unmanaged (so a fresh tree keeps TheRock's
        own default rather than being forced)."""
        requested = self._requested_build_type_vars(args)
        revert_to = requested.get("CMAKE_BUILD_TYPE") or DEFAULT_BUILD_TYPE
        desired = dict(requested)
        for name in current:
            if name in desired:
                continue
            if name == "CMAKE_BUILD_TYPE":
                desired[name] = DEFAULT_BUILD_TYPE
            elif name.endswith("_BUILD_TYPE"):
                desired[name] = revert_to
        return desired

    def _apply_build_type_flags(
        self, env: dict[str, str], info: dict[str, str],
        args: argparse.Namespace,
    ) -> None:
        """Translate --build-type into TheRock configure-time -D flags.

        TheRock gates build types at *configure* time via cache vars (global
        CMAKE_BUILD_TYPE and per-project <name>_BUILD_TYPE, e.g.
        -Damd-llvm_BUILD_TYPE=Debug), not the per-task BUILD_TYPE env (inert for
        a ninja build). The desired state (see _desired_build_types) is compared
        against the current CMakeCache: if every value already matches it is a
        no-op (so the option can live on the command line across incremental
        builds), and dropping a previously-set build type is itself a change
        because the scope reverts to the default. Only a real change needs
        --reconfigure -- one without it is a hard error; otherwise the -D flags
        are appended to SROCK_CMAKE_EXTRA for the (re)configure."""
        current = self._current_build_types(info["BUILD_DIR"])
        desired = self._desired_build_types(args, current)
        changes = {k: v for k, v in desired.items() if current.get(k) != v}
        if not changes:
            return  # already configured this way; nothing to do
        if not bool(getattr(args, "reconfigure", False)):
            pretty = ", ".join(f"{k}={v}" for k, v in sorted(changes.items()))
            core._fail(
                "--build-type changes the current TheRock configuration "
                f"({pretty}) but no --reconfigure was given.\n"
                "  Re-run with --reconfigure to apply it (reconfigures TheRock "
                f"and regenerates the build maps). Note: an unset build type "
                f"defaults to {DEFAULT_BUILD_TYPE}, so dropping a previously-set "
                "--build-type is a change; re-add it to keep that value."
            )
        flags = [f"-D{k}={v}" for k, v in sorted(desired.items())]
        extra = env.get("SROCK_CMAKE_EXTRA", "")
        env["SROCK_CMAKE_EXTRA"] = f"{extra} {' '.join(flags)}".strip()
        if self._child_env is not None:
            self._child_env["SROCK_CMAKE_EXTRA"] = env["SROCK_CMAKE_EXTRA"]

    def _current_build_testing(self, build_dir: str) -> bool | None:
        """Whether the configured tree is *known* to build test suites, read from
        THEROCK_BUILD_TESTING in CMakeCache.txt.

        None means the setting is unmanaged: either the tree is not configured
        yet, or it was configured without the variable -- straight from
        setup_srock.sh, or before this option existed -- and so runs on TheRock's
        own default (ON, via CTest's BUILD_TESTING). Such trees are deliberately
        left alone instead of being reported as a change: their tests were being
        built all along, so failing every invocation until the user reconfigures
        would cost them a configure to fix nothing. The default takes hold at
        their next configure, like any other unmanaged cmake setting."""
        path = os.path.join(build_dir, "CMakeCache.txt")
        try:
            with open(path, encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line or line.startswith(("#", "//")):
                        continue
                    key, sep, value = line.partition("=")
                    if not sep or ":" not in key:
                        continue
                    if key.split(":", 1)[0] == "THEROCK_BUILD_TESTING":
                        return _cmake_is_true(value)
        except OSError:
            return None
        return None

    def _assert_build_testing_ok(
        self, info: dict[str, str], args: argparse.Namespace,
    ) -> None:
        """Reject a --build-tests flip that no configure would pick up.

        The flag is emitted on every configure by build_child_env, but it is a
        configure-time gate: on an already-configured tree, accepting a flip
        without --reconfigure would leave the caller believing tests had been
        switched when nothing changed. Same contract as --build-type, and
        idempotent, so --build-tests can sit on the command line across
        incremental builds."""
        current = self._current_build_testing(info["BUILD_DIR"])
        desired = _wants_build_tests(args)
        if current is None or current == desired:
            return  # unmanaged (see _current_build_testing), or already so
        if bool(getattr(args, "reconfigure", False)):
            return
        if desired:
            core._fail(
                "--build-tests turns THEROCK_BUILD_TESTING on but the tree is "
                "configured without tests and no --reconfigure was given.\n"
                "  Re-run with --reconfigure to apply it (reconfigures TheRock "
                "and regenerates the build maps)."
            )
        core._fail(
            "the tree is configured with THEROCK_BUILD_TESTING on, and omitting "
            "--build-tests turns it off, but no --reconfigure was given.\n"
            "  Re-run with --reconfigure to apply it, or pass --build-tests to "
            "keep the test suites enabled."
        )

    def discover_env(self, env: dict[str, str]) -> dict[str, str]:
        """Source srock_common_vars to learn the TheRock dir, install dir and a
        usable PATH (cmake/ninja/venv), honoring -s/-i/-p overrides."""
        if self._env_info is not None:
            return self._env_info

        therock_dir = ""
        install_dir = ""
        link = ""
        repos = env.get("SROCK_REPOS", "")
        sourced_path = env.get("PATH", DEFAULT_CHILD_PATH)
        if os.path.isfile(SROCK_COMMON_VARS):
            snippet = (
                f'. "{SROCK_COMMON_VARS}" >/dev/null 2>&1; '
                'printf "%s\\n" "$SROCK_THEROCK_DIR" "$SROCK_INSTALL_DIR" '
                '"$SROCK_LINK" "$SROCK_REPOS" "$PATH"'
            )
            proc = subprocess.run(
                ["bash", "-c", snippet], capture_output=True, text=True, env=env
            )
            out = proc.stdout.splitlines()
            out += [""] * (5 - len(out))
            therock_dir, install_dir, link, repos, sourced_path = out[:5]

        # An explicit --therock-dir wins (srock_common_vars hardwires
        # SROCK_THEROCK_DIR=$SROCK_REPOS/TheRock).
        override = env.get("SROCK_THEROCK_DIR")
        if override:
            therock_dir = override
        if not therock_dir:
            base = repos or os.path.join(os.path.expanduser("~"), "git", "srock-repos")
            therock_dir = os.path.join(base, "TheRock")

        build_dir = os.path.join(therock_dir, "build")

        # Layer the venv and srock's supplemental tool dirs on top of the
        # (already caller-derived) PATH so child cmake/ninja runs find the venv
        # python and the srock-provided cmake/ninja. `sourced_path` is the PATH
        # after srock_common_vars augmented it with $SROCK_SUPP/{cmake,ninja}/bin.
        path_parts = []
        venv_bin = os.path.join(therock_dir, ".venv", "bin")
        if os.path.isdir(venv_bin):
            path_parts.append(venv_bin)
        path_parts.append(sourced_path or env["PATH"])
        new_path = os.pathsep.join(path_parts)
        env["PATH"] = new_path
        if self._child_env is not None:
            self._child_env["PATH"] = new_path

        self._env_info = {
            "BUILD_DIR": build_dir,
            "SROCK_THEROCK_DIR": therock_dir,
            "SROCK_INSTALL_DIR": install_dir,
            "SROCK_LINK": link,
            "SROCK_REPOS": repos,
        }
        return self._env_info

    # --- bootstrap ------------------------------------------------------- #
    def _ensure_subproject_map(
        self, env: dict[str, str], info: dict[str, str]
    ) -> str:
        """Return the path to subproject_map.json, (re)configuring if asked.

        A missing map is a hard error (with guidance) unless --reconfigure is
        given, because generating it means a full TheRock cmake configure (which
        clones/fetches sources on first use) -- too heavy to trigger implicitly
        from a plain `list`/dry-run.
        """
        build_dir = info["BUILD_DIR"]
        json_path = os.path.join(build_dir, SUBPROJECT_MAP)
        # --fresh-configure is --reconfigure plus removing the build dir, so it
        # implies one.
        fresh = bool(getattr(self._args, "fresh_configure", False))
        reconfigure = bool(getattr(self._args, "reconfigure", False)) or fresh

        # Detect a source-config switch: the shared checkout is reused across
        # source configs, so if it does not already reflect the requested config
        # the sources must be switched (branch checkout + submodule resync, done
        # by setup_srock.sh) and the tree reconfigured. Force a reconfigure for
        # the switch even if the caller did not ask for one (otherwise we'd build
        # the previous config's sources against the new config's name).
        #
        # The marker records the full config identity (it also captures the
        # compiler-submodule branch, which the super-repo branch does not). When
        # it is absent -- a checkout set up directly by setup_srock.sh or
        # predating source configs -- fall back to the *actual* checked-out
        # super-repo branch so `-c <name>` still takes effect on existing trees.
        therock_dir = info["SROCK_THEROCK_DIR"]
        srcfg = _resolve_source_config(self._args)
        desired_cfg = srcfg.name
        current_cfg = self._read_source_marker(therock_dir)
        if current_cfg is not None:
            switch_needed = current_cfg != desired_cfg
            from_label = current_cfg
        else:
            branch = self._checked_out_branch(therock_dir)
            switch_needed = branch is not None and branch != srcfg.therock_branch
            from_label = f"branch {branch}" if branch else "unknown"
        if switch_needed and not reconfigure:
            print(
                f"{core.PROG}: source config switch "
                f"({from_label}) -> '{desired_cfg}' requires reconfigure; "
                f"forcing it.", flush=True,
            )
            reconfigure = True

        # A switch is destructive (branch checkout + working-tree reset). Never
        # perform it during a dry run -- preview against the current sources.
        if switch_needed and getattr(self._args, "dry_run", False):
            print(
                f"{core.PROG}: (dry-run) would switch sources "
                f"({from_label}) -> '{desired_cfg}' (branch checkout + "
                f"reconfigure); previewing with the current sources.", flush=True,
            )
            if os.path.isfile(json_path):
                return json_path
            core._fail(
                "(dry-run) cannot preview a switch with no existing "
                "subproject_map.json; configure once without -n first."
            )

        if os.path.isfile(json_path) and not reconfigure:
            return json_path
        if not reconfigure:
            core._fail(
                f"no {json_path}.\n"
                f"  TheRock must be configured with -DTHEROCK_INTROSPECTION=ON "
                f"first (PR #1234).\n"
                f"  Re-run with --reconfigure to do so via srock-bin, or set it "
                f"up manually:\n"
                f"    SROCK_CMAKE_EXTRA=-DTHEROCK_INTROSPECTION=ON "
                f"{SETUP_SROCK}"
            )

        if not os.path.isfile(SETUP_SROCK):
            core._fail(f"cannot --reconfigure: missing {SETUP_SROCK}")

        run_env = dict(env)
        extra = run_env.get("SROCK_CMAKE_EXTRA", "")
        if "THEROCK_INTROSPECTION" not in extra:
            run_env["SROCK_CMAKE_EXTRA"] = (
                f"{extra} -DTHEROCK_INTROSPECTION=ON".strip()
            )

        # Lay down the correct sources *before* injecting introspection, because
        # both source-laying steps revert the tracked files we inject into
        # (CMakeLists.txt + cmake/therock_artifacts.cmake):
        #   * First use: clone + fetch sources (a full setup).
        #   * A source-config switch on an existing checkout: setup_srock.sh
        #     restart performs `git checkout` of the new branch, reverting any
        #     previously injected introspection and resyncing submodules.
        # Their configure lacks introspection (stock TheRock has none), so it
        # won't emit the map -- that's fine; we inject and reconfigure below.
        if not os.path.isdir(therock_dir):
            print(
                f"{core.PROG}: setting up TheRock (clone + fetch sources): "
                f"{SETUP_SROCK}", flush=True,
            )
            rc = subprocess.run(["bash", SETUP_SROCK], env=run_env).returncode
            if rc != 0:
                core._fail(f"TheRock setup failed (rc={rc})")
        elif switch_needed:
            # Refuse to silently discard the user's uncommitted changes / local
            # commits; confirm (or abort) before the destructive switch.
            self._assert_switch_safe(therock_dir)
            print(
                f"{core.PROG}: switching TheRock sources to '{desired_cfg}': "
                f"{SETUP_SROCK} restart", flush=True,
            )
            rc = subprocess.run(
                ["bash", SETUP_SROCK, "restart"], env=run_env
            ).returncode
            if rc != 0:
                core._fail(f"TheRock source switch failed (rc={rc})")

        # Ensure PR #1234's introspection support is present in the checkout
        # (re-applied here after any source switch above reverted it).
        self._inject_introspection(therock_dir)

        # Reconfigure (reusing fetched sources) now that introspection is wired
        # in and -DTHEROCK_INTROSPECTION=ON is passed. `restart` reconfigures in
        # place; `restart clean` removes the build dir first. Only --fresh-
        # configure asks for the latter, and only when something is left to
        # remove: a first-ever setup starts from nothing, and a source switch has
        # already discarded the build dir (it described the pre-switch sources).
        restart = ["bash", SETUP_SROCK, "restart"]
        if fresh and not switch_needed and os.path.isdir(build_dir):
            restart.append("clean")
        print(
            f"{core.PROG}: reconfiguring TheRock with introspection: "
            f"{' '.join(restart[1:])}", flush=True,
        )
        # Reconfiguring in place keeps the previous configure's map, so "the file
        # exists" would no longer prove this configure produced it -- a configure
        # that succeeded without running the introspection would leave us building
        # against a stale graph. Drop the map first so its presence afterwards is
        # proof. It is a generated file, and a configure that fails to write it is
        # a hard error either way.
        try:
            os.remove(json_path)
        except OSError:
            pass
        rc = subprocess.run(restart, env=run_env).returncode
        if rc != 0:
            core._fail(f"TheRock configure failed (rc={rc})")
        if not os.path.isfile(json_path):
            core._fail(
                f"configure did not produce {json_path}; introspection injection "
                f"may have failed (see {therock_dir}/CMakeLists.txt)."
            )
        # Record the source config the checkout now reflects, so a later run with
        # a different --config detects the switch (see top of this method).
        self._write_source_marker(therock_dir, desired_cfg)
        return json_path

    def _has_local_commits(self, repo: str) -> bool:
        """True if ``repo``'s HEAD has commits not on any remote-tracking branch.

        These are unpushed local commits -- the user's work that a branch switch
        could leave behind. Requires remote refs to exist to be meaningful; with
        none we cannot classify commits as "unpushed" and conservatively report
        False (avoids flagging every commit in a remote-less checkout)."""
        if not source_layout._git_out(repo, "for-each-ref", "refs/remotes"):
            return False
        out = source_layout._git_out(
            repo, "rev-list", "--count", "HEAD", "--not", "--remotes"
        )
        try:
            return int(out) > 0
        except ValueError:
            return False

    def _switch_safety_report(
        self, therock_dir: str
    ) -> tuple[list[str], list[str]]:
        """Scan the super-repo and its initialized submodules for at-risk work.

        Returns ``(dirty, ahead)``: repos with uncommitted modifications (which a
        switch's ``git checkout .`` would discard) and repos with unpushed local
        commits (which a branch switch could leave behind). Submodule gitlink
        changes are ignored (``--ignore-submodules=all``) so each repo reports
        only its own file changes; nested submodules are scanned in their own
        right via ``submodule status --recursive``."""
        dirty: list[str] = []
        ahead: list[str] = []
        repos: list[tuple[str, str]] = [("TheRock (super-repo)", therock_dir)]
        status = source_layout._git_out(
            therock_dir, "submodule", "status", "--recursive"
        )
        for line in status.splitlines():
            if not line:
                continue
            # " <sha> <path> (<desc>)"; leading char: ' ' ok, '+' tip differs,
            # 'U' merge conflict, '-' uninitialized (nothing checked out -> skip).
            indicator, body = line[0], line[1:]
            if indicator == "-":
                continue
            _sha, _, tail = body.partition(" ")
            path = tail.split(" (")[0].strip()
            if path:
                repos.append((path, os.path.join(therock_dir, path)))
        for label, repo in repos:
            if not source_layout.is_git_repo(repo):
                continue
            if source_layout._git_out(
                repo, "status", "--porcelain", "--ignore-submodules=all"
            ):
                dirty.append(label)
            if self._has_local_commits(repo):
                ahead.append(label)
        return dirty, ahead

    def _assert_switch_safe(self, therock_dir: str) -> None:
        """Guard a destructive in-place source-config switch.

        The switch (setup_srock.sh restart) hard-resets the super-repo and every
        submodule working tree and re-checks-out the compiler branches, silently
        discarding uncommitted changes and possibly orphaning local commits. If
        any tracked repo has uncommitted modifications or unpushed local commits,
        list them and require confirmation. -y/--yes bypasses the prompt; a
        decline (or a non-interactive session without -y) aborts the run so the
        user's work is never lost without consent."""
        dirty, ahead = self._switch_safety_report(therock_dir)
        if not dirty and not ahead:
            return
        print(
            "\nWARNING: switching the TheRock source config resets working trees "
            "and re-checks-out branches, which can lose local work:"
        )
        if ahead:
            print("  Repos with UNPUSHED LOCAL COMMITS (may be left behind):")
            for label in ahead:
                print(f"    - {label}")
        if dirty:
            print("  Repos with UNCOMMITTED CHANGES (will be discarded):")
            for label in dirty:
                print(f"    - {label}")
            print("    (note: srock applies its compiler patches as uncommitted "
                  "changes; those are expected and safe to discard.)")
        if getattr(self._args, "yes", False):
            print("  -y/--yes given; proceeding and discarding/leaving the above.")
            return
        try:
            reply = input(
                "\nProceed with the switch (commit or stash first to keep work)? "
                "[y/N] "
            ).strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            core._fail(
                "source config switch aborted to preserve local work; commit or "
                "stash your changes (or pass -y/--yes to discard them)."
            )

    def _checked_out_branch(self, therock_dir: str) -> str | None:
        """The super-repo's current branch name, or None.

        Used as the marker-less fallback for switch detection. Returns None when
        there is no git checkout, on any git error, or for a detached HEAD
        (``rev-parse`` yields the literal "HEAD") -- in which case we cannot tell
        the source config from the branch and conservatively force no switch
        (use --reconfigure to switch explicitly)."""
        if not os.path.isdir(os.path.join(therock_dir, ".git")):
            return None
        try:
            proc = subprocess.run(
                ["git", "-C", therock_dir, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True,
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        branch = proc.stdout.strip()
        if not branch or branch == "HEAD":
            return None
        return branch

    def _read_source_marker(self, therock_dir: str) -> str | None:
        """The source config name the checkout records, or None if unknown.

        None means either no checkout yet or a legacy checkout predating source
        configs (a missing marker); in both cases no switch is forced."""
        try:
            with open(
                os.path.join(therock_dir, SOURCE_CONFIG_MARKER), encoding="utf-8"
            ) as handle:
                name = handle.read().strip()
        except OSError:
            return None
        return name or None

    def _write_source_marker(self, therock_dir: str, name: str) -> None:
        """Record ``name`` as the checkout's active source config (best effort)."""
        try:
            with open(
                os.path.join(therock_dir, SOURCE_CONFIG_MARKER),
                "w", encoding="utf-8",
            ) as handle:
                handle.write(f"{name}\n")
        except OSError as exc:
            core._warn(f"could not write source config marker: {exc}")

    def _inject_introspection(self, therock_dir: str) -> None:
        """Make a stock TheRock checkout introspectable (PR #1234).

        Copies the bundled introspection module into ``<TheRock>/cmake/`` and
        appends a guarded ``include()`` + invocation to the top-level
        ``CMakeLists.txt``. Both steps are idempotent: the file is overwritten
        (so updates propagate) and the CMakeLists block is added only once
        (guarded by a marker). The block is appended at the *end* so it runs
        after every subproject is declared, independent of upstream line
        numbers.
        """
        if not os.path.isfile(INTROSPECTION_CMAKE):
            core._fail(f"missing bundled introspection cmake: {INTROSPECTION_CMAKE}")
        cmake_dir = os.path.join(therock_dir, "cmake")
        cmakelists = os.path.join(therock_dir, "CMakeLists.txt")
        if not (os.path.isdir(cmake_dir) and os.path.isfile(cmakelists)):
            core._fail(
                f"{therock_dir} does not look like a TheRock checkout "
                f"(missing cmake/ or CMakeLists.txt)"
            )
        shutil.copyfile(
            INTROSPECTION_CMAKE,
            os.path.join(cmake_dir, "therock_subproject_introspection.cmake"),
        )
        # Record artifact->subproject deps so the introspection can emit
        # artifact_map.json (drives group-based shard subproject selection).
        self._inject_artifact_deps_property(cmake_dir)
        try:
            with open(cmakelists, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            core._fail(f"cannot read {cmakelists}: {exc}")
        if INTROSPECTION_MARKER in text:
            return
        block = (
            f"\n# >>> {INTROSPECTION_MARKER} (added by {core.PROG}; PR #1234) >>>\n"
            "include(therock_subproject_introspection)\n"
            "if(THEROCK_INTROSPECTION)\n"
            "  therock_introspect_subprojects()\n"
            "endif()\n"
            f"# <<< {INTROSPECTION_MARKER} <<<\n"
        )
        with open(cmakelists, "a", encoding="utf-8") as handle:
            handle.write(block)
        print(f"{core.PROG}: injected build introspection into {cmakelists}")

    def _inject_artifact_deps_property(self, cmake_dir: str) -> None:
        """Make ``therock_provide_artifact`` record its SUBPROJECT_DEPS.

        The artifact->subproject linkage lives only inside this function (as the
        ``SUBPROJECT_DEPS`` argument); cmake does not otherwise expose it. We add
        a single ``set_property`` so every ``artifact-<slice>`` target carries
        ``THEROCK_ARTIFACT_SUBPROJECT_DEPS``, which the bundled introspection
        reads to emit ``artifact_map.json``. Idempotent (marker-guarded) and
        inserted right after the target is wired into ``therock-artifacts`` (so
        it exists in both the topology and fallback branches). Best-effort: if
        the file or anchor is absent (upstream drift), we skip and the backend
        runs without group-shard subproject selection."""
        path = os.path.join(cmake_dir, "therock_artifacts.cmake")
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            return
        if ARTIFACT_DEPS_MARKER in text:
            return
        anchor = '  add_dependencies(therock-artifacts "${_target_name}")\n'
        if anchor not in text:
            core._warn(
                f"could not record artifact subproject deps in {path} "
                f"(anchor not found); group-shard subproject selection disabled"
            )
            return
        block = (
            f"  # >>> {ARTIFACT_DEPS_MARKER} (added by {core.PROG}) >>>\n"
            '  set_property(TARGET "${_target_name}" PROPERTY\n'
            '    THEROCK_ARTIFACT_SUBPROJECT_DEPS "${ARG_SUBPROJECT_DEPS}")\n'
            f"  # <<< {ARTIFACT_DEPS_MARKER} <<<\n"
        )
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text.replace(anchor, anchor + block, 1))
        except OSError as exc:
            core._warn(f"cannot record artifact subproject deps in {path}: {exc}")
            return
        print(f"{core.PROG}: recorded artifact subproject deps in {path}")

    # --- task elaboration ------------------------------------------------- #
    def component_configs(self, comp: str, env: dict[str, str]) -> list[str]:
        # TheRock subprojects are config-less: a single build configuration is
        # baked in at configure time. Tasks therefore use the short comp/stage
        # name and always run (no variant filtering).
        return []

    def list_component_tasks(
        self, comp: str, env: dict[str, str]
    ) -> list[RawTask]:
        meta = self._meta.get(comp, {})
        available = set(meta.get("actions", []))
        build_dir = self._env_info["BUILD_DIR"] if self._env_info else \
            self.discover_env(env)["BUILD_DIR"]
        # -a/--all exposes every advertised action (expunge/.../dist); the
        # default pipeline is just configure/build/stage.
        want_all = self._args is not None and getattr(self._args, "all", False)
        actions = ALL_ACTIONS if want_all else FORWARD_ACTIONS
        raw: list[RawTask] = []
        for action in actions:
            if action in available:
                raw.append(
                    (action, None, {"target": f"{comp}+{action}", "bin": build_dir})
                )
        return raw

    def leading_tasks(
        self, components: list[str], env: dict[str, str]
    ) -> list[Task]:
        """The `therock/prereq` pseudo-task: build the cmake/ninja prerequisite
        toolchain via srock-bin/build_cmake.sh, run first so its output lands in
        a per-task log rather than spamming the console. build_cmake.sh
        self-checks and is a cheap no-op when the tools are already built."""
        if not os.path.isfile(BUILD_CMAKE):
            return []
        return [
            Task(
                comp="therock", action="prereq", cfgname=None,
                single_config=True, payload={"argv": ["bash", BUILD_CMAKE]},
            )
        ]

    def trailing_tasks(
        self, components: list[str], env: dict[str, str]
    ) -> list[Task]:
        """The whole-tree dist + install pseudo-tasks (see WHOLE_TREE_TASKS).

        Appended after all per-subproject tasks so a full build ends by
        assembling the combined dist tree and installing it, and so they can be
        run on their own (e.g. ``therock_build.py therock/install``)."""
        build_dir = (self._env_info or self.discover_env(env))["BUILD_DIR"]
        tasks: list[Task] = []
        for comp, action, target in WHOLE_TREE_TASKS:
            tasks.append(
                Task(
                    comp=comp, action=action, cfgname=None, single_config=True,
                    payload={"target": target, "bin": build_dir},
                )
            )
        return tasks

    # --- execution -------------------------------------------------------- #
    def task_command(
        self, task: Task, env: dict[str, str]
    ) -> tuple[list[str], dict[str, str]]:
        # Pseudo-tasks may carry a literal command (e.g. the prereq toolchain
        # build) instead of a ninja target.
        argv = task.payload.get("argv")
        if argv is not None:
            return list(argv), {}
        bin_dir = task.payload["bin"]
        target = task.payload["target"]
        # TheRock "Option 1": for the per-subproject build stage, once the
        # subproject has been configured (its own build.ninja exists), run ninja
        # directly in that build dir. The super-level `<comp>+build` relies on
        # stamp tracking that does not re-scan a subproject's sources, so edits
        # to large components (e.g. amd-llvm) are otherwise not rebuilt. The
        # subproject's `therock-touch` ALL target then marks stage.stamp stale,
        # so the following `<comp>+stage` re-stages. Opt out with
        # --superproject-build.
        delegate = self._args is None or getattr(self._args, "delegate", True)
        if delegate and task.action == "build":
            sub_bin = self._meta.get(task.comp, {}).get("bin", "")
            if sub_bin:
                sub_dir = os.path.join(bin_dir, sub_bin)
                if os.path.isfile(os.path.join(sub_dir, "build.ninja")):
                    cmd = ["ninja", "-C", sub_dir]
                    if self._args is not None and self._args.jobs is not None:
                        cmd += ["-j", str(self._args.jobs)]
                    return cmd, {}
        cmd = ["ninja", "-C", bin_dir, target]
        if self._args is not None and self._args.jobs is not None:
            cmd += ["-j", str(self._args.jobs)]
        return cmd, {}

    # --- manifest / clean ------------------------------------------------- #
    def component_src_dir(self, comp: str, env: dict[str, str]) -> str:
        src = self._meta.get(comp, {}).get("src", "")
        if not src:
            return ""
        therock_dir = (self._env_info or self.discover_env(env))["SROCK_THEROCK_DIR"]
        return os.path.join(therock_dir, src)

    def external_repos(self, env: dict[str, str]) -> dict[str, str]:
        # Record the TheRock super-repo itself (its submodule SHAs are captured
        # per-subproject via component_src_dir).
        therock_dir = (self._env_info or self.discover_env(env))["SROCK_THEROCK_DIR"]
        return {"TheRock": therock_dir} if therock_dir else {}

    def floating_components(self) -> set[str]:
        return set(FLOATING_COMPONENTS)

    def install_clean_task(self, env_info: dict[str, str]) -> Task:
        return Task(
            comp="install", action="clean", cfgname=None,
            single_config=True, builtin="install_clean",
            targets=[env_info.get("SROCK_INSTALL_DIR", ""),
                     env_info.get("SROCK_LINK", "")],
        )

    # --- prebuilt / incremental focus ------------------------------------ #
    def _stage_relpath(self, comp: str) -> str:
        """The component's stage dir, relative to <build> (what buildctl.py's
        regexes match against). TheRock places a subproject's stage dir beside
        its build dir: ``.../build`` -> ``.../stage``."""
        bin_rel = self._meta.get(comp, {}).get("bin", "")
        if not bin_rel:
            return ""
        bin_rel = bin_rel.replace(os.sep, "/").rstrip("/")
        if bin_rel.endswith("/build"):
            return bin_rel[: -len("/build")] + "/stage"
        if bin_rel.endswith("build"):
            return bin_rel[: -len("build")] + "stage"
        head = bin_rel.rsplit("/", 1)[0] if "/" in bin_rel else ""
        return (head + "/stage") if head else "stage"

    def provision_preconfig(self, args: argparse.Namespace) -> int | None:
        """--migrate-aomp REPODIR: MOVE the shared standalone repos out of the
        AOMP checkout REPODIR into this TheRock checkout's submodule slots
        (-s/--source resolves to SROCK_THEROCK_DIR), converting each into a
        submodule gitdir. Destructive; gated by a confirmation prompt.

        Runs *before* load_config -- and hence before the cmake configure /
        fetch_sources that would otherwise populate (and so block) the submodule
        slots -- and then ends the run. The seeded gitdirs already hold the AOMP
        objects, so a subsequent `--reconfigure` build reuses them (only a fast
        checkout to the pinned SHA, no fresh clone). Returns None when
        --migrate-aomp is not requested so a normal run proceeds."""
        repodir = getattr(args, "migrate_aomp", None)
        if not repodir:
            return None

        aomp_repodir = os.path.abspath(os.path.expanduser(repodir))
        env = self.build_child_env(args)
        # discover_env mutates env["PATH"] in place to prepend the TheRock venv
        # bin, so `env` below runs fetch_sources with the venv python.
        therock_dir = self.discover_env(env)["SROCK_THEROCK_DIR"]
        dry = getattr(args, "dry_run", False)

        if not source_layout.is_git_repo(therock_dir):
            core._fail(
                f"--migrate-aomp: destination '{therock_dir}' is not a git "
                f"checkout of TheRock (set -s/--source or --therock-dir)"
            )

        plan = source_layout.migrate_plan(therock_dir, aomp_repodir)
        print(f"--- migrate AOMP sources: {aomp_repodir} -> {therock_dir} ---")
        ready = [a for a in plan if a.status == "ready"]
        reason = {
            "missing-source": "no such repo in the AOMP checkout",
            "not-git": "not a git repository",
            "dest-occupied": "destination slot already populated",
        }
        for act in plan:
            if act.status == "ready":
                print(f"  move {act.comp.aomp_dir} -> {act.comp.therock_path}")
                for w in act.warnings:
                    print(f"      warning: {w}")
            else:
                print(f"  skip {act.comp.aomp_dir}: {reason[act.status]}")

        if not ready:
            print("nothing to migrate.")
            return 0

        if dry:
            print("(dry-run) no directories moved.")
            return 0

        if not getattr(args, "yes", False):
            print(f"\nThis MOVES {len(ready)} repo(s) out of the AOMP checkout "
                  f"into TheRock's submodule slots (the AOMP locations will no "
                  f"longer exist).")
            try:
                reply = input("Proceed? [y/N] ").strip().lower()
            except EOFError:
                reply = ""
            if reply not in ("y", "yes"):
                print("aborted.")
                return 1

        source_layout.apply_migration(therock_dir, plan, dry_run=dry)

        # The migrate only seeds the shared 1:1 repos; TheRock's other submodules
        # (rocm-systems / rocm-libraries monorepos, etc.) still need fetching for
        # the configure to succeed. fetch_sources.py treats the seeded slots as
        # already-initialized (their .git gitlink exists) and only checks them
        # out -- it does not re-clone over them -- while cloning the rest.
        fetch = os.path.join(therock_dir, "build_tools", "fetch_sources.py")
        if os.path.isfile(fetch):
            print("\n--- fetching remaining TheRock submodules "
                  "(build_tools/fetch_sources.py) ---")
            rc = subprocess.run(["python", fetch], cwd=therock_dir, env=env).returncode
            if rc != 0:
                core._warn(
                    "fetch_sources.py failed; re-run it in the TheRock checkout "
                    "before configuring"
                )
                return rc
        else:
            core._warn(
                f"{fetch} not found; run TheRock's fetch_sources before "
                f"--reconfigure"
            )

        src = args.source or "<src>"
        print("\nmigration complete (shared repos seeded, other submodules "
              "fetched). Next: configure + build with --reconfigure, e.g.\n"
              f"  therock_build.py --backend therock -s {src} --reconfigure\n"
              "The configure reuses the seeded gitdirs (fast checkout of the "
              "pinned SHAs, no re-clone of the migrated repos).")
        return 0

    def prepare_run(
        self, selected_comps: set[str], env: dict[str, str],
        args: argparse.Namespace,
    ) -> None:
        """Manage TheRock's ``.prebuilt`` markers around the upcoming run.

        Default behavior (incremental focus): when a strict, non-empty subset
        of subprojects is being built, mark every *other* component prebuilt so
        neither this build nor a later whole-tree install rebuilds dependents
        the user is not working on. This wraps TheRock's own
        ``build_tools/buildctl.py``: ``enable <working-set>`` makes exactly the
        working set buildable and marks all other (already-built) components
        prebuilt -- and only components that have actually been staged can be
        pinned, so anything not yet built stays buildable and is produced if a
        dependency needs it.

        Overrides:
          * --unpin-all   -> ``buildctl.py enable`` (clear all markers) and stop.
          * --no-auto-pin -> leave markers untouched.
        """
        if getattr(args, "unpin_all", False):
            self._run_buildctl(
                ["enable"], env, dry_run=args.dry_run,
                note="clearing all prebuilt markers (every component buildable)",
            )
            return
        if getattr(args, "no_auto_pin", False):
            return

        subprojects = set(self._meta.keys())
        working = selected_comps & subprojects
        # Nothing to focus on a full build (or a whole-tree-only selection such
        # as just 'therock/install'): leave existing markers as they are.
        if not working or working == subprojects:
            return

        patterns = sorted({p for p in (self._stage_relpath(c) for c in working) if p})
        patterns = ["^" + re.escape(p) + "$" for p in patterns]
        if not patterns:
            return
        self._run_buildctl(
            ["enable", *patterns], env, dry_run=args.dry_run,
            note=(
                f"auto-pin: focusing on {len(working)} component(s) "
                f"({', '.join(sorted(working))}); marking other already-built "
                f"components prebuilt so dependents are not rebuilt "
                f"(--rdeps to rebuild dependents too, --unpin-all to undo, "
                f"--no-auto-pin to disable)"
            ),
        )

    def _run_buildctl(
        self, sub_args: list[str], env: dict[str, str], dry_run: bool, note: str,
    ) -> None:
        info = self._env_info or self.discover_env(env)
        therock_dir = info["SROCK_THEROCK_DIR"]
        build_dir = info["BUILD_DIR"]
        buildctl = os.path.join(therock_dir, "build_tools", "buildctl.py")
        if not os.path.isfile(buildctl):
            print(
                f"{core.PROG}: note: {buildctl} not found; "
                f"skipping prebuilt management"
            )
            return
        # buildctl.py reconfigures TheRock to pick up marker changes; it needs a
        # configured tree (CMakeCache.txt), which load_config has ensured.
        cmd = ["python", buildctl, *sub_args, "--build-dir", build_dir]
        print(f"{core.PROG}: {note}")
        if dry_run:
            print(f"    would run: {' '.join(cmd)}")
            return
        rc = subprocess.run(cmd, cwd=therock_dir, env=env).returncode
        if rc != 0:
            core._fail(f"buildctl.py failed (rc={rc}): {' '.join(cmd)}")

    def built_components(self, env: dict[str, str]) -> set[str] | None:
        """Components with a valid (existing, non-empty) stage dir -- exactly the
        ones buildctl.py would treat as built and could mark prebuilt. Mirrors
        buildctl's is_valid_stage_dir check against each comp's stage relpath."""
        info = self._env_info or self.discover_env(env)
        build_dir = info["BUILD_DIR"]
        built: set[str] = set()
        for comp in self._meta:
            rel = self._stage_relpath(comp)
            if not rel:
                continue
            stage_dir = os.path.join(build_dir, rel)
            try:
                if os.path.isdir(stage_dir) and os.listdir(stage_dir):
                    built.add(comp)
            except OSError:
                continue
        return built

    # --- sharding -------------------------------------------------------- #
    def _load_topology(self, env: dict[str, str]):
        """The BuildTopology for this checkout, or None when unavailable."""
        info = self._env_info or self.discover_env(env)
        return topology.load_build_topology(info["SROCK_THEROCK_DIR"])

    def _subproject_group_map(self, topo) -> dict[str, set[str]]:
        """Map each configured subproject to the artifact group(s) it composes.

        Joins ``artifact_map.json`` (artifact -> composing cmake subprojects)
        with BUILD_TOPOLOGY.toml (artifact -> ``artifact_group``). A subproject
        can belong to more than one group when its stage install feeds artifacts
        in different groups. Subprojects absent from the artifact map, or whose
        artifacts are unknown to the topology, are omitted -- so the map is empty
        on checkouts whose ``therock_artifacts.cmake`` lacks the recorded deps
        property (group-shard subproject selection then degrades to nothing)."""
        out: dict[str, set[str]] = {}
        try:
            artifacts = topo.artifacts
        except Exception:
            return out
        for artifact_name, subs in self._artifact_map.items():
            art = artifacts.get(artifact_name)
            group = getattr(art, "artifact_group", "") if art is not None else ""
            if not group:
                continue
            for sub in subs:
                if sub in self._meta:
                    out.setdefault(sub, set()).add(group)
        return out

    def list_shards(self, env: dict[str, str]) -> list[dict] | None:
        """The TheRock artifact groups as shard rows (for `list-shards`).

        Each row reports a group's description, the subprojects it builds (from
        the artifact->subproject introspection), the source sets it fetches, the
        groups it depends on (import before building), and produced/inbound
        artifact counts. Groups are listed in dependency (build) order. The
        ``subprojects`` list reflects the current configure (only enabled
        artifacts appear in artifact_map.json), so an empty list means the
        group's subprojects are not enabled in this profile. Returns None when no
        topology is available (so the core prints a clear note)."""
        topo = self._load_topology(env)
        if topo is None:
            return None
        sub_group = self._subproject_group_map(topo)
        rows: list[dict] = []
        for name in topology.group_names(topo):
            group = topo.artifact_groups.get(name)
            subprojects = sorted(c for c, gs in sub_group.items() if name in gs)
            rows.append({
                "name": name,
                "description": getattr(group, "description", ""),
                "subprojects": subprojects,
                "configured": len(subprojects),
                "source_sets": topology.group_source_sets(topo, name),
                "depends_on": topology.group_dependencies(topo, name),
                "produced": len(topology.group_produced_names(topo, name)),
                "inbound": len(topology.group_inbound_names(topo, name)),
            })
        return rows

    def rest_build_shards(
        self, import_shards: list[str], env: dict[str, str],
    ) -> list[str] | None:
        """Every configured artifact group not in ``import_shards``, in build
        order (for -f/--fill).

        "Configured" means the group has at least one subproject in the current
        configure -- exactly what `list-shards` reports as ``configured`` (an
        empty group is feature-disabled in this profile). Imported groups are
        provided as artifacts, so they are excluded from the build set (the
        shard pipeline still pins them). Returns None when no topology is
        available."""
        topo = self._load_topology(env)
        if topo is None:
            return None
        sub_group = self._subproject_group_map(topo)
        configured = {g for groups in sub_group.values() for g in groups}
        skip = set(import_shards)
        return [
            g for g in topology.group_names(topo)
            if g in configured and g not in skip
        ]

    def shard_tasks(
        self, tasks: list[Task], import_shards: list[str],
        build_shards: list[str], export_shards: list[str],
        env: dict[str, str], args: argparse.Namespace,
    ) -> list[Task] | None:
        """Assemble the import -> build -> export pipeline for the named groups.

        Each shard is an artifact group. Order: fetch the build groups' source
        sets, import the requested producer groups' artifacts (as prebuilt), pin
        + reconfigure, build the build groups' subprojects and their group
        artifact targets, then export. Uses TheRock's own tools (fetch_sources.py
        --source-sets, buildctl.py bootstrap/enable, ninja artifact-group-<g>)
        plus the shard_artifacts.py helper for filtered import/export. Returns
        None when no topology is available."""
        topo = self._load_topology(env)
        if topo is None:
            return None
        info = self._env_info or self.discover_env(env)
        therock_dir = info["SROCK_THEROCK_DIR"]
        build_dir = info["BUILD_DIR"]

        known = set(topology.group_names(topo))
        for name in (*import_shards, *build_shards, *export_shards):
            if name not in known:
                core._fail(
                    f"unknown shard '{name}' (see `list-shards`); known groups: "
                    f"{', '.join(sorted(known))}"
                )

        store = getattr(args, "shard_store", None) or os.path.join(
            build_dir, "shard-artifacts"
        )
        families = getattr(args, "shard_families", None)

        fetch = os.path.join(therock_dir, "build_tools", "fetch_sources.py")
        buildctl = os.path.join(therock_dir, "build_tools", "buildctl.py")

        def pseudo(action: str, argv: list[str]) -> Task:
            return Task(
                comp="therock", action=action, cfgname=None,
                single_config=True, payload={"argv": argv},
            )

        sub_group = self._subproject_group_map(topo)
        build_set = set(build_shards)
        build_comps = [
            c for c in self._meta if sub_group.get(c, set()) & build_set
        ]

        pipeline: list[Task] = []

        # 1. Fetch sources for the build groups (fetch_sources.py --source-sets),
        #    so a sharded checkout only pulls the submodules those groups need.
        #    Groups with no source sets (e.g. vendored third-party-sysdeps) add
        #    nothing; if the union is empty the fetch is skipped entirely.
        #    "python" resolves to TheRock's venv (discover_env prepends its bin).
        source_sets: list[str] = []
        for shard in build_shards:
            for s in topology.group_source_sets(topo, shard):
                if s not in source_sets:
                    source_sets.append(s)
        if source_sets:
            pipeline.append(pseudo(
                "fetch-sources",
                ["python", fetch, "--source-sets", *source_sets],
            ))

        # 2. Import the requested producer groups' artifacts as prebuilt. The
        #    helper filters the shared store to each group's produced-artifact
        #    names (computed here from the topology) and runs buildctl bootstrap.
        for shard in import_shards:
            names = sorted(topology.group_produced_names(topo, shard))
            argv = [
                "python", SHARD_ARTIFACTS, "import-bootstrap",
                "--buildctl", buildctl,
                "--build-dir", build_dir,
                "--store", store,
                "--names", ",".join(names),
            ]
            if families:
                argv += ["--target-families", families]
            pipeline.append(pseudo(f"import-{shard}", argv))

        # 3. Pin + reconfigure before building. buildctl bootstrap (import) only
        #    drops .prebuilt markers and stages files; TheRock honors them at
        #    *configure* time, so a reconfigure must follow the imports (and any
        #    fetch) for the build to treat the imported subprojects as prebuilt
        #    rather than rebuild them. `buildctl enable <build-group stages>`
        #    makes exactly the build groups' subprojects buildable and pins every
        #    other now-staged component (the imports), then reconfigures. This
        #    replaces the generic auto-pin (prepare_run), which would otherwise
        #    run too early -- before the imports stage anything. Skipped when the
        #    build groups map to no known subprojects (an empty `enable` would
        #    clear all markers).
        #
        #    --force-reconfigure is essential: bootstrap created the .prebuilt
        #    markers *outside* buildctl's tracking, so `enable` sees no marker
        #    change and would otherwise skip the reconfigure -- leaving the build
        #    graph (configured before the imports) still building the imports.
        if build_comps:
            patterns = sorted({
                p for p in (self._stage_relpath(c) for c in build_comps) if p
            })
            pipeline.append(pseudo(
                "shard-pin",
                ["python", buildctl, "enable",
                 *["^" + re.escape(p) + "$" for p in patterns],
                 "--force-reconfigure", "--build-dir", build_dir],
            ))

        # 4. Build: the build groups' own subprojects (in dependency order, from
        #    the elaborated task list) followed by each group's artifact target
        #    (ninja artifact-group-<g>, the native aggregate over the group's
        #    artifacts).
        for task in tasks:
            if sub_group.get(task.comp, set()) & build_set:
                pipeline.append(task)
        for shard in build_shards:
            pipeline.append(Task(
                comp="therock", action=f"artifact-group-{shard}", cfgname=None,
                single_config=True,
                payload={"target": f"artifact-group-{shard}", "bin": build_dir},
            ))

        # 5. Export: copy each producer group's built artifacts into the store
        #    (name-filtered, mirroring the import). Uses the helper rather than
        #    artifact_manager.py push, which is stage- (not group-) scoped.
        for shard in export_shards:
            names = sorted(topology.group_produced_names(topo, shard))
            pipeline.append(pseudo(
                f"export-{shard}",
                [
                    "python", SHARD_ARTIFACTS, "export-local",
                    "--build-dir", build_dir,
                    "--store", store,
                    "--names", ",".join(names),
                ],
            ))

        return pipeline
