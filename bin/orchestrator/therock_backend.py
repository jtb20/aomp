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

from . import core, topology
from .backend import Backend
from .model import Config, Package, RawTask, Task

# bin/ holds this package; srock-bin is a sibling of bin/ at the repo root.
BIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BIN_DIR)
SROCK_BIN_DIR = os.path.join(REPO_ROOT, "srock-bin")
SROCK_COMMON_VARS = os.path.join(SROCK_BIN_DIR, "srock_common_vars")
SETUP_SROCK = os.path.join(SROCK_BIN_DIR, "setup_srock.sh")

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

# The build type a component takes when --build-type does not name one. Dropping
# a previously-set --build-type therefore reverts that scope to this default
# (rather than retaining the cached value), which is treated as a configuration
# change requiring --reconfigure.
DEFAULT_BUILD_TYPE = "Release"

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
    """A config *name* the user passed via -c/--config, or None.

    -c/--config is unsupported for TheRock builds (the build set is chosen with
    --add). The aomp_build entry defaults --config to a .cudf *path* and the
    therock_build entry defaults it to None; empty/path/.cudf values are the
    inherited defaults, not an explicit srock config name, so they return None."""
    config = getattr(args, "config", None)
    if not config or os.sep in config or config.endswith(".cudf"):
        return None
    return config


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
        json_path = self._ensure_subproject_map(env, info)
        # The feature catalog is regenerated by the (re)configure above.
        self._features_meta = self._read_feature_map(
            os.path.join(info["BUILD_DIR"], FEATURE_MAP)
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
        return _srock_config(args)

    # --- feature selection (THEROCK_ENABLE_*) ----------------------------- #
    def _read_feature_map(self, path: str) -> dict[str, dict]:
        """Best-effort read of feature_map.json; {} if missing/malformed."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

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

    def _apply_feature_flags(
        self, env: dict[str, str], info: dict[str, str],
        args: argparse.Namespace,
    ) -> None:
        """Validate requested features and, under --reconfigure, append their
        -DTHEROCK_ENABLE_<X>=ON flags to SROCK_CMAKE_EXTRA so the (re)configure
        picks them up. Without --reconfigure, requesting a feature that is not
        already enabled is a hard error (a reconfigure would wipe build/)."""
        catalog = self._read_feature_map(
            os.path.join(info["BUILD_DIR"], FEATURE_MAP)
        )
        requested = self._requested_features(args, catalog)
        if not requested:
            return
        if not bool(getattr(args, "reconfigure", False)):
            disabled = sorted(
                f for f in requested
                if not catalog.get(f, {}).get("enabled", False)
            )
            if disabled:
                names = ", ".join(f.lower() for f in disabled)
                core._fail(
                    f"requested feature(s) not enabled in the current TheRock "
                    f"configuration: {names}.\n"
                    f"  Re-run with --reconfigure to apply them (reconfigures "
                    f"TheRock and regenerates the build maps)."
                )
            return  # already enabled; nothing to (re)configure
        flags = " ".join(
            f"-DTHEROCK_ENABLE_{f}=ON" for f in sorted(requested)
        )
        extra = env.get("SROCK_CMAKE_EXTRA", "")
        env["SROCK_CMAKE_EXTRA"] = f"{extra} {flags}".strip()
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

        # SROCK_CONFIG comes from --add (all / all-debug; minimal default).
        # -c/--config is unsupported for TheRock; warn and ignore if given.
        explicit = _explicit_config(args)
        if explicit:
            core._warn(
                f"-c/--config '{explicit}' is not supported for TheRock builds "
                "and is ignored; select the build set with --add instead "
                "(--add all | --add all-debug; minimal is the default)."
            )
        env["SROCK_CONFIG"] = _srock_config(args)
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
        reconfigure = bool(getattr(self._args, "reconfigure", False))

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

        therock_dir = info["SROCK_THEROCK_DIR"]
        run_env = dict(env)
        extra = run_env.get("SROCK_CMAKE_EXTRA", "")
        if "THEROCK_INTROSPECTION" not in extra:
            run_env["SROCK_CMAKE_EXTRA"] = (
                f"{extra} -DTHEROCK_INTROSPECTION=ON".strip()
            )

        # First use: clone + fetch sources (a full setup). Stock TheRock has no
        # introspection support yet, so this initial configure won't emit the
        # map -- that's fine; we inject support and reconfigure below.
        if not os.path.isdir(therock_dir):
            print(
                f"{core.PROG}: setting up TheRock (clone + fetch sources): "
                f"{SETUP_SROCK}", flush=True,
            )
            rc = subprocess.run(["bash", SETUP_SROCK], env=run_env).returncode
            if rc != 0:
                core._fail(f"TheRock setup failed (rc={rc})")

        # Ensure PR #1234's introspection support is present in the checkout.
        self._inject_introspection(therock_dir)

        # Reconfigure (reusing fetched sources) now that introspection is wired
        # in and -DTHEROCK_INTROSPECTION=ON is passed.
        print(
            f"{core.PROG}: reconfiguring TheRock with introspection: "
            f"{SETUP_SROCK} restart", flush=True,
        )
        rc = subprocess.run(
            ["bash", SETUP_SROCK, "restart"], env=run_env
        ).returncode
        if rc != 0:
            core._fail(f"TheRock configure failed (rc={rc})")
        if not os.path.isfile(json_path):
            core._fail(
                f"configure did not produce {json_path}; introspection injection "
                f"may have failed (see {therock_dir}/CMakeLists.txt)."
            )
        return json_path

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
        raw: list[RawTask] = []
        for action in FORWARD_ACTIONS:
            if action in available:
                raw.append(
                    (action, None, {"target": f"{comp}+{action}", "bin": build_dir})
                )
        return raw

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
        bin_dir = task.payload["bin"]
        target = task.payload["target"]
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
    def shard_run_lengths(
        self, tasks: list[Task], env: dict[str, str]
    ) -> list[int] | None:
        """Group the task list into contiguous runs by TheRock build *stage* so
        --shard cut points fall on stage boundaries. Best-effort: a subproject
        is mapped to a stage via the submodule named in its source path; tasks
        that don't map land in a trailing run. Returns None (count-based shard)
        when no topology is available."""
        info = self._env_info or self.discover_env(env)
        topo = topology.load_build_topology(info["SROCK_THEROCK_DIR"])
        if topo is None:
            return None
        rank_map = topology.submodule_stage_rank(topo)
        if not rank_map:
            return None

        unknown = max(rank_map.values()) + 1

        def task_rank(task: Task) -> int:
            src = self._meta.get(task.comp, {}).get("src", "")
            parts = set(src.split("/"))
            best: int | None = None
            for submodule, idx in rank_map.items():
                if submodule in parts:
                    best = idx if best is None else min(best, idx)
            return best if best is not None else unknown

        runs: list[int] = []
        prev: int | None = None
        for task in tasks:
            rank = task_rank(task)
            if prev is not None and rank == prev:
                runs[-1] += 1
            else:
                runs.append(1)
                prev = rank
        return runs
