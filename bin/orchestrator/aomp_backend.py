"""AOMP backend: the per-component ``build_<name>.sh`` contract.

Each component is a shell script speaking the command_dispatcher interface from
``aomp_utils``: it answers ``list_configs`` / ``list`` / ``show_src_dir`` for
introspection and runs ``task_<action> [cfg]`` for execution. The component
graph comes from a CUDF-style config (``configs/aomp.cudf``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import core, source_layout
from .backend import Backend
from .model import Config, Package, RawTask, Task

# bin/ directory (parent of this package), where the build_<name>.sh scripts
# and aomp_utils / aomp_common_vars live.
BIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(BIN_DIR, "configs", "aomp.cudf")

# Child build scripts run in an isolated environment built from scratch by this
# orchestrator, so that what gets built is under the orchestrator's control and
# not at the mercy of whatever the caller happened to have exported. Only the
# variables below are passed through unchanged: these are identity / locale /
# terminal settings that affect *how* things look or *who* git acts as, not
# *what* gets compiled. Everything else (CC, CXX, LD_LIBRARY_PATH, PKG_CONFIG_*,
# AOMP_*, ROCM_*, ...) is dropped unless the orchestrator sets it explicitly or
# the user opts in with --pass-env. Any variable named LC_* is also passed
# through (locale categories).
ENV_PASSTHROUGH = (
    "HOME", "USER", "LOGNAME", "SHELL", "TERM",
    "LANG", "LANGUAGE", "TZ", "TMPDIR",
    "DISPLAY", "XAUTHORITY",
    "SSH_AUTH_SOCK", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY",
)

# Deterministic PATH handed to child build scripts. Standard system locations
# only, so tool resolution is predictable and not shadowed by whatever the
# caller put earlier on their PATH (e.g. a Homebrew pkg-config that cannot see
# the system .pc files). Override with --inherit-path to use the caller's PATH.
DEFAULT_CHILD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Repositories consumed by component builds (e.g. via LLVM_EXTERNAL_PROJECTS)
# that are not standalone components. Paths are relative to AOMP_REPOS. These
# are recorded under the manifest "externals" section and restored on import
# because they are tightly coupled to the LLVM/comgr toolchain and are a
# frequent source of build breakage, so their exact versions matter.
EXTERNAL_REPOS = {
    "SPIRV-LLVM-Translator": "SPIRV-LLVM-Translator",
}

# Components whose source must never be rolled back on import: their checkout
# is meant to track HEAD. "extras" is the AOMP build-scripts repo (this very
# tree) -- pinning it would change the build logic mid-flight, and the scripts
# are intended to (eventually) build arbitrary AOMP/ROCm versions.
FLOATING_COMPONENTS = {"extras"}


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_cudf(path: str) -> Config:
    """Parse a CUDF-style config file into a Config.

    Stanzas are separated by blank lines; '#' lines are comments. A line whose
    first token has no trailing ':' is treated as a continuation of the
    previous key's value (used to wrap long comma-separated lists).
    """
    cfg = Config()
    order = 0

    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        core._fail(f"cannot read config '{path}': {exc}")

    stanza: dict[str, str] = {}

    def flush(stanza: dict[str, str]) -> None:
        nonlocal order
        if not stanza:
            return
        if "package" in stanza:
            name = stanza["package"]
            pkg = Package(
                name=name,
                depends=_split_list(stanza.get("depends", "")),
                xdir=stanza.get("x-dir", ".") or ".",
                order=order,
            )
            cfg.packages[name] = pkg
            order += 1
        elif "feature" in stanza:
            cfg.features[stanza["feature"]] = _split_list(stanza.get("expands", ""))
        elif "request" in stanza or "install" in stanza:
            cfg.request = _split_list(stanza.get("install", ""))

    last_key: str | None = None
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            flush(stanza)
            stanza = {}
            last_key = None
            continue
        if stripped.startswith("#"):
            continue
        # Continuation line: no "key:" at the start.
        if ":" not in stripped.split(" ", 1)[0] and last_key is not None:
            stanza[last_key] = (stanza.get(last_key, "") + " " + stripped).strip()
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        stanza[key] = value.strip()
        last_key = key
    flush(stanza)

    if not cfg.packages:
        core._fail(f"no packages found in config '{path}'")
    return cfg


class AompBackend(Backend):
    name = "aomp"

    def __init__(self) -> None:
        self._cfg: Config | None = None

    # --- configuration & environment ------------------------------------- #
    def load_config(self, args: argparse.Namespace) -> Config:
        self._cfg = parse_cudf(args.config)
        return self._cfg

    def config_name(self, args: argparse.Namespace) -> str:
        return os.path.splitext(os.path.basename(args.config))[0]

    def build_child_env(self, args: argparse.Namespace) -> dict[str, str]:
        """Construct the isolated environment for child build scripts.

        Rather than inheriting the caller's environment wholesale, the env is
        built from scratch: a curated pass-through of identity/locale vars
        (ENV_PASSTHROUGH plus LC_*), a controlled PATH, and the build knobs
        derived from CLI flags. --inherit-path leaks the caller's PATH through;
        --pass-env leaks named vars.
        """
        src = os.environ
        env: dict[str, str] = {}

        for name in ENV_PASSTHROUGH:
            if name in src:
                env[name] = src[name]
        for name, value in src.items():
            if name.startswith("LC_"):
                env[name] = value

        # PATH: controlled by default, optionally inherited from the caller.
        if args.inherit_path:
            env["PATH"] = src.get("PATH", DEFAULT_CHILD_PATH)
        else:
            env["PATH"] = DEFAULT_CHILD_PATH

        # Escape hatch: explicitly leak named variables for machine-specific needs.
        for name in args.pass_env:
            for var in (n.strip() for n in name.split(",")):
                if var and var in src:
                    env[var] = src[var]

        def setenv(name: str, value: str) -> None:
            env[name] = value

        def setdir(name: str, value: str | None) -> None:
            # Directory knobs are made absolute (with ~ expansion) so child
            # scripts resolve them identically regardless of working directory.
            if value is not None:
                env[name] = os.path.abspath(os.path.expanduser(value))

        setdir("AOMP_REPOS", args.source)
        setdir("AOMP", args.install)
        setdir("BUILD_AOMP", args.build)
        setdir("AOMP_SUPP", args.prereq)

        if args.jobs is not None:
            setenv("AOMP_JOB_THREADS", str(args.jobs))
        if args.ninja is not None:
            setenv("AOMP_USE_NINJA", "1" if args.ninja else "0")
        if args.ccache is not None:
            setenv("AOMP_USE_CCACHE", "1" if args.ccache else "0")
        if args.gfx:
            setenv("GFXLIST", args.gfx)
        # BUILD_TYPE is applied per-component at execution time (see run_tasks),
        # so it is intentionally not set in the shared child environment here.
        if args.sudo:
            setenv("SUDO", "yes")
        return env

    def discover_env(self, env: dict[str, str]) -> dict[str, str]:
        """Source aomp_utils + aomp_common_vars to learn BUILD_DIR/AOMP_REPOS.

        The same isolated child environment used for the build is passed in, so
        the discovered values reflect any -i/-b/-p directory overrides (e.g.
        BUILD_DIR follows --build, used for the default log/manifest locations).
        """
        snippet = (
            f'. "{BIN_DIR}/aomp_utils" >/dev/null 2>&1; '
            f'. "{BIN_DIR}/aomp_common_vars" >/dev/null 2>&1; '
            'printf "%s\\n" "$BUILD_DIR" "$AOMP_REPOS" "$AOMP_REPO_NAME" '
            '"$AOMP_INSTALL_DIR" "$AOMP"'
        )
        proc = subprocess.run(
            ["bash", "-c", snippet], capture_output=True, text=True, env=env
        )
        out = proc.stdout.splitlines()
        out += [""] * (5 - len(out))
        return {
            "BUILD_DIR": out[0] or os.path.join(os.path.expanduser("~"), "git", "aomp"),
            "AOMP_REPOS": out[1] or os.path.join(os.path.expanduser("~"), "git", "aomp"),
            "AOMP_REPO_NAME": out[2] or "aomp",
            # The versioned install dir (the real target the scripts symlink to)
            # and the symlink itself; used by -C/--clean to wipe a stale install.
            "AOMP_INSTALL_DIR": out[3],
            "AOMP": out[4],
        }

    # --- script invocation helpers --------------------------------------- #
    def _script_path(self, comp: str) -> str:
        assert self._cfg is not None
        xdir = self._cfg.packages[comp].xdir
        if xdir in (".", "", None):
            return os.path.join(BIN_DIR, f"build_{comp}.sh")
        return os.path.join(BIN_DIR, xdir, f"build_{comp}.sh")

    @staticmethod
    def _capture(path: str, args: list[str], env: dict[str, str]) -> str:
        """Run a build script and return stdout (used for introspection)."""
        proc = subprocess.run(
            ["bash", path, *args], capture_output=True, text=True, env=env,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            core._fail(
                f"'{os.path.basename(path)} {' '.join(args)}' "
                f"failed (rc={proc.returncode})"
            )
        return proc.stdout

    # --- task elaboration ------------------------------------------------- #
    def component_configs(self, comp: str, env: dict[str, str]) -> list[str]:
        script = self._script_path(comp)
        if not os.path.exists(script):
            core._fail(f"missing build script for '{comp}': {script}")
        out = self._capture(script, ["list_configs"], env)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def list_component_tasks(
        self, comp: str, env: dict[str, str]
    ) -> list[RawTask]:
        script = self._script_path(comp)
        listing = self._capture(script, ["list"], env).splitlines()
        raw: list[RawTask] = []
        for line in listing:
            toks = line.split()
            if not toks:
                continue
            action_tok = toks[0]
            if not action_tok.startswith("task_"):
                continue
            action = action_tok[len("task_"):]
            taskcfg = toks[1] if len(toks) > 1 else None
            raw.append(
                (action, taskcfg, {"script": script, "script_args": toks})
            )
        return raw

    # --- execution -------------------------------------------------------- #
    def task_command(
        self, task: Task, env: dict[str, str]
    ) -> tuple[list[str], dict[str, str]]:
        script = task.payload["script"]
        script_args = task.payload["script_args"]
        return ["bash", script, *script_args], {}

    # --- manifest / clean ------------------------------------------------- #
    def component_src_dir(self, comp: str, env: dict[str, str]) -> str:
        script = self._script_path(comp)
        if not os.path.exists(script):
            return ""
        return self._capture(script, ["show_src_dir"], env).strip()

    def external_repos(self, env: dict[str, str]) -> dict[str, str]:
        repos = self.discover_env(env)["AOMP_REPOS"]
        return {name: os.path.join(repos, rel) for name, rel in EXTERNAL_REPOS.items()}

    def floating_components(self) -> set[str]:
        return set(FLOATING_COMPONENTS)

    # --- source provisioning --------------------------------------------- #
    def _needs_rocmlibs(self, components: list[str]) -> bool:
        """True if any selected component is backed by the rocmlibs checkout
        (x-dir: rocmlibs in the config)."""
        assert self._cfg is not None
        for comp in components:
            pkg = self._cfg.packages.get(comp)
            if pkg is not None and pkg.xdir == "rocmlibs":
                return True
        return False

    def _run_clone_script(
        self, rel_script: str, env: dict[str, str], dry_run: bool
    ) -> int:
        script = os.path.join(BIN_DIR, rel_script)
        if not os.path.exists(script):
            core._warn(f"clone script not found: {script}")
            return 0
        print(f"--- {os.path.basename(script)} ---")
        if dry_run:
            print(f"  (dry-run) would run: bash {script}")
            return 0
        return subprocess.run(["bash", script], env=env).returncode

    def provision_sources(
        self, args: argparse.Namespace, env: dict[str, str],
        components: list[str],
    ) -> int:
        symlinks = getattr(args, "therock_symlinks", None)
        clone = getattr(args, "clone", False)
        if not symlinks and not clone:
            return 0

        repos = self.discover_env(env)["AOMP_REPOS"]
        dry = getattr(args, "dry_run", False)

        if symlinks:
            therock_dir = os.path.abspath(os.path.expanduser(symlinks))
            print(f"--- symlinking shared sources from {therock_dir} "
                  f"into {repos} ---")
            plan = source_layout.symlink_plan(repos, therock_dir)
            for act in plan:
                if act.status in ("skip-existing", "missing-target"):
                    core._warn(f"{act.comp.aomp_dir}: {act.note}")
                elif act.status == "already-linked":
                    print(f"  {act.comp.aomp_dir}: already linked")
            source_layout.apply_symlinks(plan, dry_run=dry)

        # Both --clone and --therock-symlinks fill in the remaining AOMP-only
        # repos via clone_aomp.sh (which skips any symlinked dirs).
        rc = self._run_clone_script("clone_aomp.sh", env, dry)
        if rc != 0:
            return rc
        if self._needs_rocmlibs(components):
            rc = self._run_clone_script(
                os.path.join("rocmlibs", "clone_rocmlibs.sh"), env, dry
            )
            if rc != 0:
                return rc
        return 0

    def install_clean_task(self, env_info: dict[str, str]) -> Task:
        """The -C/--clean pseudo-task: wipe the install directory (the versioned
        symlink *target*, not just the symlink). Inserted at the front of the
        task list so a stale/partially-installed tree is removed before building.
        """
        install_dir = env_info.get("AOMP_INSTALL_DIR", "")
        symlink = env_info.get("AOMP", "")
        return Task(
            comp="install", action="clean", cfgname=None,
            single_config=True, builtin="install_clean",
            targets=[install_dir, symlink],
        )
