#!/usr/bin/env python3
"""Tests for the TheRock orchestrator backend.

Synthetic-fixture tests (no real TheRock build) covering the full flow:
introspection JSON -> Config graph -> elaborated tasks -> ninja commands ->
group-based sharding. Run directly or via unittest:

    python3 bin/orchestrator/tests/test_therock_backend.py
    python3 -m unittest discover -s bin/orchestrator/tests
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

# Make `import orchestrator` resolve (bin/ is two levels up from this file).
BIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

from orchestrator import core, shard_artifacts, therock_backend, topology  # noqa: E402
from orchestrator.model import Task  # noqa: E402
from orchestrator.therock_backend import (  # noqa: E402
    DEFAULT_CHILD_PATH, DEFAULT_CONFIG, TheRockBackend,
)


class TaskIsDoneTests(unittest.TestCase):
    """`continue`/`list` completion honors both stamps and built_components."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _task(self, comp: str) -> Task:
        return Task(comp=comp, action="build", cfgname=None, single_config=True)

    def test_done_via_stamp(self) -> None:
        task = self._task("amd-llvm")
        with open(core.stamp_path(self.tmp, task, "done"), "w") as fh:
            fh.write("x")
        self.assertTrue(core.task_is_done(self.tmp, task, built=None))

    def test_done_via_built_components_without_stamp(self) -> None:
        # No stamp, but the backend reports the component as built (staged) ->
        # done, so bare `continue` skips it instead of rebuilding.
        task = self._task("amd-llvm")
        self.assertEqual(core.task_state(self.tmp, task), "none")
        self.assertTrue(core.task_is_done(self.tmp, task, built={"amd-llvm"}))

    def test_not_done_when_neither(self) -> None:
        task = self._task("amd-llvm")
        self.assertFalse(core.task_is_done(self.tmp, task, built={"rocm-cmake"}))
        self.assertFalse(core.task_is_done(self.tmp, task, built=None))

# A real TheRock checkout (with BUILD_TOPOLOGY.toml + build_tools) for the
# topology tests; skipped if not present.
THEROCK_SRC = os.environ.get("THEROCK_SRC", "/work3/julbrown/code/src/TheRock")

# A small but representative introspection map. Exercises:
#  * dependency ordering (rocm-cmake -> amd-llvm -> hipBLAS),
#  * build_deps that are not themselves subprojects (therock-googletest),
#  * NOTFOUND normalization (hipBLAS compiler_toolchain),
#  * pool/toolchain feature synthesis,
#  * the expunge action being dropped from the forward pipeline.
FIXTURE = {
    "amd-llvm": {
        "src": "compiler/amd-llvm", "bin": "compiler/amd-llvm/build",
        "install_dest": "lib/llvm", "build_deps": ["rocm-cmake"],
        "runtime_deps": [], "build_pool": "", "compiler_toolchain": "",
        "actions": ["configure", "build", "stage", "dist", "expunge"],
    },
    "rocm-cmake": {
        "src": "base/rocm-cmake", "bin": "base/rocm-cmake/build",
        "install_dest": "", "build_deps": [],
        "runtime_deps": [], "build_pool": "", "compiler_toolchain": "",
        "actions": ["configure", "build", "stage", "dist", "expunge"],
    },
    "hipBLAS": {
        "src": "rocm-libraries/projects/hipblas",
        "bin": "math-libs/BLAS/hipBLAS/build", "install_dest": "",
        "build_deps": ["rocm-cmake", "amd-llvm", "therock-googletest"],
        "runtime_deps": ["amd-comgr", "hipcc"],
        "build_pool": "therock_background",
        "compiler_toolchain": "THEROCK_COMPILER_TOOLCHAIN-NOTFOUND",
        "actions": ["configure", "build", "stage", "dist", "expunge"],
    },
}

# artifact_map.json companion: maps each topology artifact to the FIXTURE
# subprojects that compose it. Joined with BUILD_TOPOLOGY.toml's artifact->group
# relation, this maps subprojects onto artifact groups (the shard unit):
#   amd-llvm   -> group 'compiler', rocm-cmake -> 'base', hipBLAS -> 'math-libs'.
ARTIFACT_FIXTURE = {
    "amd-llvm": ["amd-llvm"],
    "base": ["rocm-cmake"],
    "blas": ["hipBLAS"],
}


def make_args(therock_dir: str, repos: str, *selectors: str):
    # The therock_build entry passes default_config=None (-c/--config is
    # unsupported for TheRock; the build set is chosen with --add).
    parser = core.build_arg_parser(
        "therock_build.py", None, inherit_path_note=DEFAULT_CHILD_PATH,
    )
    core.add_backend_options(parser, default_backend="therock")
    argv = ["--therock-dir", therock_dir, "-s", repos, *selectors]
    return parser.parse_args(argv)


class TheRockFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-test-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self):
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        cfg = backend.load_config(args)
        return backend, args, cfg

    def test_config_graph(self) -> None:
        backend, _, cfg = self._load()
        self.assertEqual(set(cfg.packages), {"amd-llvm", "rocm-cmake", "hipBLAS"})
        # build_deps drive ordering; non-subproject deps are filtered out.
        self.assertEqual(cfg.packages["amd-llvm"].depends, ["rocm-cmake"])
        self.assertEqual(
            sorted(cfg.packages["hipBLAS"].depends), ["amd-llvm", "rocm-cmake"]
        )
        self.assertNotIn("therock-googletest", cfg.packages["hipBLAS"].depends)
        # NOTFOUND properties normalize to empty.
        self.assertEqual(backend._meta["hipBLAS"]["compiler_toolchain"], "")
        # Convenience features from pool / toolchain.
        self.assertEqual(cfg.features.get("pool-therock_background"), ["hipBLAS"])
        # Whole stack requested by default.
        self.assertEqual(set(cfg.request), set(FIXTURE))

    def test_task_elaboration_order_and_names(self) -> None:
        backend, args, cfg = self._load()
        env = backend.build_child_env(args)
        components = core.resolve_components(cfg, args.add, args.remove)
        # Dependency order: rocm-cmake before amd-llvm before hipBLAS.
        self.assertEqual(components, ["rocm-cmake", "amd-llvm", "hipBLAS"])
        tasks = core.elaborate_tasks(backend, cfg, components, env, [], {})
        names = [t.name for t in tasks]
        # 3 forward actions per component (dist + expunge dropped), config-less.
        self.assertEqual(len(names), 9)
        self.assertEqual(
            names[:3],
            ["rocm-cmake/configure", "rocm-cmake/build", "rocm-cmake/stage"],
        )
        # dist (whole-distribution assembly) and expunge (destructive) excluded.
        self.assertTrue(all("/dist" not in n and "expunge" not in n for n in names))
        self.assertTrue(all(t.cfgname is None for t in tasks))

    def test_all_flag_exposes_every_action(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "-a", "list")
        cfg = backend.load_config(args)
        env = backend.build_child_env(args)
        components = core.resolve_components(cfg, args.add, args.remove)
        tasks = core.elaborate_tasks(backend, cfg, components, env, [], {})
        names = [t.name for t in tasks]
        # With --all every advertised action appears, in lifecycle order
        # (expunge -> configure -> build -> stage -> dist) per component.
        self.assertEqual(
            names[:5],
            [
                "rocm-cmake/expunge", "rocm-cmake/configure",
                "rocm-cmake/build", "rocm-cmake/stage", "rocm-cmake/dist",
            ],
        )
        # 5 actions x 3 components.
        self.assertEqual(len(names), 15)
        self.assertTrue(any(n.endswith("/dist") for n in names))
        self.assertTrue(any(n.endswith("/expunge") for n in names))

    def test_task_command_is_ninja_target(self) -> None:
        backend, args, cfg = self._load()
        env = backend.build_child_env(args)
        components = core.resolve_components(cfg, args.add, args.remove)
        tasks = core.elaborate_tasks(backend, cfg, components, env, [], {})
        task = next(t for t in tasks if t.name == "hipBLAS/build")
        cmd, extra = backend.task_command(task, env)
        self.assertEqual(cmd[:3], ["ninja", "-C", self.build])
        self.assertEqual(cmd[3], "hipBLAS+build")
        self.assertEqual(extra, {})

    def test_src_dir_and_externals(self) -> None:
        backend, args, _ = self._load()
        env = backend.build_child_env(args)
        self.assertEqual(
            backend.component_src_dir("amd-llvm", env),
            os.path.join(self.therock, "compiler/amd-llvm"),
        )
        self.assertEqual(backend.external_repos(env), {"TheRock": self.therock})

    def test_missing_map_without_reconfigure_fails(self) -> None:
        os.remove(os.path.join(self.build, "subproject_map.json"))
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        with self.assertRaises(SystemExit):
            backend.load_config(args)

    def test_trailing_tasks_are_whole_tree_dist_and_install(self) -> None:
        backend, args, _ = self._load()
        env = backend.build_child_env(args)
        trailing = backend.trailing_tasks(["amd-llvm"], env)
        self.assertEqual(
            [t.name for t in trailing], ["therock/dist", "therock/install"]
        )
        # Each runs a whole-tree ninja target in <build>.
        dist_cmd, _ = backend.task_command(trailing[0], env)
        self.assertEqual(dist_cmd[:3], ["ninja", "-C", self.build])
        self.assertEqual(dist_cmd[3], "therock-dist")
        install_cmd, _ = backend.task_command(trailing[1], env)
        self.assertEqual(install_cmd[3], "install")

    def test_leading_task_is_prereq_running_build_cmake(self) -> None:
        backend, args, _ = self._load()
        env = backend.build_child_env(args)
        leading = backend.leading_tasks(["amd-llvm"], env)
        self.assertEqual([t.name for t in leading], ["therock/prereq"])
        cmd, _ = backend.task_command(leading[0], env)
        self.assertEqual(cmd[0], "bash")
        self.assertTrue(cmd[1].endswith("build_cmake.sh"))

    def test_sysdeps_via_add_drives_cmake_extra(self) -> None:
        # Default: bundle OFF (host system deps). `--add sysdeps`: bundle ON.
        # The value is appended to SROCK_CMAKE_EXTRA (srock passes it last, so
        # it wins over the config block).
        env_off = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "list")
        )
        self.assertIn(
            "-DTHEROCK_BUNDLE_SYSDEPS=OFF", env_off["SROCK_CMAKE_EXTRA"]
        )
        env_on = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "--add", "sysdeps", "list")
        )
        self.assertIn("-DTHEROCK_BUNDLE_SYSDEPS=ON", env_on["SROCK_CMAKE_EXTRA"])

    def test_sysdeps_is_a_recognized_add_token(self) -> None:
        # `--add sysdeps` must not be rejected as an unknown component even when
        # no bundled sysdep components are present in the (OFF) map.
        _, _, cfg = self._load()
        self.assertIn("sysdeps", cfg.features)
        core.resolve_components(cfg, ["sysdeps"], [])

    def test_config_defaults_to_minimal(self) -> None:
        env = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "list")
        )
        self.assertEqual(env["SROCK_CONFIG"], DEFAULT_CONFIG)  # "minimal"

    def test_add_all_selects_all_config(self) -> None:
        env = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "--add", "all", "list")
        )
        self.assertEqual(env["SROCK_CONFIG"], "all")

    def test_add_all_debug_with_sysdeps(self) -> None:
        # all-debug wins, and a comma-joined sysdeps toggle is honored too.
        env = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "--add", "all-debug,sysdeps", "list")
        )
        self.assertEqual(env["SROCK_CONFIG"], "all-debug")
        self.assertIn("-DTHEROCK_BUNDLE_SYSDEPS=ON", env["SROCK_CMAKE_EXTRA"])

    def test_all_debug_beats_all(self) -> None:
        env = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "--add", "all,all-debug", "list")
        )
        self.assertEqual(env["SROCK_CONFIG"], "all-debug")

    def test_config_tokens_are_recognized_add(self) -> None:
        _, _, cfg = self._load()
        self.assertIn("all", cfg.features)
        self.assertIn("all-debug", cfg.features)
        core.resolve_components(cfg, ["all"], [])
        core.resolve_components(cfg, ["all-debug"], [])

    def test_default_source_config_sets_amd_staging_branches(self) -> None:
        # No -c/--config: the default source config (amd-staging) drives the
        # srock branch env vars, matching the historical srock_common_vars
        # defaults. Build scope is independent (minimal here).
        env = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "list")
        )
        self.assertEqual(env["SROCK_THEROCK_BRANCH"], "compiler/amd-staging")
        self.assertEqual(env["SROCK_COMPILER_BRANCH"], "amd-staging")
        self.assertEqual(env["SROCK_CONFIG"], DEFAULT_CONFIG)  # "minimal"

    def test_develop_source_config_sets_native_branches(self) -> None:
        # -c develop selects native upstream TheRock: main super-repo branch and
        # the develop sentinel (so setup_srock.sh skips the compiler override).
        env = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos, "-c", "develop", "list")
        )
        self.assertEqual(env["SROCK_THEROCK_BRANCH"], "main")
        self.assertEqual(env["SROCK_COMPILER_BRANCH"], "develop")

    def test_source_config_is_independent_of_build_scope(self) -> None:
        # --add all (scope) does not change the source config branches; -c
        # develop (source) does not change SROCK_CONFIG.
        env = TheRockBackend().build_child_env(
            make_args(self.therock, self.repos,
                      "-c", "develop", "--add", "all", "list")
        )
        self.assertEqual(env["SROCK_CONFIG"], "all")
        self.assertEqual(env["SROCK_THEROCK_BRANCH"], "main")
        self.assertEqual(env["SROCK_COMPILER_BRANCH"], "develop")

    def test_unknown_source_config_warns_and_falls_back(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf):
            env = TheRockBackend().build_child_env(
                make_args(self.therock, self.repos, "-c", "bogus", "list")
            )
        self.assertIn("unknown source config", buf.getvalue())
        # Falls back to the default (amd-staging) branches.
        self.assertEqual(env["SROCK_THEROCK_BRANCH"], "compiler/amd-staging")
        self.assertEqual(env["SROCK_COMPILER_BRANCH"], "amd-staging")

    def test_config_name_folds_source_config_and_scope(self) -> None:
        backend = TheRockBackend()
        self.assertEqual(
            backend.config_name(make_args(self.therock, self.repos, "list")),
            f"amd-staging-{DEFAULT_CONFIG}",
        )
        self.assertEqual(
            backend.config_name(make_args(
                self.therock, self.repos, "-c", "develop", "--add", "all", "list")),
            "develop-all",
        )

    def test_list_source_configs_reports_catalog(self) -> None:
        rows = TheRockBackend().list_source_configs()
        by_name = {r["name"]: r for r in rows}
        self.assertIn("amd-staging", by_name)
        self.assertIn("develop", by_name)
        self.assertTrue(by_name["amd-staging"]["default"])
        self.assertFalse(by_name["develop"]["default"])
        self.assertEqual(by_name["develop"]["therock_branch"], "main")

    def test_stage_relpath_maps_build_to_stage(self) -> None:
        backend, _, _ = self._load()
        self.assertEqual(
            backend._stage_relpath("amd-llvm"), "compiler/amd-llvm/stage"
        )
        self.assertEqual(
            backend._stage_relpath("hipBLAS"), "math-libs/BLAS/hipBLAS/stage"
        )


class SourceConfigSwitchTest(unittest.TestCase):
    """The shared checkout's source-config marker drives in-place switching.

    A configured checkout (subproject_map.json present) is normally reused as-is
    without --reconfigure. But if its recorded source config differs from the
    requested one, load_config must force a reconfigure (driving the branch
    switch in setup_srock.sh) and rewrite the marker."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-switch-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        # Minimal checkout shape so _inject_introspection's preconditions pass.
        os.makedirs(os.path.join(self.therock, "cmake"))
        os.makedirs(self.build)
        with open(os.path.join(self.therock, "CMakeLists.txt"), "w") as fh:
            fh.write("# fixture\n")
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)
        self.marker = os.path.join(self.therock, ".srock-source-config")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _marker(self) -> str | None:
        try:
            with open(self.marker) as fh:
                return fh.read().strip()
        except OSError:
            return None

    def _prime(self, backend: TheRockBackend, args) -> None:
        # discover_env shells out to source srock_common_vars; run that once now
        # (real) so its result is cached and a later subprocess.run mock only
        # sees the reconfigure call (not the env discovery).
        backend.discover_env(backend.build_child_env(args))

    def test_matching_marker_reuses_without_reconfigure(self) -> None:
        # Marker matches the (default) requested config -> no switch, the
        # existing map is reused and no configure subprocess runs.
        with open(self.marker, "w") as fh:
            fh.write("amd-staging\n")
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        self._prime(backend, args)
        from unittest import mock
        with mock.patch.object(therock_backend.subprocess, "run") as run:
            backend.load_config(args)
        run.assert_not_called()

    def test_mismatched_marker_forces_reconfigure_and_rewrites(self) -> None:
        # Marker says develop, but the default request is amd-staging -> a switch
        # is detected, a reconfigure is forced (setup_srock.sh restart), and the
        # marker is rewritten to the new config.
        with open(self.marker, "w") as fh:
            fh.write("develop\n")
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        self._prime(backend, args)
        from unittest import mock
        real_run = therock_backend.subprocess.run

        # Patching subprocess.run patches the module globally (incl. the safety
        # check's git queries), so let real git through and mock only the heavy
        # bash reconfigure. The fixture is not a git repo, so the safety scan
        # finds nothing and does not prompt.
        def dispatch(cmd, *a, **kw):
            if cmd[:1] == ["git"]:
                return real_run(cmd, *a, **kw)
            return mock.Mock(returncode=0)

        buf = io.StringIO()
        with mock.patch.object(therock_backend.subprocess, "run",
                               side_effect=dispatch) as run, \
                redirect_stdout(buf):
            backend.load_config(args)
        # A reconfigure (restart) ran despite no --reconfigure. A switch runs
        # setup_srock.sh restart *twice*: once to switch the branch/sources
        # (which reverts injected introspection), then again after re-injecting
        # so the introspection survives and the map is produced.
        self.assertTrue(run.called)
        restarts = [c for c in run.call_args_list
                    if c.args[0][:1] == ["bash"] and "restart" in " ".join(c.args[0])]
        self.assertEqual(len(restarts), 2)
        self.assertIn("source config switch", buf.getvalue())
        # Marker now reflects the requested config.
        self.assertEqual(self._marker(), "amd-staging")

    def test_missing_marker_and_no_git_does_not_force_switch(self) -> None:
        # A marker-less checkout that is also not a git repo: nothing to compare
        # against, so it is reused as-is; no forced reconfigure.
        self.assertIsNone(self._marker())
        self.assertFalse(os.path.isdir(os.path.join(self.therock, ".git")))
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        self._prime(backend, args)
        from unittest import mock
        with mock.patch.object(therock_backend.subprocess, "run") as run:
            backend.load_config(args)
        run.assert_not_called()

    def _git_init(self, branch: str) -> None:
        import subprocess as sp
        sp.run(["git", "init", "-q", self.therock], check=True)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        sp.run(["git", "-C", self.therock, "commit", "--allow-empty", "-q",
                "-m", "x"], check=True, env=env)
        sp.run(["git", "-C", self.therock, "branch", "-m", branch], check=True)

    def test_checked_out_branch_reads_git(self) -> None:
        self._git_init("compiler/amd-staging")
        backend = TheRockBackend()
        self.assertEqual(
            backend._checked_out_branch(self.therock), "compiler/amd-staging"
        )

    def test_marker_less_switch_falls_back_to_git_branch(self) -> None:
        # No marker, but the checkout is on `main` (develop's branch); the
        # default request (amd-staging -> compiler/amd-staging) differs, so the
        # actual git branch drives the switch and forces a reconfigure.
        self._git_init("main")
        self.assertIsNone(self._marker())
        backend = TheRockBackend()
        # -y so the destructive-switch safety guard (the fixture's untracked
        # files read as uncommitted changes) does not block; that guard is
        # covered by its own tests below.
        args = make_args(self.therock, self.repos, "-y", "list")
        self._prime(backend, args)
        from unittest import mock
        real_run = therock_backend.subprocess.run

        def dispatch(cmd, *a, **kw):
            # Let real git queries through; mock the heavy reconfigure (bash).
            if cmd[:1] == ["git"]:
                return real_run(cmd, *a, **kw)
            return mock.Mock(returncode=0)

        buf = io.StringIO()
        with mock.patch.object(therock_backend.subprocess, "run",
                               side_effect=dispatch) as run, \
                redirect_stdout(buf):
            backend.load_config(args)
        self.assertIn("source config switch", buf.getvalue())
        self.assertIn("branch main", buf.getvalue())
        # Two restarts: the branch switch, then the post-injection reconfigure.
        restarts = [c for c in run.call_args_list
                    if c.args[0][:1] == ["bash"] and "restart" in " ".join(c.args[0])]
        self.assertEqual(len(restarts), 2)
        # Marker is written to reflect the now-current config.
        self.assertEqual(self._marker(), "amd-staging")

    def test_marker_less_matching_git_branch_no_switch(self) -> None:
        # No marker, checkout already on the default config's branch -> no switch.
        self._git_init("compiler/amd-staging")
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        self._prime(backend, args)
        from unittest import mock
        real_run = therock_backend.subprocess.run

        def dispatch(cmd, *a, **kw):
            if cmd[:1] == ["git"]:
                return real_run(cmd, *a, **kw)
            return mock.Mock(returncode=0)

        with mock.patch.object(therock_backend.subprocess, "run",
                               side_effect=dispatch) as run:
            backend.load_config(args)
        self.assertFalse(
            any(c.args[0][:1] == ["bash"] for c in run.call_args_list)
        )

    # --- destructive-switch safety guard ------------------------------------ #
    def _git_init_committed(self, branch: str) -> None:
        import subprocess as sp
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        sp.run(["git", "init", "-q", self.therock], check=True)
        sp.run(["git", "-C", self.therock, "add", "-A"], check=True)
        sp.run(["git", "-C", self.therock, "commit", "-q", "-m", "init"],
               check=True, env=env)
        sp.run(["git", "-C", self.therock, "branch", "-m", branch], check=True)

    def _git_init_with_remote(self, branch: str) -> None:
        import subprocess as sp
        self._git_init_committed(branch)
        bare = os.path.join(self.tmp, "origin.git")
        sp.run(["git", "init", "-q", "--bare", bare], check=True)
        sp.run(["git", "-C", self.therock, "remote", "add", "origin", bare],
               check=True)
        sp.run(["git", "-C", self.therock, "push", "-q", "-u", "origin", branch],
               check=True)

    def test_safety_report_clean_tree_is_empty(self) -> None:
        self._git_init_committed("main")
        dirty, ahead = TheRockBackend()._switch_safety_report(self.therock)
        self.assertEqual((dirty, ahead), ([], []))

    def test_safety_report_flags_uncommitted_changes(self) -> None:
        self._git_init_committed("main")
        with open(os.path.join(self.therock, "CMakeLists.txt"), "a") as fh:
            fh.write("# local edit\n")
        dirty, ahead = TheRockBackend()._switch_safety_report(self.therock)
        self.assertIn("TheRock (super-repo)", dirty)
        self.assertEqual(ahead, [])

    def test_safety_report_flags_local_commits(self) -> None:
        self._git_init_with_remote("main")
        import subprocess as sp
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        sp.run(["git", "-C", self.therock, "commit", "--allow-empty", "-q",
                "-m", "local work"], check=True, env=env)
        dirty, ahead = TheRockBackend()._switch_safety_report(self.therock)
        self.assertIn("TheRock (super-repo)", ahead)

    def test_assert_switch_safe_aborts_on_decline(self) -> None:
        self._git_init_committed("main")
        with open(os.path.join(self.therock, "CMakeLists.txt"), "a") as fh:
            fh.write("# local edit\n")
        backend = TheRockBackend()
        backend._args = make_args(self.therock, self.repos, "list")  # no -y
        from unittest import mock
        with mock.patch("builtins.input", return_value="n"), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                backend._assert_switch_safe(self.therock)

    def test_assert_switch_safe_aborts_when_noninteractive(self) -> None:
        self._git_init_committed("main")
        with open(os.path.join(self.therock, "CMakeLists.txt"), "a") as fh:
            fh.write("# local edit\n")
        backend = TheRockBackend()
        backend._args = make_args(self.therock, self.repos, "list")  # no -y
        from unittest import mock
        with mock.patch("builtins.input", side_effect=EOFError), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                backend._assert_switch_safe(self.therock)

    def test_assert_switch_safe_proceeds_with_yes(self) -> None:
        self._git_init_committed("main")
        with open(os.path.join(self.therock, "CMakeLists.txt"), "a") as fh:
            fh.write("# local edit\n")
        backend = TheRockBackend()
        backend._args = make_args(self.therock, self.repos, "-y", "list")
        # -y: no prompt, no abort even though the tree is dirty.
        with redirect_stdout(io.StringIO()):
            backend._assert_switch_safe(self.therock)

    def test_assert_switch_safe_proceeds_on_confirm(self) -> None:
        self._git_init_committed("main")
        with open(os.path.join(self.therock, "CMakeLists.txt"), "a") as fh:
            fh.write("# local edit\n")
        backend = TheRockBackend()
        backend._args = make_args(self.therock, self.repos, "list")
        from unittest import mock
        with mock.patch("builtins.input", return_value="y"), \
                redirect_stdout(io.StringIO()):
            backend._assert_switch_safe(self.therock)  # no raise

    def test_assert_switch_safe_clean_tree_no_prompt(self) -> None:
        self._git_init_committed("main")
        backend = TheRockBackend()
        backend._args = make_args(self.therock, self.repos, "list")
        from unittest import mock
        # input must never be called for a clean tree.
        with mock.patch("builtins.input",
                        side_effect=AssertionError("should not prompt")):
            backend._assert_switch_safe(self.therock)

    def test_dirty_switch_aborts_before_running_setup(self) -> None:
        # End-to-end: a marker mismatch + dirty tree, no -y, declined -> abort
        # before any setup_srock.sh runs (work is preserved).
        self._git_init_committed("main")
        with open(self.marker, "w") as fh:
            fh.write("develop\n")  # forces switch to default amd-staging
        with open(os.path.join(self.therock, "CMakeLists.txt"), "a") as fh:
            fh.write("# local edit\n")
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")  # no -y
        self._prime(backend, args)
        from unittest import mock
        real_run = therock_backend.subprocess.run

        def dispatch(cmd, *a, **kw):
            if cmd[:1] == ["git"]:
                return real_run(cmd, *a, **kw)
            return mock.Mock(returncode=0)

        with mock.patch.object(therock_backend.subprocess, "run",
                               side_effect=dispatch) as run, \
                mock.patch("builtins.input", return_value="n"), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                backend.load_config(args)
        # No setup_srock.sh (bash) invocation happened -> nothing was reset.
        self.assertFalse(
            any(c.args[0][:1] == ["bash"] for c in run.call_args_list)
        )


class DefaultRequestTest(unittest.TestCase):
    """The default request mirrors a native build: real ROCm components plus
    their build+runtime closure, excluding unused vendored third-party libs."""

    # therock-simde is a build_dep of a real component (-> requested via
    # closure); therock-msgpack-cxx is a runtime_dep of a real component (->
    # requested via closure); therock-boost is an orphan vendored lib that no
    # built component needs (-> declared but NOT requested).
    FIXTURE = {
        "amd-llvm": {
            "src": "compiler/amd-llvm", "bin": "compiler/amd-llvm/build",
            "install_dest": "lib/llvm", "build_deps": ["therock-simde"],
            "runtime_deps": [], "build_pool": "", "compiler_toolchain": "",
            "actions": ["configure", "build", "stage"],
        },
        "rocm-kpack": {
            "src": "base/rocm-kpack", "bin": "base/rocm-kpack/build",
            "install_dest": "", "build_deps": [],
            "runtime_deps": ["therock-msgpack-cxx"], "build_pool": "",
            "compiler_toolchain": "", "actions": ["configure", "build", "stage"],
        },
        "therock-simde": {
            "src": "third-party/simde", "bin": "third-party/simde/build",
            "install_dest": "", "build_deps": [], "runtime_deps": [],
            "build_pool": "", "compiler_toolchain": "",
            "actions": ["configure", "build", "stage"],
        },
        "therock-msgpack-cxx": {
            "src": "third-party/msgpack", "bin": "third-party/msgpack/build",
            "install_dest": "", "build_deps": [], "runtime_deps": [],
            "build_pool": "", "compiler_toolchain": "",
            "actions": ["configure", "build", "stage"],
        },
        "therock-boost": {
            "src": "third-party/boost", "bin": "third-party/boost/build",
            "install_dest": "", "build_deps": [], "runtime_deps": [],
            "build_pool": "", "compiler_toolchain": "",
            "actions": ["configure", "build", "stage"],
        },
    }

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-req-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(self.FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self):
        backend = TheRockBackend()
        cfg = backend.load_config(make_args(self.therock, self.repos, "list"))
        return backend, cfg

    def test_request_is_real_components_plus_needed_closure(self) -> None:
        _, cfg = self._load()
        # All subprojects remain declared so --add can opt any of them in.
        self.assertEqual(set(cfg.packages), set(self.FIXTURE))
        # Real components + the vendored libs they depend on (build & runtime);
        # the orphan vendored lib (therock-boost) is excluded.
        self.assertEqual(
            set(cfg.request),
            {"amd-llvm", "rocm-kpack", "therock-simde", "therock-msgpack-cxx"},
        )
        self.assertNotIn("therock-boost", cfg.request)

    def test_thirdparty_feature_lists_all_vendored(self) -> None:
        _, cfg = self._load()
        self.assertEqual(
            cfg.features.get("thirdparty"),
            ["therock-boost", "therock-msgpack-cxx", "therock-simde"],
        )

    def test_add_thirdparty_opts_orphan_back_in(self) -> None:
        backend, cfg = self._load()
        components = core.resolve_components(cfg, ["thirdparty"], [])
        self.assertIn("therock-boost", components)


class BuildDelegationTest(unittest.TestCase):
    """The 'build' stage delegates to in-subproject-dir ninja once configured.

    TheRock "Option 1": once a subproject's own build.ninja exists, building it
    runs `ninja -C <subproject build dir>` (which detects source edits the
    super-project's stamp tracking misses) instead of `ninja <comp>+build`.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-delegate-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _task(self, backend, args, name: str):
        env = backend.build_child_env(args)
        cfg = backend.load_config(args)
        components = core.resolve_components(cfg, args.add, args.remove)
        tasks = core.elaborate_tasks(backend, cfg, components, env, [], {})
        return next(t for t in tasks if t.name == name), env

    def _make_build_ninja(self, comp: str) -> str:
        sub_dir = os.path.join(self.build, FIXTURE[comp]["bin"])
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "build.ninja"), "w") as fh:
            fh.write("# fake\n")
        return sub_dir

    def test_build_without_ninja_uses_superlevel(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        task, env = self._task(backend, args, "amd-llvm/build")
        cmd, _ = backend.task_command(task, env)
        self.assertEqual(cmd, ["ninja", "-C", self.build, "amd-llvm+build"])

    def test_build_with_ninja_delegates_to_subdir(self) -> None:
        sub_dir = self._make_build_ninja("amd-llvm")
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        task, env = self._task(backend, args, "amd-llvm/build")
        cmd, _ = backend.task_command(task, env)
        self.assertEqual(cmd, ["ninja", "-C", sub_dir])
        self.assertNotIn("amd-llvm+build", cmd)

    def test_superproject_build_flag_disables_delegation(self) -> None:
        self._make_build_ninja("amd-llvm")
        backend = TheRockBackend()
        args = make_args(
            self.therock, self.repos, "--superproject-build", "list",
        )
        task, env = self._task(backend, args, "amd-llvm/build")
        cmd, _ = backend.task_command(task, env)
        self.assertEqual(cmd, ["ninja", "-C", self.build, "amd-llvm+build"])

    def test_configure_and_stage_never_delegate(self) -> None:
        # Even with build.ninja present, only the build stage delegates.
        self._make_build_ninja("amd-llvm")
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        for stage, target in (("configure", "amd-llvm+configure"),
                               ("stage", "amd-llvm+stage")):
            task, env = self._task(backend, args, f"amd-llvm/{stage}")
            cmd, _ = backend.task_command(task, env)
            self.assertEqual(cmd, ["ninja", "-C", self.build, target])

    def test_delegated_build_appends_jobs(self) -> None:
        sub_dir = self._make_build_ninja("amd-llvm")
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "-j", "8", "list")
        task, env = self._task(backend, args, "amd-llvm/build")
        cmd, _ = backend.task_command(task, env)
        self.assertEqual(cmd, ["ninja", "-C", sub_dir, "-j", "8"])


class FeatureSelectionTest(unittest.TestCase):
    """THEROCK_ENABLE_* feature selection via --add + the feature catalog."""

    FEATURE_FIXTURE = {
        "HIPDNN": {
            "enabled": False, "description": "Enables hipdnn", "feature": True,
            "requires": ["CORE_RUNTIME", "HIP_RUNTIME"],
        },
        "ML_LIBS": {
            "enabled": False, "description": "Enable building of ML libraries",
            "feature": False, "requires": [],
        },
        "COMPILER": {
            "enabled": True, "description": "Enable building of the compiler",
            "feature": True, "requires": [],
        },
        "AMD_DBGAPI": {
            "enabled": True, "description": "Enable amd-dbgapi",
            "feature": True, "requires": [],
        },
        "ROCGDB": {
            "enabled": True, "description": "Enable rocgdb",
            "feature": True, "requires": ["AMD_DBGAPI"],
        },
    }

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-feat-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)
        with open(os.path.join(self.build, "feature_map.json"), "w") as fh:
            json.dump(self.FEATURE_FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _backend_env(self, *selectors: str):
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, *selectors)
        env = backend.build_child_env(args)
        info = backend.discover_env(env)
        return backend, args, env, info

    def test_add_feature_with_reconfigure_appends_enable_flag(self) -> None:
        backend, args, env, info = self._backend_env(
            "--add", "hipdnn", "--reconfigure", "list",
        )
        backend._apply_feature_flags(env, info, args)
        self.assertIn("-DTHEROCK_ENABLE_HIPDNN=ON", env["SROCK_CMAKE_EXTRA"])

    def test_feature_token_is_normalized(self) -> None:
        # `ml-libs` -> THEROCK_ENABLE_ML_LIBS.
        backend, args, env, info = self._backend_env(
            "--add", "ml-libs", "--reconfigure", "list",
        )
        backend._apply_feature_flags(env, info, args)
        self.assertIn("-DTHEROCK_ENABLE_ML_LIBS=ON", env["SROCK_CMAKE_EXTRA"])

    def test_disabled_feature_without_reconfigure_fails(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "--add", "hipdnn", "list")
        with self.assertRaises(SystemExit):
            backend.load_config(args)

    def test_enabled_feature_without_reconfigure_ok(self) -> None:
        # COMPILER is already enabled, so requesting it needs no reconfigure and
        # no enable flag is appended.
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "--add", "compiler", "list")
        cfg = backend.load_config(args)
        self.assertNotIn(
            "-DTHEROCK_ENABLE_COMPILER=ON",
            backend.build_child_env(args).get("SROCK_CMAKE_EXTRA", ""),
        )
        # Feature tokens (and their spellings) are recognized by resolve_components.
        self.assertIn("compiler", cfg.features)
        self.assertIn("ml-libs", cfg.features)
        core.resolve_components(cfg, ["hipdnn", "ml-libs"], [])

    def test_remove_feature_with_reconfigure_appends_off_flag(self) -> None:
        # Removing rocgdb (no dependents) disables only ROCGDB.
        backend, args, env, info = self._backend_env(
            "--remove", "rocgdb", "--reconfigure", "list",
        )
        backend._apply_feature_flags(env, info, args)
        self.assertIn("-DTHEROCK_ENABLE_ROCGDB=OFF", env["SROCK_CMAKE_EXTRA"])
        self.assertNotIn(
            "-DTHEROCK_ENABLE_AMD_DBGAPI=OFF", env["SROCK_CMAKE_EXTRA"]
        )

    def test_remove_feature_cascades_to_dependents(self) -> None:
        # Removing amd-dbgapi also disables ROCGDB, which requires it.
        backend, args, env, info = self._backend_env(
            "--remove", "amd-dbgapi", "--reconfigure", "list",
        )
        backend._apply_feature_flags(env, info, args)
        self.assertIn(
            "-DTHEROCK_ENABLE_AMD_DBGAPI=OFF", env["SROCK_CMAKE_EXTRA"]
        )
        self.assertIn("-DTHEROCK_ENABLE_ROCGDB=OFF", env["SROCK_CMAKE_EXTRA"])

    def test_remove_enabled_feature_without_reconfigure_fails(self) -> None:
        backend = TheRockBackend()
        args = make_args(
            self.therock, self.repos, "--remove", "amd-dbgapi", "list"
        )
        with self.assertRaises(SystemExit):
            backend.load_config(args)

    def test_add_and_remove_same_feature_conflicts(self) -> None:
        backend, args, env, info = self._backend_env(
            "--add", "rocgdb", "--remove", "rocgdb", "--reconfigure", "list",
        )
        with self.assertRaises(SystemExit):
            backend._apply_feature_flags(env, info, args)

    def test_remove_already_disabled_feature_is_noop(self) -> None:
        # hipdnn is not enabled, so removing it changes nothing and needs no
        # reconfigure.
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "--remove", "hipdnn", "list")
        backend.load_config(args)
        self.assertNotIn(
            "-DTHEROCK_ENABLE_HIPDNN=OFF",
            backend.build_child_env(args).get("SROCK_CMAKE_EXTRA", ""),
        )

    def test_remove_unknown_feature_is_noop(self) -> None:
        backend, args, env, info = self._backend_env(
            "--remove", "does-not-exist", "--reconfigure", "list",
        )
        backend._apply_feature_flags(env, info, args)
        self.assertNotIn(
            "-DTHEROCK_ENABLE_DOES_NOT_EXIST=OFF",
            env.get("SROCK_CMAKE_EXTRA", ""),
        )

    def test_list_features_rows(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list-features")
        backend.load_config(args)
        rows = backend.list_features(backend.build_child_env(args))
        by_name = {r["name"]: r for r in rows}
        self.assertFalse(by_name["HIPDNN"]["enabled"])
        self.assertEqual(
            by_name["HIPDNN"]["requires"], ["CORE_RUNTIME", "HIP_RUNTIME"]
        )
        self.assertTrue(by_name["COMPILER"]["enabled"])

    def test_no_feature_map_yields_empty_catalog(self) -> None:
        # A configure that predates feature_map.json: list_features is empty,
        # and feature tokens are simply not registered.
        os.remove(os.path.join(self.build, "feature_map.json"))
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        backend.load_config(args)
        self.assertEqual(backend.list_features(backend.build_child_env(args)), [])


class GfxNormalizeTest(unittest.TestCase):
    """--gfx accepts comma- and/or space-separated targets -> GFXLIST form."""

    def test_normalizes_to_space_separated(self) -> None:
        self.assertEqual(
            core.normalize_gfx_list("gfx90a,gfx942"), "gfx90a gfx942"
        )
        self.assertEqual(
            core.normalize_gfx_list("gfx90a gfx942"), "gfx90a gfx942"
        )
        # Mixed separators and extra whitespace collapse to single spaces.
        self.assertEqual(
            core.normalize_gfx_list(" gfx90a , gfx942,gfx1100 "),
            "gfx90a gfx942 gfx1100",
        )

    def test_applied_by_arg_parser(self) -> None:
        # Parsing doesn't touch the filesystem; dummy dirs are fine here.
        args = make_args("/x", "/y", "--gfx", "gfx90a,gfx942", "list")
        self.assertEqual(args.gfx, "gfx90a gfx942")


class BuildTypeTest(unittest.TestCase):
    """--build-type -> TheRock configure-time -D flags (requires --reconfigure)."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-bt-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_cache(self, entries: dict[str, str]) -> None:
        with open(os.path.join(self.build, "CMakeCache.txt"), "w") as fh:
            fh.write("# CMakeCache test fixture\n")
            for name, value in entries.items():
                fh.write(f"{name}:STRING={value}\n")

    def _apply(self, *flags: str):
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, *flags, "list")
        env = backend.build_child_env(args)
        backend._apply_build_type_flags(env, {"BUILD_DIR": self.build}, args)
        return env

    def test_reconfigure_appends_global_and_per_comp_flags(self) -> None:
        # No (matching) cache -> these are real changes; --reconfigure applies.
        env = self._apply(
            "--reconfigure", "--build-type", "Release",
            "--build-type", "amd-llvm=Debug",
        )
        extra = env["SROCK_CMAKE_EXTRA"]
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", extra)
        self.assertIn("-Damd-llvm_BUILD_TYPE=Debug", extra)

    def test_change_without_reconfigure_is_an_error(self) -> None:
        # Cache is Debug; requesting Release is a change -> needs --reconfigure.
        self._write_cache({"CMAKE_BUILD_TYPE": "Debug"})
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "--build-type", "Release", "list")
        env = backend.build_child_env(args)
        with self.assertRaises(SystemExit):
            backend._apply_build_type_flags(env, {"BUILD_DIR": self.build}, args)

    def test_matching_cache_is_idempotent_noop(self) -> None:
        # Requested values already match the cache -> no error, no flags, even
        # without --reconfigure (safe to leave --build-type on the command line).
        self._write_cache(
            {"CMAKE_BUILD_TYPE": "Release", "amd-llvm_BUILD_TYPE": "Debug"}
        )
        env = self._apply(
            "--build-type", "Release", "--build-type", "amd-llvm=Debug",
        )
        self.assertNotIn("-DCMAKE_BUILD_TYPE", env["SROCK_CMAKE_EXTRA"])
        self.assertNotIn("_BUILD_TYPE", env["SROCK_CMAKE_EXTRA"])

    def test_dropping_per_comp_override_is_a_change(self) -> None:
        # Cache still carries a per-component override but the command line no
        # longer asks for it -> the scope must revert to the default (Release),
        # which is a change and needs --reconfigure.
        self._write_cache({"amd-llvm_BUILD_TYPE": "Debug"})
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")  # no --build-type
        env = backend.build_child_env(args)
        with self.assertRaises(SystemExit):
            backend._apply_build_type_flags(env, {"BUILD_DIR": self.build}, args)

    def test_dropping_per_comp_override_reverts_to_default_on_reconfigure(self) -> None:
        self._write_cache({"amd-llvm_BUILD_TYPE": "Debug"})
        env = self._apply("--reconfigure")  # no --build-type
        self.assertIn("-Damd-llvm_BUILD_TYPE=Release", env["SROCK_CMAKE_EXTRA"])

    def test_no_overrides_no_request_is_noop(self) -> None:
        # The common case: no --build-type and no cached overrides -> nothing to
        # do, no error, no build-type flags (clean trees are unaffected).
        self._write_cache({"CMAKE_BUILD_TYPE": "Release"})
        env = self._apply()  # no --build-type, no --reconfigure
        self.assertNotIn("_BUILD_TYPE", env["SROCK_CMAKE_EXTRA"])

    def test_partial_change_against_cache_needs_reconfigure(self) -> None:
        # Global matches but a per-comp differs -> still a change -> error.
        self._write_cache(
            {"CMAKE_BUILD_TYPE": "Release", "amd-llvm_BUILD_TYPE": "Release"}
        )
        backend = TheRockBackend()
        args = make_args(
            self.therock, self.repos, "--build-type", "Release",
            "--build-type", "amd-llvm=Debug", "list",
        )
        env = backend.build_child_env(args)
        with self.assertRaises(SystemExit):
            backend._apply_build_type_flags(env, {"BUILD_DIR": self.build}, args)

    def test_unknown_component_warns(self) -> None:
        backend = TheRockBackend()
        cfg = backend.load_config(make_args(self.therock, self.repos, "list"))
        args = make_args(
            self.therock, self.repos, "--reconfigure",
            "--build-type", "nope=Debug", "list",
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            backend._warn_unknown_build_type_comps(cfg, args)
        self.assertIn("nope", buf.getvalue())
        self.assertIn("not a known subproject", buf.getvalue())


class PrepareRunTest(unittest.TestCase):
    """Auto-pin / unpin wiring around buildctl.py (dry-run, no real build)."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-pin-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        # buildctl.py must exist for prepare_run to emit a (dry-run) command.
        os.makedirs(os.path.join(self.therock, "build_tools"))
        open(os.path.join(self.therock, "build_tools", "buildctl.py"), "w").close()
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _backend(self, *selectors: str):
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "-n", *selectors)
        backend.load_config(args)
        return backend, args

    def test_subset_auto_pins_complement_via_enable(self) -> None:
        backend, args = self._backend("amd-llvm/build")
        env = backend.build_child_env(args)
        buf = io.StringIO()
        with redirect_stdout(buf):
            backend.prepare_run({"amd-llvm"}, env, args)
        out = buf.getvalue()
        # enable <working-set> makes only the working set buildable (pins rest).
        self.assertIn("would run:", out)
        self.assertIn("buildctl.py enable", out)
        pat = "^" + re.escape("compiler/amd-llvm/stage") + "$"
        self.assertIn(pat, out)
        # The other components' stage dirs are NOT named (they get pinned).
        self.assertNotIn("hipBLAS", out)

    def test_full_build_does_not_pin(self) -> None:
        backend, args = self._backend()
        env = backend.build_child_env(args)
        buf = io.StringIO()
        with redirect_stdout(buf):
            backend.prepare_run(set(FIXTURE), env, args)
        self.assertNotIn("would run:", buf.getvalue())

    def test_no_auto_pin_flag_disables(self) -> None:
        backend, args = self._backend("--no-auto-pin", "amd-llvm/build")
        env = backend.build_child_env(args)
        buf = io.StringIO()
        with redirect_stdout(buf):
            backend.prepare_run({"amd-llvm"}, env, args)
        self.assertNotIn("would run:", buf.getvalue())

    def test_unpin_all_clears_markers(self) -> None:
        backend, args = self._backend("--unpin-all", "amd-llvm/build")
        env = backend.build_child_env(args)
        buf = io.StringIO()
        with redirect_stdout(buf):
            backend.prepare_run({"amd-llvm"}, env, args)
        out = buf.getvalue()
        self.assertIn("buildctl.py enable --build-dir", out)
        # A bare 'enable' (no patterns) clears all markers.
        self.assertNotIn("/stage$", out)


def _make_stage(build: str, relpath: str, empty: bool = False) -> None:
    """Create a (by default non-empty) stage dir under build at relpath."""
    stage = os.path.join(build, *relpath.split("/"))
    os.makedirs(stage, exist_ok=True)
    if not empty:
        open(os.path.join(stage, "marker"), "w").close()


# Stage relpaths derived from the FIXTURE bin paths (build -> stage sibling).
STAGE_RELPATHS = {
    "amd-llvm": "compiler/amd-llvm/stage",
    "rocm-cmake": "base/rocm-cmake/stage",
    "hipBLAS": "math-libs/BLAS/hipBLAS/stage",
}


class BuiltComponentsTest(unittest.TestCase):
    """built_components() reports comps with a valid (non-empty) stage dir."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-built-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_only_valid_stage_dirs_count_as_built(self) -> None:
        _make_stage(self.build, STAGE_RELPATHS["amd-llvm"])
        _make_stage(self.build, STAGE_RELPATHS["rocm-cmake"])
        # hipBLAS has an *empty* stage dir -> not built; (no dir is also not).
        _make_stage(self.build, STAGE_RELPATHS["hipBLAS"], empty=True)
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        backend.load_config(args)
        built = backend.built_components(backend.build_child_env(args))
        self.assertEqual(built, {"amd-llvm", "rocm-cmake"})


class ReverseDepClosureTest(unittest.TestCase):
    """reverse_dep_closure walks dependents over the build-dep graph."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-rdep-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)
        backend = TheRockBackend()
        self.cfg = backend.load_config(make_args(self.therock, self.repos, "list"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_closure_pulls_in_transitive_dependents(self) -> None:
        # Graph: rocm-cmake <- amd-llvm <- hipBLAS (hipBLAS also <- rocm-cmake).
        self.assertEqual(
            core.reverse_dep_closure(self.cfg, {"rocm-cmake"}),
            {"rocm-cmake", "amd-llvm", "hipBLAS"},
        )
        self.assertEqual(
            core.reverse_dep_closure(self.cfg, {"amd-llvm"}),
            {"amd-llvm", "hipBLAS"},
        )
        self.assertEqual(
            core.reverse_dep_closure(self.cfg, {"hipBLAS"}), {"hipBLAS"}
        )

    def test_runtime_dep_edges_propagate(self) -> None:
        # rocgdb consumes amd-llvm only via runtime_deps (build_deps empty), as
        # in the real introspection map. A compiler change must still pull
        # rocgdb into the rebuild set, so the backend records a runtime_depends
        # edge and reverse_dep_closure follows it.
        amd = dict(FIXTURE["amd-llvm"], build_deps=[], runtime_deps=[])
        m = {
            "amd-llvm": amd,
            "rocgdb": {
                "src": "tools/rocgdb", "bin": "tools/rocgdb/build",
                "install_dest": "", "build_deps": [],
                "runtime_deps": ["amd-llvm"], "build_pool": "",
                "compiler_toolchain": "",
                "actions": ["configure", "build", "stage", "dist", "expunge"],
            },
        }
        tmp = tempfile.mkdtemp(prefix="therock-rt-")
        try:
            therock = os.path.join(tmp, "TheRock")
            build = os.path.join(therock, "build")
            os.makedirs(build)
            with open(os.path.join(build, "subproject_map.json"), "w") as fh:
                json.dump(m, fh)
            cfg = TheRockBackend().load_config(
                make_args(therock, os.path.join(tmp, "repos"), "list")
            )
            self.assertEqual(cfg.packages["rocgdb"].depends, [])
            self.assertEqual(
                cfg.packages["rocgdb"].runtime_depends, ["amd-llvm"]
            )
            self.assertEqual(
                core.reverse_dep_closure(cfg, {"amd-llvm"}),
                {"amd-llvm", "rocgdb"},
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RdepsRunTest(unittest.TestCase):
    """--rdeps expands a subset run to its dependents (dry-run, via core.run)."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-rdeps-run-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        os.makedirs(os.path.join(self.therock, "build_tools"))
        open(os.path.join(self.therock, "build_tools", "buildctl.py"), "w").close()
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *selectors: str) -> str:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "-n", *selectors)
        buf = io.StringIO()
        with redirect_stdout(buf):
            core.run(args, backend)
        return buf.getvalue()

    def test_rdeps_rebuilds_dependents(self) -> None:
        out = self._run("--rdeps", "amd-llvm/build")
        # hipBLAS depends on amd-llvm, so the auto-pin working set now includes
        # it (focusing on 2 components) and its tasks are scheduled.
        self.assertIn("hipBLAS", out)
        self.assertIn("amd-llvm", out)

    def test_without_rdeps_dependents_are_pinned(self) -> None:
        out = self._run("amd-llvm/build")
        # Only amd-llvm builds; hipBLAS is left out (pinned), not named anywhere.
        self.assertNotIn("hipBLAS", out)


class ListPinnedTest(unittest.TestCase):
    """`list` shows [pinned] for built comps outside the previewed set."""

    CHECK = "\u2713"

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-listpin-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)
        for rel in STAGE_RELPATHS.values():  # every comp is built
            _make_stage(self.build, rel)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _list(self, *preview: str, rdeps: bool = False) -> dict[str, str]:
        backend = TheRockBackend()
        flags = ("--rdeps",) if rdeps else ()
        # Optional flags must precede the positional selectors for argparse.
        args = make_args(self.therock, self.repos, *flags, "list", *preview)
        buf = io.StringIO()
        with redirect_stdout(buf):
            core.run(args, backend)
        # Map component -> its first task line for easy assertions.
        rows: dict[str, str] = {}
        for line in buf.getvalue().splitlines():
            for comp in FIXTURE:
                if f"] {comp}/" in line and comp not in rows:
                    rows[comp] = line
        return rows

    def test_preview_marks_excluded_built_comps_pinned(self) -> None:
        rows = self._list("amd-llvm")
        self.assertNotIn("[pinned]", rows["amd-llvm"])
        # rocm-cmake (a dep) and hipBLAS (a dependent) are both built + outside
        # the focused set -> pinned.
        self.assertIn("[pinned]", rows["rocm-cmake"])
        self.assertIn("[pinned]", rows["hipBLAS"])
        # Built comps render with a done check, never blank-and-pinned.
        self.assertIn(self.CHECK, rows["hipBLAS"])

    def test_full_list_pins_nothing(self) -> None:
        rows = self._list()
        for line in rows.values():
            self.assertNotIn("[pinned]", line)

    def test_rdeps_preview_unpins_dependents(self) -> None:
        rows = self._list("amd-llvm", rdeps=True)
        # With --rdeps the dependent hipBLAS joins the build set (not pinned);
        # the forward dep rocm-cmake stays pinned.
        self.assertNotIn("[pinned]", rows["hipBLAS"])
        self.assertIn("[pinned]", rows["rocm-cmake"])


class InjectIntrospectionTest(unittest.TestCase):
    """The bootstrap must make a stock TheRock checkout introspectable."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-inject-")
        self.therock = os.path.join(self.tmp, "TheRock")
        os.makedirs(os.path.join(self.therock, "cmake"))
        # A stock-ish top-level CMakeLists with no introspection support.
        self.cmakelists = os.path.join(self.therock, "CMakeLists.txt")
        with open(self.cmakelists, "w") as fh:
            fh.write("cmake_minimum_required(VERSION 3.25)\nproject(TheRock)\n")
        # A stock-ish therock_artifacts.cmake carrying the anchor line that the
        # artifact-deps property injection hooks onto.
        self.artifacts_cmake = os.path.join(
            self.therock, "cmake", "therock_artifacts.cmake")
        with open(self.artifacts_cmake, "w") as fh:
            fh.write(
                "function(therock_provide_artifact slice_name)\n"
                '  set(_target_name "artifact-${slice_name}")\n'
                '  add_dependencies(therock-artifacts "${_target_name}")\n'
                "endfunction()\n"
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_copies_module_and_appends_hook_once(self) -> None:
        backend = TheRockBackend()
        backend._inject_introspection(self.therock)
        # Module copied into cmake/.
        self.assertTrue(os.path.isfile(os.path.join(
            self.therock, "cmake", "therock_subproject_introspection.cmake")))
        text = open(self.cmakelists).read()
        self.assertIn("include(therock_subproject_introspection)", text)
        self.assertIn("therock_introspect_subprojects()", text)
        # Idempotent: a second injection does not duplicate the hook.
        backend._inject_introspection(self.therock)
        text2 = open(self.cmakelists).read()
        self.assertEqual(
            text2.count("therock_introspect_subprojects()"), 1
        )

    def test_records_artifact_subproject_deps_once(self) -> None:
        backend = TheRockBackend()
        backend._inject_introspection(self.therock)
        text = open(self.artifacts_cmake).read()
        # The property is recorded right after the anchor.
        self.assertIn("THEROCK_ARTIFACT_SUBPROJECT_DEPS", text)
        anchor = '  add_dependencies(therock-artifacts "${_target_name}")\n'
        self.assertIn(anchor + "  # >>> srock orchestrator", text)
        # Idempotent: a second injection does not duplicate the property.
        backend._inject_introspection(self.therock)
        text2 = open(self.artifacts_cmake).read()
        self.assertEqual(text2.count("THEROCK_ARTIFACT_SUBPROJECT_DEPS"), 1)

    def test_artifact_deps_injection_best_effort_without_file(self) -> None:
        # No therock_artifacts.cmake -> injection is skipped, not fatal.
        os.remove(self.artifacts_cmake)
        backend = TheRockBackend()
        backend._inject_introspection(self.therock)
        self.assertFalse(os.path.isfile(self.artifacts_cmake))

    def test_rejects_non_checkout(self) -> None:
        backend = TheRockBackend()
        with self.assertRaises(SystemExit):
            backend._inject_introspection(os.path.join(self.tmp, "nope"))


class ShardArtifactsHelperTest(unittest.TestCase):
    """The import-bootstrap store filter selects only the named producers."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="shard-store-")
        # A nested store as artifact_manager push would write it (run-id /
        # platform prefix), mixing archives and an exploded dir.
        self.store = os.path.join(self.tmp, "store")
        nested = os.path.join(self.store, "local", "linux")
        os.makedirs(nested)
        for fname in (
            "core-runtime_lib_generic.tar.zst",
            "amd-llvm_compiler_generic.tar.xz",
            "blas_lib_gfx94X.tar.zst",
        ):
            open(os.path.join(nested, fname), "w").close()
        os.makedirs(os.path.join(nested, "hip-clr_dev_generic"))  # exploded dir
        open(os.path.join(nested, "notanartifact.txt"), "w").close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_matches_by_artifact_name_archives_and_dirs(self) -> None:
        got = shard_artifacts.find_matching_artifacts(
            self.store, {"core-runtime", "hip-clr"}
        )
        names = sorted(os.path.basename(p) for p in got)
        self.assertEqual(
            names, ["core-runtime_lib_generic.tar.zst", "hip-clr_dev_generic"]
        )

    def test_ignores_unrelated_and_nonartifact_files(self) -> None:
        got = shard_artifacts.find_matching_artifacts(self.store, {"blas"})
        self.assertEqual(
            [os.path.basename(p) for p in got], ["blas_lib_gfx94X.tar.zst"]
        )
        # A name with no matching archive/dir yields nothing.
        self.assertEqual(
            shard_artifacts.find_matching_artifacts(self.store, {"nope"}), []
        )

    def test_export_local_copies_named_artifacts(self) -> None:
        # build/artifacts holds exploded {name}_{component}_{family} dirs; only
        # those whose name is in --names are copied into the (flat) store.
        build_dir = os.path.join(self.tmp, "build")
        artifacts = os.path.join(build_dir, "artifacts")
        os.makedirs(artifacts)
        for d in ("compiler_lib_generic", "compiler_dev_generic",
                  "base_lib_generic"):
            os.makedirs(os.path.join(artifacts, d))
            open(os.path.join(artifacts, d, "artifact_manifest.txt"), "w").close()
        open(os.path.join(artifacts, "scratch.fprint"), "w").close()
        out_store = os.path.join(self.tmp, "out-store")
        rc = shard_artifacts.main([
            "export-local", "--build-dir", build_dir,
            "--store", out_store, "--names", "compiler",
        ])
        self.assertEqual(rc, 0)
        self.assertEqual(
            sorted(os.listdir(out_store)),
            ["compiler_dev_generic", "compiler_lib_generic"],
        )
        # The copied dir keeps its contents; the round-trips through the store
        # matcher recover the artifact name.
        self.assertTrue(os.path.isfile(os.path.join(
            out_store, "compiler_lib_generic", "artifact_manifest.txt")))
        self.assertEqual(
            sorted(os.path.basename(p) for p in
                   shard_artifacts.find_matching_artifacts(out_store, {"compiler"})),
            ["compiler_dev_generic", "compiler_lib_generic"],
        )

    def test_export_local_no_match_fails(self) -> None:
        build_dir = os.path.join(self.tmp, "build2")
        os.makedirs(os.path.join(build_dir, "artifacts"))
        rc = shard_artifacts.main([
            "export-local", "--build-dir", build_dir,
            "--store", os.path.join(self.tmp, "s2"), "--names", "nope",
        ])
        self.assertEqual(rc, 1)


@unittest.skipUnless(
    os.path.isdir(THEROCK_SRC)
    and os.path.isfile(os.path.join(THEROCK_SRC, "BUILD_TOPOLOGY.toml")),
    f"no TheRock checkout at {THEROCK_SRC}",
)
class TopologyShardTest(unittest.TestCase):
    """Exercises the real build_topology adapter and group-based sharding."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="therock-topo-")
        self.repos = os.path.join(self.tmp, "repos")
        self.therock = os.path.join(self.repos, "TheRock")
        self.build = os.path.join(self.therock, "build")
        os.makedirs(self.build)
        os.makedirs(os.path.join(self.therock, "build_tools", "_therock_utils"))
        # Copy just the two files topology.load_build_topology needs.
        shutil.copy(
            os.path.join(THEROCK_SRC, "BUILD_TOPOLOGY.toml"),
            os.path.join(self.therock, "BUILD_TOPOLOGY.toml"),
        )
        shutil.copy(
            os.path.join(THEROCK_SRC, "build_tools", "_therock_utils",
                         "build_topology.py"),
            os.path.join(self.therock, "build_tools", "_therock_utils",
                         "build_topology.py"),
        )
        with open(os.path.join(self.build, "subproject_map.json"), "w") as fh:
            json.dump(FIXTURE, fh)
        # artifact -> composing subprojects, joined with the real topology's
        # artifact->group relation to map the FIXTURE subprojects onto groups:
        #   amd-llvm  -> artifact 'amd-llvm' (group 'compiler')
        #   rocm-cmake-> artifact 'base'     (group 'base')
        #   hipBLAS   -> artifact 'blas'     (group 'math-libs')
        with open(os.path.join(self.build, "artifact_map.json"), "w") as fh:
            json.dump(ARTIFACT_FIXTURE, fh)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_adapter_loads_and_ranks(self) -> None:
        topo = topology.load_build_topology(self.therock)
        self.assertIsNotNone(topo)
        ranks = topology.submodule_stage_rank(topo)
        self.assertTrue(ranks, "expected a non-empty submodule->stage rank map")
        # rocm-libraries is a known submodule in the topology.
        self.assertIn("rocm-libraries", ranks)

    def test_subproject_stage_maps_submodules_to_stages(self) -> None:
        topo = topology.load_build_topology(self.therock)
        sub_stage = topology.subproject_stage(topo)
        stages = set(topology.stage_names(topo))
        self.assertIn("rocm-libraries", sub_stage)
        # Every mapped value is a real build stage.
        self.assertTrue(set(sub_stage.values()) <= stages)

    def test_subproject_group_map(self) -> None:
        # artifact_map.json + topology artifact->group should map the FIXTURE
        # subprojects onto their groups.
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        backend.load_config(args)
        topo = topology.load_build_topology(self.therock)
        sub_group = backend._subproject_group_map(topo)
        self.assertEqual(sub_group.get("amd-llvm"), {"compiler"})
        self.assertEqual(sub_group.get("rocm-cmake"), {"base"})
        self.assertEqual(sub_group.get("hipBLAS"), {"math-libs"})

    def test_list_shards_rows(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list-shards")
        backend.load_config(args)
        rows = backend.list_shards(backend.build_child_env(args))
        self.assertIsNotNone(rows)
        self.assertTrue(rows)
        topo = topology.load_build_topology(self.therock)
        # Shards are the artifact groups, in dependency (build) order.
        self.assertEqual([r["name"] for r in rows], topology.group_names(topo))
        for r in rows:
            self.assertIn("name", r)
            self.assertIsInstance(r["produced"], int)
            self.assertIsInstance(r["inbound"], int)
            self.assertIsInstance(r["subprojects"], list)
            self.assertIsInstance(r["configured"], int)
            self.assertIsInstance(r["source_sets"], list)
            self.assertIsInstance(r["depends_on"], list)
        by_name = {r["name"]: r for r in rows}
        # The FIXTURE subprojects appear under their groups (config-derived).
        self.assertIn("amd-llvm", by_name["compiler"]["subprojects"])
        self.assertIn("rocm-cmake", by_name["base"]["subprojects"])
        self.assertIn("hipBLAS", by_name["math-libs"]["subprojects"])
        # The first group in build order depends on nothing upstream.
        self.assertEqual(rows[0]["depends_on"], [])
        # depends_on matches the topology's group dependencies.
        for r in rows:
            self.assertEqual(
                r["depends_on"], topology.group_dependencies(topo, r["name"])
            )

    def test_shard_tasks_import_build_export_pipeline(self) -> None:
        backend = TheRockBackend()
        topo = topology.load_build_topology(self.therock)
        # Build 'compiler' (amd-llvm), importing its dependency group(s).
        build_group = "compiler"
        import_group = topology.group_dependencies(topo, build_group)[0]
        args = make_args(
            self.therock, self.repos,
            "--import-shard", import_group,
            "--build-shard", build_group,
            "--export-shards",
            "list",
        )
        cfg = backend.load_config(args)
        env = backend.build_child_env(args)
        backend.discover_env(env)
        components = core.resolve_components(cfg, args.add, args.remove)
        tasks = core.elaborate_tasks(backend, cfg, components, env, [], {})
        pipeline = backend.shard_tasks(
            tasks, [import_group], [build_group], [build_group], env, args,
        )
        self.assertIsNotNone(pipeline)
        names = [t.name for t in pipeline]
        # Fetch precedes import precedes export; the build group's artifact
        # target is produced via the native artifact-group-<g> ninja target.
        self.assertIn("therock/fetch-sources", names)
        self.assertIn(f"therock/import-{import_group}", names)
        self.assertIn(f"therock/export-{build_group}", names)
        self.assertIn(f"therock/artifact-group-{build_group}", names)
        self.assertLess(
            names.index("therock/fetch-sources"),
            names.index(f"therock/import-{import_group}"),
        )
        self.assertLess(
            names.index(f"therock/import-{import_group}"),
            names.index(f"therock/export-{build_group}"),
        )
        # fetch uses --source-sets (group fetch), not --stage.
        fetch = next(t for t in pipeline if t.name == "therock/fetch-sources")
        self.assertIn("--source-sets", fetch.payload["argv"])
        self.assertNotIn("--stage", fetch.payload["argv"])
        # The import task argv carries the producer group's artifact names.
        imp = next(t for t in pipeline
                   if t.name == f"therock/import-{import_group}")
        argv = imp.payload["argv"]
        self.assertIn("import-bootstrap", argv)
        self.assertIn("--names", argv)
        produced = topology.group_produced_names(topo, import_group)
        names_arg = argv[argv.index("--names") + 1]
        self.assertEqual(set(names_arg.split(",")), produced)
        # The export task is the helper's export-local for the build group.
        exp = next(t for t in pipeline
                   if t.name == f"therock/export-{build_group}")
        self.assertIn("export-local", exp.payload["argv"])
        exp_names = exp.payload["argv"][exp.payload["argv"].index("--names") + 1]
        self.assertEqual(
            set(exp_names.split(",")),
            topology.group_produced_names(topo, build_group),
        )

    def test_unknown_shard_fails(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "--build-shard", "nope", "list")
        cfg = backend.load_config(args)
        env = backend.build_child_env(args)
        with self.assertRaises(SystemExit):
            backend.shard_tasks([], [], ["nope"], [], env, args)

    def test_shard_pin_follows_imports_before_build(self) -> None:
        # When the build group maps to real subprojects, a 'shard-pin' task
        # (buildctl enable + reconfigure) is inserted after the imports and
        # before the build subproject tasks, so the just-imported groups are
        # pinned prebuilt for the build.
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        cfg = backend.load_config(args)
        env = backend.build_child_env(args)
        backend.discover_env(env)
        topo = topology.load_build_topology(self.therock)
        # 'compiler' (amd-llvm) is a build group; import one of its deps.
        build_group = "compiler"
        import_group = topology.group_dependencies(topo, build_group)[0]
        components = core.resolve_components(cfg, args.add, args.remove)
        tasks = core.elaborate_tasks(backend, cfg, components, env, [], {})
        pipeline = backend.shard_tasks(
            tasks, [import_group], [build_group], [], env, args,
        )
        names = [t.name for t in pipeline]
        self.assertIn("therock/shard-pin", names)
        # pin sits after the import and before the first build subproject task.
        pin_i = names.index("therock/shard-pin")
        self.assertLess(names.index(f"therock/import-{import_group}"), pin_i)
        sub_group = backend._subproject_group_map(topo)
        build_comps = [c for c, gs in sub_group.items() if build_group in gs]
        first_build = min(
            i for i, t in enumerate(pipeline) if t.comp in build_comps
        )
        self.assertLess(pin_i, first_build)
        # The pin task is a buildctl enable scoped to the build comps' stages.
        pin = pipeline[pin_i]
        self.assertEqual(pin.payload["argv"][1:3],
                         [os.path.join(self.therock, "build_tools",
                                       "buildctl.py"), "enable"])

    def test_rest_build_shards_excludes_imports(self) -> None:
        # The FIXTURE configures groups {base, compiler, math-libs}; importing
        # 'base' leaves the other two as the fill set, in build order.
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        backend.load_config(args)
        env = backend.build_child_env(args)
        rest = backend.rest_build_shards(["base"], env)
        self.assertEqual(set(rest), {"compiler", "math-libs"})
        self.assertNotIn("base", rest)
        # Build order: a subset of group_names, in the same relative order.
        order = topology.group_names(topology.load_build_topology(self.therock))
        self.assertEqual(rest, [g for g in order if g in set(rest)])

    def test_rest_build_shards_without_imports_is_all_configured(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        backend.load_config(args)
        env = backend.build_child_env(args)
        rest = backend.rest_build_shards([], env)
        self.assertEqual(set(rest), {"base", "compiler", "math-libs"})

    def test_fill_implies_deploy_and_excludes_imported_group(self) -> None:
        # `--import-shard base -f list` previews a pipeline that builds the fill
        # groups (compiler/math-libs), deploys (therock/install present), and
        # builds nothing from the imported 'base' group (rocm-cmake).
        backend = TheRockBackend()
        args = make_args(
            self.therock, self.repos, "--import-shard", "base", "-f", "list",
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = core.run(args, backend)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("therock/install", out)         # deploy implied
        self.assertIn("therock/import-base", out)      # import preserved
        self.assertIn("amd-llvm/", out)                # compiler group built
        self.assertIn("hipBLAS/", out)                 # math-libs group built
        # rocm-cmake (the imported 'base' group) is not built.
        self.assertNotIn("rocm-cmake/", out)
        # The computed-fill note is printed.
        self.assertIn("--fill: building", out)

    def test_fill_conflicts_with_build_shard(self) -> None:
        backend = TheRockBackend()
        args = make_args(
            self.therock, self.repos,
            "-f", "--build-shard", "compiler", "list",
        )
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):
                core.run(args, backend)

    def test_fill_with_no_configured_shards_fails(self) -> None:
        # Without artifact_map.json the subproject->group map is empty, so no
        # group is "configured" and --fill has nothing to build.
        os.remove(os.path.join(self.build, "artifact_map.json"))
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "-f", "list")
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):
                core.run(args, backend)


class TtyStream(io.StringIO):
    """A StringIO that claims to be a TTY, for exercising StatusLine output."""

    def isatty(self) -> bool:  # noqa: D401 - simple override
        return True


class StatusLineTests(unittest.TestCase):
    def test_noop_on_non_tty(self) -> None:
        # A plain StringIO is not a TTY: show()/clear() must write nothing so
        # piped/CI output stays free of carriage returns and ANSI escapes.
        buf = io.StringIO()
        status = core.StatusLine(buf)
        self.assertFalse(status.enabled)
        status.show("  Building amd-llvm [1/3] 33%")
        status.clear()
        self.assertEqual(buf.getvalue(), "")

    def test_show_and_clear_on_tty(self) -> None:
        buf = TtyStream()
        status = core.StatusLine(buf)
        self.assertTrue(status.enabled)
        status.show("  Building amd-llvm [1/3] 33%")
        out = buf.getvalue()
        # Mid-grey ANSI colour, the text, a reset, and an erase-line sequence.
        self.assertIn("\033[38;5;244m", out)
        self.assertIn("Building amd-llvm [1/3] 33%", out)
        self.assertIn("\033[0m", out)
        self.assertIn("\033[2K", out)
        self.assertTrue(status.active)
        status.clear()
        self.assertFalse(status.active)
        # clear() emits another erase sequence after the shown line.
        self.assertEqual(buf.getvalue().count("\033[2K"), 2)

    def test_clear_without_show_is_noop(self) -> None:
        buf = TtyStream()
        status = core.StatusLine(buf)
        status.clear()
        self.assertEqual(buf.getvalue(), "")


class TailLastLineTests(unittest.TestCase):
    def test_returns_last_non_empty_line(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.log")
            with open(p, "w", encoding="utf-8") as f:
                f.write("first\nsecond\nthird\n\n  \n")
            self.assertEqual(core.tail_last_line(p), "third")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(core.tail_last_line("/no/such/file.log"), "")

    def test_only_reads_trailing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.log")
            with open(p, "w", encoding="utf-8") as f:
                f.write("A" * 100000 + "\n")
                f.write("tail-line\n")
            self.assertEqual(core.tail_last_line(p, maxbytes=64), "tail-line")


class FmtElapsedTests(unittest.TestCase):
    def test_formats(self) -> None:
        self.assertEqual(core._fmt_elapsed(0), "0s")
        self.assertEqual(core._fmt_elapsed(45), "45s")
        self.assertEqual(core._fmt_elapsed(123), "2m03s")
        self.assertEqual(core._fmt_elapsed(3600 + 7 * 60), "1h07m")


class RunWithProgressTests(unittest.TestCase):
    def test_live_tail_updates_status(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "build.log")
            buf = TtyStream()
            status = core.StatusLine(buf)
            # A command that writes two lines with a pause so the poll loop has
            # a chance to observe the growing log and update the status line.
            script = (
                "import sys, time\n"
                "print('compiling foo.cpp', flush=True)\n"
                "time.sleep(0.25)\n"
                "print('linking libfoo', flush=True)\n"
                "time.sleep(0.25)\n"
            )
            with open(log_path, "w", encoding="utf-8") as log:
                rc = core.run_with_progress(
                    [sys.executable, "-c", script], dict(os.environ),
                    log, log_path, status, poll=0.05,
                )
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            # The status line should have surfaced at least one log line.
            self.assertTrue(
                "compiling foo.cpp" in out or "linking libfoo" in out,
                msg=f"status output did not include a log line: {out!r}",
            )
            # An elapsed clock is shown, and the line is cleared on completion.
            self.assertRegex(out, r"\[\d+s\]")
            self.assertFalse(status.active)

    def test_clock_ticks_while_log_is_quiet(self) -> None:
        # A process that writes nothing should still produce a ticking elapsed
        # clock (so a quiet-but-working step never looks frozen/hung).
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "build.log")
            buf = TtyStream()
            status = core.StatusLine(buf)
            with open(log_path, "w", encoding="utf-8") as log:
                rc = core.run_with_progress(
                    [sys.executable, "-c", "import time; time.sleep(1.2)"],
                    dict(os.environ), log, log_path, status, poll=0.05,
                )
            self.assertEqual(rc, 0)
            # At least one elapsed-clock draw happened despite an empty log.
            self.assertRegex(buf.getvalue(), r"\[\d+s\]")

    def test_returns_nonzero_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "build.log")
            status = core.StatusLine(io.StringIO())  # non-TTY: no draws
            with open(log_path, "w", encoding="utf-8") as log:
                rc = core.run_with_progress(
                    [sys.executable, "-c", "import sys; sys.exit(3)"],
                    dict(os.environ), log, log_path, status, poll=0.05,
                )
            self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
