#!/usr/bin/env python3
"""Tests for the TheRock orchestrator backend.

Synthetic-fixture tests (no real TheRock build) covering the full flow:
introspection JSON -> Config graph -> elaborated tasks -> ninja commands ->
sharding. Run directly or via unittest:

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

from orchestrator import core, topology  # noqa: E402
from orchestrator.therock_backend import (  # noqa: E402
    DEFAULT_CHILD_PATH, DEFAULT_CONFIG, TheRockBackend,
)

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

    def test_count_based_shard_partition(self) -> None:
        # 9 tasks (3 components x 3 actions) into 3 shards -> 3 each.
        self.assertEqual(core.partition_shard(9, 1, 3), [0, 1, 2])
        self.assertEqual(core.partition_shard(9, 2, 3), [3, 4, 5])
        self.assertEqual(core.partition_shard(9, 3, 3), [6, 7, 8])
        union: list[int] = []
        for k in (1, 2, 3):
            union += core.partition_shard(9, k, 3)
        self.assertEqual(union, list(range(9)))

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

    def test_explicit_config_warns_and_is_ignored(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf):
            env = TheRockBackend().build_child_env(
                make_args(self.therock, self.repos, "-c", "all", "list")
            )
        self.assertIn("not supported", buf.getvalue())
        # -c is ignored; with no --add toggle the config falls back to minimal.
        self.assertEqual(env["SROCK_CONFIG"], DEFAULT_CONFIG)

    def test_stage_relpath_maps_build_to_stage(self) -> None:
        backend, _, _ = self._load()
        self.assertEqual(
            backend._stage_relpath("amd-llvm"), "compiler/amd-llvm/stage"
        )
        self.assertEqual(
            backend._stage_relpath("hipBLAS"), "math-libs/BLAS/hipBLAS/stage"
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

    def test_rejects_non_checkout(self) -> None:
        backend = TheRockBackend()
        with self.assertRaises(SystemExit):
            backend._inject_introspection(os.path.join(self.tmp, "nope"))


class PartitionAlignedTest(unittest.TestCase):
    def test_cuts_snap_to_run_boundaries(self) -> None:
        # Runs of sizes [3,3,3,3] = 12 tasks; 2 shards should cut at task 6.
        shard1 = core.partition_shard_aligned(12, [3, 3, 3, 3], 1, 2)
        shard2 = core.partition_shard_aligned(12, [3, 3, 3, 3], 2, 2)
        self.assertEqual(shard1, list(range(0, 6)))
        self.assertEqual(shard2, list(range(6, 12)))

    def test_uneven_runs_prefer_boundaries(self) -> None:
        # Runs [5,1,6] = 12; ideal cut at 6 snaps to boundary 6 (5+1).
        shard1 = core.partition_shard_aligned(12, [5, 1, 6], 1, 2)
        self.assertEqual(shard1, list(range(0, 6)))

    def test_falls_back_when_runs_invalid(self) -> None:
        # Run lengths that don't sum to num_tasks -> count-based partition.
        self.assertEqual(
            core.partition_shard_aligned(12, [3, 3], 1, 3),
            core.partition_shard(12, 1, 3),
        )


@unittest.skipUnless(
    os.path.isdir(THEROCK_SRC)
    and os.path.isfile(os.path.join(THEROCK_SRC, "BUILD_TOPOLOGY.toml")),
    f"no TheRock checkout at {THEROCK_SRC}",
)
class TopologyShardTest(unittest.TestCase):
    """Exercises the real build_topology adapter and stage-aligned sharding."""

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

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_adapter_loads_and_ranks(self) -> None:
        topo = topology.load_build_topology(self.therock)
        self.assertIsNotNone(topo)
        ranks = topology.submodule_stage_rank(topo)
        self.assertTrue(ranks, "expected a non-empty submodule->stage rank map")
        # rocm-libraries is a known submodule in the topology.
        self.assertIn("rocm-libraries", ranks)

    def test_shard_run_lengths_partition_full_task_list(self) -> None:
        backend = TheRockBackend()
        args = make_args(self.therock, self.repos, "list")
        cfg = backend.load_config(args)
        env = backend.build_child_env(args)
        components = core.resolve_components(cfg, args.add, args.remove)
        tasks = core.elaborate_tasks(backend, cfg, components, env, [], {})
        runs = backend.shard_run_lengths(tasks, env)
        self.assertIsNotNone(runs)
        # Runs are contiguous and cover exactly the task list.
        self.assertEqual(sum(runs), len(tasks))
        self.assertTrue(all(r > 0 for r in runs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
