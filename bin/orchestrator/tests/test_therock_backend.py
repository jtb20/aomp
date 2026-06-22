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
