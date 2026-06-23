#!/usr/bin/env python3
"""Tests for the shared-source layout (symlink + migrate) helpers.

    python3 bin/orchestrator/tests/test_source_layout.py
    python3 -m unittest discover -s bin/orchestrator/tests
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

BIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

from orchestrator import source_layout  # noqa: E402
from orchestrator.aomp_backend import AompBackend, parse_cudf, DEFAULT_CONFIG  # noqa: E402


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True,
    )


def _init_repo(path, branch="amd-staging"):
    os.makedirs(path, exist_ok=True)
    _git(path, "init", "-q", "-b", branch)
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    with open(os.path.join(path, "README"), "w") as fh:
        fh.write("x\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")


# A shared component to exercise (llvm-project: aomp_dir llvm-project ->
# therock compiler/amd-llvm, module name llvm-project, branch amd-staging).
LLVM = next(c for c in source_layout.SHARED_COMPONENTS if c.aomp_dir == "llvm-project")


class SymlinkPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.therock = os.path.join(self.tmp, "TheRock")
        self.aomp = os.path.join(self.tmp, "aomp_repos")
        os.makedirs(self.aomp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _populate_target(self, comp):
        tgt = os.path.join(self.therock, comp.therock_path)
        os.makedirs(tgt)
        with open(os.path.join(tgt, "f"), "w") as fh:
            fh.write("x")
        return tgt

    def test_link_when_target_present(self):
        self._populate_target(LLVM)
        plan = source_layout.symlink_plan(self.aomp, self.therock)
        act = next(a for a in plan if a.comp is LLVM)
        self.assertEqual(act.status, "link")
        source_layout.apply_symlinks(plan)
        link = os.path.join(self.aomp, LLVM.aomp_dir)
        self.assertTrue(os.path.islink(link))
        self.assertTrue(os.path.exists(os.path.join(link, "f")))

    def test_missing_target(self):
        plan = source_layout.symlink_plan(self.aomp, self.therock)
        act = next(a for a in plan if a.comp is LLVM)
        self.assertEqual(act.status, "missing-target")

    def test_skip_real_dir(self):
        self._populate_target(LLVM)
        real = os.path.join(self.aomp, LLVM.aomp_dir)
        os.makedirs(real)
        plan = source_layout.symlink_plan(self.aomp, self.therock)
        act = next(a for a in plan if a.comp is LLVM)
        self.assertEqual(act.status, "skip-existing")
        source_layout.apply_symlinks(plan)
        self.assertFalse(os.path.islink(real))

    def test_already_linked_is_idempotent(self):
        tgt = self._populate_target(LLVM)
        link = os.path.join(self.aomp, LLVM.aomp_dir)
        os.symlink(tgt, link)
        plan = source_layout.symlink_plan(self.aomp, self.therock)
        act = next(a for a in plan if a.comp is LLVM)
        self.assertEqual(act.status, "already-linked")


class MigratePlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.therock = os.path.join(self.tmp, "TheRock")
        self.aomp = os.path.join(self.tmp, "aomp_repos")
        _init_repo(self.therock, branch="main")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ready_and_branch_warning(self):
        # On a branch other than TheRock's expected -> ready + warning.
        _init_repo(os.path.join(self.aomp, LLVM.aomp_dir), branch="my-feature")
        plan = source_layout.migrate_plan(self.therock, self.aomp)
        act = next(a for a in plan if a.comp is LLVM)
        self.assertEqual(act.status, "ready")
        self.assertTrue(any("my-feature" in w for w in act.warnings))

    def test_missing_and_not_git(self):
        # ROCgdb absent -> missing-source; a non-git dir for rocm-cmake.
        rcm = next(c for c in source_layout.SHARED_COMPONENTS
                   if c.aomp_dir == "rocm-cmake")
        os.makedirs(os.path.join(self.aomp, rcm.aomp_dir))
        plan = source_layout.migrate_plan(self.therock, self.aomp)
        rcm_act = next(a for a in plan if a.comp is rcm)
        self.assertEqual(rcm_act.status, "not-git")
        llvm_act = next(a for a in plan if a.comp is LLVM)
        self.assertEqual(llvm_act.status, "missing-source")

    def test_dest_occupied(self):
        _init_repo(os.path.join(self.aomp, LLVM.aomp_dir))
        dest = os.path.join(self.therock, LLVM.therock_path)
        os.makedirs(dest)
        with open(os.path.join(dest, "f"), "w") as fh:
            fh.write("x")
        plan = source_layout.migrate_plan(self.therock, self.aomp)
        act = next(a for a in plan if a.comp is LLVM)
        self.assertEqual(act.status, "dest-occupied")

    def test_apply_migration_converts_to_submodule_gitdir(self):
        src = os.path.join(self.aomp, LLVM.aomp_dir)
        _init_repo(src)
        plan = source_layout.migrate_plan(self.therock, self.aomp)
        source_layout.apply_migration(self.therock, plan)

        dest = os.path.join(self.therock, LLVM.therock_path)
        self.assertFalse(os.path.exists(src), "source should have moved")
        self.assertTrue(os.path.exists(os.path.join(dest, "README")))
        # .git is now a gitlink file pointing into .git/modules/<name>.
        gitlink = os.path.join(dest, ".git")
        self.assertTrue(os.path.isfile(gitlink))
        with open(gitlink) as fh:
            content = fh.read()
        self.assertIn(f"modules/{LLVM.submodule_name}", content)
        module_dir = os.path.join(self.therock, ".git", "modules",
                                  LLVM.submodule_name)
        self.assertTrue(os.path.isdir(module_dir))
        # The moved tree is still a valid working tree.
        self.assertEqual(
            source_layout._git_out(dest, "rev-parse", "--is-inside-work-tree"),
            "true",
        )

    def test_apply_into_empty_placeholder_slot(self):
        # TheRock checkouts have the submodule path as an empty placeholder dir
        # (unfetched submodule). The repo must land *at* dest, not nested inside.
        src = os.path.join(self.aomp, LLVM.aomp_dir)
        _init_repo(src)
        dest = os.path.join(self.therock, LLVM.therock_path)
        os.makedirs(dest)  # empty placeholder
        plan = source_layout.migrate_plan(self.therock, self.aomp)
        self.assertEqual(next(a for a in plan if a.comp is LLVM).status, "ready")
        source_layout.apply_migration(self.therock, plan)
        self.assertTrue(os.path.isfile(os.path.join(dest, "README")))
        self.assertTrue(os.path.isfile(os.path.join(dest, ".git")))
        self.assertFalse(os.path.exists(os.path.join(dest, "llvm-project")),
                         "repo must not be nested inside the placeholder")

    def test_dry_run_moves_nothing(self):
        src = os.path.join(self.aomp, LLVM.aomp_dir)
        _init_repo(src)
        plan = source_layout.migrate_plan(self.therock, self.aomp)
        source_layout.apply_migration(self.therock, plan, dry_run=True)
        self.assertTrue(os.path.exists(src))
        self.assertFalse(os.path.exists(os.path.join(self.therock,
                                                     LLVM.therock_path)))


class RocmlibsDetectionTests(unittest.TestCase):
    def test_detects_rocmlibs_components(self):
        cfg = parse_cudf(DEFAULT_CONFIG)
        backend = AompBackend()
        backend._cfg = cfg
        rocmlibs = [n for n, p in cfg.packages.items() if p.xdir == "rocmlibs"]
        self.assertTrue(rocmlibs, "fixture config should have rocmlibs comps")
        self.assertTrue(backend._needs_rocmlibs(rocmlibs[:1]))
        non = [n for n, p in cfg.packages.items() if p.xdir != "rocmlibs"]
        self.assertFalse(backend._needs_rocmlibs(non))


if __name__ == "__main__":
    unittest.main(verbosity=2)
