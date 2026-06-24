#!/usr/bin/env python3
"""Tests for the source-config loader (orchestrator.source_config).

Covers parsing/validation of config TOML files, the default config, the unknown
-name error, the bundled amd-staging/develop configs, and the catalog. Run
directly or via unittest:

    python3 bin/orchestrator/tests/test_source_config.py
    python3 -m unittest discover -s bin/orchestrator/tests
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

# Make `import orchestrator` resolve (bin/ is two levels up from this file).
BIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

from orchestrator import source_config as sc  # noqa: E402


class BundledConfigsTest(unittest.TestCase):
    """The configs shipped under srock-bin/source-configs."""

    def test_default_is_amd_staging(self) -> None:
        self.assertEqual(sc.DEFAULT_SOURCE_CONFIG, "amd-staging")

    def test_amd_staging_branches(self) -> None:
        cfg = sc.load("amd-staging")
        self.assertEqual(cfg.therock_branch, "compiler/amd-staging")
        self.assertEqual(cfg.compiler_branch, "amd-staging")
        # patch_tag defaults to the compiler branch.
        self.assertEqual(cfg.patch_tag, "amd-staging")

    def test_develop_is_native(self) -> None:
        cfg = sc.load("develop")
        self.assertEqual(cfg.therock_branch, "main")
        # The develop sentinel disables the srock compiler override.
        self.assertEqual(cfg.compiler_branch, "develop")

    def test_available_lists_bundled(self) -> None:
        names = sc.available()
        self.assertIn("amd-staging", names)
        self.assertIn("develop", names)

    def test_catalog_loads_all(self) -> None:
        names = {c.name for c in sc.catalog()}
        self.assertEqual(names, set(sc.available()))


class CustomDirTest(unittest.TestCase):
    """Parsing/validation against ad-hoc config files in a temp dir."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="srccfg-")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name: str, body: str) -> None:
        with open(os.path.join(self.tmp, f"{name}.toml"), "w") as fh:
            fh.write(body)

    def test_full_config_parsed(self) -> None:
        self._write("mix", (
            'description = "mixed"\n'
            'therock_branch = "release/rocm-rel-7.2"\n'
            'compiler_branch = "amd-staging"\n'
            'patch_tag = "custom"\n'
        ))
        cfg = sc.load("mix", config_dir=self.tmp)
        self.assertEqual(cfg.name, "mix")
        self.assertEqual(cfg.description, "mixed")
        self.assertEqual(cfg.therock_branch, "release/rocm-rel-7.2")
        self.assertEqual(cfg.compiler_branch, "amd-staging")
        # Explicit patch_tag overrides the compiler-branch default.
        self.assertEqual(cfg.patch_tag, "custom")

    def test_patch_tag_defaults_to_compiler_branch(self) -> None:
        self._write("p", (
            'therock_branch = "main"\n'
            'compiler_branch = "develop"\n'
        ))
        self.assertEqual(sc.load("p", config_dir=self.tmp).patch_tag, "develop")

    def test_unknown_name_raises(self) -> None:
        with self.assertRaises(sc.SourceConfigError):
            sc.load("nope", config_dir=self.tmp)

    def test_missing_required_branch_raises(self) -> None:
        self._write("bad", 'description = "no branches"\n')
        with self.assertRaises(sc.SourceConfigError):
            sc.load("bad", config_dir=self.tmp)

    def test_partial_branch_raises(self) -> None:
        self._write("half", 'therock_branch = "main"\n')
        with self.assertRaises(sc.SourceConfigError):
            sc.load("half", config_dir=self.tmp)

    def test_available_empty_for_missing_dir(self) -> None:
        self.assertEqual(sc.available(os.path.join(self.tmp, "nope")), [])

    def test_catalog_skips_malformed(self) -> None:
        self._write("good", (
            'therock_branch = "main"\n'
            'compiler_branch = "develop"\n'
        ))
        self._write("broken", 'therock_branch = "main"\n')  # missing compiler
        names = {c.name for c in sc.catalog(config_dir=self.tmp)}
        self.assertEqual(names, {"good"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
