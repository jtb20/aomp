#!/usr/bin/env python3
"""Tests for the AOMP orchestrator backend's listing selectors.

Synthetic-fixture tests (no real AOMP build) covering `list-features` (CUDF
component groups + components marked by the effective build set) and
`list-variants` (components advertising non-default build configs), plus the
core.py printers. Run directly or via unittest:

    python3 bin/orchestrator/tests/test_aomp_backend.py
    python3 -m unittest discover -s bin/orchestrator/tests
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

# Make `import orchestrator` resolve (bin/ is two levels up from this file).
BIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

from orchestrator import core  # noqa: E402
from orchestrator.aomp_backend import AompBackend  # noqa: E402
from orchestrator.model import Config, Package  # noqa: E402


def _cfg() -> Config:
    """A small graph: prereq <- llvm <- {flang, hip}; dbgapi <- gdb.

    request builds prereq, llvm, flang, hip (the default set); dbgapi/gdb are
    off by default. Feature groups: 'compiler' (llvm+flang+hip, all on by
    default) and 'debug' (dbgapi+gdb, off by default)."""
    cfg = Config()
    names = ["prereq", "llvm", "flang", "hip", "dbgapi", "gdb"]
    deps = {
        "prereq": [],
        "llvm": ["prereq"],
        "flang": ["llvm"],
        "hip": ["llvm"],
        "dbgapi": ["prereq"],
        "gdb": ["dbgapi"],
    }
    for i, n in enumerate(names):
        cfg.packages[n] = Package(name=n, depends=deps[n], order=i)
    cfg.features = {
        "compiler": ["llvm", "flang", "hip"],
        "debug": ["dbgapi", "gdb"],
    }
    cfg.request = ["prereq", "llvm", "flang", "hip"]
    return cfg


def _backend(adds=None, removes=None) -> AompBackend:
    be = AompBackend()
    be._cfg = _cfg()
    be._args = argparse.Namespace(add=adds or [], remove=removes or [])
    return be


class ListFeaturesTests(unittest.TestCase):
    def _rows(self, **kw):
        return _backend(**kw).list_features({})

    def test_groups_and_components_present(self) -> None:
        rows = self._rows()
        groups = {r["name"]: r for r in rows if r["kind"] == "group"}
        comps = {r["name"]: r for r in rows if r["kind"] == "component"}
        self.assertEqual(set(groups), {"compiler", "debug"})
        self.assertEqual(set(comps), {"prereq", "llvm", "flang", "hip",
                                      "dbgapi", "gdb"})

    def test_groups_before_components_and_cudf_order(self) -> None:
        rows = self._rows()
        kinds = [r["kind"] for r in rows]
        self.assertEqual(kinds, ["group"] * 2 + ["component"] * 6)
        comp_names = [r["name"] for r in rows if r["kind"] == "component"]
        self.assertEqual(comp_names,
                         ["prereq", "llvm", "flang", "hip", "dbgapi", "gdb"])

    def test_enabled_reflects_default_request(self) -> None:
        rows = self._rows()
        by = {(r["kind"], r["name"]): r for r in rows}
        self.assertTrue(by[("group", "compiler")]["enabled"])
        self.assertFalse(by[("group", "debug")]["enabled"])
        self.assertTrue(by[("component", "hip")]["enabled"])
        self.assertFalse(by[("component", "gdb")]["enabled"])

    def test_add_flips_a_group_on(self) -> None:
        rows = self._rows(adds=["debug"])
        by = {(r["kind"], r["name"]): r for r in rows}
        self.assertTrue(by[("group", "debug")]["enabled"])
        self.assertTrue(by[("component", "gdb")]["enabled"])
        self.assertTrue(by[("component", "dbgapi")]["enabled"])

    def test_remove_cascades_to_dependents(self) -> None:
        # Removing llvm drops flang and hip (they depend on it).
        rows = self._rows(removes=["llvm"])
        by = {(r["kind"], r["name"]): r for r in rows}
        self.assertFalse(by[("component", "llvm")]["enabled"])
        self.assertFalse(by[("component", "flang")]["enabled"])
        self.assertFalse(by[("component", "hip")]["enabled"])
        self.assertFalse(by[("group", "compiler")]["enabled"])
        self.assertTrue(by[("component", "prereq")]["enabled"])

    def test_partial_group_when_one_member_added(self) -> None:
        # Add only dbgapi: the 'debug' group is partially on (1/2).
        rows = self._rows(adds=["dbgapi"])
        debug = next(r for r in rows
                     if r["kind"] == "group" and r["name"] == "debug")
        self.assertFalse(debug["enabled"])
        self.assertEqual((debug["present"], debug["total"]), (1, 2))


class ListVariantsTests(unittest.TestCase):
    def _backend_with_configs(self, mapping, raise_on=()):
        be = _backend()

        def fake(comp, env):
            if comp in raise_on:
                raise SystemExit(1)
            return mapping.get(comp)

        be._list_configs_optional = fake  # type: ignore[assignment]
        return be

    def test_only_non_default_variants_kept_in_order(self) -> None:
        be = self._backend_with_configs({
            "prereq": ["default"],
            "llvm": ["default", "asan"],
            "flang": None,
            "hip": ["default", "asan", "perf"],
            "dbgapi": ["default"],
            "gdb": ["asan"],
        })
        rows = be.list_variants({})
        self.assertEqual([r["component"] for r in rows], ["llvm", "hip", "gdb"])
        self.assertEqual(rows[0]["variants"], ["default", "asan"])

    def test_component_with_only_default_is_skipped(self) -> None:
        be = self._backend_with_configs({n: ["default"]
                                         for n in _cfg().packages})
        self.assertEqual(be.list_variants({}), [])

    def test_unavailable_script_skipped(self) -> None:
        # _list_configs_optional returning None (missing/failed) is skipped.
        be = self._backend_with_configs({"llvm": ["default", "asan"]})
        rows = be.list_variants({})
        self.assertEqual([r["component"] for r in rows], ["llvm"])


class PrinterTests(unittest.TestCase):
    def test_print_feature_rows_aomp_sections(self) -> None:
        rows = _backend().list_features({})
        buf = io.StringIO()
        with redirect_stdout(buf):
            core._print_feature_rows(rows)
        out = buf.getvalue()
        self.assertIn("Component groups", out)
        self.assertIn("Components", out)
        self.assertIn("compiler", out)
        # group members are shown
        self.assertIn("llvm, flang, hip", out)
        # legend present
        self.assertIn("removable", out)
        self.assertIn("addable", out)

    def test_print_feature_rows_partial_note(self) -> None:
        rows = _backend(adds=["dbgapi"]).list_features({})
        buf = io.StringIO()
        with redirect_stdout(buf):
            core._print_feature_rows(rows)
        self.assertIn("partial: 1/2", buf.getvalue())

    def test_print_feature_rows_therock_flat(self) -> None:
        rows = [
            {"name": "hipblaslt", "enabled": True,
             "requires": ["hip"], "description": "BLAS"},
            {"name": "rocgdb", "enabled": False},
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            core._print_feature_rows(rows)
        out = buf.getvalue()
        self.assertIn("hipblaslt", out)
        self.assertIn("requires: hip", out)
        self.assertIn("BLAS", out)
        self.assertNotIn("Component groups", out)

    def test_print_variant_catalog(self) -> None:
        rows = [
            {"component": "llvm", "variants": ["default", "asan"]},
            {"component": "hip", "variants": ["default", "asan", "perf"]},
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            core.print_variant_catalog(rows)
        out = buf.getvalue()
        self.assertIn("--variant", out)
        self.assertIn("AOMP_BUILD_SANITIZER", out)
        self.assertIn("llvm", out)
        self.assertIn("default, asan, perf", out)


if __name__ == "__main__":
    unittest.main()
