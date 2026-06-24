#!/usr/bin/env python3
"""Helper invoked by the orchestrator's TheRock group-based shard tasks.

The orchestrator drives import/build/export of TheRock artifact *groups*
(shards) with TheRock's own tooling. Source fetch is a direct
``fetch_sources.py --source-sets`` (no wrapper needed), but *import* and *export*
need this shim, both keyed by an explicit set of producer artifact names (the
group's products, computed from BUILD_TOPOLOGY.toml at planning time) so this
script stays free of any TheRock import and is unit-testable on its own:

* ``import-bootstrap``: ``buildctl.py bootstrap`` imports everything in its
  ``--artifact-dir`` (or, with ``--stage``, a *consuming* stage's inbound set).
  The orchestrator's ``--import-shard`` instead names *producer* groups, so we
  must import exactly the artifacts those groups produce. This materializes a
  filtered view of the shared store -- symlinks to just the matching artifact
  archives/dirs -- and hands that to ``buildctl.py bootstrap``.
* ``export-local``: copies a group's built artifacts out of the build dir into
  the shared store (``artifact_manager.py push`` is stage-, not group-, scoped).

Artifact archives/dirs are named ``{name}_{component}_{target_family}`` (with a
``.tar.zst`` / ``.tar.xz`` suffix for archives); the leading ``{name}`` is the
topology artifact name, which is what ``--names`` matches against.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

# {name}_{component}_{target_family} with an optional archive suffix.
_ARCHIVE_RE = re.compile(r"^([^_]+)_([^_]+)_([^_]+)\.tar\.(?:zst|xz)$")
_DIR_RE = re.compile(r"^([^_]+)_([^_]+)_([^_]+)$")


def _artifact_name(entry_name: str, is_dir: bool) -> str | None:
    """The topology artifact name for a store entry, or None if not an artifact."""
    m = (_DIR_RE if is_dir else _ARCHIVE_RE).match(entry_name)
    return m.group(1) if m else None


def find_matching_artifacts(store: str, names: set[str]) -> list[str]:
    """Absolute paths of store entries whose artifact name is in ``names``.

    Walks ``store`` (a local artifact staging dir, possibly nested under a
    run-id/platform prefix). Matches both compressed archives and exploded
    artifact directories. Returns a sorted, de-duplicated list of *absolute*
    paths (so symlinks built from them resolve regardless of the symlink dir).
    """
    found: dict[str, str] = {}
    for root, dirs, files in os.walk(os.path.abspath(store)):
        for fname in files:
            name = _artifact_name(fname, is_dir=False)
            if name in names:
                found[fname] = os.path.join(root, fname)
        # Exploded artifact dirs are leaves; match them but don't descend into
        # their internal layout looking for more "artifacts".
        keep: list[str] = []
        for dname in dirs:
            name = _artifact_name(dname, is_dir=True)
            if name in names:
                found.setdefault(dname, os.path.join(root, dname))
            else:
                keep.append(dname)
        dirs[:] = keep
    return [found[k] for k in sorted(found)]


def import_bootstrap(args: argparse.Namespace) -> int:
    names = {n.strip() for n in args.names.split(",") if n.strip()}
    if not names:
        print("shard_artifacts: no artifact names to import", file=sys.stderr)
        return 0
    if not os.path.isdir(args.store):
        print(
            f"shard_artifacts: artifact store not found: {args.store}\n"
            f"  export the producer shard(s) first (--export-shard / "
            f"--export-shards).",
            file=sys.stderr,
        )
        return 1
    matches = find_matching_artifacts(args.store, names)
    if not matches:
        print(
            f"shard_artifacts: no artifacts for {sorted(names)} found under "
            f"{args.store}",
            file=sys.stderr,
        )
        return 1

    staging = tempfile.mkdtemp(prefix="shard-import-")
    try:
        for path in matches:
            link = os.path.join(staging, os.path.basename(path))
            try:
                os.symlink(path, link)
            except OSError:
                # Fall back to a hard copy when symlinks are unavailable.
                import shutil

                if os.path.isdir(path):
                    shutil.copytree(path, link)
                else:
                    shutil.copy2(path, link)
        cmd = [
            sys.executable, args.buildctl, "bootstrap",
            "--build-dir", args.build_dir,
            "--artifact-dir", staging,
        ]
        if args.target_families:
            cmd += ["--target-families", args.target_families]
        print(f"shard_artifacts: importing {len(matches)} artifact(s) for "
              f"{sorted(names)}")
        return subprocess.run(cmd).returncode
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)


def export_local(args: argparse.Namespace) -> int:
    """Copy a producer group's built artifacts into the shared store.

    The group-shard pipeline's export step. ``ninja artifact-group-<g>``
    populates ``<build-dir>/artifacts`` with exploded ``{name}_{component}_
    {family}`` directories; this copies the ones whose artifact name is in
    ``--names`` into ``--store`` (the same flat layout ``import-bootstrap`` reads
    back). Replaces ``artifact_manager.py push``, which is stage- (not group-)
    scoped. Existing store entries of the same name are replaced.
    """
    import shutil

    names = {n.strip() for n in args.names.split(",") if n.strip()}
    if not names:
        print("shard_artifacts: no artifact names to export", file=sys.stderr)
        return 0
    artifacts_dir = os.path.join(args.build_dir, "artifacts")
    if not os.path.isdir(artifacts_dir):
        print(
            f"shard_artifacts: no artifacts directory {artifacts_dir}\n"
            f"  build the group first (its artifact-group-* target).",
            file=sys.stderr,
        )
        return 1

    matched: list[str] = []
    for entry in sorted(os.listdir(artifacts_dir)):
        path = os.path.join(artifacts_dir, entry)
        is_dir = os.path.isdir(path)
        name = _artifact_name(entry, is_dir=is_dir)
        if name in names:
            matched.append(entry)
    if not matched:
        print(
            f"shard_artifacts: no built artifacts for {sorted(names)} under "
            f"{artifacts_dir}",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.store, exist_ok=True)
    for entry in matched:
        src = os.path.join(artifacts_dir, entry)
        dst = os.path.join(args.store, entry)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    print(f"shard_artifacts: exported {len(matched)} artifact(s) for "
          f"{sorted(names)} to {args.store}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="shard_artifacts.py")
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser(
        "import-bootstrap",
        help="filter a store to producer artifacts and run buildctl bootstrap",
    )
    imp.add_argument("--buildctl", required=True,
                     help="path to TheRock's build_tools/buildctl.py")
    imp.add_argument("--build-dir", required=True,
                     help="CMake build directory to populate")
    imp.add_argument("--store", required=True,
                     help="local artifact store (THEROCK_LOCAL_STAGING_DIR)")
    imp.add_argument("--names", required=True,
                     help="comma-separated topology artifact names to import")
    imp.add_argument("--target-families", default=None,
                     help="comma-separated target families to allow "
                          "(in addition to 'generic')")
    imp.set_defaults(func=import_bootstrap)

    exp = sub.add_parser(
        "export-local",
        help="copy a group's built artifacts from the build dir into a store",
    )
    exp.add_argument("--build-dir", required=True,
                     help="CMake build directory (its artifacts/ is the source)")
    exp.add_argument("--store", required=True,
                     help="local artifact store to copy into")
    exp.add_argument("--names", required=True,
                     help="comma-separated topology artifact names to export")
    exp.set_defaults(func=export_local)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
