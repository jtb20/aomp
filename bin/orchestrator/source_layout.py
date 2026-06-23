"""Shared-source mapping between the AOMP and TheRock checkout layouts.

TheRock is treated as the canonical layout. Only the components that are
*standalone git repositories in both* layouts can be shared/moved 1:1; the
monorepo-backed components (TheRock's ``rocm-systems`` / ``rocm-libraries``
submodules, which AOMP keeps as separate repos) are out of scope and stay
managed by ``clone_aomp.sh`` / ``clone_rocmlibs.sh`` (AOMP) and
``fetch_sources.py`` (TheRock).

This module is backend-agnostic: it computes *plans* (pure, testable) and
applies them. The backends do the user-facing messaging.

Two operations:
  * symlink_plan / apply_symlinks  -- AOMP builds reuse a TheRock checkout's
    sources via whole-repo directory symlinks (robust, unlike a per-file lndir
    shadow: git, edits and add/remove all resolve to the real repo).
  * migrate_plan / apply_migration -- pre-seed TheRock's submodule slots by
    *moving* repos out of an AOMP checkout and converting each standalone repo
    into a submodule gitdir (``.git/modules/<name>`` + a gitlink ``.git`` file).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SharedComponent:
    """A component that is a standalone git repo in both layouts."""
    aomp_dir: str        # path under AOMP_REPOS (e.g. "llvm-project", "rocmlibs/half")
    therock_path: str    # path under the TheRock checkout (the submodule working tree)
    submodule_name: str  # .gitmodules name == .git/modules/<name>
    therock_branch: str  # TheRock's expected submodule branch (for mismatch warnings)


# The canonical 1:1 set. llvm-project also backs AOMP's project/comgr/hipcc via
# subpaths (llvm/, amd/comgr, amd/hipcc), which resolve through the single
# directory symlink.
SHARED_COMPONENTS: tuple[SharedComponent, ...] = (
    SharedComponent("llvm-project", "compiler/amd-llvm", "llvm-project", "amd-staging"),
    SharedComponent("rocm-cmake", "base/rocm-cmake", "rocm-cmake", "mainline"),
    SharedComponent("hipify", "compiler/hipify", "HIPIFY", "amd-staging"),
    SharedComponent(
        "ROCgdb", "debug-tools/rocgdb/source", "rocgdb",
        "amd-staging-rocgdb-16",
    ),
    SharedComponent("rocmlibs/half", "base/half", "half", "rocm"),
    SharedComponent(
        "SPIRV-LLVM-Translator", "compiler/spirv-llvm-translator",
        "spirv-llvm-translator", "amd-staging",
    ),
)


def _abs(*parts: str) -> str:
    return os.path.abspath(os.path.join(*parts))


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True,
    )


def _git_out(repo: str, *args: str) -> str:
    proc = _git(repo, *args)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def is_git_repo(path: str) -> bool:
    return os.path.isdir(path) and _git(path, "rev-parse", "--git-dir").returncode == 0


def _is_dirty(path: str) -> bool:
    return bool(_git_out(path, "status", "--porcelain"))


def _current_branch(path: str) -> str:
    return _git_out(path, "rev-parse", "--abbrev-ref", "HEAD")


def _dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))


# --------------------------------------------------------------------------- #
# Symlink plan (AOMP <- TheRock canonical sources)
# --------------------------------------------------------------------------- #
@dataclass
class SymlinkAction:
    comp: SharedComponent
    link_path: str   # abs AOMP_REPOS/<aomp_dir>
    target: str      # abs <therock>/<therock_path>
    status: str      # "link" | "already-linked" | "skip-existing" | "missing-target"
    note: str = ""


def symlink_plan(dest_aomp_repos: str, therock_dir: str) -> list[SymlinkAction]:
    """Plan whole-repo directory symlinks from the AOMP layout into a TheRock
    checkout. dest_aomp_repos is -s/--source (AOMP_REPOS)."""
    actions: list[SymlinkAction] = []
    for comp in SHARED_COMPONENTS:
        link_path = _abs(dest_aomp_repos, comp.aomp_dir)
        target = _abs(therock_dir, comp.therock_path)
        if not _dir_nonempty(target):
            status, note = "missing-target", (
                "TheRock submodule not checked out (run TheRock's "
                "fetch_sources.py first)"
            )
        elif os.path.islink(link_path):
            if _abs(os.path.realpath(link_path)) == _abs(os.path.realpath(target)):
                status, note = "already-linked", ""
            else:
                status, note = "link", "replacing stale symlink"
        elif os.path.exists(link_path):
            status, note = "skip-existing", (
                "a real directory already exists here; not clobbering"
            )
        else:
            status, note = "link", ""
        actions.append(SymlinkAction(comp, link_path, target, status, note))
    return actions


def apply_symlinks(
    actions: list[SymlinkAction], dry_run: bool = False, log=print,
) -> None:
    for act in actions:
        if act.status not in ("link",):
            continue
        rel = os.path.relpath(act.target, os.path.dirname(act.link_path))
        log(f"  symlink {act.comp.aomp_dir} -> {rel}")
        if dry_run:
            continue
        os.makedirs(os.path.dirname(act.link_path), exist_ok=True)
        if os.path.islink(act.link_path) or os.path.exists(act.link_path):
            # Only ever remove a symlink we manage (status "link" excludes real dirs).
            if os.path.islink(act.link_path):
                os.unlink(act.link_path)
        os.symlink(act.target, act.link_path)


# --------------------------------------------------------------------------- #
# Migrate plan (move AOMP repos -> TheRock submodule slots)
# --------------------------------------------------------------------------- #
@dataclass
class MigrateAction:
    comp: SharedComponent
    src_repo: str    # abs <aomp_repodir>/<aomp_dir>
    dest_path: str   # abs <therock>/<therock_path>
    status: str      # "ready" | "missing-source" | "not-git" | "dest-occupied"
    warnings: list[str] = field(default_factory=list)


def migrate_plan(dest_therock_dir: str, aomp_repodir: str) -> list[MigrateAction]:
    """Plan moving the shared repos out of an AOMP checkout (aomp_repodir) into
    the TheRock checkout's submodule slots (dest_therock_dir == the resolved
    TheRock dir, from -s/--source or --therock-dir)."""
    git_dir = _git_out(dest_therock_dir, "rev-parse", "--absolute-git-dir")
    actions: list[MigrateAction] = []
    for comp in SHARED_COMPONENTS:
        src = _abs(aomp_repodir, comp.aomp_dir)
        dest = _abs(dest_therock_dir, comp.therock_path)
        warnings: list[str] = []
        if not os.path.isdir(src):
            actions.append(MigrateAction(comp, src, dest, "missing-source"))
            continue
        if not is_git_repo(src):
            actions.append(MigrateAction(comp, src, dest, "not-git"))
            continue
        module_dir = os.path.join(git_dir, "modules", comp.submodule_name) \
            if git_dir else ""
        if _dir_nonempty(dest) or (module_dir and os.path.exists(module_dir)):
            actions.append(MigrateAction(comp, src, dest, "dest-occupied"))
            continue
        if _is_dirty(src):
            warnings.append("uncommitted changes (they move with the tree)")
        branch = _current_branch(src)
        if branch and branch != comp.therock_branch:
            warnings.append(
                f"on branch '{branch}', TheRock expects '{comp.therock_branch}' "
                f"(SHAs reconcile on the next TheRock fetch/configure)"
            )
        actions.append(MigrateAction(comp, src, dest, "ready", warnings))
    return actions


def apply_migration(
    dest_therock_dir: str, actions: list[MigrateAction],
    dry_run: bool = False, log=print,
) -> None:
    """Move + convert each "ready" action's standalone repo into a TheRock
    submodule gitdir. Best-effort registration into the superproject config."""
    git_dir = _git_out(dest_therock_dir, "rev-parse", "--absolute-git-dir")
    for act in actions:
        if act.status != "ready":
            continue
        log(f"  move {act.comp.aomp_dir} -> {act.comp.therock_path}")
        if dry_run:
            continue
        # 1. Move the whole working tree (incl. its .git directory) into place.
        os.makedirs(os.path.dirname(act.dest_path), exist_ok=True)
        shutil.move(act.src_repo, act.dest_path)
        # 2. Relocate the repo's .git dir into the superproject's modules store.
        module_dir = os.path.join(git_dir, "modules", act.comp.submodule_name)
        inner_git = os.path.join(act.dest_path, ".git")
        os.makedirs(os.path.dirname(module_dir), exist_ok=True)
        shutil.move(inner_git, module_dir)
        # 3. Write the gitlink file and reconnect worktree <-> gitdir.
        with open(inner_git, "w", encoding="utf-8") as fh:
            fh.write(f"gitdir: {os.path.relpath(module_dir, act.dest_path)}\n")
        subprocess.run(
            ["git", "config", "--file", os.path.join(module_dir, "config"),
             "core.worktree", os.path.relpath(act.dest_path, module_dir)],
            check=False,
        )
        # 4. Register the submodule in the superproject config (url/active from
        #    .gitmodules). Best-effort: the working tree is already usable, and
        #    TheRock's fetch_sources/configure reconciles the recorded SHA.
        init = _git(dest_therock_dir, "submodule", "init", "--",
                    act.comp.therock_path)
        if init.returncode != 0:
            log(f"    note: 'git submodule init {act.comp.therock_path}' "
                f"failed; reconcile with TheRock's fetch_sources.py")
