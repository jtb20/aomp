"""Named source configurations for the TheRock backend.

A *source config* selects which TheRock sources a build uses -- the git branches
the srock scripts check out -- without ever pinning a SHA by hand. TheRock
records each branch's submodule pins as gitlink SHAs in that branch's own tree,
and ``build_tools/fetch_sources.py`` checks submodules out at those recorded
SHAs (it does not pass ``--remote``). So a config only names a *branch*; checking
that branch out and running ``fetch_sources.py`` always uses whatever pins the
branch currently records. ``branch`` values may also be a tag or commit, which
gives a reproducible "sticky" config for free.

This is orthogonal to the build *scope* (``minimal`` / ``all`` / ``all-debug``,
selected with ``--add`` and surfaced as ``SROCK_CONFIG``): a source config maps
to the srock branch env vars (``SROCK_THEROCK_BRANCH`` / ``SROCK_COMPILER_BRANCH``)
only.

Configs are TOML files under ``srock-bin/source-configs/<name>.toml`` with:

    description      = "..."          # shown by `list-configs`
    therock_branch   = "main"         # TheRock super-repo branch (or tag/commit)
    compiler_branch  = "develop"      # compiler-submodule branch; "develop" means
                                      # native (no srock compiler override/patches)
    patch_tag        = "amd-staging"  # optional; defaults to compiler_branch
    # [overrides]                     # reserved for future per-submodule refs

This module is backend-agnostic and side-effect free: it discovers and parses
config files into ``SourceConfig`` records. The TheRock backend translates a
selected config into child-environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    try:
        import tomli as _toml  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        _toml = None  # type: ignore

# srock-bin/source-configs, resolved from this file (bin/orchestrator/...).
_BIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_BIN_DIR)
SOURCE_CONFIG_DIR = os.path.join(_REPO_ROOT, "srock-bin", "source-configs")

# The config used when --config is not given. Preserves the historical srock
# defaults (SROCK_THEROCK_BRANCH=compiler/amd-staging, SROCK_COMPILER_BRANCH=
# amd-staging), so a bare `therock_build.py` builds amd-staging exactly as before.
DEFAULT_SOURCE_CONFIG = "amd-staging"


class SourceConfigError(Exception):
    """Raised for an unknown config name or a malformed config file."""


@dataclass(frozen=True)
class SourceConfig:
    """A named TheRock source selection (branches only, never SHAs)."""
    name: str
    description: str
    therock_branch: str   # super-repo branch/tag/commit (SROCK_THEROCK_BRANCH)
    compiler_branch: str  # compiler-submodule branch (SROCK_COMPILER_BRANCH);
                          # "develop" => native TheRock (no compiler override)
    patch_tag: str        # srock patch set name (defaults to compiler_branch)
    overrides: dict = field(default_factory=dict)  # reserved for future use


def _config_path(name: str, config_dir: str = SOURCE_CONFIG_DIR) -> str:
    return os.path.join(config_dir, f"{name}.toml")


def available(config_dir: str = SOURCE_CONFIG_DIR) -> list[str]:
    """The config names discoverable under ``config_dir``, sorted."""
    try:
        entries = os.listdir(config_dir)
    except OSError:
        return []
    return sorted(
        os.path.splitext(e)[0] for e in entries if e.endswith(".toml")
    )


def load(name: str, config_dir: str = SOURCE_CONFIG_DIR) -> SourceConfig:
    """Parse and return the named source config.

    Raises ``SourceConfigError`` for an unknown name, a missing TOML parser, or a
    malformed/incomplete file (missing ``therock_branch``/``compiler_branch``).
    """
    if _toml is None:
        raise SourceConfigError(
            "no TOML parser available (need Python 3.11+ tomllib or the tomli "
            "package) to read source configs"
        )
    path = _config_path(name, config_dir)
    if not os.path.isfile(path):
        known = ", ".join(available(config_dir)) or "(none found)"
        raise SourceConfigError(
            f"unknown source config '{name}'; available: {known}"
        )
    try:
        with open(path, "rb") as handle:
            data = _toml.load(handle)
    except Exception as exc:  # parse error, IO error
        raise SourceConfigError(f"cannot read source config '{path}': {exc}")

    therock_branch = str(data.get("therock_branch", "")).strip()
    compiler_branch = str(data.get("compiler_branch", "")).strip()
    if not therock_branch or not compiler_branch:
        raise SourceConfigError(
            f"source config '{path}' must set both 'therock_branch' and "
            f"'compiler_branch'"
        )
    patch_tag = str(data.get("patch_tag", "") or compiler_branch).strip()
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}
    return SourceConfig(
        name=name,
        description=str(data.get("description", "")).strip(),
        therock_branch=therock_branch,
        compiler_branch=compiler_branch,
        patch_tag=patch_tag,
        overrides=overrides,
    )


def catalog(config_dir: str = SOURCE_CONFIG_DIR) -> list[SourceConfig]:
    """All loadable configs (skipping malformed ones), in name order."""
    out: list[SourceConfig] = []
    for name in available(config_dir):
        try:
            out.append(load(name, config_dir))
        except SourceConfigError:
            continue
    return out
