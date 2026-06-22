"""Generic build orchestration: component resolution, topological ordering,
task elaboration, the amd-build-style selector grammar, the logging/stamp
execution loop, and the git-fingerprint version manifest.

Everything backend-specific (how a config is loaded, how a component's tasks
are listed, how one task runs) lives behind the Backend interface; this module
contains no AOMP- or TheRock-specific knowledge.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time

from .backend import Backend
from .model import Config, Task

# Program name used in diagnostics; entry points may override it (e.g.
# "therock_build") so messages match the invoked tool.
PROG = "aomp_build"


def _fail(msg: str) -> "None":
    sys.exit(f"{PROG}: {msg}")


def _warn(msg: str) -> None:
    print(f"{PROG}: warning: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Feature expansion + dependency resolution
# --------------------------------------------------------------------------- #
def expand_names(names: list[str], cfg: Config) -> set[str]:
    """Expand component/feature names into a set of components.

    Each entry may itself be a comma-separated list (so `--add a,b` and
    `--add a --add b` are equivalent), and any entry naming a feature is
    expanded to that feature's components.
    """
    result: set[str] = set()
    for entry in names:
        for name in (n.strip() for n in entry.split(",")):
            if not name:
                continue
            if name in cfg.features:
                result.update(cfg.features[name])
            else:
                result.add(name)
    return result


def resolve_components(
    cfg: Config, adds: list[str], removes: list[str]
) -> list[str]:
    """Resolve the final, topologically ordered component list.

    request +/- (features|components), then transitive dependency closure with
    removal cascade: removing a component also drops anything that depends on
    it (directly or transitively).
    """
    requested = set(cfg.request)
    requested |= expand_names(adds, cfg)
    removed = expand_names(removes, cfg)
    requested -= removed

    unknown = {n for n in requested | removed if n not in cfg.packages}
    if unknown:
        _fail("unknown component(s): " + ", ".join(sorted(unknown)))

    # Transitive dependency closure with removal cascade.
    final = set(requested)
    changed = True
    while changed:
        changed = False
        for comp in list(final):
            for dep in cfg.packages[comp].depends:
                if dep in removed:
                    final.discard(comp)
                    removed.add(comp)
                    changed = True
                    break
                if dep not in final:
                    final.add(dep)
                    changed = True

    return topo_sort(cfg, final)


def reverse_dep_closure(cfg: Config, seed: set[str]) -> set[str]:
    """The seed plus every component that transitively *depends on* it.

    Walks the inverse of the build-dependency graph (cfg.packages[c].depends):
    if rocgdb depends on amd-llvm, then amd-llvm's closure includes rocgdb (and
    anything depending on rocgdb, transitively). Used by --rdeps to rebuild a
    subset's dependents instead of pinning them."""
    closure = {c for c in seed if c in cfg.packages}
    changed = True
    while changed:
        changed = False
        for comp, pkg in cfg.packages.items():
            if comp in closure:
                continue
            if any(dep in closure for dep in pkg.depends):
                closure.add(comp)
                changed = True
    return closure


def topo_sort(cfg: Config, comps: set[str]) -> list[str]:
    """Deterministic Kahn topological sort with declaration-order tie-break."""
    indeg = {c: 0 for c in comps}
    succ: dict[str, list[str]] = {c: [] for c in comps}
    for comp in comps:
        for dep in cfg.packages[comp].depends:
            if dep in comps:
                indeg[comp] += 1
                succ[dep].append(comp)

    ready = sorted((c for c in comps if indeg[c] == 0),
                   key=lambda c: cfg.packages[c].order)
    ordered: list[str] = []
    while ready:
        comp = ready.pop(0)
        ordered.append(comp)
        for nxt in succ[comp]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda c: cfg.packages[c].order)

    if len(ordered) != len(comps):
        cycle = comps - set(ordered)
        _fail("dependency cycle among: " + ", ".join(sorted(cycle)))
    return ordered


# --------------------------------------------------------------------------- #
# Task elaboration
# --------------------------------------------------------------------------- #
def select_variants(
    comp: str, available: list[str], global_variants: list[str],
    per_comp: dict[str, list[str]],
) -> list[str]:
    """Decide which build configs to elaborate for a component.

    Components advertise their configs via the backend. Two styles exist:
    style-A always offers a plain "default" plus opt-in variants (asan/debug);
    style-B (e.g. the runtimes) derives its config set from the environment and
    has no "default" because its default build is produced by the compiler
    (project) build itself.

    The "default" config is the baseline an installable build needs, so it is
    always built when the component offers it.

    Selection:
      * No --variant anywhere: build *every* advertised config (the full build:
        default plus all variants the component offers).
      * --variant given: build "default" (when offered) plus the requested
        variants the component advertises. Requested variants apply globally,
        or to a single component via "comp=cfg" (which overrides the global
        list for that component). A component that offers neither "default" nor
        any requested variant yields an empty list and is skipped entirely - so
        `--variant default` builds just the default config and skips components
        that have none (e.g. the runtimes, built by the project build).
    """
    if not global_variants and not per_comp:
        # No filter at all: build everything the component advertises.
        return list(available)

    requested = per_comp[comp] if comp in per_comp else global_variants

    wanted: list[str] = []
    if "default" in available:
        wanted.append("default")
    for variant in requested:
        if variant in available and variant not in wanted:
            wanted.append(variant)
    return wanted


def elaborate_tasks(
    backend: Backend, cfg: Config, components: list[str], env: dict[str, str],
    global_variants: list[str], per_comp_variants: dict[str, list[str]],
) -> list[Task]:
    tasks: list[Task] = []
    for comp in components:
        available = backend.component_configs(comp, env)
        wanted = select_variants(
            comp, available, global_variants, per_comp_variants
        )
        # An explicit --variant filter that matches none of this component's
        # configs skips the component outright (no init/fini tasks either).
        if available and not wanted:
            continue
        # Components whose only advertised config is "default" use the short
        # two-element task name (comp/stage) instead of comp/default/stage.
        single_config = available == ["default"]
        for action, taskcfg, payload in backend.list_component_tasks(comp, env):
            # Variant filter: config-bearing tasks must match a wanted config;
            # config-less tasks (init/fini such as patch/unpatch) always run.
            if taskcfg is not None and taskcfg not in wanted:
                continue
            tasks.append(
                Task(
                    comp=comp,
                    action=action,
                    cfgname=taskcfg,
                    single_config=single_config,
                    payload=payload,
                )
            )
    return tasks


# --------------------------------------------------------------------------- #
# Selector grammar (amd-build style)
# --------------------------------------------------------------------------- #
def brace_expand(token: str) -> list[str]:
    """Minimal brace expansion: a{b,c}d -> [abd, acd]. Single level."""
    match = re.search(r"\{([^{}]*)\}", token)
    if not match:
        return [token]
    pre, post = token[: match.start()], token[match.end():]
    out: list[str] = []
    for part in match.group(1).split(","):
        out.extend(brace_expand(pre + part + post))
    return out


def select_tasks(tasks: list[Task], selectors: list[str]) -> list[int]:
    """Return the indices (into tasks) selected by the positional selectors.

    Grammar (mirrors amd-build, with a trailing `continue`):
      * no selectors            -> all tasks
      * N                       -> task number N (1-based)
      * N--M                    -> inclusive range
      * glob (with {a,b} braces)-> substring/glob match on
                                   "comp/variant/stage" (or "comp/stage")
      * ... X continue          -> trailing `continue` turns the preceding
                                   selector X into a "from X to the end" anchor;
                                   any earlier selectors are selected normally.
                                   e.g. `comp1 comp2 continue` runs comp1's
                                   tasks then everything from comp2 onward.
    """
    if not selectors:
        return list(range(len(tasks)))

    names = [t.name for t in tasks]
    selectors = list(selectors)

    # `continue` is a trailing modifier: it must be the final token and turns
    # the selector immediately before it into a continue-to-the-end anchor.
    continue_from_last = False
    if selectors and selectors[-1] == "continue":
        selectors.pop()
        if not selectors:
            _fail(
                "trailing 'continue' requires a preceding task "
                "number or name (e.g. 'comp2 continue')"
            )
        continue_from_last = True
    if "continue" in selectors:
        _fail(
            "'continue' must be the last selector "
            "(e.g. 'comp1 comp2 continue' builds comp1 then continues from comp2)"
        )

    selected: list[int] = []

    def add(idx: int) -> None:
        if 0 <= idx < len(tasks) and idx not in selected:
            selected.append(idx)

    def match_indices(sel: str) -> list[int]:
        """Indices matched by a single (non-continue) selector."""
        if sel.isdigit():
            return [int(sel) - 1]
        if "--" in sel:
            lo_s, hi_s = sel.split("--", 1)
            lo, hi = match_point(lo_s), match_point(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            return list(range(lo, hi + 1))
        out: list[int] = []
        for pat in brace_expand(sel):
            for idx, nm in enumerate(names):
                if fnmatch.fnmatch(nm, pat) or pat == nm or pat in nm:
                    out.append(idx)
        if not out:
            _fail(f"selector '{sel}' matched no task")
        return out

    def match_point(token: str) -> int:
        if token.isdigit():
            return int(token) - 1
        for i, nm in enumerate(names):
            if token == nm or fnmatch.fnmatch(nm, token) or token in nm:
                return i
        _fail(f"selector '{token}' matched no task")

    last = len(selectors) - 1
    for i, sel in enumerate(selectors):
        if continue_from_last and i == last:
            # Anchor: from this selector's first matched task to the end.
            start = min(match_indices(sel))
            for idx in range(start, len(tasks)):
                add(idx)
        else:
            for idx in match_indices(sel):
                add(idx)

    selected.sort()
    return selected


def partition_shard(num_tasks: int, k: int, n: int) -> list[int]:
    """Indices belonging to shard k of n over a topologically ordered task list.

    The flat task list is already in dependency order, so contiguous segments
    are dependency-respecting: running shards 1..n in order reproduces a full
    build, and a single shard's segment can run on its own machine provided the
    earlier shards' outputs (shared source/build/install tree) are present.
    Segments are balanced by task count (earlier shards get the +1 remainder).
    """
    if n <= 0:
        _fail("--shard N must be >= 1")
    if not (1 <= k <= n):
        _fail(f"--shard k must be in 1..{n} (got {k})")
    base, extra = divmod(num_tasks, n)
    # Segment sizes: first `extra` shards get base+1, the rest get base.
    start = 0
    bounds: list[tuple[int, int]] = []
    for i in range(n):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    lo, hi = bounds[k - 1]
    return list(range(lo, hi))


def partition_shard_aligned(
    num_tasks: int, run_lengths: list[int], k: int, n: int
) -> list[int]:
    """Like partition_shard, but snap the N cut points to run boundaries.

    `run_lengths` are the lengths of contiguous runs (summing to num_tasks) that
    a shard boundary should not split -- for TheRock these are build stages. We
    compute the ideal balanced cut positions (i*num_tasks/n) and move each to the
    nearest cumulative run boundary, keeping cuts strictly increasing so every
    shard is a contiguous, non-empty-where-possible segment in dependency order.
    Falls back to the plain count-based partition when the runs are unusable.
    """
    if n <= 0:
        _fail("--shard N must be >= 1")
    if not (1 <= k <= n):
        _fail(f"--shard k must be in 1..{n} (got {k})")
    if sum(run_lengths) != num_tasks or any(r <= 0 for r in run_lengths):
        return partition_shard(num_tasks, k, n)

    # Cumulative run boundaries (candidate cut positions), excluding 0/num_tasks.
    boundaries = []
    acc = 0
    for length in run_lengths[:-1]:
        acc += length
        boundaries.append(acc)

    cuts = [0]
    used: set[int] = set()
    for i in range(1, n):
        ideal = round(i * num_tasks / n)
        # Snap to the nearest unused boundary; if all are taken, keep splitting
        # at the ideal position so we still produce n segments.
        candidate = min(
            (b for b in boundaries if b not in used),
            key=lambda b: (abs(b - ideal), b),
            default=ideal,
        )
        if candidate <= cuts[-1]:
            candidate = min(cuts[-1] + 1, num_tasks)
        used.add(candidate)
        cuts.append(candidate)
    cuts.append(num_tasks)
    lo, hi = cuts[k - 1], cuts[k]
    return list(range(lo, hi))


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    """Parse a --shard 'k/N' spec into (k, N), or None when not given."""
    if not spec:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", spec)
    if not m:
        _fail(f"--shard expects 'k/N' (e.g. 2/4), got '{spec}'")
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def tail_file(path: str, n: int = 40) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ""
    return "".join(lines[-n:])


def tail_last_line(path: str, maxbytes: int = 8192) -> str:
    """Last non-empty line of `path` (reading only the trailing `maxbytes`), or
    "" if unreadable/empty. Used to surface live build progress without slurping
    the whole (potentially huge) log on every poll."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            start = max(0, handle.tell() - maxbytes)
            handle.seek(start)
            data = handle.read()
    except OSError:
        return ""
    text = data.decode("utf-8", "replace")
    for line in reversed(text.splitlines()):
        stripped = line.rstrip()
        if stripped.strip():
            return stripped
    return ""


def run_with_progress(
    cmd: list[str], env: dict[str, str], log, log_path: str,
    status: "StatusLine", poll: float = 0.1,
) -> int:
    """Run `cmd` (stdout+stderr -> the open file `log`) while updating `status`
    with the clipped last line of the growing log (two-space indented).

    Polls at most every `poll` seconds (~10 Hz) and only redraws when the log
    has actually grown and its last line changed, so an idle/quiet build does no
    extra terminal I/O. Returns the process return code."""
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env
    )
    last_line: str | None = None
    last_size = -1
    while proc.poll() is None:
        try:
            size = os.path.getsize(log_path)
        except OSError:
            size = last_size
        if size != last_size:
            last_size = size
            line = tail_last_line(log_path)
            if line and line != last_line:
                last_line = line
                status.show(f"  {line}")
        time.sleep(poll)
    return proc.returncode


def stamp_path(stamp_dir: str, task: Task, kind: str) -> str:
    """Path of a task's stamp file (keyed by its stable name). kind is one of
    'start' (written when the task begins) or 'done' (written on success)."""
    return os.path.join(stamp_dir, task.name.replace("/", "-") + "." + kind)


def task_state(stamp_dir: str | None, task: Task) -> str:
    """Completion state inferred from the stamps:

      'done'       -> the task finished (a 'done' stamp is present)
      'incomplete' -> it started but did not finish ('start' only)
      'none'       -> no stamp at all
    """
    if not stamp_dir:
        return "none"
    if os.path.exists(stamp_path(stamp_dir, task, "done")):
        return "done"
    if os.path.exists(stamp_path(stamp_dir, task, "start")):
        return "incomplete"
    return "none"


def render_mark(state: str) -> str:
    """A mark for a task state: green check (done), red cross (incomplete),
    or a blank of the same width (none)."""
    tty = sys.stdout.isatty()
    if state == "done":
        return "\033[32m\u2713\033[0m" if tty else "\u2713"
    if state == "incomplete":
        return "\033[31m\u2717\033[0m" if tty else "\u2717"
    return " "


class StatusLine:
    """An ephemeral one-line build-progress indicator on a TTY.

    On a non-TTY stdout (pipe, file, CI log) every method is a no-op, so output
    stays clean. On a TTY, show() draws a transient mid-grey line at the cursor
    (no trailing newline) and clear() erases it. Callers MUST clear() before any
    real print so the transient line never blends into permanent output -- it is
    meant to be wiped by the next line print, an error, or completion."""

    GREY = "\033[38;5;244m"
    RESET = "\033[0m"
    # Carriage return + "erase entire line": resets the cursor to column 0 and
    # clears whatever the status line drew.
    _ERASE = "\r\033[2K"

    def __init__(self, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.active = False

    def show(self, text: str) -> None:
        if not self.enabled:
            return
        # Keep it to a single physical line: truncate to the terminal width so
        # it never wraps (a wrapped line can't be cleared with one erase).
        cols = shutil.get_terminal_size((80, 24)).columns
        text = text[: max(0, cols - 1)]
        self.stream.write(f"{self._ERASE}{self.GREY}{text}{self.RESET}")
        self.stream.flush()
        self.active = True

    def clear(self) -> None:
        if not self.enabled or not self.active:
            return
        self.stream.write(self._ERASE)
        self.stream.flush()
        self.active = False


def clear_stamps_from(stamp_dir: str, tasks: list[Task], start_index: int) -> None:
    """Remove both stamps for every task at index >= start_index."""
    for task in tasks[start_index:]:
        for kind in ("start", "done"):
            try:
                os.remove(stamp_path(stamp_dir, task, kind))
            except FileNotFoundError:
                pass


def run_install_clean(targets: list[str], env: dict[str, str], log) -> int:
    """Wipe the install dir (targets[0]) and drop the symlink (targets[1]) if it
    is a distinct symlink. Honors SUDO (the install may be root-owned)."""
    install_dir = targets[0] if targets else ""
    symlink = targets[1] if len(targets) > 1 else ""

    abs_install = os.path.abspath(install_dir) if install_dir else ""
    if not abs_install or abs_install in ("/", os.path.abspath(os.path.expanduser("~"))):
        log.write(f"ERROR: refusing to wipe unsafe install dir '{install_dir}'\n")
        log.flush()
        return 1

    sudo = env.get("SUDO", "")
    prefix = ["sudo"] if sudo in ("set", "yes", "YES") else []

    def run(cmd: list[str]) -> int:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        return subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env
        ).returncode

    rc = run(prefix + ["rm", "-rf", "--", install_dir])
    # If the symlink is distinct from the target, remove the dangling link too.
    if rc == 0 and symlink and symlink != install_dir and os.path.islink(symlink):
        rc = run(prefix + ["rm", "-f", "--", symlink])
    return rc


def run_tasks(
    backend: Backend, tasks: list[Task], indices: list[int],
    env: dict[str, str], log_dir: str, dry_run: bool,
    build_type_global: str | None = None,
    build_type_per_comp: dict[str, str] | None = None,
    log_base: str | None = None, stamp_dir: str | None = None,
) -> int:
    build_type_per_comp = build_type_per_comp or {}
    os.makedirs(log_dir, exist_ok=True)
    # Any run clears stamps from the lowest task index onward (a full run thus
    # resets everything), so the stamps reflect only the current build attempt.
    if stamp_dir and not dry_run:
        os.makedirs(stamp_dir, exist_ok=True)
        if indices:
            clear_stamps_from(stamp_dir, tasks, min(indices))
    # The progress counter is relative to the whole build: num is the task's
    # absolute position (1-based) in the full elaborated task list and total is
    # the full count, so a selected subset still reports its real task numbers.
    total = len(tasks)
    width = len(str(total))
    status = StatusLine()
    try:
        return _run_task_loop(
            backend, tasks, indices, env, log_dir, dry_run,
            build_type_global, build_type_per_comp, log_base, stamp_dir,
            total, width, status,
        )
    finally:
        status.clear()


def _run_task_loop(
    backend: Backend, tasks: list[Task], indices: list[int],
    env: dict[str, str], log_dir: str, dry_run: bool,
    build_type_global: str | None,
    build_type_per_comp: dict[str, str],
    log_base: str | None, stamp_dir: str | None,
    total: int, width: int, status: "StatusLine",
) -> int:
    for idx in indices:
        task = tasks[idx]
        num = idx + 1
        safe = task.name.replace("/", "-")
        log_path = os.path.join(log_dir, f"{num:03d}-{safe}.log")
        # Per-component BUILD_TYPE override (per-component wins over global).
        build_type = build_type_per_comp.get(task.comp, build_type_global)
        task_env = env
        extra_env: dict[str, str] = {}
        if task.builtin == "install_clean":
            cmd = ["rm", "-rf", *(t for t in task.targets if t)]
        else:
            cmd, extra_env = backend.task_command(task, env)
        if build_type or extra_env:
            task_env = dict(env)
            task_env.update(extra_env)
            if build_type:
                task_env["BUILD_TYPE"] = build_type
        header = f"[{num:0{width}d}/{total}] {task.name}"
        bt_note = f"  BUILD_TYPE={build_type}" if build_type else ""
        # Show the log path relative to the build root (where logs live) for
        # brevity; fall back to the absolute path if it lies elsewhere.
        rel_log = log_path
        if log_base:
            candidate = os.path.relpath(log_path, log_base)
            if not candidate.startswith(".."):
                rel_log = candidate
        # Wipe any leftover status line before emitting a permanent line.
        status.clear()
        if dry_run:
            print(f"{header}\n    {' '.join(cmd)}{bt_note}  > {rel_log}")
            continue
        print(f"{header}{bt_note} -> {rel_log}", flush=True)
        start = datetime.datetime.now()
        # Mark the task as started (start stamp without a done stamp == an
        # incomplete/failed build until the done stamp is written below).
        if stamp_dir:
            with open(stamp_path(stamp_dir, task, "start"), "w",
                      encoding="utf-8") as st:
                st.write(start.isoformat() + "\n")
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"### task {num}: {task.name}\n")
            log.write(f"### command: {' '.join(cmd)}\n")
            if build_type:
                log.write(f"### env: BUILD_TYPE={build_type}\n")
            log.write(f"### start: {start.isoformat()}\n\n")
            log.flush()
            if task.builtin == "install_clean":
                rc = run_install_clean(task.targets, task_env, log)
            else:
                rc = run_with_progress(
                    cmd, task_env, log, log_path, status
                )
            end = datetime.datetime.now()
            log.write(f"\n### end: {end.isoformat()} (rc={rc})\n")
        if rc != 0:
            status.clear()
            print(
                f"\n{PROG}: FAILED task {num} ({task.name}), rc={rc}",
                file=sys.stderr,
            )
            print(f"--- tail of {log_path} ---", file=sys.stderr)
            print(tail_file(log_path), file=sys.stderr)
            return rc
        # Mark the task complete.
        if stamp_dir:
            with open(stamp_path(stamp_dir, task, "done"), "w",
                      encoding="utf-8") as st:
                st.write(end.isoformat() + "\n")
    return 0


# --------------------------------------------------------------------------- #
# Version manifest (git fingerprint)
# --------------------------------------------------------------------------- #
def _git(src: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", src, *args], capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_toplevel(src: str) -> str:
    """Absolute path of the git repository containing `src` (or "")."""
    if not src or not os.path.isdir(src):
        return ""
    return _git(src, "rev-parse", "--show-toplevel")


def git_facts(src: str) -> dict | None:
    """Git fingerprint of the repository containing `src`.

    `src` may be a subdirectory of its repository: the LLVM project, comgr,
    hipcc and the standalone runtimes all build from different subdirs of the
    single llvm-project checkout. We resolve the enclosing repository root via
    `git rev-parse --show-toplevel` so each such component is recorded, and
    note the build subdir (relative to that root) when it is not the root.
    """
    top = git_toplevel(src)
    if not top:
        return None

    facts = {
        "sha": _git(src, "rev-parse", "HEAD"),
        "repo": _git(src, "config", "--get", "remote.origin.url"),
        "branch": _git(src, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(src, "status", "--porcelain")),
    }
    # Build subdir relative to the repo root (handles src being a subdir of a
    # shared repo, e.g. llvm-project/{llvm,amd/comgr,amd/hipcc,runtimes}).
    # `--show-prefix` is symlink-safe, unlike relpath against --show-toplevel.
    subdir = _git(src, "rev-parse", "--show-prefix").rstrip("/")
    if subdir:
        facts["subdir"] = subdir
    return facts


def export_manifest(
    backend: Backend, components: list[str], env: dict[str, str], path: str,
    config_name: str,
) -> None:
    manifest = {
        "generated": datetime.datetime.now().isoformat(),
        "config": config_name,
        "order": components,
        "components": {},
        "externals": {},
    }
    for comp in components:
        src = backend.component_src_dir(comp, env)
        facts = git_facts(src)
        if facts is not None:
            manifest["components"][comp] = facts

    # External (non-component) repos consumed by the toolchain.
    for name, src in backend.external_repos(env).items():
        facts = git_facts(src)
        if facts is not None:
            manifest["externals"][name] = facts

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"{PROG}: wrote manifest for {len(manifest['components'])} "
          f"component(s) and {len(manifest['externals'])} external repo(s) "
          f"to {path}")


def import_manifest(
    backend: Backend, components: list[str], env: dict[str, str], path: str,
) -> int:
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read manifest '{path}': {exc}")

    recorded = manifest.get("components", {})
    recorded_ext = manifest.get("externals", {})
    floating = backend.floating_components()

    # First pass: build a checkout plan and refuse if any target repo is dirty.
    # Several components (project, comgr, hipcc, runtimes) share the single
    # llvm-project checkout, so we dedupe by repository root to avoid repeated
    # (and redundant) checkouts/dirty reports.
    plan: list[tuple[str, str, str]] = []  # (label, src, sha)
    dirty: list[str] = []
    seen_tops: set[str] = set()

    def schedule(label: str, src: str, sha: str) -> None:
        facts = git_facts(src)
        if facts is None:
            print(f"{PROG}: skipping '{label}' (no git source at {src})")
            return
        top = git_toplevel(src)
        if top in seen_tops:
            return
        seen_tops.add(top)
        if facts["dirty"]:
            dirty.append(label)
        plan.append((label, src, sha))

    # Components, in build order. Floating components track HEAD and are never
    # rolled back.
    for comp in components:
        if comp not in recorded:
            continue
        if comp in floating:
            print(f"{PROG}: keeping '{comp}' at HEAD (floating; not rolled back)")
            continue
        schedule(comp, backend.component_src_dir(comp, env), recorded[comp]["sha"])

    # External repos (e.g. SPIRV-LLVM-Translator).
    for name, src in backend.external_repos(env).items():
        if name not in recorded_ext:
            continue
        schedule(name, src, recorded_ext[name]["sha"])

    if dirty:
        _fail(
            "refusing to import manifest; local modifications in: "
            + ", ".join(dirty)
        )

    # Second pass: check out recorded SHAs.
    for label, src, sha in plan:
        if not sha:
            print(f"{PROG}: skipping '{label}' (no recorded sha)")
            continue
        print(f"{PROG}: checking out {label} @ {sha[:12]} in {src}")
        proc = subprocess.run(["git", "-C", src, "checkout", sha])
        if proc.returncode != 0:
            _fail(f"git checkout failed for '{label}'")
    return 0


# --------------------------------------------------------------------------- #
# Scoped option parsing (--variant / --build-type)
# --------------------------------------------------------------------------- #
def parse_scoped_specs(specs: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Split scoped option values into global values and per-component overrides.

    Used by both --variant and --build-type. Each value may be a comma-separated
    list, and the option is repeatable, so these are equivalent:
        --variant debug,asan
        --variant debug --variant asan
    An entry of the form "comp=value" applies to a single component; bare
    entries apply globally. Both forms can be mixed in one value, e.g.
        --variant comp1=debug,comp2=asan
    Multiple "comp=value" entries for the same component accumulate, in order.
    """
    global_values: list[str] = []
    per_comp: dict[str, list[str]] = {}
    for spec in specs:
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                comp, _, val = entry.partition("=")
                per_comp.setdefault(comp.strip(), []).append(val.strip())
            else:
                global_values.append(entry)
    return global_values, per_comp


def normalize_gfx_list(value: str) -> str:
    """Normalize a --gfx value to the space-separated GFXLIST form.

    Accepts comma- and/or whitespace-separated GPU targets (so the user can
    write `--gfx gfx90a,gfx942` without quoting) and returns them joined by
    single spaces, the form every downstream srock / build_*.sh consumer
    expects (e.g. `gfx90a gfx942`)."""
    return " ".join(value.replace(",", " ").split())


def parse_build_type_specs(specs: list[str]) -> tuple[str | None, dict[str, str]]:
    """Resolve --build-type into a single global type and per-component types.

    Same grammar as --variant (global value or 'comp=type', comma-separated
    and/or repeatable), but a build type is a single value per scope, so the
    last value wins if several are given for the same scope.
    """
    global_values, per_comp_lists = parse_scoped_specs(specs)
    global_bt = global_values[-1] if global_values else None
    per_comp_bt = {comp: vals[-1] for comp, vals in per_comp_lists.items()}
    return global_bt, per_comp_bt


# --------------------------------------------------------------------------- #
# Argument parser (generic flags shared by all backends)
# --------------------------------------------------------------------------- #
def build_arg_parser(
    prog: str, default_config: str,
    description: str = "Unified component build orchestrator.",
    inherit_path_note: str = "",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Selectors (positional, amd-build style):\n"
            "  (none)        run all elaborated tasks\n"
            "  list          print the numbered task list and exit\n"
            "  list-features print the backend's configurable features and exit\n"
            "                  (TheRock: THEROCK_ENABLE_* features; enable one\n"
            "                  with --add <name> --reconfigure)\n"
            "  N             run task number N (1-based)\n"
            "  N--M          run the inclusive range of tasks N..M\n"
            "  comp/variant/stage  glob/substring match (supports {a,b} braces);\n"
            "                  config-less init/fini tasks are comp/stage\n"
            "  continue        on its own, resume from the first task not marked\n"
            "                  complete (by its stamp) through to the end\n"
            "  ... X continue  trailing 'continue' makes the preceding selector\n"
            "                  X a 'from X to the end' anchor, e.g.\n"
            "                  'comp1 comp2 continue' builds comp1 then continues\n"
            "                  from comp2 onward\n"
            "\n"
            "Any run clears completion stamps from the lowest selected task\n"
            "onward; 'list' shows a green check (done) or red cross (incomplete).\n"
        ),
    )
    parser.add_argument("selectors", nargs="*", help="task selector(s); see below")
    parser.add_argument("-c", "--config", default=default_config,
                        help=f"CUDF config file (default: {default_config})")
    # Core directory layout (exported to child build scripts).
    parser.add_argument("-s", "--source", default=None, metavar="DIR",
                        help="source/repo root (AOMP_REPOS) holding the cloned "
                             "component repos. The build dir defaults to it "
                             "unless -b is given. Default: $HOME/git/aomp<ver>")
    parser.add_argument("-i", "--install", default=None, metavar="DIR",
                        help="installation directory root (AOMP); the versioned "
                             "install dir AOMP_<version> derives from it. "
                             "Default: $HOME/rocm/aomp")
    parser.add_argument("-b", "--build", default=None, metavar="DIR",
                        help="directory where builds run / object files go "
                             "(BUILD_AOMP). Default: the repo dir (AOMP_REPOS)")
    parser.add_argument("-p", "--prereq", default=None, metavar="DIR",
                        help="prerequisite/supplemental component root (AOMP_SUPP); "
                             "its build/install subdirs and the prereq cmake "
                             "derive from it. Default: $HOME/local")
    parser.add_argument("--add", action="append", default=[], metavar="NAMES",
                        help="add component(s)/feature(s); comma-separated and/or "
                             "repeatable")
    parser.add_argument("--remove", action="append", default=[], metavar="NAMES",
                        help="remove component(s)/feature(s); comma-separated "
                             "and/or repeatable")
    parser.add_argument("--variant", action="append", default=[], metavar="SPEC",
                        help="variant filter: 'cfg' (global) or 'comp=cfg' "
                             "(per-component). Comma-separated and/or repeatable "
                             "(e.g. 'debug,asan'). 'default' is always built when "
                             "offered, so '--variant debug' means default+debug; "
                             "components offering neither default nor a requested "
                             "variant are skipped (so '--variant default' skips "
                             "the runtimes). With no --variant, all advertised "
                             "configs are built.")
    parser.add_argument("--shard", default=None, metavar="k/N",
                        help="run shard k of N: split the elaborated, dependency-"
                             "ordered task list into N balanced contiguous "
                             "segments and run the k-th (e.g. '2/4'). Composes "
                             "with selectors and stamps.")
    parser.add_argument("-C", "--clean", action="store_true",
                        help="prepend an 'install/clean' task that wipes the "
                             "install directory (the versioned symlink target, "
                             "not just the symlink) before building. Per-"
                             "component 'clean' tasks (build dirs) are always "
                             "listed and run like any other task.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="show what would run without executing")
    parser.add_argument("--components", action="store_true",
                        help="print the resolved, ordered component list and exit")
    parser.add_argument("--log-dir", default=None,
                        help="directory for per-task logs "
                             "(default: <BUILD_DIR>/aomp_build_logs)")
    # Environment knobs passed to child build scripts.
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="parallel build threads (AOMP_JOB_THREADS)")
    parser.add_argument("--ninja", dest="ninja", action="store_true", default=None,
                        help="use ninja (AOMP_USE_NINJA=1)")
    parser.add_argument("--no-ninja", dest="ninja", action="store_false",
                        help="do not use ninja (AOMP_USE_NINJA=0)")
    parser.add_argument("--ccache", dest="ccache", action="store_true", default=None,
                        help="use ccache (AOMP_USE_CCACHE=1)")
    parser.add_argument("--no-ccache", dest="ccache", action="store_false",
                        help="do not use ccache (AOMP_USE_CCACHE=0)")
    parser.add_argument("--gfx", default=None, metavar="LIST",
                        type=normalize_gfx_list,
                        help="GPU target list (GFXLIST); comma- or "
                             "space-separated, e.g. 'gfx90a,gfx942'")
    parser.add_argument("--build-type", action="append", default=[],
                        metavar="SPEC",
                        help="CMake build type (BUILD_TYPE): 'type' (global) or "
                             "'comp=type' (per-component). Comma-separated and/or "
                             "repeatable, e.g. 'project=Debug,comgr=Debug'.")
    parser.add_argument("--sudo", action="store_true",
                        help="install with sudo (SUDO=yes)")
    # Environment isolation.
    parser.add_argument("--inherit-path", action="store_true",
                        help="use the caller's PATH for child build scripts "
                             "instead of the controlled default"
                             + (f" ({inherit_path_note})" if inherit_path_note else ""))
    parser.add_argument("--pass-env", action="append", default=[], metavar="VARS",
                        help="leak named environment variable(s) from the caller "
                             "into the otherwise-isolated child environment; "
                             "comma-separated and/or repeatable "
                             "(e.g. --pass-env CC,CXX,LD_LIBRARY_PATH)")
    # Version manifest.
    parser.add_argument("--export-manifest", nargs="?", const="", metavar="FILE",
                        help="export a git fingerprint manifest and exit "
                             "(default path under <BUILD_DIR>/manifests)")
    parser.add_argument("--import-manifest", metavar="FILE",
                        help="check out recorded git SHAs before building")
    return parser


def add_backend_options(
    parser: argparse.ArgumentParser, default_backend: str = "aomp"
) -> None:
    """Add the backend selector and TheRock-specific knobs to a parser.

    Kept separate from build_arg_parser so the generic flags stay backend-
    agnostic. The TheRock options are no-ops for the AOMP backend.
    """
    parser.add_argument(
        "--backend", choices=["aomp", "therock"], default=default_backend,
        help=f"build backend to drive (default: {default_backend}). 'aomp' runs "
             "the per-component build_<name>.sh scripts; 'therock' drives "
             "TheRock's CMake super-build via subproject_map.json introspection.",
    )
    group = parser.add_argument_group("TheRock backend (--backend therock)")
    group.add_argument(
        "--reconfigure", action="store_true",
        help="force a fresh TheRock cmake configure (with "
             "-DTHEROCK_INTROSPECTION=ON) to regenerate subproject_map.json, "
             "even if a cached one exists.",
    )
    group.add_argument(
        "--therock-dir", default=None, metavar="DIR",
        help="path to the TheRock checkout (overrides SROCK_THEROCK_DIR).",
    )
    group.add_argument(
        "--no-auto-pin", action="store_true",
        help="do not auto-mark out-of-scope components as prebuilt when "
             "building a subset. By default a focused subset build marks every "
             "other (already-built) component prebuilt via buildctl.py so the "
             "build -- and any later whole-tree install -- does not rebuild "
             "dependents you are not working on.",
    )
    group.add_argument(
        "--unpin-all", action="store_true",
        help="clear all prebuilt markers (buildctl.py enable) so every "
             "component is buildable again, then proceed normally.",
    )
    group.add_argument(
        "--rdeps", action="store_true",
        help="when building a subset, also rebuild the components that "
             "(transitively) depend on it instead of pinning them. By default "
             "only the explicitly-selected components rebuild and their "
             "dependents are left prebuilt; with --rdeps the reverse-dependency "
             "closure is built too (e.g. rebuilding amd-llvm also rebuilds "
             "rocgdb).",
    )


def make_backend(args: argparse.Namespace) -> Backend:
    """Instantiate the backend selected by --backend (defaults to aomp)."""
    backend = getattr(args, "backend", "aomp")
    if backend == "therock":
        from .therock_backend import TheRockBackend
        return TheRockBackend()
    from .aomp_backend import AompBackend
    return AompBackend()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace, backend: Backend) -> int:
    cfg = backend.load_config(args)
    config_name = backend.config_name(args)
    child_env = backend.build_child_env(args)
    env_info = backend.discover_env(child_env)

    # `list-features` selector: print the backend's feature catalog and exit.
    # Same positional syntax as `list`; a green check marks an enabled feature,
    # a red cross a disabled one (enable a disabled feature with --add <name>
    # --reconfigure).
    if args.selectors and args.selectors[0] == "list-features":
        rows = backend.list_features(child_env)
        if rows is None:
            print(f"{PROG}: this backend has no configurable features")
            return 0
        if not rows:
            print(f"{PROG}: no features found "
                  f"(run with --reconfigure to generate the feature catalog)")
            return 0
        width = max(len(r["name"]) for r in rows)
        for r in rows:
            mark = render_mark("done" if r["enabled"] else "incomplete")
            line = f"[{mark}] {r['name']:<{width}}"
            if r.get("requires"):
                line += f"  requires: {', '.join(r['requires'])}"
            if r.get("description"):
                line += f"  - {r['description']}"
            print(line)
        return 0

    components = resolve_components(cfg, args.add, args.remove)

    if args.components:
        for comp in components:
            print(comp)
        return 0

    build_dir = env_info["BUILD_DIR"]
    log_dir = args.log_dir or os.path.join(build_dir, "aomp_build_logs")
    # Completion stamps live alongside the log dir.
    stamp_dir = os.path.join(os.path.dirname(os.path.abspath(log_dir)), "stamps")
    manifest_dir = os.path.join(build_dir, "manifests")

    # Manifest export is a standalone action.
    if args.export_manifest is not None:
        path = args.export_manifest or os.path.join(
            manifest_dir, f"{config_name}-manifest.json"
        )
        export_manifest(backend, components, child_env, path, config_name)
        return 0

    # Manifest import runs as a pre-step before any task execution.
    if args.import_manifest:
        rc = import_manifest(backend, components, child_env, args.import_manifest)
        if rc != 0:
            return rc

    global_variants, per_comp_variants = parse_scoped_specs(args.variant)
    build_type_global, build_type_per_comp = parse_build_type_specs(args.build_type)
    tasks = elaborate_tasks(
        backend, cfg, components, child_env, global_variants, per_comp_variants,
    )
    # -C/--clean prepends a pseudo-task that wipes the install directory so a
    # stale install is removed before anything builds. Per-component build-dir
    # "clean" tasks are always in the list and run like any other task.
    if args.clean:
        clean_task = backend.install_clean_task(env_info)
        if clean_task is None:
            _fail("this backend does not support -C/--clean")
        tasks.insert(0, clean_task)

    # Whole-build pseudo-tasks (e.g. TheRock's combined dist + final install)
    # are appended after every per-component task so they run last on a full
    # build and can be selected by name (e.g. 'therock/install').
    tasks += backend.trailing_tasks(components, child_env)

    # `list` selector: print the numbered task list and exit. A green check
    # marks completed tasks (or ones already built in the backend), a red cross
    # marks started-but-unfinished ones. Trailing selectors after `list` preview
    # a focused build: components not in that set (and already built) are shown
    # with a [pinned] suffix, since a real run would mark them prebuilt.
    if args.selectors and args.selectors[0] == "list":
        preview = args.selectors[1:]
        explicit = {tasks[i].comp for i in select_tasks(tasks, preview)}
        if args.rdeps:
            explicit |= reverse_dep_closure(cfg, explicit)
        built = backend.built_components(child_env)
        # Only a strict, non-empty preview pins anything (a full/empty preview
        # builds everything, pinning nothing). Pinned subset is the already-built
        # complement of the focused set.
        all_comps = {t.comp for t in tasks}
        focused = bool(preview) and explicit and explicit != all_comps
        pinned: set[str] = (built - explicit) if (built and focused) else set()
        width = len(str(len(tasks)))
        for i, task in enumerate(tasks, start=1):
            state = task_state(stamp_dir, task)
            if state != "done" and built and task.comp in built:
                state = "done"
            mark = render_mark(state)
            suffix = "  [pinned]" if task.comp in pinned else ""
            print(f"[{i:0{width}d}] [{mark}] {task.name}{suffix}")
        return 0

    # Bare `continue`: resume from the first task that is not yet done.
    if args.selectors == ["continue"]:
        resume = next(
            (i for i, t in enumerate(tasks)
             if task_state(stamp_dir, t) != "done"),
            None,
        )
        if resume is None:
            print(f"{PROG}: all tasks already complete")
            return 0
        indices = list(range(resume, len(tasks)))
    else:
        indices = select_tasks(tasks, args.selectors)

    # --shard k/N narrows the selection to the k-th dependency-ordered segment.
    # A backend may supply contiguous run boundaries (e.g. TheRock build stages)
    # for the cut points to snap to; otherwise the segments are balanced purely
    # by task count.
    shard = parse_shard(args.shard)
    if shard is not None:
        k, n = shard
        runs = backend.shard_run_lengths(tasks, child_env)
        if runs:
            shard_idx = set(partition_shard_aligned(len(tasks), runs, k, n))
        else:
            shard_idx = set(partition_shard(len(tasks), k, n))
        indices = [i for i in indices if i in shard_idx]

    if not indices:
        print(f"{PROG}: no tasks selected")
        return 0

    # --rdeps: when building a strict subset, also run the tasks of every
    # component that (transitively) depends on the selected set, so dependents
    # are rebuilt rather than left pinned. The reverse-dependency closure pulls
    # those components' tasks into the run (kept in dependency order).
    if args.rdeps:
        selected_comps = {tasks[i].comp for i in indices}
        all_comps = {t.comp for t in tasks}
        if selected_comps and selected_comps != all_comps:
            expanded = reverse_dep_closure(cfg, selected_comps)
            if expanded != selected_comps:
                idx_set = set(indices)
                idx_set |= {i for i, t in enumerate(tasks) if t.comp in expanded}
                indices = sorted(idx_set)

    # Let the backend adjust build state for exactly the components about to
    # run (e.g. TheRock marks out-of-scope components prebuilt so a focused
    # subset build does not cascade rebuilds into dependents).
    backend.prepare_run({tasks[i].comp for i in indices}, child_env, args)

    return run_tasks(
        backend, tasks, indices, child_env, log_dir, args.dry_run,
        build_type_global=build_type_global,
        build_type_per_comp=build_type_per_comp,
        log_base=build_dir, stamp_dir=stamp_dir,
    )
