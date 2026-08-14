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
import textwrap
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

    Walks the inverse of the dependency graph using both build edges
    (`depends`) and runtime/link edges (`runtime_depends`): if rocgdb depends on
    amd-llvm (even only at link/runtime), then amd-llvm's closure includes
    rocgdb (and anything depending on rocgdb, transitively). Used by --rdeps to
    rebuild a subset's dependents instead of pinning them -- so changing the
    compiler propagates a rebuild to everything built against it."""
    closure = {c for c in seed if c in cfg.packages}
    changed = True
    while changed:
        changed = False
        for comp, pkg in cfg.packages.items():
            if comp in closure:
                continue
            if any(dep in closure
                   for dep in (*pkg.depends, *pkg.runtime_depends)):
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


def parse_shard_list(spec: str | None) -> list[str]:
    """Split a comma-separated shard option into an ordered, de-duplicated list.

    Empty/whitespace entries are dropped. Returns [] for None or "" so callers
    can treat "option not given" and "given empty" alike.
    """
    if not spec:
        return []
    out: list[str] = []
    for tok in spec.split(","):
        name = tok.strip()
        if name and name not in out:
            out.append(name)
    return out


def print_shard_catalog(rows: list[dict]) -> None:
    """Pretty-print the shard catalog for `list-shards`.

    Shards are TheRock artifact groups, printed in dependency (build) order as a
    vertical block each so the output stays within the terminal width. A short
    legend explains the fields, and long comma-lists wrap with hanging indent.
    """
    cols = shutil.get_terminal_size((80, 24)).columns
    label_w = len("artifacts")  # widest field label, for aligned ": " columns
    indent = "  "
    hang = indent + " " * (label_w + 2)  # continuation indent past "label : "

    def field(label: str, value: str) -> None:
        prefix = f"{indent}{label:<{label_w}} : "
        avail = max(20, cols - len(prefix))
        wrapped = textwrap.wrap(value, width=avail) or [""]
        print(prefix + wrapped[0])
        for cont in wrapped[1:]:
            print(hang + cont)

    print("Shards are TheRock artifact groups, in build order. Pick names for")
    print("--import-shard / --build-shard / --export-shard.\n")
    print("  builds    subprojects this group builds (from the current "
          "configure)")
    print("  imports   dependency groups to import first (use as --import-shard)")
    print("  sources   source sets fetched for this group (fetch_sources.py)")
    print("  artifacts count produced (exported) / consumed from upstream\n")

    for r in rows:
        name = r["name"]
        header = name
        if r.get("description"):
            header += f"  - {r['description']}"
        print(header)

        subs = r.get("subprojects") or []
        if subs:
            field("builds", ", ".join(subs) + f"  ({len(subs)})")
        else:
            # artifact_map.json only carries enabled artifacts, so an empty set
            # means this group's subprojects are off in the current profile.
            field("builds", "(no subprojects in the current configure)")

        deps = r.get("depends_on") or []
        field("imports", ", ".join(deps) if deps else "(none)")

        sources = r.get("source_sets") or []
        if sources:
            field("sources", ", ".join(sources))

        field("artifacts",
              f"{r.get('produced', 0)} produced, {r.get('inbound', 0)} inbound")
        print()


def _print_feature_rows(rows: list[dict]) -> None:
    """Print the `list-features` catalog.

    Two row shapes are supported. AOMP rows carry a "kind" ("group" or
    "component") and are printed in two labelled sections with a built-by-
    default mark (and group member lists); a check means the row is in the
    current build set (removable), a cross means it is addable. Rows without a
    "kind" use the flat TheRock formatting (name + requires + description)."""
    if any("kind" in r for r in rows):
        groups = [r for r in rows if r.get("kind") == "group"]
        comps = [r for r in rows if r.get("kind") == "component"]
        gwidth = max((len(r["name"]) for r in groups), default=0)
        if groups:
            print("Component groups (pass to --add / --remove):")
            for r in groups:
                mark = render_mark("done" if r["enabled"] else "incomplete")
                line = f"  [{mark}] {r['name']:<{gwidth}}"
                members = r.get("members") or []
                if members:
                    line += f"  -> {', '.join(members)}"
                if r.get("partial") or (
                    0 < r.get("present", 0) < r.get("total", 0)
                ):
                    line += f"  (partial: {r['present']}/{r['total']})"
                print(line)
            print()
        if comps:
            cwidth = max(len(r["name"]) for r in comps)
            print("Components (pass to --add / --remove):")
            for r in comps:
                mark = render_mark("done" if r["enabled"] else "incomplete")
                print(f"  [{mark}] {r['name']:<{cwidth}}")
            print()
        print("[\u2713] = in the current build set (removable);  "
              "[\u2717] = not built (addable).")
        return
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        mark = render_mark("done" if r["enabled"] else "incomplete")
        line = f"[{mark}] {r['name']:<{width}}"
        if r.get("requires"):
            line += f"  requires: {', '.join(r['requires'])}"
        if r.get("description"):
            line += f"  - {r['description']}"
        print(line)


def print_variant_catalog(rows: list[dict]) -> None:
    """Print the `list-variants` catalog: components advertising a non-default
    build variant, one per line, with a short note on how variants are selected
    and that the advertised set is environment-gated."""
    print("Build variants are selected with --variant:")
    print("  --variant <cfg>          apply to every component that offers it")
    print("  --variant <comp>=<cfg>   apply only to that component")
    print("'default' is always built when a component offers it, so e.g. "
          "'--variant asan' builds default+asan.")
    print("Advertised variants are environment-gated "
          "(AOMP_BUILD_SANITIZER / AOMP_BUILD_DEBUG / AOMP_BUILD_PERF); set "
          "them (e.g. via --pass-env) to expose asan/debug/perf.\n")
    width = max(len(r["component"]) for r in rows)
    for r in rows:
        print(f"  {r['component']:<{width}}  {', '.join(r['variants'])}")


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


def _fmt_elapsed(secs: float) -> str:
    """Compact elapsed time: '45s', '2m03s', '1h07m'."""
    s = int(secs)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def run_with_progress(
    cmd: list[str], env: dict[str, str], log, log_path: str,
    status: "StatusLine", poll: float = 0.1,
) -> int:
    """Run `cmd` (stdout+stderr -> the open file `log`) while updating `status`
    with the clipped last line of the growing log, prefixed by a live elapsed
    clock: ``  [2m03s] <last log line>`` (two-space indented).

    The line is redrawn when the log gains a new last line, and at least once a
    second so the clock keeps ticking -- a quiet-but-still-working step (e.g. a
    long compile whose output has not yet flushed to the log) never looks hung.
    Completion is detected immediately via wait(), and the transient line is
    cleared the moment the task finishes. Returns the process return code."""
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env
    )
    # No TTY -> nothing to draw; just wait (and reap) the process.
    if not status.enabled:
        return proc.wait()
    start = time.monotonic()
    last_line = ""
    last_size = -1
    last_draw = -1.0
    try:
        while True:
            try:
                rc = proc.wait(timeout=poll)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            changed = False
            try:
                size = os.path.getsize(log_path)
            except OSError:
                size = last_size
            if size != last_size:
                last_size = size
                line = tail_last_line(log_path)
                if line and line != last_line:
                    last_line = line
                    changed = True
            # Redraw on a new log line or ~once a second to tick the clock.
            if changed or now - last_draw >= 1.0:
                last_draw = now
                elapsed = _fmt_elapsed(now - start)
                text = f"  [{elapsed}] {last_line}" if last_line \
                    else f"  [{elapsed}]"
                status.show(text)
    finally:
        # The task is done: drop the transient line right away rather than
        # leaving its last line frozen until the next header prints.
        status.clear()
    return rc


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


def task_is_done(
    stamp_dir: str | None, task: Task, built: set[str] | None
) -> bool:
    """Whether a task counts as complete for `list`/`continue`.

    A task is done if its orchestrator 'done' stamp is present *or* the backend
    considers its component already built (a valid stage dir, via
    built_components). Both sources are honored so the `list` tick-box and bare
    `continue` agree -- e.g. a component staged outside the orchestrator shows as
    done and is skipped by `continue`."""
    if task_state(stamp_dir, task) == "done":
        return True
    return bool(built) and task.comp in built


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
            "                  with --add <name> --reconfigure. AOMP: CUDF\n"
            "                  component groups + components for --add/--remove)\n"
            "  list-variants print components with non-default build variants\n"
            "                  and exit (AOMP only; select with --variant)\n"
            "  list-shards   print the backend's shard catalog and exit\n"
            "                  (TheRock: BUILD_TOPOLOGY.toml artifact groups;\n"
            "                  drive one with --import/build/export-shard)\n"
            "  list-configs  print the backend's source configs and exit\n"
            "                  (TheRock: which branches -c/--config selects)\n"
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
                        help="config selector: a CUDF config file (AOMP "
                             "backend) or a source-config name (TheRock backend; "
                             "see `list-configs`). "
                             f"Default: {default_config}")
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
        "--superproject-build", dest="delegate", action="store_false",
        default=True,
        help="for the per-subproject 'build' stage, always run the super-level "
             "`ninja <comp>+build` instead of running ninja directly in the "
             "subproject's build dir. By default, once a subproject has been "
             "configured (its build/build.ninja exists), its build runs in-dir "
             "so local source edits are detected (TheRock 'Option 1'); the "
             "super-project's stamp tracking can otherwise miss them for large "
             "components.",
    )
    group.add_argument(
        "-a", "--all", action="store_true",
        help="elaborate every advertised per-component action "
             "(expunge/configure/build/stage/dist) instead of just the default "
             "configure/build/stage. Intended for `list` and for targeted "
             "selection while untangling a build (e.g. 'amd-llvm/expunge'); note "
             "that per-component 'dist' triggers whole-tree distribution "
             "assembly and 'expunge' is a destructive clean, so a bare --all "
             "run would clean/dist every component.",
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
    group.add_argument(
        "--build-tests", action="store_true",
        help="build each component's test suites (THEROCK_BUILD_TESTING=ON). "
             "Off by default: TheRock would otherwise build test-only "
             "subprojects such as rocPRIM_tests, which dominate build time for "
             "a Debug compiler. Being a configure-time gate, changing it needs "
             "--reconfigure.",
    )

    shard = parser.add_argument_group(
        "group-based sharding (--backend therock)"
    )
    shard.add_argument(
        "--import-shard", default=None, metavar="LIST",
        help="comma-separated producer shard(s) (artifact groups) whose "
             "artifacts to import into the build tree before building, e.g. "
             "'compiler'. Pulled from --shard-store and unpacked as prebuilt "
             "via TheRock's buildctl.py bootstrap. See `list-shards`.",
    )
    shard.add_argument(
        "--build-shard", default=None, metavar="LIST",
        help="comma-separated shard(s) (artifact groups) to build, e.g. "
             "'hip-runtime'. Only the named groups' subprojects build; their "
             "sources are fetched (fetch_sources.py --source-sets) for just "
             "those groups.",
    )
    shard.add_argument(
        "-f", "--fill", action="store_true",
        help="build every configured shard (artifact group) NOT named by "
             "--import-shard, so you need not hand-calculate the inverse of "
             "--import-shard. Implies --deploy (assemble + install). TheRock "
             "only; mutually exclusive with --build-shard. See `list-shards`.",
    )
    shard.add_argument(
        "--export-shard", default=None, metavar="LIST",
        help="comma-separated producer shard(s) (artifact groups) whose built "
             "artifacts to export to --shard-store after building.",
    )
    shard.add_argument(
        "--export-shards", action="store_true",
        help="export the artifacts of every shard named by --build-shard "
             "(sugar so the build set need not be repeated in --export-shard).",
    )
    shard.add_argument(
        "--shard-store", default=None, metavar="DIR",
        help="local artifact store for shard import/export (TheRock's "
             "THEROCK_LOCAL_STAGING_DIR). Default: <BUILD_DIR>/shard-artifacts.",
    )
    shard.add_argument(
        "--shard-run-id", default="local", metavar="LABEL",
        help="run-id namespace under --shard-store for push/fetch "
             "(default: 'local').",
    )
    shard.add_argument(
        "--shard-families", default=None, metavar="LIST",
        help="comma-separated target families to import in addition to "
             "'generic' (e.g. 'gfx94X'), for per-arch artifacts on "
             "--import-shard. Default: generic only.",
    )
    shard.add_argument(
        "--deploy", action="store_true",
        help="in shard mode, also assemble the imported + built shards into the "
             "combined dist tree and the final install dir (re-enables the "
             "trailing therock/dist + therock/install steps that a shard run "
             "otherwise skips). No effect outside shard mode.",
    )

    prov = parser.add_argument_group(
        "source provisioning (run once, before building)"
    )
    prov.add_argument(
        "--clone", action="store_true",
        help="[aomp] clone the AOMP sources into -s/--source via clone_aomp.sh "
             "(and rocmlibs/clone_rocmlibs.sh when the selected set includes "
             "rocmlibs components), then build as usual.",
    )
    prov.add_argument(
        "--therock-symlinks", default=None, metavar="DIR",
        help="[aomp] provision -s/--source by symlinking the shared standalone "
             "repos (llvm-project, rocm-cmake, hipify, ROCgdb, half, "
             "SPIRV-LLVM-Translator) from the TheRock checkout DIR, then "
             "clone the remaining AOMP-only repos via clone_aomp.sh.",
    )
    prov.add_argument(
        "--migrate-aomp", default=None, metavar="REPODIR",
        help="[therock] MOVE the shared standalone repos out of the AOMP "
             "checkout REPODIR into -s/--source's TheRock submodule slots, "
             "converting each standalone repo into a submodule gitdir. "
             "Destructive (moves directories); prompts unless -y/--yes.",
    )
    prov.add_argument(
        "-y", "--yes", action="store_true",
        help="skip the confirmation prompt for destructive provisioning "
             "(--migrate-aomp).",
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
    # Pre-config source provisioning (e.g. TheRock's --migrate-aomp) runs before
    # load_config: for TheRock, load_config triggers the cmake configure which
    # fetches submodule sources, so seeding the submodule slots has to happen
    # first. A handled pre-config step returns an exit code and ends the run.
    preconfig_rc = backend.provision_preconfig(args)
    if preconfig_rc is not None:
        return preconfig_rc

    # `list-configs` selector: print the backend's source-config catalog (which
    # sources -c/--config can select) and exit. Static (reads the config files),
    # so it runs before load_config -- no checkout or configure required.
    if args.selectors and args.selectors[0] == "list-configs":
        rows = backend.list_source_configs()
        if rows is None:
            print(f"{PROG}: this backend has no source-config concept")
            return 0
        if not rows:
            print(f"{PROG}: no source configs found")
            return 0
        width = max(len(r["name"]) for r in rows)
        for r in rows:
            tag = " (default)" if r.get("default") else ""
            line = f"{r['name']:<{width}}{tag}"
            branches = []
            if r.get("therock_branch"):
                branches.append(f"therock={r['therock_branch']}")
            if r.get("compiler_branch"):
                branches.append(f"compiler={r['compiler_branch']}")
            if branches:
                line += f"  [{', '.join(branches)}]"
            if r.get("description"):
                line += f"  - {r['description']}"
            print(line)
        return 0

    cfg = backend.load_config(args)
    config_name = backend.config_name(args)
    child_env = backend.build_child_env(args)
    env_info = backend.discover_env(child_env)

    # `list-features` selector: print the backend's feature catalog and exit.
    # Same positional syntax as `list`; a green check marks an enabled feature,
    # a red cross a disabled one. For TheRock these are THEROCK_ENABLE_* toggles
    # (enable with --add <name> --reconfigure); for AOMP they are the CUDF
    # component groups plus the individual components addable/removable via
    # --add/--remove (a check means it is in the current build set).
    if args.selectors and args.selectors[0] == "list-features":
        rows = backend.list_features(child_env)
        if rows is None:
            print(f"{PROG}: this backend has no configurable features")
            return 0
        if not rows:
            print(f"{PROG}: no features found "
                  f"(run with --reconfigure to generate the feature catalog)")
            return 0
        _print_feature_rows(rows)
        return 0

    # `list-variants` selector: print the components that advertise a
    # non-default build variant and how to select them. AOMP only (TheRock
    # subprojects are config-less).
    if args.selectors and args.selectors[0] == "list-variants":
        rows = backend.list_variants(child_env)
        if rows is None:
            print(f"{PROG}: this backend has no build-variant concept")
            return 0
        if not rows:
            print(f"{PROG}: no components advertise non-default variants "
                  f"(variants are environment-gated; e.g. set "
                  f"AOMP_BUILD_SANITIZER=1 and pass it with --pass-env to "
                  f"expose 'asan').")
            return 0
        print_variant_catalog(rows)
        return 0

    # `list-shards` selector: print the backend's shard catalog (TheRock
    # artifact groups) and exit. Shows each group's subprojects, dependency
    # groups, and artifact counts so the user can pick shard names.
    if args.selectors and args.selectors[0] == "list-shards":
        rows = backend.list_shards(child_env)
        if rows is None:
            print(f"{PROG}: this backend has no shard concept")
            return 0
        if not rows:
            print(f"{PROG}: no shards found "
                  f"(needs a TheRock checkout with BUILD_TOPOLOGY.toml)")
            return 0
        print_shard_catalog(rows)
        return 0

    components = resolve_components(cfg, args.add, args.remove)

    # Source provisioning (--clone / --therock-symlinks / --migrate-aomp) runs
    # once here, after the component set is known and before any task work.
    prov_rc = backend.provision_sources(args, child_env, components)
    if prov_rc != 0:
        return prov_rc

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

    # Group-based sharding: when any --import/build/export-shard option is given,
    # the task list becomes the backend's import->build->export pipeline for the
    # named artifact groups instead of the normal whole-tree run. --export-shards
    # is sugar for "export every build shard". The leading prereq still runs (it
    # only builds the cmake/ninja toolchain); the trailing whole-tree dist/
    # install is suppressed (a shard run produces/pushes artifacts, it does not
    # assemble the full SDK) unless --deploy asks to assemble them.
    shard_import = parse_shard_list(getattr(args, "import_shard", None))
    shard_build = parse_shard_list(getattr(args, "build_shard", None))
    shard_export = parse_shard_list(getattr(args, "export_shard", None))

    # -f/--fill: build every configured shard not named by --import-shard, so
    # the user need not hand-calculate the inverse build set. It owns the build
    # set (mutually exclusive with --build-shard) and implies --deploy so the
    # imported + freshly-built shards are assembled and installed.
    if getattr(args, "fill", False):
        if shard_build:
            _fail("--fill is mutually exclusive with --build-shard (it builds "
                  "every configured shard not imported).")
        rest = backend.rest_build_shards(shard_import, child_env)
        if rest is None:
            _fail("this backend does not support group-based sharding "
                  "(--fill / --import-shard / --build-shard / --export-shard)")
        if not rest:
            _fail("--fill found no configured shards to build (all configured "
                  "groups were imported, or the build is not configured / "
                  "artifact_map.json is missing). Re-run with --reconfigure or "
                  "drop --fill.")
        shard_build = rest
        print(f"{PROG}: --fill: building {len(rest)} shard(s) not imported: "
              f"{', '.join(rest)}")

    if getattr(args, "export_shards", False):
        shard_export += [s for s in shard_build if s not in shard_export]
    shard_mode = bool(shard_import or shard_build or shard_export)
    deploy = bool(getattr(args, "deploy", False) or getattr(args, "fill", False))

    if shard_mode:
        pipeline = backend.shard_tasks(
            tasks, shard_import, shard_build, shard_export, child_env, args,
        )
        if pipeline is None:
            _fail("this backend does not support group-based sharding "
                  "(--import-shard / --build-shard / --export-shard)")
        tasks = backend.leading_tasks(components, child_env) + pipeline
        # --deploy (or its implied form via --fill) re-enables the trailing
        # whole-tree dist + install steps so a shard run also assembles the
        # imported + freshly-built shards into the combined dist tree and the
        # final install dir (otherwise a shard run only produces/pushes
        # artifacts).
        if deploy:
            tasks += backend.trailing_tasks(components, child_env)
    else:
        # Leading whole-build pseudo-tasks (e.g. TheRock's `therock/prereq`,
        # which builds the cmake/ninja toolchain) run before every per-component
        # task so a full build sets up its prerequisites first, with output
        # captured to a log.
        tasks = backend.leading_tasks(components, child_env) + tasks

        # Whole-build pseudo-tasks (e.g. TheRock's combined dist + final
        # install) are appended after every per-component task so they run last
        # on a full build and can be selected by name (e.g. 'therock/install').
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
            state = "done" if task_is_done(stamp_dir, task, built) \
                else task_state(stamp_dir, task)
            mark = render_mark(state)
            suffix = "  [pinned]" if task.comp in pinned else ""
            print(f"[{i:0{width}d}] [{mark}] {task.name}{suffix}")
        return 0

    # Bare `continue`: resume from the first task that is not yet done. "Done"
    # honors both the orchestrator stamp and the backend's already-built set
    # (the stage dir), matching the `list` tick-box -- so a component staged
    # outside the orchestrator is skipped rather than rebuilt.
    if args.selectors == ["continue"]:
        built = backend.built_components(child_env)
        resume = next(
            (i for i, t in enumerate(tasks)
             if not task_is_done(stamp_dir, t, built)),
            None,
        )
        if resume is None:
            print(f"{PROG}: all tasks already complete")
            return 0
        indices = list(range(resume, len(tasks)))
    else:
        indices = select_tasks(tasks, args.selectors)

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
    # subset build does not cascade rebuilds into dependents). Skipped in shard
    # mode: the shard pipeline pins + reconfigures *after* its imports (the
    # generic auto-pin here would run too early, before anything is staged).
    if not shard_mode:
        backend.prepare_run({tasks[i].comp for i in indices}, child_env, args)

    return run_tasks(
        backend, tasks, indices, child_env, log_dir, args.dry_run,
        build_type_global=build_type_global,
        build_type_per_comp=build_type_per_comp,
        log_base=build_dir, stamp_dir=stamp_dir,
    )
