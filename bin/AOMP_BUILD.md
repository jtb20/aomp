# `aomp_build.py` — AOMP build orchestrator

`aomp_build.py` is a unified, introspectable driver for building AOMP. It pulls
the individual taskified component build scripts (`build_<name>.sh`) into a
single workflow: it resolves which components to build from a config file,
orders them by their dependencies, breaks each component into fine-grained
tasks, and runs those tasks (all of them, a numbered range, a glob, or from a
chosen point onward) with per-task logging.

It is the successor workflow to running `build_aomp.sh` directly: the same
components, the same order, but introspectable, incremental, and scriptable.

- Orchestrator: [`bin/aomp_build.py`](aomp_build.py)
- Default config: [`bin/configs/aomp.cudf`](configs/aomp.cudf)
- Per-component scripts: `bin/build_<name>.sh` and `bin/rocmlibs/build_<name>.sh`

---

## Table of contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Command-line reference](#command-line-reference)
- [Selectors](#selectors)
- [Build variants](#build-variants)
- [Components and features](#components-and-features)
- [The version manifest](#the-version-manifest)
- [Sharding](#sharding)
- [TheRock backend](#therock-backend)
- [Logs](#logs)
- [Sharing sources between AOMP and TheRock](#sharing-sources-between-aomp-and-therock)
- [User workflows](#user-workflows)
- [Internals](#internals)
  - [The taskified component interface](#the-taskified-component-interface)
  - [The CUDF config format](#the-cudf-config-format)
  - [Resolution pipeline](#resolution-pipeline)
  - [Task elaboration](#task-elaboration)
  - [Execution](#execution)
  - [Environment discovery and child environment](#environment-discovery-and-child-environment)
- [Troubleshooting](#troubleshooting)
- [Extending](#extending)

---

## Requirements

- Python 3 (standard library only; no third-party packages).
- `bash` and the usual AOMP build prerequisites (the orchestrator simply
  invokes the existing `build_<name>.sh` scripts).
- A working AOMP environment as set up by `aomp_common_vars` (repos cloned
  under `$AOMP_REPOS`, etc.). `aomp_build.py` itself only reads a few
  variables (`BUILD_DIR`, `AOMP_REPOS`, `AOMP_REPO_NAME`); each build script
  sources `aomp_common_vars` on its own.

`aomp_build.py` does not need to be run from any particular directory — it
locates the build scripts relative to its own location in `bin/`.

---

## Quick start

```bash
cd $AOMP_REPOS/aomp/bin

# See the components that would be built (resolved + dependency-ordered).
./aomp_build.py --components

# See every task that would run, numbered.
./aomp_build.py list

# Dry-run everything (prints commands + log paths, runs nothing).
./aomp_build.py -n

# Build everything.
./aomp_build.py

# Build just one component's tasks.
./aomp_build.py 'comgr/*'

# Resume from a component after fixing a failure (trailing 'continue').
./aomp_build.py comgr/default/cmake continue
```

---

## Concepts

| Term | Meaning |
|------|---------|
| **Component** | A buildable unit with a `build_<name>.sh` script (e.g. `project`, `comgr`, `flang`, `rocBLAS`). |
| **Feature** | A named alias expanding to a set of components (e.g. `flang`, `rocmlibs`), used by `--add`/`--remove`. |
| **Config / variant** | A build configuration a component advertises (e.g. `default`, `asan`, `perf`, `debug`, `*-devicertl`). |
| **Task** | A single step of a component build: `precheck`, `patch`, `clean`, `cmake`, `build`, `install`, `postinstall`, `unpatch`. A `clean` task wipes that component's build dir and is listed right before its `cmake`. |
| **Request** | The default set of components to build, declared in the config's `request:` stanza. |
| **Selector** | A positional argument that picks which elaborated tasks to run. |

The pipeline, end to end:

```
parse argv ─▶ load CUDF config ─▶ expand features; apply --add/--remove
           ─▶ dependency closure + topological sort
           ─▶ elaborate tasks (query each build_<name>.sh list, filter by variant)
           ─▶ select tasks (selector grammar)
           ─▶ list, or run with per-task logs
```

---

## Command-line reference

```
aomp_build.py [options] [selector ...]
```

### Actions / output

| Option | Description |
|--------|-------------|
| `list` (selector) | Print the numbered task list and exit (`[NNN] [✓] component/stage`; the tick marks completed tasks, see [Completion stamps](#completion-stamps)). Trailing selectors preview a focused build: `list amd-llvm` marks every other already-built component `[pinned]` (TheRock only, see [Incremental focus](#incremental-focus-auto-pin-out-of-scope-components)). |
| `list-features` (selector) | Print the backend's configurable features and exit (TheRock only: the `THEROCK_ENABLE_*` flags, with ✓/✗ enabled state). Enable one with `--add <name> --reconfigure`. |
| `list-shards` (selector) | Print the backend's shard catalog and exit (TheRock only: the `BUILD_TOPOLOGY.toml` artifact groups, with their subprojects, dependency groups, and artifact counts). Drive one with `--import-shard` / `--build-shard` / `--export-shard(s)`. See [Sharding](#sharding). |
| `list-configs` (selector) | Print the backend's source configs and exit (TheRock only: the branches each `-c/--config` selects, with the default marked). See [Source config](#source-config-which-sources-to-build-via--c--config). |
| `-a`, `--all` | (TheRock only) Elaborate *every* advertised per-subproject action (`expunge/configure/build/stage/dist`) instead of the default `configure/build/stage`. Use with `list` to see the full capability set, or with a selector to run a normally-hidden action (e.g. `-a amd-llvm/expunge`). See [TheRock backend](#therock-backend). |
| `--components` | Print the resolved, dependency-ordered component list and exit. |
| `-n`, `--dry-run` | Show what would run (command + log path per task) without executing. |
| `--export-manifest [FILE]` | Write a git fingerprint manifest and exit. Default path: `<BUILD_DIR>/manifests/<config>-manifest.json`. |
| `--import-manifest FILE` | Check out recorded git SHAs before building (refuses if any repo is dirty; `extras` is kept at HEAD). |

### Component selection

| Option | Description |
|--------|-------------|
| `-c`, `--config` | **AOMP backend:** CUDF config file (default `bin/configs/aomp.cudf`). **TheRock backend:** source-config name selecting which TheRock branches to build (default `amd-staging`; see [Source config](#source-config-which-sources-to-build-via--c--config) and `list-configs`). |
| `--add NAMES` | Add component(s) or feature(s). Comma-separated and/or repeatable. |
| `--remove NAMES` | Remove component(s) or feature(s) (cascades to dependents). Comma-separated and/or repeatable. |
| `--variant SPEC` | Variant filter: `cfg` (global) or `comp=cfg` (per-component). Comma-separated and/or repeatable. `default` is always built when offered (so `--variant debug` = default+debug); components offering neither are skipped (so `--variant default` skips the runtimes). See [Build variants](#build-variants). |
| `-C`, `--clean` | Prepend an `install/clean` task that wipes the install directory — the versioned symlink *target* (`AOMP_INSTALL_DIR`), then the symlink itself — before building. Per-component `clean` tasks (build dirs) are always listed and run like any other task. |

### Directory layout (exported to child build scripts)

| Option | Environment variable | Notes |
|--------|----------------------|-------|
| `-s`, `--source DIR` | `AOMP_REPOS` | Source/repo root holding the cloned component repos. The build dir (`BUILD_AOMP`) defaults to it unless `-b` is given. Default: `$HOME/git/aomp<version>`. |
| `-i`, `--install DIR` | `AOMP` | Install root. The versioned install dir `AOMP_<version>` (`AOMP_INSTALL_DIR`) derives from it. Default: `$HOME/rocm/aomp`. |
| `-b`, `--build DIR` | `BUILD_AOMP` | Where cmake/make run and object files go (also `BUILD_DIR`, used for the default log/manifest locations). Default: the repo dir (`AOMP_REPOS`). |
| `-p`, `--prereq DIR` | `AOMP_SUPP` | Prerequisite/supplemental root. Its build (`AOMP_SUPP_BUILD`), install (`AOMP_SUPP_INSTALL`), and the prereq `cmake` all derive from it. Default: `$HOME/local`. |

Each is expanded to an absolute path (with `~`) before being handed to the
child scripts, so they resolve identically regardless of working directory.

### Build environment knobs (exported to child build scripts)

| Option | Environment variable | Notes |
|--------|----------------------|-------|
| `-j`, `--jobs N` | `AOMP_JOB_THREADS` | Parallel build threads. |
| `--ninja` / `--no-ninja` | `AOMP_USE_NINJA=1` / `0` | Use the Ninja generator. |
| `--ccache` / `--no-ccache` | `AOMP_USE_CCACHE=1` / `0` | Use ccache. |
| `--gfx LIST` | `GFXLIST` | GPU target list; comma- or space-separated (e.g. `gfx90a,gfx942`), normalized to the space-separated `GFXLIST` form. |
| `--build-type SPEC` | `BUILD_TYPE` | CMake build type; global or per-component (see below). |
| `--sudo` | `SUDO=yes` | Install with sudo. |

`--build-type` uses the same scoped grammar as `--variant`: a bare value applies
globally, while `comp=type` sets the build type for one component only. It is
comma-separated and/or repeatable, and per-component values override the global
one. The type is applied to each component's tasks via the `BUILD_TYPE`
environment variable at execution time (it does not affect other components):

```bash
./aomp_build.py --build-type RelWithDebInfo            # global
./aomp_build.py --build-type project=Debug,comgr=Debug # per-component
./aomp_build.py --build-type Release,project=Debug     # global Release, project Debug
```

For the **TheRock** backend, build types are not an execution-time env var:
TheRock gates them at *configure* time via cmake cache variables. The
orchestrator therefore translates `--build-type` into `-D` flags on the
configure (`SROCK_CMAKE_EXTRA`) -- a bare value becomes `-DCMAKE_BUILD_TYPE=<T>`
and `comp=type` becomes `-D<comp>_BUILD_TYPE=<T>` (the component name is the
subproject name, e.g. `-Damd-llvm_BUILD_TYPE=Debug`).

The desired state is compared against the **current `CMakeCache.txt`**, so it is
idempotent: if every value already matches the configured cache it is a no-op,
and you can leave `--build-type` on the command line across incremental builds
with no `--reconfigure`. An **unset** build type defaults to `Release` rather
than retaining whatever is cached, so *dropping* a previously-set `--build-type`
reverts that scope to the default -- which is itself a configuration change.
Only a value that actually *changes* the configuration needs `--reconfigure`
(build types take hold at configure time); requesting a change without it is a
hard error naming the differing settings:

```bash
# Set a per-component type (reconfigures to apply).
therock_build.py --reconfigure --build-type ROCR-Runtime=Debug ROCR-Runtime/build

# Keep it on the command line -> no-op, no --reconfigure needed.
therock_build.py --build-type ROCR-Runtime=Debug ROCR-Runtime/build

# Drop it -> ROCR-Runtime reverts to the Release default, which is a change:
#   error unless --reconfigure is given.
therock_build.py --reconfigure ROCR-Runtime/build
```

Scopes that are neither requested nor already present in the cache are left
unmanaged (a fresh tree keeps TheRock's own default). A `comp=type` whose
component is not a known subproject is warned about (the `-D<comp>_BUILD_TYPE`
flag would have no effect), but is not fatal.

### Environment isolation

Child build scripts run in an **isolated environment built from scratch** by the
orchestrator, not a copy of your shell's environment. This keeps *what* gets
built under the orchestrator's control and prevents stray exports (a Homebrew
`pkg-config` ahead of the system one, a leftover `CC`/`LD_LIBRARY_PATH`, a stale
`PKG_CONFIG_PATH`, etc.) from silently changing the build.

What the child environment contains:

- A controlled `PATH` of standard system locations only:
  `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`.
- A curated pass-through of identity/locale/terminal variables that do not
  affect *what* is compiled: `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`,
  `LANGUAGE`, `LC_*`, `TZ`, `TMPDIR`, `DISPLAY`, `XAUTHORITY`, and the `SSH_*`
  variables (so git-over-ssh and your `~/.gitconfig` keep working).
- The build knobs set by the flags above (`AOMP_JOB_THREADS`, `GFXLIST`, ...).

Everything else (`CC`, `CXX`, `LD_LIBRARY_PATH`, `PKG_CONFIG_*`, `AOMP_*`,
`ROCM_*`, ...) is **dropped** unless the orchestrator sets it from a flag or you
opt in explicitly:

| Option | Description |
|--------|-------------|
| `--inherit-path` | Use your shell's `PATH` for child scripts instead of the controlled default. |
| `--pass-env VARS` | Leak named variable(s) into the child environment. Comma-separated and/or repeatable, e.g. `--pass-env CC,CXX,LD_LIBRARY_PATH`. |

```bash
./aomp_build.py --inherit-path                     # let my PATH through
./aomp_build.py --pass-env LD_LIBRARY_PATH         # leak one var
./aomp_build.py --inherit-path --pass-env CC,CXX   # both
```

Because the classic `AOMP_*` overrides are no longer inherited automatically,
drive non-flag knobs either with `--pass-env` or by exporting them and letting a
flag carry them. (`AOMP_JOB_THREADS=32 ./aomp_build.py` no longer leaks through;
use `-j 32` instead.)

### Logging

| Option | Description |
|--------|-------------|
| `--log-dir DIR` | Directory for per-task logs. Default: `<BUILD_DIR>/aomp_build_logs`. |

---

## Selectors

Selectors are positional arguments that decide which of the elaborated tasks to
run. The grammar mirrors `amd-build`:

| Selector | Meaning |
|----------|---------|
| *(none)* | Run all elaborated tasks. |
| `list` | Print the numbered task list and exit (does not run anything). Trailing selectors preview a focused build, marking out-of-scope built components `[pinned]` (TheRock). |
| `list-features` | Print the backend's configurable features and exit (TheRock: the `THEROCK_ENABLE_*` flags). Does not run anything. |
| `list-shards` | Print the backend's shard catalog and exit (TheRock: the `BUILD_TOPOLOGY.toml` artifact groups). Does not run anything. |
| `list-configs` | Print the backend's source configs and exit (TheRock: the branches each `-c/--config` selects). Does not run anything. |
| `N` | Run task number `N` (1-based, as shown by `list`). |
| `N--M` | Run the inclusive range of tasks `N` through `M`. |
| `comp/variant/stage` | Glob/substring match on task names; supports `{a,b}` brace expansion. |
| `continue` | On its own, resume from the first task that is not marked complete (see [Completion stamps](#completion-stamps)) through to the end. |
| `... X continue` | Trailing `continue` turns the preceding selector `X` into a "from `X` to the end" anchor. Any earlier selectors are selected normally. |

Multiple selectors can be combined; the union of matched tasks runs in task
order. Task names take the form `component/variant/stage` for build tasks
(e.g. `comgr/default/cmake`, `llvm_runtimes_standalone/asan/build`). The variant
segment is dropped (giving `component/stage`) for:

- the config-less init/fini tasks (`precheck`, `patch`, `unpatch` — e.g.
  `comgr/patch`), and
- components whose only advertised config is `default` (e.g. `prereq/build`,
  `rocminfo/cmake`).

Putting the variant in the middle makes it easy to glob a whole variant across
components, e.g. `*/asan/*`.

`continue` is a **trailing** keyword: it must be the last argument, and it
applies to the selector immediately before it. `X` may be a task number or a
task name; if it matches several tasks (e.g. a component name), continuation
starts from its first task.

Examples:

```bash
./aomp_build.py 5                      # just task 5
./aomp_build.py 5--12                  # tasks 5 through 12
./aomp_build.py 30 continue            # from task 30 to the end
./aomp_build.py rocr/default/cmake continue   # from rocr's cmake task onward
./aomp_build.py comgr continue         # from comgr's first task to the end
./aomp_build.py project/default/build comgr continue  # project's build, then comgr onward
./aomp_build.py 'comgr/*'              # every comgr task
./aomp_build.py '*/install'            # every install task (any component/variant)
./aomp_build.py '*/asan/*'             # every asan-variant task
./aomp_build.py '{comgr,rocr}/*/build' # comgr build + rocr build
```

> Tip: quote selectors containing `*`, `?`, or `{}` so your shell doesn't try
> to expand them first.

---

## Build variants

Each component advertises one or more *configs* (variants) through its
`list_configs` command. There are two common styles:

- **Style A** (most components): always offer a plain `default` plus opt-in
  variants such as `asan` and `debug`.
- **Style B** (e.g. `llvm_runtimes_standalone`): derive their config set from
  the environment (`AOMP_BUILD_SANITIZER`, `AOMP_BUILD_PERF`,
  `AOMP_BUILD_DEBUG`) and have **no** plain `default` — their default runtime
  libraries are produced by the `project`/LLVM build itself. The advertised set
  is the extra instrumented variants (e.g. `asan`, `perf`, `perf+asan`,
  `debug`, plus `*-devicertl` device-runtime passes).

The `default` config is the baseline an installable build needs, so it is
**always built when a component offers it**.

Selection policy:

- **No `--variant`:** build **every** advertised config for every component
  (the full build — `default` plus all variants such as `asan`/`debug`/`perf`,
  including the runtimes matrix).
- **`--variant` (any explicit value):** build `default` (where offered) **plus**
  the requested variants each component advertises. So `--variant debug` means
  `default,debug` everywhere: default-only components (e.g. `comgr`) still build
  their `default`, components offering `debug` add it, and components offering
  no `default` (e.g. the runtimes) build just their `debug`. A component that
  offers **neither** `default` **nor** any requested variant is **skipped
  entirely** (no tasks, not even `precheck`/`patch`). In particular,
  `--variant default` builds only the `default` config of each component and
  therefore **skips `llvm_runtimes_standalone`** (its default runtime libraries
  are produced by the `project`/LLVM build itself).

Forms (all combinable, comma-separated and/or repeated):

| Form | Meaning |
|------|---------|
| `--variant cfg` | Apply `cfg` globally (every component that advertises it). |
| `--variant cfg1,cfg2` | Multiple global variants (e.g. `debug,asan`). |
| `--variant comp=cfg` | Override one component only. |
| `--variant comp1=cfg1,comp2=cfg2` | Per-component overrides in one value. |

When only per-component overrides are given (no global value), components not
named build their normal default.

Examples:

```bash
./aomp_build.py --variant default          # default everywhere; skips the runtimes
./aomp_build.py --variant debug,asan        # default + debug + asan (where offered)
./aomp_build.py --variant asan list         # default + asan
./aomp_build.py --variant llvm_runtimes_standalone=perf list
./aomp_build.py --variant rocr=debug,llvm_runtimes_standalone=asan
```

---

## Components and features

The default component set is the `request:` stanza of the config — the
standalone x86_64 AOMP build from `build_aomp.sh` (including `hipfort` and the
`rocdbgapi`/`rocgdb` debugger), **omitting the deprecated classic Flang stack**
(`llvm-classic`, `flang-classic`, `pgmath`, `flang`, `flang_runtime`) and the
ROCm math libraries. Re-enable classic Flang with `--add flang` and the math
libraries with `--add rocmlibs`.

Adjust it with `--add` / `--remove`, which accept **component** names or
**feature** names. Each option is repeatable and also accepts a comma-separated
list, so `--add rocmlibs,debug` and `--add rocmlibs --add debug` are
equivalent. Features defined in the default config:

| Feature | Expands to |
|---------|-----------|
| `flang` | `llvm-classic`, `flang-classic`, `pgmath`, `flang`, `flang_runtime` (deprecated classic Flang; not in the default set) |
| `hip` | `hipcc`, `hipamd`, `hipify` |
| `debug` | `rocdbgapi`, `rocgdb` (the ROCm debugger; in the default set — drop with `--remove debug`) |
| `profiler` | `rocprofiler-register`, `rocprofiler-sdk` |
| `rocmlibs` | `rocm-cmake`, `rocBLAS`, `rocPRIM`, `rocSPARSE`, `rocSOLVER`, `hipBLAS-common`, `hipBLAS`, `rocRAND`, `hipRAND`, `rccl`, `half`, `hipSOLVER` |

Resolution rules:

- `--add` pulls in the named component(s)/feature(s) **and their transitive
  dependencies**.
- `--remove` drops the named component(s)/feature(s) **and anything that
  depends on them** (the removal cascades).

```bash
./aomp_build.py --add rocmlibs --components       # default set + ROCm math libraries
./aomp_build.py --remove flang --components       # drop the whole Flang group
./aomp_build.py --remove debug --components         # drop the debugger (rocgdb + rocdbgapi)
./aomp_build.py --add rocmlibs --remove debug --components  # combine adds and removes
```

---

## The version manifest

The manifest captures the exact git state of every component's source repo, so
a build can be reproduced later.

Export:

```bash
./aomp_build.py --export-manifest                       # default path
./aomp_build.py --add rocmlibs --export-manifest m.json # explicit path + set
```

The JSON records, per component that has a git source:

```json
{
  "generated": "2026-06-17T10:00:00",
  "config": "aomp",
  "order": ["prereq", "project", "..."],
  "components": {
    "project": {
      "sha": "…",
      "repo": "https://github.com/…",
      "branch": "amd-staging",
      "dirty": false,
      "subdir": "llvm"
    },
    "comgr": {
      "sha": "…",
      "repo": "https://github.com/…",
      "branch": "amd-staging",
      "dirty": false,
      "subdir": "amd/comgr"
    }
  },
  "externals": {
    "SPIRV-LLVM-Translator": {
      "sha": "…",
      "repo": "https://github.com/…",
      "branch": "amd-staging-npi",
      "dirty": false
    }
  }
}
```

- The git fingerprint is taken from the repository that **contains** a
  component's source, so components that build from a subdirectory of a shared
  checkout are all recorded. The LLVM `project`, `comgr`, `hipcc` and
  `llvm_runtimes_standalone` all live in the single `llvm-project` repo (same
  `sha`/`branch`), distinguished by their `subdir` (`llvm`, `amd/comgr`,
  `amd/hipcc`, `runtimes`).
- `externals` records repos that are pulled into a component build but are not
  standalone components (e.g. `SPIRV-LLVM-Translator`, consumed by the LLVM
  build via `LLVM_EXTERNAL_PROJECTS`). These are tightly coupled to the
  LLVM/comgr toolchain and a frequent source of build breakage, so their exact
  versions are worth recording alongside LLVM and comgr.

Import (a pre-step before building):

```bash
./aomp_build.py --import-manifest m.json [selectors...]
```

- For each component in the manifest that is also in the resolved set (plus the
  recorded `externals`), the recorded SHA is checked out (`git checkout <sha>`).
  Components that share a repository (the `llvm-project` family) are deduped, so
  the shared checkout is moved once.
- **`extras` stays at HEAD.** The `extras` component is the AOMP build-scripts
  repo (this tree); it is never rolled back, since the scripts are meant to
  build arbitrary AOMP/ROCm versions and pinning them would change the build
  logic mid-flight.
- **Safety:** if any target repo has local modifications, the import is
  **refused** entirely (nothing is checked out) and the dirty repos are listed.
  `extras` being dirty does not block the import.
- Components without a git source (e.g. `prereq`) are skipped.

Manifests live under `<BUILD_DIR>/manifests/` by default
(`BUILD_DIR` = `$BUILD_AOMP`, normally `$AOMP_REPOS`).

---

## Sharding

Sharding splits a large build across machines (or separate invocations) along
**TheRock's artifact groups** — the `[artifact_groups]` of `BUILD_TOPOLOGY.toml`
(e.g. `third-party-sysdeps`, `compiler`, `core-runtime`, `math-libs`). A shard
*is* an artifact group. Groups are the finest unit TheRock tracks dependencies
between, so this lets you, for example, import the slow-moving
`third-party-sysdeps` and `third-party-libs` once and rebuild only `compiler`
day to day, instead of treating the whole `compiler-runtime` stage as one lump.

Sharding is therefore a **TheRock-backend feature** (the aomp backend has no
group/artifact model). List the available shards — in dependency (build) order
— with `list-shards`:

```bash
therock_build.py list-shards
# compiler  - AMD LLVM toolchain and compiler infrastructure
#   builds    : amd-llvm, amd-comgr, hipcc  (3)
#   imports   : third-party-sysdeps
#   sources   : compilers
#   artifacts : 2 produced, 9 inbound
# core-runtime  - Core runtime (ROCR-Runtime, rocminfo)
#   builds    : ROCR-Runtime, rocminfo  (2)
#   imports   : third-party-sysdeps, base
#   ...
```

`builds` reflects the *current configure* (only enabled subprojects appear); the
other fields are intrinsic to the topology.

Each shard verb is its own option, all taking a comma-separated group list:

| Option | Effect |
| --- | --- |
| `--build-shard LIST` | Build these groups' subprojects (and their `artifact-group-<g>` targets). Sources are fetched for **only** these groups' source sets (`fetch_sources.py --source-sets`). |
| `--import-shard LIST` | Before building, import these (producer) groups' artifacts into the build tree as prebuilt, via `buildctl.py bootstrap`. |
| `--export-shard LIST` | After building, copy these (producer) groups' artifacts to the store (the `shard_artifacts.py export-local` helper). |
| `--export-shards` | Sugar for "export every group named by `--build-shard`" (so the build set need not be repeated). |
| `--shard-store DIR` | Local artifact store shared between import/export (TheRock's `THEROCK_LOCAL_STAGING_DIR`). Default: `<BUILD_DIR>/shard-artifacts`. |
| `--shard-run-id LABEL` | Run-id namespace under the store for push/import (default `local`). |
| `--shard-families LIST` | Extra target families (e.g. `gfx94X`) to import for per-arch artifacts, in addition to `generic`. |
| `--deploy` | Also assemble the imported + built shards into the combined dist tree and final install dir (re-enables the trailing `therock/dist` + `therock/install` steps a shard run otherwise skips). No effect outside shard mode. |

When any shard option is given the run becomes the
**import → build → export** pipeline for the named groups (the leading
`therock/prereq` toolchain step still runs; the trailing whole-tree
dist/install is skipped — a shard produces and pushes artifacts, it does not
assemble the full SDK). Pass `--deploy` to re-enable that assembly so the
imported and freshly-built shards land in the dist/install tree on this
machine.

```bash
# Machine A: build the compiler group and publish its artifacts to the store.
# (third-party-sysdeps is imported; see list-shards for a group's deps.)
therock_build.py --import-shard third-party-sysdeps --build-shard compiler \
    --export-shards --shard-store /shared/rocm-artifacts

# Machine B: pull the compiler artifacts, then build math-libs against them
# and publish math-libs too. Only math-libs' sources are fetched here.
therock_build.py --import-shard compiler --build-shard math-libs \
    --export-shards --shard-store /shared/rocm-artifacts --shard-families gfx94X

# Same, but also assemble the imported compiler + built math-libs into the
# local dist/install tree (not just artifacts) with --deploy.
therock_build.py --import-shard compiler --build-shard math-libs \
    --deploy --shard-store /shared/rocm-artifacts --shard-families gfx94X

# Export an already-built group on its own (no rebuild).
therock_build.py --export-shard math-libs --shard-store /shared/rocm-artifacts
```

`--import-shard`/`--export-shard` name **producer** groups: import pulls a
group's *produced* artifacts; export pushes them. Import resolves each producer
group to its artifact names from the topology and hands a filtered view of the
store to `buildctl.py bootstrap`.

Because `bootstrap` only drops `.prebuilt` markers and stages files (TheRock
honors them at *configure* time), the pipeline inserts a `therock/shard-pin`
step after the imports — `buildctl.py enable <build-group subprojects>
--force-reconfigure` — which reconfigures so the imported subprojects are
treated as prebuilt and the build groups' subprojects build against them rather
than rebuilding the imports. (In shard mode this replaces the normal
[auto-pin](#incremental-focus-auto-pin-out-of-scope-components), which would
otherwise run before anything is staged.)

Mapping subprojects to groups requires an **artifact → subproject** map
(`artifact_map.json`), emitted by the bundled introspection alongside
`subproject_map.json` once TheRock is (re)configured with introspection. On a
checkout that has not yet been reconfigured, `list-shards` shows groups with no
`builds` subprojects; run a build (or `--reconfigure`) to populate it.

---

## TheRock backend

The orchestrator is split into a backend-agnostic core (the `orchestrator`
package) and pluggable backends. The default **aomp** backend drives the
per-component `build_<name>.sh` scripts described throughout this document. The
**therock** backend instead drives [TheRock](https://github.com/ROCm/TheRock)'s
single CMake super-build, exposing the *same* workflow (resolve, order,
elaborate, list/run/continue, log, stamp, manifest) plus group-based sharding.

Select it with `--backend therock`, or use the dedicated entry point
[`bin/therock_build.py`](therock_build.py) (identical, but defaulting to the
TheRock backend):

```bash
therock_build.py list                 # numbered subproject+action task list
therock_build.py -n                   # dry-run the whole build
therock_build.py 'amd-llvm/*'         # just the compiler's tasks
therock_build.py list-shards          # list artifact groups (shards)
therock_build.py --build-shard math-libs --import-shard compiler --export-shards
therock_build.py list-configs         # list source configs (branch sets)
therock_build.py -c develop --reconfigure   # build native upstream TheRock
```

### Where the build graph comes from

TheRock can emit an introspection file describing its build graph when it is
configured with `-DTHEROCK_INTROSPECTION=ON`:

```
<TheRock>/build/subproject_map.json
```

This support lives in TheRock PR #1234 and is not yet in upstream `ROCm/TheRock`
(which is what `setup_srock.sh` clones). The orchestrator is self-sufficient: it
bundles the introspection module (`srock-bin/therock_subproject_introspection.cmake`)
and, during `--reconfigure`, injects it into the checkout — copying it into
`<TheRock>/cmake/` and appending a guarded `include()` + invocation at the end of
the top-level `CMakeLists.txt` (idempotent, line-number-independent). The
required internals (`therock_get_all_targets` and the `THEROCK_*` target
properties) already exist in upstream TheRock, so this works on a stock clone.

For each subproject it records the source/build directories, the build-time and
runtime dependencies, the build pool and compiler toolchain, and the available
ninja action targets. The backend reads this file as its component graph:

- **Components** are TheRock subprojects.
- **Dependencies** that drive ordering are the **build-time** deps
  (`build_deps`). Runtime deps (`runtime_deps`) are recorded as metadata but do
  *not* affect build order (they describe what a subproject needs at run time,
  not what must be built before it).
- **Tasks** are `subproject/<action>` for the forward pipeline actions
  `configure`, `build`, `stage`, each running as:

  ```
  ninja -C <TheRock>/build <subproject>+<action>
  ```

  These actions depend only on a subproject's *genuine build prerequisites* (its
  own configure→build, its build-deps' configure, and its compiler toolchain),
  so running a subset respects real ordering without doing unrelated work.

  Two actions are deliberately **excluded** from the per-subproject task list:

  - `dist` — in TheRock a subproject's `+dist` target is wired into whole
    *distribution* assembly (artifacts/distributions add themselves as its
    dependencies), so `ninja <sub>+dist` builds unrelated subprojects across the
    runtime-dependency graph (e.g. dist-ing a base sysdep like `bzip2` drags in
    llvm). That is a whole-tree packaging step, not a per-component one. The
    per-subproject `+stage` already populates the combined local dist directory
    with this + transitive stage installs (a file copy, not extra builds), so an
    in-order build still produces a usable staged tree. The combined dist tree
    and the final install are produced by the **whole-tree pseudo-tasks** below.
  - `expunge` — destructive clean; would wipe a subproject mid-build. Clean an
    install with `-C`/`--clean`, or run `ninja <subproject>+expunge` by hand.

  **`-a`/`--all`** overrides this exclusion and elaborates *every* advertised
  per-subproject action in lifecycle order
  (`expunge → configure → build → stage → dist`). This is meant for `list`
  (to see the full capability set) and for targeted selection while untangling a
  build — e.g. `therock_build.py -a amd-llvm/expunge` to wipe just that
  subproject, or `-a amd-llvm/dist`. The `dist`/`expunge` caveats above still
  apply (per-subproject `dist` triggers whole-tree assembly; `expunge` is
  destructive), so a *bare* `--all` run with no selectors would clean and dist
  every component — prefer it with `list` or an explicit selector.

TheRock subprojects are config-less (a single configuration is baked in at
configure time), so tasks use the short `subproject/stage` names and the
`--variant` filter does not apply.

### Default request set (match the native build)

The introspection map declares **every** subproject TheRock knows about,
including the vendored third-party / system libraries (the `therock-*`
projects: boost, eigen, googletest, fmt, grpc, …). A native
`cmake --build build` does **not** build all of them — TheRock marks each
subproject `EXCLUDE_FROM_ALL` and pulls only what it needs through its
`therock-priority-build` / distribution targets — so requesting the literal
full map would over-build vendored libs that nothing in the build actually
needs.

To mirror the native build, the default request is the **real ROCm components
plus their full build- and runtime-dependency closure**:

- Seed = every non-`therock-*` subproject (the real ROCm components: amd-llvm,
  ROCR-Runtime, rocm-core, hip-clr, rocgdb, …).
- Closure adds any subproject they need via `build_deps` *or* `runtime_deps`
  (so genuinely-required vendored libs like `therock-simde` and
  `therock-msgpack-cxx` are pulled in), transitively.
- Vendored libs that no built component depends on (boost, eigen, googletest,
  fmt, fftw, grpc, …) are **left out** of the default build.

On the current minimal config this trims the default request from 36 declared
subprojects to 19 (17 real components + the 2 vendored libs they need). The
excluded libs stay **declared**, so you can opt any of them in explicitly:

- `--add <name>` (e.g. `--add therock-boost`) — adds that lib and its deps.
- `--add thirdparty` — a convenience feature listing **all** `therock-*`
  vendored libs, restoring the build-everything behavior.

Anything a *distribution* still requires is assembled by the whole-tree
`therock/dist` task below regardless of the per-subproject request, so the
final SDK is unchanged.

### Selecting build features (`THEROCK_ENABLE_*`)

Which subprojects even *exist* in the introspection map is decided by TheRock's
own configure-time **feature** flags — the `THEROCK_ENABLE_*` cache variables
(coarse group options like `THEROCK_ENABLE_ML_LIBS`, and per-artifact features
like `THEROCK_ENABLE_HIPDNN`). A component gated off by a disabled feature does
not appear in `subproject_map.json` at all, so it cannot be selected with `--add
<subproject>` until the feature that produces it is turned on.

The backend exposes these features as `--add` tokens. During introspection it
emits a companion `feature_map.json` next to `subproject_map.json` cataloging
every `THEROCK_ENABLE_*` flag (its enabled state, description, and `requires`
list), and surfaces them as selectable tokens:

```
therock_build.py list-features          # list every feature + on/off state
```

`list-features` prints one row per feature, a green check (✓) for enabled and a
red cross (✗) for disabled, with its `requires` dependencies and description.

To turn a feature on, name it with `--add` (case-insensitive; dashes and
underscores are interchangeable, so `hipdnn`, `HIPDNN`, and `ml-libs` all work)
**together with `--reconfigure`**:

```
therock_build.py --add hipdnn --reconfigure list   # enable HIPDNN, then list
```

`--reconfigure` is required because features are configure-time gates: the
backend appends `-DTHEROCK_ENABLE_<NAME>=ON` to the cmake invocation and lets
TheRock reconfigure, which is what makes the newly-enabled subprojects show up
in the regenerated map. Requesting a feature that is **not** already enabled
*without* `--reconfigure` is a hard error (a reconfigure would otherwise be
needed to honor it, and silently ignoring the request would be misleading):

```
therock_build.py --add hipdnn list
# error: requested feature(s) not enabled in the current TheRock
#        configuration: hipdnn.
#          Re-run with --reconfigure to apply them (...).
```

Requesting a feature that is already enabled is a no-op (no flag appended, no
reconfigure forced), so `--add <feature>` is safe to leave in a command line
across incremental builds.

### Leading pseudo-task (prereq toolchain)

A `therock/prereq` pseudo-task is prepended **before** every per-subproject task.
It runs `srock-bin/build_cmake.sh` to build the cmake/ninja prerequisite
toolchain, so that output is captured to a per-task log
(`<build>/aomp_build_logs/001-therock-prereq.log`) instead of spamming the
console mid-build. `build_cmake.sh` self-checks, so it's a cheap no-op once the
tools are built. It appears as task 1 in `list` and can be run on its own
(`therock_build.py therock/prereq`).

### Whole-tree pseudo-tasks (dist + install)

Because `dist`/install are whole-tree operations (not per-subproject), the
backend appends two pseudo-tasks **after** every per-subproject task — the
TheRock equivalents of amd-build's final "install all components":

- `therock/dist` → `ninja -C <build> therock-dist` — assembles the combined
  distribution tree under `<build>/dist` (e.g. `<build>/dist/rocm`, the complete
  ROCm SDK). Same step CI runs as `cmake --build build --target therock-dist`.
- `therock/install` → `ninja -C <build> install` — installs that tree to the
  final install dir (`CMAKE_INSTALL_PREFIX` = `$SROCK_INSTALL_DIR`), exactly as
  `srock-bin/build_srock.sh`'s `ninja install` does.

They appear last in `list` and run last on a full build (build everything, then
assemble + install). You can also run them on their own, e.g.
`therock_build.py therock/install`.

### Incremental focus (auto-pin out-of-scope components)

When you build only a **subset** of subprojects (e.g. iterating on `amd-llvm`),
the backend marks every *other* already-built component **prebuilt** before
running, by wrapping TheRock's own `build_tools/buildctl.py`:

```
buildctl.py enable <working-set>     # working set buildable; everything else pinned
```

A `.prebuilt` marker makes CMake trust a component's existing `stage/` and skip
its configure/build, so neither the subset build nor a later `therock/install`
rebuilds dependents you are not working on (e.g. rebuilding `amd-llvm` won't drag
`rocgdb` along, even though it technically depends on the compiler). This is the
"work on one thing, don't cascade" workflow, integrated so you don't manage
markers by hand. Only components that have actually been staged can be pinned, so
anything not yet built stays buildable and is produced if a dependency needs it.

- It triggers only for a strict, non-empty subset of subprojects (a full build,
  or a selection of only the whole-tree pseudo-tasks, pins nothing).
- `--rdeps` rebuilds the **reverse-dependency closure** of the subset instead of
  pinning it: with `amd-llvm --rdeps`, every component that (transitively)
  depends on `amd-llvm` is pulled into the build and rebuilt, and only the
  remaining built components are pinned. The closure follows **both** build
  dependencies (`build_deps`, e.g. `ROCR-Runtime`) **and** runtime/link
  dependencies (`runtime_deps`, e.g. `rocgdb`, `amd-comgr`, `hipcc`, which link
  the compiler's libraries but list no build-dep on it) -- so changing the
  compiler propagates a rebuild to everything built against it. Forward
  dependencies the subset *needs* (e.g. `rocm-cmake`) are still left prebuilt
  either way.
- `--unpin-all` clears all markers (`buildctl.py enable` with no args) so every
  component builds again, then proceeds normally.
- `--no-auto-pin` leaves markers untouched for one run.

Note that `buildctl.py` reconfigures TheRock to pick up marker changes, so a
focused subset run does a (cheap) cmake reconfigure first.

#### Previewing pins with `list`

`list` can preview which components a focused build would pin. Trailing
selectors after `list` are treated as the focused set; every already-built
component **outside** that set is shown with a `[pinned]` suffix:

```
therock_build.py list amd-llvm
[01] [✓] rocm-cmake/configure   [pinned]
...
[07] [✓] amd-llvm/build
...
[12] [✓] hipBLAS/build          [pinned]
```

A component is reported `[pinned]` only when it has a valid (built) TheRock
`stage/` dir — exactly the components `buildctl.py` would mark prebuilt. The
checkbox reflects TheRock's build state too: a component with a valid stage dir
renders as done (`[✓]`) even if this orchestrator never ran its task, so a
`[pinned]` row is never blank. Bare `continue` uses the same combined notion of
"done" (orchestrator stamp **or** a valid stage dir), so it resumes past
components TheRock already staged rather than rebuilding them. `--rdeps` applies
to the preview as well (place flags before the
selectors, e.g. `therock_build.py --rdeps list amd-llvm`): the dependents then
show as buildable rather than `[pinned]`. A bare `list` (no trailing selectors)
previews a full build and pins nothing.

### Bootstrap

Generating `subproject_map.json` means a full TheRock cmake configure (which on
first use clones the repo and fetches submodule sources). This is delegated to
the existing `srock-bin` scripts (`srock_common_vars` / `setup_srock.sh`) rather
than reimplemented, so the orchestrator and the two-script srock workflow stay
consistent.

- If a `subproject_map.json` already exists, it is used as-is.
- `--reconfigure` forces a fresh configure (with `-DTHEROCK_INTROSPECTION=ON`)
  via `setup_srock.sh`, regenerating the map.
- A missing map without `--reconfigure` is a hard error with guidance — the
  heavy clone/fetch/configure is never triggered implicitly by a `list` or
  dry-run.

The build is controlled by the same `SROCK_*` conventions as the srock scripts:
`SROCK_REPOS` (set by `-s`), the install dir / symlink (`-i` → `SROCK_LINK`),
the supplemental tools dir (`-p` → `SROCK_SUPP`), and `GFXLIST` (`--gfx`).
`--therock-dir` overrides the TheRock checkout location.

#### Source config (which sources to build) via `-c/--config`

For TheRock, `-c/--config` selects a **source config**: *which TheRock sources a
build uses* — the git branches the srock scripts check out. This is orthogonal
to the build *scope* (`SROCK_CONFIG`, below), which selects *how much* to build.
The default is **`amd-staging`** (no option needed):

| `-c/--config` | TheRock branch | Compiler submodules | Meaning |
|---|---|---|---|
| *(none)* / `amd-staging` | `compiler/amd-staging` | `amd-staging` + srock patches | AMD staging compiler (srock default) |
| `develop` | `main` | native (no override/patches) | Native upstream TheRock |

```
therock_build.py list-configs                  # list source configs + branches
./aomp_build.py --backend therock --reconfigure              # amd-staging (default)
./aomp_build.py --backend therock -c develop --reconfigure   # native upstream
```

A source config names **branches only** — never a pinned SHA. TheRock records
each branch's submodule pins as gitlink SHAs in that branch's own tree, and
`fetch_sources.py` checks submodules out at exactly those recorded SHAs, so a
config always tracks whatever the branch currently points at (no manual pin
updates). Configs live in `srock-bin/source-configs/<name>.toml`; add a new one
by dropping in a TOML file with `therock_branch` / `compiler_branch` (and an
optional `description` / `patch_tag`).

The TheRock checkout is **shared** across source configs and switched **in
place**: a marker file (`<TheRock>/.srock-source-config`) records the config the
sources currently reflect. Requesting a different `-c/--config` than the checkout
currently reflects detects a switch and **forces a reconfigure** (no
`--reconfigure` needed), driving `setup_srock.sh` to check out the new super-repo
branch, resync submodules to its pins, and (for `amd-staging`) reapply the
compiler override and patch set. When the marker is **absent** — a checkout set
up directly by `setup_srock.sh` or predating this feature — the switch detection
falls back to the checkout's **actual super-repo git branch**, so `-c <name>`
still takes effect on existing trees (a detached HEAD is left untouched; use
`--reconfigure` to switch it explicitly). The selected source config is also
folded into the export manifest name (e.g. `amd-staging-minimal` vs
`develop-minimal`) so distinct selections never collide.

#### Build set (SROCK_CONFIG) via `--add`

The build set (`SROCK_CONFIG`) is a *configure toggle surfaced through `--add`*,
with **`minimal`** as the default (no option needed):

| `--add` toggle | `SROCK_CONFIG` | Meaning |
|---|---|---|
| *(none)* | `minimal` | Compiler-developer stack (default) |
| `--add all` | `all` | Full build **minus** known-failing (MIOpen, CK, FFT off) |
| `--add all-debug` | `all-debug` | Full build, **may include** failing components |

`all-debug` wins over `all` if both are given. Toggles combine with each other
and with normal `--add`/`--remove`, e.g. `--add all-debug,sysdeps` or
`--add all --add rocmlibs`. Like `sysdeps`, these change the **configured**
component set, so they only take effect at **(re)configure** time — pair them
with `--reconfigure`:

```
./aomp_build.py --backend therock --reconfigure                 # minimal (default)
./aomp_build.py --backend therock --add all --reconfigure       # full minus failing
./aomp_build.py --backend therock --add all-debug,sysdeps --reconfigure
```

`--add sysdeps` toggles TheRock's bundled system dependencies
(`THEROCK_BUNDLE_SYSDEPS`, the `therock-*` sysdep components that make the
install self-contained and portable). The **default is off** — system libraries
are resolved from the host. `sysdeps` is a *configure toggle surfaced through
`--add`* rather than a normal component group: it is not a `--sysdeps` flag, you
request it the same way you request any other extra (`--add sysdeps`,
optionally combined like `--add sysdeps,rocmlibs`).

```
./aomp_build.py --backend therock --reconfigure              # default: no bundled sysdeps
./aomp_build.py --backend therock --add sysdeps --reconfigure # bundle system deps
```

When set, `-DTHEROCK_BUNDLE_SYSDEPS=ON` is appended to `SROCK_CMAKE_EXTRA`
(which srock passes to cmake last, so it wins over the config block) and emitted
explicitly in both directions so the toggle survives CMake's cache; the bundled
libs that real components need are then pulled into the build by the default
request closure. Because it changes the **configured** component set, it only
takes effect at **(re)configure** time — pair `--add sysdeps` with
`--reconfigure` to apply it (and regenerate `subproject_map.json`).

Like the aomp backend, the TheRock backend builds with an **isolated `PATH`**
(the venv plus srock's supplemental `cmake`/`ninja` dirs are layered on top in
environment discovery). TheRock's CMake configure discovers its build tools —
`cmake`, `ninja`, `patchelf`, `meson` — from `PATH` via `find_program`, and the
srock minimal config enables `THEROCK_BUNDLE_SYSDEPS`/`THEROCK_ENABLE_ROCGDB`,
which **require `patchelf` and `meson`** on Linux. These are provided from the
venv automatically: `srock_venv_activate` (`srock-bin/srock_common_vars`) runs
`pip install patchelf meson` when setting up the venv, so the build is
self-contained and needs no system packages or user prefixes. If you maintain
those tools elsewhere, `--inherit-path` (use your `PATH`) and
`--pass-env VAR[,VAR...]` remain available.

### Group-based sharding

TheRock ships a static `BUILD_TOPOLOGY.toml` (parsed via its own
`build_topology` module) describing the artifact *groups*, the artifacts each
group produces/consumes, and the source sets (git submodules) each group needs.
The therock backend exposes those groups as **shards**: `list-shards` enumerates
them in dependency order, and `--import-shard` / `--build-shard` /
`--export-shard(s)` drive the native import → build → export pipeline (using
`fetch_sources.py --source-sets`, `buildctl.py bootstrap`, the per-group
`ninja artifact-group-<g>` targets, and the `shard_artifacts.py export-local`
helper). To select and pin exactly each group's subprojects, the backend reads
an `artifact_map.json` (artifact → composing subprojects) emitted by the bundled
introspection — for which it records a `THEROCK_ARTIFACT_SUBPROJECT_DEPS`
property on each artifact target in `cmake/therock_artifacts.cmake`. See
[Sharding](#sharding) for the full option set and examples. Sharding is
unavailable when the topology cannot be loaded.

### Manifest

Manifest export/import works as for the aomp backend, recording a git
fingerprint per subproject (resolved to the enclosing submodule repository) plus
the TheRock super-repo itself. The compiler submodules (`amd-llvm`, `hipify`,
`spirv-llvm-translator`), which the srock workflow keeps on a moving
`amd-staging` branch, are treated as **floating** and are not rolled back on
import.

---

## Logs

Each executed task writes a numbered log:

```
<log-dir>/NNN-component-variant-stage.log
```

(config-less init/fini tasks and default-only components are
`NNN-component-stage.log`.)

where `NNN` is the global task number (matching `list`) and `<log-dir>`
defaults to `<BUILD_DIR>/aomp_build_logs`. Each log begins with the task
number, the exact command, and a start timestamp, and ends with an end
timestamp and the return code.

On failure, the orchestrator prints the failing task and tails the log to
stderr, then stops (non-zero exit). Re-run with `<failed-task> continue` after
fixing the problem.

### Progress line

When stdout is a terminal, an ephemeral mid-grey status line is drawn at the
bottom while each task runs, showing a live elapsed clock plus the clipped last
line of that task's log (two-space indented) so you can watch the build move:

```
  [2m03s] [ 47%] Building CXX object lib/.../Foo.cpp.o
```

The trailing text is the current tail of `aomp_build_logs/NNN-<task>.log`,
polled ~10 times per second and redrawn when that line changes. The clock keeps
ticking (redrawn ~once a second) even when the log is quiet, so a step that
produces no output for a while -- e.g. a long compile whose output has not yet
flushed to the log -- never looks hung. The line is transient: it is cleared the
moment the task finishes, before the next task's header is printed, and before a
failure tail. It is never emitted when stdout is not a TTY (pipe, file, CI log),
so captured output stays free of carriage returns and ANSI escapes.

### Completion stamps

Each task records its progress with two stamp files in a `stamps/` directory
alongside the log dir (e.g. `<BUILD_DIR>/stamps/`):

- `component-stage.start` — written when the task begins, and
- `component-stage.done` — written when it finishes successfully.

The `list` selector reflects this with a colored mark:

| Mark | Meaning | Stamps |
|------|---------|--------|
| green `✓` | completed | `.done` present |
| red `✗` | started but did not finish (failed/interrupted) | `.start` only |
| (blank) | not built | neither |

```
[005] [✓] project/cmake
[006] [✗] project/build
[007] [ ] project/install
```

**Clearing.** Any run clears the stamps for every task from the lowest task
index being run through to the end, *before* executing. So a full build (no
selector) resets all stamps, and running a subset or `continue` resets that
point onward. Deleting the `stamps/` directory also resets everything.

---

## Sharing sources between AOMP and TheRock

AOMP keeps each component as its own standalone repo under `AOMP_REPOS`; TheRock
keeps them as submodules of a single checkout. TheRock is treated as the
canonical layout (AOMP builds move toward it, and it is mostly a superset).
Sharing comes in two tiers, defined by `SHARED_COMPONENTS` in
[bin/orchestrator/source_layout.py](orchestrator/source_layout.py):

**Tier 1 — standalone repos in both layouts (symlink *and* migrate).** A handful
of components are standalone git repos on both sides, so they can be both
symlinked into an AOMP checkout and migrated 1:1 into TheRock's submodule slots:

| AOMP `AOMP_REPOS/…`     | TheRock path                       | submodule              | branch                  |
| ----------------------- | ---------------------------------- | ---------------------- | ----------------------- |
| `llvm-project`          | `compiler/amd-llvm`                | `llvm-project`         | `amd-staging`           |
| `rocm-cmake`            | `base/rocm-cmake`                  | `rocm-cmake`           | `mainline`              |
| `hipify`                | `compiler/hipify`                  | `HIPIFY`               | `amd-staging`           |
| `ROCgdb`                | `debug-tools/rocgdb/source`        | `rocgdb`               | `amd-staging-rocgdb-16` |
| `rocmlibs/half`         | `base/half`                        | `half`                 | `rocm`                  |
| `SPIRV-LLVM-Translator` | `compiler/spirv-llvm-translator`   | `spirv-llvm-translator`| `amd-staging`           |

`llvm-project` also backs AOMP's `project` / `comgr` / `hipcc` via subpaths, so
those resolve through the single shared checkout.

**Tier 2 — monorepo-backed components (symlink *only*).** Most remaining AOMP
repos exist in TheRock too, but as a *subdirectory* of the `rocm-systems` /
`rocm-libraries` monorepo submodule (or vendored under `third-party/`). A
whole-repo directory symlink still works (the AOMP build then compiles whatever
TheRock's monorepo checkout contains), but they **cannot be migrated** — you
can't move a standalone AOMP repo into a slot the monorepo owns. So `--migrate-aomp`
ignores them; only `--therock-symlinks` covers them.

| AOMP `AOMP_REPOS/…`        | TheRock path                          |
| -------------------------- | ------------------------------------- |
| `amdsmi`                   | `rocm-systems/projects/amdsmi`        |
| `clr`                      | `rocm-systems/projects/clr`           |
| `hip`                      | `rocm-systems/projects/hip`           |
| `ROCdbgapi`                | `rocm-systems/projects/rocdbgapi`     |
| `rocminfo`                 | `rocm-systems/projects/rocminfo`      |
| `rocm_smi_lib`             | `rocm-systems/projects/rocm-smi-lib`  |
| `rocprofiler-register`     | `rocm-systems/projects/rocprofiler-register` |
| `rocprofiler-sdk`          | `rocm-systems/projects/rocprofiler-sdk` |
| `rocr-runtime`             | `rocm-systems/projects/rocr-runtime`  |
| `rocmlibs/rccl`            | `rocm-systems/projects/rccl`          |
| `rocmlibs/rocBLAS`         | `rocm-libraries/projects/rocblas`     |
| `rocmlibs/rocPRIM`         | `rocm-libraries/projects/rocprim`     |
| `rocmlibs/rocSPARSE`       | `rocm-libraries/projects/rocsparse`   |
| `rocmlibs/rocSOLVER`       | `rocm-libraries/projects/rocsolver`   |
| `rocmlibs/hipBLAS-common`  | `rocm-libraries/projects/hipblas-common` |
| `rocmlibs/hipBLAS`         | `rocm-libraries/projects/hipblas`     |
| `rocmlibs/rocRAND`         | `rocm-libraries/projects/rocrand`     |
| `rocmlibs/hipRAND`         | `rocm-libraries/projects/hiprand`     |
| `rocmlibs/hipSOLVER`       | `rocm-libraries/projects/hipsolver`   |
| `simde`                    | `third-party/simde`                   |

Note the target paths handle the casing/underscore differences
(`ROCdbgapi`→`rocdbgapi`, `rocm_smi_lib`→`rocm-smi-lib`) and cross-repo placement
(`rccl` lives in `rocm-systems`, not `rocm-libraries`). Consequences of sharing
these: the AOMP build compiles TheRock's monorepo-pinned versions (the monorepos
track `develop`, not AOMP's per-component branches), git fingerprints in the
manifest resolve to the enclosing monorepo SHA, and AOMP's transient `patchrepo`
edits apply to the shared tree (the same as the Tier 1 shared repos). Components
with **no** TheRock counterpart (notably `hipfort`) stay AOMP-only and are always
cloned.

These provisioning steps run **once**, after component resolution and before any
build work. `-s/--source` is always the **destination** root.

### `--clone` (AOMP)

Clone the AOMP sources into `-s/--source` via `clone_aomp.sh`. When the selected
set includes any `rocmlibs` components, `rocmlibs/clone_rocmlibs.sh` is run too.

```bash
./aomp_build.py -s ~/git/aomp --clone list      # provision, then list
```

### `--therock-symlinks DIR` (AOMP)

Provision `-s/--source` by **symlinking** the shared repos (both tiers above —
the standalone repos *and* the monorepo-subdir / vendored components) from the
TheRock checkout `DIR`, then clone the remaining AOMP-only repos with
`clone_aomp.sh` / `clone_rocmlibs.sh` (both skip any dir that is already a
symlink, so the canonical TheRock tree is never mutated by a clone/pull). Whole-repo
directory symlinks are used (not a per-file shadow tree): git, edits and
add/remove all resolve to the real repo.

```bash
./aomp_build.py -s ~/git/aomp --therock-symlinks ~/code/TheRock
```

A shared slot that is missing in the TheRock checkout (submodule not fetched) is
warned and left to `clone_aomp.sh`; an existing **real** directory in
`AOMP_REPOS` is never clobbered.

### `--migrate-aomp REPODIR` (TheRock)

Pre-seed TheRock's submodule slots by **moving** the shared repos out of an
existing AOMP checkout `REPODIR` into `-s/--source`'s TheRock checkout,
converting each standalone repo into a submodule gitdir (the repo's `.git` moves
to `<therock>/.git/modules/<name>` and a gitlink `.git` file is written). This is
**destructive** (directories are moved) and prompts for confirmation unless
`-y/--yes` is given; `-n/--dry-run` previews the plan without moving anything.

`--migrate-aomp` is a **standalone setup step**: it runs *before* the cmake
configure and then exits without building. It seeds only the shared 1:1 repos,
so afterwards it runs TheRock's `build_tools/fetch_sources.py` to populate the
*other* submodules (the `rocm-systems` / `rocm-libraries` monorepos, etc.) that
the configure needs. `fetch_sources.py` sees the seeded slots as already
initialized (their gitlink `.git` exists) and only checks them out — it does
**not** re-clone over them — while cloning the rest. Run it, then build:

```bash
therock_build.py -s ~/git/srock --migrate-aomp ~/git/aomp -n   # preview only
therock_build.py -s ~/git/srock --migrate-aomp ~/git/aomp      # move + fetch rest
therock_build.py -s ~/git/srock --reconfigure                  # then configure + build
```

The `--reconfigure` build reuses the seeded gitdirs (a fast checkout of the
pinned SHA, no re-clone of the migrated repos).

Per-repo pre-flight skips a slot that is missing in the AOMP checkout, is not a
git repo, or whose TheRock slot is already populated. Uncommitted changes move
with the tree, and a branch that differs from TheRock's expected branch is
warned but allowed — the recorded submodule SHA is reconciled by that next
`--reconfigure` (fast local checkout, no re-clone).

---

## User workflows

### Full build from scratch

```bash
cd $AOMP_REPOS/aomp/bin
./aomp_build.py            # builds the default component set in order
```

### Inspect before building

```bash
./aomp_build.py --components   # which components, in what order
./aomp_build.py list           # every task, numbered
./aomp_build.py -n             # full dry-run (commands + log paths)
```

### Rebuild a single component

```bash
./aomp_build.py 'comgr/*'              # all comgr tasks (includes its clean)
./aomp_build.py comgr/default/cmake continue  # reconfigure comgr, then onward
./aomp_build.py -C                     # wipe the install dir, then full build
./aomp_build.py comgr/default/build    # just re-run comgr's build step
```

### Resume after a failure

A task fails; its log is tailed to your terminal. Fix the issue, then resume
with a trailing `continue`:

```bash
./aomp_build.py comgr/default/cmake continue   # re-run from comgr's cmake onward
# or by number, using the value shown in `list`:
./aomp_build.py 30 continue
```

### Build extra component sets

```bash
./aomp_build.py --add rocmlibs                 # default set + ROCm math libs
./aomp_build.py --add rocmlibs 'rocBLAS/*'     # only rocBLAS tasks (deps still resolved)
./aomp_build.py --remove debug                 # drop the debugger (rocdbgapi + rocgdb)
```

### Build a sanitizer / perf / debug variant

```bash
./aomp_build.py --variant asan                 # asan everywhere it's offered
./aomp_build.py --variant llvm_runtimes_standalone=debug
```

### Tune the build environment

```bash
./aomp_build.py -j 32 --ninja --ccache         # threads + ninja + ccache
./aomp_build.py --gfx "gfx90a;gfx942"          # specific GPU targets
./aomp_build.py --inherit-path                 # use my PATH (not the clean default)
./aomp_build.py --pass-env CC,CXX,LD_LIBRARY_PATH   # leak specific env vars
```

### Use custom source / install / build / prereq directories

```bash
./aomp_build.py -s /scratch/aomp-src -i /opt/aomp -b /scratch/aompbuild -p /opt/aomp-supp
```

`-s/--source` sets the repo/source root (`AOMP_REPOS`), `-i/--install` sets the
install root (`AOMP`), `-b/--build` sets where builds run and object files go
(`BUILD_AOMP`, and the default log/manifest location), and `-p/--prereq` sets the
supplemental/prerequisite root (`AOMP_SUPP`). All are made absolute and exported
to every child script. Note `BUILD_AOMP` defaults to `AOMP_REPOS`, so `-s`
without `-b` puts builds under the source root.

### Reproduce an earlier build

```bash
# On the reference machine:
./aomp_build.py --export-manifest good.json
# Later / elsewhere (repos must be clean):
./aomp_build.py --import-manifest good.json
```

---

## Internals

### The taskified component interface

Every component script (`build_<name>.sh`) speaks a common interface provided
by `command_dispatcher` in [`bin/aomp_utils`](aomp_utils). The orchestrator
relies on exactly these commands:

| Command | Purpose |
|---------|---------|
| `list_configs` | Print the configs/variants this component offers, one per line. |
| `list` | Print every task as `task_<action> [cfg]`, one per line (init tasks, then per-config tasks, then fini tasks). |
| `task_<action> [cfg]` | Run a single task for a given config. |
| `show_src_dir` | Print the component's git source directory (for the manifest). |
| `show_build_dir <cfg>` / `show_install_dir <cfg>` | Print build/install directories (informational). |

Because `list` emits each line already in `task_<action> [cfg]` form, the
orchestrator can pass a line straight back to the script to execute it.

`build_supp.sh` (and its `build_prereq.sh` symlink) is modeled as a single
coarse component: one `default` config, a `task_build` that runs the existing
"build all prerequisite/supplemental components" logic, and a no-op
`task_install`. It still supports its legacy direct invocation
(`build_supp.sh openmpi`, no-arg, `install`, `-h`).

### The CUDF config format

The config (default [`bin/configs/aomp.cudf`](configs/aomp.cudf)) is a
CUDF-style (Common Upgradeability Description Format) text file. Stanzas are
separated by blank lines; `#` lines are comments. Within a stanza each line is
`key: value`; a line whose first token has no `:` is treated as a continuation
of the previous value (used to wrap long comma-separated lists).

Three stanza kinds:

```
package: comgr            # a buildable component
version: 1                # stanza schema version (always 1)
depends: project, rocr    # components that must build first (a DAG)
x-dir: .                  # bin subdir with build_<name>.sh: "." or "rocmlibs"

feature: flang            # a named alias for a set of components
expands: llvm-classic, flang-classic, pgmath, flang, flang_runtime

request:                  # the default requested build set
install: prereq, project, ...
```

The package list, ordering, and grouping are derived from
[`bin/build_aomp.sh`](build_aomp.sh) (standalone x86_64 build) and
[`bin/rocmlibs/build_rocmlibs.sh`](rocmlibs/build_rocmlibs.sh). Dependencies are
expressed as a DAG; packages are declared in canonical build order so the
topological sort reproduces that order (see below).

The parser is intentionally simple and lives entirely in `parse_cudf()`. It
leaves room for a future external version *solver* (the "CUDF" framing): today
the `request:` set plus `--add`/`--remove` is resolved directly, but the same
config could later feed a real dependency/version solver.

### Resolution pipeline

`resolve_components()` turns the config + `--add`/`--remove` into an ordered
component list:

1. Start from `request.install`.
2. `requested |= expand(--add)` and `removed = expand(--remove)`; features are
   expanded to their component sets via `expand_names()`.
3. `requested -= removed`; unknown names are rejected.
4. **Transitive closure with removal cascade** (fixed-point loop): for each
   component, pull in its `depends`; if a dependency is in the removed set,
   drop the dependent (and add it to the removed set so its own dependents
   cascade too). This is why `--remove comgr` also drops `hipamd`,
   `rocdbgapi`, etc.
5. **Topological sort** (`topo_sort()`): a deterministic Kahn's algorithm where
   ready nodes are always taken in *declaration order*. Because packages are
   declared in canonical `build_aomp.sh` order, the result reproduces that
   canonical order while still honoring real dependency edges. A cycle is a
   hard error.

### Task elaboration

`elaborate_tasks()` expands the ordered component list into a flat task list:

1. For each component, locate its script via `script_path()` (honoring
   `x-dir`).
2. Query `list_configs`; pick the configs to build via `select_variants()`
   (see [Build variants](#build-variants)). If an explicit `--variant` filter
   matches none of the component's configs, the component is skipped entirely.
3. Query `list`; parse each `task_<action> [cfg]` line.
4. Keep a task if either it has no config (init/fini tasks such as `precheck`,
   `patch`, `unpatch` always run) or its config is in the selected set.

`clean` tasks (which wipe a component's build dir) are kept like any other
task — they sit right before each `cmake`. With `-C/--clean`, a single
`install/clean` pseudo-task is prepended to the whole list; it wipes the
install directory (the versioned symlink target, then the symlink) and is run
by the orchestrator itself (no backing script).

Each surviving task is recorded as a `Task` (component, action, config, script
path, and the exact `script_args` to pass back). Its display name is
`component/variant/stage`, shortened to `component/stage` when the task has no
config or when the component advertises only the `default` config.

### Execution

`select_tasks()` applies the selector grammar to produce a list of task
indices; `run_tasks()` runs them:

- Creates the log directory.
- For each task, opens `NNN-component-variant-stage.log`, writes a header
  (number, command, start time), runs `bash build_<name>.sh task_<action>
  [cfg]` with stdout+stderr redirected to the log, and writes a footer (end
  time, return code).
- On a non-zero return code, prints the failure and tails the log to stderr,
  then returns that code (stops the run).
- `--dry-run` prints the planned command and log path instead of executing.

### Environment discovery and child environment

- `discover_env()` sources `aomp_utils` + `aomp_common_vars` in a subshell and
  reads back `BUILD_DIR`, `AOMP_REPOS`, and `AOMP_REPO_NAME` — used only to
  locate the default log and manifest directories.
- `build_child_env()` builds the child environment **from scratch** (it does not
  copy the caller's environment): a curated pass-through of identity/locale vars
  (`ENV_PASSTHROUGH` + `LC_*`), a controlled `PATH` (`DEFAULT_CHILD_PATH`, or the
  caller's PATH with `--inherit-path`), any vars named by `--pass-env`, the
  directory knobs (`-s`/`AOMP_REPOS`, `-i`/`AOMP`, `-b`/`BUILD_AOMP`,
  `-p`/`AOMP_SUPP`, made absolute), and then the values implied by the global build-knob flags (`-j`,
  `--ninja`, `--ccache`, `--gfx`, `--sudo`). This same environment is used for every child
  script invocation (introspection, execution, and git/manifest probes), so
  introspection sees the same configs that will actually be built — and isolates
  the build from stray exports on the caller's side.
- `--build-type` is resolved separately into a global value plus per-component
  overrides and applied as the `BUILD_TYPE` variable on each task's subprocess
  at execution time (`run_tasks`), so different components can build with
  different CMake build types.

---

## Troubleshooting

**A component shows no tasks in `list`.**
Its `list` produced no `task_*` lines — usually the component self-disables in
the current environment (e.g. `llvm-classic`/`flang-classic` when the classic
LLVM source is absent). Such components emit a warning on **stderr** and exit
cleanly, so they simply contribute nothing. Confirm by running the script
directly: `bash build_<name>.sh list`.

**`unknown component(s): …`.**
A name passed to `--add`/`--remove` (or in the config's `request:`) is neither
a package nor a feature. Check spelling against `--components` and the config.

**`dependency cycle among: …`.**
The `depends:` edges in the config form a cycle. Fix the config.

**Import refuses with "local modifications".**
A target repo is dirty. Commit/stash/clean it, or drop it from the resolved set
(`--remove`), then retry the import.

**A task fails.**
Read the tailed log (path printed on failure, under the log dir). After fixing,
resume with `<task> continue`.

**A tool isn't found, or the wrong one is picked up.**
Child scripts use a controlled `PATH` and a clean environment by default (see
[Environment isolation](#environment-isolation)), so a tool living only in a
non-standard location (e.g. a Homebrew prefix) won't be on `PATH`, and
build-affecting exports like `PKG_CONFIG_PATH`, `CC`, or `LD_LIBRARY_PATH` are
not inherited. Use `--inherit-path` to restore your `PATH`, and/or
`--pass-env VAR[,VAR...]` to leak the specific variables the build needs.

---

## Extending

- **Add a component:** create `build_<name>.sh` implementing the
  [task interface](#the-taskified-component-interface), then add a `package:`
  stanza (with `depends:` and `x-dir:`) to the config. Declare it in canonical
  build position so the topological sort places it correctly.
- **Add a feature:** add a `feature:` stanza with an `expands:` list.
- **Change the default set:** edit the `request:` stanza, or keep the config
  fixed and use `--add`/`--remove` at the command line.
- **Use a different config:** `-c /path/to/other.cudf`. The manifest's default
  filename follows the config's basename.
- **Add a backend:** implement the `Backend` interface in
  [`bin/orchestrator/backend.py`](orchestrator/backend.py) (config loading,
  environment, task listing, and per-task command) and register it in
  `core.make_backend()`. The generic core (resolution, ordering, selectors,
  execution, stamps, manifest, and the `list-shards` / `--*-shard` wiring that
  defers to the backend's optional `list_shards`/`shard_tasks` hooks) is reused
  unchanged — see the
  [TheRock backend](orchestrator/therock_backend.py) for a worked example that
  is driven by introspection JSON rather than per-component scripts.
