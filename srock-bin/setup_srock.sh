#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  setup_srock.sh: Clone and initialize TheRock Repo. 
#
# --- Start standard header to set SROCK environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
# shellcheck disable=1091
. "$thisdir/srock_common_vars"
# --- end standard header ----
#

# Accept a command as the first argument.  Only "restart" is accepted so far:
# reconfigure with the existing sources, reusing the existing build directory
# (cmake is designed to be re-run over one).  An optional second argument,
# "clean", removes the build directory first for a guaranteed-fresh configure.
ARG=$1
ARG2=$2
export _build_srock_mode="$ARG"

if [ -n "$ARG2" ] && [ "$ARG2" != "clean" ]; then
   echo " ERROR: unrecognized second argument '$ARG2' (expected 'clean')"
   exit 1
fi

# Set to 1 below when an existing checkout's super-repo branch differs from the
# requested SROCK_THEROCK_BRANCH (a source-config switch, e.g. amd-staging <->
# develop). A switch makes the "restart" path also resync sources and re-apply
# the compiler override, so the in-place checkout matches the requested config.
_do_branch_switch=0

if [ -d "$SROCK_THEROCK_DIR" ] && [ "$ARG" != "restart" ]; then
   echo " ERROR:  $0 requires that $SROCK_THEROCK_DIR NOT exist"
   echo "         Delete or move that directory to run $0"
   echo "         Alternatively, try '$0 restart' to reconfigure with"
   echo "         existing sources, reusing the build dir ('$0 restart clean'"
   echo "         to remove the build dir and configure from scratch)."
   exit 1
fi

_curdir=$PWD
_start_date=$(date)
_start_secs=$(date +%s)

# Print the start banner similar to DONE banner, useful if fails
echo
echo "===== START $0 on $_start_date"
echo "      THEROCK targets:   $_gfxsemicolons"
echo "      THEROCK families:  $_gfamsemicolons"
echo "      ROCm install dir:  $SROCK_INSTALL_DIR"
echo "      TheRock Dir:       $SROCK_THEROCK_DIR"
echo "      TheRock branch:    $SROCK_THEROCK_BRANCH"
echo "      Compiler branch:   $SROCK_COMPILER_BRANCH"
echo "      SROCK config name: $SROCK_CONFIG"
echo "      cmake:             $SROCK_CMAKE"
echo "      cmake args:        ${_cmake_args[*]}"

# Run srock prebuild which includes finding suitable cmake
echo
echo "===== Sourcing prebuild_srock.sh"
. "$thisdir/prebuild_srock.sh"
echo "===== DONE Sourcing prebuild_srock.sh"

if [ "$ARG" != "restart" ]; then
   cd "$SROCK_REPOS" || exit
   echo
   echo "===== git clone https://github.com/ROCm/TheRock.git -b $SROCK_THEROCK_BRANCH TheRock"
   git clone https://github.com/ROCm/TheRock.git -b "$SROCK_THEROCK_BRANCH" TheRock
   if [ -f TheRock/version.json ] ; then
      _quoted=$(cat TheRock/version.json | grep rocm-version | cut -d: -f2)
      _rocm_version=${_quoted//\"/}
      echo " ROCm components version : $_rocm_version"
      echo " SROCK_VERSION_STRING    :  $SROCK_VERSION_STRING (Compiler dev version)"
   fi
fi

cd "$SROCK_THEROCK_DIR" || exit

srock_venv_activate

# In-place source-config switch: on "restart" against an existing checkout whose
# super-repo branch differs from the requested SROCK_THEROCK_BRANCH, switch the
# super-repo branch here. fetch_sources.py and the compiler-submodule override
# below are then forced to run (despite "restart") so submodules are repinned to
# the new branch's recorded SHAs and the matching patch set is reapplied. Without
# this, "restart" reuses whatever branch was first cloned.
if [ "$ARG" = "restart" ]; then
   _current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
   if [ -n "$SROCK_THEROCK_BRANCH" ] && [ -n "$_current_branch" ] && \
      [ "$_current_branch" != "$SROCK_THEROCK_BRANCH" ]; then
      _do_branch_switch=1
      echo
      echo "===== Source config switch: super-repo branch $_current_branch -> $SROCK_THEROCK_BRANCH"
      echo "      --- discarding local changes in submodules so they can be repinned"
      git submodule foreach --recursive 'git checkout . 2>/dev/null || true'
      echo "      --- git checkout . (super-repo)"
      git checkout .
      echo "      --- git fetch origin $SROCK_THEROCK_BRANCH"
      git fetch origin "$SROCK_THEROCK_BRANCH"
      echo "      --- git checkout $SROCK_THEROCK_BRANCH"
      git checkout "$SROCK_THEROCK_BRANCH"
      echo "      --- git pull (most recent $SROCK_THEROCK_BRANCH)"
      git pull
   fi
fi

if [ "$ARG" != "restart" ] || [ "$_do_branch_switch" = 1 ]; then
   echo
   echo "===== Running python ./build_tools/fetch_sources.py ====="
   python ./build_tools/fetch_sources.py
   echo "=====  Done running python ./build_tools/fetch_sources.py"
fi

# Build directory handling for "restart".  The default is an in-place cmake
# reconfigure, keeping the build dir: object files, each subproject's own build
# and stage dirs, TheRock's stage.prebuilt markers and any imported artifacts all
# survive.  Removing it means rebuilding everything from scratch -- upwards of a
# day for a debug compiler -- which a reconfigure does not require.
#
# It is removed only when reuse would be unsound, or when explicitly requested:
#   * a source-config switch: the checkout above moved to different branches and a
#     different patch set, so the existing build state describes sources that are
#     no longer there.
#   * "restart clean": the caller wants a configure that inherits nothing (CI and
#     release builds, or recovering a build dir whose state has gone bad).  Note
#     that an in-place reconfigure keeps cache entries this run does not set, so
#     this is the way to guarantee the configuration comes only from the current
#     arguments.
if [ "$ARG" = "restart" ]; then
   if [ "$_do_branch_switch" = 1 ]; then
      echo "==== Removing build dir: it describes the pre-switch sources ====="
      echo "rm -rf $SROCK_THEROCK_DIR/build"
      rm -rf "$SROCK_THEROCK_DIR/build"
   elif [ "$ARG2" = "clean" ]; then
      echo "==== Removing build dir for clean restart ====="
      echo "rm -rf $SROCK_THEROCK_DIR/build"
      rm -rf "$SROCK_THEROCK_DIR/build"
   elif [ -d "$SROCK_THEROCK_DIR/build" ]; then
      echo "==== Reusing build dir (cmake reconfigures in place) ====="
      echo "     $SROCK_THEROCK_DIR/build"
      echo "     Run '$0 restart clean' to configure from scratch instead."
   fi
fi

echo "cd $SROCK_THEROCK_DIR" 
cd "$SROCK_THEROCK_DIR" || exit
if [ "$ARG" != "restart" ] && [ -d build ]; then
   echo "WARNING build directory $SROCK_THEROCK_DIR/build should not exist "
fi

echo 
echo "===== Running build_tools/setup_ccache.py"
eval "$(python3 ./build_tools/setup_ccache.py)"

# Make updates to compiler submodules unless this is native TheRock build.
# Runs on a fresh setup, and on "restart" only when switching source config
# (above) -- so a config switch into amd-staging re-applies the override/patches,
# while a config switch into develop (SROCK_COMPILER_BRANCH=develop) skips it,
# leaving the native sources fetch_sources.py just repinned.
if [ "$SROCK_COMPILER_BRANCH" != "develop" ] && \
   { [ "$ARG" != "restart" ] || [ "$_do_branch_switch" = 1 ]; }; then
   # FIXME: Before wiping out current amd-staging changes, 
   #        to save current changes in the patches directory. 
   #        Otherwise, this is not a real development environment"
   echo
   echo "===== Switch to $SROCK_COMPILER_BRANCH branch for compiler components"
   echo "      --- cd $SROCK_THEROCK_DIR/compiler/hipify"
   cd "$SROCK_THEROCK_DIR/compiler/hipify" || exit
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH (WARNING: This may leave commits behind"
   git checkout "$SROCK_COMPILER_BRANCH" 
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   echo "      --- cd $SROCK_THEROCK_DIR/compiler/spirv-llvm-translator"
   cd "$SROCK_THEROCK_DIR/compiler/spirv-llvm-translator" || exit
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH"
   git checkout "$SROCK_COMPILER_BRANCH"
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   echo "      --- cd $SROCK_THEROCK_DIR/compiler/amd-llvm"
   cd "$SROCK_THEROCK_DIR/compiler/amd-llvm" || exit
   echo "      --- git checkout ."
   git checkout .
   echo "      --- git checkout $SROCK_COMPILER_BRANCH (WARNING: This leaves commits behind for amd-llvm)"
   git checkout "$SROCK_COMPILER_BRANCH"
   echo "      --- git pull (gets most recent updates to $SROCK_COMPILER_BRANCH)"
   git pull

   if [ -d "$thisdir/patches/$SROCK_COMPILER_BRANCH" ] ; then 
      cd "$SROCK_THEROCK_DIR" || exit
      shopt -s nullglob
      # shellcheck disable=SC2206 # word splitting on file glob intended
      _patches=( $thisdir/patches/$SROCK_COMPILER_BRANCH/_TheRock*.patch )
      shopt -u nullglob
      for _patch_file in "${_patches[@]}"; do
         test_apply_patch
      done
      _tmpfile=/tmp/submod$$
      git submodule > "$_tmpfile"
      echo "tmpfile:$_tmpfile"
      while read -r _line ; do
	 _subdir=$(echo "$_line" | cut -d" " -f2)
	 _subdirfull="$SROCK_THEROCK_DIR/$_subdir"
	 if [ ! -d "$_subdirfull" ] ; then 
            echo "Directory $_subdirfull does not exist "
         else 
            cd "$_subdirfull" || exit
	    _subdirname=$(echo "$_subdir" | tr "/" "_")
            shopt -s nullglob
            # shellcheck disable=SC2206 # word splitting on file glob intended
            _patches=( $thisdir/patches/$SROCK_COMPILER_BRANCH/${_subdirname}*.patch )
            shopt -u nullglob
            for _patch_file in "${_patches[@]}"; do
               test_apply_patch
            done
	 fi
      done < $_tmpfile
      rm "$_tmpfile"
   fi

   # Reconstruct compiler/.amd-llvm.smrev from the current HEAD before the cmake
   # configure below. TheRock's compiler/CMakeLists.txt reads this file at
   # configure time and forces it as the compiler's VC revision (clang
   # --version). srock's fetch_sources.py run (no --patch-tag) removes the file,
   # and the amd-staging checkout + patches leave amd-llvm non-pristine/dirty, so
   # without this LLVM would auto-compute a less precise (possibly "-dirty")
   # revision. Doing it here (vs build_srock.sh, which ran it after configure)
   # means both the orchestrator and the two-script srock workflow get a clean,
   # deterministic revision. Skipped for native develop builds (handled by the
   # enclosing SROCK_COMPILER_BRANCH != develop guard).
   echo "      --- reconstructing compiler/.amd-llvm.smrev from current HEAD"
   (
      cd "$SROCK_THEROCK_DIR/compiler/amd-llvm" || exit
      _smrev="../.amd-llvm.smrev"
      git config --get remote.origin.url > "$_smrev"
      _smsha=$(git rev-parse HEAD)
      echo "${_smsha}${LLVM_SHA_EXTRA}" >> "$_smrev"
   )

echo "      --- end compiler submodule updates for $SROCK_COMPILER_BRANCH"
fi

cd "$SROCK_THEROCK_DIR" || exit
echo 
echo "===== cmake CMD: $SROCK_CMAKE ${_cmake_args[*]}"
$SROCK_CMAKE "${_cmake_args[@]}"
_rc=$? && [ "$_rc" != 0 ] && cd "$_curdir" && exit "$_rc"

_setup_secs=$(date +%s)
_secs_to_setup=$(( _setup_secs - _start_secs ))

echo
echo "===== DONE $0 on $_start_date"
echo "   THEROCK targets:      $_gfxsemicolons"
echo "   ROCm comp version:    $_rocm_version"
echo "   SROCK_VERSION_STRING: $SROCK_VERSION_STRING (Compiler dev version)"
echo "   THEROCK families:     $_gfamsemicolons"
echo "   ROCm install dir:     $SROCK_INSTALL_DIR"
echo "   TheRock Dir:          $SROCK_THEROCK_DIR"
echo "   TheRock branch:       $SROCK_THEROCK_BRANCH"
echo "   Compiler branch:      $SROCK_COMPILER_BRANCH"
echo "   SROCK config name:    $SROCK_CONFIG"
echo "   Setup time:           $_secs_to_setup (seconds)"
echo "   cmake args:           ${_cmake_args[*]}"
echo 
echo " Next step, run this command: $thisdir/build_srock.sh" 
echo 

