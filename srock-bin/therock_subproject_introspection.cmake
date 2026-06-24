# Generate a JSON file containing key metadata about the current build
# configuration for processing by other tools.
#
# Bundled with the AOMP srock scripts (from TheRock PR #1234) so the TheRock
# orchestrator backend can introspect a stock TheRock checkout that does not yet
# carry this support. The TheRock backend (bin/orchestrator/therock_backend.py)
# copies this file into <TheRock>/cmake/ and appends an include + invocation to
# the top-level CMakeLists.txt, gated by -DTHEROCK_INTROSPECTION=ON.

function(therock_introspect_subprojects)
  include(therock_subproject_utils)
  therock_get_all_targets(all_targets "${CMAKE_CURRENT_SOURCE_DIR}")
  set(info "{")
  set(first TRUE)
  foreach(target ${all_targets})
    get_target_property(_is_subproject ${target} THEROCK_SUBPROJECT)
    if (NOT _is_subproject STREQUAL "cmake")
      string(FIND ${target} "+" plusloc)
      if (NOT plusloc EQUAL -1)
        string(LENGTH ${target} strlen)
        string(SUBSTRING ${target} 0 ${plusloc} subproject_part)
        math(EXPR plusloc "${plusloc}+1")
        string(SUBSTRING ${target} ${plusloc} ${strlen} action_part)
        list(APPEND "actions_${subproject_part}" "\"${action_part}\"")
      endif()
    endif()
  endforeach()
  foreach(target ${all_targets})
    get_target_property(_is_subproject ${target} THEROCK_SUBPROJECT)
    if (_is_subproject STREQUAL "cmake")
      if(NOT ${first})
        set(info "${info},\n")
      endif()
      set(info "${info}\"${target}\":\n")
      get_target_property(src_dir ${target} THEROCK_CMAKE_SOURCE_DIR)
      get_target_property(bin_dir ${target} THEROCK_BINARY_DIR)
      get_target_property(install_dest ${target} THEROCK_INSTALL_DESTINATION)
      get_target_property(build_deps ${target} THEROCK_BUILD_DEPS)
      get_target_property(runtime_deps ${target} THEROCK_RUNTIME_DEPS)
      get_target_property(build_pool ${target} THEROCK_BUILD_POOL)
      get_target_property(compiler_toolchain ${target} THEROCK_COMPILER_TOOLCHAIN)
      file(RELATIVE_PATH rel_srcdir ${CMAKE_SOURCE_DIR} ${src_dir})
      file(RELATIVE_PATH rel_bindir ${CMAKE_BINARY_DIR} ${bin_dir})
      set(info "${info}{ \"src\": \"${rel_srcdir}\",\n")
      set(info "${info}  \"bin\": \"${rel_bindir}\",\n")
      set(info "${info}  \"install_dest\": \"${install_dest}\",\n")
      set(quoted_build_deps)
      foreach(dep ${build_deps})
        list(APPEND quoted_build_deps "\"${dep}\"")
      endforeach()
      list(JOIN quoted_build_deps "," deplist)
      set(info "${info}  \"build_deps\": [ ${deplist} ],\n")
      set(quoted_runtime_deps)
      foreach(dep ${runtime_deps})
        list(APPEND quoted_runtime_deps "\"${dep}\"")
      endforeach()
      list(JOIN quoted_runtime_deps "," rundeplist)
      set(info "${info}  \"runtime_deps\": [ ${rundeplist} ],\n")
      set(info "${info}  \"build_pool\": \"${build_pool}\",\n")
      set(info "${info}  \"compiler_toolchain\": \"${compiler_toolchain}\",\n")
      list(JOIN "actions_${target}" "," actionlist)
      set(info "${info}  \"actions\": [ ${actionlist} ] }")
      set(first FALSE)
    endif()
  endforeach()
  set(info "${info}\n}")

  file(WRITE ${CMAKE_BINARY_DIR}/subproject_map.json "${info}")

  therock_introspect_features()
  therock_introspect_artifacts()
endfunction()

# Emit artifact_map.json: maps each topology artifact to the cmake subprojects
# that compose it (the SUBPROJECT_DEPS passed to therock_provide_artifact). The
# orchestrator turns this into a subproject->artifact-group mapping (the group
# comes from BUILD_TOPOLOGY.toml) so group-based shards can select and pin
# exactly the right subprojects. Requires therock_provide_artifact to record
# THEROCK_ARTIFACT_SUBPROJECT_DEPS on each artifact-<slice> target (the srock
# orchestrator injects this into cmake/therock_artifacts.cmake); artifacts
# without that property (e.g. the artifact-group-* aggregate targets) are
# skipped.
function(therock_introspect_artifacts)
  include(therock_subproject_utils)
  therock_get_all_targets(all_targets "${CMAKE_CURRENT_SOURCE_DIR}")
  set(info "{")
  set(first TRUE)
  foreach(target ${all_targets})
    string(FIND "${target}" "artifact-" _loc)
    if(NOT _loc EQUAL 0)
      continue()
    endif()
    get_target_property(_deps "${target}" THEROCK_ARTIFACT_SUBPROJECT_DEPS)
    if(NOT _deps)
      continue()
    endif()
    string(SUBSTRING "${target}" 9 -1 artifact_name)  # strip "artifact-"
    set(quoted_deps)
    foreach(dep ${_deps})
      list(APPEND quoted_deps "\"${dep}\"")
    endforeach()
    list(JOIN quoted_deps "," deplist)
    if(NOT ${first})
      set(info "${info},\n")
    endif()
    set(info "${info}\"${artifact_name}\": [ ${deplist} ]")
    set(first FALSE)
  endforeach()
  set(info "${info}\n}")
  file(WRITE ${CMAKE_BINARY_DIR}/artifact_map.json "${info}")
endfunction()

# Emit feature_map.json: every THEROCK_ENABLE_* cache variable (the build's
# selectable "features"), so tools can list and toggle them. Covers both the
# coarse group options (e.g. THEROCK_ENABLE_ML_LIBS) and the per-artifact
# features generated from BUILD_TOPOLOGY.toml (e.g. THEROCK_ENABLE_HIPDNN).
# Runs in the top-level CMakeLists scope (after therock_finalize_features), so
# the THEROCK_ALL_FEATURES list and the THEROCK_REQUIRES_<name> variables it
# populates are visible here.
function(therock_introspect_features)
  get_cmake_property(_all_cache_vars CACHE_VARIABLES)
  set(info "{")
  set(first TRUE)
  foreach(var ${_all_cache_vars})
    string(FIND "${var}" "THEROCK_ENABLE_" _loc)
    if(NOT _loc EQUAL 0)
      continue()
    endif()
    string(SUBSTRING "${var}" 15 -1 name)  # strip "THEROCK_ENABLE_"
    if(name STREQUAL "")
      continue()
    endif()
    if(${var})
      set(enabled "true")
    else()
      set(enabled "false")
    endif()
    get_property(desc CACHE "${var}" PROPERTY HELPSTRING)
    string(REPLACE "\\" "\\\\" desc "${desc}")
    string(REPLACE "\"" "\\\"" desc "${desc}")
    # A feature declared via therock_add_feature() carries a REQUIRES list and
    # appears in THEROCK_ALL_FEATURES; the coarse group options do not.
    set(in_group "false")
    if(name IN_LIST THEROCK_ALL_FEATURES)
      set(in_group "true")
    endif()
    set(quoted_requires)
    foreach(req ${THEROCK_REQUIRES_${name}})
      list(APPEND quoted_requires "\"${req}\"")
    endforeach()
    list(JOIN quoted_requires "," reqlist)
    if(NOT ${first})
      set(info "${info},\n")
    endif()
    set(info "${info}\"${name}\":\n")
    set(info "${info}{ \"enabled\": ${enabled},\n")
    set(info "${info}  \"description\": \"${desc}\",\n")
    set(info "${info}  \"feature\": ${in_group},\n")
    set(info "${info}  \"requires\": [ ${reqlist} ] }")
    set(first FALSE)
  endforeach()
  set(info "${info}\n}")
  file(WRITE ${CMAKE_BINARY_DIR}/feature_map.json "${info}")
endfunction()
