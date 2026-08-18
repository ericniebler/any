# Adds a `coverage` build target that instruments TARGET with clang's
# source-based code coverage, runs it, and produces a text summary plus an
# HTML report under ${CMAKE_BINARY_DIR}/coverage/html. Clang only.
function(any_add_coverage_target TARGET)
  if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(WARNING "ANY_ENABLE_COVERAGE requires Clang; ignoring for ${CMAKE_CXX_COMPILER_ID}")
    return()
  endif()

  target_compile_options(${TARGET} PRIVATE -fprofile-instr-generate -fcoverage-mapping -g)
  target_link_options(${TARGET} PRIVATE -fprofile-instr-generate -fcoverage-mapping)

  string(REPLACE "." ";" _clang_version_list ${CMAKE_CXX_COMPILER_VERSION})
  list(GET _clang_version_list 0 _clang_major_version)

  find_program(LLVM_PROFDATA NAMES llvm-profdata-${_clang_major_version} llvm-profdata)
  find_program(LLVM_COV NAMES llvm-cov-${_clang_major_version} llvm-cov)
  if (NOT LLVM_PROFDATA OR NOT LLVM_COV)
    message(WARNING "llvm-profdata/llvm-cov not found; `coverage` target will not be available")
    return()
  endif()

  set(_coverage_dir ${CMAKE_BINARY_DIR}/coverage)
  set(_profraw ${_coverage_dir}/${TARGET}.profraw)
  set(_profdata ${_coverage_dir}/${TARGET}.profdata)

  add_custom_target(coverage
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_coverage_dir}
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE=${_profraw} $<TARGET_FILE:${TARGET}>
    COMMAND ${LLVM_PROFDATA} merge -sparse ${_profraw} -o ${_profdata}
    COMMAND ${LLVM_COV} report $<TARGET_FILE:${TARGET}>
            -instr-profile=${_profdata}
            -ignore-filename-regex=${CMAKE_BINARY_DIR}
    COMMAND ${LLVM_COV} show $<TARGET_FILE:${TARGET}>
            -instr-profile=${_profdata}
            -format=html
            -output-dir=${_coverage_dir}/html
            -ignore-filename-regex=${CMAKE_BINARY_DIR}
    DEPENDS ${TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running ${TARGET} and generating coverage report"
    VERBATIM)
endfunction()
