# Ling-3.0-flash (bailingmoe3) model package.
add_library(gufo_ling_3_flash STATIC
  chat_template.cpp
  config.cpp
  gpu_dump.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_ling_3_flash PUBLIC ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Numerics identity of the compiled kernels: a SHA-256 over every kernel
# source, the HIP compiler version, HIP flags, target architectures, the ROCm
# release and the prefill batch size. It is part of the disk-cache identity, so
# snapshots computed by different numerics are never loaded. It changes when
# any of those change (configure re-runs when a kernel file changes).
set(LING_HIP_COMPILE_OPTIONS -O3 -ffast-math -fno-finite-math-only
  -Wno-unused-command-line-argument)
file(GLOB_RECURSE LING_NUMERIC_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*")
# layout.hpp fixes the state/cache layout and the config the kernels read.
list(APPEND LING_NUMERIC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/layout.hpp")
list(SORT LING_NUMERIC_SOURCES)
set(LING_NUMERIC_ACCUMULATOR "")
foreach(f IN LISTS LING_NUMERIC_SOURCES)
  file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
  file(SHA256 "${f}" digest)
  string(APPEND LING_NUMERIC_ACCUMULATOR "${rel}:${digest}\n")
endforeach()
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/engine.hpp" LING_BATCH_LINE
  REGEX "kCliPrefillBatch = [0-9]+")
string(REGEX MATCH "[0-9]+" LING_PREFILL_BATCH "${LING_BATCH_LINE}")
string(SHA256 LING_NUMERICS_HASH
  "${LING_NUMERIC_ACCUMULATOR}|hip=${CMAKE_HIP_COMPILER_VERSION}|rocm=${hip_VERSION}|flags=${LING_HIP_COMPILE_OPTIONS}|${CMAKE_HIP_FLAGS}|${CMAKE_HIP_FLAGS_RELEASE}|${CMAKE_HIP_FLAGS_RELWITHDEBINFO}|arch=${CMAKE_HIP_ARCHITECTURES}|batch=${LING_PREFILL_BATCH}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/numerics_id.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/ling_3_flash_numerics_id.hpp @ONLY)

target_link_libraries(gufo_ling_3_flash PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  set(LING_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  # One translation unit per kernel group.
  set(LING_HIP_SOURCES
    ${LING_ROCM_DIR}/dense.hip.cpp
    ${LING_ROCM_DIR}/kda.hip.cpp
    ${LING_ROCM_DIR}/mla.hip.cpp
    ${LING_ROCM_DIR}/moe.hip.cpp
    ${LING_ROCM_DIR}/mtp.hip.cpp)
  target_sources(gufo_ling_3_flash PRIVATE ${LING_HIP_SOURCES}
    ${LING_ROCM_DIR}/device_model.cpp
    ${LING_ROCM_DIR}/executor.cpp
    ${LING_ROCM_DIR}/state_io.cpp
    engine.cpp)
  set_source_files_properties(${LING_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_ling_3_flash PRIVATE
    "$<$<COMPILE_LANGUAGE:HIP>:${LING_HIP_COMPILE_OPTIONS}>")
  target_link_libraries(gufo_ling_3_flash PUBLIC hip::host roc::hipblas)
endif()

# Scalar float32 oracle; built only by explicit validation targets, never by
# production or routine checks. It keeps -O2 in every configuration.
add_library(gufo_ling_3_flash_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_ling_3_flash_reference PUBLIC gufo_ling_3_flash)
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")
