# Ling-3.0-flash (bailingmoe3) model package.
add_library(gufo_ling_3_flash STATIC
  chat_template.cpp
  config.cpp
  gpu_dump.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_ling_3_flash PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ling_3_flash PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  set(LING_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  # One translation unit per kernel group.
  set(LING_HIP_SOURCES
    ${LING_ROCM_DIR}/dense.hip.cpp
    ${LING_ROCM_DIR}/kda.hip.cpp
    ${LING_ROCM_DIR}/mla.hip.cpp
    ${LING_ROCM_DIR}/moe.hip.cpp)
  target_sources(gufo_ling_3_flash PRIVATE ${LING_HIP_SOURCES}
    ${LING_ROCM_DIR}/device_model.cpp
    ${LING_ROCM_DIR}/executor.cpp
    engine.cpp)
  set_source_files_properties(${LING_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_ling_3_flash PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_ling_3_flash PUBLIC hip::host roc::hipblas)
endif()

# Scalar float32 oracle; built only by explicit validation targets, never by
# production or routine checks. It keeps -O2 in every configuration.
add_library(gufo_ling_3_flash_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_ling_3_flash_reference PUBLIC gufo_ling_3_flash)
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")
