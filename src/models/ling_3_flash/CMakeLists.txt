# Ling-3.0-flash (bailingmoe3) model package.
add_library(gufo_ling_3_flash STATIC
  chat_template.cpp
  config.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_ling_3_flash PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ling_3_flash PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  set(LING_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  # One translation unit per kernel group (kda, mla, moe, dense); each
  # workstream appends its own file here.
  set(LING_HIP_SOURCES)
  if (LING_HIP_SOURCES)
    target_sources(gufo_ling_3_flash PRIVATE ${LING_HIP_SOURCES})
    set_source_files_properties(${LING_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    target_compile_options(gufo_ling_3_flash PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  endif()
  target_link_libraries(gufo_ling_3_flash PUBLIC hip::host roc::hipblas)
endif()
