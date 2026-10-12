add_custom_target(ling_3_flash_tests)

function(add_ling_3_flash_test name library labels)
  set(target ling_3_flash_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME ling_3_flash.${name} COMMAND ${target})
  set_tests_properties(ling_3_flash.${name} PROPERTIES
    LABELS "ling_3_flash;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(ling_3_flash_tests ${target})
endfunction()

set(LING_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

add_ling_3_flash_test(config gufo_ling_3_flash "cpu")
add_ling_3_flash_test(tokenizer gufo_ling_3_flash "cpu")
target_compile_definitions(ling_3_flash_tokenizer_test PRIVATE
  GUFO_LING_VOCAB_GGUF="${LING_FIXTURES}/ling-vocab.gguf.fixture"
  GUFO_LING_TOKENIZER_GOLDENS="${LING_FIXTURES}/tokenizer_goldens.json")
add_ling_3_flash_test(chat_template gufo_ling_3_flash "cpu")
target_compile_definitions(ling_3_flash_chat_template_test PRIVATE
  GUFO_LING_VOCAB_GGUF="${LING_FIXTURES}/ling-vocab.gguf.fixture"
  GUFO_LING_CHAT_TEMPLATE="${LING_FIXTURES}/chat_template.jinja"
  GUFO_LING_CHAT_GOLDENS="${LING_FIXTURES}/chat_template_goldens.json")

if (ENGINE_ENABLE_HIP)
  foreach(ops dense kda mla moe)
    add_ling_3_flash_test(${ops}_ops gufo_ling_3_flash "hip;gpu;gfx1151")
  endforeach()
  set_tests_properties(ling_3_flash.dense_ops PROPERTIES TIMEOUT 900)
  set_tests_properties(ling_3_flash.moe_ops PROPERTIES TIMEOUT 1800)
  # Fast bit-exactness subset of the prefill MoE path.
  add_test(NAME ling_3_flash.moe_ops_invariance
    COMMAND ling_3_flash_moe_ops_test --invariance)
  set_tests_properties(ling_3_flash.moe_ops_invariance PROPERTIES
    LABELS "ling_3_flash;hip;gpu;gfx1151" SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()

# Full-model checks. Built explicitly (target ling_3_flash_model_tests), never
# by the default build. Each binary exits 77 without --model; the CTest entries
# exist only when a model is named: -DGUFO_LING_MODEL=/path/first-shard.gguf.
if (ENGINE_ENABLE_HIP)
  set(LING_MODEL_BINARIES)
  foreach(name IN ITEMS session_test logit_dump layer_check b2 decode_digest
                        handoff_study)
    set(target ling_3_flash_${name})
    add_executable(${target} EXCLUDE_FROM_ALL ${name}.cpp)
    target_link_libraries(${target} PRIVATE gufo_ling_3_flash hip::host)
    list(APPEND LING_MODEL_BINARIES ${target})
    set(model_args --model ${GUFO_LING_MODEL})
    if (name STREQUAL "decode_digest")
      # Graph replay vs eager, bit for bit.
      list(APPEND model_args --graph-compare)
    endif()
    if (GUFO_LING_MODEL)
      add_test(NAME ling_3_flash.model_${name}
        COMMAND ${target} ${model_args})
      set_tests_properties(ling_3_flash.model_${name} PROPERTIES
        LABELS "ling_3_flash;hip;gpu;gfx1151;model" SKIP_RETURN_CODE 77
        TIMEOUT 3600)
    endif()
  endforeach()
  if (GUFO_LING_MODEL)
    # Crosses the clamped 96-split attention plan (24,576 keys) between the
    # step that captures a graph and the steps that replay it.
    add_test(NAME ling_3_flash.model_decode_digest_long
      COMMAND ling_3_flash_decode_digest --model ${GUFO_LING_MODEL}
        --graph-compare --layers 8 --max-batch 512 --max-context 25000
        --prompt-tokens 24570 --decode-tokens 12)
    set_tests_properties(ling_3_flash.model_decode_digest_long PROPERTIES
      LABELS "ling_3_flash;hip;gpu;gfx1151;model" SKIP_RETURN_CODE 77
      TIMEOUT 3600)
  endif()
  add_custom_target(ling_3_flash_model_tests DEPENDS ${LING_MODEL_BINARIES})
endif()

# Oracle probe: per-layer dumps and logits of the float32 reference. Built
# explicitly (`--target ling_3_flash_reference_probe`), never by routine checks.
add_executable(ling_3_flash_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(ling_3_flash_reference_probe PRIVATE
  gufo_ling_3_flash_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

# Synthetic operator checks of the scalar reference; explicit target, run
# `ling_3_flash_cpu_ops_test` by hand (it is not part of routine ctest).
add_executable(ling_3_flash_cpu_ops_test EXCLUDE_FROM_ALL cpu_ops_test.cpp)
target_link_libraries(ling_3_flash_cpu_ops_test PRIVATE
  gufo_ling_3_flash_reference)
