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

set(LING_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
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
  # --- WS4a: kda ---
  add_ling_3_flash_test(kda_ops gufo_ling_3_flash "hip;gpu;gfx1151")
  # --- end WS4a ---
  # --- mla (WS4b) ---
  add_ling_3_flash_test(mla_ops gufo_ling_3_flash "hip;gpu;gfx1151")
  # --- end mla ---
  # --- WS4c: MoE routing and routed-expert op tests ---
  add_ling_3_flash_test(moe_ops gufo_ling_3_flash "hip;gpu;gfx1151")
  # --- end WS4c ---
  # --- WS4d dense kernel op tests ---
  add_ling_3_flash_test(dense_ops gufo_ling_3_flash "hip;gpu;gfx1151")
  set_tests_properties(ling_3_flash.dense_ops PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 600)
  # --- end WS4d ---
endif()

# Full-model checks. Built explicitly (target ling_3_flash_model_tests), never
# by the default build. Each binary exits 77 without --model; the CTest entries
# exist only when a model is named: -DGUFO_LING_MODEL=/path/first-shard.gguf.
if (ENGINE_ENABLE_HIP)
  set(LING_MODEL_BINARIES)
  foreach(name IN ITEMS session_test logit_dump layer_check b2)
    set(target ling_3_flash_${name})
    add_executable(${target} EXCLUDE_FROM_ALL ${name}.cpp)
    target_link_libraries(${target} PRIVATE gufo_ling_3_flash hip::host)
    list(APPEND LING_MODEL_BINARIES ${target})
    if (GUFO_LING_MODEL)
      add_test(NAME ling_3_flash.model_${name}
        COMMAND ${target} --model ${GUFO_LING_MODEL})
      set_tests_properties(ling_3_flash.model_${name} PROPERTIES
        LABELS "ling_3_flash;hip;gpu;gfx1151;model" SKIP_RETURN_CODE 77
        TIMEOUT 3600)
    endif()
  endforeach()
  add_custom_target(ling_3_flash_model_tests DEPENDS ${LING_MODEL_BINARIES})
endif()
