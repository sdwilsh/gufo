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
endif()
