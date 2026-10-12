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
add_ling_3_flash_test(snapshot_format gufo_ling_3_flash "cpu")
add_ling_3_flash_test(mtp_policy gufo_ling_3_flash "cpu")
add_ling_3_flash_test(mtp_sampled_chain gufo_ling_3_flash "cpu")
# Sampled MTP under the server's Bailing tool grammar; parses complete calls
# with the server's Bailing parser.
add_ling_3_flash_test(mtp_constrained_chain gufo_ling_3_flash "cpu")
target_sources(ling_3_flash_mtp_constrained_chain_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/cli/serve/bailing_tool_calls.cpp)
target_link_libraries(ling_3_flash_mtp_constrained_chain_test PRIVATE gufo_core)
set_tests_properties(ling_3_flash.mtp_constrained_chain PROPERTIES TIMEOUT 1800)
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
  foreach(name IN ITEMS session_test snapshot_test interleave_test restore_tail_test
                        logit_dump layer_check b2 decode_digest handoff_study
                        batch_segments_test reload_test)
    set(target ling_3_flash_${name})
    add_executable(${target} EXCLUDE_FROM_ALL ${name}.cpp)
    target_link_libraries(${target} PRIVATE gufo_ling_3_flash hip::host)
    list(APPEND LING_MODEL_BINARIES ${target})
    set(model_args --model ${GUFO_LING_MODEL})
    if (name STREQUAL "decode_digest")
      # Graph replay vs eager, bit for bit.
      list(APPEND model_args --graph-compare)
    endif()
    if (name STREQUAL "mtp_test" OR name STREQUAL "batch_test")
      # serving_strict_tool_* (mtp_test), serving_* (batch_test): the
      # InferenceBackend path (scheduler, runner pool, Ling runner; grammar
      # binding, MTP decode, Bailing parser), compiled in as for the
      # Flash-Next session test.
      target_sources(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/cli/serve/bailing_tool_calls.cpp
        ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
        ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
        ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
        ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
      target_link_libraries(${target} PRIVATE gufo_logging
        gufo_qwen38_flash_next gufo_deepseek_v4_flash)
    endif()
    if (name STREQUAL "mtp_distribution")
      # Tool-call arguments are read back with the server's Bailing parser.
      target_sources(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/cli/serve/bailing_tool_calls.cpp)
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
    # Batched decode without any persistent expanded K/V store, and with the
    # MTP block (its cache rows and carry are part of the compared state).
    add_test(NAME ling_3_flash.model_batch_test_no_store
      COMMAND ling_3_flash_batch_test --model ${GUFO_LING_MODEL}
        --expanded-kv-bytes 0)
    add_test(NAME ling_3_flash.model_batch_test_mtp
      COMMAND ling_3_flash_batch_test --model ${GUFO_LING_MODEL} --mtp)
    add_test(NAME ling_3_flash.model_batch_segments_test_mtp
      COMMAND ling_3_flash_batch_segments_test --model ${GUFO_LING_MODEL} --mtp)
    set_tests_properties(ling_3_flash.model_batch_test_no_store
      ling_3_flash.model_batch_test_mtp
      ling_3_flash.model_batch_segments_test_mtp PROPERTIES
      LABELS "ling_3_flash;hip;gpu;gfx1151;model" SKIP_RETURN_CODE 77
      TIMEOUT 3600)
    # A restore must also be exact across a 2048-key expanded K/V block edge
    # (Session::OnStateReplaced invalidates the derived rows).
    add_test(NAME ling_3_flash.model_snapshot_test_block_edge
      COMMAND ling_3_flash_snapshot_test --model ${GUFO_LING_MODEL}
        --prompt-tokens 2300 --max-context 4096)
    set_tests_properties(ling_3_flash.model_snapshot_test_block_edge PROPERTIES
      LABELS "ling_3_flash;hip;gpu;gfx1151;model" SKIP_RETURN_CODE 77
      TIMEOUT 3600)
  endif()
  # Draft-block op test: light model (embedding, head, block 41, block 42), runs
  # on any gfx11 GPU, compares with the CPU reference and its golden numbers.
  add_executable(ling_3_flash_mtp_ops_test EXCLUDE_FROM_ALL mtp_ops_test.cpp)
  target_link_libraries(ling_3_flash_mtp_ops_test PRIVATE
    gufo_ling_3_flash_reference hip::host)
  target_include_directories(ling_3_flash_mtp_ops_test PRIVATE
    ${PROJECT_SOURCE_DIR})
  list(APPEND LING_MODEL_BINARIES ling_3_flash_mtp_ops_test)
  if (GUFO_LING_MODEL)
    add_test(NAME ling_3_flash.model_mtp_ops
      COMMAND ling_3_flash_mtp_ops_test --model ${GUFO_LING_MODEL}
        --golden ${LING_FIXTURES}/mtp_reference_golden.txt)
    set_tests_properties(ling_3_flash.model_mtp_ops PROPERTIES
      LABELS "ling_3_flash;hip;gpu;gfx1151;model" SKIP_RETURN_CODE 77
      TIMEOUT 1800)
  endif()
  # Draft-head A/B (F16 vs Q8_0) on real text: acceptance and draft step time;
  # needs --corpus files, so it has no CTest entry.
  add_executable(ling_3_flash_mtp_head_ab EXCLUDE_FROM_ALL mtp_head_ab.cpp)
  target_link_libraries(ling_3_flash_mtp_head_ab PRIVATE gufo_ling_3_flash
    hip::host)
  list(APPEND LING_MODEL_BINARIES ling_3_flash_mtp_head_ab)
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

# MTP draft-step reference on the real block-42 weights (synthetic hidden
# inputs, no trunk): explicit target, runs only when a model is named.
add_executable(ling_3_flash_mtp_reference_test EXCLUDE_FROM_ALL
  mtp_reference_test.cpp)
target_link_libraries(ling_3_flash_mtp_reference_test PRIVATE
  gufo_ling_3_flash_reference)
target_include_directories(ling_3_flash_mtp_reference_test PRIVATE
  ${PROJECT_SOURCE_DIR})
set_source_files_properties(mtp_reference_test.cpp PROPERTIES COMPILE_OPTIONS "-O2")
if (GUFO_LING_MODEL)
  add_test(NAME ling_3_flash.model_mtp_reference
    COMMAND ling_3_flash_mtp_reference_test --model ${GUFO_LING_MODEL}
      --golden ${LING_FIXTURES}/mtp_reference_golden.txt)
  set_tests_properties(ling_3_flash.model_mtp_reference PROPERTIES
    LABELS "ling_3_flash;cpu;model" SKIP_RETURN_CODE 77 TIMEOUT 1800)
endif()

# CPU-reference MTP acceptance and expert-union probe on real text (explicit
# target; about 10 minutes per 100 generated tokens).
add_executable(ling_3_flash_mtp_acceptance_probe EXCLUDE_FROM_ALL
  mtp_acceptance_probe.cpp)
target_link_libraries(ling_3_flash_mtp_acceptance_probe PRIVATE
  gufo_ling_3_flash_reference)
target_include_directories(ling_3_flash_mtp_acceptance_probe PRIVATE
  ${PROJECT_SOURCE_DIR})
set_source_files_properties(mtp_acceptance_probe.cpp PROPERTIES
  COMPILE_OPTIONS "-O2")
