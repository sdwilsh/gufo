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

add_ling_3_flash_test(config gufo_ling_3_flash "cpu")
