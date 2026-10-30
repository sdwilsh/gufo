# Cargo builds the decoder; CMake imports the static library it produces.
# Nix builds point CARGO_HOME at vendored crates, see
# .devops/nix/image-decode-cargo.nix. Other builds fetch the locked crates
# from crates.io.
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(cargo_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(image_decode_library
  "${cargo_target_dir}/release/${CMAKE_STATIC_LIBRARY_PREFIX}gufo_image_decode${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE image_decode_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs")

add_custom_command(
  OUTPUT "${image_decode_library}"
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${cargo_target_dir}"
    "${CARGO_EXECUTABLE}" build --release --locked
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS ${image_decode_sources}
    "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock"
  COMMENT "Building the Rust image decoder"
  VERBATIM)
add_custom_target(gufo_image_decode_cargo DEPENDS "${image_decode_library}")

add_library(gufo_image_decode STATIC IMPORTED GLOBAL)
set_target_properties(gufo_image_decode PROPERTIES
  IMPORTED_LOCATION "${image_decode_library}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_dependencies(gufo_image_decode gufo_image_decode_cargo)
# Rust's standard library needs these system libraries when linked statically.
if (WIN32)
  target_link_libraries(gufo_image_decode INTERFACE
    ws2_32 userenv ntdll bcrypt advapi32)
else()
  target_link_libraries(gufo_image_decode INTERFACE
    Threads::Threads ${CMAKE_DL_LIBS} m)
endif()
