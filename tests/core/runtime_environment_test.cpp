#include "src/core/runtime_environment.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

bool Contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  using gufo::core::CheckTempDirectory;
  const fs::path root =
      fs::temp_directory_path() /
      ("gufo_runtime_environment_test_" + std::to_string(::getpid()));
  fs::create_directories(root);
  const std::string good = (root / "tmp").string();
  fs::create_directories(good);
  const std::string missing = (root / "missing").string();
  const std::string file = (root / "file").string();
  std::ofstream(file) << "x";

  // Unset or empty leaves the runtime's default (/tmp) alone.
  Expect(CheckTempDirectory("").empty(), "empty TMPDIR");
  Expect(CheckTempDirectory(good).empty(), "existing writable directory");

  // A missing directory names the variable and path and says how to fix it.
  auto error = CheckTempDirectory(missing);
  Expect(Contains(error, "TMPDIR=" + missing) && Contains(error, "mkdir -p") &&
             error.find('\n') == std::string::npos,
         "missing TMPDIR is a one-line error naming the variable and path");
  Expect(!fs::exists(missing), "the directory is not created");
  error = CheckTempDirectory(file);
  Expect(Contains(error, "TMPDIR=" + file), "a file is not a directory");

  // A read-only directory fails the writability check (skipped for root).
  if (::geteuid() != 0) {
    const std::string readonly = (root / "readonly").string();
    fs::create_directories(readonly);
    fs::permissions(readonly, fs::perms::owner_read | fs::perms::owner_exec);
    error = CheckTempDirectory(readonly);
    Expect(Contains(error, "not a writable directory"),
           "read-only directory is rejected");
    fs::permissions(readonly, fs::perms::owner_all);
  }

  // The environment form reads the process variables.
  ::setenv("TMPDIR", missing.c_str(), 1);
  Expect(!gufo::core::RequireTempDirectory(),
         "RequireTempDirectory fails for a missing TMPDIR");
  ::setenv("TMPDIR", good.c_str(), 1);
  Expect(gufo::core::RequireTempDirectory(),
         "RequireTempDirectory passes for an existing TMPDIR");
  ::unsetenv("TMPDIR");
  Expect(gufo::core::RequireTempDirectory(), "unset TMPDIR passes");

  std::error_code ignored;
  fs::remove_all(root, ignored);
  if (failures == 0)
    std::puts("runtime_environment_test: all passed");
  return failures == 0 ? 0 : 1;
}
