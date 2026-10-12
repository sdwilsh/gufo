#include "src/core/runtime_environment.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <system_error>

namespace gufo::core {
namespace {

std::string CheckDirectory(const char* name, std::string_view value) {
  const std::string path(value);
  std::error_code ignored;
  if (!std::filesystem::is_directory(path, ignored)) {
    return std::string(name) + "=" + path +
           " is not an existing directory; HIP/comgr crash on first use "
           "without it: run `mkdir -p " +
           path + "` or unset " + name;
  }
  if (::access(path.c_str(), W_OK | X_OK) != 0) {
    return std::string(name) + "=" + path +
           " is not a writable directory; HIP/comgr need to write there: fix "
           "its permissions or point " +
           name + " elsewhere";
  }
  return {};
}

}  // namespace

std::string CheckTempDirectory(std::string_view tmpdir) {
  return tmpdir.empty() ? std::string() : CheckDirectory("TMPDIR", tmpdir);
}

bool RequireTempDirectory() {
  const char* tmpdir = std::getenv("TMPDIR");
  const auto error = CheckTempDirectory(tmpdir != nullptr ? tmpdir : "");
  if (error.empty())
    return true;
  std::cerr << "gufo: " << error << '\n';
  return false;
}

}  // namespace gufo::core
