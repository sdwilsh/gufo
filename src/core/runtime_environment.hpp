#ifndef GUFO_CORE_RUNTIME_ENVIRONMENT_HPP_
#define GUFO_CORE_RUNTIME_ENVIRONMENT_HPP_

#include <string>
#include <string_view>

namespace gufo::core {

/// The HIP runtime and comgr take their scratch directory from $TMPDIR and
/// crash inside the first HIP call (hipHostMalloc during weight upload) when it
/// names a directory that does not exist. Returns an empty string when TMPDIR
/// is unset, is empty, or names an existing writable directory. An unset or
/// empty TMPDIR makes the runtime use /tmp. Otherwise returns a one-line error
/// naming the variable and the path. The function never creates the
/// directory, because that would hide a deployment mistake. It does not check
/// $XDG_CACHE_HOME: the GGUF digest cache and comgr create what they need and
/// run without a cache when they cannot.
[[nodiscard]] std::string CheckTempDirectory(std::string_view tmpdir);

/// Runs CheckTempDirectory() on the process environment. Call it before the
/// first HIP API call. On a problem it prints "gufo: <error>" to stderr and
/// returns false, and the caller exits non-zero.
[[nodiscard]] bool RequireTempDirectory();

}  // namespace gufo::core

#endif  // GUFO_CORE_RUNTIME_ENVIRONMENT_HPP_
