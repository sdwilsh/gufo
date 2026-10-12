#ifndef GUFO_CORE_RUNTIME_ENVIRONMENT_HPP_
#define GUFO_CORE_RUNTIME_ENVIRONMENT_HPP_

#include <string>
#include <string_view>

namespace gufo::core {

/// The HIP runtime and comgr take their scratch directory from $TMPDIR and
/// crash inside the first HIP call (hipHostMalloc during weight upload) when it
/// names a directory that does not exist. Returns an empty string when TMPDIR
/// is unset or empty (the runtime then uses /tmp) or names an existing
/// writable directory, otherwise a one-line error naming the variable and the
/// path. The directory is never created here: that would hide a deployment
/// mistake. ($XDG_CACHE_HOME is not checked: the GGUF digest cache and comgr
/// create what they need and run without a cache when they cannot.)
[[nodiscard]] std::string CheckTempDirectory(std::string_view tmpdir);

/// CheckTempDirectory() on the process environment. Call it before the
/// first HIP API call; on a problem it prints "gufo: <error>" to stderr and
/// returns false (the caller exits non-zero).
[[nodiscard]] bool RequireTempDirectory();

}  // namespace gufo::core

#endif  // GUFO_CORE_RUNTIME_ENVIRONMENT_HPP_
