#include "src/cli/serve/logging.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace gufo::server {
namespace {

std::mutex& LogMutex() {
  static std::mutex mutex;
  return mutex;
}

// Process-wide verbosity threshold. Connection workers, the text scheduler and
// video workers all log through Logger, so the filter has to be shared state
// that is safe to read while another thread is still starting up.
std::atomic<LogLevel>& LevelStorage() {
  static std::atomic<LogLevel> level{LogLevel::kInfo};
  return level;
}

std::string SafeLine(std::string_view input, std::size_t limit = 8192) {
  std::string output;
  for (const unsigned char c : input.substr(0, limit)) {
    if (c < 0x20 || c == 0x7f) {
      constexpr char hex[] = "0123456789abcdef";
      output += "\\x";
      output += hex[c >> 4];
      output += hex[c & 15];
    } else {
      output += static_cast<char>(c);
    }
  }
  if (input.size() > limit)
    output += "...";
  return output;
}

std::string CurrentTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm calendar{};
  ::localtime_r(&seconds, &calendar);
  std::array<char, 32> buffer{};
  const auto written = std::strftime(buffer.data(), buffer.size(),
                                     "%Y-%m-%d %H:%M:%S", &calendar);
  return {buffer.data(), written};
}

std::size_t ReadMemoryKiB(const char* path, std::string_view field) {
  std::ifstream input(path);
  for (std::string line; std::getline(input, line);) {
    if (line.starts_with(field)) {
      std::istringstream value(line.substr(field.size()));
      std::size_t kib = 0;
      if (value >> kib)
        return kib;
    }
  }
  return 0;
}

}  // namespace

std::optional<std::uint64_t> MeminfoAvailableBytes(std::string_view meminfo) {
  std::optional<std::uint64_t> available_kib;
  std::uint64_t cma_free_kib = 0;
  std::istringstream lines{std::string(meminfo)};
  for (std::string line; std::getline(lines, line);) {
    std::istringstream fields(line);
    std::string key;
    std::uint64_t kib = 0;
    if (!(fields >> key >> kib))
      continue;
    if (key == "MemAvailable:")
      available_kib = kib;
    else if (key == "CmaFree:")
      cma_free_kib = kib;
  }
  if (!available_kib)
    return std::nullopt;
  return (*available_kib > cma_free_kib ? *available_kib - cma_free_kib : 0) *
         1024;
}

std::uint64_t CgroupAvailableBytes(const std::filesystem::path& root,
                                   std::string_view membership_text,
                                   std::uint64_t host_available) {
  std::uint64_t available = host_available;
  // A cgroup limit can be much smaller than the host's available memory.
  // Walk parents too: a child may say "max" beneath a limited ancestor.
  std::istringstream membership{std::string(membership_text)};
  for (std::string line; std::getline(membership, line);) {
    if (!line.starts_with("0::/"))
      continue;
    const auto relative = std::filesystem::path(line.substr(4));
    if (std::ranges::any_of(relative,
                            [](const auto& part) { return part == ".."; }))
      continue;
    for (auto path = root / relative;; path = path.parent_path()) {
      std::ifstream limit_file(path / "memory.max");
      std::ifstream used_file(path / "memory.current");
      std::uint64_t limit = 0, used = 0;
      if ((limit_file >> limit) && (used_file >> used)) {
        // memory.current includes page cache the kernel reclaims under
        // pressure. Count inactive file pages as free; the active ones may
        // still be in use, so they stay charged. Dirty and writeback pages
        // cannot be dropped until written, so they stay charged too. A
        // malformed memory.stat is ignored.
        std::ifstream stat_file(path / "memory.stat");
        std::uint64_t inactive_file = 0, unreclaimable = 0;
        bool found = false, valid = true;
        for (std::string key; stat_file >> key;) {
          std::uint64_t value = 0;
          if (!(stat_file >> value)) {
            valid = false;
            break;
          }
          if (key == "inactive_file") {
            inactive_file = value;
            found = true;
          } else if (key == "file_dirty" || key == "file_writeback") {
            unreclaimable += value;
          }
        }
        if (found && valid)
          used -= std::min(
              used, inactive_file - std::min(inactive_file, unreclaimable));
        available = std::min(available, limit > used ? limit - used : 0);
      }
      if (path == root)
        break;
    }
    break;
  }
  return available;
}

// Unified-memory device allocations draw from the same RAM, so HIP's device
// capacity is not used.
std::uint64_t EffectiveHostAvailableBytes() {
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return 0;
  std::uint64_t available = std::uint64_t(pages) * page_size;
  std::ifstream meminfo("/proc/meminfo");
  std::ostringstream meminfo_text;
  meminfo_text << meminfo.rdbuf();
  if (const auto parsed = MeminfoAvailableBytes(meminfo_text.str()))
    available = *parsed;
  std::ifstream membership("/proc/self/cgroup");
  std::ostringstream membership_text;
  membership_text << membership.rdbuf();
  return CgroupAvailableBytes("/sys/fs/cgroup", membership_text.str(),
                              available);
}

std::optional<LogLevel> LogLevelFromName(std::string_view name) {
  if (name == "debug")
    return LogLevel::kDebug;
  if (name == "info")
    return LogLevel::kInfo;
  if (name == "warn")
    return LogLevel::kWarn;
  if (name == "error")
    return LogLevel::kError;
  return std::nullopt;
}

std::string_view LogLevelName(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "debug";
    case LogLevel::kInfo:
      return "info";
    case LogLevel::kWarn:
      return "warn";
    case LogLevel::kError:
      return "error";
  }
  return "info";
}

void Logger::SetLevel(LogLevel level) {
  LevelStorage().store(level, std::memory_order_relaxed);
}

LogLevel Logger::Level() {
  return LevelStorage().load(std::memory_order_relaxed);
}

bool Logger::Enabled(LogLevel level) {
  return level >= Level();
}

void Logger::Log(LogLevel level, std::string_view component,
                 std::string_view message) {
  if (!Enabled(level)) {
    return;
  }
  static const bool color =
      std::getenv("NO_COLOR") == nullptr && ::isatty(STDERR_FILENO) != 0;
  const char* tag = level == LogLevel::kError   ? "ERROR"
                    : level == LogLevel::kWarn  ? "WARN"
                    : level == LogLevel::kDebug ? "DEBUG"
                                                : "INFO";
  const char* tint = level == LogLevel::kError   ? "\033[31m"
                     : level == LogLevel::kWarn  ? "\033[33m"
                     : level == LogLevel::kDebug ? "\033[2;36m"
                                                 : "\033[32m";
  std::ostringstream output;
  output << CurrentTimestamp() << ' ';
  if (color)
    output << tint;
  output << '[' << tag << ']';
  if (color)
    output << "\033[0m";
  output << " [" << SafeLine(component, 64) << "] " << SafeLine(message)
         << '\n';
  const std::lock_guard lock(LogMutex());
  std::clog << output.str() << std::flush;
}

void Logger::LogFormatted(LogLevel level, std::string_view component,
                          const char* format, ...) {
  if (!Enabled(level)) {
    return;
  }
  // A fixed buffer is enough for these process-authored lines and avoids an
  // allocation that a filtered call site would otherwise pay for. vsnprintf
  // reports the untruncated length, so clamp it to what was written.
  char buffer[1024];
  va_list args;
  va_start(args, format);
  const int written = std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  const std::size_t length =
      written < 0 ? 0
                  : std::min<std::size_t>(static_cast<std::size_t>(written),
                                          sizeof(buffer) - 1);
  Log(level, component, std::string_view(buffer, length));
}

std::string Logger::MemoryStatus() {
  std::ostringstream output;
  output << "rss_mib=" << ReadMemoryKiB("/proc/self/status", "VmRSS:") / 1024
         << " host_available_mib="
         << ReadMemoryKiB("/proc/meminfo", "MemAvailable:") / 1024
         << " effective_available_mib="
         << EffectiveHostAvailableBytes() / (1024 * 1024);
  return output.str();
}

void Logger::LogRequest(std::string_view id, std::string_view method,
                        std::string_view path, int status_code,
                        double duration_ms, std::string_view details,
                        std::string_view outcome, LogLevel success_level) {
  // Escalation stays upward-only: a debug-tier poll that failed still logs at
  // WARN/ERROR under the default threshold.
  const LogLevel level =
      status_code >= 500 || outcome == "stream_error"   ? LogLevel::kError
      : status_code >= 400 || outcome == "disconnected" ? LogLevel::kWarn
                                                        : success_level;
  // A filtered line must not pay for MemoryStatus(): its /proc reads and
  // cgroup walk add up on every quiet poll the tier suppresses.
  if (!Enabled(level)) {
    return;
  }
  std::ostringstream message;
  message << "request=" << id << " event=completed method=" << method
          << " path=" << path << " status=" << status_code
          << " duration_ms=" << std::fixed << std::setprecision(1)
          << duration_ms << " outcome=" << outcome;
  if (!details.empty())
    message << ' ' << details;
  message << ' ' << MemoryStatus();
  Log(level, "http", message.str());
}

}  // namespace gufo::server
