#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_MODEL_TOOL_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_MODEL_TOOL_HPP_

// Shared code of the full-model Ling-3.0-Flash binaries (session_test,
// logit_dump, layer_check): argument parsing, one JSON report on stdout,
// stderr progress, a watchdog so a hung GPU cannot stall a pod forever, and
// `--hold-seconds` so the pod can stay up for `kubectl cp` after the run.
//
// Common options: --model FILE (absent: exit 77), --max-context N,
// --max-batch N, --layers N / --layer-first M (debug: only blocks
// [M, M+N) are uploaded; forward runs need M = 0),
// --mtp (also upload the MTP block and run the MTP session path),
// --expanded-kv-bytes N (the sessions' shared persistent expanded K/V budget,
// default ModelOptions::expanded_kv_bytes),
// --timeout-seconds S (default 3300), --hold-seconds S (default 0),
// --partial-file F (default $TMPDIR/<tool>_partial.json: the latest report,
// rewritten at every update; a fatal signal prints it as {"status":"crashed"}).
// Exit codes: 0 pass, 1 failed check or error, 3 watchdog timeout, 77 skip.

#include <hip/hip_runtime.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/engine.hpp"

namespace gufo::tests::ling {

namespace json = gufo::json;
namespace l3 = gufo::models::ling_3_flash;
using l3::ReadF32Dump;

/// Unsigned integer to JSON (the json::Value constructors are ambiguous).
[[nodiscard]] inline json::Value U(std::uint64_t v) {
  return json::Value(static_cast<double>(v));
}

class Args {
public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        positional_.push_back(a);
        continue;
      }
      const auto eq = a.find('=');
      if (eq != std::string::npos) {
        kv_[a.substr(0, eq)] = a.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[a] = argv[++i];
      } else {
        kv_[a] = "";
      }
    }
  }
  [[nodiscard]] bool Has(const std::string& key) const {
    return kv_.count(key) > 0;
  }
  [[nodiscard]] std::string Str(const std::string& key,
                                const std::string& def = "") const {
    const auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second;
  }
  [[nodiscard]] double Num(const std::string& key, double def) const {
    const auto it = kv_.find(key);
    return it == kv_.end() || it->second.empty()
               ? def
               : std::atof(it->second.c_str());
  }

private:
  std::map<std::string, std::string> kv_;
  std::vector<std::string> positional_;
};

[[nodiscard]] inline std::string TmpDir() {
  const char* t = std::getenv("TMPDIR");
  return t != nullptr && *t != '\0' ? t : "/tmp";
}

/// Rounds to `digits` significant digits and maps non-finite values to null.
[[nodiscard]] inline json::Value Round(double v, int digits = 4) {
  if (!std::isfinite(v)) {
    return json::Value();
  }
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%.*g", digits, v);
  return json::Value(std::strtod(buf, nullptr));
}

[[nodiscard]] inline long PeakRssKiB() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      return std::atol(line.c_str() + 6);
    }
  }
  return -1;
}

[[nodiscard]] inline double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// Thread-safe JSON report, printed once on stdout. A child report mirrors
/// every update into its parent under `section`, so partial results of a
/// sub-tool survive a watchdog timeout of the whole run.
class Report {
public:
  Report() : root_(json::Value::object()) {}
  Report(Report* parent, std::string section)
      : root_(json::Value::object()),
        parent_(parent),
        section_(std::move(section)) {}
  void Set(const std::string& key, json::Value value) {
    std::lock_guard lock(mutex_);
    root_[key] = std::move(value);
    if (parent_ != nullptr) {
      parent_->Set(section_, root_);
    } else {
      Render();
    }
  }
  /// Best-effort dump of the latest report from a fatal signal handler: a
  /// crashed or killed run still returns whatever was measured so far.
  static void WriteCrashDump(int signal_number) {
    const std::string& marker = StdoutMarker();
    if (!marker.empty()) {
      (void)!::write(1, marker.data(), marker.size());
    }
    const char head[] = "{\"status\":\"crashed\",\"signal\":";
    const std::string num = std::to_string(signal_number);
    const char mid[] = ",\"partial\":";
    (void)!::write(1, head, sizeof(head) - 1);
    (void)!::write(1, num.data(), num.size());
    (void)!::write(1, mid, sizeof(mid) - 1);
    const std::string& last = LastRender();
    if (last.empty()) {
      (void)!::write(1, "null", 4);
    } else {
      (void)!::write(1, last.data(), last.size());
    }
    (void)!::write(1, "}\n", 2);
  }
  static std::string& LastRender() {
    static std::string text;
    return text;
  }
  /// When set, the report is written once, on stdout only, as one line that
  /// starts with this marker (run-suite strips it); otherwise it is written to
  /// stdout and echoed to stderr after `[report-json] `.
  static std::string& StdoutMarker() {
    static std::string marker;
    return marker;
  }
  static std::string& PartialFile() {
    static std::string path;
    return path;
  }
  [[nodiscard]] std::string Status() {
    std::lock_guard lock(mutex_);
    const json::Value* s = root_.find("status");
    return s != nullptr ? s->str() : std::string();
  }
  /// Prints the report once; later calls do nothing. Returns whether this call
  /// printed.
  bool Emit() {
    std::lock_guard lock(mutex_);
    if (emitted_ || parent_ != nullptr) {
      return false;
    }
    emitted_ = true;
    root_["host_peak_rss_mib"] =
        Round(static_cast<double>(PeakRssKiB()) / 1024.0, 6);
    std::string text;
    try {
      text = root_.dump();
    } catch (const std::exception& e) {
      text = std::string("{\"status\":\"error\",\"error\":\"report: ") +
             e.what() + "\"}";
    }
    std::fflush(stderr);
    std::fputs(StdoutMarker().c_str(), stdout);
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    if (!StdoutMarker().empty()) {
      return true;
    }
    // run-suite keeps a tool's stdout until the whole suite ends; echo the
    // report to stderr (pod logs) so it is readable before any hold.
    std::fputs("[report-json] ", stderr);
    std::fwrite(text.data(), 1, text.size(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    return true;
  }

private:
  /// Keeps the latest rendering for crashes and mirrors it to a file.
  void Render() {
    try {
      LastRender() = root_.dump();
    } catch (const std::exception&) {
      return;
    }
    if (!PartialFile().empty()) {
      std::ofstream out(PartialFile() + ".tmp", std::ios::binary);
      out << LastRender() << '\n';
      out.close();
      std::error_code ec;
      std::filesystem::rename(PartialFile() + ".tmp", PartialFile(), ec);
    }
  }

  std::mutex mutex_;
  json::Value root_;
  Report* parent_{nullptr};
  std::string section_;
  bool emitted_{false};
};

inline void Hold(const Args& args) {
  const double hold = args.Num("--hold-seconds", 0.0);
  if (hold > 0.0) {
    std::fprintf(stderr, "[hold] sleeping %.0f s for file collection\n", hold);
    std::this_thread::sleep_for(std::chrono::duration<double>(hold));
  }
}

class Watchdog {
public:
  Watchdog(Report& report, const Args& args, double seconds)
      : thread_([this, &report, &args, seconds] {
          std::unique_lock lock(mutex_);
          if (cv_.wait_for(lock, std::chrono::duration<double>(seconds),
                           [this] { return done_; })) {
            return;
          }
          std::fprintf(stderr, "[watchdog] no completion after %.0f s\n",
                       seconds);
          report.Set("status", "timeout");
          report.Set("watchdog_seconds", Round(seconds, 6));
          report.Emit();
          Hold(args);
          std::_Exit(3);
        }) {}
  ~Watchdog() { Stop(); }
  void Stop() {
    {
      std::lock_guard lock(mutex_);
      done_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool done_{false};
  std::thread thread_;
};

/// Loads the model with the common options and records memory and timing.
[[nodiscard]] inline std::shared_ptr<l3::Model> LoadModel(
    const Args& args, Report& report, std::uint32_t default_batch,
    std::uint32_t default_context, bool force_mtp = false,
    std::uint64_t default_expanded_kv_bytes =
        l3::ModelOptions{}.expanded_kv_bytes) {
  l3::ModelOptions options;
  options.max_batch =
      static_cast<std::uint32_t>(args.Num("--max-batch", default_batch));
  options.max_context =
      static_cast<std::uint32_t>(args.Num("--max-context", default_context));
  options.layer_first =
      static_cast<std::uint32_t>(args.Num("--layer-first", 0));
  options.layer_limit = static_cast<std::uint32_t>(args.Num("--layers", 0));
  options.mtp = force_mtp || args.Has("--mtp");
  options.expanded_kv_bytes = static_cast<std::uint64_t>(args.Num(
      "--expanded-kv-bytes", static_cast<double>(default_expanded_kv_bytes)));
  std::string error;
  auto model = l3::Model::Load(args.Str("--model"), options, &error);
  if (!model) {
    std::fprintf(stderr, "model load failed: %s\n", error.c_str());
    report.Set("status", "error");
    report.Set("error", "model load failed: " + error);
    return nullptr;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  (void)hipMemGetInfo(&free_bytes, &total_bytes);
  auto info = json::Value::object();
  info["path"] = args.Str("--model");
  info["layers_resident"] = U(model->LayerCount());
  info["first_layer"] = U(model->FirstLayer());
  info["load_seconds"] = Round(model->LoadSeconds(), 5);
  info["upload_seconds"] = Round(model->UploadSeconds(), 5);
  info["weights_gib"] =
      Round(static_cast<double>(model->ResidentBytes()) / (1ULL << 30), 5);
  info["scratch_gib"] =
      Round(static_cast<double>(model->ScratchBytes()) / (1ULL << 30), 5);
  info["max_batch"] = U(options.max_batch);
  info["max_context"] = U(options.max_context);
  info["device_free_gib"] =
      Round(static_cast<double>(free_bytes) / (1ULL << 30), 5);
  info["device_total_gib"] =
      Round(static_cast<double>(total_bytes) / (1ULL << 30), 5);
  report.Set("model", std::move(info));
  std::fprintf(stderr, "[model] loaded in %.1f s (peak RSS %.2f GiB)\n",
               model->LoadSeconds(),
               static_cast<double>(PeakRssKiB()) / (1024.0 * 1024.0));
  return model;
}

using ToolBody = int (*)(const Args&, Report&);

/// Runs a tool body: skip without --model, watchdog, exception guard, one JSON
/// report on stdout, optional hold.
inline int RunTool(int argc, char** argv, const char* tool, ToolBody body) {
  const Args args(argc, argv);
  if (!args.Has("--model")) {
    std::fprintf(stderr, "%s: no --model given, skipping\n", tool);
    return 77;
  }
  // Before the first HIP call: a missing $TMPDIR crashes the runtime.
  if (!gufo::core::RequireTempDirectory())
    return 1;
  Report report;
  Report::PartialFile() =
      args.Str("--partial-file", TmpDir() + "/" + tool + "_partial.json");
  for (const int sig : {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGTERM}) {
    struct sigaction sa{};
    sa.sa_handler = [](int n) {
      Report::WriteCrashDump(n);
      ::signal(n, SIG_DFL);
      ::raise(n);
    };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    ::sigaction(sig, &sa, nullptr);
  }
  report.Set("tool", tool);
  report.Set("status", "running");
  Watchdog watchdog(report, args, args.Num("--timeout-seconds", 3300.0));
  int code = 1;
  try {
    code = body(args, report);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: exception: %s\n", tool, e.what());
    report.Set("status", "error");
    report.Set("error", std::string(e.what()));
    code = 1;
  }
  watchdog.Stop();
  report.Emit();
  Hold(args);
  return code;
}

[[nodiscard]] inline std::vector<std::int32_t> ParseIds(
    const std::string& text) {
  std::vector<std::int32_t> out;
  std::string cur;
  for (const char ch : text + ",") {
    if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r' ||
        ch == '[' || ch == ']') {
      if (!cur.empty()) {
        out.push_back(static_cast<std::int32_t>(std::atoll(cur.c_str())));
        cur.clear();
      }
    } else {
      cur.push_back(ch);
    }
  }
  return out;
}

[[nodiscard]] inline std::string ReadText(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

/// ||a - b|| / ||b|| in double precision.
[[nodiscard]] inline double RelL2(std::span<const float> a,
                                  std::span<const float> b) {
  double num = 0.0;
  double den = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num / std::max(den, 1e-300));
}

[[nodiscard]] inline bool AllFinite(std::span<const float> v) {
  return std::all_of(v.begin(), v.end(),
                     [](float x) { return std::isfinite(x); });
}

}  // namespace gufo::tests::ling

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_MODEL_TOOL_HPP_
