#ifndef GUFO_CLI_BENCH_HPP_
#define GUFO_CLI_BENCH_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::cli {

struct BenchOptions {
  std::string model_path;
  std::vector<std::size_t> n_prompts{2048};
  std::vector<std::size_t> n_gens{128};
  std::vector<std::size_t> n_depths{0};
  std::vector<std::size_t> concurrency{1};
  std::size_t repetitions{1};
  /// Ling-3.0-flash tokens per forward pass; 0 = ling::kCliPrefillBatch.
  std::size_t max_batch{0};
  /// Ling-3.0-flash persistent expanded K/V budget in bytes
  /// (ModelOptions::expanded_kv_bytes).
  std::uint64_t expanded_kv_bytes{std::uint64_t{12} << 30};
  std::size_t validate_prefill_tokens{0};
  std::string speculative_backend{""};
  /// True when --speculative was given (including "off"); otherwise Ling
  /// selects MTP when its GGUF carries the block-42 weights.
  bool speculative_explicit{false};
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  sampling::SamplingConfig sampling{.seed = 0};
  /// Ling-3.0-flash: context tokens come from this text file (tokenized, not
  /// repeated) instead of pseudo-random tokens.
  std::string text_file;
  std::string output_format{"md"};
  bool verbose{false};
  /// Ling-3.0-flash (-o jsonl): per-operation device-time records of one eager
  /// decode run and one serialised prefill chunk per point.
  bool op_profile{false};
};

void PrintBenchHelp(std::string_view program_name);
std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);
int RunBench(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BENCH_HPP_
