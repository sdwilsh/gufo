#include "src/cli/bench/bench.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/testing/compare/logit_comparator.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/ling_3_flash/engine.hpp"
#include "src/models/ling_3_flash/weights.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#endif

namespace gufo::cli {

namespace {

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << load_seconds << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

std::optional<std::vector<std::size_t>> ParseCommaSeparatedSizes(
    std::string_view text, bool keep_zero = false) {
  if (text.empty() || text.back() == ',')
    return std::nullopt;
  std::vector<std::size_t> result;
  for (std::size_t start = 0; start < text.size();) {
    const auto comma = text.find(',', start);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto part = text.substr(start, end - start);
    std::size_t value = 0;
    const auto [ptr, ec] =
        std::from_chars(part.data(), part.data() + part.size(), value);
    if (ec != std::errc{} || ptr != part.data() + part.size())
      return std::nullopt;
    if (value != 0 || keep_zero)
      result.push_back(value);
    start = end + 1;
  }
  return result;
}

void RegisterBenchOptions(ArgParser& parser, BenchOptions& opt,
                          bool& explicit_p, bool& explicit_n,
                          bool& speculative_explicit) {
  parser.AddOption("-m", "--model", "PATH",
                   "Path to GGUF model file (required)", "Model",
                   &opt.model_path);

  parser.AddCustomOption(
      "-p", "--n-prompt", "n,n,...",
      "Prompt token lengths to benchmark (default: 2048)", "Workload",
      [&opt, &explicit_p](std::string_view flag, std::string_view val,
                          std::string* error) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val);
        if (!sizes) {
          if (error)
            *error = "Invalid token counts for " + std::string(flag);
          return false;
        }
        opt.n_prompts = std::move(*sizes);
        explicit_p = true;
        return true;
      });

  parser.AddCustomOption(
      "-n", "--n-gen", "n,n,...",
      "Number of text generation tokens to benchmark (default: 128)",
      "Workload",
      [&opt, &explicit_n](std::string_view flag, std::string_view val,
                          std::string* error) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val);
        if (!sizes) {
          if (error)
            *error = "Invalid token counts for " + std::string(flag);
          return false;
        }
        opt.n_gens = std::move(*sizes);
        explicit_n = true;
        return true;
      });

  parser.AddCustomOption(
      "-d", "--n-depth", "n,n,...",
      "Context depths prepared before timed region (default: 0)", "Workload",
      [&opt](std::string_view flag_name, std::string_view val,
             std::string* err) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val, true);
        if (!sizes) {
          if (err != nullptr) {
            *err = "Invalid argument for " + std::string(flag_name);
          }
          return false;
        }
        opt.n_depths = std::move(*sizes);
        return true;
      });

  parser.AddCustomOption(
      "-c", "--concurrency", "n,n,...",
      "DS4 simultaneous requests, 1..8 (default: 1); pp is aggregate, tg per "
      "user",
      "Workload",
      [&opt](std::string_view, std::string_view value, std::string* error) {
        auto sizes = ParseCommaSeparatedSizes(value, true);
        if (!sizes || sizes->empty() ||
            std::any_of(sizes->begin(), sizes->end(),
                        [](auto size) { return size < 1 || size > 8; })) {
          if (error)
            *error =
                "--concurrency requires DS4 request counts between 1 and 8";
          return false;
        }
        opt.concurrency = std::move(*sizes);
        return true;
      });

  parser.AddCustomOption(
      "-o", "--output", "FORMAT",
      "Result format: md (default) or jsonl (Ling-3.0-flash only; "
      "llama-bench compatible records with per-repetition samples)",
      "Workload",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        if (value != "md" && value != "jsonl") {
          if (error != nullptr)
            *error = "--output must be md or jsonl";
          return false;
        }
        opt.output_format = value;
        return true;
      });

  parser.AddOption(
      "", "--text-file", "PATH",
      "Ling-3.0-flash: take the context tokens from this text file (tokenized, "
      "never repeated; it must cover the deepest depth plus the workload) "
      "instead of pseudo-random tokens. Needed for MTP acceptance that "
      "reflects real text: greedy continuation of pseudo-random tokens "
      "over-states it",
      "Workload", &opt.text_file);
  parser.AddOption("", "--max-batch", "N",
                   "Ling-3.0-flash prompt tokens per forward pass (default: "
                   "512, the CLI/server prefill chunk; llama.cpp's n_ubatch)",
                   "Workload", &opt.max_batch);
  parser.AddOption("", "--expanded-kv-bytes", "N",
                   "Ling-3.0-flash device budget for persistent expanded K/V "
                   "rows of the MLA layers (112 KiB per cached token; "
                   "default: 12 GiB, 0 disables)",
                   "Workload", &opt.expanded_kv_bytes);

  parser.AddOption(
      "-r", "--repetitions", "N",
      "Repetitions per test point for variance reduction (default: 1)",
      "Workload", &opt.repetitions);

  parser.AddCustomOption(
      "", "--validate-prefill", "N",
      "Compare batched prefill logits against sequential reference",
      "Validation",
      [&opt](std::string_view, std::string_view val, std::string* err) -> bool {
        std::size_t num = 0;
        const auto [ptr, ec] =
            std::from_chars(val.data(), val.data() + val.size(), num);
        if (ec != std::errc{} || ptr != val.data() + val.size() || num == 0) {
          if (err != nullptr) {
            *err = "Invalid argument for --validate-prefill";
          }
          return false;
        }
        opt.validate_prefill_tokens = num;
        return true;
      });

  const auto parse_speculative_backend =
      [&opt, &speculative_explicit](std::string_view, std::string_view value,
                                    std::string* error) -> bool {
    speculative_explicit = true;
    if (value == "off") {
      opt.speculative_backend.clear();
    } else if (value == "dspark" || value == "dflash2" || value == "mtp") {
      opt.speculative_backend = value;
    } else {
      if (error != nullptr)
        *error = "Unknown speculative backend: " + std::string(value);
      return false;
    }
    return true;
  };
  parser.AddCustomOption("", "--speculative", "MODE",
                         "Draft backend: dflash2, mtp, dspark, or off",
                         "Speculative", parse_speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to Qwen DFlash2 GGUF file", "Speculative",
                   &opt.dflash_model_path);
  parser.AddOption(
      "", "--draft-policy", "POLICY",
      "DFlash2 block length: fixed or adaptive (default: adaptive)",
      "Speculative", &opt.draft_policy);
  parser.AddOption("", "--dspark-model", "PATH",
                   "DeepSeek V4 Flash DSpark support GGUF", "Speculative",
                   &opt.dspark_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to quantized Qwen MTP draft head GGUF file (Ling-3.0-"
                   "flash carries its MTP block in the main GGUF: "
                   "--speculative mtp alone enables it)",
                   "Speculative", &opt.mtp_model_path);
  parser.AddCustomOption(
      "", "--draft-tokens", "N",
      "Maximum speculative draft tokens per verification step (default: 7)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error = "Invalid argument for --draft-tokens";
          }
          return false;
        }
        opt.draft_tokens = count;
        return true;
      });

  parser.AddCustomOption(
      "", "--min-draft-tokens", "N", "Adaptive draft floor (default: 1)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error = "Invalid argument for --min-draft-tokens";
          }
          return false;
        }
        opt.min_draft_tokens = count;
        return true;
      });
  RegisterSamplingOptions(parser, &opt.sampling, "Sampling", false);
  parser.AddFlag("", "--op-profile",
                 "Ling-3.0-flash (-o jsonl): also emit op_profile records, "
                 "the per-operation device time of one eager decode run and "
                 "of one prefill chunk per point (serialised, so slower than "
                 "the timed runs; the MLA prefill attention is broken down as "
                 "mla.attention/...)",
                 "Workload", &opt.op_profile);
  parser.AddFlag("-v", "--verbose",
                 "Print detailed timing, latency breakdown, and tok/s metrics",
                 "General", &opt.verbose);

  parser.SetPositionalHandler(
      [&opt](std::string_view arg, std::string*) -> bool {
        opt.model_path = std::string(arg);
        return true;
      });
}

#if defined(ENGINE_ENABLE_HIP)
struct BenchStats {
  double mean{0.0};
  double stddev{0.0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) {
    return {.mean = 0.0, .stddev = 0.0};
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (values.size() <= 1) {
    return {.mean = mean, .stddev = 0.0};
  }
  double sq_sum = 0.0;
  for (const double v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

std::vector<tokenization::TokenId> MakeBenchmarkTokens(
    std::size_t count, std::size_t start_pos = 0) {
  std::vector<tokenization::TokenId> tokens(count);
  for (std::size_t i = 0; i < count; ++i) {
    tokens[i] =
        static_cast<tokenization::TokenId>(((start_pos + i) % 1000) + 100);
  }
  return tokens;
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
}

bool ValidatePrefill(hip::QwenGpuExecutor& executor,
                     std::size_t prompt_length) {
  if (prompt_length == 0 || prompt_length > executor.GetMaxPromptBatch()) {
    std::cerr << "Error: --validate-prefill must be between 1 and "
              << executor.GetMaxPromptBatch() << " tokens.\n";
    return false;
  }

  const auto prompt_tokens = MakeBenchmarkTokens(prompt_length);

  executor.Reset();
  tokenization::TokenId sequential_token = 0;
  for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
    sequential_token =
        executor.ForwardToken(prompt_tokens[i], static_cast<std::uint32_t>(i),
                              i + 1 == prompt_tokens.size());
  }
  const auto sequential_logits_view = executor.CopyLastLogits();
  const std::vector<float> sequential_logits(sequential_logits_view.begin(),
                                             sequential_logits_view.end());

  executor.Reset();
  const auto batched_token = executor.ForwardPromptBatch(prompt_tokens);
  const auto batched_logits = executor.CopyLastLogits();
  const auto comparison =
      testing::CompareLogits(sequential_logits, batched_logits);

  std::cout << std::fixed << std::setprecision(8)
            << "[Prefill Validation] tokens=" << prompt_length
            << " sequential_top1=" << sequential_token
            << " batched_top1=" << batched_token
            << " top1_match=" << (comparison.top1_match ? "yes" : "no")
            << " finite=" << (comparison.finite ? "yes" : "no") << '\n'
            << "  max_abs_diff=" << comparison.max_abs_diff
            << " mean_abs_diff=" << comparison.mean_abs_diff
            << " rmse=" << comparison.root_mean_square_error
            << " cosine_similarity=" << comparison.cosine_similarity << '\n';

  return comparison.finite && comparison.top1_match &&
         sequential_token == batched_token;
}

bool IsDeepSeekV4Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "deepseek4";
}

std::vector<int> MakeDeepSeekBenchmarkTokens(
    const models::deepseek_v4_flash::Model& model, std::size_t count) {
  const auto pattern = model.Tokenize(
      "The quick brown fox jumps over the lazy dog. "
      "Strix Halo executes this deterministic benchmark sequence. ");
  if (pattern.empty()) {
    throw std::runtime_error("DeepSeek benchmark token pattern is empty");
  }
  std::vector<int> tokens(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens[index] = pattern[index % pattern.size()];
  }
  return tokens;
}

int RunDeepSeekBenchmark(
    const BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for DeepSeek V4 Flash\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  constexpr std::size_t context_limit = std::numeric_limits<int>::max() - 1;
  if (max_depth > context_limit || max_prompt > context_limit - max_depth ||
      max_generation > context_limit - std::max<std::size_t>(max_depth, 16) ||
      options.validate_prefill_tokens > context_limit) {
    std::cerr << "Error: requested DeepSeek benchmark context is too large\n";
    return 1;
  }
  const std::size_t required_context =
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1,
                options.validate_prefill_tokens + 1});
  const bool dspark = options.speculative_backend == "dspark";
  if (!options.speculative_backend.empty() && !dspark) {
    std::cerr << "Error: DeepSeek supports only --speculative dspark or off\n";
    return 1;
  }
  if (dspark && options.dspark_model_path.empty()) {
    std::cerr << "Error: --speculative dspark requires --dspark-model\n";
    return 1;
  }
  if (dspark && options.min_draft_tokens != 1) {
    std::cerr << "Error: DSpark uses model-owned adaptive drafting; custom "
                 "draft floors are unsupported\n";
    return 1;
  }

  // Both paths use the server's target distribution. Stochastic DSpark
  // preserves that distribution, with its own reproducible RNG trace.
  const bool sampled = !options.sampling.can_use_unmodified_argmax();
  const auto sampling_config = options.sampling;
  const auto sample_next = [&](models::deepseek_v4_flash::Session& session,
                               sampling::SamplerState& sampler,
                               std::string* error_msg) -> int {
    int token = session.TakePendingDsparkToken();
    if (token < 0) {
      const auto logits = session.CopyLogits(error_msg);
      if (logits.empty())
        return -1;
      token = static_cast<int>(sampler.Sample(logits));
    }
    sampler.Accept(static_cast<sampling::TokenId>(token));
    return token;
  };

  std::string error;
  auto model = models::deepseek_v4_flash::Model::Load(
      options.model_path,
      models::deepseek_v4_flash::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .dspark_model_path = dspark ? options.dspark_model_path : "",
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating DeepSeek V4 Flash model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto tokens = MakeDeepSeekBenchmarkTokens(*model, required_context);
  if (options.validate_prefill_tokens != 0) {
    auto sequential =
        model->CreateSession(options.speculative_backend.empty()
                                 ? gufo::core::SessionMode::kAutoregressive
                                 : gufo::core::SessionMode::kSpeculative,
                             required_context, &error);
    auto batched =
        model->CreateSession(options.speculative_backend.empty()
                                 ? gufo::core::SessionMode::kAutoregressive
                                 : gufo::core::SessionMode::kSpeculative,
                             required_context, &error);
    const auto prefix =
        std::span(tokens).first(options.validate_prefill_tokens);
    if (!sequential || !batched || !sequential->Sync(prefix.first(1), &error) ||
        !batched->Sync(prefix, &error)) {
      std::cerr << "DeepSeek prefill validation failed: " << error << '\n';
      return 1;
    }
    for (const int token : prefix.subspan(1)) {
      if (!sequential->Evaluate(token, &error)) {
        std::cerr << "DeepSeek scalar reference failed: " << error << '\n';
        return 1;
      }
    }
    const auto reference = sequential->CopyLogits(&error);
    const auto candidate = batched->CopyLogits(&error);
    if (reference.empty() || reference.size() != candidate.size()) {
      std::cerr << "DeepSeek validation logit readback failed: " << error
                << '\n';
      return 1;
    }
    const auto comparison = testing::CompareLogits(reference, candidate);
    const std::size_t rank =
        1 + std::count_if(candidate.begin(), candidate.end(), [&](float value) {
          return value > candidate[comparison.reference_argmax];
        });
    std::cout << "[Prefill Validation] tokens=" << prefix.size()
              << " finite=" << comparison.finite
              << " scalar_winner_rank=" << rank
              << " rmse=" << comparison.root_mean_square_error
              << " cosine=" << comparison.cosine_similarity
              << " max_error=" << comparison.max_abs_diff << '\n';
    if (!comparison.finite || rank > 3 ||
        comparison.root_mean_square_error > 1.12F ||
        comparison.cosine_similarity < 0.979F ||
        comparison.max_abs_diff > 5.0F) {
      return 1;
    }
  }
  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto model_name = model->ModelName();

  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";

  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model_name << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };

  for (const std::size_t concurrency : options.concurrency) {
    if (concurrency > 1) {
      using models::deepseek_v4_flash::Session;
      for (const std::size_t depth : options.n_depths) {
        auto base = model->CreateSession(
            options.speculative_backend.empty()
                ? gufo::core::SessionMode::kAutoregressive
                : gufo::core::SessionMode::kSpeculative,
            static_cast<uint32_t>(required_context), &error);
        const std::size_t prefix = depth > 0 ? depth : 16;
        if (!base || !base->Sync(std::span(tokens).first(prefix), &error)) {
          std::cerr << "Error preparing concurrent depth: " << error << '\n';
          return 1;
        }
        auto snapshot = base->SaveSnapshot(&error);
        if (!snapshot) {
          std::cerr << "Error preparing concurrent snapshot: " << error << '\n';
          return 1;
        }
        base.reset();
        for (const bool generate : {false, true}) {
          for (const std::size_t count :
               generate ? options.n_gens : options.n_prompts) {
            std::vector<double> runs;
            for (std::size_t repeat = 0; repeat < options.repetitions;
                 ++repeat) {
              std::vector<std::unique_ptr<Session>> sessions;
              std::vector<std::vector<int>> generated(concurrency);
              const std::vector<sampling::TokenId> history(
                  tokens.begin(), tokens.begin() + prefix);
              std::vector<sampling::SamplerState> samplers(
                  concurrency,
                  sampling::SamplerState(sampling_config, history));
              for (std::size_t i = 0; i < concurrency; ++i) {
                auto session = model->CreateSession(
                    options.speculative_backend.empty()
                        ? gufo::core::SessionMode::kAutoregressive
                        : gufo::core::SessionMode::kSpeculative,
                    static_cast<uint32_t>(required_context), &error);
                if (!session ||
                    ((generate || depth > 0) &&
                     !session->RestoreSnapshot(*snapshot, &error))) {
                  std::cerr << "Error restoring concurrent depth: " << error
                            << '\n';
                  return 1;
                }
                sessions.push_back(std::move(session));
                generated[i].reserve(count);
              }
              const auto start = std::chrono::steady_clock::now();
              if (!generate) {
                for (auto& session : sessions) {
                  if (!session->Sync(std::span(tokens).first(depth + count),
                                     &error)) {
                    std::cerr << "Error in concurrent prefill: " << error
                              << '\n';
                    return 1;
                  }
                }
              } else if (dspark) {
                std::vector<std::vector<int>> emitted(concurrency);
                while (true) {
                  std::vector<models::deepseek_v4_flash::SessionDsparkBatchItem>
                      items;
                  std::vector<std::size_t> active;
                  std::vector<std::unique_ptr<
                      models::deepseek_v4_flash::DsparkSamplerBridge>>
                      bridges;
                  for (std::size_t i = 0; i < concurrency; ++i) {
                    if (generated[i].size() == count)
                      continue;
                    active.push_back(i);
                    const ds4_dspark_sampler* hook = nullptr;
                    if (sampled) {
                      bridges.push_back(
                          std::make_unique<
                              models::deepseek_v4_flash::DsparkSamplerBridge>(
                              samplers[i]));
                      hook = bridges.back()->hook();
                    }
                    items.push_back({.session = sessions[i].get(),
                                     .max_tokens = count - generated[i].size(),
                                     .max_draft_tokens = options.draft_tokens,
                                     .emitted = &emitted[i],
                                     .sampler = hook});
                  }
                  if (items.empty())
                    break;
                  if (!model->DsparkStepBatch(items, &error)) {
                    std::cerr << "Error in concurrent DSpark: " << error
                              << '\n';
                    return 1;
                  }
                  for (std::size_t slot = 0; slot < active.size(); ++slot) {
                    const auto i = active[slot];
                    if (emitted[i].empty() ||
                        emitted[i].size() > count - generated[i].size()) {
                      std::cerr << "Error: DSpark returned an invalid "
                                   "benchmark token count\n";
                      return 1;
                    }
                    if (sampled) {
                      samplers[i].SetRngState(bridges[slot]->rng_state());
                      for (const int token : emitted[i]) {
                        samplers[i].Accept(
                            static_cast<sampling::TokenId>(token));
                      }
                    }
                    generated[i].insert(generated[i].end(), emitted[i].begin(),
                                        emitted[i].end());
                  }
                }
              } else {
                for (std::size_t step = 0; step < count; ++step) {
                  std::vector<models::deepseek_v4_flash::SessionBatchItem>
                      items;
                  for (std::size_t i = 0; i < concurrency; ++i) {
                    const int token =
                        sampled ? sample_next(*sessions[i], samplers[i], &error)
                                : sessions[i]->SelectNext(0.0F, nullptr);
                    if (token < 0) {
                      std::cerr << "Error in concurrent sampling: " << error
                                << '\n';
                      return 1;
                    }
                    generated[i].push_back(token);
                    items.push_back(
                        {.session = sessions[i].get(), .token = token});
                  }
                  if (!model->EvaluateBatch(items, &error)) {
                    std::cerr << "Error in concurrent target decode: " << error
                              << '\n';
                    return 1;
                  }
                }
              }
              const double seconds =
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count();
              runs.push_back(
                  static_cast<double>(count) *
                  (generate ? 1.0 : static_cast<double>(concurrency)) /
                  seconds);
              if (options.verbose && generate) {
                for (std::size_t i = 0; i < concurrency; ++i) {
                  const auto stats = sessions[i]->DsparkStatistics();
                  const auto* bytes =
                      reinterpret_cast<const uint8_t*>(generated[i].data());
                  std::cerr << "DeepSeek tg C=" << concurrency
                            << " depth=" << depth << " request=" << i
                            << " accepted=" << stats.support_accepted
                            << " drafted=" << stats.support_drafted
                            << " steps=" << stats.steps
                            << " skipped=" << stats.skipped << " output_sha256="
                            << crypto::Sha256Hex(std::span(
                                   bytes, generated[i].size() * sizeof(int)))
                            << '\n';
                }
              }
            }
            print_result(MakeTestName(generate ? "tg" : "pp", count, depth) +
                             " C" + std::to_string(concurrency),
                         ComputeStats(runs));
          }
        }
      }
      continue;
    }
    std::unique_ptr<models::deepseek_v4_flash::Session> prepared_session;
    std::size_t prepared_depth = 0;
    for (const std::size_t depth : options.n_depths) {
      std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
      if (depth > 0) {
        if (prepared_session == nullptr || depth < prepared_depth) {
          prepared_session = model->CreateSession(
              options.speculative_backend.empty()
                  ? gufo::core::SessionMode::kAutoregressive
                  : gufo::core::SessionMode::kSpeculative,
              static_cast<std::uint32_t>(required_context), &error);
          prepared_depth = 0;
        }
        if (prepared_session == nullptr ||
            !prepared_session->Sync(std::span(tokens.data(), depth), &error)) {
          std::cerr << "Error preparing DeepSeek depth " << depth << ": "
                    << error << '\n';
          return 1;
        }
        prepared_depth = depth;
        snapshot = prepared_session->SaveSnapshot(&error);
        if (snapshot == nullptr) {
          std::cerr << "Error snapshotting DeepSeek depth " << depth << ": "
                    << error << '\n';
          return 1;
        }
        if (options.verbose) {
          std::cerr << "Prepared DeepSeek depth " << depth
                    << " snapshot_bytes=" << snapshot->SizeBytes() << '\n';
        }
      }

      for (const std::size_t prompt_length : options.n_prompts) {
        if (depth + prompt_length >= required_context) {
          std::cerr << "Error: DeepSeek prompt benchmark exceeds context\n";
          return 1;
        }
        std::vector<double> runs;
        runs.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions;
             ++repetition) {
          std::unique_ptr<models::deepseek_v4_flash::Session> local_session;
          models::deepseek_v4_flash::Session* session = prepared_session.get();
          if (snapshot == nullptr) {
            local_session = model->CreateSession(
                options.speculative_backend.empty()
                    ? gufo::core::SessionMode::kAutoregressive
                    : gufo::core::SessionMode::kSpeculative,
                static_cast<std::uint32_t>(required_context), &error);
            session = local_session.get();
            if (session != nullptr && depth > 0 &&
                !session->Sync(std::span(tokens).first(depth), &error)) {
              session = nullptr;
            }
          } else if (!prepared_session->RestoreSnapshot(*snapshot, &error)) {
            std::cerr << "Error restoring DeepSeek depth: " << error << '\n';
            return 1;
          }
          if (session == nullptr) {
            std::cerr << "Error creating DeepSeek session: " << error << '\n';
            return 1;
          }
          const auto start = std::chrono::steady_clock::now();
          if (!session->Sync(std::span(tokens.data(), depth + prompt_length),
                             &error)) {
            std::cerr << "Error running DeepSeek prefill: " << error << '\n';
            return 1;
          }
          const double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
          runs.push_back(static_cast<double>(prompt_length) / seconds);
          if (options.verbose) {
            std::cerr << "DeepSeek pp depth=" << depth
                      << " payload_bytes=" << session->PayloadBytes() << '\n';
          }
        }
        print_result(MakeTestName("pp", prompt_length, depth),
                     ComputeStats(runs));
      }

      for (const std::size_t generation_length : options.n_gens) {
        const std::size_t prefix_length = depth > 0 ? depth : 16;
        if (prefix_length + generation_length >= required_context) {
          std::cerr << "Error: DeepSeek generation benchmark exceeds context\n";
          return 1;
        }
        std::vector<double> runs;
        runs.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions;
             ++repetition) {
          std::unique_ptr<models::deepseek_v4_flash::Session> local_session;
          models::deepseek_v4_flash::Session* session = prepared_session.get();
          if (snapshot == nullptr) {
            local_session = model->CreateSession(
                options.speculative_backend.empty()
                    ? gufo::core::SessionMode::kAutoregressive
                    : gufo::core::SessionMode::kSpeculative,
                static_cast<std::uint32_t>(required_context), &error);
            session = local_session.get();
            if (session != nullptr &&
                !session->Sync(std::span(tokens.data(), prefix_length),
                               &error)) {
              session = nullptr;
            }
          } else if (!prepared_session->RestoreSnapshot(*snapshot, &error)) {
            session = nullptr;
          }
          if (session == nullptr) {
            std::cerr << "Error preparing DeepSeek generation: " << error
                      << '\n';
            return 1;
          }

          std::vector<int> generated;
          generated.reserve(generation_length);
          const std::vector<sampling::TokenId> history(
              tokens.begin(), tokens.begin() + prefix_length);
          sampling::SamplerState sampler(sampling_config, history);
          const auto start = std::chrono::steady_clock::now();
          for (std::size_t step = 0; step < generation_length;) {
            if (dspark) {
              std::vector<int> emitted;
              std::optional<models::deepseek_v4_flash::DsparkSamplerBridge>
                  bridge;
              if (sampled)
                bridge.emplace(sampler);
              if (!session->DsparkStep(generation_length - step,
                                       options.draft_tokens, &emitted, &error,
                                       bridge ? bridge->hook() : nullptr) ||
                  emitted.empty()) {
                std::cerr << "Error running DSpark decode: " << error << '\n';
                return 1;
              }
              if (bridge) {
                sampler.SetRngState(bridge->rng_state());
                for (const int token : emitted) {
                  sampler.Accept(static_cast<sampling::TokenId>(token));
                }
              }
              generated.insert(generated.end(), emitted.begin(), emitted.end());
              step += emitted.size();
            } else {
              const int token = sampled ? sample_next(*session, sampler, &error)
                                        : session->SelectNext(0.0F, nullptr);
              if (token < 0 || !session->Evaluate(token, &error)) {
                std::cerr << "Error running DeepSeek decode: " << error << '\n';
                return 1;
              }
              generated.push_back(token);
              ++step;
            }
          }
          const double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
          runs.push_back(static_cast<double>(generation_length) / seconds);
          if (options.verbose) {
            const auto stats = session->DsparkStatistics();
            const auto* bytes =
                reinterpret_cast<const uint8_t*>(generated.data());
            std::cerr << "DeepSeek tg C=1 depth=" << depth
                      << " request=0 accepted=" << stats.support_accepted
                      << " drafted=" << stats.support_drafted
                      << " steps=" << stats.steps
                      << " skipped=" << stats.skipped << " output_sha256="
                      << crypto::Sha256Hex(
                             std::span(bytes, generated.size() * sizeof(int)))
                      << '\n';
          }
        }
        print_result(MakeTestName("tg", generation_length, depth),
                     ComputeStats(runs));
      }

      if (snapshot && !prepared_session->RestoreSnapshot(*snapshot, &error)) {
        std::cerr << "Error restoring final DeepSeek depth: " << error << '\n';
        return 1;
      }
    }
  }
  std::cout << '\n';
  return 0;
}

bool IsQwen38FlashNext(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "qwen4exp";
}

int RunQwen38FlashNextBenchmark(
    const BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  namespace qfn = models::qwen38_flash_next;
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Qwen3.8-Flash-Next\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  if (std::max({max_depth, max_prompt, max_generation,
                options.validate_prefill_tokens}) >
      std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: benchmark workload exceeds the context range\n";
    return 1;
  }
  const std::size_t required_context =
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1,
                options.validate_prefill_tokens + 1});
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Qwen3.8-Flash-Next supports only --speculative mtp "
                 "or off\n";
    return 1;
  }
  if (mtp && options.mtp_model_path.empty()) {
    std::cerr << "Error: --speculative mtp requires --mtp-model\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Flash-Next bench supports C1; use the serving "
                 "benchmark for concurrent requests\n";
    return 1;
  }
  if (mtp && options.min_draft_tokens != 1) {
    std::cerr << "Error: Flash-Next MTP requires --min-draft-tokens 1\n";
    return 1;
  }
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: Flash-Next context is out of range\n";
    return 1;
  }

  std::string error;
  auto model = qfn::Model::Load(
      options.model_path,
      qfn::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .mtp_model_path = mtp ? options.mtp_model_path : "",
          .max_draft_tokens = std::max<std::uint32_t>(1, options.draft_tokens),
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating Qwen3.8-Flash-Next model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  // Repeat a fixed token pattern for reproducible timing. It uses far fewer
  // distinct PLE rows than varied requests.
  std::vector<std::int32_t> tokens;
  {
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    if (pattern.empty()) {
      std::cerr << "Error: benchmark token pattern is empty\n";
      return 1;
    }
    tokens.resize(required_context);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = pattern[i % pattern.size()];
    }
  }

  if (options.validate_prefill_tokens != 0) {
    // Batched prefill against one-token-at-a-time evaluation of the same
    // prefix; both run on the GPU, so this checks the batched kernels.
    auto sequential = model->CreateSession(
        options.speculative_backend.empty()
            ? gufo::core::SessionMode::kAutoregressive
            : gufo::core::SessionMode::kSpeculative,
        static_cast<std::uint32_t>(required_context), &error);
    auto batched = model->CreateSession(
        options.speculative_backend.empty()
            ? gufo::core::SessionMode::kAutoregressive
            : gufo::core::SessionMode::kSpeculative,
        static_cast<std::uint32_t>(required_context), &error);
    const auto prefix =
        std::span(tokens).first(options.validate_prefill_tokens);
    if (!sequential || !batched || !sequential->Sync(prefix.first(1), &error) ||
        !batched->Sync(prefix, &error)) {
      std::cerr << "Qwen3.8-Flash-Next prefill validation failed: " << error
                << '\n';
      return 1;
    }
    for (const std::int32_t token : prefix.subspan(1)) {
      if (!sequential->Evaluate(token, &error)) {
        std::cerr << "Qwen3.8-Flash-Next sequential reference failed: " << error
                  << '\n';
        return 1;
      }
    }
    const auto reference = sequential->Logits();
    const auto candidate = batched->Logits();
    const auto comparison = testing::CompareLogits(reference, candidate);
    const std::size_t rank =
        1 + std::count_if(candidate.begin(), candidate.end(), [&](float value) {
          return value > candidate[comparison.reference_argmax];
        });
    std::cout << "[Prefill Validation] tokens=" << prefix.size()
              << " finite=" << comparison.finite
              << " scalar_winner_rank=" << rank
              << " rmse=" << comparison.root_mean_square_error
              << " cosine=" << comparison.cosine_similarity
              << " max_error=" << comparison.max_abs_diff << '\n';
    if (!comparison.finite || rank > 3 ||
        comparison.root_mean_square_error > 1.12F ||
        comparison.cosine_similarity < 0.979F ||
        comparison.max_abs_diff > 5.0F) {
      return 1;
    }
  }

  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto model_name = model->ModelName();
  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";
  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model_name << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };

  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      // A full aligned prefix already warms the same chunk shapes and GEMM
      // plans. Other shapes still need an untimed pass before measurement.
      const auto chunk = model->PrefillCapacity();
      const bool warm_prefix = chunk > 0 && depth >= chunk &&
                               depth % chunk == 0 && prompt_length > 0 &&
                               prompt_length % chunk == 0;
      for (std::size_t repetition = warm_prefix ? 1 : 0;
           repetition <= options.repetitions; ++repetition) {
        auto session = model->CreateSession(
            options.speculative_backend.empty()
                ? gufo::core::SessionMode::kAutoregressive
                : gufo::core::SessionMode::kSpeculative,
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            (depth > 0 &&
             !session->Sync(std::span(tokens).first(depth), &error))) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!session->Sync(std::span(tokens).first(depth + prompt_length),
                           &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (repetition > 0) {
          runs.push_back(static_cast<double>(prompt_length) / seconds);
        }
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }

    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      if (prefix_length + generation_length >= required_context) {
        std::cerr << "Error: generation benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            options.speculative_backend.empty()
                ? gufo::core::SessionMode::kAutoregressive
                : gufo::core::SessionMode::kSpeculative,
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            !session->Sync(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        std::vector<std::int32_t> generated;
        const std::vector<sampling::TokenId> history(
            tokens.begin(), tokens.begin() + prefix_length);
        sampling::SamplerState sampler(options.sampling, history);
        const auto start = std::chrono::steady_clock::now();
        while (generated.size() < generation_length) {
          qfn::Session::DecodeResult decoded;
          if (!session->DecodeStep(generation_length - generated.size(),
                                   sampler, &decoded, &error, false) ||
              decoded.tokens.empty()) {
            std::cerr << "Error running Flash-Next decode: " << error << '\n';
            return 1;
          }
          generated.insert(generated.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          const auto stats = session->Statistics();
          std::cerr << "Qwen3.8-Flash-Next tg depth=" << depth
                    << " cycles=" << stats.cycles
                    << " drafted=" << stats.drafted
                    << " accepted=" << stats.accepted << " output_sha256="
                    << crypto::Sha256Hex(
                           std::span(reinterpret_cast<const std::uint8_t*>(
                                         generated.data()),
                                     generated.size() * sizeof(std::int32_t)))
                    << " text="
                    << model->Decode(std::span(generated).first(
                           std::min<std::size_t>(generated.size(), 48)))
                    << '\n';
        }
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
    }
  }
  std::cout << '\n';
  return 0;
}

bool IsLing3Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "bailingmoe3";
}

std::string JsonString(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char escaped[8];
          std::snprintf(escaped, sizeof(escaped), "\\u%04x",
                        static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += escaped;
        } else {
          out += c;
        }
    }
  }
  return out + '"';
}

std::string UtcNow() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

// Deterministic pseudo-random text tokens: like llama-bench, no repeated
// pattern, so MoE routing sees realistic expert diversity.
std::vector<std::int32_t> LingBenchTokens(
    const models::ling_3_flash::Model& model, std::size_t count) {
  std::vector<std::int32_t> tokens(count);
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  const std::uint32_t range =
      std::min<std::uint32_t>(model.VocabSize(), 150000);
  for (auto& token : tokens) {
    do {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      token = static_cast<std::int32_t>((state >> 11) % range);
    } while (model.tokenizer().IsSpecialToken(
        static_cast<models::ling_3_flash::TokenId>(token)));
  }
  return tokens;
}

// Context tokens a Ling MTP generation starts from at depth 0.
constexpr std::size_t kLingMtpSeedTokens = 16;

// JSON array of op_profile entries divided by `per` (steps), with the
// effective bandwidth of each (bytes it must read / its device time).
std::string LingOpList(const std::vector<models::ling_3_flash::OpTiming>& ops,
                       double per) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(4) << "[";
  for (std::size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];
    out << (i ? ", " : "") << "{\"name\": " << JsonString(op.name)
        << ", \"count\": " << op.count / per << ", \"ms\": " << op.ms / per
        << ", \"mbytes\": " << op.bytes / per / 1e6
        << ", \"gbps\": " << (op.ms > 0 ? op.bytes / (op.ms * 1e6) : 0.0)
        << "}";
  }
  out << "]";
  return out.str();
}

double LingOpTotal(const std::vector<models::ling_3_flash::OpTiming>& ops) {
  double total = 0.0;
  for (const auto& op : ops) {
    // "name/part" entries break an operation down; they are inside it.
    if (op.name.find('/') == std::string::npos)
      total += op.ms;
  }
  return total;
}

// Host issue / wall time and device span of a series of steps.
struct LingLaunchSeries {
  std::vector<double> issue, wall, device, host_wall;
  void Add(const models::ling_3_flash::LaunchTiming& t, double host_ms) {
    issue.push_back(t.issue_ms);
    wall.push_back(t.wall_ms);
    device.push_back(t.device_ms);
    host_wall.push_back(host_ms);
  }
  static double Median(std::vector<double> v) {
    if (v.empty())
      return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }
  static double Mean(const std::vector<double>& v) {
    return v.empty() ? 0.0
                     : std::accumulate(v.begin(), v.end(), 0.0) /
                           static_cast<double>(v.size());
  }
  // issue: entry until the last launch was issued; wall: entry until done
  // (inside the engine); call: the caller's clock around the whole call.
  std::string Json() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << "{\"steps\": " << wall.size()
        << ", \"issue_ms\": " << Median(issue)
        << ", \"wall_ms\": " << Median(wall)
        << ", \"device_ms\": " << Median(device)
        << ", \"call_ms\": " << Median(host_wall)
        << ", \"mean_issue_ms\": " << Mean(issue)
        << ", \"mean_device_ms\": " << Mean(device)
        << ", \"mean_call_ms\": " << Mean(host_wall) << "}";
    return out.str();
  }
};

int RunLingBenchmark(const BenchOptions& options,
                     const std::shared_ptr<const core::GgufReader>& reader,
                     std::chrono::steady_clock::time_point model_load_start) {
  namespace ling = models::ling_3_flash;
  const bool jsonl = options.output_format == "jsonl";
  // Ling's default is MTP when the GGUF carries the block-42 weights and the
  // run is a greedy, single-floor one; `--speculative off` keeps AR.
  const bool default_mtp = !options.speculative_explicit &&
                           options.mtp_model_path.empty() &&
                           options.min_draft_tokens == 1 &&
                           options.sampling.can_use_unmodified_argmax() &&
                           models::ling_3_flash::HasMtpBlock(*reader);
  if (!options.speculative_explicit && !default_mtp) {
    std::cerr << "[ling] speculative=off: "
              << (models::ling_3_flash::HasMtpBlock(*reader)
                      ? "run is not greedy MTP-compatible\n"
                      : "GGUF has no MTP block\n");
  }
  const bool mtp = default_mtp || options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() &&
      options.speculative_backend != "mtp") {
    std::cerr << "Error: Ling-3.0-flash supports --speculative mtp only\n";
    return 1;
  }
  if (mtp && !options.mtp_model_path.empty()) {
    std::cerr << "Error: Ling-3.0-flash carries its MTP block in the main "
                 "GGUF; drop --mtp-model\n";
    return 1;
  }
  if (mtp && options.min_draft_tokens != 1) {
    std::cerr << "Error: Ling-3.0-flash MTP requires --min-draft-tokens 1; "
                 "bound blocks with --draft-tokens\n";
    return 1;
  }
  if (mtp && !options.sampling.can_use_unmodified_argmax()) {
    std::cerr << "Error: Ling-3.0-flash MTP bench measures greedy decoding\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Ling-3.0-flash bench supports C1 only\n";
    return 1;
  }
  if (options.validate_prefill_tokens != 0) {
    std::cerr << "Error: --validate-prefill is not available for "
                 "Ling-3.0-flash\n";
    return 1;
  }
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Ling-3.0-flash\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  std::vector<std::size_t> depths = options.n_depths;
  std::sort(depths.begin(), depths.end());
  depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(depths);
  const std::size_t workload =
      std::max(max_or_zero(options.n_prompts), max_or_zero(options.n_gens));
  const std::size_t required_context = max_depth + workload + 1;
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: benchmark workload exceeds the context range\n";
    return 1;
  }

  std::string error;
  auto model = ling::Model::Load(
      options.model_path,
      ling::ModelOptions{
          .max_context = static_cast<std::uint32_t>(
              std::max<std::size_t>(required_context, 4096)),
          .max_batch = options.max_batch != 0
                           ? static_cast<std::uint32_t>(options.max_batch)
                           : ling::kCliPrefillBatch,
          .expanded_kv_bytes = options.expanded_kv_bytes,
          .mtp = mtp},
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating Ling-3.0-flash model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);
  auto session = model->CreateSession(
      static_cast<std::uint32_t>(required_context), &error);
  if (!session) {
    std::cerr << "Error creating Ling-3.0-flash session: " << error << '\n';
    return 1;
  }
  session->Reset();

  const std::size_t chunk = model->PrefillCapacity();
  const std::string model_type = "bailingmoe3 (Ling-3.0-flash)";
  const std::string gpu_info =
      std::string(prop.name) + " (" + prop.gcnArchName + ", ROCm/HIP)";
#ifdef GUFO_REVISION
  const std::string revision = GUFO_REVISION;
#else
  const std::string revision = "unknown";
#endif
  // Fields that the llama-bench baseline sets and gufo cannot or does not
  // match. Printed once so a comparison states them.
  const std::vector<std::pair<std::string, std::string>> unmatched{
      {"type_k/type_v",
       "gufo caches the MLA latent in F16 (llama.cpp baseline: q8_0, about "
       "half the cache bytes); KDA state is F32"},
      {"flash_attn",
       "no flag: gufo's own MLA and KDA kernels (baseline: flash_attn=1)"},
      {"n_batch",
       "gufo prefill chunk equals n_ubatch (512 unless --max-batch); "
       "baseline n_batch=2048, n_ubatch=512"},
      {"n_threads/cpu_mask/poll", "not applicable: device-only execution"},
      {"split_mode/tensor_split/n_cpu_moe/devices",
       "not applicable: one GPU, all layers resident"},
      {"graph/sampling",
       "decode feeds fixed pseudo-random tokens like llama-bench; gufo "
       "copies the last-row logits to the host each step; decode steps "
       "replay per-session HIP graphs (stage_ms_per_rep comes from one extra "
       "eager repetition)"},
      {"depth_build",
       "depth is built by real prefill of d tokens in 512-token chunks, "
       "once per run and ascending; each repetition restores an exact copy "
       "of the recurrent state and truncates the latent cache to d"},
      {"warmup", "one untimed repetition per test point"},
      {"speculative",
       mtp ? "mtp: tg generates greedily with Ling's MTP block (up to "
             "--draft-tokens drafts per verified step, adaptive); depth 0 "
             "starts from a 16-token seed; see the mtp records for acceptance"
           : "off"},
      {"context_tokens",
       options.text_file.empty()
           ? "pseudo-random tokens (MTP acceptance on them is not "
             "representative)"
           : "tokenized --text-file"}};
  if (jsonl) {
    std::cout << "{\"record\": \"header\", \"build_commit\": "
              << JsonString(revision)
              << ", \"gpu_info\": " << JsonString(gpu_info)
              << ", \"backends\": \"ROCm (HIP)\", \"model_filename\": "
              << JsonString(options.model_path)
              << ", \"model_type\": " << JsonString(model_type)
              << ", \"model_size\": " << reader->GetSize()
              << ", \"n_batch\": " << chunk << ", \"n_ubatch\": " << chunk
              << ", \"type_k\": \"f16\", \"type_v\": \"f16\", "
                 "\"kv_cache\": \"MLA latent F16 (7 layers x 576)\", "
                 "\"kda_state\": \"f32\", \"n_gpu_layers\": \"all\", "
                 "\"flash_attn\": \"n/a (gufo kernels)\", \"repetitions\": "
              << options.repetitions
              << ", \"warmup\": 1, \"max_context\": " << required_context
              << ", \"weights_gib\": "
              << static_cast<double>(model->ResidentBytes()) /
                     (1024.0 * 1024.0 * 1024.0)
              << ", \"unmatched\": {";
    for (std::size_t i = 0; i < unmatched.size(); ++i) {
      std::cout << (i ? ", " : "") << JsonString(unmatched[i].first) << ": "
                << JsonString(unmatched[i].second);
    }
    std::cout << "}}\n" << std::flush;
  } else {
    std::cout << "# gufo bench: " << model_type << ", " << gpu_info
              << ", build " << revision << '\n'
              << "# KV cache: MLA latent F16, KDA state F32; own kernels, "
                 "all layers on GPU; n_batch=n_ubatch="
              << chunk << ", repetitions=" << options.repetitions
              << ", warmup=1\n";
    for (const auto& [key, text] : unmatched)
      std::cout << "# unmatched " << key << ": " << text << '\n';
    std::cout << "| test | t/s | samples t/s |\n| --- | ---: | --- |\n"
              << std::flush;
  }

  std::vector<std::int32_t> tokens;
  if (options.text_file.empty()) {
    tokens = LingBenchTokens(*model, required_context);
  } else {
    std::ifstream in(options.text_file, std::ios::binary);
    if (!in) {
      std::cerr << "Error: cannot read --text-file " << options.text_file
                << '\n';
      return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    tokens = model->Tokenize(text);
    if (tokens.size() < required_context) {
      std::cerr << "Error: --text-file has " << tokens.size()
                << " tokens, the workload needs " << required_context << '\n';
      return 1;
    }
    tokens.resize(required_context);
  }
  // Device-event time per stage, summed over the chunks (pp) or steps (tg) of
  // one repetition and averaged over the timed repetitions.
  struct StageMs {
    double total{0}, head{0}, kda{0}, mla{0}, dense{0}, host{0};
    std::size_t forwards{0};
    std::size_t reps{0};  ///< repetitions behind the sums (0: the timed ones)
    double graph_forwards{0};  ///< graph replays per timed repetition
    void Add(const std::vector<ling::StepTiming>& timings) {
      for (const auto& t : timings) {
        total += t.total_ms;
        head += t.head_ms;
        kda += t.kda_ms;
        mla += t.mla_ms;
        dense += t.dense_ms;
        host += t.host_ms;
        ++forwards;
      }
    }
  };
  const auto emit = [&](std::size_t n_prompt, std::size_t n_gen,
                        std::size_t depth, const std::vector<double>& ns,
                        const StageMs& stages) {
    std::vector<double> ts;
    const double count = static_cast<double>(n_prompt + n_gen);
    for (const double value : ns)
      ts.push_back(count * 1e9 / value);
    const auto ns_stats = ComputeStats(ns);
    const auto ts_stats = ComputeStats(ts);
    if (jsonl) {
      const auto list = [](const std::vector<double>& values, int precision) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(precision) << "[";
        for (std::size_t i = 0; i < values.size(); ++i)
          out << (i ? ", " : " ") << values[i];
        out << " ]";
        return out.str();
      };
      std::cout << "{\"record\": \"result\", \"n_prompt\": " << n_prompt
                << ", \"n_gen\": " << n_gen << ", \"n_depth\": " << depth
                << ", \"test_time\": \"" << UtcNow() << "\", \"avg_ns\": "
                << static_cast<std::uint64_t>(ns_stats.mean)
                << ", \"stddev_ns\": "
                << static_cast<std::uint64_t>(ns_stats.stddev)
                << ", \"avg_ts\": " << std::fixed << std::setprecision(6)
                << ts_stats.mean << ", \"stddev_ts\": " << ts_stats.stddev
                << ", \"samples_ns\": " << list(ns, 0)
                << ", \"samples_ts\": " << list(ts, 3);
      const double reps =
          static_cast<double>(stages.reps != 0 ? stages.reps : ns.size());
      std::cout << ", \"stage_ms_per_rep\": {\"forwards\": "
                << stages.forwards / static_cast<std::size_t>(reps)
                << ", \"graph_forwards\": " << stages.graph_forwards
                << ", \"device_total\": " << stages.total / reps
                << ", \"mla_blocks_5_11_17_23_29_35_41\": " << stages.mla / reps
                << ", \"kda_blocks_2_to_41\": " << stages.kda / reps
                << ", \"blocks_0_1\": " << stages.dense / reps
                << ", \"output_head\": " << stages.head / reps
                << ", \"host_wall\": " << stages.host / reps << "}}\n"
                << std::flush;
    } else {
      std::ostringstream samples;
      samples << std::fixed << std::setprecision(2);
      for (std::size_t i = 0; i < ts.size(); ++i)
        samples << (i ? ", " : "") << ts[i];
      std::cout << "| "
                << MakeTestName(n_prompt != 0 ? "pp" : "tg",
                                n_prompt != 0 ? n_prompt : n_gen, depth)
                << " | " << std::fixed << std::setprecision(2) << ts_stats.mean
                << " ± " << ts_stats.stddev << " | " << samples.str() << " |\n"
                << std::flush;
    }
  };

  const auto seconds_since = [](std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
  };
  // MTP decode-cycle profile at one depth (--op-profile, jsonl, MTP): one
  // "mtp_profile" record per part. Each part starts from the depth's state
  // (plus the 16-token seed at depth 0) and stays within the n_gen tokens the
  // context holds past the depth.
  //   ar      one-token steps replayed as graphs vs eager (Feed with timings),
  //           host issue / wall time and device span, and the per-op device
  //           times of eager steps (the decode bandwidth reference);
  //   draft   one MTP draft step (block 42 + head, chained like DecodeStep),
  //           launch timing and per-op device times;
  //   verify  the verify forward at widths 2..8 (always eager: rollback
  //           points), launch timing, Rollback(1) wall and per-op times;
  //   cycle   fixed-width greedy cycles (anchor argmax, width - 1 drafts,
  //           verify, acceptance, rollback) split into their host phases with
  //           the device spans of the drafts and the verify.
  const auto profile_mtp = [&](std::size_t depth, std::size_t room,
                               const auto& restore) {
    constexpr int kSteps = 16;
    const auto prepare = [&]() {
      if (!restore())
        return false;
      if (depth == 0 &&
          !session->Feed(std::span(tokens).first(kLingMtpSeedTokens), {},
                         &error)) {
        std::cerr << "Error seeding the MTP profile: " << error << '\n';
        return false;
      }
      return true;
    };
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto ms_since = [](std::chrono::steady_clock::time_point t) {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t)
          .count();
    };
    // Next context token for a forced step at the current position.
    const auto next_token = [&]() {
      return tokens[std::min<std::size_t>(session->Position(),
                                          tokens.size() - 1)];
    };
    const auto fail = [&](const char* what) {
      std::cerr << "Error in the MTP profile (" << what << "): " << error
                << '\n';
      return false;
    };
    const auto head = [&](const char* part) {
      std::ostringstream out;
      out << "{\"record\": \"mtp_profile\", \"part\": \"" << part
          << "\", \"n_depth\": " << depth;
      return out.str();
    };
    const std::size_t steps =
        std::min<std::size_t>(kSteps, room > 8 ? (room - 8) / 3 : 1);

    // ar: graph replay vs eager one-token steps.
    {
      if (!prepare())
        return false;
      LingLaunchSeries graph, eager;
      // The first occurrence of a step shape runs eagerly, the second
      // captures; replays follow.
      for (std::size_t i = 0; i < steps + 2; ++i) {
        const std::int32_t token = next_token();
        const auto t = now();
        if (!session->Feed(std::span(&token, 1), {}, &error))
          return fail("ar graph step");
        if (i >= 2)
          graph.Add(session->LastLaunchTiming(), ms_since(t));
      }
      for (std::size_t i = 0; i < steps; ++i) {
        const std::int32_t token = next_token();
        std::vector<ling::StepTiming> timing;
        const auto t = now();
        if (!session->Feed(std::span(&token, 1), {.timings = &timing}, &error))
          return fail("ar eager step");
        eager.Add(session->LastLaunchTiming(), ms_since(t));
      }
      std::vector<ling::OpTiming> ops;
      constexpr int kOpSteps = 4;
      for (int i = 0; i < kOpSteps; ++i) {
        const std::int32_t token = next_token();
        if (!session->Feed(std::span(&token, 1), {.op_profile = &ops}, &error))
          return fail("ar op profile");
      }
      std::cout << head("ar") << ", \"graph\": " << graph.Json()
                << ", \"eager\": " << eager.Json()
                << ", \"ms_per_step\": " << std::fixed << std::setprecision(4)
                << LingOpTotal(ops) / kOpSteps
                << ", \"ops\": " << LingOpList(ops, kOpSteps) << "}\n"
                << std::flush;
    }
    // draft: chained draft steps from the committed position.
    {
      if (!prepare())
        return false;
      LingLaunchSeries series;
      std::vector<ling::OpTiming> ops;
      int profiled = 0;
      const std::int32_t anchor = session->Argmax();
      for (int round = 0; round < 4; ++round) {
        std::int32_t token = anchor;
        for (std::uint32_t j = 0; j + 1 < ling::kMaxVerifyTokens; ++j) {
          std::int32_t index = -1;
          const bool profile = round == 3;
          const auto t = now();
          if (!session->Draft(token, j > 0, {}, &index, &error,
                              profile ? &ops : nullptr))
            return fail("draft");
          if (round > 0 && !profile)
            series.Add(session->LastLaunchTiming(), ms_since(t));
          profiled += profile ? 1 : 0;
          token = index;
        }
      }
      std::cout << head("draft") << ", \"launch\": " << series.Json()
                << ", \"ms_per_step\": " << std::fixed << std::setprecision(4)
                << LingOpTotal(ops) / profiled
                << ", \"ops\": " << LingOpList(ops, profiled) << "}\n"
                << std::flush;
    }
    // verify: every width, then Rollback(1).
    for (std::uint32_t width = 2; width <= ling::kMaxVerifyTokens; ++width) {
      if (!prepare())
        return false;
      LingLaunchSeries series;
      std::vector<double> rollback_ms;
      std::vector<ling::OpTiming> ops;
      const std::size_t reps = std::max<std::size_t>(2, steps / 2);
      for (std::size_t rep = 0; rep < reps + 2; ++rep) {
        std::vector<std::int32_t> chunk{session->Argmax()};
        for (std::uint32_t j = 1; j < width; ++j)
          chunk.push_back(tokens[std::min<std::size_t>(session->Position() + j,
                                                       tokens.size() - 1)]);
        std::array<std::int32_t, ling::kMaxVerifyTokens> argmax{};
        const bool profile = rep == reps + 1;
        const auto t = now();
        if (!session->Verify(chunk, argmax, nullptr, &error,
                             profile ? &ops : nullptr))
          return fail("verify");
        const double call = ms_since(t);
        if (rep > 0 && !profile)
          series.Add(session->LastLaunchTiming(), call);
        const auto r = now();
        if (!session->Rollback(1, &error))
          return fail("rollback");
        if (rep > 0 && !profile)
          rollback_ms.push_back(ms_since(r));
      }
      std::cout << head("verify") << ", \"width\": " << width
                << ", \"launch\": " << series.Json()
                << ", \"rollback_ms\": " << std::fixed << std::setprecision(4)
                << LingLaunchSeries::Median(rollback_ms)
                << ", \"ms_per_step\": " << LingOpTotal(ops)
                << ", \"ops\": " << LingOpList(ops, 1) << "}\n"
                << std::flush;
    }
    // cycle: fixed-width greedy cycles.
    for (std::uint32_t width = 2; width <= ling::kMaxVerifyTokens; ++width) {
      if (!prepare())
        return false;
      double sample = 0, draft = 0, verify = 0, accept = 0, rollback = 0;
      double draft_device = 0, draft_issue = 0, verify_device = 0;
      double verify_issue = 0;
      std::vector<double> cycle;
      std::size_t committed = 0;
      std::size_t used = 0;
      for (std::size_t c = 0; c < steps + 1; ++c) {
        if (used + width + 1 > room)
          break;
        const auto t0 = now();
        const std::int32_t anchor = session->Argmax();
        const double t_sample = ms_since(t0);
        const auto t1 = now();
        std::vector<std::int32_t> chain{anchor};
        double d_device = 0, d_issue = 0;
        for (std::uint32_t j = 0; j + 1 < width; ++j) {
          std::int32_t index = -1;
          if (!session->Draft(chain.back(), j > 0, {}, &index, &error))
            return fail("cycle draft");
          d_device += session->LastLaunchTiming().device_ms;
          d_issue += session->LastLaunchTiming().issue_ms;
          chain.push_back(index);
        }
        const double t_draft = ms_since(t1);
        const auto t2 = now();
        std::array<std::int32_t, ling::kMaxVerifyTokens> argmax{};
        if (!session->Verify(chain, argmax, nullptr, &error))
          return fail("cycle verify");
        const auto verify_launch = session->LastLaunchTiming();
        const double t_verify = ms_since(t2);
        const auto t3 = now();
        std::uint32_t keep = 1;
        while (keep < width && argmax[keep - 1] == chain[keep])
          ++keep;
        const double t_accept = ms_since(t3);
        const auto t4 = now();
        if (!session->Rollback(keep, &error))
          return fail("cycle rollback");
        const double t_rollback = ms_since(t4);
        const double total = ms_since(t0);
        used += keep;
        if (c == 0)
          continue;  // warm-up cycle
        sample += t_sample;
        draft += t_draft;
        verify += t_verify;
        accept += t_accept;
        rollback += t_rollback;
        draft_device += d_device;
        draft_issue += d_issue;
        verify_device += verify_launch.device_ms;
        verify_issue += verify_launch.issue_ms;
        cycle.push_back(total);
        committed += keep;
      }
      const double cycles = static_cast<double>(cycle.size());
      const double sum = std::accumulate(cycle.begin(), cycle.end(), 0.0);
      const double per = cycles > 0 ? 1.0 / cycles : 0.0;
      std::cout << head("cycle") << ", \"width\": " << width
                << ", \"cycles\": " << cycle.size() << std::fixed
                << std::setprecision(4) << ", \"tokens_per_cycle\": "
                << static_cast<double>(committed) * per
                << ", \"cycle_ms\": " << sum * per
                << ", \"median_cycle_ms\": " << LingLaunchSeries::Median(cycle)
                << ", \"tokens_per_s\": "
                << (sum > 0 ? static_cast<double>(committed) * 1e3 / sum : 0.0)
                << ", \"phase_ms\": {\"sample\": " << sample * per
                << ", \"draft\": " << draft * per
                << ", \"verify\": " << verify * per
                << ", \"accept\": " << accept * per
                << ", \"rollback\": " << rollback * per
                << "}, \"draft_device_ms\": " << draft_device * per
                << ", \"draft_issue_ms\": " << draft_issue * per
                << ", \"verify_device_ms\": " << verify_device * per
                << ", \"verify_issue_ms\": " << verify_issue * per
                << ", \"host_overhead_ms\": "
                << (sum - draft_device - verify_device) * per << "}\n"
                << std::flush;
    }
    return true;
  };
  std::unique_ptr<ling::SessionSnapshot> snapshot;
  std::size_t built = 0;
  for (const std::size_t depth : depths) {
    // Real prefill of the missing tokens, continuing the chunk grid of the
    // previous depth (multiples of 512 keep the layout identical to a
    // direct prefill of d tokens).
    if (depth > built) {
      if (built % chunk != 0 && built != 0) {
        std::cerr << "Warning: depth " << built
                  << " is not a multiple of the prefill chunk; the chunk "
                     "layout of deeper depths differs from a direct prefill\n";
      }
      // Earlier tests left their tokens in the session; return to the
      // previous depth first (fresh state at depth 0).
      if (built == 0) {
        session->Reset();  // an empty session has no snapshot
      } else if (!session->RestoreSnapshot(*snapshot, &error)) {
        std::cerr << "Error restoring depth " << built << ": " << error << '\n';
        return 1;
      }
      if (session->Position() != built) {
        std::cerr << "Error: session is at " << session->Position()
                  << ", expected depth " << built << '\n';
        return 1;
      }
      const auto start = std::chrono::steady_clock::now();
      if (!session->Feed(std::span(tokens).subspan(built, depth - built), {},
                         &error)) {
        std::cerr << "Error building depth " << depth << ": " << error << '\n';
        return 1;
      }
      const double seconds = seconds_since(start);
      std::cerr << "[depth " << depth << "] prefilled " << depth - built
                << " tokens in " << seconds << " s ("
                << static_cast<double>(depth - built) / seconds
                << " tok/s, from depth " << built << ")\n";
      if (jsonl) {
        std::cout << "{\"record\": \"depth_fill\", \"n_from\": " << built
                  << ", \"n_to\": " << depth << ", \"seconds\": " << seconds
                  << ", \"ts\": "
                  << static_cast<double>(depth - built) / seconds << "}\n"
                  << std::flush;
      }
      built = depth;
    }
    // Depth 0 is the empty session: restoring it is a reset (a snapshot needs
    // at least one token).
    if (depth > 0) {
      snapshot = session->SaveSnapshot(&error);
      if (!snapshot) {
        std::cerr << "Error saving depth " << depth << ": " << error << '\n';
        return 1;
      }
    }
    const auto restore = [&]() {
      if (depth == 0) {
        session->Reset();
        return true;
      }
      if (!session->RestoreSnapshot(*snapshot, &error)) {
        std::cerr << "Error restoring depth " << depth << ": " << error << '\n';
        return false;
      }
      return true;
    };

    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> ns;
      StageMs stages;
      for (std::size_t repetition = 0; repetition <= options.repetitions;
           ++repetition) {
        if (!restore())
          return 1;
        std::vector<ling::StepTiming> timings;
        const auto start = std::chrono::steady_clock::now();
        if (!session->Feed(std::span(tokens).subspan(depth, prompt_length),
                           {.timings = &timings}, &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        if (repetition > 0) {
          ns.push_back(seconds_since(start) * 1e9);
          stages.Add(timings);
        }
      }
      emit(prompt_length, 0, depth, ns, stages);
      if (options.op_profile && jsonl) {
        if (!restore())
          return 1;
        std::vector<ling::OpTiming> ops;
        if (!session->Feed(std::span(tokens).subspan(depth, prompt_length),
                           {.op_profile = &ops}, &error)) {
          std::cerr << "Error running prefill profile: " << error << '\n';
          return 1;
        }
        double total_ms = 0.0;
        for (const auto& op : ops) {
          // "name/part" entries break an operation down; they are inside it.
          if (op.name.find('/') == std::string::npos)
            total_ms += op.ms;
        }
        std::cout << "{\"record\": \"op_profile\", \"n_prompt\": "
                  << prompt_length << ", \"n_depth\": " << depth
                  << ", \"ms\": " << std::fixed << std::setprecision(4)
                  << total_ms << ", \"ops\": [";
        for (std::size_t i = 0; i < ops.size(); ++i) {
          const auto& op = ops[i];
          std::cout << (i ? ", " : "") << "{\"name\": \"" << op.name
                    << "\", \"count\": " << op.count << ", \"ms\": " << op.ms
                    << ", \"mbytes\": " << op.bytes / 1e6 << "}";
        }
        std::cout << "]}\n" << std::flush;
      }
    }
    for (const std::size_t generation_length : options.n_gens) {
      if (depth + generation_length >= required_context) {
        std::cerr << "Error: generation benchmark exceeds context\n";
        return 1;
      }
      if (mtp) {
        // Greedy speculative generation from the depth: tokens are generated
        // (not fed), so acceptance reflects the context. Depth 0 starts from a
        // 16-token seed (a speculative session needs a prompt).
        std::vector<double> ns;
        ling::Session::DraftStats timed{};
        std::size_t emitted = 0;
        for (std::size_t repetition = 0; repetition <= options.repetitions;
             ++repetition) {
          if (!restore())
            return 1;
          if (depth == 0 &&
              !session->Feed(std::span(tokens).first(kLingMtpSeedTokens), {},
                             &error)) {
            std::cerr << "Error seeding the MTP session: " << error << '\n';
            return 1;
          }
          session->ResetDraftPolicy();
          session->SetMaxDrafts(options.draft_tokens);
          sampling::SamplerState sampler(options.sampling);
          const auto before = session->Statistics();
          const auto start = std::chrono::steady_clock::now();
          std::size_t generated = 0;
          while (generated < generation_length) {
            ling::Session::DecodeResult step;
            if (!session->DecodeStep(
                    std::min<std::size_t>(ling::kMaxVerifyTokens,
                                          generation_length - generated),
                    sampler, &step, &error)) {
              std::cerr << "Error running speculative decode: " << error
                        << '\n';
              return 1;
            }
            // DecodeStep already accepted every returned token.
            generated += step.tokens.size();
            if (step.stop) {
              std::cerr << "Error: the model emitted a stop token after "
                        << generated << " of " << generation_length
                        << " tokens; use a longer or different --text-file\n";
              return 1;
            }
          }
          if (repetition > 0) {
            ns.push_back(seconds_since(start) * 1e9);
            const auto after = session->Statistics();
            timed.cycles += after.cycles - before.cycles;
            timed.drafted += after.drafted - before.drafted;
            timed.accepted += after.accepted - before.accepted;
            emitted += generated;
          }
        }
        emit(0, generation_length, depth, ns, StageMs{});
        const double acceptance = timed.drafted != 0
                                      ? static_cast<double>(timed.accepted) /
                                            static_cast<double>(timed.drafted)
                                      : 0.0;
        const double per_cycle =
            timed.cycles != 0
                ? static_cast<double>(timed.accepted + timed.cycles) /
                      static_cast<double>(timed.cycles)
                : 0.0;
        if (jsonl) {
          std::cout << "{\"record\": \"mtp\", \"n_gen\": " << generation_length
                    << ", \"n_depth\": " << depth
                    << ", \"max_drafts\": " << options.draft_tokens
                    << ", \"verified_cycles\": " << timed.cycles
                    << ", \"drafted\": " << timed.drafted
                    << ", \"accepted\": " << timed.accepted
                    << ", \"acceptance\": " << std::fixed
                    << std::setprecision(4) << acceptance
                    << ", \"tokens_per_verified_cycle\": " << per_cycle
                    << ", \"generated_tokens\": " << emitted << "}\n"
                    << std::flush;
        } else {
          std::cout << "# mtp tg" << generation_length << " @ d" << depth
                    << ": verified cycles " << timed.cycles << ", drafted "
                    << timed.drafted << ", accepted " << timed.accepted << " ("
                    << std::fixed << std::setprecision(1) << 100.0 * acceptance
                    << "%), " << std::setprecision(2) << per_cycle
                    << " tokens per verified cycle\n"
                    << std::flush;
        }
        if (jsonl && options.op_profile &&
            !profile_mtp(depth, generation_length, restore)) {
          return 1;
        }
        continue;
      }
      std::vector<double> ns;
      StageMs stages;
      for (std::size_t repetition = 0; repetition <= options.repetitions;
           ++repetition) {
        if (!restore())
          return 1;
        const std::uint64_t replays_before = model->GraphReplays();
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < generation_length; ++i) {
          // Same path as Session::Evaluate; decode steps replay HIP graphs.
          if (!session->Feed(std::span(tokens).subspan(depth + i, 1), {},
                             &error)) {
            std::cerr << "Error running decode: " << error << '\n';
            return 1;
          }
        }
        if (repetition > 0) {
          ns.push_back(seconds_since(start) * 1e9);
          stages.graph_forwards +=
              static_cast<double>(model->GraphReplays() - replays_before) /
              static_cast<double>(options.repetitions);
        }
      }
      // One more repetition, eager, for the per-block device split (a
      // replayed graph is only measured as a whole); not part of the speed.
      if (!restore())
        return 1;
      {
        std::vector<ling::StepTiming> timings;
        timings.reserve(generation_length);
        for (std::size_t i = 0; i < generation_length; ++i) {
          if (!session->Feed(std::span(tokens).subspan(depth + i, 1),
                             {.timings = &timings}, &error)) {
            std::cerr << "Error running decode: " << error << '\n';
            return 1;
          }
        }
        stages.Add(timings);
        stages.reps = 1;
      }
      emit(0, generation_length, depth, ns, stages);
      if (jsonl && options.op_profile) {
        // Per-operation device time of one more eager decode run (rocprofv3
        // is not available in every pod): which launches the step spends its
        // time in, and the bandwidth each reaches on the bytes it must read.
        if (!restore())
          return 1;
        std::vector<ling::OpTiming> ops;
        for (std::size_t i = 0; i < generation_length; ++i) {
          if (!session->Feed(std::span(tokens).subspan(depth + i, 1),
                             {.op_profile = &ops}, &error)) {
            std::cerr << "Error running decode profile: " << error << '\n';
            return 1;
          }
        }
        double total_ms = 0.0;
        for (const auto& op : ops) {
          // "name/part" entries break an operation down; they are inside it.
          if (op.name.find('/') == std::string::npos)
            total_ms += op.ms;
        }
        const double steps = static_cast<double>(generation_length);
        std::cout << "{\"record\": \"op_profile\", \"n_gen\": "
                  << generation_length << ", \"n_depth\": " << depth
                  << ", \"ms_per_step\": " << std::fixed << std::setprecision(4)
                  << total_ms / steps << ", \"ops\": [";
        for (std::size_t i = 0; i < ops.size(); ++i) {
          const auto& op = ops[i];
          std::cout << (i ? ", " : "") << "{\"name\": \"" << op.name
                    << "\", \"count\": " << op.count / generation_length
                    << ", \"ms\": " << op.ms / steps
                    << ", \"mbytes\": " << op.bytes / steps / 1e6
                    << ", \"gbps\": "
                    << (op.ms > 0 ? op.bytes / (op.ms * 1e6) : 0.0) << "}";
        }
        std::cout << "]}\n" << std::flush;
      }
    }
  }
  return 0;
}

#endif

}  // namespace

void PrintBenchHelp(std::string_view program_name) {
  BenchOptions opt;
  bool explicit_p = false;
  bool explicit_n = false;
  bool speculative_explicit = false;
  ArgParser parser(
      std::string(program_name) + " bench",
      "Benchmark prompt processing (pp) and token generation (tg) throughput.");
  RegisterBenchOptions(parser, opt, explicit_p, explicit_n,
                       speculative_explicit);
  parser.PrintHelp();
}

std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg) {
  BenchOptions opt;
  bool explicit_p = false;
  bool explicit_n = false;

  gufo::cli::ArgParser parser(
      "gufo bench",
      "Benchmark prompt processing (pp) and token generation (tg) "
      "throughput.");
  bool speculative_explicit = false;
  RegisterBenchOptions(parser, opt, explicit_p, explicit_n,
                       speculative_explicit);

  if (!parser.Parse(args, error_msg)) {
    return std::nullopt;
  }
  if (parser.IsHelpRequested()) {
    return std::nullopt;
  }

  if (explicit_p && !explicit_n) {
    opt.n_gens.clear();
  } else if (!explicit_p && explicit_n) {
    opt.n_prompts.clear();
  }

  if (opt.repetitions == 0 || (opt.n_prompts.empty() && opt.n_gens.empty() &&
                               opt.validate_prefill_tokens == 0)) {
    if (error_msg)
      *error_msg =
          "Benchmark needs positive repetitions and at least one workload";
    return std::nullopt;
  }

  opt.speculative_explicit = speculative_explicit;
  if (!speculative_explicit && !opt.dspark_model_path.empty()) {
    opt.speculative_backend = "dspark";
  }

  if (opt.draft_tokens == 0 || opt.min_draft_tokens == 0 ||
      opt.min_draft_tokens > opt.draft_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "min-draft-tokens cannot exceed draft-tokens";
    }
    return std::nullopt;
  }
  try {
    opt.sampling.Validate();
    if (opt.sampling.seed < 0) {
      throw std::invalid_argument("benchmark seed must be non-negative");
    }
  } catch (const std::invalid_argument& error) {
    if (error_msg)
      *error_msg = error.what();
    return std::nullopt;
  }
  if (!opt.draft_policy.empty() &&
      ((opt.draft_policy != "fixed" && opt.draft_policy != "adaptive") ||
       opt.speculative_backend != "dflash2")) {
    if (error_msg != nullptr)
      *error_msg = "--draft-policy requires DFlash2 and fixed or adaptive";
    return std::nullopt;
  }
  if (opt.min_draft_tokens != 1 && opt.speculative_backend == "dflash2") {
    if (error_msg != nullptr)
      *error_msg =
          "DFlash2 requires --min-draft-tokens 1; bound blocks with "
          "--draft-tokens";
    return std::nullopt;
  }

  return opt;
}

int RunBench(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParseBenchOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintBenchHelp("gufo");
      return 2;
    }
    PrintBenchHelp("gufo");
    return 0;
  }

  const auto& opt = *opt_res;
  if (opt.model_path.empty()) {
    std::cerr << "Error: --model <PATH> is required\n";
    return 2;
  }

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  auto reader_owner = gufo::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader_owner) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));

#if defined(ENGINE_ENABLE_HIP)
  if (opt.output_format == "jsonl" && !IsLing3Flash(*reader)) {
    std::cerr << "Error: --output jsonl is available for Ling-3.0-flash only\n";
    return 2;
  }
  if (IsDeepSeekV4Flash(*reader)) {
    return RunDeepSeekBenchmark(opt, reader, model_load_start);
  }
  if (IsQwen38FlashNext(*reader)) {
    return RunQwen38FlashNextBenchmark(opt, reader, model_load_start);
  }
  if (IsLing3Flash(*reader)) {
    return RunLingBenchmark(opt, reader, model_load_start);
  }

  if (opt.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: concurrent model benchmarks currently support DS4 "
                 "only; use the serving benchmark for Qwen\n";
    return 1;
  }
  if (!opt.sampling.can_use_unmodified_argmax()) {
    std::cerr << "Error: sampled model benchmarks support DS4 and Flash-Next; "
                 "use the serving benchmark for Qwen\n";
    return 1;
  }

  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: No HIP GPU devices available for benchmarking.\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const double total_vram_mib =
      static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0);

  std::cout << "HIP: found " << device_count << " ROCm devices (Total VRAM: "
            << static_cast<std::size_t>(total_vram_mib) << " MiB):\n"
            << "  Device 0: " << prop.name << ", " << prop.gcnArchName
            << ", Wave Size: " << prop.warpSize
            << ", VRAM: " << static_cast<std::size_t>(total_vram_mib)
            << " MiB\n";

  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(opt.n_depths);
  const std::size_t max_test_tokens =
      std::max({max_or_zero(opt.n_prompts), max_or_zero(opt.n_gens),
                opt.validate_prefill_tokens});
  if (max_depth > std::numeric_limits<std::uint32_t>::max() - max_test_tokens) {
    std::cerr << "Error: requested benchmark context is too large.\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::size_t required_context =
      std::max<std::size_t>(4096, max_depth + max_test_tokens);

  auto gpu_exec = hip::QwenGpuExecutor::CreateFromGguf(
      reader, &err, static_cast<std::uint32_t>(required_context));
  if (!gpu_exec) {
    std::cerr << "Error creating Qwen GPU executor: " << err << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto& config = gpu_exec->GetConfig();
  std::shared_ptr<const hip::QwenMtpGpuModel> mtp_gpu_model;
  if (opt.speculative_backend == "mtp") {
    std::string mtp_path = opt.mtp_model_path;
    if (mtp_path.empty()) {
      std::cerr << "MTP requires --mtp-model\n";
      return 1;
    }
    auto mtp_reader_owner = core::GgufReader::OpenFile(mtp_path, &err);
    if (mtp_reader_owner == nullptr) {
      std::cerr << "Error loading MTP GGUF model: " << err << '\n';
      return 1;
    }
    std::shared_ptr<const core::GgufReader> mtp_reader(
        std::move(mtp_reader_owner));
    mtp_gpu_model = hip::QwenMtpGpuModel::Create(
        std::move(mtp_reader), gpu_exec->GetSharedModel(), &err);
    if (mtp_gpu_model == nullptr) {
      std::cerr << "Error creating GPU MTP model: " << err << '\n';
      return 1;
    }
    if (opt.verbose) {
      std::cerr << "Packed GPU MTP view: "
                << mtp_gpu_model->GetPackedWeightBytes() << " bytes in "
                << mtp_gpu_model->GetPackTimeSeconds() << " s\n";
    }
  }
  // A DFlash2 pp row includes the same feature capture and draft injection as
  // prompt/serving. Keep one backend alive across warmup, depth restores and
  // TG.
  std::unique_ptr<speculative::SpeculativeVerifier> dflash_verifier;
  if (opt.speculative_backend == "dflash2") {
    hip::QwenDFlashGpuDraftConfig draft_config{
        .max_context = static_cast<std::uint32_t>(required_context),
        .max_draft_tokens = opt.draft_tokens,
        .policy = speculative::ParseDFlashDraftPolicy(opt.draft_policy),
    };
    auto draft = hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
        opt.dflash_model_path, gpu_exec->GetSharedModel(), draft_config, &err);
    if (!draft) {
      throw std::runtime_error("DFlash2 initialization failed: " + err);
    }
    speculative::SpeculativeOptions options;
    options.max_draft_tokens = opt.draft_tokens;
    options.min_draft_tokens = opt.min_draft_tokens;
    options.initial_draft_tokens = opt.draft_tokens;
    options.enable_adaptive_draft_length = false;
    options.use_batched_verification = true;
    dflash_verifier = std::make_unique<speculative::SpeculativeVerifier>(
        *gpu_exec, std::move(draft), options);
  }
  const std::string model_name =
      config.model_name + " " + reader->GetQuantizationLabel();
  const double model_size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  std::uint64_t parameter_count = 0;
  for (const auto& tensor : reader->GetTensors()) {
    parameter_count += tensor.ElementCount();
  }
  const double model_params_b =
      static_cast<double>(parameter_count) / 1'000'000'000.0;

  if (opt.validate_prefill_tokens > 0 &&
      !ValidatePrefill(*gpu_exec, opt.validate_prefill_tokens)) {
    return 1;
  }

  std::cout << "| " << std::left << std::setw(30) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::right << std::setw(10) << "params"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(15) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n";
  std::cout << "| " << std::string(30, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(10, '-')
            << " | " << std::string(15, '-') << " | " << std::string(21, '-')
            << " |\n";

  // Warmup run
  {
    const auto warmup_tokens = MakeBenchmarkTokens(32);
    gpu_exec->Reset();
    (void)gpu_exec->ForwardPromptBatch(warmup_tokens);
    HIP_CHECK(hipDeviceSynchronize());
  }

  std::ostringstream ss_size, ss_params;
  ss_size << std::fixed << std::setprecision(2) << model_size_gib << " GiB";
  ss_params << std::fixed << std::setprecision(2) << model_params_b << " B";

  const std::string_view backend_name = "ROCm (HIP)";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream ss_ts;
    ss_ts << std::fixed << std::setprecision(2) << stats.mean << " ± "
          << stats.stddev;

    std::cout << "| " << std::left << std::setw(30) << model_name << " | "
              << std::right << std::setw(10) << ss_size.str() << " | "
              << std::right << std::setw(10) << ss_params.str() << " | "
              << std::left << std::setw(10) << backend_name << " | "
              << std::right << std::setw(15) << test_name << " | " << std::right
              << std::setw(21) << ss_ts.str() << " |\n"
              << std::flush;
  };

  // Verification owns the executor's SaveState rollback slot. A benchmark
  // prefix must have an independent snapshot across speculative TG repeats.
  std::unique_ptr<hip::QwenGpuSnapshot> prepared_target;
  std::unique_ptr<speculative::SpeculativeVerifierSnapshot> prepared_draft;
  std::size_t prepared_depth = 0;
  bool has_prepared_depth = false;
  tokenization::TokenId prepared_next_token = 0;
  bool has_prepared_next_token = false;
  for (const std::size_t depth : opt.n_depths) {
    if (depth > 0) {
      if (opt.verbose) {
        std::cerr << "Preparing context depth " << depth << "...\n";
      }
      std::size_t preparation_start = 0;
      double restore_ms = 0.0;
      if (has_prepared_depth && depth >= prepared_depth) {
        const auto restore_start = std::chrono::steady_clock::now();
        gpu_exec->RestoreSnapshot(*prepared_target);
        if (dflash_verifier)
          dflash_verifier->RestoreSnapshot(*prepared_draft);
        const auto restore_end = std::chrono::steady_clock::now();
        restore_ms = std::chrono::duration<double, std::milli>(restore_end -
                                                               restore_start)
                         .count();
        preparation_start = prepared_depth;
      } else {
        gpu_exec->Reset();
      }
      HIP_CHECK(hipDeviceSynchronize());
      const auto depth_tokens =
          MakeBenchmarkTokens(depth - preparation_start, preparation_start);
      const auto preparation_begin = std::chrono::steady_clock::now();
      const bool compute_next_token = !opt.n_gens.empty();
      if (!depth_tokens.empty()) {
        const auto next_token =
            dflash_verifier
                ? (preparation_start == 0
                       ? dflash_verifier->Prime(depth_tokens,
                                                compute_next_token)
                       : dflash_verifier->ExtendPrompt(
                             depth_tokens,
                             static_cast<std::uint32_t>(preparation_start),
                             compute_next_token))
                : gpu_exec->ForwardPromptBatch(
                      depth_tokens,
                      static_cast<std::uint32_t>(preparation_start),
                      compute_next_token);
        if (compute_next_token) {
          prepared_next_token = next_token;
          has_prepared_next_token = true;
        }
      }
      HIP_CHECK(hipDeviceSynchronize());
      const auto preparation_end = std::chrono::steady_clock::now();
      const auto cache_begin = preparation_end;
      prepared_target =
          gpu_exec->SaveSnapshot(static_cast<std::uint32_t>(depth));
      if (dflash_verifier)
        prepared_draft = dflash_verifier->Snapshot();
      const auto cache_end = std::chrono::steady_clock::now();
      if (opt.verbose) {
        const double preparation_seconds =
            std::chrono::duration<double>(preparation_end - preparation_begin)
                .count();
        const double cache_ms =
            std::chrono::duration<double, std::milli>(cache_end - cache_begin)
                .count();
        std::cerr << "Prepared " << (depth - preparation_start)
                  << " new tokens in " << std::fixed << std::setprecision(3)
                  << preparation_seconds << " s; cached depth " << depth
                  << " in " << std::setprecision(2) << cache_ms << " ms";
        if (preparation_start > 0) {
          std::cerr << " after restoring depth " << preparation_start << " in "
                    << restore_ms << " ms";
        }
        std::cerr << "\n";
      }
      prepared_depth = depth;
      has_prepared_depth = true;
    }

    const auto restore_depth = [&] {
      if (depth == 0) {
        gpu_exec->Reset();
      } else {
        gpu_exec->RestoreSnapshot(*prepared_target);
        if (dflash_verifier)
          dflash_verifier->RestoreSnapshot(*prepared_draft);
      }
      HIP_CHECK(hipDeviceSynchronize());
    };

    for (const std::size_t p_len : opt.n_prompts) {
      const auto prompt_tokens = MakeBenchmarkTokens(p_len, depth);

      const auto prefill = [&] {
        if (dflash_verifier) {
          if (depth == 0) {
            (void)dflash_verifier->Prime(prompt_tokens);
          } else {
            (void)dflash_verifier->ExtendPrompt(
                prompt_tokens, static_cast<std::uint32_t>(depth));
          }
        } else {
          (void)gpu_exec->ForwardPromptBatch(prompt_tokens,
                                             static_cast<std::uint32_t>(depth));
        }
        HIP_CHECK(hipDeviceSynchronize());
      };
      restore_depth();
      prefill();

      std::vector<double> runs;
      runs.reserve(opt.repetitions);
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        restore_depth();

        const auto t0 = std::chrono::high_resolution_clock::now();
        prefill();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double elapsed_sec =
            std::chrono::duration<double>(t1 - t0).count();
        if (elapsed_sec > 0.0) {
          runs.push_back(static_cast<double>(p_len) / elapsed_sec);
        }
      }

      print_result(MakeTestName("pp", p_len, depth), ComputeStats(runs));
    }

    for (const std::size_t g_len : opt.n_gens) {
      restore_depth();
      (void)gpu_exec->ForwardToken(
          static_cast<tokenization::TokenId>((depth % 1000) + 100),
          static_cast<std::uint32_t>(depth));
      HIP_CHECK(hipDeviceSynchronize());

      std::vector<double> runs;
      runs.reserve(opt.repetitions);
      std::vector<double> acceptance_runs;
      acceptance_runs.reserve(opt.repetitions);
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        restore_depth();

        std::unique_ptr<speculative::IDraftBackend> draft_backend;
        if (opt.speculative_backend == "mtp") {
          hip::QwenMtpGpuDraftConfig cfg{
              .max_context = static_cast<std::uint32_t>(required_context),
              .max_draft_tokens = opt.draft_tokens,
          };
          draft_backend =
              hip::QwenMtpGpuDraftBackend::Create(mtp_gpu_model, cfg, &err);
          if (draft_backend == nullptr) {
            throw std::runtime_error("MTP initialization failed: " + err);
          }
        }

        std::unique_ptr<speculative::SpeculativeVerifier> owned_spec_verifier;
        auto* spec_verifier = dflash_verifier.get();
        if (draft_backend) {
          speculative::SpeculativeOptions s_opts;
          s_opts.max_draft_tokens = opt.draft_tokens;
          s_opts.min_draft_tokens = opt.min_draft_tokens;
          s_opts.initial_draft_tokens = opt.draft_tokens;
          s_opts.use_batched_verification = true;
          owned_spec_verifier =
              std::make_unique<speculative::SpeculativeVerifier>(
                  *gpu_exec, std::move(draft_backend), s_opts);
          spec_verifier = owned_spec_verifier.get();
        }

        std::vector<tokenization::TokenId> speculative_sequence;
        tokenization::TokenId speculative_current_token = 0;
        std::uint32_t speculative_current_pos = 0;
        if (spec_verifier) {
          speculative_sequence =
              MakeBenchmarkTokens(depth > 0 ? depth : std::size_t{16});
          speculative_current_token =
              dflash_verifier && depth > 0
                  ? prepared_next_token
                  : spec_verifier->Prime(speculative_sequence);
          speculative_current_pos =
              static_cast<std::uint32_t>(speculative_sequence.size());
          speculative_sequence.push_back(speculative_current_token);
        }
        tokenization::TokenId greedy_current_token = 0;
        std::uint32_t greedy_current_pos = 0;
        if (!spec_verifier) {
          if (depth > 0) {
            if (!has_prepared_next_token) {
              throw std::logic_error(
                  "prepared generation depth has no target token");
            }
            greedy_current_token = prepared_next_token;
            greedy_current_pos = static_cast<std::uint32_t>(depth);
          } else {
            const auto prompt = MakeBenchmarkTokens(16);
            greedy_current_token = gpu_exec->ForwardPromptBatch(prompt);
            greedy_current_pos = static_cast<std::uint32_t>(prompt.size());
          }
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<tokenization::TokenId> generated;
        if (opt.verbose)
          generated.reserve(g_len);
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (spec_verifier) {
          std::size_t emitted = 0;
          while (emitted < g_len) {
            const auto step_res = spec_verifier->VerifyStep(
                speculative_sequence, speculative_current_pos,
                speculative_current_token, 999999,
                static_cast<std::uint32_t>(g_len - emitted));
            for (const auto t : step_res.emitted_tokens) {
              speculative_sequence.push_back(t);
              if (opt.verbose)
                generated.push_back(t);
              ++speculative_current_pos;
              ++emitted;
              if (emitted >= g_len) {
                break;
              }
            }
            speculative_current_token = step_res.next_token;
          }
          acceptance_runs.push_back(spec_verifier->GetStats().AcceptanceRate());
        } else {
          for (std::size_t step = 0; step < g_len; ++step) {
            greedy_current_token = gpu_exec->ForwardToken(greedy_current_token,
                                                          greedy_current_pos);
            ++greedy_current_pos;
            if (opt.verbose)
              generated.push_back(greedy_current_token);
          }
        }
        HIP_CHECK(hipDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (opt.verbose) {
          const auto* bytes =
              reinterpret_cast<const std::uint8_t*>(generated.data());
          std::cerr << "[QwenBenchTrace]: depth=" << depth
                    << " count=" << generated.size() << " sha256="
                    << crypto::Sha256Hex(std::span(
                           bytes,
                           generated.size() * sizeof(tokenization::TokenId)))
                    << '\n';
        }

        const double elapsed_sec =
            std::chrono::duration<double>(t1 - t0).count();
        if (elapsed_sec > 0.0) {
          runs.push_back(static_cast<double>(g_len) / elapsed_sec);
        }
      }

      std::string test_name = MakeTestName("tg", g_len, depth);
      if (!opt.speculative_backend.empty()) {
        test_name += "-" + opt.speculative_backend;
      }
      print_result(test_name, ComputeStats(runs));
      if (opt.verbose && !acceptance_runs.empty()) {
        const auto acceptance = ComputeStats(acceptance_runs);
        std::cerr << test_name << " acceptance=" << std::fixed
                  << std::setprecision(3) << acceptance.mean << " +/- "
                  << acceptance.stddev << '\n';
      }
    }
  }

  std::cout << "\n";
  return 0;
#else
  std::cerr << "Error: Gufo GPU benchmark requires ENGINE_ENABLE_HIP=ON\n";
  PrintModelLoadTime(model_load_start, false);
  return 1;
#endif
}

}  // namespace gufo::cli
