// Runs the float32 Ling-3.0-flash reference over token ids and writes
// per-layer dumps plus logits for the oracle and GPU comparison tools.
//
//   reference_probe --model FIRST_SHARD.gguf
//                   (--tokens 1,2,3 | --text-ids-file FILE)
//                   --dump-dir DIR [--layers all|none|0,5,41]
//                   [--teacher-input-dir DIR2] [--generate N]
//                   [--logits all|last|none] [--logits-from-pos N]
//                   [--chunk N] [--threads N] [--f16-cache] [--no-mla-check]
//                   [--emulate-gpu-f16 [--emulate-f16-head]]
//                   [--f64] [--dump-names a,b,c] [--top-k N]
//
// --f64 accumulates every dot product, norm, softmax, the router, the MLA
// attention and the KDA recurrence in double (weights are the same dequantised
// floats; activations between operators stay float). Test-only tie-break
// reference. --dump-names keeps only the named intermediates (for example
// router_ids,router_top_ids,router_top_sel). --top-k prints the N best
// log-probabilities (double log-softmax of the float logits) of every position
// whose logits are evaluated.
//
// Free-running mode feeds the tokens through the whole trunk and writes, per
// selected block, the intermediates listed in reference.hpp
// (`<name>.lNN.f32`, shape [tokens, n]) plus `logits.f32` and `index.json`.
// `--layers none` writes logits only. `--chunk N` clears all state every N
// tokens (llama-perplexity chunks) and
// `--logits-from-pos N` only evaluates the output matrix at in-chunk
// positions >= N. Teacher mode reads `h_out.l(il-1).f32` (`embd.l00.f32` for
// block 0) from DIR2 and runs each selected block alone on those inputs.

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/reference.hpp"
#include "src/models/ling_3_flash/weights.hpp"

namespace l3 = gufo::models::ling_3_flash;

namespace {

std::vector<std::int64_t> ParseList(const std::string& text) {
  std::vector<std::int64_t> out;
  std::string cur;
  for (char ch : text + ",") {
    if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') {
      if (!cur.empty()) {
        out.push_back(std::atoll(cur.c_str()));
        cur.clear();
      }
    } else {
      cur.push_back(ch);
    }
  }
  return out;
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

long PeakRssKiB() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      return std::atol(line.c_str() + 6);
    }
  }
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string tokens_arg;
  std::string ids_file;
  std::string dump_dir;
  std::string layers_arg = "all";
  std::string teacher_dir;
  std::string logits_mode = "all";
  int generate = 0;
  std::int64_t logits_from = 0;
  std::int64_t chunk = 0;
  unsigned threads = 8;
  bool f16_cache = false;
  bool emulate_f16 = false;
  bool emulate_head = false;
  bool mla_check = true;
  bool f64 = false;
  int top_k = 0;
  std::string dump_names;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--tokens") {
      tokens_arg = next();
    } else if (arg == "--text-ids-file") {
      ids_file = next();
    } else if (arg == "--dump-dir") {
      dump_dir = next();
    } else if (arg == "--layers") {
      layers_arg = next();
    } else if (arg == "--teacher-input-dir") {
      teacher_dir = next();
    } else if (arg == "--logits") {
      logits_mode = next();
    } else if (arg == "--logits-from-pos") {
      logits_from = std::atoll(next().c_str());
    } else if (arg == "--chunk") {
      chunk = std::atoll(next().c_str());
    } else if (arg == "--generate") {
      generate = std::atoi(next().c_str());
    } else if (arg == "--threads") {
      threads = static_cast<unsigned>(std::atoi(next().c_str()));
    } else if (arg == "--f16-cache") {
      f16_cache = true;
    } else if (arg == "--emulate-gpu-f16") {
      emulate_f16 = true;
    } else if (arg == "--emulate-f16-head") {
      emulate_head = true;
    } else if (arg == "--f64") {
      f64 = true;
    } else if (arg == "--top-k") {
      top_k = std::atoi(next().c_str());
    } else if (arg == "--dump-names") {
      dump_names = next();
    } else if (arg == "--no-mla-check") {
      mla_check = false;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty() || dump_dir.empty()) {
    std::fprintf(stderr, "--model and --dump-dir are required\n");
    return 2;
  }

  std::vector<int> layers;
  if (layers_arg == "none") {
    layers.push_back(-2);  // matches no block; only logits are written
  } else if (layers_arg != "all") {
    for (auto v : ParseList(layers_arg)) {
      layers.push_back(static_cast<int>(v));
    }
    layers.push_back(-1);  // model-level tensors follow an explicit list
  }

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  // Expert slices are touched sparsely; undo the reader's sequential hint.
  for (const auto& region : reader->GetMappedRegions()) {
    (void)madvise(const_cast<void*>(region.data), region.size, MADV_RANDOM);
  }
  auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  const auto& c = weights->config;

  l3::DumpRecorder recorder(layers);
  if (!dump_names.empty()) {
    std::vector<std::string> names;
    std::string cur;
    for (char ch : dump_names + ",") {
      if (ch == ',') {
        if (!cur.empty()) {
          names.push_back(cur);
          cur.clear();
        }
      } else {
        cur.push_back(ch);
      }
    }
    recorder.SetNames(std::move(names));
  }
  std::string extra_entries;
  std::string extra_fields;
  const auto t0 = std::chrono::steady_clock::now();

  if (!teacher_dir.empty()) {
    // ---- Teacher mode ------------------------------------------------------
    std::vector<int> run;
    for (int il : layers) {
      if (il >= 0) {
        run.push_back(il);
      }
    }
    if (run.empty()) {
      for (std::uint32_t il = 0; il < c.num_layers; ++il) {
        run.push_back(static_cast<int>(il));
      }
    }
    std::sort(run.begin(), run.end());
    l3::ReferenceModel::Options options;
    options.max_context = 1;
    options.threads = threads;
    options.f16_cache = f16_cache;
    options.emulate_gpu_f16 = emulate_f16;
    options.emulate_f16_head = emulate_head;
    options.check_mla = mla_check;
    options.f64 = f64;
    std::uint32_t tokens = 0;
    for (int il : run) {
      char name[64];
      if (il == 0) {
        std::snprintf(name, sizeof(name), "embd.l00.f32");
      } else {
        std::snprintf(name, sizeof(name), "h_out.l%02d.f32", il - 1);
      }
      std::vector<float> in;
      if (!l3::ReadF32File(teacher_dir + "/" + name, &in) ||
          in.size() % l3::kHidden != 0 || in.empty()) {
        std::fprintf(stderr, "cannot read teacher input %s/%s\n",
                     teacher_dir.c_str(), name);
        return 1;
      }
      tokens = static_cast<std::uint32_t>(in.size() / l3::kHidden);
      options.max_context = std::max(options.max_context, tokens);
    }
    l3::ReferenceModel model(*weights, options);
    for (int il : run) {
      char name[64];
      if (il == 0) {
        std::snprintf(name, sizeof(name), "embd.l00.f32");
      } else {
        std::snprintf(name, sizeof(name), "h_out.l%02d.f32", il - 1);
      }
      std::vector<float> in;
      (void)l3::ReadF32File(teacher_dir + "/" + name, &in);
      tokens = static_cast<std::uint32_t>(in.size() / l3::kHidden);
      std::vector<float> out(l3::kHidden);
      for (std::uint32_t t = 0; t < tokens; ++t) {
        if (!model.RunLayer(
                static_cast<std::uint32_t>(il),
                {in.data() + std::size_t{t} * l3::kHidden, l3::kHidden}, out,
                &recorder, &error)) {
          std::fprintf(stderr, "layer %d token %u failed: %s\n", il, t,
                       error.c_str());
          return 1;
        }
      }
      std::printf("teacher layer %d: %u tokens\n", il, tokens);
      std::fflush(stdout);
    }
    extra_fields = "  \"mode\": \"teacher\",\n  \"teacher_input_dir\": \"" +
                   teacher_dir + "\",\n  \"hidden\": 2560,\n";
    if (mla_check) {
      const auto& chk = model.mla_check();
      extra_fields +=
          "  \"mla_check\": {\"layer\": " + std::to_string(chk.layer) +
          ", \"comparisons\": " + std::to_string(chk.comparisons) +
          ", \"max_rel_l2\": " + std::to_string(chk.max_rel_l2) + "},\n";
      std::printf(
          "mla absorbed vs non-absorbed: layer %d, %u comparisons, max rel-L2 "
          "%.3e\n",
          chk.layer, chk.comparisons, chk.max_rel_l2);
    }
  } else {
    // ---- Free-running mode -------------------------------------------------
    std::vector<std::int32_t> tokens;
    if (!tokens_arg.empty()) {
      for (auto v : ParseList(tokens_arg)) {
        tokens.push_back(static_cast<std::int32_t>(v));
      }
    } else if (!ids_file.empty()) {
      for (auto v : ParseList(ReadFile(ids_file))) {
        tokens.push_back(static_cast<std::int32_t>(v));
      }
    }
    if (tokens.empty()) {
      std::fprintf(stderr, "--tokens or --text-ids-file is required\n");
      return 2;
    }
    const std::size_t prompt_len = tokens.size();
    const std::size_t total = prompt_len + static_cast<std::size_t>(generate);
    l3::ReferenceModel::Options options;
    options.max_context = chunk > 0 ? static_cast<std::uint32_t>(chunk)
                                    : static_cast<std::uint32_t>(total);
    options.threads = threads;
    options.f16_cache = f16_cache;
    options.emulate_gpu_f16 = emulate_f16;
    options.emulate_f16_head = emulate_head;
    options.check_mla = mla_check;
    options.f64 = f64;
    l3::ReferenceModel model(*weights, options);

    std::ofstream logits_out;
    std::vector<std::size_t> logit_positions;
    if (logits_mode != "none") {
      std::filesystem::create_directories(dump_dir);
      logits_out.open(dump_dir + "/logits.f32", std::ios::binary);
    }
    std::vector<float> logits(c.vocab_size);
    std::vector<std::int32_t> argmaxes;
    for (std::size_t i = 0; i < total; ++i) {
      if (chunk > 0 && i > 0 && i % static_cast<std::size_t>(chunk) == 0) {
        model.Reset();
      }
      const std::size_t in_chunk =
          chunk > 0 ? i % static_cast<std::size_t>(chunk) : i;
      if (i >= tokens.size()) {
        tokens.push_back(argmaxes.back());
      }
      bool want_logits = false;
      if (logits_mode == "all") {
        want_logits = static_cast<std::int64_t>(in_chunk) >= logits_from;
      } else if (logits_mode == "last") {
        want_logits = i + 1 == total;
      }
      if (generate > 0 && i + 1 >= prompt_len && i + 1 < total) {
        want_logits = true;  // generation needs the argmax
      }
      const auto s = std::chrono::steady_clock::now();
      if (!model.Step(
              tokens[i],
              want_logits ? std::span<float>(logits) : std::span<float>(),
              &recorder, &error)) {
        std::fprintf(stderr, "step %zu failed: %s\n", i, error.c_str());
        return 1;
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - s)
                            .count();
      if (want_logits) {
        const auto top = static_cast<std::int32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        argmaxes.push_back(top);
        if (top_k > 0) {
          double mx = logits[static_cast<std::size_t>(top)];
          double sum = 0.0;
          for (float v : logits) {
            sum += std::exp(static_cast<double>(v) - mx);
          }
          std::vector<std::int32_t> idx(logits.size());
          std::iota(idx.begin(), idx.end(), 0);
          const auto kk = static_cast<std::ptrdiff_t>(std::min<std::size_t>(
              static_cast<std::size_t>(top_k), logits.size()));
          std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                            [&](std::int32_t a, std::int32_t b) {
                              return logits[static_cast<std::size_t>(a)] >
                                     logits[static_cast<std::size_t>(b)];
                            });
          std::printf("top pos %zu:", i);
          for (std::ptrdiff_t r = 0; r < kk; ++r) {
            const double lg = logits[static_cast<std::size_t>(idx[r])];
            std::printf(" %d %.6f", idx[r], lg - mx - std::log(sum));
          }
          std::printf("\n");
        }
        std::printf("pos %zu token %d -> argmax %d (%.2f) [%.0f ms]\n", i,
                    tokens[i], top, logits[static_cast<std::size_t>(top)], ms);
        if (logits_out.is_open() && logits_mode != "none") {
          logits_out.write(reinterpret_cast<const char*>(logits.data()),
                           static_cast<std::streamsize>(logits.size() * 4));
          logit_positions.push_back(i);
        }
      } else {
        argmaxes.push_back(-1);
        std::printf("pos %zu token %d [%.0f ms]\n", i, tokens[i], ms);
      }
      std::fflush(stdout);
    }
    extra_fields = "  \"mode\": \"free\",\n  \"hidden\": 2560,\n  \"vocab\": " +
                   std::to_string(c.vocab_size) +
                   ",\n  \"chunk\": " + std::to_string(chunk) +
                   ",\n  \"tokens\": [";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      extra_fields += (i ? ", " : "") + std::to_string(tokens[i]);
    }
    extra_fields += "],\n";
    if (mla_check) {
      const auto& chk = model.mla_check();
      extra_fields +=
          "  \"mla_check\": {\"layer\": " + std::to_string(chk.layer) +
          ", \"comparisons\": " + std::to_string(chk.comparisons) +
          ", \"max_rel_l2\": " + std::to_string(chk.max_rel_l2) + "},\n";
      std::printf(
          "mla absorbed vs non-absorbed: layer %d, %u comparisons, max rel-L2 "
          "%.3e\n",
          chk.layer, chk.comparisons, chk.max_rel_l2);
      if (chk.comparisons > 0 && chk.max_rel_l2 >= 1e-5) {
        std::fprintf(stderr, "FAIL: absorbed/non-absorbed MLA disagree\n");
      }
    }
    if (logits_out.is_open()) {
      logits_out.close();
      std::string positions;
      for (std::size_t i = 0; i < logit_positions.size(); ++i) {
        positions += (i ? ", " : "") + std::to_string(logit_positions[i]);
      }
      extra_entries =
          "    {\"name\": \"logits\", \"layer\": -1, \"file\": "
          "\"logits.f32\", \"dtype\": \"f32\", \"shape\": [" +
          std::to_string(logit_positions.size()) + ", " +
          std::to_string(c.vocab_size) + "], \"positions\": [" + positions +
          "]}";
    }
    if (generate > 0) {
      std::printf("generated:");
      for (std::size_t i = tokens.size() - static_cast<std::size_t>(generate);
           i < tokens.size(); ++i) {
        std::printf(" %d", tokens[i]);
      }
      std::printf("\n");
    }
  }

  if (!recorder.Write(dump_dir, extra_entries, extra_fields, &error)) {
    std::fprintf(stderr, "dump failed: %s\n", error.c_str());
    return 1;
  }
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::printf("done in %.1f s, peak RSS %.2f GiB\n", secs,
              static_cast<double>(PeakRssKiB()) / (1024.0 * 1024.0));
  return 0;
}
