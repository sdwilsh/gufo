// A/B of the MTP draft head on real text: the trunk's F16 output head against
// the Q8_0 copy the draft step owns (mtp.hpp, "Draft head"). One model load;
// for each text window and each head it prefills the same prompt, generates
// greedily with a fixed number of drafts per cycle and reports draft
// acceptance, tokens per cycle and the draft step's device and wall time.
// Greedy output must be identical for both heads (verify uses the F16 head),
// which is checked and fails the run otherwise. Exit 77 without --model.
//
//   mtp_head_ab --model FIRST_SHARD.gguf --corpus LABEL=TEXT_FILE [--corpus ..]
//               [--windows W] [--prompt-tokens P] [--generate N] [--drafts D]
//               [--draft-steps S] [--max-batch N]
//
// Windows are W evenly spaced P-token slices of each text (raw text, control
// tokens not recognised). One JSON line per (text, head) summary and a final
// {"record": "summary"} line with the Q8/F16 ratios. Timings are only
// meaningful on the gfx1151 node.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/engine.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace sampling = gufo::sampling;

namespace {

struct Tally {
  std::uint64_t cycles{0}, drafted{0}, accepted{0}, tokens{0};
  std::vector<double> device_ms, wall_ms;
};

double Median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct WindowRun {
  std::vector<std::int32_t> tokens;
  l3::Session::DraftStats stats;
  std::vector<double> device_ms, wall_ms;
  bool ok{true};
};

WindowRun RunWindow(l3::Model& model, std::span<const std::int32_t> prompt,
                    std::uint32_t generate, std::uint32_t drafts,
                    std::uint32_t draft_steps, bool q8) {
  WindowRun run;
  model.SetMtpDraftHeadQ8(q8);
  std::string error;
  auto session = model.CreateSession(model.MaxContext(), &error);
  if (!session || !session->Feed(prompt, {}, &error)) {
    std::fprintf(stderr, "setup: %s\n", error.c_str());
    run.ok = false;
    return run;
  }
  session->SetMaxDrafts(7);
  session->SetDraftHooks({.fixed_drafts = drafts});
  sampling::SamplerState sampler{sampling::SamplingConfig{}};
  while (run.tokens.size() < generate) {
    const std::size_t budget = std::min<std::size_t>(
        l3::kMaxVerifyTokens, generate - run.tokens.size());
    l3::Session::DecodeResult step;
    if (!session->DecodeStep(budget, sampler, &step, &error, false)) {
      std::fprintf(stderr, "decode: %s\n", error.c_str());
      run.ok = false;
      return run;
    }
    for (const auto token : step.tokens) {
      sampler.Accept(static_cast<sampling::TokenId>(token));
      run.tokens.push_back(token);
    }
    if (step.stop) {
      break;
    }
  }
  run.stats = session->Statistics();
  // Draft step cost: chained drafts of the width above from the committed
  // position (the first round warms the kernels).
  const std::int32_t anchor = session->Argmax();
  for (std::uint32_t round = 0; round * drafts < draft_steps + drafts;
       ++round) {
    std::int32_t token = anchor;
    for (std::uint32_t j = 0; j < drafts; ++j) {
      std::int32_t index = -1;
      if (!session->Draft(token, j > 0, {}, &index, &error)) {
        std::fprintf(stderr, "draft: %s\n", error.c_str());
        run.ok = false;
        return run;
      }
      token = index;
      if (round > 0) {
        const auto t = session->LastLaunchTiming();
        run.device_ms.push_back(t.device_ms);
        run.wall_ms.push_back(t.wall_ms);
      }
    }
  }
  return run;
}

}  // namespace

int main(int argc, char** argv) {
  if (!gufo::core::RequireTempDirectory()) {
    return 1;
  }
  std::string model_path;
  std::vector<std::pair<std::string, std::string>> corpora;
  std::uint32_t windows = 4, prompt_tokens = 1024, generate = 192, drafts = 3,
                draft_steps = 48, max_batch = 512;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : "";
    };
    if (a == "--model") {
      model_path = next();
    } else if (a == "--corpus") {
      const std::string v = next();
      const auto eq = v.find('=');
      corpora.emplace_back(eq == std::string::npos ? v : v.substr(0, eq),
                           eq == std::string::npos ? v : v.substr(eq + 1));
    } else if (a == "--windows") {
      windows = std::stoul(next());
    } else if (a == "--prompt-tokens") {
      prompt_tokens = std::stoul(next());
    } else if (a == "--generate") {
      generate = std::stoul(next());
    } else if (a == "--drafts") {
      drafts = std::stoul(next());
    } else if (a == "--draft-steps") {
      draft_steps = std::stoul(next());
    } else if (a == "--max-batch") {
      max_batch = std::stoul(next());
    }
  }
  if (model_path.empty() || corpora.empty()) {
    std::fprintf(stderr, "no --model or --corpus: skipping\n");
    return 77;
  }
  drafts = std::clamp<std::uint32_t>(drafts, 2, 7);
  l3::ModelOptions options;
  options.mtp = true;
  options.max_batch = max_batch;
  options.max_context = prompt_tokens + generate + 128;
  options.expanded_kv_bytes = 0;
  std::string error;
  auto model = l3::Model::Load(model_path, options, &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  bool ok = true;
  std::map<std::string, Tally> total;  // by head
  for (const auto& [label, path] : corpora) {
    std::ifstream in(path, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(in), {}};
    const auto ids = model->Tokenize(text);
    if (ids.size() < prompt_tokens + 1) {
      std::fprintf(stderr, "%s: only %zu tokens\n", label.c_str(), ids.size());
      return 1;
    }
    Tally tally[2];  // 0 = F16 head, 1 = Q8 head
    std::size_t identical = 0;
    const std::size_t span = ids.size() - prompt_tokens;
    for (std::uint32_t w = 0; w < windows; ++w) {
      const std::size_t start = windows > 1 ? span * w / (windows - 1) : 0;
      const std::span<const std::int32_t> prompt(ids.data() + start,
                                                 prompt_tokens);
      WindowRun runs[2];
      for (int head = 0; head < 2; ++head) {
        runs[head] =
            RunWindow(*model, prompt, generate, drafts, draft_steps, head == 1);
        ok = ok && runs[head].ok;
        Tally& t = tally[head];
        t.cycles += runs[head].stats.cycles;
        t.drafted += runs[head].stats.drafted;
        t.accepted += runs[head].stats.accepted;
        t.tokens += runs[head].tokens.size();
        t.device_ms.insert(t.device_ms.end(), runs[head].device_ms.begin(),
                           runs[head].device_ms.end());
        t.wall_ms.insert(t.wall_ms.end(), runs[head].wall_ms.begin(),
                         runs[head].wall_ms.end());
      }
      const bool same = runs[0].tokens == runs[1].tokens;
      identical += same ? 1 : 0;
      ok = ok && same;
      std::fprintf(stderr, "[%s window %u] f16 %llu/%llu q8 %llu/%llu %s\n",
                   label.c_str(), w,
                   static_cast<unsigned long long>(runs[0].stats.accepted),
                   static_cast<unsigned long long>(runs[0].stats.drafted),
                   static_cast<unsigned long long>(runs[1].stats.accepted),
                   static_cast<unsigned long long>(runs[1].stats.drafted),
                   same ? "tokens identical" : "TOKENS DIFFER");
    }
    for (int head = 0; head < 2; ++head) {
      const Tally& t = tally[head];
      Tally& all = total[head == 0 ? "f16" : "q8"];
      all.cycles += t.cycles;
      all.drafted += t.drafted;
      all.accepted += t.accepted;
      all.tokens += t.tokens;
      all.device_ms.insert(all.device_ms.end(), t.device_ms.begin(),
                           t.device_ms.end());
      all.wall_ms.insert(all.wall_ms.end(), t.wall_ms.begin(), t.wall_ms.end());
      std::printf(
          "{\"record\": \"head\", \"text\": \"%s\", \"head\": \"%s\", "
          "\"windows\": %u, \"prompt_tokens\": %u, \"generated\": %llu, "
          "\"drafts_per_cycle\": %u, \"cycles\": %llu, \"drafted\": %llu, "
          "\"accepted\": %llu, \"acceptance\": %.4f, \"tokens_per_cycle\": "
          "%.4f, \"draft_device_ms\": %.4f, \"draft_wall_ms\": %.4f, "
          "\"tokens_identical_windows\": %zu}\n",
          label.c_str(), head == 0 ? "f16" : "q8", windows, prompt_tokens,
          static_cast<unsigned long long>(t.tokens), drafts,
          static_cast<unsigned long long>(t.cycles),
          static_cast<unsigned long long>(t.drafted),
          static_cast<unsigned long long>(t.accepted),
          t.drafted ? static_cast<double>(t.accepted) / t.drafted : 0.0,
          t.cycles ? static_cast<double>(t.accepted + t.cycles) / t.cycles
                   : 0.0,
          Median(t.device_ms), Median(t.wall_ms), identical);
    }
    std::fflush(stdout);
  }
  const auto rate = [](const Tally& t) {
    return t.drafted ? static_cast<double>(t.accepted) / t.drafted : 0.0;
  };
  const Tally &f16 = total["f16"], &q8 = total["q8"];
  std::printf(
      "{\"record\": \"summary\", \"pass\": %s, \"acceptance_f16\": %.4f, "
      "\"acceptance_q8\": %.4f, \"acceptance_ratio\": %.4f, "
      "\"draft_device_ms_f16\": %.4f, \"draft_device_ms_q8\": %.4f, "
      "\"draft_wall_ms_f16\": %.4f, \"draft_wall_ms_q8\": %.4f}\n",
      ok ? "true" : "false", rate(f16), rate(q8),
      rate(f16) > 0 ? rate(q8) / rate(f16) : 0.0, Median(f16.device_ms),
      Median(q8.device_ms), Median(f16.wall_ms), Median(q8.wall_ms));
  return ok ? 0 : 1;
}
