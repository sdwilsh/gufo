// Host-only check that sampled MTP decoding, cycle by cycle as
// Session::DecodeStep and the server drive it, draws every committed token
// from the autoregressive target distribution. A synthetic model supplies
// prefix-dependent target and draft logits (two stop tokens, a stop
// probability growing with the length), so whole sequences can be generated
// with the production proposal and verification code (MtpProposer,
// VerifySampledChain), the deferred residual draw, a varying draft count and
// the server's working-copy sampler, under the Ling preset, with penalties,
// with top-p/min-p and with an unfiltered support wider than the 64 proposal
// candidates. Every sequence is then teacher-forced through a plain sampler.
// The randomized PIT values must be U(0, 1) (Kolmogorov-Smirnov), the stop and
// top-1 counts must match their exact expectations, and no committed token may
// have p = 0 (sampled_distribution.hpp). Plain autoregressive sampling runs
// through the same checks as a control of the statistics.
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/ling_3_flash/mtp.hpp"
#include "src/models/ling_3_flash/mtp_sampling.hpp"
#include "tests/models/ling_3_flash/sampled_distribution.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace sampling = gufo::sampling;
using gufo::tests::ling::Calibration;
using gufo::tests::ling::kStopOutcome;
using gufo::tests::ling::KsUniform;
using gufo::tests::ling::Pit;
using gufo::tests::ling::Sequence;

namespace {

int g_failures = 0;
void Expect(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
  if (!ok) {
    ++g_failures;
  }
}

constexpr std::size_t kVocab = 200;
bool IsStop(sampling::TokenId token) {
  return token == 0 || token == 1;
}

std::uint64_t Mix(std::uint64_t x) {
  x ^= x >> 33U;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33U;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33U;
  return x;
}

double Unit(std::uint64_t h) {
  return static_cast<double>(h >> 11U) * 0x1.0p-53;
}

/// Context key: the last two tokens and the generated length.
std::uint64_t Key(std::span<const std::int32_t> context, std::size_t generated,
                  std::uint64_t salt) {
  std::uint64_t h = Mix(salt + 0x9E3779B97F4A7C15ULL * (generated + 1));
  for (std::size_t i = context.size() > 2 ? context.size() - 2 : 0;
       i < context.size(); ++i) {
    h = Mix(h ^ static_cast<std::uint64_t>(context[i] + 7));
  }
  return h;
}

/// Target logits after `context` (`generated` tokens of the response).
std::vector<float> Target(std::span<const std::int32_t> context,
                          std::size_t generated) {
  const std::uint64_t h = Key(context, generated, 1);
  std::vector<float> logits(kVocab);
  for (std::size_t i = 0; i < kVocab; ++i) {
    // Zipf-like base, context-dependent jitter, a few strong candidates.
    const double jitter = Unit(Mix(h + i)) - 0.5;
    logits[i] = static_cast<float>(-1.1 * std::log(1.0 + i) + 3.0 * jitter +
                                   ((Mix(h ^ (i * 31)) & 15U) == 0 ? 2.5 : 0));
  }
  logits[0] = static_cast<float>(-4.0 + 0.6 * static_cast<double>(generated));
  logits[1] = static_cast<float>(-5.0 + 0.3 * static_cast<double>(generated));
  return logits;
}

/// Draft logits: the target with context-dependent noise (sometimes none).
std::vector<float> Draft(std::span<const std::int32_t> context,
                         std::size_t generated) {
  auto logits = Target(context, generated);
  const std::uint64_t h = Key(context, generated, 2);
  const double sigma = (h & 3U) == 0 ? 0.0 : 1.5;
  for (std::size_t i = 0; i < kVocab; ++i) {
    logits[i] += static_cast<float>(sigma * (Unit(Mix(h + 977 * i)) - 0.5));
  }
  return logits;
}

/// Unrelated drafts (another context's target): almost every draft is
/// rejected, so nearly every token after the first comes from a residual draw
/// (the regime of a truncated trunk).
enum class Mode {
  kAutoregressive,
  kVaryingDrafts,
  kController,
  kUnrelatedDrafts
};

struct DraftStats {
  std::uint64_t cycles{0};
  std::uint64_t drafted{0};
  std::uint64_t accepted{0};
};

/// One request as the server runs it: the pool's sampler accepts committed
/// tokens; each step works on a copy whose draw state (RNG, deferred residual
/// draw) is published afterwards (LingTextRunner::DecodeStep).
Sequence Generate(const sampling::SamplingConfig& config,
                  std::span<const std::int32_t> prompt, std::size_t length,
                  Mode mode, DraftStats* stats) {
  std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState pool(config, history);
  std::vector<std::int32_t> context(prompt.begin(), prompt.end());
  l3::MtpLengthController controller;
  Sequence out;
  std::uint64_t cycle = 0;
  while (out.tokens.size() < length) {
    const std::size_t budget =
        std::min<std::size_t>(l3::kMaxVerifyTokens, length - out.tokens.size());
    sampling::SamplerState working = pool;
    const std::size_t generated = out.tokens.size();
    const auto anchor = working.Sample(Target(context, generated));
    if (IsStop(anchor)) {
      out.stopped = true;
      break;
    }
    std::uint32_t drafts = 0;
    if (mode == Mode::kVaryingDrafts || mode == Mode::kUnrelatedDrafts) {
      drafts = static_cast<std::uint32_t>(
          Mix(static_cast<std::uint64_t>(config.seed) * 131 + cycle) % 8);
    } else if (mode == Mode::kController) {
      drafts = controller.Choose(7, false, 0);
    }
    drafts =
        std::min<std::uint32_t>(drafts, static_cast<std::uint32_t>(budget - 1));
    ++cycle;
    std::vector<std::int32_t> committed;
    bool stop = false;
    if (drafts == 0) {
      working.Accept(anchor);
      committed = {static_cast<std::int32_t>(anchor)};
      controller.ObservePlainToken();
    } else {
      l3::MtpProposer proposer(working, anchor);
      std::vector<std::int32_t> chain{static_cast<std::int32_t>(anchor)};
      std::vector<std::vector<float>> rows;
      for (std::uint32_t j = 0; j < drafts; ++j) {
        auto with = context;
        with.insert(with.end(), chain.begin(), chain.end());
        chain.push_back(static_cast<std::int32_t>(
            proposer.Propose(mode == Mode::kUnrelatedDrafts
                                 ? Target(with, generated + chain.size() + 1000)
                                 : Draft(with, generated + chain.size()))));
      }
      for (std::size_t j = 0; j < chain.size(); ++j) {
        auto with = context;
        with.insert(with.end(), chain.begin(), chain.begin() + j + 1);
        rows.push_back(Target(with, generated + j + 1));
      }
      working.Accept(anchor);
      const auto verified = l3::VerifySampledChain(
          [&](std::uint32_t row) { return std::span<const float>(rows[row]); },
          proposer.proposals(), working, IsStop);
      committed.assign(chain.begin(), chain.begin() + verified.keep);
      stop = verified.stop;
      if (verified.correction) {
        working.DeferSample(*verified.correction);
      }
      controller.Observe(verified.keep - 1, stop ? verified.keep - 1 : drafts,
                         false, 0);
      if (stats != nullptr) {
        ++stats->cycles;
        stats->drafted += drafts;
        stats->accepted += verified.keep - 1;
      }
    }
    pool.CopyDrawStateFrom(working);
    for (const auto token : committed) {
      pool.Accept(static_cast<sampling::TokenId>(token));
      context.push_back(token);
      out.tokens.push_back(token);
    }
    if (stop) {
      out.stopped = true;
      break;
    }
  }
  return out;
}

struct Evidence {
  std::vector<double> pit;
  std::size_t impossible{0};
  Calibration stops;
  Calibration top;
};

/// Teacher-forces `s` through a plain sampler with the request's history.
void Score(const sampling::SamplingConfig& config,
           std::span<const std::int32_t> prompt, const Sequence& s,
           std::mt19937_64& rng, Evidence* e) {
  std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState plain(config, history);
  std::vector<std::int32_t> context(prompt.begin(), prompt.end());
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (std::size_t i = 0; i <= s.tokens.size(); ++i) {
    if (i == s.tokens.size() && !s.stopped) {
      break;
    }
    const auto p = plain.Distribution(Target(context, i));
    const std::int64_t outcome =
        i < s.tokens.size() ? s.tokens[i] : kStopOutcome;
    const auto v = Pit(p, outcome, IsStop, unit(rng));
    e->pit.push_back(v.u);
    e->impossible += v.possible ? 0 : 1;
    e->stops.Add(outcome == kStopOutcome, v.p_stop);
    e->top.Add(v.top, v.p_top);
    if (i < s.tokens.size()) {
      plain.Accept(static_cast<sampling::TokenId>(s.tokens[i]));
      context.push_back(static_cast<std::int32_t>(s.tokens[i]));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t sequences = argc > 1 ? std::stoul(argv[1]) : 6000;
  constexpr std::size_t kLength = 10;
  const std::array<std::int32_t, 12> prompt{17, 4, 99, 23, 4, 150,
                                            61, 8, 4,  77, 5, 42};
  struct Case {
    const char* name;
    sampling::SamplingConfig config;
  };
  std::vector<Case> cases;
  {
    sampling::SamplingConfig c;  // Ling preset
    c.temperature = 0.6F;
    c.top_p = 0.95F;
    c.top_k = 20;
    cases.push_back({"ling_preset", c});
    c.presence_penalty = 1.5F;
    c.frequency_penalty = 0.5F;
    c.repeat_penalty = 1.3F;
    c.repeat_last_n = 8;
    cases.push_back({"ling_preset_penalties", c});
    sampling::SamplingConfig wide;
    wide.temperature = 1.0F;
    wide.top_p = 0.9F;
    wide.min_p = 0.01F;
    wide.presence_penalty = 0.5F;
    cases.push_back({"top_p_min_p", wide});
    sampling::SamplingConfig flat;  // support wider than 64 candidates
    flat.temperature = 1.3F;
    cases.push_back({"unfiltered_wide_support", flat});
  }
  constexpr double kAlpha = 1e-4;
  constexpr double kMaxZ = 4.5;
  for (const auto& c : cases) {
    for (const Mode mode : {Mode::kAutoregressive, Mode::kVaryingDrafts,
                            Mode::kController, Mode::kUnrelatedDrafts}) {
      const char* mode_name =
          mode == Mode::kAutoregressive  ? "ar"
          : mode == Mode::kVaryingDrafts ? "mtp_varying_drafts"
          : mode == Mode::kController    ? "mtp_controller"
                                         : "mtp_unrelated_drafts";
      Evidence e;
      DraftStats stats;
      std::mt19937_64 rng(12345);
      std::size_t stopped = 0;
      std::size_t tokens = 0;
      for (std::size_t n = 0; n < sequences; ++n) {
        auto config = c.config;
        // Small consecutive seeds, as the agent harness sends (seed 1000 +
        // task): the xorshift stream starts from a sparse state.
        config.seed = static_cast<std::int64_t>(
            1000 + 100000 * static_cast<std::size_t>(mode) + n);
        const auto s = Generate(config, prompt, kLength, mode, &stats);
        stopped += s.stopped ? 1 : 0;
        tokens += s.tokens.size();
        Score(c.config, prompt, s, rng, &e);
      }
      const auto ks = KsUniform(e.pit);
      std::printf(
          "       %s/%s: %zu pit values, KS D=%.4f p=%.3g, stops %zu "
          "(z=%.2f), top-1 z=%.2f, impossible %zu, mean length %.2f, "
          "drafted %llu accepted %llu\n",
          c.name, mode_name, ks.n, ks.d, ks.p, stopped, e.stops.z(), e.top.z(),
          e.impossible, static_cast<double>(tokens) / sequences,
          static_cast<unsigned long long>(stats.drafted),
          static_cast<unsigned long long>(stats.accepted));
      const std::string label = std::string(c.name) + "/" + mode_name;
      Expect(e.impossible == 0, label + ": no committed token has p = 0");
      Expect(ks.p >= kAlpha, label + ": PIT values are uniform");
      Expect(std::fabs(e.stops.z()) <= kMaxZ,
             label + ": stop count matches sum p(stop)");
      Expect(std::fabs(e.top.z()) <= kMaxZ,
             label + ": top-1 count matches sum p(top)");
      if (mode == Mode::kUnrelatedDrafts) {
        Expect(stats.drafted > 0 && stats.accepted * 10 < stats.drafted,
               label + ": drafts are proposed and mostly rejected");
      } else if (mode != Mode::kAutoregressive) {
        Expect(stats.accepted > stats.drafted / 10 &&
                   stats.accepted < stats.drafted,
               label + ": drafts are accepted and rejected");
      }
    }
  }
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
