// Host-only checks of the MTP draft-length controller (mtp.hpp) and of the
// lossless p/q sampling rule (mtp_sampling.hpp): fallback to plain decoding
// with exponential backoff when drafts do not pay, growth with perfect
// acceptance, deterministic choices from the committed history, and an
// empirical test that proposal + acceptance + residual reproduces the target
// distribution.
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/ling_3_flash/mtp.hpp"
#include "src/models/ling_3_flash/mtp_sampling.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace sampling = gufo::sampling;

namespace {
int g_failures = 0;
void Expect(bool ok, const char* what) {
  std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) {
    ++g_failures;
  }
}
}  // namespace

int main() {
  // Batched cycles (MtpCohortController): the calibrated prior and measured
  // costs decide between drafting and the batched AR step (width 0).
  {
    l3::MtpLengthController perfect(7);
    for (int i = 0; i < 40; ++i) {
      perfect.Observe(3, 3, true, 0);
    }
    l3::MtpLengthController poor(7);
    for (int i = 0; i < 40; ++i) {
      poor.Observe(0, 1, true, 0);
    }
    const l3::MtpCohortController fresh;
    const std::array<l3::MtpCohortController::Row, 2> good = {
        {{&perfect, 3}, {&perfect, 3}}};
    const std::array<l3::MtpCohortController::Row, 2> bad = {
        {{&poor, 3}, {&poor, 3}}};
    Expect(fresh.Choose(good, 2, 0, 8, 0) >= 1,
           "cohort: accepted drafts beat batched AR on the prior");
    Expect(fresh.Choose(bad, 2, 0, 8, 0) == 0,
           "cohort: rejected drafts keep the batched AR step");
    Expect(fresh.Choose(good, 2, 0, 8, 0) <= 3, "cohort: width within budget");
    // Measured: width 1 costs three AR steps; drafting must stop paying.
    l3::MtpCohortController measured;
    for (int i = 0; i < 4; ++i) {
      measured.Observe(2, 0, 0, 35.0F);
      measured.Observe(2, 0, 1, 105.0F);
      measured.Observe(2, 0, 2, 140.0F);
      measured.Observe(2, 0, 3, 175.0F);
    }
    // 16 cycles observed: no exploration step; width 0 scores best
    // (2 / 35 against 4 / 105, 6 / 140 and 8 / 175 tokens per ms).
    Expect(measured.Choose(good, 2, 0, 8, 0) == 0,
           "cohort: measured costs override the prior");
    // Two greedy lanes beside 4 rows of other lanes fit width 1 in one
    // 8-row forward; width 3 needs a second forward, and when that is what
    // the measurements favour it is chosen (the cycle then splits).
    l3::MtpCohortController split;
    for (int i = 0; i < 4; ++i) {
      split.Observe(4, 0, 0, 50.0F);
      split.Observe(4, 0, 1, 200.0F);
      split.Observe(4, 0, 2, 200.0F);
      split.Observe(4, 0, 3, 60.0F);
    }
    Expect(split.Choose(good, 4, 4, 8, 0) == 3,
           "cohort: a wider width across two verify forwards can win");
  }
  // Perfect acceptance: drafts are used and the depth grows toward the limit.
  {
    l3::MtpLengthController c(7);
    std::uint32_t last = 0;
    for (int i = 0; i < 60; ++i) {
      last = c.Choose(7, false, 4096);
      c.Observe(last, last, false, 4096);
    }
    Expect(last >= 3, "perfect acceptance reaches a depth of at least 3");
  }
  // Zero acceptance: after the first rejections plain decoding with backoff
  // (16, 32, 64 ... tokens between single probes).
  {
    l3::MtpLengthController c(7);
    std::vector<int> probe_gaps;
    int plain = 0;
    for (int step = 0; step < 700; ++step) {
      const std::uint32_t k = c.Choose(7, false, 4096);
      if (k == 0) {
        c.ObservePlainToken();
        ++plain;
      } else {
        if (!probe_gaps.empty() || plain > 0) {
          probe_gaps.push_back(plain);
        } else {
          probe_gaps.push_back(0);
        }
        plain = 0;
        c.Observe(0, k, false, 4096);
      }
    }
    // Skip the warm-up cycles before the first fallback.
    std::vector<int> gaps;
    for (int g : probe_gaps) {
      if (g > 0)
        gaps.push_back(g);
    }
    bool backoff = gaps.size() >= 4 && gaps[0] == 16 && gaps[1] == 32 &&
                   gaps[2] == 64 && gaps[3] == 128;
    Expect(backoff, "no acceptance: plain decoding with doubling probe gaps");
    int drafted_cycles = static_cast<int>(probe_gaps.size());
    Expect(drafted_cycles < 20, "no acceptance: few speculative cycles in 700");
  }
  // Moderate acceptance (0.3) does not beat plain decoding.
  {
    l3::MtpLengthController c(7);
    int speculative = 0;
    int steps = 0;
    for (int i = 0; i < 400; ++i) {
      const std::uint32_t k = c.Choose(7, false, 0);
      ++steps;
      if (k == 0) {
        c.ObservePlainToken();
        continue;
      }
      ++speculative;
      // Deterministic pattern with a first-depth acceptance of 30%.
      const std::size_t accepted = (i % 10) < 3 ? 1 : 0;
      c.Observe(accepted, k, false, 0);
    }
    Expect(speculative * 10 < steps * 3,
           "30% acceptance mostly falls back to plain decoding");
  }
  // Same history, same choices (seeded replay of sampled requests).
  {
    l3::MtpLengthController a(7), b(7);
    bool same = true;
    for (int i = 0; i < 200 && same; ++i) {
      const auto ka = a.Choose(7, false, 1000 + i);
      const auto kb = b.Choose(7, false, 1000 + i);
      same = ka == kb;
      if (ka == 0) {
        a.ObservePlainToken();
        b.ObservePlainToken();
      } else {
        const std::size_t acc = static_cast<std::size_t>((i * 7) % (ka + 1));
        a.Observe(acc, ka, false, 1000 + i);
        b.Observe(acc, ka, false, 1000 + i);
      }
    }
    Expect(same, "choices depend only on the committed history");
  }
  Expect(l3::MtpDraftCostPrior(65536) > l3::MtpDraftCostPrior(0) &&
             l3::MtpDraftCostPrior(1u << 30) == l3::MtpDraftCostPrior(65536),
         "the cost prior grows with the context and saturates at 64k");

  // Measured costs override the prior (greedy): an expensive width is avoided.
  {
    l3::MtpLengthController c(7);
    for (int i = 0; i < 4; ++i) {
      c.ObserveCost(0, 28.0F);
      c.ObserveCost(1, 90.0F);  // slow cycles with one draft
    }
    Expect(c.RelativeCost(1, true, 0) > 3.0F, "measured cycle cost is used");
    Expect(c.RelativeCost(1, false, 0) < 1.5F,
           "sampled requests keep the prior");
  }

  // Lossless sampling: draft x ~ q (64-candidate support), accepted with
  // min(1, p/q) or replaced by a residual sample, yields tokens ~ p.
  {
    constexpr std::size_t kVocab = 100;
    std::vector<float> target(kVocab), draft(kVocab);
    for (std::size_t i = 0; i < kVocab; ++i) {
      target[i] = 3.0F * std::sin(0.7F * static_cast<float>(i)) +
                  0.02F * static_cast<float>(i);
      draft[i] = 2.5F * std::sin(0.7F * static_cast<float>(i) + 0.3F) -
                 0.01F * static_cast<float>(i);
    }
    sampling::SamplingConfig config;
    config.temperature = 1.0F;
    config.seed = 1;
    sampling::SamplerState sampler(config);
    const auto candidates = l3::TopCandidates(draft);
    std::uint64_t draft_rng = 12345;
    constexpr int kTrials = 300000;
    std::vector<double> histogram(kVocab, 0.0);
    int accepted = 0;
    for (int t = 0; t < kTrials; ++t) {
      const auto proposal =
          l3::SampleMtpProposal(candidates, sampler, &draft_rng);
      const auto verified = l3::VerifyMtpProposal(target, proposal, sampler);
      histogram[verified.token] += 1.0;
      accepted += verified.accepted ? 1 : 0;
    }
    const auto p = sampler.Distribution(target);
    double worst = 0.0;
    for (const auto& entry : p.entries()) {
      worst = std::max(
          worst, std::fabs(histogram[entry.token] / kTrials - entry.value));
    }
    std::printf("       worst |empirical - p| = %.4f, accepted %.3f\n", worst,
                static_cast<double>(accepted) / kTrials);
    Expect(worst < 0.004,
           "proposal + acceptance + residual reproduces the target");
    Expect(accepted > kTrials / 10 && accepted < kTrials,
           "the draft is accepted sometimes and rejected sometimes");
  }
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
