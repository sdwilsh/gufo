#ifndef GUFO_MODELS_LING_3_FLASH_MTP_HPP_
#define GUFO_MODELS_LING_3_FLASH_MTP_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/ling_3_flash/layout.hpp"

/// Ling-3.0-flash MTP (multi-token prediction) speculative decoding: design
/// note and the draft-length controller.
///
/// Math of the draft step (llama.cpp `bailingmoe3` graph_mtp, GGUF block 42).
/// Positions index the trunk token stream t_0, t_1, ...; the trunk at position
/// i consumes t_i and its output-normed hidden state h_i = rmsnorm(h;
/// output_norm) predicts t_{i+1}. The MTP block at position i takes (h_i,
/// t_{i+1}) and predicts t_{i+2}:
///
///   x   = eh_proj([ rmsnorm(embed(t_{i+1}); enorm) | rmsnorm(h_i; hnorm) ])
///                                    (Q8_0 [5120 -> 2560]; the token half
///                                    first)
///   x  += mla_i(rmsnorm(x; attn_norm))   block 42's own MLA with its own
///   latent
///                                    cache (row i written, rows 0..i attended)
///   x  += moe(rmsnorm(x; ffn_norm))      512 experts, shared expert, no clamps
///   h'  = rmsnorm(x; layer_output_norm)  (the shared-head norm)
///   logits = output . h'                 (draft: a Q8_0 copy of the F16
///                                         lm_head; verify: the F16 head)
///
/// The block is exactly a trunk MLA + MoE block (same kernels), preceded by the
/// eh_proj input stage and followed by its own norm; the residual stream starts
/// from the eh_proj output. h' also feeds the next draft of a chain as its `h`
/// input (position i + 1, token = the previous draft).
///
/// Cache and state. The MTP latent cache row i is a function of (h_i, t_{i+1})
/// only (attn_norm -> kv_a -> norm -> rope), so it is filled for every
/// committed position without running attention or MoE: every forward pass
/// (prefill chunk, decode step, verify step) appends rows first_pos - 1 ..
/// first_pos + n - 2 (cheap: eh_proj and kv_a GEMMs) and carries the last
/// trunk hidden state h_{first_pos + n - 1} for the next call. Rows written by
/// a draft chain beyond the committed frontier are speculative; the next
/// forward rewrites them from the trunk hidden states, and a rollback just
/// lowers the position. Draft cost per step: eh_proj 13 MB + block 42 weights
/// (MLA 0.24 GB, 8 routed experts + shared expert and router per token) + the
/// 0.8 GB lm_head, about a quarter of a trunk step.
///
/// Draft head. The draft's numerics only decide ACCEPTANCE: every emitted token
/// is scored by the verify step with the trunk's F16 head, and a sampled
/// request's proposal q is computed from the very logits row the draft token
/// was drawn from (MtpProposer), so any head keeps both greedy output and the
/// p/q rejection test exact. The draft step is bandwidth bound on that head
/// (157,184 x 2,560 F16 = 805 MB, ~3.4 ms of a ~4.2 ms step), so the draft owns
/// a Q8_0 copy (DeviceMtp::draft_head, 427 MB, re-encoded from the resident F16
/// weights at load when MTP is enabled; ggml Q8_0 rounding) read by the
/// existing two-plane int8 Q8_0 GEMV that already serves the other draft
/// projections. Weights +0.40 GiB (see Memory). Not stored in snapshots or in
/// any numerics identity (drafts are recomputed). Model::SetMtpDraftHeadQ8
/// switches back to the F16 head for A/B measurements.
///
/// Cycle (Session::DecodeStep). The session holds the trunk logits of the last
/// committed position. anchor = sample(logits) (always committed). k drafts
/// d_1..d_k come from the MTP chain (token anchor at position P - 1, then each
/// draft at the next position). One (k+1)-token forward verifies
/// [anchor, d_1..d_k] (positions P..P+k): it is in the small-batch numeric
/// family (k + 1 <= 8, at most kMaxDrafts = 7 drafts), so every row is
/// bit-identical to the same token decoded alone and greedy output is
/// token-identical to plain greedy decoding. The loop accepts d_j while the
/// target sample of row j - 1 equals it (greedy: the argmax; sampled: the
/// lossless p/q test of mtp_sampling.hpp with the 64-candidate proposal q; a
/// deferred residual draw becomes the next anchor). A (k+1)-token verify loads
/// the union of the experts its rows route to, so on this MoE it costs more
/// than a dense target would: the controller below decides k from measured
/// costs.
///
/// Rollback. Rejected rows must not change the recurrent state. While a verify
/// step runs, the KDA layers store the state (2 MiB per layer) and the conv
/// window after every row but the last (kernels kda::RecurrentSequence /
/// Conv3Checkpoint, about 0.5 GiB of executor scratch for 7 rows) and the step
/// keeps the output-normed hidden rows; MtpRollback(keep) restores the state
/// after row keep - 1 (one 75 MiB copy; nothing when all rows are kept) and the
/// hidden carry. MLA caches (7 trunk layers and block 42) are truncated by the
/// position alone. The restored state is bit-identical to feeding the kept rows
/// alone (they use the same arithmetic), which the rollback tests check.
///
/// Memory: 0.5 GiB shared checkpoint scratch, per session 1152 bytes per
/// context token (the MTP latent cache, 75 MiB at 64k) plus 10 KiB, and the
/// block-42 weights (about 1.2 GiB: Q8_0 MLA/shared/eh_proj, IQ4_XS/Q4_K
/// experts) plus the 0.40 GiB Q8_0 draft head. Snapshots carry the MTP rows and
/// hidden carry so restored sessions keep drafting.
///
/// Wiring. The block lives in the main GGUF, so `--speculative mtp` needs no
/// `--mtp-model` for Ling (the option is refused for this architecture): the
/// server and the bench enable the block when the artifact has it.
namespace gufo::models::ling_3_flash {

/// Cost of one more draft (draft step plus the extra verified row) relative to
/// a plain decode step, used by sampled requests and until measurements replace
/// it (greedy). From the measured per-op costs of the gfx1151 decode step
/// (27.9 ms at d0, 31.4 ms at 64k; lm_head 3.42 ms; 0.82 ms per additional
/// distinct expert per layer, with the 13.0 / 17.3 / 21.1 distinct experts per
/// layer measured for 2 / 3 / 4 consecutive tokens on real text; 3.46 ms of MLA
/// attention per extra row at 64k) and the estimated block-42, rollback and
/// launch costs (tools/models/ling_3_flash/mtp/breakeven.py): about 0.33 of a
/// step at depth 0 growing to 0.44 at 64k. Estimates, not measurements of this
/// code: the greedy controller replaces them with measured cycle times.
[[nodiscard]] constexpr float MtpDraftCostPrior(
    std::uint32_t context) noexcept {
  const float depth = context < 65536 ? static_cast<float>(context) : 65536.0F;
  return 0.33F + 0.11F * depth / 65536.0F;
}

/// Draft-length controller. Acceptance of the depth-d draft is estimated from
/// the committed history only (decayed counts; the first rejected draft is a
/// failure, its unverified tail is not observed); the choice maximises
/// expected tokens per cost: sum_{j<=k} prod_{i<=j} a_i + 1 over cost(k).
/// Sampled requests keep the fixed prior costs so the length depends on the
/// request's own history alone (seeded replay); greedy requests use the
/// measured cycle time per k (decayed means, prior scaled by the measured
/// plain-step time until two samples exist).
class MtpLengthController {
public:
  explicit MtpLengthController(std::uint32_t limit = kMaxDrafts)
      : limit_(std::clamp(limit, 1U, kMaxDrafts)) {
    Reset();
  }

  void Reset() noexcept {
    successes_.fill(1.5F);
    failures_.fill(0.5F);
    retry_tokens_ = 0;
    retry_interval_ = 16;
    probe_depth_ = 0;
    explored_depth_ = 0;
    probe_delay_ = 0;
    failed_depths_ = 0;
    cycles_ = 0;
    ms_.fill(0.0F);
    samples_.fill(0);
  }

  /// Drafts for the next cycle (0 = plain decoding), at most `cap` and the
  /// limit. `measured` lets the measured cycle costs steer the choice.
  [[nodiscard]] std::uint32_t Choose(std::uint32_t cap, bool measured,
                                     std::uint32_t context = 0) const noexcept {
    cap = std::min(cap, limit_);
    if (cap == 0 || retry_tokens_ != 0) {
      return 0;
    }
    // After a bounded plain-decoding interval one proposal tests recovery.
    std::uint32_t best =
        std::max({1U, Best(cap, measured, context), probe_depth_});
    if (measured && cycles_ % 16 == 8) {
      // Refresh the costs one width above the current choice.
      best = std::max(1U, best + 1);
    }
    return std::min(cap, best);
  }

  /// Committed outcome of a cycle that proposed `drafted` and accepted
  /// `accepted` (a stop ends a cycle: pass drafted = accepted). `measured` as
  /// for Choose.
  void Observe(std::size_t accepted, std::size_t drafted, bool measured,
               std::uint32_t context = 0) noexcept {
    ++cycles_;
    if (drafted == 0) {
      return;
    }
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    const std::size_t observed = accepted + (accepted < drafted ? 1 : 0);
    for (std::size_t d = 0; d < observed; ++d) {
      successes_[d] = 0.75F * successes_[d] + (d < accepted ? 1.0F : 0.0F);
      failures_[d] = 0.75F * failures_[d] + (d == accepted ? 1.0F : 0.0F);
    }
    // A fully accepted chain says nothing about its tail: probe deeper.
    probe_depth_ = 0;
    if (accepted == drafted && drafted < limit_) {
      if (drafted >= explored_depth_ || probe_delay_ == 0) {
        probe_depth_ = std::min<std::uint32_t>(2 * drafted, limit_);
        probe_delay_ = 16;
      } else {
        --probe_delay_;
      }
    }
    explored_depth_ = std::max<std::uint32_t>(
        explored_depth_, static_cast<std::uint32_t>(observed));
    if (accepted < drafted) {
      failed_depths_ |= 1U << accepted;
    }
    // Plain decoding wins: back off exponentially (16, 32 .. 256 tokens) so
    // text on which MTP never helps pays only a fraction of a percent for the
    // recovery probes; any profitable estimate resets the interval.
    if (Best(limit_, measured, context) == 0) {
      retry_tokens_ = retry_interval_;
      retry_interval_ = std::min(retry_interval_ * 2, 256U);
    } else {
      retry_interval_ = 16;
    }
  }

  /// A plain decoding step happened (bounded interval before the next probe).
  void ObservePlainToken() noexcept {
    ++cycles_;
    if (retry_tokens_ != 0) {
      --retry_tokens_;
    }
  }

  /// Wall time of a cycle with `drafts` drafts (0 = plain decoding step).
  void ObserveCost(std::uint32_t drafts, float ms) noexcept {
    if (drafts > kMaxDrafts || !std::isfinite(ms) || ms <= 0.0F) {
      return;
    }
    auto& n = samples_[drafts];
    auto& e = ms_[drafts];
    // The first sample of a width carries lazy initialisation: replace it.
    e = n < 2 ? ms : e + 0.25F * (ms - e);
    n = std::min<std::uint32_t>(n + 1, 1000000);
  }

  /// Expected tokens of a cycle with `drafts` drafts under the current
  /// acceptance estimates (anchor included).
  [[nodiscard]] float ExpectedTokens(std::uint32_t drafts) const noexcept {
    float expected = 1.0F;
    float survival = 1.0F;
    for (std::uint32_t d = 0; d < std::min(drafts, limit_); ++d) {
      survival *= Probability(d);
      expected += survival;
    }
    return expected;
  }

  /// Cost of a cycle with k drafts relative to a plain step.
  [[nodiscard]] float RelativeCost(std::uint32_t drafts, bool measured,
                                   std::uint32_t context = 0) const noexcept {
    const float per_draft = MtpDraftCostPrior(context);
    const float prior = 1.0F + per_draft * static_cast<float>(drafts);
    if (!measured || samples_[0] == 0) {
      return prior;
    }
    if (samples_[drafts] >= 2) {
      return ms_[drafts] / ms_[0];
    }
    // Unmeasured width: the prior, shifted by what the measured widths show.
    float scale = 1.0F;
    for (std::uint32_t d = 1; d <= kMaxDrafts; ++d) {
      if (samples_[d] >= 2) {
        scale = (ms_[d] / ms_[0]) / (1.0F + per_draft * static_cast<float>(d));
        break;
      }
    }
    return prior * scale;
  }

  [[nodiscard]] std::uint32_t limit() const noexcept { return limit_; }

private:
  [[nodiscard]] float Probability(std::uint32_t d) const noexcept {
    return d < explored_depth_ && (failed_depths_ & (1U << d)) == 0
               ? 1.0F
               : successes_[d] / (successes_[d] + failures_[d]);
  }
  [[nodiscard]] std::uint32_t Best(std::uint32_t cap, bool measured,
                                   std::uint32_t context) const noexcept {
    float best_score = 1.0F;
    std::uint32_t best = 0;
    float survival = 1.0F;
    float expected = 1.0F;
    for (std::uint32_t k = 1; k <= cap; ++k) {
      survival *= Probability(k - 1);
      expected += survival;
      const float score = expected / RelativeCost(k, measured, context);
      if (score > best_score * 1.01F) {
        best_score = score;
        best = k;
      }
    }
    return best;
  }

  std::uint32_t limit_;
  std::array<float, kMaxDrafts> successes_{};
  std::array<float, kMaxDrafts> failures_{};
  std::uint32_t retry_tokens_{0};
  std::uint32_t retry_interval_{16};
  std::uint32_t probe_depth_{0};
  std::uint32_t explored_depth_{0};
  std::uint32_t probe_delay_{0};
  std::uint32_t failed_depths_{0};
  std::uint64_t cycles_{0};
  std::array<float, kMaxDrafts + 1> ms_{};
  std::array<std::uint32_t, kMaxDrafts + 1> samples_{};
};

/// Prior wall time (ms) of a batched decode forward over `rows` rows of
/// distinct sessions at `context`: gfx1151 batched AR steps measured on the
/// full model (m1-bench, rc2 base; aggregate 35.9 / 56.7 / 81.7 / 105.6 t/s
/// at 1 / 2 / 4 / 8 sessions at depth 0, 32.0 / 48.3 / 66.7 / 83.0 at
/// 65,536), linear between the measured widths and in depth. Rows of one
/// session's verify share more experts than rows of distinct sessions, so
/// this is the high side for verify rows.
[[nodiscard]] inline float MtpBatchStepMsPrior(float rows,
                                               std::uint32_t context) noexcept {
  constexpr std::array<float, 4> kRows = {1.0F, 2.0F, 4.0F, 8.0F};
  constexpr std::array<float, 4> kDepth0 = {27.86F, 35.27F, 48.96F, 75.76F};
  constexpr std::array<float, 4> kDepth64k = {31.25F, 41.41F, 59.97F, 96.39F};
  const float depth =
      static_cast<float>(context < 65536 ? context : 65536) / 65536.0F;
  const auto at = [&](std::size_t i) {
    return kDepth0[i] + depth * (kDepth64k[i] - kDepth0[i]);
  };
  rows = std::max(rows, 1.0F);
  std::size_t i = 0;
  while (i + 2 < kRows.size() && rows > kRows[i + 1]) {
    ++i;
  }
  const float t = (rows - kRows[i]) / (kRows[i + 1] - kRows[i]);
  return at(i) + t * (at(i + 1) - at(i));
}

/// Prior wall time (ms) of one batched draft step over `lanes` chains (block
/// 42 and the Q8_0 draft head once; one expert union per lane): about 4 ms
/// for one lane on gfx1151.
[[nodiscard]] constexpr float MtpDraftStepMsPrior(std::size_t lanes) noexcept {
  return 3.4F + 0.8F * static_cast<float>(lanes);
}

/// Draft length of the greedy lanes of a batched decode cycle
/// (Session::DecodeBatch): one width for all of them, chosen from their own
/// acceptance estimates and the cost of whole cycles per (occupancy, context
/// bin), the way Flash-Next's MtpBatchController does. Width 0 is the batched
/// AR step, so a cohort drafts only when that beats batched AR on measured (or
/// prior) costs; it never falls back to serial decoding. Greedy tokens do not
/// depend on the width, so cohort-measured costs are allowed here; sampled
/// lanes keep their own-history MtpLengthController choice so seeded replay
/// never depends on the peers.
class MtpCohortController {
public:
  struct Row {
    const MtpLengthController* acceptance;
    std::uint32_t budget;  ///< Most drafts this lane may take.
  };

  /// `occupancy` lanes in the cycle, `rows` of them greedy; the other lanes
  /// take `other_rows` verify rows in all. A cycle's rows go to verify
  /// forwards of at most `row_limit` rows: a width whose rows do not fit
  /// beside the others costs a further forward, which a wider greedy width
  /// may still repay.
  [[nodiscard]] std::uint32_t Choose(std::span<const Row> rows,
                                     std::size_t occupancy,
                                     std::uint32_t other_rows,
                                     std::uint32_t row_limit,
                                     std::uint32_t context) const noexcept {
    if (rows.empty() || occupancy == 0 || occupancy > 8) {
      return 0;
    }
    std::uint32_t cap = kMaxDrafts;
    for (const Row& row : rows) {
      cap = std::min(cap, row.budget);
    }
    if (cap == 0) {
      return 0;
    }
    const Cell& cell = CellFor(occupancy, context);
    // Periodic plain cycles measure the alternative to drafting.
    const bool stable = last_occupancy_ == occupancy &&
                        last_bin_ == ContextBin(context) && stable_cycles_ >= 2;
    if (stable && (cell.samples[0] == 0 || cell.cycles % 32 < 2)) {
      return 0;
    }
    const auto greedy = static_cast<float>(rows.size());
    const auto verify_ms = [&](std::uint32_t total) {
      // Full forwards of row_limit rows and the remainder (the packing is
      // first fit; this is its cost when no row is wasted).
      const std::uint32_t limit = std::max(row_limit, 1U);
      float ms = static_cast<float>(total / limit) *
                 MtpBatchStepMsPrior(static_cast<float>(limit), context);
      if (total % limit != 0) {
        ms += MtpBatchStepMsPrior(static_cast<float>(total % limit), context);
      }
      return ms;
    };
    const auto prior = [&](std::uint32_t width) {
      return verify_ms(other_rows +
                       static_cast<std::uint32_t>(greedy) * (width + 1)) +
             static_cast<float>(width) * MtpDraftStepMsPrior(rows.size());
    };
    const float scale = cell.samples[0] >= 2 ? cell.ms[0] / prior(0) : 1.0F;
    const auto cost = [&](std::uint32_t width) {
      return cell.samples[width] >= 2 ? cell.ms[width] : scale * prior(width);
    };
    const auto others = static_cast<float>(occupancy - rows.size());
    float best_score = (others + greedy) / cost(0);
    std::uint32_t best = 0;
    for (std::uint32_t width = 1; width <= cap; ++width) {
      float expected = others;
      for (const Row& row : rows) {
        expected += row.acceptance->ExpectedTokens(width);
      }
      const float score = expected / cost(width);
      if (score > best_score * 1.01F) {
        best_score = score;
        best = width;
      }
    }
    // Bounded exploration learns the cost one width wider.
    if (cell.cycles % 16 == 2) {
      return std::min(cap, std::max(1U, best + 1));
    }
    return best;
  }

  /// Wall time of a cycle whose greedy lanes drafted `drafts` each.
  void Observe(std::size_t occupancy, std::uint32_t context,
               std::uint32_t drafts, float ms) noexcept {
    if (occupancy == 0 || occupancy > 8 || drafts > kMaxDrafts ||
        !std::isfinite(ms) || ms <= 0.0F) {
      return;
    }
    const unsigned bin = ContextBin(context);
    const bool same = last_occupancy_ == occupancy && last_bin_ == bin;
    stable_cycles_ = same ? std::min(stable_cycles_ + 1, 1000000U) : 1;
    last_occupancy_ = occupancy;
    last_bin_ = bin;
    Cell& cell = CellFor(occupancy, context);
    ++cell.cycles;
    auto& n = cell.samples[drafts];
    auto& e = cell.ms[drafts];
    e = n < 2 ? ms : e + 0.25F * (ms - e);
    n = std::min<std::uint32_t>(n + 1, 1000000);
  }

private:
  struct Cell {
    std::array<float, kMaxDrafts + 1> ms{};
    std::array<std::uint32_t, kMaxDrafts + 1> samples{};
    std::uint64_t cycles{0};
  };
  static unsigned ContextBin(std::uint32_t context) noexcept {
    unsigned bin = 0;
    for (std::uint32_t bound = 4096; context >= bound && bin < 7; bound *= 2) {
      ++bin;
    }
    return bin;
  }
  Cell& CellFor(std::size_t occupancy, std::uint32_t context) noexcept {
    return cells_[occupancy - 1][ContextBin(context)];
  }
  const Cell& CellFor(std::size_t occupancy,
                      std::uint32_t context) const noexcept {
    return cells_[occupancy - 1][ContextBin(context)];
  }
  std::array<std::array<Cell, 8>, 8> cells_{};
  std::size_t last_occupancy_{0};
  unsigned last_bin_{0};
  unsigned stable_cycles_{0};
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_MTP_HPP_
