#ifndef GUFO_MODELS_LING_3_FLASH_MTP_HPP_
#define GUFO_MODELS_LING_3_FLASH_MTP_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

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
///   logits = output . h'                 (the trunk's F16 lm_head)
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
/// experts). Snapshots carry the MTP rows and hidden carry so restored sessions
/// keep drafting.
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

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_MTP_HPP_
