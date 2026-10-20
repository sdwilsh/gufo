#ifndef GUFO_MODELS_LING_3_FLASH_MTP_SAMPLING_HPP_
#define GUFO_MODELS_LING_3_FLASH_MTP_SAMPLING_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::models::ling_3_flash {

// Proposal sampling and lossless verification of the MTP draft. The logic is
// that of the Qwen3.8-Flash-Next MTP (the repository keeps model code with its
// model, so it is duplicated rather than shared): the proposal q has bounded
// support (the 64 best draft logits, exact masses summing to 2^24 units), the
// target distribution p stays complete, a draft token x is accepted with
// probability min(1, p(x) / q(x)) and a rejection resamples from the residual
// max(p - q, 0).

enum class DraftDecision { kAccept, kReject, kStop };

inline constexpr std::size_t kMtpCandidates = 64;

struct MtpCandidateLogits {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> logits{};
  std::size_t size{0};
};

/// The kMtpCandidates largest finite logits (descending, lower index first on
/// ties): the support of a sampled proposal.
inline MtpCandidateLogits TopCandidates(std::span<const float> logits) {
  std::vector<std::uint32_t> order;
  order.reserve(logits.size());
  for (std::uint32_t i = 0; i < logits.size(); ++i) {
    if (std::isfinite(logits[i])) {
      order.push_back(i);
    }
  }
  const std::size_t keep = std::min(kMtpCandidates, order.size());
  const auto better = [&](std::uint32_t a, std::uint32_t b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  };
  std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep),
                    order.end(), better);
  MtpCandidateLogits out;
  out.size = keep;
  for (std::size_t i = 0; i < keep; ++i) {
    out.ids[i] = static_cast<sampling::TokenId>(order[i]);
    out.logits[i] = logits[order[i]];
  }
  return out;
}

struct MtpProposal {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> probabilities{};
  std::size_t size{0};
  sampling::TokenId token{0};
  float probability{0.0F};
};

/// The proposal has bounded support; the target distribution stays complete.
/// Integer probability masses sum to 2^24, so the exported F32 q sums to
/// exactly one and is exactly the distribution used to draw the proposal.
inline MtpProposal SampleMtpProposal(const MtpCandidateLogits& candidates,
                                     const sampling::SamplerState& sampler,
                                     std::uint64_t* rng) {
  if (candidates.size == 0 || candidates.size > kMtpCandidates) {
    throw std::invalid_argument("invalid MTP candidate count");
  }
  const auto distribution =
      sampler.Distribution(std::span(candidates.logits).first(candidates.size),
                           std::span(candidates.ids).first(candidates.size));
  constexpr std::uint32_t units = 1U << 24;
  std::array<std::uint32_t, kMtpCandidates> mass{};
  MtpProposal proposal;
  proposal.size = distribution.entries().size();
  std::uint32_t total = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    const auto& entry = distribution.entries()[i];
    proposal.ids[i] = candidates.ids[entry.token];
    mass[i] = static_cast<std::uint32_t>(std::floor(entry.value * units));
    total += mass[i];
  }
  mass[0] += units - total;
  const auto draw = static_cast<std::uint32_t>(sampling::Uniform(rng) * units);
  std::uint32_t cumulative = 0;
  bool selected = false;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    proposal.probabilities[i] = static_cast<float>(mass[i]) / units;
    cumulative += mass[i];
    if (!selected && draw < cumulative) {
      proposal.token = proposal.ids[i];
      proposal.probability = proposal.probabilities[i];
      selected = true;
    }
  }
  return proposal;
}

/// A prompt-lookup copy as a proposal: the point mass on the copied token
/// (q(x) = 1). The copy depends on the committed tokens alone, so the generic
/// rule stays exact: x is accepted with probability min(1, p(x) / 1) = p(x),
/// and a rejection draws from norm(max(p - q, 0)), which is p without x. It
/// has no draft logits and uses no draw of the request's proposal stream.
inline MtpProposal CopyProposal(sampling::TokenId token) {
  MtpProposal proposal;
  proposal.ids[0] = token;
  proposal.probabilities[0] = 1.0F;
  proposal.size = 1;
  proposal.token = token;
  proposal.probability = 1.0F;
  return proposal;
}

struct MtpVerification {
  sampling::TokenId token;
  bool accepted;
};

/// The same FP64 target distribution drives AR draws, support, acceptance
/// and residual correction. Only proposal q uses the compact F32 masses.
inline MtpVerification VerifyMtpProposal(std::span<const float> logits,
                                         const MtpProposal& proposal,
                                         sampling::SamplerState& sampler) {
  if (proposal.size == 0 || proposal.size > kMtpCandidates ||
      proposal.token >= logits.size() || !std::isfinite(proposal.probability) ||
      proposal.probability <= 0 || proposal.probability > 1)
    throw std::invalid_argument("invalid MTP proposal");
  double total = 0;
  double selected = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    if (proposal.ids[i] >= logits.size() ||
        !std::isfinite(proposal.probabilities[i]) ||
        proposal.probabilities[i] < 0)
      throw std::invalid_argument("invalid MTP proposal mass");
    total += proposal.probabilities[i];
    if (proposal.ids[i] == proposal.token)
      selected += proposal.probabilities[i];
  }
  if (total != 1 || selected != proposal.probability)
    throw std::invalid_argument("MTP proposal masses do not match the draw");
  const auto target = sampler.Distribution(logits);
  if (sampler.Uniform() * proposal.probability <
      target.probability(proposal.token))
    return {proposal.token, true};
  return {target.SampleResidual(
              std::span(proposal.ids).first(proposal.size),
              std::span(proposal.probabilities).first(proposal.size),
              sampler.mutable_rng_state()),
          false};
}

/// Draft side of a sampled cycle. The proposals use their own stream, seeded
/// with one draw of the request's RNG so target verification keeps
/// independent draws, and an unconstrained copy of the request's sampler that
/// follows the chain (anchor, then each proposal) for the penalty history.
class MtpProposer {
public:
  MtpProposer(sampling::SamplerState& sampler, sampling::TokenId anchor)
      : rng_(sampling::NextRandom(sampler.mutable_rng_state())),
        draft_(sampler.WithoutConstraint()) {
    draft_.Accept(anchor);
  }

  /// Draws the next draft token from the draft block's logits row.
  sampling::TokenId Propose(std::span<const float> draft_logits) {
    proposals_.push_back(
        SampleMtpProposal(TopCandidates(draft_logits), draft_, &rng_));
    draft_.Accept(proposals_.back().token);
    return proposals_.back().token;
  }

  [[nodiscard]] std::span<const MtpProposal> proposals() const noexcept {
    return proposals_;
  }

private:
  std::uint64_t rng_;
  sampling::SamplerState draft_;
  std::vector<MtpProposal> proposals_;
};

struct SampledVerification {
  /// Committed chain tokens (anchor included).
  std::uint32_t keep{1};
  /// The target sample after the kept tokens is a stop token.
  bool stop{false};
  /// Residual draw after the first rejection: the next cycle's anchor.
  std::optional<sampling::TokenId> correction;
};

/// Target side of a sampled cycle over [anchor, proposals]: row_logits(j) are
/// the target logits after chain token j, and `sampler` has accepted the
/// anchor. Accepted proposals enter the sampler's history in order before the
/// next row's p is computed; a rejected proposal never does.
template<typename RowLogits, typename IsStop>
SampledVerification VerifySampledChain(RowLogits&& row_logits,
                                       std::span<const MtpProposal> proposals,
                                       sampling::SamplerState& sampler,
                                       IsStop&& is_stop) {
  SampledVerification out;
  while (out.keep <= proposals.size()) {
    const auto verified = VerifyMtpProposal(row_logits(out.keep - 1),
                                            proposals[out.keep - 1], sampler);
    if (is_stop(verified.token)) {
      out.stop = true;
      break;
    }
    if (!verified.accepted) {
      out.correction = verified.token;
      break;
    }
    sampler.Accept(verified.token);
    ++out.keep;
  }
  return out;
}

/// The draft is deterministic, so an exact target sample decides acceptance.
/// A rejected draw is deferred to the next cycle: restore its RNG so that
/// stopping at this prefix consumes exactly the draws made by ordinary decode.
template<typename IsStop>
DraftDecision VerifyDraft(std::span<const float> logits, std::int32_t draft,
                          sampling::SamplerState& sampler, IsStop is_stop) {
  const auto rng = sampler.rng_state();
  const auto token = sampler.Sample(logits);
  if (is_stop(static_cast<std::int32_t>(token))) {
    return DraftDecision::kStop;
  }
  if (token != static_cast<sampling::TokenId>(draft)) {
    sampler.SetRngState(rng);
    return DraftDecision::kReject;
  }
  sampler.Accept(token);
  return DraftDecision::kAccept;
}

}  // namespace gufo::models::ling_3_flash

#endif
