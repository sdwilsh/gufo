#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_SAMPLED_DISTRIBUTION_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_SAMPLED_DISTRIBUTION_HPP_

// Statistics of the sampled-MTP distribution checks (mtp_sampled_chain_test on
// a synthetic model, mtp_distribution on the real one). Sampled speculative
// decoding is correct when every committed token is distributed as the plain
// autoregressive sampler would draw it given the committed prefix (penalty
// history included). Two kinds of evidence:
//   * exact reference: for each generated sequence, the autoregressive target
//     distribution p(. | prefix) of every position is recomputed (teacher
//     forcing). The randomized probability integral transform of each
//     outcome is then i.i.d. U(0, 1) along every trajectory (Rosenblatt), so
//     a Kolmogorov-Smirnov test of the pooled values against U(0, 1) checks
//     every conditional at once. Categories are ordered by descending p, so a
//     sampler that is too greedy (or too flat) moves mass to small (large) u.
//     The counts of stops and of top-1 outcomes are compared with their exact
//     expectations sum p (z-scores), and an outcome with p = 0 is impossible.
//   * two-sample: MTP sequences against independent autoregressive ones,
//     per-position token distributions (total variation summed over
//     positions), stop rate and length distribution, with permutation p-values
//     (whole sequences are permuted, so within-sequence dependence is kept).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::tests::ling {

/// Outcome of a position: a token ID, a stop (any stop token, merged) or the
/// end of a sequence that stopped earlier.
inline constexpr std::int64_t kStopOutcome = -1;
inline constexpr std::int64_t kEndedOutcome = -2;

struct PitValue {
  bool possible{false};  ///< p(outcome) > 0
  double u{0.0};         ///< randomized PIT, U(0, 1) under the null
  double p_outcome{0.0};
  double p_stop{0.0};  ///< merged probability of the stop tokens
  double p_top{0.0};   ///< probability of the most likely category
  bool top{false};     ///< the outcome is the most likely category
};

/// Randomized PIT of `outcome` (token ID or kStopOutcome) under `p`;
/// `stop_aware` merges the stop tokens into one category (generation ends
/// there), otherwise every token is its own category. `v` ~ U(0, 1).
template<typename IsStop>
[[nodiscard]] PitValue Pit(const sampling::SamplingDistribution& p,
                           std::int64_t outcome, IsStop is_stop, double v) {
  std::vector<std::pair<std::int64_t, double>> categories;
  categories.reserve(p.entries().size() + 1);
  double stop = 0.0;
  for (const auto& entry : p.entries()) {
    if (is_stop(entry.token)) {
      stop += entry.value;
    } else {
      categories.emplace_back(static_cast<std::int64_t>(entry.token),
                              entry.value);
    }
  }
  if (stop > 0.0) {
    categories.emplace_back(kStopOutcome, stop);
  }
  std::ranges::sort(categories, [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  PitValue out;
  out.p_stop = stop;
  out.p_top = categories.empty() ? 0.0 : categories.front().second;
  double below = 0.0;
  for (const auto& [id, prob] : categories) {
    if (id == outcome) {
      out.possible = prob > 0.0;
      out.p_outcome = prob;
      out.top = id == categories.front().first;
      out.u = below + v * prob;
      return out;
    }
    below += prob;
  }
  return out;
}

struct KsResult {
  std::size_t n{0};
  double d{0.0};
  double p{1.0};
};

/// Asymptotic Kolmogorov tail probability (Stephens' small-sample factor).
[[nodiscard]] inline double KolmogorovTail(double d, std::size_t n) {
  if (n == 0) {
    return 1.0;
  }
  const double root = std::sqrt(static_cast<double>(n));
  const double lambda = (root + 0.12 + 0.11 / root) * d;
  if (lambda < 0.2) {
    return 1.0;
  }
  double sum = 0.0;
  for (int k = 1; k <= 100; ++k) {
    const double term = std::exp(-2.0 * k * k * lambda * lambda);
    sum += (k % 2 == 1 ? 2.0 : -2.0) * term;
    if (term < 1e-12) {
      break;
    }
  }
  return std::clamp(sum, 0.0, 1.0);
}

/// One-sample Kolmogorov-Smirnov test against U(0, 1).
[[nodiscard]] inline KsResult KsUniform(std::vector<double> values) {
  KsResult out;
  out.n = values.size();
  if (values.empty()) {
    return out;
  }
  std::ranges::sort(values);
  const double n = static_cast<double>(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out.d = std::max({out.d, static_cast<double>(i + 1) / n - values[i],
                      values[i] - static_cast<double>(i) / n});
  }
  out.p = KolmogorovTail(out.d, out.n);
  return out;
}

/// Observed count of an event against the sum of its exact probabilities
/// (Poisson-binomial mean and variance).
struct Calibration {
  double observed{0.0};
  double expected{0.0};
  double variance{0.0};
  void Add(bool hit, double probability) {
    observed += hit ? 1.0 : 0.0;
    expected += probability;
    variance += probability * (1.0 - probability);
  }
  [[nodiscard]] double z() const {
    return variance > 0.0         ? (observed - expected) / std::sqrt(variance)
           : observed == expected ? 0.0
                                  : 1e9;
  }
};

/// Generated sequence: `length` tokens, then a stop or the length limit.
struct Sequence {
  std::vector<std::int64_t> tokens;
  bool stopped{false};
};

/// Outcome of position `i` (token, stop, or ended).
[[nodiscard]] inline std::int64_t OutcomeAt(const Sequence& s, std::size_t i) {
  if (i < s.tokens.size()) {
    return s.tokens[i];
  }
  return i == s.tokens.size() && s.stopped ? kStopOutcome : kEndedOutcome;
}

/// Each item is a vector of features (positions, or one feature); the
/// statistic is the sum over features of the total variation between the
/// two groups' category frequencies.
class PermutationTest {
public:
  explicit PermutationTest(
      const std::vector<std::vector<std::int64_t>>& items) {
    const std::size_t features = items.empty() ? 0 : items.front().size();
    dense_.assign(items.size(), std::vector<std::uint32_t>(features));
    categories_.assign(features, 0);
    for (std::size_t f = 0; f < features; ++f) {
      std::map<std::int64_t, std::uint32_t> ids;
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto [it, inserted] = ids.try_emplace(
            items[i][f], static_cast<std::uint32_t>(ids.size()));
        dense_[i][f] = it->second;
      }
      categories_[f] = static_cast<std::uint32_t>(ids.size());
    }
  }

  /// Statistic for the split: items [0, n_a) against the rest, in `order`.
  [[nodiscard]] double Statistic(std::span<const std::uint32_t> order,
                                 std::size_t n_a) const {
    const std::size_t n_b = order.size() - n_a;
    double total = 0.0;
    std::vector<double> diff;
    for (std::size_t f = 0; f < categories_.size(); ++f) {
      diff.assign(categories_[f], 0.0);
      for (std::size_t i = 0; i < order.size(); ++i) {
        diff[dense_[order[i]][f]] += i < n_a ? 1.0 / static_cast<double>(n_a)
                                             : -1.0 / static_cast<double>(n_b);
      }
      double tv = 0.0;
      for (const double d : diff) {
        tv += std::fabs(d);
      }
      total += 0.5 * tv;
    }
    return total;
  }

  struct Result {
    double statistic{0.0};
    double p{1.0};
    std::size_t permutations{0};
  };

  /// Groups: items [0, n_a) and [n_a, size); `permutations` relabellings.
  [[nodiscard]] Result Run(std::size_t n_a, std::size_t permutations,
                           std::uint64_t seed) const {
    std::vector<std::uint32_t> order(dense_.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    Result out;
    out.permutations = permutations;
    out.statistic = Statistic(order, n_a);
    std::mt19937_64 rng(seed);
    std::size_t at_least = 0;
    for (std::size_t k = 0; k < permutations; ++k) {
      std::ranges::shuffle(order, rng);
      // Ties count as at least as extreme (exact for discrete statistics).
      at_least += Statistic(order, n_a) >= out.statistic - 1e-12 ? 1 : 0;
    }
    out.p = static_cast<double>(at_least + 1) /
            static_cast<double>(permutations + 1);
    return out;
  }

private:
  std::vector<std::vector<std::uint32_t>> dense_;
  std::vector<std::uint32_t> categories_;
};

/// Two-sample comparison of sequences: per-position outcome distributions
/// (TV summed over positions), stop rate and length distribution.
struct TwoSample {
  PermutationTest::Result positions;
  PermutationTest::Result stop_rate;
  PermutationTest::Result length;
  std::vector<double> position_tv;  ///< observed TV per position
};

[[nodiscard]] inline TwoSample CompareSequences(const std::vector<Sequence>& a,
                                                const std::vector<Sequence>& b,
                                                std::size_t positions,
                                                std::size_t permutations,
                                                std::uint64_t seed) {
  std::vector<std::vector<std::int64_t>> by_position;
  std::vector<std::vector<std::int64_t>> stops;
  std::vector<std::vector<std::int64_t>> lengths;
  for (const auto* group : {&a, &b}) {
    for (const auto& s : *group) {
      std::vector<std::int64_t> row(positions);
      for (std::size_t i = 0; i < positions; ++i) {
        row[i] = OutcomeAt(s, i);
      }
      by_position.push_back(std::move(row));
      stops.push_back({s.stopped ? 1 : 0});
      lengths.push_back({static_cast<std::int64_t>(s.tokens.size()) * 2 +
                         (s.stopped ? 1 : 0)});
    }
  }
  TwoSample out;
  const PermutationTest per_position(by_position);
  out.positions = per_position.Run(a.size(), permutations, seed);
  out.stop_rate = PermutationTest(stops).Run(a.size(), permutations, seed + 1);
  out.length = PermutationTest(lengths).Run(a.size(), permutations, seed + 2);
  for (std::size_t i = 0; i < positions; ++i) {
    std::vector<std::vector<std::int64_t>> one;
    for (const auto& row : by_position) {
      one.push_back({row[i]});
    }
    std::vector<std::uint32_t> order(one.size());
    for (std::uint32_t k = 0; k < order.size(); ++k) {
      order[k] = k;
    }
    out.position_tv.push_back(PermutationTest(one).Statistic(order, a.size()));
  }
  return out;
}

/// Two-sided permutation test of a per-sequence statistic (difference of the
/// group means): long-output summaries such as the mean log-probability of
/// the committed tokens or the repeated n-gram rate.
[[nodiscard]] inline PermutationTest::Result MeanDifferenceTest(
    const std::vector<double>& a, const std::vector<double>& b,
    std::size_t permutations, std::uint64_t seed) {
  std::vector<double> all(a);
  all.insert(all.end(), b.begin(), b.end());
  const auto statistic = [&](std::span<const std::uint32_t> order) {
    double sa = 0.0;
    double sb = 0.0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      (i < a.size() ? sa : sb) += all[order[i]];
    }
    return std::fabs(
        sa / static_cast<double>(std::max<std::size_t>(1, a.size())) -
        sb / static_cast<double>(std::max<std::size_t>(1, b.size())));
  };
  std::vector<std::uint32_t> order(all.size());
  for (std::uint32_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  PermutationTest::Result out;
  out.permutations = permutations;
  out.statistic = statistic(order);
  std::mt19937_64 rng(seed);
  std::size_t at_least = 0;
  for (std::size_t k = 0; k < permutations; ++k) {
    std::ranges::shuffle(order, rng);
    at_least += statistic(order) >= out.statistic - 1e-12 ? 1 : 0;
  }
  out.p =
      static_cast<double>(at_least + 1) / static_cast<double>(permutations + 1);
  return out;
}

/// Permutation test of the pooled token frequencies (total variation between
/// the groups' unigram distributions; whole sequences are permuted).
[[nodiscard]] inline PermutationTest::Result UnigramTest(
    const std::vector<Sequence>& a, const std::vector<Sequence>& b,
    std::size_t permutations, std::uint64_t seed) {
  std::map<std::int64_t, std::uint32_t> ids;
  std::vector<std::vector<std::uint32_t>> items;
  for (const auto* group : {&a, &b}) {
    for (const auto& s : *group) {
      std::vector<std::uint32_t> row;
      for (const auto token : s.tokens) {
        row.push_back(
            ids.try_emplace(token, static_cast<std::uint32_t>(ids.size()))
                .first->second);
      }
      items.push_back(std::move(row));
    }
  }
  std::vector<double> ca(ids.size());
  std::vector<double> cb(ids.size());
  const auto statistic = [&](std::span<const std::uint32_t> order) {
    std::ranges::fill(ca, 0.0);
    std::ranges::fill(cb, 0.0);
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      auto& counts = i < a.size() ? ca : cb;
      for (const auto id : items[order[i]]) {
        counts[id] += 1.0;
      }
      (i < a.size() ? na : nb) += static_cast<double>(items[order[i]].size());
    }
    double tv = 0.0;
    for (std::size_t k = 0; k < ca.size(); ++k) {
      tv += std::fabs(ca[k] / std::max(na, 1.0) - cb[k] / std::max(nb, 1.0));
    }
    return 0.5 * tv;
  };
  std::vector<std::uint32_t> order(items.size());
  for (std::uint32_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  PermutationTest::Result out;
  out.permutations = permutations;
  out.statistic = statistic(order);
  std::mt19937_64 rng(seed);
  std::size_t at_least = 0;
  for (std::size_t k = 0; k < permutations; ++k) {
    std::ranges::shuffle(order, rng);
    at_least += statistic(order) >= out.statistic - 1e-12 ? 1 : 0;
  }
  out.p =
      static_cast<double>(at_least + 1) / static_cast<double>(permutations + 1);
  return out;
}

/// Fraction of positions whose n-gram (ending there) already occurred earlier
/// in the same sequence: degenerate repetition over long outputs.
[[nodiscard]] inline double RepeatedNgramRate(const Sequence& s,
                                              std::size_t n = 4) {
  if (s.tokens.size() <= n) {
    return 0.0;
  }
  std::map<std::vector<std::int64_t>, int> seen;
  std::size_t repeated = 0;
  for (std::size_t i = n - 1; i < s.tokens.size(); ++i) {
    std::vector<std::int64_t> gram(s.tokens.begin() + (i + 1 - n),
                                   s.tokens.begin() + (i + 1));
    repeated += seen[gram]++ > 0 ? 1 : 0;
  }
  return static_cast<double>(repeated) /
         static_cast<double>(s.tokens.size() + 1 - n);
}

}  // namespace gufo::tests::ling

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_SAMPLED_DISTRIBUTION_HPP_
