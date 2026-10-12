// Logit dump and oracle comparison for Ling-3.0-flash on the GPU.
//
//   logit_dump --model FIRST_SHARD.gguf
//       [--kl-base kl_base.bin]  [--kl-chunks N]
//       [--greedy greedy.jsonl]  [--greedy-tokens 64]
//       [--tokens-file ids.txt --positions all|last|3,5,9 [--dump-dir DIR]]
//       [--max-batch N] [--max-context N] [--layers N]
//       [--timeout-seconds S] [--hold-seconds S]
//
// * --kl-base: llama.cpp `--kl-divergence-base` file (format in
//   tools/models/ling_3_flash/kl_base.py). Every chunk is evaluated from an
//   empty context exactly as llama-perplexity does (the stored tokens are fed
//   verbatim; Ling has add_bos=false so llama.cpp inserts no BOS) and the
//   second half of the chunk is scored. Per position the KL against the base
//   distribution (llama.cpp definition: sum over base log-prob > -16 of
//   p_base * (log p_base - log p_gufo)), top-1 ids and log-probs are emitted.
// * --greedy: greedy.jsonl of the oracle (3 prompts x 64 tokens with top-5
//   log-probs). Free-running greedy decode plus a teacher-forced pass.
// * --tokens-file: writes full-vocab F32 logits of the requested positions to
//   DIR/logits.f32 with an index.json in the reference dump format.
//
// One JSON report on stdout (summary, per-position KL rows, greedy detail).
// Exit 77 without --model, 1 when a gate fails.

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>

#include "tests/models/ling_3_flash/model_tool.hpp"

namespace gufo::tests::ling::logit_tool {

constexpr double kKlMeanGate = 0.01;
constexpr double kTop1Gate = 0.98;
constexpr double kNearTieNats = 0.05;
constexpr std::uint32_t kGreedyIdentical = 32;

// ---- llama.cpp KL base reader ---------------------------------------------

class KlBase {
public:
  ~KlBase() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  bool Open(const std::string& path, std::string* error) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
      *error = "cannot open " + path;
      return false;
    }
    char magic[8];
    std::uint32_t head[3];
    if (::pread(fd_, magic, 8, 0) != 8 ||
        std::memcmp(magic, "_logits_", 8) != 0 ||
        ::pread(fd_, head, 12, 8) != 12) {
      *error = path + " is not a llama.cpp kl-divergence base file";
      return false;
    }
    n_ctx = head[0];
    n_vocab = static_cast<std::int32_t>(head[1]);
    n_chunk = static_cast<std::int32_t>(head[2]);
    if (n_ctx < 4 || n_vocab != static_cast<std::int32_t>(l3::kVocab) ||
        n_chunk <= 0) {
      *error = "unexpected base header: n_ctx " + std::to_string(n_ctx) +
               " n_vocab " + std::to_string(n_vocab) + " n_chunk " +
               std::to_string(n_chunk);
      return false;
    }
    tokens.resize(std::size_t{n_ctx} * static_cast<std::size_t>(n_chunk));
    const std::size_t token_bytes = tokens.size() * 4;
    if (::pread(fd_, tokens.data(), token_bytes, 20) !=
        static_cast<ssize_t>(token_bytes)) {
      *error = "truncated token block in " + path;
      return false;
    }
    first = n_ctx / 2;
    rows_per_chunk = n_ctx - 1 - first;
    nv = 2U * ((static_cast<std::uint32_t>(n_vocab) + 1U) / 2U) + 4U;
    data_offset = 20 + token_bytes;
    const off_t size = ::lseek(fd_, 0, SEEK_END);
    const std::uint64_t expect =
        data_offset + std::uint64_t{2} * nv * rows_per_chunk *
                          static_cast<std::uint64_t>(n_chunk);
    if (static_cast<std::uint64_t>(size) != expect) {
      *error = path + ": size " + std::to_string(size) + " != expected " +
               std::to_string(expect) + " (wrong n_ctx/chunk layout)";
      return false;
    }
    raw_.resize(nv);
    return true;
  }
  /// Log-probabilities of one scored row of one chunk (float32 like numpy).
  bool Row(std::int32_t chunk, std::uint32_t row, std::vector<float>* lp) {
    const std::uint64_t offset =
        data_offset +
        std::uint64_t{2} * nv *
            (static_cast<std::uint64_t>(chunk) * rows_per_chunk + row);
    const std::size_t bytes = std::size_t{nv} * 2;
    if (::pread(fd_, raw_.data(), bytes, static_cast<off_t>(offset)) !=
        static_cast<ssize_t>(bytes)) {
      return false;
    }
    float head[2];
    std::memcpy(head, raw_.data(), 8);
    lp->resize(static_cast<std::size_t>(n_vocab));
    for (std::int32_t v = 0; v < n_vocab; ++v) {
      (*lp)[static_cast<std::size_t>(v)] =
          static_cast<float>(raw_[4 + static_cast<std::size_t>(v)]) * head[0] +
          head[1];
    }
    return true;
  }
  std::uint32_t n_ctx{0};
  std::int32_t n_vocab{0}, n_chunk{0};
  std::vector<std::int32_t> tokens;
  std::uint32_t first{0}, rows_per_chunk{0}, nv{0};
  std::uint64_t data_offset{0};

private:
  int fd_{-1};
  std::vector<std::uint16_t> raw_;
};

/// log-softmax in double; returns the lowest index of the maximum.
std::int32_t LogSoftmax(std::span<const float> logits,
                        std::vector<double>* out) {
  out->resize(logits.size());
  std::int32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[static_cast<std::size_t>(best)]) {
      best = static_cast<std::int32_t>(i);
    }
  }
  const double m = logits[static_cast<std::size_t>(best)];
  double sum = 0.0;
  for (const float x : logits) {
    sum += std::exp(static_cast<double>(x) - m);
  }
  const double lse = m + std::log(sum);
  for (std::size_t i = 0; i < logits.size(); ++i) {
    (*out)[i] = static_cast<double>(logits[i]) - lse;
  }
  return best;
}

struct KlRow {
  std::int32_t chunk, pos, top1_test, top1_base;
  double kl, lp_test_of_base_top1, nll_test, nll_base;
};

double Percentile(std::vector<double> v, double q) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const double idx = q * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(idx));
  const auto hi = std::min(v.size() - 1, lo + 1);
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

json::Value KlSummary(const std::vector<KlRow>& rows, std::int32_t chunks_done,
                      const KlBase& base) {
  auto s = json::Value::object();
  std::vector<double> kl;
  double sum = 0.0;
  double nll_t = 0.0;
  double nll_b = 0.0;
  std::size_t same = 0;
  for (const auto& r : rows) {
    kl.push_back(r.kl);
    sum += r.kl;
    nll_t += r.nll_test;
    nll_b += r.nll_base;
    same += r.top1_test == r.top1_base ? 1 : 0;
  }
  const double n = static_cast<double>(std::max<std::size_t>(rows.size(), 1));
  const double mean = sum / n;
  double var = 0.0;
  for (const double k : kl) {
    var += (k - mean) * (k - mean);
  }
  s["chunks_done"] = chunks_done;
  s["chunks_total"] = base.n_chunk;
  s["n_ctx"] = U(base.n_ctx);
  s["positions"] = rows.size();
  s["kl_mean"] = Round(mean, 6);
  s["kl_mean_stderr"] = Round(std::sqrt(var / std::max(n - 1.0, 1.0) / n), 4);
  s["kl_median"] = Round(Percentile(kl, 0.5), 5);
  s["kl_p99"] = Round(Percentile(kl, 0.99), 5);
  s["kl_max"] =
      Round(kl.empty() ? 0.0 : *std::max_element(kl.begin(), kl.end()), 5);
  s["top1_agreement"] = Round(static_cast<double>(same) / n, 5);
  s["nll_base_mean"] = Round(nll_b / n, 5);
  s["nll_test_mean"] = Round(nll_t / n, 5);
  s["gate_kl_mean_le"] = kKlMeanGate;
  s["gate_top1_ge"] = kTop1Gate;
  s["gate_pass"] = !rows.empty() && mean <= kKlMeanGate &&
                   static_cast<double>(same) / n >= kTop1Gate &&
                   chunks_done == base.n_chunk;
  // Worst positions by KL.
  std::vector<std::size_t> order(rows.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return rows[a].kl > rows[b].kl;
  });
  auto worst = json::Value::array();
  for (std::size_t i = 0; i < std::min<std::size_t>(order.size(), 12); ++i) {
    const KlRow& r = rows[order[i]];
    auto o = json::Value::object();
    o["chunk"] = r.chunk;
    o["pos"] = r.pos;
    o["kl"] = Round(r.kl, 5);
    o["top1_test"] = r.top1_test;
    o["top1_base"] = r.top1_base;
    worst.push_back(std::move(o));
  }
  s["worst_positions"] = std::move(worst);
  return s;
}

bool RunKl(const Args& args, Report& report, l3::Model& model, KlBase& base) {
  std::string error;
  auto session = model.CreateSession(model.MaxContext(), &error);
  if (!session) {
    report.Set("kl_error", error);
    return false;
  }
  const std::int32_t chunks = std::min<std::int32_t>(
      base.n_chunk, static_cast<std::int32_t>(args.Num(
                        "--kl-chunks", static_cast<double>(base.n_chunk))));
  std::vector<KlRow> rows;
  std::vector<float> logits;
  std::vector<float> lp_base;
  std::vector<double> lp_test;
  for (std::int32_t c = 0; c < chunks; ++c) {
    const double t0 = Now();
    session->Reset();
    const std::span<const std::int32_t> tokens(
        base.tokens.data() +
            std::size_t{base.n_ctx} * static_cast<std::size_t>(c),
        base.n_ctx);
    if (!session->Feed(tokens,
                       {.logits_from = base.first, .all_logits = &logits},
                       &error)) {
      std::fprintf(stderr, "kl chunk %d failed: %s\n", c, error.c_str());
      report.Set("kl_error", "chunk " + std::to_string(c) + ": " + error);
      report.Set("kl", KlSummary(rows, c, base));
      return false;
    }
    const double t_fwd = Now() - t0;
    for (std::uint32_t i = 0; i < base.rows_per_chunk; ++i) {
      const std::span<const float> row(
          logits.data() + std::size_t{i} * l3::kVocab, l3::kVocab);
      if (!AllFinite(row)) {
        report.Set("kl_error", "non-finite logits at chunk " +
                                   std::to_string(c) + " row " +
                                   std::to_string(i));
        report.Set("kl", KlSummary(rows, c, base));
        return false;
      }
      if (!base.Row(c, i, &lp_base)) {
        report.Set("kl_error", "short read of the base file");
        return false;
      }
      const std::int32_t top_test = LogSoftmax(row, &lp_test);
      std::int32_t top_base = 0;
      for (std::int32_t v = 1; v < base.n_vocab; ++v) {
        if (lp_base[static_cast<std::size_t>(v)] >
            lp_base[static_cast<std::size_t>(top_base)]) {
          top_base = v;
        }
      }
      double kl = 0.0;
      for (std::int32_t v = 0; v < base.n_vocab; ++v) {
        const double lb = lp_base[static_cast<std::size_t>(v)];
        if (lb > -16.0) {
          kl += std::exp(lb) * (lb - lp_test[static_cast<std::size_t>(v)]);
        }
      }
      const std::uint32_t pos = base.first + i;
      const std::int32_t target = tokens[pos + 1];
      rows.push_back(
          {c, static_cast<std::int32_t>(pos), top_test, top_base, kl,
           lp_test[static_cast<std::size_t>(top_base)],
           -lp_test[static_cast<std::size_t>(target)],
           -static_cast<double>(lp_base[static_cast<std::size_t>(target)])});
    }
    double sum = 0.0;
    for (const auto& r : rows) {
      sum += r.kl;
    }
    std::fprintf(stderr,
                 "[kl] chunk %d/%d: forward %.2f s, total %.2f s, running KL "
                 "mean %.5f over %zu positions\n",
                 c + 1, chunks, t_fwd, Now() - t0,
                 sum / static_cast<double>(rows.size()), rows.size());
    report.Set("kl", KlSummary(rows, c + 1, base));
  }
  auto summary = KlSummary(rows, chunks, base);
  // Per-position rows: [chunk, pos, top1_gufo, top1_oracle, kl,
  // gufo_logprob_of_oracle_top1, nll_gufo, nll_oracle].
  auto table = json::Value::array();
  for (const auto& r : rows) {
    auto row = json::Value::array();
    row.push_back(r.chunk);
    row.push_back(r.pos);
    row.push_back(r.top1_test);
    row.push_back(r.top1_base);
    row.push_back(Round(r.kl));
    row.push_back(Round(r.lp_test_of_base_top1));
    row.push_back(Round(r.nll_test));
    row.push_back(Round(r.nll_base));
    table.push_back(std::move(row));
  }
  summary["row_columns"] =
      "chunk,pos,top1_gufo,top1_oracle,kl,gufo_logprob_of_oracle_top1,nll_gufo,"
      "nll_oracle";
  summary["rows"] = std::move(table);
  report.Set("kl", std::move(summary));
  const double n = static_cast<double>(std::max<std::size_t>(rows.size(), 1));
  double kl_sum = 0.0;
  std::size_t same = 0;
  for (const auto& r : rows) {
    kl_sum += r.kl;
    same += r.top1_test == r.top1_base ? 1 : 0;
  }
  return chunks != base.n_chunk || (kl_sum / n <= kKlMeanGate &&
                                    static_cast<double>(same) / n >= kTop1Gate);
}

// ---- greedy ----------------------------------------------------------------

struct TopTwo {
  std::int32_t first{-1}, second{-1};
  double gap{0.0};
};

TopTwo Top2(std::span<const float> logits) {
  TopTwo t;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (t.first < 0 || logits[i] > logits[static_cast<std::size_t>(t.first)]) {
      t.second = t.first;
      t.first = static_cast<std::int32_t>(i);
    } else if (t.second < 0 ||
               logits[i] > logits[static_cast<std::size_t>(t.second)]) {
      t.second = static_cast<std::int32_t>(i);
    }
  }
  if (t.second >= 0) {
    t.gap = static_cast<double>(logits[static_cast<std::size_t>(t.first)]) -
            static_cast<double>(logits[static_cast<std::size_t>(t.second)]);
  }
  return t;
}

json::Value IntArray(const std::vector<std::int32_t>& v) {
  auto a = json::Value::array();
  for (const auto x : v) {
    a.push_back(x);
  }
  return a;
}

bool RunGreedy(const Args& args, Report& report, l3::Model& model) {
  std::string error;
  const std::string path = args.Str("--greedy");
  const auto n_tokens =
      static_cast<std::size_t>(args.Num("--greedy-tokens", 64));
  std::istringstream lines(ReadText(path));
  std::string line;
  auto results = json::Value::array();
  bool all_pass = true;
  bool any = false;
  auto session = model.CreateSession(model.MaxContext(), &error);
  if (!session) {
    report.Set("greedy_error", error);
    return false;
  }
  std::vector<double> decode_ms;
  while (std::getline(lines, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    const json::Value rec = json::parse(line);
    const std::string name = rec.member_str("name");
    std::vector<std::int32_t> prompt;
    std::vector<std::int32_t> oracle;
    if (const auto* p = rec.find("prompt_ids")) {
      for (const auto& v : p->items()) {
        prompt.push_back(static_cast<std::int32_t>(v.as_double()));
      }
    }
    if (const auto* p = rec.find("tokens")) {
      for (const auto& v : p->items()) {
        oracle.push_back(static_cast<std::int32_t>(v.as_double()));
      }
    }
    const std::size_t n = std::min(n_tokens, oracle.size());
    if (prompt.empty() || n == 0) {
      report.Set("greedy_error", "record " + name + " lacks prompt_ids/tokens");
      return false;
    }
    any = true;
    auto out = json::Value::object();
    out["name"] = name;
    out["prompt_tokens"] = prompt.size();

    // ---- free-running greedy ----
    session->Reset();
    std::vector<l3::StepTiming> timings;
    double t0 = Now();
    if (!session->Feed(prompt, {.timings = &timings}, &error)) {
      report.Set("greedy_error", name + " prefill: " + error);
      return false;
    }
    out["prefill_ms"] = Round((Now() - t0) * 1e3);
    std::vector<std::int32_t> mine;
    std::vector<TopTwo> tops;
    std::int32_t divergence = -1;
    std::vector<float> divergence_logits_gap;
    for (std::size_t i = 0; i < n; ++i) {
      const TopTwo t2 = Top2(session->Logits());
      tops.push_back(t2);
      mine.push_back(t2.first);
      if (divergence < 0 && t2.first != oracle[i]) {
        divergence = static_cast<std::int32_t>(i);
        const auto lg = session->Logits();
        out["divergence_gufo_gap_to_oracle_token"] =
            Round(static_cast<double>(lg[static_cast<std::size_t>(t2.first)]) -
                  static_cast<double>(lg[static_cast<std::size_t>(oracle[i])]));
      }
      if (i + 1 < n) {
        std::vector<l3::StepTiming> st;
        t0 = Now();
        if (!session->Feed(std::span<const std::int32_t>(&mine.back(), 1),
                           {.timings = &st}, &error)) {
          report.Set("greedy_error", name + " decode: " + error);
          return false;
        }
        const double ms = (Now() - t0) * 1e3;
        decode_ms.push_back(ms);
        std::fprintf(stderr, "[greedy] %s step %zu: %.1f ms (device %.1f ms)\n",
                     name.c_str(), i + 1, ms,
                     st.empty() ? 0.0 : st[0].total_ms);
      }
    }
    out["tokens"] = IntArray(mine);
    out["text"] = model.Decode(mine);
    out["first_divergence"] = divergence;
    double oracle_gap = -1.0;
    if (divergence >= 0) {
      const auto d = static_cast<std::size_t>(divergence);
      out["divergence_gufo_token"] = mine[d];
      out["divergence_oracle_token"] = oracle[d];
      out["divergence_gufo_top2_gap"] = Round(tops[d].gap);
      if (const auto* top = rec.find("top");
          top != nullptr && d < top->items().size() &&
          top->items()[d].items().size() >= 2) {
        const auto& t = top->items()[d].items();
        oracle_gap =
            t[0].member_double("logprob") - t[1].member_double("logprob");
        out["divergence_oracle_top2_gap"] = Round(oracle_gap);
      }
    }
    const double to_oracle =
        divergence >= 0
            ? out.find("divergence_gufo_gap_to_oracle_token")->as_double()
            : 0.0;
    const bool near_tie =
        divergence >= 0 && (to_oracle < kNearTieNats ||
                            (oracle_gap >= 0.0 && oracle_gap < kNearTieNats));
    const bool free_pass =
        divergence < 0 ||
        static_cast<std::uint32_t>(divergence) >= kGreedyIdentical || near_tie;
    out["free_run_pass"] = free_pass;
    out["near_tie_divergence"] = near_tie;

    // ---- teacher-forced: oracle tokens in, compare argmax at each step ----
    session->Reset();
    if (!session->Feed(prompt, {}, &error)) {
      report.Set("greedy_error", name + " teacher prefill: " + error);
      return false;
    }
    std::size_t agree = 0;
    double lp_diff_sum = 0.0;
    std::size_t lp_diff_n = 0;
    auto mismatches = json::Value::array();
    std::vector<double> lsm;
    for (std::size_t i = 0; i < n; ++i) {
      const TopTwo t2 = Top2(session->Logits());
      (void)LogSoftmax(session->Logits(), &lsm);
      if (const auto* top = rec.find("top"); top != nullptr &&
                                             i < top->items().size() &&
                                             !top->items()[i].items().empty()) {
        const auto& first = top->items()[i].items()[0];
        const auto id = static_cast<std::size_t>(first.member_double("id"));
        if (id < lsm.size()) {
          lp_diff_sum += std::fabs(lsm[id] - first.member_double("logprob"));
          ++lp_diff_n;
        }
      }
      if (t2.first == oracle[i]) {
        ++agree;
      } else if (mismatches.size() < 16) {
        auto m = json::Value::object();
        m["index"] = i;
        m["gufo"] = t2.first;
        m["oracle"] = oracle[i];
        m["gufo_top2_gap"] = Round(t2.gap);
        m["gufo_gap_to_oracle_token"] =
            Round(static_cast<double>(
                      session->Logits()[static_cast<std::size_t>(t2.first)]) -
                  static_cast<double>(
                      session->Logits()[static_cast<std::size_t>(oracle[i])]));
        mismatches.push_back(std::move(m));
      }
      if (i + 1 < n &&
          !session->Feed(std::span<const std::int32_t>(&oracle[i], 1), {},
                         &error)) {
        report.Set("greedy_error", name + " teacher decode: " + error);
        return false;
      }
    }
    out["teacher_forced_agree"] = agree;
    out["teacher_forced_total"] = n;
    out["teacher_forced_mismatches"] = std::move(mismatches);
    out["oracle_top1_logprob_abs_diff_mean"] = Round(
        lp_diff_n == 0 ? 0.0 : lp_diff_sum / static_cast<double>(lp_diff_n));
    std::fprintf(
        stderr,
        "[greedy] %s: first divergence %d, teacher-forced agree %zu/%zu\n",
        name.c_str(), divergence, agree, n);
    all_pass = all_pass && free_pass;
    results.push_back(std::move(out));
  }
  std::sort(decode_ms.begin(), decode_ms.end());
  auto g = json::Value::object();
  g["prompts"] = std::move(results);
  g["gate_identical_tokens"] = U(kGreedyIdentical);
  g["gate_near_tie_nats"] = kNearTieNats;
  g["gate_pass"] = any && all_pass;
  g["decode_ms_median"] =
      Round(decode_ms.empty() ? 0.0 : decode_ms[decode_ms.size() / 2]);
  report.Set("greedy", std::move(g));
  return any && all_pass;
}

// ---- plain logit files -----------------------------------------------------

bool RunDump(const Args& args, Report& report, l3::Model& model) {
  std::string error;
  const auto ids = ParseIds(ReadText(args.Str("--tokens-file")));
  if (ids.empty()) {
    report.Set("dump_error", "--tokens-file is empty or unreadable");
    return false;
  }
  const std::string spec = args.Str("--positions", "last");
  std::vector<std::uint32_t> positions;
  if (spec == "all") {
    for (std::uint32_t i = 0; i < ids.size(); ++i) {
      positions.push_back(i);
    }
  } else if (spec == "last") {
    positions.push_back(static_cast<std::uint32_t>(ids.size() - 1));
  } else {
    for (const auto p : ParseIds(spec)) {
      positions.push_back(static_cast<std::uint32_t>(p));
    }
  }
  std::sort(positions.begin(), positions.end());
  if (positions.empty() || positions.back() >= ids.size()) {
    report.Set("dump_error", "positions outside the token list");
    return false;
  }
  auto session = model.CreateSession(model.MaxContext(), &error);
  if (!session) {
    report.Set("dump_error", error);
    return false;
  }
  std::vector<float> logits;
  const double t0 = Now();
  if (!session->Feed(ids,
                     {.logits_from = positions.front(), .all_logits = &logits},
                     &error)) {
    report.Set("dump_error", error);
    return false;
  }
  std::fprintf(stderr, "[dump] %zu tokens in %.2f s\n", ids.size(), Now() - t0);
  const std::string dir = args.Str("--dump-dir", TmpDir() + "/logit_dump");
  std::vector<float> selected;
  std::string plist;
  for (const auto p : positions) {
    const std::size_t row = p - positions.front();
    selected.insert(
        selected.end(),
        logits.begin() + static_cast<std::ptrdiff_t>(row * l3::kVocab),
        logits.begin() + static_cast<std::ptrdiff_t>((row + 1) * l3::kVocab));
    plist += (plist.empty() ? "" : ", ") + std::to_string(p);
  }
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::ofstream out(dir + "/logits.f32", std::ios::binary);
  out.write(reinterpret_cast<const char*>(selected.data()),
            static_cast<std::streamsize>(selected.size() * 4));
  out.close();
  std::string tokens_json;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    tokens_json += (i ? ", " : "") + std::to_string(ids[i]);
  }
  l3::GpuDumpRecorder recorder;
  const std::string entry =
      "    {\"name\": \"logits\", \"layer\": -1, \"file\": \"logits.f32\", "
      "\"dtype\": \"f32\", \"shape\": [" +
      std::to_string(positions.size()) + ", " + std::to_string(l3::kVocab) +
      "], \"positions\": [" + plist + "]}";
  if (!recorder.Write(
          dir, entry,
          "  \"mode\": \"gpu\",\n  \"tokens\": [" + tokens_json + "],\n",
          &error) ||
      !out) {
    report.Set("dump_error", error);
    return false;
  }
  auto d = json::Value::object();
  d["dir"] = dir;
  d["positions"] = positions.size();
  d["finite"] = AllFinite(selected);
  auto top = json::Value::array();
  for (std::size_t r = 0; r < positions.size() && r < 16; ++r) {
    const TopTwo t2 = Top2(
        std::span<const float>(selected).subspan(r * l3::kVocab, l3::kVocab));
    auto o = json::Value::object();
    o["pos"] = U(positions[r]);
    o["top1"] = t2.first;
    o["gap"] = Round(t2.gap);
    top.push_back(std::move(o));
  }
  d["top1"] = std::move(top);
  report.Set("dump", std::move(d));
  return AllFinite(selected);
}

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  KlBase base;
  std::string error;
  if (args.Has("--kl-base")) {
    if (!base.Open(args.Str("--kl-base"), &error)) {
      std::fprintf(stderr, "logit_dump: %s\n", error.c_str());
      report.Set("status", "error");
      report.Set("error", error);
      return 1;
    }
    if (base.n_ctx > model->PrefillCapacity() ||
        base.n_ctx > model->MaxContext()) {
      error = "--max-batch/--max-context must be at least the base's n_ctx " +
              std::to_string(base.n_ctx);
      report.Set("status", "error");
      report.Set("error", error);
      return 1;
    }
  }
  if (!args.Has("--kl-base") && !args.Has("--greedy") &&
      !args.Has("--tokens-file")) {
    report.Set("status", "error");
    report.Set("error",
               "nothing to do: give --kl-base, --greedy and/or --tokens-file");
    return 1;
  }
  bool ok = true;
  if (args.Has("--tokens-file")) {
    ok = RunDump(args, report, *model) && ok;
  }
  if (args.Has("--greedy")) {
    ok = RunGreedy(args, report, *model) && ok;
  }
  if (args.Has("--kl-base")) {
    ok = RunKl(args, report, *model, base) && ok;
  }
  report.Set("status", ok ? "pass" : "fail");
  return ok ? 0 : 1;
}

}  // namespace gufo::tests::ling::logit_tool

#ifndef LING_TOOL_NO_MAIN
int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "logit_dump",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        // The base file's n_ctx (default 512) must fit one forward pass.
        auto model = gufo::tests::ling::LoadModel(args, report, 512, 1024);
        return model ? gufo::tests::ling::logit_tool::Run(args, report, model)
                     : 1;
      });
}
#endif
