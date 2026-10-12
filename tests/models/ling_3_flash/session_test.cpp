// Full-model session checks for Ling-3.0-flash on the GPU: chunk invariance
// within one numeric family, gated at exact equality (prefill family: two
// chunks of more than 8 tokens == one chunk; small-batch family: chunks of 3,
// 5 or 8 == token by token), cross-family agreement (small-batch vs prefill,
// bounded at logits level) and the snapshot/restore round trip. Compares
// logits, the recurrent KDA state, the conv windows and the latent cache rows.
// Exit 77 without --model.
//
//   session_test --model FIRST_SHARD.gguf [--prompt-tokens N] [--decode-tokens
//   K]
//                [--max-context N] [--max-batch N] [--layers N]
//                [--timeout-seconds S] [--hold-seconds S]
//
// Prints one JSON report on stdout (checks[] with metrics and limits).

#include <algorithm>
#include <cstring>
#include <functional>

#include "tests/models/ling_3_flash/model_tool.hpp"

namespace gufo::tests::ling::session_tool {

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1U;
  const std::uint32_t exp = (h >> 10) & 0x1FU;
  const std::uint32_t man = h & 0x3FFU;
  float value;
  if (exp == 0) {
    value = std::ldexp(static_cast<float>(man), -24);
  } else if (exp == 31) {
    value = man == 0 ? INFINITY : NAN;
  } else {
    value = std::ldexp(static_cast<float>(man | 0x400U),
                       static_cast<int>(exp) - 25);
  }
  return sign ? -value : value;
}

struct Capture {
  std::vector<std::uint32_t> kda_layers;
  std::vector<std::vector<float>> state, conv;
  std::vector<std::uint32_t> mla_layers;
  std::vector<std::vector<float>> cache;
  std::vector<float> logits;
};

bool Take(l3::Session& s, std::uint32_t layers, Capture* out,
          std::string* error) {
  out->logits.assign(s.Logits().begin(), s.Logits().end());
  for (std::uint32_t il = 0; il < layers; ++il) {
    if (l3::IsMlaLayer(il)) {
      std::vector<std::uint16_t> raw;
      if (!s.ReadMlaCache(il, s.Position(), &raw, error)) {
        return false;
      }
      std::vector<float> f(raw.size());
      std::transform(raw.begin(), raw.end(), f.begin(), HalfToFloat);
      out->mla_layers.push_back(il);
      out->cache.push_back(std::move(f));
    } else {
      std::vector<float> st, cv;
      if (!s.ReadKdaState(il, &st, &cv, error)) {
        return false;
      }
      out->kda_layers.push_back(il);
      out->state.push_back(std::move(st));
      out->conv.push_back(std::move(cv));
    }
  }
  return true;
}

struct Diff {
  double logits{0.0};
  double max_abs_logit{0.0};
  bool top1_same{false};
  double kl{0.0};          ///< KL(reference || candidate) of the softmax
  double ref_margin{0.0};  ///< reference top-1 minus top-2 logit
  double state{0.0}, conv{0.0}, cache{0.0};  ///< worst layer rel-L2
  int worst_state_layer{-1}, worst_cache_layer{-1};
  bool bitwise{false};
  bool finite{true};
};

Diff Compare(const Capture& a, const Capture& b) {
  Diff d;
  d.logits = RelL2(a.logits, b.logits);
  for (std::size_t i = 0; i < a.logits.size() && i < b.logits.size(); ++i) {
    d.max_abs_logit =
        std::max<double>(d.max_abs_logit, std::fabs(a.logits[i] - b.logits[i]));
  }
  if (!a.logits.empty() && !b.logits.empty()) {
    d.top1_same =
        std::max_element(a.logits.begin(), a.logits.end()) - a.logits.begin() ==
        std::max_element(b.logits.begin(), b.logits.end()) - b.logits.begin();
  }
  d.finite = AllFinite(a.logits) && AllFinite(b.logits);
  if (d.finite && !a.logits.empty() && a.logits.size() == b.logits.size()) {
    const auto log_sum_exp = [](const std::vector<float>& v) {
      const double m = *std::max_element(v.begin(), v.end());
      double sum = 0.0;
      for (const float x : v) {
        sum += std::exp(x - m);
      }
      return m + std::log(sum);
    };
    const double lse_a = log_sum_exp(a.logits);
    const double lse_b = log_sum_exp(b.logits);
    for (std::size_t i = 0; i < a.logits.size(); ++i) {
      const double lp_b = b.logits[i] - lse_b;
      d.kl += std::exp(lp_b) * (lp_b - (a.logits[i] - lse_a));
    }
    auto top2 = b.logits;
    std::partial_sort(top2.begin(), top2.begin() + 2, top2.end(),
                      std::greater<float>());
    d.ref_margin = static_cast<double>(top2[0]) - top2[1];
  }
  bool bits = a.logits == b.logits;
  for (std::size_t i = 0; i < a.state.size(); ++i) {
    const double rs = RelL2(a.state[i], b.state[i]);
    if (rs >= d.state) {
      d.state = rs;
      d.worst_state_layer = static_cast<int>(a.kda_layers[i]);
    }
    d.conv = std::max(d.conv, RelL2(a.conv[i], b.conv[i]));
    bits = bits && a.state[i] == b.state[i] && a.conv[i] == b.conv[i];
    d.finite = d.finite && AllFinite(a.state[i]) && AllFinite(b.state[i]);
  }
  for (std::size_t i = 0; i < a.cache.size(); ++i) {
    const double rc = RelL2(a.cache[i], b.cache[i]);
    if (rc >= d.cache) {
      d.cache = rc;
      d.worst_cache_layer = static_cast<int>(a.mla_layers[i]);
    }
    bits = bits && a.cache[i] == b.cache[i];
    d.finite = d.finite && AllFinite(a.cache[i]);
  }
  d.bitwise = bits;
  return d;
}

json::Value ToJson(const Diff& d) {
  auto o = json::Value::object();
  o["logits_rel_l2"] = Round(d.logits);
  o["logits_max_abs"] = Round(d.max_abs_logit);
  o["top1_same"] = d.top1_same;
  o["kda_state_rel_l2"] = Round(d.state);
  o["kda_state_worst_layer"] = d.worst_state_layer;
  o["kda_conv_rel_l2"] = Round(d.conv);
  o["mla_cache_rel_l2"] = Round(d.cache);
  o["mla_cache_worst_layer"] = d.worst_cache_layer;
  o["bitwise_identical"] = d.bitwise;
  o["finite"] = d.finite;
  return o;
}

// Numeric families (see the reproducibility contract in engine.hpp). Chunks of
// 1..8 tokens are the small-batch/decode family (two-plane int8 Q8_0 GEMV,
// decode MoE, one decode-attention launch per token); chunks above 8 tokens
// are the prefill family (F16 WMMA GEMM, prefill attention). They are
// different numeric formats by design. Inside one family the result is
// independent of the chunk layout, so those comparisons are gated at exact
// (bitwise) equality of the logits, the recurrent state, the conv windows and
// the cache. Across families the logits differ by the format error of the two
// paths compounded over all layers, which is not a state-carry signal, so only
// a logits-level bound is gated there and state/cache numbers are reported.
constexpr double kCrossFamilyLogits = 2e-2;  // B2r on gfx1151 measured 4.9e-3

bool CrossFamilyOk(const Diff& d) {
  return d.finite && d.logits <= kCrossFamilyLogits;
}

bool Exact(const Diff& d) {
  return d.finite && d.bitwise && d.logits == 0.0 && d.top1_same;
}

const char* kText =
    "The history of computing is a story of steadily shrinking machines and "
    "steadily growing ambitions. Early computers filled entire rooms, drew "
    "enormous power and needed teams of engineers to keep their vacuum tubes "
    "alive. The transistor changed that, and the integrated circuit changed "
    "it again, packing thousands and later billions of switches onto a sliver "
    "of silicon. Software followed the same arc: from hand-wired programs to "
    "assembly language, then compilers, operating systems, networks and the "
    "web. Today a phone in a pocket outperforms the supercomputers of the "
    "nineteen eighties, and language models trained on vast amounts of text "
    "can write code, summarize documents and answer questions. Running such "
    "a model efficiently means moving as few bytes as possible for every "
    "token that is produced; ";

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  const std::uint32_t layers = model->LayerCount();
  const auto want_prompt =
      static_cast<std::uint32_t>(args.Num("--prompt-tokens", 100));
  const auto want_decode =
      static_cast<std::uint32_t>(args.Num("--decode-tokens", 6));
  std::string text;
  std::vector<std::int32_t> ids;
  while (ids.size() < want_prompt + want_decode) {
    text += kText;
    ids = model->Tokenize(text);
  }
  ids.resize(want_prompt + want_decode);
  const std::uint32_t N = want_prompt;
  const std::uint32_t K = want_decode;
  const std::uint32_t total = N + K;
  const std::uint32_t ctx = model->MaxContext();
  std::string error;
  auto checks = json::Value::array();
  int failures = 0;
  const auto finish_check = [&](const char* name, bool pass,
                                json::Value metrics) {
    metrics["name"] = name;
    metrics["pass"] = pass;
    checks.push_back(std::move(metrics));
    std::fprintf(stderr, "[check] %-28s %s\n", name, pass ? "PASS" : "FAIL");
    failures += pass ? 0 : 1;
  };
  const auto fail_with = [&](const char* what) {
    std::fprintf(stderr, "session_test: %s: %s\n", what, error.c_str());
    report.Set("status", "error");
    report.Set("error", std::string(what) + ": " + error);
    report.Set("checks", std::move(checks));
    return 1;
  };

  // Reference: the whole sequence in one prefill chunk.
  auto s1 = model->CreateSession(ctx, &error);
  if (!s1) {
    return fail_with("create session");
  }
  std::vector<l3::StepTiming> timings;
  double t0 = Now();
  if (!s1->Feed(ids, {.timings = &timings}, &error)) {
    return fail_with("single-chunk prefill");
  }
  std::fprintf(stderr, "[prefill] %u tokens in %.1f ms\n", total,
               (Now() - t0) * 1e3);
  Capture ref;
  if (!Take(*s1, layers, &ref, &error)) {
    return fail_with("capture reference");
  }
  {
    auto o = json::Value::object();
    o["tokens"] = U(total);
    o["top1_token"] = s1->Argmax();
    o["top1_text"] = model->TokenText(s1->Argmax());
    o["logits_finite"] = AllFinite(ref.logits);
    o["prefill_ms"] = Round(timings.empty() ? 0.0 : timings[0].total_ms);
    o["prefill_kda_ms"] = Round(timings.empty() ? 0.0 : timings[0].kda_ms);
    o["prefill_mla_ms"] = Round(timings.empty() ? 0.0 : timings[0].mla_ms);
    finish_check("reference_prefill",
                 AllFinite(ref.logits) && s1->Argmax() >= 0, std::move(o));
  }

  // Prefill family: two chunks (both above 8 tokens) == one chunk, exactly.
  for (const std::uint32_t a :
       {std::uint32_t{9}, std::max<std::uint32_t>(9, total * 37 / 100),
        total / 2, total - 9}) {
    if (a <= 8 || total - a <= 8) {
      continue;
    }
    auto s2 = model->CreateSession(ctx, &error);
    if (!s2) {
      return fail_with("create session");
    }
    if (!s2->Feed(std::span(ids).first(a), {}, &error) ||
        !s2->Feed(std::span(ids).subspan(a), {}, &error)) {
      return fail_with("two-chunk prefill");
    }
    Capture cap;
    if (!Take(*s2, layers, &cap, &error)) {
      return fail_with("capture two-chunk");
    }
    const Diff d = Compare(cap, ref);
    auto o = ToJson(d);
    o["split_at"] = U(a);
    o["family"] = "prefill";
    o["gate"] = "bitwise";
    const std::string name = "prefill_split_" + std::to_string(a) + "+" +
                             std::to_string(total - a) + "_vs_1";
    finish_check(name.c_str(), Exact(d), std::move(o));
  }

  // Token by token: the small-batch reference.
  Capture cap1;
  {
    auto s1t = model->CreateSession(ctx, &error);
    if (!s1t) {
      return fail_with("create session");
    }
    for (std::uint32_t at = 0; at < total; ++at) {
      if (!s1t->Feed(std::span(ids).subspan(at, 1), {}, &error)) {
        return fail_with("token-by-token");
      }
    }
    if (!Take(*s1t, layers, &cap1, &error)) {
      return fail_with("capture token-by-token");
    }
  }

  // Small-batch family: chunks of 3, 5 and 8 tokens == token by token, exactly.
  Capture cap5;
  for (const std::uint32_t chunk : {5U, 3U, 8U}) {
    auto sc = model->CreateSession(ctx, &error);
    if (!sc) {
      return fail_with("create session");
    }
    for (std::uint32_t at = 0; at < total; at += chunk) {
      const std::uint32_t len = std::min<std::uint32_t>(chunk, total - at);
      if (!sc->Feed(std::span(ids).subspan(at, len), {}, &error)) {
        return fail_with("short-chunk prefill");
      }
    }
    Capture cap;
    if (!Take(*sc, layers, &cap, &error)) {
      return fail_with("capture short chunks");
    }
    const Diff d = Compare(cap, cap1);
    auto o = ToJson(d);
    o["family"] = "small-batch";
    o["gate"] = "bitwise";
    o["chunk"] = U(chunk);
    const std::string name =
        "chunks_of_" + std::to_string(chunk) + "_vs_token_by_token";
    finish_check(name.c_str(), Exact(d), std::move(o));
    if (chunk == 5) {
      cap5 = std::move(cap);
    }
  }

  // Cross-family (report-only state/cache, logits-level gate): small-batch
  // chunks vs the single prefill chunk.
  {
    const Diff d = Compare(cap5, ref);
    auto o = ToJson(d);
    o["family"] = "cross (small-batch vs prefill)";
    o["gate"] = "logits_rel_l2 only";
    o["limit_logits_rel_l2"] = kCrossFamilyLogits;
    finish_check("chunks_of_5_vs_prefill", CrossFamilyOk(d), std::move(o));
  }

  // Prefill N then decode k == prefill N+k.
  {
    auto s3 = model->CreateSession(ctx, &error);
    if (!s3) {
      return fail_with("create session");
    }
    std::vector<l3::StepTiming> steps;
    if (!s3->Feed(std::span(ids).first(N), {}, &error)) {
      return fail_with("prefill N");
    }
    for (std::uint32_t i = 0; i < K; ++i) {
      if (!s3->Feed(std::span(ids).subspan(N + i, 1), {.timings = &steps},
                    &error)) {
        return fail_with("decode step");
      }
    }
    Capture cap;
    if (!Take(*s3, layers, &cap, &error)) {
      return fail_with("capture decode");
    }
    const Diff d = Compare(cap, ref);
    auto o = ToJson(d);
    // Decode uses the two-plane int8 Q8_0 GEMV; prefill F16 activations.
    o["family"] = "cross (small-batch vs prefill)";
    o["gate"] = "logits_rel_l2 only";
    o["limit_logits_rel_l2"] = kCrossFamilyLogits;
    std::vector<double> ms;
    for (const auto& st : steps) {
      ms.push_back(st.total_ms);
    }
    std::sort(ms.begin(), ms.end());
    o["decode_steps"] = U(K);
    o["decode_ms_median"] = Round(ms.empty() ? 0.0 : ms[ms.size() / 2]);
    finish_check("prefill_then_decode", CrossFamilyOk(d), std::move(o));
  }

  // Snapshot / restore round trip.
  {
    auto s4 = model->CreateSession(ctx, &error);
    if (!s4 || !s4->Feed(std::span(ids).first(N), {}, &error)) {
      return fail_with("snapshot prefill");
    }
    Capture before;
    if (!Take(*s4, layers, &before, &error)) {
      return fail_with("capture before snapshot");
    }
    auto snap = s4->SaveSnapshot(&error);
    if (!snap) {
      return fail_with("save snapshot");
    }
    std::vector<float> first_logits;
    std::vector<std::int32_t> chain;
    for (std::uint32_t i = 0; i < K; ++i) {
      if (!s4->Feed(std::span(ids).subspan(N + i, 1), {}, &error)) {
        return fail_with("decode after snapshot");
      }
      chain.push_back(s4->Argmax());
    }
    first_logits.assign(s4->Logits().begin(), s4->Logits().end());
    if (!s4->RestoreSnapshot(*snap, &error)) {
      return fail_with("restore snapshot");
    }
    Capture after;
    if (!Take(*s4, layers, &after, &error)) {
      return fail_with("capture after restore");
    }
    const Diff restored = Compare(after, before);
    bool replay_ok = true;
    for (std::uint32_t i = 0; i < K; ++i) {
      replay_ok =
          replay_ok && s4->Feed(std::span(ids).subspan(N + i, 1), {}, &error);
    }
    if (!replay_ok) {
      return fail_with("replay after restore");
    }
    const double replay_rel = RelL2(s4->Logits(), first_logits);
    const bool replay_top1 = s4->Argmax() == chain.back();
    // A divergent history must be rejected.
    std::vector<std::int32_t> other(ids.begin(), ids.begin() + N);
    other[N / 2] = (other[N / 2] + 7) % 1000;
    std::string reject_error;
    s4->Reset();
    const bool refed = s4->Feed(other, {}, &error);
    const bool rejected = refed && !s4->RestoreSnapshot(*snap, &reject_error);
    auto o = ToJson(restored);
    o["restored_position"] = U(snap->Position());
    o["snapshot_bytes"] = static_cast<double>(snap->SizeBytes());
    o["replay_logits_rel_l2"] = Round(replay_rel);
    o["replay_top1_same"] = replay_top1;
    o["divergent_history_rejected"] = rejected;
    const bool pass =
        restored.bitwise && replay_rel <= 1e-4 && replay_top1 && rejected;
    finish_check("snapshot_restore", pass, std::move(o));
  }

  report.Set("checks", std::move(checks));
  report.Set("failures", failures);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

}  // namespace gufo::tests::ling::session_tool

#ifndef LING_TOOL_NO_MAIN
int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "session_test",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model = gufo::tests::ling::LoadModel(args, report, 256, 512);
        return model ? gufo::tests::ling::session_tool::Run(args, report, model)
                     : 1;
      });
}
#endif
