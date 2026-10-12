// Prefill to decode handoff study on the full model. For each prompt of N+k
// tokens, within one build, the study compares path A (prefill N tokens, then k
// single-token decode steps) with path R (one prefill of all N+k tokens). It
// measures:
//   * logits rel-L2 and KL(R || A) at the end,
//   * per layer, the rel-L2 of the block output (h_out) over the k last tokens,
//     the KDA state and conv rel-L2, the MLA cache rel-L2, and the number of
//     those tokens whose top-8 expert set differs (routing flips),
// and writes one JSON report. study-verdict.py (container) compares the
// reports of two builds. The study uses only the public engine API, so it does
// not depend on the build.
//
//   handoff_study --model FIRST_SHARD.gguf --greedy greedy.jsonl
//       --text-tokens ids.txt --novel-text file.txt [--decode-tokens 6]
//       [--max-batch 512] [--max-context 1024]

#include <algorithm>
#include <random>

#include "src/models/ling_3_flash/gpu_dump.hpp"
#include "tests/models/ling_3_flash/model_tool.hpp"

namespace gufo::tests::ling::study {

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

class Collector final : public l3::GpuTensorSink {
public:
  [[nodiscard]] bool Wants(std::string_view name, int) const override {
    return name == "h_out" || name == "router_ids";
  }
  void Float(std::string_view, int layer, std::uint32_t,
             std::span<const float> rows) override {
    auto& v = h[layer];
    v.insert(v.end(), rows.begin(), rows.end());
  }
  void Int(std::string_view, int layer, std::uint32_t,
           std::span<const std::int32_t> rows) override {
    auto& v = ids[layer];
    v.insert(v.end(), rows.begin(), rows.end());
  }
  std::map<int, std::vector<float>> h;
  std::map<int, std::vector<std::int32_t>> ids;
};

struct Prompt {
  std::string name;
  std::vector<std::int32_t> ids;  // N + k tokens
  std::uint32_t n{0};
};

double Kl(std::span<const float> p_logits, std::span<const float> q_logits) {
  const auto lse = [](std::span<const float> x) {
    double m = *std::max_element(x.begin(), x.end());
    double s = 0;
    for (const float v : x) {
      s += std::exp(v - m);
    }
    return m + std::log(s);
  };
  const double lp = lse(p_logits);
  const double lq = lse(q_logits);
  double kl = 0;
  for (std::size_t i = 0; i < p_logits.size(); ++i) {
    const double lpi = p_logits[i] - lp;
    kl += std::exp(lpi) * (lpi - (q_logits[i] - lq));
  }
  return kl;
}

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  const auto k = static_cast<std::uint32_t>(args.Num("--decode-tokens", 6));
  const std::uint32_t layers = model->LayerCount();
  std::vector<Prompt> prompts;
  // Oracle prompts followed by their greedy continuation.
  {
    std::istringstream in(ReadText(args.Str("--greedy")));
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) {
        continue;
      }
      const json::Value j = json::parse(line);
      Prompt p;
      p.name = "oracle_" + j.member_str("name", "");
      for (const auto& v : j.find("prompt_ids")->items()) {
        p.ids.push_back(static_cast<std::int32_t>(v.as_double()));
      }
      p.n = static_cast<std::uint32_t>(p.ids.size());
      const auto& toks = j.find("tokens")->items();
      for (std::uint32_t i = 0; i < k && i < toks.size(); ++i) {
        p.ids.push_back(static_cast<std::int32_t>(toks[i].as_double()));
      }
      prompts.push_back(std::move(p));
    }
  }
  const auto windows = [&](const char* tag,
                           const std::vector<std::int32_t>& stream,
                           std::uint32_t salt) {
    std::uint32_t i = 0;
    for (const std::uint32_t n : {9U, 16U, 17U, 31U, 64U, 100U, 200U, 300U}) {
      Prompt p;
      p.name = std::string(tag) + "_" + std::to_string(n);
      p.n = n;
      const std::size_t off = (salt + 211U * i++) % (stream.size() - n - k);
      p.ids.assign(stream.begin() + off, stream.begin() + off + n + k);
      prompts.push_back(std::move(p));
    }
  };
  windows("text", ParseIds(ReadText(args.Str("--text-tokens"))), 17);
  windows("novel", model->Tokenize(ReadText(args.Str("--novel-text"))), 401);
  {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<std::int32_t> d(1000, 100000);
    for (const std::uint32_t n : {9U, 16U, 31U, 64U, 128U}) {
      Prompt p;
      p.name = "random_" + std::to_string(n);
      p.n = n;
      for (std::uint32_t i = 0; i < n + k; ++i) {
        p.ids.push_back(d(rng));
      }
      prompts.push_back(std::move(p));
    }
  }

  std::string error;
  const std::uint32_t ctx = model->MaxContext();
  auto out = json::Value::array();
  for (const Prompt& p : prompts) {
    const std::uint32_t total = p.n + k;
    if (p.ids.size() < total) {
      continue;
    }
    std::fprintf(stderr, "[study] %s N=%u k=%u\n", p.name.c_str(), p.n, k);
    Collector ca;
    Collector cr;
    std::vector<float> logits_a;
    std::vector<float> logits_r;
    std::vector<std::vector<float>> state_a, conv_a, cache_a, state_r, conv_r,
        cache_r;
    const auto take = [&](l3::Session& s, std::vector<std::vector<float>>& st,
                          std::vector<std::vector<float>>& cv,
                          std::vector<std::vector<float>>& ca_) {
      st.resize(layers);
      cv.resize(layers);
      ca_.resize(layers);
      for (std::uint32_t il = 0; il < layers; ++il) {
        if (l3::IsMlaLayer(il)) {
          std::vector<std::uint16_t> raw;
          if (!s.ReadMlaCache(il, s.Position(), &raw, &error)) {
            return false;
          }
          ca_[il].resize(raw.size());
          std::transform(raw.begin(), raw.end(), ca_[il].begin(), HalfToFloat);
        } else if (!s.ReadKdaState(il, &st[il], &cv[il], &error)) {
          return false;
        }
      }
      return true;
    };
    {
      auto s = model->CreateSession(ctx, &error);
      if (!s || !s->Feed(std::span(p.ids).first(p.n), {}, &error)) {
        std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
        return 1;
      }
      for (std::uint32_t t = p.n; t < total; ++t) {
        if (!s->Feed(std::span(p.ids).subspan(t, 1), {.sink = &ca}, &error)) {
          std::fprintf(stderr, "decode failed: %s\n", error.c_str());
          return 1;
        }
      }
      logits_a.assign(s->Logits().begin(), s->Logits().end());
      if (!take(*s, state_a, conv_a, cache_a)) {
        return 1;
      }
    }
    {
      auto s = model->CreateSession(ctx, &error);
      if (!s ||
          !s->Feed(std::span(p.ids).first(total), {.sink = &cr}, &error)) {
        std::fprintf(stderr, "reference prefill failed: %s\n", error.c_str());
        return 1;
      }
      logits_r.assign(s->Logits().begin(), s->Logits().end());
      if (!take(*s, state_r, conv_r, cache_r)) {
        return 1;
      }
    }
    auto o = json::Value::object();
    o["name"] = p.name;
    o["n"] = U(p.n);
    o["k"] = U(k);
    o["logits_rel_l2"] = Round(RelL2(logits_a, logits_r), 6);
    o["kl"] = Round(Kl(logits_r, logits_a), 6);
    o["top1_same"] =
        std::max_element(logits_a.begin(), logits_a.end()) - logits_a.begin() ==
        std::max_element(logits_r.begin(), logits_r.end()) - logits_r.begin();
    auto d_h = json::Value::array();
    auto d_state = json::Value::array();
    auto d_conv = json::Value::array();
    auto d_cache = json::Value::array();
    auto flips = json::Value::array();
    for (std::uint32_t il = 0; il < layers; ++il) {
      const auto& ha = ca.h[static_cast<int>(il)];
      const auto& hr = cr.h[static_cast<int>(il)];
      const std::size_t keep = ha.size();  // k rows
      d_h.push_back(Round(
          hr.size() >= keep ? RelL2(ha, std::span(hr).last(keep)) : -1.0, 6));
      d_state.push_back(Round(
          l3::IsMlaLayer(il) ? -1.0 : RelL2(state_a[il], state_r[il]), 6));
      d_conv.push_back(
          Round(l3::IsMlaLayer(il) ? -1.0 : RelL2(conv_a[il], conv_r[il]), 6));
      d_cache.push_back(Round(
          l3::IsMlaLayer(il) ? RelL2(cache_a[il], cache_r[il]) : -1.0, 6));
      int nf = -1;
      const auto ia = ca.ids.find(static_cast<int>(il));
      const auto ir = cr.ids.find(static_cast<int>(il));
      if (ia != ca.ids.end() && ir != cr.ids.end()) {
        nf = 0;
        const std::size_t w = 8;
        for (std::uint32_t t = 0; t < k; ++t) {
          std::vector<std::int32_t> x(ia->second.begin() + t * w,
                                      ia->second.begin() + (t + 1) * w);
          const std::size_t base = ir->second.size() - std::size_t{k} * w;
          std::vector<std::int32_t> y(ir->second.begin() + base + t * w,
                                      ir->second.begin() + base + (t + 1) * w);
          std::sort(x.begin(), x.end());
          std::sort(y.begin(), y.end());
          nf += x != y ? 1 : 0;
        }
      }
      flips.push_back(nf);
    }
    o["d_h"] = std::move(d_h);
    o["d_state"] = std::move(d_state);
    o["d_conv"] = std::move(d_conv);
    o["d_cache"] = std::move(d_cache);
    o["flips"] = std::move(flips);
    out.push_back(std::move(o));
  }
  report.Set("prompts", std::move(out));
  report.Set("status", "pass");
  return 0;
}

}  // namespace gufo::tests::ling::study

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "handoff_study",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model = gufo::tests::ling::LoadModel(args, report, 512, 1024);
        return model ? gufo::tests::ling::study::Run(args, report, model) : 1;
      });
}
