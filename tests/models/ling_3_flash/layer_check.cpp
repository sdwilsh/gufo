// Per-layer GPU check of Ling-3.0-Flash on a short prompt (16 tokens).
//
//   layer_check --model FIRST_SHARD.gguf
//       [--reference-dir DIR]  CPU reference dump (reference_probe free-run:
//                              index.json, embd.l00.f32, h_out.lNN.f32, ...)
//       [--tokens 1,2,3 | --tokens-file F]   default: tokens of the reference
//                              index.json, else a built-in sentence (16 ids)
//       [--dump-dir DIR]       default $TMPDIR/layer_check
//       [--layers N] [--max-batch N] [--timeout-seconds S] [--hold-seconds S]
//       [--invariance [--n 106] [--layer-first F --chain-in FILE]
//        [--chain-out FILE]]   numeric-family bit-identity per block (below)
//
// 1. Free run: the GPU feeds the tokens through the trunk in one prefill chunk
//    and dumps per-block tensors in the reference dump format
//    (DIR/free: embd, mixer_out, h_mid, h_out, kda_o, router_ids, router_w).
//    With a reference, per-layer free-running rel-L2 is reported (not gated:
//    routing flips compound).
// 2. Teacher forcing: every block runs alone on the reference's input
//    (h_out of the previous block, embd for block 0) and its output is
//    compared with the reference's block output. Two passes: "prefill" (one
//    chunk of all tokens, the kernels used for prompts) and "decode" (one
//    token at a time, the single-token kernels). Gate: local rel-L2 < 1e-2.
//    Tokens whose top-8 expert set differs from the reference (near-tie routing
//    flips) are excluded from the gated number and counted separately.
//    DIR/teacher_prefill and DIR/teacher_decode hold the outputs.
// Without --reference-dir the GPU's own free-run dump is both input and
// expected output (plumbing self-check: prefill-teacher must reproduce the
// free run almost exactly).

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

#include "tests/models/ling_3_flash/model_tool.hpp"

namespace gufo::tests::ling::layer_tool {

constexpr double kLocalGate = 1e-2;
constexpr double kSelfPrefillGate = 1e-3;
constexpr double kSelfDecodeGate = 5e-2;

std::string LayerFile(const std::string& dir, const char* name, int layer) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "/%s.l%02d.f32", name, layer);
  return dir + buf;
}

struct Metric {
  double rel{0.0};
  double worst_token{0.0};
  double rel_all{0.0};
  std::uint32_t flipped{0};
  bool valid{false};
};

/// `skip[t]` excludes tokens whose expert set differs from the reference (a
/// near-tie routing flip changes the block output by construction); `rel` is
/// measured over the remaining tokens, `rel_all` over every token.
Metric Compare(std::span<const float> a, std::span<const float> b,
               const std::vector<bool>& skip = {}) {
  Metric m;
  if (a.empty() || a.size() != b.size() || a.size() % l3::kHidden != 0) {
    return m;
  }
  m.valid = true;
  m.rel_all = RelL2(a, b);
  std::vector<float> ka;
  std::vector<float> kb;
  for (std::size_t t = 0; t < a.size() / l3::kHidden; ++t) {
    if (t < skip.size() && skip[t]) {
      ++m.flipped;
      continue;
    }
    ka.insert(ka.end(), a.begin() + t * l3::kHidden,
              a.begin() + (t + 1) * l3::kHidden);
    kb.insert(kb.end(), b.begin() + t * l3::kHidden,
              b.begin() + (t + 1) * l3::kHidden);
  }
  m.rel = ka.empty() ? 0.0 : RelL2(ka, kb);
  for (std::size_t t = 0; t < a.size() / l3::kHidden; ++t) {
    if (t < skip.size() && skip[t]) {
      continue;
    }
    m.worst_token =
        std::max(m.worst_token, RelL2(a.subspan(t * l3::kHidden, l3::kHidden),
                                      b.subspan(t * l3::kHidden, l3::kHidden)));
  }
  return m;
}

json::Value ToJson(const Metric& m) {
  if (!m.valid) {
    return json::Value();
  }
  auto o = json::Value::object();
  o["rel_l2"] = Round(m.rel, 4);
  o["worst_token_rel_l2"] = Round(m.worst_token, 4);
  o["rel_l2_all_tokens"] = Round(m.rel_all, 4);
  o["routing_flipped_tokens"] = U(m.flipped);
  return o;
}

// ---- numeric-family invariance (--invariance) -----------------------------
// Block by block over the resident window, with identical inputs (each block
// takes the previous block's single-chunk prefill output), the check requires
// the block output, the KDA recurrent state and conv window, and the MLA cache
// rows to be bit-identical across chunk layouts within a numeric family:
//   prefill family (chunks > 8 tokens): one chunk against two chunks, at any
//   split point.
//   small-batch family (chunks <= 8 tokens): token by token against chunks of
//   3, 5, and 8.
// Windows chain through --chain-in and --chain-out (raw F32 n x hidden), so the
// whole trunk can be covered on a GPU that holds only some blocks at a time.
struct Variant {
  std::string name;
  bool prefill_family;
  std::vector<std::uint32_t> chunks;
};

struct BlockSnap {
  std::vector<float> out;
  std::vector<float> state, conv;
  std::vector<std::uint16_t> cache;
};

bool SameBits(const BlockSnap& a, const BlockSnap& b) {
  const auto eq = [](const auto& x, const auto& y) {
    return x.size() == y.size() &&
           std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0;
  };
  return eq(a.out, b.out) && eq(a.state, b.state) && eq(a.conv, b.conv) &&
         eq(a.cache, b.cache);
}

int RunInvariance(const Args& args, Report& report,
                  std::shared_ptr<l3::Model> model) {
  std::string error;
  std::vector<std::int32_t> tokens;
  if (args.Has("--tokens")) {
    tokens = ParseIds(args.Str("--tokens"));
  } else {
    const auto want = static_cast<std::size_t>(args.Num("--n", 106));
    std::string text;
    while (tokens.size() < want) {
      text +=
          "The history of computing is a story of steadily shrinking "
          "machines and steadily growing ambitions. ";
      tokens = model->Tokenize(text);
    }
    tokens.resize(want);
  }
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t first = model->FirstLayer();
  const std::uint32_t layers = model->LayerCount();
  const std::size_t want = std::size_t{n} * l3::kHidden;
  if (n < 20) {
    report.Set("status", "error");
    report.Set("error", "--invariance needs at least 20 tokens");
    return 1;
  }
  auto session = model->CreateSession(model->MaxContext(), &error);
  if (!session) {
    report.Set("status", "error");
    report.Set("error", error);
    return 1;
  }
  std::vector<float> cur;
  if (first == 0) {
    l3::GpuDumpRecorder rec({}, {"embd"});
    if (!session->Feed(tokens, {.sink = &rec}, &error)) {
      report.Set("status", "error");
      report.Set("error", "embedding run: " + error);
      return 1;
    }
    const auto e = rec.Get("embd", 0);
    cur.assign(e.begin(), e.end());
  } else if (!ReadF32Dump(args.Str("--chain-in"), &cur)) {
    report.Set("status", "error");
    report.Set("error", "--layer-first needs --chain-in (F32 n x hidden)");
    return 1;
  }
  if (cur.size() < want) {
    report.Set("status", "error");
    report.Set("error", "chain input is shorter than n x hidden");
    return 1;
  }
  cur.resize(want);

  std::vector<Variant> variants;
  variants.push_back({"prefill_1_chunk", true, {n}});
  for (const std::uint32_t a : {9U, n * 37 / 100, n / 2, n - 9}) {
    if (a > 8 && n - a > 8) {
      variants.push_back(
          {"prefill_split_" + std::to_string(a) + "+" + std::to_string(n - a),
           true,
           {a, n - a}});
    }
  }
  variants.push_back(
      {"small_token_by_token", false, std::vector<std::uint32_t>(n, 1)});
  for (const std::uint32_t c : {3U, 5U, 8U}) {
    Variant v{"small_chunks_of_" + std::to_string(c), false, {}};
    for (std::uint32_t t = 0; t < n; t += c) {
      v.chunks.push_back(std::min(c, n - t));
    }
    variants.push_back(std::move(v));
  }

  auto rows = json::Value::array();
  std::vector<std::uint32_t> mismatches(variants.size(), 0);
  double worst_cross = 0.0;
  for (std::uint32_t k = 0; k < layers; ++k) {
    const std::uint32_t il = first + k;
    const double t0 = Now();
    std::vector<BlockSnap> snaps(variants.size());
    for (std::size_t vi = 0; vi < variants.size(); ++vi) {
      session->Reset();
      BlockSnap& sn = snaps[vi];
      sn.out.assign(want, 0.0F);
      std::uint32_t pos = 0;
      for (const std::uint32_t len : variants[vi].chunks) {
        if (!session->RunBlock(il, pos,
                               std::span<const float>(cur).subspan(
                                   std::size_t{pos} * l3::kHidden,
                                   std::size_t{len} * l3::kHidden),
                               std::span<float>(sn.out).subspan(
                                   std::size_t{pos} * l3::kHidden,
                                   std::size_t{len} * l3::kHidden),
                               nullptr, &error)) {
          report.Set("status", "error");
          report.Set("error", variants[vi].name + " layer " +
                                  std::to_string(il) + ": " + error);
          return 1;
        }
        pos += len;
      }
      bool ok = true;
      if (l3::IsMlaLayer(il)) {
        ok = session->ReadMlaCache(il, n, &sn.cache, &error);
      } else {
        ok = session->ReadKdaState(il, &sn.state, &sn.conv, &error);
      }
      if (!ok) {
        report.Set("status", "error");
        report.Set("error", "state read: " + error);
        return 1;
      }
    }
    auto row = json::Value::object();
    row["layer"] = U(il);
    row["kind"] = l3::IsMlaLayer(il) ? "mla" : "kda";
    row["ffn"] = l3::IsDenseLayer(il) ? "dense" : "moe";
    const BlockSnap* ref_prefill = &snaps[0];
    const BlockSnap* ref_small = nullptr;
    for (std::size_t vi = 0; vi < variants.size(); ++vi) {
      if (!variants[vi].prefill_family && ref_small == nullptr) {
        ref_small = &snaps[vi];
      }
    }
    auto bits = json::Value::object();
    for (std::size_t vi = 0; vi < variants.size(); ++vi) {
      const BlockSnap& ref =
          variants[vi].prefill_family ? *ref_prefill : *ref_small;
      const bool same = SameBits(snaps[vi], ref);
      bits[variants[vi].name] = same;
      if (!same) {
        ++mismatches[vi];
        std::fprintf(stderr,
                     "[invariance] layer %u %s DIFFERS (out rel-L2 %.3e)\n", il,
                     variants[vi].name.c_str(), RelL2(snaps[vi].out, ref.out));
      }
    }
    row["bitwise_identical_to_family_reference"] = std::move(bits);
    const double cross = RelL2(ref_small->out, ref_prefill->out);
    row["cross_family_rel_l2_report_only"] = Round(cross, 4);
    worst_cross = std::max(worst_cross, cross);
    rows.push_back(std::move(row));
    cur = snaps[0].out;  // chain: next block's input
    std::fprintf(stderr, "[invariance] layer %2u (%s) done [%.0f ms]\n", il,
                 l3::IsMlaLayer(il) ? "MLA" : "KDA", (Now() - t0) * 1e3);
  }
  if (args.Has("--chain-out")) {
    std::ofstream f(args.Str("--chain-out"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(cur.data()),
            static_cast<std::streamsize>(cur.size() * sizeof(float)));
  }
  auto summary = json::Value::object();
  summary["tokens"] = U(n);
  summary["first_layer"] = U(first);
  summary["layers"] = U(layers);
  std::uint32_t total_bad = 0;
  auto mm = json::Value::object();
  for (std::size_t vi = 0; vi < variants.size(); ++vi) {
    mm[variants[vi].name] = U(mismatches[vi]);
    total_bad += mismatches[vi];
  }
  summary["non_identical_layers_per_variant"] = std::move(mm);
  summary["worst_cross_family_rel_l2"] = Round(worst_cross, 4);
  summary["gate_pass"] = total_bad == 0;
  report.Set("layers", std::move(rows));
  report.Set("summary", std::move(summary));
  report.Set("status", total_bad == 0 ? "pass" : "fail");
  return total_bad == 0 ? 0 : 1;
}

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  if (args.Has("--invariance")) {
    return RunInvariance(args, report, std::move(model));
  }
  const std::string ref_dir = args.Str("--reference-dir");
  const std::string out_dir = args.Str("--dump-dir", TmpDir() + "/layer_check");
  const bool self_mode = ref_dir.empty();
  std::string error;

  std::vector<std::int32_t> tokens;
  if (args.Has("--tokens")) {
    tokens = ParseIds(args.Str("--tokens"));
  } else if (args.Has("--tokens-file")) {
    tokens = ParseIds(ReadText(args.Str("--tokens-file")));
  } else if (!self_mode) {
    try {
      const json::Value index = json::parse(ReadText(ref_dir + "/index.json"));
      if (const auto* t = index.find("tokens")) {
        for (const auto& v : t->items()) {
          tokens.push_back(static_cast<std::int32_t>(v.as_double()));
        }
      }
    } catch (const std::exception& e) {
      report.Set("status", "error");
      report.Set("error",
                 std::string("cannot read reference index.json: ") + e.what());
      return 1;
    }
  }

  if (tokens.empty()) {
    tokens = model->Tokenize(
        "The quick brown fox jumps over the lazy dog while the evening sun "
        "slowly sets behind the distant hills and the river keeps flowing.");
    tokens.resize(std::min<std::size_t>(tokens.size(), 16));
  }
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t first = model->FirstLayer();
  const std::uint32_t layers = model->LayerCount();
  const bool free_run = first == 0;
  if (!free_run && self_mode) {
    report.Set("status", "error");
    report.Set(
        "error",
        "--layer-first needs --reference-dir (no free run to self-check)");
    return 1;
  }
  if (n > model->PrefillCapacity()) {
    report.Set("status", "error");
    report.Set("error", "more tokens than --max-batch");
    return 1;
  }
  auto tj = json::Value::array();
  for (const auto t : tokens) {
    tj.push_back(t);
  }
  report.Set("tokens", std::move(tj));

  // ---- free run ----
  auto session = model->CreateSession(model->MaxContext(), &error);
  if (!session) {
    report.Set("status", "error");
    report.Set("error", error);
    return 1;
  }
  l3::GpuDumpRecorder free_dump;
  std::vector<l3::StepTiming> timings;
  if (free_run &&
      !session->Feed(tokens, {.sink = &free_dump, .timings = &timings},
                     &error)) {
    report.Set("status", "error");
    report.Set("error", "free run: " + error);
    return 1;
  }
  const auto top = free_run ? session->Argmax() : -1;
  std::fprintf(
      stderr,
      "[layer_check] free run %u tokens: %.1f ms, last-token top-1 %d\n", n,
      timings.empty() ? 0.0 : timings[0].total_ms, top);
  const std::string free_path = out_dir + "/free";
  std::string tokens_json;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens_json += (i ? ", " : "") + std::to_string(tokens[i]);
  }
  if (free_run &&
      !free_dump.Write(free_path, "",
                       "  \"mode\": \"free\",\n  \"source\": \"gpu\",\n  "
                       "\"hidden\": 2560,\n  \"tokens\": [" +
                           tokens_json + "],\n",
                       &error)) {
    report.Set("status", "error");
    report.Set("error", error);
    return 1;
  }
  if (free_run) {
    report.Set("free_dump_dir", free_path);
    report.Set("free_run_ms",
               Round(timings.empty() ? 0.0 : timings[0].total_ms));
    report.Set("last_token_top1", top);
  }

  // Inputs and expectations of the teacher passes.
  const auto embd = [&]() -> std::vector<float> {
    std::vector<float> v;
    if (self_mode) {
      const auto s = free_dump.Get("embd", 0);
      v.assign(s.begin(), s.end());
    } else {
      (void)ReadF32Dump(LayerFile(ref_dir, "embd", 0), &v);
    }
    return v;
  };
  const auto h_out_of = [&](int il) -> std::vector<float> {
    std::vector<float> v;
    if (self_mode) {
      const auto s = free_dump.Get("h_out", il);
      v.assign(s.begin(), s.end());
    } else {
      (void)ReadF32Dump(LayerFile(ref_dir, "h_out", il), &v);
    }
    return v;
  };
  const std::size_t want = std::size_t{n} * l3::kHidden;

  // Indexed by k = block - first.
  std::vector<std::vector<float>> inputs(layers);
  std::vector<std::vector<float>> expected(layers);
  for (std::uint32_t k = 0; k < layers; ++k) {
    const std::uint32_t il = first + k;
    inputs[k] = il == 0 ? embd() : h_out_of(static_cast<int>(il) - 1);
    expected[k] = h_out_of(static_cast<int>(il));
    if (inputs[k].size() < want || expected[k].size() < want) {
      report.Set("status", "error");
      report.Set("error", "missing or short reference tensors for layer " +
                              std::to_string(il) + " (need " +
                              std::to_string(n) + " tokens)");
      return 1;
    }
    inputs[k].resize(want);
    expected[k].resize(want);
  }

  // ---- teacher passes ----
  struct Pass {
    explicit Pass(const char* n)
        : name(n),
          dump({}, {"h_out", "router_ids", "mixer_out", "h_mid", "ffn_out",
                    "ffn_routed", "ffn_shared", "ffn_in", "ffn_gate", "ffn_up",
                    "ffn_act", "mla_q", "mla_kva", "mla_kv", "mla_qabs",
                    "mla_lat", "mla_vb"}) {}
    const char* name;
    std::vector<Metric> metrics;
    std::vector<json::Value> comps;
    l3::GpuDumpRecorder dump;
  };
  Pass prefill("teacher_prefill");
  Pass decode("teacher_decode");
  // Two prefill chunks (state carried, positions > 0, partial tiles).
  Pass split("teacher_prefill_split");
  // Chunks of 5 tokens (small-batch kernels) for a bitwise diff against the
  // token-by-token pass on identical inputs.
  Pass chunk5("teacher_chunks_of_5");
  const auto split_at = static_cast<std::uint32_t>(std::clamp<double>(
      args.Num("--split", static_cast<double>(n * 2 / 5)), 1.0, n - 1.0));
  for (Pass* pass : {&prefill, &decode, &split, &chunk5}) {
    session->Reset();
    const bool one_by_one = pass == &decode;
    for (std::uint32_t k = 0; k < layers; ++k) {
      const std::uint32_t il = first + k;
      std::vector<float> out(want);
      const double t0 = Now();
      if (pass == &chunk5) {
        for (std::uint32_t t = 0; t < n; t += 5) {
          const std::uint32_t len = std::min<std::uint32_t>(5, n - t);
          std::span<const float> in(
              inputs[k].data() + std::size_t{t} * l3::kHidden,
              std::size_t{len} * l3::kHidden);
          std::span<float> o(out.data() + std::size_t{t} * l3::kHidden,
                             std::size_t{len} * l3::kHidden);
          if (!session->RunBlock(il, t, in, o, &pass->dump, &error)) {
            report.Set("status", "error");
            report.Set("error", std::string(pass->name) + " layer " +
                                    std::to_string(il) + ": " + error);
            return 1;
          }
        }
      } else if (pass == &split) {
        const std::size_t cut = std::size_t{split_at} * l3::kHidden;
        std::span<const float> in_all(inputs[k]);
        std::span<float> out_all(out);
        if (!session->RunBlock(il, 0, in_all.first(cut), out_all.first(cut),
                               &pass->dump, &error) ||
            !session->RunBlock(il, split_at, in_all.subspan(cut),
                               out_all.subspan(cut), &pass->dump, &error)) {
          report.Set("status", "error");
          report.Set("error", std::string(pass->name) + " layer " +
                                  std::to_string(il) + ": " + error);
          return 1;
        }
      } else if (!one_by_one) {
        if (!session->RunBlock(il, 0, inputs[k], out, &pass->dump, &error)) {
          report.Set("status", "error");
          report.Set("error", std::string(pass->name) + " layer " +
                                  std::to_string(il) + ": " + error);
          return 1;
        }
      } else {
        for (std::uint32_t t = 0; t < n; ++t) {
          std::span<const float> in(
              inputs[k].data() + std::size_t{t} * l3::kHidden, l3::kHidden);
          std::span<float> o(out.data() + std::size_t{t} * l3::kHidden,
                             l3::kHidden);
          if (!session->RunBlock(il, t, in, o, &pass->dump, &error)) {
            report.Set("status", "error");
            report.Set("error", std::string(pass->name) + " layer " +
                                    std::to_string(il) + " token " +
                                    std::to_string(t) + ": " + error);
            return 1;
          }
        }
      }
      std::vector<bool> flipped;
      const auto got_ids =
          pass->dump.GetInt("router_ids", static_cast<int>(il));
      if (!self_mode && !got_ids.empty()) {
        std::ifstream f(ref_dir + "/router_ids.l" + (il < 10 ? "0" : "") +
                            std::to_string(il) + ".i32",
                        std::ios::binary);
        std::vector<std::int32_t> ref((std::size_t{n}) * l3::kExpertsUsed);
        f.read(reinterpret_cast<char*>(ref.data()),
               static_cast<std::streamsize>(ref.size() * 4));
        if (f && got_ids.size() >= ref.size()) {
          flipped.assign(n, false);
          for (std::uint32_t t = 0; t < n; ++t) {
            std::set<std::int32_t> a(ref.begin() + t * 8,
                                     ref.begin() + (t + 1) * 8);
            std::set<std::int32_t> b(got_ids.begin() + t * 8,
                                     got_ids.begin() + (t + 1) * 8);
            flipped[t] = a != b;
          }
        }
      }
      pass->metrics.push_back(Compare(out, expected[k], flipped));
      {
        // Component errors against the reference: the mixer (KDA or MLA,
        // including its output projection) and the FFN contribution
        // (h_out - h_mid: dense, or routed experts plus shared expert).
        auto c = json::Value::object();
        std::vector<float> rm;
        const int layer = static_cast<int>(il);
        if (!self_mode &&
            ReadF32Dump(LayerFile(ref_dir, "mixer_out", layer), &rm)) {
          rm.resize(want);
          const auto gm = pass->dump.Get("mixer_out", layer);
          const auto last = [&](std::span<const float> a,
                                std::span<const float> b) {
            return RelL2(
                a.subspan((n - 1) * std::size_t{l3::kHidden}, l3::kHidden),
                b.subspan((n - 1) * std::size_t{l3::kHidden}, l3::kHidden));
          };
          c["mixer"] = ToJson(Compare(gm, rm));
          c["mixer_last_token"] = Round(last(gm, rm), 4);
          // The FFN contribution is compared directly, without residual-stream
          // cancellation: the total, and routed and shared where dumped.
          for (const char* name : {"ffn_out", "ffn_routed", "ffn_shared",
                                   "ffn_in", "ffn_gate", "ffn_up", "ffn_act"}) {
            std::vector<float> rf;
            const auto gf = pass->dump.Get(name, layer);
            if (!gf.empty() &&
                ReadF32Dump(LayerFile(ref_dir, name, layer), &rf)) {
              rf.resize(want);
              const std::string key =
                  std::string(name) == "ffn_out" ? "ffn" : name;
              c[key] = ToJson(Compare(gf, rf, flipped));
              if (key == "ffn")
                c["ffn_last_token"] = Round(last(gf, rf), 4);
            }
          }
          c["h_out_last_token"] = Round(last(out, expected[k]), 4);
          c["last_token_routing_flipped"] = !flipped.empty() && flipped[n - 1];
        }
        pass->comps.push_back(std::move(c));
      }
      std::fprintf(stderr,
                   "[layer_check] %s layer %2u (%s): rel-L2 %.3e [%.0f ms]\n",
                   pass->name, il, l3::IsMlaLayer(il) ? "MLA" : "KDA",
                   pass->metrics.back().rel, (Now() - t0) * 1e3);
    }
    (void)pass->dump.Write(
        out_dir + "/" + pass->name, "",
        std::string("  \"mode\": \"teacher\",\n  \"source\": \"gpu\",\n  "
                    "\"hidden\": 2560,\n  \"tokens\": [") +
            tokens_json + "],\n",
        &error);
  }

  // ---- report ----
  auto rows = json::Value::array();
  double worst_prefill = 0.0;
  double worst_decode = 0.0;
  double worst_split = 0.0;
  int worst_split_layer = -1;
  double worst_free = 0.0;
  int worst_prefill_layer = -1;
  int worst_decode_layer = -1;
  for (std::uint32_t k = 0; k < layers; ++k) {
    const std::uint32_t il = first + k;
    auto row = json::Value::object();
    row["layer"] = U(il);
    row["kind"] = l3::IsMlaLayer(il) ? "mla" : "kda";
    row["ffn"] = l3::IsDenseLayer(il) ? "dense" : "moe";
    row["teacher_prefill"] = ToJson(prefill.metrics[k]);
    row["teacher_decode"] = ToJson(decode.metrics[k]);
    row["components_prefill"] = prefill.comps[k];
    row["components_decode"] = decode.comps[k];
    row["teacher_prefill_split"] = ToJson(split.metrics[k]);
    row["components_prefill_split"] = split.comps[k];
    if (split.metrics[k].rel >= worst_split) {
      worst_split = split.metrics[k].rel;
      worst_split_layer = static_cast<int>(il);
    }
    if (prefill.metrics[k].rel >= worst_prefill) {
      worst_prefill = prefill.metrics[k].rel;
      worst_prefill_layer = static_cast<int>(il);
    }
    if (decode.metrics[k].rel >= worst_decode) {
      worst_decode = decode.metrics[k].rel;
      worst_decode_layer = static_cast<int>(il);
    }
    if (!self_mode && free_run) {
      // Free-running comparisons against the reference's intermediates.
      const auto free_h = free_dump.Get("h_out", static_cast<int>(il));
      row["free_h_out"] = ToJson(Compare(free_h, expected[k]));
      worst_free = std::max(worst_free, Compare(free_h, expected[k]).rel);
      for (const char* name : {"mixer_out", "h_mid"}) {
        std::vector<float> ref;
        if (ReadF32Dump(LayerFile(ref_dir, name, static_cast<int>(il)), &ref)) {
          ref.resize(std::min(ref.size(), want));
          row[std::string("free_") + name] =
              ToJson(Compare(free_dump.Get(name, static_cast<int>(il)), ref));
        }
      }
      if (!l3::IsMlaLayer(il)) {
        std::vector<float> ref;
        if (ReadF32Dump(LayerFile(ref_dir, "kda_o", static_cast<int>(il)),
                        &ref)) {
          ref.resize(std::min<std::size_t>(ref.size(),
                                           std::size_t{n} * l3::kKdaInner));
          const auto got = free_dump.Get("kda_o", static_cast<int>(il));
          if (got.size() == ref.size() && !ref.empty()) {
            auto o = json::Value::object();
            o["rel_l2"] = Round(RelL2(got, ref), 4);
            row["free_kda_o"] = std::move(o);
          }
        }
      }
    }
    rows.push_back(std::move(row));
  }
  report.Set("layers", std::move(rows));

  auto summary = json::Value::object();
  summary["mode"] =
      self_mode ? "self (GPU free run is its own reference)" : "cpu_reference";
  summary["tokens"] = U(n);
  summary["layers"] = U(layers);
  summary["first_layer"] = U(first);
  summary["teacher_prefill_worst_rel_l2"] = Round(worst_prefill);
  summary["teacher_prefill_worst_layer"] = worst_prefill_layer;
  summary["teacher_decode_worst_rel_l2"] = Round(worst_decode);
  summary["teacher_decode_worst_layer"] = worst_decode_layer;
  summary["teacher_prefill_split_worst_rel_l2"] = Round(worst_split);
  summary["teacher_prefill_split_worst_layer"] = worst_split_layer;
  summary["split_at"] = U(split_at);
  if (!self_mode && free_run) {
    summary["free_running_worst_rel_l2"] = Round(worst_free);
  }
  const double gate_prefill = self_mode ? kSelfPrefillGate : kLocalGate;
  const double gate_decode = self_mode ? kSelfDecodeGate : kLocalGate;
  summary["gate_prefill_lt"] = gate_prefill;
  summary["gate_decode_lt"] = gate_decode;
  const bool finite =
      std::isfinite(worst_prefill) && std::isfinite(worst_decode);
  const bool pass = finite && worst_prefill < gate_prefill &&
                    worst_decode < gate_decode && worst_split < gate_prefill;
  summary["gate_pass"] = pass;
  report.Set("summary", std::move(summary));
  report.Set("status", pass ? "pass" : "fail");
  return pass ? 0 : 1;
}

}  // namespace gufo::tests::ling::layer_tool

#ifndef LING_TOOL_NO_MAIN
int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "layer_check",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model = gufo::tests::ling::LoadModel(args, report, 64, 256);
        return model ? gufo::tests::ling::layer_tool::Run(args, report, model)
                     : 1;
      });
}
#endif
