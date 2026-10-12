// Decode-path digest of Ling-3.0-flash: runs a scripted prefill, token-by-token
// decode, small-batch chunks (2..8 tokens), a snapshot rollback and two
// interleaved sessions, and prints one FNV-1a digest per step (logits bits and
// device argmax). With `--graph-compare` the script runs twice, with HIP-graph
// decode on and off (ModelOptions::graphs), and the digests must be identical;
// the printed digests also let two builds be compared bit for bit. After the
// script, four sessions at different depths decode in batched cohorts
// (Session::EvaluateBatch, widths 2 and 4, varying order); their per-session
// digests ("batch_digests", kept apart so "digests" stays comparable with
// builds without batching) must equal those of the same tokens decoded by
// each session alone. Exit 77 without --model.
//
//   decode_digest --model FIRST_SHARD.gguf [--mtp] [--layers N] [--max-context
//   N]
//                 [--prompt-tokens P] [--decode-tokens K] [--graph-compare]
//                 [--max-batch N] [--expanded-kv-bytes N]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/engine.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace sampling = gufo::sampling;

namespace {

std::uint64_t Fnv(const void* data, std::size_t bytes, std::uint64_t h) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    h = (h ^ p[i]) * 1099511628211ULL;
  }
  return h;
}

struct Script {
  std::vector<std::string> lines;
  std::vector<std::string> batch_lines;  ///< per session, batched cohorts
  std::size_t batch_vs_solo_differ{0};
  bool ok{true};
  std::uint64_t graph_replays{0};  ///< HIP graph replays of the whole script
};

std::int32_t Tok(std::uint32_t i) {
  return static_cast<std::int32_t>((i * 2654435761U + 12345U) % 150000U);
}

std::string DigestLine(Script* out, const char* what, std::uint32_t step,
                       const l3::Session& s) {
  const auto logits = s.Logits();
  std::uint64_t h = Fnv(logits.data(), logits.size() * sizeof(float),
                        14695981039346656037ULL);
  const auto host = static_cast<std::int32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
  if (host != s.Argmax()) {
    std::fprintf(stderr, "argmax mismatch at %s %u: device %d host %d\n", what,
                 step, s.Argmax(), host);
    out->ok = false;
  }
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s %u pos=%u argmax=%d logits=%016llx", what,
                step, s.Position(), s.Argmax(),
                static_cast<unsigned long long>(h));
  return buf;
}

l3::Session::BatchLane Item(l3::Session* session, std::int32_t token) {
  l3::Session::BatchLane lane;
  lane.session = session;
  lane.token = token;
  return lane;
}

void Digest(Script* out, const char* what, std::uint32_t step,
            const l3::Session& s) {
  out->lines.push_back(DigestLine(out, what, step, s));
}

/// Batched cohorts: lane i decodes BatchTok(i, j) for j = 0..3, in cohorts of
/// varying width and order; each lane's digest after each step must equal the
/// digest of the same tokens on one session alone (`solo`, restored from the
/// lane's start). Lanes may be dropped right after graph replays; the
/// graph-compare run's second load relies on their teardown releasing the
/// model's memory (rocm::Session::DestroyGraphs, reload_test).
void BatchCohorts(l3::Model& model, const l3::ModelOptions& options,
                  std::uint32_t prompt, l3::Session& solo, Script* out) {
  constexpr std::uint32_t kLanes = 4;
  constexpr std::uint32_t kSteps = 4;
  const auto batch_tok = [](std::uint32_t lane, std::uint32_t j) {
    return Tok(5000 + 17 * lane + j);
  };
  const std::uint32_t depths[kLanes] = {prompt, prompt / 2 + 3, 9, 1};
  std::string error;
  std::vector<std::unique_ptr<l3::Session>> lanes;
  std::vector<std::unique_ptr<l3::SessionSnapshot>> start;
  for (std::uint32_t i = 0; i < kLanes; ++i) {
    // Only the context a lane needs: its expanded K/V blocks come from the
    // shared budget and the device (the long form runs at 24,570 tokens).
    auto s = model.CreateSession(
        std::min(options.max_context, depths[i] + kSteps + 8), &error);
    std::vector<std::int32_t> toks;
    for (std::uint32_t t = 0; t < depths[i]; ++t)
      toks.push_back(Tok(3000 + 101 * i + t));
    if (!s || !s->Feed(toks, {}, &error)) {
      std::fprintf(stderr, "batch prefill failed: %s\n", error.c_str());
      out->ok = false;
      return;
    }
    start.push_back(s->SaveSnapshot(&error));
    if (!start.back()) {
      out->ok = false;
      return;
    }
    lanes.push_back(std::move(s));
  }
  std::vector<std::vector<std::string>> solo_lines(kLanes);
  for (std::uint32_t i = 0; i < kLanes; ++i) {
    if (!solo.RestoreSnapshot(*start[i], &error)) {
      out->ok = false;
      return;
    }
    const std::string label = "lane-" + std::to_string(i);
    for (std::uint32_t j = 0; j < kSteps; ++j) {
      if (!solo.Evaluate(batch_tok(i, j), &error)) {
        out->ok = false;
        return;
      }
      solo_lines[i].push_back(DigestLine(out, label.c_str(), j, solo));
    }
  }
  // Every lane takes kSteps steps: widths 4, 4 (reversed), 2, 2, 4.
  const std::vector<std::vector<std::uint32_t>> cohorts = {
      {0, 1, 2, 3}, {3, 2, 1, 0}, {1, 3}, {0, 2}, {2, 0, 3, 1}};
  std::uint32_t step[kLanes] = {};
  for (const auto& cohort : cohorts) {
    std::vector<l3::Session::BatchLane> batch;
    for (const std::uint32_t i : cohort)
      batch.push_back(Item(lanes[i].get(), batch_tok(i, step[i])));
    l3::Session::EvaluateBatch(batch);
    for (std::size_t k = 0; k < cohort.size(); ++k) {
      const std::uint32_t i = cohort[k];
      if (!batch[k].ok) {
        std::fprintf(stderr, "batched step failed: %s\n",
                     batch[k].error.c_str());
        out->ok = false;
        return;
      }
      const std::string label = "lane-" + std::to_string(i);
      std::string line = DigestLine(out, label.c_str(), step[i], *lanes[i]);
      if (line != solo_lines[i][step[i]]) {
        ++out->batch_vs_solo_differ;
        std::fprintf(
            stderr, "BATCH MISMATCH (width %zu)\n  batch: %s\n  solo: %s\n",
            cohort.size(), line.c_str(), solo_lines[i][step[i]].c_str());
      }
      out->batch_lines.push_back("w" + std::to_string(cohort.size()) + " " +
                                 line);
      ++step[i];
    }
  }
  // Batched MTP cycles (Session::DecodeBatch, greedy, adaptive widths): each
  // session's tokens and final logits equal greedy DecodeStep of it alone.
  // Cycle sizes differ (the cohort chooses the greedy width), so the digest is
  // per session at the end.
  if (model.HasMtp()) {
    constexpr std::uint32_t kTokens = 12;
    const auto final_line = [&](std::uint32_t i, const l3::Session& s,
                                const std::vector<std::int32_t>& toks) {
      std::uint64_t h = Fnv(toks.data(), toks.size() * sizeof(std::int32_t),
                            14695981039346656037ULL);
      const auto logits = s.Logits();
      h = Fnv(logits.data(), logits.size() * sizeof(float), h);
      char buf[160];
      std::snprintf(buf, sizeof buf, "mtp-lane-%u pos=%u final=%016llx", i,
                    s.Position(), static_cast<unsigned long long>(h));
      return std::string(buf);
    };
    const auto bad = [&](const char* what) {
      std::fprintf(stderr, "batched MTP digest: %s: %s\n", what, error.c_str());
      out->ok = false;
    };
    sampling::SamplingConfig greedy;
    greedy.temperature = 0.0F;
    std::vector<std::string> want(kLanes);
    for (std::uint32_t i = 0; i < kLanes; ++i) {
      if (!solo.RestoreSnapshot(*start[i], &error)) {
        return bad("solo restore");
      }
      solo.ResetDraftPolicy();
      sampling::SamplerState sampler(greedy);
      std::vector<std::int32_t> toks;
      while (toks.size() < kTokens) {
        l3::Session::DecodeResult r;
        if (!solo.DecodeStep(kTokens - toks.size(), sampler, &r, &error,
                             false)) {
          return bad("solo decode");
        }
        toks.insert(toks.end(), r.tokens.begin(), r.tokens.end());
      }
      want[i] = final_line(i, solo, toks);
    }
    std::vector<sampling::SamplerState> samplers(
        kLanes, sampling::SamplerState(greedy));
    std::vector<std::vector<std::int32_t>> toks(kLanes);
    for (std::uint32_t i = 0; i < kLanes; ++i) {
      if (!lanes[i]->RestoreSnapshot(*start[i], &error)) {
        return bad("lane restore");
      }
      lanes[i]->ResetDraftPolicy();
    }
    while (true) {
      std::vector<l3::Session::BatchDecode> batch;
      std::vector<std::uint32_t> which;
      for (std::uint32_t i = 0; i < kLanes; ++i) {
        if (toks[i].size() < kTokens) {
          l3::Session::BatchDecode d;
          d.session = lanes[i].get();
          d.max_tokens = kTokens - toks[i].size();
          d.sampler = &samplers[i];
          d.stop_at_eos = false;
          batch.push_back(std::move(d));
          which.push_back(i);
        }
      }
      if (batch.empty()) {
        break;
      }
      l3::Session::DecodeBatch(batch);
      for (std::size_t k = 0; k < batch.size(); ++k) {
        if (!batch[k].ok || batch[k].result.tokens.empty()) {
          std::fprintf(stderr, "batched MTP failed: %s\n",
                       batch[k].error.c_str());
          out->ok = false;
          return;
        }
        auto& t = toks[which[k]];
        t.insert(t.end(), batch[k].result.tokens.begin(),
                 batch[k].result.tokens.end());
      }
    }
    for (std::uint32_t i = 0; i < kLanes; ++i) {
      const std::string line = final_line(i, *lanes[i], toks[i]);
      if (line != want[i]) {
        ++out->batch_vs_solo_differ;
        std::fprintf(stderr, "BATCH MTP MISMATCH\n  batch: %s\n  solo: %s\n",
                     line.c_str(), want[i].c_str());
      }
      out->batch_lines.push_back(line);
    }
  }
  out->ok = out->ok && out->batch_vs_solo_differ == 0;
}

Script Run(const std::string& path, const l3::ModelOptions& options,
           std::uint32_t prompt, std::uint32_t decode) {
  Script out;
  std::string error;
  auto model = l3::Model::Load(path, options, &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    out.ok = false;
    return out;
  }
  auto a = model->CreateSession(options.max_context, &error);
  auto b = model->CreateSession(options.max_context, &error);
  if (!a || !b) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    out.ok = false;
    return out;
  }
  std::vector<std::int32_t> toks;
  for (std::uint32_t i = 0; i < prompt; ++i)
    toks.push_back(Tok(i));
  if (!a->Feed(toks, {}, &error) || !b->Feed(toks, {}, &error)) {
    std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
    out.ok = false;
    return out;
  }
  Digest(&out, "prefill", 0, *a);
  std::uint32_t next = prompt;
  // Token-by-token decode, the two sessions interleaved (each owns its graph).
  for (std::uint32_t i = 0; i < decode; ++i) {
    if (!a->Evaluate(Tok(next + i), &error) ||
        !b->Evaluate(Tok(next + 7 * i + 3), &error)) {
      std::fprintf(stderr, "decode failed: %s\n", error.c_str());
      out.ok = false;
      return out;
    }
    Digest(&out, "decode-a", i, *a);
    Digest(&out, "decode-b", i, *b);
  }
  next += decode;
  // Small-batch chunks of every size (own graph per size, each after a warmup).
  for (std::uint32_t rep = 0; rep < 2; ++rep) {
    for (std::uint32_t n = 2; n <= 8; ++n) {
      std::vector<std::int32_t> chunk;
      for (std::uint32_t i = 0; i < n; ++i)
        chunk.push_back(Tok(next++));
      if (!a->Feed(chunk, {}, &error)) {
        std::fprintf(stderr, "chunk failed: %s\n", error.c_str());
        out.ok = false;
        return out;
      }
      Digest(&out, "chunk", n + 10 * rep, *a);
    }
  }
  // Rollback and replay with a different mix.
  auto snap = a->SaveSnapshot(&error);
  for (std::uint32_t i = 0; i < 4; ++i) {
    if (!a->Evaluate(Tok(next + i), &error))
      out.ok = false;
  }
  Digest(&out, "post-snap", 0, *a);
  if (!snap || !a->RestoreSnapshot(*snap, &error))
    out.ok = false;
  for (std::uint32_t i = 0; i < 4; ++i) {
    if (!a->Evaluate(Tok(next + i), &error))
      out.ok = false;
  }
  Digest(&out, "replay", 0, *a);
  // Reset and reuse (same session object, graphs persist).
  a->Reset();
  std::vector<std::int32_t> again;  // 12 tokens whatever the prompt length
  for (std::uint32_t i = 0; i < 12; ++i)
    again.push_back(Tok(i));
  if (!a->Feed(again, {}, &error)) {
    std::fprintf(stderr, "feed after reset failed: %s\n", error.c_str());
    out.ok = false;
  }
  for (std::uint32_t i = 0; i < 3; ++i) {
    if (!a->Evaluate(Tok(100 + i), &error)) {
      std::fprintf(stderr, "evaluate after reset failed: %s\n", error.c_str());
      out.ok = false;
    }
    Digest(&out, "after-reset", i, *a);
  }
  // Profiled (eager, per-operation events) step == plain step, bit for bit.
  {
    auto snap2 = a->SaveSnapshot(&error);
    std::vector<l3::OpTiming> ops;
    if (!snap2 || !a->Feed(std::vector<std::int32_t>{Tok(900)},
                           {.op_profile = &ops}, &error)) {
      out.ok = false;
    } else {
      const std::vector<float> profiled(a->Logits().begin(), a->Logits().end());
      double ms = 0.0, mb = 0.0;
      for (const auto& op : ops) {
        ms += op.ms;
        mb += op.bytes / 1e6;
        std::fprintf(stderr, "[op] %-24s x%-3u %8.3f ms %9.2f MB\n",
                     op.name.c_str(), op.count, op.ms, op.bytes / 1e6);
      }
      std::fprintf(stderr, "[op] total %.3f ms, %.1f MB\n", ms, mb);
      if (ops.empty() || !a->RestoreSnapshot(*snap2, &error) ||
          !a->Evaluate(Tok(900), &error)) {
        out.ok = false;
      } else {
        const auto plain = a->Logits();
        out.ok = out.ok && plain.size() == profiled.size() &&
                 std::memcmp(plain.data(), profiled.data(),
                             profiled.size() * sizeof(float)) == 0;
      }
    }
  }
  // The same for a prefill-family chunk (n > 8): the op profile also breaks
  // the MLA attention down, and the profiled chunk equals the plain one.
  {
    std::vector<std::int32_t> chunk;
    for (std::uint32_t i = 0; i < 40; ++i)
      chunk.push_back(Tok(1000 + i));
    auto snap3 = a->SaveSnapshot(&error);
    std::vector<l3::OpTiming> ops;
    if (!snap3 || !a->Feed(chunk, {.op_profile = &ops}, &error)) {
      std::fprintf(stderr, "profiled prefill chunk failed: %s\n",
                   error.c_str());
      out.ok = false;
    } else {
      const std::vector<float> profiled(a->Logits().begin(), a->Logits().end());
      bool attention = false;
      bool breakdown = false;
      for (const auto& op : ops) {
        attention = attention || op.name == "mla.attention";
        breakdown = breakdown || op.name == "mla.attention/flash";
      }
      if (attention != breakdown) {
        std::fprintf(stderr, "prefill op profile lacks the MLA breakdown\n");
        out.ok = false;
      }
      if (ops.empty() || !a->RestoreSnapshot(*snap3, &error) ||
          !a->Feed(chunk, {}, &error)) {
        out.ok = false;
      } else {
        const auto plain = a->Logits();
        out.ok = out.ok && plain.size() == profiled.size() &&
                 std::memcmp(plain.data(), profiled.data(),
                             profiled.size() * sizeof(float)) == 0;
      }
    }
  }
  BatchCohorts(*model, options, prompt, *b, &out);
  out.graph_replays = model->GraphReplays();
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  // Before the first HIP call: a missing $TMPDIR crashes the runtime.
  if (!gufo::core::RequireTempDirectory())
    return 1;
  std::string model;
  l3::ModelOptions options;
  options.max_context = 512;
  options.max_batch = 128;
  std::uint32_t prompt = 40;
  std::uint32_t decode = 6;
  bool compare = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : "";
    };
    if (a == "--model")
      model = next();
    else if (a == "--layers")
      options.layer_limit = std::stoul(next());
    else if (a == "--max-context")
      options.max_context = std::stoul(next());
    else if (a == "--max-batch")
      options.max_batch = std::stoul(next());
    else if (a == "--expanded-kv-bytes")
      options.expanded_kv_bytes = std::stoull(next());
    else if (a == "--prompt-tokens")
      prompt = std::stoul(next());
    else if (a == "--decode-tokens")
      decode = std::stoul(next());
    else if (a == "--graph-compare")
      compare = true;
    else if (a == "--mtp")
      options.mtp = true;  // the digests must equal those of a run without
  }
  if (model.empty()) {
    std::fprintf(stderr, "no --model: skipping\n");
    return 77;
  }
  options.graphs = true;
  Script graph = Run(model, options, prompt, decode);
  bool ok = graph.ok;
  std::size_t diff = 0;
  std::size_t eager_steps = 0;
  if (compare) {
    options.graphs = false;
    Script eager = Run(model, options, prompt, decode);
    eager_steps = eager.lines.size();
    for (std::size_t i = 0;
         i < std::min(graph.lines.size(), eager.lines.size()); ++i) {
      if (graph.lines[i] != eager.lines[i]) {
        ++diff;
        std::fprintf(stderr, "MISMATCH\n  graph: %s\n  eager: %s\n",
                     graph.lines[i].c_str(), eager.lines[i].c_str());
      }
    }
    ok = ok && eager.ok && diff == 0 &&
         graph.lines.size() == eager.lines.size() && eager.graph_replays == 0 &&
         graph.graph_replays > 0 && graph.batch_lines == eager.batch_lines;
  }
  // One JSON document on stdout: the digests (comparable across builds) and
  // the graph-vs-eager verdict.
  std::printf(
      "{\"test\": \"decode_digest\", \"pass\": %s, \"steps\": %zu, "
      "\"graph_replays\": %llu",
      ok ? "true" : "false", graph.lines.size(),
      static_cast<unsigned long long>(graph.graph_replays));
  if (compare) {
    std::printf(", \"graph_vs_eager_differ\": %zu, \"eager_steps\": %zu", diff,
                eager_steps);
  }
  std::printf(", \"digests\": [");
  for (std::size_t i = 0; i < graph.lines.size(); ++i) {
    std::printf("%s\"%s\"", i ? ", " : "", graph.lines[i].c_str());
  }
  std::printf("], \"batch_vs_solo_differ\": %zu, \"batch_digests\": [",
              graph.batch_vs_solo_differ);
  for (std::size_t i = 0; i < graph.batch_lines.size(); ++i) {
    std::printf("%s\"%s\"", i ? ", " : "", graph.batch_lines[i].c_str());
  }
  std::printf("]}\n");
  return ok ? 0 : 1;
}
