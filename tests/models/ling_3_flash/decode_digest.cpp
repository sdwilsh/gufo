// Decode-path digest of Ling-3.0-flash: runs a scripted prefill, token-by-token
// decode, small-batch chunks (2..8 tokens), a snapshot rollback and two
// interleaved sessions, and prints one FNV-1a digest per step (logits bits and
// device argmax). With `--graph-compare` the script runs twice, with HIP-graph
// decode on and off (ModelOptions::graphs), and the digests must be identical;
// the printed digests also let two builds be compared bit for bit. Exit 77
// without --model.
//
//   decode_digest --model FIRST_SHARD.gguf [--layers N] [--max-context N]
//                 [--prompt-tokens P] [--decode-tokens K] [--graph-compare]
//                 [--max-batch N] [--expanded-kv-bytes N]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/engine.hpp"

namespace l3 = gufo::models::ling_3_flash;

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
  bool ok{true};
  std::uint64_t graph_replays{0};  ///< HIP graph replays of the whole script
};

std::int32_t Tok(std::uint32_t i) {
  return static_cast<std::int32_t>((i * 2654435761U + 12345U) % 150000U);
}

void Digest(Script* out, const char* what, std::uint32_t step,
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
  out->lines.emplace_back(buf);
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
         graph.graph_replays > 0;
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
  std::printf("]}\n");
  return ok ? 0 : 1;
}
