// A Model's teardown returns its device memory, so a process can load the
// model again (decode_digest --graph-compare loads it twice). Each round loads
// the model, decodes on a first session and on a session created after it
// until both replay HIP graphs (with --mtp also a few speculative steps on the
// later one), drops the later session right after its last step (no device
// synchronisation in between, which is when ROCm kept the weights referenced),
// then destroys everything. The device's free memory must come back to within
// --slack-gib of the free memory before the first load, and the next round's
// load must succeed. Prints one JSON document on stdout. Exit 77 without
// --model.
//
//   reload_test --model FIRST_SHARD.gguf [--layers N] [--mtp] [--rounds N]
//               [--slack-gib G]

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/engine.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace sampling = gufo::sampling;

namespace {

double FreeGiB() {
  (void)hipDeviceSynchronize();
  std::size_t free = 0;
  std::size_t total = 0;
  if (hipMemGetInfo(&free, &total) != hipSuccess) {
    return -1.0;
  }
  return static_cast<double>(free) / static_cast<double>(1ULL << 30);
}

std::int32_t Tok(std::uint32_t i) {
  return static_cast<std::int32_t>((i * 2654435761U + 12345U) % 150000U);
}

/// Feeds `prompt` tokens, then decodes `steps` tokens one at a time (the
/// first step of a size runs eagerly, the second captures its graph, later
/// ones replay it).
bool Decode(l3::Session& s, std::uint32_t prompt, std::uint32_t steps,
            std::uint32_t seed, std::string* error) {
  std::vector<std::int32_t> toks;
  for (std::uint32_t i = 0; i < prompt; ++i) {
    toks.push_back(Tok(seed + i));
  }
  if (!s.Feed(toks, {}, error)) {
    return false;
  }
  for (std::uint32_t i = 0; i < steps; ++i) {
    if (!s.Evaluate(Tok(seed + prompt + i), error)) {
      return false;
    }
  }
  return true;
}

struct Round {
  bool ok{false};
  std::string error;
  double loaded_gib{0.0};
  double after_gib{0.0};
  std::uint64_t graph_replays{0};
};

Round RunRound(const std::string& path, const l3::ModelOptions& options) {
  Round r;
  auto model = l3::Model::Load(path, options, &r.error);
  if (!model) {
    r.error = "load: " + r.error;
    return r;
  }
  r.loaded_gib = FreeGiB();
  auto first = model->CreateSession(options.max_context, &r.error);
  if (!first || !Decode(*first, 40, 4, 0, &r.error)) {
    r.error = "first session: " + r.error;
    return r;
  }
  {
    auto later = model->CreateSession(options.max_context, &r.error);
    if (!later || !Decode(*later, 20, 4, 500, &r.error)) {
      r.error = "later session: " + r.error;
      return r;
    }
    if (model->HasMtp()) {
      sampling::SamplingConfig greedy;
      greedy.temperature = 0.0F;
      sampling::SamplerState sampler(greedy);
      for (int i = 0; i < 4; ++i) {
        l3::Session::DecodeResult step;
        if (!later->DecodeStep(4, sampler, &step, &r.error, false)) {
          r.error = "speculative step: " + r.error;
          return r;
        }
      }
    }
    // Dropped here, right after its last step.
  }
  r.graph_replays = model->GraphReplays();
  if (r.graph_replays == 0) {
    r.error = "no HIP graph replayed (the check needs graph decode)";
    return r;
  }
  first.reset();
  model.reset();
  r.after_gib = FreeGiB();
  r.ok = true;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  // Before the first HIP call: a missing $TMPDIR crashes the runtime.
  if (!gufo::core::RequireTempDirectory()) {
    return 1;
  }
  std::string path;
  l3::ModelOptions options;
  options.max_context = 512;
  options.max_batch = 128;
  std::uint32_t rounds = 2;
  double slack = 1.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : "";
    };
    if (a == "--model") {
      path = next();
    } else if (a == "--layers") {
      options.layer_limit = std::stoul(next());
    } else if (a == "--mtp") {
      options.mtp = true;
    } else if (a == "--rounds") {
      rounds = std::stoul(next());
    } else if (a == "--slack-gib") {
      slack = std::stod(next());
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, "no --model: skipping\n");
    return 77;
  }
  const double before = FreeGiB();
  bool ok = before >= 0.0 && rounds >= 2;
  std::string doc;
  for (std::uint32_t i = 0; i < rounds; ++i) {
    const Round r = RunRound(path, options);
    const double lost = before - r.after_gib;
    const bool pass = r.ok && lost <= slack;
    std::fprintf(
        stderr,
        "[check] round %u: %s (free %.2f GiB before the first load, "
        "%.2f loaded, %.2f after teardown; %.2f GiB not returned)%s%s\n",
        i, pass ? "PASS" : "FAIL", before, r.loaded_gib, r.after_gib,
        r.ok ? lost : 0.0, r.error.empty() ? "" : ": ", r.error.c_str());
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "%s{\"round\": %u, \"pass\": %s, \"loaded_free_gib\": %.3f, "
                  "\"after_free_gib\": %.3f, \"graph_replays\": %llu}",
                  i ? ", " : "", i, pass ? "true" : "false", r.loaded_gib,
                  r.after_gib,
                  static_cast<unsigned long long>(r.graph_replays));
    doc += buf;
    ok = ok && pass;
    if (!r.ok) {
      break;
    }
  }
  std::printf(
      "{\"test\": \"reload_test\", \"pass\": %s, \"free_before_gib\": %.3f, "
      "\"slack_gib\": %.3f, \"rounds\": [%s]}\n",
      ok ? "true" : "false", before, slack, doc.c_str());
  return ok ? 0 : 1;
}
