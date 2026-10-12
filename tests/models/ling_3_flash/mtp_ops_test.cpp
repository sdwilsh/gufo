// Op test of the MTP draft block (block 42 + eh_proj input stage + shared-head
// norm + lm_head) on the GPU against the CPU reference (ReferenceModel::
// MtpStep, F32) and the golden numbers of mtp_reference_test. A draft step
// needs only (token, trunk hidden), so this runs on synthetic hiddens with a
// light model: the embedding, the lm_head, block 41 (the smallest resident
// trunk range) and block 42 are uploaded, about 3.5 GiB. It runs on any gfx11
// GPU (correctness only, timings are meaningless).
//
//   mtp_ops_test --model FIRST_SHARD.gguf [--golden FILE]
//
// Checks per step: finite logits; device argmax equals the host argmax of the
// logits; logits and h_next rel L2 vs the CPU reference below the tolerance of
// the decode numeric family (int8 two-plane Q8_0 activations, F16 lm_head);
// top-1 equals the reference whenever its lead exceeds twice the worst logit
// error; the draft attends over the rows written by earlier steps (a chain of 4
// steps reproduces the CPU chain). Exit 77 without --model or a HIP device.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/engine.hpp"
#include "src/models/ling_3_flash/reference.hpp"
#include "src/models/ling_3_flash/weights.hpp"
#include "tests/models/ling_3_flash/mtp_synthetic.hpp"

namespace l3 = gufo::models::ling_3_flash;
using gufo::tests::ling::kMtpSyntheticSteps;
using gufo::tests::ling::MtpSyntheticHidden;
using gufo::tests::ling::MtpSyntheticToken;

namespace {

int g_failures = 0;
void Expect(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
  if (!ok) {
    ++g_failures;
  }
}

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

std::int32_t HostArgmax(const std::vector<float>& v) {
  return static_cast<std::int32_t>(std::max_element(v.begin(), v.end()) -
                                   v.begin());
}

}  // namespace

int main(int argc, char** argv) {
  if (!gufo::core::RequireTempDirectory()) {
    return 1;
  }
  std::string model_path;
  std::string golden_path;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (a == "--golden" && i + 1 < argc) {
      golden_path = argv[++i];
    }
  }
  int devices = 0;
  if (model_path.empty() || hipGetDeviceCount(&devices) != hipSuccess ||
      devices == 0) {
    std::printf("SKIP: no --model or no HIP device\n");
    return 77;
  }

  // CPU reference first (host memory only).
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::printf("open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights || !weights->mtp) {
    std::printf("bind failed or no MTP block: %s\n", error.c_str());
    return weights ? 77 : 1;
  }
  l3::ReferenceModel::Options ropt;
  ropt.max_context = 64;
  ropt.threads = 4;
  ropt.check_mla = false;
  l3::ReferenceModel ref(*weights, ropt);
  struct Out {
    std::vector<float> logits, h_next;
  };
  std::vector<Out> cpu(kMtpSyntheticSteps);
  for (std::uint32_t step = 0; step < kMtpSyntheticSteps; ++step) {
    cpu[step].logits.resize(l3::kVocab);
    cpu[step].h_next.resize(l3::kHidden);
    if (!ref.MtpStep(MtpSyntheticToken(step),
                     MtpSyntheticHidden(step, l3::kHidden), cpu[step].logits,
                     cpu[step].h_next, nullptr, &error)) {
      std::printf("reference failed: %s\n", error.c_str());
      return 1;
    }
  }

  l3::ModelOptions options;
  options.max_batch = 16;
  options.max_context = 64;
  options.expanded_kv_bytes = 0;
  options.layer_first = 41;
  options.layer_limit = 1;
  options.mtp = true;
  auto model = l3::Model::Load(model_path, options, &error);
  if (!model) {
    std::printf("model load failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(64, &error);
  if (!session) {
    std::printf("session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::vector<float>> gpu_logits;
  for (std::uint32_t step = 0; step < kMtpSyntheticSteps; ++step) {
    std::vector<float> logits(l3::kVocab);
    std::vector<float> h_next(l3::kHidden);
    std::int32_t argmax = -1;
    if (!session->DraftWithHidden(MtpSyntheticToken(step),
                                  MtpSyntheticHidden(step, l3::kHidden), step,
                                  logits, h_next, &argmax, &error)) {
      std::printf("draft step %u failed: %s\n", step, error.c_str());
      return 1;
    }
    const std::string tag = "step " + std::to_string(step);
    const bool finite =
        std::ranges::all_of(logits, [](float v) { return std::isfinite(v); });
    Expect(finite, tag + ": logits are finite");
    Expect(argmax == HostArgmax(logits),
           tag + ": device argmax equals the host argmax (" +
               std::to_string(argmax) + ")");
    const double rel_logits = RelL2(logits, cpu[step].logits);
    const double rel_h = RelL2(h_next, cpu[step].h_next);
    double max_abs = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
      max_abs =
          std::max<double>(max_abs, std::fabs(logits[i] - cpu[step].logits[i]));
    }
    std::printf("       logits rel L2 %.3g, max abs %.3g, h_next rel L2 %.3g\n",
                rel_logits, max_abs, rel_h);
    Expect(rel_logits < 3e-3, tag + ": logits match the CPU reference");
    Expect(rel_h < 3e-3, tag + ": h_next matches the CPU reference");
    auto top = cpu[step].logits;
    std::partial_sort(top.begin(), top.begin() + 2, top.end(),
                      std::greater<float>());
    const double margin = static_cast<double>(top[0]) - top[1];
    Expect(HostArgmax(cpu[step].logits) == argmax || margin <= 2.0 * max_abs,
           tag + ": top-1 equals the reference (margin " +
               std::to_string(margin) + ")");
    gpu_logits.push_back(std::move(logits));
  }

  // Golden numbers (the same file the CPU test writes).
  if (!golden_path.empty()) {
    std::ifstream in(golden_path);
    std::string line;
    std::uint32_t seen = 0;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream ls(line);
      std::uint32_t step = 0;
      std::int32_t id[5];
      float val[5];
      ls >> step;
      for (auto& v : id)
        ls >> v;
      for (auto& v : val)
        ls >> v;
      if (step >= gpu_logits.size())
        continue;
      const auto& g = gpu_logits[step];
      bool ok = HostArgmax(g) == id[0];
      for (int k = 0; k < 5; ++k) {
        ok = ok &&
             std::fabs(g[id[k]] - val[k]) < 0.05 * (1.0 + std::fabs(val[k]));
      }
      Expect(ok, "step " + std::to_string(step) + ": golden top-5 ids/logits");
      ++seen;
    }
    Expect(seen == kMtpSyntheticSteps, "golden covers every step");
  }
  std::printf("%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILED",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
