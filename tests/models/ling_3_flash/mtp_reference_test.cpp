// CPU reference of the MTP draft step (ReferenceModel::MtpStep, mtp.hpp) on the
// real block-42 weights. The draft step needs only (token, trunk hidden), so it
// runs on synthetic hiddens (mtp_synthetic.hpp) without the 42-block trunk.
//
//   ling_3_flash_mtp_reference_test --model FIRST_SHARD.gguf
//       [--golden FILE] [--update] [--threads N]
//
// Checks: finite outputs and the shared-head norm structure; an F64 reference
// (every dot product, norm and softmax accumulated in double) agrees with the
// F32 one; rewinding the MTP cache and recomputing a step reproduces it
// exactly; a chain equals single steps fed the previous h_next; the results
// match a golden file (written with --update) so the GPU op test and later
// refactors compare against fixed numbers. Exit 77 without --model.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
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

struct StepOut {
  std::vector<float> logits;
  std::vector<float> h_next;
};

struct Golden {
  std::int32_t top_id[5];
  float top_logit[5];
  double h_l2;
  double h_sum;
};

Golden Summarize(const StepOut& o) {
  Golden g{};
  std::vector<std::int32_t> order(o.logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                    [&](std::int32_t a, std::int32_t b) {
                      return o.logits[a] > o.logits[b] ||
                             (o.logits[a] == o.logits[b] && a < b);
                    });
  for (int i = 0; i < 5; ++i) {
    g.top_id[i] = order[i];
    g.top_logit[i] = o.logits[order[i]];
  }
  for (float v : o.h_next) {
    g.h_l2 += static_cast<double>(v) * v;
    g.h_sum += v;
  }
  g.h_l2 = std::sqrt(g.h_l2);
  return g;
}

bool RunChain(l3::ReferenceModel& ref, std::vector<StepOut>* out) {
  std::string error;
  out->clear();
  for (std::uint32_t step = 0; step < kMtpSyntheticSteps; ++step) {
    StepOut o;
    o.logits.resize(l3::kVocab);
    o.h_next.resize(l3::kHidden);
    const auto h = MtpSyntheticHidden(step, l3::kHidden);
    if (!ref.MtpStep(MtpSyntheticToken(step), h, o.logits, o.h_next, nullptr,
                     &error)) {
      std::printf("MtpStep failed: %s\n", error.c_str());
      return false;
    }
    out->push_back(std::move(o));
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  std::string golden_path;
  bool update = false;
  unsigned threads = 4;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else if (a == "--golden" && i + 1 < argc) {
      golden_path = argv[++i];
    } else if (a == "--update") {
      update = true;
    } else if (a == "--threads" && i + 1 < argc) {
      threads = static_cast<unsigned>(std::atoi(argv[++i]));
    }
  }
  if (model.empty()) {
    std::printf("no --model: skip\n");
    return 77;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!reader) {
    std::printf("open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights || !weights->mtp) {
    std::printf("bind failed or no MTP block: %s\n", error.c_str());
    return weights ? 77 : 1;
  }

  l3::ReferenceModel::Options opt;
  opt.max_context = 64;
  opt.threads = threads;
  opt.check_mla = false;
  l3::ReferenceModel f32(*weights, opt);
  std::vector<StepOut> a;
  if (!RunChain(f32, &a)) {
    return 1;
  }
  Expect(f32.MtpPosition() == kMtpSyntheticSteps,
         "the MTP cache holds one row per step");
  bool finite = true;
  for (const auto& o : a) {
    finite =
        finite &&
        std::ranges::all_of(o.logits,
                            [](float v) { return std::isfinite(v); }) &&
        std::ranges::all_of(o.h_next, [](float v) { return std::isfinite(v); });
  }
  Expect(finite, "logits and h_next are finite");

  // h_next is rmsnorm(x; layer_output_norm): dividing by the norm weights must
  // give a unit-RMS row. Weights are F32 [2560] in the file.
  {
    const auto* gamma =
        static_cast<const float*>(weights->mtp->layer_output_norm.data);
    double ss = 0.0;
    for (std::uint32_t i = 0; i < l3::kHidden; ++i) {
      const double v = a.back().h_next[i] / gamma[i];
      ss += v * v;
    }
    const double rms = std::sqrt(ss / l3::kHidden);
    Expect(std::fabs(rms - 1.0) < 2e-3,
           "h_next / layer_output_norm has unit RMS (" + std::to_string(rms) +
               ")");
  }

  // Rewind and recompute step 2: exact.
  {
    f32.MtpRewind(2);
    StepOut again;
    again.logits.resize(l3::kVocab);
    again.h_next.resize(l3::kHidden);
    const auto h = MtpSyntheticHidden(2, l3::kHidden);
    (void)f32.MtpStep(MtpSyntheticToken(2), h, again.logits, again.h_next);
    Expect(again.logits == a[2].logits && again.h_next == a[2].h_next,
           "rewind + recompute reproduces a step exactly");
    Expect(f32.MtpPosition() == 3, "position follows the rewind");
  }

  // Chain: step 1 fed the previous step's h_next equals a fresh model doing
  // the same (the state that matters is the cache and the given h).
  {
    l3::ReferenceModel chain(*weights, opt);
    StepOut s0;
    s0.logits.resize(l3::kVocab);
    s0.h_next.resize(l3::kHidden);
    (void)chain.MtpStep(MtpSyntheticToken(0),
                        MtpSyntheticHidden(0, l3::kHidden), s0.logits,
                        s0.h_next);
    StepOut s1;
    s1.logits.resize(l3::kVocab);
    s1.h_next.resize(l3::kHidden);
    (void)chain.MtpStep(MtpSyntheticToken(1), s0.h_next, s1.logits, s1.h_next);
    l3::ReferenceModel other(*weights, opt);
    (void)other.MtpStep(MtpSyntheticToken(0),
                        MtpSyntheticHidden(0, l3::kHidden), {}, {});
    StepOut t1;
    t1.logits.resize(l3::kVocab);
    t1.h_next.resize(l3::kHidden);
    (void)other.MtpStep(MtpSyntheticToken(1), s0.h_next, t1.logits, t1.h_next);
    Expect(s1.logits == t1.logits,
           "a chain step depends only on the cache rows and its inputs");
    // The attention of step 1 must actually see step 0's row.
    StepOut lone;
    lone.logits.resize(l3::kVocab);
    lone.h_next.resize(l3::kHidden);
    l3::ReferenceModel fresh(*weights, opt);
    (void)fresh.MtpStep(MtpSyntheticToken(1), s0.h_next, lone.logits,
                        lone.h_next);
    Expect(RelL2(lone.logits, s1.logits) > 1e-4,
           "the draft attends to earlier MTP cache rows");
  }

  // F64 reference agrees with F32.
  {
    l3::ReferenceModel::Options o64 = opt;
    o64.f64 = true;
    l3::ReferenceModel f64(*weights, o64);
    std::vector<StepOut> b;
    if (!RunChain(f64, &b)) {
      return 1;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      worst = std::max({worst, RelL2(a[i].logits, b[i].logits),
                        RelL2(a[i].h_next, b[i].h_next)});
    }
    Expect(worst < 2e-3, "F32 and F64 references agree (worst rel L2 " +
                             std::to_string(worst) + ")");
  }

  // Golden numbers.
  if (!golden_path.empty()) {
    if (update) {
      std::ofstream out(golden_path);
      out.precision(9);
      out << "# MTP reference golden (F32): step top5 ids, top5 logits, "
             "|h_next|, sum(h_next)\n";
      for (std::uint32_t i = 0; i < a.size(); ++i) {
        const Golden g = Summarize(a[i]);
        out << i;
        for (int k = 0; k < 5; ++k)
          out << ' ' << g.top_id[k];
        for (int k = 0; k < 5; ++k)
          out << ' ' << g.top_logit[k];
        out << ' ' << g.h_l2 << ' ' << g.h_sum << '\n';
      }
      std::printf("wrote %s\n", golden_path.c_str());
    } else {
      std::ifstream in(golden_path);
      std::string line;
      std::uint32_t seen = 0;
      while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
          continue;
        std::istringstream ls(line);
        std::uint32_t step = 0;
        Golden g{};
        ls >> step;
        for (int k = 0; k < 5; ++k)
          ls >> g.top_id[k];
        for (int k = 0; k < 5; ++k)
          ls >> g.top_logit[k];
        ls >> g.h_l2 >> g.h_sum;
        if (step >= a.size())
          continue;
        const Golden now = Summarize(a[step]);
        bool same = true;
        for (int k = 0; k < 5; ++k) {
          same = same && g.top_id[k] == now.top_id[k] &&
                 std::fabs(g.top_logit[k] - now.top_logit[k]) <
                     1e-3 * (1.0 + std::fabs(g.top_logit[k]));
        }
        same =
            same && std::fabs(g.h_l2 - now.h_l2) < 1e-3 * g.h_l2 &&
            std::fabs(g.h_sum - now.h_sum) < 1e-3 * (1.0 + std::fabs(g.h_sum));
        Expect(same, "step " + std::to_string(step) + " matches the golden");
        ++seen;
      }
      Expect(seen == kMtpSyntheticSteps, "golden covers every step");
    }
  }
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
