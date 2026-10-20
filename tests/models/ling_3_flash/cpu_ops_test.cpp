// Synthetic checks of the scalar reference operators (no weights needed).
#include "src/models/ling_3_flash/cpu_ops.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace l3 = gufo::models::ling_3_flash;
namespace cpu = gufo::models::ling_3_flash::cpu;

namespace {

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

std::vector<float> Random(std::size_t n, std::uint32_t seed,
                          float scale = 1.0F) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0F, scale);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

l3::TensorRef F32Tensor(const std::vector<float>& v, std::uint64_t cols,
                        std::uint64_t rows, std::uint64_t depth) {
  l3::TensorRef t;
  t.data = v.data();
  t.type = gufo::core::GgmlType::kF32;
  t.cols = cols;
  t.rows = rows;
  t.depth = depth;
  return t;
}

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
  }
  return std::sqrt(num / den);
}

void TestL2Norm() {
  // Epsilon sits on the un-averaged sum: x / sqrt(sum(x^2) + eps).
  std::vector<float> x = {3.0F, 4.0F};
  cpu::L2Norm(x, 1.0F);
  const float s = 1.0F / std::sqrt(26.0F);
  Check(
      std::fabs(x[0] - 3.0F * s) < 1e-6F && std::fabs(x[1] - 4.0F * s) < 1e-6F,
      "L2Norm eps on the sum");
}

void TestSwiGlu() {
  std::vector<float> g = {10.0F, -1.0F, 2.0F};
  std::vector<float> u = {9.0F, 2.0F, -9.0F};
  std::vector<float> out(3);
  cpu::SwiGlu(g, u, 0.0F, out);
  Check(std::fabs(out[0] - cpu::Silu(10.0F) * 9.0F) < 1e-3F,
        "SwiGlu unclamped");
  cpu::SwiGlu(g, u, 4.0F, out);
  Check(std::fabs(out[0] - 4.0F * 4.0F) < 1e-5F, "SwiGlu clamps act and up");
  Check(std::fabs(out[1] - cpu::Silu(-1.0F) * 2.0F) < 1e-6F, "SwiGlu small");
  Check(std::fabs(out[2] - cpu::Silu(2.0F) * -4.0F) < 1e-5F,
        "SwiGlu clamps up");
}

void TestRope() {
  std::vector<float> x = Random(2 * 192, 1);
  const std::vector<float> orig = x;
  cpu::NormRope(x.data() + 128, 2, 192, 64, 0, 6e6F);
  Check(RelL2(x, orig) < 1e-7, "rope pos 0 is identity");
  cpu::NormRope(x.data() + 128, 2, 192, 64, 7, 6e6F);
  // Pair 0 rotates by pos * 1 radian; nope part untouched.
  const float c = std::cos(7.0F);
  const float s = std::sin(7.0F);
  Check(std::fabs(x[128] - (orig[128] * c - orig[129] * s)) < 1e-5F &&
            std::fabs(x[129] - (orig[128] * s + orig[129] * c)) < 1e-5F,
        "rope adjacent pair rotation");
  Check(x[0] == orig[0] && x[127] == orig[127], "rope leaves nope dims");
}

void TestKda() {
  // Decay 0 (g very negative), beta 1: S = k v^T, o = (q.k) v / sqrt(128).
  std::vector<float> state(l3::kKdaStateFloats, 5.0F);
  std::vector<float> q = Random(l3::kKdaInner, 2);
  std::vector<float> k = Random(l3::kKdaInner, 3);
  std::vector<float> v = Random(l3::kKdaInner, 4);
  std::vector<float> g(l3::kKdaInner, -1e4F);
  std::vector<float> beta(l3::kHeads, 1.0F);
  std::vector<float> o(l3::kKdaInner);
  cpu::KdaStep(state.data(), q.data(), k.data(), v.data(), g.data(),
               beta.data(), o.data());
  double err = 0.0;
  for (std::uint32_t h = 0; h < l3::kHeads; ++h) {
    float qk = 0.0F;
    for (int i = 0; i < 128; ++i) {
      qk += q[h * 128 + i] * k[h * 128 + i];
    }
    for (int j = 0; j < 128; ++j) {
      const float want = qk * v[h * 128 + j] / std::sqrt(128.0F);
      err =
          std::max(err, static_cast<double>(std::fabs(o[h * 128 + j] - want)));
    }
  }
  Check(err < 1e-4, "KDA rank-1 recurrence");
  // Key-axis decay: state row i is scaled by exp(g[i]) only.
  std::fill(state.begin(), state.end(), 0.0F);
  state[0 * 128 + 1] = 2.0F;  // head 0, i = 0, j = 1
  state[1 * 128 + 1] = 3.0F;  // head 0, i = 1, j = 1
  std::fill(g.begin(), g.end(), 0.0F);
  g[0] = std::log(0.5F);
  std::fill(k.begin(), k.end(), 0.0F);  // no write, only decay and read
  std::fill(q.begin(), q.end(), 0.0F);
  q[0] = 1.0F;
  q[1] = 10.0F;
  cpu::KdaStep(state.data(), q.data(), k.data(), v.data(), g.data(),
               beta.data(), o.data());
  const float want = (2.0F * 0.5F * 1.0F + 3.0F * 10.0F) / std::sqrt(128.0F);
  Check(std::fabs(o[1] - want) < 1e-5F, "KDA decays the key axis");
}

void TestConv() {
  const std::size_t c = 3;
  std::vector<float> w = {1, 2, 3, 4, 0, 0, 0, 1, 1, 1, 1, 1};  // [c][4]
  std::vector<float> hist = {1, 1, 1, 2, 2, 2, 3, 3, 3};        // [3][c]
  std::vector<float> raw = {10, 10, 10};
  std::vector<float> out(c);
  cpu::CausalConv4Silu(w.data(), hist.data(), raw.data(), out.data(), c);
  // channel 0: 1*1 + 2*2 + 3*3 + 4*10 = 54 (tap 3 meets the current token).
  Check(std::fabs(out[0] - cpu::Silu(54.0F)) < 1e-3F, "conv taps");
  Check(hist[0] == 2.0F && hist[6] == 10.0F, "conv history shift");
}

void TestRoute() {
  std::vector<float> logits(512, -4.0F);
  std::vector<float> bias(512, 0.0F);
  // Group 0 holds the two strongest raw scores; group 7 wins through bias.
  logits[0] = 5.0F;
  logits[1] = 4.0F;
  for (int i = 0; i < 64; ++i) {
    bias[7 * 64 + i] = 3.0F;
  }
  logits[7 * 64 + 5] = 0.0F;
  cpu::Routing r{};
  cpu::Route(logits, bias.data(), 8, 4, 8, 2.5F, 6.1035e-5F, r);
  bool first_is_biased = r.ids[0] / 64 == 7;
  Check(first_is_biased, "bias steers selection");
  float sum = 0.0F;
  for (float w : r.weights) {
    sum += w;
  }
  Check(std::fabs(sum - 2.5F) < 1e-4F, "weights normalised to the scale");
  cpu::Routing r2{};
  std::vector<float> probs(512);
  cpu::Route(logits, bias.data(), 8, 4, 8, 1.0F, 6.1035e-5F, r2, probs.data());
  const float p = 1.0F / (1.0F + std::exp(0.0F));
  Check(std::fabs(probs[7 * 64 + 5] - p) < 1e-6F, "probs are unbiased");
}

void TestMla() {
  for (std::size_t n : {1U, 5U, 33U}) {
    auto kb = Random(32 * 512 * 128, 10, 0.05F);  // [h][r][d]
    auto vb = Random(32 * 128 * 512, 11, 0.05F);  // [h][j][r]
    auto q = Random(32 * 192, 12);
    auto cache = Random(n * 576, 13);
    const auto kb_t = F32Tensor(kb, 128, 512, 32);
    const auto vb_t = F32Tensor(vb, 512, 128, 32);
    std::vector<float> a(32 * 128);
    cpu::MlaAttendAbsorbed(kb_t, vb_t, q.data(), cache.data(), n, a.data());

    std::vector<float> k_nope(n * 32 * 128);
    std::vector<float> v(n * 32 * 128);
    std::vector<float> k_pe(n * 64);
    for (std::size_t p = 0; p < n; ++p) {
      cpu::MlaExpand(kb.data(), vb.data(), cache.data() + p * 576,
                     k_nope.data() + p * 32 * 128, v.data() + p * 32 * 128);
      std::copy(cache.begin() + static_cast<std::ptrdiff_t>(p * 576 + 512),
                cache.begin() + static_cast<std::ptrdiff_t>((p + 1) * 576),
                k_pe.begin() + static_cast<std::ptrdiff_t>(p * 64));
    }
    std::vector<float> b(32 * 128);
    cpu::MlaAttendExpanded(q.data(), k_nope.data(), k_pe.data(), v.data(), n,
                           b.data());
    Check(RelL2(a, b) < 1e-5, "absorbed == non-absorbed MLA");
  }
}

// F64 mode against F32 mode on a small case: every operator must agree to
// ~1e-5 relative (they differ only by accumulation precision).
void TestF64Agreement() {
  const std::size_t n = 2560;
  const auto x = Random(n, 20);
  const auto w = Random(n, 21);
  const auto mat = Random(96 * n, 22, 0.05F);
  const auto mat_t = F32Tensor(mat, n, 96, 1);

  std::vector<float> y32(96);
  std::vector<float> y64(96);
  std::vector<float> norm32(n);
  std::vector<float> norm64(n);
  std::vector<float> l2_32 = x;
  std::vector<float> l2_64 = x;
  std::vector<float> sw32(n);
  std::vector<float> sw64(n);
  std::vector<float> rope32 = Random(2 * 192, 23);
  std::vector<float> rope64 = rope32;
  const auto gate = Random(n, 24, 3.0F);

  const auto run = [&](bool f64, std::vector<float>& y,
                       std::vector<float>& norm, std::vector<float>& l2,
                       std::vector<float>& sw, std::vector<float>& rope) {
    cpu::SetF64(f64);
    cpu::MatVec(mat_t, 0, x, y);
    cpu::RmsNorm(x, w.data(), 1e-6F, norm);
    cpu::L2Norm(l2, 1e-6F);
    cpu::SwiGlu(gate, x, 7.0F, sw);
    cpu::NormRope(rope.data() + 128, 2, 192, 64, 55, 6e6F);
  };
  run(false, y32, norm32, l2_32, sw32, rope32);
  run(true, y64, norm64, l2_64, sw64, rope64);
  Check(RelL2(y32, y64) < 1e-5, "F64 MatVec agrees with F32");
  Check(RelL2(norm32, norm64) < 1e-5, "F64 RmsNorm agrees with F32");
  Check(RelL2(l2_32, l2_64) < 1e-5, "F64 L2Norm agrees with F32");
  Check(RelL2(sw32, sw64) < 1e-5, "F64 SwiGlu agrees with F32");
  Check(RelL2(rope32, rope64) < 1e-5, "F64 rope agrees with F32");

  // The F64 MatVec must equal an exact double dot of the same weights.
  cpu::SetF64(true);
  double worst = 0.0;
  for (std::size_t r = 0; r < 96; ++r) {
    double ref = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
      ref += static_cast<double>(mat[r * n + c]) * x[c];
    }
    worst = std::max(worst, std::fabs(ref - y64[r]));
  }
  Check(worst < 1e-5, "F64 MatVec equals the double dot");

  // KDA recurrence: five tokens, float state vs double state.
  std::vector<float> state32(l3::kKdaStateFloats, 0.0F);
  std::vector<double> state64(l3::kKdaStateFloats, 0.0);
  std::vector<float> o32(l3::kKdaInner);
  std::vector<float> o64(l3::kKdaInner);
  for (std::uint32_t t = 0; t < 5; ++t) {
    auto q = Random(l3::kKdaInner, 30 + t);
    auto k = Random(l3::kKdaInner, 40 + t);
    const auto v = Random(l3::kKdaInner, 50 + t);
    auto g = Random(l3::kKdaInner, 60 + t, 0.5F);
    for (float& e : g) {
      e = -std::fabs(e);
    }
    const std::vector<float> beta(l3::kHeads, 0.7F);
    for (std::uint32_t h = 0; h < l3::kHeads; ++h) {
      cpu::SetF64(false);
      cpu::L2Norm({q.data() + h * 128, 128}, 1e-6F);
      cpu::L2Norm({k.data() + h * 128, 128}, 1e-6F);
    }
    cpu::SetF64(false);
    cpu::KdaStep(state32.data(), q.data(), k.data(), v.data(), g.data(),
                 beta.data(), o32.data());
    cpu::KdaStepF64(state64.data(), q.data(), k.data(), v.data(), g.data(),
                    beta.data(), o64.data());
  }
  Check(RelL2(o32, o64) < 1e-5, "F64 KDA recurrence agrees with F32");

  // MLA attention and router.
  const std::size_t tokens = 9;
  const auto kb = Random(32 * 512 * 128, 10, 0.05F);
  const auto vb = Random(32 * 128 * 512, 11, 0.05F);
  const auto q = Random(32 * 192, 12);
  const auto cache = Random(tokens * 576, 13);
  const auto kb_t = F32Tensor(kb, 128, 512, 32);
  const auto vb_t = F32Tensor(vb, 512, 128, 32);
  std::vector<float> m32(32 * 128);
  std::vector<float> m64(32 * 128);
  cpu::SetF64(false);
  cpu::MlaAttendAbsorbed(kb_t, vb_t, q.data(), cache.data(), tokens,
                         m32.data());
  cpu::MlaAttendAbsorbedF64(kb_t, vb_t, q.data(), cache.data(), tokens,
                            m64.data());
  Check(RelL2(m32, m64) < 1e-5, "F64 MLA attention agrees with F32");

  const auto lg = Random(256, 70, 2.0F);
  const auto bias = Random(256, 71, 0.1F);
  std::vector<double> lg_d(lg.begin(), lg.end());
  cpu::Routing r32{};
  cpu::Routing r64{};
  cpu::Route(lg, bias.data(), 8, 4, 8, 2.5F, 6.1035e-5F, r32);
  cpu::RouteF64(lg_d, bias.data(), 8, 4, 8, 2.5F, 6.1035e-5F, r64);
  bool same = true;
  float wdiff = 0.0F;
  for (int i = 0; i < 8; ++i) {
    same = same && r32.ids[i] == r64.ids[i];
    wdiff = std::max(wdiff, std::fabs(r32.weights[i] - r64.weights[i]));
  }
  Check(same && wdiff < 1e-5F, "F64 router agrees with F32");
  Check(r64.top_ids[8] != r64.top_ids[7] && r64.top_sel[7] >= r64.top_sel[8],
        "router reports the first rejected experts");
  cpu::SetF64(false);
}

}  // namespace

int main() {
  cpu::SetThreads(4);
  TestL2Norm();
  TestSwiGlu();
  TestRope();
  TestKda();
  TestConv();
  TestRoute();
  TestMla();
  TestF64Agreement();
  if (failures == 0) {
    std::printf("ling_3_flash cpu_ops: all checks passed\n");
  }
  return failures == 0 ? 0 : 1;
}
