// Op tests of the Ling-3.0-flash KDA kernels against scalar host references
// (double accumulation, straight from PLAN.md section 1). Random inputs with
// fixed seeds. Exit 77 when no GPU is present.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"

namespace k = gufo::models::ling_3_flash::rocm::kda;
namespace l = gufo::models::ling_3_flash;
namespace {

constexpr std::uint32_t kD = l::kKdaHeadDim;
constexpr std::uint32_t kInner = l::kKdaInner;
constexpr std::uint32_t kH = l::kHeads;
constexpr std::uint32_t kProj = l::kKdaProjRows;
constexpr std::uint32_t kCh = 3 * kInner;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

template<typename T>
class Dev {
public:
  explicit Dev(std::size_t n) : n_(n) {
    Check(hipMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)), "hipMalloc");
  }
  explicit Dev(const std::vector<T>& v) : Dev(v.size()) { Put(v); }
  ~Dev() { (void)hipFree(p_); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  T* get() const { return p_; }
  void Put(const std::vector<T>& v) {
    Check(hipMemcpy(p_, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
          "H2D");
  }
  std::vector<T> Get() const {
    std::vector<T> v(n_);
    Check(hipMemcpy(v.data(), p_, n_ * sizeof(T), hipMemcpyDeviceToHost),
          "D2H");
    return v;
  }

private:
  T* p_{nullptr};
  std::size_t n_;
};

std::vector<float> Randn(std::mt19937& rng, std::size_t n, float scale,
                         float mean = 0.0F) {
  std::normal_distribution<float> d(mean, scale);
  std::vector<float> v(n);
  for (auto& x : v) {
    x = d(rng);
  }
  return v;
}

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}
double Silu(double x) {
  return x * Sigmoid(x);
}

struct Err {
  double abs{0};
  double ref_max{0};
  [[nodiscard]] double rel() const { return abs / std::max(ref_max, 1e-30); }
};
Err Compare(const float* got, const float* want, std::size_t n) {
  Err e;
  for (std::size_t i = 0; i < n; ++i) {
    e.abs = std::max(e.abs, std::fabs(static_cast<double>(got[i]) - want[i]));
    e.ref_max = std::max(e.ref_max, std::fabs(static_cast<double>(want[i])));
  }
  return e;
}
Err Compare(const std::vector<float>& a, const std::vector<float>& b) {
  return Compare(a.data(), b.data(), std::min(a.size(), b.size()));
}

int g_failures = 0;
void Report(const char* name, const Err& e, double rel_tol) {
  const bool ok = std::isfinite(e.abs) && e.rel() <= rel_tol;
  std::printf("%-44s %s  max_abs=%.3e  max_rel=%.3e (ref_max=%.3g, tol=%.0e)\n",
              name, ok ? "PASS" : "FAIL", e.abs, e.rel(), e.ref_max, rel_tol);
  if (!ok) {
    ++g_failures;
  }
}

// ---- host references ------------------------------------------------------

// proj [n][kProj]; w[3][4096*4]; state [3][3][4096] (raw history). Returns
// out [n][3*4096] and advances state.
std::vector<float> HostConv(const std::vector<float>& proj,
                            const std::vector<float> w[3],
                            std::vector<float>& state, std::uint32_t n) {
  std::vector<float> out(static_cast<std::size_t>(n) * kCh);
  std::vector<float> hist = state;  // [s][pos][c] oldest first
  for (std::uint32_t t = 0; t < n; ++t) {
    for (std::uint32_t s = 0; s < 3; ++s) {
      for (std::uint32_t c = 0; c < kInner; ++c) {
        const float cur =
            proj[static_cast<std::size_t>(t) * kProj + s * kInner + c];
        float* h = &hist[(static_cast<std::size_t>(s) * 3) * kInner + c];
        const double y = static_cast<double>(w[s][c * 4 + 0]) * h[0] +
                         static_cast<double>(w[s][c * 4 + 1]) * h[kInner] +
                         static_cast<double>(w[s][c * 4 + 2]) * h[2 * kInner] +
                         static_cast<double>(w[s][c * 4 + 3]) * cur;
        out[static_cast<std::size_t>(t) * kCh + s * kInner + c] =
            static_cast<float>(Silu(y));
        h[0] = h[kInner];
        h[kInner] = h[2 * kInner];
        h[2 * kInner] = cur;
      }
    }
  }
  state = hist;
  return out;
}

// state [h][i][j]; q,k,v,g rows with strides; writes o [n][4096].
void HostRecurrent(std::vector<float>& state, const float* q, std::size_t ld_q,
                   const float* kk, std::size_t ld_k, const float* v,
                   std::size_t ld_v, const float* g, const float* beta,
                   std::vector<float>& o, std::uint32_t n) {
  std::vector<double> S(state.begin(), state.end());
  o.assign(static_cast<std::size_t>(n) * kInner, 0.0F);
  std::vector<double> delta(kD);
  for (std::uint32_t t = 0; t < n; ++t) {
    for (std::uint32_t h = 0; h < kH; ++h) {
      double* s = &S[static_cast<std::size_t>(h) * kD * kD];
      const float* qt = q + t * ld_q + h * kD;
      const float* kt = kk + t * ld_k + h * kD;
      const float* vt = v + t * ld_v + h * kD;
      const float* gt = g + static_cast<std::size_t>(t) * kInner + h * kD;
      const double b = beta[static_cast<std::size_t>(t) * kH + h];
      for (std::uint32_t i = 0; i < kD; ++i) {
        const double e = std::exp(static_cast<double>(gt[i]));
        for (std::uint32_t j = 0; j < kD; ++j) {
          s[i * kD + j] *= e;
        }
      }
      for (std::uint32_t j = 0; j < kD; ++j) {
        double u = 0;
        for (std::uint32_t i = 0; i < kD; ++i) {
          u += s[i * kD + j] * kt[i];
        }
        delta[j] = (vt[j] - u) * b;
      }
      for (std::uint32_t i = 0; i < kD; ++i) {
        for (std::uint32_t j = 0; j < kD; ++j) {
          s[i * kD + j] += static_cast<double>(kt[i]) * delta[j];
        }
      }
      for (std::uint32_t j = 0; j < kD; ++j) {
        double acc = 0;
        for (std::uint32_t i = 0; i < kD; ++i) {
          acc += s[i * kD + j] * qt[i];
        }
        o[static_cast<std::size_t>(t) * kInner + h * kD + j] =
            static_cast<float>(acc / std::sqrt(128.0));
      }
    }
  }
  for (std::size_t i = 0; i < state.size(); ++i) {
    state[i] = static_cast<float>(S[i]);
  }
}

void HostL2(std::vector<float>& x, std::uint32_t ld, std::uint32_t rows) {
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t h = 0; h < kH; ++h) {
      float* p = &x[static_cast<std::size_t>(r) * ld + h * kD];
      double ss = 0;
      for (std::uint32_t i = 0; i < kD; ++i) {
        ss += static_cast<double>(p[i]) * p[i];
      }
      const double inv = 1.0 / std::sqrt(ss + 1e-6);
      for (std::uint32_t i = 0; i < kD; ++i) {
        p[i] = static_cast<float>(p[i] * inv);
      }
    }
  }
}

// ---- input generators -----------------------------------------------------

// Unit-ish q/k rows [n][ld] (head-normalised random), v ~ N(0, 0.7).
struct Inputs {
  std::vector<float> qkv;   // [n][3*4096], q|k|v, q and k L2-normalised
  std::vector<float> g;     // [n][4096]
  std::vector<float> beta;  // [n][32]
};

Inputs MakeInputs(std::mt19937& rng, std::uint32_t n) {
  Inputs in;
  in.qkv = Randn(rng, static_cast<std::size_t>(n) * kCh, 1.0F);
  for (std::size_t i = 0; i < in.qkv.size(); ++i) {
    if (i % kCh >= 2 * kInner) {
      in.qkv[i] *= 0.7F;
    }
  }
  // Normalise q and k heads in place (strided rows).
  for (std::uint32_t t = 0; t < n; ++t) {
    for (std::uint32_t s = 0; s < 2; ++s) {
      for (std::uint32_t h = 0; h < kH; ++h) {
        float* p =
            &in.qkv[static_cast<std::size_t>(t) * kCh + s * kInner + h * kD];
        double ss = 0;
        for (std::uint32_t i = 0; i < kD; ++i) {
          ss += static_cast<double>(p[i]) * p[i];
        }
        for (std::uint32_t i = 0; i < kD; ++i) {
          p[i] = static_cast<float>(p[i] / std::sqrt(ss + 1e-6));
        }
      }
    }
  }
  in.g.resize(static_cast<std::size_t>(n) * kInner);
  std::normal_distribution<float> nd(0.0F, 2.0F);
  std::uniform_real_distribution<float> ud(0.0F, 1.0F);
  for (std::size_t i = 0; i < in.g.size(); ++i) {
    const std::uint32_t h = (i % kInner) / kD;
    if (h % 4 == 0) {
      in.g[i] = -5.0F + 0.5F * ud(rng) * 0.2F;  // near -5
    } else if (h % 4 == 1) {
      in.g[i] = -0.01F * ud(rng);  // near 0
    } else {
      in.g[i] = -5.0F * static_cast<float>(Sigmoid(nd(rng)));
    }
  }
  in.beta.resize(static_cast<std::size_t>(n) * kH);
  for (auto& b : in.beta) {
    b = static_cast<float>(Sigmoid(nd(rng) * 0.5F));
  }
  return in;
}

// ---- tests ----------------------------------------------------------------

void TestConv() {
  std::mt19937 rng(1234);
  const auto wq = Randn(rng, kInner * 4, 0.5F);
  const auto wk = Randn(rng, kInner * 4, 0.5F);
  const auto wv = Randn(rng, kInner * 4, 0.5F);
  const std::vector<float> w[3] = {wq, wk, wv};
  Dev<float> dwq(wq);
  Dev<float> dwk(wk);
  Dev<float> dwv(wv);
  const auto init_state = Randn(rng, l::kKdaConvStateFloats, 1.0F);

  auto run_prefill = [&](const std::vector<float>& proj, std::uint32_t n,
                         std::vector<float>& state_io) {
    Dev<float> dp(proj);
    Dev<float> ds(state_io);
    Dev<float> dout(static_cast<std::size_t>(n) * kCh);
    k::Conv3Prefill(dp.get(), dwq.get(), dwk.get(), dwv.get(), ds.get(),
                    dout.get(), n, nullptr);
    Check(hipDeviceSynchronize(), "sync");
    state_io = ds.Get();
    return dout.Get();
  };

  for (const std::uint32_t n : {1U, 2U, 3U, 37U, 200U}) {
    const auto proj = Randn(rng, static_cast<std::size_t>(n) * kProj, 1.0F);
    std::vector<float> hs = init_state;
    const auto want = HostConv(proj, w, hs, n);
    std::vector<float> gs = init_state;
    const auto got = run_prefill(proj, n, gs);
    const std::string nm = "conv3 prefill n=" + std::to_string(n);
    Report((nm + " out").c_str(), Compare(got, want), 1e-5);
    Report((nm + " state").c_str(), Compare(gs, hs), 1e-6);
  }

  // Prefill n tokens == n decode steps; split prefill == one prefill.
  const std::uint32_t n = 40;
  const auto proj = Randn(rng, static_cast<std::size_t>(n) * kProj, 1.0F);
  std::vector<float> s_all = init_state;
  const auto out_all = run_prefill(proj, n, s_all);

  std::vector<float> s_dec = init_state;
  std::vector<float> out_dec;
  {
    Dev<float> dp(proj);
    Dev<float> ds(s_dec);
    Dev<float> dout(kCh);
    for (std::uint32_t t = 0; t < n; ++t) {
      k::Conv3Decode(dp.get() + static_cast<std::size_t>(t) * kProj, dwq.get(),
                     dwk.get(), dwv.get(), ds.get(), dout.get(), nullptr);
      Check(hipDeviceSynchronize(), "sync");
      const auto o = dout.Get();
      out_dec.insert(out_dec.end(), o.begin(), o.end());
    }
    s_dec = ds.Get();
  }
  Report("conv3 prefill(40) == 40 decode steps out", Compare(out_dec, out_all),
         0.0);
  Report("conv3 prefill(40) == 40 decode steps state", Compare(s_dec, s_all),
         0.0);

  for (const std::uint32_t a : {1U, 2U, 5U, 17U}) {
    std::vector<float> s_split = init_state;
    const std::vector<float> pa(
        proj.begin(), proj.begin() + static_cast<std::size_t>(a) * kProj);
    const std::vector<float> pb(
        proj.begin() + static_cast<std::size_t>(a) * kProj, proj.end());
    auto oa = run_prefill(pa, a, s_split);
    const auto ob = run_prefill(pb, n - a, s_split);
    oa.insert(oa.end(), ob.begin(), ob.end());
    const std::string nm = "conv3 split " + std::to_string(a) + "+" +
                           std::to_string(n - a) + " == whole";
    Report((nm + " out").c_str(), Compare(oa, out_all), 0.0);
    Report((nm + " state").c_str(), Compare(s_split, s_all), 0.0);
  }
}

void TestPrep() {
  std::mt19937 rng(77);
  const std::uint32_t n = 9;
  const auto proj = Randn(rng, static_cast<std::size_t>(n) * kProj, 1.5F);
  std::vector<float> ssm_a(kH);
  std::uniform_real_distribution<float> ud(0.05F, 4.0F);  // exp(A_log) > 0
  for (auto& a : ssm_a) {
    a = ud(rng);
  }
  const auto dt = Randn(rng, kInner, 1.0F);
  Dev<float> dp(proj);
  Dev<float> da(ssm_a);
  Dev<float> ddt(dt);
  Dev<float> dg(static_cast<std::size_t>(n) * kInner);
  Dev<float> db(static_cast<std::size_t>(n) * kH);
  k::GateBetaPrep(dp.get(), da.get(), ddt.get(), dg.get(), db.get(), n,
                  nullptr);
  Check(hipDeviceSynchronize(), "sync");
  std::vector<float> wg(static_cast<std::size_t>(n) * kInner);
  std::vector<float> wb(static_cast<std::size_t>(n) * kH);
  for (std::uint32_t t = 0; t < n; ++t) {
    for (std::uint32_t c = 0; c < kInner; ++c) {
      const double f =
          proj[static_cast<std::size_t>(t) * kProj + l::kKdaFaRow + c];
      wg[static_cast<std::size_t>(t) * kInner + c] = static_cast<float>(
          -5.0 * Sigmoid(static_cast<double>(ssm_a[c / kD]) * (f + dt[c])));
    }
    for (std::uint32_t h = 0; h < kH; ++h) {
      wb[static_cast<std::size_t>(t) * kH + h] = static_cast<float>(Sigmoid(
          proj[static_cast<std::size_t>(t) * kProj + l::kKdaBetaRow + h]));
    }
  }
  Report("gate_beta_prep g", Compare(dg.Get(), wg), 1e-6);
  Report("gate_beta_prep beta", Compare(db.Get(), wb), 1e-6);

  // L2 norm in place with row stride 3*4096 (q portion only).
  const std::uint32_t rows = 5;
  auto x = Randn(rng, static_cast<std::size_t>(rows) * kCh, 3.0F);
  auto want = x;
  HostL2(want, kCh, rows);
  Dev<float> dx(x);
  k::L2NormHeads(dx.get(), kCh, rows, nullptr);
  Check(hipDeviceSynchronize(), "sync");
  const auto got = dx.Get();
  // Rows are processed only over the first 4096 channels (one q or k block).
  Err e;
  Err untouched;
  for (std::uint32_t r = 0; r < rows; ++r) {
    const auto base = static_cast<std::size_t>(r) * kCh;
    const Err a = Compare(&got[base], &want[base], kInner);
    e.abs = std::max(e.abs, a.abs);
    e.ref_max = std::max(e.ref_max, a.ref_max);
    const Err u = Compare(&got[base + kInner], &x[base + kInner], kCh - kInner);
    untouched.abs = std::max(untouched.abs, u.abs);
    untouched.ref_max = 1;
  }
  Report("l2norm_heads (ld=12288)", e, 1e-6);
  Report("l2norm_heads leaves other columns", untouched, 0.0);
}

void TestRecurrent() {
  std::mt19937 rng(4242);
  const std::size_t sf = l::kKdaStateFloats;
  // State magnitudes from earlier tokens are O(1) per entry at most.
  const auto s0 = Randn(rng, sf, 0.3F);
  const std::uint32_t kMax = 300;
  const auto in = MakeInputs(rng, kMax);
  const float* q = in.qkv.data();
  const float* kk = in.qkv.data() + kInner;
  const float* v = in.qkv.data() + 2 * kInner;

  // 1 token and 64 sequential decode tokens.
  for (const std::uint32_t n : {1U, 64U}) {
    std::vector<float> hs = s0;
    std::vector<float> ho;
    HostRecurrent(hs, q, kCh, kk, kCh, v, kCh, in.g.data(), in.beta.data(), ho,
                  n);
    Dev<float> ds(s0);
    Dev<float> dq(kInner);
    Dev<float> dk(kInner);
    Dev<float> dv(kInner);
    Dev<float> dg(kInner);
    Dev<float> db(kH);
    Dev<float> dout(kInner);
    Err oe;
    for (std::uint32_t t = 0; t < n; ++t) {
      auto row = [&](const float* base) {
        return std::vector<float>(
            base + static_cast<std::size_t>(t) * kCh,
            base + static_cast<std::size_t>(t) * kCh + kInner);
      };
      dq.Put(row(q));
      dk.Put(row(kk));
      dv.Put(row(v));
      dg.Put(std::vector<float>(
          in.g.begin() + static_cast<std::size_t>(t) * kInner,
          in.g.begin() + static_cast<std::size_t>(t + 1) * kInner));
      db.Put(std::vector<float>(
          in.beta.begin() + static_cast<std::size_t>(t) * kH,
          in.beta.begin() + static_cast<std::size_t>(t + 1) * kH));
      k::RecurrentDecode(ds.get(), dq.get(), dk.get(), dv.get(), dg.get(),
                         db.get(), dout.get(), nullptr);
      Check(hipDeviceSynchronize(), "sync");
      const auto o = dout.Get();
      const Err e =
          Compare(o.data(), &ho[static_cast<std::size_t>(t) * kInner], kInner);
      oe.abs = std::max(oe.abs, e.abs);
      oe.ref_max = std::max(oe.ref_max, e.ref_max);
    }
    const std::string nm = "recurrent decode x" + std::to_string(n);
    Report((nm + " o (worst step)").c_str(), oe, 1e-4);
    Report((nm + " state").c_str(), Compare(ds.Get(), hs), 1e-4);
  }

  // Small-batch family: RecurrentSequence over strided q|k|v rows vs host and
  // bit-identical to n RecurrentDecode launches.
  for (const std::uint32_t n : {1U, 7U, 8U}) {
    std::vector<float> hs = s0;
    std::vector<float> ho;
    HostRecurrent(hs, q, kCh, kk, kCh, v, kCh, in.g.data(), in.beta.data(), ho,
                  n);
    Dev<float> dqkv(in.qkv);
    Dev<float> dg(in.g);
    Dev<float> db(in.beta);
    Dev<float> ds(s0);
    Dev<float> dout(static_cast<std::size_t>(n) * kInner);
    k::RecurrentSequence(ds.get(), dqkv.get(), kCh, dqkv.get() + kInner, kCh,
                         dqkv.get() + 2 * kInner, kCh, dg.get(), db.get(),
                         dout.get(), n, nullptr);
    Check(hipDeviceSynchronize(), "sync");
    const auto got_o = dout.Get();
    const auto got_s = ds.Get();
    const std::string nm = "recurrent sequence n=" + std::to_string(n);
    Report((nm + " o vs host").c_str(), Compare(got_o, ho), 1e-4);
    Report((nm + " state vs host").c_str(), Compare(got_s, hs), 1e-4);

    Dev<float> ds2(s0);
    Dev<float> dout2(kInner);
    Err e;
    for (std::uint32_t t = 0; t < n; ++t) {
      k::RecurrentDecode(
          ds2.get(), dqkv.get() + static_cast<std::size_t>(t) * kCh,
          dqkv.get() + static_cast<std::size_t>(t) * kCh + kInner,
          dqkv.get() + static_cast<std::size_t>(t) * kCh + 2 * kInner,
          dg.get() + static_cast<std::size_t>(t) * kInner,
          db.get() + static_cast<std::size_t>(t) * kH, dout2.get(), nullptr);
      Check(hipDeviceSynchronize(), "sync");
      const auto o = dout2.Get();
      const Err x = Compare(
          o.data(), &got_o[static_cast<std::size_t>(t) * kInner], kInner);
      e.abs = std::max(e.abs, x.abs);
      e.ref_max = std::max(e.ref_max, x.ref_max);
    }
    Report((nm + " == n decode calls o").c_str(), e, 0.0);
    Report((nm + " == n decode calls state").c_str(), Compare(ds2.Get(), got_s),
           0.0);
  }
}

// Prefill family: PrefillSequence over N in {9 ... 4096} tokens vs the scalar
// double recurrence (host reference carried through the same sizes), a long
// 4096-token run for drift, and bit-exact split invariance.
void TestPrefill() {
  std::mt19937 rng(2024);
  const auto s0 = Randn(rng, l::kKdaStateFloats, 0.3F);
  const std::vector<std::uint32_t> sizes = {9, 16, 17, 64, 300, 2048, 4096};
  const std::uint32_t kMax = sizes.back();
  const auto in = MakeInputs(rng, kMax);
  Dev<float> dqkv(in.qkv);
  Dev<float> dg(in.g);
  Dev<float> db(in.beta);
  const auto run = [&](Dev<float>& ds, Dev<float>& dout, std::uint32_t from,
                       std::uint32_t n) {
    k::PrefillSequence(
        ds.get(), dqkv.get() + static_cast<std::size_t>(from) * kCh, kCh,
        dqkv.get() + static_cast<std::size_t>(from) * kCh + kInner, kCh,
        dqkv.get() + static_cast<std::size_t>(from) * kCh + 2 * kInner, kCh,
        dg.get() + static_cast<std::size_t>(from) * kInner,
        db.get() + static_cast<std::size_t>(from) * kH,
        dout.get() + static_cast<std::size_t>(from) * kInner, n, nullptr);
  };

  // Host reference: one pass, state snapshots at every size (float, like the
  // device state); outputs of all kMax tokens.
  std::vector<float> hs = s0;
  std::vector<float> ho_all(static_cast<std::size_t>(kMax) * kInner);
  std::vector<std::vector<float>> snaps;
  {
    std::uint32_t at = 0;
    for (const std::uint32_t n : sizes) {
      std::vector<float> o;
      HostRecurrent(
          hs, in.qkv.data() + static_cast<std::size_t>(at) * kCh, kCh,
          in.qkv.data() + static_cast<std::size_t>(at) * kCh + kInner, kCh,
          in.qkv.data() + static_cast<std::size_t>(at) * kCh + 2 * kInner, kCh,
          in.g.data() + static_cast<std::size_t>(at) * kInner,
          in.beta.data() + static_cast<std::size_t>(at) * kH, o, n - at);
      std::copy(o.begin(), o.end(),
                ho_all.begin() + static_cast<std::size_t>(at) * kInner);
      snaps.push_back(hs);
      at = n;
    }
  }
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const std::uint32_t n = sizes[i];
    Dev<float> ds(s0);
    Dev<float> dout(static_cast<std::size_t>(kMax) * kInner);
    run(ds, dout, 0, n);
    Check(hipDeviceSynchronize(), "sync");
    const auto got_o = dout.Get();
    const std::string nm = "prefill n=" + std::to_string(n);
    Report((nm + " o vs host").c_str(),
           Compare(got_o.data(), ho_all.data(),
                   static_cast<std::size_t>(n) * kInner),
           1e-4);
    Report((nm + " state vs host").c_str(), Compare(ds.Get(), snaps[i]), 1e-4);
    if (n <= 300) {
      // Against the small-batch family kernel: same math, other sum order.
      Dev<float> ds2(s0);
      Dev<float> dout2(static_cast<std::size_t>(n) * kInner);
      k::RecurrentSequence(ds2.get(), dqkv.get(), kCh, dqkv.get() + kInner, kCh,
                           dqkv.get() + 2 * kInner, kCh, dg.get(), db.get(),
                           dout2.get(), n, nullptr);
      Check(hipDeviceSynchronize(), "sync");
      Report((nm + " vs recurrent sequence o").c_str(),
             Compare(got_o.data(), dout2.Get().data(),
                     static_cast<std::size_t>(n) * kInner),
             1e-5);
      Report((nm + " vs recurrent sequence state").c_str(),
             Compare(ds.Get(), ds2.Get()), 1e-5);
    }
  }

  // Split invariance: every split of a run into calls, at any token, is
  // bit-identical to the single call (outputs and state).
  for (const std::uint32_t total : {300U, 2048U, 4096U}) {
    Dev<float> sa(s0);
    Dev<float> oa(static_cast<std::size_t>(kMax) * kInner);
    run(sa, oa, 0, total);
    Check(hipDeviceSynchronize(), "sync");
    const auto want_o = oa.Get();
    const auto want_s = sa.Get();
    const std::vector<std::vector<std::uint32_t>> cuts = {
        {1}, {5}, {8}, {9}, {16}, {17}, {100}, {total - 1}, {7, 20, 129}};
    for (const auto& cut : cuts) {
      Dev<float> sb(s0);
      Dev<float> ob(static_cast<std::size_t>(kMax) * kInner);
      std::uint32_t at = 0;
      std::string nm = "prefill split " + std::to_string(total) + " at";
      for (const std::uint32_t c : cut) {
        run(sb, ob, at, c - at);
        at = c;
        nm += " " + std::to_string(c);
      }
      run(sb, ob, at, total - at);
      Check(hipDeviceSynchronize(), "sync");
      Report((nm + " o == whole").c_str(),
             Compare(ob.Get().data(), want_o.data(),
                     static_cast<std::size_t>(total) * kInner),
             0.0);
      Report((nm + " state == whole").c_str(), Compare(sb.Get(), want_s), 0.0);
    }
  }
}

void TestOutputNormGate() {
  std::mt19937 rng(9);
  const std::uint32_t n = 6;
  const auto o = Randn(rng, static_cast<std::size_t>(n) * kInner, 0.4F);
  const auto proj = Randn(rng, static_cast<std::size_t>(n) * kProj, 1.5F);
  const auto gamma = Randn(rng, kD, 0.3F, 1.0F);
  Dev<float> dout(static_cast<std::size_t>(n) * kInner);
  Dev<float> d_o(o);
  Dev<float> dp(proj);
  Dev<float> dg(gamma);
  k::OutputNormGate(d_o.get(), dp.get(), dg.get(), dout.get(), n, nullptr);
  Check(hipDeviceSynchronize(), "sync");
  std::vector<float> want(o.size());
  for (std::uint32_t t = 0; t < n; ++t) {
    for (std::uint32_t h = 0; h < kH; ++h) {
      const float* p = &o[static_cast<std::size_t>(t) * kInner + h * kD];
      double ss = 0;
      for (std::uint32_t i = 0; i < kD; ++i) {
        ss += static_cast<double>(p[i]) * p[i];
      }
      const double r = 1.0 / std::sqrt(ss / kD + 1e-6);
      for (std::uint32_t i = 0; i < kD; ++i) {
        const double gate = Sigmoid(proj[static_cast<std::size_t>(t) * kProj +
                                         l::kKdaGaRow + h * kD + i]);
        want[static_cast<std::size_t>(t) * kInner + h * kD + i] =
            static_cast<float>(p[i] * r * gamma[i] * gate);
      }
    }
  }
  Report("output_norm_gate", Compare(dout.Get(), want), 1e-5);
}

void TestSnapshot() {
  std::mt19937 rng(5);
  const std::size_t ns = l::kKdaLayers * l::kKdaStateFloats;
  const std::size_t nc = l::kKdaLayers * l::kKdaConvStateFloats;
  const auto state = Randn(rng, ns, 1.0F);
  const auto conv = Randn(rng, nc, 1.0F);
  Dev<float> ds(state);
  Dev<float> dc(conv);
  Dev<float> ss(ns);
  Dev<float> sc(nc);
  k::SnapshotState(ds.get(), dc.get(), ss.get(), sc.get(), nullptr);
  Check(hipDeviceSynchronize(), "sync");
  Report("snapshot state", Compare(ss.Get(), state), 0.0);
  Report("snapshot conv", Compare(sc.Get(), conv), 0.0);
  ds.Put(std::vector<float>(ns, 0.0F));
  dc.Put(std::vector<float>(nc, 0.0F));
  k::RestoreState(ds.get(), dc.get(), ss.get(), sc.get(), nullptr);
  Check(hipDeviceSynchronize(), "sync");
  Report("restore state", Compare(ds.Get(), state), 0.0);
  Report("restore conv", Compare(dc.Get(), conv), 0.0);
}

// The decode fusions must equal the separate launches bit for bit.
void ExactVerdict(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok)
    ++g_failures;
}

bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Rollback points of a small verify chunk (MTP): the state / conv window stored
// after row t of an n-row call must equal, bit for bit, the state after t + 1
// rows of an ordinary call, and the call itself must be unchanged.
void TestCheckpoints() {
  std::mt19937 rng(777);
  const std::size_t sf = l::kKdaStateFloats;
  const std::size_t cf = l::kKdaConvStateFloats;
  const std::uint32_t kN = 6;
  const auto s0 = Randn(rng, sf, 0.3F);
  const auto c0 = Randn(rng, cf, 1.0F);
  const auto in = MakeInputs(rng, kN);
  const float* q = in.qkv.data();
  const float* kk = in.qkv.data() + kInner;
  const float* v = in.qkv.data() + 2 * kInner;
  Dev<float> dq(std::vector<float>(in.qkv.begin(), in.qkv.end()));
  Dev<float> dg(in.g);
  Dev<float> db(in.beta);
  const auto run = [&](std::uint32_t rows, float* ckpt, std::vector<float>* out,
                       std::vector<float>* state) {
    Dev<float> ds(s0);
    Dev<float> dout(static_cast<std::size_t>(kN) * kInner);
    k::RecurrentSequence(ds.get(), dq.get(), kCh, dq.get() + kInner, kCh,
                         dq.get() + 2 * kInner, kCh, dg.get(), db.get(),
                         dout.get(), rows, nullptr, ckpt, sf);
    Check(hipDeviceSynchronize(), "sync");
    *out = dout.Get();
    *state = ds.Get();
  };
  (void)q;
  (void)kk;
  (void)v;
  Dev<float> dckpt(sf * (kN - 1));
  std::vector<float> o_plain, s_plain, o_ck, s_ck;
  run(kN, nullptr, &o_plain, &s_plain);
  run(kN, dckpt.get(), &o_ck, &s_ck);
  ExactVerdict(SameBits(o_plain, o_ck) && SameBits(s_plain, s_ck),
               "RecurrentSequence with rollback points == without");
  const auto ck = dckpt.Get();
  bool all = true;
  for (std::uint32_t t = 0; t + 1 < kN; ++t) {
    std::vector<float> o, st;
    run(t + 1, nullptr, &o, &st);
    all = all && std::memcmp(st.data(), ck.data() + t * sf, sf * 4) == 0;
  }
  ExactVerdict(all, "state rollback point t == state after t + 1 rows");

  // Conv windows.
  const auto proj = Randn(rng, static_cast<std::size_t>(kN) * kProj, 1.0F);
  std::vector<float> w[3];
  for (auto& x : w)
    x = Randn(rng, kInner * 4, 0.5F);
  Dev<float> dp(proj), dw0(w[0]), dw1(w[1]), dw2(w[2]);
  Dev<float> dout(static_cast<std::size_t>(kN) * kCh);
  Dev<float> dcc(cf * (kN - 1));
  {
    Dev<float> dcs(c0);
    k::Conv3Checkpoint(dp.get(), dcs.get(), dcc.get(), cf, kN, nullptr);
    k::Conv3Prefill(dp.get(), dw0.get(), dw1.get(), dw2.get(), dcs.get(),
                    dout.get(), kN, nullptr);
    Check(hipDeviceSynchronize(), "sync");
  }
  const auto cck = dcc.Get();
  bool conv_ok = true;
  for (std::uint32_t t = 0; t + 1 < kN; ++t) {
    Dev<float> dcs(c0);
    k::Conv3Prefill(dp.get(), dw0.get(), dw1.get(), dw2.get(), dcs.get(),
                    dout.get(), t + 1, nullptr);
    Check(hipDeviceSynchronize(), "sync");
    const auto st = dcs.Get();
    conv_ok =
        conv_ok && std::memcmp(st.data(), cck.data() + t * cf, cf * 4) == 0;
  }
  ExactVerdict(conv_ok, "conv rollback point t == window after t + 1 rows");
}

void TestFusedDecode() {
  std::mt19937 rng(777);
  const auto wq = Randn(rng, kInner * 4, 0.5F);
  const auto wk = Randn(rng, kInner * 4, 0.5F);
  const auto wv = Randn(rng, kInner * 4, 0.5F);
  Dev<float> dwq(wq), dwk(wk), dwv(wv);
  const auto ssm_a = Randn(rng, kH, 0.5F, -1.0F);
  const auto dt = Randn(rng, kInner, 1.0F);
  Dev<float> da(ssm_a), ddt(dt);
  // Every decode-family chunk size, with and without the rollback windows of
  // a verify chunk; round 2 uses large (saturating) projections.
  const std::size_t cf = l::kKdaConvStateFloats;
  for (const std::uint32_t n : {1U, 2U, 3U, 5U, 8U}) {
    for (int round = 0; round < 3; ++round) {
      const bool ckpt = round == 1 && n > 1;
      const auto proj =
          Randn(rng, std::size_t{n} * kProj, round == 2 ? 6.0F : 1.5F);
      const auto state0 = Randn(rng, cf, 1.0F);
      Dev<float> dp(proj);
      // Reference: Conv3Checkpoint, Conv3Prefill, GateBetaPrep, L2NormHeads.
      Dev<float> s_ref(state0), conv_ref(std::size_t{n} * kCh),
          g_ref(std::size_t{n} * kInner), b_ref(std::size_t{n} * kH),
          ck_ref(cf * 8);
      Dev<float> s_new(state0), conv_new(std::size_t{n} * kCh),
          g_new(std::size_t{n} * kInner), b_new(std::size_t{n} * kH),
          ck_new(cf * 8);
      Check(hipMemset(ck_ref.get(), 0, cf * 8 * sizeof(float)), "memset");
      Check(hipMemset(ck_new.get(), 0, cf * 8 * sizeof(float)), "memset");
      if (ckpt)
        k::Conv3Checkpoint(dp.get(), s_ref.get(), ck_ref.get(), cf, n, nullptr);
      k::Conv3Prefill(dp.get(), dwq.get(), dwk.get(), dwv.get(), s_ref.get(),
                      conv_ref.get(), n, nullptr);
      k::GateBetaPrep(dp.get(), da.get(), ddt.get(), g_ref.get(), b_ref.get(),
                      n, nullptr);
      k::L2NormHeads(conv_ref.get(), kCh, n, nullptr);
      k::L2NormHeads(conv_ref.get() + kInner, kCh, n, nullptr);
      k::PrepDecode(dp.get(), dwq.get(), dwk.get(), dwv.get(), s_new.get(),
                    da.get(), ddt.get(), conv_new.get(), g_new.get(),
                    b_new.get(), n, ckpt ? ck_new.get() : nullptr, cf, nullptr);
      Check(hipDeviceSynchronize(), "sync");
      const std::string r = " n=" + std::to_string(n) + " round " +
                            std::to_string(round) + (ckpt ? " +ckpt" : "");
      ExactVerdict(SameBits(conv_ref.Get(), conv_new.Get()),
                   "PrepDecode conv/q/k/v == separate" + r);
      ExactVerdict(SameBits(s_ref.Get(), s_new.Get()),
                   "PrepDecode conv state == separate" + r);
      ExactVerdict(SameBits(g_ref.Get(), g_new.Get()),
                   "PrepDecode gate == separate" + r);
      ExactVerdict(SameBits(b_ref.Get(), b_new.Get()),
                   "PrepDecode beta == separate" + r);
      if (ckpt)
        ExactVerdict(SameBits(ck_ref.Get(), ck_new.Get()),
                     "PrepDecode rollback windows == Conv3Checkpoint" + r);
    }
  }
  // OutputNormGateQuant == OutputNormGate + QuantizeActivations.
  const auto gamma = Randn(rng, kD, 0.4F, 1.0F);
  Dev<float> dgamma(gamma);
  for (const std::uint32_t n : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
    const auto o = Randn(rng, std::size_t{n} * kInner, 2.0F);
    const auto proj = Randn(rng, std::size_t{n} * kProj, 2.0F);
    Dev<float> dO(o), dP(proj), out_ref(std::size_t{n} * kInner),
        out_new(std::size_t{n} * kInner);
    const std::size_t qb =
        gufo::models::ling_3_flash::rocm::dense::QuantActBytes(n, kInner);
    Dev<std::uint8_t> q_ref(qb), q_new(qb);
    Check(hipMemset(q_ref.get(), 0, qb), "memset");
    Check(hipMemset(q_new.get(), 0, qb), "memset");
    k::OutputNormGate(dO.get(), dP.get(), dgamma.get(), out_ref.get(), n,
                      nullptr);
    gufo::models::ling_3_flash::rocm::dense::QuantizeActivations(
        out_ref.get(), n, kInner, q_ref.get(), nullptr);
    k::OutputNormGateQuant(dO.get(), dP.get(), dgamma.get(), out_new.get(), n,
                           q_new.get(), nullptr);
    Check(hipDeviceSynchronize(), "sync");
    const auto a = q_ref.Get();
    const auto b = q_new.Get();
    ExactVerdict(
        SameBits(out_ref.Get(), out_new.Get()) && a == b,
        "OutputNormGateQuant == gate + quantise n=" + std::to_string(n));
    // Through the Q8_0 GEMV (2560 x 4096): quantised path == standalone path.
    {
      namespace dn = gufo::models::ling_3_flash::rocm::dense;
      constexpr std::uint32_t kRows = 2560;
      std::vector<std::uint8_t> wq(std::size_t{kRows} * (kInner / 32) * 34);
      std::mt19937 wr(99);
      for (auto& v : wq)
        v = static_cast<std::uint8_t>(wr() & 0x3F);
      for (std::size_t blk = 0; blk < wq.size() / 34; ++blk) {
        wq[blk * 34] = 0x00;  // fp16 scale = 0x3000 (0.125)
        wq[blk * 34 + 1] = 0x30;
      }
      Dev<std::uint8_t> dw(wq);
      Dev<std::uint8_t> scratch(dn::ScratchBytes(
          gufo::models::ling_3_flash::rocm::WeightType::kQ8_0, 8, kInner));
      Dev<float> y_ref(std::size_t{n} * kRows), y_new(std::size_t{n} * kRows);
      dn::GemvQ8_0(dw.get(), out_ref.get(), y_ref.get(), n, kInner, kRows,
                   scratch.get(), nullptr);
      const dn::Q8Group group{dw.get(), y_new.get(), kRows};
      dn::GemvQ8_0Quantized(&group, 1, q_new.get(), n, kInner, nullptr);
      Check(hipDeviceSynchronize(), "sync");
      ExactVerdict(SameBits(y_ref.Get(), y_new.Get()),
                   "gate+quantise -> GemvQ8_0 chain n=" + std::to_string(n));
    }
  }
}

// Prefill -> decode handoff through the whole mixer pipeline (conv, gate/beta,
// L2, recurrence): prefill N tokens (family dispatch as in the executor: n > 8
// PrefillSequence), then K decode steps, against N + K single-token steps.
// The carried KDA state and conv window after N tokens, and the outputs of the
// following steps, must agree to rounding.
void TestHandoff() {
  std::mt19937 rng(606);
  const auto wq = Randn(rng, kInner * 4, 0.5F);
  const auto wk = Randn(rng, kInner * 4, 0.5F);
  const auto wv = Randn(rng, kInner * 4, 0.5F);
  Dev<float> dwq(wq);
  Dev<float> dwk(wk);
  Dev<float> dwv(wv);
  std::vector<float> ssm_a(kH);
  std::uniform_real_distribution<float> ud(0.05F, 4.0F);
  for (auto& a : ssm_a) {
    a = ud(rng);
  }
  Dev<float> da(ssm_a);
  Dev<float> ddt(Randn(rng, kInner, 1.0F));
  const auto s0 = Randn(rng, l::kKdaStateFloats, 0.3F);
  const auto c0 = Randn(rng, l::kKdaConvStateFloats, 1.0F);
  constexpr std::uint32_t kSteps = 8;
  for (const std::uint32_t n : {9U, 16U, 17U, 30U, 31U, 64U, 300U}) {
    const std::uint32_t total = n + kSteps;
    const auto proj = Randn(rng, static_cast<std::size_t>(total) * kProj, 1.5F);
    Dev<float> dproj(proj);
    // Runs tokens [from, from + len) through the pipeline in one call.
    const auto run = [&](Dev<float>& st, Dev<float>& cs, Dev<float>& out,
                         std::uint32_t from, std::uint32_t len) {
      const float* p = dproj.get() + static_cast<std::size_t>(from) * kProj;
      Dev<float> conv(static_cast<std::size_t>(len) * kCh);
      Dev<float> g(static_cast<std::size_t>(len) * kInner);
      Dev<float> b(static_cast<std::size_t>(len) * kH);
      k::Conv3Prefill(p, dwq.get(), dwk.get(), dwv.get(), cs.get(), conv.get(),
                      len, nullptr);
      k::GateBetaPrep(p, da.get(), ddt.get(), g.get(), b.get(), len, nullptr);
      k::L2NormHeads(conv.get(), kCh, len, nullptr);
      k::L2NormHeads(conv.get() + kInner, kCh, len, nullptr);
      float* o = out.get() + static_cast<std::size_t>(from) * kInner;
      if (len == 1) {
        k::RecurrentDecode(st.get(), conv.get(), conv.get() + kInner,
                           conv.get() + 2 * kInner, g.get(), b.get(), o,
                           nullptr);
      } else if (len <= 8) {
        k::RecurrentSequence(st.get(), conv.get(), kCh, conv.get() + kInner,
                             kCh, conv.get() + 2 * kInner, kCh, g.get(),
                             b.get(), o, len, nullptr);
      } else {
        k::PrefillSequence(st.get(), conv.get(), kCh, conv.get() + kInner, kCh,
                           conv.get() + 2 * kInner, kCh, g.get(), b.get(), o,
                           len, nullptr);
      }
      Check(hipDeviceSynchronize(), "sync");
    };
    Dev<float> sa(s0);
    Dev<float> ca(c0);
    Dev<float> oa(static_cast<std::size_t>(total) * kInner);
    run(sa, ca, oa, 0, n);  // prefill family
    const auto state_a = sa.Get();
    const auto conv_a = ca.Get();
    for (std::uint32_t t = n; t < total; ++t) {
      run(sa, ca, oa, t, 1);
    }
    Dev<float> sb(s0);
    Dev<float> cb(c0);
    Dev<float> ob(static_cast<std::size_t>(total) * kInner);
    std::vector<float> state_b;
    std::vector<float> conv_b;
    for (std::uint32_t t = 0; t < total; ++t) {
      run(sb, cb, ob, t, 1);
      if (t + 1 == n) {
        state_b = sb.Get();
        conv_b = cb.Get();
      }
    }
    const std::string nm = "handoff n=" + std::to_string(n);
    Report((nm + " state after prefill").c_str(), Compare(state_a, state_b),
           5e-5);
    Report((nm + " conv after prefill").c_str(), Compare(conv_a, conv_b), 0.0);
    Report((nm + " o of prefill").c_str(),
           Compare(oa.Get().data(), ob.Get().data(),
                   static_cast<std::size_t>(n) * kInner),
           5e-5);
    Report((nm + " o of 8 decode steps").c_str(),
           Compare(oa.Get().data() + static_cast<std::size_t>(n) * kInner,
                   ob.Get().data() + static_cast<std::size_t>(n) * kInner,
                   static_cast<std::size_t>(kSteps) * kInner),
           5e-5);
    Report((nm + " final state").c_str(), Compare(sa.Get(), sb.Get()), 5e-5);
    Report((nm + " final conv").c_str(), Compare(ca.Get(), cb.Get()), 0.0);
  }
}

// Informational device timings (best of 7) of one layer's KDA prefill stages.
template<typename F>
double TimeMs(F&& f) {
  hipEvent_t a;
  hipEvent_t b;
  Check(hipEventCreate(&a), "event");
  Check(hipEventCreate(&b), "event");
  // Spin the clocks up first: a few short launches measure the idle DPM state.
  for (int r = 0; r < 20; ++r) {
    f();
  }
  Check(hipDeviceSynchronize(), "sync");
  constexpr int kReps = 7;
  float best = 1e30F;
  for (int r = 0; r < kReps; ++r) {
    Check(hipEventRecord(a, nullptr), "record");
    f();
    Check(hipEventRecord(b, nullptr), "record");
    Check(hipEventSynchronize(b), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, a, b), "elapsed");
    best = std::min(best, ms);
  }
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return best;
}

void BenchStages() {
  std::mt19937 rng(31);
  for (const std::uint32_t n : {512U, 2048U}) {
    const auto in = MakeInputs(rng, n);
    Dev<float> dqkv(in.qkv);
    Dev<float> dg(in.g);
    Dev<float> db(in.beta);
    Dev<float> ds(Randn(rng, l::kKdaStateFloats, 0.1F));
    Dev<float> dout(static_cast<std::size_t>(n) * kInner);
    Dev<float> dproj(static_cast<std::size_t>(n) * kProj);
    Dev<float> dcs(Randn(rng, l::kKdaConvStateFloats, 1.0F));
    Dev<float> dw(Randn(rng, kInner * 4, 0.5F));
    Dev<float> dssm(Randn(rng, kD, 1.0F));
    const double rec = TimeMs([&] {
      k::PrefillSequence(ds.get(), dqkv.get(), kCh, dqkv.get() + kInner, kCh,
                         dqkv.get() + 2 * kInner, kCh, dg.get(), db.get(),
                         dout.get(), n, nullptr);
    });
    const double conv = TimeMs([&] {
      k::Conv3Prefill(dproj.get(), dw.get(), dw.get(), dw.get(), dcs.get(),
                      dqkv.get(), n, nullptr);
    });
    const double prep = TimeMs([&] {
      k::GateBetaPrep(dproj.get(), dssm.get(), dw.get(), dg.get(), db.get(), n,
                      nullptr);
      k::L2NormHeads(dqkv.get(), kCh, n, nullptr);
      k::L2NormHeads(dqkv.get() + kInner, kCh, n, nullptr);
    });
    const double onorm = TimeMs([&] {
      k::OutputNormGate(dout.get(), dproj.get(), dssm.get(), dout.get(), n,
                        nullptr);
    });
    std::printf(
        "TIMING n=%u: recurrence %.3f ms, conv %.3f ms, gate+l2 %.3f ms, "
        "out-norm-gate %.3f ms\n",
        n, rec, conv, prep, onorm);
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Before the first HIP call: a missing $TMPDIR crashes the runtime.
  if (!gufo::core::RequireTempDirectory())
    return 1;
  // --bench prints informational stage timings (not part of correctness runs).
  const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  try {
    TestConv();
    TestPrep();
    TestRecurrent();
    TestPrefill();
    TestHandoff();
    TestOutputNormGate();
    TestSnapshot();
    TestCheckpoints();
    TestFusedDecode();
    if (bench) {
      BenchStages();
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILED",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
