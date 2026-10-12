// Op tests of the Ling-3.0-flash KDA kernels against scalar host references
// (double accumulation, straight from PLAN.md section 1). Random inputs with
// fixed seeds. Exit 77 when no GPU is present.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

  // Row-split prefill over strided q|k|v rows vs host and vs N decodes.
  for (const std::uint32_t n : {1U, 7U, 64U, 300U}) {
    std::vector<float> hs = s0;
    std::vector<float> ho;
    HostRecurrent(hs, q, kCh, kk, kCh, v, kCh, in.g.data(), in.beta.data(), ho,
                  n);
    Dev<float> dqkv(in.qkv);
    Dev<float> dg(in.g);
    Dev<float> db(in.beta);
    Dev<float> ds(s0);
    Dev<float> dout(static_cast<std::size_t>(n) * kInner);
    k::PrefillRowSplit(ds.get(), dqkv.get(), kCh, dqkv.get() + kInner, kCh,
                       dqkv.get() + 2 * kInner, kCh, dg.get(), db.get(),
                       dout.get(), n, nullptr);
    Check(hipDeviceSynchronize(), "sync");
    const auto got_o = dout.Get();
    const auto got_s = ds.Get();
    const std::string nm = "prefill row-split n=" + std::to_string(n);
    Report((nm + " o vs host").c_str(), Compare(got_o, ho), 1e-4);
    Report((nm + " state vs host").c_str(), Compare(got_s, hs), 1e-4);

    // Same through n RecurrentDecode launches (device-side comparison).
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
    Report((nm + " == n decode calls o").c_str(), e, 1e-6);
    Report((nm + " == n decode calls state").c_str(), Compare(ds2.Get(), got_s),
           1e-6);
  }

  // Chunk invariance: 100 + 200 tokens == 300 tokens.
  {
    Dev<float> dqkv(in.qkv);
    Dev<float> dg(in.g);
    Dev<float> db(in.beta);
    auto run = [&](Dev<float>& ds, Dev<float>& dout, std::uint32_t from,
                   std::uint32_t n) {
      k::PrefillRowSplit(
          ds.get(), dqkv.get() + static_cast<std::size_t>(from) * kCh, kCh,
          dqkv.get() + static_cast<std::size_t>(from) * kCh + kInner, kCh,
          dqkv.get() + static_cast<std::size_t>(from) * kCh + 2 * kInner, kCh,
          dg.get() + static_cast<std::size_t>(from) * kInner,
          db.get() + static_cast<std::size_t>(from) * kH,
          dout.get() + static_cast<std::size_t>(from) * kInner, n, nullptr);
    };
    Dev<float> sa(s0);
    Dev<float> oa(static_cast<std::size_t>(kMax) * kInner);
    run(sa, oa, 0, kMax);
    Dev<float> sb(s0);
    Dev<float> ob(static_cast<std::size_t>(kMax) * kInner);
    run(sb, ob, 0, 100);
    run(sb, ob, 100, 200);
    Check(hipDeviceSynchronize(), "sync");
    Report("prefill 100+200 == 300 o", Compare(ob.Get(), oa.Get()), 0.0);
    Report("prefill 100+200 == 300 state", Compare(sb.Get(), sa.Get()), 0.0);
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

}  // namespace

int main() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  try {
    TestConv();
    TestPrep();
    TestRecurrent();
    TestOutputNormGate();
    TestSnapshot();
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILED",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
