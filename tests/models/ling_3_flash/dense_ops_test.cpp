// Op tests of the Ling-3.0-flash dense kernels against scalar host references.
// Needs a GPU; exits 77 (skip) without one.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"

namespace dn = gufo::models::ling_3_flash::rocm::dense;
using gufo::models::ling_3_flash::rocm::WeightType;

namespace {

int g_failures = 0;
std::mt19937 g_rng(20260101);

#define HIP_OK(call)                                                         \
  do {                                                                       \
    const hipError_t e_ = (call);                                            \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
                   __FILE__, __LINE__);                                      \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

template<typename T>
struct Dev {
  T* p{nullptr};
  std::size_t n{0};
  explicit Dev(std::size_t count) : n(count) {
    HIP_OK(hipMalloc(reinterpret_cast<void**>(&p),
                     std::max<std::size_t>(count, 1) * sizeof(T)));
  }
  Dev(const std::vector<T>& v) : Dev(v.size()) {
    HIP_OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  }
  ~Dev() { (void)hipFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  std::vector<T> Get() const {
    std::vector<T> v(n);
    HIP_OK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};

std::vector<float> Randn(std::size_t n, float sigma = 1.0F) {
  std::normal_distribution<float> d(0.0F, sigma);
  std::vector<float> v(n);
  for (auto& x : v)
    x = d(g_rng);
  return v;
}

std::uint16_t HalfBits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

/// Quantises F32 rows (k % 32 == 0) to GGUF Q8_0: 34-byte blocks.
std::vector<std::uint8_t> QuantizeQ8_0(const std::vector<float>& w,
                                       std::size_t k) {
  const std::size_t nblocks = w.size() / 32;
  std::vector<std::uint8_t> out(nblocks * 34);
  for (std::size_t b = 0; b < nblocks; ++b) {
    float amax = 0;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(w[b * 32 + i]));
    const std::uint16_t db = HalfBits(amax / 127.0F);
    _Float16 dh;
    std::memcpy(&dh, &db, 2);
    const float d = static_cast<float>(dh);
    std::memcpy(&out[b * 34], &db, 2);
    for (int i = 0; i < 32; ++i) {
      const int q =
          d > 0 ? static_cast<int>(std::lrintf(w[b * 32 + i] / d)) : 0;
      out[b * 34 + 2 + i] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(std::clamp(q, -127, 127)));
    }
  }
  (void)k;
  return out;
}

std::vector<float> DequantQ8_0Rows(const std::vector<std::uint8_t>& q,
                                   std::size_t rows, std::size_t k) {
  std::vector<float> w(rows * k);
  const std::size_t rb = k / 32 * 34;
  for (std::size_t r = 0; r < rows; ++r)
    gufo::quant::DequantizeQ8_0(q.data() + r * rb, w.data() + r * k, k);
  return w;
}

template<typename F>
void ParallelFor(std::size_t n, F f) {
  const unsigned nt = std::max(1U, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      for (std::size_t i = t; i < n; i += nt)
        f(i);
    });
  for (auto& x : th)
    x.join();
}

/// ref[t * rows + r] = sum_c w[r][c] * x[t][c] in double.
std::vector<float> RefGemm(const std::vector<float>& w,
                           const std::vector<float>& x, std::size_t n_tokens,
                           std::size_t k, std::size_t rows) {
  std::vector<float> ref(n_tokens * rows);
  ParallelFor(rows, [&](std::size_t r) {
    for (std::size_t t = 0; t < n_tokens; ++t) {
      double s = 0;
      for (std::size_t c = 0; c < k; ++c)
        s += double{w[r * k + c]} * x[t * k + c];
      ref[t * rows + r] = static_cast<float>(s);
    }
  });
  return ref;
}

/// Reports max |got - ref| relative to rms(ref); returns pass.
bool Check(const char* name, const std::vector<float>& got, const float* ref,
           std::size_t n, double tol) {
  double max_abs = 0, ss = 0;
  bool finite = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(got[i]))
      finite = false;
    max_abs = std::max(max_abs, std::fabs(double{got[i]} - ref[i]));
    ss += double{ref[i]} * ref[i];
  }
  const double rms = std::sqrt(ss / std::max<std::size_t>(n, 1));
  const double rel = rms > 0 ? max_abs / rms : max_abs;
  const bool ok = finite && rel <= tol;
  std::printf("%-4s %-52s max_abs=%.3e rms_ref=%.3e rel=%.3e tol=%.1e\n",
              ok ? "PASS" : "FAIL", name, max_abs, rms, rel, tol);
  if (!ok)
    ++g_failures;
  return ok;
}

void Name(char* buf, std::size_t n, const char* what, std::size_t a,
          std::size_t b, std::size_t c) {
  std::snprintf(buf, n, "%s rows=%zu k=%zu n=%zu", what, a, b, c);
}

// ---- Q8_0 ------------------------------------------------------------------
void TestQ8(std::uint32_t rows, std::uint32_t k) {
  constexpr std::uint32_t kTok = 67;
  const auto wf = Randn(std::size_t{rows} * k, 0.5F);
  const auto wq = QuantizeQ8_0(wf, k);
  const auto wd = DequantQ8_0Rows(wq, rows, k);
  const auto x = Randn(std::size_t{kTok} * k);
  const auto ref = RefGemm(wd, x, kTok, k, rows);
  Dev<std::uint8_t> dw(wq);
  Dev<float> dx(x);
  Dev<float> dy(std::size_t{kTok} * rows);
  Dev<std::uint8_t> scratch(dn::ScratchBytes(WeightType::kQ8_0, kTok, k));
  char name[128];

  for (std::uint32_t n : {1U, 3U, 8U, 9U}) {
    HIP_OK(hipMemset(dy.p, 0xFF, dy.n * sizeof(float)));
    dn::GemvQ8_0(dw.p, dx.p, dy.p, n, k, rows, scratch.p, nullptr);
    HIP_OK(hipDeviceSynchronize());
    auto got = dy.Get();
    got.resize(std::size_t{n} * rows);
    Name(name, sizeof name, "GemvQ8_0", rows, k, n);
    // Two int8 activation planes: as accurate as the F16-activation GEMM.
    // Every n <= 8 runs the int8 kernel (in LDS-sized sub-batches).
    const bool gemv = n <= 8;
    Check(name, got, ref.data(), got.size(), gemv ? 1e-3 : 3e-3);
  }
  {  // Activations already on the int8 grid: the int8 path is then exact
    // (the residual plane is zero).
    constexpr std::uint32_t n = 8;
    std::vector<float> xe(std::size_t{n} * k);
    std::uniform_int_distribution<int> qd(-127, 127);
    for (std::size_t b = 0; b < xe.size() / 32; ++b) {
      const float d = 0.01F * static_cast<float>(1 + b % 7);
      for (int i = 0; i < 32; ++i)
        xe[b * 32 + i] = d * static_cast<float>(i == 5 ? 127 : qd(g_rng));
    }
    const auto refe = RefGemm(wd, xe, n, k, rows);
    Dev<float> dxe(xe);
    dn::GemvQ8_0(dw.p, dxe.p, dy.p, n, k, rows, scratch.p, nullptr);
    HIP_OK(hipDeviceSynchronize());
    auto got = dy.Get();
    got.resize(std::size_t{n} * rows);
    Name(name, sizeof name, "GemvQ8_0 int8-exact", rows, k, n);
    Check(name, got, refe.data(), got.size(), 2e-5);
  }
  {
    HIP_OK(hipMemset(dy.p, 0xFF, dy.n * sizeof(float)));
    dn::GemmQ8_0(dw.p, dx.p, dy.p, kTok, k, rows, nullptr, nullptr);
    HIP_OK(hipDeviceSynchronize());
    Name(name, sizeof name, "GemmQ8_0", rows, k, kTok);
    Check(name, dy.Get(), ref.data(), ref.size(), 3e-3);
  }
}

// Batch invariance: token t of an n-token call (n <= 8) must be bit-identical
// to the same token in a 1-token call, including when n exceeds the LDS
// sub-batch size (k = 6144 fits only 4 tokens).
void TestQ8BatchInvariant(std::uint32_t rows, std::uint32_t k) {
  const auto wq = QuantizeQ8_0(Randn(std::size_t{rows} * k, 0.5F), k);
  const auto x = Randn(std::size_t{8} * k);
  Dev<std::uint8_t> dw(wq);
  Dev<float> dx(x);
  Dev<float> dy(std::size_t{8} * rows);
  Dev<std::uint8_t> scratch(dn::ScratchBytes(WeightType::kQ8_0, 8, k));
  std::vector<float> one(std::size_t{8} * rows);
  for (std::uint32_t t = 0; t < 8; ++t) {
    dn::GemvQ8_0(dw.p, dx.p + std::size_t{t} * k, dy.p, 1, k, rows, scratch.p,
                 nullptr);
    HIP_OK(hipDeviceSynchronize());
    const auto g = dy.Get();
    std::copy_n(g.begin(), rows, one.begin() + std::size_t{t} * rows);
  }
  for (std::uint32_t n = 2; n <= 8; ++n) {
    HIP_OK(hipMemset(dy.p, 0xFF, dy.n * sizeof(float)));
    dn::GemvQ8_0(dw.p, dx.p, dy.p, n, k, rows, scratch.p, nullptr);
    HIP_OK(hipDeviceSynchronize());
    auto got = dy.Get();
    got.resize(std::size_t{n} * rows);
    char name[128];
    Name(name, sizeof name, "GemvQ8_0 batch-invariant", rows, k, n);
    const bool ok =
        std::memcmp(got.data(), one.data(), got.size() * sizeof(float)) == 0;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok)
      ++g_failures;
  }
}

void TestQ8Strided() {
  // Writes the 6144-row q block of a stacked [t][20512] projection at row
  // offset 4096; everything else must stay untouched.
  constexpr std::uint32_t rows = 6144, k = 2560, ldy = 20512, n = 70,
                          off = 4096;
  const auto wq = QuantizeQ8_0(Randn(std::size_t{rows} * k, 0.5F), k);
  const auto wd = DequantQ8_0Rows(wq, rows, k);
  const auto x = Randn(std::size_t{n} * k);
  const auto ref = RefGemm(wd, x, n, k, rows);
  Dev<std::uint8_t> dw(wq);
  Dev<float> dx(x);
  Dev<float> dy(std::size_t{n} * ldy);
  HIP_OK(hipMemset(dy.p, 0, dy.n * sizeof(float)));
  dn::GemmQ8_0Strided(dw.p, dx.p, dy.p + off, n, k, rows, ldy, nullptr,
                      nullptr);
  HIP_OK(hipDeviceSynchronize());
  const auto got = dy.Get();
  std::vector<float> sel(std::size_t{n} * rows);
  bool untouched = true;
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t c = 0; c < ldy; ++c) {
      if (c >= off && c < off + rows)
        sel[t * rows + (c - off)] = got[t * ldy + c];
      else if (got[t * ldy + c] != 0.0F)
        untouched = false;
    }
  Check("GemmQ8_0Strided rows=6144 ldy=20512 off=4096 n=70", sel, ref.data(),
        ref.size(), 3e-3);
  std::printf("%-4s GemmQ8_0Strided leaves other columns untouched\n",
              untouched ? "PASS" : "FAIL");
  if (!untouched)
    ++g_failures;
}

// ---- F32 / F16 --------------------------------------------------------------
void TestF32Router() {
  constexpr std::uint32_t rows = 512, k = 2560;
  const auto w = Randn(std::size_t{rows} * k, 0.05F);
  const auto x = Randn(std::size_t{19} * k);
  const auto ref = RefGemm(w, x, 19, k, rows);
  Dev<float> dw(w), dx(x), dy(std::size_t{19} * rows);
  for (std::uint32_t n : {1U, 4U, 8U, 19U}) {
    dn::GemvF32(dw.p, dx.p, dy.p, n, k, rows, nullptr);
    HIP_OK(hipDeviceSynchronize());
    auto got = dy.Get();
    got.resize(std::size_t{n} * rows);
    char name[96];
    Name(name, sizeof name, "GemvF32", rows, k, n);
    Check(name, got, ref.data(), got.size(), 1e-5);
  }
}

void TestF16(std::uint32_t rows, std::uint32_t k) {
  std::vector<std::uint16_t> wh(std::size_t{rows} * k);
  {
    std::normal_distribution<float> d(0.0F, 0.05F);
    for (auto& h : wh)
      h = HalfBits(d(g_rng));
  }
  constexpr std::uint32_t kTok = 67;
  const auto x = Randn(std::size_t{kTok} * k);
  std::vector<float> ref(std::size_t{kTok} * rows);
  ParallelFor(rows, [&](std::size_t r) {
    std::vector<float> row(k);
    for (std::uint32_t c = 0; c < k; ++c) {
      _Float16 h;
      std::memcpy(&h, &wh[r * k + c], 2);
      row[c] = static_cast<float>(h);
    }
    for (std::size_t t = 0; t < kTok; ++t) {
      double s = 0;
      for (std::uint32_t c = 0; c < k; ++c)
        s += double{row[c]} * x[t * k + c];
      ref[t * rows + r] = static_cast<float>(s);
    }
  });
  Dev<std::uint16_t> dw(wh);
  Dev<float> dx(x), dy(std::size_t{kTok} * rows);
  const auto* w = reinterpret_cast<const __half*>(dw.p);
  char name[96];
  for (std::uint32_t n : {1U, 2U, 8U}) {
    dn::GemvF16(w, dx.p, dy.p, n, k, rows, nullptr);
    HIP_OK(hipDeviceSynchronize());
    auto got = dy.Get();
    got.resize(std::size_t{n} * rows);
    Name(name, sizeof name, "GemvF16", rows, k, n);
    Check(name, got, ref.data(), got.size(), 1e-5);
  }
  dn::GemmF16(w, dx.p, dy.p, kTok, k, rows, nullptr);
  HIP_OK(hipDeviceSynchronize());
  Name(name, sizeof name, "GemmF16", rows, k, kTok);
  Check(name, dy.Get(), ref.data(), ref.size(), 2e-3);
}

// ---- Q4_K embedding
// -----------------------------------------------------------
void TestEmbed() {
  constexpr std::uint32_t vocab = 157184, hidden = 2560;
  constexpr std::size_t row_bytes = hidden / 256 * 144;
  std::vector<std::uint8_t> table(std::size_t{vocab} * row_bytes);
  {
    std::uniform_int_distribution<int> bd(0, 255);
    std::uniform_real_distribution<float> dd(0.001F, 0.02F);
    for (auto& b : table)
      b = static_cast<std::uint8_t>(bd(g_rng));
    for (std::size_t sb = 0; sb < table.size() / 144; ++sb) {
      const std::uint16_t d = HalfBits(dd(g_rng)), m = HalfBits(dd(g_rng));
      std::memcpy(&table[sb * 144], &d, 2);
      std::memcpy(&table[sb * 144 + 2], &m, 2);
    }
  }
  const std::vector<std::int32_t> tokens = {0,     1,     157183, 12345,
                                            99999, 12345, 777,    156000};
  Dev<std::uint8_t> dt(table);
  Dev<std::int32_t> dtok(tokens);
  Dev<float> dout(tokens.size() * hidden);
  dn::EmbedQ4K(dt.p, dtok.p, dout.p, static_cast<std::uint32_t>(tokens.size()),
               hidden, nullptr);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> ref(tokens.size() * hidden);
  for (std::size_t t = 0; t < tokens.size(); ++t)
    gufo::quant::DequantizeQ4_K(
        table.data() + std::size_t(tokens[t]) * row_bytes,
        ref.data() + t * hidden, hidden);
  Check("EmbedQ4K vocab=157184 hidden=2560 n=8", dout.Get(), ref.data(),
        ref.size(), 1e-6);
}

// ---- Norm / residual / SwiGLU ----------------------------------------------
void TestRmsNorm(std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 bool gamma, bool in_place) {
  const std::uint32_t d = dim / groups;
  const auto x = Randn(std::size_t{n_rows} * dim, 3.0F);
  const auto g = Randn(d, 1.0F);
  constexpr float eps = 1e-6F;
  std::vector<float> ref(x.size());
  for (std::size_t r = 0; r < n_rows; ++r)
    for (std::uint32_t gi = 0; gi < groups; ++gi) {
      const float* p = &x[r * dim + std::size_t{gi} * d];
      double ss = 0;
      for (std::uint32_t i = 0; i < d; ++i)
        ss += double{p[i]} * p[i];
      const double inv = 1.0 / std::sqrt(ss / d + eps);
      for (std::uint32_t i = 0; i < d; ++i)
        ref[r * dim + std::size_t{gi} * d + i] =
            static_cast<float>(p[i] * inv * (gamma ? g[i] : 1.0F));
    }
  Dev<float> dx(x), dg(g), dout(x.size());
  float* out = in_place ? dx.p : dout.p;
  dn::RmsNormRows(dx.p, gamma ? dg.p : nullptr, out, n_rows, dim, groups, eps,
                  nullptr);
  HIP_OK(hipDeviceSynchronize());
  const auto got = in_place ? dx.Get() : dout.Get();
  char name[128];
  std::snprintf(name, sizeof name,
                "RmsNormRows n=%u dim=%u groups=%u gamma=%d inplace=%d", n_rows,
                dim, groups, gamma, in_place);
  Check(name, got, ref.data(), ref.size(), 1e-5);
}

void TestResidualSwiGlu() {
  constexpr std::size_t n = 3 * 2560 + 7;
  const auto a = Randn(n), b = Randn(n);
  Dev<float> dr(a), db(b);
  dn::AddResidual(dr.p, db.p, n, nullptr);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> ref(n);
  for (std::size_t i = 0; i < n; ++i)
    ref[i] = a[i] + b[i];
  Check("AddResidual n=7687", dr.Get(), ref.data(), n, 1e-7);

  const auto gate = Randn(n, 6.0F), up = Randn(n, 6.0F);
  Dev<float> dgate(gate), dup(up), dout(n);
  for (float L : {0.0F, 5.0F, 7.0F}) {
    dn::SwiGluClamped(dgate.p, dup.p, dout.p, n, L, nullptr);
    HIP_OK(hipDeviceSynchronize());
    for (std::size_t i = 0; i < n; ++i) {
      double s = double{gate[i]} / (1.0 + std::exp(-double{gate[i]}));
      double u = up[i];
      if (L > 0) {
        s = std::min<double>(s, L);
        u = std::clamp<double>(u, -L, L);
      }
      ref[i] = static_cast<float>(s * u);
    }
    char name[64];
    std::snprintf(name, sizeof name, "SwiGluClamped L=%g", double{L});
    Check(name, dout.Get(), ref.data(), n, 1e-5);
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::puts("SKIP: no GPU");
    return 77;
  }
  const std::pair<std::uint32_t, std::uint32_t> q8_shapes[] = {
      {20512, 2560}, {6144, 2560}, {576, 2560}, {32, 2560},
      {2560, 4096},  {2560, 6144}, {768, 2560}, {2560, 768}};
  for (const auto& [rows, k] : q8_shapes)
    TestQ8(rows, k);
  TestQ8BatchInvariant(6144, 6144);  // dense FFN down: 4-token sub-batches
  TestQ8BatchInvariant(2560, 2560);
  TestQ8BatchInvariant(576, 4096);
  TestQ8Strided();
  TestF32Router();
  TestF16(157184, 2560);
  TestF16(4097, 2560);
  TestEmbed();
  TestRmsNorm(5, 2560, 1, true, false);
  TestRmsNorm(3, 2560, 1, false, true);
  TestRmsNorm(7, 4096, 32, true, false);     // per-head norm, d = 128
  TestRmsNorm(2, 32 * 192, 32, true, true);  // d = 192
  TestRmsNorm(2, 512, 1, true, false);       // MLA latent
  TestResidualSwiGlu();
  std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "ALL PASSED",
              g_failures);
  return g_failures ? 1 : 0;
}
