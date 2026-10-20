// Op tests of the Ling-3.0-flash dense kernels against scalar host references.
// Needs a GPU; exits 77 (skip) without one.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"
#include "tests/models/ling_3_flash/parent_digests.hpp"
#include "tests/models/ling_3_flash/test_rng.hpp"

namespace dn = gufo::models::ling_3_flash::rocm::dense;
using gufo::models::ling_3_flash::rocm::WeightType;

namespace {

int g_failures = 0;
gufo::tests::ling::Rng g_rng(20260101);

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
  std::vector<float> v(n);
  for (auto& x : v)
    x = g_rng.Normal(0.0F, sigma);
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
    for (std::size_t b = 0; b < xe.size() / 32; ++b) {
      const float d = 0.01F * static_cast<float>(1 + b % 7);
      for (int i = 0; i < 32; ++i)
        xe[b * 32 + i] =
            d * static_cast<float>(i == 5 ? 127 : g_rng.Int(-127, 127));
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
    dn::GemmQ8_0(dw.p, dx.p, dy.p, kTok, k, rows, scratch.p, nullptr);
    HIP_OK(hipDeviceSynchronize());
    Name(name, sizeof name, "GemmQ8_0", rows, k, kTok);
    Check(name, dy.Get(), ref.data(), ref.size(), 3e-3);
    // The kernel rounds activations and dequantised weights (d * q) to F16
    // and accumulates in F32: emulate exactly that on the host.
    std::vector<float> w16(wd.size());
    for (std::size_t i = 0; i < wd.size(); ++i)
      w16[i] = static_cast<float>(static_cast<_Float16>(wd[i]));
    std::vector<float> x16(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      x16[i] = static_cast<float>(static_cast<_Float16>(x[i]));
    // The RDNA3 WMMA unit adds each group of 8 products to the F32
    // accumulator with truncation toward zero (a plain F32 sum is up to 10x
    // further away: see the mean error toward zero). Emulate that too.
    std::vector<float> ref16(std::size_t{kTok} * rows);
    for (std::size_t t = 0; t < kTok; ++t)
      for (std::size_t r = 0; r < rows; ++r) {
        float acc = 0.0F;
        for (std::size_t kb = 0; kb < k; kb += 8) {
          double sum = 0;
          for (int i = 0; i < 8; ++i)
            sum += double{w16[r * k + kb + i]} * x16[t * k + kb + i];
          const double tot = double{acc} + sum;
          float f = static_cast<float>(tot);
          if (std::fabs(double{f}) > std::fabs(tot))
            f = std::nextafterf(f, 0.0F);
          acc = f;
        }
        ref16[t * rows + r] = acc;
      }
    Name(name, sizeof name, "GemmQ8_0 vs F16+RTZ-8 emulation", rows, k, kTok);
    Check(name, dy.Get(), ref16.data(), ref16.size(), 3e-5);
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
  Dev<std::uint8_t> scratch(dn::ScratchBytes(WeightType::kQ8_0, n, k));
  dn::GemmQ8_0Strided(dw.p, dx.p, dy.p + off, n, k, rows, ldy, scratch.p,
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

// F16 -> Q8_0 re-encoding of the MTP draft head: ggml's reference rounding
// (d = absmax / 127 stored as F16, q = round(x / d)), checked per block against
// the F16 source, and the Q8_0 GEMV against the F16 GEMV on the same weights.
void TestQuantizeF16ToQ8(std::uint32_t rows, std::uint32_t k) {
  std::vector<std::uint16_t> wh(std::size_t{rows} * k);
  {
    for (std::size_t i = 0; i < wh.size(); ++i) {
      // Mix scales and a few all-zero blocks.
      const std::size_t b = i / 32;
      const float scale = b % 17 == 0 ? 0.0F : (b % 5 == 0 ? 20.0F : 1.0F);
      wh[i] = HalfBits(scale * g_rng.Normal(0.0F, 0.05F));
    }
  }
  Dev<std::uint16_t> dw(wh);
  const std::size_t bytes = std::size_t{rows} * (k / 32) * 34;
  Dev<std::uint8_t> dq(bytes);
  HIP_OK(hipMemset(dq.p, 0xA5, bytes));
  dn::QuantizeF16ToQ8_0(reinterpret_cast<const __half*>(dw.p), dq.p, rows, k,
                        nullptr);
  HIP_OK(hipDeviceSynchronize());
  const auto q = dq.Get();
  std::size_t bad = 0, off_by_one = 0;
  for (std::size_t b = 0; b < wh.size() / 32; ++b) {
    float v[32];
    float amax = 0;
    for (int i = 0; i < 32; ++i) {
      _Float16 h;
      std::memcpy(&h, &wh[b * 32 + i], 2);
      v[i] = static_cast<float>(h);
      amax = std::max(amax, std::fabs(v[i]));
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    std::uint16_t db;
    std::memcpy(&db, &q[b * 34], 2);
    if (db != HalfBits(d)) {
      ++bad;
      continue;
    }
    for (int i = 0; i < 32; ++i) {
      const int want = static_cast<int>(std::lroundf(v[i] * id));
      const int got = static_cast<std::int8_t>(q[b * 34 + 2 + i]);
      if (got != want) {
        // 1/d may round differently on the device (fast-math reciprocal).
        const int diff = got > want ? got - want : want - got;
        if (diff == 1)
          ++off_by_one;
        else
          ++bad;
      }
    }
  }
  std::printf("%-4s QuantizeF16ToQ8_0 %ux%u: %zu bad, %zu off-by-one of %zu\n",
              bad == 0 && off_by_one * 1000 < wh.size() ? "PASS" : "FAIL", rows,
              k, bad, off_by_one, wh.size());
  if (bad != 0 || off_by_one * 1000 >= wh.size())
    ++g_failures;

  // GEMV agreement with the F16 head on the same weights.
  constexpr std::uint32_t kTok = 1;
  const auto x = Randn(std::size_t{kTok} * k);
  Dev<float> dx(x), y16(std::size_t{kTok} * rows), y8(std::size_t{kTok} * rows);
  Dev<std::uint8_t> scratch(std::max(
      dn::ScratchBytes(WeightType::kQ8_0, kTok, k), 2 * std::size_t{kTok} * k));
  dn::GemvF16(reinterpret_cast<const __half*>(dw.p), dx.p, y16.p, kTok, k, rows,
              scratch.p, nullptr);
  dn::GemvQ8_0(dq.p, dx.p, y8.p, kTok, k, rows, scratch.p, nullptr);
  HIP_OK(hipDeviceSynchronize());
  const auto a = y16.Get(), b = y8.Get();
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (double{a[i]} - b[i]) * (double{a[i]} - b[i]);
    den += double{a[i]} * a[i];
  }
  const double rel = std::sqrt(num / std::max(den, 1e-30));
  std::printf(
      "%-4s GemvQ8_0 on re-encoded head vs GemvF16 %ux%u: rel L2 %.3g\n",
      rel < 1e-2 ? "PASS" : "FAIL", rows, k, rel);
  if (rel >= 1e-2)
    ++g_failures;
}

void TestF16(std::uint32_t rows, std::uint32_t k) {
  std::vector<std::uint16_t> wh(std::size_t{rows} * k);
  {
    for (auto& h : wh)
      h = HalfBits(g_rng.Normal(0.0F, 0.05F));
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
  Dev<std::uint8_t> scratch(2 * std::size_t{kTok} * k);
  const auto* w = reinterpret_cast<const __half*>(dw.p);
  char name[96];
  for (std::uint32_t n : {1U, 2U, 8U}) {
    dn::GemvF16(w, dx.p, dy.p, n, k, rows, scratch.p, nullptr);
    HIP_OK(hipDeviceSynchronize());
    auto got = dy.Get();
    got.resize(std::size_t{n} * rows);
    Name(name, sizeof name, "GemvF16", rows, k, n);
    Check(name, got, ref.data(), got.size(), 1e-5);
  }
  dn::GemmF16(w, dx.p, dy.p, kTok, k, rows, scratch.p, nullptr);
  HIP_OK(hipDeviceSynchronize());
  Name(name, sizeof name, "GemmF16", rows, k, kTok);
  Check(name, dy.Get(), ref.data(), ref.size(), 2e-3);
}

// ---- prefill GEMM: split / tile-shape invariance and large-n accuracy -------
std::uint64_t Fnv(const void* p, std::size_t bytes) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < bytes; ++i)
    h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

/// Host emulation of the WMMA GEMM for one output: F16-rounded operands,
/// F32 accumulation adding 8 products at a time with truncation toward zero.
float EmulateDot(const float* w, const float* x, std::size_t k) {
  float acc = 0.0F;
  for (std::size_t kb = 0; kb < k; kb += 8) {
    double sum = 0;
    for (int i = 0; i < 8; ++i)
      sum += double{static_cast<float>(static_cast<_Float16>(w[kb + i]))} *
             static_cast<float>(static_cast<_Float16>(x[kb + i]));
    const double tot = double{acc} + sum;
    float f = static_cast<float>(tot);
    if (std::fabs(double{f}) > std::fabs(tot))
      f = std::nextafterf(f, 0.0F);
    acc = f;
  }
  return acc;
}

/// One weight matrix, the same activations: the n = N call must equal (bit
/// for bit) every chunked split of the same tokens and every tile shape, and
/// match the host emulation on sampled outputs.
void TestGemmInvariance(bool q8, std::uint32_t rows, std::uint32_t k,
                        std::uint32_t n_total) {
  // Fixed per-case seed: the digest must not depend on the test order.
  g_rng.Seed(0x5eed0000U + rows * 31U + k * 7U + n_total + (q8 ? 1U : 0U));
  std::vector<std::uint8_t> wbytes;
  std::vector<float> wd;  // dequantised (F16-exact for the F16 case)
  if (q8) {
    wbytes = QuantizeQ8_0(Randn(std::size_t{rows} * k, 0.5F), k);
    wd = DequantQ8_0Rows(wbytes, rows, k);
  } else {
    wbytes.resize(std::size_t{rows} * k * 2);
    wd.resize(std::size_t{rows} * k);
    for (std::size_t i = 0; i < wd.size(); ++i) {
      const std::uint16_t h = HalfBits(g_rng.Normal(0.0F, 0.05F));
      std::memcpy(&wbytes[i * 2], &h, 2);
      _Float16 f;
      std::memcpy(&f, &h, 2);
      wd[i] = static_cast<float>(f);
    }
  }
  const auto x = Randn(std::size_t{n_total} * k);
  Dev<std::uint8_t> dw(wbytes);
  Dev<float> dx(x);
  Dev<float> dy(std::size_t{n_total} * rows);
  Dev<std::uint8_t> scratch(dn::ScratchBytes(WeightType::kQ8_0, n_total, k));
  const auto* wh = reinterpret_cast<const __half*>(dw.p);
  const auto run = [&](std::uint32_t t0, std::uint32_t n, int shape,
                       bool use_scratch = true) {
    void* sc = use_scratch ? scratch.p : nullptr;
    float* y = dy.p + std::size_t{t0} * rows;
    const float* xs = dx.p + std::size_t{t0} * k;
    if (q8) {
      if (shape < 0)
        dn::GemmQ8_0(dw.p, xs, y, n, k, rows, sc, nullptr);
      else
        dn::GemmQ8_0Shape(dw.p, xs, y, n, k, rows, rows, shape, sc, nullptr);
    } else {
      if (shape < 0)
        dn::GemmF16(wh, xs, y, n, k, rows, sc, nullptr);
      else
        dn::GemmF16Shape(wh, xs, y, n, k, rows, shape, sc, nullptr);
    }
  };
  const auto fetch = [&] {
    HIP_OK(hipDeviceSynchronize());
    return dy.Get();
  };
  char name[160];
  const char* what = q8 ? "GemmQ8_0" : "GemmF16";
  run(0, n_total, -1);
  const auto full = fetch();
  std::printf("DIGEST %s rows=%u k=%u n=%u %016llx\n", what, rows, k, n_total,
              static_cast<unsigned long long>(
                  Fnv(full.data(), full.size() * sizeof(float))));
  {  // golden digest of the parent kernels
    const auto digest = Fnv(full.data(), full.size() * sizeof(float));
    bool found = false, ok = true;
    for (const auto& g : parent_digests::kDense)
      if (g.rows == rows && g.k == k && g.n == n_total &&
          std::strcmp(g.op, what) == 0) {
        found = true;
        ok = g.digest == digest;
      }
    if (gufo::tests::ling::PrintGoldens()) {
      std::printf("    {\"%s\", %u, %u, %u, 0x%016llxULL},\n", what, rows, k,
                  n_total, static_cast<unsigned long long>(digest));
    } else if (found) {
      std::snprintf(name, sizeof name, "%s == parent digest rows=%u k=%u n=%u",
                    what, rows, k, n_total);
      std::printf("%-4s %s\n", ok ? "PASS" : "FAIL", name);
      if (!ok)
        ++g_failures;
    }
  }
  const auto same = [&](const char* tag, const std::vector<float>& got) {
    const bool ok =
        std::memcmp(got.data(), full.data(), got.size() * sizeof(float)) == 0;
    std::snprintf(name, sizeof name, "%s %s rows=%u k=%u n=%u", what, tag, rows,
                  k, n_total);
    std::printf("%-4s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok)
      ++g_failures;
  };
  HIP_OK(hipMemset(dy.p, 0xFF, dy.n * sizeof(float)));
  run(0, n_total, -1, false);
  same("scratch-free path bit-identical", fetch());
  for (int shape = 0; shape < dn::kGemmShapes; ++shape) {
    HIP_OK(hipMemset(dy.p, 0xFF, dy.n * sizeof(float)));
    run(0, n_total, shape);
    char tag[40];
    std::snprintf(tag, sizeof tag, "tile shape %d bit-identical", shape);
    same(tag, fetch());
  }
  // Chunk prefixes; the remainder of the tokens goes in one last call.
  const std::vector<std::vector<std::uint32_t>> splits = {
      {9},
      {n_total > 9 ? n_total - 9 : 9},
      {97, 128},
      {512, 512, 512},
      {64, 64, 64, 64}};
  for (std::size_t si = 0; si < splits.size(); ++si) {
    HIP_OK(hipMemset(dy.p, 0xFF, dy.n * sizeof(float)));
    std::uint32_t t0 = 0;
    for (std::uint32_t n : splits[si]) {
      n = std::min(n, n_total - t0);
      if (n == 0)
        break;
      run(t0, n, -1);
      t0 += n;
    }
    if (t0 < n_total)
      run(t0, n_total - t0, -1);
    char tag[40];
    std::snprintf(tag, sizeof tag, "split %zu bit-identical", si);
    same(tag, fetch());
  }
  // Sampled accuracy against the host emulation (every tile edge case: the
  // first/last tokens and rows are always included).
  std::vector<std::uint32_t> ts{0, n_total - 1}, rs{0, rows - 1};
  for (int i = 0; i < 46; ++i) {
    ts.push_back(g_rng.Next() % n_total);
    rs.push_back(g_rng.Next() % rows);
  }
  std::vector<float> got, ref;
  for (auto t : ts)
    for (auto r : rs) {
      got.push_back(full[std::size_t{t} * rows + r]);
      ref.push_back(
          EmulateDot(&wd[std::size_t{r} * k], &x[std::size_t{t} * k], k));
    }
  std::snprintf(name, sizeof name,
                "%s vs F16+RTZ-8 emulation (sampled) rows=%u k=%u n=%u", what,
                rows, k, n_total);
  Check(name, got, ref.data(), ref.size(), 3e-5);
}

// ---- --bench ----------------------------------------------------------------
void BenchGemm(bool q8, std::uint32_t rows, std::uint32_t k, std::uint32_t n,
               const std::vector<std::uint8_t>& wbytes) {
  Dev<std::uint8_t> dw(wbytes);
  Dev<float> dx(Randn(std::size_t{n} * k));
  Dev<float> dy(std::size_t{n} * rows);
  Dev<std::uint8_t> scratch(dn::ScratchBytes(WeightType::kQ8_0, n, k));
  const auto* wh = reinterpret_cast<const __half*>(dw.p);
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  const auto launch = [&](int shape) {
    if (shape < 0) {  // default dispatch
      if (q8)
        dn::GemmQ8_0(dw.p, dx.p, dy.p, n, k, rows, scratch.p, nullptr);
      else
        dn::GemmF16(wh, dx.p, dy.p, n, k, rows, scratch.p, nullptr);
    } else if (q8) {
      dn::GemmQ8_0Shape(dw.p, dx.p, dy.p, n, k, rows, rows, shape, scratch.p,
                        nullptr);
    } else {
      dn::GemmF16Shape(wh, dx.p, dy.p, n, k, rows, shape, scratch.p, nullptr);
    }
  };
  const auto time_ms = [&](int shape) {
    // Whole-batch timing of >= 20 launches covering >= 5 ms.
    int reps = 20;
    for (;;) {
      HIP_OK(hipEventRecord(e0));
      for (int i = 0; i < reps; ++i)
        launch(shape);
      HIP_OK(hipEventRecord(e1));
      HIP_OK(hipEventSynchronize(e1));
      float ms = 0;
      HIP_OK(hipEventElapsedTime(&ms, e0, e1));
      if (ms >= 5.0F)
        return static_cast<double>(ms) / reps;
      reps *= 2;
    }
  };
  // Tiny kernels are dominated by clock ramp-up and the first launch of each
  // configuration: warm up the GPU on every configuration, then time two
  // passes (forward, reverse order) and keep the minimum per configuration.
  std::vector<int> order{-1};
  for (int s2 = 0; s2 < dn::kGemmShapes; ++s2)
    order.push_back(s2);
  for (int s2 : order)
    (void)time_ms(s2);
  std::vector<double> best(order.size(), 1e30);
  for (int pass = 0; pass < 2; ++pass)
    for (std::size_t i = 0; i < order.size(); ++i) {
      const std::size_t j = pass == 0 ? i : order.size() - 1 - i;
      best[j] = std::min(best[j], time_ms(order[j]));
    }
  const double flop = 2.0 * rows * k * n;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const std::string shape =
        order[i] < 0
            ? "auto(chosen=" + std::to_string(dn::GemmShapeFor(n, rows)) + ")"
            : std::to_string(order[i]);
    std::printf("BENCH %s rows=%u k=%u n=%u shape=%s ms=%.4f tflops=%.2f\n",
                q8 ? "GemmQ8_0" : "GemmF16", rows, k, n, shape.c_str(), best[i],
                flop / (best[i] * 1e-3) / 1e12);
  }
  HIP_OK(hipEventDestroy(e0));
  HIP_OK(hipEventDestroy(e1));
}

void RunBench() {
  const std::pair<std::uint32_t, std::uint32_t> shapes[] = {
      {20512, 2560}, {6144, 2560}, {2560, 4096},
      {2560, 6144},  {768, 2560},  {2560, 768}};
  for (const auto& [rows, k] : shapes) {
    const auto wq = QuantizeQ8_0(Randn(std::size_t{rows} * k, 0.5F), k);
    for (std::uint32_t n : {16U, 32U, 64U, 128U, 512U, 2048U})
      BenchGemm(true, rows, k, n, wq);
  }
}

// ---- Q4_K embedding
// -----------------------------------------------------------
void TestEmbed() {
  constexpr std::uint32_t vocab = 157184, hidden = 2560;
  constexpr std::size_t row_bytes = hidden / 256 * 144;
  std::vector<std::uint8_t> table(std::size_t{vocab} * row_bytes);
  {
    for (auto& b : table)
      b = static_cast<std::uint8_t>(g_rng.Int(0, 255));
    for (std::size_t sb = 0; sb < table.size() / 144; ++sb) {
      const std::uint16_t d = HalfBits(g_rng.Uniform(0.001F, 0.02F)),
                          m = HalfBits(g_rng.Uniform(0.001F, 0.02F));
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

// ---- Q8_0 embedding (AD-Q4_K_S / AD-Q5_K_M token_embd)
void TestEmbedQ8_0() {
  constexpr std::uint32_t vocab = 157184, hidden = 2560;
  constexpr std::size_t row_bytes = hidden / 32 * 34;
  std::vector<std::uint8_t> table(std::size_t{vocab} * row_bytes);
  {
    for (auto& b : table)
      b = static_cast<std::uint8_t>(g_rng.Int(0, 255));
    for (std::size_t blk = 0; blk < table.size() / 34; ++blk) {
      const std::uint16_t d = HalfBits(g_rng.Uniform(0.0001F, 0.01F));
      std::memcpy(&table[blk * 34], &d, 2);
    }
  }
  const std::vector<std::int32_t> tokens = {0,     1,     157183, 12345,
                                            99999, 12345, 777,    156000};
  Dev<std::uint8_t> dt(table);
  Dev<std::int32_t> dtok(tokens);
  Dev<float> dout(tokens.size() * hidden);
  dn::Embed(WeightType::kQ8_0, dt.p, dtok.p, dout.p,
            static_cast<std::uint32_t>(tokens.size()), hidden, nullptr);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> ref(tokens.size() * hidden);
  for (std::size_t t = 0; t < tokens.size(); ++t)
    gufo::quant::DequantizeQ8_0(
        table.data() + std::size_t(tokens[t]) * row_bytes,
        ref.data() + t * hidden, hidden);
  Check("EmbedQ8_0 vocab=157184 hidden=2560 n=8", dout.Get(), ref.data(),
        ref.size(), 0.0);
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

// ---------------------------------------------------------------------------
// Decode fusions must reproduce the separate launches bit for bit.
// ---------------------------------------------------------------------------
bool SameBytes(const void* a, const void* b, std::size_t bytes) {
  return std::memcmp(a, b, bytes) == 0;
}

void Verdict(bool ok, const char* what, std::uint32_t n) {
  std::printf("%s %s n=%u\n", ok ? "PASS" : "FAIL", what, n);
  if (!ok)
    ++g_failures;
}

void TestFusions() {
  constexpr std::uint32_t kH = 2560;
  const auto gamma = Randn(kH, 0.3F);
  std::vector<float> gamma1 = gamma;
  for (auto& g : gamma1)
    g += 1.0F;
  Dev<float> dg(gamma1);
  for (std::uint32_t n = 1; n <= 8; ++n) {
    const auto res0 = Randn(std::size_t{n} * kH, 2.0F);
    const auto add = Randn(std::size_t{n} * kH, 1.5F);
    const auto eo = Randn(std::size_t{n} * 8 * kH, 0.7F);
    auto wts = Randn(std::size_t{n} * 8, 0.4F);
    const auto shared = Randn(std::size_t{n} * kH, 0.5F);
    const std::size_t qb = dn::QuantActBytes(n, kH);
    Dev<std::uint8_t> q_ref(qb), q_new(qb);
    Dev<float> x_ref(std::size_t{n} * kH), x_new(std::size_t{n} * kH);
    const auto planes_equal = [&](const char* what) {
      HIP_OK(hipDeviceSynchronize());
      const auto a = q_ref.Get();
      const auto b = q_new.Get();
      const auto xa = x_ref.Get();
      const auto xb = x_new.Get();
      Verdict(SameBytes(a.data(), b.data(), qb) &&
                  SameBytes(xa.data(), xb.data(), xa.size() * 4),
              what, n);
    };
    // Plain: x = res.
    {
      Dev<float> res(res0);
      HIP_OK(hipMemset(q_ref.p, 0, qb));
      HIP_OK(hipMemset(q_new.p, 0, qb));
      dn::RmsNormRows(res.p, dg.p, x_ref.p, n, kH, 1, 1e-6F, nullptr);
      dn::QuantizeActivations(x_ref.p, n, kH, q_ref.p, nullptr);
      dn::RmsNormQuantRows(res.p, dg.p, x_new.p, n, 1e-6F, q_new.p, nullptr);
      planes_equal("RmsNormQuantRows == norm + quantise");
    }
    // Residual add.
    {
      Dev<float> r_ref(res0), r_new(res0), d_add(add);
      HIP_OK(hipMemset(q_ref.p, 0, qb));
      HIP_OK(hipMemset(q_new.p, 0, qb));
      dn::AddResidual(r_ref.p, d_add.p, std::size_t{n} * kH, nullptr);
      dn::RmsNormRows(r_ref.p, dg.p, x_ref.p, n, kH, 1, 1e-6F, nullptr);
      dn::QuantizeActivations(x_ref.p, n, kH, q_ref.p, nullptr);
      dn::AddResidualRmsNormQuant(r_new.p, d_add.p, dg.p, x_new.p, n, 1e-6F,
                                  q_new.p, nullptr);
      planes_equal("AddResidualRmsNormQuant == add + norm + quantise");
      const auto a = r_ref.Get();
      const auto b = r_new.Get();
      Verdict(SameBytes(a.data(), b.data(), a.size() * 4),
              "AddResidualRmsNormQuant residual", n);
    }
    // MoE combine.
    {
      Dev<float> r_ref(res0), r_new(res0), d_eo(eo), d_w(wts), d_sh(shared);
      HIP_OK(hipMemset(q_ref.p, 0, qb));
      HIP_OK(hipMemset(q_new.p, 0, qb));
      gufo::models::ling_3_flash::rocm::moe::Combine(r_ref.p, d_eo.p, d_w.p,
                                                     d_sh.p, n, nullptr);
      dn::RmsNormRows(r_ref.p, dg.p, x_ref.p, n, kH, 1, 1e-6F, nullptr);
      dn::QuantizeActivations(x_ref.p, n, kH, q_ref.p, nullptr);
      dn::CombineRmsNormQuant(r_new.p, d_eo.p, d_w.p, d_sh.p, dg.p, x_new.p, n,
                              1e-6F, q_new.p, nullptr);
      planes_equal("CombineRmsNormQuant == combine + norm + quantise");
      const auto a = r_ref.Get();
      const auto b = r_new.Get();
      Verdict(SameBytes(a.data(), b.data(), a.size() * 4),
              "CombineRmsNormQuant residual", n);
    }
    // SwiGLU with planes (shared expert: 768 wide, and a clamp).
    for (const float limit : {0.0F, 7.0F}) {
      constexpr std::uint32_t kF = 768;
      const auto gate = Randn(std::size_t{n} * kF, 4.0F);
      const auto up = Randn(std::size_t{n} * kF, 4.0F);
      Dev<float> dgate(gate), dup(up), o_ref(std::size_t{n} * kF),
          o_new(std::size_t{n} * kF);
      const std::size_t qf = dn::QuantActBytes(n, kF);
      Dev<std::uint8_t> a_ref(qf), a_new(qf);
      HIP_OK(hipMemset(a_ref.p, 0, qf));
      HIP_OK(hipMemset(a_new.p, 0, qf));
      dn::SwiGluClamped(dgate.p, dup.p, o_ref.p, std::size_t{n} * kF, limit,
                        nullptr);
      dn::QuantizeActivations(o_ref.p, n, kF, a_ref.p, nullptr);
      dn::SwiGluClampedQuant(dgate.p, dup.p, o_new.p, std::size_t{n} * kF, kF,
                             limit, a_new.p, nullptr);
      HIP_OK(hipDeviceSynchronize());
      const auto pa = a_ref.Get();
      const auto pb = a_new.Get();
      const auto oa = o_ref.Get();
      const auto ob = o_new.Get();
      Verdict(SameBytes(pa.data(), pb.data(), qf) &&
                  SameBytes(oa.data(), ob.data(), oa.size() * 4),
              limit > 0 ? "SwiGluClampedQuant (clamped)" : "SwiGluClampedQuant",
              n);
    }
  }
}

// Three weight matrices over one set of planes in one launch == three
// separate GemvQ8_0 launches, bit for bit (including 4-token sub-batches).
void TestGroupedGemv() {
  struct Shape {
    std::uint32_t rows, k;
  };
  for (const std::uint32_t k : {2560U, 4096U, 768U}) {
    const std::uint32_t rows[3] = {6144, 576, 32};
    std::vector<std::vector<std::uint8_t>> wq;
    std::vector<std::unique_ptr<Dev<std::uint8_t>>> dw;
    for (int i = 0; i < 3; ++i) {
      wq.push_back(QuantizeQ8_0(Randn(std::size_t{rows[i]} * k, 0.5F), k));
      dw.push_back(std::make_unique<Dev<std::uint8_t>>(wq.back()));
    }
    for (std::uint32_t n : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
      const auto x = Randn(std::size_t{n} * k);
      Dev<float> dx(x);
      Dev<std::uint8_t> scratch(dn::ScratchBytes(WeightType::kQ8_0, 8, k));
      Dev<std::uint8_t> quant(dn::QuantActBytes(n, k));
      std::vector<std::unique_ptr<Dev<float>>> y_ref, y_new;
      for (int i = 0; i < 3; ++i) {
        y_ref.push_back(std::make_unique<Dev<float>>(std::size_t{n} * rows[i]));
        y_new.push_back(std::make_unique<Dev<float>>(std::size_t{n} * rows[i]));
        dn::GemvQ8_0(dw[i]->p, dx.p, y_ref[i]->p, n, k, rows[i], scratch.p,
                     nullptr);
      }
      dn::QuantizeActivations(dx.p, n, k, quant.p, nullptr);
      dn::Q8Group groups[3] = {{dw[0]->p, y_new[0]->p, rows[0]},
                               {dw[1]->p, y_new[1]->p, rows[1]},
                               {dw[2]->p, y_new[2]->p, rows[2]}};
      dn::GemvQ8_0Quantized(groups, 3, quant.p, n, k, nullptr);
      HIP_OK(hipDeviceSynchronize());
      bool ok = true;
      for (int i = 0; i < 3; ++i) {
        const auto a = y_ref[i]->Get();
        const auto b = y_new[i]->Get();
        ok = ok && SameBytes(a.data(), b.data(), a.size() * 4);
      }
      char name[96];
      std::snprintf(name, sizeof name, "grouped GemvQ8_0Quantized k=%u", k);
      Verdict(ok, name, n);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Before the first HIP call: a missing $TMPDIR crashes the runtime.
  if (!gufo::core::RequireTempDirectory())
    return 1;
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::puts("SKIP: no GPU");
    return 77;
  }
  if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
    RunBench();
    return 0;
  }
  const std::pair<std::uint32_t, std::uint32_t> q8_shapes[] = {
      {20512, 2560}, {6144, 2560}, {576, 2560}, {32, 2560},
      {2560, 4096},  {2560, 6144}, {768, 2560}, {2560, 768}};
  for (const auto& [rows, k] : q8_shapes)
    TestQ8(rows, k);
  // Dispatch contract: small chunks (agent turns of 50-150 tokens) must not
  // take the 128x128 tile; chunks of up to 32 tokens take the 128-row tiles
  // only when those give every WGP a block (gfx1151: 20 WGPs, so 768 rows
  // take 64 x 64), as on this device.
  int device_wgps = 0;
  HIP_OK(hipDeviceGetAttribute(&device_wgps,
                               hipDeviceAttributeMultiprocessorCount, 0));
  for (const int wgps : {20, 4, device_wgps}) {
    for (std::uint32_t rows : {576U, 768U, 2560U, 6144U, 20512U}) {
      const bool tall = (rows + 127) / 128 >= static_cast<std::uint32_t>(wgps);
      const int want[][2] = {{9, tall ? 3 : 1},
                             {16, tall ? 3 : 1},
                             {17, tall ? 2 : 1},
                             {32, tall ? 2 : 1},
                             {33, 1},
                             {64, 1}};
      for (const auto& [n, shape] : want) {
        const int got = wgps == device_wgps ? dn::GemmShapeFor(n, rows)
                                            : dn::GemmShapeFor(n, rows, wgps);
        const bool ok = got == shape;
        std::printf("%-4s GemmShapeFor(n=%d, rows=%u, wgps=%d) == %d\n",
                    ok ? "PASS" : "FAIL", n, rows, wgps, shape);
        if (!ok)
          ++g_failures;
      }
    }
  }
  {  // gfx1151 at the bench shapes: 768 x 2560 n=16 is the 64 x 64 tile.
    const bool ok = dn::GemmShapeFor(16, 768, 20) == 1 &&
                    dn::GemmShapeFor(16, 2560, 20) == 3;
    std::printf("%-4s GemmShapeFor gfx1151 n=16: 768 rows -> 1, 2560 -> 3\n",
                ok ? "PASS" : "FAIL");
    if (!ok)
      ++g_failures;
  }
  // Prefill GEMM at real shapes and n in {9, 64, 512, 2048}: tile-shape and
  // split invariance (bit-exact) plus sampled accuracy.
  // Small chunks whose dispatch depends on the WGP count: every shape gives
  // the same bits there too.
  for (std::uint32_t n : {16U, 32U}) {
    TestGemmInvariance(true, 768, 2560, n);
    TestGemmInvariance(true, 576, 2560, n);
  }
  for (std::uint32_t n : {9U, 64U, 512U, 2048U}) {
    TestGemmInvariance(true, 20512, 2560, n);
    TestGemmInvariance(true, 2560, 4096, n);
    TestGemmInvariance(true, 768, 2560, n);
    TestGemmInvariance(true, 2560, 768, n);
    TestGemmInvariance(true, 6144, 2560, n);
    TestGemmInvariance(true, 2560, 6144, n);
    TestGemmInvariance(false, 4097, 2560, n);
  }
  TestQ8BatchInvariant(6144, 6144);  // dense FFN down: 4-token sub-batches
  TestQ8BatchInvariant(2560, 2560);
  TestQ8BatchInvariant(576, 4096);
  TestQ8Strided();
  TestF32Router();
  TestF16(157184, 2560);
  TestF16(4097, 2560);
  TestQuantizeF16ToQ8(157184, 2560);
  TestQuantizeF16ToQ8(33, 2560);
  TestEmbed();
  TestEmbedQ8_0();
  TestRmsNorm(5, 2560, 1, true, false);
  TestRmsNorm(3, 2560, 1, false, true);
  TestRmsNorm(7, 4096, 32, true, false);     // per-head norm, d = 128
  TestRmsNorm(2, 32 * 192, 32, true, true);  // d = 192
  TestRmsNorm(2, 512, 1, true, false);       // MLA latent
  TestResidualSwiGlu();
  TestFusions();
  TestGroupedGemv();
  std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "ALL PASSED",
              g_failures);
  return g_failures ? 1 : 0;
}
