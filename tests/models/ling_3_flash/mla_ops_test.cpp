// Op tests of the Ling-3.0-flash MLA kernels (absorbed and non-absorbed)
// against scalar host references computed in double, fixed seeds. Prints max
// abs / relative errors (relative = max abs error / max |reference|). Exit 77
// without a GPU.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"
#include "src/models/ling_3_flash/layout.hpp"
#include "tests/models/ling_3_flash/parent_digests.hpp"

namespace L = gufo::models::ling_3_flash;
namespace mla = gufo::models::ling_3_flash::rocm::mla;

namespace {

constexpr std::uint32_t kH = L::kHeads;
constexpr std::uint32_t kLat = L::kMlaLatent;
constexpr std::uint32_t kW = L::kMlaCacheWidth;
constexpr std::uint32_t kNope = L::kMlaNope;
constexpr std::uint32_t kRopeDim = L::kMlaRope;
constexpr std::uint32_t kQH = L::kMlaQHeadDim;
constexpr std::uint32_t kV = L::kMlaValueDim;

int g_failures = 0;

void CheckHip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

template<typename T>
class Dev {
public:
  explicit Dev(std::size_t n) : n_(n) {
    CheckHip(hipMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)),
             "hipMalloc");
    CheckHip(hipMemset(p_, 0, n * sizeof(T)), "hipMemset");
  }
  explicit Dev(const std::vector<T>& v) : Dev(v.size()) { Upload(v); }
  ~Dev() { (void)hipFree(p_); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  T* get() { return p_; }
  void Upload(const std::vector<T>& v) {
    CheckHip(
        hipMemcpy(p_, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  }
  std::vector<T> Download() const {
    std::vector<T> v(n_);
    CheckHip(hipMemcpy(v.data(), p_, n_ * sizeof(T), hipMemcpyDeviceToHost),
             "download");
    return v;
  }

private:
  T* p_{nullptr};
  std::size_t n_;
};

std::vector<float> Normal(std::size_t n, std::uint32_t seed, float sigma) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, sigma);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(gen);
  }
  return v;
}

std::vector<__half> ToHalf(const std::vector<float>& v) {
  std::vector<__half> h(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    h[i] = __float2half(v[i]);
  }
  return h;
}

float RoundHalf(double x) {
  return __half2float(__float2half(static_cast<float>(x)));
}

template<typename F>
void ParallelFor(std::size_t n, F f) {
  const std::size_t threads = std::min<std::size_t>(4, n);
  std::vector<std::thread> pool;
  for (std::size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (std::size_t i = t; i < n; i += threads) {
        f(i);
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
}

struct Err {
  double max_abs{0};
  double max_ref{0};
  void Add(double got, double want) {
    max_abs = std::max(max_abs, std::abs(got - want));
    max_ref = std::max(max_ref, std::abs(want));
  }
  [[nodiscard]] double Rel() const {
    return max_abs / std::max(max_ref, 1e-30);
  }
};

void Report(const std::string& name, const Err& e, double rel_limit) {
  const bool ok = e.Rel() <= rel_limit && std::isfinite(e.max_abs);
  std::printf(
      "%-44s max_abs=%.3e max_rel=%.3e (|ref|max=%.3e, limit %.1e) %s\n",
      name.c_str(), e.max_abs, e.Rel(), e.max_ref, rel_limit,
      ok ? "PASS" : "FAIL");
  if (!ok) {
    ++g_failures;
  }
}

// ---- host references ------------------------------------------------------

void HostRope(const double* in, double* out, std::int32_t pos) {
  for (std::uint32_t i = 0; i < kRopeDim / 2; ++i) {
    const double freq = std::pow(6.0e6, -2.0 * i / kRopeDim);
    const double a = static_cast<double>(pos) * freq;
    const double c = std::cos(a);
    const double s = std::sin(a);
    out[2 * i] = (in[2 * i] * c) - (in[(2 * i) + 1] * s);
    out[(2 * i) + 1] = (in[2 * i] * s) + (in[(2 * i) + 1] * c);
  }
}

// One cache row [latent | k_pe] in double from a kv_a row.
std::vector<double> HostLatentRow(const float* kv_a, const float* gamma,
                                  std::int32_t pos) {
  std::vector<double> row(kW);
  double ss = 0;
  for (std::uint32_t c = 0; c < kLat; ++c) {
    ss += static_cast<double>(kv_a[c]) * kv_a[c];
  }
  const double scale = 1.0 / std::sqrt((ss / kLat) + 1e-6);
  for (std::uint32_t c = 0; c < kLat; ++c) {
    row[c] = kv_a[c] * scale * gamma[c];
  }
  std::vector<double> pe(kRopeDim);
  for (std::uint32_t i = 0; i < kRopeDim; ++i) {
    pe[i] = kv_a[kLat + i];
  }
  HostRope(pe.data(), row.data() + kLat, pos);
  return row;
}

// Q8_0 weights: random `rows x k` floats per head quantised to 34-byte blocks.
struct Q8Weights {
  std::vector<std::uint8_t> bytes;
  std::vector<float> dequant;  // [head][row][k]
};

Q8Weights MakeQ8(std::uint32_t rows, std::uint32_t k, std::uint32_t seed,
                 float sigma) {
  const std::vector<float> src =
      Normal(std::size_t{kH} * rows * k, seed, sigma);
  Q8Weights w;
  w.dequant.resize(src.size());
  const std::size_t blocks = src.size() / 32;
  w.bytes.resize(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    float amax = 0;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::abs(src[(b * 32) + j]));
    }
    const __half d = __float2half(amax / 127.0F);
    const float df = __half2float(d);
    std::memcpy(&w.bytes[b * 34], &d, 2);
    for (std::size_t j = 0; j < 32; ++j) {
      const int qv =
          df > 0 ? static_cast<int>(std::lround(src[(b * 32) + j] / df)) : 0;
      w.bytes[(b * 34) + 2 + j] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(std::clamp(qv, -127, 127)));
      w.dequant[(b * 32) + j] =
          df * static_cast<float>(std::clamp(qv, -127, 127));
    }
  }
  return w;
}

// Absorbed attention of one query row set against `n_keys` cache rows.
// q_abs: [32][576] floats; cache rows F16. Returns [32][512] (double).
void HostAbsorbedAttention(const float* q_abs, const __half* cache,
                           std::uint32_t n_keys, std::vector<double>* out) {
  out->assign(std::size_t{kH} * kLat, 0.0);
  std::vector<float> cf(std::size_t{n_keys} * kW);
  for (std::size_t i = 0; i < cf.size(); ++i) {
    cf[i] = __half2float(cache[i]);
  }
  const double scale = 1.0 / std::sqrt(192.0);
  for (std::uint32_t h = 0; h < kH; ++h) {
    std::vector<double> s(n_keys);
    double mx = -1e300;
    for (std::uint32_t p = 0; p < n_keys; ++p) {
      double acc = 0;
      const float* kr = &cf[std::size_t{p} * kW];
      const float* qr = q_abs + (std::size_t{h} * kW);
      for (std::uint32_t c = 0; c < kW; ++c) {
        acc += static_cast<double>(qr[c]) * kr[c];
      }
      s[p] = acc * scale;
      mx = std::max(mx, s[p]);
    }
    double sum = 0;
    for (double& v : s) {
      v = std::exp(v - mx);
      sum += v;
    }
    double* o = out->data() + (std::size_t{h} * kLat);
    for (std::uint32_t p = 0; p < n_keys; ++p) {
      const double w = s[p] / sum;
      const float* kr = &cf[std::size_t{p} * kW];
      for (std::uint32_t c = 0; c < kLat; ++c) {
        o[c] += w * kr[c];
      }
    }
  }
}

// Realistic cache rows: latent ~ N(0,1), k_pe ~ N(0,1.5).
std::vector<__half> MakeCache(std::size_t rows, std::uint32_t seed) {
  std::vector<float> v = Normal(rows * kW, seed, 1.0F);
  const std::vector<float> pe = Normal(rows * kRopeDim, seed + 1, 1.5F);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::uint32_t i = 0; i < kRopeDim; ++i) {
      v[(r * kW) + kLat + i] = pe[(r * kRopeDim) + i];
    }
  }
  return ToHalf(v);
}

// ---- tests ----------------------------------------------------------------

void TestLatentNormRopeAndAppend() {
  for (const std::int32_t base_pos : {0, 120000}) {
    const std::uint32_t n = 37;
    std::vector<float> kv_a = Normal(std::size_t{n} * kW, 11, 3.0F);
    std::vector<float> gamma = Normal(kLat, 12, 0.2F);
    for (float& g : gamma) {
      g += 1.0F;
    }
    std::vector<std::int32_t> pos(n);
    for (std::uint32_t t = 0; t < n; ++t) {
      pos[t] = base_pos + static_cast<std::int32_t>(t);
    }
    Dev<float> d_kv(kv_a);
    Dev<float> d_gamma(gamma);
    Dev<std::int32_t> d_pos(pos);
    Dev<float> d_out(std::size_t{n} * kW);
    mla::LatentNormRope(d_kv.get(), d_gamma.get(), d_pos.get(), d_out.get(), n,
                        nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
    const auto got = d_out.Download();
    Err err;
    for (std::uint32_t t = 0; t < n; ++t) {
      const auto row =
          HostLatentRow(&kv_a[std::size_t{t} * kW], gamma.data(), pos[t]);
      for (std::uint32_t c = 0; c < kW; ++c) {
        err.Add(got[(std::size_t{t} * kW) + c], row[c]);
      }
    }
    Report("LatentNormRope pos0=" + std::to_string(base_pos), err, 2e-6);

    // CacheAppend into the middle of a layer of a two-layer cache.
    const std::uint32_t capacity = 64;
    const std::uint32_t first_pos = 5;
    std::vector<__half> sentinel(std::size_t{2} * capacity * kW,
                                 __float2half(7.0F));
    Dev<__half> d_cache(sentinel);
    __half* layer1 = d_cache.get() + (std::size_t{capacity} * kW);
    std::vector<std::int32_t> append_pos(n);
    for (std::uint32_t t = 0; t < n; ++t) {
      append_pos[t] = static_cast<std::int32_t>(first_pos + t);
    }
    Dev<std::int32_t> d_append_pos(append_pos);
    mla::CacheAppend(layer1, d_out.get(), d_append_pos.get(), n, nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
    const auto cache = d_cache.Download();
    bool exact = true;
    for (std::size_t i = 0; i < cache.size(); ++i) {
      const std::size_t layer_off = std::size_t{capacity} * kW;
      __half want = __float2half(7.0F);
      if (i >= layer_off + (std::size_t{first_pos} * kW) &&
          i < layer_off + (std::size_t{first_pos + n} * kW)) {
        want = __float2half(got[i - layer_off - (std::size_t{first_pos} * kW)]);
      }
      std::uint16_t a = 0;
      std::uint16_t b = 0;
      std::memcpy(&a, &cache[i], 2);
      std::memcpy(&b, &want, 2);
      exact = exact && a == b;
    }
    std::printf("%-44s %s\n", "CacheAppend (exact, neighbours untouched)",
                exact ? "PASS" : "FAIL");
    g_failures += exact ? 0 : 1;
  }
}

void TestQAbsorb() {
  const Q8Weights kb = MakeQ8(kLat, kNope, 21, 0.05F);
  Dev<std::uint8_t> d_kb(kb.bytes);
  for (const std::uint32_t n : {1U, 33U}) {
    const std::vector<float> q =
        Normal(std::size_t{n} * kH * kQH, 22 + n, 1.5F);
    std::vector<std::int32_t> pos(n);
    for (std::uint32_t t = 0; t < n; ++t) {
      pos[t] = 1000 + 7 * static_cast<std::int32_t>(t);
    }
    Dev<float> d_q(q);
    Dev<std::int32_t> d_pos(pos);
    Dev<float> d_out(std::size_t{n} * kH * kW);
    mla::QAbsorb(d_q.get(), d_kb.get(), d_pos.get(), d_out.get(), n, nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
    const auto got = d_out.Download();
    Err err_lat;
    Err err_pe;
    for (std::uint32_t t = 0; t < n; ++t) {
      for (std::uint32_t h = 0; h < kH; ++h) {
        const float* qh = &q[((std::size_t{t} * kH) + h) * kQH];
        const float* go = &got[((std::size_t{t} * kH) + h) * kW];
        for (std::uint32_t c = 0; c < kLat; ++c) {
          double acc = 0;
          const float* w = &kb.dequant[((std::size_t{h} * kLat) + c) * kNope];
          for (std::uint32_t d = 0; d < kNope; ++d) {
            acc += static_cast<double>(qh[d]) * w[d];
          }
          err_lat.Add(go[c], acc);
        }
        std::vector<double> pe(kRopeDim);
        std::vector<double> rot(kRopeDim);
        for (std::uint32_t i = 0; i < kRopeDim; ++i) {
          pe[i] = qh[kNope + i];
        }
        HostRope(pe.data(), rot.data(), pos[t]);
        for (std::uint32_t i = 0; i < kRopeDim; ++i) {
          err_pe.Add(go[kLat + i], rot[i]);
        }
      }
    }
    Report("QAbsorb latent T=" + std::to_string(n), err_lat, 1e-5);
    Report("QAbsorb q_pe rope T=" + std::to_string(n), err_pe, 2e-6);
  }
}

void TestDecode() {
  for (const std::uint32_t context : {1U, 17U, 4096U, 70000U}) {
    // Two-layer cache, test layer 1, to check the layer base addressing.
    const std::size_t layer_elems = std::size_t{context} * kW;
    std::vector<__half> layer = MakeCache(context, 31 + context);
    std::vector<__half> both(2 * layer_elems, __float2half(5.0F));
    std::copy(layer.begin(), layer.end(), both.begin() + layer_elems);
    Dev<__half> d_cache(both);
    const std::vector<float> q =
        Normal(std::size_t{kH} * kW, 32 + context, 1.7F);
    Dev<float> d_q(q);
    Dev<float> d_out(std::size_t{kH} * kLat);
    Dev<std::uint8_t> d_scratch(mla::DecodeScratchBytes(70000));
    Dev<std::int32_t> d_position(
        std::vector<std::int32_t>{static_cast<std::int32_t>(context - 1)});
    mla::DecodeAttention(d_q.get(), d_cache.get() + layer_elems,
                         d_position.get(), d_out.get(), d_scratch.get(),
                         nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
    const auto got = d_out.Download();
    std::vector<double> want;
    HostAbsorbedAttention(q.data(), layer.data(), context, &want);
    Err err;
    for (std::size_t i = 0; i < want.size(); ++i) {
      err.Add(got[i], want[i]);
    }
    Report("DecodeAttention context=" + std::to_string(context), err, 1.5e-3);
  }
}

std::size_t CountDiff(const float* a, const float* b, std::size_t n) {
  std::size_t d = 0;
  for (std::size_t i = 0; i < n; ++i)
    d += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return d;
}

// ---- non-absorbed prefill ---------------------------------------------------

// k_b / v_b as the engine holds them: Q8_0 on the device plus the F16 expansion
// weights built from them.
struct NaWeights {
  Q8Weights kb;
  Q8Weights vb;
  Dev<std::uint8_t> d_kb;
  Dev<std::uint8_t> d_vb;
  Dev<__half> d_w16;
  explicit NaWeights(std::uint32_t seed)
      : kb(MakeQ8(kLat, kNope, seed, 0.05F)),
        vb(MakeQ8(kV, kLat, seed + 1, 0.05F)),
        d_kb(kb.bytes),
        d_vb(vb.bytes),
        d_w16(mla::ExpansionWeightBytes() / sizeof(__half)) {
    mla::BuildExpansionWeights(d_kb.get(), d_vb.get(), d_w16.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
  }
};

// Device non-absorbed prefill of `n` tokens at `first` from raw q
// ([t][32][192], un-roped). Returns [t][32*128].
std::vector<float> RunNa(NaWeights& w, __half* cache, std::uint32_t first,
                         std::uint32_t n, Dev<std::uint8_t>& scratch,
                         Dev<float>& d_qr, Dev<float>& d_out,
                         const float* d_q_src, const float* d_gate,
                         __half* store = nullptr, std::uint32_t rows = 0) {
  std::vector<std::int32_t> pos(n);
  for (std::uint32_t t = 0; t < n; ++t) {
    pos[t] = static_cast<std::int32_t>(first + t);
  }
  Dev<std::int32_t> d_pos(pos);
  mla::QPrepare(d_q_src, d_pos.get(), d_qr.get(), n, nullptr);
  if (rows > 0) {
    mla::ExpandRows(cache, w.d_w16.get(), store, rows, 0,
                    std::min(first + n, rows), nullptr);
  }
  mla::PrefillAttentionNonAbsorbed(d_qr.get(), cache, w.d_w16.get(), store,
                                   rows, d_gate, first, n, d_out.get(),
                                   scratch.get(), nullptr);
  CheckHip(hipDeviceSynchronize(), "sync");
  return d_out.Download();
}

// Reference for one token: absorb q on the host, attend over the F16 cache in
// double, expand with v_b. Equals the non-absorbed math before F16 rounding of
// the expanded K/V. Returns [32][128].
std::vector<double> HostNaToken(const NaWeights& w, const float* q_raw,
                                std::int32_t pos, const __half* cache,
                                std::uint32_t n_keys) {
  std::vector<float> q_abs(std::size_t{kH} * kW);
  for (std::uint32_t h = 0; h < kH; ++h) {
    const float* qh = q_raw + (std::size_t{h} * kQH);
    for (std::uint32_t c = 0; c < kLat; ++c) {
      double acc = 0;
      const float* k = &w.kb.dequant[((std::size_t{h} * kLat) + c) * kNope];
      for (std::uint32_t d = 0; d < kNope; ++d) {
        acc += static_cast<double>(qh[d]) * k[d];
      }
      q_abs[(std::size_t{h} * kW) + c] = static_cast<float>(acc);
    }
    std::vector<double> pe(kRopeDim);
    std::vector<double> rot(kRopeDim);
    for (std::uint32_t i = 0; i < kRopeDim; ++i) {
      pe[i] = qh[kNope + i];
    }
    HostRope(pe.data(), rot.data(), pos);
    for (std::uint32_t i = 0; i < kRopeDim; ++i) {
      q_abs[(std::size_t{h} * kW) + kLat + i] = static_cast<float>(rot[i]);
    }
  }
  std::vector<double> lat;
  HostAbsorbedAttention(q_abs.data(), cache, n_keys, &lat);
  std::vector<double> out(std::size_t{kH} * kV);
  for (std::uint32_t h = 0; h < kH; ++h) {
    for (std::uint32_t e = 0; e < kV; ++e) {
      const float* v = &w.vb.dequant[((std::size_t{h} * kV) + e) * kLat];
      double acc = 0;
      for (std::uint32_t c = 0; c < kLat; ++c) {
        acc += lat[(std::size_t{h} * kLat) + c] * v[c];
      }
      out[(std::size_t{h} * kV) + e] = acc;
    }
  }
  return out;
}

void TestNonAbsorbedPrefill() {
  NaWeights w(71);
  struct Case {
    std::uint32_t n;
    std::uint32_t first;
  };
  for (const Case c :
       {Case{9, 0}, Case{9, 1000}, Case{9, 65536}, Case{64, 0}, Case{64, 1000},
        Case{64, 65536}, Case{512, 0}, Case{512, 1000}, Case{512, 65536},
        Case{40, 4090}, Case{2048, 0}}) {
    const std::uint32_t total = c.first + c.n;
    const std::vector<__half> cache = MakeCache(total, 141 + total);
    const std::vector<float> q =
        Normal(std::size_t{c.n} * kH * kQH, 142 + c.n + c.first, 1.7F);
    const std::vector<float> gate = Normal(std::size_t{c.n} * kH, 143, 1.5F);
    Dev<__half> d_cache(cache);
    Dev<float> d_q(q);
    Dev<float> d_gate(gate);
    Dev<float> d_qr(std::size_t{c.n} * kH * kQH);
    Dev<float> d_out(std::size_t{c.n} * kH * kV);
    Dev<std::uint8_t> d_scratch(mla::NonAbsorbedScratchBytes(c.n));
    const auto got = RunNa(w, d_cache.get(), c.first, c.n, d_scratch, d_qr,
                           d_out, d_q.get(), d_gate.get());
    std::vector<std::uint32_t> tokens;
    for (const std::uint32_t t :
         {0U, 1U, 2U, c.n / 3, c.n / 2, c.n - 2, c.n - 1}) {
      if (t < c.n && std::find(tokens.begin(), tokens.end(), t) == tokens.end())
        tokens.push_back(t);
    }
    if (c.first < 2000) {
      for (const std::uint32_t t : {15U, 16U, 17U, 31U, 32U, 33U, 63U, 64U}) {
        if (t < c.n &&
            std::find(tokens.begin(), tokens.end(), t) == tokens.end())
          tokens.push_back(t);
      }
    }
    std::vector<Err> errs(tokens.size());
    ParallelFor(tokens.size(), [&](std::size_t i) {
      const std::uint32_t t = tokens[i];
      const auto want = HostNaToken(w, &q[std::size_t{t} * kH * kQH],
                                    static_cast<std::int32_t>(c.first + t),
                                    cache.data(), c.first + t + 1);
      for (std::uint32_t h = 0; h < kH; ++h) {
        const double sg = 1.0 / (1.0 + std::exp(-static_cast<double>(
                                           gate[(std::size_t{t} * kH) + h])));
        for (std::uint32_t e = 0; e < kV; ++e) {
          errs[i].Add(got[(std::size_t{t} * kH * kV) + (h * kV) + e],
                      want[(std::size_t{h} * kV) + e] * sg);
        }
      }
    });
    Err err;
    for (const Err& e : errs) {
      err.max_abs = std::max(err.max_abs, e.max_abs);
      err.max_ref = std::max(err.max_ref, e.max_ref);
    }
    Report("NonAbsorbed prefill T=" + std::to_string(c.n) +
               " first_pos=" + std::to_string(c.first),
           err, 3e-3);
  }
}

// Bitwise split invariance of the non-absorbed prefill: one chunk of the whole
// run (transient expansion only) against two and three chunks, at every split
// point (all points for short runs, a spread incl. tile / key-block edges for
// long ones), from first positions that put chunk boundaries at odd and even
// offsets of a 16-key tile. Every layout also changes where the expanded rows
// live: none, 2048 or 4096 persistent rows (expanded incrementally like the
// engine does), so the transient / persistent equivalence and the fold over
// key blocks are covered together. The cache beyond a part's end holds the
// other part's rows, so masked partner keys differ between the layouts (the
// WMMA key-pair quirk).
void TestNonAbsorbedInvariance() {
  NaWeights w(81);
  constexpr std::size_t kTok = kH * kV;
  struct Case {
    std::uint32_t base;
    std::uint32_t total;
    bool all;
  };
  constexpr std::uint32_t kStoreRows[3] = {0, 2048, 4096};
  Dev<__half> stores[3] = {Dev<__half>(1),
                           Dev<__half>(mla::ExpandedLayerBytes(2048) / 2),
                           Dev<__half>(mla::ExpandedLayerBytes(4096) / 2)};
  for (const Case c :
       {Case{0, 106, true}, Case{0, 97, true}, Case{0, 160, true},
        Case{0, 33, true}, Case{0, 300, true}, Case{1000, 97, true},
        Case{2025, 70, true}, Case{4090, 40, true}, Case{0, 2100, false},
        Case{3000, 300, false}, Case{0, 2048, false},
        Case{4000, 2100, false}}) {
    const std::uint32_t rows = c.base + c.total;
    const std::vector<__half> cache = MakeCache(rows, 191 + rows);
    const std::vector<float> q =
        Normal(std::size_t{c.total} * kH * kQH, 192 + rows, 1.7F);
    const std::vector<float> gate =
        Normal(std::size_t{c.total} * kH, 193, 1.5F);
    Dev<__half> d_cache(cache);
    Dev<float> d_q(q);
    Dev<float> d_gate(gate);
    Dev<float> d_qr(std::size_t{c.total} * kH * kQH);
    Dev<float> d_one(std::size_t{c.total} * kTok);
    Dev<float> d_two(std::size_t{c.total} * kTok);
    Dev<std::uint8_t> d_scratch(mla::NonAbsorbedScratchBytes(c.total));
    const auto run = [&](Dev<float>& dst, std::uint32_t t0, std::uint32_t n,
                         int cfg) {
      std::vector<std::int32_t> pos(n);
      for (std::uint32_t t = 0; t < n; ++t) {
        pos[t] = static_cast<std::int32_t>(c.base + t0 + t);
      }
      Dev<std::int32_t> d_pos(pos);
      mla::QPrepare(d_q.get() + (std::size_t{t0} * kH * kQH), d_pos.get(),
                    d_qr.get() + (std::size_t{t0} * kH * kQH), n, nullptr);
      if (kStoreRows[cfg] > 0) {
        mla::ExpandRows(d_cache.get(), w.d_w16.get(), stores[cfg].get(),
                        kStoreRows[cfg], t0 == 0 ? 0 : c.base + t0,
                        c.base + t0 + n, nullptr);
      }
      mla::PrefillAttentionNonAbsorbed(
          d_qr.get() + (std::size_t{t0} * kH * kQH), d_cache.get(),
          w.d_w16.get(), stores[cfg].get(), kStoreRows[cfg],
          d_gate.get() + (std::size_t{t0} * kH), c.base + t0, n,
          dst.get() + (std::size_t{t0} * kTok), d_scratch.get(), nullptr);
    };
    run(d_one, 0, c.total, 0);
    CheckHip(hipDeviceSynchronize(), "sync");
    const auto one = d_one.Download();
    std::vector<std::uint32_t> splits;
    if (c.all) {
      for (std::uint32_t s = 1; s < c.total; ++s)
        splits.push_back(s);
    } else {
      for (const std::uint32_t s :
           {1U,    8U,    9U,    15U,   16U,   17U,   31U,   33U,
            63U,   64U,   65U,   100U,  1000U, 2023U, 2024U, 2025U,
            2047U, 2048U, 2049U, 2087U, 2088U, 2089U}) {
        if (s < c.total)
          splits.push_back(s);
      }
    }
    std::size_t bad = 0;
    std::uint32_t first_bad = 0;
    for (std::size_t i = 0; i < splits.size(); ++i) {
      const std::uint32_t s = splits[i];
      const int cfg = static_cast<int>(i % 3);
      run(d_two, 0, s, cfg);
      run(d_two, s, c.total - s, cfg);
      CheckHip(hipDeviceSynchronize(), "sync");
      const auto two = d_two.Download();
      if (CountDiff(one.data(), two.data(), one.size()) != 0) {
        if (bad++ == 0)
          first_bad = s;
      }
    }
    // Three chunks (a, b, rest), store layouts rotating.
    std::size_t bad3 = 0;
    std::size_t tried3 = 0;
    for (const std::uint32_t a : {9U, 16U, 33U, 100U}) {
      for (const std::uint32_t b : {9U, 17U, 64U, 1000U}) {
        if (a + b >= c.total)
          continue;
        const int cfg = static_cast<int>(tried3++ % 3);
        run(d_two, 0, a, cfg);
        run(d_two, a, b, cfg);
        run(d_two, a + b, c.total - a - b, cfg);
        CheckHip(hipDeviceSynchronize(), "sync");
        const auto two = d_two.Download();
        bad3 += CountDiff(one.data(), two.data(), one.size()) != 0;
      }
    }
    const std::string label =
        "base=" + std::to_string(c.base) + " N=" + std::to_string(c.total);
    std::printf("%-44s differing splits=%zu of %zu (first %u) %s\n",
                ("NonAbsorbed split == single, " + label).c_str(), bad,
                splits.size(), first_bad, bad == 0 ? "PASS" : "FAIL");
    std::printf("%-44s differing layouts=%zu of %zu %s\n",
                ("NonAbsorbed 3 chunks == single, " + label).c_str(), bad3,
                tried3, bad3 == 0 ? "PASS" : "FAIL");
    g_failures += (bad == 0 ? 0 : 1) + (bad3 == 0 ? 0 : 1);
  }
}

std::uint64_t Fnv(const void* p, std::size_t bytes) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < bytes; ++i)
    h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

// Output bits of the non-absorbed prefill against the parent kernels (golden
// digests): chunks of 9..130 tokens (partially live query waves), from first
// positions on and across 2048-key block edges, with the keys in the
// persistent store, in the transient buffer, or split between them.
void TestNonAbsorbedParentDigest() {
  NaWeights w(91);
  for (const auto& c : parent_digests::kMlaPrefill) {
    if (c.n == 0)
      break;
    const std::uint32_t total = c.first + c.n;
    const std::vector<__half> cache = MakeCache(total, 301 + total);
    const std::vector<float> q =
        Normal(std::size_t{c.n} * kH * kQH, 302 + c.n, 1.7F);
    const std::vector<float> gate = Normal(std::size_t{c.n} * kH, 303, 1.5F);
    Dev<__half> d_cache(cache);
    Dev<float> d_q(q);
    Dev<float> d_gate(gate);
    Dev<float> d_qr(std::size_t{c.n} * kH * kQH);
    Dev<float> d_out(std::size_t{c.n} * kH * kV);
    Dev<std::uint8_t> d_scratch(mla::NonAbsorbedScratchBytes(c.n));
    Dev<__half> store(
        c.store_rows == 0 ? 1 : mla::ExpandedLayerBytes(c.store_rows) / 2);
    const auto got =
        RunNa(w, d_cache.get(), c.first, c.n, d_scratch, d_qr, d_out, d_q.get(),
              d_gate.get(), store.get(), c.store_rows);
    const auto digest = static_cast<unsigned long long>(
        Fnv(got.data(), got.size() * sizeof(float)));
    const bool ok = digest == c.digest;
    std::printf(
        "%-4s NonAbsorbed == parent digest first=%u n=%u store=%u "
        "(%016llx)\n",
        ok ? "PASS" : "FAIL", c.first, c.n, c.store_rows, digest);
    if (!ok)
      ++g_failures;
  }
}

void TestVbUpProjGate() {
  const Q8Weights vb = MakeQ8(kV, kLat, 51, 0.05F);
  Dev<std::uint8_t> d_vb(vb.bytes);
  for (const std::uint32_t n : {1U, 40U}) {
    const std::vector<float> lat =
        Normal(std::size_t{n} * kH * kLat, 52 + n, 0.5F);
    const std::vector<float> gate = Normal(std::size_t{n} * kH, 53 + n, 2.0F);
    Dev<float> d_lat(lat);
    Dev<float> d_gate(gate);
    Dev<float> d_out(std::size_t{n} * kH * kV);
    mla::VbUpProjGate(d_lat.get(), d_vb.get(), d_gate.get(), d_out.get(), n,
                      nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
    const auto got = d_out.Download();
    Err err;
    for (std::uint32_t t = 0; t < n; ++t) {
      for (std::uint32_t h = 0; h < kH; ++h) {
        const double sg =
            1.0 / (1.0 + std::exp(-static_cast<double>(gate[(t * kH) + h])));
        const float* l = &lat[((std::size_t{t} * kH) + h) * kLat];
        for (std::uint32_t d = 0; d < kV; ++d) {
          const float* w = &vb.dequant[((std::size_t{h} * kV) + d) * kLat];
          double acc = 0;
          for (std::uint32_t c = 0; c < kLat; ++c) {
            acc += static_cast<double>(l[c]) * w[c];
          }
          err.Add(got[(std::size_t{t} * kH * kV) + (h * kV) + d], acc * sg);
        }
      }
    }
    Report("VbUpProjGate T=" + std::to_string(n), err, 1e-5);
  }
}

// One MLA layer core: absorbed GPU path (prefill chunk, then one decode
// token) against a host NON-absorbed reference that expands k and v per head.
void TestLayerCoreVsNonAbsorbed() {
  const std::uint32_t n_pre = 40;
  const std::uint32_t total = n_pre + 1;
  const std::vector<float> kv_a = Normal(std::size_t{total} * kW, 61, 2.0F);
  std::vector<float> gamma = Normal(kLat, 62, 0.2F);
  for (float& g : gamma) {
    g += 1.0F;
  }
  const std::vector<float> q = Normal(std::size_t{total} * kH * kQH, 63, 2.0F);
  const std::vector<float> gate = Normal(std::size_t{total} * kH, 64, 1.0F);
  const Q8Weights kb = MakeQ8(kLat, kNope, 65, 0.04F);
  const Q8Weights vb = MakeQ8(kV, kLat, 66, 0.04F);
  std::vector<std::int32_t> pos(total);
  for (std::uint32_t t = 0; t < total; ++t) {
    pos[t] = static_cast<std::int32_t>(t);
  }

  Dev<float> d_kv_a(kv_a);
  Dev<float> d_gamma(gamma);
  Dev<float> d_q(q);
  Dev<float> d_gate(gate);
  Dev<std::uint8_t> d_kb(kb.bytes);
  Dev<std::uint8_t> d_vb(vb.bytes);
  Dev<std::int32_t> d_pos(pos);
  Dev<float> d_kv(std::size_t{total} * kW);
  Dev<__half> d_cache(std::size_t{128} * kW);
  Dev<float> d_qabs(std::size_t{total} * kH * kW);
  Dev<float> d_lat(std::size_t{total} * kH * kLat);
  Dev<float> d_out(std::size_t{total} * kH * kV);
  Dev<std::uint8_t> d_scratch(mla::DecodeScratchBytes(128));
  // Non-absorbed prefill of the same 40-token chunk (engine path for n > 8).
  Dev<__half> d_w16(mla::ExpansionWeightBytes() / sizeof(__half));
  mla::BuildExpansionWeights(d_kb.get(), d_vb.get(), d_w16.get(), nullptr);
  Dev<__half> d_cache_na(std::size_t{128} * kW);
  const std::size_t last = n_pre;
  Dev<float> d_qr(std::size_t{total} * kH * kQH);
  Dev<float> d_out_na(std::size_t{total} * kH * kV);
  Dev<std::uint8_t> d_scratch_na(mla::NonAbsorbedScratchBytes(n_pre));

  // Prefill chunk (tokens 0..39), then the decode token 40.
  mla::LatentNormRope(d_kv_a.get(), d_gamma.get(), d_pos.get(), d_kv.get(),
                      n_pre, nullptr);
  mla::CacheAppend(d_cache.get(), d_kv.get(), d_pos.get(), n_pre, nullptr);
  mla::LatentNormRope(d_kv_a.get() + (last * kW), d_gamma.get(),
                      d_pos.get() + last, d_kv.get() + (last * kW), 1, nullptr);
  mla::CacheAppend(d_cache.get(), d_kv.get() + (last * kW), d_pos.get() + last,
                   1, nullptr);
  mla::QAbsorb(d_q.get() + (last * kH * kQH), d_kb.get(), d_pos.get() + last,
               d_qabs.get() + (last * kH * kW), 1, nullptr);
  mla::DecodeAttention(d_qabs.get() + (last * kH * kW), d_cache.get(),
                       d_pos.get() + last, d_lat.get() + (last * kH * kLat),
                       d_scratch.get(), nullptr);
  mla::VbUpProjGate(d_lat.get() + (last * kH * kLat), d_vb.get(),
                    d_gate.get() + (last * kH), d_out.get() + (last * kH * kV),
                    1, nullptr);
  mla::CacheAppend(d_cache_na.get(), d_kv.get(), d_pos.get(), n_pre, nullptr);
  mla::QPrepare(d_q.get(), d_pos.get(), d_qr.get(), n_pre, nullptr);
  mla::PrefillAttentionNonAbsorbed(d_qr.get(), d_cache_na.get(), d_w16.get(),
                                   nullptr, 0, d_gate.get(), 0, n_pre,
                                   d_out_na.get(), d_scratch_na.get(), nullptr);
  // Token 40 through the non-absorbed path too (n = 1 chunk at position 40).
  mla::CacheAppend(d_cache_na.get(), d_kv.get() + (last * kW),
                   d_pos.get() + last, 1, nullptr);
  mla::QPrepare(d_q.get() + (last * kH * kQH), d_pos.get() + last,
                d_qr.get() + (last * kH * kQH), 1, nullptr);
  mla::PrefillAttentionNonAbsorbed(
      d_qr.get() + (last * kH * kQH), d_cache_na.get(), d_w16.get(), nullptr, 0,
      d_gate.get() + (last * kH), n_pre, 1, d_out_na.get() + (last * kH * kV),
      d_scratch_na.get(), nullptr);
  CheckHip(hipDeviceSynchronize(), "sync");
  const auto got = d_out.Download();
  const auto got_na = d_out_na.Download();

  // Host: cache rows as stored (F16-rounded), then expand per head.
  std::vector<std::vector<double>> latent(total);
  std::vector<std::vector<double>> k_pe(total);
  for (std::uint32_t p = 0; p < total; ++p) {
    const auto row =
        HostLatentRow(&kv_a[std::size_t{p} * kW], gamma.data(), pos[p]);
    latent[p].resize(kLat);
    k_pe[p].resize(kRopeDim);
    for (std::uint32_t c = 0; c < kLat; ++c) {
      latent[p][c] = RoundHalf(row[c]);
    }
    for (std::uint32_t i = 0; i < kRopeDim; ++i) {
      k_pe[p][i] = RoundHalf(row[kLat + i]);
    }
  }
  // k_nope[p][h][d] = sum_c lat[c] k_b[h][c][d]; v[p][h][d] = sum_c lat[c]
  // v_b[h][d][c]
  std::vector<double> k_nope(std::size_t{total} * kH * kNope);
  std::vector<double> v_exp(std::size_t{total} * kH * kV);
  ParallelFor(total, [&](std::size_t p) {
    for (std::uint32_t h = 0; h < kH; ++h) {
      for (std::uint32_t c = 0; c < kLat; ++c) {
        const double l = latent[p][c];
        const float* kbr = &kb.dequant[((std::size_t{h} * kLat) + c) * kNope];
        for (std::uint32_t d = 0; d < kNope; ++d) {
          k_nope[((p * kH) + h) * kNope + d] += l * kbr[d];
        }
      }
      for (std::uint32_t d = 0; d < kV; ++d) {
        const float* vbr = &vb.dequant[((std::size_t{h} * kV) + d) * kLat];
        double acc = 0;
        for (std::uint32_t c = 0; c < kLat; ++c) {
          acc += latent[p][c] * vbr[c];
        }
        v_exp[((p * kH) + h) * kV + d] = acc;
      }
    }
  });

  Err err_decode;
  Err err_na;
  Err err_na_vs_abs;
  for (std::uint32_t t = 0; t < total; ++t) {
    for (std::uint32_t h = 0; h < kH; ++h) {
      const float* qh = &q[((std::size_t{t} * kH) + h) * kQH];
      std::vector<double> pe(kRopeDim);
      std::vector<double> rot(kRopeDim);
      for (std::uint32_t i = 0; i < kRopeDim; ++i) {
        pe[i] = qh[kNope + i];
      }
      HostRope(pe.data(), rot.data(), pos[t]);
      std::vector<double> s(t + 1);
      double mx = -1e300;
      for (std::uint32_t p = 0; p <= t; ++p) {
        double acc = 0;
        for (std::uint32_t d = 0; d < kNope; ++d) {
          acc +=
              static_cast<double>(qh[d]) * k_nope[((p * kH) + h) * kNope + d];
        }
        for (std::uint32_t i = 0; i < kRopeDim; ++i) {
          acc += rot[i] * k_pe[p][i];
        }
        s[p] = acc / std::sqrt(192.0);
        mx = std::max(mx, s[p]);
      }
      double sum = 0;
      for (double& x : s) {
        x = std::exp(x - mx);
        sum += x;
      }
      const double sg =
          1.0 / (1.0 + std::exp(-static_cast<double>(gate[(t * kH) + h])));
      for (std::uint32_t d = 0; d < kV; ++d) {
        double acc = 0;
        for (std::uint32_t p = 0; p <= t; ++p) {
          acc += s[p] / sum * v_exp[((p * kH) + h) * kV + d];
        }
        const std::size_t at = (std::size_t{t} * kH * kV) + (h * kV) + d;
        if (t == n_pre) {
          err_decode.Add(got[at], acc * sg);
          err_na_vs_abs.Add(got_na[at], got[at]);
        }
        err_na.Add(got_na[at], acc * sg);
      }
    }
  }
  Report("Layer core absorbed(decode) vs non-absorbed", err_decode, 1e-3);
  Report("Layer core non-absorbed GPU vs host", err_na, 3e-3);
  Report("Layer core non-absorbed vs absorbed decode (GPU)", err_na_vs_abs,
         3e-3);
}

}  // namespace

int main() {
  // Before the first HIP call: a missing $TMPDIR crashes the runtime.
  if (!gufo::core::RequireTempDirectory())
    return 1;
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cerr << "no HIP device; skipping\n";
    return 77;
  }
  try {
    TestLatentNormRopeAndAppend();
    TestQAbsorb();
    TestVbUpProjGate();
    TestDecode();
    TestNonAbsorbedPrefill();
    TestNonAbsorbedInvariance();
    TestNonAbsorbedParentDigest();
    TestLayerCoreVsNonAbsorbed();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
