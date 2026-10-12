// Op tests of the Ling-3.0-flash MLA kernels (absorbed form) against scalar
// host references computed in double, fixed seeds. Prints max abs / relative
// errors (relative = max abs error / max |reference|). Exit 77 without a GPU.
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
    mla::CacheAppend(layer1, d_out.get(), first_pos, n, nullptr);
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
    mla::DecodeAttention(d_q.get(), d_cache.get() + layer_elems, context,
                         d_out.get(), d_scratch.get(), nullptr);
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

void TestPrefill() {
  struct Case {
    std::uint32_t n;
    std::uint32_t first;
  };
  for (const Case c : {Case{1, 0}, Case{1, 1000}, Case{64, 0}, Case{64, 1000},
                       Case{512, 0}, Case{512, 1000}}) {
    const std::uint32_t total = c.first + c.n;
    const std::vector<__half> cache = MakeCache(total, 41 + total);
    const std::vector<float> q =
        Normal(std::size_t{c.n} * kH * kW, 42 + c.n + c.first, 1.7F);
    Dev<__half> d_cache(cache);
    Dev<float> d_q(q);
    Dev<float> d_out(std::size_t{c.n} * kH * kLat);
    mla::PrefillAttention(d_q.get(), d_cache.get(), c.first, c.n, d_out.get(),
                          nullptr, nullptr);
    CheckHip(hipDeviceSynchronize(), "sync");
    const auto got = d_out.Download();
    // Every token for small chunks, a spread sample (incl. tile edges) else.
    std::vector<std::uint32_t> tokens;
    if (c.n <= 64) {
      for (std::uint32_t t = 0; t < c.n; ++t) {
        tokens.push_back(t);
      }
    } else {
      for (const std::uint32_t t :
           {0U, 1U, 15U, 16U, 17U, 100U, 255U, 256U, 301U, 480U, 510U, 511U}) {
        tokens.push_back(t);
      }
    }
    std::vector<Err> errs(tokens.size());
    ParallelFor(tokens.size(), [&](std::size_t i) {
      const std::uint32_t t = tokens[i];
      std::vector<double> want;
      HostAbsorbedAttention(&q[std::size_t{t} * kH * kW], cache.data(),
                            c.first + t + 1, &want);
      for (std::size_t j = 0; j < want.size(); ++j) {
        errs[i].Add(got[(std::size_t{t} * kH * kLat) + j], want[j]);
      }
    });
    Err err;
    for (const Err& e : errs) {
      err.max_abs = std::max(err.max_abs, e.max_abs);
      err.max_ref = std::max(err.max_ref, e.max_ref);
    }
    Report("PrefillAttention T=" + std::to_string(c.n) +
               " first_pos=" + std::to_string(c.first),
           err, 1.5e-3);

    if (c.n == 512) {
      // Chunked prefill must equal the single launch (same tile sweep).
      Dev<float> d_chunked(std::size_t{c.n} * kH * kLat);
      for (std::uint32_t t0 = 0; t0 < c.n; t0 += 64) {
        mla::PrefillAttention(d_q.get() + (std::size_t{t0} * kH * kW),
                              d_cache.get(), c.first + t0, 64,
                              d_chunked.get() + (std::size_t{t0} * kH * kLat),
                              nullptr, nullptr);
      }
      CheckHip(hipDeviceSynchronize(), "sync");
      const auto chunked = d_chunked.Download();
      Err same;
      for (std::size_t i = 0; i < got.size(); ++i) {
        same.Add(chunked[i], got[i]);
      }
      std::printf(
          "%-44s max_abs_diff=%.3e %s\n",
          ("Prefill chunked(64) == single, first=" + std::to_string(c.first))
              .c_str(),
          same.max_abs, same.max_abs == 0.0 ? "PASS" : "FAIL");
      g_failures += same.max_abs == 0.0 ? 0 : 1;
    }
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

  // Prefill chunk (tokens 0..39), then the decode token 40.
  mla::LatentNormRope(d_kv_a.get(), d_gamma.get(), d_pos.get(), d_kv.get(),
                      n_pre, nullptr);
  mla::CacheAppend(d_cache.get(), d_kv.get(), 0, n_pre, nullptr);
  mla::QAbsorb(d_q.get(), d_kb.get(), d_pos.get(), d_qabs.get(), n_pre,
               nullptr);
  mla::PrefillAttention(d_qabs.get(), d_cache.get(), 0, n_pre, d_lat.get(),
                        nullptr, nullptr);
  mla::VbUpProjGate(d_lat.get(), d_vb.get(), d_gate.get(), d_out.get(), n_pre,
                    nullptr);
  const std::size_t last = n_pre;
  mla::LatentNormRope(d_kv_a.get() + (last * kW), d_gamma.get(),
                      d_pos.get() + last, d_kv.get() + (last * kW), 1, nullptr);
  mla::CacheAppend(d_cache.get(), d_kv.get() + (last * kW), n_pre, 1, nullptr);
  mla::QAbsorb(d_q.get() + (last * kH * kQH), d_kb.get(), d_pos.get() + last,
               d_qabs.get() + (last * kH * kW), 1, nullptr);
  mla::DecodeAttention(d_qabs.get() + (last * kH * kW), d_cache.get(), total,
                       d_lat.get() + (last * kH * kLat), d_scratch.get(),
                       nullptr);
  mla::VbUpProjGate(d_lat.get() + (last * kH * kLat), d_vb.get(),
                    d_gate.get() + (last * kH), d_out.get() + (last * kH * kV),
                    1, nullptr);
  CheckHip(hipDeviceSynchronize(), "sync");
  const auto got = d_out.Download();

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

  Err err_prefill;
  Err err_decode;
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
        (t < n_pre ? err_prefill : err_decode)
            .Add(got[(std::size_t{t} * kH * kV) + (h * kV) + d], acc * sg);
      }
    }
  }
  Report("Layer core absorbed(prefill) vs non-absorbed", err_prefill, 1e-3);
  Report("Layer core absorbed(decode) vs non-absorbed", err_decode, 1e-3);
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
    TestPrefill();
    TestLayerCoreVsNonAbsorbed();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
