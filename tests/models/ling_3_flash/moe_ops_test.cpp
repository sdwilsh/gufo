// Op tests of the Ling-3.0-flash MoE kernels against scalar host references.
// Expert weights are random but valid quant blocks; references dequantise with
// the repo's CPU dequantisers (src/core/quant/ggml_dequant.*) and accumulate in
// double. Fixed seeds. Exit 77 when no GPU is present.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/runtime_environment.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"
#include "src/models/ling_3_flash/layout.hpp"
#include "tests/models/ling_3_flash/parent_digests.hpp"

namespace {

namespace rocm = gufo::models::ling_3_flash::rocm;
namespace quant = gufo::quant;
using rocm::WeightType;

constexpr int kHidden = 2560;
constexpr int kFf = 768;
constexpr int kUsed = 8;
constexpr int kExperts = 512;

void CheckHip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
  std::uint64_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float Uniform() {
    return static_cast<float>((Next() >> 40) * (1.0 / 16777216.0));
  }
  float Normal() {
    const float u1 = std::max(Uniform(), 1e-7F), u2 = Uniform();
    return std::sqrt(-2.0F * std::log(u1)) * std::cos(6.2831853F * u2);
  }
};

template<typename T>
class Dev {
public:
  explicit Dev(std::size_t count) : count_(count) {
    CheckHip(hipMalloc(&p_, std::max<std::size_t>(count * sizeof(T), 16)),
             "hipMalloc");
  }
  Dev(const std::vector<T>& v) : Dev(v.size()) {
    CheckHip(
        hipMemcpy(p_, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
        "H2D");
  }
  Dev(const Dev&) = delete;
  ~Dev() { (void)hipFree(p_); }
  T* get() const { return static_cast<T*>(p_); }
  std::vector<T> Read() const {
    std::vector<T> v(count_);
    CheckHip(hipMemcpy(v.data(), p_, count_ * sizeof(T), hipMemcpyDeviceToHost),
             "D2H");
    return v;
  }

private:
  void* p_{nullptr};
  std::size_t count_;
};

// ---- quant helpers --------------------------------------------------------
std::size_t BlockBytes(WeightType t) {
  switch (t) {
    case WeightType::kIQ4_XS:
      return 136;
    case WeightType::kQ4_K:
      return 144;
    case WeightType::kQ5_K:
      return 176;
    case WeightType::kQ6_K:
      return 210;
    default:
      throw std::runtime_error("type");
  }
}
const char* Name(WeightType t) {
  switch (t) {
    case WeightType::kIQ4_XS:
      return "IQ4_XS";
    case WeightType::kQ4_K:
      return "Q4_K";
    case WeightType::kQ5_K:
      return "Q5_K";
    case WeightType::kQ6_K:
      return "Q6_K";
    default:
      return "?";
  }
}
void DequantRow(WeightType t, const void* src, float* dst, std::size_t k) {
  switch (t) {
    case WeightType::kIQ4_XS:
      quant::DequantizeIQ4_XS(src, dst, k);
      break;
    case WeightType::kQ4_K:
      quant::DequantizeQ4_K(src, dst, k);
      break;
    case WeightType::kQ5_K:
      quant::DequantizeQ5_K(src, dst, k);
      break;
    case WeightType::kQ6_K:
      quant::DequantizeQ6_K(src, dst, k);
      break;
    default:
      throw std::runtime_error("type");
  }
}
std::uint16_t HalfBits(float f) {
  return std::bit_cast<std::uint16_t>(static_cast<_Float16>(f));
}
void SetScales(WeightType t, std::uint8_t* blk, float d, float dmin) {
  std::size_t off_d = 0, off_m = 2;
  if (t == WeightType::kQ6_K) {
    off_d = 208;
  }
  const std::uint16_t hd = HalfBits(d), hm = HalfBits(dmin);
  std::memcpy(blk + off_d, &hd, 2);
  if (t == WeightType::kQ4_K || t == WeightType::kQ5_K) {
    std::memcpy(blk + off_m, &hm, 2);
  }
}

/// Random bytes with sane fp16 super-scales; `base` multiplies the scales.
void FillBlocks(WeightType t, std::uint8_t* dst, std::size_t blocks, float base,
                Rng& rng) {
  const std::size_t bb = BlockBytes(t);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* p = dst + b * bb;
    for (std::size_t i = 0; i < bb; i += 8) {
      const std::uint64_t r = rng.Next();
      std::memcpy(p + i, &r, std::min<std::size_t>(8, bb - i));
    }
    SetScales(t, p, base * (0.5F + rng.Uniform()),
              base * (0.5F + rng.Uniform()));
  }
}

/// Scale so that dequantised weights have rms ~= target.
float BaseFor(WeightType t, float target) {
  Rng rng(7);
  std::vector<std::uint8_t> buf(64 * 10 * BlockBytes(t));
  FillBlocks(t, buf.data(), 64 * 10, 1e-3F, rng);
  std::vector<float> row(2560);
  double ss = 0;
  for (int r = 0; r < 64; ++r) {
    DequantRow(t, buf.data() + r * 10 * BlockBytes(t), row.data(), 2560);
    for (float v : row)
      ss += static_cast<double>(v) * v;
  }
  const double rms = std::sqrt(ss / (64.0 * 2560));
  return static_cast<float>(1e-3 * target / rms);
}

struct Experts {
  WeightType type;
  int count, rows, k;
  std::vector<std::uint8_t> bytes;
  std::map<int, std::vector<float>> cache;
  std::size_t RowBytes() const { return (k / 256) * BlockBytes(type); }
  Experts(WeightType t, int n, int rows_, int k_, float target_rms,
          std::uint64_t seed)
      : type(t), count(n), rows(rows_), k(k_) {
    Rng rng(seed);
    bytes.resize(static_cast<std::size_t>(n) * rows * RowBytes());
    FillBlocks(t, bytes.data(), bytes.size() / BlockBytes(t),
               BaseFor(t, target_rms), rng);
  }
  const std::vector<float>& Dequant(int e) {
    auto it = cache.find(e);
    if (it != cache.end())
      return it->second;
    std::vector<float> m(static_cast<std::size_t>(rows) * k);
    for (int r = 0; r < rows; ++r) {
      DequantRow(
          type,
          bytes.data() + (static_cast<std::size_t>(e) * rows + r) * RowBytes(),
          m.data() + static_cast<std::size_t>(r) * k, k);
    }
    return cache.emplace(e, std::move(m)).first->second;
  }
};

void ParallelFor(int n, const std::function<void(int)>& fn) {
  const int nt =
      std::min<int>(4, std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (int w = 0; w < nt; ++w) {
    th.emplace_back([&, w] {
      for (int i = w; i < n; i += nt)
        fn(i);
    });
  }
  for (auto& t : th)
    t.join();
}

double SwigluRef(double g, double u, double limit) {
  double a = g / (1.0 + std::exp(-g));
  if (limit != 0.0) {
    a = std::min(a, limit);
    u = std::min(std::max(u, -limit), limit);
  }
  return a * u;
}

int g_failures = 0;
void Report(const std::string& name, double max_abs, double rms_ref, double tol,
            std::size_t checked) {
  const double rel = max_abs / std::max(rms_ref, 1e-12);
  const bool ok = rel <= tol && std::isfinite(rel);
  std::printf(
      "%-52s %s  max_abs=%.3e  rel_to_rms=%.3e  (tol %.1e, %zu elems)\n",
      name.c_str(), ok ? "PASS" : "FAIL", max_abs, rel, tol, checked);
  if (!ok)
    ++g_failures;
}

std::vector<int> SampleTokens(int n, int max_checked, Rng& rng) {
  std::vector<int> v;
  if (n <= max_checked) {
    v.resize(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
  }
  std::set<int> s{0, n - 1};
  while (static_cast<int>(s.size()) < max_checked)
    s.insert(rng.Next() % n);
  return {s.begin(), s.end()};
}

/// ids[t][8]: 8 distinct experts per token, drawn from [0, pool).
/// mode 1: every token selects the same 8 experts.
std::vector<std::int32_t> MakeIds(int n, int pool, int mode, Rng& rng) {
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n) * kUsed);
  std::vector<int> first;
  for (int t = 0; t < n; ++t) {
    std::vector<int> ch;
    if (mode == 1 && t > 0) {
      ch = first;
    } else {
      while (static_cast<int>(ch.size()) < kUsed) {
        // skew toward low experts so buckets are uneven
        const int e = static_cast<int>(
            (rng.Next() % 3 == 0 ? rng.Next() % pool
                                 : rng.Next() % std::max(1, pool / 4)));
        if (std::find(ch.begin(), ch.end(), e) == ch.end())
          ch.push_back(e);
      }
      if (t == 0)
        first = ch;
    }
    std::copy(ch.begin(), ch.end(),
              ids.begin() + static_cast<std::size_t>(t) * kUsed);
  }
  return ids;
}

// ---- route ----------------------------------------------------------------
struct RouteRef {
  std::int32_t ids[8];
  float w[8];
};
RouteRef RouteHost(const float* lg, const float* bias) {
  float p[512], sel[512];
  for (int i = 0; i < 512; ++i) {
    p[i] = 1.0F / (1.0F + std::exp(-lg[i]));
    sel[i] = p[i] + bias[i];
  }
  float gs[8];
  for (int g = 0; g < 8; ++g) {
    float a = -INFINITY, b = -INFINITY;
    for (int i = 0; i < 64; ++i) {
      const float v = sel[g * 64 + i];
      if (v > a) {
        b = a;
        a = v;
      } else if (v > b) {
        b = v;
      }
    }
    gs[g] = a + b;
  }
  std::vector<int> go(8);
  std::iota(go.begin(), go.end(), 0);
  std::stable_sort(go.begin(), go.end(),
                   [&](int x, int y) { return gs[x] > gs[y]; });
  std::vector<int> cand;
  for (int r = 0; r < 4; ++r)
    for (int i = 0; i < 64; ++i)
      cand.push_back(go[r] * 64 + i);
  std::stable_sort(cand.begin(), cand.end(), [&](int x, int y) {
    return sel[x] > sel[y] || (sel[x] == sel[y] && x < y);
  });
  RouteRef r{};
  float sum = 0;
  for (int k = 0; k < 8; ++k) {
    r.ids[k] = cand[k];
    sum += p[cand[k]];
  }
  const float denom = std::max(sum, 6.1035e-5F);
  for (int k = 0; k < 8; ++k)
    r.w[k] = p[cand[k]] / denom * 2.5F;
  return r;
}

void TestRoute(int n, hipStream_t st) {
  Rng rng(1000 + n);
  std::vector<float> lg(static_cast<std::size_t>(n) * 512), bias(512);
  for (auto& v : bias)
    v = 1.5F * rng.Normal();
  for (auto& v : lg)
    v = 2.0F * rng.Normal();
  const std::string tag = "route n=" + std::to_string(n);
  // Row classes (cycled): random; heavy ties (3 distinct values, zero bias
  // handled by a separate bias vector below); all-equal; group inversion.
  std::vector<float> zero_bias(512, 0.0F);
  std::vector<float> bias_flat(512, 0.0F);
  int tie_rows = 0;
  std::vector<int> row_bias(n, 0);  // 0 = random bias, 1 = zero bias
  for (int t = 0; t < n; ++t) {
    float* row = lg.data() + static_cast<std::size_t>(t) * 512;
    switch (t % 5) {
      case 1:  // ties: logits from {-1,0,1}, zero bias
        for (int i = 0; i < 512; ++i)
          row[i] = static_cast<float>(static_cast<int>(rng.Next() % 3) - 1);
        row_bias[t] = 1;
        ++tie_rows;
        break;
      case 2:  // all equal, zero bias: ids must be 0..7
        for (int i = 0; i < 512; ++i)
          row[i] = 0.25F;
        row_bias[t] = 1;
        ++tie_rows;
        break;
      case 3:  // group 0 has the best unbiased logits but a very negative bias
        for (int i = 0; i < 512; ++i)
          row[i] = (i < 64) ? 3.0F + 0.01F * rng.Normal() : 0.5F * rng.Normal();
        row_bias[t] = 2;
        break;
      default:
        break;
    }
  }
  // Router has one bias vector; run the three bias classes as separate calls.
  std::vector<float> bias_inv(512, 0.0F);
  for (int i = 0; i < 64; ++i)
    bias_inv[i] = -3.0F;
  Dev<float> dl(lg);
  std::vector<std::int32_t> got_ids(static_cast<std::size_t>(n) * 8);
  std::vector<float> got_w(static_cast<std::size_t>(n) * 8);
  const std::vector<float>* biases[3] = {&bias, &zero_bias, &bias_inv};
  int bad_ids = 0;
  double max_w = 0;
  for (int cls = 0; cls < 3; ++cls) {
    Dev<float> db(*biases[cls]);
    Dev<std::int32_t> di(static_cast<std::size_t>(n) * 8);
    Dev<float> dw(static_cast<std::size_t>(n) * 8);
    rocm::moe::Route(dl.get(), db.get(), di.get(), dw.get(), n, st);
    CheckHip(hipStreamSynchronize(st), "route sync");
    const auto ids = di.Read();
    const auto w = dw.Read();
    for (int t = 0; t < n; ++t) {
      if (row_bias[t] != cls)
        continue;
      const RouteRef ref = RouteHost(
          lg.data() + static_cast<std::size_t>(t) * 512, biases[cls]->data());
      for (int k = 0; k < 8; ++k) {
        if (ids[t * 8 + k] != ref.ids[k])
          ++bad_ids;
        max_w = std::max(
            max_w, static_cast<double>(std::fabs(w[t * 8 + k] - ref.w[k])));
      }
    }
  }
  std::printf("%-52s %s  id_mismatches=%d  max_weight_err=%.3e (tie rows %d)\n",
              tag.c_str(), (bad_ids == 0 && max_w <= 1e-6) ? "PASS" : "FAIL",
              bad_ids, max_w, tie_rows);
  if (bad_ids != 0 || max_w > 1e-6)
    ++g_failures;
}

// ---- experts --------------------------------------------------------------
struct Setup {
  Experts gate, up, down;
  Dev<std::uint8_t> dgate, dup, ddown;
  Setup(WeightType gu, WeightType dn, int e, std::uint64_t seed, float gu_rms)
      : gate(gu, e, kFf, kHidden, gu_rms, seed),
        up(gu, e, kFf, kHidden, gu_rms, seed + 1),
        down(dn, e, kHidden, kFf, 0.02F, seed + 2),
        dgate(gate.bytes),
        dup(up.bytes),
        ddown(down.bytes) {}
};

void TestExperts(Setup& s, int n, int pool, int mode, float limit, bool decode,
                 int max_check, hipStream_t st, const std::string& label) {
  Rng rng(5000 + n * 7 + pool + mode);
  const auto ids = MakeIds(n, pool, mode, rng);
  std::vector<float> x(static_cast<std::size_t>(n) * kHidden);
  for (auto& v : x)
    v = rng.Normal();
  std::vector<float> act_in(static_cast<std::size_t>(n) * kUsed * kFf);
  for (auto& v : act_in)
    v = rng.Normal();
  Dev<float> dx(x), dact_in(act_in);
  Dev<std::int32_t> dids(ids);
  Dev<float> dact(act_in.size()),
      dout(static_cast<std::size_t>(n) * kUsed * kHidden);
  const std::size_t sb = decode ? rocm::moe::DecodeScratchBytes()
                                : rocm::moe::PrefillScratchBytes(n);
  Dev<std::uint8_t> scratch(sb);
  if (decode) {
    rocm::moe::ExpertGateUpDecode(s.dgate.get(), s.dup.get(), s.gate.type,
                                  dx.get(), dids.get(), n, limit, dact.get(),
                                  scratch.get(), st);
    rocm::moe::ExpertDownDecode(s.ddown.get(), s.down.type, dact_in.get(),
                                dids.get(), n, dout.get(), scratch.get(), st);
  } else {
    rocm::moe::ExpertGateUpPrefill(s.dgate.get(), s.dup.get(), s.gate.type,
                                   dx.get(), dids.get(), n, limit, dact.get(),
                                   scratch.get(), st);
    rocm::moe::ExpertDownPrefill(s.ddown.get(), s.down.type, dact_in.get(),
                                 dids.get(), n, dout.get(), scratch.get(), st);
  }
  CheckHip(hipStreamSynchronize(st), "expert sync");
  const auto act = dact.Read();
  const auto out = dout.Read();

  const auto toks = SampleTokens(n, max_check, rng);
  std::set<int> need;
  for (int t : toks)
    for (int k = 0; k < kUsed; ++k)
      need.insert(ids[t * kUsed + k]);
  for (int e : need) {
    s.gate.Dequant(e);
    s.up.Dequant(e);
    s.down.Dequant(e);
  }
  const int npairs = static_cast<int>(toks.size()) * kUsed;
  // Reference with weights/activations optionally rounded to F16 (the prefill
  // kernels stage F16 tiles; the emulation isolates real indexing/accumulation
  // bugs from that deliberate rounding).
  auto run_ref = [&](bool f16, const std::string& suffix, double tol) {
    auto rd = [f16](float v) -> double {
      return f16 ? static_cast<double>(
                       static_cast<float>(static_cast<_Float16>(v)))
                 : static_cast<double>(v);
    };
    std::vector<double> ref_act(static_cast<std::size_t>(npairs) * kFf);
    std::vector<double> ref_out(static_cast<std::size_t>(npairs) * kHidden);
    ParallelFor(npairs, [&](int pi) {
      const int t = toks[pi / kUsed], k = pi % kUsed;
      const int e = ids[t * kUsed + k];
      const auto& g = s.gate.cache.at(e);
      const auto& u = s.up.cache.at(e);
      const auto& d = s.down.cache.at(e);
      const float* xt = x.data() + static_cast<std::size_t>(t) * kHidden;
      for (int r = 0; r < kFf; ++r) {
        double a = 0, b = 0;
        for (int c = 0; c < kHidden; ++c) {
          const double xv = rd(xt[c]);
          a += rd(g[static_cast<std::size_t>(r) * kHidden + c]) * xv;
          b += rd(u[static_cast<std::size_t>(r) * kHidden + c]) * xv;
        }
        ref_act[static_cast<std::size_t>(pi) * kFf + r] =
            SwigluRef(a, b, limit);
      }
      const float* ai =
          act_in.data() + (static_cast<std::size_t>(t) * kUsed + k) * kFf;
      for (int r = 0; r < kHidden; ++r) {
        double a = 0;
        for (int c = 0; c < kFf; ++c)
          a += rd(d[static_cast<std::size_t>(r) * kFf + c]) * rd(ai[c]);
        ref_out[static_cast<std::size_t>(pi) * kHidden + r] = a;
      }
    });
    double ea = 0, eo = 0, sa = 0, so = 0;
    for (int pi = 0; pi < npairs; ++pi) {
      const std::size_t pr =
          static_cast<std::size_t>(toks[pi / kUsed]) * kUsed + pi % kUsed;
      for (int r = 0; r < kFf; ++r) {
        const double ref = ref_act[static_cast<std::size_t>(pi) * kFf + r];
        ea = std::max(ea, std::fabs(act[pr * kFf + r] - ref));
        sa += ref * ref;
      }
      for (int r = 0; r < kHidden; ++r) {
        const double ref = ref_out[static_cast<std::size_t>(pi) * kHidden + r];
        eo = std::max(eo, std::fabs(out[pr * kHidden + r] - ref));
        so += ref * ref;
      }
    }
    Report(label + " gate_up/" + Name(s.gate.type) + suffix, ea,
           std::sqrt(sa / (npairs * kFf)), tol,
           static_cast<std::size_t>(npairs) * kFf);
    Report(label + " down/" + Name(s.down.type) + suffix, eo,
           std::sqrt(so / (npairs * double(kHidden))), tol,
           static_cast<std::size_t>(npairs) * kHidden);
  };
  if (decode) {
    run_ref(false, "", 1e-4);
  } else {
    run_ref(true, " [f16-emu]", 5e-4);
    run_ref(false, " [f32-ref]", 2e-2);
  }
  s.gate.cache.clear();
  s.up.cache.clear();
  s.down.cache.clear();
}

void TestCombine(int n, hipStream_t st) {
  Rng rng(9000 + n);
  std::vector<float> res(static_cast<std::size_t>(n) * kHidden),
      eo(static_cast<std::size_t>(n) * 8 * kHidden),
      w(static_cast<std::size_t>(n) * 8), sh(res.size());
  for (auto* v : {&res, &eo, &w, &sh})
    for (auto& f : *v)
      f = rng.Normal();
  Dev<float> dres(res), deo(eo), dw(w), dsh(sh);
  rocm::moe::Combine(dres.get(), deo.get(), dw.get(), dsh.get(), n, st);
  CheckHip(hipStreamSynchronize(st), "combine");
  const auto got = dres.Read();
  double err = 0, ss = 0;
  for (int t = 0; t < n; ++t)
    for (int c = 0; c < kHidden; ++c) {
      double a = res[static_cast<std::size_t>(t) * kHidden + c] +
                 sh[static_cast<std::size_t>(t) * kHidden + c];
      for (int k = 0; k < 8; ++k)
        a += static_cast<double>(w[t * 8 + k]) *
             eo[(static_cast<std::size_t>(t) * 8 + k) * kHidden + c];
      err = std::max(
          err, std::fabs(got[static_cast<std::size_t>(t) * kHidden + c] - a));
      ss += a * a;
    }
  Report("combine n=" + std::to_string(n), err,
         std::sqrt(ss / (double(n) * kHidden)), 1e-5,
         static_cast<std::size_t>(n) * kHidden);
}

// ---- prefill: split / tile-variant invariance (bit-exact) -------------------
std::uint64_t Fnv(const void* p, std::size_t bytes) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < bytes; ++i)
    h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

/// Runs gate/up + down over `n` tokens in one call (default dispatch), then
/// again with every token tile variant, then as several chunks of the same
/// tokens. Every run must match the first one bit for bit, because an output
/// element depends only on its own token, expert, and weights.
void TestPrefillInvariance(Setup& s, int n, int pool, float limit,
                           hipStream_t st, const std::string& label) {
  Rng rng(7000 + n * 3 + pool);
  const auto ids = MakeIds(n, pool, 0, rng);
  std::vector<float> x(static_cast<std::size_t>(n) * kHidden);
  for (auto& v : x)
    v = rng.Normal();
  std::vector<float> act_in(static_cast<std::size_t>(n) * kUsed * kFf);
  for (auto& v : act_in)
    v = rng.Normal();
  Dev<float> dx(x), dact_in(act_in);
  Dev<std::int32_t> dids(ids);
  Dev<float> dact(act_in.size()),
      dout(static_cast<std::size_t>(n) * kUsed * kHidden);
  Dev<std::uint8_t> scratch(rocm::moe::PrefillScratchBytes(n));
  using Runner = std::function<void(int t0, int cnt, unsigned nf)>;
  const Runner run = [&](int t0, int cnt, unsigned nf) {
    rocm::moe::ExpertGateUpPrefillTile(
        s.dgate.get(), s.dup.get(), s.gate.type,
        dx.get() + static_cast<std::size_t>(t0) * kHidden,
        dids.get() + static_cast<std::size_t>(t0) * kUsed, cnt, limit,
        dact.get() + static_cast<std::size_t>(t0) * kUsed * kFf, scratch.get(),
        nf, st);
    rocm::moe::ExpertDownPrefillTile(
        s.ddown.get(), s.down.type,
        dact_in.get() + static_cast<std::size_t>(t0) * kUsed * kFf,
        dids.get() + static_cast<std::size_t>(t0) * kUsed, cnt,
        dout.get() + static_cast<std::size_t>(t0) * kUsed * kHidden,
        scratch.get(), nf, st);
  };
  const auto fetch = [&] {
    CheckHip(hipStreamSynchronize(st), "inv sync");
    return std::make_pair(dact.Read(), dout.Read());
  };
  run(0, n, 0);
  const auto full = fetch();
  std::printf("DIGEST %s %s/%s n=%d %016llx %016llx\n", label.c_str(),
              Name(s.gate.type), Name(s.down.type), n,
              static_cast<unsigned long long>(
                  Fnv(full.first.data(), full.first.size() * 4)),
              static_cast<unsigned long long>(
                  Fnv(full.second.data(), full.second.size() * 4)));
  {  // golden digests of the parent kernels
    const std::string key = label + " " + Name(s.gate.type) + "/" +
                            Name(s.down.type) + " n=" + std::to_string(n);
    const auto da = Fnv(full.first.data(), full.first.size() * 4);
    const auto dop = Fnv(full.second.data(), full.second.size() * 4);
    for (const auto& g : parent_digests::kMoe)
      if (key == g.key) {
        const bool ok = g.act == da && g.out == dop;
        std::printf("%-52s %s\n", (key + " == parent digest").c_str(),
                    ok ? "PASS" : "FAIL");
        if (!ok)
          ++g_failures;
      }
  }
  const auto same = [&](const std::string& what) {
    const auto got = fetch();
    const bool ok = std::memcmp(got.first.data(), full.first.data(),
                                got.first.size() * 4) == 0 &&
                    std::memcmp(got.second.data(), full.second.data(),
                                got.second.size() * 4) == 0;
    std::printf("%-52s %s\n",
                (label + " " + Name(s.gate.type) + "/" + Name(s.down.type) +
                 " n=" + std::to_string(n) + " " + what + " bit-identical")
                    .c_str(),
                ok ? "PASS" : "FAIL");
    if (!ok)
      ++g_failures;
  };
  for (unsigned nf : {1U, 2U}) {
    CheckHip(hipMemset(dact.get(), 0xFF, act_in.size() * 4), "memset");
    CheckHip(hipMemset(dout.get(), 0xFF, dout.Read().size() * 4), "memset");
    run(0, n, nf);
    same("nf=" + std::to_string(nf));
  }
  const std::vector<std::vector<int>> splits = {
      {9}, {std::max(9, n - 9)}, {97, 128}, {512, 512, 512}, {64, 64, 64, 64}};
  for (std::size_t si = 0; si < splits.size(); ++si) {
    CheckHip(hipMemset(dact.get(), 0xFF, act_in.size() * 4), "memset");
    CheckHip(hipMemset(dout.get(), 0xFF, dout.Read().size() * 4), "memset");
    int t0 = 0;
    for (int c : splits[si]) {
      c = std::min(c, n - t0);
      if (c <= 0)
        break;
      run(t0, c, 0);
      t0 += c;
    }
    if (t0 < n)
      run(t0, n - t0, 0);
    same("split " + std::to_string(si));
  }
}

// ---- --bench ----------------------------------------------------------------
/// Uniform routing: 8 distinct experts per token out of `kExperts`.
std::vector<std::int32_t> UniformIds(int n, Rng& rng) {
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n) * kUsed);
  for (int t = 0; t < n; ++t) {
    int ch[kUsed];
    for (int k = 0; k < kUsed; ++k) {
      int e;
      bool dup;
      do {
        e = static_cast<int>(rng.Next() % kExperts);
        dup = false;
        for (int j = 0; j < k; ++j)
          dup = dup || ch[j] == e;
      } while (dup);
      ch[k] = e;
      ids[static_cast<std::size_t>(t) * kUsed + k] = e;
    }
  }
  return ids;
}

void BenchExperts(Setup& s, int n, hipStream_t st) {
  Rng rng(321 + n);
  const auto ids = UniformIds(n, rng);
  std::vector<float> x(static_cast<std::size_t>(n) * kHidden);
  for (auto& v : x)
    v = rng.Normal();
  std::vector<float> act_in(static_cast<std::size_t>(n) * kUsed * kFf);
  for (auto& v : act_in)
    v = rng.Normal();
  Dev<float> dx(x), dact_in(act_in);
  Dev<std::int32_t> dids(ids);
  Dev<float> dact(act_in.size()),
      dout(static_cast<std::size_t>(n) * kUsed * kHidden);
  Dev<std::uint8_t> scratch(rocm::moe::PrefillScratchBytes(n));
  std::set<int> touched(ids.begin(), ids.end());
  const double gu_bytes = 2.0 * touched.size() * kFf * s.gate.RowBytes();
  const double dn_bytes = 1.0 * touched.size() * kHidden * s.down.RowBytes();
  hipEvent_t e0, e1;
  CheckHip(hipEventCreate(&e0), "ev");
  CheckHip(hipEventCreate(&e1), "ev");
  const auto time = [&](const std::function<void()>& f) {
    for (int i = 0; i < 3; ++i)
      f();
    CheckHip(hipEventRecord(e0, st), "rec");
    for (int i = 0; i < 10; ++i)
      f();
    CheckHip(hipEventRecord(e1, st), "rec");
    CheckHip(hipEventSynchronize(e1), "sync");
    float ms = 0;
    CheckHip(hipEventElapsedTime(&ms, e0, e1), "elapsed");
    return ms / 10.0;
  };
  struct V {
    unsigned nf;
  };
  for (const V v : {V{0}, V{1}, V{2}}) {
    const double gu = time([&] {
      rocm::moe::ExpertGateUpPrefillTile(
          s.dgate.get(), s.dup.get(), s.gate.type, dx.get(), dids.get(), n,
          4.0F, dact.get(), scratch.get(), v.nf, st);
    });
    const double dn = time([&] {
      rocm::moe::ExpertDownPrefillTile(s.ddown.get(), s.down.type,
                                       dact_in.get(), dids.get(), n, dout.get(),
                                       scratch.get(), v.nf, st);
    });
    const double flop_gu = 2.0 * 2 * n * kUsed * kFf * kHidden;
    const double flop_dn = 2.0 * n * kUsed * kFf * kHidden;
    std::printf(
        "BENCH moe-prefill %s/%s n=%d nf=%u gate_up_ms=%.3f "
        "(%.0f GB/s %.2f TFLOPS) down_ms=%.3f (%.0f GB/s %.2f TFLOPS) "
        "experts=%zu\n",
        Name(s.gate.type), Name(s.down.type), n, v.nf, gu,
        gu_bytes / (gu * 1e-3) / 1e9, flop_gu / (gu * 1e-3) / 1e12, dn,
        dn_bytes / (dn * 1e-3) / 1e9, flop_dn / (dn * 1e-3) / 1e12,
        touched.size());
  }
  CheckHip(hipEventDestroy(e0), "ev");
  CheckHip(hipEventDestroy(e1), "ev");
}

}  // namespace

int main(int argc, char** argv) {
  // Before the first HIP call: a missing $TMPDIR crashes the runtime.
  if (!gufo::core::RequireTempDirectory())
    return 1;
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::puts("no GPU; skipping");
    return 77;
  }
  try {
    hipStream_t st;
    CheckHip(hipStreamCreate(&st), "stream");
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
      struct Pair {
        WeightType gu, dn;
      };
      for (const Pair pr : {Pair{WeightType::kIQ4_XS, WeightType::kQ4_K},
                            Pair{WeightType::kQ5_K, WeightType::kQ6_K}}) {
        Setup s(pr.gu, pr.dn, kExperts, 99, 0.06F);
        for (int n : {64, 512, 2048})
          BenchExperts(s, n, st);
      }
      return 0;
    }
    // --invariance: only the (fast) bit-exactness checks of the prefill path.
    const bool quick = argc > 1 && std::strcmp(argv[1], "--invariance") == 0;
    if (!quick) {
      for (int n : {1, 2, 8, 64, 513})
        TestRoute(n, st);
      for (int n : {1, 8, 64})
        TestCombine(n, st);
    }

    struct Pair {
      WeightType gu, dn;
    };
    for (const Pair pr : {Pair{WeightType::kIQ4_XS, WeightType::kQ4_K},
                          Pair{WeightType::kQ5_K, WeightType::kQ6_K}}) {
      Setup s(pr.gu, pr.dn, 16, 42, 0.06F);
      for (float limit : {0.0F, 4.0F}) {
        if (quick)
          break;
        for (int n : {1, 2, 8}) {
          TestExperts(s, n, 16, 0, limit, true, 8, st,
                      "decode n=" + std::to_string(n) +
                          " L=" + std::to_string(int(limit)));
        }
        TestExperts(s, 8, 16, 1, limit, true, 8, st,
                    "decode n=8 same-ids L=" + std::to_string(int(limit)));
        for (int n : {1, 2, 8, 64, 513}) {
          TestExperts(s, n, 16, 0, limit, false, n > 64 ? 24 : 64, st,
                      "prefill n=" + std::to_string(n) +
                          " L=" + std::to_string(int(limit)));
        }
      }
      if (pr.gu == WeightType::kIQ4_XS && !quick) {
        TestExperts(s, 4096, 16, 0, 4.0F, false, 6, st, "prefill n=4096 L=4");
      }
      for (int n : {9, 64, 513})
        TestPrefillInvariance(s, n, 16, 4.0F, st, "pool16");
    }
    {
      // Real tensor shapes: 512 stacked experts, ids over the full range.
      Setup s(WeightType::kIQ4_XS, WeightType::kQ4_K, kExperts, 77, 0.06F);
      if (!quick) {
        TestExperts(s, 8, kExperts, 0, 4.0F, true, 3, st,
                    "real-shape 512x768x2560 decode n=8 L=4");
        TestExperts(s, 64, kExperts, 0, 4.0F, false, 3, st,
                    "real-shape 512x768x2560 prefill n=64 L=4");
      }
      for (int n : {9, 64, 512, 2048})
        TestPrefillInvariance(s, n, kExperts, 4.0F, st, "real-shape");
    }
    {
      // Real shapes for the second expert type pair (layers 2, 3, 39-41).
      Setup s(WeightType::kQ5_K, WeightType::kQ6_K, kExperts, 78, 0.06F);
      for (int n : {9, 64, 512, 2048})
        TestPrefillInvariance(s, n, kExperts, 4.0F, st, "real-shape");
    }
    CheckHip(hipStreamDestroy(st), "stream destroy");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILED",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
