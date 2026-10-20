#include "src/models/ling_3_flash/cpu_ops.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <type_traits>

#include "src/core/quant/ggml_gemm.hpp"

namespace gufo::models::ling_3_flash::cpu {
namespace {

constexpr std::uint32_t kHeadDim = kKdaHeadDim;

// ---- Thread pool -----------------------------------------------------------

thread_local bool g_in_worker = false;

class Pool {
public:
  static Pool& Get() {
    static Pool pool;
    return pool;
  }

  ~Pool() { Stop(); }

  void Resize(unsigned threads) {
    threads = std::max(1U, threads);
    std::lock_guard<std::mutex> resize_lock(run_mutex_);
    Stop();
    stop_ = false;
    threads_ = threads;
    for (unsigned i = 1; i < threads; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  [[nodiscard]] unsigned Threads() const noexcept { return threads_; }

  void Run(std::size_t n,
           const std::function<void(std::size_t, std::size_t)>& fn) {
    if (n == 0) {
      return;
    }
    if (g_in_worker || threads_ <= 1 || n == 1) {
      fn(0, n);
      return;
    }
    std::lock_guard<std::mutex> run_lock(run_mutex_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fn_ = &fn;
      n_ = n;
      chunk_ =
          std::max<std::size_t>(1, (n + threads_ * 4 - 1) / (threads_ * 4));
      next_.store(0);
      active_ = static_cast<unsigned>(workers_.size());
      ++generation_;
    }
    wake_.notify_all();
    Drain();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return active_ == 0; });
  }

private:
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& w : workers_) {
      w.join();
    }
    workers_.clear();
  }

  void Drain() {
    const bool was = g_in_worker;
    g_in_worker = true;
    for (;;) {
      const std::size_t begin = next_.fetch_add(chunk_);
      if (begin >= n_) {
        break;
      }
      (*fn_)(begin, std::min(n_, begin + chunk_));
    }
    g_in_worker = was;
  }

  void WorkerLoop() {
    std::uint64_t seen = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      seen = generation_;
    }
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
      }
      Drain();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--active_ == 0) {
          done_.notify_one();
        }
      }
    }
  }

  std::mutex run_mutex_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::vector<std::thread> workers_;
  const std::function<void(std::size_t, std::size_t)>* fn_{nullptr};
  std::size_t n_{0};
  std::size_t chunk_{1};
  std::atomic<std::size_t> next_{0};
  unsigned active_{0};
  unsigned threads_{1};
  std::uint64_t generation_{0};
  bool stop_{false};
};

std::atomic<bool> g_f64{false};

float Dot(const float* a, const float* b, std::size_t n) {
  if (g_f64.load(std::memory_order_relaxed)) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      s += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(s);
  }
  float s = 0.0F;
  for (std::size_t i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

double SigmoidD(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}
double SiluD(double x) {
  return x / (1.0 + std::exp(-x));
}

}  // namespace

void SetF64(bool on) noexcept {
  g_f64.store(on, std::memory_order_relaxed);
}
bool F64() noexcept {
  return g_f64.load(std::memory_order_relaxed);
}

void SetThreads(unsigned threads) {
  Pool::Get().Resize(threads);
}
unsigned Threads() noexcept {
  return Pool::Get().Threads();
}

void ParallelFor(std::size_t n,
                 const std::function<void(std::size_t, std::size_t)>& fn) {
  Pool::Get().Run(n, fn);
}

void ReleasePages(const void* p, std::size_t bytes) noexcept {
  static const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
  const auto begin = reinterpret_cast<std::uintptr_t>(p);
  const std::uintptr_t first = (begin + page - 1) & ~(page - 1);
  const std::uintptr_t last = (begin + bytes) & ~(page - 1);
  if (last > first) {
    (void)madvise(reinterpret_cast<void*>(first), last - first, MADV_DONTNEED);
  }
}

void DequantizeRow(const TensorRef& t, std::uint64_t slice, std::uint64_t row,
                   float* out) {
  quant::Dequantize(t.type, t.Slice(slice) + row * t.RowBytes(), out, t.cols);
}

std::vector<float> DequantizeAll(const TensorRef& t) {
  std::vector<float> out(t.ElementCount());
  for (std::uint64_t s = 0; s < t.depth; ++s) {
    for (std::uint64_t r = 0; r < t.rows; ++r) {
      DequantizeRow(t, s, r, out.data() + (s * t.rows + r) * t.cols);
    }
  }
  return out;
}

void MatVec(const TensorRef& t, std::uint64_t slice, std::span<const float> x,
            std::span<float> y, bool parallel) {
  const std::uint8_t* base = t.Slice(slice);
  const std::size_t row_bytes = t.RowBytes();
  const bool f64 = F64();
  auto body = [&](std::size_t begin, std::size_t end) {
    if (f64) {
      thread_local std::vector<float> row;
      row.resize(t.cols);
      for (std::size_t r = begin; r < end; ++r) {
        quant::Dequantize(t.type, base + r * row_bytes, row.data(), t.cols);
        double sum = 0.0;
        for (std::size_t c = 0; c < t.cols; ++c) {
          sum += static_cast<double>(row[c]) * x[c];
        }
        y[r] = static_cast<float>(sum);
      }
      return;
    }
    for (std::size_t r = begin; r < end; ++r) {
      y[r] = quant::Dot(t.type, base + r * row_bytes, x, t.cols);
    }
  };
  if (parallel && t.rows >= 64) {
    ParallelFor(t.rows, body);
  } else {
    body(0, t.rows);
  }
}

void MatVecD(const TensorRef& t, std::uint64_t slice, std::span<const double> x,
             std::span<double> y, bool parallel) {
  const std::uint8_t* base = t.Slice(slice);
  const std::size_t row_bytes = t.RowBytes();
  auto body = [&](std::size_t begin, std::size_t end) {
    thread_local std::vector<float> row;
    row.resize(t.cols);
    for (std::size_t r = begin; r < end; ++r) {
      quant::Dequantize(t.type, base + r * row_bytes, row.data(), t.cols);
      double sum = 0.0;
      for (std::size_t c = 0; c < t.cols; ++c) {
        sum += static_cast<double>(row[c]) * x[c];
      }
      y[r] = sum;
    }
  };
  if (parallel && t.rows >= 64) {
    ParallelFor(t.rows, body);
  } else {
    body(0, t.rows);
  }
}

float Sigmoid(float x) noexcept {
  if (F64()) {
    return static_cast<float>(SigmoidD(x));
  }
  return 1.0F / (1.0F + std::exp(-x));
}
float Silu(float x) noexcept {
  if (F64()) {
    return static_cast<float>(SiluD(x));
  }
  return x / (1.0F + std::exp(-x));
}

void RmsNorm(std::span<const float> x, const float* w, float eps,
             std::span<float> out) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  if (F64()) {
    const double scale =
        1.0 / std::sqrt(ss / static_cast<double>(x.size()) + eps);
    for (std::size_t i = 0; i < x.size(); ++i) {
      out[i] = static_cast<float>(x[i] * scale * (w != nullptr ? w[i] : 1.0F));
    }
    return;
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(ss / static_cast<double>(x.size())) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * scale * (w != nullptr ? w[i] : 1.0F);
  }
}

void L2Norm(std::span<float> x, float eps) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  if (F64()) {
    const double scale = 1.0 / std::sqrt(ss + eps);
    for (float& v : x) {
      v = static_cast<float>(v * scale);
    }
    return;
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(ss) + eps);
  for (float& v : x) {
    v *= scale;
  }
}

void SwiGlu(std::span<const float> gate, std::span<const float> up, float limit,
            std::span<float> out) {
  const bool clamp = limit > 1e-6F;
  if (F64()) {
    for (std::size_t i = 0; i < gate.size(); ++i) {
      double u = up[i];
      double a = SiluD(gate[i]);
      if (clamp) {
        u = std::clamp(u, -static_cast<double>(limit),
                       static_cast<double>(limit));
        a = std::min(a, static_cast<double>(limit));
      }
      out[i] = static_cast<float>(a * u);
    }
    return;
  }
  for (std::size_t i = 0; i < gate.size(); ++i) {
    float u = up[i];
    float a = Silu(gate[i]);
    if (clamp) {
      u = std::clamp(u, -limit, limit);
      a = std::min(a, limit);
    }
    out[i] = a * u;
  }
}

void CausalConv4Silu(const float* weight, float* history, const float* raw,
                     float* out, std::size_t channels) {
  for (std::size_t c = 0; c < channels; ++c) {
    const float* w = weight + c * kKdaConvKernel;
    if (F64()) {
      const double acc = static_cast<double>(w[0]) * history[0 * channels + c] +
                         static_cast<double>(w[1]) * history[1 * channels + c] +
                         static_cast<double>(w[2]) * history[2 * channels + c] +
                         static_cast<double>(w[3]) * raw[c];
      out[c] = static_cast<float>(SiluD(acc));
      continue;
    }
    const float acc = w[0] * history[0 * channels + c] +
                      w[1] * history[1 * channels + c] +
                      w[2] * history[2 * channels + c] + w[3] * raw[c];
    out[c] = Silu(acc);
  }
  for (std::size_t c = 0; c < channels; ++c) {
    history[0 * channels + c] = history[1 * channels + c];
    history[1 * channels + c] = history[2 * channels + c];
    history[2 * channels + c] = raw[c];
  }
}

void KdaGate(const float* fa, const float* dt_bias, const float* a,
             float lower_bound, float* g) {
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      const std::size_t idx = std::size_t{h} * kHeadDim + i;
      if (F64()) {
        g[idx] = static_cast<float>(
            static_cast<double>(lower_bound) *
            SigmoidD(static_cast<double>(a[h]) *
                     (static_cast<double>(fa[idx]) + dt_bias[idx])));
        continue;
      }
      g[idx] = lower_bound * Sigmoid(a[h] * (fa[idx] + dt_bias[idx]));
    }
  }
}

void KdaStep(float* state, const float* q, const float* k, const float* v,
             const float* g, const float* beta, float* o) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  ParallelFor(kHeads, [&](std::size_t h0, std::size_t h1) {
    for (std::size_t h = h0; h < h1; ++h) {
      float* s = state + h * kHeadDim * kHeadDim;  // [i][j]
      const float* qh = q + h * kHeadDim;
      const float* kh = k + h * kHeadDim;
      const float* vh = v + h * kHeadDim;
      const float* gh = g + h * kHeadDim;
      float* oh = o + h * kHeadDim;
      float delta[kHeadDim];
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        const float decay = std::exp(gh[i]);
        for (std::uint32_t j = 0; j < kHeadDim; ++j) {
          s[i * kHeadDim + j] *= decay;
        }
      }
      for (std::uint32_t j = 0; j < kHeadDim; ++j) {
        float sum = 0.0F;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          sum += s[i * kHeadDim + j] * kh[i];
        }
        delta[j] = (vh[j] - sum) * beta[h];
      }
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        for (std::uint32_t j = 0; j < kHeadDim; ++j) {
          s[i * kHeadDim + j] += kh[i] * delta[j];
        }
      }
      for (std::uint32_t j = 0; j < kHeadDim; ++j) {
        float sum = 0.0F;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          sum += s[i * kHeadDim + j] * qh[i];
        }
        oh[j] = sum * scale;
      }
    }
  });
}

void KdaStepF64(double* state, const float* q, const float* k, const float* v,
                const float* g, const float* beta, float* o) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  ParallelFor(kHeads, [&](std::size_t h0, std::size_t h1) {
    for (std::size_t h = h0; h < h1; ++h) {
      double* s = state + h * kHeadDim * kHeadDim;  // [i][j]
      const float* qh = q + h * kHeadDim;
      const float* kh = k + h * kHeadDim;
      const float* vh = v + h * kHeadDim;
      const float* gh = g + h * kHeadDim;
      float* oh = o + h * kHeadDim;
      double delta[kHeadDim];
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        const double decay = std::exp(static_cast<double>(gh[i]));
        for (std::uint32_t j = 0; j < kHeadDim; ++j) {
          s[i * kHeadDim + j] *= decay;
        }
      }
      for (std::uint32_t j = 0; j < kHeadDim; ++j) {
        double sum = 0.0;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          sum += s[i * kHeadDim + j] * kh[i];
        }
        delta[j] = (static_cast<double>(vh[j]) - sum) * beta[h];
      }
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        for (std::uint32_t j = 0; j < kHeadDim; ++j) {
          s[i * kHeadDim + j] += static_cast<double>(kh[i]) * delta[j];
        }
      }
      for (std::uint32_t j = 0; j < kHeadDim; ++j) {
        double sum = 0.0;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          sum += s[i * kHeadDim + j] * qh[i];
        }
        oh[j] = static_cast<float>(sum * scale);
      }
    }
  });
}

void KdaOutputGate(float* o, const float* norm_w, const float* gate,
                   float eps) {
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    float* oh = o + std::size_t{h} * kHeadDim;
    RmsNorm({oh, kHeadDim}, norm_w, eps, {oh, kHeadDim});
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      if (F64()) {
        oh[d] = static_cast<float>(
            oh[d] * SigmoidD(gate[std::size_t{h} * kHeadDim + d]));
        continue;
      }
      oh[d] *= Sigmoid(gate[std::size_t{h} * kHeadDim + d]);
    }
  }
}

void NormRope(float* x, std::uint32_t heads, std::uint32_t stride,
              std::uint32_t n_rot, std::uint32_t pos, float base) {
  for (std::uint32_t h = 0; h < heads; ++h) {
    float* v = x + std::size_t{h} * stride;
    for (std::uint32_t i = 0; i < n_rot / 2; ++i) {
      if (F64()) {
        const double theta_d =
            static_cast<double>(pos) *
            std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) /
                                                    static_cast<double>(n_rot));
        const double cd = std::cos(theta_d);
        const double sd = std::sin(theta_d);
        const double a0 = v[2 * i];
        const double a1 = v[2 * i + 1];
        v[2 * i] = static_cast<float>(a0 * cd - a1 * sd);
        v[2 * i + 1] = static_cast<float>(a0 * sd + a1 * cd);
        continue;
      }
      const float theta = static_cast<float>(pos) *
                          std::pow(base, -2.0F * static_cast<float>(i) /
                                             static_cast<float>(n_rot));
      const float c = std::cos(theta);
      const float s = std::sin(theta);
      const float x0 = v[2 * i];
      const float x1 = v[2 * i + 1];
      v[2 * i] = x0 * c - x1 * s;
      v[2 * i + 1] = x0 * s + x1 * c;
    }
  }
}

namespace {

void SoftmaxInPlace(std::vector<float>& s) {
  const float m = *std::max_element(s.begin(), s.end());
  double sum = 0.0;
  for (float& v : s) {
    v = std::exp(v - m);
    sum += v;
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (float& v : s) {
    v *= inv;
  }
}

}  // namespace

void MlaAttendAbsorbed(const TensorRef& k_b, const TensorRef& v_b,
                       const float* q, const float* cache, std::size_t n,
                       float* out, float* latent_out) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kMlaQHeadDim));
  ParallelFor(kHeads, [&](std::size_t h0, std::size_t h1) {
    std::vector<float> q_abs(kMlaCacheWidth);
    std::vector<float> scores(n);
    std::vector<float> lat(kMlaLatent);
    for (std::size_t h = h0; h < h1; ++h) {
      const float* qh = q + h * kMlaQHeadDim;
      // q_lat[r] = sum_d k_b[h][r][d] * q_nope[d]
      MatVec(k_b, h, {qh, kMlaNope}, {q_abs.data(), kMlaLatent}, false);
      std::memcpy(q_abs.data() + kMlaLatent, qh + kMlaNope,
                  kMlaRope * sizeof(float));
      for (std::size_t p = 0; p < n; ++p) {
        scores[p] =
            Dot(q_abs.data(), cache + p * kMlaCacheWidth, kMlaCacheWidth) *
            scale;
      }
      SoftmaxInPlace(scores);
      std::fill(lat.begin(), lat.end(), 0.0F);
      for (std::size_t p = 0; p < n; ++p) {
        const float* row = cache + p * kMlaCacheWidth;
        for (std::uint32_t r = 0; r < kMlaLatent; ++r) {
          lat[r] += scores[p] * row[r];
        }
      }
      if (latent_out != nullptr) {
        std::memcpy(latent_out + h * kMlaLatent, lat.data(),
                    kMlaLatent * sizeof(float));
      }
      MatVec(v_b, h, lat, {out + h * kMlaValueDim, kMlaValueDim}, false);
    }
  });
}

void MlaAttendAbsorbedF64(const TensorRef& k_b, const TensorRef& v_b,
                          const float* q, const float* cache, std::size_t n,
                          float* out, float* latent_out) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(kMlaQHeadDim));
  ParallelFor(kHeads, [&](std::size_t h0, std::size_t h1) {
    std::vector<double> qn(kMlaNope);
    std::vector<double> q_abs(kMlaCacheWidth);
    std::vector<double> scores(n);
    std::vector<double> lat(kMlaLatent);
    std::vector<double> o(kMlaValueDim);
    for (std::size_t h = h0; h < h1; ++h) {
      const float* qh = q + h * kMlaQHeadDim;
      for (std::uint32_t d = 0; d < kMlaNope; ++d) {
        qn[d] = qh[d];
      }
      MatVecD(k_b, h, qn, {q_abs.data(), kMlaLatent}, false);
      for (std::uint32_t d = 0; d < kMlaRope; ++d) {
        q_abs[kMlaLatent + d] = qh[kMlaNope + d];
      }
      double m = -std::numeric_limits<double>::infinity();
      for (std::size_t p = 0; p < n; ++p) {
        const float* row = cache + p * kMlaCacheWidth;
        double sum = 0.0;
        for (std::size_t c = 0; c < kMlaCacheWidth; ++c) {
          sum += q_abs[c] * row[c];
        }
        scores[p] = sum * scale;
        m = std::max(m, scores[p]);
      }
      double denom = 0.0;
      for (double& sc : scores) {
        sc = std::exp(sc - m);
        denom += sc;
      }
      std::fill(lat.begin(), lat.end(), 0.0);
      for (std::size_t p = 0; p < n; ++p) {
        const float* row = cache + p * kMlaCacheWidth;
        const double wgt = scores[p] / denom;
        for (std::uint32_t r = 0; r < kMlaLatent; ++r) {
          lat[r] += wgt * row[r];
        }
      }
      if (latent_out != nullptr) {
        for (std::uint32_t r = 0; r < kMlaLatent; ++r) {
          latent_out[h * kMlaLatent + r] = static_cast<float>(lat[r]);
        }
      }
      MatVecD(v_b, h, lat, o, false);
      for (std::uint32_t j = 0; j < kMlaValueDim; ++j) {
        out[h * kMlaValueDim + j] = static_cast<float>(o[j]);
      }
    }
  });
}

void MlaExpand(const float* wk, const float* wv, const float* latent,
               float* k_nope, float* v) {
  ParallelFor(kHeads, [&](std::size_t h0, std::size_t h1) {
    for (std::size_t h = h0; h < h1; ++h) {
      float* kh = k_nope + h * kMlaNope;
      std::fill(kh, kh + kMlaNope, 0.0F);
      const float* wkh = wk + h * kMlaLatent * kMlaNope;
      for (std::uint32_t r = 0; r < kMlaLatent; ++r) {
        for (std::uint32_t d = 0; d < kMlaNope; ++d) {
          kh[d] += latent[r] * wkh[r * kMlaNope + d];
        }
      }
      const float* wvh = wv + h * kMlaValueDim * kMlaLatent;
      for (std::uint32_t j = 0; j < kMlaValueDim; ++j) {
        v[h * kMlaValueDim + j] = Dot(wvh + j * kMlaLatent, latent, kMlaLatent);
      }
    }
  });
}

void MlaAttendExpanded(const float* q, const float* k_nope, const float* k_pe,
                       const float* v, std::size_t n, float* out) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kMlaQHeadDim));
  ParallelFor(kHeads, [&](std::size_t h0, std::size_t h1) {
    std::vector<float> scores(n);
    for (std::size_t h = h0; h < h1; ++h) {
      const float* qh = q + h * kMlaQHeadDim;
      for (std::size_t p = 0; p < n; ++p) {
        scores[p] = (Dot(qh, k_nope + (p * kHeads + h) * kMlaNope, kMlaNope) +
                     Dot(qh + kMlaNope, k_pe + p * kMlaRope, kMlaRope)) *
                    scale;
      }
      SoftmaxInPlace(scores);
      float* oh = out + h * kMlaValueDim;
      std::fill(oh, oh + kMlaValueDim, 0.0F);
      for (std::size_t p = 0; p < n; ++p) {
        const float* vp = v + (p * kHeads + h) * kMlaValueDim;
        for (std::uint32_t j = 0; j < kMlaValueDim; ++j) {
          oh[j] += scores[p] * vp[j];
        }
      }
    }
  });
}

namespace {

template<typename T>
void RouteImpl(std::span<const T> logits, const float* bias,
               std::uint32_t groups, std::uint32_t groups_used,
               std::uint32_t experts_used, float scale, float floor,
               Routing& out, float* probs) {
  const std::size_t n = logits.size();
  std::vector<T> p(n);
  std::vector<T> sel(n);
  for (std::size_t i = 0; i < n; ++i) {
    if constexpr (std::is_same_v<T, double>) {
      p[i] = SigmoidD(logits[i]);
    } else {
      p[i] = Sigmoid(logits[i]);
    }
    sel[i] = p[i] + (bias != nullptr ? static_cast<T>(bias[i]) : T{0});
  }
  if (probs != nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
      probs[i] = static_cast<float>(p[i]);
    }
  }
  const std::size_t per_group = n / groups;
  std::vector<std::pair<T, std::uint32_t>> group_score(groups);
  for (std::uint32_t g = 0; g < groups; ++g) {
    std::vector<T> s(
        sel.begin() + static_cast<std::ptrdiff_t>(g * per_group),
        sel.begin() + static_cast<std::ptrdiff_t>((g + 1) * per_group));
    std::partial_sort(s.begin(), s.begin() + 2, s.end(), std::greater<>());
    group_score[g] = {s[0] + s[1], g};
  }
  std::stable_sort(
      group_score.begin(), group_score.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<T> masked(n, -std::numeric_limits<T>::infinity());
  for (std::uint32_t k = 0; k < groups_used; ++k) {
    const std::size_t g = group_score[k].second;
    std::copy(sel.begin() + static_cast<std::ptrdiff_t>(g * per_group),
              sel.begin() + static_cast<std::ptrdiff_t>((g + 1) * per_group),
              masked.begin() + static_cast<std::ptrdiff_t>(g * per_group));
  }
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return masked[a] > masked[b]; });
  for (std::size_t k = 0; k < 10 && k < n; ++k) {
    out.top_ids[k] = static_cast<std::int32_t>(order[k]);
    out.top_sel[k] = static_cast<float>(masked[order[k]]);
  }
  double sum = 0.0;
  for (std::uint32_t k = 0; k < experts_used; ++k) {
    out.ids[k] = static_cast<std::int32_t>(order[k]);
    out.weights[k] = static_cast<float>(p[order[k]]);
    sum += p[order[k]];
  }
  const T denom = std::max(static_cast<T>(sum), static_cast<T>(floor));
  for (std::uint32_t k = 0; k < experts_used; ++k) {
    const T w = static_cast<T>(p[order[k]]) / denom * static_cast<T>(scale);
    out.weights[k] = static_cast<float>(w);
    out.weights_d[k] = static_cast<double>(w);
  }
}

}  // namespace

void Route(std::span<const float> logits, const float* bias,
           std::uint32_t groups, std::uint32_t groups_used,
           std::uint32_t experts_used, float scale, float floor, Routing& out,
           float* probs) {
  RouteImpl<float>(logits, bias, groups, groups_used, experts_used, scale,
                   floor, out, probs);
}

void RouteF64(std::span<const double> logits, const float* bias,
              std::uint32_t groups, std::uint32_t groups_used,
              std::uint32_t experts_used, float scale, float floor,
              Routing& out, float* probs) {
  RouteImpl<double>(logits, bias, groups, groups_used, experts_used, scale,
                    floor, out, probs);
}

}  // namespace gufo::models::ling_3_flash::cpu
