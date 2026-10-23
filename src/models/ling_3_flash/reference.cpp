#include "src/models/ling_3_flash/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/ling_3_flash/cpu_ops.hpp"

namespace gufo::models::ling_3_flash {
namespace {

constexpr std::size_t kReleaseMinBytes = 256 * 1024;

const float* F32Data(const TensorRef& t) {
  return static_cast<const float*>(t.data);
}

float RoundF16(float v) {
  return static_cast<float>(static_cast<_Float16>(v));
}

/// float(v) rounded toward zero. The RDNA3 WMMA unit adds every group of 8
/// products to the F32 accumulator with truncation (measured against the GPU
/// GEMM in dense_ops_test); the emulation reproduces that.
float RoundTowardZero(double v) {
  float f = static_cast<float>(v);
  if (std::fabs(static_cast<double>(f)) > std::fabs(v)) {
    f = std::nextafter(f, 0.0F);
  }
  return f;
}

std::string LayerSuffix(int layer) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), ".l%02d", layer);
  return buf;
}

}  // namespace

// ---- DumpRecorder ----------------------------------------------------------

bool DumpRecorder::Wants(std::string_view name, int layer) const {
  if (!names_.empty() &&
      std::find(names_.begin(), names_.end(), name) == names_.end()) {
    return false;
  }
  return layers_.empty() ||
         std::find(layers_.begin(), layers_.end(), layer) != layers_.end();
}

void DumpRecorder::Float(std::string_view name, int layer, std::uint32_t token,
                         std::span<const float> data) {
  if (!Wants(name, layer)) {
    return;
  }
  Series& s = series_[{std::string(name), layer}];
  if (s.width == 0) {
    s.width = static_cast<std::uint32_t>(data.size());
  }
  // Tokens may arrive sparsely or repeat; missing rows stay zero.
  if (token >= s.tokens) {
    s.tokens = token + 1;
    s.f.resize(std::size_t{s.tokens} * s.width, 0.0F);
  }
  std::copy(
      data.begin(), data.end(),
      s.f.begin() + static_cast<std::ptrdiff_t>(std::size_t{token} * s.width));
}

void DumpRecorder::Int(std::string_view name, int layer, std::uint32_t token,
                       std::span<const std::int32_t> data) {
  if (!Wants(name, layer)) {
    return;
  }
  Series& s = series_[{std::string(name), layer}];
  s.is_int = true;
  if (s.width == 0) {
    s.width = static_cast<std::uint32_t>(data.size());
  }
  if (token >= s.tokens) {
    s.tokens = token + 1;
    s.i.resize(std::size_t{s.tokens} * s.width, 0);
  }
  std::copy(
      data.begin(), data.end(),
      s.i.begin() + static_cast<std::ptrdiff_t>(std::size_t{token} * s.width));
}

bool DumpRecorder::Write(const std::string& dir,
                         const std::string& extra_entries,
                         const std::string& extra_fields,
                         std::string* error_msg) const {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    if (error_msg != nullptr) {
      *error_msg = "cannot create " + dir + ": " + ec.message();
    }
    return false;
  }
  std::string entries;
  for (const auto& [key, s] : series_) {
    const std::string file = key.first +
                             (key.second >= 0 ? LayerSuffix(key.second) : "") +
                             (s.is_int ? ".i32" : ".f32");
    std::ofstream out(dir + "/" + file, std::ios::binary);
    if (s.is_int) {
      out.write(
          reinterpret_cast<const char*>(s.i.data()),
          static_cast<std::streamsize>(s.i.size() * sizeof(std::int32_t)));
    } else {
      out.write(reinterpret_cast<const char*>(s.f.data()),
                static_cast<std::streamsize>(s.f.size() * sizeof(float)));
    }
    if (!out) {
      if (error_msg != nullptr) {
        *error_msg = "write failed: " + file;
      }
      return false;
    }
    if (!entries.empty()) {
      entries += ",\n";
    }
    entries += "    {\"name\": \"" + key.first +
               "\", \"layer\": " + std::to_string(key.second) +
               ", \"file\": \"" + file + "\", \"dtype\": \"" +
               (s.is_int ? "i32" : "f32") + "\", \"shape\": [" +
               std::to_string(s.tokens) + ", " + std::to_string(s.width) +
               "], \"tokens\": " + std::to_string(s.tokens) + "}";
  }
  if (!extra_entries.empty()) {
    if (!entries.empty()) {
      entries += ",\n";
    }
    entries += extra_entries;
  }
  std::ofstream index(dir + "/index.json");
  index << "{\n  \"format\": \"gufo-ling-dump-v1\",\n"
        << extra_fields << "  \"entries\": [\n"
        << entries << "\n  ]\n}\n";
  if (!index) {
    if (error_msg != nullptr) {
      *error_msg = "write failed: index.json";
    }
    return false;
  }
  return true;
}

// ---- ReferenceModel --------------------------------------------------------

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               const Options& options)
    : w_(weights), c_(weights.config), opt_(options) {
  cpu::SetThreads(opt_.threads);
  cpu::SetF64(opt_.f64);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsMla(il)) {
      check_layer_ = static_cast<int>(il);
      break;
    }
  }
  Reset();
}

void ReferenceModel::Reset() {
  const std::uint32_t layers = StateLayers();
  layer_pos_.assign(layers, 0);
  kda_state_.assign(layers, {});
  kda_state64_.assign(layers, {});
  kda_conv_.assign(layers, {});
  mla_cache_.assign(layers, {});
  for (std::uint32_t il = 0; il < layers; ++il) {
    if (c_.IsMla(il)) {
      mla_cache_[il].reserve(std::size_t{opt_.max_context} * kMlaCacheWidth);
    } else {
      if (opt_.f64) {
        kda_state64_[il].assign(kKdaStateFloats, 0.0);
      } else {
        kda_state_[il].assign(kKdaStateFloats, 0.0F);
      }
      kda_conv_[il].assign(kKdaConvStateFloats, 0.0F);
    }
  }
  check_k_nope_.clear();
  check_v_.clear();
  check_k_pe_.clear();
  check_ = {};
  check_.layer = check_layer_;
}

void ReferenceModel::Release(const TensorRef& t, std::uint64_t slice) const {
  const std::size_t bytes = t.RowBytes() * t.rows;
  if (opt_.release_pages && bytes >= kReleaseMinBytes) {
    cpu::ReleasePages(t.Slice(slice), bytes);
  }
}

void ReferenceModel::ReleaseAll(const TensorRef& t) const {
  if (opt_.release_pages && t.SizeBytes() >= kReleaseMinBytes) {
    cpu::ReleasePages(t.data, t.SizeBytes());
  }
}

void ReferenceModel::Mv(const TensorRef& t, std::uint64_t slice,
                        std::span<const float> x, std::span<float> y,
                        bool parallel) const {
  if (opt_.emulate_gpu_f16 && t.type != core::GgmlType::kF32 &&
      (&t != &w_.output || opt_.emulate_f16_head)) {
    MvGpuF16(t, slice, x, y, parallel);
    return;
  }
  cpu::MatVec(t, slice, x, y, parallel);
  Release(t, slice);
}

void ReferenceModel::MvGpuF16(const TensorRef& t, std::uint64_t slice,
                              std::span<const float> x, std::span<float> y,
                              bool parallel) const {
  std::vector<float> xr(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    xr[i] = RoundF16(x[i]);
  }
  const auto rows = [&](std::size_t r0, std::size_t r1) {
    std::vector<float> row(t.cols);
    for (std::size_t r = r0; r < r1; ++r) {
      cpu::DequantizeRow(t, slice, r, row.data());
      float acc = 0.0F;
      for (std::size_t c0 = 0; c0 < t.cols; c0 += 8) {
        double sum = 0.0;
        for (std::size_t c = c0; c < std::min<std::size_t>(c0 + 8, t.cols);
             ++c) {
          sum += static_cast<double>(RoundF16(row[c])) * xr[c];
        }
        acc = RoundTowardZero(static_cast<double>(acc) + sum);
      }
      y[r] = acc;
    }
  };
  if (parallel) {
    cpu::ParallelFor(t.rows, rows);
  } else {
    rows(0, t.rows);
  }
  Release(t, slice);
}

void ReferenceModel::MlaAttendGpuF16(const LayerWeights& l, const float* q,
                                     const float* cache, std::size_t n,
                                     float* out) const {
  constexpr float kScale = 0.07216878364870323F;  // 1 / sqrt(192), as the GPU
  constexpr std::size_t kTile = 16;
  cpu::ParallelFor(kHeads, [&](std::size_t h0, std::size_t h1) {
    std::vector<float> q_abs(kMlaCacheWidth);
    std::vector<float> q16(kMlaCacheWidth);
    std::vector<float> o(kMlaLatent);
    std::vector<float> lat(kMlaLatent);
    for (std::size_t h = h0; h < h1; ++h) {
      const float* qh = q + h * kMlaQHeadDim;
      cpu::MatVec(l.attn_k_b, h, {qh, kMlaNope}, {q_abs.data(), kMlaLatent},
                  false);
      std::memcpy(q_abs.data() + kMlaLatent, qh + kMlaNope,
                  kMlaRope * sizeof(float));
      for (std::size_t c = 0; c < kMlaCacheWidth; ++c) {
        q16[c] = RoundF16(q_abs[c] * kScale);
      }
      float run_max = -INFINITY;
      float run_sum = 0.0F;
      std::fill(o.begin(), o.end(), 0.0F);
      for (std::size_t kb = 0; kb < n; kb += kTile) {
        const std::size_t count = std::min(kTile, n - kb);
        float s[kTile];
        float part_max = -INFINITY;
        for (std::size_t j = 0; j < count; ++j) {
          const float* row = cache + (kb + j) * kMlaCacheWidth;
          // Four waves each accumulate 144 channels, then add in order.
          float part[4];
          for (std::size_t wv = 0; wv < 4; ++wv) {
            float acc = 0.0F;
            for (std::size_t c0 = wv * 144; c0 < (wv + 1) * 144; c0 += 8) {
              double sum = 0.0;
              for (std::size_t c = c0; c < c0 + 8; ++c) {
                sum += static_cast<double>(q16[c]) * row[c];
              }
              acc = RoundTowardZero(static_cast<double>(acc) + sum);
            }
            part[wv] = acc;
          }
          s[j] = ((part[0] + part[1]) + part[2]) + part[3];
          part_max = std::max(part_max, s[j]);
        }
        const float next_max = std::max(run_max, part_max);
        const float prior =
            std::isinf(run_max) ? 0.0F : std::exp(run_max - next_max);
        float p[kTile];
        float part_sum = 0.0F;
        for (std::size_t j = 0; j < count; ++j) {
          p[j] = RoundF16(std::exp(s[j] - next_max));
          part_sum += p[j];
        }
        run_max = next_max;
        run_sum = run_sum * prior + part_sum;
        for (std::size_t r = 0; r < kMlaLatent; ++r) {
          float acc = o[r] * prior;
          for (std::size_t j0 = 0; j0 < count; j0 += 8) {
            double sum = 0.0;
            for (std::size_t j = j0; j < std::min(j0 + 8, count); ++j) {
              sum += static_cast<double>(p[j]) *
                     cache[(kb + j) * kMlaCacheWidth + r];
            }
            acc = RoundTowardZero(static_cast<double>(acc) + sum);
          }
          o[r] = acc;
        }
      }
      for (std::size_t r = 0; r < kMlaLatent; ++r) {
        lat[r] = o[r] / run_sum;
      }
      cpu::MatVec(l.attn_v_b, h, lat, {out + h * kMlaValueDim, kMlaValueDim},
                  false);
    }
  });
}

bool ReferenceModel::Embed(std::int32_t token, std::span<float> out,
                           std::string* error_msg) const {
  if (token < 0 || static_cast<std::uint64_t>(token) >= w_.token_embd.rows) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range: " + std::to_string(token);
    }
    return false;
  }
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     out.data());
  return true;
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          DumpSink* sink, std::string* error_msg) {
  if (layer_pos_[0] >= opt_.max_context) {
    if (error_msg != nullptr) {
      *error_msg = "max_context exceeded";
    }
    return false;
  }
  std::vector<float> h(kHidden);
  if (!Embed(token, h, error_msg)) {
    return false;
  }
  const std::uint32_t pos = layer_pos_[0];
  if (sink != nullptr && sink->Wants("embd", 0)) {
    sink->Float("embd", 0, pos, h);
  }
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    Block(il, h, sink);
  }
  last_hidden_.resize(kHidden);
  cpu::RmsNorm(h, F32Data(w_.output_norm), c_.rms_eps, last_hidden_);
  if (!logits.empty()) {
    if (sink != nullptr && sink->Wants("result_norm", -1)) {
      sink->Float("result_norm", -1, pos, last_hidden_);
    }
    Mv(w_.output, 0, last_hidden_, logits);
  }
  return true;
}

bool ReferenceModel::MtpStep(std::int32_t token, std::span<const float> h,
                             std::span<float> logits, std::span<float> h_next,
                             DumpSink* sink, std::string* error_msg) {
  const auto fail = [&](const char* message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return false;
  };
  if (!w_.mtp) {
    return fail("the artifact has no MTP block");
  }
  if (h.size() != kHidden || (!h_next.empty() && h_next.size() != kHidden)) {
    return fail("MTP hidden rows must have the model width");
  }
  if (layer_pos_[kMtpBlock] >= opt_.max_context) {
    return fail("max_context exceeded");
  }
  const MtpWeights& m = *w_.mtp;
  std::vector<float> emb(kHidden);
  if (!Embed(token, emb, error_msg)) {
    return false;
  }
  // x = eh_proj([ rmsnorm(embed; enorm) | rmsnorm(h; hnorm) ]).
  std::vector<float> cat(2 * std::size_t{kHidden});
  cpu::RmsNorm(emb, F32Data(m.enorm), c_.rms_eps, {cat.data(), kHidden});
  cpu::RmsNorm(h, F32Data(m.hnorm), c_.rms_eps,
               {cat.data() + kHidden, kHidden});
  std::vector<float> x(kHidden);
  Mv(m.eh_proj, 0, cat, x);
  Block(kMtpBlock, x, sink);
  std::vector<float> normed(kHidden);
  cpu::RmsNorm(x, F32Data(m.layer_output_norm), c_.rms_eps, normed);
  if (sink != nullptr && sink->Wants("mtp_h_next", -1)) {
    sink->Float("mtp_h_next", -1, layer_pos_[kMtpBlock] - 1, normed);
  }
  if (!h_next.empty()) {
    std::copy(normed.begin(), normed.end(), h_next.begin());
  }
  if (!logits.empty()) {
    Mv(w_.output, 0, normed, logits);
  }
  return true;
}

void ReferenceModel::MtpRewind(std::uint32_t position) {
  if (!w_.mtp || position >= layer_pos_[kMtpBlock]) {
    return;
  }
  layer_pos_[kMtpBlock] = position;
  mla_cache_[kMtpBlock].resize(std::size_t{position} * kMlaCacheWidth);
}

bool ReferenceModel::RunLayer(std::uint32_t il, std::span<const float> in,
                              std::span<float> out, DumpSink* sink,
                              std::string* error_msg) {
  if (il >= c_.num_layers || in.size() != kHidden || out.size() != kHidden) {
    if (error_msg != nullptr) {
      *error_msg = "bad layer or hidden size";
    }
    return false;
  }
  if (layer_pos_[il] >= opt_.max_context) {
    if (error_msg != nullptr) {
      *error_msg = "max_context exceeded";
    }
    return false;
  }
  std::copy(in.begin(), in.end(), out.begin());
  Block(il, out, sink);
  return true;
}

void ReferenceModel::Block(std::uint32_t il, std::span<float> h,
                           DumpSink* sink) {
  const LayerWeights& l = Layer(il);
  const std::uint32_t token = layer_pos_[il];
  const int layer = static_cast<int>(il);
  std::vector<float> x(kHidden);
  std::vector<float> mixer(kHidden);
  std::vector<float> mid(kHidden);
  std::vector<float> y(kHidden);
  std::vector<float> ffn(kHidden);

  cpu::RmsNorm(h, F32Data(l.attn_norm), c_.rms_eps, x);
  if (l.kind == LayerKind::kKda) {
    Kda(il, x, mixer, token, sink);
  } else {
    Mla(il, x, mixer, token, sink);
  }
  for (std::size_t i = 0; i < kHidden; ++i) {
    mid[i] = h[i] + mixer[i];
  }
  cpu::RmsNorm(mid, F32Data(l.ffn_norm), c_.rms_eps, y);
  if (l.moe) {
    Moe(il, y, ffn, token, sink);
  } else {
    Ffn(l.ffn_gate, l.ffn_up, l.ffn_down, c_.swiglu_clamp_shexp[il], y, ffn,
        sink, layer, token);
  }
  for (std::size_t i = 0; i < kHidden; ++i) {
    h[i] = mid[i] + ffn[i];
  }
  if (sink != nullptr) {
    if (sink->Wants("mixer_out", layer))
      sink->Float("mixer_out", layer, token, mixer);
    if (sink->Wants("h_mid", layer))
      sink->Float("h_mid", layer, token, mid);
    if (sink->Wants("ffn_out", layer))
      sink->Float("ffn_out", layer, token, ffn);
    if (sink->Wants("h_out", layer))
      sink->Float("h_out", layer, token, h);
  }
  ++layer_pos_[il];
}

void ReferenceModel::Kda(std::uint32_t il, std::span<const float> x,
                         std::span<float> out, std::uint32_t token,
                         DumpSink* sink) {
  const LayerWeights& l = Layer(il);
  constexpr std::size_t kInner = kKdaInner;
  std::vector<float> raw_q(kInner);
  std::vector<float> raw_k(kInner);
  std::vector<float> raw_v(kInner);
  Mv(l.attn_q, 0, x, raw_q);
  Mv(l.attn_k, 0, x, raw_k);
  Mv(l.attn_v, 0, x, raw_v);

  float* conv = kda_conv_[il].data();
  std::vector<float> q(kInner);
  std::vector<float> k(kInner);
  std::vector<float> v(kInner);
  const std::size_t hist = std::size_t{kKdaConvHistory} * kInner;
  cpu::CausalConv4Silu(F32Data(l.ssm_conv1d_q), conv, raw_q.data(), q.data(),
                       kInner);
  cpu::CausalConv4Silu(F32Data(l.ssm_conv1d_k), conv + hist, raw_k.data(),
                       k.data(), kInner);
  cpu::CausalConv4Silu(F32Data(l.ssm_conv1d_v), conv + 2 * hist, raw_v.data(),
                       v.data(), kInner);

  std::vector<float> fa(kInner);
  std::vector<float> g(kInner);
  Mv(l.ssm_f_a, 0, x, fa);
  cpu::KdaGate(fa.data(), F32Data(l.ssm_dt), F32Data(l.ssm_a),
               c_.kda_gate_lower_bound, g.data());
  std::vector<float> beta(kHeads);
  Mv(l.ssm_beta, 0, x, beta);
  for (float& b : beta) {
    b = cpu::Sigmoid(b);
  }
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    cpu::L2Norm({q.data() + std::size_t{h} * kKdaHeadDim, kKdaHeadDim},
                kKdaL2Eps);
    cpu::L2Norm({k.data() + std::size_t{h} * kKdaHeadDim, kKdaHeadDim},
                kKdaL2Eps);
  }

  std::vector<float> o(kInner);
  if (opt_.f64) {
    cpu::KdaStepF64(kda_state64_[il].data(), q.data(), k.data(), v.data(),
                    g.data(), beta.data(), o.data());
  } else {
    cpu::KdaStep(kda_state_[il].data(), q.data(), k.data(), v.data(), g.data(),
                 beta.data(), o.data());
  }
  const int layer = static_cast<int>(il);
  if (sink != nullptr && sink->Wants("kda_o", layer)) {
    sink->Float("kda_o", layer, token, o);
  }

  std::vector<float> gate(kInner);
  Mv(l.ssm_g_a, 0, x, gate);
  cpu::KdaOutputGate(o.data(), F32Data(l.ssm_norm), gate.data(), c_.rms_eps);
  Mv(l.attn_output, 0, o, out);
}

void ReferenceModel::Mla(std::uint32_t il, std::span<const float> x,
                         std::span<float> out, std::uint32_t token,
                         DumpSink* sink) {
  const LayerWeights& l = Layer(il);
  std::vector<float> q(kMlaQRows);
  std::vector<float> kv(kMlaCacheWidth);
  Mv(l.attn_q, 0, x, q);
  Mv(l.attn_kv_a_mqa, 0, x, kv);

  cpu::NormRope(q.data() + kMlaNope, kHeads, kMlaQHeadDim, kMlaRope, token,
                c_.rope_theta);
  cpu::NormRope(kv.data() + kMlaLatent, 1, kMlaRope, kMlaRope, token,
                c_.rope_theta);
  cpu::RmsNorm({kv.data(), kMlaLatent}, F32Data(l.attn_kv_a_norm), c_.rms_eps,
               {kv.data(), kMlaLatent});
  if (opt_.f16_cache || opt_.emulate_gpu_f16) {
    for (float& v : kv) {
      v = RoundF16(v);
    }
  }

  auto& cache = mla_cache_[il];
  cache.insert(cache.end(), kv.begin(), kv.end());
  const std::size_t n = std::size_t{token} + 1;

  std::vector<float> ctx(kHeads * kMlaValueDim);
  if (opt_.emulate_gpu_f16) {
    MlaAttendGpuF16(l, q.data(), cache.data(), n, ctx.data());
  } else if (opt_.f64) {
    cpu::MlaAttendAbsorbedF64(l.attn_k_b, l.attn_v_b, q.data(), cache.data(), n,
                              ctx.data());
  } else {
    cpu::MlaAttendAbsorbed(l.attn_k_b, l.attn_v_b, q.data(), cache.data(), n,
                           ctx.data());
  }
  ReleaseAll(l.attn_k_b);
  ReleaseAll(l.attn_v_b);

  if (opt_.check_mla && !opt_.emulate_gpu_f16 &&
      static_cast<int>(il) == check_layer_) {
    if (check_wk_.empty()) {
      check_wk_ = cpu::DequantizeAll(l.attn_k_b);
      check_wv_ = cpu::DequantizeAll(l.attn_v_b);
    }
    // Expanded K/V of this token from the (possibly F16-rounded) cache row.
    const std::size_t kv_floats = std::size_t{kHeads} * kMlaNope;
    check_k_nope_.resize(n * kv_floats);
    check_v_.resize(n * kv_floats);
    check_k_pe_.resize(n * kMlaRope);
    cpu::MlaExpand(check_wk_.data(), check_wv_.data(), kv.data(),
                   check_k_nope_.data() + token * kv_floats,
                   check_v_.data() + token * kv_floats);
    std::copy(
        kv.begin() + kMlaLatent, kv.end(),
        check_k_pe_.begin() + static_cast<std::ptrdiff_t>(token * kMlaRope));
    std::vector<float> ctx2(kHeads * kMlaValueDim);
    cpu::MlaAttendExpanded(q.data(), check_k_nope_.data(), check_k_pe_.data(),
                           check_v_.data(), n, ctx2.data());
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < ctx.size(); ++i) {
      const double d = static_cast<double>(ctx[i]) - ctx2[i];
      num += d * d;
      den += static_cast<double>(ctx2[i]) * ctx2[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    check_.max_rel_l2 = std::max(check_.max_rel_l2, rel);
    ++check_.comparisons;
  }

  const int layer = static_cast<int>(il);
  if (sink != nullptr && sink->Wants("mla_ctx", layer)) {
    sink->Float("mla_ctx", layer, token, ctx);
  }
  std::vector<float> gate(kHeads);
  Mv(l.attn_gate, 0, x, gate);
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const float s = cpu::Sigmoid(gate[h]);
    for (std::uint32_t j = 0; j < kMlaValueDim; ++j) {
      ctx[std::size_t{h} * kMlaValueDim + j] *= s;
    }
  }
  Mv(l.attn_output, 0, ctx, out);
}

void ReferenceModel::Ffn(const TensorRef& gate, const TensorRef& up,
                         const TensorRef& down, float clamp,
                         std::span<const float> x, std::span<float> out,
                         DumpSink* sink, int layer, std::uint32_t token) const {
  std::vector<float> g(gate.rows);
  std::vector<float> u(up.rows);
  std::vector<float> a(gate.rows);
  Mv(gate, 0, x, g);
  Mv(up, 0, x, u);
  cpu::SwiGlu(g, u, clamp, a);
  if (sink != nullptr && layer >= 0) {
    if (sink->Wants("ffn_in", layer))
      sink->Float("ffn_in", layer, token, x);
    if (sink->Wants("ffn_gate", layer))
      sink->Float("ffn_gate", layer, token, g);
    if (sink->Wants("ffn_up", layer))
      sink->Float("ffn_up", layer, token, u);
    if (sink->Wants("ffn_act", layer))
      sink->Float("ffn_act", layer, token, a);
  }
  Mv(down, 0, a, out);
}

void ReferenceModel::Moe(std::uint32_t il, std::span<const float> x,
                         std::span<float> out, std::uint32_t token,
                         DumpSink* sink) {
  const LayerWeights& l = Layer(il);
  cpu::Routing route{};
  if (opt_.f64) {
    std::vector<double> xd(x.begin(), x.end());
    std::vector<double> logits(c_.num_experts);
    cpu::MatVecD(l.router, 0, xd, logits);
    cpu::RouteF64(logits, F32Data(l.exp_probs_b), c_.expert_groups,
                  c_.expert_groups_used, c_.num_experts_used,
                  c_.expert_weights_scale, kExpertWeightsNormFloor, route);
  } else {
    std::vector<float> logits(c_.num_experts);
    Mv(l.router, 0, x, logits);
    cpu::Route(logits, F32Data(l.exp_probs_b), c_.expert_groups,
               c_.expert_groups_used, c_.num_experts_used,
               c_.expert_weights_scale, kExpertWeightsNormFloor, route);
  }
  const int layer = static_cast<int>(il);
  if (sink != nullptr) {
    if (sink->Wants("router_ids", layer)) {
      sink->Int("router_ids", layer, token, route.ids);
    }
    if (sink->Wants("router_w", layer)) {
      sink->Float("router_w", layer, token, route.weights);
    }
    if (sink->Wants("router_top_ids", layer)) {
      sink->Int("router_top_ids", layer, token, route.top_ids);
    }
    if (sink->Wants("router_top_sel", layer)) {
      sink->Float("router_top_sel", layer, token, route.top_sel);
    }
  }

  const std::uint32_t used = c_.num_experts_used;
  std::vector<float> expert_out(std::size_t{used} * kHidden);
  const float clamp = c_.swiglu_clamp_exp[il];
  cpu::ParallelFor(used, [&](std::size_t k0, std::size_t k1) {
    std::vector<float> g(kExpertFf);
    std::vector<float> u(kExpertFf);
    std::vector<float> a(kExpertFf);
    for (std::size_t k = k0; k < k1; ++k) {
      const auto e = static_cast<std::uint64_t>(route.ids[k]);
      Mv(l.ffn_gate_exps, e, x, g, false);
      Mv(l.ffn_up_exps, e, x, u, false);
      cpu::SwiGlu(g, u, clamp, a);
      Mv(l.ffn_down_exps, e, a, {expert_out.data() + k * kHidden, kHidden},
         false);
    }
  });
  if (opt_.f64) {
    for (std::size_t i = 0; i < kHidden; ++i) {
      double acc = 0.0;
      for (std::uint32_t k = 0; k < used; ++k) {
        acc += route.weights_d[k] * expert_out[std::size_t{k} * kHidden + i];
      }
      out[i] = static_cast<float>(acc);
    }
  } else {
    std::fill(out.begin(), out.end(), 0.0F);
    for (std::uint32_t k = 0; k < used; ++k) {
      for (std::size_t i = 0; i < kHidden; ++i) {
        out[i] += route.weights[k] * expert_out[std::size_t{k} * kHidden + i];
      }
    }
  }
  if (sink != nullptr && sink->Wants("ffn_routed", layer)) {
    sink->Float("ffn_routed", layer, token, out);
  }
  std::vector<float> shared(kHidden);
  Ffn(l.ffn_gate_shexp, l.ffn_up_shexp, l.ffn_down_shexp,
      c_.swiglu_clamp_shexp[il], x, shared);
  if (sink != nullptr && sink->Wants("ffn_shared", layer)) {
    sink->Float("ffn_shared", layer, token, shared);
  }
  for (std::size_t i = 0; i < kHidden; ++i) {
    out[i] += shared[i];
  }
}

}  // namespace gufo::models::ling_3_flash
