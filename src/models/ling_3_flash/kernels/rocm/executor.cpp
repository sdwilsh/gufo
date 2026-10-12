#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"

namespace gufo::models::ling_3_flash::rocm {
namespace {

bool Check(hipError_t status, const char* what, std::string* error) {
  if (status == hipSuccess) {
    return true;
  }
  if (error != nullptr) {
    *error = std::string(what) + ": " + hipGetErrorString(status);
  }
  return false;
}

template<class T>
bool Alloc(T** out, std::size_t count, const char* what, std::size_t* total,
           std::string* error) {
  void* ptr = nullptr;
  const std::size_t bytes = count * sizeof(T);
  if (!Check(hipMalloc(&ptr, std::max<std::size_t>(bytes, 16)), what, error)) {
    if (error != nullptr) {
      *error += " (" + std::to_string(bytes) + " bytes)";
    }
    return false;
  }
  *out = static_cast<T*>(ptr);
  if (total != nullptr) {
    *total += bytes;
  }
  return true;
}

constexpr std::uint32_t kMaxDecodeTokens = 8;  ///< small-batch numeric family
constexpr std::uint32_t kVocabRows = kVocab;
constexpr std::uint32_t kQkvStride = 3 * kKdaInner;  ///< q|k|v conv output row

}  // namespace

// ---- Session ---------------------------------------------------------------

Session::~Session() {
  (void)hipFree(state_.kda_state);
  (void)hipFree(state_.kda_conv);
  (void)hipFree(state_.mla_cache);
}

bool Session::Reset(std::string* error_msg) {
  const auto sizes = SessionBytes(state_.cache.capacity);
  state_.position = 0;
  // hipMemset may return before the fill completes; the executor's stream
  // must not start the next token on stale state.
  return Check(hipMemset(state_.kda_state, 0, sizes.kda_state), "reset state",
               error_msg) &&
         Check(hipMemset(state_.kda_conv, 0, sizes.kda_conv), "reset conv",
               error_msg) &&
         Check(hipDeviceSynchronize(), "reset sync", error_msg);
}

RecurrentSnapshot::~RecurrentSnapshot() {
  (void)hipFree(state_);
  (void)hipFree(conv_);
}

std::size_t RecurrentSnapshot::bytes() const noexcept {
  return SessionBytes(0).snapshot;
}

// ---- Executor --------------------------------------------------------------

struct Executor::Buffers {
  float *h{}, *x{}, *proj{}, *conv{}, *g{}, *beta{}, *o{}, *kout{}, *mixer{};
  float *q{}, *kva{}, *kv{}, *gate{}, *qabs{}, *lat{};
  float *ga{}, *gb{}, *act{};
  float *rlogits{}, *rweights{}, *moe_act{}, *exp_out{};
  std::int32_t *ids{}, *tokens{}, *positions{};
  float* logits{};
  std::uint8_t* scratch{};
  std::size_t scratch_bytes{0};
  std::vector<void*> all;
  std::vector<std::size_t> sizes;
};

Executor::Executor(const DeviceModel& model, const Options& options)
    : model_(model), options_(options), b_(std::make_unique<Buffers>()) {}

Executor::~Executor() {
  for (hipEvent_t e : events_) {
    (void)hipEventDestroy(e);
  }
  for (void* p : b_->all) {
    (void)hipFree(p);
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           const Options& options,
                                           std::string* error_msg) {
  if (options.max_batch == 0 || options.max_context == 0 ||
      options.logit_group == 0) {
    if (error_msg != nullptr) {
      *error_msg = "executor limits must be positive";
    }
    return nullptr;
  }
  std::unique_ptr<Executor> e(new Executor(model, options));
  if (!e->Init(error_msg)) {
    return nullptr;
  }
  return e;
}

bool Executor::Init(std::string* error) {
  if (!Check(hipStreamCreate(&stream_), "hipStreamCreate", error)) {
    return false;
  }
  const std::size_t B = options_.max_batch;
  Buffers& b = *b_;
  std::size_t total = 0;
  const auto f = [&](float** p, std::size_t count, const char* what) {
    const bool ok = Alloc(p, count, what, &total, error);
    if (ok) {
      b.all.push_back(*p);
      b.sizes.push_back(count * sizeof(float));
    }
    return ok;
  };
  bool ok =
      f(&b.h, B * kHidden, "h") && f(&b.x, B * kHidden, "x") &&
      f(&b.proj, B * kKdaProjRows, "proj") &&
      f(&b.conv, B * kQkvStride, "conv") && f(&b.g, B * kKdaInner, "g") &&
      f(&b.beta, B * kHeads, "beta") && f(&b.o, B * kKdaInner, "o") &&
      f(&b.kout, B * kKdaInner, "kout") && f(&b.mixer, B * kHidden, "mixer") &&
      f(&b.q, B * kMlaQRows, "q") && f(&b.kva, B * kMlaCacheWidth, "kva") &&
      f(&b.kv, B * kMlaCacheWidth, "kv") && f(&b.gate, B * kHeads, "gate") &&
      f(&b.qabs, B * kMlaQAbsorbedRowFloats, "qabs") &&
      f(&b.lat, B * kHeads * kMlaLatent, "lat") &&
      f(&b.ga, B * kDenseFf, "ga") && f(&b.gb, B * kDenseFf, "gb") &&
      f(&b.act, B * kDenseFf, "act") &&
      f(&b.rlogits, B * kExperts, "rlogits") &&
      f(&b.rweights, B * kExpertsUsed, "rweights") &&
      f(&b.moe_act, B * kExpertsUsed * kExpertFf, "moe_act") &&
      f(&b.exp_out, B * kExpertsUsed * kHidden, "exp_out") &&
      f(&b.logits, std::size_t{options_.logit_group} * kVocabRows, "logits");
  const auto fi = [&](std::int32_t** p, std::size_t count, const char* what) {
    const bool good = Alloc(p, count, what, &total, error);
    if (good) {
      b.all.push_back(*p);
      b.sizes.push_back(count * sizeof(std::int32_t));
    }
    return good;
  };
  ok = ok && fi(&b.ids, B * kExpertsUsed, "ids") &&
       fi(&b.tokens, B, "tokens") && fi(&b.positions, B, "positions");
  if (!ok) {
    return false;
  }
  std::size_t scratch = 0;
  for (std::uint32_t k : {kHidden, kKdaInner, kDenseFf, kSharedFf}) {
    scratch = std::max(
        scratch, dense::ScratchBytes(WeightType::kQ8_0, options_.max_batch, k));
  }
  scratch = std::max(scratch, mla::DecodeScratchBytes(options_.max_context));
  scratch = std::max(scratch, moe::PrefillScratchBytes(options_.max_batch));
  scratch = std::max(scratch, moe::DecodeScratchBytes());
  b.scratch_bytes = scratch;
  if (!Alloc(&b.scratch, scratch, "kernel scratch", &total, error)) {
    return false;
  }
  b.all.push_back(b.scratch);
  b.sizes.push_back(scratch);
  scratch_total_ = total;
  if (const char* poison = std::getenv("GUFO_LING_POISON");
      poison != nullptr && *poison == '1') {
    // Debug aid: fill every activation buffer with NaN/-1 so a kernel that
    // reads memory it was not given shows up as non-finite output.
    (void)hipDeviceSynchronize();
    std::fprintf(stderr, "[ling] poisoning activation buffers\n");
    for (std::size_t i = 0; i < b.all.size(); ++i) {
      (void)hipMemset(b.all[i], 0xFF, b.sizes[i]);
    }
    (void)hipDeviceSynchronize();
  }
  events_.resize(layer_count() + 2);
  for (auto& e : events_) {
    if (!Check(hipEventCreate(&e), "hipEventCreate", error)) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<Session> Executor::CreateSession(std::uint32_t max_context,
                                                 std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    if (error_msg != nullptr) {
      *error_msg = "session context " + std::to_string(max_context) +
                   " exceeds the executor limit " +
                   std::to_string(options_.max_context);
    }
    return nullptr;
  }
  std::unique_ptr<Session> s(new Session());
  const auto sizes = SessionBytes(max_context);
  s->state_.cache.capacity = max_context;
  if (!Alloc(&s->state_.kda_state, sizes.kda_state / sizeof(float), "KDA state",
             nullptr, error_msg) ||
      !Alloc(&s->state_.kda_conv, sizes.kda_conv / sizeof(float), "KDA conv",
             nullptr, error_msg)) {
    return nullptr;
  }
  void* cache = nullptr;
  if (!Check(hipMalloc(&cache, sizes.mla_cache), "MLA cache", error_msg)) {
    return nullptr;
  }
  s->state_.mla_cache = cache;
  // Zero everything: padded attention tiles may touch rows past the context.
  if (!Check(hipMemset(cache, 0, sizes.mla_cache), "clear MLA cache",
             error_msg) ||
      !Check(hipDeviceSynchronize(), "clear MLA cache sync", error_msg) ||
      !s->Reset(error_msg)) {
    return nullptr;
  }
  return s;
}

bool Executor::UploadInputs(std::span<const std::int32_t> tokens,
                            std::uint32_t n, std::uint32_t first_pos,
                            std::string* error) {
  std::vector<std::int32_t> positions(n);
  for (std::uint32_t t = 0; t < n; ++t) {
    positions[t] = static_cast<std::int32_t>(first_pos + t);
  }
  if (!tokens.empty() &&
      !Check(hipMemcpyAsync(b_->tokens, tokens.data(), n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "upload tokens", error)) {
    return false;
  }
  return Check(hipMemcpyAsync(b_->positions, positions.data(),
                              n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                              stream_),
               "upload positions", error) &&
         Check(hipStreamSynchronize(stream_), "upload sync", error);
}

void Executor::DumpFloat(GpuTensorSink* sink, const char* name, int layer,
                         const float* device, std::uint32_t rows,
                         std::uint32_t width) {
  if (sink == nullptr || !sink->Wants(name, layer)) {
    return;
  }
  std::vector<float> host(std::size_t{rows} * width);
  (void)hipMemcpyAsync(host.data(), device, host.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream_);
  (void)hipStreamSynchronize(stream_);
  sink->Float(name, layer, width, host);
}

void Executor::DumpInt(GpuTensorSink* sink, const char* name, int layer,
                       const std::int32_t* device, std::uint32_t rows,
                       std::uint32_t width) {
  if (sink == nullptr || !sink->Wants(name, layer)) {
    return;
  }
  std::vector<std::int32_t> host(std::size_t{rows} * width);
  (void)hipMemcpyAsync(host.data(), device, host.size() * sizeof(std::int32_t),
                       hipMemcpyDeviceToHost, stream_);
  (void)hipStreamSynchronize(stream_);
  sink->Int(name, layer, width, host);
}

void Executor::Q8(const DeviceTensor& w, const float* x, float* y,
                  std::uint32_t n) {
  static const bool force_gemm = std::getenv("GUFO_LING_FORCE_GEMM") != nullptr;
  if (n <= kMaxDecodeTokens && !force_gemm) {
    dense::GemvQ8_0(w.data, x, y, n, w.cols, w.rows, b_->scratch, stream_);
  } else {
    dense::GemmQ8_0(w.data, x, y, n, w.cols, w.rows, b_->scratch, stream_);
  }
}

void Executor::Kda(const DeviceLayer& l, const Session& session,
                   std::uint32_t il, std::uint32_t n, GpuTensorSink* sink) {
  Buffers& b = *b_;
  const std::uint32_t k = KdaIndex(il);
  float* state = session.state_.kda_state + std::size_t{k} * kKdaStateFloats;
  float* conv_state =
      session.state_.kda_conv + std::size_t{k} * kKdaConvStateFloats;
  Q8(l.kda_proj, b.x, b.proj, n);
  if (n == 1) {
    kda::Conv3Decode(b.proj, l.conv_q.f32(), l.conv_k.f32(), l.conv_v.f32(),
                     conv_state, b.conv, stream_);
  } else {
    kda::Conv3Prefill(b.proj, l.conv_q.f32(), l.conv_k.f32(), l.conv_v.f32(),
                      conv_state, b.conv, n, stream_);
  }
  kda::GateBetaPrep(b.proj, l.ssm_a.f32(), l.ssm_dt.f32(), b.g, b.beta, n,
                    stream_);
  kda::L2NormHeads(b.conv, kQkvStride, n, stream_);
  kda::L2NormHeads(b.conv + kKdaInner, kQkvStride, n, stream_);
  float* q = b.conv;
  float* kk = b.conv + kKdaInner;
  float* v = b.conv + 2 * kKdaInner;
  if (n == 1) {
    kda::RecurrentDecode(state, q, kk, v, b.g, b.beta, b.o, stream_);
  } else {
    kda::PrefillRowSplit(state, q, kQkvStride, kk, kQkvStride, v, kQkvStride,
                         b.g, b.beta, b.o, n, stream_);
  }
  DumpFloat(sink, "kda_o", static_cast<int>(il), b.o, n, kKdaInner);
  kda::OutputNormGate(b.o, b.proj, l.ssm_norm.f32(), b.kout, n, stream_);
  Q8(l.attn_output, b.kout, b.mixer, n);
}

void Executor::Mla(const DeviceLayer& l, const Session& session,
                   std::uint32_t il, std::uint32_t first_pos, std::uint32_t n) {
  Buffers& b = *b_;
  const SessionState& st = session.state_;
  auto* cache = static_cast<__half*>(st.mla_cache) +
                std::size_t{MlaIndex(il)} * st.cache.LayerStride();
  Q8(l.attn_q, b.x, b.q, n);
  Q8(l.kv_a, b.x, b.kva, n);
  Q8(l.attn_gate, b.x, b.gate, n);
  mla::LatentNormRope(b.kva, l.kv_a_norm.f32(), b.positions, b.kv, n, stream_);
  mla::CacheAppend(cache, b.kv, first_pos, n, stream_);
  mla::QAbsorb(b.q, l.k_b.data, b.positions, b.qabs, n, stream_);
  if (n <= kMaxDecodeTokens) {
    // Small-batch family: every token runs the decode kernel on its own
    // context, so a chunk of n gives the same bits as n single-token steps
    // whatever the context length (split-K plans depend only on the context).
    for (std::uint32_t t = 0; t < n; ++t)
      mla::DecodeAttention(b.qabs + (std::size_t{t} * kHeads * kMlaCacheWidth),
                           cache, first_pos + t + 1,
                           b.lat + (std::size_t{t} * kHeads * kMlaLatent),
                           b.scratch, stream_);
  } else {
    mla::PrefillAttention(b.qabs, cache, first_pos, n, b.lat, b.scratch,
                          stream_);
  }
  mla::VbUpProjGate(b.lat, l.v_b.data, b.gate, b.kout, n, stream_);
  Q8(l.attn_output, b.kout, b.mixer, n);
}

void Executor::Ffn(const DeviceLayer& l, std::uint32_t il, std::uint32_t n,
                   GpuTensorSink* sink) {
  Buffers& b = *b_;
  const Config& c = model_.config();
  const float shexp_limit = c.swiglu_clamp_shexp[il];
  dense::RmsNormRows(b.h, l.ffn_norm.f32(), b.x, n, kHidden, 1, kRmsEps,
                     stream_);
  if (!l.moe) {
    Q8(l.ffn_gate, b.x, b.ga, n);
    Q8(l.ffn_up, b.x, b.gb, n);
    dense::SwiGluClamped(b.ga, b.gb, b.act, std::size_t{n} * kDenseFf,
                         shexp_limit, stream_);
    Q8(l.ffn_down, b.act, b.mixer, n);
    dense::AddResidual(b.h, b.mixer, std::size_t{n} * kHidden, stream_);
    return;
  }
  dense::GemvF32(l.router.f32(), b.x, b.rlogits, n, kHidden, kExperts, stream_);
  moe::Route(b.rlogits, l.exp_probs_b.f32(), b.ids, b.rweights, n, stream_);
  DumpInt(sink, "router_ids", static_cast<int>(il), b.ids, n, kExpertsUsed);
  DumpFloat(sink, "router_w", static_cast<int>(il), b.rweights, n,
            kExpertsUsed);
  const float exp_limit = c.swiglu_clamp_exp[il];
  if (n <= kMaxDecodeTokens) {
    moe::ExpertGateUpDecode(l.gate_exps.data, l.up_exps.data, l.gate_exps.type,
                            b.x, b.ids, n, exp_limit, b.moe_act, b.scratch,
                            stream_);
    moe::ExpertDownDecode(l.down_exps.data, l.down_exps.type, b.moe_act, b.ids,
                          n, b.exp_out, b.scratch, stream_);
  } else {
    moe::ExpertGateUpPrefill(l.gate_exps.data, l.up_exps.data, l.gate_exps.type,
                             b.x, b.ids, n, exp_limit, b.moe_act, b.scratch,
                             stream_);
    moe::ExpertDownPrefill(l.down_exps.data, l.down_exps.type, b.moe_act, b.ids,
                           n, b.exp_out, b.scratch, stream_);
  }
  Q8(l.gate_shexp, b.x, b.ga, n);
  Q8(l.up_shexp, b.x, b.gb, n);
  dense::SwiGluClamped(b.ga, b.gb, b.act, std::size_t{n} * kSharedFf,
                       shexp_limit, stream_);
  Q8(l.down_shexp, b.act, b.mixer, n);
  moe::Combine(b.h, b.exp_out, b.rweights, b.mixer, n, stream_);
}

bool Executor::Block(const Session& session, std::uint32_t il,
                     std::uint32_t first_pos, std::uint32_t n,
                     GpuTensorSink* sink, std::string* error) {
  Buffers& b = *b_;
  const DeviceLayer& l = model_.layers()[il - model_.first_layer()];
  const int layer = static_cast<int>(il);
  dense::RmsNormRows(b.h, l.attn_norm.f32(), b.x, n, kHidden, 1, kRmsEps,
                     stream_);
  if (l.kind == LayerKind::kKda) {
    Kda(l, session, il, n, sink);
  } else {
    Mla(l, session, il, first_pos, n);
  }
  DumpFloat(sink, "mixer_out", layer, b.mixer, n, kHidden);
  dense::AddResidual(b.h, b.mixer, std::size_t{n} * kHidden, stream_);
  DumpFloat(sink, "h_mid", layer, b.h, n, kHidden);
  Ffn(l, il, n, sink);
  DumpFloat(sink, "h_out", layer, b.h, n, kHidden);
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    if (error != nullptr) {
      *error = "layer " + std::to_string(il) +
               " launch failed: " + hipGetErrorString(status);
    }
    return false;
  }
  return true;
}

bool Executor::Head(std::uint32_t n, std::uint32_t from,
                    std::span<float> logits, std::string* error) {
  Buffers& b = *b_;
  const std::uint32_t rows = n - from;
  dense::RmsNormRows(b.h + std::size_t{from} * kHidden,
                     model_.output_norm().f32(), b.x, rows, kHidden, 1, kRmsEps,
                     stream_);
  for (std::uint32_t g0 = 0; g0 < rows; g0 += options_.logit_group) {
    const std::uint32_t gn = std::min(options_.logit_group, rows - g0);
    dense::GemvF16(model_.output().f16(), b.x + std::size_t{g0} * kHidden,
                   b.logits, gn, kHidden, kVocabRows, stream_);
    if (!Check(hipMemcpyAsync(logits.data() + std::size_t{g0} * kVocabRows,
                              b.logits, std::size_t{gn} * kVocabRows * 4,
                              hipMemcpyDeviceToHost, stream_),
               "copy logits", error) ||
        !Check(hipStreamSynchronize(stream_), "logits sync", error)) {
      return false;
    }
  }
  return true;
}

bool Executor::Forward(Session& session, const ForwardRequest& r,
                       std::string* error) {
  const auto n = static_cast<std::uint32_t>(r.tokens.size());
  const std::uint32_t first_pos = session.position();
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (first_layer() != 0) {
    return fail("forward needs the trunk from block 0 (layer_first is set)");
  }
  if (n == 0 || n > options_.max_batch) {
    return fail("forward batch of " + std::to_string(n) +
                " tokens outside 1.." + std::to_string(options_.max_batch));
  }
  if (first_pos + n > session.capacity()) {
    return fail("context overflow: " + std::to_string(first_pos) + " + " +
                std::to_string(n) + " > " + std::to_string(session.capacity()));
  }
  std::uint32_t from = n;
  if (!r.logits.empty()) {
    from = r.logits_from == UINT32_MAX ? n - 1 : r.logits_from;
    if (from >= n || r.logits.size() < std::size_t{n - from} * kVocabRows) {
      return fail("logits buffer too small for rows " + std::to_string(from) +
                  ".." + std::to_string(n));
    }
  }
  for (const std::int32_t t : r.tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= kVocab) {
      return fail("token id " + std::to_string(t) + " outside the vocabulary");
    }
  }
  if (!UploadInputs(r.tokens, n, first_pos, error)) {
    return false;
  }
  Buffers& b = *b_;
  const std::uint32_t layers = layer_count();
  (void)hipEventRecord(events_[0], stream_);
  dense::EmbedQ4K(model_.token_embd().data, b.tokens, b.h, n, kHidden, stream_);
  DumpFloat(r.sink, "embd", 0, b.h, n, kHidden);
  for (std::uint32_t il = 0; il < layers; ++il) {
    if (!Block(session, il, first_pos, n, r.sink, error)) {
      (void)hipStreamSynchronize(stream_);
      return false;
    }
    (void)hipEventRecord(events_[il + 1], stream_);
  }
  if (!r.logits.empty() && !Head(n, from, r.logits, error)) {
    return false;
  }
  (void)hipEventRecord(events_[layers + 1], stream_);
  if (!Check(hipStreamSynchronize(stream_), "forward", error) ||
      !Check(hipGetLastError(), "forward", error)) {
    return false;
  }
  timing_.layer_ms.assign(layers, 0.0F);
  for (std::uint32_t il = 0; il < layers; ++il) {
    // Dumps synchronise mid-stream; events still bracket each block.
    (void)hipEventElapsedTime(&timing_.layer_ms[il], events_[il],
                              events_[il + 1]);
  }
  float total = 0.0F;
  float head = 0.0F;
  (void)hipEventElapsedTime(&total, events_[0], events_[layers + 1]);
  (void)hipEventElapsedTime(&head, events_[layers], events_[layers + 1]);
  timing_.total_ms = total;
  timing_.head_ms = head;
  session.set_position(first_pos + n);
  return true;
}

bool Executor::RunBlock(Session& session, std::uint32_t il,
                        std::uint32_t first_pos, std::span<const float> in,
                        std::span<float> out, GpuTensorSink* sink,
                        std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (il < first_layer() || il >= first_layer() + layer_count()) {
    return fail("block " + std::to_string(il) + " is not resident");
  }
  if (in.empty() || in.size() % kHidden != 0 || out.size() != in.size()) {
    return fail("block input must be whole hidden rows and match the output");
  }
  const auto n = static_cast<std::uint32_t>(in.size() / kHidden);
  if (n > options_.max_batch || first_pos + n > session.capacity()) {
    return fail("block batch exceeds the executor or session limits");
  }
  Buffers& b = *b_;
  if (!UploadInputs({}, n, first_pos, error) ||
      !Check(hipMemcpyAsync(b.h, in.data(), in.size_bytes(),
                            hipMemcpyHostToDevice, stream_),
             "upload block input", error)) {
    return false;
  }
  if (!Block(session, il, first_pos, n, sink, error)) {
    (void)hipStreamSynchronize(stream_);
    return false;
  }
  return Check(hipMemcpyAsync(out.data(), b.h, in.size_bytes(),
                              hipMemcpyDeviceToHost, stream_),
               "download block output", error) &&
         Check(hipStreamSynchronize(stream_), "block", error);
}

std::unique_ptr<RecurrentSnapshot> Executor::SaveRecurrent(
    const Session& session, std::string* error) {
  std::unique_ptr<RecurrentSnapshot> s(new RecurrentSnapshot());
  const auto sizes = SessionBytes(0);
  if (!Alloc(&s->state_, sizes.kda_state / 4, "snapshot state", nullptr,
             error) ||
      !Alloc(&s->conv_, sizes.kda_conv / 4, "snapshot conv", nullptr, error)) {
    return nullptr;
  }
  kda::SnapshotState(session.state_.kda_state, session.state_.kda_conv,
                     s->state_, s->conv_, stream_);
  if (!Check(hipStreamSynchronize(stream_), "snapshot", error)) {
    return nullptr;
  }
  return s;
}

bool Executor::RestoreRecurrent(Session& session,
                                const RecurrentSnapshot& snapshot,
                                std::string* error) {
  kda::RestoreState(session.state_.kda_state, session.state_.kda_conv,
                    snapshot.state_, snapshot.conv_, stream_);
  return Check(hipStreamSynchronize(stream_), "restore", error);
}

bool Executor::ReadKdaState(const Session& session, std::uint32_t il,
                            std::vector<float>* state, std::vector<float>* conv,
                            std::string* error) {
  if (il >= kTrunkLayers || IsMlaLayer(il)) {
    if (error != nullptr) {
      *error = "block " + std::to_string(il) + " is not a KDA layer";
    }
    return false;
  }
  const std::uint32_t k = KdaIndex(il);
  state->resize(kKdaStateFloats);
  conv->resize(kKdaConvStateFloats);
  return Check(hipMemcpy(
                   state->data(),
                   session.state_.kda_state + std::size_t{k} * kKdaStateFloats,
                   kKdaStateFloats * 4, hipMemcpyDeviceToHost),
               "read state", error) &&
         Check(hipMemcpy(conv->data(),
                         session.state_.kda_conv +
                             std::size_t{k} * kKdaConvStateFloats,
                         kKdaConvStateFloats * 4, hipMemcpyDeviceToHost),
               "read conv", error);
}

bool Executor::ReadMlaCache(const Session& session, std::uint32_t il,
                            std::uint32_t rows, std::vector<std::uint16_t>* out,
                            std::string* error) {
  if (il >= kTrunkLayers || !IsMlaLayer(il) || rows > session.capacity()) {
    if (error != nullptr) {
      *error = "block " + std::to_string(il) + " is not an MLA layer";
    }
    return false;
  }
  out->resize(std::size_t{rows} * kMlaCacheWidth);
  const auto* src =
      static_cast<const std::uint16_t*>(session.state_.mla_cache) +
      std::size_t{MlaIndex(il)} * session.state_.cache.LayerStride();
  return Check(
      hipMemcpy(out->data(), src, out->size() * 2, hipMemcpyDeviceToHost),
      "read cache", error);
}

}  // namespace gufo::models::ling_3_flash::rocm
