#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

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

/// Bytes of one weight tensor (all `depth` slices) in its GGUF encoding.
std::size_t TensorBytes(const DeviceTensor& t) {
  double per_element = 0.0;
  switch (t.type) {
    case WeightType::kF32:
      per_element = 4.0;
      break;
    case WeightType::kF16:
      per_element = 2.0;
      break;
    case WeightType::kQ8_0:
      per_element = 34.0 / 32.0;
      break;
    case WeightType::kQ4_K:
      per_element = 144.0 / 256.0;
      break;
    case WeightType::kQ5_K:
      per_element = 176.0 / 256.0;
      break;
    case WeightType::kQ6_K:
      per_element = 210.0 / 256.0;
      break;
    case WeightType::kIQ4_XS:
      per_element = 136.0 / 256.0;
      break;
  }
  return static_cast<std::size_t>(per_element * t.rows * t.cols * t.depth);
}

constexpr std::uint32_t kQkvStride = 3 * kKdaInner;  ///< q|k|v conv output row

}  // namespace

// ---- Session ---------------------------------------------------------------

void Session::DestroyGraphs() noexcept {
  if (!graphs_.empty()) {
    // ROCm keeps the memory a graph's kernels use referenced when the graph
    // is destroyed right after its launch completed on the stream (a session
    // dropped after its first replays): the model's later hipFree calls then
    // succeed but its weights stay allocated, and a second Model::Load in the
    // process runs out of memory. Draining the device first releases them
    // (reload_test); ~Session's hipFree calls synchronise anyway.
    (void)hipDeviceSynchronize();
    (void)hipGetLastError();
  }
  for (auto& [key, exec] : graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  graphs_.clear();
  warmed_.clear();
}

Session::~Session() {
  // A background snapshot copy still reads the cache rows freed below. After a
  // device loss a copy may be wedged in a HIP call, and then this wait (and a
  // shutdown that destroys sessions) waits with it; nothing is leaked or timed
  // out on purpose, so rows are never freed under a running copy.
  for (;;) {
    try {
      AwaitRowHolds(0);
      break;
    } catch (...) {
      // condition_variable::wait failed (resources): a destructor must not
      // throw, and the rows must not be freed under a running copy. Retry.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  DestroyGraphs();
  if (forward_done_ != nullptr) {
    (void)hipEventDestroy(forward_done_);
  }
  (void)hipFree(state_.kda_state);
  (void)hipFree(state_.kda_conv);
  (void)hipFree(state_.mla_cache);
  (void)hipFree(state_.mtp_cache);
  (void)hipFree(state_.mtp_h);
  (void)hipFree(state_.mla_expanded);
  if (expanded_pool_ != nullptr) {
    expanded_pool_->fetch_add(expanded_bytes_);
  }
}

bool Session::Reset(std::string* error_msg) {
  const auto sizes = SessionBytes(state_.cache.capacity);
  set_position(0);  // also drops every expanded K/V watermark
  mtp_valid_ = false;
  // hipMemset may return before the fill completes; the executor's stream
  // must not start the next token on stale state.
  return Check(hipMemset(state_.kda_state, 0, sizes.kda_state), "reset state",
               error_msg) &&
         Check(hipMemset(state_.kda_conv, 0, sizes.kda_conv), "reset conv",
               error_msg) &&
         Check(hipDeviceSynchronize(), "reset sync", error_msg);
}

// ---- Executor --------------------------------------------------------------

struct Executor::Buffers {
  float *h{}, *x{}, *proj{}, *conv{}, *g{}, *beta{}, *o{}, *kout{}, *mixer{};
  float *q{}, *kva{}, *kv{}, *gate{}, *qabs{}, *lat{};
  float *ga{}, *gb{}, *act{};
  float *rlogits{}, *rweights{}, *moe_act{}, *exp_out{};
  std::int32_t *ids{}, *tokens{}, *positions{};
  float* logits{};
  /// MTP: [0] carry copy, [1 + t] output-normed trunk row t of the last
  /// forward ([max_batch + 1][kHidden]); draft chain hidden; rollback points
  /// ([kMaxDrafts][35][state floats] and conv); device argmax results.
  float *hn{}, *mtp_hn{}, *kda_ckpt{}, *conv_ckpt{}, *argmax_val{};
  std::int32_t* argmax_idx{};
  std::uint32_t* route_counter{};    ///< zero between uses (RouteDecode)
  std::uint8_t *xq{}, *kq{}, *aq{};  ///< activation planes (fused norms)
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
  for (hipEvent_t e : mark_events_) {
    (void)hipEventDestroy(e);
  }
  for (hipEvent_t e : draft_events_) {
    if (e != nullptr) {
      (void)hipEventDestroy(e);
    }
  }
  for (void* p : b_->all) {
    (void)hipFree(p);
  }
  if (fork_event_ != nullptr) {
    (void)hipEventDestroy(fork_event_);
  }
  if (join_event_ != nullptr) {
    (void)hipEventDestroy(join_event_);
  }
  if (branch_stream_ != nullptr) {
    (void)hipStreamDestroy(branch_stream_);
  }
  if (staging_ != nullptr) {
    (void)hipHostFree(staging_);
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
  if (transfer_stream_ != nullptr) {
    (void)hipStreamDestroy(transfer_stream_);
  }
  if (transfer_staging_ != nullptr) {
    (void)hipHostFree(transfer_staging_);
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
  e->expanded_pool_ =
      std::make_shared<std::atomic<std::uint64_t>>(options.expanded_kv_bytes);
  if (!e->Init(error_msg)) {
    return nullptr;
  }
  return e;
}

bool Executor::Init(std::string* error) {
  if (!Check(hipStreamCreate(&stream_), "hipStreamCreate", error) ||
      !Check(hipStreamCreateWithFlags(&transfer_stream_, hipStreamNonBlocking),
             "transfer stream", error) ||
      !Check(hipHostMalloc(&transfer_staging_, kTransferStagingBytes),
             "pinned snapshot staging", error)) {
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
  if (ok && options_.mtp) {
    if (model_.mtp() == nullptr) {
      if (error != nullptr) {
        *error = "MTP requested but the device model has no MTP block";
      }
      return false;
    }
    ok = f(&b.hn, (B + 1) * kHidden, "mtp hn") &&
         f(&b.mtp_hn, std::size_t{kMaxDecodeTokens} * kHidden,
           "mtp chain hidden") &&
         f(&b.kda_ckpt, std::size_t{kMaxDrafts} * kKdaLayers * kKdaStateFloats,
           "kda rollback points") &&
         f(&b.conv_ckpt,
           std::size_t{kMaxDrafts} * kKdaLayers * kKdaConvStateFloats,
           "conv rollback points");
  }
  ok = ok && f(&b.argmax_val, options_.logit_group, "argmax values");
  const auto fi = [&](std::int32_t** p, std::size_t count, const char* what) {
    const bool good = Alloc(p, count, what, &total, error);
    if (good) {
      b.all.push_back(*p);
      b.sizes.push_back(count * sizeof(std::int32_t));
    }
    return good;
  };
  const auto fu = [&](std::uint8_t** p, std::size_t bytes, const char* what) {
    const bool good = Alloc(p, bytes, what, &total, error);
    if (good) {
      b.all.push_back(*p);
      b.sizes.push_back(bytes);
    }
    return good;
  };
  ok = ok && fu(&b.xq, dense::QuantActBytes(kMaxDecodeTokens, kHidden), "xq") &&
       fu(&b.kq, dense::QuantActBytes(kMaxDecodeTokens, kKdaInner), "kq") &&
       fu(&b.aq, dense::QuantActBytes(kMaxDecodeTokens, kSharedFf), "aq");
  ok = ok && fi(&b.ids, B * kExpertsUsed, "ids") &&
       fi(&b.tokens, B, "tokens") && fi(&b.positions, B, "positions") &&
       fi(&b.argmax_idx, options_.logit_group, "argmax indices");
  ok = ok && Alloc(&b.route_counter, 1, "route counter", &total, error);
  if (ok) {
    b.all.push_back(b.route_counter);
    b.sizes.push_back(sizeof(std::uint32_t));
    ok = Check(hipMemset(b.route_counter, 0, sizeof(std::uint32_t)),
               "route counter init", error);
  }
  if (!ok) {
    return false;
  }
  std::size_t scratch = 0;
  for (std::uint32_t k :
       {kHidden, kKdaInner, kDenseFf, kSharedFf, 2 * kHidden}) {
    scratch = std::max(
        scratch, dense::ScratchBytes(WeightType::kQ8_0, options_.max_batch, k));
  }
  scratch = std::max(scratch, mla::DecodeScratchBytes(options_.max_context));
  scratch = std::max(scratch, mla::NonAbsorbedScratchBytes(options_.max_batch));
  scratch = std::max(scratch, moe::PrefillScratchBytes(options_.max_batch));
  scratch = std::max(scratch, moe::DecodeScratchBytes());
  b.scratch_bytes = scratch;
  if (!Alloc(&b.scratch, scratch, "kernel scratch", &total, error)) {
    return false;
  }
  b.all.push_back(b.scratch);
  b.sizes.push_back(scratch);
  scratch_total_ = total;
  graphs_enabled_ = options_.graphs;
  if (!Check(hipStreamCreateWithFlags(&branch_stream_, hipStreamNonBlocking),
             "branch stream", error) ||
      !Check(hipEventCreateWithFlags(&fork_event_, hipEventDisableTiming),
             "fork event", error) ||
      !Check(hipEventCreateWithFlags(&join_event_, hipEventDisableTiming),
             "join event", error)) {
    return false;
  }
  // tokens[8], positions[8], argmax[1] for decode steps.
  if (!Check(hipHostMalloc(reinterpret_cast<void**>(&staging_),
                           32 * sizeof(std::int32_t), hipHostMallocDefault),
             "pinned staging", error)) {
    return false;
  }
  events_.resize(layer_count() + 2);
  for (auto& e : events_) {
    if (!Check(hipEventCreate(&e), "hipEventCreate", error)) {
      return false;
    }
  }
  for (auto& e : draft_events_) {
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
  if (!Check(hipEventCreateWithFlags(&s->forward_done_, hipEventDisableTiming),
             "session event", error_msg)) {
    return nullptr;
  }
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
  if (options_.mtp) {
    const std::size_t mtp_cache_bytes =
        std::size_t{max_context} * kMlaCacheWidth * sizeof(__half);
    void* mtp_cache = nullptr;
    if (!Check(hipMalloc(&mtp_cache, mtp_cache_bytes), "MTP cache",
               error_msg)) {
      return nullptr;
    }
    s->state_.mtp_cache = mtp_cache;
    if (!Alloc(&s->state_.mtp_h, kHidden, "MTP hidden carry", nullptr,
               error_msg) ||
        !Check(hipMemset(mtp_cache, 0, mtp_cache_bytes), "clear MTP cache",
               error_msg) ||
        !Check(hipMemset(s->state_.mtp_h, 0, kHidden * sizeof(float)),
               "clear MTP carry", error_msg)) {
      return nullptr;
    }
  }
  // Persistent expanded K/V: whole 2048-key blocks of this context that the
  // shared budget still affords (all-or-nothing per block, first come first
  // served). Zeroed so rows no append has expanded yet read as finite.
  {
    const std::uint32_t block_rows = 2048;
    const std::uint64_t block_bytes =
        std::uint64_t{block_rows} * kMlaLayers * mla::ExpandedLayerBytes(1);
    const std::uint32_t want = (max_context + block_rows - 1) / block_rows;
    std::uint32_t got = 0;
    std::uint64_t free_bytes = expanded_pool_->load();
    while (got < want) {
      const std::uint32_t take = want - got;
      const std::uint64_t need = block_bytes * take;
      const std::uint64_t grant =
          std::min<std::uint64_t>(need, free_bytes / block_bytes * block_bytes);
      if (grant == 0) {
        break;
      }
      if (expanded_pool_->compare_exchange_weak(free_bytes,
                                                free_bytes - grant)) {
        got = static_cast<std::uint32_t>(grant / block_bytes);
        s->expanded_pool_ = expanded_pool_;
        s->expanded_bytes_ = grant;
        break;
      }
    }
    if (got > 0) {
      void* store = nullptr;
      if (!Check(hipMalloc(&store, s->expanded_bytes_), "expanded K/V store",
                 error_msg)) {
        return nullptr;
      }
      s->state_.mla_expanded = store;
      s->state_.expanded_rows = got * block_rows;
      if (!Check(hipMemset(store, 0, s->expanded_bytes_), "clear expanded K/V",
                 error_msg)) {
        return nullptr;
      }
    }
  }
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

bool Executor::UploadRows(std::span<const std::int32_t> tokens,
                          std::span<const std::int32_t> positions,
                          std::string* error) {
  return Check(hipMemcpyAsync(b_->tokens, tokens.data(), tokens.size_bytes(),
                              hipMemcpyHostToDevice, stream_),
               "upload tokens", error) &&
         Check(hipMemcpyAsync(b_->positions, positions.data(),
                              positions.size_bytes(), hipMemcpyHostToDevice,
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

void Executor::Mark(const char* label, std::size_t bytes) {
  if (!profiling_) {
    return;
  }
  if (mark_events_.size() < marks_.size() + 2) {
    hipEvent_t e = nullptr;
    (void)hipEventCreate(&e);
    mark_events_.push_back(e);
  }
  marks_.push_back({label, bytes});
  (void)hipEventRecord(mark_events_[marks_.size()], stream_);
}

void Executor::BeginProfile() {
  profiling_ = true;
  marks_.clear();
  if (mark_events_.empty()) {
    mark_events_.resize(1);
    (void)hipEventCreate(&mark_events_[0]);
  }
  (void)hipEventRecord(mark_events_[0], stream_);
}

double Executor::StepMs() const {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - step_start_)
      .count();
}

void Executor::FinishProfile(std::vector<OpTiming>* out) {
  profiling_ = false;
  for (std::size_t i = 0; i < marks_.size(); ++i) {
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, mark_events_[i], mark_events_[i + 1]);
    auto it = std::find_if(out->begin(), out->end(), [&](const OpTiming& t) {
      return t.name == marks_[i].label;
    });
    if (it == out->end()) {
      out->push_back({marks_[i].label, 0, 0.0, 0.0});
      it = out->end() - 1;
    }
    ++it->count;
    it->ms += ms;
    it->bytes += static_cast<double>(marks_[i].bytes);
  }
  marks_.clear();
  for (const auto& [name, ms] : detail_ops_) {
    auto it = std::find_if(out->begin(), out->end(),
                           [&](const OpTiming& t) { return t.name == name; });
    if (it == out->end()) {
      out->push_back({name, 0, 0.0, 0.0});
      it = out->end() - 1;
    }
    ++it->count;
    it->ms += ms;
  }
  detail_ops_.clear();
}

void Executor::Q8(const DeviceTensor& w, const float* x, float* y,
                  std::uint32_t n) {
  if (n <= kMaxDecodeTokens) {
    dense::GemvQ8_0(w.data, x, y, n, w.cols, w.rows, b_->scratch, stream_);
  } else {
    dense::GemmQ8_0(w.data, x, y, n, w.cols, w.rows, b_->scratch, stream_);
  }
}

void Executor::Q8Quantized(
    std::initializer_list<std::pair<const DeviceTensor*, float*>> groups,
    const void* quant, std::uint32_t n, std::uint32_t k) {
  assert(groups.size() >= 1 && groups.size() <= 3);
  dense::Q8Group g[3];
  int count = 0;
  for (const auto& [w, y] : groups) {
    g[count++] = {w->data, y, w->rows};
  }
  dense::GemvQ8_0Quantized(g, count, quant, n, k, stream_);
}

void Executor::Kda(const DeviceLayer& l, std::span<const Lane> lanes,
                   std::uint32_t il, std::uint32_t n, bool fuse,
                   GpuTensorSink* sink) {
  Buffers& b = *b_;
  const std::uint32_t k = KdaIndex(il);
  // Rollback point t of a lane: checkpoint ckpt0 + t, this layer's slice.
  const auto conv_ckpt_at = [k](const Lane& lane) {
    return std::size_t{lane.ckpt0} * kKdaLayers * kKdaConvStateFloats +
           std::size_t{k} * kKdaConvStateFloats;
  };
  const auto state_ckpt_at = [k](const Lane& lane) {
    return std::size_t{lane.ckpt0} * kKdaLayers * kKdaStateFloats +
           std::size_t{k} * kKdaStateFloats;
  };
  if (fuse) {
    Q8Quantized({{&l.kda_proj, b.proj}}, b.xq, n, kHidden);
  } else {
    Q8(l.kda_proj, b.x, b.proj, n);
  }
  Mark("kda.proj_gemv", TensorBytes(l.kda_proj));
  // Conv window and recurrence: per lane, on that session's own state and
  // rows, with the launches a forward of the lane's tokens alone would use.
  for (const Lane& lane : lanes) {
    const std::uint32_t m = lane.n;
    float* conv_state =
        lane.session->state_.kda_conv + std::size_t{k} * kKdaConvStateFloats;
    const float* proj = b.proj + std::size_t{lane.row0} * kKdaProjRows;
    float* conv = b.conv + std::size_t{lane.row0} * kQkvStride;
    float* g = b.g + std::size_t{lane.row0} * kKdaInner;
    float* beta = b.beta + std::size_t{lane.row0} * kHeads;
    if (fuse) {
      kda::PrepDecode(
          proj, l.conv_q.f32(), l.conv_k.f32(), l.conv_v.f32(), conv_state,
          l.ssm_a.f32(), l.ssm_dt.f32(), conv, g, beta, m,
          lane.rollback && m > 1 ? b.conv_ckpt + conv_ckpt_at(lane) : nullptr,
          kKdaLayers * kKdaConvStateFloats, stream_);
      continue;
    }
    if (m == 1) {
      kda::Conv3Decode(proj, l.conv_q.f32(), l.conv_k.f32(), l.conv_v.f32(),
                       conv_state, conv, stream_);
    } else {
      if (lane.rollback) {
        kda::Conv3Checkpoint(proj, conv_state, b.conv_ckpt + conv_ckpt_at(lane),
                             kKdaLayers * kKdaConvStateFloats, m, stream_);
      }
      kda::Conv3Prefill(proj, l.conv_q.f32(), l.conv_k.f32(), l.conv_v.f32(),
                        conv_state, conv, m, stream_);
    }
    kda::GateBetaPrep(proj, l.ssm_a.f32(), l.ssm_dt.f32(), g, beta, m, stream_);
    kda::L2NormHeads(conv, kQkvStride, m, stream_);
    kda::L2NormHeads(conv + kKdaInner, kQkvStride, m, stream_);
  }
  Mark("kda.conv_gate_l2", 0);
  for (const Lane& lane : lanes) {
    const std::uint32_t m = lane.n;
    float* state =
        lane.session->state_.kda_state + std::size_t{k} * kKdaStateFloats;
    float* q = b.conv + std::size_t{lane.row0} * kQkvStride;
    float* kk = q + kKdaInner;
    float* v = q + 2 * kKdaInner;
    const float* g = b.g + std::size_t{lane.row0} * kKdaInner;
    const float* beta = b.beta + std::size_t{lane.row0} * kHeads;
    float* o = b.o + std::size_t{lane.row0} * kKdaInner;
    if (m == 1) {
      kda::RecurrentDecode(state, q, kk, v, g, beta, o, stream_);
    } else if (m <= kMaxDecodeTokens) {
      kda::RecurrentSequence(
          state, q, kQkvStride, kk, kQkvStride, v, kQkvStride, g, beta, o, m,
          stream_, lane.rollback ? b.kda_ckpt + state_ckpt_at(lane) : nullptr,
          kKdaLayers * kKdaStateFloats);
    } else {
      kda::PrefillSequence(state, q, kQkvStride, kk, kQkvStride, v, kQkvStride,
                           g, beta, o, m, stream_);
    }
  }
  Mark("kda.recurrent", lanes.size() * 2 * kKdaStateFloats * sizeof(float));
  DumpFloat(sink, "kda_o", static_cast<int>(il), b.o, n, kKdaInner);
  if (fuse) {
    kda::OutputNormGateQuant(b.o, b.proj, l.ssm_norm.f32(), b.kout, n, b.kq,
                             stream_);
    Mark("kda.outgate_quant", 0);
    Q8Quantized({{&l.attn_output, b.mixer}}, b.kq, n, kKdaInner);
  } else {
    kda::OutputNormGate(b.o, b.proj, l.ssm_norm.f32(), b.kout, n, stream_);
    Mark("kda.outgate_quant", 0);
    Q8(l.attn_output, b.kout, b.mixer, n);
  }
  Mark("kda.out_gemv", TensorBytes(l.attn_output));
}

void Executor::Mla(const DeviceLayer& l, std::span<const Lane> lanes,
                   std::uint32_t il, std::uint32_t n, bool fuse,
                   GpuTensorSink* sink) {
  Buffers& b = *b_;
  const bool mtp_block = il == kMtpBlock;
  const auto cache_of = [&](const Lane& lane) {
    const SessionState& st = lane.session->state_;
    return mtp_block ? static_cast<__half*>(st.mtp_cache)
                     : static_cast<__half*>(st.mla_cache) +
                           std::size_t{MlaIndex(il)} * st.cache.LayerStride();
  };
  if (fuse) {
    Q8Quantized({{&l.attn_q, b.q}, {&l.kv_a, b.kva}, {&l.attn_gate, b.gate}},
                b.xq, n, kHidden);
  } else {
    Q8(l.attn_q, b.x, b.q, n);
    Q8(l.kv_a, b.x, b.kva, n);
    Q8(l.attn_gate, b.x, b.gate, n);
  }
  Mark("mla.qkv_gemv",
       TensorBytes(l.attn_q) + TensorBytes(l.kv_a) + TensorBytes(l.attn_gate));
  const int ly = static_cast<int>(il);
  DumpFloat(sink, "mla_q", ly, b.q, n, kMlaQRows);
  DumpFloat(sink, "mla_kva", ly, b.kva, n, kMlaCacheWidth);
  mla::LatentNormRope(b.kva, l.kv_a_norm.f32(), b.positions, b.kv, n, stream_);
  std::size_t attended_rows = 0;
  for (const Lane& lane : lanes) {
    mla::CacheAppend(cache_of(lane),
                     b.kv + std::size_t{lane.row0} * kMlaCacheWidth,
                     b.positions + lane.row0, lane.n, stream_);
    // Rows from first_pos on are rewritten: their persistent expansion is
    // stale. The MTP block never runs the prefill family and has no store.
    if (!mtp_block) {
      std::uint32_t& expanded_to =
          lane.session->state_.mla_expanded_to[MlaIndex(il)];
      expanded_to = std::min(expanded_to, lane.first_pos);
    }
    attended_rows += lane.first_pos + lane.n;
  }
  Mark("mla.latent_rope_append", 0);
  DumpFloat(sink, "mla_kv", ly, b.kv, n, kMlaCacheWidth);
  if (n <= kMaxDecodeTokens) {
    // Small-batch family: every token runs the absorbed decode kernel on its
    // own session's context, so a chunk of n gives the same bits as n
    // single-token steps whatever the context length (split-K plans depend
    // only on the context), and so does a batch of several sessions' rows.
    mla::QAbsorb(b.q, l.k_b.data, b.positions, b.qabs, n, stream_);
    Mark("mla.qabsorb", TensorBytes(l.k_b));
    for (const Lane& lane : lanes) {
      const __half* cache = cache_of(lane);
      for (std::uint32_t t = lane.row0; t < lane.row0 + lane.n; ++t)
        mla::DecodeAttention(
            b.qabs + (std::size_t{t} * kHeads * kMlaCacheWidth), cache,
            b.positions + t, b.lat + (std::size_t{t} * kHeads * kMlaLatent),
            b.scratch, stream_);
    }
    Mark("mla.attention", attended_rows * kMlaCacheWidth * sizeof(__half));
    DumpFloat(sink, "mla_qabs", ly, b.qabs, n, kHeads * kMlaCacheWidth);
    DumpFloat(sink, "mla_lat", ly, b.lat, n, kHeads * kMlaLatent);
    mla::VbUpProjGate(b.lat, l.v_b.data, b.gate, b.kout, n, stream_);
    Mark("mla.vb_gate", TensorBytes(l.v_b));
  } else {
    // Prefill family: non-absorbed. Latent rows expand per head (bounded
    // 2048-key blocks) and the gated attention output is written directly.
    // Always used for n > 8, so the bits do not depend on the chunk layout.
    // One lane: only Forward runs this family.
    assert(lanes.size() == 1);
    const Lane& lane = lanes.front();
    const SessionState& st = lane.session->state_;
    const std::uint32_t first_pos = lane.first_pos;
    const __half* cache = cache_of(lane);
    std::uint32_t unused_watermark = 0;
    std::uint32_t& expanded_to =
        mtp_block ? unused_watermark : st.mla_expanded_to[MlaIndex(il)];
    mla::QPrepare(b.q, b.positions, b.qabs, n, stream_);
    Mark("mla.q_prep", 0);
    __half* expanded = nullptr;
    if (st.expanded_rows > 0) {
      expanded = static_cast<__half*>(st.mla_expanded) +
                 (std::size_t{MlaIndex(il)} *
                  (mla::ExpandedLayerBytes(st.expanded_rows) / sizeof(__half)));
      const std::uint32_t upto = std::min(first_pos + n, st.expanded_rows);
      if (expanded_to < upto) {
        mla::ExpandRows(cache, l.mla_w16.f16(), expanded, st.expanded_rows,
                        expanded_to, upto, stream_);
        expanded_to = upto;
      }
    }
    Mark("mla.expand_rows", 0);
    // With an op profile each launch inside the call is also timed (waited
    // for); the parts are reported under "mla.attention/..." and are not
    // added to the step total (they are inside "mla.attention").
    mla::PrefillTimes times;
    mla::PrefillAttentionNonAbsorbed(b.qabs, cache, l.mla_w16.f16(), expanded,
                                     st.expanded_rows, b.gate, first_pos, n,
                                     b.kout, b.scratch, stream_,
                                     profiling_ ? &times : nullptr);
    Mark("mla.attention",
         std::size_t{first_pos + n} * kMlaCacheWidth * sizeof(__half));
    if (profiling_) {
      detail_ops_.push_back(
          {"mla.attention/expand_transient", times.expand_ms});
      detail_ops_.push_back({"mla.attention/flash", times.attn_ms});
      detail_ops_.push_back({"mla.attention/merge", times.merge_ms});
    }
  }
  DumpFloat(sink, "mla_vb", ly, b.kout, n, kKdaInner);
  Q8(l.attn_output, b.kout, b.mixer, n);
  Mark("mla.out_gemv", TensorBytes(l.attn_output));
}

void Executor::Ffn(const DeviceLayer& l, std::uint32_t il, std::uint32_t n,
                   bool fuse, const DeviceLayer* next, GpuTensorSink* sink) {
  Buffers& b = *b_;
  const Config& c = model_.config();
  const float shexp_limit = c.swiglu_clamp_shexp[il];
  // With `fuse` the caller already produced x (and its planes) together with
  // the residual add.
  if (!fuse) {
    dense::RmsNormRows(b.h, l.ffn_norm.f32(), b.x, n, kHidden, 1, kRmsEps,
                       stream_);
  }
  if (!l.moe) {
    if (fuse) {
      Q8Quantized({{&l.ffn_gate, b.ga}, {&l.ffn_up, b.gb}}, b.xq, n, kHidden);
    } else {
      Q8(l.ffn_gate, b.x, b.ga, n);
      Q8(l.ffn_up, b.x, b.gb, n);
    }
    Mark("dense.gate_up_gemv", TensorBytes(l.ffn_gate) + TensorBytes(l.ffn_up));
    dense::SwiGluClamped(b.ga, b.gb, b.act, std::size_t{n} * kDenseFf,
                         shexp_limit, stream_);
    Mark("dense.swiglu", 0);
    DumpFloat(sink, "ffn_in", static_cast<int>(il), b.x, n, kHidden);
    DumpFloat(sink, "ffn_gate", static_cast<int>(il), b.ga, n, kDenseFf);
    DumpFloat(sink, "ffn_up", static_cast<int>(il), b.gb, n, kDenseFf);
    DumpFloat(sink, "ffn_act", static_cast<int>(il), b.act, n, kDenseFf);
    Q8(l.ffn_down, b.act, b.mixer, n);
    Mark("dense.down_gemv", TensorBytes(l.ffn_down));
    DumpFloat(sink, "ffn_out", static_cast<int>(il), b.mixer, n, kHidden);
    if (next != nullptr) {
      dense::AddResidualRmsNormQuant(b.h, b.mixer, next->attn_norm.f32(), b.x,
                                     n, kRmsEps, b.xq, stream_);
    } else {
      dense::AddResidual(b.h, b.mixer, std::size_t{n} * kHidden, stream_);
    }
    Mark("dense.combine_norm", 0);
    return;
  }
  // The shared expert is independent of routing: in the decode family it runs
  // as a parallel branch (fork after the norm, join before the combine), in a
  // captured graph and in eager steps (MTP verification, drafts) alike. It
  // reads x and its planes, writes ga/gb/act/aq/mixer and uses no b.scratch,
  // so results are the same as in the serial order. Profiled steps and debug
  // dumps stay serial so per-op times and dumps are attributable.
  const auto shexp_fused = [&]() {
    Q8Quantized({{&l.gate_shexp, b.ga}, {&l.up_shexp, b.gb}}, b.xq, n, kHidden);
    Mark("shexp.gate_up_gemv",
         TensorBytes(l.gate_shexp) + TensorBytes(l.up_shexp));
    dense::SwiGluClampedQuant(b.ga, b.gb, b.act, std::size_t{n} * kSharedFf,
                              kSharedFf, shexp_limit, b.aq, stream_);
    Mark("shexp.swiglu_quant", 0);
    Q8Quantized({{&l.down_shexp, b.mixer}}, b.aq, n, kSharedFf);
  };
  const bool branch = (capturing_ || (!profiling_ && sink == nullptr)) &&
                      fuse && n <= kMaxDecodeTokens;
  if (branch) {
    const hipStream_t main_stream = stream_;
    (void)hipEventRecord(fork_event_, main_stream);
    (void)hipStreamWaitEvent(branch_stream_, fork_event_, 0);
    stream_ = branch_stream_;
    shexp_fused();
    (void)hipEventRecord(join_event_, branch_stream_);
    stream_ = main_stream;
  }
  dense::GemvF32(l.router.f32(), b.x, b.rlogits, n, kHidden, kExperts, stream_);
  Mark("moe.router_gemv", TensorBytes(l.router));
  if (fuse) {
    // Routing also leaves the expert plan in b.scratch for the two expert
    // launches below (nothing else may write b.scratch until they ran).
    moe::RouteDecode(b.rlogits, l.exp_probs_b.f32(), b.ids, b.rweights, n,
                     b.scratch, b.route_counter, stream_);
  } else {
    moe::Route(b.rlogits, l.exp_probs_b.f32(), b.ids, b.rweights, n, stream_);
  }
  Mark("moe.route", 0);
  DumpInt(sink, "router_ids", static_cast<int>(il), b.ids, n, kExpertsUsed);
  DumpFloat(sink, "router_w", static_cast<int>(il), b.rweights, n,
            kExpertsUsed);
  const float exp_limit = c.swiglu_clamp_exp[il];
  if (n <= kMaxDecodeTokens) {
    if (fuse) {
      moe::ExpertGateUpDecodePlanned(l.gate_exps.data, l.up_exps.data,
                                     l.gate_exps.type, b.x, b.ids, n, exp_limit,
                                     b.moe_act, b.scratch, stream_);
    } else {
      moe::ExpertGateUpDecode(l.gate_exps.data, l.up_exps.data,
                              l.gate_exps.type, b.x, b.ids, n, exp_limit,
                              b.moe_act, b.scratch, stream_);
    }
    Mark("moe.expert_gate_up",
         kExpertsUsed * (TensorBytes(l.gate_exps) + TensorBytes(l.up_exps)) /
             l.gate_exps.depth);
    if (fuse) {
      // Same ids and scratch as the gate/up launch: its expert plan is reused.
      // Nothing may write b.scratch (attention partials, GEMV planes) between
      // the two launches; keep them adjacent.
      moe::ExpertDownDecodePlanned(l.down_exps.data, l.down_exps.type,
                                   b.moe_act, b.ids, n, b.exp_out, b.scratch,
                                   stream_);
    } else {
      moe::ExpertDownDecode(l.down_exps.data, l.down_exps.type, b.moe_act,
                            b.ids, n, b.exp_out, b.scratch, stream_);
    }
    Mark("moe.expert_down",
         kExpertsUsed * TensorBytes(l.down_exps) / l.down_exps.depth);
  } else {
    moe::ExpertGateUpPrefill(l.gate_exps.data, l.up_exps.data, l.gate_exps.type,
                             b.x, b.ids, n, exp_limit, b.moe_act, b.scratch,
                             stream_);
    moe::ExpertDownPrefill(l.down_exps.data, l.down_exps.type, b.moe_act, b.ids,
                           n, b.exp_out, b.scratch, stream_);
  }
  if (branch) {
    (void)hipStreamWaitEvent(stream_, join_event_, 0);
  } else if (fuse) {
    shexp_fused();
  } else {
    Q8(l.gate_shexp, b.x, b.ga, n);
    Q8(l.up_shexp, b.x, b.gb, n);
    dense::SwiGluClamped(b.ga, b.gb, b.act, std::size_t{n} * kSharedFf,
                         shexp_limit, stream_);
    Q8(l.down_shexp, b.act, b.mixer, n);
  }
  Mark("shexp.down_gemv", TensorBytes(l.down_shexp));
  if (sink != nullptr && (sink->Wants("ffn_routed", static_cast<int>(il)) ||
                          sink->Wants("ffn_out", static_cast<int>(il)))) {
    // Debug split of the FFN contribution (routed experts, shared expert,
    // total) so it can be compared without the residual stream's cancellation.
    const int layer = static_cast<int>(il);
    std::vector<float> eo(std::size_t{n} * kExpertsUsed * kHidden);
    std::vector<float> w(std::size_t{n} * kExpertsUsed);
    std::vector<float> shared(std::size_t{n} * kHidden);
    (void)hipMemcpyAsync(eo.data(), b.exp_out, eo.size() * 4,
                         hipMemcpyDeviceToHost, stream_);
    (void)hipMemcpyAsync(w.data(), b.rweights, w.size() * 4,
                         hipMemcpyDeviceToHost, stream_);
    (void)hipMemcpyAsync(shared.data(), b.mixer, shared.size() * 4,
                         hipMemcpyDeviceToHost, stream_);
    (void)hipStreamSynchronize(stream_);
    std::vector<float> routed(std::size_t{n} * kHidden, 0.0F);
    std::vector<float> total(std::size_t{n} * kHidden);
    for (std::uint32_t t = 0; t < n; ++t) {
      for (std::uint32_t k = 0; k < kExpertsUsed; ++k) {
        const float wk = w[std::size_t{t} * kExpertsUsed + k];
        const float* src =
            eo.data() + (std::size_t{t} * kExpertsUsed + k) * kHidden;
        for (std::uint32_t i = 0; i < kHidden; ++i) {
          routed[std::size_t{t} * kHidden + i] += wk * src[i];
        }
      }
    }
    for (std::size_t i = 0; i < total.size(); ++i) {
      total[i] = routed[i] + shared[i];
    }
    if (sink->Wants("ffn_routed", layer))
      sink->Float("ffn_routed", layer, kHidden, routed);
    if (sink->Wants("ffn_shared", layer))
      sink->Float("ffn_shared", layer, kHidden, shared);
    if (sink->Wants("ffn_out", layer))
      sink->Float("ffn_out", layer, kHidden, total);
  }
  if (next != nullptr) {
    dense::CombineRmsNormQuant(b.h, b.exp_out, b.rweights, b.mixer,
                               next->attn_norm.f32(), b.x, n, kRmsEps, b.xq,
                               stream_);
  } else {
    moe::Combine(b.h, b.exp_out, b.rweights, b.mixer, n, stream_);
  }
  Mark("moe.combine_norm", 0);
}

bool Executor::Block(std::span<const Lane> lanes, std::uint32_t il,
                     std::uint32_t n, GpuTensorSink* sink, bool fuse,
                     bool x_ready, const DeviceLayer* next,
                     std::string* error) {
  return BlockOf(model_.layers()[il - model_.first_layer()], lanes, il, n, sink,
                 fuse, x_ready, next, error);
}

bool Executor::BlockOf(const DeviceLayer& l, std::span<const Lane> lanes,
                       std::uint32_t il, std::uint32_t n, GpuTensorSink* sink,
                       bool fuse, bool x_ready, const DeviceLayer* next,
                       std::string* error) {
  Buffers& b = *b_;
  const int layer = static_cast<int>(il);
  if (!x_ready) {
    if (fuse) {
      dense::RmsNormQuantRows(b.h, l.attn_norm.f32(), b.x, n, kRmsEps, b.xq,
                              stream_);
    } else {
      dense::RmsNormRows(b.h, l.attn_norm.f32(), b.x, n, kHidden, 1, kRmsEps,
                         stream_);
    }
    Mark("norm.attn", 0);
  }
  if (l.kind == LayerKind::kKda) {
    Kda(l, lanes, il, n, fuse, sink);
  } else {
    Mla(l, lanes, il, n, fuse, sink);
  }
  DumpFloat(sink, "mixer_out", layer, b.mixer, n, kHidden);
  if (fuse) {
    dense::AddResidualRmsNormQuant(b.h, b.mixer, l.ffn_norm.f32(), b.x, n,
                                   kRmsEps, b.xq, stream_);
  } else {
    dense::AddResidual(b.h, b.mixer, std::size_t{n} * kHidden, stream_);
  }
  Mark("norm.resid_ffn", 0);
  DumpFloat(sink, "h_mid", layer, b.h, n, kHidden);
  Ffn(l, il, n, fuse, fuse ? next : nullptr, sink);
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

void Executor::HeadLaunch(std::uint32_t n, std::uint32_t from) {
  // At most logit_group rows: norm and logits GEMV.
  Buffers& b = *b_;
  const std::uint32_t rows = n - from;
  dense::RmsNormRows(b.h + std::size_t{from} * kHidden,
                     model_.output_norm().f32(), b.x, rows, kHidden, 1, kRmsEps,
                     stream_);
  dense::GemvF16(model_.output().f16(), b.x, b.logits, rows, kHidden,
                 kVocabRows, b.scratch, stream_);
}

bool Executor::Head(std::uint32_t n, std::uint32_t from,
                    std::span<float> logits, std::span<std::int32_t> argmax,
                    std::string* error) {
  Buffers& b = *b_;
  const std::uint32_t rows = n - from;
  dense::RmsNormRows(b.h + std::size_t{from} * kHidden,
                     model_.output_norm().f32(), b.x, rows, kHidden, 1, kRmsEps,
                     stream_);
  for (std::uint32_t g0 = 0; g0 < rows; g0 += options_.logit_group) {
    const std::uint32_t gn = std::min(options_.logit_group, rows - g0);
    dense::GemvF16(model_.output().f16(), b.x + std::size_t{g0} * kHidden,
                   b.logits, gn, kHidden, kVocabRows, b.scratch, stream_);
    Mark("head.logits_gemv", TensorBytes(model_.output()));
    if (!argmax.empty()) {
      // At most one group (checked by Forward): rows 0..gn of b.logits.
      mtp::ArgmaxRows(b.logits, gn, kVocabRows, b.argmax_idx, b.argmax_val,
                      stream_);
      if (!Check(hipMemcpyAsync(argmax.data() + g0, b.argmax_idx,
                                gn * sizeof(std::int32_t),
                                hipMemcpyDeviceToHost, stream_),
                 "copy argmax", error) ||
          !Check(hipStreamSynchronize(stream_), "argmax sync", error)) {
        return false;
      }
    }
    if (!logits.empty() &&
        (!Check(hipMemcpyAsync(logits.data() + std::size_t{g0} * kVocabRows,
                               b.logits, std::size_t{gn} * kVocabRows * 4,
                               hipMemcpyDeviceToHost, stream_),
                "copy logits", error) ||
         !Check(hipStreamSynchronize(stream_), "logits sync", error))) {
      return false;
    }
  }
  return true;
}

bool Executor::Trunk(std::span<const Lane> lanes, std::uint32_t n,
                     std::uint32_t head_from, bool events, GpuTensorSink* sink,
                     std::string* error) {
  Buffers& b = *b_;
  const std::uint32_t layers = layer_count();
  if (events) {
    (void)hipEventRecord(events_[0], stream_);
  }
  dense::EmbedQ4K(model_.token_embd().data, b.tokens, b.h, n, kHidden, stream_);
  Mark("embed", 0);
  DumpFloat(sink, "embd", 0, b.h, n, kHidden);
  // Decode family: fused launches (norm + planes + residual), same bits.
  const bool fuse = n <= kMaxDecodeTokens && dense::GemvQ8_0Fits(n, kHidden);
  bool ready = false;
  for (std::uint32_t il = 0; il < layers; ++il) {
    const DeviceLayer* next =
        il + 1 < layers ? &model_.layers()[il + 1] : nullptr;
    if (!Block(lanes, il, n, sink, fuse, ready, next, error)) {
      (void)hipStreamSynchronize(stream_);
      (void)hipStreamSynchronize(branch_stream_);
      return false;
    }
    ready = fuse && next != nullptr;
    if (events) {
      (void)hipEventRecord(events_[il + 1], stream_);
    }
  }
  if (head_from < n) {
    HeadLaunch(n, head_from);
  }
  return true;
}

bool Executor::FinishForward(Session& session, const ForwardRequest& r,
                             std::uint32_t n, std::uint32_t from,
                             std::uint32_t layers, bool graph,
                             std::string* error) {
  Buffers& b = *b_;
  const std::uint32_t first_pos = session.position();
  if (graph && from < n && !r.logits.empty() &&
      !Check(hipMemcpyAsync(r.logits.data(), b.logits,
                            std::size_t{n - from} * kVocabRows * 4,
                            hipMemcpyDeviceToHost, stream_),
             "copy logits", error)) {
    return false;
  }
  (void)hipEventRecord(events_[layers + 1], stream_);
  (void)hipEventRecord(session.forward_done_, stream_);
  if (!Check(hipStreamSynchronize(stream_), "forward", error) ||
      !Check(hipGetLastError(), "forward", error)) {
    return false;
  }
  timing_.layer_ms.assign(layers, 0.0F);
  float total = 0.0F;
  float head = 0.0F;
  if (!graph) {
    for (std::uint32_t il = 0; il < layers; ++il) {
      // Dumps synchronise mid-stream; events still bracket each block.
      (void)hipEventElapsedTime(&timing_.layer_ms[il], events_[il],
                                events_[il + 1]);
    }
    (void)hipEventElapsedTime(&head, events_[layers], events_[layers + 1]);
  }
  (void)hipEventElapsedTime(&total, events_[0], events_[layers + 1]);
  timing_.total_ms = total;
  timing_.head_ms = head;
  launch_.wall_ms = StepMs();
  launch_.device_ms = total;
  launch_.graph = graph;
  session.set_position(first_pos + n);
  return true;
}

bool Executor::ForwardGraph(Session& session, const ForwardRequest& r,
                            std::uint32_t n, std::uint32_t first_pos,
                            std::uint32_t from, bool* launched,
                            std::string* error) {
  // Caller guarantees: 1 <= n <= kMaxDecodeTokens, no sink, rows fit one
  // logits group. On false with *launched unset only a capture failed: it
  // recorded launches without running them (stray branch launches of a
  // broken capture are awaited below), so the session state is untouched
  // and the caller falls back to eager.
  const std::uint32_t key = (n << 4U) | from;
  const std::uint32_t layers = layer_count();
  hipGraphExec_t exec = nullptr;
  if (const auto it = session.graphs_.find(key); it != session.graphs_.end()) {
    exec = it->second;
  } else {
    hipGraph_t captured = nullptr;
    // No other thread may wait on an event of this stream while it captures
    // (capture_mutex_, executor.hpp): snapshot workers run beside decode.
    std::unique_lock<std::mutex> capture_lock(capture_mutex_);
    if (!Check(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
               "graph capture", error)) {
      return false;
    }
    if (capture_breaks_for_test_ > 0) {
      // Test hook: the wait a snapshot worker used to issue here, from
      // another thread and without the mutex, which invalidates the capture.
      --capture_breaks_for_test_;
      std::thread([&session] {
        (void)hipEventSynchronize(session.forward_done_);
        (void)hipGetLastError();
      }).join();
    }
    capturing_ = true;
    const Lane lane{&session, first_pos, n, 0};
    const bool ok = Trunk({&lane, 1}, n, from, false, nullptr, error);
    capturing_ = false;
    const hipError_t end = hipStreamEndCapture(stream_, &captured);
    capture_invalidated_ = end == hipErrorStreamCaptureInvalidated;
    if (!ok || end != hipSuccess || captured == nullptr) {
      if (captured != nullptr) {
        (void)hipGraphDestroy(captured);
      }
      if (ok && error != nullptr) {
        *error = std::string("graph capture end: ") + hipGetErrorString(end);
      }
      // ROCm leaves a stream whose capture was invalidated in capture mode
      // after hipStreamEndCapture has reported it: every later launch or copy
      // on it fails ("operation failed due to a previous error during
      // capture"), so the eager fallback and every later step of every
      // session would fail. An empty capture resets the stream; it runs under
      // the capture mutex so no worker wait can break it in turn
      // (tools/models/ling_3_flash/capture_probe.hip, mode event-once).
      hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
      if (hipStreamIsCapturing(stream_, &status) == hipSuccess &&
          status != hipStreamCaptureStatusNone) {
        hipGraph_t empty = nullptr;
        if (hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal) ==
            hipSuccess) {
          (void)hipStreamEndCapture(stream_, &empty);
        }
        if (empty != nullptr) {
          (void)hipGraphDestroy(empty);
        }
      }
      capture_lock.unlock();
      // A broken capture can leave launches of its shared-expert branch
      // executing outside the graph; they write activation scratch, so they
      // must finish before the eager fallback reuses it.
      (void)hipStreamSynchronize(branch_stream_);
      (void)hipGetLastError();
      return false;
    }
    capture_lock.unlock();
    const hipError_t inst =
        hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0);
    (void)hipGraphDestroy(captured);
    if (inst != hipSuccess || exec == nullptr) {
      if (error != nullptr) {
        *error = std::string("graph instantiate: ") + hipGetErrorString(inst);
      }
      return false;
    }
    session.graphs_.emplace(key, exec);
  }
  // From here on a failure is a device error, not a capture problem: report
  // it as a failed step (no eager retry that could mask a lost device, and
  // graphs stay enabled).
  *launched = true;
  // Step inputs: staged in pinned memory, copied in stream order (the
  // staging area is not reused before this forward's final synchronisation).
  for (std::uint32_t t = 0; t < n; ++t) {
    staging_[t] = r.tokens.empty() ? 0 : r.tokens[t];
    staging_[8 + t] = static_cast<std::int32_t>(first_pos + t);
  }
  Buffers& b = *b_;
  if (!Check(hipMemcpyAsync(b.tokens, staging_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "upload tokens", error) ||
      !Check(hipMemcpyAsync(b.positions, staging_ + 8, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "upload positions", error)) {
    return false;
  }
  (void)hipEventRecord(events_[0], stream_);
  if (!Check(hipGraphLaunch(exec, stream_), "graph launch", error)) {
    return false;
  }
  launch_.issue_ms = StepMs();
  ++graph_replays_;
  return MtpAfterForward(session, n, first_pos, 0, 0, error) &&
         FinishForward(session, r, n, from, layers, true, error);
}

bool Executor::Forward(Session& session, const ForwardRequest& r,
                       std::string* error) {
  step_start_ = std::chrono::steady_clock::now();
  launch_ = {};
  const auto n = static_cast<std::uint32_t>(r.tokens.size());
  const std::uint32_t first_pos = session.position();
  // Rows below a held snapshot copy are never rewritten while it runs; a
  // forward at or past them (every decode and prefill step) does not wait.
  session.AwaitRowHolds(first_pos);
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
  if (!r.logits.empty() || !r.argmax.empty()) {
    from = r.logits_from == UINT32_MAX ? n - 1 : r.logits_from;
    if (from >= n || (!r.logits.empty() &&
                      r.logits.size() < std::size_t{n - from} * kVocabRows)) {
      return fail("logits buffer too small for rows " + std::to_string(from) +
                  ".." + std::to_string(n));
    }
    if (!r.argmax.empty() &&
        (r.argmax.size() < n - from || n - from > options_.logit_group)) {
      return fail("argmax buffer too small for rows " + std::to_string(from) +
                  ".." + std::to_string(n));
    }
  }
  if (r.rollback &&
      (!options_.mtp || n < 2 || n > kMaxDecodeTokens || r.sink != nullptr)) {
    return fail("rollback points need MTP and a verify chunk of 2.." +
                std::to_string(kMaxDecodeTokens) + " tokens");
  }
  for (const std::int32_t t : r.tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= kVocab) {
      return fail("token id " + std::to_string(t) + " outside the vocabulary");
    }
  }
  const std::uint32_t layers = layer_count();
  // Small-batch steps replay a per-session graph from their second occurrence
  // (the first runs eagerly so lazy module loads happen before capture).
  if (graphs_enabled_ && !r.block_timing && r.op_profile == nullptr &&
      !r.rollback && r.argmax.empty() && n <= kMaxDecodeTokens &&
      r.sink == nullptr && n - from <= options_.logit_group) {
    const std::uint32_t key = (n << 4U) | from;
    if (session.warmed_.contains(key)) {
      std::string graph_error;
      bool launched = false;
      if (ForwardGraph(session, r, n, first_pos, from, &launched,
                       &graph_error)) {
        return true;
      }
      if (launched) {
        // Inputs were being uploaded or the graph ran: a device failure.
        return fail(graph_error);
      }
      // Capture is not executed, so the eager path below starts from the
      // untouched state. A capture another thread's runtime call invalidated
      // is transient: stay eager for this step and retry on the next one; any
      // other failure turns graphs off for good.
      (void)hipGetLastError();
      if (!capture_invalidated_) {
        std::fprintf(stderr, "[ling] HIP graph decode disabled: %s\n",
                     graph_error.c_str());
        graphs_enabled_ = false;
      } else if (const std::uint64_t n = ++capture_invalidations_;
                 (n & (n - 1)) == 0) {
        // Not silent: a capture that keeps breaking leaves the session eager
        // (and re-capturing) on every step.
        std::fprintf(stderr,
                     "[ling] HIP graph capture invalidated (%llu so far), "
                     "step ran eagerly: %s\n",
                     static_cast<unsigned long long>(n), graph_error.c_str());
      }
      capture_invalidated_ = false;
    } else {
      session.warmed_.insert(key);
    }
  }
  if (!UploadInputs(r.tokens, n, first_pos, error)) {
    return false;
  }
  if (r.op_profile != nullptr) {
    BeginProfile();
  }
  const Lane lane{.session = &session,
                  .first_pos = first_pos,
                  .n = n,
                  .row0 = 0,
                  .rollback = r.rollback};
  const bool trunk_ok =
      Trunk({&lane, 1}, n, /*head_from=*/n, true, r.sink, error);
  launch_.issue_ms = StepMs();
  if (!trunk_ok) {
    profiling_ = false;
    return false;
  }
  if (from < n && !Head(n, from, r.logits, r.argmax, error)) {
    return false;
  }
  if (!MtpAfterForward(session, n, first_pos, 0, 0, error)) {
    return false;
  }
  if (!FinishForward(session, r, n, from, layers, false, error)) {
    profiling_ = false;
    return false;
  }
  if (r.op_profile != nullptr) {
    FinishProfile(r.op_profile);
  }
  return true;
}

bool Executor::ForwardBatch(std::span<BatchSegment> segments,
                            std::string* error) {
  using Status = BatchSegment::Status;
  step_start_ = std::chrono::steady_clock::now();
  launch_ = {};
  const std::uint32_t row_limit = batch_row_limit();
  std::vector<Lane> lanes;
  std::vector<BatchSegment*> members;  // accepted, in row order
  std::vector<std::int32_t> tokens;
  std::vector<std::int32_t> positions;
  std::vector<std::uint32_t> froms;  // first logits row per member
  std::vector<std::uint32_t> hn0s;   // MTP hidden rows per member
  lanes.reserve(segments.size());
  members.reserve(segments.size());
  std::uint32_t rows = 0;
  std::uint32_t ckpts = 0;  // rollback points taken so far
  std::uint32_t head_from = UINT32_MAX;
  // Validation only reads host state: a refused segment's session is never
  // touched, and nothing is launched before every segment was checked.
  for (std::size_t i = 0; i < segments.size(); ++i) {
    BatchSegment& s = segments[i];
    s.status = Status::kRejected;
    s.error.clear();
    const auto n = static_cast<std::uint32_t>(s.tokens.size());
    if (first_layer() != 0) {
      s.error = "forward needs the trunk from block 0 (layer_first is set)";
      continue;
    }
    if (s.session == nullptr) {
      s.error = "batch segment has no session";
      continue;
    }
    s.session->AwaitRowHolds(s.session->position());
    if (std::any_of(
            segments.begin(), segments.begin() + i,
            [&](const BatchSegment& o) { return o.session == s.session; })) {
      s.error = "session appears twice in one batched forward";
      continue;
    }
    // The pool finishes a member's snapshot before batching it; a transfer
    // in flight would copy state this pass rewrites. Refused, not asserted:
    // the caller runs such a lane alone, as it did before batching.
    if (s.session->transfers_.load(std::memory_order_acquire) != 0) {
      s.error = "session has a snapshot transfer in flight";
      continue;
    }
    if (n == 0 || n > kMaxDecodeTokens) {
      s.error = "batch segment of " + std::to_string(n) +
                " tokens outside 1.." + std::to_string(kMaxDecodeTokens);
      continue;
    }
    if (rows + n > row_limit) {
      s.error =
          "batched forward exceeds " + std::to_string(row_limit) + " rows";
      continue;
    }
    const std::uint32_t first_pos = s.session->position();
    if (first_pos + n > s.session->capacity()) {
      s.error = "context overflow: " + std::to_string(first_pos) + " + " +
                std::to_string(n) + " > " +
                std::to_string(s.session->capacity());
      continue;
    }
    if (const auto bad = std::find_if(
            s.tokens.begin(), s.tokens.end(),
            [](std::int32_t t) {
              return t < 0 || static_cast<std::uint32_t>(t) >= kVocab;
            });
        bad != s.tokens.end()) {
      s.error = "token id " + std::to_string(*bad) + " outside the vocabulary";
      continue;
    }
    const std::uint32_t from =
        s.logits_from == UINT32_MAX ? n - 1 : s.logits_from;
    if (!s.logits.empty() &&
        (from >= n || s.logits.size() < std::size_t{n - from} * kVocabRows)) {
      s.error = "logits buffer too small for rows " + std::to_string(from) +
                ".." + std::to_string(n);
      continue;
    }
    if (!s.argmax.empty() && (from >= n || s.argmax.size() < n - from)) {
      s.error = "argmax buffer too small for rows " + std::to_string(from) +
                ".." + std::to_string(n);
      continue;
    }
    // Rollback points of the segments follow each other in the checkpoint
    // scratch (kMaxDrafts points, enough for any 8-row step).
    if (s.rollback && (!options_.mtp || n < 2 || ckpts + n - 1 > kMaxDrafts)) {
      s.error = "rollback points need MTP, 2..8 tokens and free checkpoints";
      continue;
    }
    // MTP hidden rows: the carry and n rows per segment (MtpRollback).
    const std::uint32_t hn0 = rows + static_cast<std::uint32_t>(lanes.size());
    if (options_.mtp && hn0 + n + 1 > options_.max_batch + 1) {
      s.error = "batched forward exceeds the MTP hidden rows";
      continue;
    }
    s.status = Status::kPending;
    lanes.push_back({.session = s.session,
                     .first_pos = first_pos,
                     .n = n,
                     .row0 = rows,
                     .rollback = s.rollback,
                     .ckpt0 = ckpts});
    if (s.rollback) {
      ckpts += n - 1;
    }
    members.push_back(&s);
    froms.push_back(from);
    hn0s.push_back(hn0);
    for (std::uint32_t t = 0; t < n; ++t) {
      tokens.push_back(s.tokens[t]);
      positions.push_back(static_cast<std::int32_t>(first_pos + t));
    }
    if (!s.logits.empty() || !s.argmax.empty()) {
      head_from = std::min(head_from, rows + from);
    }
    rows += n;
  }
  if (lanes.empty()) {
    return true;
  }
  head_from = std::min(head_from, rows);  // rows: no segment wants logits
  bool argmax = false;
  for (std::size_t i = 0; i < members.size(); ++i) {
    BatchSegment& s = *members[i];
    argmax = argmax || !s.argmax.empty();
    s.placement = {.ckpt0 = lanes[i].ckpt0,
                   .hn0 = hn0s[i],
                   .logit_row = s.logits.empty() && s.argmax.empty()
                                    ? 0
                                    : lanes[i].row0 + froms[i] - head_from};
  }
  std::string message;
  bool launched = false;  // the trunk is queued: members' state is in flight
  const auto fail = [&]() {
    for (BatchSegment* s : members) {
      s->status = Status::kFailed;
      s->error = message;
      // Later snapshot copies into a member wait for whatever was queued.
      if (launched) {
        (void)hipEventRecord(s->session->forward_done_, stream_);
      }
    }
    if (launched) {
      // As Trunk() on failure: nothing queued (including the eager
      // shared-expert branch) may still run when the caller reuses scratch.
      (void)hipStreamSynchronize(stream_);
      (void)hipStreamSynchronize(branch_stream_);
    }
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  const std::uint32_t layers = layer_count();
  if (!UploadRows(tokens, positions, &message) ||
      !Trunk(lanes, rows, head_from, true, nullptr, &message)) {
    return fail();
  }
  launched = true;
  if (batch_failures_for_test_ > 0 &&
      batch_failure_point_ == BatchFailurePoint::kBeforeSync) {
    --batch_failures_for_test_;
    message = "batched forward: injected failure before the wait (test hook)";
    return fail();
  }
  launch_.issue_ms = StepMs();
  Buffers& b = *b_;
  if (argmax) {
    // Row-wise, as the single-session head's argmax of its rows.
    mtp::ArgmaxRows(b.logits, rows - head_from, kVocabRows, b.argmax_idx,
                    b.argmax_val, stream_);
  }
  for (std::size_t i = 0; i < members.size(); ++i) {
    BatchSegment& s = *members[i];
    if (!s.argmax.empty() &&
        !Check(hipMemcpyAsync(s.argmax.data(),
                              b.argmax_idx + s.placement.logit_row,
                              (lanes[i].n - froms[i]) * sizeof(std::int32_t),
                              hipMemcpyDeviceToHost, stream_),
               "copy argmax", &message)) {
      return fail();
    }
    if (s.logits.empty()) {
      continue;
    }
    const Lane& lane = lanes[i];
    const std::uint32_t first_row = lane.row0 + froms[i] - head_from;
    if (!Check(hipMemcpyAsync(s.logits.data(),
                              b.logits + std::size_t{first_row} * kVocabRows,
                              std::size_t{lane.n - froms[i]} * kVocabRows * 4,
                              hipMemcpyDeviceToHost, stream_),
               "copy logits", &message)) {
      return fail();
    }
  }
  // Ascending rows (MtpAfterForward overwrites the rows below row0).
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    if (!MtpAfterForward(*members[i]->session, lanes[i].n, lanes[i].first_pos,
                         lanes[i].row0, hn0s[i], &message)) {
      return fail();
    }
  }
  (void)hipEventRecord(events_[layers + 1], stream_);
  // Every member's snapshot copies wait for this pass.
  for (BatchSegment* s : members) {
    (void)hipEventRecord(s->session->forward_done_, stream_);
  }
  if (!Check(hipStreamSynchronize(stream_), "batched forward", &message) ||
      !Check(hipGetLastError(), "batched forward", &message)) {
    return fail();
  }
  if (batch_failures_for_test_ > 0) {
    --batch_failures_for_test_;
    message = "batched forward: injected failure after the wait (test hook)";
    return fail();
  }
  timing_.layer_ms.assign(layers, 0.0F);
  for (std::uint32_t il = 0; il < layers; ++il) {
    (void)hipEventElapsedTime(&timing_.layer_ms[il], events_[il],
                              events_[il + 1]);
  }
  float total = 0.0F;
  float head = 0.0F;
  (void)hipEventElapsedTime(&head, events_[layers], events_[layers + 1]);
  (void)hipEventElapsedTime(&total, events_[0], events_[layers + 1]);
  timing_.total_ms = total;
  timing_.head_ms = head;
  launch_.wall_ms = StepMs();
  launch_.device_ms = total;
  launch_.graph = false;
  for (std::size_t i = 0; i < members.size(); ++i) {
    members[i]->session->set_position(lanes[i].first_pos + lanes[i].n);
    members[i]->status = Status::kDone;
  }
  return true;
}

bool Executor::MtpAfterForward(Session& session, std::uint32_t n,
                               std::uint32_t first_pos, std::uint32_t row0,
                               std::uint32_t hn0, std::string* error) {
  SessionState& st = session.state_;
  if (st.mtp_cache == nullptr) {
    return true;
  }
  if (first_pos > 0 && !session.mtp_valid_) {
    return true;  // state of the earlier positions is unknown: stay invalid
  }
  const DeviceMtp& m = *model_.mtp();
  Buffers& b = *b_;
  float* const hn = b.hn + std::size_t{hn0} * kHidden;
  // hn[0] = h of position first_pos - 1 (the carry), hn[1 + t] = the
  // output-normed trunk row row0 + t. Row r of the fill pairs hn[skip + r]
  // with the token of trunk row row0 + skip + r: MTP cache row q = (h_q,
  // token_{q + 1}).
  if (first_pos > 0) {
    (void)hipMemcpyAsync(hn, st.mtp_h, kHidden * sizeof(float),
                         hipMemcpyDeviceToDevice, stream_);
  }
  dense::RmsNormRows(b.h + std::size_t{row0} * kHidden,
                     model_.output_norm().f32(), hn + kHidden, n, kHidden, 1,
                     kRmsEps, stream_);
  Mark("mtp.hidden_norm", 0);
  const std::uint32_t skip = first_pos == 0 ? 1U : 0U;
  const std::uint32_t rows = n - skip;
  if (rows > 0) {
    dense::EmbedQ4K(model_.token_embd().data, b.tokens + row0 + skip, b.ga,
                    rows, kHidden, stream_);
    mtp::ConcatNormRows(b.ga, hn + std::size_t{skip} * kHidden, m.enorm.f32(),
                        m.hnorm.f32(), b.gb, rows, kRmsEps, stream_);
    Q8(m.eh_proj, b.gb, b.h, rows);
    dense::RmsNormRows(b.h, m.block.attn_norm.f32(), b.x, rows, kHidden, 1,
                       kRmsEps, stream_);
    Q8(m.block.kv_a, b.x, b.kva, rows);
    mtp::FillPositions(b.positions,
                       static_cast<std::int32_t>(first_pos + skip) - 1, rows,
                       stream_);
    mla::LatentNormRope(b.kva, m.block.kv_a_norm.f32(), b.positions, b.kv, rows,
                        stream_);
    mla::CacheAppend(static_cast<__half*>(st.mtp_cache), b.kv, b.positions,
                     rows, stream_);
    Mark("mtp.cache_fill", 0);
  }
  (void)hipMemcpyAsync(st.mtp_h, hn + std::size_t{n} * kHidden,
                       kHidden * sizeof(float), hipMemcpyDeviceToDevice,
                       stream_);
  session.mtp_valid_ = true;
  return Check(hipGetLastError(), "MTP cache fill", error);
}

bool Executor::MtpDraft(Session& session, const DraftRequest& r,
                        std::int32_t* argmax, std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  SessionState& st = session.state_;
  const bool host_h = !r.h_host.empty();
  if (!options_.mtp || st.mtp_cache == nullptr ||
      (!host_h && !session.mtp_valid_) ||
      (host_h && (r.h_host.size() != kHidden ||
                  (!r.h_out.empty() && r.h_out.size() != kHidden)))) {
    return fail("MTP draft needs an MTP session with a valid cache");
  }
  if (r.token < 0 || static_cast<std::uint32_t>(r.token) >= kVocab ||
      r.position >= session.capacity() ||
      (!r.logits.empty() && r.logits.size() < kVocabRows)) {
    return fail("MTP draft request is out of range");
  }
  session.AwaitRowHolds(r.position + 1);  // MTP row `position`
  const DeviceMtp& m = *model_.mtp();
  Buffers& b = *b_;
  step_start_ = std::chrono::steady_clock::now();
  launch_ = {};
  staging_[0] = r.token;
  staging_[8] = static_cast<std::int32_t>(r.position);
  if (!Check(hipMemcpyAsync(b.tokens, staging_, sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "upload draft token", error) ||
      !Check(hipMemcpyAsync(b.positions, staging_ + 8, sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "upload draft position", error)) {
    return false;
  }
  if (host_h &&
      !Check(hipMemcpyAsync(b.mtp_hn, r.h_host.data(), kHidden * sizeof(float),
                            hipMemcpyHostToDevice, stream_),
             "upload draft hidden", error)) {
    return false;
  }
  (void)hipEventRecord(draft_events_[0], stream_);
  if (r.op_profile != nullptr) {
    BeginProfile();
  }
  dense::EmbedQ4K(model_.token_embd().data, b.tokens, b.ga, 1, kHidden,
                  stream_);
  const float* h_in = r.chain || host_h ? b.mtp_hn : st.mtp_h;
  mtp::ConcatNormRows(b.ga, h_in, m.enorm.f32(), m.hnorm.f32(), b.gb, 1,
                      kRmsEps, stream_);
  Q8(m.eh_proj, b.gb, b.h, 1);
  Mark("draft.embed_eh_proj", TensorBytes(m.eh_proj));
  const bool fuse = dense::GemvQ8_0Fits(1, kHidden);
  const Lane lane{&session, r.position, 1, 0};
  if (!BlockOf(m.block, {&lane, 1}, kMtpBlock, 1, nullptr, fuse, false, nullptr,
               error)) {
    profiling_ = false;
    (void)hipStreamSynchronize(stream_);
    return false;
  }
  dense::RmsNormRows(b.h, m.layer_output_norm.f32(), b.mtp_hn, 1, kHidden, 1,
                     kRmsEps, stream_);
  // The draft scores with its own Q8_0 copy of the head (acceptance only: the
  // verify step and the sampled p use the F16 head); logits, argmax and the
  // proposal q all come from this one row.
  if (draft_head_q8_) {
    dense::GemvQ8_0(m.draft_head.data, b.mtp_hn, b.logits, 1, kHidden,
                    kVocabRows, b.scratch, stream_);
    Mark("draft.head_logits_gemv", TensorBytes(m.draft_head));
  } else {
    dense::GemvF16(model_.output().f16(), b.mtp_hn, b.logits, 1, kHidden,
                   kVocabRows, b.scratch, stream_);
    Mark("draft.head_logits_gemv", TensorBytes(model_.output()));
  }
  mtp::ArgmaxRows(b.logits, 1, kVocabRows, b.argmax_idx, b.argmax_val, stream_);
  Mark("draft.argmax", 0);
  (void)hipEventRecord(draft_events_[1], stream_);
  launch_.issue_ms = StepMs();
  std::int32_t index = -1;
  if (!Check(hipMemcpyAsync(&index, b.argmax_idx, sizeof(index),
                            hipMemcpyDeviceToHost, stream_),
             "copy draft argmax", error) ||
      (!r.logits.empty() &&
       !Check(hipMemcpyAsync(r.logits.data(), b.logits,
                             std::size_t{kVocabRows} * sizeof(float),
                             hipMemcpyDeviceToHost, stream_),
              "copy draft logits", error)) ||
      (!r.h_out.empty() &&
       !Check(hipMemcpyAsync(r.h_out.data(), b.mtp_hn, kHidden * sizeof(float),
                             hipMemcpyDeviceToHost, stream_),
              "copy draft hidden", error)) ||
      !Check(hipStreamSynchronize(stream_), "draft", error) ||
      !Check(hipGetLastError(), "draft", error)) {
    profiling_ = false;
    return false;
  }
  launch_.wall_ms = StepMs();
  {
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, draft_events_[0], draft_events_[1]);
    launch_.device_ms = ms;
  }
  if (r.op_profile != nullptr) {
    FinishProfile(r.op_profile);
  }
  if (index < 0) {
    return fail("MTP draft logits are not comparable");
  }
  if (argmax != nullptr) {
    *argmax = index;
  }
  return true;
}

bool Executor::MtpDraftBatch(std::span<DraftLane> lanes, std::string* error) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  const auto count = static_cast<std::uint32_t>(lanes.size());
  if (!options_.mtp || count == 0 || count > batch_row_limit()) {
    return fail("MTP draft batch needs an MTP executor and 1..8 lanes");
  }
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    const DraftLane& d = lanes[i];
    if (d.session == nullptr || d.session->state_.mtp_cache == nullptr ||
        !d.session->mtp_valid_ ||
        std::any_of(lanes.begin(), lanes.begin() + i, [&](const DraftLane& o) {
          return o.session == d.session;
        })) {
      return fail("MTP draft batch needs distinct sessions with valid caches");
    }
    if (d.token < 0 || static_cast<std::uint32_t>(d.token) >= kVocab ||
        d.position >= d.session->capacity() ||
        (!d.logits.empty() && d.logits.size() < kVocabRows)) {
      return fail("MTP draft batch request is out of range");
    }
    d.session->AwaitRowHolds(d.position + 1);  // MTP row `position`
  }
  const DeviceMtp& m = *model_.mtp();
  Buffers& b = *b_;
  step_start_ = std::chrono::steady_clock::now();
  launch_ = {};
  std::vector<std::int32_t> tokens(count);
  std::vector<std::int32_t> positions(count);
  std::vector<Lane> block_lanes(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    tokens[i] = lanes[i].token;
    positions[i] = static_cast<std::int32_t>(lanes[i].position);
    block_lanes[i] = {.session = lanes[i].session,
                      .first_pos = lanes[i].position,
                      .n = 1,
                      .row0 = i};
  }
  std::string message;
  if (!UploadRows(tokens, positions, &message)) {
    return fail(message);
  }
  (void)hipEventRecord(draft_events_[0], stream_);
  // The launches of MtpDraft over `count` rows: row-wise kernels and GEMVs
  // whose rows do not depend on each other; the hidden input per lane.
  dense::EmbedQ4K(model_.token_embd().data, b.tokens, b.ga, count, kHidden,
                  stream_);
  for (std::uint32_t i = 0; i < count; ++i) {
    const float* h_in = lanes[i].chain ? b.mtp_hn + std::size_t{i} * kHidden
                                       : lanes[i].session->state_.mtp_h;
    mtp::ConcatNormRows(b.ga + std::size_t{i} * kHidden, h_in, m.enorm.f32(),
                        m.hnorm.f32(), b.gb + std::size_t{i} * 2 * kHidden, 1,
                        kRmsEps, stream_);
  }
  Q8(m.eh_proj, b.gb, b.h, count);
  const bool fuse = dense::GemvQ8_0Fits(count, kHidden);
  if (!BlockOf(m.block, block_lanes, kMtpBlock, count, nullptr, fuse, false,
               nullptr, &message)) {
    (void)hipStreamSynchronize(stream_);
    (void)hipStreamSynchronize(branch_stream_);
    return fail(message);
  }
  dense::RmsNormRows(b.h, m.layer_output_norm.f32(), b.mtp_hn, count, kHidden,
                     1, kRmsEps, stream_);
  if (draft_head_q8_) {
    dense::GemvQ8_0(m.draft_head.data, b.mtp_hn, b.logits, count, kHidden,
                    kVocabRows, b.scratch, stream_);
  } else {
    dense::GemvF16(model_.output().f16(), b.mtp_hn, b.logits, count, kHidden,
                   kVocabRows, b.scratch, stream_);
  }
  mtp::ArgmaxRows(b.logits, count, kVocabRows, b.argmax_idx, b.argmax_val,
                  stream_);
  (void)hipEventRecord(draft_events_[1], stream_);
  launch_.issue_ms = StepMs();
  std::vector<std::int32_t> index(count, -1);
  bool ok = Check(
      hipMemcpyAsync(index.data(), b.argmax_idx, count * sizeof(std::int32_t),
                     hipMemcpyDeviceToHost, stream_),
      "copy draft argmax", &message);
  for (std::uint32_t i = 0; ok && i < count; ++i) {
    ok = lanes[i].logits.empty() ||
         Check(hipMemcpyAsync(lanes[i].logits.data(),
                              b.logits + std::size_t{i} * kVocabRows,
                              std::size_t{kVocabRows} * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "copy draft logits", &message);
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    (void)hipEventRecord(lanes[i].session->forward_done_, stream_);
  }
  if (!ok || !Check(hipStreamSynchronize(stream_), "draft batch", &message) ||
      !Check(hipGetLastError(), "draft batch", &message)) {
    (void)hipStreamSynchronize(stream_);
    (void)hipStreamSynchronize(branch_stream_);
    return fail(message);
  }
  if (draft_failure_in_ > 0 && --draft_failure_in_ == 0) {
    return fail("MTP draft batch: injected failure after launch (test hook)");
  }
  launch_.wall_ms = StepMs();
  {
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, draft_events_[0], draft_events_[1]);
    launch_.device_ms = ms;
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    if (index[i] < 0) {
      return fail("MTP draft logits are not comparable");
    }
    lanes[i].argmax = index[i];
  }
  return true;
}

bool Executor::MtpRollback(Session& session, std::uint32_t first_pos,
                           std::uint32_t n, std::uint32_t keep,
                           std::string* error, BatchSegment::Placement at) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (!options_.mtp || keep == 0 || keep > n || n > kMaxDecodeTokens ||
      at.ckpt0 + n - 1 > kMaxDrafts || session.position() != first_pos + n) {
    return fail("rollback does not match the last verify step");
  }
  if (keep == n) {
    return true;
  }
  Buffers& b = *b_;
  SessionState& st = session.state_;
  const std::size_t state_floats = kKdaLayers * kKdaStateFloats;
  const std::size_t conv_floats = kKdaLayers * kKdaConvStateFloats;
  const std::size_t point = std::size_t{at.ckpt0} + keep - 1;
  kda::CopyFloats(b.kda_ckpt + point * state_floats, st.kda_state, state_floats,
                  stream_);
  kda::CopyFloats(b.conv_ckpt + point * conv_floats, st.kda_conv, conv_floats,
                  stream_);
  if (st.mtp_cache != nullptr) {
    (void)hipMemcpyAsync(
        st.mtp_h, b.hn + (std::size_t{at.hn0} + keep) * kHidden,
        kHidden * sizeof(float), hipMemcpyDeviceToDevice, stream_);
  }
  (void)hipEventRecord(session.forward_done_, stream_);
  if (!Check(hipStreamSynchronize(stream_), "rollback", error) ||
      !Check(hipGetLastError(), "rollback", error)) {
    return false;
  }
  session.set_position(first_pos + keep);
  return true;
}

bool Executor::ReadLogitRow(std::uint32_t row, std::span<float> out,
                            std::string* error) {
  if (row >= options_.logit_group || out.size() < kVocabRows) {
    if (error != nullptr) {
      *error = "logit row out of range";
    }
    return false;
  }
  return Check(
      hipMemcpy(out.data(), b_->logits + std::size_t{row} * kVocabRows,
                std::size_t{kVocabRows} * sizeof(float), hipMemcpyDeviceToHost),
      "read logit row", error);
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
  session.AwaitRowHolds(first_pos);
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
  const Lane lane{&session, first_pos, n, 0};
  if (!Block({&lane, 1}, il, n, sink, false, false, nullptr, error)) {
    (void)hipStreamSynchronize(stream_);
    return false;
  }
  return Check(hipMemcpyAsync(out.data(), b.h, in.size_bytes(),
                              hipMemcpyDeviceToHost, stream_),
               "download block output", error) &&
         Check(hipStreamSynchronize(stream_), "block", error);
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

bool Executor::ReadMtpCache(const Session& session, std::uint32_t rows,
                            std::vector<std::uint16_t>* out,
                            std::string* error) {
  if (session.state_.mtp_cache == nullptr || rows > session.capacity()) {
    if (error != nullptr) {
      *error = "the session has no MTP cache of that size";
    }
    return false;
  }
  out->resize(std::size_t{rows} * kMlaCacheWidth);
  return Check(hipMemcpy(out->data(), session.state_.mtp_cache, out->size() * 2,
                         hipMemcpyDeviceToHost),
               "read MTP cache", error);
}

bool Executor::ReadMtpHidden(const Session& session, std::vector<float>* out,
                             std::string* error) {
  if (session.state_.mtp_h == nullptr) {
    if (error != nullptr) {
      *error = "the session has no MTP state";
    }
    return false;
  }
  out->resize(kHidden);
  return Check(hipMemcpy(out->data(), session.state_.mtp_h,
                         kHidden * sizeof(float), hipMemcpyDeviceToHost),
               "read MTP hidden", error);
}

}  // namespace gufo::models::ling_3_flash::rocm
