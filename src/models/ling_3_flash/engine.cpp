#include "src/models/ling_3_flash/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <system_error>
#include <thread>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/kernels/rocm/device_model.hpp"
#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"
#include "src/models/ling_3_flash/mtp_sampling.hpp"
#include "src/models/ling_3_flash/weights.hpp"

namespace gufo::models::ling_3_flash {
namespace {

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

double SecondsSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
      .count();
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  const auto begin = std::chrono::steady_clock::now();
  std::shared_ptr<Model> m(new Model());
  if (options.max_batch == 0 || options.max_context == 0) {
    AssignError(error_msg, "max_batch and max_context must be positive");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  if (options.layer_first + options.layer_limit > c.num_layers) {
    AssignError(error_msg, "layer range exceeds the trunk's " +
                               std::to_string(c.num_layers) + " layers");
    return nullptr;
  }
  m->tokenizer_ = LingTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  std::fprintf(stderr, "[ling] bound %s in %.2f s; uploading weights\n",
               model_path.c_str(), SecondsSince(begin));
  m->device_ =
      rocm::DeviceModel::Upload(*m->weights_, *m->reader_, options.layer_first,
                                options.layer_limit, error_msg, options.mtp);
  if (!m->device_) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.max_batch = options.max_batch;
  exec.expanded_kv_bytes = options.expanded_kv_bytes;
  exec.max_context = options.max_context;
  exec.graphs = options.graphs;
  exec.mtp = options.mtp;
  m->executor_ = rocm::Executor::Create(*m->device_, exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  m->load_seconds_ = SecondsSince(begin);
  std::fprintf(
      stderr, "[ling] ready: weights %.2f GiB, scratch %.2f GiB, load %.1f s\n",
      static_cast<double>(m->ResidentBytes()) / (1ULL << 30),
      static_cast<double>(m->ScratchBytes()) / (1ULL << 30), m->load_seconds_);
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  auto state = executor_->CreateSession(max_context, error_msg);
  if (!state) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(state)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  const auto ids = tokenizer_->Encode(text);
  return {ids.begin(), ids.end()};
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeToken(static_cast<TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->EosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token == EosToken();
}

std::uint32_t Model::VocabSize() const noexcept {
  return kVocab;
}
std::uint32_t Model::FirstLayer() const noexcept {
  return device_->first_layer();
}
std::uint32_t Model::LayerCount() const noexcept {
  return executor_->layer_count();
}
const Config& Model::config() const noexcept {
  return weights_->config;
}
std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes();
}
std::size_t Model::ScratchBytes() const noexcept {
  return executor_->scratch_bytes();
}
void Model::SetMtpDraftHeadQ8(bool q8) noexcept {
  executor_->SetMtpDraftHeadQ8(q8);
}

bool Model::MtpDraftHeadQ8() const noexcept {
  return executor_->MtpDraftHeadQ8();
}

std::size_t Model::SessionBytes(std::uint32_t context, bool mtp) noexcept {
  return ling_3_flash::SessionBytes(context, mtp).Total();
}
std::uint64_t Model::GraphReplays() const noexcept {
  return executor_->graph_replays();
}

std::uint64_t Model::CaptureInvalidations() const noexcept {
  return executor_->capture_invalidations();
}

void Model::BreakCapturesForTest(std::uint32_t count) noexcept {
  executor_->BreakCapturesForTest(count);
}

std::uint32_t Model::BatchRowLimit() const noexcept {
  return executor_->batch_row_limit();
}

void Model::HoldSnapshotRowCopiesForTest(bool hold) noexcept {
  executor_->HoldRowCopiesForTest(hold);
}

void Model::FailSnapshotRowCopiesForTest(std::uint32_t count) noexcept {
  executor_->FailRowCopiesForTest(count);
}

void Model::FailDraftBatchForTest(std::uint32_t skip) noexcept {
  executor_->FailDraftBatchForTest(skip);
}

void Model::FailBatchesForTest(std::uint32_t count, bool before_wait) noexcept {
  executor_->FailBatchesForTest(
      count, before_wait ? rocm::Executor::BatchFailurePoint::kBeforeSync
                         : rocm::Executor::BatchFailurePoint::kAfterSync);
}

double Model::UploadSeconds() const noexcept {
  return device_->stats().seconds;
}

// ---- Session ---------------------------------------------------------------

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)), session_(std::move(session)) {}

Session::~Session() = default;

std::uint32_t Session::ExpandedRowsValid() const noexcept {
  return session_->expanded_rows_valid();
}

std::uint32_t Session::ExpandedStoreRows() const noexcept {
  return session_->expanded_store_rows();
}

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->bytes();
}

void Session::Reset() {
  Invalidate();
  valid_ = session_->Reset();
  reset_pending_ = !valid_;
}

void Session::Invalidate() noexcept {
  tokens_.clear();
  logits_.clear();
  pending_ = {};
  // Controller state (successes, retry backoff, measured costs) must not leak
  // into the next request, or seeded sampled replay would depend on history.
  ResetDraftPolicy();
  draft_position_ = UINT32_MAX;
  // Host-side position and derived state only: position 0 drops every
  // expanded K/V watermark, and the MTP rows no longer match any tokens.
  session_->set_position(0);
  session_->set_mtp_valid(false);
  valid_ = true;
  reset_pending_ = true;
}

std::int32_t Session::Argmax() const noexcept {
  if (!valid_ || logits_.empty()) {
    return -1;
  }
  return static_cast<std::int32_t>(
      std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "empty prompt");
    return false;
  }
  const bool prefix =
      valid_ && tokens_.size() <= prompt.size() &&
      std::equal(tokens_.begin(), tokens_.end(), prompt.begin());
  if (!prefix) {
    Reset();
    if (!valid_) {
      AssignError(error_msg, "session reset failed");
      return false;
    }
  }
  if (tokens_.size() == prompt.size() && !logits_.empty()) {
    return true;
  }
  if (tokens_.size() == prompt.size()) {
    // Same tokens but no logits (cannot happen after Feed); redo from scratch.
    Reset();
  }
  return Feed(prompt.subspan(tokens_.size()), {}, error_msg);
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  return Feed(std::span<const std::int32_t>(&token, 1), {}, error_msg);
}

void Session::EvaluateBatch(std::span<BatchLane> lanes) {
  std::vector<rocm::BatchSegment> segments;
  std::vector<BatchLane*> members;
  std::vector<bool> had_logits;
  segments.reserve(lanes.size());
  members.reserve(lanes.size());
  const Model* model = nullptr;
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    BatchLane& lane = lanes[i];
    lane.ok = false;
    lane.deferred = false;
    lane.error.clear();
    Session* s = lane.session;
    if (s == nullptr) {
      lane.error = "batch lane has no session";
      continue;
    }
    if (model == nullptr) {
      model = s->model_.get();
    }
    if (s->model_.get() != model) {
      lane.error = "batch lane belongs to another model";
      continue;
    }
    if (std::any_of(lanes.begin(), lanes.begin() + i,
                    [&](const BatchLane& o) { return o.session == s; })) {
      lane.error = "session appears twice in one batch";
      continue;
    }
    if (segments.size() == model->executor_->batch_row_limit()) {
      lane.error = "a batch holds at most " +
                   std::to_string(model->executor_->batch_row_limit()) +
                   " sessions";
      lane.deferred = true;
      continue;
    }
    // Evaluate's own checks (Feed), in its order and with its effects.
    if (!s->valid_) {
      lane.error = "session is invalid after an earlier failure";
      continue;
    }
    if (s->reset_pending_) {
      if (!s->session_->Reset(&lane.error)) {
        s->valid_ = false;
        continue;
      }
      s->reset_pending_ = false;
    }
    if (s->pending_.active) {
      lane.error = "a verify step is waiting for Rollback()";
      continue;
    }
    const std::uint32_t position = s->session_->position();
    if (position + 1 > s->session_->capacity()) {
      lane.error = "context overflow: " + std::to_string(position) + " + 1 > " +
                   std::to_string(s->session_->capacity());
      continue;
    }
    // Refusals of the forward itself leave an Evaluate'd session invalid.
    if (model->executor_->first_layer() != 0 || lane.token < 0 ||
        static_cast<std::uint32_t>(lane.token) >= kVocab) {
      lane.error = model->executor_->first_layer() != 0
                       ? "forward needs the trunk from block 0 (layer_first "
                         "is set)"
                       : "token id " + std::to_string(lane.token) +
                             " outside the vocabulary";
      s->valid_ = false;
      s->logits_.clear();
      continue;
    }
    had_logits.push_back(s->logits_.size() == kVocab);
    s->logits_.resize(kVocab);
    segments.push_back({.session = s->session_.get(),
                        .tokens = std::span<const std::int32_t>(&lane.token, 1),
                        .logits = s->logits_});
    members.push_back(&lane);
  }
  if (segments.empty()) {
    return;
  }
  (void)model->executor_->ForwardBatch(segments);
  for (std::size_t i = 0; i < members.size(); ++i) {
    BatchLane& lane = *members[i];
    Session& s = *lane.session;
    switch (segments[i].status) {
      case rocm::BatchSegment::Status::kDone:
        s.tokens_.push_back(lane.token);
        lane.ok = true;
        break;
      case rocm::BatchSegment::Status::kRejected:
        // Batch-only refusal (row budget, a snapshot in flight): nothing ran.
        if (!had_logits[i]) {
          s.logits_.clear();
        }
        lane.error = segments[i].error;
        lane.deferred = true;
        break;
      default:
        // Launched and failed: the state is lost and the device may be too.
        s.Invalidate();
        lane.error = segments[i].error.empty() ? "batched forward failed"
                                               : segments[i].error;
        break;
    }
  }
}

bool Session::Feed(std::span<const std::int32_t> tokens,
                   const FeedOptions& options, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg, "session is invalid after an earlier failure");
    return false;
  }
  if (reset_pending_) {
    if (!session_->Reset(error_msg)) {
      valid_ = false;
      return false;
    }
    reset_pending_ = false;
  }
  if (pending_.active) {
    AssignError(error_msg, "a verify step is waiting for Rollback()");
    return false;
  }
  if (tokens.empty()) {
    AssignError(error_msg, "no tokens to feed");
    return false;
  }
  const auto& exec = model_->executor_;
  const std::uint32_t chunk_cap = model_->options_.max_batch;
  const std::uint32_t total = static_cast<std::uint32_t>(tokens.size());
  if (session_->position() + total > session_->capacity()) {
    AssignError(error_msg,
                "context overflow: " + std::to_string(session_->position()) +
                    " + " + std::to_string(total) + " > " +
                    std::to_string(session_->capacity()));
    return false;
  }
  const bool collect =
      options.logits_from != UINT32_MAX && options.all_logits != nullptr;
  if (collect) {
    options.all_logits->clear();
  }
  std::vector<float> chunk_logits;
  // A call of more than 8 tokens stays in the prefill numeric family: when
  // the capacity split would leave a final chunk of 1..8 tokens (the
  // small-batch family), the last cap + tail tokens are split in two halves.
  const std::uint32_t tail = total % chunk_cap;
  const bool balance =
      total > chunk_cap && tail >= 1 && tail <= 8 && (chunk_cap + tail) / 2 > 8;
  for (std::uint32_t start = 0, n = 0; start < total; start += n) {
    const std::uint32_t remaining = total - start;
    n = balance && remaining == chunk_cap + tail
            ? remaining / 2
            : std::min(chunk_cap, remaining);
    const bool last = start + n == total;
    // Rows of this chunk whose logits are needed: everything from `want_from`.
    std::uint32_t want_from = n;
    if (collect && options.logits_from < start + n) {
      want_from = options.logits_from > start ? options.logits_from - start : 0;
    }
    if (last) {
      want_from = std::min(want_from, n - 1);
    }
    rocm::ForwardRequest request;
    request.tokens = tokens.subspan(start, n);
    request.sink = options.sink;
    request.block_timing = options.timings != nullptr;
    request.op_profile = options.op_profile;
    if (want_from < n) {
      chunk_logits.resize(std::size_t{n - want_from} * kVocab);
      request.logits = chunk_logits;
      request.logits_from = want_from;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!exec->Forward(*session_, request, error_msg)) {
      valid_ = false;
      logits_.clear();
      return false;
    }
    tokens_.insert(tokens_.end(), request.tokens.begin(), request.tokens.end());
    if (want_from < n) {
      if (collect) {
        // Rows requested by the caller are the ones at or past logits_from.
        const std::uint32_t first_row =
            options.logits_from > start ? options.logits_from - start : 0;
        if (first_row < n) {
          const auto skip = std::size_t{first_row - want_from} * kVocab;
          options.all_logits->insert(options.all_logits->end(),
                                     chunk_logits.begin() + skip,
                                     chunk_logits.end());
        }
      }
      if (last) {
        logits_.assign(chunk_logits.end() - kVocab, chunk_logits.end());
      }
    }
    if (options.timings != nullptr) {
      const rocm::ForwardTiming& ft = exec->last_timing();
      StepTiming st;
      st.tokens = n;
      st.total_ms = ft.total_ms;
      st.head_ms = ft.head_ms;
      for (std::uint32_t il = 0; il < ft.layer_ms.size(); ++il) {
        (IsMlaLayer(il)     ? st.mla_ms
         : IsDenseLayer(il) ? st.dense_ms
                            : st.kda_ms) += ft.layer_ms[il];
      }
      st.host_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
      options.timings->push_back(st);
    }
  }
  return true;
}

std::size_t Session::SnapshotBytes() const noexcept {
  return valid_ && !pending_.active && !tokens_.empty() && !logits_.empty()
             ? SnapshotPayloadBytes(static_cast<std::uint32_t>(tokens_.size()),
                                    session_->mtp_enabled())
             : 0;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  return SaveSnapshotImpl(error_msg, false);
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshotSynchronouslyForTest(
    std::string* error_msg) const {
  return SaveSnapshotImpl(error_msg, true);
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshotImpl(
    std::string* error_msg, bool synchronous) const {
  if (!valid_ || pending_.active || tokens_.empty() ||
      logits_.size() != kVocab || tokens_.size() != session_->position()) {
    AssignError(error_msg, "snapshot needs a valid, non-empty session");
    return nullptr;
  }
  const auto position = static_cast<std::uint32_t>(tokens_.size());
  const bool mtp = session_->mtp_enabled() && session_->mtp_valid();
  if (session_->mtp_enabled() && !mtp) {
    AssignError(error_msg, "the session's MTP state is not valid");
    return nullptr;
  }
  const SnapshotSections at = SnapshotLayout(position, mtp);
  std::unique_ptr<SessionSnapshot> s(new SessionSnapshot(position, at.total));
  std::uint8_t* base = s->data_;
  const SnapshotHeader header{
      .magic = {kSnapshotMagic[0], kSnapshotMagic[1], kSnapshotMagic[2],
                kSnapshotMagic[3], kSnapshotMagic[4], kSnapshotMagic[5],
                kSnapshotMagic[6], kSnapshotMagic[7]},
      .version = kSnapshotPayloadVersion,
      .header_bytes = static_cast<std::uint32_t>(kSnapshotHeaderBytes),
      .vocab = kVocab,
      .position = position,
      .kda_layers = kKdaLayers,
      .mla_layers = kMlaLayers,
      .mla_width = kMlaCacheWidth,
      .reserved = mtp ? kSnapshotHasMtp : 0U,
      .kda_state_floats = kKdaStateFloats,
      .kda_conv_floats = kKdaConvStateFloats,
      .payload_bytes = at.total};
  std::memcpy(base, &header, sizeof(header));
  std::memcpy(base + at.tokens, tokens_.data(),
              tokens_.size() * sizeof(std::int32_t));
  std::memcpy(base + at.logits, logits_.data(), logits_.size() * sizeof(float));
  const rocm::Executor::SnapshotBuffers buffers{
      .kda_state = base + at.kda_state,
      .kda_conv = base + at.kda_conv,
      .mla = base + at.mla,
      .mtp_h = mtp ? base + at.mtp_h : nullptr,
      .mtp = mtp ? base + at.mtp : nullptr};
  if (synchronous) {
    // The one-call executor path (both phases back to back, no worker).
    if (!model_->executor_->SaveSnapshot(*session_, position, buffers,
                                         error_msg)) {
      return nullptr;
    }
    s->StartCopy([](const std::atomic<bool>&, std::string*) { return true; },
                 [] {});
    return s;
  }
  rocm::RowHold hold;
  if (!model_->executor_->CaptureSnapshotState(*session_, position, buffers,
                                               &hold, error_msg)) {
    return nullptr;
  }
  // The worker needs the executor and the session's device buffers only until
  // it releases the hold; the session's destructor waits for that.
  const auto held = std::make_shared<rocm::RowHold>(std::move(hold));
  s->StartCopy(
      [executor = model_->executor_.get(), device = session_.get(), position,
       buffers](const std::atomic<bool>& abort, std::string* error) {
        return executor->CopySnapshotRows(*device, position, buffers, &abort,
                                          error);
      },
      [held] { held->Release(); });
  return s;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  if (!snapshot.Wait(error_msg)) {
    return false;  // the session is untouched
  }
  return RestoreSnapshot(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  SnapshotHeader h{};
  if (payload.size() < sizeof(h)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&h, payload.data(), sizeof(h));
  if (std::memcmp(h.magic, kSnapshotMagic.data(), kSnapshotMagic.size()) != 0 ||
      h.version != kSnapshotPayloadVersion ||
      h.header_bytes != kSnapshotHeaderBytes) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  const bool has_mtp = (h.reserved & kSnapshotHasMtp) != 0;
  if (h.vocab != kVocab || h.kda_layers != kKdaLayers ||
      h.mla_layers != kMlaLayers || h.mla_width != kMlaCacheWidth ||
      h.kda_state_floats != kKdaStateFloats ||
      h.kda_conv_floats != kKdaConvStateFloats || h.position == 0 ||
      h.position > session_->capacity() || (h.reserved & ~kSnapshotHasMtp) ||
      h.payload_bytes != SnapshotPayloadBytes(h.position, has_mtp) ||
      payload.size() != h.payload_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  const SnapshotSections at = SnapshotLayout(h.position, has_mtp);
  const std::uint8_t* base = payload.data();
  std::vector<std::int32_t> tokens(h.position);
  std::memcpy(tokens.data(), base + at.tokens,
              tokens.size() * sizeof(std::int32_t));
  std::vector<float> logits(kVocab);
  std::memcpy(logits.data(), base + at.logits, logits.size() * sizeof(float));
  // The payload may come from a disk file: refuse token ids outside the
  // vocabulary and non-finite logits before the session is touched.
  if (!std::ranges::all_of(tokens,
                           [](std::int32_t t) {
                             return t >= 0 &&
                                    t < static_cast<std::int32_t>(kVocab);
                           }) ||
      !std::ranges::all_of(logits, [](float v) { return std::isfinite(v); })) {
    AssignError(error_msg, "session snapshot holds invalid tokens or logits");
    return false;
  }
  if (!model_->executor_->RestoreSnapshot(
          *session_, h.position,
          {.kda_state = base + at.kda_state,
           .kda_conv = base + at.kda_conv,
           .mla = base + at.mla,
           .mtp_h = has_mtp ? base + at.mtp_h : nullptr,
           .mtp = has_mtp ? base + at.mtp : nullptr},
          error_msg)) {
    valid_ = false;
    return false;
  }
  pending_ = {};
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  valid_ = true;
  reset_pending_ = false;  // every recurrent state was overwritten
  return true;
}

bool Session::RunBlock(std::uint32_t il, std::uint32_t first_pos,
                       std::span<const float> in, std::span<float> out,
                       GpuTensorSink* sink, std::string* error_msg) {
  return model_->executor_->RunBlock(*session_, il, first_pos, in, out, sink,
                                     error_msg);
}

bool Session::ReadKdaState(std::uint32_t il, std::vector<float>* state,
                           std::vector<float>* conv,
                           std::string* error_msg) const {
  return model_->executor_->ReadKdaState(*session_, il, state, conv, error_msg);
}

bool Session::ReadMlaCache(std::uint32_t il, std::uint32_t rows,
                           std::vector<std::uint16_t>* out,
                           std::string* error_msg) const {
  return model_->executor_->ReadMlaCache(*session_, il, rows, out, error_msg);
}

// ---- MTP -------------------------------------------------------------------

bool Session::MtpActive() const noexcept {
  return model_->HasMtp() && session_->mtp_enabled() && session_->mtp_valid() &&
         valid_ && !tokens_.empty();
}

void Session::ResetDraftPolicy() noexcept {
  draft_length_.Reset();
}

void Session::SetMaxDrafts(std::uint32_t drafts) noexcept {
  draft_length_ = MtpLengthController(drafts);
}

bool Session::Verify(std::span<const std::int32_t> tokens,
                     std::span<std::int32_t> argmax, std::vector<float>* rows,
                     std::string* error_msg,
                     std::vector<OpTiming>* op_profile) {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (!valid_ || pending_.active || tokens_.empty() || !MtpActive()) {
    AssignError(error_msg, "verify needs a valid session with MTP state");
    return false;
  }
  if (n < 2 || n > kMaxVerifyTokens || argmax.size() < n ||
      session_->position() + n > session_->capacity()) {
    AssignError(error_msg,
                "verify chunk is outside 2..8 tokens or the context");
    return false;
  }
  rocm::ForwardRequest request;
  request.tokens = tokens;
  request.rollback = true;
  request.logits_from = 0;
  request.argmax = argmax.first(n);
  request.op_profile = op_profile;
  if (rows != nullptr) {
    rows->resize(std::size_t{n} * kVocab);
    request.logits = *rows;
  }
  const std::uint32_t first_pos = session_->position();
  if (!model_->executor_->Forward(*session_, request, error_msg)) {
    valid_ = false;
    logits_.clear();
    return false;
  }
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  logits_.clear();
  pending_ = {.active = true,
              .first_pos = first_pos,
              .n = n,
              .have_rows = rows != nullptr};
  if (rows != nullptr && rows != &verify_rows_) {
    verify_rows_ = *rows;
  }
  return true;
}

bool Session::Rollback(std::uint32_t keep, std::string* error_msg) {
  if (!pending_.active || keep == 0 || keep > pending_.n) {
    AssignError(error_msg, "rollback needs a pending verify and 1..n rows");
    return false;
  }
  const PendingVerify v = pending_;
  if (!model_->executor_->MtpRollback(
          *session_, v.first_pos, v.n, keep, error_msg,
          {.ckpt0 = v.ckpt0, .hn0 = v.hn0, .logit_row = v.logit_row})) {
    valid_ = false;
    return false;
  }
  tokens_.resize(v.first_pos + keep);
  logits_.resize(kVocab);
  if (v.have_rows) {
    std::copy_n(verify_rows_.begin() + std::size_t{keep - 1} * kVocab, kVocab,
                logits_.begin());
  } else if (!model_->executor_->ReadLogitRow(v.logit_row + keep - 1, logits_,
                                              error_msg)) {
    valid_ = false;
    logits_.clear();
    return false;
  }
  pending_ = {};
  return true;
}

bool Session::Draft(std::int32_t token, bool chain, std::span<float> logits,
                    std::int32_t* argmax, std::string* error_msg,
                    std::vector<OpTiming>* op_profile) {
  if (!MtpActive() || pending_.active ||
      (chain && draft_position_ == UINT32_MAX)) {
    AssignError(error_msg, "draft needs a valid session with MTP state");
    return false;
  }
  rocm::DraftRequest request;
  request.token = token;
  request.chain = chain;
  request.position = chain ? draft_position_ + 1 : session_->position() - 1;
  request.logits = logits;
  request.op_profile = op_profile;
  if (!model_->executor_->MtpDraft(*session_, request, argmax, error_msg)) {
    return false;
  }
  draft_position_ = request.position;
  return true;
}

LaunchTiming Session::LastLaunchTiming() const noexcept {
  return model_->executor_->last_launch();
}

bool Session::DraftWithHidden(std::int32_t token, std::span<const float> h,
                              std::uint32_t position, std::span<float> logits,
                              std::span<float> h_next, std::int32_t* argmax,
                              std::string* error_msg) {
  rocm::DraftRequest request;
  request.token = token;
  request.position = position;
  request.logits = logits;
  request.h_host = h;
  request.h_out = h_next;
  return model_->executor_->MtpDraft(*session_, request, argmax, error_msg);
}

bool Session::ReadMtpCache(std::uint32_t rows, std::vector<std::uint16_t>* out,
                           std::string* error_msg) const {
  return model_->executor_->ReadMtpCache(*session_, rows, out, error_msg);
}

bool Session::ReadMtpHidden(std::vector<float>* out,
                            std::string* error_msg) const {
  return model_->executor_->ReadMtpHidden(*session_, out, error_msg);
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  if (!valid_ || pending_.active) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (result == nullptr || max_tokens == 0 || tokens_.empty() ||
      logits_.size() != kVocab) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && IsStop(token);
  };
  const std::size_t room = session_->capacity() - session_->position();
  if (room == 0) {
    result->stop = true;
    return true;
  }
  const auto begin = std::chrono::steady_clock::now();
  const auto cap = static_cast<std::uint32_t>(
      std::min<std::size_t>({max_tokens, room, kMaxVerifyTokens}));
  const bool sampled = sampler.config().uses_random_sampling();
  const bool greedy_fast = sampler.config().temperature == 0.0F &&
                           sampler.config().can_use_unmodified_argmax();
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  std::uint32_t drafts =
      MtpActive() && cap > 1
          ? draft_length_.Choose(cap - 1, !sampled, session_->position())
          : 0;
  if (hooks_.fixed_drafts != 0 && cap > 1 && MtpActive()) {
    drafts = std::min(hooks_.fixed_drafts, cap - 1);
  }
  const auto elapsed_ms = [&] {
    return std::chrono::duration<float, std::milli>(
               std::chrono::steady_clock::now() - begin)
        .count();
  };
  if (drafts == 0) {
    if (!Evaluate(anchor, error_msg)) {
      return false;
    }
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens = {anchor};
    if (MtpActive()) {
      draft_length_.ObservePlainToken();
      if (!sampled) {
        draft_length_.ObserveCost(0, elapsed_ms());
      }
    }
    return true;
  }

  // Draft chain. A sampled proposal uses its own stream (MtpProposer).
  Cycle c;
  c.anchor = anchor;
  c.sampled = sampled;
  c.greedy_fast = greedy_fast;
  c.stop_at_eos = stop_at_eos;
  if (sampled) {
    c.proposer.emplace(sampler, static_cast<sampling::TokenId>(anchor));
  }
  c.chain = {anchor};
  if (sampled && draft_logits_.size() != kVocab) {
    draft_logits_.resize(kVocab);
  }
  for (std::uint32_t j = 0; j < drafts; ++j) {
    std::int32_t index = -1;
    if (hooks_.source && !sampled) {
      c.chain.push_back(hooks_.source(c.chain));
      continue;
    }
    if (!Draft(c.chain.back(), j > 0,
               sampled ? std::span<float>(draft_logits_) : std::span<float>{},
               &index, error_msg)) {
      return false;  // the trunk state is untouched by draft steps
    }
    if (sampled) {
      index = static_cast<std::int32_t>(c.proposer->Propose(draft_logits_));
    }
    c.chain.push_back(index);
  }

  // Verify [anchor, drafts] as one small-batch chunk.
  std::vector<float>* rows = greedy_fast ? nullptr : &verify_rows_;
  if (!Verify(c.chain, c.argmax, rows, error_msg)) {
    return false;
  }
  return FinishCycle(c, sampler, result, error_msg, &begin);
}

bool Session::IsStop(std::int32_t token) const noexcept {
  return model_->IsStopToken(token) ||
         (hooks_.stop_token >= 0 && token == hooks_.stop_token);
}

bool Session::FinishCycle(Cycle& c, sampling::SamplerState& sampler,
                          DecodeResult* result, std::string* error_msg,
                          const std::chrono::steady_clock::time_point* begin) {
  const auto is_stop = [&](std::int32_t token) {
    return c.stop_at_eos && IsStop(token);
  };
  const bool sampled = c.sampled;
  const auto& chain = c.chain;
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto row_logits = [&](std::uint32_t row) {
    return std::span<const float>(verify_rows_)
        .subspan(std::size_t{row} * kVocab, kVocab);
  };
  sampler.Accept(static_cast<sampling::TokenId>(c.anchor));
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  // Sampling can throw between Verify() and Rollback(); the rejected rows
  // would otherwise stay in the trunk state of a session that still looks
  // valid, so the session is invalidated before the exception propagates.
  try {
    if (hooks_.after_verify) {
      hooks_.after_verify();
    }
    if (sampled) {
      const auto verified =
          VerifySampledChain(row_logits, c.proposer->proposals(), sampler,
                             [&](sampling::TokenId token) {
                               return is_stop(static_cast<std::int32_t>(token));
                             });
      keep = verified.keep;
      result->stop = verified.stop;
      if (verified.correction) {
        correction = static_cast<std::int32_t>(*verified.correction);
      }
    }
    while (!sampled && keep < k) {
      if (c.greedy_fast) {
        const std::int32_t predicted = c.argmax[keep - 1];
        if (predicted < 0) {
          AssignError(error_msg,
                      "logit distribution contains no finite values");
          valid_ = false;
          return false;
        }
        if (is_stop(predicted)) {
          result->stop = true;
          break;
        }
        if (predicted != chain[keep]) {
          break;
        }
        sampler.Accept(static_cast<sampling::TokenId>(predicted));
        ++keep;
        continue;
      }
      const auto decision =
          VerifyDraft(row_logits(keep - 1), chain[keep], sampler, is_stop);
      if (decision != DraftDecision::kAccept) {
        result->stop = decision == DraftDecision::kStop;
        break;
      }
      ++keep;
    }
  } catch (...) {
    pending_ = {};
    valid_ = false;
    throw;
  }
  if (!Rollback(keep, error_msg)) {
    return false;
  }
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  ++stats_.cycles;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions.
  draft_length_.Observe(keep - 1, result->stop ? keep - 1 : k - 1, !sampled,
                        session_->position());
  if (!sampled && begin != nullptr) {
    draft_length_.ObserveCost(k - 1,
                              std::chrono::duration<float, std::milli>(
                                  std::chrono::steady_clock::now() - *begin)
                                  .count());
  }
  if (correction) {
    // The residual draw is the next cycle's anchor; resampling would bias it.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

void Session::DecodeBatch(std::span<BatchDecode> lanes) {
  using Clock = std::chrono::steady_clock;
  const auto ms_since = [](Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  };
  const auto call_start = Clock::now();
  struct Work {
    BatchDecode* d;
    Session* s;
    Cycle c;
    std::uint32_t k{0};
    bool cohort{false};  // greedy lane taking the cohort's draft width
    bool live{true};
  };
  std::vector<Work> work;
  work.reserve(lanes.size());
  struct Ready {
    BatchDecode* d;
    Session* s;
    std::uint32_t cap;
  };
  std::vector<Ready> ready;
  Model* model = nullptr;
  const auto fail = [](BatchDecode& d, std::string message) {
    d.ok = false;
    d.error = std::move(message);
  };
  // ---- anchors and draft widths, host only (DecodeStep's own checks) ------
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    BatchDecode& d = lanes[i];
    d.ok = false;
    d.deferred = false;
    d.error.clear();
    d.result = {};
    d.ms = 0.0;
    d.width = 1;
    Session* s = d.session;
    if (s == nullptr || d.sampler == nullptr) {
      fail(d, "batch lane needs a session and a sampler");
      continue;
    }
    if (model == nullptr) {
      model = s->model_.get();
    }
    if (s->model_.get() != model) {
      fail(d, "batch lane belongs to another model");
      continue;
    }
    if (std::any_of(lanes.begin(), lanes.begin() + i,
                    [&](const BatchDecode& o) { return o.session == s; })) {
      fail(d, "session appears twice in one batch");
      continue;
    }
    if (ready.size() == model->executor_->batch_row_limit()) {
      fail(d, "a batch holds at most " +
                  std::to_string(model->executor_->batch_row_limit()) +
                  " sessions");
      d.deferred = true;
      continue;
    }
    const auto t0 = Clock::now();
    if (!s->valid_ || s->pending_.active) {
      fail(d, "session needs a successful Sync after a failed operation");
      continue;
    }
    if (d.max_tokens == 0 || s->tokens_.empty() ||
        s->logits_.size() != kVocab) {
      fail(d, "decode needs an output, a positive budget and a synced prompt");
      continue;
    }
    const std::size_t room = s->session_->capacity() - s->session_->position();
    if (room == 0) {
      d.result.stop = true;
      d.ok = true;
      continue;
    }
    const auto cap = static_cast<std::uint32_t>(
        std::min<std::size_t>({d.max_tokens, room, kMaxVerifyTokens}));
    ready.push_back({.d = &d, .s = s, .cap = cap});
    d.ms += ms_since(t0);
  }
  if (ready.size() == 1) {
    // One lane: DecodeStep itself (its graph-replayed plain steps).
    BatchDecode& d = *ready.front().d;
    const auto t0 = Clock::now();
    try {
      d.ok = ready.front().s->DecodeStep(d.max_tokens, *d.sampler, &d.result,
                                         &d.error, d.stop_at_eos);
    } catch (const std::exception& e) {
      fail(d, e.what());
    }
    d.ms += ms_since(t0);
    return;
  }
  for (const Ready& r : ready) {
    BatchDecode& d = *r.d;
    Session* s = r.s;
    const std::uint32_t cap = r.cap;
    const auto t0 = Clock::now();
    sampling::SamplerState& sampler = *d.sampler;
    Work w{.d = &d, .s = s};
    w.c.sampled = sampler.config().uses_random_sampling();
    w.c.greedy_fast = sampler.config().temperature == 0.0F &&
                      sampler.config().can_use_unmodified_argmax();
    w.c.stop_at_eos = d.stop_at_eos;
    try {
      w.c.anchor = static_cast<std::int32_t>(sampler.Sample(s->logits_));
    } catch (const std::exception& e) {
      fail(d, e.what());
      continue;
    }
    if (d.stop_at_eos && s->IsStop(w.c.anchor)) {
      d.result.stop = true;
      d.ok = true;
      continue;
    }
    w.c.chain = {w.c.anchor};
    if (s->MtpActive() && cap > 1) {
      if (s->hooks_.fixed_drafts != 0) {
        w.k = std::min(s->hooks_.fixed_drafts, cap - 1);
      } else if (w.c.sampled) {
        // The width of the lane's own history, as DecodeStep (seeded replay).
        w.k = s->draft_length_.Choose(cap - 1, false, s->session_->position());
      } else {
        w.cohort = true;
        w.k = cap - 1;  // the cohort width, capped below
      }
    }
    d.ms += ms_since(t0);
    work.push_back(std::move(w));
  }
  if (work.empty()) {
    return;
  }
  rocm::Executor& exec = *model->executor_;
  const std::uint32_t row_limit = exec.batch_row_limit();
  // ---- the greedy lanes' common width -------------------------------------
  std::vector<MtpCohortController::Row> rows;
  std::uint32_t other_rows = 0;
  std::uint32_t context = 0;
  for (const Work& w : work) {
    if (w.cohort) {
      rows.push_back({&w.s->draft_length_, w.k});
      context = std::max(context, w.s->session_->position());
    } else {
      other_rows += w.k + 1;
    }
  }
  std::uint32_t cohort_width = 0;
  if (!rows.empty()) {
    const auto g = static_cast<std::uint32_t>(rows.size());
    // The greedy lanes' rows fit one forward by themselves at most; whether
    // they also fit beside the other lanes (one verify forward) or take a
    // forward of their own is the controller's cost trade (sampled widths
    // are never cut; the grouping below splits).
    const std::uint32_t fit = row_limit / g - 1;
    for (auto& row : rows) {
      row.budget = std::min(row.budget, fit);
    }
    cohort_width = model->cohort_.Choose(rows, work.size(), other_rows,
                                         row_limit, context);
    cohort_width = std::min(cohort_width, fit);
    for (Work& w : work) {
      if (w.cohort) {
        w.k = std::min(w.k, cohort_width);
      }
    }
  }
  // ---- draft chains: one batched draft step per depth ---------------------
  for (Work& w : work) {
    const auto t0 = Clock::now();
    if (w.k > 0 && w.c.sampled) {
      w.c.proposer.emplace(*w.d->sampler,
                           static_cast<sampling::TokenId>(w.c.anchor));
      if (w.s->draft_logits_.size() != kVocab) {
        w.s->draft_logits_.resize(kVocab);
      }
    }
    if (w.k > 0 && !w.c.sampled && w.s->hooks_.source) {
      for (std::uint32_t j = 0; j < w.k; ++j) {
        w.c.chain.push_back(w.s->hooks_.source(w.c.chain));
      }
    }
    w.d->ms += ms_since(t0);
  }
  // Chains in descending length, so each depth's lanes are a prefix and keep
  // their row (the chained hidden input stays at its index).
  std::vector<Work*> drafting;
  for (Work& w : work) {
    if (w.k > 0 && !(!w.c.sampled && w.s->hooks_.source)) {
      drafting.push_back(&w);
    }
  }
  std::stable_sort(drafting.begin(), drafting.end(),
                   [](const Work* a, const Work* b) { return a->k > b->k; });
  std::vector<std::uint32_t> draft_position(drafting.size(), 0);
  bool step_failed = false;
  for (std::uint32_t j = 0;
       !step_failed && !drafting.empty() && j < drafting.front()->k; ++j) {
    std::vector<rocm::DraftLane> step;
    std::vector<Work*> members;
    // A lane whose proposal failed keeps its row as a placeholder (outputs
    // ignored) so the chains after it keep their rows.
    for (std::size_t i = 0; i < drafting.size() && drafting[i]->k > j; ++i) {
      Work& w = *drafting[i];
      rocm::DraftLane lane;
      lane.session = w.s->session_.get();
      lane.token = w.c.chain.back();
      lane.position =
          j == 0 ? w.s->session_->position() - 1 : draft_position[i] + 1;
      lane.chain = j > 0;
      if (w.c.sampled) {
        lane.logits = w.s->draft_logits_;
      }
      step.push_back(lane);
      members.push_back(&w);
    }
    if (step.empty()) {
      break;
    }
    const auto t0 = Clock::now();
    std::string error;
    const bool ok = exec.MtpDraftBatch(step, &error);
    const double ms = ms_since(t0);
    step_failed = !ok;
    for (std::size_t i = 0; i < members.size(); ++i) {
      Work& w = *members[i];
      if (!w.live) {
        draft_position[i] = step[i].position;
        continue;
      }
      w.d->ms += ms;
      w.d->width = std::max<std::uint32_t>(
          w.d->width, static_cast<std::uint32_t>(members.size()));
      if (!ok) {
        // As DecodeStep: a failed draft leaves the trunk state untouched.
        fail(*w.d, error);
        w.live = false;
        continue;
      }
      const auto t1 = Clock::now();
      draft_position[i] = step[i].position;
      w.s->draft_position_ = step[i].position;
      try {
        w.c.chain.push_back(w.c.sampled
                                ? static_cast<std::int32_t>(
                                      w.c.proposer->Propose(w.s->draft_logits_))
                                : step[i].argmax);
      } catch (const std::exception& e) {
        fail(*w.d, e.what());
        w.live = false;
      }
      w.d->ms += ms_since(t1);
    }
  }
  // ---- verify forwards (at most row_limit rows each) and acceptance -------
  std::vector<std::vector<Work*>> groups;
  std::vector<std::uint32_t> group_rows;
  for (Work& w : work) {
    if (!w.live) {
      continue;
    }
    const auto n = static_cast<std::uint32_t>(w.c.chain.size());
    std::size_t g = 0;
    while (g < groups.size() && group_rows[g] + n > row_limit) {
      ++g;
    }
    if (g == groups.size()) {
      groups.emplace_back();
      group_rows.push_back(0);
    }
    groups[g].push_back(&w);
    group_rows[g] += n;
  }
  bool any_failure = false;
  for (const auto& group : groups) {
    std::vector<rocm::BatchSegment> segments;
    std::vector<Work*> members;
    for (Work* wp : group) {
      Work& w = *wp;
      Session& s = *w.s;
      const auto n = static_cast<std::uint32_t>(w.c.chain.size());
      // Verify's (or, for a plain token, Feed's) own checks.
      if (!s.valid_ || s.pending_.active ||
          s.session_->position() + n > s.session_->capacity() ||
          (n > 1 && !s.MtpActive())) {
        fail(*w.d, n > 1 ? "verify needs a valid session with MTP state"
                         : "session is invalid after an earlier failure");
        continue;
      }
      rocm::BatchSegment seg;
      seg.session = s.session_.get();
      seg.tokens = w.c.chain;
      if (n == 1) {
        seg.logits = s.logits_;
      } else {
        seg.rollback = true;
        seg.logits_from = 0;
        seg.argmax = std::span<std::int32_t>(w.c.argmax.data(), n);
        if (!w.c.greedy_fast) {
          s.verify_rows_.resize(std::size_t{n} * kVocab);
          seg.logits = s.verify_rows_;
        }
      }
      segments.push_back(std::move(seg));
      members.push_back(&w);
    }
    if (segments.empty()) {
      continue;
    }
    const auto t0 = Clock::now();
    (void)exec.ForwardBatch(segments);
    const double ms = ms_since(t0);
    for (std::size_t i = 0; i < members.size(); ++i) {
      Work& w = *members[i];
      Session& s = *w.s;
      BatchDecode& d = *w.d;
      const rocm::BatchSegment& seg = segments[i];
      const auto n = static_cast<std::uint32_t>(w.c.chain.size());
      if (seg.status == rocm::BatchSegment::Status::kRejected) {
        d.deferred = true;
        fail(d, seg.error);
        continue;
      }
      d.ms += ms;
      d.width = std::max<std::uint32_t>(
          d.width, static_cast<std::uint32_t>(members.size()));
      if (seg.status != rocm::BatchSegment::Status::kDone) {
        // Launched and failed: the state is lost and the device may be too.
        s.Invalidate();
        fail(d, seg.error.empty() ? "batched verify failed" : seg.error);
        any_failure = true;
        continue;
      }
      const auto t1 = Clock::now();
      sampling::SamplerState& sampler = *d.sampler;
      if (n == 1) {
        // DecodeStep's plain step (Evaluate).
        s.tokens_.push_back(w.c.anchor);
        sampler.Accept(static_cast<sampling::TokenId>(w.c.anchor));
        d.result.tokens = {w.c.anchor};
        if (s.MtpActive()) {
          s.draft_length_.ObservePlainToken();
        }
        d.ok = true;
      } else {
        // Verify's bookkeeping, then DecodeStep's acceptance and rollback.
        const std::uint32_t first_pos = s.session_->position() - n;
        s.tokens_.insert(s.tokens_.end(), w.c.chain.begin(), w.c.chain.end());
        s.logits_.clear();
        s.pending_ = {.active = true,
                      .first_pos = first_pos,
                      .n = n,
                      .have_rows = !w.c.greedy_fast,
                      .ckpt0 = seg.placement.ckpt0,
                      .hn0 = seg.placement.hn0,
                      .logit_row = seg.placement.logit_row};
        std::string error;
        try {
          d.ok = s.FinishCycle(w.c, sampler, &d.result, &error, nullptr);
        } catch (const std::exception& e) {
          d.ok = false;
          error = e.what();
        }
        if (!d.ok) {
          fail(d, error);
          any_failure = true;
        }
      }
      d.ms += ms_since(t1);
    }
  }
  if (!rows.empty() && !any_failure) {
    model->cohort_.Observe(work.size(), context, cohort_width,
                           static_cast<float>(ms_since(call_start)));
  }
}

// ---- SessionSnapshot -------------------------------------------------------

/// The background row copy of one snapshot, shared by the snapshot and its
/// worker; it also owns the payload buffer the worker writes. The worker holds
/// no reference to the snapshot and releases its RowHold last, after
/// publishing the result, because the session may be destroyed right after.
/// Destroying the snapshot only sets `abort` and detaches the worker: joining
/// could stall the thread that evicts a cache entry behind another session's
/// restore, which holds the executor's transfer mutex for a whole restore.
/// The detached worker stops at its next chunk (until then it keeps the host
/// buffer, though a cache eviction or reap already counted the bytes as freed);
/// its device reads stay safe because ~rocm::Session waits for the RowHold it
/// still owns.
struct SessionSnapshot::AsyncCopy {
  explicit AsyncCopy(std::size_t size) : buffer(new std::uint8_t[size]) {}
  // Uninitialized: every byte is written by SaveSnapshot.
  std::unique_ptr<std::uint8_t[]> buffer;
  mutable std::mutex mutex;
  mutable std::condition_variable changed;
  bool done{false};
  bool ok{false};
  std::string error;
  std::atomic<bool> abort{false};
  std::thread worker;
};

SessionSnapshot::SessionSnapshot(std::uint32_t position, std::size_t size)
    : size_(size), position_(position), copy_(new AsyncCopy(size)) {
  data_ = copy_->buffer.get();
}

SessionSnapshot::~SessionSnapshot() {
  if (copy_ != nullptr && copy_->worker.joinable()) {
    copy_->abort.store(true);
    copy_->worker.detach();
  }
}

bool SessionSnapshot::Ready() const noexcept {
  const std::lock_guard<std::mutex> lock(copy_->mutex);
  return copy_->done;
}

bool SessionSnapshot::Failed() const noexcept {
  const std::lock_guard<std::mutex> lock(copy_->mutex);
  return copy_->done && !copy_->ok;
}

bool SessionSnapshot::Wait(std::string* error_msg) const {
  std::unique_lock<std::mutex> lock(copy_->mutex);
  copy_->changed.wait(lock, [this] { return copy_->done; });
  if (!copy_->ok) {
    AssignError(error_msg, copy_->error);
  }
  return copy_->ok;
}

std::span<const std::uint8_t> SessionSnapshot::bytes() const noexcept {
  return Wait() ? std::span<const std::uint8_t>(data_, size_)
                : std::span<const std::uint8_t>();
}

void SessionSnapshot::StartCopy(
    std::function<bool(const std::atomic<bool>&, std::string*)> copy,
    std::function<void()> release) {
  const std::shared_ptr<AsyncCopy> state = copy_;
  const auto run = [state, copy = std::move(copy),
                    release = std::move(release)] {
    bool ok = false;
    std::string error;
    try {
      ok = copy(state->abort, &error);
    } catch (...) {
      ok = false;
      error = "snapshot row copy threw";
    }
    // Publish in every path, then release the hold last: a missing release
    // would make ~rocm::Session wait forever.
    try {
      const std::lock_guard<std::mutex> lock(state->mutex);
      state->ok = ok;
      try {
        state->error = std::move(error);
      } catch (...) {
        state->error.clear();
      }
      state->done = true;
      state->changed.notify_all();
    } catch (...) {
      // The mutex itself failed; nothing more can be published safely.
    }
    try {
      release();
    } catch (...) {
    }
  };
  try {
    state->worker = std::thread(run);
  } catch (const std::system_error&) {
    run();
  }
}

}  // namespace gufo::models::ling_3_flash
