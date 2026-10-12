#include "src/models/ling_3_flash/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>

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
  draft_position_ = 0;
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
  std::uint8_t* base = s->data_.get();
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
  if (!model_->executor_->SaveSnapshot(
          *session_, position,
          {.kda_state = base + at.kda_state,
           .kda_conv = base + at.kda_conv,
           .mla = base + at.mla,
           .mtp_h = mtp ? base + at.mtp_h : nullptr,
           .mtp = mtp ? base + at.mtp : nullptr},
          error_msg)) {
    return nullptr;
  }
  return s;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
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
                     std::string* error_msg) {
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
  if (!model_->executor_->MtpRollback(*session_, v.first_pos, v.n, keep,
                                      error_msg)) {
    valid_ = false;
    return false;
  }
  tokens_.resize(v.first_pos + keep);
  logits_.resize(kVocab);
  if (v.have_rows) {
    std::copy_n(verify_rows_.begin() + std::size_t{keep - 1} * kVocab, kVocab,
                logits_.begin());
  } else if (!model_->executor_->ReadLogitRow(keep - 1, logits_, error_msg)) {
    valid_ = false;
    logits_.clear();
    return false;
  }
  pending_ = {};
  return true;
}

bool Session::Draft(std::int32_t token, bool chain, std::span<float> logits,
                    std::int32_t* argmax, std::string* error_msg) {
  if (!MtpActive() || pending_.active || (chain && draft_position_ == 0)) {
    AssignError(error_msg, "draft needs a valid session with MTP state");
    return false;
  }
  rocm::DraftRequest request;
  request.token = token;
  request.chain = chain;
  request.position = chain ? draft_position_ + 1 : session_->position() - 1;
  request.logits = logits;
  if (!model_->executor_->MtpDraft(*session_, request, argmax, error_msg)) {
    return false;
  }
  draft_position_ = request.position;
  return true;
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
    return stop_at_eos && model_->IsStopToken(token);
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
  std::optional<MtpProposer> proposer;
  if (sampled) {
    proposer.emplace(sampler, static_cast<sampling::TokenId>(anchor));
  }
  std::vector<std::int32_t> chain{anchor};
  if (sampled && draft_logits_.size() != kVocab) {
    draft_logits_.resize(kVocab);
  }
  for (std::uint32_t j = 0; j < drafts; ++j) {
    std::int32_t index = -1;
    if (hooks_.source && !sampled) {
      chain.push_back(hooks_.source(chain));
      continue;
    }
    if (!Draft(chain.back(), j > 0,
               sampled ? std::span<float>(draft_logits_) : std::span<float>{},
               &index, error_msg)) {
      return false;  // the trunk state is untouched by draft steps
    }
    if (sampled) {
      index = static_cast<std::int32_t>(proposer->Propose(draft_logits_));
    }
    chain.push_back(index);
  }

  // Verify [anchor, drafts] as one small-batch chunk.
  const auto k = static_cast<std::uint32_t>(chain.size());
  std::array<std::int32_t, kMaxVerifyTokens> argmax{};
  std::vector<float>* rows = greedy_fast ? nullptr : &verify_rows_;
  if (!Verify(chain, argmax, rows, error_msg)) {
    return false;
  }
  const auto row_logits = [&](std::uint32_t row) {
    return std::span<const float>(verify_rows_)
        .subspan(std::size_t{row} * kVocab, kVocab);
  };
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
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
          VerifySampledChain(row_logits, proposer->proposals(), sampler,
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
      if (greedy_fast) {
        const std::int32_t predicted = argmax[keep - 1];
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
  if (!sampled) {
    draft_length_.ObserveCost(k - 1, elapsed_ms());
  }
  if (correction) {
    // The residual draw is the next cycle's anchor; resampling would bias it.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

// ---- SessionSnapshot -------------------------------------------------------

SessionSnapshot::SessionSnapshot(std::uint32_t position, std::size_t size)
    // Uninitialized: every byte is written by SaveSnapshot.
    : data_(new std::uint8_t[size]), size_(size), position_(position) {}
SessionSnapshot::~SessionSnapshot() = default;

}  // namespace gufo::models::ling_3_flash
