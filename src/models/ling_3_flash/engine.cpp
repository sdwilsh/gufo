#include "src/models/ling_3_flash/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/kernels/rocm/device_model.hpp"
#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"
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
                                options.layer_limit, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.max_batch = options.max_batch;
  exec.expanded_kv_bytes = options.expanded_kv_bytes;
  exec.max_context = options.max_context;
  exec.graphs = options.graphs;
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
std::size_t Model::SessionBytes(std::uint32_t context) noexcept {
  return ling_3_flash::SessionBytes(context).Total();
}
std::uint64_t Model::GraphReplays() const noexcept {
  return executor_->graph_replays();
}

double Model::UploadSeconds() const noexcept {
  return device_->stats().seconds;
}

// ---- Session ---------------------------------------------------------------

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)), session_(std::move(session)) {}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->bytes();
}

void Session::Reset() {
  tokens_.clear();
  logits_.clear();
  valid_ = session_->Reset();
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
    // Rows of this chunk whose logits we need: everything from `want_from`.
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
  return valid_ && !tokens_.empty() && !logits_.empty()
             ? SnapshotPayloadBytes(static_cast<std::uint32_t>(tokens_.size()))
             : 0;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty() || logits_.size() != kVocab ||
      tokens_.size() != session_->position()) {
    AssignError(error_msg, "snapshot needs a valid, non-empty session");
    return nullptr;
  }
  const auto position = static_cast<std::uint32_t>(tokens_.size());
  const SnapshotSections at = SnapshotLayout(position);
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
      .reserved = 0,
      .kda_state_floats = kKdaStateFloats,
      .kda_conv_floats = kKdaConvStateFloats,
      .payload_bytes = at.total};
  std::memcpy(base, &header, sizeof(header));
  std::memcpy(base + at.tokens, tokens_.data(),
              tokens_.size() * sizeof(std::int32_t));
  std::memcpy(base + at.logits, logits_.data(), logits_.size() * sizeof(float));
  if (!model_->executor_->SaveSnapshot(*session_, position,
                                       {.kda_state = base + at.kda_state,
                                        .kda_conv = base + at.kda_conv,
                                        .mla = base + at.mla},
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
  if (h.vocab != kVocab || h.kda_layers != kKdaLayers ||
      h.mla_layers != kMlaLayers || h.mla_width != kMlaCacheWidth ||
      h.kda_state_floats != kKdaStateFloats ||
      h.kda_conv_floats != kKdaConvStateFloats || h.position == 0 ||
      h.position > session_->capacity() ||
      h.payload_bytes != SnapshotPayloadBytes(h.position) ||
      payload.size() != h.payload_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  const SnapshotSections at = SnapshotLayout(h.position);
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
  if (!model_->executor_->RestoreSnapshot(*session_, h.position,
                                          {.kda_state = base + at.kda_state,
                                           .kda_conv = base + at.kda_conv,
                                           .mla = base + at.mla},
                                          error_msg)) {
    valid_ = false;
    return false;
  }
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  valid_ = true;
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

// ---- SessionSnapshot -------------------------------------------------------

SessionSnapshot::SessionSnapshot(std::uint32_t position, std::size_t size)
    // Uninitialized: every byte is written by SaveSnapshot.
    : data_(new std::uint8_t[size]), size_(size), position_(position) {}
SessionSnapshot::~SessionSnapshot() = default;

}  // namespace gufo::models::ling_3_flash
