#include "src/models/ling_3_flash/continuation_adapter.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/transfer_pool.hpp"
#include "src/models/ling_3_flash/continuation_hooks.hpp"
#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"
#include "src/models/ling_3_flash/kernels/rocm/snapshot_copy.hpp"
#include "src/models/ling_3_flash/snapshot_format.hpp"

namespace gufo::models::ling_3_flash {
namespace {
using cache::Rows;
using cache::TransferResult;
constexpr std::size_t kLatentRowBytes = kMlaCacheWidth * sizeof(std::uint16_t);
constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

void Check(hipError_t result) {
  if (result != hipSuccess) {
    rocm::Executor::NoteDeviceError();
    throw std::runtime_error(hipGetErrorString(result));
  }
}
void RefuseSuspectDevice() {
  if (rocm::Executor::DeviceSuspect())
    throw std::runtime_error(
        "Ling cache transfer refused: a HIP call failed since the last "
        "successful forward");
}
bool HasRows(ContinuationPart part) {
  return part == ContinuationPart::kTokens || part == ContinuationPart::kMla ||
         part == ContinuationPart::kMtpRows;
}
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{};
  std::uint32_t mtp{};
  std::uint32_t mtp_valid{};
  std::uint64_t layout_hash{};
  std::uint32_t context{};
  std::uint32_t prefill_capacity{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

struct ContinuationAdapter::Piece {
  std::byte* pointer;
  std::size_t offset, bytes;
  bool host{false};
};

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> owned)
      : adapter(adapter), guard(guard), session(std::move(owned)) {
    Check(hipStreamCreateWithFlags(&control, hipStreamNonBlocking));
    try {
      Check(hipMalloc(&mismatch, sizeof(std::uint32_t) * kMlaLayers));
    } catch (...) {
      (void)hipStreamDestroy(control);
      throw;
    }
    session->session_->continuation_hooks_ = this;
  }
  ~State() override {
    {
      const std::lock_guard lock(jobs_mutex);
      for (const auto& job : jobs)
        (void)job->Wait();
    }
    Release();
    session->session_->continuation_hooks_ = nullptr;
    (void)hipFree(mismatch);
    (void)hipStreamDestroy(control);
  }

  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (*result == TransferResult::kFailed) {
          rocm::Executor::NoteDeviceError();
          if (load)
            owner->failed = true;
        }
        owner->Settle();
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };

  rocm::Session& Native() const { return *session->session_; }
  [[nodiscard]] bool IsValid() const noexcept override {
    return !restoring && !failed && (guarding || session->IsValid());
  }
  [[nodiscard]] bool Ready() const noexcept override {
    return !restoring && !failed && pending.load() == 0;
  }
  void Submitted() {
    ++pending;
    Native().transfers_.fetch_add(1, std::memory_order_acq_rel);
  }
  void Settle() noexcept {
    --pending;
    Native().transfers_.fetch_sub(1, std::memory_order_acq_rel);
  }
  // Rows below `first` keep their bytes; preserve borrowers of every row the
  // write may reach that is still intact, then lower the intact frontier.
  void BeforeWrite(std::uint32_t first, std::uint32_t end,
                   std::uint32_t mtp_first, std::uint32_t mtp_end) override {
    if (!Ready())
      throw std::logic_error("Ling slot is not ready for execution");
    const std::uint32_t position = Native().position();
    GuardScope scope(*this);
    if (first < end) {
      const auto intact = std::max(intact_mla.load(), position);
      if (first < intact) {
        for (const auto& c : adapter.layout_.Components())
          if (c.part == ContinuationPart::kMla)
            guard.BeforeOverwrite(c.descriptor.id, first, intact);
      }
      intact_mla = std::min(intact, first);
    }
    if (adapter.layout_.mtp() && mtp_first < mtp_end) {
      const auto intact =
          std::max(intact_mtp.load(), position > 0 ? position - 1 : 0U);
      if (mtp_first < intact)
        for (const auto& c : adapter.layout_.Components())
          if (c.part == ContinuationPart::kMtpRows)
            guard.BeforeOverwrite(c.descriptor.id, mtp_first, intact);
      intact_mtp = std::min(intact, mtp_first);
    }
  }
  // Every row component, before a restore replaces it (may refuse).
  void GuardAll() {
    GuardScope scope(*this);
    for (const auto& c : adapter.layout_.Components())
      if (HasRows(c.part))
        guard.BeforeOverwrite(c.descriptor.id, 0, adapter.context_);
  }
  // Release cannot refuse: borrowers are preserved, or retired when their
  // copy fails (for example after a device error, RefuseSuspectDevice).
  void Release() noexcept {
    GuardScope scope(*this);
    for (const auto& c : adapter.layout_.Components())
      if (HasRows(c.part))
        guard.BeforeRelease(c.descriptor.id, 0, adapter.context_);
  }
  // Rows of an append component whose device bytes are known unchanged.
  [[nodiscard]] Rows Intact(ContinuationPart part) const {
    const std::uint32_t position = Native().position();
    switch (part) {
      case ContinuationPart::kTokens:
        return session->tokens_.size();
      case ContinuationPart::kMla:
        return std::max(intact_mla.load(), position);
      case ContinuationPart::kMtpRows:
        return std::max(intact_mtp.load(), position > 0 ? position - 1 : 0U);
      default:
        return 0;
    }
  }
  struct GuardScope {
    explicit GuardScope(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~GuardScope() { state.guarding = previous; }
    State& state;
    bool previous;
  };

  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  hipStream_t control{};
  std::uint32_t* mismatch{};  ///< device [kMlaLayers]
  bool restoring{false}, guarding{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  // Intact frontiers (rows) of the MLA layers and MTP rows: positions only
  // fall below them after Invalidate or a rollback; writes lower them.
  mutable std::atomic<std::uint32_t> intact_mla{0}, intact_mtp{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
  std::array<std::uint32_t, kMlaLayers> watermark{}, fallback{}, kept{};
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         std::uint32_t context,
                                         std::uint32_t max_drafts,
                                         cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      max_drafts_(max_drafts),
      layout_(model_ && model_->HasMtp(), context),
      identity_(std::move(identity)) {
  if (!model_ || identity_.empty() || context > model_->MaxContext())
    throw std::invalid_argument("invalid Ling continuation adapter");
  const auto put = [&](std::uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  // Everything that changes what restored bytes mean, beyond the caller's
  // artifact identity: layout, mode, context, prefill capacity (numeric
  // family rebalancing), verify width and the resident trunk range.
  constexpr std::string_view kSchema = "ling-continuation-adapter-v1";
  identity_.insert(identity_.end(), kSchema.begin(), kSchema.end());
  put(kContinuationLayoutVersion, 4);
  put(layout_.Hash(), 8);
  put(layout_.mtp(), 1);
  put(context, 4);
  put(model_->PrefillCapacity(), 4);
  put(kMaxVerifyTokens, 4);
  put(model_->FirstLayer(), 4);
  put(model_->LayerCount(), 4);
  put(kSnapshotPayloadVersion, 4);
  identity_.insert(identity_.end(), kNumericsId.begin(), kNumericsId.end());
  for (const auto& c : layout_.Components())
    descriptors_.push_back(c.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;

ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Ling adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  if (!state.Ready())
    throw std::logic_error("Ling slot is not ready for execution");
  return *state.session;
}
std::uint32_t ContinuationAdapter::KeptExpansionRows(
    const cache::Slot& slot) const {
  const auto& kept = As(slot).kept;
  return *std::ranges::max_element(kept);
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(context_, &error);
  if (!session)
    throw std::runtime_error("Ling slot session: " + error);
  session->Reset();
  if (model_->HasMtp())
    session->SetMaxDrafts(max_drafts_);
  return std::make_unique<State>(*this, guard, std::move(session));
}

const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid Ling component ID");
  return layout_.Components()[id.value - 1];
}

std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  const auto& session = *state.session;
  const auto& native = state.Native();
  if (!state.IsValid() || state.restoring || session.pending_.active)
    throw std::logic_error("cannot query an invalid or speculative Ling slot");
  const auto position = native.position();
  if (position != 0 &&
      (session.tokens_.size() != position || !session.valid_ ||
       session.reset_pending_ || session.logits_.size() != kVocab ||
       (layout_.mtp() && !native.mtp_valid())))
    throw std::logic_error("Ling slot holds no coherent frontier");
  // Every row below the frontier is intact; raise the borrowable bound.
  state.intact_mla = std::max(state.intact_mla.load(), position);
  if (position > 0)
    state.intact_mtp = std::max(state.intact_mtp.load(), position - 1);
  return layout_.Positions(position);
}

std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  (void)As(slot);
  if (first > end || end > context_)
    throw std::invalid_argument("invalid Ling prefill plan");
  return PlannedPrefillPasses(first, end, model_->PrefillCapacity());
}

std::byte* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  auto& st = state.Native().state_;
  switch (component.part) {
    case ContinuationPart::kTokens:
      return reinterpret_cast<std::byte*>(state.session->tokens_.data());
    case ContinuationPart::kMla:
      return static_cast<std::byte*>(st.mla_cache) +
             component.layer * st.cache.LayerStride() * sizeof(std::uint16_t);
    case ContinuationPart::kMtpRows:
      return static_cast<std::byte*>(st.mtp_cache);
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

std::vector<ContinuationAdapter::Piece> ContinuationAdapter::PrivatePieces(
    State& state, const ContinuationComponent& component) const {
  auto& st = state.Native().state_;
  const auto bytes = component.descriptor.state_bytes;
  switch (component.part) {
    case ContinuationPart::kKdaState:
      return {
          {reinterpret_cast<std::byte*>(
               st.kda_state + std::size_t{component.layer} * kKdaStateFloats),
           0, bytes}};
    case ContinuationPart::kKdaConv:
      return {{reinterpret_cast<std::byte*>(st.kda_conv), 0, bytes}};
    case ContinuationPart::kMtpHidden:
      return {{reinterpret_cast<std::byte*>(st.mtp_h), 0, bytes}};
    case ContinuationPart::kLogits:
      return {{reinterpret_cast<std::byte*>(state.session->logits_.data()), 0,
               bytes, true}};
    case ContinuationPart::kMetadata:
      return {{state.metadata.data(), 0, bytes, true}};
    default:
      throw std::invalid_argument("component is not private state");
  }
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  state.Submitted();
  try {
    std::string error;
    std::exception_ptr failure;
    // Reads wait for the session's last forward; every submission stays
    // outside a decode-graph capture of the executor stream.
    if (!model_->executor_->WithSessionIdle(
            state.Native(),
            [&] {
              try {
                submit(hip_stream);
              } catch (...) {
                failure = std::current_exception();
              }
            },
            &error))
      throw std::runtime_error("Ling cache transfer: " + error);
    if (failure)
      std::rethrow_exception(failure);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    {
      const std::lock_guard lock(job->mutex);
      job->result = TransferResult::kFailed;
    }
    state.Settle();
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  auto& session = *state.session;
  if (!state.Ready() || session.pending_.active)
    throw std::logic_error("Ling slot is not ready for a restore");
  // No device work once a HIP call failed: the restore falls back to a cold
  // prefill (or the server reports the loss) instead of waiting on it.
  RefuseSuspectDevice();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Ling restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Ling frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  const auto target = ordered.front().valid_rows;  // tokens
  if (target > context_ ||
      ordered != layout_.Positions(static_cast<std::uint32_t>(target)))
    throw std::invalid_argument("inconsistent Ling component frontiers");
  // Host capacity first: a refusal leaves execution valid. Then preserve
  // every borrower while the old rows are still readable.
  session.tokens_.reserve(target);
  session.logits_.reserve(kVocab);
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  auto& native = state.Native();
  // Expansion watermarks of the rows the slot holds now; the compare-then-
  // copy loads keep the expansion of rows the restore leaves identical.
  for (std::uint32_t m = 0; m < kMlaLayers; ++m) {
    state.watermark[m] = native.state_.mla_expanded_to[m];
    state.fallback[m] = kNone;
    state.kept[m] = 0;
  }
  std::string error;
  if (!model_->executor_->WithSessionIdle(
          native,
          [&] {
            Check(hipMemsetAsync(state.mismatch, 0xff,
                                 sizeof(std::uint32_t) * kMlaLayers,
                                 state.control));
            Check(hipStreamSynchronize(state.control));
          },
          &error))
    throw std::runtime_error("Ling restore preparation: " + error);
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
  state.intact_mla = 0;
  state.intact_mtp = 0;
  session.valid_ = false;
  session.pending_ = {};
  session.lookup_.Clear();
  session.draft_position_ = kNone;
  session.tokens_.resize(target);
  session.logits_.resize(kVocab);
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Ling private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  RefuseSuspectDevice();
  (void)Positions(slot);  // coherence checks
  const auto position = state.Native().position();
  if (position == 0)
    throw std::logic_error("cannot capture an empty Ling slot");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Ling private piece");
  std::array<std::byte, kContinuationMetadataBytes> image{};
  auto pieces = PrivatePieces(state, component);
  if (component.part == ContinuationPart::kMetadata) {
    const Metadata metadata{.position = position,
                            .mtp = layout_.mtp(),
                            .mtp_valid = state.Native().mtp_valid(),
                            .layout_hash = layout_.Hash(),
                            .context = context_,
                            .prefill_capacity = model_->PrefillCapacity()};
    std::memcpy(image.data(), &metadata, sizeof(metadata));
    pieces = {{image.data(), 0, image.size(), true}};
  }
  return Transfer(state, false, stream, [&](hip::TransferStream& transfer) {
    for (const auto& piece : pieces) {
      const auto first = std::max(offset, piece.offset);
      const auto end =
          std::min(offset + bytes.size(), piece.offset + piece.bytes);
      if (first >= end)
        continue;
      auto* source = piece.pointer + (first - piece.offset);
      auto* destination = bytes.data() + (first - offset);
      if (piece.host)
        std::memcpy(destination, source, end - first);
      else
        Check(rocm::CopySnapshotToHost(destination, source, end - first,
                                       transfer.Native()));
    }
  });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Ling private buffer size");
  } catch (...) {
    state.failed = true;
    throw;
  }
  return LoadPrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Ling load requires an unfailed restore");
  try {
    RefuseSuspectDevice();
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Ling private piece");
    const auto pieces = PrivatePieces(state, component);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(state, true, stream, [&](hip::TransferStream& transfer) {
      for (const auto& piece : pieces) {
        const auto first = std::max(offset, piece.offset);
        const auto end =
            std::min(offset + bytes.size(), piece.offset + piece.bytes);
        if (first >= end)
          continue;
        auto* destination = piece.pointer + (first - piece.offset);
        const auto* source = bytes.data() + (first - offset);
        if (piece.host)
          std::memcpy(destination, source, end - first);
        else
          Check(hipMemcpyAsync(destination, source, end - first,
                               hipMemcpyDefault, transfer.Native()));
      }
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Ling slot");
  RefuseSuspectDevice();
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > state.Intact(component.part) ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Ling row capture");
  auto* source =
      bytes.empty() ? nullptr : RowData(state, component) + first * row;
  return Transfer(state, false, stream, [&](hip::TransferStream& transfer) {
    if (bytes.empty())
      return;
    if (component.part == ContinuationPart::kTokens)
      std::memcpy(bytes.data(), source, bytes.size());
    else
      Check(rocm::CopySnapshotToHost(bytes.data(), source, bytes.size(),
                                     transfer.Native()));
  });
}

std::span<const std::byte> ContinuationAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  const auto& state = As(slot);
  const auto& component = Component(id);
  // The host token vector moves when execution appends; device latent rows
  // stay fixed for the session and are guarded before any overwrite.
  if (component.part == ContinuationPart::kTokens ||
      rocm::Executor::DeviceSuspect())
    return {};
  if (!state.IsValid() ||
      component.descriptor.kind != cache::ComponentKind::kAppendRows)
    throw std::logic_error("cannot borrow invalid Ling rows");
  const auto row = component.descriptor.row_bytes;
  if (first >= end || end > state.Intact(component.part))
    throw std::invalid_argument("invalid Ling borrowed range");
  return {RowData(const_cast<State&>(state), component) + first * row,
          (end - first) * row};
}

cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Ling load requires an unfailed restore");
  try {
    RefuseSuspectDevice();
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Ling row restore");
    auto* destination =
        bytes.empty() ? nullptr : RowData(state, component) + first * row;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    // MLA rows below the slot's expansion watermark: compare, and rewrite only
    // rows that differ (their expansion is then dropped in Validate).
    const auto m = component.layer;
    const bool mla = component.part == ContinuationPart::kMla;
    const Rows compare_end =
        mla ? std::min<Rows>(end, state.watermark[m]) : first;
    const Rows compare = compare_end > first ? compare_end - first : 0;
    const void* readable = compare && !bytes.empty()
                               ? rocm::DeviceReadable(bytes.data())
                               : nullptr;
    if (compare && !readable) {
      // Pageable source: copy plainly and keep no expansion from here on.
      const std::lock_guard lock(state.jobs_mutex);
      state.fallback[m] =
          std::min(state.fallback[m], static_cast<std::uint32_t>(first));
    }
    return Transfer(state, true, stream, [&](hip::TransferStream& transfer) {
      if (bytes.empty())
        return;
      if (component.part == ContinuationPart::kTokens)
        std::memcpy(destination, bytes.data(), bytes.size());
      else if (readable)
        Check(rocm::CompareCopyRows(destination, readable, end - first, compare,
                                    row, static_cast<std::uint32_t>(first),
                                    state.mismatch + m, transfer.Native()));
      else
        Check(hipMemcpyAsync(destination, bytes.data(), bytes.size(),
                             hipMemcpyDefault, transfer.Native()));
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Ling validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  auto& session = *state.session;
  auto& native = state.Native();
  const auto target = static_cast<std::uint32_t>(state.expected[0].valid_rows);
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  if (metadata.version != 1 || metadata.position != target ||
      metadata.mtp != layout_.mtp() || metadata.layout_hash != layout_.Hash() ||
      metadata.mtp_valid != (layout_.mtp() && target > 0))
    return refuse();
  // Restored bytes may come from disk: refuse tokens outside the vocabulary
  // and non-finite logits before anything executes.
  if (!std::ranges::all_of(session.tokens_,
                           [](std::int32_t t) {
                             return t >= 0 &&
                                    t < static_cast<std::int32_t>(kVocab);
                           }) ||
      (target > 0 && !std::ranges::all_of(session.logits_, [](float v) {
         return std::isfinite(v);
       })))
    return refuse();
  std::array<std::uint32_t, kMlaLayers> mismatch{};
  std::string error;
  if (rocm::Executor::DeviceSuspect())
    return refuse();
  try {
    if (!model_->executor_->WithSessionIdle(
            native,
            [&] {
              Check(hipMemcpyAsync(mismatch.data(), state.mismatch,
                                   sizeof(mismatch), hipMemcpyDeviceToHost,
                                   state.control));
              Check(hipStreamSynchronize(state.control));
            },
            &error))
      return refuse();
  } catch (...) {
    return refuse();
  }
  std::uint32_t same[kMlaLayers]{};
  for (std::uint32_t m = 0; m < kMlaLayers; ++m) {
    same[m] =
        std::min({state.watermark[m], mismatch[m], state.fallback[m], target});
    state.kept[m] = same[m];
  }
  native.set_position(target);
  native.OnStateReplaced(target, same);
  native.set_mtp_valid(layout_.mtp() && target > 0);
  session.pending_ = {};
  session.lookup_.Clear();
  session.draft_position_ = kNone;
  session.ResetDraftPolicy();
  session.valid_ = true;
  session.reset_pending_ = false;  // every recurrent state was loaded
  if (target == 0)
    session.logits_.clear();
  state.intact_mla = target;
  state.intact_mtp = target > 0 ? target - 1 : 0;
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}

bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    resetting = &state;
    state.Release();
    // Host only (no device call after a failure, the device may be lost);
    // the next prefill zeroes the recurrent state. The latent rows stay as
    // they are, so a later restore can keep the expansion of identical rows.
    auto& native = state.Native();
    const bool rows_intact = !state.restoring && !state.failed;
    std::array<std::uint32_t, kMlaLayers> kept{};
    for (std::uint32_t m = 0; m < kMlaLayers; ++m)
      kept[m] = native.state_.mla_expanded_to[m];
    state.session->Invalidate();
    if (rows_intact)
      for (std::uint32_t m = 0; m < kMlaLayers; ++m)
        native.state_.mla_expanded_to[m] = kept[m];
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting)
      resetting->failed = true;
    return false;
  }
}

}  // namespace gufo::models::ling_3_flash
