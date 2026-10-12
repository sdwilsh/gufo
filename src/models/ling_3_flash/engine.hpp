#ifndef GUFO_MODELS_LING_3_FLASH_ENGINE_HPP_
#define GUFO_MODELS_LING_3_FLASH_ENGINE_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/ling_3_flash/config.hpp"
#include "src/models/ling_3_flash/gpu_dump.hpp"
#include "src/models/ling_3_flash/mtp.hpp"
#include "src/models/ling_3_flash/mtp_sampling.hpp"
#include "src/models/ling_3_flash/snapshot_format.hpp"
#include "src/models/ling_3_flash/tokenizer.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::ling_3_flash {

struct ModelWeights;
namespace rocm {
class DeviceModel;
class Executor;
class Session;
struct ForwardTiming;
}  // namespace rocm

/// Prefill chunk (ModelOptions::max_batch) the CLI, the bench and the server
/// pass (llama.cpp's n_ubatch). ModelOptions::max_batch itself defaults to
/// 2048, the capacity library users get; the executables choose 512. The
/// server's `--prefill-chunk` is a separate scheduling budget between decode
/// rounds, not this value. Chunks larger than 8 tokens give bit-identical
/// results whatever their size (see the reproducibility contract below).
inline constexpr std::uint32_t kCliPrefillBatch = 512;

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Tokens per forward pass (device buffers are sized for it); longer
  /// prompts are fed in chunks of this size.
  std::uint32_t max_batch = 2048;
  /// Device bytes, shared by all sessions, for persistent expanded K/V of the
  /// MLA layers (112 KiB per cached token, filled lazily by prefill-family
  /// chunks). Rows past a session's share are expanded per chunk with
  /// identical bits, so this is a speed/memory trade only. 0 disables it.
  std::uint64_t expanded_kv_bytes = std::uint64_t{12} << 30;
  /// Debug aids for GPUs too small for the artifact: upload only blocks
  /// [layer_first, layer_first + layer_limit) (limit 0 = to the end of the
  /// trunk). A truncated trunk has meaningless logits; Forward requires
  /// layer_first == 0, block-isolated runs (Session::RunBlock) work for any
  /// resident block.
  std::uint32_t layer_first = 0;
  std::uint32_t layer_limit = 0;
  /// Replay small-batch decode steps as HIP graphs (default). Tests clear it to
  /// compare with the eager launch sequence; results are bit-identical.
  bool graphs = true;
  /// Upload the MTP block (block 42, about 1.2 GiB) and run speculative
  /// decoding (Session::DecodeStep drafts and verifies; every forward keeps the
  /// session's MTP cache filled). Costs 0.5 GiB of verify rollback scratch and
  /// 1152 bytes per context token and session. Without it nothing changes: the
  /// default path is bit-identical to a build without MTP. See mtp.hpp.
  bool mtp = false;
};

/// Timing of one forward pass (device events), kept per chunk.
struct StepTiming {
  std::uint32_t tokens{0};
  double total_ms{0.0};
  double head_ms{0.0};
  double kda_ms{0.0};    ///< Summed over KDA blocks.
  double mla_ms{0.0};    ///< Summed over MLA blocks.
  double dense_ms{0.0};  ///< Summed over the (resident) blocks 0-1.
  double host_ms{0.0};   ///< Wall clock including copies and dumps.
};

struct FeedOptions {
  /// First row (index into the tokens of this call) whose logits go to
  /// `all_logits`; UINT32_MAX = none. The last row's logits always become
  /// Session::Logits().
  std::uint32_t logits_from{UINT32_MAX};
  /// Receives rows [logits_from, n), `vocab` floats each, back to back.
  std::vector<float>* all_logits{nullptr};
  GpuTensorSink* sink{nullptr};
  std::vector<StepTiming>* timings{nullptr};
  /// Merges per-operation device times (see OpTiming) of the chunks of this
  /// call into the vector; such steps run eagerly. For profiling.
  std::vector<OpTiming>* op_profile{nullptr};
};

class Session;
class SessionSnapshot;

/// Gufo-owned API over the ROCm runtime: one resident model, sessions with
/// independent context state.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// Opens the (first shard of the) GGUF, validates it, streams the weights
  /// to the GPU and allocates the activation scratch. Progress and timings go
  /// to stderr. Weights never pass through host memory beyond the small
  /// pinned staging slots of core::hip::WeightUpload.
  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept {
    return options_.max_batch;
  }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::uint32_t LayerCount() const noexcept;  ///< Resident.
  [[nodiscard]] std::uint32_t FirstLayer() const noexcept;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const LingTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  /// Device bytes of the uploaded weights.
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] std::size_t ScratchBytes() const noexcept;
  [[nodiscard]] static std::size_t SessionBytes(std::uint32_t context,
                                                bool mtp = false) noexcept;
  [[nodiscard]] bool HasMtp() const noexcept { return options_.mtp; }
  /// Head of the MTP draft step: true (default) scores drafts with the Q8_0
  /// copy of the output head built at load, false with the trunk's F16 head.
  /// Only draft acceptance differs; the verify path and every emitted token
  /// are independent of it. For A/B measurements and tests, between requests.
  void SetMtpDraftHeadQ8(bool q8) noexcept;
  [[nodiscard]] bool MtpDraftHeadQ8() const noexcept;
  [[nodiscard]] double LoadSeconds() const noexcept { return load_seconds_; }
  [[nodiscard]] double UploadSeconds() const noexcept;
  /// HIP-graph decode replays so far (tests and the bench check the path ran).
  [[nodiscard]] std::uint64_t GraphReplays() const noexcept;
  /// Decode-graph captures that another thread's runtime call invalidated
  /// (each such step ran eagerly and the capture is retried on the next one).
  [[nodiscard]] std::uint64_t CaptureInvalidations() const noexcept;
  /// Test hook: invalidates the next `count` decode-graph captures (as an
  /// unserialized wait from another thread would); each such step must still
  /// run eagerly with identical bits and leave the executor usable.
  void BreakCapturesForTest(std::uint32_t count) noexcept;
  /// Most sessions one Session::EvaluateBatch pass takes (8, or fewer with a
  /// smaller max_batch).
  [[nodiscard]] std::uint32_t BatchRowLimit() const noexcept;
  /// Test hook: the next `count` Session::EvaluateBatch passes fail after
  /// their launches, at the final wait or (`before_wait`) right after the
  /// trunk launches. Host bookkeeping only: no device fault is injected.
  void FailBatchesForTest(std::uint32_t count,
                          bool before_wait = false) noexcept;
  /// Test hook: the batched draft step after the next `skip` ones fails after
  /// launch (Session::DecodeBatch; bookkeeping only).
  void FailDraftBatchForTest(std::uint32_t skip) noexcept;
  /// Test hooks for the background row copy of Session::SaveSnapshot: while
  /// held, the copy pauses before each staging chunk (the snapshot stays
  /// not ready); the next `count` copies fail after their first chunk.
  void HoldSnapshotRowCopiesForTest(bool hold) noexcept;
  void FailSnapshotRowCopiesForTest(std::uint32_t count) noexcept;

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<LingTokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  /// Draft width of the greedy lanes of batched cycles (Session::DecodeBatch;
  /// sessions of one model are used from one thread at a time).
  MtpCohortController cohort_;
  double load_seconds_{0.0};

  friend class Session;
};

/// Reproducibility contract (numeric families).
///
/// A forward chunk of n tokens runs in one of two numeric families, chosen by
/// n alone:
///   * small-batch (decode) family, n <= 8: two-plane int8 Q8_0 GEMV, decode
///     MoE kernels, and the decode attention kernel launched once per token on
///     that token's own context. Every token is computed exactly as if it were
///     fed alone, so any split of a token run into chunks of 1..8 tokens gives
///     bit-identical logits, recurrent state, conv windows and cache rows.
///   * prefill family, n > 8: F16 WMMA GEMM, prefill MoE and non-absorbed MLA
///     prefill attention (cached latent rows expand per head to F16 K/V,
///     once at first use into a budgeted persistent per-session store, or per
///     call in transient 2048-key blocks beyond the budget; flash attention
///     over 192/128 dims whose per-block partials fold in absolute block
///     order; the absorbed form is used only by the small-batch family). The
///     attention rule depends on n alone, never on the context length, the
///     chunk layout or the store budget, and an expanded row has the same bits
///     wherever it was expanded. Any split of a token run into chunks that are
///     all larger than 8 tokens (for example 39 + 67 or one chunk of 106) is
///     bit-identical. The KDA recurrence of this family is restructured for
///     throughput but stays token-sequential from the carried state with a
///     reduction tree independent of n and of where a call starts, so it adds
///     no split dependence.
/// The two families are different numeric formats by design: they agree to
/// about 1e-3 per layer (logits rel-L2 well under 2e-2 over the whole trunk,
/// top-1 usually the same) but not bit for bit.
///
/// What this means for callers (prompt cache, snapshots, resumed sessions):
///   * The state after a token sequence depends on the tokens and on the family
///     each token was computed in, never on where chunk boundaries fall inside
///     a family. A resumed session reproduces an uninterrupted one exactly iff
///     every token is computed in the same family as in the uninterrupted run.
///   * A snapshot taken after a prefill-family run (more than 8 tokens in each
///     chunk) and continued by n > 8 new tokens reproduces the uninterrupted
///     prefill of the whole sequence exactly; continued by n <= 8 tokens
///     (including token-by-token decode) it reproduces the uninterrupted run
///     that processed those tokens in the small-batch family (a prompt prefill
///     followed by decode), which is what generation does anyway.
///   * A cached prefix is therefore reusable for exact reproduction only when
///     the tokens that follow it are processed in the family they would have
///     had without the cache: extending a cached prefix with a suffix of 1..8
///     tokens computes that suffix in the small-batch family, so it does not
///     match a cold prefill of the same prompt. A caller that needs a cold
///     prefill's exact bits must either feed more than 8 tokens at once (back
///     the snapshot up so the replayed tail exceeds 8 tokens) or accept the
///     cross-family difference above. A prefix computed with a chunk of 1..8
///     tokens (for example a 5-token system prompt) belongs to the
///     small-batch family.
///   * Feed() never produces a chunk of 1..8 tokens from a longer call: when
///     the max_batch split would leave such a tail, the last two chunks are
///     rebalanced, so a call of more than 8 tokens is wholly prefill-family
///     whatever max_batch is. The chunk layout inside the family is irrelevant.
///   * Logits rows: the last token's logits (Session::Logits) use the F16 GEMV
///     when at most 8 rows reach the output head in one group and the F16 GEMM
///     otherwise. They are single-row by default; a FeedOptions::logits_from
///     that collects more than 8 rows moves the final row (and the all_logits
///     rows) to the GEMM format, so use the same logits_from class when
///     comparing logits.
///   * Snapshots restore state bit for bit (recurrent state, conv windows and
///     every MLA cache row, into any session, not only the origin); the
///     contract concerns only how later tokens are chunked.
///
/// What the server prompt cache (RAM, disk, learned shared prefixes) therefore
/// guarantees:
///   * Restore is exact: a restored session is bit-identical to the session
///     the snapshot was taken from, whatever it held before.
///   * A request answered from a snapshot at position p reproduces the cold
///     (uncached) run of the same prompt exactly iff every token after p is
///     computed in the family it has in the cold run. The server therefore
///     keeps every checkpoint it plans inside a prompt (grid, shared-prefix,
///     learned and stable-prefix boundaries) at least 9 tokens before the
///     prompt end (TextRunnerCapabilities::min_checkpoint_tail_tokens), and
///     its prefill never leaves a 1..8-token tail before a checkpoint when it
///     can rebalance (snapshot_format.hpp PrefillChunkTokens, the Feed rule).
///     A prompt's own state (and its checkpoints) are then prefill-family
///     states, like a cold prefill's. Every pair of consecutive cut points
///     (reused frontier, snapshot and stable positions, learned boundaries,
///     grid and shared checkpoints, prompt end) is kept at least 9 tokens
///     apart; a point that would be closer is dropped. Residual differences,
///     all inherent: a prompt (or the rest of one) shorter than 9 tokens past
///     a restored frontier runs in the small-batch family, decoded tokens
///     always do, and a client grant below two tails (--prefill-chunk) is
///     raised to two tails, so very small values do not shrink chunks below 18.
///   * Cached frontiers computed in different families are different numeric
///     states of the same text (agreement about 1e-3 per layer, not bitwise);
///     the disk-cache identity carries kSnapshotPayloadVersion so a change of
///     numerics invalidates old files.
/// Verified bit-exact over all 42 blocks (block output, KDA state/conv, MLA
/// cache rows) by `layer_check --invariance` and end to end by session_test.
///
/// Batch invariance (Session::EvaluateBatch, rocm::Executor::ForwardBatch).
/// The family rule holds per session: a batched decode step gives each
/// session a segment of 1..8 tokens computed in the small-batch family,
/// whatever else shares the forward pass. A session's committed tokens,
/// logits, KDA state, conv windows, MLA and MTP cache rows and MTP carry after
/// a batched step are bit-identical to the same step run alone, whatever the
/// number of sessions, their order, depths or segment widths, and whether
/// peers were refused. The decode-family kernels compute every row
/// independently of the rows beside it (weights are shared, arithmetic is
/// per row; routed-expert plans dot each (token, slot) pair on its own), and
/// every per-session part (conv window, recurrence, cache append, attention
/// over the session's own context, MTP cache fill) runs on that session's
/// rows alone. A forward never mixes families. Verified by
/// ling_3_flash_batch_test and the batched cohorts of decode_digest.
///
/// One conversation's context. Prefill/Sync feed prompts, Evaluate appends one
/// token; the logits of the last token are kept (greedy decoding is
/// `Evaluate(Argmax())`). Sessions of one model share its executor: use them
/// from one thread at a time.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: when the session's tokens are
  /// a prefix of `prompt` only the rest is processed, otherwise the state is
  /// reset and the whole prompt is processed (recurrent state cannot be cut
  /// back to a shared prefix; use snapshots for that).
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Appends `tokens` in chunks of Model::PrefillCapacity().
  [[nodiscard]] bool Feed(std::span<const std::int32_t> tokens,
                          const FeedOptions& options = {},
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);

  /// One lane of EvaluateBatch: `token` is appended to `session`.
  struct BatchLane {
    Session* session{nullptr};
    std::int32_t token{0};
    bool ok{false};  ///< Out.
    /// Out: not run for a batch-only reason (no row left, a snapshot transfer
    /// in flight); the session is untouched and can be evaluated alone.
    bool deferred{false};
    std::string error;  ///< Out, when !ok.
  };
  /// Evaluate for several sessions of one model in one forward pass (at most
  /// Model::BatchRowLimit() lanes, one per session). Each lane ends exactly as
  /// its own Evaluate would (batch invariance, above): a lane that Evaluate
  /// would refuse (an invalid session, a pending verify, a full context, a
  /// token outside the vocabulary, another model, a session listed twice)
  /// fails alone, with the same effect on its session, and the others still
  /// run. A lane the batch alone cannot take is `deferred`, untouched. If the
  /// pass fails after launch, every launched lane fails and its session is
  /// emptied without device calls (Invalidate(): the device may be lost).
  static void EvaluateBatch(std::span<BatchLane> lanes);

  /// Logits of the last token; empty when invalid or nothing was fed.
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  /// Index of the largest logit (lowest index on ties); -1 without logits.
  [[nodiscard]] std::int32_t Argmax() const noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  /// Leading cache rows whose persistent expanded K/V is current (the most of
  /// any MLA layer) (tests: how much of a restore had to be re-expanded).
  [[nodiscard]] std::uint32_t ExpandedRowsValid() const noexcept;
  /// Rows of the session's persistent expanded K/V store (whole 2048-row
  /// blocks of the shared --expanded-kv-bytes budget; tests).
  [[nodiscard]] std::uint32_t ExpandedStoreRows() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return tokens_;
  }
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  /// Empties the session now and zeroes its recurrent state on the device.
  void Reset();
  /// Empties the session without any device call; the recurrent state is
  /// zeroed by the next Feed (a restore overwrites it anyway). Safe after a
  /// failed step on a lost GPU, where device cleanup can block forever.
  void Invalidate() noexcept;

  /// Result of one speculative decode step (DecodeStep).
  struct DecodeResult {
    /// Committed tokens, already part of the session (anchor first). Empty
    /// with `stop` when the next token would stop generation.
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  struct DraftStats {
    std::uint64_t cycles{0};    ///< Cycles that verified at least one draft.
    std::uint64_t drafted{0};   ///< Draft tokens verified.
    std::uint64_t accepted{0};  ///< Draft tokens accepted.
  };
  /// One decode step of at most `max_tokens` committed tokens (max_tokens 1
  /// is plain decoding): samples the anchor token from the session's logits
  /// with `sampler`, proposes up to 7 drafts with the MTP block (the count
  /// from the adaptive controller), verifies [anchor, drafts] in one forward
  /// pass and rolls the session back to the accepted prefix (mtp.hpp). Greedy
  /// results equal plain greedy decoding token for token; sampled results
  /// follow the lossless p/q rule of mtp_sampling.hpp. `sampler` receives the
  /// accepted tokens (history, RNG, deferred residual draw). Without MTP
  /// (Model::HasMtp() false or an invalid MTP state) it decodes one token.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  [[nodiscard]] const DraftStats& Statistics() const noexcept { return stats_; }

  /// One lane of DecodeBatch: DecodeStep's arguments and outcome.
  struct BatchDecode {
    Session* session{nullptr};
    std::size_t max_tokens{0};
    sampling::SamplerState* sampler{nullptr};
    bool stop_at_eos{true};
    DecodeResult result;  ///< Out.
    bool ok{false};       ///< Out: as DecodeStep's return value.
    /// Out: not run for a batch-only reason (a snapshot transfer in flight,
    /// no room in the step); nothing was committed and the sampler is as
    /// given to the lane's draw (callers pass a copy and DecodeStep alone).
    bool deferred{false};
    std::string error;  ///< Out, when !ok.
    /// Out: wall time of the device steps this lane took part in (shared
    /// draft and verify steps count in full; waiting for steps of other lanes
    /// does not), plus its host work: what a serial DecodeStep would report
    /// for its own cycle.
    double ms{0.0};
    /// Out: the most lanes that shared one forward with this lane (1 = it
    /// ran alone).
    std::uint32_t width{1};
  };
  /// DecodeStep for several sessions of one model, sharing the draft steps and
  /// the verify forward (at most 8 rows per forward; a cohort that needs more
  /// splits into several verify forwards). Every lane's committed tokens,
  /// logits, state and sampler effects equal its own DecodeStep's (batch
  /// invariance): greedy lanes take a common draft width from the model's
  /// cohort controller (greedy tokens do not depend on it), sampled lanes the
  /// width of their own history (MtpLengthController, prior costs), never cut
  /// to fit a step. Draft counters (Statistics) are outside the contract.
  /// A lane DecodeStep would refuse fails alone; a pass that fails after
  /// launch fails its lanes and empties their sessions without device calls.
  static void DecodeBatch(std::span<BatchDecode> lanes);
  /// Forgets the acceptance history (a new request over retained state).
  void ResetDraftPolicy() noexcept;
  /// Largest number of drafts per step (1..7, default 7).
  void SetMaxDrafts(std::uint32_t drafts) noexcept;
  /// Test hooks (greedy only): a fixed number of drafts per cycle instead of
  /// the controller's choice, and a draft source replacing the MTP chain
  /// (called with the chain so far, anchor first, it returns the next draft
  /// token) so tests can force full, partial and zero acceptance.
  struct DraftHooks {
    std::uint32_t fixed_drafts{0};  ///< 0 = adaptive
    std::function<std::int32_t(std::span<const std::int32_t>)> source;
    /// Test seam, called between Verify() and Rollback() (may throw).
    std::function<void()> after_verify;
    /// Also a stop token for decoding with stop_at_eos (tests: a stop at a
    /// chosen point of a truncated trunk's text); -1 = none.
    std::int32_t stop_token{-1};
  };
  void SetDraftHooks(DraftHooks hooks) { hooks_ = std::move(hooks); }
  /// True when drafting is possible: an MTP model and a consistent MTP state.
  [[nodiscard]] bool MtpActive() const noexcept;

  // MTP building blocks, public for tests and tools (DecodeStep composes them).
  /// Appends `tokens` (2..8) as one verify step that records rollback points.
  /// Fills `argmax` (one entry per token) with the device argmax of every row,
  /// and `rows` (when given) with all rows' logits (`tokens.size()` * vocab).
  /// The session then holds the whole chunk until Rollback(). `op_profile`
  /// (profiling) receives the per-operation device times of the step.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens,
                            std::span<std::int32_t> argmax,
                            std::vector<float>* rows,
                            std::string* error_msg = nullptr,
                            std::vector<OpTiming>* op_profile = nullptr);
  /// Completes a Verify keeping its first `keep` tokens (1..n): the recurrent
  /// state, position, tokens and logits become those after row keep - 1.
  [[nodiscard]] bool Rollback(std::uint32_t keep,
                              std::string* error_msg = nullptr);
  /// One MTP draft step for `token` (the token after the last committed one,
  /// or after the previous draft when `chain`); see rocm::DraftRequest.
  /// `op_profile` (profiling) receives the per-operation device times.
  [[nodiscard]] bool Draft(std::int32_t token, bool chain,
                           std::span<float> logits, std::int32_t* argmax,
                           std::string* error_msg = nullptr,
                           std::vector<OpTiming>* op_profile = nullptr);
  /// Host issue / wall time and device span of the model's last forward step
  /// (Feed chunk, Evaluate, Verify) or draft step (profiling; sessions of one
  /// model share it).
  [[nodiscard]] LaunchTiming LastLaunchTiming() const noexcept;
  /// Test hook: draft block alone on a host hidden row at an explicit MTP
  /// position (no trunk state needed; writes MTP cache row `position`).
  [[nodiscard]] bool DraftWithHidden(
      std::int32_t token, std::span<const float> h, std::uint32_t position,
      std::span<float> logits, std::span<float> h_next, std::int32_t* argmax,
      std::string* error_msg = nullptr);
  [[nodiscard]] bool ReadMtpCache(std::uint32_t rows,
                                  std::vector<std::uint16_t>* out,
                                  std::string* error_msg = nullptr) const;
  /// Output-normed trunk hidden carry (the MTP input of the next draft).
  [[nodiscard]] bool ReadMtpHidden(std::vector<float>* out,
                                   std::string* error_msg = nullptr) const;

  /// Complete, self-contained copy of the context at Position(): KDA state,
  /// conv windows, the MLA latent rows [0, Position()), the tokens and the
  /// last logits, in host memory (snapshot_format.hpp; SnapshotBytes() is its
  /// exact size). It reads only this session's buffers on a private stream, so
  /// it may run on a worker thread while another session executes; this
  /// session must be idle at the call. The recurrent state is copied before
  /// the call returns; the latent rows (the bulk) are copied by a background
  /// thread while this session may already decode (see SessionSnapshot).
  /// Fails on an invalid or empty session.
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  /// Reference for tests: the same payload through Executor::SaveSnapshot
  /// (capture and row copy back to back, no background worker).
  [[nodiscard]] std::unique_ptr<SessionSnapshot>
  SaveSnapshotSynchronouslyForTest(std::string* error_msg = nullptr) const;
  /// Payload size SaveSnapshot would produce now (0 when it would fail).
  [[nodiscard]] std::size_t SnapshotBytes() const noexcept;
  /// Makes this session's context equal to the snapshot's, bit for bit, in any
  /// session whose capacity holds the snapshot (not only the one it came
  /// from): recurrent state, conv windows and cache rows are overwritten, so
  /// no property of the session's previous contents survives. State derived
  /// from the rows or positions is told through rocm::Session::OnStateReplaced
  /// (contract in kernels/rocm/executor.hpp: device-side positions, the WS12
  /// expanded K/V watermark, lock-protected when pooled) and rebuilt from the
  /// rows, never stored in a snapshot.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  /// Same from serialized bytes (the disk-cache payload); validates magic,
  /// version, geometry and exact size before touching the session.
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

  // Debug and test access.
  [[nodiscard]] bool RunBlock(std::uint32_t il, std::uint32_t first_pos,
                              std::span<const float> in, std::span<float> out,
                              GpuTensorSink* sink,
                              std::string* error_msg = nullptr);
  [[nodiscard]] bool ReadKdaState(std::uint32_t il, std::vector<float>* state,
                                  std::vector<float>* conv,
                                  std::string* error_msg = nullptr) const;
  [[nodiscard]] bool ReadMlaCache(std::uint32_t il, std::uint32_t rows,
                                  std::vector<std::uint16_t>* out,
                                  std::string* error_msg = nullptr) const;

private:
  friend class Model;
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshotImpl(
      std::string* error_msg, bool synchronous) const;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  bool valid_{true};
  /// Set by Invalidate(): the device state still holds the old context.
  bool reset_pending_{false};

  // MTP.
  struct PendingVerify {
    bool active{false};
    std::uint32_t first_pos{0};
    std::uint32_t n{0};
    bool have_rows{false};
    /// Where a batched verify left this lane's rollback points, MTP hidden
    /// rows and head rows (rocm::BatchSegment::Placement; 0 for Verify).
    std::uint32_t ckpt0{0};
    std::uint32_t hn0{0};
    std::uint32_t logit_row{0};
  };
  /// A drafted cycle up to its verify (DecodeStep, DecodeBatch).
  struct Cycle {
    std::int32_t anchor{0};
    std::vector<std::int32_t> chain;
    std::array<std::int32_t, kMaxVerifyTokens> argmax{};
    bool sampled{false};
    bool greedy_fast{false};
    bool stop_at_eos{true};
    std::optional<MtpProposer> proposer;
  };
  /// After a verify of `c.chain`: acceptance with `sampler`, Rollback, the
  /// result, statistics and the length controller (ObserveCost from `begin`
  /// when given: serial cycles only).
  [[nodiscard]] bool IsStop(std::int32_t token) const noexcept;
  bool FinishCycle(Cycle& c, sampling::SamplerState& sampler,
                   DecodeResult* result, std::string* error_msg,
                   const std::chrono::steady_clock::time_point* begin);
  PendingVerify pending_;
  std::vector<float> verify_rows_;
  std::vector<float> draft_logits_;
  /// Position of the last draft step (UINT32_MAX: none since Invalidate).
  std::uint32_t draft_position_{UINT32_MAX};
  MtpLengthController draft_length_;
  DraftStats stats_;
  DraftHooks hooks_;
};

/// Host-held snapshot payload of one session (snapshot_format.hpp).
///
/// Session::SaveSnapshot copies the state a decode step mutates before it
/// returns and fills the append-only latent rows in the background, beside the
/// session's own decode (the rows below Position() are never rewritten while
/// the copy runs; a reset, restore or destruction of the session waits for it).
/// The payload is complete, and immutable, once Wait() returned true.
/// Position() and SizeBytes() are known immediately; bytes() is the payload
/// and waits for the copy. Destroying the snapshot aborts its copy without
/// waiting for it: the worker keeps the payload buffer alive until it stops (at
/// its next chunk).
class SessionSnapshot final {
public:
  ~SessionSnapshot();
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  [[nodiscard]] std::size_t SizeBytes() const noexcept { return size_; }
  /// True once the background copy finished, successfully or not.
  [[nodiscard]] bool Ready() const noexcept;
  /// True once the background copy failed (the payload is then unusable).
  [[nodiscard]] bool Failed() const noexcept;
  /// Waits for the background copy. False (with the reason in `error_msg`)
  /// when it failed.
  [[nodiscard]] bool Wait(std::string* error_msg = nullptr) const;
  /// The serialized payload (RAM entry and disk file are these same bytes),
  /// once complete; empty when the copy failed. Waits for the copy.
  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept;

private:
  struct AsyncCopy;
  SessionSnapshot(std::uint32_t position, std::size_t size);
  /// Runs `copy` (returns false with a reason on failure; stops early when
  /// the flag is set) on a worker, then `release`. Inline when no thread can
  /// be started.
  void StartCopy(
      std::function<bool(const std::atomic<bool>&, std::string*)> copy,
      std::function<void()> release);
  std::uint8_t* data_{nullptr};  ///< Owned by `copy_`, which a worker shares.
  std::size_t size_{0};
  std::uint32_t position_{0};
  std::shared_ptr<AsyncCopy> copy_;
  friend class Session;
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_ENGINE_HPP_
