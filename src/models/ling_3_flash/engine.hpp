#ifndef GUFO_MODELS_LING_3_FLASH_ENGINE_HPP_
#define GUFO_MODELS_LING_3_FLASH_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/ling_3_flash/config.hpp"
#include "src/models/ling_3_flash/gpu_dump.hpp"
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
class RecurrentSnapshot;
struct ForwardTiming;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Tokens per forward pass; longer prompts are fed in chunks of this size.
  std::uint32_t max_batch = 2048;
  /// Debug aids for GPUs too small for the artifact: upload only blocks
  /// [layer_first, layer_first + layer_limit) (limit 0 = to the end of the
  /// trunk). A truncated trunk has meaningless logits; Forward requires
  /// layer_first == 0, block-isolated runs (Session::RunBlock) work for any
  /// resident block.
  std::uint32_t layer_first = 0;
  std::uint32_t layer_limit = 0;
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
  [[nodiscard]] static std::size_t SessionBytes(std::uint32_t context) noexcept;
  [[nodiscard]] double LoadSeconds() const noexcept { return load_seconds_; }
  [[nodiscard]] double UploadSeconds() const noexcept;

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<LingTokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
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
///   * prefill family, n > 8: F16 WMMA GEMM, prefill MoE and prefill
///     attention. Any split of a token run into chunks that are all larger than
///     8 tokens (for example 39 + 67 or one chunk of 106) is bit-identical.
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
///   * Snapshots restore state bit for bit; the contract concerns only how
///     later tokens are chunked.
/// Verified bit-exact over all 42 blocks (block output, KDA state/conv, MLA
/// cache rows) by `layer_check --invariance` and end to end by session_test.
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

  /// Logits of the last token; empty when invalid or nothing was fed.
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  /// Index of the largest logit (lowest index on ties); -1 without logits.
  [[nodiscard]] std::int32_t Argmax() const noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return tokens_;
  }
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  void Reset();

  /// Rollback point: the recurrent state (device copy), position, tokens and
  /// last logits. The MLA cache is not copied; rows past the position are
  /// rewritten before use.
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  /// Restores a snapshot of this session. The session's tokens must still
  /// start with the snapshot's tokens (the cache rows below the snapshot
  /// position are then intact); otherwise it fails without changing state.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
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
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  bool valid_{true};
};

/// Host-held handle to a rollback point of one session.
class SessionSnapshot final {
public:
  ~SessionSnapshot();
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }
  /// Device plus host bytes.
  [[nodiscard]] std::size_t SizeBytes() const noexcept;

private:
  SessionSnapshot();
  std::unique_ptr<rocm::RecurrentSnapshot> recurrent_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  friend class Session;
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_ENGINE_HPP_
