#ifndef GUFO_MODELS_LING_3_FLASH_SNAPSHOT_FORMAT_HPP_
#define GUFO_MODELS_LING_3_FLASH_SNAPSHOT_FORMAT_HPP_

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "ling_3_flash_numerics_id.hpp"
#include "src/models/ling_3_flash/layout.hpp"

namespace gufo::models::ling_3_flash {

static_assert(std::endian::native == std::endian::little,
              "snapshot payloads are little-endian by memcpy");

/// Self-contained, host-resident session snapshot (RAM cache payload and disk
/// payload are the same bytes). Little-endian, all sections packed back to
/// back after the header:
///
///   header                      kSnapshotHeaderBytes
///   tokens   int32  [position]
///   logits   float  [vocab]     logits of the last token (decode can resume)
///   KDA state float [35][kKdaStateFloats]
///   KDA conv  float [35][kKdaConvStateFloats]
///   MLA rows  uint16 [7 mla layers][position][kMlaCacheWidth]
///             (the live F16 latent rows 0..position-1, packed: no capacity
///             padding, so a snapshot restores into a session of any context
///             size >= position)
///
/// The WS12 expanded K/V cache is a pure function of the MLA rows and is never
/// stored; restore invalidates it (rocm::Session::InvalidateDerived).
inline constexpr std::array<char, 8> kSnapshotMagic = {'L', 'I', 'N', 'G',
                                                       '3', 'S', 'N', 'P'};
/// Bump on any change to the byte layout or to the numerics of what the bytes
/// mean (state or cache row format). Part of the disk-cache identity.
inline constexpr std::uint32_t kSnapshotPayloadVersion = 1;

/// Identity of the compiled numerics (kernel sources, HIP compiler, flags,
/// architectures, ROCm release, prefill batch), derived by CMake. Part of the
/// disk-cache identity; kSnapshotPayloadVersion covers only the byte layout.
inline constexpr std::string_view kNumericsId = GUFO_LING_NUMERICS_SHA256;

struct SnapshotHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t vocab;
  std::uint32_t position;  ///< Tokens (and MLA rows) in the snapshot.
  std::uint32_t kda_layers;
  std::uint32_t mla_layers;
  std::uint32_t mla_width;
  std::uint32_t reserved;
  std::uint64_t kda_state_floats;  ///< Per layer.
  std::uint64_t kda_conv_floats;   ///< Per layer.
  std::uint64_t payload_bytes;     ///< Whole payload including the header.
};
inline constexpr std::size_t kSnapshotHeaderBytes = sizeof(SnapshotHeader);
static_assert(kSnapshotHeaderBytes == 64);

struct SnapshotSections {
  std::size_t tokens, logits, kda_state, kda_conv, mla, total;
};

/// Byte offsets of the sections for `position` tokens.
[[nodiscard]] constexpr SnapshotSections SnapshotLayout(
    std::uint32_t position) noexcept {
  SnapshotSections s{};
  s.tokens = kSnapshotHeaderBytes;
  s.logits = s.tokens + std::size_t{position} * sizeof(std::int32_t);
  s.kda_state = s.logits + std::size_t{kVocab} * sizeof(float);
  s.kda_conv = s.kda_state + kKdaLayers * kKdaStateFloats * sizeof(float);
  s.mla = s.kda_conv + kKdaLayers * kKdaConvStateFloats * sizeof(float);
  s.total = s.mla + std::size_t{kMlaLayers} * position * kMlaCacheWidth *
                        sizeof(std::uint16_t);
  return s;
}

/// Exact payload size of a snapshot taken at `position`: about 75 MiB of
/// recurrent state, 0.6 MiB of logits and 8068 bytes per token.
[[nodiscard]] constexpr std::size_t SnapshotPayloadBytes(
    std::uint32_t position) noexcept {
  return SnapshotLayout(position).total;
}

/// Largest chunk of tokens of the small-batch numeric family (engine.hpp).
inline constexpr std::size_t kSmallBatchMaxTokens = 8;

/// Tokens a prefill call should consume next so that checkpoints keep their
/// numeric family. `left` is the number of prompt tokens up to the next cache
/// boundary (checkpoint, learned prefix end or prompt end), `grant` the most
/// the scheduler allows now and `capacity` the model's chunk limit. The result
/// is min(left, grant, capacity), except that it never leaves a tail of 1..8
/// tokens before the boundary when the chunk can give tokens up (the same
/// rebalance as Session::Feed): such a tail would be computed in the
/// small-batch family, which differs from the prefill family an uncached run
/// of the same prompt uses. A boundary less than 9 tokens away, or a grant
/// that cannot be rebalanced, keeps the plain minimum.
[[nodiscard]] constexpr std::size_t PrefillChunkTokens(
    std::size_t left, std::size_t grant, std::size_t capacity) noexcept {
  std::size_t chunk = left;
  if (grant < chunk) {
    chunk = grant;
  }
  if (capacity < chunk) {
    chunk = capacity;
  }
  const std::size_t tail = left - chunk;
  if (tail >= 1 && tail <= kSmallBatchMaxTokens &&
      chunk > kSmallBatchMaxTokens + 1) {
    const std::size_t want = kSmallBatchMaxTokens + 1 - tail;
    const std::size_t give = chunk - kSmallBatchMaxTokens - 1;
    chunk -= want < give ? want : give;
  }
  return chunk;
}

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_SNAPSHOT_FORMAT_HPP_
