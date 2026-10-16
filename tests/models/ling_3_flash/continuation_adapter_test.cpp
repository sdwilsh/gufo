// Model-local checks of the Ling-3.0-Flash component-cache adapter on a real
// artifact. A truncated trunk is enough: --layers N with N >= 6 keeps one MLA
// layer resident. The test runs AR, or MTP with --mtp. Exit 77 without
// --model.
//
//   round_trip        every component byte-for-byte after a cross-slot restore
//                     (odd pieces, reversed order), next-step logits exact
//   continuation      restored + 40-token suffix equals the uninterrupted
//                     session: logits, greedy and seeded tokens, MTP drafts
//   edit_before       a shorter boundary restored into a slot holding a longer
//                     different history continues exactly
//   keep_expansion    a same-slot restore keeps the expansion of identical
//                     rows (watermark > 0) and a > 8-token suffix stays exact
//   guards            borrowed rows are preserved before every overwrite path:
//                     engine Invalidate then release, a forward at 0 after an
//                     engine Invalidate, BeginRestore; decode never guards
//   batch_lanes       DecodeBatch/EvaluateBatch: a lane whose slot is not
//                     ready fails alone, a lane whose guard refuses (as a
//                     cancelled lease) decodes, peers equal their solo runs
//   graph_after_restore graph-replayed decode after a restore equals the
//                     source's
//   latching          misuse throws, failed loads latch until Invalidate,
//                     the device-suspect latch retires borrowers without
//                     device work
//
//   ling_3_flash_continuation_adapter_test --model FIRST_SHARD.gguf
//       [--layers N (12)] [--mtp] [--max-context N (4096)]
#include "src/models/ling_3_flash/continuation_adapter.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/hip/transfer_pool.hpp"
#include "src/core/sampling.hpp"
#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace cache = gufo::cache;
namespace sampling = gufo::sampling;

namespace {

void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& call, const std::string& what) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "accepted invalid operation: " + what);
}
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}

std::vector<std::int32_t> Tokens(std::size_t count, std::uint32_t seed) {
  std::vector<std::int32_t> tokens(count);
  std::uint32_t x = seed * 2654435761U + 1;
  for (auto& t : tokens) {
    x = x * 1664525U + 1013904223U;
    t = static_cast<std::int32_t>(1000 + (x >> 8) % 120000);
  }
  return tokens;
}

struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};

constexpr std::size_t kPiece = (1 << 20) + 13;
constexpr cache::Rows kRowPiece = 97;

Checkpoint Capture(l3::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams, bool registered = true) {
  Checkpoint result{adapter.Positions(slot), {}};
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    void* mapped{};
    if (registered && !bytes.empty())
      Require(hipHostMalloc(&mapped, bytes.size(), hipHostMallocDefault) ==
                  hipSuccess,
              "registered capture allocation failed");
    const std::unique_ptr<void, decltype(&hipHostFree)> mapping(mapped,
                                                                hipHostFree);
    const auto buffer =
        mapped ? std::span(static_cast<std::byte*>(mapped), bytes.size())
               : std::span(bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + kRowPiece);
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            buffer.subspan(first * component.row_bytes,
                           (end - first) * component.row_bytes),
            *stream));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(kPiece, bytes.size() - offset);
        auto stream = streams.TryAcquire();
        Require(bool(stream), "stream exhausted");
        Wait(adapter.CapturePrivatePiece(slot, component.id, offset,
                                         buffer.subspan(offset, count),
                                         *stream));
        offset += count;
      }
    }
    if (mapped)
      std::memcpy(bytes.data(), mapped, bytes.size());
  }
  return result;
}

// Reverse component and piece order, on independent streams. `pinned` loads
// from registered host memory (the compare-then-copy path), else pageable.
void Load(l3::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams,
          bool pinned = true) {
  adapter.BeginRestore(slot, checkpoint.positions);
  Require(!slot.IsValid(), "BeginRestore exposed executable state");
  Reject([&] { (void)adapter.GetSession(slot); }, "execution during restore");
  std::vector<void*> registered;
  std::vector<cache::Completion> pending;
  const auto settle = [&] {
    for (auto& completion : pending)
      Wait(std::move(completion));
    pending.clear();
  };
  const auto acquire = [&] {
    auto stream = streams.TryAcquire();
    if (!stream) {
      settle();
      stream = streams.TryAcquire();
    }
    Require(bool(stream), "stream exhausted");
    return stream;
  };
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    std::span<const std::byte> bytes(checkpoint.bytes[i]);
    if (pinned && !bytes.empty()) {
      void* mapped{};
      Require(hipHostMalloc(&mapped, bytes.size(), hipHostMallocDefault) ==
                  hipSuccess,
              "registered load allocation failed");
      std::memcpy(mapped, bytes.data(), bytes.size());
      registered.push_back(mapped);
      bytes = {static_cast<const std::byte*>(mapped), bytes.size()};
    }
    if (component.row_bytes) {
      const auto rows = checkpoint.positions[i].valid_rows;
      for (cache::Rows end = rows; end;) {
        const auto first = end > kRowPiece ? end - kRowPiece : 0;
        auto stream = acquire();
        pending.push_back(adapter.CopyRowsIn(
            slot, component.id, first, end,
            bytes.subspan(first * component.row_bytes,
                          (end - first) * component.row_bytes),
            *stream));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > kPiece ? end - kPiece : 0;
        auto stream = acquire();
        pending.push_back(adapter.LoadPrivatePiece(
            slot, component.id, first, bytes.subspan(first, end - first),
            *stream));
        end = first;
      }
    }
  }
  settle();
  for (void* p : registered)
    (void)hipHostFree(p);
  Require(adapter.Validate(slot, checkpoint.positions), "validation failed");
}

// Borrows one checkpoint's rows the way the cache does: before any overwrite
// or release of a borrowed range, copy it out and check it is unchanged.
struct Guard final : cache::MutationGuard {
  l3::ContinuationAdapter* adapter{};
  cache::Slot* slot{};
  gufo::hip::TransferPool* streams{};
  const Checkpoint* borrowed{};
  std::vector<bool> detached;
  std::size_t overwrites{}, preserved{}, releases{}, resets{};
  bool release_failed{false};
  bool refuse{false};  // as a cancelled lease (CheckStop)
  void Borrow(const Checkpoint& checkpoint) {
    borrowed = &checkpoint;
    detached.assign(checkpoint.bytes.size(), false);
  }
  void Preserve(cache::ComponentId id, cache::Rows first, cache::Rows end) {
    if (!borrowed || detached[id.value - 1])
      return;
    const auto& component = adapter->Components()[id.value - 1];
    const auto rows = borrowed->positions[id.value - 1].valid_rows;
    if (!component.row_bytes || !rows || first >= rows || end <= first ||
        component.id.value == 1)  // tokens are copied, never borrowed
      return;
    std::vector<std::byte> bytes(rows * component.row_bytes);
    auto stream = streams->TryAcquire();
    Require(bool(stream), "guard stream exhausted");
    Wait(adapter->CopyRowsOut(*slot, id, 0, rows, bytes, *stream));
    Require(bytes == borrowed->bytes[id.value - 1],
            "borrowed rows overwritten before preservation");
    detached[id.value - 1] = true;
    ++preserved;
  }
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first,
                       cache::Rows end) override {
    ++overwrites;
    if (refuse)
      throw std::runtime_error("slot lease cancelled");
    Preserve(id, first, end);
  }
  void BeforeRelease(cache::ComponentId id, cache::Rows first,
                     cache::Rows end) noexcept override {
    ++releases;
    try {
      Preserve(id, first, end);
    } catch (...) {
      release_failed = true;  // the cache would retire the borrower
    }
  }
  void AfterReset() noexcept override { ++resets; }
  [[nodiscard]] bool AllPreserved() const {
    for (std::size_t i = 0; i < detached.size(); ++i) {
      const auto& c = adapter->Components()[i];
      if (c.row_bytes && c.id.value != 1 && borrowed->positions[i].valid_rows &&
          !detached[i])
        return false;
    }
    return true;
  }
};

struct Slot {
  Guard guard;
  std::unique_ptr<cache::Slot> slot;
  Slot(l3::ContinuationAdapter& adapter, gufo::hip::TransferPool& streams) {
    guard.adapter = &adapter;
    guard.streams = &streams;
    slot = adapter.CreateSlot(guard);
    guard.slot = slot.get();
  }
};

void Feed(l3::Session& session, std::span<const std::int32_t> tokens) {
  std::string error;
  Require(session.Feed(tokens, {}, &error), "feed: " + error);
}

void SameLogits(const l3::Session& a, const l3::Session& b,
                const std::string& what) {
  Require(a.Logits().size() == b.Logits().size() && !a.Logits().empty() &&
              std::memcmp(a.Logits().data(), b.Logits().data(),
                          a.Logits().size_bytes()) == 0,
          what + ": logits differ");
}

struct Decoded {
  std::vector<std::int32_t> tokens;
  l3::Session::DraftStats stats;
};
Decoded Decode(l3::Session& session, std::size_t count, bool sampled) {
  const auto history = session.Tokens();
  std::vector<sampling::TokenId> initial(history.begin(), history.end());
  sampling::SamplerState sampler({.temperature = sampled ? 0.8F : 0.0F,
                                  .top_k = 20,
                                  .top_p = 0.95F,
                                  .seed = 73},
                                 initial);
  Decoded out;
  const auto before = session.Statistics();
  std::string error;
  while (out.tokens.size() < count) {
    l3::Session::DecodeResult result;
    Require(session.DecodeStep(count - out.tokens.size(), sampler, &result,
                               &error, false),
            "decode: " + error);
    out.tokens.insert(out.tokens.end(), result.tokens.begin(),
                      result.tokens.end());
  }
  out.stats = session.Statistics();
  out.stats.cycles -= before.cycles;
  out.stats.drafted -= before.drafted;
  out.stats.accepted -= before.accepted;
  return out;
}
void SameDecode(const Decoded& a, const Decoded& b, const std::string& what) {
  Require(a.tokens == b.tokens, what + ": decoded tokens differ");
  Require(a.stats.drafted == b.stats.drafted &&
              a.stats.accepted == b.stats.accepted,
          what + ": drafts differ");
}

struct Fixture {
  std::shared_ptr<l3::Model> model;
  std::unique_ptr<l3::ContinuationAdapter> adapter;
  gufo::hip::TransferPool streams{16};
  bool mtp{false};
};

void Report(const char* name) {
  std::fprintf(stderr, "[adapter] %s: pass\n", name);
}

void RoundTrip(Fixture& f) {
  auto& adapter = *f.adapter;
  Slot a(adapter, f.streams), b(adapter, f.streams);
  const auto prompt = Tokens(1300, 1);
  Feed(adapter.GetSession(*a.slot), prompt);
  const auto checkpoint = Capture(adapter, *a.slot, f.streams);
  // The frontiers match the layout's positions.
  Require(checkpoint.positions == adapter.layout().Positions(1300),
          "unexpected frontiers");
  // A pageable source skips compare-then-copy and must still be exact.
  Load(adapter, *b.slot, checkpoint, f.streams, false);
  const auto again = Capture(adapter, *b.slot, f.streams);
  Require(again.positions == checkpoint.positions, "restored frontiers differ");
  for (std::size_t i = 0; i < again.bytes.size(); ++i)
    Require(again.bytes[i] == checkpoint.bytes[i],
            "component " + std::to_string(i + 1) + " differs after restore");
  SameLogits(adapter.GetSession(*a.slot), adapter.GetSession(*b.slot),
             "round trip");
  Require(b.guard.resets == 0, "restore must not reset");
  Report("round_trip");
}

void Continuation(Fixture& f) {
  auto& adapter = *f.adapter;
  Slot a(adapter, f.streams), b(adapter, f.streams);
  const auto prompt = Tokens(1300, 2);
  const auto suffix = Tokens(40, 3);
  for (const bool sampled : {false, true}) {
    Require(adapter.Invalidate(*a.slot) && adapter.Invalidate(*b.slot),
            "invalidate");
    auto& source = adapter.GetSession(*a.slot);
    Feed(source, prompt);
    const auto checkpoint = Capture(adapter, *a.slot, f.streams);
    Load(adapter, *b.slot, checkpoint, f.streams);
    auto& restored = adapter.GetSession(*b.slot);
    Feed(source, suffix);
    Feed(restored, suffix);
    SameLogits(source, restored, "restored suffix");
    source.ResetDraftPolicy();
    const auto d1 = Decode(source, 24, sampled);
    const auto d2 = Decode(restored, 24, sampled);
    SameDecode(d1, d2, sampled ? "seeded decode" : "greedy decode");
    if (f.mtp && !sampled)
      Require(d2.stats.drafted > 0, "no MTP drafts were proposed");
  }
  // The uninterrupted single-call prefill is the same prefill-family state.
  Slot c(adapter, f.streams);
  auto& cold = adapter.GetSession(*c.slot);
  std::vector<std::int32_t> whole = prompt;
  whole.insert(whole.end(), suffix.begin(), suffix.end());
  Feed(cold, whole);
  Require(adapter.Invalidate(*b.slot), "invalidate");
  {
    auto& source = adapter.GetSession(*a.slot);
    Require(adapter.Invalidate(*a.slot), "invalidate");
    Feed(source, prompt);
  }
  const auto checkpoint = Capture(adapter, *a.slot, f.streams);
  Load(adapter, *b.slot, checkpoint, f.streams);
  Feed(adapter.GetSession(*b.slot), suffix);
  SameLogits(cold, adapter.GetSession(*b.slot), "cold vs restored");
  Report("continuation");
}

void EditBefore(Fixture& f) {
  auto& adapter = *f.adapter;
  Slot a(adapter, f.streams), b(adapter, f.streams), ref(adapter, f.streams);
  const auto prompt = Tokens(1100, 4);
  const auto suffix = Tokens(30, 5);
  Feed(adapter.GetSession(*a.slot), prompt);
  const auto checkpoint = Capture(adapter, *a.slot, f.streams);
  // b holds a longer, different history before the restore.
  Feed(adapter.GetSession(*b.slot), Tokens(1700, 6));
  Load(adapter, *b.slot, checkpoint, f.streams);
  auto& restored = adapter.GetSession(*b.slot);
  Require(restored.Position() == 1100, "restored position");
  Feed(restored, suffix);
  auto& reference = adapter.GetSession(*ref.slot);
  Feed(reference, prompt);
  Feed(reference, suffix);
  SameLogits(reference, restored, "edit before checkpoint");
  Report("edit_before");
}

void KeepExpansion(Fixture& f) {
  auto& adapter = *f.adapter;
  Slot a(adapter, f.streams), ref(adapter, f.streams);
  auto& session = adapter.GetSession(*a.slot);
  if (session.ExpandedStoreRows() == 0) {
    std::fprintf(stderr, "[adapter] keep_expansion: no expanded store, skip\n");
    return;
  }
  const auto prompt = Tokens(1300, 7);
  const auto later = Tokens(600, 8);
  const auto suffix = Tokens(20, 9);
  Feed(session, prompt);
  const auto checkpoint = Capture(adapter, *a.slot, f.streams);
  Feed(session, later);  // a longer history in the same slot
  Require(session.ExpandedRowsValid() > 0, "expansion was never populated");
  // Same-slot cold reassignment (the C=1 edit/retry path): release, reset,
  // restore the earlier boundary.
  Require(adapter.Invalidate(*a.slot), "invalidate");
  Load(adapter, *a.slot, checkpoint, f.streams);
  Require(adapter.KeptExpansionRows(*a.slot) > 0,
          "identical restored rows lost their expansion");
  Require(adapter.GetSession(*a.slot).ExpandedRowsValid() ==
              adapter.KeptExpansionRows(*a.slot),
          "kept watermark not installed");
  Feed(adapter.GetSession(*a.slot), suffix);
  auto& reference = adapter.GetSession(*ref.slot);
  Feed(reference, prompt);
  Feed(reference, suffix);
  SameLogits(reference, adapter.GetSession(*a.slot), "kept expansion suffix");
  // A different history in the slot keeps only the identical leading rows.
  Require(adapter.Invalidate(*a.slot), "invalidate");
  Feed(adapter.GetSession(*a.slot), Tokens(1400, 10));
  Require(adapter.Invalidate(*a.slot), "invalidate");
  Load(adapter, *a.slot, checkpoint, f.streams);
  Require(adapter.KeptExpansionRows(*a.slot) == 0,
          "differing rows kept their expansion");
  Feed(adapter.GetSession(*a.slot), suffix);
  SameLogits(reference, adapter.GetSession(*a.slot), "re-expanded suffix");
  Report("keep_expansion");
}

void Guards(Fixture& f) {
  auto& adapter = *f.adapter;
  Slot a(adapter, f.streams), b(adapter, f.streams);
  const auto prompt = Tokens(1300, 11);
  // Decode appends never guard (no CheckStop exposure), even with a borrow.
  auto& session = adapter.GetSession(*a.slot);
  Feed(session, prompt);
  auto checkpoint = Capture(adapter, *a.slot, f.streams);
  a.guard.Borrow(checkpoint);
  const auto overwrites = a.guard.overwrites;
  (void)Decode(session, 12, false);
  Feed(session, Tokens(50, 12));
  Require(a.guard.overwrites == overwrites, "appends called the guard");
  // Engine Invalidate (a failed step) then adapter release: rows preserved.
  session.Invalidate();
  Require(adapter.Invalidate(*a.slot), "invalidate");
  Require(!a.guard.release_failed && a.guard.AllPreserved(),
          "release after engine Invalidate lost borrowed rows");
  Require(a.guard.resets == 1, "AfterReset once per Invalidate");
  // Engine Invalidate, then a forward from 0 guards and preserves first.
  Feed(adapter.GetSession(*a.slot), prompt);
  checkpoint = Capture(adapter, *a.slot, f.streams);
  a.guard.Borrow(checkpoint);
  adapter.GetSession(*a.slot).Invalidate();
  Feed(adapter.GetSession(*a.slot), Tokens(600, 13));
  Require(a.guard.AllPreserved(), "forward at 0 overwrote borrowed rows");
  // BeginRestore preserves the slot's borrowers before loads.
  checkpoint = Capture(adapter, *a.slot, f.streams);
  a.guard.Borrow(checkpoint);
  Feed(adapter.GetSession(*b.slot), prompt);
  const auto other = Capture(adapter, *b.slot, f.streams);
  Load(adapter, *a.slot, other, f.streams);
  Require(a.guard.AllPreserved(), "restore overwrote borrowed rows");
  a.guard.borrowed = nullptr;
  Report("guards");
}

void BatchLanes(Fixture& f) {
  auto& adapter = *f.adapter;
  constexpr std::size_t kLanes = 3, kCount = 16;
  std::vector<std::unique_ptr<Slot>> lanes, solo;
  std::vector<l3::Session*> sessions;
  for (std::size_t i = 0; i < kLanes; ++i) {
    lanes.push_back(std::make_unique<Slot>(adapter, f.streams));
    solo.push_back(std::make_unique<Slot>(adapter, f.streams));
    const auto prompt = Tokens(600 + 100 * i, 20 + i);
    sessions.push_back(&adapter.GetSession(*lanes[i]->slot));
    Feed(*sessions.back(), prompt);
    Feed(adapter.GetSession(*solo[i]->slot), prompt);
  }
  // Lane 1 borrows rows and its guard refuses everything (a cancelled lease):
  // batched appends must never reach it. Lane 2 is mid-restore (not ready):
  // it must fail alone.
  const auto borrowed = Capture(adapter, *lanes[1]->slot, f.streams);
  lanes[1]->guard.Borrow(borrowed);
  lanes[1]->guard.refuse = true;
  const auto other = Capture(adapter, *solo[2]->slot, f.streams);
  adapter.BeginRestore(*lanes[2]->slot, other.positions);
  std::vector<sampling::SamplerState> samplers(kLanes);
  std::vector<std::vector<std::int32_t>> produced(kLanes);
  for (int guard = 0; guard < 64 && (produced[0].size() < kCount ||
                                     produced[1].size() < kCount);
       ++guard) {
    std::vector<l3::Session::BatchDecode> batch(kLanes);
    for (std::size_t i = 0; i < kLanes; ++i)
      batch[i] = {.session = sessions[i],
                  .max_tokens = 8,
                  .sampler = &samplers[i],
                  .stop_at_eos = false};
    l3::Session::DecodeBatch(batch);
    for (std::size_t i = 0; i < 2; ++i) {
      Require(batch[i].ok, "healthy lane failed: " + batch[i].error);
      produced[i].insert(produced[i].end(), batch[i].result.tokens.begin(),
                         batch[i].result.tokens.end());
    }
    Require(!batch[2].ok && !batch[2].deferred,
            "a lane whose slot is restoring was not failed alone");
  }
  for (std::size_t i = 0; i < 2; ++i) {
    const auto alone =
        Decode(adapter.GetSession(*solo[i]->slot), kCount, false);
    Require(produced[i].size() >= kCount &&
                std::equal(alone.tokens.begin(), alone.tokens.end(),
                           produced[i].begin()),
            "batched lane differs from its solo run");
  }
  // EvaluateBatch: the same isolation for AR steps.
  {
    std::vector<l3::Session::BatchLane> step(kLanes);
    for (std::size_t i = 0; i < kLanes; ++i)
      step[i] = {.session = sessions[i], .token = 1234};
    l3::Session::EvaluateBatch(step);
    Require(step[0].ok && step[1].ok, "healthy EvaluateBatch lane failed");
    Require(!step[2].ok, "restoring lane evaluated");
  }
  Require(lanes[1]->guard.overwrites == 0, "batched appends called the guard");
  Require(adapter.Invalidate(*lanes[2]->slot), "abandon restore");
  lanes[1]->guard.refuse = false;
  lanes[1]->guard.borrowed = nullptr;
  Report("batch_lanes");
}

void GraphAfterRestore(Fixture& f) {
  auto& adapter = *f.adapter;
  Slot a(adapter, f.streams), b(adapter, f.streams);
  const auto prompt = Tokens(900, 30);
  Feed(adapter.GetSession(*a.slot), prompt);
  const auto checkpoint = Capture(adapter, *a.slot, f.streams);
  Load(adapter, *b.slot, checkpoint, f.streams);
  const auto replays = f.model->GraphReplays();
  // Plain token steps (max_tokens 1) replay the decode graphs.
  std::vector<std::int32_t> ta, tb;
  for (int i = 0; i < 12; ++i) {
    auto& sa = adapter.GetSession(*a.slot);
    auto& sb = adapter.GetSession(*b.slot);
    std::string error;
    const auto x = sa.Argmax(), y = sb.Argmax();
    Require(x == y, "argmax differs after restore");
    Require(sa.Evaluate(x, &error) && sb.Evaluate(y, &error), error);
    SameLogits(sa, sb, "graph decode");
  }
  Require(f.model->GraphReplays() > replays, "decode graphs did not replay");
  Report("graph_after_restore");
}

void Latching(Fixture& f) {
  auto& adapter = *f.adapter;
  Slot a(adapter, f.streams), b(adapter, f.streams);
  Feed(adapter.GetSession(*a.slot), Tokens(700, 40));
  const auto checkpoint = Capture(adapter, *a.slot, f.streams);
  Reject([&] { (void)adapter.Validate(*b.slot, checkpoint.positions); },
         "Validate outside a restore");
  // Incomplete coverage: validation fails and latches until Invalidate.
  adapter.BeginRestore(*b.slot, checkpoint.positions);
  Require(!adapter.Validate(*b.slot, checkpoint.positions),
          "incomplete restore validated");
  Reject([&] { (void)adapter.GetSession(*b.slot); }, "execution after failure");
  Require(adapter.Invalidate(*b.slot), "invalidate after failure");
  Feed(adapter.GetSession(*b.slot), Tokens(100, 41));  // cold prefill works
  // Wrong private size latches the load.
  adapter.BeginRestore(*b.slot, checkpoint.positions);
  {
    auto stream = f.streams.TryAcquire();
    std::vector<std::byte> small(16);
    Reject(
        [&] {
          (void)adapter.LoadPrivate(*b.slot, adapter.Components().back().id,
                                    small, *stream);
        },
        "short private load");
  }
  Require(!adapter.Validate(*b.slot, checkpoint.positions),
          "latched failure validated");
  Require(adapter.Invalidate(*b.slot), "invalidate");
  // Device-suspect latch: release retires instead of copying.
  a.guard.Borrow(checkpoint);
  l3::rocm::Executor::NoteDeviceError();
  {
    auto stream = f.streams.TryAcquire();
    std::vector<std::byte> bytes(1152);
    Reject(
        [&] {
          (void)adapter.CopyRowsOut(*a.slot, adapter.Components()[1].id, 0, 1,
                                    bytes, *stream);
        },
        "copy while the device is suspect");
  }
  Require(adapter.Invalidate(*a.slot), "invalidate while suspect");
  Require(a.guard.release_failed, "suspect release did not retire");
  l3::rocm::Executor::ClearDeviceSuspect();
  a.guard.borrowed = nullptr;
  Feed(adapter.GetSession(*a.slot), Tokens(64, 42));  // recovers
  Report("latching");
}

std::string Arg(int argc, char** argv, const std::string& key,
                const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i])
      return argv[i + 1];
  return def;
}
bool Flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i)
    if (key == argv[i])
      return true;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const auto model_path = Arg(argc, argv, "--model");
  if (model_path.empty()) {
    std::fprintf(stderr, "skip: no --model\n");
    return 77;
  }
  Fixture f;
  f.mtp = Flag(argc, argv, "--mtp");
  l3::ModelOptions options;
  options.max_context = static_cast<std::uint32_t>(
      std::stoul(Arg(argc, argv, "--max-context", "4096")));
  options.max_batch = 512;
  options.layer_limit =
      static_cast<std::uint32_t>(std::stoul(Arg(argc, argv, "--layers", "12")));
  options.mtp = f.mtp;
  options.expanded_kv_bytes = std::uint64_t{4} << 30;
  std::string error;
  f.model = l3::Model::Load(model_path, options, &error);
  if (!f.model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  f.adapter = std::make_unique<l3::ContinuationAdapter>(
      f.model, options.max_context, 7, cache::Identity{'t', 'e', 's', 't'});
  const std::vector<std::pair<const char*, std::function<void(Fixture&)>>>
      cases = {{"round_trip", RoundTrip},
               {"continuation", Continuation},
               {"edit_before", EditBefore},
               {"keep_expansion", KeepExpansion},
               {"guards", Guards},
               {"batch_lanes", BatchLanes},
               {"graph_after_restore", GraphAfterRestore},
               {"latching", Latching}};
  int failures = 0;
  for (const auto& [name, run] : cases) {
    try {
      run(f);
    } catch (const std::exception& e) {
      ++failures;
      std::fprintf(stderr, "[adapter] %s: FAIL: %s\n", name, e.what());
    }
  }
  std::printf(
      "{\"test\": \"continuation_adapter\", \"mtp\": %s, "
      "\"pass\": %s, \"failures\": %d}\n",
      f.mtp ? "true" : "false", failures ? "false" : "true", failures);
  return failures ? 1 : 0;
}
