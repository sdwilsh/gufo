// Full-model snapshot contract checks for Ling-3.0-flash on the GPU (see the
// reproducibility contract in engine.hpp). Every comparison is bitwise over
// the logits, the KDA state, the conv windows and the MLA cache rows:
//   * restore into the same session, a fresh session, a session holding a
//     divergent history, and from the serialized bytes (disk payload);
//   * continue after a restore with the tokens of the uninterrupted run, in the
//     same family (prefill after prefill, decode after prefill) == the
//     uninterrupted run, exactly;
//   * truncation: restore an earlier checkpoint after running past it, then
//     continue;
//   * the payload size matches Session::SnapshotBytes and 8068 bytes/token;
//   * malformed payloads are rejected without touching the session;
//   * the background row copy (the latent rows are copied after
//     SaveSnapshot returned): bytes equal the synchronous copy while the
//     session decodes beside the copy, a reset/rewrite of the rows and the
//     session's destruction wait for it, destroying the snapshot aborts it, a
//     failed copy is unusable and leaves restore targets untouched, batched
//     decode proceeds beside it, and the host-only Invalidate never waits.
// Exit 77 without --model.
//
// After WS12 (persistent expanded K/V in 2048-key blocks) merges, also run with
// --prompt-tokens 2300 --max-context 4096: the continuation checks then attend
// across an expanded block boundary after a restore and must still equal the
// cold run bit for bit (restore invalidates the derived rows through
// rocm::Session::OnStateReplaced).
//
//   snapshot_test --model FIRST_SHARD.gguf [--prompt-tokens N]
//                 [--decode-tokens K] [--max-context N] [--max-batch N]
//                 [--layers N] [--timeout-seconds S] [--hold-seconds S]

#define LING_TOOL_NO_MAIN
#include <chrono>
#include <future>

#include "tests/models/ling_3_flash/session_test.cpp"

namespace gufo::tests::ling::snapshot_tool {

using session_tool::Capture;
using session_tool::Compare;
using session_tool::Diff;
using session_tool::Take;
using session_tool::ToJson;

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  const std::uint32_t layers = model->LayerCount();
  const auto N = static_cast<std::uint32_t>(args.Num("--prompt-tokens", 100));
  const auto K = static_cast<std::uint32_t>(args.Num("--decode-tokens", 6));
  std::string text;
  std::vector<std::int32_t> ids;
  while (ids.size() < N + K) {
    text += session_tool::kText;
    ids = model->Tokenize(text);
  }
  ids.resize(N + K);
  const std::uint32_t total = N + K;
  const std::uint32_t ctx = model->MaxContext();
  std::string error;
  auto checks = json::Value::array();
  int failures = 0;
  const auto check = [&](const std::string& name, bool pass,
                         json::Value metrics) {
    metrics["name"] = name;
    metrics["pass"] = pass;
    checks.push_back(std::move(metrics));
    std::fprintf(stderr, "[check] %-44s %s\n", name.c_str(),
                 pass ? "PASS" : "FAIL");
    failures += pass ? 0 : 1;
  };
  const auto fail_with = [&](const char* what) {
    std::fprintf(stderr, "snapshot_test: %s: %s\n", what, error.c_str());
    report.Set("status", "error");
    report.Set("error", std::string(what) + ": " + error);
    report.Set("checks", std::move(checks));
    return 1;
  };
  const auto exact = [&](const Capture& a, const Capture& b) {
    const Diff d = Compare(a, b);
    return std::pair{session_tool::Exact(d), ToJson(d)};
  };
  const auto fresh = [&]() { return model->CreateSession(ctx, &error); };
  const auto feed = [&](l3::Session& s, std::size_t from, std::size_t to) {
    return s.Feed(std::span(ids).subspan(from, to - from), {}, &error);
  };
  const auto feed_each = [&](l3::Session& s, std::size_t from, std::size_t to) {
    for (std::size_t i = from; i < to; ++i) {
      if (!feed(s, i, i + 1)) {
        return false;
      }
    }
    return true;
  };

  // Uninterrupted references: one prefill chunk of all tokens, and a prefill
  // of N followed by token-by-token decode (what generation does).
  auto s_ref = fresh();
  auto s_gen = fresh();
  if (!s_ref || !s_gen || !feed(*s_ref, 0, total) || !feed(*s_gen, 0, N) ||
      !feed_each(*s_gen, N, total)) {
    return fail_with("reference runs");
  }
  Capture ref_prefill, ref_gen;
  if (!Take(*s_ref, layers, &ref_prefill, &error) ||
      !Take(*s_gen, layers, &ref_gen, &error)) {
    return fail_with("capture references");
  }
  // The persistent expanded K/V budget is shared by every live session, so
  // sessions are released as soon as their last use has passed; at large
  // contexts the later expansion checks would otherwise find the pool empty.
  s_ref.reset();
  s_gen.reset();

  // Snapshot at N after a prefill-family run.
  auto s_a = fresh();
  if (!s_a || !feed(*s_a, 0, N)) {
    return fail_with("prefill to snapshot point");
  }
  Capture at_n;
  if (!Take(*s_a, layers, &at_n, &error)) {
    return fail_with("capture at N");
  }
  auto snap = s_a->SaveSnapshot(&error);
  if (!snap) {
    return fail_with("save snapshot");
  }
  {
    auto o = json::Value::object();
    o["position"] = U(snap->Position());
    o["bytes"] = U(snap->SizeBytes());
    o["bytes_per_token"] =
        U(l3::SnapshotPayloadBytes(2) - l3::SnapshotPayloadBytes(1));
    o["fixed_bytes"] = U(l3::SnapshotPayloadBytes(0));
    check("payload_size",
          snap->SizeBytes() == s_a->SnapshotBytes() &&
              snap->SizeBytes() ==
                  l3::SnapshotPayloadBytes(N, args.Has("--mtp")) &&
              snap->bytes().size() == snap->SizeBytes(),
          std::move(o));
  }

  // 1. Restoring into a session that holds a different history is exact.
  //    (a) same session after running on; (b) fresh session; (c) divergent.
  auto verify_restore = [&](const char* name, l3::Session& s) {
    Capture got;
    if (!Take(s, layers, &got, &error)) {
      return false;
    }
    auto [ok, o] = exact(got, at_n);
    o["position"] = U(s.Position());
    check(name, ok && s.Position() == N, std::move(o));
    return true;
  };
  if (!feed_each(*s_a, N, total) || !s_a->RestoreSnapshot(*snap, &error) ||
      !verify_restore("restore_same_session_after_running_on", *s_a)) {
    return fail_with("restore same session");
  }
  auto s_b = fresh();
  if (!s_b || !s_b->RestoreSnapshot(*snap, &error) ||
      !verify_restore("restore_fresh_session", *s_b)) {
    return fail_with("restore fresh session");
  }
  auto s_c = fresh();
  std::vector<std::int32_t> other(ids.begin(), ids.begin() + total);
  for (auto& t : other) {
    t = (t * 7 + 13) % 1000;
  }
  if (!s_c || !s_c->Feed(other, {}, &error) ||
      !s_c->RestoreSnapshot(*snap, &error) ||
      !verify_restore("restore_divergent_history", *s_c)) {
    return fail_with("restore divergent session");
  }
  // From serialized bytes, as the disk cache does.
  std::vector<std::uint8_t> payload(snap->bytes().begin(), snap->bytes().end());
  auto s_d = fresh();
  if (!s_d || !s_d->RestoreSnapshot(payload, &error) ||
      !verify_restore("restore_from_serialized_bytes", *s_d)) {
    return fail_with("restore from bytes");
  }

  // 2. Continuation after restore == the uninterrupted run, same family.
  for (auto* s : {&*s_a, &*s_b, &*s_c, &*s_d}) {
    // Decode continuation after a prefill-family prefix.
    if (!feed_each(*s, N, total)) {
      return fail_with("decode continuation");
    }
    Capture got;
    if (!Take(*s, layers, &got, &error)) {
      return fail_with("capture continuation");
    }
    auto [ok, o] = exact(got, ref_gen);
    o["family"] = "prefill prefix + decode";
    check(std::string("continue_decode_") +
              (s == &*s_a   ? "same_session"
               : s == &*s_b ? "fresh_session"
               : s == &*s_c ? "divergent_session"
                            : "from_bytes") +
              "_vs_uninterrupted",
          ok, std::move(o));
  }
  s_a.reset();
  s_b.reset();
  s_c.reset();
  s_d.reset();

  // 3. Prefill-family continuation: snapshots at several prefill positions,
  //    each continued by a prefill-family chunk (> 8 tokens) == one chunk.
  for (const std::uint32_t a :
       {std::uint32_t{9}, std::max<std::uint32_t>(9, total * 37 / 100),
        total / 2, total - 9}) {
    if (a <= 8 || total - a <= 8) {
      continue;
    }
    auto sp = fresh();
    auto sr = fresh();
    if (!sp || !sr || !feed(*sp, 0, a)) {
      return fail_with("prefix prefill");
    }
    auto sn = sp->SaveSnapshot(&error);
    if (!sn || !sr->RestoreSnapshot(*sn, &error) || !feed(*sr, a, total)) {
      return fail_with("continue after restore");
    }
    Capture got;
    if (!Take(*sr, layers, &got, &error)) {
      return fail_with("capture prefill continuation");
    }
    auto [ok, o] = exact(got, ref_prefill);
    o["split_at"] = U(a);
    o["family"] = "prefill + prefill";
    check("continue_prefill_at_" + std::to_string(a) + "_vs_one_chunk", ok,
          std::move(o));

    // The same, restored into a session that already holds a different
    // history: its persistent expanded K/V rows (derived state) describe the
    // old cache rows, so the restore hook must invalidate them
    // (rocm::Session::OnStateReplaced), or this continuation would attend over
    // stale expansions.
    auto sdv = fresh();
    if (!sdv || !sdv->Feed(other, {}, &error) ||
        !sdv->RestoreSnapshot(*sn, &error) || !feed(*sdv, a, total)) {
      return fail_with("continue after restore into a divergent session");
    }
    Capture got_divergent;
    if (!Take(*sdv, layers, &got_divergent, &error)) {
      return fail_with("capture divergent prefill continuation");
    }
    auto [ok_divergent, o_divergent] = exact(got_divergent, ref_prefill);
    o_divergent["split_at"] = U(a);
    o_divergent["family"] = "prefill + prefill, divergent session";
    check("continue_prefill_at_" + std::to_string(a) +
              "_divergent_session_vs_one_chunk",
          ok_divergent, std::move(o_divergent));
  }

  // 3b. Expanded K/V reuse across a restore. The persistent expansion of a
  //     cache row is a pure function of the row, so a restore into a session
  //     whose history shares a prefix (same bytes) keeps the shared rows
  //     expanded and re-expands only the rest; a divergent history loses all
  //     of it. Either way the continuation equals the cold run bit for bit.
  {
    const std::uint32_t a = total / 2;
    auto sp = fresh();
    auto ss = fresh();
    auto sdv = fresh();
    if (!sp || !ss || !sdv || !feed(*sp, 0, a) || !feed(*ss, 0, total) ||
        !sdv->Feed(other, {}, &error)) {
      return fail_with("expansion reuse setup");
    }
    // Precondition: ss must have a store covering the whole prompt (whole
    // 2048-row blocks), or its expansion is never built and the check below
    // would measure the shared budget rather than the restore.
    constexpr std::uint32_t kBlockRows = 2048;
    constexpr std::uint64_t kBlockBytes =
        std::uint64_t{kBlockRows} * 7 * 16384;  // 7 MLA layers, 16 KiB/row
    const std::uint32_t needed_rows =
        (total + kBlockRows - 1) / kBlockRows * kBlockRows;
    const std::uint32_t first_layer =
        static_cast<std::uint32_t>(args.Num("--layer-first", 0));
    bool mla_resident = false;
    for (std::uint32_t il = first_layer; il < first_layer + layers; ++il) {
      mla_resident =
          mla_resident || (il + 1) % 6 == 0;  // MLA blocks 5, 11, ...
    }
    if (!mla_resident) {
      std::fprintf(stderr,
                   "snapshot_test: no MLA layer resident (layers=%u, "
                   "layer_first=%u); expansion checks need at least 6 layers\n",
                   layers, first_layer);
      auto o = json::Value::object();
      o["layers"] = U(layers);
      o["layer_first"] = U(first_layer);
      o["error"] = "no MLA layer resident; expansion checks need 6 layers";
      check("mla_layer_resident", false, std::move(o));
    }
    const std::uint32_t store_rows = ss->ExpandedStoreRows();
    if (store_rows < needed_rows) {
      std::fprintf(stderr,
                   "snapshot_test: expanded K/V store too small: "
                   "store_rows=%u, needed_rows=%u; raise --expanded-kv-bytes "
                   "to at least %llu MiB for the live sessions\n",
                   store_rows, needed_rows,
                   static_cast<unsigned long long>(
                       (3 * needed_rows / kBlockRows * kBlockBytes) >> 20));
      auto o = json::Value::object();
      o["store_rows"] = U(store_rows);
      o["needed_rows"] = U(needed_rows);
      o["error"] = "expanded K/V store too small; raise --expanded-kv-bytes";
      check("expanded_store_covers_prompt", false, std::move(o));
    }
    auto sn = sp->SaveSnapshot(&error);
    const std::uint32_t before = ss->ExpandedRowsValid();
    if (!sn || !ss->RestoreSnapshot(*sn, &error) ||
        !sdv->RestoreSnapshot(*sn, &error)) {
      return fail_with("expansion reuse restore");
    }
    const std::uint32_t shared_valid = ss->ExpandedRowsValid();
    const std::uint32_t divergent_valid = sdv->ExpandedRowsValid();
    auto o = json::Value::object();
    o["split_at"] = U(a);
    o["valid_before"] = U(before);
    o["valid_after_shared_prefix"] = U(shared_valid);
    o["valid_after_divergent"] = U(divergent_valid);
    check("restore_keeps_shared_expansion",
          before >= a && shared_valid == a && divergent_valid == 0,
          std::move(o));
    for (auto* s2 : {&*ss, &*sdv}) {
      Capture got;
      if (!feed(*s2, a, total) || !Take(*s2, layers, &got, &error)) {
        return fail_with("continue after expansion reuse restore");
      }
      auto [ok, oo] = exact(got, ref_prefill);
      check(s2 == &*ss ? "continue_after_shared_expansion_vs_one_chunk"
                       : "continue_after_divergent_expansion_vs_one_chunk",
            ok, std::move(oo));
    }
  }

  // 4. Truncation to an earlier checkpoint after running past it: take
  //    checkpoints on the way, restore the first and continue with the same
  //    tokens; then restore it again and continue with a different tail.
  {
    const std::uint32_t c = std::max<std::uint32_t>(9, N / 2);
    auto st = fresh();
    if (!st || !feed(*st, 0, c)) {
      return fail_with("truncation prefix");
    }
    auto cp = st->SaveSnapshot(&error);
    if (!cp || !feed(*st, c, N)) {
      return fail_with("truncation run past checkpoint");
    }
    // Different tail first, to dirty cache rows past the checkpoint.
    if (!st->RestoreSnapshot(*cp, &error) ||
        !st->Feed(std::span(other).subspan(c, N - c), {}, &error)) {
      return fail_with("truncation divergent tail");
    }
    if (!st->RestoreSnapshot(*cp, &error) || !feed(*st, c, total)) {
      return fail_with("truncation continue");
    }
    Capture got;
    if (!Take(*st, layers, &got, &error)) {
      return fail_with("capture truncation");
    }
    auto [ok, o] = exact(got, ref_prefill);
    o["checkpoint"] = U(c);
    check("truncate_to_checkpoint_then_continue_vs_one_chunk", ok,
          std::move(o));
  }

  // 5. Capacity: a snapshot restores into a session whose context is only as
  //    large as the snapshot needs; a smaller one rejects it, unchanged.
  {
    auto small = model->CreateSession(N + K, &error);
    auto tiny = model->CreateSession(N - 1, &error);
    bool ok = small && tiny && small->RestoreSnapshot(*snap, &error);
    Capture got;
    ok = ok && Take(*small, layers, &got, &error) && exact(got, at_n).first;
    std::string reject;
    const bool rejected =
        tiny && !tiny->RestoreSnapshot(*snap, &reject) && tiny->Position() == 0;
    auto o = json::Value::object();
    o["small_context_restores"] = ok;
    o["too_small_rejected"] = rejected;
    check("capacity_checks", ok && rejected, std::move(o));
  }

  // 6. Malformed payloads are rejected and leave the session untouched.
  {
    auto sm = fresh();
    if (!sm || !feed(*sm, 0, 12)) {
      return fail_with("malformed setup");
    }
    Capture before;
    if (!Take(*sm, layers, &before, &error)) {
      return fail_with("malformed capture");
    }
    std::string e;
    auto bad = payload;
    bad[0] ^= 1;  // magic
    auto truncated = std::vector<std::uint8_t>(
        payload.begin(), payload.begin() + payload.size() - 1);
    auto versioned = payload;
    versioned[8] ^= 1;
    auto grown = payload;
    grown.push_back(0);
    auto empty = std::vector<std::uint8_t>{};
    const auto at = l3::SnapshotLayout(N);
    auto bad_token = payload;
    const std::int32_t huge = 0x7fffffff;
    std::memcpy(bad_token.data() + at.tokens, &huge, sizeof(huge));
    auto negative_token = payload;
    const std::int32_t negative = -1;
    std::memcpy(negative_token.data() + at.tokens, &negative, sizeof(negative));
    auto bad_logits = payload;
    const float nan = std::nanf("");
    std::memcpy(bad_logits.data() + at.logits + 40, &nan, sizeof(nan));
    const bool rejected =
        !sm->RestoreSnapshot(bad, &e) && !sm->RestoreSnapshot(truncated, &e) &&
        !sm->RestoreSnapshot(versioned, &e) &&
        !sm->RestoreSnapshot(grown, &e) && !sm->RestoreSnapshot(empty, &e) &&
        !sm->RestoreSnapshot(bad_token, &e) &&
        !sm->RestoreSnapshot(negative_token, &e) &&
        !sm->RestoreSnapshot(bad_logits, &e);
    Capture after;
    const bool same = Take(*sm, layers, &after, &error) &&
                      exact(after, before).first && sm->Position() == 12;
    auto o = json::Value::object();
    o["rejected"] = rejected;
    o["session_untouched"] = same;
    check("malformed_payloads_rejected", rejected && same, std::move(o));
  }

  // 7. An empty session cannot be snapshotted.
  {
    auto se = fresh();
    const bool none =
        se && !se->SaveSnapshot(&error) && se->SnapshotBytes() == 0;
    auto o = json::Value::object();
    check("empty_session_has_no_snapshot", none, std::move(o));
  }

  // 8. Background row copy. The gate pauses the copy before each staging
  //    chunk so the session can be driven while it is provably in flight.
  {
    struct Gate {
      l3::Model& m;
      explicit Gate(l3::Model& model) : m(model) {
        m.HoldSnapshotRowCopiesForTest(true);
      }
      void Open() { m.HoldSnapshotRowCopiesForTest(false); }
      ~Gate() { Open(); }
    };
    using namespace std::chrono_literals;
    const auto blocked = [](auto& future) {
      return future.wait_for(300ms) != std::future_status::ready;
    };
    const auto same_bytes = [&](const l3::SessionSnapshot& a,
                                const l3::SessionSnapshot& b) {
      const auto x = a.bytes();
      const auto y = b.bytes();
      return x.size() == y.size() && !x.empty() &&
             std::memcmp(x.data(), y.data(), x.size()) == 0;
    };

    // 8a. Decode beside the in-flight copy: it does not wait, the decoded
    //     state equals the uninterrupted run, and the payload equals the
    //     copy taken with nothing in flight (`snap`).
    {
      auto sd = fresh();
      Gate gate(*model);
      std::unique_ptr<l3::SessionSnapshot> held;
      bool in_flight = false, size_known = false, decode_ok = false;
      bool still_in_flight = false;
      if (sd && feed(*sd, 0, N)) {
        held = sd->SaveSnapshot(&error);
        in_flight = held && !held->Ready();
        size_known = held && held->SizeBytes() == sd->SnapshotBytes() &&
                     held->SizeBytes() == snap->SizeBytes();
        decode_ok = held && feed_each(*sd, N, total);
        still_in_flight = held && !held->Ready();
      }
      Capture got;
      const bool state_ok = decode_ok && Take(*sd, layers, &got, &error) &&
                            exact(got, ref_gen).first;
      gate.Open();
      std::string wait_error;
      const bool done = held && held->Wait(&wait_error) && !held->Failed();
      auto o = json::Value::object();
      o["in_flight_after_save"] = in_flight;
      o["size_known_while_pending"] = size_known;
      o["decode_while_pending"] = decode_ok;
      o["still_pending_after_decode"] = still_in_flight;
      o["decoded_state_equals_uninterrupted"] = state_ok;
      o["payload_equals_earlier_copy"] = done && same_bytes(*held, *snap);
      // The reference: Executor::SaveSnapshot (both phases back to back) on
      // an identically built session, compared byte for byte.
      bool equals_sync = false;
      if (done) {
        auto sref = fresh();
        auto sync = sref && feed(*sref, 0, N)
                        ? sref->SaveSnapshotSynchronouslyForTest(&error)
                        : nullptr;
        equals_sync = sync && sync->Wait() && same_bytes(*held, *sync);
      }
      o["payload_equals_executor_save_snapshot"] = equals_sync;
      check("async_copy_decode_beside_copy_bytes_exact",
            in_flight && size_known && decode_ok && still_in_flight &&
                state_ok && done && same_bytes(*held, *snap) && equals_sync,
            std::move(o));
    }

    // 8b. Rewriting the rows waits: invalidate (host only, never waits), then
    //     refill the session from scratch with different tokens. The refill
    //     blocks until the copy ends, and the payload is still the original.
    {
      auto sd = fresh();
      Gate gate(*model);
      std::unique_ptr<l3::SessionSnapshot> held;
      bool invalidate_returned = false, refill_blocked = false;
      bool refilled = false;
      if (sd && feed(*sd, 0, N)) {
        held = sd->SaveSnapshot(&error);
      }
      if (held) {
        sd->Invalidate();
        invalidate_returned = !held->Ready() && sd->Position() == 0;
        auto refill = std::async(std::launch::async, [&] {
          std::string e;
          return sd->Feed(std::span(other).subspan(0, N), {}, &e);
        });
        refill_blocked = blocked(refill) && !held->Ready();
        gate.Open();
        refilled = refill.get();
      }
      std::string wait_error;
      const bool done = held && held->Wait(&wait_error);
      auto o = json::Value::object();
      o["invalidate_returned_without_waiting"] = invalidate_returned;
      o["refill_waited_for_copy"] = refill_blocked;
      o["refill_ok"] = refilled;
      o["payload_unchanged_by_refill"] = done && same_bytes(*held, *snap);
      check("async_copy_rewrite_waits_invalidate_does_not",
            invalidate_returned && refill_blocked && refilled && done &&
                same_bytes(*held, *snap),
            std::move(o));
    }

    // 8c. Restoring the session over its own in-flight snapshot, and a
    //     restore of the pending snapshot elsewhere, wait for the copy.
    {
      auto sd = fresh();
      auto st = fresh();
      Gate gate(*model);
      std::unique_ptr<l3::SessionSnapshot> held;
      bool blocked_restore = false, restored = false, same_session = false;
      if (sd && st && feed(*sd, 0, N)) {
        held = sd->SaveSnapshot(&error);
      }
      if (held) {
        auto restore = std::async(std::launch::async, [&] {
          std::string e;
          return st->RestoreSnapshot(*held, &e);
        });
        blocked_restore = blocked(restore) && st->Position() == 0;
        gate.Open();
        restored = restore.get();
        same_session = sd->RestoreSnapshot(*held, &error);
      }
      Capture got_other, got_same;
      const bool ok =
          restored && same_session && Take(*st, layers, &got_other, &error) &&
          Take(*sd, layers, &got_same, &error) &&
          exact(got_other, at_n).first && exact(got_same, at_n).first;
      auto o = json::Value::object();
      o["restore_waited_for_copy"] = blocked_restore;
      check("async_copy_restore_waits_and_is_exact", blocked_restore && ok,
            std::move(o));
    }

    // 8d. Destroying the snapshot aborts its copy; the session is usable and
    //     a rewrite no longer waits. Destroying the session waits for the
    //     copy of a snapshot that outlives it.
    {
      auto sd = fresh();
      Gate gate(*model);
      bool aborted_fast = false, rewrite_ok = false, destroy_waited = false;
      bool outlived_exact = false;
      if (sd && feed(*sd, 0, N)) {
        auto gone = sd->SaveSnapshot(&error);
        const auto t0 = std::chrono::steady_clock::now();
        gone.reset();  // abort + join, though the gate is still closed
        aborted_fast = std::chrono::steady_clock::now() - t0 < 5s;
        sd->Invalidate();
        rewrite_ok = feed(*sd, 0, N);  // no hold left: does not wait
        auto outlive = sd->SaveSnapshot(&error);
        if (outlive) {
          auto destroy = std::async(std::launch::async, [&] { sd.reset(); });
          destroy_waited = blocked(destroy) && !outlive->Ready();
          gate.Open();
          destroy.get();
          outlived_exact = outlive->Wait() && same_bytes(*outlive, *snap);
        }
      }
      auto o = json::Value::object();
      o["abort_on_destroy"] = aborted_fast;
      o["rewrite_after_abort"] = rewrite_ok;
      o["session_destroy_waited_for_copy"] = destroy_waited;
      o["snapshot_outlives_session_exact"] = outlived_exact;
      check("async_copy_destroy_snapshot_aborts_destroy_session_waits",
            aborted_fast && rewrite_ok && destroy_waited && outlived_exact,
            std::move(o));
    }

    // 8e. A failed copy is unusable: no bytes, restore refused with the
    //     target untouched, and the failing copy does not disturb the next.
    {
      auto sd = fresh();
      auto st = fresh();
      bool failed = false, no_bytes = false, refused = false;
      bool target_untouched = false, next_ok = false;
      if (sd && st && feed(*sd, 0, N) && feed(*st, 0, 12)) {
        model->FailSnapshotRowCopiesForTest(1);
        auto bad = sd->SaveSnapshot(&error);
        std::string why;
        failed = bad && !bad->Wait(&why) && bad->Failed() && bad->Ready() &&
                 !why.empty();
        no_bytes = bad && bad->bytes().empty() &&
                   bad->SizeBytes() == snap->SizeBytes();
        std::string refusal;
        refused =
            bad && !st->RestoreSnapshot(*bad, &refusal) && !refusal.empty();
        target_untouched = st->Position() == 12;
        auto good = sd->SaveSnapshot(&error);
        next_ok = good && good->Wait() && same_bytes(*good, *snap);
      }
      auto o = json::Value::object();
      o["failed_copy_reports_failure"] = failed;
      o["failed_copy_has_no_bytes"] = no_bytes;
      o["restore_refused"] = refused;
      o["target_untouched"] = target_untouched;
      o["next_copy_exact"] = next_ok;
      check("async_copy_failure_is_unusable_and_contained",
            failed && no_bytes && refused && target_untouched && next_ok,
            std::move(o));
    }

    // 8f. Batched decode beside in-flight copies: every lane runs in the
    //     batch (none is deferred) and ends bit-identical to the same steps
    //     taken alone.
    {
      const std::uint32_t lanes_n =
          std::min<std::uint32_t>(model->BatchRowLimit(), 3);
      std::vector<std::unique_ptr<l3::Session>> alone, batched;
      std::vector<std::unique_ptr<l3::SessionSnapshot>> held;
      bool setup = lanes_n >= 2;
      for (std::uint32_t i = 0; setup && i < lanes_n; ++i) {
        alone.push_back(fresh());
        batched.push_back(fresh());
        const auto len = N - 3 * i;
        setup = alone.back() && batched.back() && feed(*alone.back(), 0, len) &&
                feed(*batched.back(), 0, len);
      }
      Gate gate(*model);
      bool none_deferred = setup, all_ok = setup, equal = setup;
      bool pending = setup;
      for (std::uint32_t i = 0; setup && i < lanes_n; ++i) {
        held.push_back(batched[i]->SaveSnapshot(&error));
        pending = pending && held.back() && !held.back()->Ready();
      }
      for (int step = 0; setup && step < 3; ++step) {
        std::vector<l3::Session::BatchLane> lanes(lanes_n);
        for (std::uint32_t i = 0; i < lanes_n; ++i) {
          lanes[i].session = batched[i].get();
          lanes[i].token = ids[N + step];
          all_ok = all_ok && alone[i]->Evaluate(ids[N + step], &error);
        }
        l3::Session::EvaluateBatch(lanes);
        for (const auto& lane : lanes) {
          none_deferred = none_deferred && !lane.deferred;
          all_ok = all_ok && lane.ok;
        }
      }
      for (std::uint32_t i = 0; setup && i < lanes_n; ++i) {
        Capture a, b;
        equal = equal && Take(*alone[i], layers, &a, &error) &&
                Take(*batched[i], layers, &b, &error) && exact(b, a).first;
      }
      pending = pending && !held.empty() && !held.back()->Ready();
      gate.Open();
      bool exact_payloads = setup;
      for (std::uint32_t i = 0; setup && i < lanes_n; ++i) {
        auto ref = fresh();
        auto again =
            feed(*ref, 0, N - 3 * i) ? ref->SaveSnapshot(&error) : nullptr;
        exact_payloads = exact_payloads && again && again->Wait() &&
                         held[i]->Wait() && same_bytes(*held[i], *again);
      }
      auto o = json::Value::object();
      o["lanes"] = U(lanes_n);
      o["copies_pending_during_batch"] = pending;
      o["no_lane_deferred"] = none_deferred;
      o["batched_equals_alone"] = equal;
      o["payloads_exact"] = exact_payloads;
      check("async_copy_batched_decode_beside_copy",
            setup && pending && none_deferred && all_ok && equal &&
                exact_payloads,
            std::move(o));
    }
  }

  report.Set("checks", std::move(checks));
  report.Set("failures", failures);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

}  // namespace gufo::tests::ling::snapshot_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "snapshot_test",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model = gufo::tests::ling::LoadModel(args, report, 256, 512);
        return model
                   ? gufo::tests::ling::snapshot_tool::Run(args, report, model)
                   : 1;
      });
}
