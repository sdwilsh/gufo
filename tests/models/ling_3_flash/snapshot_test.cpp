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
//   * malformed payloads are rejected without touching the session.
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
              snap->SizeBytes() == l3::SnapshotPayloadBytes(N) &&
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
