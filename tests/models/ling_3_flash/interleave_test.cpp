// Cross-session isolation of Ling-3.0-flash sessions sharing one executor, as
// the server drives them: several sessions restore the same shared-prefix
// snapshot, then their prefill chunks, snapshot captures (on worker threads,
// overlapping other sessions' forward passes and decode-graph captures, like
// the RAM cache's post-token captures) and decode steps interleave in a seeded
// random order. Every session must end bit-identical (logits, KDA state, conv
// windows, MLA cache rows and the greedy tokens) to the same work done alone
// on an otherwise idle session, and every worker snapshot must equal the
// snapshot an idle session takes at the same point. After the greedy steps
// each session feeds forced tokens in small-batch chunks of 1..8 tokens, three
// of each size, so every round captures and replays a decode graph per chunk
// size on fresh sessions (the reference feeds them one by one; the contract
// makes any split bit-identical). A background thread keeps snapshotting an
// unrelated idle session throughout, so worker snapshot calls overlap those
// HIP graph captures on the executor stream (a worker that waited on an event
// of that stream used to break the capture and let its shared-expert branch
// run outside the graph, racing the eager fallback). No worker snapshot may
// fail and no capture may be invalidated: the executor serializes those waits
// with its captures, so either is a regression even when the bits survive.
// Exit 77 without --model.
//
//   interleave_test --model FIRST_SHARD.gguf [--layers N] [--sessions S]
//                   [--prefix-tokens P] [--tail-tokens T] [--decode-tokens K]
//                   [--rounds R] [--seed N] [--max-context N] [--max-batch N]
//                   [--no-background]
//
// Prints one JSON report on stdout (checks[] with metrics).

#define LING_TOOL_NO_MAIN
#include <atomic>
#include <future>
#include <random>

#include "tests/models/ling_3_flash/session_test.cpp"

namespace gufo::tests::ling::interleave_tool {

using session_tool::Capture;
using session_tool::Compare;
using session_tool::Diff;
using session_tool::Take;
using session_tool::ToJson;

struct Expected {
  std::vector<std::int32_t> tokens;
  Capture state;
  std::vector<float> logits_after_greedy;
  std::vector<std::uint8_t> snapshot;  ///< payload after the first chunk
};

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  const std::uint32_t layers = model->LayerCount();
  const auto S = static_cast<std::uint32_t>(args.Num("--sessions", 4));
  const auto P = static_cast<std::uint32_t>(args.Num("--prefix-tokens", 600));
  const auto T = static_cast<std::uint32_t>(args.Num("--tail-tokens", 19));
  const auto K = static_cast<std::uint32_t>(args.Num("--decode-tokens", 12));
  const auto R = static_cast<std::uint32_t>(args.Num("--rounds", 6));
  const auto seed = static_cast<std::uint32_t>(args.Num("--seed", 1));
  const bool background = !args.Has("--no-background");
  // Two prefill-family chunks (more than 8 tokens each), as the server splits
  // a short tail around its stable checkpoint.
  const std::uint32_t c1 = T / 2 + T % 2;
  if (T / 2 <= 8 || S < 2) {
    report.Set("status", "error");
    report.Set("error", "need --tail-tokens >= 18 and --sessions >= 2");
    return 1;
  }
  // Forced small-batch chunks: three of every size 1..8 (warm, capture,
  // replay of each decode-graph key).
  std::vector<std::uint32_t> chunks;
  for (std::uint32_t n = 1; n <= l3::kSmallBatchMaxTokens; ++n) {
    chunks.insert(chunks.end(), 3, n);
  }
  std::uint32_t forced_total = 0;
  for (const auto n : chunks) {
    forced_total += n;
  }
  const std::uint32_t ctx = model->MaxContext();
  if (P + 64 * S + T + K + forced_total + 64 > ctx) {
    report.Set("status", "error");
    report.Set("error", "--max-context too small for the prefix and tails");
    return 1;
  }
  std::string text;
  std::vector<std::int32_t> ids;
  while (ids.size() < P + 64 * S + T + forced_total) {
    text += session_tool::kText;
    ids = model->Tokenize(text);
  }
  const std::vector<std::int32_t> prefix(ids.begin(), ids.begin() + P);
  std::vector<std::vector<std::int32_t>> tails(S);
  for (std::uint32_t i = 0; i < S; ++i) {
    const auto at = ids.begin() + P + 64 * i;
    tails[i].assign(at, at + T);
  }
  const auto forced_at = ids.begin() + P + 64 * S + T;
  const std::vector<std::int32_t> forced(forced_at, forced_at + forced_total);

  std::string error;
  auto checks = json::Value::array();
  int failures = 0;
  const auto check = [&](const std::string& name, bool pass,
                         json::Value metrics) {
    metrics["name"] = name;
    metrics["pass"] = pass;
    checks.push_back(std::move(metrics));
    std::fprintf(stderr, "[check] %-52s %s\n", name.c_str(),
                 pass ? "PASS" : "FAIL");
    failures += pass ? 0 : 1;
    report.Set("checks", checks);
  };
  const auto fail_with = [&](const std::string& what) {
    std::fprintf(stderr, "interleave_test: %s: %s\n", what.c_str(),
                 error.c_str());
    report.Set("status", "error");
    report.Set("error", what + ": " + error);
    report.Set("checks", checks);
    return 1;
  };
  const auto fresh = [&]() { return model->CreateSession(ctx, &error); };

  // The shared prefix, prefilled once and published as a snapshot.
  auto leader = fresh();
  if (!leader || !leader->Feed(prefix, {}, &error)) {
    return fail_with("leader prefill");
  }
  auto shared = leader->SaveSnapshot(&error);
  if (!shared) {
    return fail_with("shared snapshot");
  }
  leader.reset();

  // References: each task alone on an idle session.
  std::vector<Expected> expected(S);
  {
    auto ref = fresh();
    for (std::uint32_t i = 0; i < S; ++i) {
      auto& e = expected[i];
      if (!ref || !ref->RestoreSnapshot(*shared, &error) ||
          !ref->Feed(std::span(tails[i]).first(c1), {}, &error)) {
        return fail_with("reference prefill " + std::to_string(i));
      }
      auto snap = ref->SaveSnapshot(&error);
      if (!snap || !ref->Feed(std::span(tails[i]).subspan(c1), {}, &error)) {
        return fail_with("reference continue " + std::to_string(i));
      }
      e.snapshot.assign(snap->bytes().begin(), snap->bytes().end());
      for (std::uint32_t k = 0; k < K; ++k) {
        const std::int32_t token = ref->Argmax();
        e.tokens.push_back(token);
        if (!ref->Evaluate(token, &error)) {
          return fail_with("reference decode " + std::to_string(i));
        }
      }
      e.logits_after_greedy.assign(ref->Logits().begin(), ref->Logits().end());
      for (const std::int32_t token : forced) {
        if (!ref->Evaluate(token, &error)) {
          return fail_with("reference forced decode " + std::to_string(i));
        }
      }
      if (!Take(*ref, layers, &e.state, &error)) {
        return fail_with("reference capture " + std::to_string(i));
      }
    }
  }

  // A capture that does break (the test hook synchronizes the stream from the
  // capturing thread) must leave the executor usable: the step runs eagerly
  // with the same bits and the next step captures again. ROCm keeps such a
  // stream in capture mode, which used to fail that step and every later one.
  {
    auto s = fresh();
    const std::uint64_t replays = model->GraphReplays();
    bool ok = s && s->RestoreSnapshot(*shared, &error) &&
              s->Feed(std::span(tails[0]).first(c1), {}, &error) &&
              s->Feed(std::span(tails[0]).subspan(c1), {}, &error);
    model->BreakCapturesForTest(1);
    std::vector<std::int32_t> tokens;
    std::vector<float> logits_after_greedy;
    for (std::uint32_t k = 0; ok && k < K; ++k) {
      const std::int32_t token = s->Argmax();
      tokens.push_back(token);
      ok = s->Evaluate(token, &error);
    }
    if (ok) {
      logits_after_greedy.assign(s->Logits().begin(), s->Logits().end());
    }
    // The reference fed the forced tokens one by one as well.
    for (std::size_t at = 0; ok && at < forced.size(); ++at) {
      ok = s->Evaluate(forced[at], &error);
    }
    model->BreakCapturesForTest(0);
    auto o = json::Value::object();
    o["capture_invalidations"] = U(model->CaptureInvalidations());
    o["graph_replays"] = U(model->GraphReplays() - replays);
    bool same = false;
    if (ok) {
      Capture got;
      if (!Take(*s, layers, &got, &error)) {
        return fail_with("capture after the broken capture");
      }
      const Diff d = Compare(got, expected[0].state);
      o = ToJson(d);
      o["capture_invalidations"] = U(model->CaptureInvalidations());
      o["graph_replays"] = U(model->GraphReplays() - replays);
      same = session_tool::Exact(d) && tokens == expected[0].tokens &&
             logits_after_greedy == expected[0].logits_after_greedy;
      o["tokens_same"] = same;
    } else {
      o["error"] = error;
      std::fprintf(stderr, "interleave_test: broken capture: %s\n",
                   error.c_str());
    }
    check("broken_capture_recovers",
          ok && same && model->CaptureInvalidations() == 1 &&
              model->GraphReplays() > replays,
          std::move(o));
  }
  const std::uint64_t injected_invalidations = model->CaptureInvalidations();

  // Fresh serving sessions every round (so decode graphs are captured during
  // the measured work). They first hold other conversations (expanded K/V
  // rows past the shared prefix's end); even ones share half of the prefix,
  // so a restore keeps part of their expansion.
  std::vector<std::unique_ptr<l3::Session>> sessions;
  const auto make_sessions = [&]() {
    sessions.clear();
    for (std::uint32_t j = 0; j < S; ++j) {
      auto s = fresh();
      std::vector<std::int32_t> other(ids.begin(), ids.begin() + P + 40);
      const std::uint32_t keep = j % 2 == 0 ? P / 2 : 0;
      for (std::size_t t = keep; t < other.size(); ++t) {
        other[t] = (other[t] * (7 + static_cast<std::int32_t>(j)) + 13) % 1000;
      }
      if (!s || !s->Feed(other, {}, &error)) {
        return false;
      }
      sessions.push_back(std::move(s));
    }
    return true;
  };

  // Worker snapshots of an unrelated idle session for the whole run.
  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> background_snapshots{0};
  std::atomic<std::uint64_t> background_failures{0};
  std::atomic<std::uint64_t> background_mismatches{0};
  std::unique_ptr<l3::Session> idle;
  std::vector<std::uint8_t> idle_bytes;
  std::thread worker;
  if (background) {
    idle = fresh();
    if (!idle || !idle->RestoreSnapshot(*shared, &error) ||
        !idle->Feed(std::span(tails[0]).first(c1), {}, &error)) {
      return fail_with("idle session");
    }
    auto first = idle->SaveSnapshot(&error);
    if (!first) {
      return fail_with("idle snapshot");
    }
    idle_bytes.assign(first->bytes().begin(), first->bytes().end());
    worker = std::thread([&] {
      while (!stop.load()) {
        std::string e;
        auto s = idle->SaveSnapshot(&e);
        if (!s) {
          ++background_failures;
          continue;
        }
        ++background_snapshots;
        if (!std::ranges::equal(s->bytes(), idle_bytes)) {
          ++background_mismatches;
        }
      }
    });
  }
  const auto stop_worker = [&] {
    stop = true;
    if (worker.joinable()) {
      worker.join();
    }
  };

  std::mt19937 rng(seed);
  std::uint64_t worker_snapshot_failures = 0;
  for (std::uint32_t round = 0; round < R; ++round) {
    if (!make_sessions()) {
      stop_worker();
      return fail_with("serving sessions");
    }
    for (auto& s : sessions) {
      if (!s->RestoreSnapshot(*shared, &error)) {
        stop_worker();
        return fail_with("restore shared prefix");
      }
    }
    // Per session: chunk 1, worker snapshot (the session is parked until it
    // completes), chunk 2, K greedy decode steps, the forced chunks.
    struct Progress {
      std::uint32_t step{0};
      std::uint32_t forced_at{0};
      std::vector<float> logits_after_greedy;
      std::string failure;
      std::future<std::unique_ptr<l3::SessionSnapshot>> pending;
      std::unique_ptr<l3::SessionSnapshot> snapshot;
      std::vector<std::int32_t> tokens;
    };
    std::vector<Progress> progress(S);
    const auto steps = static_cast<std::uint32_t>(3 + K + chunks.size());
    std::uint32_t done = 0;
    while (done < S) {
      std::vector<std::uint32_t> runnable;
      for (std::uint32_t i = 0; i < S; ++i) {
        auto& p = progress[i];
        if (p.step >= steps) {
          continue;
        }
        if (p.pending.valid()) {
          if (p.pending.wait_for(std::chrono::seconds(0)) !=
              std::future_status::ready) {
            continue;
          }
          p.snapshot = p.pending.get();
          if (!p.snapshot) {
            ++worker_snapshot_failures;
          }
        }
        runnable.push_back(i);
      }
      if (runnable.empty()) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      const std::uint32_t i = runnable[rng() % runnable.size()];
      auto& p = progress[i];
      auto& s = *sessions[i];
      bool ok = true;
      if (p.step == 0) {
        ok = s.Feed(std::span(tails[i]).first(c1), {}, &error);
      } else if (p.step == 1) {
        l3::Session* target = &s;
        p.pending = std::async(std::launch::async, [target] {
          std::string e;
          return target->SaveSnapshot(&e);
        });
      } else if (p.step == 2) {
        ok = s.Feed(std::span(tails[i]).subspan(c1), {}, &error);
      } else if (p.step < 3 + K) {
        const std::int32_t token = s.Argmax();
        p.tokens.push_back(token);
        ok = s.Evaluate(token, &error);
        if (ok && p.step == 2 + K) {
          p.logits_after_greedy.assign(s.Logits().begin(), s.Logits().end());
        }
      } else {
        const std::uint32_t n = chunks[p.step - 3 - K];
        ok = s.Feed(std::span(forced).subspan(p.forced_at, n), {}, &error);
        p.forced_at += n;
      }
      if (!ok) {
        // A failed work unit fails its request (the server answers 500 and
        // resets the state); keep the other sessions running so a silent
        // corruption next to the failure is still caught.
        p.failure = "step " + std::to_string(p.step) + ": " + error;
        p.step = steps;
        ++done;
        continue;
      }
      if (++p.step == steps) {
        ++done;
      }
    }
    for (auto& p : progress) {
      if (p.pending.valid()) {
        p.snapshot = p.pending.get();
      }
    }
    for (std::uint32_t i = 0; i < S; ++i) {
      if (!progress[i].failure.empty()) {
        auto o = json::Value::object();
        o["round"] = U(round);
        o["session"] = U(i);
        o["error"] = progress[i].failure;
        std::fprintf(stderr, "interleave_test: round %u session %u: %s\n",
                     round, i, progress[i].failure.c_str());
        check("round_" + std::to_string(round) + "_session_" +
                  std::to_string(i) + "_vs_alone",
              false, std::move(o));
        continue;
      }
      Capture got;
      if (!Take(*sessions[i], layers, &got, &error)) {
        stop_worker();
        return fail_with("capture round " + std::to_string(round));
      }
      const Diff d = Compare(got, expected[i].state);
      auto o = ToJson(d);
      const bool tokens_same =
          progress[i].tokens == expected[i].tokens &&
          progress[i].logits_after_greedy == expected[i].logits_after_greedy;
      const bool snapshot_same =
          progress[i].snapshot != nullptr &&
          std::ranges::equal(progress[i].snapshot->bytes(),
                             expected[i].snapshot);
      o["round"] = U(round);
      o["session"] = U(i);
      o["tokens_same"] = tokens_same;
      o["worker_snapshot_same"] = snapshot_same;
      o["worker_snapshot_taken"] = progress[i].snapshot != nullptr;
      check("round_" + std::to_string(round) + "_session_" + std::to_string(i) +
                "_vs_alone",
            session_tool::Exact(d) && tokens_same && snapshot_same,
            std::move(o));
    }
  }
  stop_worker();
  {
    auto o = json::Value::object();
    o["background_snapshots"] = U(background_snapshots.load());
    o["background_failures"] = U(background_failures.load());
    o["background_mismatches"] = U(background_mismatches.load());
    o["worker_snapshot_failures"] = U(worker_snapshot_failures);
    o["graph_replays"] = U(model->GraphReplays());
    o["capture_invalidations"] =
        U(model->CaptureInvalidations() - injected_invalidations);
    check("worker_snapshots_never_differ_or_fail",
          background_mismatches.load() == 0 &&
              background_failures.load() == 0 &&
              worker_snapshot_failures == 0 &&
              model->CaptureInvalidations() == injected_invalidations &&
              model->GraphReplays() > 0,
          std::move(o));
  }

  report.Set("checks", std::move(checks));
  report.Set("failures", failures);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

}  // namespace gufo::tests::ling::interleave_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "interleave_test",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model = gufo::tests::ling::LoadModel(args, report, 512, 2048);
        return model ? gufo::tests::ling::interleave_tool::Run(args, report,
                                                               model)
                     : 1;
      });
}
