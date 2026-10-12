// Time-to-first-token cost of capturing a live checkpoint on Ling-3.0-flash,
// as the server's resume path does it: a session holding `--prompt-tokens`
// tokens is snapshotted (Session::SaveSnapshot), then the request's new tokens
// (`--tail-tokens`) are fed and the first logits are read. Reports, over
// `--reps` repetitions, the milliseconds of the capture call, of the tail
// feed after it, of their sum (what the first token waits for) and of the time
// until the snapshot's bytes are usable. `control` is the tail feed with no
// capture. Exit 77 without --model.
//
//   snapshot_ttft_bench --model FIRST_SHARD.gguf [--prompt-tokens N (16384)]
//       [--tail-tokens K (32)] [--reps R (8)] [--max-context N] [--layers N]
//       [--mtp]

#define LING_TOOL_NO_MAIN
#include <algorithm>
#include <chrono>
#include <thread>

#include "tests/models/ling_3_flash/session_test.cpp"

namespace gufo::tests::ling::ttft_tool {

using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

json::Value Stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto o = json::Value::object();
  o["min"] = Round(v.front(), 3);
  o["median"] = Round(v[v.size() / 2], 3);
  o["max"] = Round(v.back(), 3);
  return o;
}

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  const auto N = static_cast<std::uint32_t>(args.Num("--prompt-tokens", 16384));
  const auto K = static_cast<std::uint32_t>(args.Num("--tail-tokens", 32));
  const auto reps = static_cast<int>(args.Num("--reps", 8));
  std::vector<std::int32_t> unit = model->Tokenize(session_tool::kText);
  std::vector<std::int32_t> ids;
  while (ids.size() < N + K) {
    ids.insert(ids.end(), unit.begin(), unit.end());
  }
  ids.resize(N + K);
  std::string error;
  auto session = model->CreateSession(model->MaxContext(), &error);
  if (!session || !session->Feed(std::span(ids).subspan(0, N), {}, &error)) {
    std::fprintf(stderr, "ttft_bench: prefill: %s\n", error.c_str());
    return 1;
  }
  // The restore point for every repetition (waits for the copy if async).
  auto base = session->SaveSnapshot(&error);
  if (!base || base->bytes().size() != base->SizeBytes()) {
    std::fprintf(stderr, "ttft_bench: base snapshot: %s\n", error.c_str());
    return 1;
  }
  const auto tail = std::span(ids).subspan(N, K);
  std::vector<double> capture, feed, total, ready, control;
  for (int rep = 0; rep < reps + 1; ++rep) {
    for (const bool capture_first : {false, true}) {
      if (!session->RestoreSnapshot(*base, &error)) {
        std::fprintf(stderr, "ttft_bench: restore: %s\n", error.c_str());
        return 1;
      }
      std::unique_ptr<l3::SessionSnapshot> snap;
      const auto t0 = Clock::now();
      if (capture_first) {
        snap = session->SaveSnapshot(&error);
        if (!snap) {
          std::fprintf(stderr, "ttft_bench: capture: %s\n", error.c_str());
          return 1;
        }
      }
      const auto t1 = Clock::now();
      if (!session->Feed(tail, {}, &error)) {
        std::fprintf(stderr, "ttft_bench: feed: %s\n", error.c_str());
        return 1;
      }
      const auto t2 = Clock::now();
      if (snap && snap->bytes().size() != snap->SizeBytes()) {
        std::fprintf(stderr, "ttft_bench: snapshot incomplete\n");
        return 1;
      }
      const auto t3 = Clock::now();
      if (rep == 0) {
        continue;  // warm-up (graph capture, store growth)
      }
      if (capture_first) {
        capture.push_back(Ms(t0, t1));
        feed.push_back(Ms(t1, t2));
        total.push_back(Ms(t0, t2));
        ready.push_back(Ms(t0, t3));
      } else {
        control.push_back(Ms(t1, t2));
      }
    }
  }
  // A restore into another session while a snapshot's background row copy is
  // running (the server restores a cached prefix right after the previous
  // request captured its checkpoint): the restore waits for the copy's
  // current staging unit, so that unit bounds the added restore time.
  std::vector<double> idle_restore, busy_restore;
  {
    auto other = model->CreateSession(model->MaxContext(), &error);
    if (!other || !other->RestoreSnapshot(*base, &error)) {
      std::fprintf(stderr, "ttft_bench: second session: %s\n", error.c_str());
      return 1;
    }
    for (int rep = 0; rep < reps + 1; ++rep) {
      for (const bool busy : {false, true}) {
        std::unique_ptr<l3::SessionSnapshot> snap;
        if (busy) {
          snap = session->SaveSnapshot(&error);
          if (!snap) {
            return 1;
          }
          // Let the worker start its first unit.
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        const auto t0 = Clock::now();
        if (!other->RestoreSnapshot(*base, &error)) {
          std::fprintf(stderr, "ttft_bench: restore: %s\n", error.c_str());
          return 1;
        }
        const auto t1 = Clock::now();
        if (snap && !snap->Wait(&error)) {
          return 1;
        }
        if (rep > 0) {
          (busy ? busy_restore : idle_restore).push_back(Ms(t0, t1));
        }
      }
    }
  }
  // The copy alone, and under a storm of restores into another session (the
  // copy must still complete: it yields once per staging unit, not until the
  // transfers stop).
  std::vector<double> copy_alone, copy_storm, storm_restores;
  {
    auto other = model->CreateSession(model->MaxContext(), &error);
    if (!other || !other->RestoreSnapshot(*base, &error)) {
      return 1;
    }
    for (int rep = 0; rep < reps + 1; ++rep) {
      auto t0 = Clock::now();
      auto snap = session->SaveSnapshot(&error);
      if (!snap || !snap->Wait(&error)) {
        return 1;
      }
      const double alone = Ms(t0, Clock::now());
      t0 = Clock::now();
      snap = session->SaveSnapshot(&error);
      if (!snap) {
        return 1;
      }
      int restores = 0;
      while (!snap->Ready()) {
        if (!other->RestoreSnapshot(*base, &error)) {
          return 1;
        }
        ++restores;
      }
      const double storm = Ms(t0, Clock::now());
      if (!snap->Wait(&error)) {
        return 1;
      }
      if (rep > 0) {
        copy_alone.push_back(alone);
        copy_storm.push_back(storm);
        storm_restores.push_back(restores);
      }
    }
  }
  auto o = json::Value::object();
  o["row_copy_alone_ms"] = Stats(copy_alone);
  o["row_copy_under_restores_ms"] = Stats(copy_storm);
  o["restores_during_copy"] = Stats(storm_restores);
  std::fprintf(stderr,
               "[copy] alone=%.1f under_restore_storm=%.1f (restores %.0f) "
               "(median ms)\n",
               copy_alone[copy_alone.size() / 2],
               copy_storm[copy_storm.size() / 2],
               storm_restores[storm_restores.size() / 2]);
  o["idle_restore_ms"] = Stats(idle_restore);
  o["restore_during_row_copy_ms"] = Stats(busy_restore);
  std::fprintf(stderr, "[restore] idle=%.1f during_row_copy=%.1f (median ms)\n",
               idle_restore[idle_restore.size() / 2],
               busy_restore[busy_restore.size() / 2]);
  o["prompt_tokens"] = U(N);
  o["tail_tokens"] = U(K);
  o["reps"] = U(static_cast<std::uint64_t>(reps));
  o["snapshot_bytes"] = U(base->SizeBytes());
  o["capture_ms"] = Stats(capture);
  o["feed_after_capture_ms"] = Stats(feed);
  o["first_token_wait_ms"] = Stats(total);
  o["snapshot_usable_ms"] = Stats(ready);
  o["control_feed_ms"] = Stats(control);
  std::fprintf(stderr,
               "[ttft] N=%u K=%u capture=%.1f feed=%.1f first_token=%.1f "
               "usable=%.1f control=%.1f (median ms)\n",
               N, K, capture[capture.size() / 2], feed[feed.size() / 2],
               total[total.size() / 2], ready[ready.size() / 2],
               control[control.size() / 2]);
  report.Set("ttft", std::move(o));
  report.Set("status", "pass");
  return 0;
}

}  // namespace gufo::tests::ling::ttft_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "snapshot_ttft_bench",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model = gufo::tests::ling::LoadModel(args, report, 512, 32768);
        return model ? gufo::tests::ling::ttft_tool::Run(args, report, model)
                     : 1;
      });
}
