// Full-model batch-invariance checks of the batched decode step
// (Session::EvaluateBatch over rocm::Executor::ForwardBatch; engine.hpp, batch
// invariance). Eight sessions at mixed depths (1 to 2048 prompt tokens, the
// 2048-key expanded K/V block edge crossed while decoding), some prefilled in
// place, some restored from snapshots into sessions that held another
// context, two forced to route like a peer (same history; same or
// alternating tokens), with and without persistent expanded K/V stores, take
// their decode tokens in cohorts of 1..8 sessions in varying order (all eight
// forward and reversed, every width 2..7, random subsets), interleaved with
// plain Evaluate steps. Every lane's logits after every step and its whole
// final state (the snapshot payload: tokens, logits, KDA state and conv
// windows, every MLA cache row, MTP rows and carry) must be byte-identical to
// the same tokens evaluated on that session alone. Then, per lane, a
// prefill-family continuation (and with --mtp a speculative decode step) after
// the batched steps must still equal the solo run; and the failure paths: a
// lane refused before launch (bad token, full context, duplicate, invalid
// session, a ninth lane) fails alone while its peers still match (a lane
// only the batch refuses is deferred, untouched), and a pass that fails after
// launch, at the final wait or right after the trunk launches, empties every
// member without device calls and leaves them reusable. That failure comes
// from Model::FailBatchesForTest: it checks the bookkeeping only, not a
// device fault (tests/functional/device_loss.py, batched-* cases, injects HIP
// errors into a batched serving decode). Last, through the real server stack
// (InferenceBackend: scheduler, runner pool, Ling runner), four concurrent
// greedy completions answer as they do one at a time, in batched cohorts
// (with --mtp: serially, as before batching).
// Exit 77 without --model.
//
//   batch_test --model FIRST_SHARD.gguf [--mtp] [--layers N]
//              [--max-context N] [--max-batch N] [--decode-tokens K]
//              [--expanded-kv-bytes N] [--timeout-seconds S]

#include <latch>
#include <numeric>
#include <thread>

#include "src/cli/serve/inference_backend.hpp"

#define LING_TOOL_NO_MAIN
#include "tests/models/ling_3_flash/session_test.cpp"

namespace gufo::tests::ling::batch_tool {

using Bytes = std::vector<std::uint8_t>;

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
  std::uint64_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  std::size_t Below(std::size_t n) {
    return static_cast<std::size_t>(Next() % n);
  }
};

/// The whole state of a session: its snapshot payload.
bool State(const l3::Session& s, Bytes* out, std::string* error) {
  auto snap = s.SaveSnapshot(error);
  if (!snap) {
    return false;
  }
  out->assign(snap->bytes().begin(), snap->bytes().end());
  return true;
}

/// Name of the first snapshot section where two payloads differ.
std::string FirstDiff(const Bytes& a, const Bytes& b, bool mtp) {
  if (a.size() != b.size()) {
    return "size";
  }
  const auto at = std::mismatch(a.begin(), a.end(), b.begin()).first;
  if (at == a.end()) {
    return "";
  }
  const auto offset = static_cast<std::size_t>(at - a.begin());
  l3::SnapshotHeader header{};
  std::memcpy(&header, a.data(), sizeof(header));
  const l3::SnapshotSections s = l3::SnapshotLayout(header.position, mtp);
  if (offset < s.tokens)
    return "header";
  if (offset < s.logits)
    return "tokens";
  if (offset < s.kda_state)
    return "logits";
  if (offset < s.kda_conv)
    return "kda_state";
  if (offset < s.mla)
    return "kda_conv";
  if (!mtp || offset < s.mtp_h)
    return "mla_cache";
  if (offset < s.mtp)
    return "mtp_carry";
  return "mtp_cache";
}

l3::Session::BatchLane Item(l3::Session* session, std::int32_t token) {
  l3::Session::BatchLane lane;
  lane.session = session;
  lane.token = token;
  return lane;
}

bool SameLogits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && !a.empty() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

struct Lane {
  std::string kind;
  std::uint32_t prompt{0};
  std::unique_ptr<l3::Session> s;
  std::unique_ptr<l3::SessionSnapshot> start;
  std::vector<std::int32_t> decode;
  std::vector<std::vector<float>> solo_logits;  // after solo step j
  Bytes solo_end;
  std::uint32_t steps{0};
  std::uint32_t batched_steps{0};
  std::uint32_t solo_steps{0};
  std::uint32_t mismatches{0};
  std::uint32_t failures{0};
};

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  const auto K = static_cast<std::uint32_t>(args.Num("--decode-tokens", 8));
  const std::uint32_t ctx = model->MaxContext();
  const bool mtp = model->HasMtp();
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
    report.Set("checks", checks);
  };
  const auto fatal = [&](const std::string& what) {
    std::fprintf(stderr, "batch_test: %s: %s\n", what.c_str(), error.c_str());
    report.Set("status", "error");
    report.Set("error", what + ": " + error);
    return 1;
  };

  // Token stream: real text, so routing and attention look like serving.
  std::string text;
  std::vector<std::int32_t> stream;
  while (stream.size() < 6000 + 16 * K) {
    text += session_tool::kText;
    stream = model->Tokenize(text);
  }
  const auto slice = [&](std::size_t from, std::size_t n) {
    return std::vector<std::int32_t>(
        stream.begin() + static_cast<long>(from),
        stream.begin() + static_cast<long>(from + n));
  };
  // depth, kind (live: prefilled in place; restored: prefilled elsewhere and
  // restored over another context; collision: restored from a peer's start
  // with the same / alternating decode tokens).
  struct Spec {
    std::uint32_t prompt;
    const char* kind;
    int peer;  // collision source lane
  };
  const std::uint32_t edge = std::min<std::uint32_t>(2046, ctx - K - 32);
  const std::vector<Spec> specs = {
      {edge, "live", -1},
      {1, "live", -1},
      {std::min<std::uint32_t>(2047, ctx - K - 32), "restored", -1},
      {9, "live", -1},
      {std::min<std::uint32_t>(1200, ctx - K - 32), "live", -1},
      {std::min<std::uint32_t>(2048, ctx - K - 32), "restored", -1},
      {edge, "collision_same", 0},
      {std::min<std::uint32_t>(1200, ctx - K - 32), "collision_alternate", 4},
  };
  const std::size_t L = specs.size();
  std::vector<Lane> lanes(L);
  // Expanded K/V blocks go first come first served: lanes 0 and 4 (the deep
  // live ones) are created first so that, with a small budget, they hold a
  // store and the rest do not.
  for (const std::size_t i : {0U, 4U, 1U, 2U, 3U, 5U, 6U, 7U}) {
    lanes[i].s = model->CreateSession(ctx, &error);
    if (!lanes[i].s) {
      return fatal("create session");
    }
  }
  auto solo = model->CreateSession(ctx, &error);
  if (!solo) {
    return fatal("create solo session");
  }
  // Setup: every lane's start state and start snapshot.
  for (std::size_t i = 0; i < L; ++i) {
    Lane& lane = lanes[i];
    const Spec& spec = specs[i];
    lane.kind = spec.kind;
    lane.prompt = spec.prompt;
    const std::size_t base = 97 * i;
    lane.decode = slice(base + spec.prompt, K);
    if (spec.peer >= 0) {
      const Lane& peer = lanes[static_cast<std::size_t>(spec.peer)];
      lane.decode = peer.decode;
      if (lane.kind == "collision_alternate") {
        for (std::uint32_t j = 1; j < K; j += 2) {
          lane.decode[j] = stream[4096 + 13 * j];
        }
      }
      if (!lane.s->Feed(slice(3000, 40), {}, &error) ||
          !lane.s->RestoreSnapshot(*peer.start, &error)) {
        return fatal("collision setup");
      }
    } else if (lane.kind == "restored") {
      solo->Reset();
      if (!lane.s->Feed(slice(3100, 50), {}, &error) ||
          !solo->Feed(slice(base, spec.prompt), {}, &error)) {
        return fatal("restored setup");
      }
      auto snap = solo->SaveSnapshot(&error);
      if (!snap || !lane.s->RestoreSnapshot(*snap, &error)) {
        return fatal("restored setup snapshot");
      }
    } else if (!lane.s->Feed(slice(base, spec.prompt), {}, &error)) {
      return fatal("prefill");
    }
    lane.start = lane.s->SaveSnapshot(&error);
    if (!lane.start) {
      return fatal("start snapshot");
    }
  }
  // Solo references: the same tokens on one session alone (Evaluate, graph
  // replays as in serving).
  for (Lane& lane : lanes) {
    if (!solo->RestoreSnapshot(*lane.start, &error)) {
      return fatal("solo restore");
    }
    for (std::uint32_t j = 0; j < K; ++j) {
      if (!solo->Evaluate(lane.decode[j], &error)) {
        return fatal("solo evaluate");
      }
      lane.solo_logits.emplace_back(solo->Logits().begin(),
                                    solo->Logits().end());
    }
    if (!State(*solo, &lane.solo_end, &error)) {
      return fatal("solo state");
    }
  }

  // ---- schedule: cohorts of every width, orders, interleaved solo steps -----
  Rng rng(20261009);
  std::vector<bool> width_seen(L + 1, false);
  bool reversed_seen = false;
  std::uint32_t cohorts = 0;
  const auto remaining = [&](std::size_t i) { return K - lanes[i].steps; };
  const auto run_cohort = [&](const std::vector<std::size_t>& ids) {
    std::vector<l3::Session::BatchLane> batch;
    for (const std::size_t i : ids) {
      batch.push_back(Item(lanes[i].s.get(), lanes[i].decode[lanes[i].steps]));
    }
    l3::Session::EvaluateBatch(batch);
    for (std::size_t j = 0; j < ids.size(); ++j) {
      Lane& lane = lanes[ids[j]];
      if (!batch[j].ok) {
        std::fprintf(stderr, "lane %zu batched step %u failed: %s\n", ids[j],
                     lane.steps, batch[j].error.c_str());
        ++lane.failures;
        lane.steps = K;  // the lane is done (and fails its check)
        continue;
      }
      if (!SameLogits(lane.s->Logits(), lane.solo_logits[lane.steps])) {
        ++lane.mismatches;
        std::fprintf(stderr,
                     "lane %zu step %u (cohort of %zu): logits differ\n",
                     ids[j], lane.steps, ids.size());
      }
      ++lane.steps;
      ++lane.batched_steps;
    }
    width_seen[ids.size()] = true;
    ++cohorts;
  };
  const auto solo_step = [&](std::size_t i) {
    Lane& lane = lanes[i];
    if (!lane.s->Evaluate(lane.decode[lane.steps], &error)) {
      ++lane.failures;
      lane.steps = K;
      return;
    }
    if (!SameLogits(lane.s->Logits(), lane.solo_logits[lane.steps])) {
      ++lane.mismatches;
    }
    ++lane.steps;
    ++lane.solo_steps;
  };
  // Lanes with the most steps left first (random ties), in random order.
  const auto pick = [&](std::size_t want) {
    std::vector<std::size_t> c;
    for (std::size_t i = 0; i < L; ++i) {
      if (remaining(i) > 0) {
        c.push_back(i);
      }
    }
    for (std::size_t i = c.size(); i > 1; --i) {
      std::swap(c[i - 1], c[rng.Below(i)]);
    }
    std::stable_sort(c.begin(), c.end(), [&](std::size_t a, std::size_t b) {
      return remaining(a) > remaining(b);
    });
    c.resize(std::min(want, c.size()));
    for (std::size_t i = c.size(); i > 1; --i) {
      std::swap(c[i - 1], c[rng.Below(i)]);
    }
    return c;
  };
  if (K >= 2) {
    std::vector<std::size_t> all(L);
    std::iota(all.begin(), all.end(), 0);
    run_cohort(all);
    std::reverse(all.begin(), all.end());
    run_cohort(all);
    reversed_seen = true;
  }
  for (std::size_t w = L - 1; w >= 1; --w) {
    const auto ids = pick(w);
    if (ids.size() == w) {
      run_cohort(ids);
    }
  }
  while (true) {
    const auto ids = pick(1 + rng.Below(L));
    if (ids.empty()) {
      break;
    }
    run_cohort(ids);
    // Some lanes outside the cohort take a plain (graph-replayed) step.
    for (std::size_t i = 0; i < L; ++i) {
      if (remaining(i) > 0 &&
          std::find(ids.begin(), ids.end(), i) == ids.end() &&
          rng.Below(4) == 0) {
        solo_step(i);
      }
    }
  }
  bool widths = reversed_seen;
  for (std::size_t w = 1; w <= L; ++w) {
    widths = widths && width_seen[w];
  }
  {
    auto o = json::Value::object();
    o["cohorts"] = U(cohorts);
    o["decode_tokens"] = U(K);
    check("schedule_covers_widths_1_to_8_and_reversed", widths, std::move(o));
  }
  auto lane_reports = json::Value::array();
  for (std::size_t i = 0; i < L; ++i) {
    Lane& lane = lanes[i];
    Bytes end;
    if (!State(*lane.s, &end, &error)) {
      return fatal("lane state");
    }
    const std::string diff = FirstDiff(end, lane.solo_end, mtp);
    auto o = json::Value::object();
    o["lane"] = U(i);
    o["kind"] = lane.kind;
    o["prompt_tokens"] = U(lane.prompt);
    o["position"] = U(lane.s->Position());
    o["expanded_store_rows"] = U(lane.s->ExpandedStoreRows());
    o["batched_steps"] = U(lane.batched_steps);
    o["solo_steps"] = U(lane.solo_steps);
    o["logit_mismatches"] = U(lane.mismatches);
    o["step_failures"] = U(lane.failures);
    o["state_first_diff"] = diff.empty() ? "none" : diff;
    lane_reports.push_back(o);
    check("lane" + std::to_string(i) + "_" + lane.kind + "_equals_solo",
          lane.steps == K && lane.mismatches == 0 && lane.failures == 0 &&
              diff.empty(),
          std::move(o));
  }
  report.Set("lanes", lane_reports);

  // ---- after the batched steps: speculative step and prefill continuation ---
  for (std::size_t i = 0; i < L; ++i) {
    Lane& lane = lanes[i];
    if (!solo->RestoreSnapshot(std::span<const std::uint8_t>(lane.solo_end),
                               &error)) {
      return fatal("solo restore end");
    }
    auto o = json::Value::object();
    o["lane"] = U(i);
    bool pass = true;
    if (mtp) {
      sampling::SamplingConfig greedy;
      l3::Session::DecodeResult a;
      l3::Session::DecodeResult b;
      lane.s->ResetDraftPolicy();
      solo->ResetDraftPolicy();
      l3::Session::DraftHooks hooks;
      hooks.fixed_drafts = 3;
      lane.s->SetDraftHooks(hooks);
      solo->SetDraftHooks(hooks);
      sampling::SamplerState sa(greedy);
      sampling::SamplerState sb(greedy);
      const bool ran = lane.s->DecodeStep(4, sa, &a, &error) &&
                       solo->DecodeStep(4, sb, &b, &error);
      lane.s->SetDraftHooks({});
      solo->SetDraftHooks({});
      o["mtp_step_tokens"] = U(a.tokens.size());
      o["mtp_step_same_tokens"] = ran && a.tokens == b.tokens;
      pass = pass && ran && a.tokens == b.tokens &&
             SameLogits(lane.s->Logits(), solo->Logits());
    }
    const auto chunk = slice(5000 + 31 * i, 24);
    Bytes x;
    Bytes y;
    const bool fed = lane.s->Feed(chunk, {}, &error) &&
                     solo->Feed(chunk, {}, &error) &&
                     State(*lane.s, &x, &error) && State(*solo, &y, &error);
    const std::string diff = fed ? FirstDiff(x, y, mtp) : "error";
    o["state_first_diff"] = diff.empty() ? "none" : diff;
    pass = pass && fed && diff.empty();
    check("lane" + std::to_string(i) + (mtp ? "_mtp_step" : "") +
              "_continuation_equals_solo",
          pass, std::move(o));
  }

  // ---- refusals before launch: the lane fails alone -------------------------
  {
    for (std::size_t i = 0; i < 4; ++i) {
      if (!lanes[i].s->RestoreSnapshot(*lanes[i].start, &error)) {
        return fatal("restore start");
      }
    }
    auto full = model->CreateSession(16, &error);
    if (!full || !full->Feed(slice(0, 16), {}, &error)) {
      return fatal("full session");
    }
    std::vector<l3::Session::BatchLane> batch = {
        Item(lanes[0].s.get(), lanes[0].decode[0]),
        Item(lanes[1].s.get(), -7),
        Item(lanes[2].s.get(), lanes[2].decode[0]),
        Item(lanes[0].s.get(), lanes[0].decode[0]),
        Item(full.get(), lanes[3].decode[0]),
        Item(nullptr, 5),
        Item(lanes[3].s.get(), lanes[3].decode[0]),
    };
    l3::Session::EvaluateBatch(batch);
    const bool peers =
        batch[0].ok && batch[2].ok && batch[6].ok &&
        SameLogits(lanes[0].s->Logits(), lanes[0].solo_logits[0]) &&
        SameLogits(lanes[2].s->Logits(), lanes[2].solo_logits[0]) &&
        SameLogits(lanes[3].s->Logits(), lanes[3].solo_logits[0]);
    const bool bad_token =
        !batch[1].ok && !batch[1].deferred && !lanes[1].s->IsValid();
    const bool duplicate =
        !batch[3].ok && lanes[0].s->Position() == lanes[0].prompt + 1;
    const bool overflow =
        !batch[4].ok && full->IsValid() && full->Position() == 16;
    const bool null_lane = !batch[5].ok;
    auto o = json::Value::object();
    o["bad_token_error"] = batch[1].error;
    o["duplicate_error"] = batch[3].error;
    o["overflow_error"] = batch[4].error;
    check("refused_lanes_fail_alone",
          peers && bad_token && duplicate && overflow && null_lane,
          std::move(o));
    // The invalid session is refused on its next step; its peer still runs.
    std::vector<l3::Session::BatchLane> next = {
        Item(lanes[1].s.get(), lanes[1].decode[0]),
        Item(lanes[2].s.get(), lanes[2].decode[1]),
    };
    l3::Session::EvaluateBatch(next);
    auto o2 = json::Value::object();
    o2["invalid_error"] = next[0].error;
    check("invalid_session_refused_peer_runs",
          !next[0].ok && next[1].ok &&
              SameLogits(lanes[2].s->Logits(), lanes[2].solo_logits[1]),
          std::move(o2));
  }
  // A ninth lane is refused; the eight others run (and match from restored
  // starts).
  {
    for (Lane& lane : lanes) {
      if (!lane.s->RestoreSnapshot(*lane.start, &error)) {
        return fatal("restore start");
      }
    }
    if (!solo->RestoreSnapshot(*lanes[0].start, &error)) {
      return fatal("restore solo");
    }
    std::vector<l3::Session::BatchLane> batch;
    for (Lane& lane : lanes) {
      batch.push_back(Item(lane.s.get(), lane.decode[0]));
    }
    batch.push_back(Item(solo.get(), lanes[0].decode[0]));
    l3::Session::EvaluateBatch(batch);
    bool eight = true;
    for (std::size_t i = 0; i < L; ++i) {
      eight = eight && batch[i].ok &&
              SameLogits(lanes[i].s->Logits(), lanes[i].solo_logits[0]);
    }
    auto o = json::Value::object();
    o["ninth_error"] = batch[L].error;
    check("ninth_lane_refused_eight_run",
          eight && !batch[L].ok && batch[L].deferred && solo->IsValid() &&
              solo->Position() == lanes[0].prompt,
          std::move(o));
  }

  // ---- failure after launch (device-loss shape): every member is emptied ----
  for (const bool before_wait : {false, true}) {
    const std::string point = before_wait ? "_before_wait" : "_after_wait";
    for (std::size_t i = 0; i < 4; ++i) {
      if (!lanes[i].s->RestoreSnapshot(*lanes[i].start, &error)) {
        return fatal("restore start");
      }
    }
    Bytes outsider_before;
    Bytes outsider_after;
    if (!State(*lanes[3].s, &outsider_before, &error)) {
      return fatal("outsider state");
    }
    model->FailBatchesForTest(1, before_wait);
    std::vector<l3::Session::BatchLane> batch = {
        Item(lanes[0].s.get(), lanes[0].decode[0]),
        Item(lanes[1].s.get(), lanes[1].decode[0]),
        Item(lanes[2].s.get(), lanes[2].decode[0]),
    };
    l3::Session::EvaluateBatch(batch);
    bool emptied = true;
    for (std::size_t i = 0; i < 3; ++i) {
      const l3::Session& s = *lanes[i].s;
      emptied = emptied && !batch[i].ok && s.IsValid() && s.Position() == 0 &&
                s.Tokens().empty() && s.Logits().empty();
    }
    const bool outsider = State(*lanes[3].s, &outsider_after, &error) &&
                          outsider_after == outsider_before;
    auto o = json::Value::object();
    o["error"] = batch[0].error;
    check("failed_pass_empties_members_only" + point, emptied && outsider,
          std::move(o));
    // Reusable: an emptied lane restores and decodes, another one starts from
    // position 0 inside a batch (the deferred reset), equal to solo.
    std::vector<std::int32_t> first = {lanes[1].decode[0]};
    if (!lanes[0].s->RestoreSnapshot(*lanes[0].start, &error)) {
      return fatal("restore after failure");
    }
    solo->Reset();
    if (!solo->Evaluate(first[0], &error)) {
      return fatal("solo from empty");
    }
    const std::vector<float> fresh(solo->Logits().begin(),
                                   solo->Logits().end());
    std::vector<l3::Session::BatchLane> again = {
        Item(lanes[0].s.get(), lanes[0].decode[0]),
        Item(lanes[1].s.get(), first[0]),
    };
    l3::Session::EvaluateBatch(again);
    check("members_reusable_after_failed_pass" + point,
          again[0].ok && again[1].ok &&
              SameLogits(lanes[0].s->Logits(), lanes[0].solo_logits[0]) &&
              SameLogits(lanes[1].s->Logits(), fresh),
          json::Value::object());
  }

  // ---- serving: the scheduler forms the batch (InferenceBackend) ------------
  // Four greedy raw completions of a fixed 64 tokens (ignore_eos, so all four
  // are still decoding when the last one starts), first one at a time, then
  // all at once. The concurrent answers must equal the serial ones; without
  // MTP every request must have decoded in batched cohorts (plan
  // "batched-wN", widest 4), with MTP too (shared draft and verify).
  {
    namespace server = gufo::server;
    constexpr std::size_t kUsers = 4;
    server::InferenceBackend backend;
    server::TextSpeculativeConfig speculative;
    if (mtp) {
      speculative.backend = server::TextSpeculativeBackend::kMtp;
    }
    if (!backend.load(model, &error, std::min<std::uint32_t>(ctx, 2048), kUsers,
                      {}, {}, speculative)) {
      return fatal("serving load");
    }
    const char* questions[kUsers] = {
        "List four rivers of Europe and one city on each.",
        "Explain in two sentences why the sky looks blue.",
        "Write a haiku about a lighthouse in winter.",
        "Name three prime numbers above one hundred.",
    };
    sampling::SamplingConfig greedy;
    greedy.temperature = 0.0F;
    const auto ask = [&](std::size_t i) {
      return backend
          .start_complete(questions[i], 64, greedy, {}, false,
                          /*ignore_eos=*/true)
          ->Wait();
    };
    std::vector<server::InferenceBackend::Result> serial;
    for (std::size_t i = 0; i < kUsers; ++i) {
      serial.push_back(ask(i));
    }
    std::vector<server::InferenceBackend::Result> together(kUsers);
    {
      std::latch go(kUsers);
      std::vector<std::thread> users;
      for (std::size_t i = 0; i < kUsers; ++i) {
        users.emplace_back([&, i] {
          go.arrive_and_wait();
          together[i] = ask(i);
        });
      }
      for (auto& user : users) {
        user.join();
      }
    }
    bool same = true;
    bool batched = true;
    std::size_t widest = 0;
    auto plans = json::Value::array();
    for (std::size_t i = 0; i < kUsers; ++i) {
      same = same && serial[i].completion_tokens == 64 &&
             together[i].tokens == serial[i].tokens &&
             together[i].text == serial[i].text;
      batched = batched && together[i].execution_plan.starts_with("batched-w");
      widest = std::max(widest, together[i].physical_execution_width);
      plans.push_back(together[i].execution_plan);
    }
    auto o = json::Value::object();
    o["plans"] = plans;
    o["widest"] = U(widest);
    o["completion_tokens"] = U(together[0].completion_tokens);
    o["same_answers"] = same;
    check(mtp ? "serving_concurrent_mtp_equals_serial"
              : "serving_concurrent_batched_equals_serial",
          same && batched && widest == kUsers, std::move(o));
  }

  report.Set("checks", checks);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

int Body(const Args& args, Report& report) {
  // A small store: the first two sessions get expanded K/V blocks, the rest
  // expand per call (--expanded-kv-bytes 0: none has one).
  constexpr std::uint64_t kBlockBytes = std::uint64_t{2048} * 112 * 1024;
  auto model = LoadModel(args, report, 512, 2304, false, 3 * kBlockBytes);
  if (!model) {
    return 1;
  }
  return Run(args, report, model);
}

}  // namespace gufo::tests::ling::batch_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(argc, argv, "batch_test",
                                    gufo::tests::ling::batch_tool::Body);
}
