// Full-model checks of batched MTP decoding (Session::DecodeBatch: shared
// draft steps, and verify forwards of at most 8 rows with per-lane rollback).
// Eight sessions on distinct real-text prompts decode T tokens each:
//   * greedy cohorts of 2, 3, 4 and 8 lanes with mixed draft widths (fixed
//     1/3/7 and the adaptive cohort width): every lane's tokens equal plain
//     greedy decoding of that session alone (Evaluate(Argmax())), and its
//     whole final state (snapshot payload: KDA state, conv windows, MLA rows,
//     MTP rows and carry, logits) is byte-identical to that run's;
//   * seeded sampled lanes, alone and mixed with greedy lanes (cohorts whose
//     own-history widths overflow one 8-row verify split it): tokens and final
//     state equal DecodeStep of the same seed alone;
//   * lanes that verify prompt-lookup copies instead of drafting
//     (DraftHooks::copy_source replays a lane's own continuation with
//     mistakes, alternating with MTP cycles on some lanes; the greedy
//     reference is plain AR, the sampled one DecodeStep alone with the same
//     hook): tokens and whole final state equal, in cohorts whose copy lanes
//     overflow one 8-row verify and split it;
//   * greedy lanes through the sampler (a repetition penalty: verify rows and
//     VerifyDraft) and lanes stopping at a stop token (DraftHooks::stop_token,
//     the stop paths of a prompt that reaches EOS) equal DecodeStep alone;
//   * a lane DecodeStep would refuse fails alone, a ninth lane is deferred
//     untouched, a verify forward that fails after launch
//     (Model::FailBatchesForTest, bookkeeping only) empties its lanes, which
//     are reusable afterwards, and a failed draft step
//     (Model::FailDraftBatchForTest) fails only the chains it carried.
// Exit 77 without --model.
//
//   batch_mtp_test --model FIRST_SHARD.gguf [--layers N] [--max-context N]
//                  [--tokens T]

#include <cstring>

#define LING_TOOL_NO_MAIN
#include "tests/models/ling_3_flash/batch_lanes.hpp"
#include "tests/models/ling_3_flash/session_test.cpp"

namespace gufo::tests::ling::batch_mtp_tool {

sampling::SamplingConfig Greedy() {
  sampling::SamplingConfig c;
  c.temperature = 0.0F;
  return c;
}

sampling::SamplingConfig Sampled(std::uint64_t seed) {
  sampling::SamplingConfig c;
  c.temperature = 0.8F;
  c.top_p = 0.95F;
  c.top_k = 40;
  c.seed = seed;
  return c;
}

struct Outcome {
  std::vector<std::int32_t> tokens;
  Bytes end;
};

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model) {
  const auto T = static_cast<std::uint32_t>(args.Num("--tokens", 24));
  const std::uint32_t ctx = model->MaxContext();
  constexpr std::size_t kLanes = 8;
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
    std::fprintf(stderr, "batch_mtp_test: %s: %s\n", what.c_str(),
                 error.c_str());
    report.Set("status", "error");
    report.Set("error", what + ": " + error);
    return 1;
  };
  std::string text;
  std::vector<std::int32_t> stream;
  while (stream.size() < 4000) {
    text += session_tool::kText;
    stream = model->Tokenize(text);
  }
  // Start states: distinct prompts of 40..600 tokens.
  std::vector<std::unique_ptr<l3::Session>> lanes;
  std::vector<std::unique_ptr<l3::SessionSnapshot>> start;
  for (std::size_t i = 0; i < kLanes + 1; ++i) {
    auto s = model->CreateSession(ctx, &error);
    const std::size_t n = 40 + 70 * i;
    const std::vector<std::int32_t> prompt(
        stream.begin() + static_cast<long>(37 * i),
        stream.begin() + static_cast<long>(37 * i + n));
    if (!s || !s->Feed(prompt, {}, &error)) {
      return fatal("prefill");
    }
    start.push_back(s->SaveSnapshot(&error));
    if (!start.back()) {
      return fatal("snapshot");
    }
    lanes.push_back(std::move(s));
  }
  auto solo = model->CreateSession(ctx, &error);
  if (!solo) {
    return fatal("solo session");
  }
  // Greedy AR reference per lane: Evaluate(Argmax()) alone.
  std::vector<Outcome> ar(kLanes);
  for (std::size_t i = 0; i < kLanes; ++i) {
    if (!solo->RestoreSnapshot(*start[i], &error)) {
      return fatal("restore");
    }
    for (std::uint32_t t = 0; t < T; ++t) {
      const std::int32_t next = solo->Argmax();
      ar[i].tokens.push_back(next);
      if (!solo->Evaluate(next, &error)) {
        return fatal("AR reference");
      }
    }
    if (!State(*solo, &ar[i].end, &error)) {
      return fatal("AR state");
    }
  }
  // Lane kinds: greedy (device-argmax fast path), seeded sampled, and greedy
  // through the sampler (a repetition penalty: the verify rows and
  // VerifyDraft path).
  enum Kind { kGreedy, kSampled, kPenalty };
  const auto config_of = [](Kind kind, std::size_t i) {
    if (kind == kSampled) {
      return Sampled(1000 + i);
    }
    sampling::SamplingConfig c = Greedy();
    if (kind == kPenalty) {
      c.repeat_penalty = 1.3F;
      c.repeat_last_n = 16;
    }
    return c;
  };
  // Forced copies for lane i: its AR continuation, every fifth token wrong;
  // `alternate` leaves every other 8 committed tokens to the MTP drafter. A
  // function of the lane's own committed tokens, so alone and batched agree.
  const auto copy_hook = [&](const l3::Session* s, std::size_t i,
                             bool alternate) {
    const auto base = static_cast<std::size_t>(start[i]->Position());
    return [s, base, i, alternate,
            &ar](std::span<const std::int32_t> chain) -> std::int32_t {
      const std::size_t committed = s->Tokens().size() - base;
      if (alternate && (committed / 8) % 2 == 1) {
        return -1;
      }
      const std::size_t at = committed + chain.size();
      if (at >= ar[i].tokens.size()) {
        return -1;
      }
      const std::int32_t truth = ar[i].tokens[at];
      return at % 5 == 4 ? (truth + 1) % static_cast<std::int32_t>(l3::kVocab)
                         : truth;
    };
  };
  // Runs a lane alone with DecodeStep, using the lane's sampler, fixed width (0
  // is adaptive and is part of a sampled lane's seeded trace), and stop token,
  // until T tokens or a stop.
  const auto solo_run = [&](std::size_t i, Kind kind, std::uint32_t fixed,
                            std::int32_t stop_token, Outcome* out,
                            int copy = 0) {
    *out = {};
    if (!solo->RestoreSnapshot(*start[i], &error)) {
      return false;
    }
    solo->ResetDraftPolicy();
    l3::Session::DraftHooks hooks;
    hooks.fixed_drafts = fixed;
    hooks.stop_token = stop_token;
    if (copy != 0) {
      hooks.copy_source = copy_hook(solo.get(), i, copy == 2);
    }
    solo->SetDraftHooks(hooks);
    sampling::SamplerState sampler(config_of(kind, i));
    bool ok = true;
    while (ok && out->tokens.size() < T) {
      l3::Session::DecodeResult r;
      ok = solo->DecodeStep(T - out->tokens.size(), sampler, &r, &error,
                            stop_token >= 0);
      out->tokens.insert(out->tokens.end(), r.tokens.begin(), r.tokens.end());
      if (r.stop) {
        break;
      }
    }
    solo->SetDraftHooks({});
    return ok && State(*solo, &out->end, &error);
  };

  // One batched run: `members` decode T tokens (or to their stop token) in
  // cohorts; lanes that are done drop out.
  struct Plan {
    std::string name;
    std::vector<std::size_t> members;
    std::vector<std::uint32_t> fixed;  // drafts per lane, 0 = adaptive
    std::vector<Kind> kind;
    bool stop{false};  // stop at the lane's 11th token (DraftHooks::stop_token)
    /// Per lane: 0 MTP, 1 forced copies, 2 forced copies alternating with MTP.
    std::vector<int> copy;
  };
  const auto run = [&](const Plan& plan) -> int {
    const std::size_t count = plan.members.size();
    std::vector<Outcome> want(count);
    std::vector<std::int32_t> stop_token(count, -1);
    const auto copy_of = [&](std::size_t m) {
      return m < plan.copy.size() ? plan.copy[m] : 0;
    };
    for (std::size_t m = 0; m < count; ++m) {
      const std::size_t i = plan.members[m];
      if (plan.kind[m] == kGreedy && !plan.stop) {
        want[m] = ar[i];  // greedy MTP equals plain greedy decoding
        continue;
      }
      if (plan.stop) {
        Outcome free;
        if (!solo_run(i, plan.kind[m], plan.fixed[m], -1, &free)) {
          return fatal("solo reference");
        }
        stop_token[m] = free.tokens.at(10);
      }
      if (!solo_run(i, plan.kind[m], plan.fixed[m], stop_token[m], &want[m],
                    copy_of(m))) {
        return fatal("solo reference");
      }
    }
    std::vector<sampling::SamplerState> samplers;
    samplers.reserve(count);
    std::vector<std::vector<std::int32_t>> got(count);
    std::vector<bool> done(count, false);
    for (std::size_t m = 0; m < count; ++m) {
      const std::size_t i = plan.members[m];
      if (!lanes[i]->RestoreSnapshot(*start[i], &error)) {
        return fatal("restore lane");
      }
      lanes[i]->ResetDraftPolicy();
      l3::Session::DraftHooks hooks;
      hooks.fixed_drafts = plan.fixed[m];
      hooks.stop_token = stop_token[m];
      if (copy_of(m) != 0) {
        hooks.copy_source = copy_hook(lanes[i].get(), i, copy_of(m) == 2);
      }
      lanes[i]->SetDraftHooks(hooks);
      samplers.emplace_back(config_of(plan.kind[m], i));
    }
    std::uint32_t cycles = 0;
    std::uint32_t widest = 0;
    bool failed = false;
    bool stopped = false;
    std::uint64_t copied = 0;
    std::uint64_t copied_accepted = 0;
    std::vector<l3::Session::DraftStats> before(count);
    for (std::size_t m = 0; m < count; ++m) {
      before[m] = lanes[plan.members[m]]->Statistics();
    }
    while (true) {
      std::vector<l3::Session::BatchDecode> batch;
      std::vector<std::size_t> which;
      for (std::size_t m = 0; m < count; ++m) {
        if (!done[m] && got[m].size() < T) {
          l3::Session::BatchDecode d;
          d.session = lanes[plan.members[m]].get();
          d.max_tokens = T - got[m].size();
          d.sampler = &samplers[m];
          d.stop_at_eos = plan.stop;
          batch.push_back(std::move(d));
          which.push_back(m);
        }
      }
      if (batch.empty()) {
        break;
      }
      l3::Session::DecodeBatch(batch);
      ++cycles;
      for (std::size_t b = 0; b < batch.size(); ++b) {
        const std::size_t m = which[b];
        if (!batch[b].ok) {
          std::fprintf(stderr, "%s lane %zu: %s\n", plan.name.c_str(),
                       plan.members[m], batch[b].error.c_str());
          failed = true;
          done[m] = true;
          continue;
        }
        widest = std::max(widest, batch[b].width);
        got[m].insert(got[m].end(), batch[b].result.tokens.begin(),
                      batch[b].result.tokens.end());
        if (batch[b].result.stop) {
          done[m] = true;
          stopped = true;
        }
      }
    }
    bool tokens_equal = !failed;
    bool states_equal = !failed;
    for (std::size_t m = 0; m < count; ++m) {
      const std::size_t i = plan.members[m];
      tokens_equal = tokens_equal && got[m] == want[m].tokens;
      const auto after = lanes[i]->Statistics();
      copied += after.copied - before[m].copied;
      copied_accepted += after.copied_accepted - before[m].copied_accepted;
      Bytes end;
      if (!State(*lanes[i], &end, &error)) {
        return fatal("lane state");
      }
      states_equal = states_equal && end == want[m].end;
      lanes[i]->SetDraftHooks({});
    }
    auto o = json::Value::object();
    o["lanes"] = U(count);
    o["cycles"] = U(cycles);
    o["widest_forward"] = U(widest);
    o["tokens_equal"] = tokens_equal;
    o["states_equal"] = states_equal;
    o["stopped"] = stopped;
    o["copied"] = U(copied);
    o["copied_accepted"] = U(copied_accepted);
    const bool has_copies = !plan.copy.empty();
    // Greedy copies replay the greedy continuation: some must be accepted.
    bool greedy_copies = false;
    for (std::size_t m = 0; m < count; ++m) {
      greedy_copies = greedy_copies || (plan.kind[m] == kGreedy && copy_of(m));
    }
    check(plan.name,
          tokens_equal && states_equal && widest >= 2 &&
              (!plan.stop || stopped) && (!has_copies || copied > 0) &&
              (!greedy_copies || copied_accepted > 0),
          std::move(o));
    return 0;
  };
  const auto lanes_of = [](std::size_t n) {
    std::vector<std::size_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = i;
    }
    return v;
  };
  const auto all = [](std::size_t n, Kind kind) {
    return std::vector<Kind>(n, kind);
  };
  const std::vector<Plan> plans = {
      {"greedy_b2_adaptive_equals_ar", lanes_of(2), {0, 0}, all(2, kGreedy)},
      {"greedy_b3_mixed_k_equals_ar", lanes_of(3), {1, 3, 7}, all(3, kGreedy)},
      {"greedy_b4_mixed_k_equals_ar",
       lanes_of(4),
       {7, 0, 3, 1},
       all(4, kGreedy)},
      {"greedy_b8_equals_ar",
       lanes_of(8),
       {0, 1, 0, 3, 0, 7, 0, 2},
       all(8, kGreedy)},
      {"sampled_b2_equals_solo", lanes_of(2), {0, 0}, all(2, kSampled)},
      // Lanes 2 and 3 of mixed_b4 and 2, 4, 6 of mixed_b8 overflow a step.
      {"mixed_b4_equals_solo",
       lanes_of(4),
       {0, 0, 7, 3},
       {kSampled, kGreedy, kSampled, kGreedy}},
      {"mixed_b8_split_equals_solo",
       lanes_of(8),
       {0, 0, 7, 0, 7, 3, 5, 7},
       {kSampled, kGreedy, kSampled, kGreedy, kSampled, kGreedy, kSampled,
        kGreedy}},
      {"penalty_greedy_b3_equals_solo",
       lanes_of(3),
       {3, 7, 0},
       all(3, kPenalty)},
      // Forced copies (prompt lookup) in cohorts with MTP lanes.
      {"copies_greedy_b4_equals_ar",
       lanes_of(4),
       {0, 0, 0, 0},
       all(4, kGreedy),
       false,
       {1, 0, 2, 1}},
      {"copies_sampled_b3_equals_solo",
       lanes_of(3),
       {0, 0, 0},
       all(3, kSampled),
       false,
       {1, 2, 0}},
      // Copy lanes (up to 8 rows each) overflow one forward and split it.
      {"copies_mixed_b8_split_equals_solo",
       lanes_of(8),
       {0, 0, 0, 0, 0, 0, 0, 0},
       {kSampled, kGreedy, kSampled, kGreedy, kPenalty, kGreedy, kSampled,
        kGreedy},
       false,
       {1, 1, 2, 0, 1, 2, 1, 0}},
      {"stop_greedy_b4_equals_solo",
       lanes_of(4),
       {7, 3, 1, 0},
       all(4, kGreedy),
       true},
      {"stop_mixed_b3_equals_solo",
       lanes_of(3),
       {0, 3, 7},
       {kSampled, kPenalty, kSampled},
       true},
  };
  for (const Plan& plan : plans) {
    if (const int code = run(plan); code != 0) {
      return code;
    }
  }

  // ---- failures ----------------------------------------------------------
  {
    // A ninth lane is deferred untouched (eight valid lanes run) ...
    const auto greedy_batch =
        [&](std::size_t count, std::vector<sampling::SamplerState>* samplers) {
          samplers->assign(count, sampling::SamplerState(Greedy()));
          std::vector<l3::Session::BatchDecode> batch(count);
          for (std::size_t i = 0; i < count; ++i) {
            batch[i].session = lanes[i].get();
            batch[i].max_tokens = 4;
            batch[i].sampler = &(*samplers)[i];
            batch[i].stop_at_eos = false;
          }
          return batch;
        };
    const auto prefix_of_ar = [&](const l3::Session::BatchDecode& d,
                                  std::size_t i) {
      const auto& t = d.result.tokens;
      return d.ok && !t.empty() &&
             std::equal(t.begin(), t.end(), ar[i].tokens.begin());
    };
    for (std::size_t i = 0; i < kLanes + 1; ++i) {
      if (!lanes[i]->RestoreSnapshot(*start[i], &error)) {
        return fatal("restore");
      }
    }
    std::vector<sampling::SamplerState> samplers;
    auto batch = greedy_batch(kLanes + 1, &samplers);
    const std::uint32_t ninth_pos = lanes[kLanes]->Position();
    l3::Session::DecodeBatch(batch);
    bool eight = true;
    for (std::size_t i = 0; i < kLanes; ++i) {
      eight = eight && prefix_of_ar(batch[i], i);
    }
    auto o = json::Value::object();
    o["ninth_error"] = batch[kLanes].error;
    check("ninth_lane_deferred_untouched",
          eight && !batch[kLanes].ok && batch[kLanes].deferred &&
              lanes[kLanes]->Position() == ninth_pos,
          std::move(o));
    // ... and an invalid session fails alone among valid peers.
    for (std::size_t i = 0; i < 4; ++i) {
      if (!lanes[i]->RestoreSnapshot(*start[i], &error)) {
        return fatal("restore");
      }
    }
    (void)lanes[1]->Evaluate(-3, &error);  // leaves lane 1 invalid
    batch = greedy_batch(4, &samplers);
    l3::Session::DecodeBatch(batch);
    auto o2 = json::Value::object();
    o2["invalid_error"] = batch[1].error;
    check("invalid_lane_fails_alone",
          prefix_of_ar(batch[0], 0) && prefix_of_ar(batch[2], 2) &&
              prefix_of_ar(batch[3], 3) && !batch[1].ok && !batch[1].deferred,
          std::move(o2));
  }
  {
    // A verify forward that fails after launch empties its own lanes.
    for (std::size_t i = 0; i < 3; ++i) {
      if (!lanes[i]->RestoreSnapshot(*start[i], &error)) {
        return fatal("restore");
      }
    }
    std::vector<sampling::SamplerState> samplers(
        3, sampling::SamplerState(Greedy()));
    std::vector<l3::Session::BatchDecode> batch(3);
    for (std::size_t i = 0; i < 3; ++i) {
      batch[i].session = lanes[i].get();
      batch[i].max_tokens = 4;
      batch[i].sampler = &samplers[i];
      batch[i].stop_at_eos = false;
    }
    // One injected failure hits one verify forward. The lanes may split across
    // several forwards (8 rows each, set by copy chains and the draft width).
    // Only the failed forward's lanes fail and are emptied. The others decode
    // on.
    model->FailBatchesForTest(1);
    l3::Session::DecodeBatch(batch);
    bool injected = false;
    bool emptied = true;
    bool others_ok = true;
    auto failed_lanes = json::Value::array();
    for (std::size_t i = 0; i < 3; ++i) {
      const auto& t = batch[i].result.tokens;
      if (batch[i].ok) {
        others_ok = others_ok && !t.empty() &&
                    std::equal(t.begin(), t.end(), ar[i].tokens.begin());
      } else {
        injected = injected ||
                   batch[i].error.find("injected failure") != std::string::npos;
        emptied = emptied && lanes[i]->IsValid() && lanes[i]->Position() == 0 &&
                  lanes[i]->Tokens().empty();
      }
      auto lane = json::Value::object();
      lane["ok"] = batch[i].ok;
      lane["position"] = U(lanes[i]->Position());
      lane["tokens"] = U(lanes[i]->Tokens().size());
      failed_lanes.push_back(std::move(lane));
    }
    // Reusable: restore and decode again, equal to AR.
    bool reusable = true;
    std::vector<sampling::SamplerState> again(3,
                                              sampling::SamplerState(Greedy()));
    for (std::size_t i = 0; i < 3; ++i) {
      reusable = reusable && lanes[i]->RestoreSnapshot(*start[i], &error);
      batch[i] = {};
      batch[i].session = lanes[i].get();
      batch[i].max_tokens = 4;
      batch[i].sampler = &again[i];
      batch[i].stop_at_eos = false;
    }
    l3::Session::DecodeBatch(batch);
    auto again_lanes = json::Value::array();
    for (std::size_t i = 0; i < 3; ++i) {
      const auto& t = batch[i].result.tokens;
      const bool equal = std::equal(t.begin(), t.end(), ar[i].tokens.begin());
      reusable = reusable && batch[i].ok && !t.empty() && equal;
      auto lane = json::Value::object();
      lane["ok"] = batch[i].ok;
      lane["tokens"] = U(t.size());
      lane["equals_ar"] = equal;
      again_lanes.push_back(std::move(lane));
    }
    auto o = json::Value::object();
    o["error"] = batch[0].error;
    o["failed"] = std::move(failed_lanes);
    o["again"] = std::move(again_lanes);
    check("failed_verify_empties_its_lanes_reusable",
          injected && emptied && others_ok && reusable, std::move(o));
  }
  {
    // The second batched draft step fails after launch. Its lane (the longer
    // chain) fails with the trunk untouched, as a failed draft alone leaves
    // it. The shorter chain has finished drafting and still verifies.
    for (std::size_t i = 0; i < 2; ++i) {
      if (!lanes[i]->RestoreSnapshot(*start[i], &error)) {
        return fatal("restore");
      }
      lanes[i]->ResetDraftPolicy();
    }
    l3::Session::DraftHooks three;
    three.fixed_drafts = 3;
    l3::Session::DraftHooks one;
    one.fixed_drafts = 1;
    lanes[0]->SetDraftHooks(three);
    lanes[1]->SetDraftHooks(one);
    std::vector<sampling::SamplerState> samplers(
        2, sampling::SamplerState(Greedy()));
    std::vector<l3::Session::BatchDecode> batch(2);
    for (std::size_t i = 0; i < 2; ++i) {
      batch[i].session = lanes[i].get();
      batch[i].max_tokens = 8;
      batch[i].sampler = &samplers[i];
      batch[i].stop_at_eos = false;
    }
    const std::uint32_t start_pos = lanes[0]->Position();
    model->FailDraftBatchForTest(1);
    l3::Session::DecodeBatch(batch);
    const auto& t = batch[1].result.tokens;
    const bool shorter = batch[1].ok && !t.empty() &&
                         std::equal(t.begin(), t.end(), ar[1].tokens.begin());
    const bool failed_untouched = !batch[0].ok && lanes[0]->IsValid() &&
                                  lanes[0]->Position() == start_pos;
    // The failed lane decodes on afterwards as it would have.
    sampling::SamplerState next(Greedy());
    l3::Session::DecodeResult r;
    const bool resumes =
        lanes[0]->DecodeStep(4, next, &r, &error, false) && !r.tokens.empty() &&
        std::equal(r.tokens.begin(), r.tokens.end(), ar[0].tokens.begin());
    lanes[0]->SetDraftHooks({});
    lanes[1]->SetDraftHooks({});
    auto o = json::Value::object();
    o["error"] = batch[0].error;
    check("failed_draft_step_fails_its_lanes_only",
          shorter && failed_untouched && resumes, std::move(o));
  }
  report.Set("checks", checks);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

int Body(const Args& args, Report& report) {
  auto model = LoadModel(args, report, 512, 1024, /*force_mtp=*/true);
  if (!model) {
    return 1;
  }
  return Run(args, report, model);
}

}  // namespace gufo::tests::ling::batch_mtp_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(argc, argv, "batch_mtp_test",
                                    gufo::tests::ling::batch_mtp_tool::Body);
}
