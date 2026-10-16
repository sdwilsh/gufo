// Executor-level batch invariance of multi-token segments
// (rocm::Executor::ForwardBatch with segments of 1..8 tokens, the shape a
// batched MTP verify will use): six sessions at depths 0 (its first forward is
// a batched segment, the MTP fill's skip path), 9, 300, 1200 (with an
// expanded K/V store), 2045 (with a store; its segments end at and start on
// the 2048-row block edge) and 2047 (an 8-token segment straddles the edge)
// take rounds of segments of mixed widths ({3,1,2,2}, {8}, {7,1}, {4,4},
// {1,3,1,2,1}, {2,5}, six 1-token lanes), each with its own logits_from (last
// row, all rows, from the middle, no head). Every segment's logits rows and,
// at the end, every lane's KDA state, conv windows, MLA cache rows and (with
// --mtp) MTP cache rows and carry must be byte-identical to Forward of the
// same chunks on a twin session alone. A segment that would exceed the
// 8-row budget is refused and leaves its session untouched.
// Exit 77 without --model.
//
//   batch_segments_test --model FIRST_SHARD.gguf [--mtp] [--layers N]
//                       [--max-context N] [--expanded-kv-bytes N]

#include <cstring>
#include <optional>
#include <tuple>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/continuation_hooks.hpp"
#include "src/models/ling_3_flash/kernels/rocm/device_model.hpp"
#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"
#include "src/models/ling_3_flash/weights.hpp"
#include "tests/models/ling_3_flash/model_tool.hpp"

namespace gufo::tests::ling::segments_tool {

namespace rocm = l3::rocm;

std::int32_t Tok(std::uint32_t i) {
  return static_cast<std::int32_t>((i * 2654435761U + 12345U) % 150000U);
}

template<class T>
bool SameBytes(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// Stands in for a slot that is not ready (or whose guard refuses).
struct NotReadyHooks final : l3::ContinuationHooks {
  bool ready{false};
  bool throws{false};
  [[nodiscard]] bool Ready() const noexcept override { return ready; }
  void BeforeWrite(std::uint32_t, std::uint32_t, std::uint32_t,
                   std::uint32_t) override {
    if (throws)
      throw std::runtime_error("refused");
  }
};

struct Lane {
  std::uint32_t depth{0};
  std::unique_ptr<rocm::Session> batch;
  std::unique_ptr<rocm::Session> solo;
  std::uint32_t next{0};  // token counter
  std::uint32_t segments{0};
  std::uint32_t mismatches{0};
};

int Body(const Args& args, Report& report) {
  const std::string path = args.Str("--model");
  const bool mtp = args.Has("--mtp");
  const auto ctx = static_cast<std::uint32_t>(args.Num("--max-context", 2304));
  constexpr std::uint64_t kBlockBytes = std::uint64_t{2048} * 112 * 1024;
  std::string error;
  const auto fatal = [&](const std::string& what) {
    std::fprintf(stderr, "batch_segments_test: %s: %s\n", what.c_str(),
                 error.c_str());
    report.Set("status", "error");
    report.Set("error", what + ": " + error);
    return 1;
  };
  auto reader = core::GgufReader::OpenFile(path, &error);
  if (!reader) {
    return fatal("open");
  }
  auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    return fatal("bind");
  }
  auto device = rocm::DeviceModel::Upload(
      *weights, *reader, 0, static_cast<std::uint32_t>(args.Num("--layers", 0)),
      &error, mtp);
  if (!device) {
    return fatal("upload");
  }
  rocm::Executor::Options options;
  options.max_batch = 512;
  options.max_context = ctx;
  options.mtp = mtp;
  options.expanded_kv_bytes = static_cast<std::uint64_t>(
      args.Num("--expanded-kv-bytes", static_cast<double>(3 * kBlockBytes)));
  auto exec = rocm::Executor::Create(*device, options, &error);
  if (!exec) {
    return fatal("executor");
  }
  const std::uint32_t edge = std::min<std::uint32_t>(2045, ctx - 64);
  const std::vector<std::uint32_t> depths = {
      edge, 0,
      9,    std::min<std::uint32_t>(1200, ctx - 64),
      300,  std::min<std::uint32_t>(2047, ctx - 64)};
  std::vector<Lane> lanes(depths.size());
  // Expanded K/V blocks go first come first served: lanes 0 and 3 first.
  for (const std::size_t i : {0U, 3U, 1U, 2U, 4U, 5U}) {
    lanes[i].depth = depths[i];
    lanes[i].batch = exec->CreateSession(ctx, &error);
    if (!lanes[i].batch) {
      return fatal("create session");
    }
  }
  for (Lane& lane : lanes) {
    lane.solo = exec->CreateSession(ctx, &error);
    if (!lane.solo) {
      return fatal("create twin");
    }
  }
  // Identical prefill-family prompts (chunks of 9..512 tokens) on both twins.
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    Lane& lane = lanes[i];
    std::vector<std::int32_t> prompt;
    for (std::uint32_t t = 0; t < lane.depth; ++t) {
      prompt.push_back(Tok(100000 * i + t));
    }
    for (std::uint32_t at = 0; at < lane.depth; at += 512) {
      const std::uint32_t n = std::min<std::uint32_t>(512, lane.depth - at);
      rocm::ForwardRequest r;
      r.tokens = std::span<const std::int32_t>(prompt).subspan(at, n);
      if (!exec->Forward(*lane.batch, r, &error) ||
          !exec->Forward(*lane.solo, r, &error)) {
        return fatal("prefill");
      }
    }
  }
  // (lane, width) segments per round; logits_from cycles per segment.
  using Round = std::vector<std::pair<std::size_t, std::uint32_t>>;
  const std::vector<Round> rounds = {
      {{0, 3}, {2, 1}, {1, 2}, {3, 2}},  // lane 1: 2 tokens from position 0
      {{5, 8}},
      {{1, 7}, {4, 1}},
      {{3, 4}, {0, 4}},
      {{2, 1}, {4, 3}, {1, 1}, {5, 2}, {0, 1}},
      {{4, 2}, {2, 5}},
      {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}},
      {{5, 3}, {1, 2}, {0, 3}},
  };
  constexpr std::uint32_t kFroms[] = {UINT32_MAX, 0, 1, 2, 3};
  std::uint32_t cycle = 0;
  std::uint32_t failures_seen = 0;
  for (const Round& round : rounds) {
    std::vector<std::vector<std::int32_t>> toks(round.size());
    std::vector<std::vector<float>> got(round.size());
    std::vector<rocm::BatchSegment> segs(round.size());
    std::vector<std::uint32_t> froms(round.size());
    for (std::size_t k = 0; k < round.size(); ++k) {
      const auto& [i, n] = round[k];
      Lane& lane = lanes[i];
      for (std::uint32_t t = 0; t < n; ++t) {
        toks[k].push_back(Tok(7000 * (i + 1) + lane.next + t));
      }
      const std::uint32_t pick = kFroms[cycle++ % std::size(kFroms)];
      // 3 = no head for this segment; others clamp into the segment.
      const bool head = pick != 3;
      const std::uint32_t from =
          pick == UINT32_MAX ? UINT32_MAX : std::min(pick, n - 1);
      froms[k] = from;
      segs[k].session = lane.batch.get();
      segs[k].tokens = toks[k];
      if (head) {
        const std::uint32_t rows = from == UINT32_MAX ? 1 : n - from;
        got[k].resize(std::size_t{rows} * l3::kVocab);
        segs[k].logits = got[k];
        segs[k].logits_from = from;
      }
    }
    if (!exec->ForwardBatch(segs, &error)) {
      return fatal("batched forward");
    }
    for (std::size_t k = 0; k < round.size(); ++k) {
      const auto& [i, n] = round[k];
      Lane& lane = lanes[i];
      if (segs[k].status != rocm::BatchSegment::Status::kDone) {
        std::fprintf(stderr, "segment lane %zu n=%u: %s\n", i, n,
                     segs[k].error.c_str());
        ++failures_seen;
        continue;
      }
      rocm::ForwardRequest r;
      r.tokens = toks[k];
      std::vector<float> want(got[k].size());
      if (!got[k].empty()) {
        r.logits = want;
        r.logits_from = froms[k];
      }
      if (!exec->Forward(*lane.solo, r, &error)) {
        return fatal("solo forward");
      }
      if (!SameBytes(got[k], want)) {
        ++lane.mismatches;
        std::fprintf(stderr, "lane %zu segment n=%u from=%u: logits differ\n",
                     i, n, froms[k]);
      }
      lane.next += n;
      ++lane.segments;
    }
  }
  // ---- MTP (--mtp): batched drafts and rollback verify segments -----------
  auto mtp_checks = json::Value::array();
  if (mtp) {
    // (A) Draft chains of lengths {3, 2, 2, 1, 1, 0} (longest first, so each
    // depth's lanes are a prefix), with the Q8_0 and the F16 draft head:
    // every lane's logits rows, argmax and MTP rows equal MtpDraft alone.
    const std::vector<std::uint32_t> chain_len = {3, 2, 2, 1, 1, 0};
    for (const bool q8 : {true, false}) {
      exec->SetMtpDraftHeadQ8(q8);
      std::uint32_t mismatches = 0;
      std::vector<std::vector<std::vector<float>>> batch_logits(lanes.size());
      std::vector<std::vector<std::int32_t>> batch_argmax(lanes.size());
      std::vector<std::uint32_t> pos(lanes.size());
      for (std::uint32_t j = 0; j < 3; ++j) {
        std::vector<rocm::DraftLane> step;
        std::vector<std::vector<float>> out;
        std::vector<std::size_t> who;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
          if (chain_len[i] > j) {
            who.push_back(i);
          }
        }
        out.resize(who.size(), std::vector<float>(l3::kVocab));
        for (std::size_t r = 0; r < who.size(); ++r) {
          const std::size_t i = who[r];
          if (j == 0) {
            pos[i] = lanes[i].batch->position() - 1;
          } else {
            ++pos[i];
          }
          rocm::DraftLane d;
          d.session = lanes[i].batch.get();
          d.token = Tok(55000 + 31 * i + j);
          d.position = pos[i];
          d.chain = j > 0;
          d.logits = out[r];
          step.push_back(d);
        }
        if (!exec->MtpDraftBatch(step, &error)) {
          return fatal("draft batch");
        }
        for (std::size_t r = 0; r < who.size(); ++r) {
          batch_logits[who[r]].push_back(out[r]);
          batch_argmax[who[r]].push_back(step[r].argmax);
        }
      }
      for (std::size_t i = 0; i < lanes.size(); ++i) {
        const std::uint32_t start = lanes[i].solo->position() - 1;
        for (std::uint32_t j = 0; j < chain_len[i]; ++j) {
          rocm::DraftRequest r;
          r.token = Tok(55000 + 31 * i + j);
          r.position = start + j;
          r.chain = j > 0;
          std::vector<float> want(l3::kVocab);
          r.logits = want;
          std::int32_t argmax = -1;
          if (!exec->MtpDraft(*lanes[i].solo, r, &argmax, &error)) {
            return fatal("solo draft");
          }
          if (!SameBytes(want, batch_logits[i][j]) ||
              argmax != batch_argmax[i][j]) {
            ++mismatches;
            std::fprintf(stderr, "draft lane %zu depth %u (q8 %d) differs\n", i,
                         j, q8 ? 1 : 0);
          }
        }
        if (chain_len[i] > 0) {
          const std::uint32_t rows = start + chain_len[i];
          std::vector<std::uint16_t> a, b;
          if (!exec->ReadMtpCache(*lanes[i].batch, rows, &a, &error) ||
              !exec->ReadMtpCache(*lanes[i].solo, rows, &b, &error)) {
            return fatal("read draft rows");
          }
          mismatches += SameBytes(a, b) ? 0 : 1;
        }
      }
      auto o = json::Value::object();
      o["name"] = std::string("draft_batch_equals_solo_") + (q8 ? "q8" : "f16");
      o["mismatches"] = U(mismatches);
      o["pass"] = mismatches == 0;
      std::fprintf(stderr, "[check] %s: %s\n", o["name"].str().c_str(),
                   mismatches == 0 ? "PASS" : "FAIL");
      failures_seen += mismatches == 0 ? 0 : 1;
      mtp_checks.push_back(std::move(o));
    }
    exec->SetMtpDraftHeadQ8(true);
    // (A2) A lane whose hooks refuse (not ready, or a throwing guard) fails
    // alone: the call succeeds, the peers equal MtpDraft alone, and the
    // refused lane's session is untouched.
    for (const bool throws : {false, true}) {
      NotReadyHooks hooks;
      hooks.ready = throws;  // not ready, or ready but BeforeWrite throws
      hooks.throws = throws;
      const std::vector<std::size_t> who = {0, 2, 3};
      Lane& refused = lanes[2];
      std::vector<std::uint16_t> rows_before, rows_after;
      const std::uint32_t refused_rows = refused.batch->position();
      if (!exec->ReadMtpCache(*refused.batch, refused_rows, &rows_before,
                              &error)) {
        return fatal("read refused rows");
      }
      refused.batch->SetContinuationHooksForTest(&hooks);
      std::vector<std::vector<float>> out(who.size(),
                                          std::vector<float>(l3::kVocab));
      std::vector<rocm::DraftLane> step;
      for (std::size_t r = 0; r < who.size(); ++r) {
        rocm::DraftLane d;
        d.session = lanes[who[r]].batch.get();
        d.token = Tok(77000 + 13 * who[r]);
        d.position = d.session->position() - 1;
        d.logits = out[r];
        step.push_back(d);
      }
      const bool ran = exec->MtpDraftBatch(step, &error);
      refused.batch->SetContinuationHooksForTest(nullptr);
      std::uint32_t bad = 0;
      if (!ran || step[0].failed || !step[1].failed || step[1].error.empty() ||
          step[2].failed || step[1].argmax != -1) {
        ++bad;
      }
      for (const std::size_t r : {std::size_t{0}, std::size_t{2}}) {
        rocm::DraftRequest req;
        req.token = step[r].token;
        req.position = step[r].position;
        std::vector<float> want(l3::kVocab);
        req.logits = want;
        std::int32_t argmax = -1;
        if (!exec->MtpDraft(*lanes[who[r]].solo, req, &argmax, &error)) {
          return fatal("solo draft of a peer");
        }
        if (!SameBytes(want, out[r]) || argmax != step[r].argmax) {
          ++bad;
        }
      }
      if (!exec->ReadMtpCache(*refused.batch, refused_rows, &rows_after,
                              &error)) {
        return fatal("read refused rows");
      }
      bad += SameBytes(rows_before, rows_after) ? 0 : 1;
      auto o = json::Value::object();
      o["name"] = throws ? "draft_batch_refusing_hook_fails_alone"
                         : "draft_batch_not_ready_lane_fails_alone";
      o["mismatches"] = U(bad);
      o["pass"] = bad == 0;
      std::fprintf(stderr, "[check] %s: %s\n", o["name"].str().c_str(),
                   bad == 0 ? "PASS" : "FAIL");
      failures_seen += bad == 0 ? 0 : 1;
      mtp_checks.push_back(std::move(o));
    }
    // (B) Verify segments with rollback points, cut back per lane: logits,
    // argmax and (at the end) the whole state equal Forward(rollback) +
    // MtpRollback alone. Cohorts {3,2,3} (5 points), {4,4}, {2,1,1,4}.
    using Verify = std::vector<std::tuple<std::size_t, std::uint32_t,
                                          std::uint32_t>>;  // lane, n, keep
    const std::vector<Verify> verifies = {
        {{0, 3, 1}, {1, 2, 2}, {2, 3, 2}},
        {{3, 4, 4}, {4, 4, 1}},
        {{5, 2, 1}, {0, 1, 1}, {1, 1, 1}, {2, 4, 3}},
    };
    std::uint32_t mismatches = 0;
    for (const Verify& v : verifies) {
      std::vector<std::vector<std::int32_t>> toks(v.size());
      std::vector<std::vector<float>> logits(v.size());
      std::vector<std::vector<std::int32_t>> argmax(v.size());
      std::vector<rocm::BatchSegment> segs(v.size());
      std::vector<std::uint32_t> first(v.size());
      for (std::size_t k = 0; k < v.size(); ++k) {
        const auto [i, n, keep] = v[k];
        for (std::uint32_t t = 0; t < n; ++t) {
          toks[k].push_back(Tok(66000 + 97 * i + lanes[i].next + t));
        }
        first[k] = lanes[i].batch->position();
        logits[k].resize(std::size_t{n} * l3::kVocab);
        argmax[k].resize(n);
        segs[k].session = lanes[i].batch.get();
        segs[k].tokens = toks[k];
        segs[k].logits = logits[k];
        segs[k].logits_from = 0;
        segs[k].argmax = argmax[k];
        segs[k].rollback = n > 1;
      }
      if (!exec->ForwardBatch(segs, &error)) {
        return fatal("verify batch");
      }
      for (std::size_t k = 0; k < v.size(); ++k) {
        const auto [i, n, keep] = v[k];
        Lane& lane = lanes[i];
        if (segs[k].status != rocm::BatchSegment::Status::kDone) {
          ++mismatches;
          std::fprintf(stderr, "verify lane %zu: %s\n", i,
                       segs[k].error.c_str());
          continue;
        }
        if (n > 1 && !exec->MtpRollback(*lane.batch, first[k], n, keep, &error,
                                        segs[k].placement)) {
          return fatal("batched rollback");
        }
        rocm::ForwardRequest r;
        r.tokens = toks[k];
        std::vector<float> want(logits[k].size());
        std::vector<std::int32_t> want_argmax(n);
        r.logits = want;
        r.logits_from = 0;
        r.argmax = want_argmax;
        r.rollback = n > 1;
        if (!exec->Forward(*lane.solo, r, &error) ||
            (n > 1 &&
             !exec->MtpRollback(*lane.solo, first[k], n, keep, &error))) {
          return fatal("solo verify");
        }
        if (!SameBytes(want, logits[k]) || want_argmax != argmax[k]) {
          ++mismatches;
          std::fprintf(stderr, "verify lane %zu n=%u: rows differ\n", i, n);
        }
        lane.next += n;
      }
    }
    auto o = json::Value::object();
    o["name"] = "rollback_verify_equals_solo";
    o["mismatches"] = U(mismatches);
    o["pass"] = mismatches == 0;
    std::fprintf(stderr, "[check] rollback_verify_equals_solo: %s\n",
                 mismatches == 0 ? "PASS" : "FAIL");
    failures_seen += mismatches == 0 ? 0 : 1;
    mtp_checks.push_back(std::move(o));
  }
  report.Set("mtp", mtp_checks);
  // Row budget: a segment past 8 rows is refused, its session untouched.
  {
    std::vector<std::int32_t> eight(8, Tok(1)), one(1, Tok(2));
    std::vector<rocm::BatchSegment> segs(2);
    segs[0].session = lanes[4].batch.get();
    segs[0].tokens = eight;
    segs[1].session = lanes[2].batch.get();
    segs[1].tokens = one;
    const std::uint32_t before = lanes[2].batch->position();
    const bool ran = exec->ForwardBatch(segs, &error);
    rocm::ForwardRequest r;
    r.tokens = eight;
    const bool solo = exec->Forward(*lanes[4].solo, r, &error);
    auto o = json::Value::object();
    o["refusal"] = segs[1].error;
    const bool pass = ran && solo &&
                      segs[0].status == rocm::BatchSegment::Status::kDone &&
                      segs[1].status == rocm::BatchSegment::Status::kRejected &&
                      lanes[2].batch->position() == before;
    o["name"] = "row_budget_refusal_leaves_session";
    o["pass"] = pass;
    report.Set("row_budget", o);
    failures_seen += pass ? 0 : 1;
  }
  // Whole state, every lane.
  auto results = json::Value::array();
  int failures = static_cast<int>(failures_seen);
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    Lane& lane = lanes[i];
    bool kda = true, mla = true, mtp_rows = true;
    const std::uint32_t pos = lane.batch->position();
    for (std::uint32_t il = 0; il < exec->layer_count(); ++il) {
      if (l3::IsMlaLayer(il)) {
        std::vector<std::uint16_t> a, b;
        if (!exec->ReadMlaCache(*lane.batch, il, pos, &a, &error) ||
            !exec->ReadMlaCache(*lane.solo, il, pos, &b, &error)) {
          return fatal("read cache");
        }
        mla = mla && SameBytes(a, b);
      } else {
        std::vector<float> sa, ca, sb, cb;
        if (!exec->ReadKdaState(*lane.batch, il, &sa, &ca, &error) ||
            !exec->ReadKdaState(*lane.solo, il, &sb, &cb, &error)) {
          return fatal("read state");
        }
        kda = kda && SameBytes(sa, sb) && SameBytes(ca, cb);
      }
    }
    if (mtp) {
      std::vector<std::uint16_t> a, b;
      std::vector<float> ha, hb;
      if (!exec->ReadMtpCache(*lane.batch, pos - 1, &a, &error) ||
          !exec->ReadMtpCache(*lane.solo, pos - 1, &b, &error) ||
          !exec->ReadMtpHidden(*lane.batch, &ha, &error) ||
          !exec->ReadMtpHidden(*lane.solo, &hb, &error)) {
        return fatal("read MTP");
      }
      mtp_rows = SameBytes(a, b) && SameBytes(ha, hb) &&
                 lane.batch->mtp_valid() && lane.solo->mtp_valid();
    }
    const bool pass = lane.mismatches == 0 && pos == lane.solo->position() &&
                      kda && mla && mtp_rows && lane.segments > 0;
    auto o = json::Value::object();
    o["lane"] = U(i);
    o["depth"] = U(lane.depth);
    o["position"] = U(pos);
    o["segments"] = U(lane.segments);
    o["expanded_store_rows"] = U(lane.batch->expanded_store_rows());
    o["logit_mismatches"] = U(lane.mismatches);
    o["kda_equal"] = kda;
    o["mla_equal"] = mla;
    o["mtp_equal"] = mtp_rows;
    o["pass"] = pass;
    std::fprintf(stderr, "[check] lane%zu depth %u -> %u: %s\n", i, lane.depth,
                 pos, pass ? "PASS" : "FAIL");
    failures += pass ? 0 : 1;
    results.push_back(std::move(o));
  }
  report.Set("lanes", results);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

}  // namespace gufo::tests::ling::segments_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(argc, argv, "batch_segments_test",
                                    gufo::tests::ling::segments_tool::Body);
}
