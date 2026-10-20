// Restore + tail prefill of Ling-3.0-flash sessions whose persistent expanded
// K/V stores and prior histories differ, as the server's fan-out drives them:
// a shared prefix that ends mid-block past 2048 rows (5,421 tokens in the
// cache-concurrency fan-out) is restored into whatever session is free, a
// short tail is prefilled in prefill-family chunks with a snapshot at the
// stable boundary between them, and the session decodes greedily. Every run
// must be bit-identical (logits after the tail, the decoded tokens, the final
// logits, KDA state, conv windows, MLA cache rows, the boundary snapshot) to
// the same work on a fresh session with a full store.
//
// Sessions: `ref` and `full` own whole stores, `partial` owns
// --partial-blocks 2048-row blocks (default: those below the prefix end, so
// the tail mixes persistent and transient blocks), `zero` owns none (the
// budget is sized for exactly that; a session's grant is first come first
// served). Each store kind runs, in order:
//   fresh               restore into a never-used session
//   direct              reset, prefill prefix + tail without a restore
//   shared_history      reset, prefill another history (the first
//                       --share-tokens tokens equal the prefix, then other
//                       text, longer than the restored context) and decode,
//                       then restore: partial re-expansion from the first
//                       differing row, stale rows past the restored position
//   same_prefix_history the history is the prefix plus other text: the
//                       restore keeps the whole prefix's expansion and the
//                       tail starts the expansion at the unaligned prefix end
//   unrelated_history   unrelated history: no row kept, every row stale
// Exit 77 without --model.
//
//   restore_tail_test --model FIRST_SHARD.gguf [--layers N]
//       [--max-context N (8192)] [--max-batch N (512)]
//       [--prefix-tokens N (5421)] [--tails A,B,... (10,9)]
//       [--decode-tokens K (8)] [--share-tokens N (1000)]
//       [--history-extra N (600)] [--partial-blocks N]
//
// Prints one JSON report on stdout (checks[] with metrics).

#define LING_TOOL_NO_MAIN
#include <sstream>

#include "tests/models/ling_3_flash/session_test.cpp"

namespace gufo::tests::ling::restore_tail_tool {

using session_tool::Capture;
using session_tool::Compare;
using session_tool::Take;
using session_tool::ToJson;

constexpr std::uint32_t kBlockRows = 2048;
/// Persistent store bytes of one 2048-row block over the 7 MLA layers (32
/// heads x (128 nope + 128 value) F16 per row and layer).
constexpr std::uint64_t kBlockBytes =
    std::uint64_t{kBlockRows} * l3::kMlaLayers * 32 * 256 * 2;

std::vector<std::uint32_t> ParseTails(const std::string& text) {
  std::vector<std::uint32_t> out;
  std::stringstream in(text);
  std::string item;
  while (std::getline(in, item, ',')) {
    if (!item.empty()) {
      out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    }
  }
  return out;
}

struct Outcome {
  std::vector<float> logits_after_tail;
  std::vector<std::uint8_t> boundary_snapshot;
  std::vector<std::int32_t> tokens;
  Capture state;
  std::uint32_t expanded_after_restore{0};
};

struct Plan {
  std::vector<std::int32_t> prefix;
  std::vector<std::int32_t> tail;
  std::vector<std::uint32_t> tails;
  std::uint32_t decode{0};
  std::uint32_t layers{0};
};

/// Restored (or directly prefilled) prefix, the tail chunks with a snapshot at
/// the first boundary, then greedy decode steps.
bool Continue(l3::Session& s, const Plan& plan, Outcome* out,
              std::string* error) {
  std::size_t at = 0;
  for (std::size_t c = 0; c < plan.tails.size(); ++c) {
    if (!s.Feed(std::span(plan.tail).subspan(at, plan.tails[c]), {}, error)) {
      return false;
    }
    at += plan.tails[c];
    if (c == 0 && plan.tails.size() > 1) {
      auto snap = s.SaveSnapshot(error);
      if (!snap) {
        return false;
      }
      out->boundary_snapshot.assign(snap->bytes().begin(), snap->bytes().end());
    }
  }
  out->logits_after_tail.assign(s.Logits().begin(), s.Logits().end());
  for (std::uint32_t k = 0; k < plan.decode; ++k) {
    const std::int32_t token = s.Argmax();
    out->tokens.push_back(token);
    if (!s.Evaluate(token, error)) {
      return false;
    }
  }
  return Take(s, plan.layers, &out->state, error);
}

/// Prefills `tokens` (in the model's max-batch chunks) and decodes three
/// greedy tokens, so the session holds expanded rows and captured graphs.
bool Occupy(l3::Session& s, std::span<const std::int32_t> tokens,
            std::string* error) {
  s.Reset();
  if (!s.IsValid() || !s.Feed(tokens, {}, error)) {
    return false;
  }
  for (int k = 0; k < 3; ++k) {
    if (!s.Evaluate(s.Argmax(), error)) {
      return false;
    }
  }
  return true;
}

int Run(const Args& args, Report& report, std::shared_ptr<l3::Model> model,
        std::uint32_t partial_blocks) {
  const std::uint32_t ctx = model->MaxContext();
  const auto P = static_cast<std::uint32_t>(args.Num("--prefix-tokens", 5421));
  const auto K = static_cast<std::uint32_t>(args.Num("--decode-tokens", 8));
  const auto share =
      static_cast<std::uint32_t>(args.Num("--share-tokens", 1000));
  const auto extra =
      static_cast<std::uint32_t>(args.Num("--history-extra", 600));
  const auto tails = ParseTails(args.Str("--tails", "10,9"));
  std::uint32_t T = 0;
  for (const auto n : tails) {
    T += n;
  }
  const std::uint32_t H = P + T + K + extra;  // prior histories
  if (tails.empty() ||
      std::ranges::any_of(
          tails,
          [](std::uint32_t n) { return n <= l3::kSmallBatchMaxTokens; }) ||
      share >= P || H + 3 > ctx) {
    report.Set("status", "error");
    report.Set("error",
               "need prefill-family tails (> 8 tokens), --share-tokens < "
               "--prefix-tokens and --max-context > prefix + tails + decode "
               "+ --history-extra + 3");
    return 1;
  }
  std::string text;
  std::vector<std::int32_t> ids;
  while (ids.size() < static_cast<std::size_t>(H) + T) {
    text += session_tool::kText;
    ids = model->Tokenize(text);
  }
  Plan plan;
  plan.prefix.assign(ids.begin(), ids.begin() + P);
  plan.tail.assign(ids.begin() + P, ids.begin() + P + T);
  plan.tails = tails;
  plan.decode = K;
  plan.layers = model->LayerCount();
  // Other conversations: the same text with remapped ids after `keep` tokens.
  const auto history = [&](std::uint32_t keep, std::int32_t salt) {
    std::vector<std::int32_t> h(ids.begin(), ids.begin() + H);
    for (std::size_t t = keep; t < h.size(); ++t) {
      h[t] = (h[t] * (7 + salt) + 13 + salt) % 50000;
    }
    return h;
  };

  std::string error;
  auto checks = json::Value::array();
  int failures = 0;
  const auto fail_with = [&](const std::string& what) {
    std::fprintf(stderr, "restore_tail_test: %s: %s\n", what.c_str(),
                 error.c_str());
    report.Set("status", "error");
    report.Set("error", what + ": " + error);
    report.Set("checks", checks);
    return 1;
  };

  // Store grants follow creation order: ref and full take whole stores,
  // partial the rest of the budget but `zero`'s nothing.
  auto ref = model->CreateSession(ctx, &error);
  auto full = ref ? model->CreateSession(ctx, &error) : nullptr;
  auto partial = full ? model->CreateSession(ctx, &error) : nullptr;
  auto zero = partial ? model->CreateSession(ctx, &error) : nullptr;
  if (!zero) {
    return fail_with("create sessions");
  }
  const std::uint32_t full_rows =
      (ctx + kBlockRows - 1) / kBlockRows * kBlockRows;
  {
    auto o = json::Value::object();
    o["ref_store_rows"] = U(ref->ExpandedStoreRows());
    o["full_store_rows"] = U(full->ExpandedStoreRows());
    o["partial_store_rows"] = U(partial->ExpandedStoreRows());
    o["zero_store_rows"] = U(zero->ExpandedStoreRows());
    o["prefix_tokens"] = U(P);
    o["prefix_end_rows"] = U(P + T);
    const bool pass =
        ref->ExpandedStoreRows() == full_rows &&
        full->ExpandedStoreRows() == full_rows &&
        partial->ExpandedStoreRows() == partial_blocks * kBlockRows &&
        zero->ExpandedStoreRows() == 0;
    o["name"] = "store_grants";
    o["pass"] = pass;
    std::fprintf(stderr, "[check] %-52s %s (rows %u/%u/%u/%u)\n",
                 "store_grants", pass ? "PASS" : "FAIL",
                 ref->ExpandedStoreRows(), full->ExpandedStoreRows(),
                 partial->ExpandedStoreRows(), zero->ExpandedStoreRows());
    failures += pass ? 0 : 1;
    checks.push_back(std::move(o));
  }

  // The shared prefix, prefilled once by a leader. With the budget spent the
  // leader has no store, like a server session past the budget.
  std::unique_ptr<l3::SessionSnapshot> shared;
  {
    auto leader = model->CreateSession(ctx, &error);
    if (!leader || !leader->Feed(plan.prefix, {}, &error)) {
      return fail_with("leader prefill");
    }
    shared = leader->SaveSnapshot(&error);
    if (!shared) {
      return fail_with("shared snapshot");
    }
  }

  Outcome expected;
  if (!ref->RestoreSnapshot(*shared, &error) ||
      !Continue(*ref, plan, &expected, &error)) {
    return fail_with("reference");
  }
  std::fprintf(stderr, "[ref] first token %d, decoded %zu tokens\n",
               expected.tokens.empty() ? -1 : expected.tokens.front(),
               expected.tokens.size());

  const auto compare = [&](const std::string& name, const Outcome& got,
                           l3::Session& s) {
    const auto d = Compare(got.state, expected.state);
    auto o = ToJson(d);
    const bool tail_same = got.logits_after_tail == expected.logits_after_tail;
    const bool tokens_same = got.tokens == expected.tokens;
    const bool snapshot_same =
        got.boundary_snapshot == expected.boundary_snapshot;
    o["logits_after_tail_same"] = tail_same;
    o["tokens_same"] = tokens_same;
    o["boundary_snapshot_same"] = snapshot_same;
    o["store_rows"] = U(s.ExpandedStoreRows());
    o["expanded_rows_after_restore"] = U(got.expanded_after_restore);
    if (!got.logits_after_tail.empty() && !expected.logits_after_tail.empty()) {
      double worst = 0.0;
      for (std::size_t i = 0; i < got.logits_after_tail.size(); ++i) {
        worst =
            std::max<double>(worst, std::fabs(got.logits_after_tail[i] -
                                              expected.logits_after_tail[i]));
      }
      o["logits_after_tail_max_abs"] = Round(worst);
    }
    const bool pass =
        session_tool::Exact(d) && tail_same && tokens_same && snapshot_same;
    o["name"] = name;
    o["pass"] = pass;
    std::fprintf(stderr, "[check] %-52s %s\n", name.c_str(),
                 pass ? "PASS" : "FAIL");
    failures += pass ? 0 : 1;
    checks.push_back(std::move(o));
    report.Set("checks", checks);
  };

  // The reference itself once more without a restore (restore is exact).
  {
    Outcome got;
    ref->Reset();
    if (!ref->Feed(plan.prefix, {}, &error) ||
        !Continue(*ref, plan, &got, &error)) {
      return fail_with("ref direct");
    }
    compare("ref_direct_vs_restored", got, *ref);
  }

  const std::pair<const char*, l3::Session*> kinds[] = {
      {"full", full.get()}, {"partial", partial.get()}, {"zero", zero.get()}};
  std::int32_t salt = 0;
  for (const auto& [kind, s] : kinds) {
    const std::string k = kind;
    {
      Outcome got;
      if (!s->RestoreSnapshot(*shared, &error)) {
        return fail_with(k + " fresh restore");
      }
      got.expanded_after_restore = s->ExpandedRowsValid();
      if (!Continue(*s, plan, &got, &error)) {
        return fail_with(k + " fresh");
      }
      compare(k + "_fresh", got, *s);
    }
    {
      Outcome got;
      s->Reset();
      if (!s->Feed(plan.prefix, {}, &error) ||
          !Continue(*s, plan, &got, &error)) {
        return fail_with(k + " direct");
      }
      compare(k + "_direct", got, *s);
    }
    const std::pair<const char*, std::uint32_t> histories[] = {
        {"shared_history", share},
        {"same_prefix_history", P},
        {"unrelated_history", 0}};
    for (const auto& [label, keep] : histories) {
      Outcome got;
      ++salt;
      if (!Occupy(*s, history(keep, salt), &error) ||
          !s->RestoreSnapshot(*shared, &error)) {
        return fail_with(k + " " + label);
      }
      got.expanded_after_restore = s->ExpandedRowsValid();
      if (!Continue(*s, plan, &got, &error)) {
        return fail_with(k + " " + label + " continue");
      }
      compare(k + "_" + label, got, *s);
    }
  }

  report.Set("checks", std::move(checks));
  report.Set("failures", failures);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

}  // namespace gufo::tests::ling::restore_tail_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "restore_tail_test",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        namespace rt = gufo::tests::ling::restore_tail_tool;
        const auto ctx =
            static_cast<std::uint32_t>(args.Num("--max-context", 8192));
        const auto prefix =
            static_cast<std::uint32_t>(args.Num("--prefix-tokens", 5421));
        const auto partial = static_cast<std::uint32_t>(
            args.Num("--partial-blocks", prefix / rt::kBlockRows));
        const std::uint32_t full = (ctx + rt::kBlockRows - 1) / rt::kBlockRows;
        if (partial == 0 || partial >= full) {
          std::fprintf(stderr,
                       "restore_tail_test: --partial-blocks must be 1..%u\n",
                       full - 1);
          return 1;
        }
        // ref + full whole, partial the remainder, zero (and the leader)
        // nothing.
        const std::uint64_t budget =
            std::uint64_t{2 * full + partial} * rt::kBlockBytes;
        auto model = gufo::tests::ling::LoadModel(args, report, 512, 8192,
                                                  false, budget);
        return model ? rt::Run(args, report, model, partial) : 1;
      });
}
