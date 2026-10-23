// Host-only checks of the prompt-lookup index (prompt_lookup.hpp) and of the
// copy controller and proposals built on it (mtp.hpp, mtp_sampling.hpp):
// matches, the minimum length, preference for the longest and then the most
// recent match, the uncommitted tail, incremental indexing equal to indexing
// from scratch (against a brute-force reference of the same rule), restarts,
// token ids beyond the key packing, the memory and cost bounds of a 200k-token
// context, and the copy controller's backoff.
#include "src/models/ling_3_flash/prompt_lookup.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "src/models/ling_3_flash/mtp.hpp"
#include "src/models/ling_3_flash/mtp_sampling.hpp"

namespace l3 = gufo::models::ling_3_flash;

namespace {

int g_failures = 0;
void Expect(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
  if (!ok) {
    ++g_failures;
  }
}

using Tokens = std::vector<std::int32_t>;

/// The rule of PromptLookup::Find by brute force: every end position whose
/// preceding 3 tokens equal the context's last 3 (newest first, the first
/// kMaxCandidates), the longest backward agreement up to kMaxMatch (the most
/// recent on ties), and the minimum length.
l3::PromptLookup::Match Reference(
    const Tokens& committed, const Tokens& tail, std::size_t min_match,
    std::size_t indexed = std::numeric_limits<std::size_t>::max()) {
  Tokens context = committed;
  context.insert(context.end(), tail.begin(), tail.end());
  const std::size_t n = context.size();
  l3::PromptLookup::Match best;
  if (n < l3::PromptLookup::kKeyTokens) {
    return best;
  }
  std::size_t seen = 0;
  for (std::size_t end = committed.size();
       end-- > l3::PromptLookup::kKeyTokens;) {
    bool key = true;
    for (std::size_t i = 1; i <= l3::PromptLookup::kKeyTokens; ++i) {
      key = key && committed[end - i] == context[n - i];
    }
    if (!key) {
      continue;
    }
    if (++seen > l3::PromptLookup::kMaxCandidates) {
      break;
    }
    std::size_t length = 0;
    while (length < l3::PromptLookup::kMaxMatch && length < end && length < n &&
           committed[end - length - 1] == context[n - length - 1]) {
      ++length;
    }
    if (length > best.length) {
      best = {end, length};
    }
  }
  return best.length >= std::max(min_match, l3::PromptLookup::kKeyTokens)
             ? best
             : l3::PromptLookup::Match{};
}

bool Same(const l3::PromptLookup::Match& a, const l3::PromptLookup::Match& b) {
  return a.start == b.start && a.length == b.length;
}

Tokens Passage(std::int32_t base, std::size_t n) {
  Tokens t;
  for (std::size_t i = 0; i < n; ++i) {
    t.push_back(base + static_cast<std::int32_t>((i * 7 + i * i) % 997));
  }
  return t;
}

}  // namespace

int main() {
  using l3::PromptLookup;
  // A repeated passage: the context ends where the passage began again; the
  // proposals are what followed it.
  {
    const Tokens a = Passage(1000, 40);
    Tokens committed = Passage(5000, 30);
    committed.insert(committed.end(), a.begin(), a.end());
    const Tokens filler = Passage(9000, 25);
    committed.insert(committed.end(), filler.begin(), filler.end());
    // The context: the first 20 tokens of the passage again.
    committed.insert(committed.end(), a.begin(), a.begin() + 19);
    const Tokens tail = {a[19]};
    PromptLookup lookup;
    lookup.Extend(committed);
    const auto match = lookup.Find(committed, tail);
    Expect(match.length == 20 && match.start == 30 + 20,
           "repeated passage: matched 20 tokens, proposals start after them");
    Expect(Same(match, Reference(committed, tail, 12)),
           "repeated passage: equals the brute-force rule");
    // Proposals are the tokens that followed.
    Expect(
        committed[match.start] == a[20] && committed[match.start + 1] == a[21],
        "repeated passage: the proposals continue the earlier passage");
    // Without the tail token the context is 19 tokens and still matches.
    Expect(lookup.Find(committed, {}).length == 19,
           "an empty tail matches the committed context");
    // A different pending token breaks the match.
    const Tokens other = {a[19] + 1};
    Expect(lookup.Find(committed, other).length == 0,
           "a diverging tail token finds nothing");
  }
  // Minimum length.
  {
    const Tokens a = Passage(100, 10);
    Tokens committed = a;
    const Tokens gap = Passage(3000, 12);
    committed.insert(committed.end(), gap.begin(), gap.end());
    committed.insert(committed.end(), a.begin(), a.begin() + 7);
    const Tokens tail = {a[7]};
    PromptLookup lookup;
    lookup.Extend(committed);
    Expect(lookup.Find(committed, tail).length == 0,
           "an 8-token match proposes nothing at the default minimum of 12");
    PromptLookup eager(6);
    eager.Extend(committed);
    const auto match = eager.Find(committed, tail);
    Expect(match.length == 8 && match.start == 8,
           "the same context matches with a minimum of 6");
    PromptLookup floor(1);
    Expect(floor.min_match() == PromptLookup::kKeyTokens,
           "the minimum is at least the key length");
  }
  // Longest match wins over the more recent one.
  {
    const Tokens long_run = Passage(200, 30);
    Tokens committed = long_run;  // earlier: long agreement
    committed.push_back(777);     // continuation 777
    const Tokens mid = Passage(4000, 20);
    committed.insert(committed.end(), mid.begin(), mid.end());
    // More recent: agrees only on the last 14 tokens of the run.
    committed.push_back(1);
    committed.insert(committed.end(), long_run.end() - 14, long_run.end());
    committed.push_back(888);  // continuation 888
    const Tokens filler = Passage(6000, 15);
    committed.insert(committed.end(), filler.begin(), filler.end());
    committed.push_back(2);
    committed.insert(committed.end(), long_run.begin(), long_run.end() - 1);
    const Tokens tail = {long_run.back()};
    PromptLookup lookup;
    lookup.Extend(committed);
    const auto match = lookup.Find(committed, tail);
    Expect(match.length >= 30 && committed[match.start] == 777,
           "the longest agreement wins over a more recent shorter one");
    Expect(Same(match, Reference(committed, tail, 12)),
           "longest-match case equals the brute-force rule");
  }
  // Ties go to the most recent occurrence.
  {
    const Tokens run = Passage(300, 14);
    Tokens committed = {50};
    committed.insert(committed.end(), run.begin(), run.end());
    committed.push_back(111);
    committed.push_back(51);
    committed.insert(committed.end(), run.begin(), run.end());
    committed.push_back(222);
    committed.push_back(52);
    committed.insert(committed.end(), run.begin(), run.begin() + 13);
    const Tokens tail = {run[13]};
    PromptLookup lookup;
    lookup.Extend(committed);
    const auto match = lookup.Find(committed, tail);
    Expect(match.length == 14 && committed[match.start] == 222,
           "equal agreement: the most recent occurrence");
    Expect(Same(match, Reference(committed, tail, 12)),
           "tie case equals the brute-force rule");
  }
  // Incremental indexing equals indexing from scratch, on text that repeats.
  {
    std::mt19937 rng(7);
    Tokens text;
    const Tokens phrases[3] = {Passage(100, 40), Passage(700, 25),
                               Passage(1500, 60)};
    for (int round = 0; round < 60; ++round) {
      const Tokens& p = phrases[rng() % 3];
      const std::size_t cut = rng() % p.size();
      text.insert(text.end(), p.begin(), p.begin() + static_cast<long>(cut));
      for (int i = 0; i < 3; ++i) {
        text.push_back(static_cast<std::int32_t>(rng() % 5000));
      }
    }
    PromptLookup incremental;
    std::size_t checked = 0;
    std::size_t matched = 0;
    bool equal = true;
    for (std::size_t size = 3; size + 1 < text.size(); size += 1 + size % 5) {
      const Tokens committed(text.begin(),
                             text.begin() + static_cast<long>(size));
      const Tokens tail = {text[size]};
      incremental.Extend(committed);
      PromptLookup fresh;
      fresh.Extend(committed);
      const auto a = incremental.Find(committed, tail);
      const auto b = fresh.Find(committed, tail);
      const auto c = Reference(committed, tail, 12);
      equal = equal && Same(a, b) && Same(a, c);
      matched += a.length != 0 ? 1 : 0;
      ++checked;
    }
    Expect(equal,
           "incremental, from-scratch and brute-force matches agree at " +
               std::to_string(checked) + " points");
    Expect(matched > 10, "the repeated text produced matches (" +
                             std::to_string(matched) + ")");
  }
  // Bounded indexing: a long prefix is indexed over several calls; Find()
  // searches the indexed prefix meanwhile (exactly what a full build would
  // find among those positions) and equals a full build once caught up.
  {
    const Tokens a = Passage(1, 300);
    Tokens committed = a;
    committed.insert(committed.end(), a.begin(), a.begin() + 40);
    const Tokens tail = {a[40]};
    PromptLookup stepped;
    std::size_t calls = 0;
    bool partial_exact = true;
    bool saw_match_while_behind = false;
    while (stepped.Size() < committed.size()) {
      stepped.Extend(committed, 100);
      ++calls;
      const auto got = stepped.Find(committed, tail);
      partial_exact = partial_exact &&
                      Same(got, Reference(committed, tail, 12, stepped.Size()));
      saw_match_while_behind =
          saw_match_while_behind ||
          (stepped.Size() < committed.size() && got.length != 0);
    }
    PromptLookup full;
    full.Extend(committed);
    Expect(calls == 4 && partial_exact,
           "a bounded Extend takes several calls and Find equals the "
           "brute-force rule over the indexed prefix");
    Expect(saw_match_while_behind,
           "Find proposes from the indexed prefix while catching up");
    Expect(Same(stepped.Find(committed, tail), full.Find(committed, tail)) &&
               stepped.Find(committed, tail).length == 41,
           "the stepped index equals a full build");
  }
  // Stepped, then appended, truncated and re-extended; and the Feed pattern
  // (n + 4096 per call, from empty and over a restored prefix).
  {
    std::mt19937 rng(5);
    Tokens text;
    const Tokens phrases[3] = {Passage(100, 40), Passage(700, 25),
                               Passage(1500, 60)};
    while (text.size() < 30000) {
      const Tokens& p = phrases[rng() % 3];
      text.insert(text.end(), p.begin(),
                  p.begin() + static_cast<long>(rng() % p.size()));
      for (int i = 0; i < 3; ++i) {
        text.push_back(static_cast<std::int32_t>(rng() % 5000));
      }
    }
    const auto agrees = [&](const PromptLookup& a, std::size_t size) {
      const Tokens committed(text.begin(),
                             text.begin() + static_cast<long>(size));
      PromptLookup fresh;
      fresh.Extend(committed);
      bool ok = a.Size() == size;
      const Tokens tail = {text[size]};
      // Several pending tokens: the true next one and the tokens that follow
      // earlier occurrences of the context's last key.
      for (std::size_t k = 0; k < 40 && ok; ++k) {
        const Tokens t = {text[size - 1 - 7 * k]};
        ok = ok && Same(a.Find(committed, t), fresh.Find(committed, t)) &&
             Same(a.Find(committed, t), Reference(committed, t, 12));
      }
      return ok && Same(a.Find(committed, tail), fresh.Find(committed, tail)) &&
             Same(a.Find(committed, tail), Reference(committed, tail, 12));
    };
    // Stepped, appended, truncated below the indexed length, extended again.
    PromptLookup lookup;
    Tokens committed(text.begin(), text.begin() + 20000);
    lookup.Extend(committed, 4096);
    lookup.Extend(committed, 4096);
    committed.insert(committed.end(), text.begin() + 20000,
                     text.begin() + 20008);
    lookup.Extend(committed, 4096);
    committed.resize(9000);  // a rollback below the indexed length restarts
    lookup.Extend(committed, 4096);
    while (lookup.Size() < committed.size()) {
      lookup.Extend(committed, 4096);
    }
    Expect(agrees(lookup, 9000),
           "stepped, appended and truncated index equals a full build");
    // Truncated but still above the indexed length (the verify rollback).
    PromptLookup rolled;
    Tokens run(text.begin(), text.begin() + 12000);
    rolled.Extend(run, 4096);
    run.insert(run.end(), text.begin() + 12000, text.begin() + 12008);
    run.resize(12003);
    while (rolled.Size() < run.size()) {
      rolled.Extend(run, 4096);
    }
    Expect(agrees(rolled, 12003),
           "a rollback above the indexed length keeps a correct index");
    // The Feed pattern: chunks of 512 with n + 4096 per call, from empty.
    PromptLookup feed;
    for (std::size_t end = 512; end <= 20480; end += 512) {
      const Tokens c(text.begin(), text.begin() + static_cast<long>(end));
      feed.Extend(c, 512 + 4096);
    }
    Expect(agrees(feed, 20480),
           "the Feed pattern from empty equals a full build");
    // Over a restored prefix: the first Extend sees the whole prefix at once.
    PromptLookup restored;
    for (std::size_t end = 20000; end <= 20000 + 512 * 8; end += 512) {
      const Tokens c(text.begin(), text.begin() + static_cast<long>(end));
      restored.Extend(c, 512 + 4096);
    }
    while (restored.Size() < 24096) {
      const Tokens c(text.begin(), text.begin() + 24096);
      restored.Extend(c, 4096);
    }
    Expect(agrees(restored, 24096),
           "the Feed pattern over a restored prefix equals a full build");
  }
  // Restarts: Clear, a shorter run, a changed history.
  {
    const Tokens a = Passage(1, 60);
    Tokens committed = a;
    committed.insert(committed.end(), a.begin(), a.begin() + 20);
    PromptLookup lookup;
    lookup.Extend(committed);
    const Tokens tail = {a[20]};
    Expect(lookup.Find(committed, tail).length == 21, "baseline match");
    // A shorter run restarts the index.
    const Tokens shorter(committed.begin(), committed.begin() + 30);
    lookup.Extend(shorter);
    Expect(lookup.Size() == shorter.size(), "a shorter run restarts the index");
    // Same length, different history: the last indexed token differs.
    Tokens changed = committed;
    changed[changed.size() - 1] += 1;
    lookup.Extend(committed);
    lookup.Extend(changed);
    Expect(Same(lookup.Find(changed, tail), Reference(changed, tail, 12)),
           "a rewritten last token restarts the index");
    lookup.Clear();
    Expect(lookup.Size() == 0 && lookup.HostBytes() == 0,
           "Clear frees the index");
    Expect(lookup.Find(committed, tail).length == 0,
           "an empty index finds nothing");
    lookup.Extend(committed);
    Expect(lookup.Find(committed, tail).length == 21, "re-indexed after Clear");
  }
  // Token ids past the 21-bit key packing only alias keys: no false match.
  {
    const std::int32_t big = 1 << 21;
    Tokens committed;
    for (int i = 0; i < 20; ++i) {
      committed.push_back(big + i);
    }
    Tokens context = committed;  // same low 21 bits, different tokens
    for (auto& t : context) {
      t += big;
    }
    Tokens all = committed;
    all.insert(all.end(), context.begin(), context.end() - 1);
    PromptLookup lookup;
    lookup.Extend(all);
    const Tokens tail = {context.back()};
    Expect(lookup.Find(all, tail).length == 0,
           "ids that alias in the key packing never match");
    Tokens negative = {-1, -2, -3, -1, -2};
    PromptLookup n;
    n.Extend(negative);
    Expect(n.Find(negative, {}).length == 0, "negative ids are harmless");
  }
  // A 200k-token context: bounded memory and per-step cost.
  {
    constexpr std::size_t kTokens = 200000;
    std::mt19937 rng(11);
    Tokens text;
    text.reserve(kTokens + 64);
    // Random ids from a 157k vocabulary: almost every 3-gram is new, the
    // worst case for the table.
    while (text.size() < kTokens) {
      text.push_back(static_cast<std::int32_t>(rng() % 157184));
    }

    PromptLookup lookup;
    const auto t0 = std::chrono::steady_clock::now();
    lookup.Extend(text);
    const auto build_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
    const double per_token = static_cast<double>(lookup.HostBytes()) /
                             static_cast<double>(text.size());
    std::printf("       200k tokens: build %.1f ms, %.1f MB (%.1f B/token)\n",
                build_ms, static_cast<double>(lookup.HostBytes()) / 1e6,
                per_token);
    // 4 B chain + 16 B per slot, at most 4 slots per key just after a doubling.
    Expect(per_token <= 68.0, "index memory is at most 68 bytes per token");
    // A step: append 8 tokens, probe.
    Tokens grown = text;
    std::size_t probes = 0;
    const auto t1 = std::chrono::steady_clock::now();
    for (int step = 0; step < 2000; ++step) {
      for (int i = 0; i < 8; ++i) {
        grown.push_back(static_cast<std::int32_t>(rng() % 157184));
      }
      lookup.Extend(grown);
      const Tokens tail = {static_cast<std::int32_t>(rng() % 157184)};
      (void)lookup.Find(grown, tail);
      ++probes;
    }
    const auto step_us = std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - t1)
                             .count() /
                         static_cast<double>(probes);
    std::printf("       per step (8 tokens appended + probe): %.2f us\n",
                step_us);
    Expect(step_us < 200.0, "a step costs well under 200 us at 200k tokens");
    // A degenerate context (one repeated token) stays bounded.
    Tokens flat(50000, 42);
    PromptLookup degenerate;
    degenerate.Extend(flat);
    const auto t2 = std::chrono::steady_clock::now();
    PromptLookup::Match m;
    for (int i = 0; i < 1000; ++i) {
      m = degenerate.Find(flat, {});
    }
    const auto flat_us = std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - t2)
                             .count() /
                         1000.0;
    Expect(m.length == PromptLookup::kMaxMatch && flat_us < 200.0,
           "a one-token context matches at the cap and probes in bounded time");
  }
  // Point-mass proposals (mtp_sampling.hpp).
  {
    const auto p = l3::CopyProposal(1234);
    Expect(p.size == 1 && p.ids[0] == 1234 && p.token == 1234 &&
               p.probabilities[0] == 1.0F && p.probability == 1.0F,
           "CopyProposal is the point mass on the copied token");
  }
  // The copy controller: starts willing, grows on success, backs off on
  // failure, and prices a draft below the MTP controller.
  {
    l3::MtpLengthController copies = l3::MtpLengthController::ForCopies(7);
    l3::MtpLengthController mtp(7);
    Expect(copies.RelativeCost(3, false, 0) < mtp.RelativeCost(3, false, 0),
           "a copy is priced without the drafter step");
    Expect(copies.Choose(7, false, 0) >= 1, "copies are tried at first");
    for (int i = 0; i < 40; ++i) {
      copies.Observe(7, 7, false, 0);
    }
    Expect(copies.Choose(7, false, 0) == 7,
           "fully accepted copies use all rows");
    l3::MtpLengthController failing = l3::MtpLengthController::ForCopies(7);
    for (int i = 0; i < 40; ++i) {
      failing.Observe(0, 3, false, 0);
    }
    Expect(failing.Choose(7, false, 0) == 0, "failing copies back off");
    for (int i = 0; i < 300; ++i) {
      failing.ObservePlainToken();
    }
    Expect(failing.Choose(7, false, 0) >= 1,
           "the backoff ends after its bounded interval");
    // A short match: never more than the copies available.
    Expect(copies.Choose(2, false, 0) == 2, "copies are capped by the match");
  }
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
