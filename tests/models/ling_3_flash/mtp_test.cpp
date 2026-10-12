// Full-model MTP (multi-token prediction) speculative decoding checks for
// Ling-3.0-flash on the GPU (mtp.hpp). Needs the model loaded with the MTP
// block (this tool always passes --mtp). Checks:
//   * greedy_equals_ar_*: greedy speculative decoding (several prompts, draft
//     limits 1, 3 and 7) is token-identical to greedy autoregressive decoding,
//     and the final KDA state, conv windows, MLA cache rows and logits are
//     bit-identical (the verify chunk is in the small-batch numeric family);
//     the drafts must actually have been proposed and accepted.
//   * rollback_*: after Verify() of 5 tokens, Rollback(keep) leaves the KDA
//     state, conv windows, MLA rows (7 trunk layers), logits, MTP cache rows
//     and hidden carry bit-identical to feeding only the kept tokens, and
//     decoding on from there stays bit-identical to a run that never saw the
//     rejected rows (real continuation drafts and unrelated garbage drafts).
//   * cache_fill_matches_draft: the MTP cache row written by a forward pass
//     equals the row a draft step writes from the same (hidden, token).
//   * constrained_*: with the server's Bailing tool grammar (strict const
//     value), greedy speculative decoding equals constrained autoregressive
//     decoding and sampled speculative decoding commits only grammar tokens.
//   * serving_strict_tool_*: InferenceBackend with MTP answers a forced
//     strict tool (the const value holds `</tool_call>`) with exactly that
//     value through the Bailing parser.
//   * serving_cached_replay_thinking_*: InferenceBackend with MTP restores a
//     repeated >= 128-token prompt whole (no prefill) after uncached requests
//     with other samplers, and replays the first answer (the openai_sdk
//     sampling-defaults sequence). "serving_cached_replay_short" reports the
//     same sequence for a prompt below Ling's snapshot floor.
//   * sampled_replay: a seeded sampled speculative run reproduces itself, also
//     from a restored snapshot (so snapshots carry the MTP state); the draft
//     path was exercised.
//   * sampled_replay_on_recycled_session: a request after Reset() of a used
//     session equals the same request on a fresh session (no controller leak).
//   * throw_after_verify_invalidates_session: an exception between Verify()
//     and Rollback() leaves the session invalid until Reset().
//   * snapshot_mtp_state: payload size and exact restore of the MTP rows and
//     hidden carry.
// Exit 77 without --model.
//
//   mtp_test --model FIRST_SHARD.gguf [--gen-tokens N] [--max-context N]
//            [--max-batch N] [--timeout-seconds S] [--hold-seconds S]

#include <array>
#include <random>
#include <stdexcept>

#define LING_TOOL_NO_MAIN
#include "tests/models/ling_3_flash/session_test.cpp"

namespace gufo::tests::ling::mtp_tool {

using session_tool::Capture;
using session_tool::Compare;
using session_tool::Diff;
using session_tool::Take;
using session_tool::ToJson;

namespace sampling = gufo::sampling;

const char* const kPrompts[] = {
    // Prose (the session tests' text).
    nullptr,
    // Code.
    "def fibonacci(n):\n    if n < 2:\n        return n\n    a, b = 0, 1\n"
    "    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n\n"
    "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n"
    "        result *= i\n    return result\n\n"
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n"
    "def lcm(a, b):\n",
    // Counting and lists (very predictable).
    "One, two, three, four, five, six, seven, eight, nine, ten, eleven, "
    "twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, "
    "nineteen, twenty, twenty-one, twenty-two, twenty-three, twenty-four, "
    "twenty-five, twenty-six, twenty-seven,",
    // Multilingual.
    "Berlin ist die Hauptstadt von Deutschland. Paris est la capitale de la "
    "France. Madrid es la capital de Espana. \xe5\x8c\x97\xe4\xba\xac\xe6"
    "\x98\xaf\xe4\xb8\xad\xe5\x9b\xbd\xe7\x9a\x84\xe9\xa6\x96\xe9\x83\xbd"
    "\xe3\x80\x82 Tokyo is the capital of Japan. Rome is the capital of",
};

struct Run {
  std::vector<std::int32_t> tokens;
  Capture end;
  std::vector<std::uint16_t> mtp_rows;
  std::vector<float> mtp_hidden;
  l3::Session::DraftStats stats;
};

bool Finish(l3::Session& s, std::uint32_t layers, Run* out,
            std::string* error) {
  if (!Take(s, layers, &out->end, error)) {
    return false;
  }
  out->stats = s.Statistics();
  return s.ReadMtpCache(s.Position() - 1, &out->mtp_rows, error) &&
         s.ReadMtpHidden(&out->mtp_hidden, error);
}

bool GenerateAr(l3::Model& model, l3::Session& s,
                const sampling::SamplingConfig& config,
                std::span<const std::int32_t> prompt, std::uint32_t count,
                std::uint32_t layers, Run* out, std::string* error) {
  if (!s.Feed(prompt, {}, error)) {
    return false;
  }
  sampling::SamplerState sampler(config);
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto token = static_cast<std::int32_t>(sampler.Sample(s.Logits()));
    if (model.IsStopToken(token)) {
      break;
    }
    out->tokens.push_back(token);
    sampler.Accept(static_cast<sampling::TokenId>(token));
    if (!s.Evaluate(token, error)) {
      return false;
    }
  }
  return Finish(s, layers, out, error);
}

bool GenerateMtp(l3::Model& model, l3::Session& s,
                 const sampling::SamplingConfig& config,
                 std::span<const std::int32_t> prompt, std::uint32_t count,
                 std::uint32_t max_drafts, std::uint32_t layers, Run* out,
                 std::string* error, bool feed_prompt = true,
                 l3::Session::DraftHooks hooks = {},
                 bool configure_drafts = true, bool stop_at_eos = true) {
  if (feed_prompt && !s.Feed(prompt, {}, error)) {
    return false;
  }
  if (configure_drafts) {
    // SetMaxDrafts() installs a fresh controller; the recycled-session check
    // skips it to prove Reset() alone clears the history.
    s.SetMaxDrafts(max_drafts);
  }
  s.SetDraftHooks(std::move(hooks));
  sampling::SamplerState sampler(config);
  while (out->tokens.size() < count) {
    const std::size_t budget =
        std::min<std::size_t>(l3::kMaxVerifyTokens, count - out->tokens.size());
    l3::Session::DecodeResult step;
    sampling::SamplerState working = sampler;
    if (!s.DecodeStep(budget, working, &step, error, stop_at_eos)) {
      return false;
    }
    sampler.CopyDrawStateFrom(working);
    for (const auto token : step.tokens) {
      sampler.Accept(static_cast<sampling::TokenId>(token));
      out->tokens.push_back(token);
    }
    if (step.stop) {
      break;
    }
  }
  return Finish(s, layers, out, error);
}

bool SameState(const Run& a, const Run& b) {
  return Compare(a.end, b.end).bitwise && a.end.logits == b.end.logits;
}

int Execute(const Args& args, Report& report,
            std::shared_ptr<l3::Model> model) {
  const std::uint32_t layers = model->LayerCount();
  const auto G = static_cast<std::uint32_t>(args.Num("--gen-tokens", 48));
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
    std::fprintf(stderr, "mtp_test: %s: %s\n", what, error.c_str());
    report.Set("status", "error");
    report.Set("error", std::string(what) + ": " + error);
    report.Set("checks", std::move(checks));
    return 1;
  };
  if (!model->HasMtp()) {
    error = "model was loaded without MTP";
    return fail_with("setup");
  }
  const auto fresh = [&]() { return model->CreateSession(ctx, &error); };

  std::vector<std::vector<std::int32_t>> prompts;
  for (const char* text : kPrompts) {
    std::string t = text != nullptr ? text : session_tool::kText;
    auto ids = model->Tokenize(t);
    if (text == nullptr && ids.size() > 120) {
      ids.resize(120);
    }
    prompts.push_back(std::move(ids));
  }

  // ---- greedy speculative == greedy autoregressive ------------------------
  sampling::SamplingConfig greedy;
  std::uint64_t total_drafted = 0;
  std::uint64_t total_accepted = 0;
  for (std::size_t p = 0; p < prompts.size(); ++p) {
    auto s_ar = fresh();
    Run ar;
    if (!s_ar || !GenerateAr(*model, *s_ar, greedy, prompts[p], G, layers, &ar,
                             &error)) {
      return fail_with("autoregressive run");
    }
    for (const std::uint32_t k : {1U, 3U, 7U}) {
      auto s = fresh();
      Run mtp;
      if (!s || !GenerateMtp(*model, *s, greedy, prompts[p], G, k, layers, &mtp,
                             &error)) {
        return fail_with("speculative run");
      }
      auto o = json::Value::object();
      o["prompt_tokens"] = U(prompts[p].size());
      o["max_drafts"] = U(k);
      o["cycles"] = U(mtp.stats.cycles);
      o["drafted"] = U(mtp.stats.drafted);
      o["accepted"] = U(mtp.stats.accepted);
      o["tokens_equal"] = ar.tokens == mtp.tokens;
      const Diff d = Compare(mtp.end, ar.end);
      o["final_state"] = ToJson(d);
      o["mtp_cache_nonzero"] =
          std::any_of(mtp.mtp_rows.begin(), mtp.mtp_rows.end(),
                      [](std::uint16_t v) { return v != 0; });
      total_drafted += mtp.stats.drafted;
      total_accepted += mtp.stats.accepted;
      check("greedy_equals_ar_p" + std::to_string(p) + "_k" + std::to_string(k),
            ar.tokens == mtp.tokens && ar.tokens.size() == G &&
                session_tool::Exact(d) && mtp.stats.drafted > 0,
            std::move(o));
    }
  }
  {
    // A truncated trunk (--layers, debug aid) has meaningless hidden states:
    // its drafts are not expected to be accepted.
    const bool full = model->LayerCount() == l3::kTrunkLayers;
    auto o = json::Value::object();
    o["drafted"] = U(total_drafted);
    o["accepted"] = U(total_accepted);
    o["full_trunk"] = full;
    check("drafts_proposed_and_accepted",
          total_drafted > 0 && (total_accepted > 0 || !full), std::move(o));
  }

  // ---- forced acceptance patterns ------------------------------------------
  // With a draft source that replays the autoregressive continuation (and
  // corrupts it at chosen depths), every cycle accepts all, some or none of
  // its drafts, whatever the quality of the MTP drafts: the verified output and
  // the final state must still be those of autoregressive decoding.
  {
    const auto& prompt = prompts[0];
    auto s_ar = fresh();
    Run ar;
    if (!s_ar ||
        !GenerateAr(*model, *s_ar, greedy, prompt, G, layers, &ar, &error)) {
      return fail_with("pattern reference run");
    }
    struct Pattern {
      const char* name;
      std::uint32_t drafts;
      int wrong_depth;  // 1-based depth to corrupt, 0 = none, -1 = every depth
    };
    for (const Pattern pat :
         {Pattern{"all_accepted_k3", 3, 0}, Pattern{"all_accepted_k7", 7, 0},
          Pattern{"none_accepted_k4", 4, -1}, Pattern{"reject_depth2_k5", 5, 2},
          Pattern{"reject_depth5_k7", 7, 5},
          Pattern{"reject_depth1_k2", 2, 1}}) {
      auto s = fresh();
      Run mtp;
      std::uint32_t emitted_base = 0;
      l3::Session::DraftHooks hooks;
      hooks.fixed_drafts = pat.drafts;
      // chain = [anchor, drafts so far]; the continuation after the anchor at
      // the committed prefix is ar.tokens[committed + chain.size()].
      hooks.source = [&](std::span<const std::int32_t> chain) {
        const std::size_t at = mtp.tokens.size() + chain.size();
        const std::int32_t truth = at < ar.tokens.size() ? ar.tokens[at] : 0;
        const auto depth = static_cast<int>(chain.size());  // draft depth
        const bool wrong = pat.wrong_depth == -1 || pat.wrong_depth == depth;
        return wrong ? (truth + 1) % static_cast<std::int32_t>(l3::kVocab)
                     : truth;
      };
      (void)emitted_base;
      if (!s || !GenerateMtp(*model, *s, greedy, prompt, G, 7, layers, &mtp,
                             &error, true, hooks)) {
        return fail_with("pattern run");
      }
      const Diff d = Compare(mtp.end, ar.end);
      auto o = json::Value::object();
      o["cycles"] = U(mtp.stats.cycles);
      o["drafted"] = U(mtp.stats.drafted);
      o["accepted"] = U(mtp.stats.accepted);
      o["final_state"] = ToJson(d);
      const bool accepted_ok =
          pat.wrong_depth == 0    ? mtp.stats.accepted == mtp.stats.drafted
          : pat.wrong_depth == -1 ? mtp.stats.accepted == 0
                                  : mtp.stats.accepted < mtp.stats.drafted;
      o["accepted_as_forced"] = accepted_ok;
      check(std::string("forced_acceptance_") + pat.name,
            ar.tokens == mtp.tokens && session_tool::Exact(d) && accepted_ok &&
                mtp.stats.cycles > 0,
            std::move(o));
    }
  }

  // ---- rollback ------------------------------------------------------------
  {
    const auto& prompt = prompts[0];
    auto s_ar = fresh();
    Run ar;
    if (!s_ar ||
        !GenerateAr(*model, *s_ar, greedy, prompt, 8, layers, &ar, &error)) {
      return fail_with("rollback reference run");
    }
    std::mt19937 rng(1234);
    const auto make_chunk = [&](bool real) {
      std::vector<std::int32_t> chunk(5);
      for (std::size_t i = 0; i < chunk.size(); ++i) {
        chunk[i] =
            real ? ar.tokens[i] : static_cast<std::int32_t>(rng() % l3::kVocab);
      }
      return chunk;
    };
    for (const bool real : {true, false}) {
      const auto chunk = make_chunk(real);
      for (std::uint32_t keep = 1; keep <= chunk.size(); ++keep) {
        auto s = fresh();
        auto ref = fresh();
        std::array<std::int32_t, l3::kMaxVerifyTokens> argmax{};
        if (!s || !ref || !s->Feed(prompt, {}, &error) ||
            !ref->Feed(prompt, {}, &error) ||
            !s->Verify(chunk, argmax, nullptr, &error) ||
            !s->Rollback(keep, &error) ||
            !ref->Feed(std::span(chunk).first(keep), {}, &error)) {
          return fail_with("rollback setup");
        }
        Run a;
        Run b;
        if (!Finish(*s, layers, &a, &error) ||
            !Finish(*ref, layers, &b, &error)) {
          return fail_with("rollback capture");
        }
        auto o = json::Value::object();
        o["kept"] = U(keep);
        o["real_continuation"] = real;
        o["state"] = ToJson(Compare(a.end, b.end));
        o["mtp_rows_equal"] = a.mtp_rows == b.mtp_rows;
        o["mtp_hidden_equal"] = a.mtp_hidden == b.mtp_hidden;
        const bool state_ok = SameState(a, b) && a.mtp_rows == b.mtp_rows &&
                              a.mtp_hidden == b.mtp_hidden &&
                              s->Position() == ref->Position();
        // Decode on from the rolled-back state: the next tokens must see the
        // state of a run that never had the rejected rows.
        bool on_ok = true;
        for (int i = 0; i < 4 && on_ok; ++i) {
          const auto token = static_cast<std::int32_t>(1000 + 37 * i);
          on_ok = s->Evaluate(token, &error) && ref->Evaluate(token, &error) &&
                  std::equal(s->Logits().begin(), s->Logits().end(),
                             ref->Logits().begin(), ref->Logits().end());
        }
        Run a2;
        Run b2;
        on_ok = on_ok && Finish(*s, layers, &a2, &error) &&
                Finish(*ref, layers, &b2, &error) && SameState(a2, b2);
        o["decode_on_equal"] = on_ok;
        check(std::string("rollback_") + (real ? "real" : "garbage") + "_keep" +
                  std::to_string(keep),
              state_ok && on_ok, std::move(o));
      }
    }
    // The device argmax of every verify row equals the host argmax of its
    // logits, and logits rows equal single decodes (family contract).
    {
      auto s = fresh();
      auto ref = fresh();
      const auto chunk = make_chunk(true);
      std::array<std::int32_t, l3::kMaxVerifyTokens> argmax{};
      std::vector<float> rows;
      bool ok = s && ref && s->Feed(prompt, {}, &error) &&
                ref->Feed(prompt, {}, &error) &&
                s->Verify(chunk, argmax, &rows, &error);
      bool rows_equal = ok;
      bool argmax_ok = ok;
      for (std::size_t i = 0; ok && i < chunk.size(); ++i) {
        ok = ref->Evaluate(chunk[i], &error);
        const std::span<const float> row(rows.data() + i * l3::kVocab,
                                         l3::kVocab);
        rows_equal = rows_equal && ok &&
                     std::equal(row.begin(), row.end(), ref->Logits().begin(),
                                ref->Logits().end());
        argmax_ok = argmax_ok && ok &&
                    argmax[i] == (std::max_element(row.begin(), row.end()) -
                                  row.begin());
      }
      auto o = json::Value::object();
      o["rows_equal_single_decode"] = rows_equal;
      o["device_argmax_equals_host"] = argmax_ok;
      check("verify_rows_equal_single_decode", ok && rows_equal && argmax_ok,
            std::move(o));
      if (s->IsValid() && !s->Rollback(chunk.size(), &error)) {
        return fail_with("rollback after rows");
      }
    }
  }

  // ---- cache fill equals draft ---------------------------------------------
  {
    const auto& prompt = prompts[0];
    auto s = fresh();
    std::vector<std::uint16_t> from_draft, from_fill;
    std::vector<float> h_draft, h_fill;
    const std::int32_t x = prompt.back() == 3 ? 5 : 3;
    std::int32_t arg = -1;
    const bool ok = s && s->Feed(prompt, {}, &error) &&
                    s->Draft(x, false, {}, &arg, &error) &&
                    s->ReadMtpCache(s->Position(), &from_draft, &error) &&
                    s->Evaluate(x, &error) &&
                    s->ReadMtpCache(s->Position() - 1, &from_fill, &error);
    const std::size_t row =
        std::size_t{static_cast<std::uint32_t>(prompt.size() - 1)} *
        l3::kMlaCacheWidth;
    const bool same =
        ok && std::equal(from_draft.begin() + row,
                         from_draft.begin() + row + l3::kMlaCacheWidth,
                         from_fill.begin() + row);
    auto o = json::Value::object();
    o["draft_argmax"] = arg;
    check("cache_fill_matches_draft", same, std::move(o));
  }

  // ---- grammar-constrained decoding ---------------------------------------
  // The Bailing tool grammar of the server (a forced, strict call whose const
  // value holds `</tool_call>`) bound to the real vocabulary. The drafts are
  // unconstrained; the verifier applies the mask (greedy: VerifyDraft with a
  // constrained Sample; sampled: p from the masked Distribution). Greedy
  // speculative output must equal constrained autoregressive output token for
  // token, and every committed token must keep the grammar viable.
  {
    // A truncated trunk may wander in the admitted whitespace before the call;
    // only the full trunk must finish it within the token budget.
    const bool full = model->LayerCount() == l3::kTrunkLayers;
    constexpr std::string_view kLiteral = "<think>literal</think></tool_call>";
    auto schema = json::parse(R"({"type":"object","properties":{
      "value":{"type":"string"}},"required":["value"],
      "additionalProperties":false})");
    schema["properties"]["value"]["const"] = std::string(kLiteral);
    using Constraint = sampling::JsonConstraint;
    const auto grammar = Constraint::WithTools(
        nullptr,
        {{"record", Constraint::ToolParameters(
                        schema, true, Constraint::ToolFormat::kBailing, true)}},
        true, false, Constraint::ToolFormat::kBailing);
    auto binding = std::make_shared<sampling::TokenConstraint>();
    binding->grammar = grammar;
    binding->vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
        model->VocabSize(), [&](std::uint32_t id) {
          const auto token = static_cast<std::int32_t>(id);
          return sampling::ConstraintVocabulary::Piece{
              model->TokenText(token), model->IsStopToken(token)};
        });
    const auto viable = [&](const Run& run, bool* complete) {
      std::string text;
      for (const auto token : run.tokens)
        text += model->TokenText(token);
      auto state = grammar->Start();
      for (const unsigned char byte : text) {
        state = grammar->Advance(state, byte);
        if (state.empty())
          return false;
      }
      *complete = text.find("<arg_value>" + std::string(kLiteral) +
                            "</arg_value>") != std::string::npos;
      return true;
    };
    sampling::SamplingConfig constrained;
    constrained.constraint = binding;
    const auto& prompt = prompts[0];
    auto s_ar = fresh();
    Run ar;
    if (!s_ar || !GenerateAr(*model, *s_ar, constrained, prompt, G, layers, &ar,
                             &error)) {
      return fail_with("constrained autoregressive run");
    }
    for (const std::uint32_t k : {1U, 3U}) {
      auto s = fresh();
      Run mtp;
      if (!s || !GenerateMtp(*model, *s, constrained, prompt, G, k, layers,
                             &mtp, &error)) {
        return fail_with("constrained speculative run");
      }
      bool complete = false;
      const bool ok = viable(mtp, &complete);
      auto o = json::Value::object();
      o["max_drafts"] = U(k);
      o["drafted"] = U(mtp.stats.drafted);
      o["accepted"] = U(mtp.stats.accepted);
      o["tokens"] = U(mtp.tokens.size());
      o["tokens_equal"] = ar.tokens == mtp.tokens;
      o["call_complete"] = complete;
      o["full_trunk"] = full;
      check("constrained_greedy_equals_ar_k" + std::to_string(k),
            ar.tokens == mtp.tokens && ok && (complete || !full) &&
                mtp.stats.drafted > 0 &&
                session_tool::Exact(Compare(mtp.end, ar.end)),
            std::move(o));
    }
    sampling::SamplingConfig sampled = constrained;
    sampled.temperature = 0.8F;
    sampled.top_p = 0.95F;
    sampled.top_k = 40;
    sampled.seed = 4321;
    auto s1 = fresh();
    auto s2 = fresh();
    Run a, b;
    if (!s1 || !s2 ||
        !GenerateMtp(*model, *s1, sampled, prompt, G, 3, layers, &a, &error) ||
        !GenerateMtp(*model, *s2, sampled, prompt, G, 3, layers, &b, &error)) {
      return fail_with("constrained sampled runs");
    }
    bool complete = false;
    const bool ok = viable(a, &complete);
    auto o = json::Value::object();
    o["drafted"] = U(a.stats.drafted);
    o["accepted"] = U(a.stats.accepted);
    o["tokens"] = U(a.tokens.size());
    o["call_complete"] = complete;
    check("constrained_sampled_obeys_grammar",
          ok && (complete || !full) && a.tokens == b.tokens &&
              a.stats.drafted > 0,
          std::move(o));

    // The serving path end to end: ConstrainChatRequest binds the Bailing
    // grammar for a forced strict tool, the MTP runner decodes under it and
    // the Bailing parser reads the const value back. The call admits only
    // optional newlines besides its fixed bytes, so it completes within the
    // budget even on a truncated trunk.
    {
      namespace server = gufo::server;
      server::InferenceBackend backend;
      server::TextSpeculativeConfig speculative;
      speculative.backend = server::TextSpeculativeBackend::kMtp;
      if (!backend.load(model, &error, std::min<std::uint32_t>(ctx, 2048), 1,
                        {}, {}, speculative)) {
        return fail_with("serving load");
      }
      auto definition = json::parse(R"({"type":"function","function":{
        "name":"record","strict":true}})");
      definition["function"]["parameters"] = schema;
      server::ChatRequest request({{gufo::tokenization::ChatRole::kUser,
                                    "Call record once with the required "
                                    "value. No explanation."}});
      request.reasoning.enabled = false;
      request.tools = {{.name = "record",
                        .parameters_json = schema.dump(),
                        .definition_json = definition.dump()}};
      request.tool_choice = server::ChatRequest::ToolChoice::kRequired;
      request.forced_tool_name = "record";
      request.parallel_tool_calls = false;
      for (const float temperature : {0.0F, 0.8F}) {
        sampling::SamplingConfig config;
        config.temperature = temperature;
        config.seed = 99;
        const auto result = backend.chat(request, 256, config);
        const auto scan = models::ling_3_flash::ScanBailingToolCalls(
            result.text, models::ling_3_flash::BailingTools(request.tools),
            true, false);
        const bool exact = scan.calls.size() == 1 &&
                           scan.calls[0].name == "record" &&
                           scan.calls[0].arguments.size() == 1 &&
                           scan.calls[0].arguments[0].name == "value" &&
                           scan.calls[0].arguments[0].value == kLiteral;
        auto serving = json::Value::object();
        serving["temperature"] = temperature;
        serving["completion_tokens"] = U(result.completion_tokens);
        serving["draft_tokens"] = U(result.draft_tokens);
        serving["text_head"] = result.text.substr(0, 400);
        check(std::string("serving_strict_tool_") +
                  (temperature == 0.0F ? "greedy" : "sampled"),
              exact, std::move(serving));
      }
    }
  }

  // ---- serving: cached replay of a repeated prompt -------------------------
  // The openai_sdk sampling-defaults sequence with MTP on: the prompt with
  // cache_prompt=false and the Ling preset, then with greedy penalties, then
  // with cache_prompt=true and the preset again. The last must restore the
  // whole prompt (no prefill) and replay the first answer. The Ling sampling
  // prompt clears the runner's 128-token snapshot floor; the bare question
  // (31 tokens) stays below it and is reported, not checked, because no prompt
  // snapshot exists and the decoded frontier cannot be cut back to it.
  {
    namespace server = gufo::server;
    server::InferenceBackend backend;
    server::TextSpeculativeConfig speculative;
    speculative.backend = server::TextSpeculativeBackend::kMtp;
    if (!backend.load(model, &error, std::min<std::uint32_t>(ctx, 2048), 2, {},
                      {}, speculative)) {
      return fail_with("serving cache load");
    }
    constexpr std::string_view kQuestion =
        "Name three animals and describe each in one sentence.";
    // tests/functional/openai_sdk.py SAMPLING_CONTEXT.
    constexpr std::string_view kContext =
        "You are helping a teacher prepare a short handout for a "
        "primary-school nature club. The club meets every Thursday afternoon "
        "in the school garden, where the children watch birds, look for "
        "insects under stones and keep a notebook of everything they find. "
        "The teacher wants the handout to be friendly, accurate and easy to "
        "read aloud, with plain words and no jargon. Each entry should say "
        "where the animal lives and one thing that makes it special, so the "
        "children can remember it on their next walk. Keep the tone warm and "
        "curious, and avoid anything frightening. The handout will be printed "
        "on a single page together with a few drawings made by the children "
        "themselves.\n\n";
    sampling::SamplingConfig preset;
    preset.temperature = 0.6F;
    preset.top_p = 0.95F;
    preset.top_k = 20;
    preset.seed = 73;
    sampling::SamplingConfig penalties;
    penalties.seed = 73;
    penalties.frequency_penalty = 0.3F;
    penalties.presence_penalty = 0.4F;
    penalties.repeat_penalty = 1.1F;
    penalties.repeat_last_n = 16;
    const auto replay = [&](const std::string& content, bool thinking) {
      server::ChatRequest request(
          {{gufo::tokenization::ChatRole::kUser, content}});
      request.reasoning.enabled = thinking;
      request.cache_prompt = false;
      const auto inherited = backend.chat(request, 8, preset);
      (void)backend.chat(request, 8, penalties);
      request.cache_prompt = true;
      const auto cached = backend.chat(request, 8, preset);
      auto o = json::Value::object();
      o["thinking"] = thinking;
      o["prompt_tokens"] = U(cached.prompt_tokens);
      o["cached_tokens"] = U(cached.cached_prompt_tokens);
      o["prefill_tokens"] = U(cached.prefill_tokens);
      o["cache_hit"] = cached.cache_hit;
      o["cache_miss_reason"] = cached.cache_miss_reason;
      o["cache_common_prefix_tokens"] = U(cached.cache_common_prefix_tokens);
      o["cache_checkpoint_tokens"] = U(cached.cache_checkpoint_tokens);
      o["draft_tokens"] = U(inherited.draft_tokens + cached.draft_tokens);
      o["same_answer"] = cached.tokens == inherited.tokens;
      const bool pass = cached.cache_hit && cached.prefill_tokens == 0 &&
                        cached.prompt_tokens >= 128 &&
                        cached.cached_prompt_tokens == cached.prompt_tokens &&
                        cached.tokens == inherited.tokens &&
                        inherited.draft_tokens + cached.draft_tokens > 0;
      return std::pair{pass, std::move(o)};
    };
    for (const bool thinking : {false, true}) {
      auto [pass, o] =
          replay(std::string(kContext) + std::string(kQuestion), thinking);
      check(std::string("serving_cached_replay_thinking_") +
                (thinking ? "on" : "off"),
            pass, std::move(o));
    }
    report.Set("serving_cached_replay_short",
               replay(std::string(kQuestion), false).second);
  }

  // ---- sampled replay ------------------------------------------------------
  {
    sampling::SamplingConfig sampled;
    sampled.temperature = 0.8F;
    sampled.top_p = 0.95F;
    sampled.top_k = 40;
    sampled.seed = 1234;
    const auto& prompt = prompts[0];
    const auto run_sampled = [&](l3::Session& s, Run* out, bool feed) {
      return GenerateMtp(*model, s, sampled, prompt, G, 3, layers, out, &error,
                         feed);
    };
    auto s1 = fresh();
    auto s2 = fresh();
    Run a, b;
    if (!s1 || !s2 || !run_sampled(*s1, &a, true) ||
        !run_sampled(*s2, &b, true)) {
      return fail_with("sampled runs");
    }
    auto o = json::Value::object();
    o["cycles"] = U(a.stats.cycles);
    o["drafted"] = U(a.stats.drafted);
    o["accepted"] = U(a.stats.accepted);
    check("sampled_replay",
          a.tokens == b.tokens && a.tokens.size() == G && a.stats.drafted > 0 &&
              SameState(a, b),
          std::move(o));

    // From a restored snapshot (state, MTP rows and carry restored).
    auto s3 = fresh();
    auto s4 = fresh();
    Run c;
    std::unique_ptr<l3::SessionSnapshot> snap;
    std::vector<std::uint16_t> rows_before, rows_after;
    std::vector<float> h_before, h_after;
    bool ok = s3 && s4 && s3->Feed(prompt, {}, &error);
    ok = ok && s3->ReadMtpCache(s3->Position() - 1, &rows_before, &error) &&
         s3->ReadMtpHidden(&h_before, &error);
    snap = ok ? s3->SaveSnapshot(&error) : nullptr;
    ok = ok && snap && s4->RestoreSnapshot(*snap, &error) &&
         s4->ReadMtpCache(s4->Position() - 1, &rows_after, &error) &&
         s4->ReadMtpHidden(&h_after, &error) && run_sampled(*s4, &c, false);
    auto o2 = json::Value::object();
    o2["snapshot_bytes"] = U(snap ? snap->SizeBytes() : 0);
    o2["expected_bytes"] = U(l3::SnapshotPayloadBytes(
        static_cast<std::uint32_t>(prompt.size()), true));
    o2["mtp_active_after_restore"] = ok && s4->MtpActive();
    check("snapshot_mtp_state",
          ok && snap &&
              snap->SizeBytes() ==
                  l3::SnapshotPayloadBytes(
                      static_cast<std::uint32_t>(prompt.size()), true) &&
              snap->SizeBytes() == s3->SnapshotBytes() &&
              rows_before == rows_after && h_before == h_after,
          std::move(o2));
    auto o3 = json::Value::object();
    o3["cycles"] = U(c.stats.cycles);
    check("sampled_replay_from_snapshot",
          ok && c.tokens == a.tokens && c.stats.drafted > 0, std::move(o3));

    // A recycled session (cache miss: Reset() reuses the object) must not
    // carry draft-controller history into the next request.
    {
      sampling::SamplingConfig other = sampled;
      other.seed = 99;
      auto s5 = fresh();
      Run first, replay;
      bool recycled =
          s5 && GenerateMtp(*model, *s5, other, prompts[1], 4 * G, 3, layers,
                            &first, &error, true, {}, true, false);
      const auto before =
          recycled ? s5->Statistics() : l3::Session::DraftStats{};
      if (recycled) {
        s5->Reset();
      }
      recycled = recycled && s5->IsValid() &&
                 GenerateMtp(*model, *s5, sampled, prompt, G, 3, layers,
                             &replay, &error, true, {}, false);
      const auto delta_cycles = replay.stats.cycles - before.cycles;
      const auto delta_drafted = replay.stats.drafted - before.drafted;
      auto o4 = json::Value::object();
      o4["first_cycles"] = U(before.cycles);
      o4["replay_cycles"] = U(delta_cycles);
      o4["first_drafted"] = U(before.drafted);
      o4["replay_drafted"] = U(delta_drafted);
      o4["fresh_drafted"] = U(a.stats.drafted);
      check("sampled_replay_on_recycled_session",
            recycled && replay.tokens == a.tokens && SameState(a, replay) &&
                delta_cycles == a.stats.cycles &&
                delta_drafted == a.stats.drafted,
            std::move(o4));
    }

    // A throw between Verify() and Rollback() must not leave a valid-looking
    // session holding rejected rows.
    {
      auto s6 = fresh();
      bool threw = false;
      bool ok6 = s6 && s6->Feed(prompt, {}, &error);
      if (ok6) {
        s6->SetMaxDrafts(3);
        s6->SetDraftHooks({.fixed_drafts = 2, .after_verify = [] {
                             throw std::runtime_error("forced sampler failure");
                           }});
        sampling::SamplerState sampler(sampled);
        l3::Session::DecodeResult step;
        try {
          (void)s6->DecodeStep(4, sampler, &step, &error);
        } catch (const std::runtime_error&) {
          threw = true;
        }
      }
      const bool invalid = ok6 && !s6->IsValid();
      Run again;
      if (ok6) {
        s6->SetDraftHooks({});
        s6->Reset();
      }
      const bool recovered = ok6 && s6->IsValid() &&
                             run_sampled(*s6, &again, true) &&
                             again.tokens == a.tokens;
      auto o5 = json::Value::object();
      o5["threw"] = threw;
      o5["invalid_after_throw"] = invalid;
      check("throw_after_verify_invalidates_session",
            ok6 && threw && invalid && recovered, std::move(o5));
    }
  }

  report.Set("checks", std::move(checks));
  report.Set("failures", failures);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

}  // namespace gufo::tests::ling::mtp_tool

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(
      argc, argv, "mtp_test",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model =
            gufo::tests::ling::LoadModel(args, report, 256, 1024, true);
        return model ? gufo::tests::ling::mtp_tool::Execute(args, report, model)
                     : 1;
      });
}
