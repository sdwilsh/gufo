// Host-only check of sampled MTP decoding under the server's Bailing tool
// grammar (JsonConstraint::ToolFormat::kBailing, the lazy `auto` grammar with
// reasoning, and a required call). Whole sequences of a few hundred tokens are
// generated cycle by cycle as Session::DecodeStep and the server drive them
// (working sampler copy, MtpProposer with an unconstrained draft sampler,
// VerifySampledChain with the request's grammar, the deferred residual draw,
// varying or controller-chosen draft counts) on a synthetic model whose
// target follows a write_file call with a long raw-string `content` argument
// (quotes, backslashes, tabs, UTF-8, `<tool_call>`, `</tool_call>` and a
// partial `</arg` inside the code) with noise, over a small byte vocabulary
// whose pieces split and join the markup.
//
// Prompt-lookup copies (prompt_lookup.hpp) run as a further mode: the copied
// tokens are a deterministic function of the context (here the draft model's
// argmax, which the grammar often forbids), verified as point-mass proposals
// (CopyProposal) by the same VerifySampledChain with the request's grammar.
//
// Checks per grammar, sampling setting and mode:
//   * grammar consistency: every committed sequence is re-validated from
//     scratch by walking its bytes through the grammar (no token masks, no
//     sampler), every committed token is admitted at its position by a second,
//     independent vocabulary, a stop is committed only where the grammar is
//     complete, and after every cycle the request sampler's grammar state
//     equals the independently walked state (advanced exactly once per
//     committed token, in order, never by a rejected draft);
//   * distribution: teacher forcing through a plain constrained sampler gives
//     randomized PIT values that must be U(0, 1) (Kolmogorov-Smirnov), stop and
//     top-1 counts that match their exact expectations, and no committed token
//     with p = 0 (sampled_distribution.hpp); plain autoregressive sampling is
//     the control of the same statistics;
//   * arguments: each complete call is parsed by the server's Bailing parser
//     and its `content` must equal the raw text between the value tags;
//   * coverage: drafts the grammar forbids must be proposed (and rejected) and
//     calls must complete.
// A power check then runs plausible bugs of the constrained cycle (the grammar
// advanced over unverified drafts, stale or unmasked row targets, an unmasked
// residual, a dropped deferred draw, a double or a rejected-draft grammar
// advance) and requires every one of them to fail at least one check.
//
//   ling_3_flash_mtp_constrained_chain_test [sequences] [bug-sequences]
//       [label-filter] [seed-shift]
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/quote_tracker.hpp"
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/sampling.hpp"
#include "src/models/ling_3_flash/mtp.hpp"
#include "src/models/ling_3_flash/mtp_sampling.hpp"
#include "src/models/ling_3_flash/tool_calls.hpp"
#include "tests/models/ling_3_flash/sampled_distribution.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace sampling = gufo::sampling;
using gufo::tests::ling::Calibration;
using gufo::tests::ling::kStopOutcome;
using gufo::tests::ling::KsUniform;
using gufo::tests::ling::Pit;
using gufo::tests::ling::Sequence;
using Constraint = sampling::JsonConstraint;

namespace {

int g_failures = 0;
void Expect(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
  if (!ok) {
    ++g_failures;
  }
}

// ---- vocabulary -------------------------------------------------------------

/// Tokens 0 and 1 are stop tokens (no text); the rest are byte pieces: every
/// printable ASCII byte, tab and newline alone (any text is representable),
/// the Bailing markup whole and split, pieces that join a closer with
/// neighbouring text, code words and UTF-8 (whole and split characters).
std::vector<std::string> Pieces() {
  std::vector<std::string> out{"", ""};
  std::set<std::string> seen;
  const auto add = [&](std::string piece) {
    if (seen.insert(piece).second) {
      out.push_back(std::move(piece));
    }
  };
  for (int c = 0x20; c < 0x7f; ++c) {
    add(std::string(1, static_cast<char>(c)));
  }
  for (const char* piece : {"\n",
                            "\t",
                            "\n\n",
                            "    ",
                            "        ",
                            "<tool_call>",
                            "</tool_call>",
                            "<arg_key>",
                            "</arg_key>",
                            "<arg_value>",
                            "</arg_value>",
                            "</think>",
                            "<think>",
                            "</",
                            "arg",
                            "_key",
                            "_value",
                            "_value>",
                            "</arg",
                            "tool",
                            "_call",
                            "think",
                            "write_file",
                            "read_file",
                            "write",
                            "_file",
                            "path",
                            "content",
                            "x</arg_value>",
                            "</arg_value>\n",
                            "</arg_value>\n</tool_call>",
                            "\n</tool_call>",
                            "</tool_call>\n",
                            "\n<arg_key>",
                            "\n<tool_call>",
                            "def",
                            " parse",
                            "(text",
                            "):",
                            " return",
                            " =",
                            " ==",
                            " +",
                            "\"\"\"",
                            "\\n",
                            "\\\\",
                            " #",
                            " ;",
                            "[",
                            "]",
                            " in",
                            " for",
                            " if",
                            " not",
                            " or",
                            " line",
                            " text",
                            "line",
                            "text",
                            " out",
                            "out",
                            " key",
                            "key",
                            " value",
                            "value",
                            " section",
                            "section",
                            ".strip",
                            ".split",
                            ".startswith",
                            ".partition",
                            "ini",
                            ".py",
                            " comment",
                            "comments",
                            "Let",
                            " me",
                            " write",
                            " the",
                            " parser",
                            " First",
                            " check",
                            " rules",
                            "\xc3\xa9",
                            "\xc3",
                            "\xa9",
                            "\xe2\x86\x92",
                            "\xe2\x86",
                            "\x92",
                            "continue",
                            " None",
                            " {}",
                            "\n        ",
                            "\n    ",
                            "TAG",
                            "PATH"}) {
    add(piece);
  }
  return out;
}

bool IsStop(sampling::TokenId token) {
  return token == 0 || token == 1;
}

std::shared_ptr<const sampling::ConstraintVocabulary> MakeVocabulary(
    const std::vector<std::string>& pieces) {
  return std::make_shared<sampling::ConstraintVocabulary>(
      static_cast<std::uint32_t>(pieces.size()), [&](std::uint32_t id) {
        return sampling::ConstraintVocabulary::Piece{pieces[id], IsStop(id)};
      });
}

// ---- grammars ---------------------------------------------------------------

constexpr std::string_view kCode =
    "def parse(text):\n"
    "    \"\"\"Parse INI text; '#' and ';' start comments.\"\"\"\n"
    "    out = {}\n"
    "    section = None\n"
    "    for line in text.split(\"\\n\"):\n"
    "        line = line.strip()\n"
    "        if not line or line.startswith((\"#\", \";\")):\n"
    "            continue\n"
    "        if line.startswith(\"[\") and line.endswith(\"]\"):\n"
    "            section = out.setdefault(line[1:-1], {})\n"
    "            continue\n"
    "        key, _, value = line.partition(\"=\")\n"
    "        value = value.split(\" ;\")[0].split(\" #\")[0]\n"
    "\tsection[key.strip()] = value.strip()  # \xc3\xa9 \xe2\x86\x92 ok\n"
    "    return out\n"
    "\n"
    "TAG = \"<tool_call>read_file</tool_call> </arg\"\n"
    "PATH = \"C:\\\\temp\\\\x.ini\"\n";

struct GrammarCase {
  std::string name;
  std::shared_ptr<const Constraint> grammar;
  std::string template_text;
};

const char* kWriteFileSchema = R"({"type":"object","properties":{
  "path":{"type":"string"},"content":{"type":"string"}},
  "required":["path","content"]})";
const char* kReadFileSchema = R"({"type":"object","properties":{
  "path":{"type":"string"}},"required":["path"]})";

// As in llama.cpp, a required call uses the same argument grammar as an
// optional one; tool_choice only changes the call envelope.
std::vector<Constraint::Tool> Tools() {
  using F = Constraint::ToolFormat;
  return {{"write_file",
           Constraint::ToolParameters(gufo::json::parse(kWriteFileSchema),
                                      false, F::kBailing)},
          {"read_file",
           Constraint::ToolParameters(gufo::json::parse(kReadFileSchema), false,
                                      F::kBailing)}};
}

std::string CallText() {
  return "<tool_call>write_file\n<arg_key>path</arg_key>\n<arg_value>ini.py"
         "</arg_value>\n<arg_key>content</arg_key>\n<arg_value>" +
         std::string(kCode) + "</arg_value>\n</tool_call>";
}

/// The server's grammars (ConstrainChatRequest): `auto` tools with thinking on
/// (lazy calls, reasoning first, parallel calls) and a required call.
std::vector<GrammarCase> Grammars() {
  using F = Constraint::ToolFormat;
  return {
      {"auto_reasoning",
       Constraint::WithReasoning(
           Constraint::WithTools(nullptr, Tools(), false, true, F::kBailing)),
       "Let me write the parser. First check the comment rules.</think>\n" +
           CallText()},
      {"required",
       Constraint::WithTools(nullptr, Tools(), true, false, F::kBailing),
       CallText()},
  };
}

// ---- synthetic model --------------------------------------------------------

std::uint64_t Mix(std::uint64_t x) {
  x ^= x >> 33U;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33U;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33U;
  return x;
}

double Unit(std::uint64_t h) {
  return static_cast<double>(h >> 11U) * 0x1.0p-53;
}

/// The model's view of a prefix: the generated count, the last two tokens and
/// the position in the template text (-1 when lost; resynchronized on the last
/// bytes written).
struct ModelState {
  std::size_t generated{0};
  std::int64_t last[2]{-1, -1};
  std::int64_t position{0};
  std::string tail;
};

class Model {
public:
  Model(const std::vector<std::string>& pieces, std::string text)
      : pieces_(pieces), text_(std::move(text)) {}

  void Advance(ModelState* s, sampling::TokenId token) const {
    const auto& piece = pieces_[token];
    ++s->generated;
    s->last[0] = s->last[1];
    s->last[1] = token;
    s->tail += piece;
    if (s->tail.size() > 24) {
      s->tail.erase(0, s->tail.size() - 24);
    }
    if (s->position >= 0 &&
        text_.compare(static_cast<std::size_t>(s->position), piece.size(),
                      piece) == 0 &&
        static_cast<std::size_t>(s->position) + piece.size() <= text_.size()) {
      s->position += static_cast<std::int64_t>(piece.size());
      return;
    }
    s->position = -1;
    for (std::size_t k = std::min<std::size_t>(16, s->tail.size()); k >= 4;
         --k) {
      const auto found = text_.find(s->tail.substr(s->tail.size() - k));
      if (found != std::string::npos) {
        s->position = static_cast<std::int64_t>(found + k);
        return;
      }
    }
  }

  /// Target logits: noise plus a pull towards the pieces that continue the
  /// template; stop strongly favoured at the template's end, rare elsewhere.
  [[nodiscard]] std::vector<float> Target(const ModelState& s,
                                          std::uint64_t salt = 1) const {
    const std::uint64_t h =
        Mix(salt * 0x9E3779B97F4A7C15ULL ^
            Mix(static_cast<std::uint64_t>(s.last[0] + 3) * 1000003 +
                static_cast<std::uint64_t>(s.last[1] + 3)) ^
            (s.generated + 1) * 0x632BE59BD9B4E019ULL);
    std::vector<float> logits(pieces_.size());
    for (std::size_t i = 2; i < pieces_.size(); ++i) {
      double v = -0.4 * std::log(1.0 + static_cast<double>(i % 37)) +
                 2.4 * (Unit(Mix(h + i)) - 0.5);
      const auto& piece = pieces_[i];
      if (s.position >= 0 &&
          static_cast<std::size_t>(s.position) + piece.size() <= text_.size() &&
          text_.compare(static_cast<std::size_t>(s.position), piece.size(),
                        piece) == 0) {
        v += 2.6 +
             0.35 * static_cast<double>(std::min<std::size_t>(piece.size(), 8));
      }
      logits[i] = static_cast<float>(v);
    }
    const bool done = s.position == static_cast<std::int64_t>(text_.size());
    logits[0] = static_cast<float>(done ? 6.0 : -3.5 + 2.0 * Unit(h ^ 77));
    logits[1] = static_cast<float>(done ? 3.0 : -4.5 + 2.0 * Unit(h ^ 99));
    return logits;
  }

  /// Draft logits: the target with context-dependent noise (sometimes none),
  /// or (unrelated) another context's noise without the template pull.
  [[nodiscard]] std::vector<float> Draft(const ModelState& s,
                                         bool unrelated) const {
    if (unrelated) {
      ModelState lost = s;
      lost.position = -1;
      return Target(lost, 17);
    }
    auto logits = Target(s);
    const std::uint64_t h = Mix(s.generated * 7919 + 13 +
                                static_cast<std::uint64_t>(s.last[1] + 3));
    const double sigma = (h & 3U) == 0 ? 0.0 : 1.5;
    for (std::size_t i = 0; i < logits.size(); ++i) {
      logits[i] += static_cast<float>(sigma * (Unit(Mix(h + 977 * i)) - 0.5));
    }
    return logits;
  }

  [[nodiscard]] const std::string& piece(sampling::TokenId t) const {
    return pieces_[t];
  }

private:
  const std::vector<std::string>& pieces_;
  std::string text_;
};

// ---- the cycle --------------------------------------------------------------

enum class Mode {
  kAutoregressive,
  kVaryingDrafts,
  kController,
  kUnrelatedDrafts,
  kCopies
};

const char* ModeName(Mode m) {
  return m == Mode::kAutoregressive  ? "ar"
         : m == Mode::kVaryingDrafts ? "mtp_varying_drafts"
         : m == Mode::kController    ? "mtp_controller"
         : m == Mode::kCopies        ? "lookup_copies"
                                     : "mtp_unrelated_drafts";
}

/// Plausible bugs of the constrained cycle (the power check).
enum class Bug {
  kNone,
  /// The grammar follows the whole draft chain before verification, so row
  /// targets are masked with the state after unverified drafts.
  kAdvanceOverDrafts,
  /// Every row target uses the grammar state after the anchor only.
  kStaleRowState,
  /// Acceptance tests p without the grammar mask.
  kUnmaskedAcceptance,
  /// The residual correction is drawn without the grammar mask.
  kUnmaskedResidual,
  /// The deferred residual draw is dropped and the next anchor resampled.
  kDroppedCorrection,
  /// The request sampler advances the grammar twice over the anchor.
  kDoubleAnchorAdvance,
  /// The request sampler's grammar also follows the first rejected draft.
  kRejectedDraftAdvance,
};

const char* BugName(Bug b) {
  switch (b) {
    case Bug::kNone:
      return "none";
    case Bug::kAdvanceOverDrafts:
      return "advance_over_drafts";
    case Bug::kStaleRowState:
      return "stale_row_state";
    case Bug::kUnmaskedAcceptance:
      return "unmasked_acceptance";
    case Bug::kUnmaskedResidual:
      return "unmasked_residual";
    case Bug::kDroppedCorrection:
      return "dropped_correction";
    case Bug::kDoubleAnchorAdvance:
      return "double_anchor_advance";
    case Bug::kRejectedDraftAdvance:
      return "rejected_draft_advance";
  }
  return "?";
}

struct CycleStats {
  std::uint64_t cycles{0};
  std::uint64_t drafted{0};
  std::uint64_t accepted{0};
  /// Proposals the grammar forbids at their position (first unverified one
  /// included), i.e. rejections that only the mask can decide.
  std::uint64_t forbidden_drafts{0};
  /// Cycles after which the request sampler's grammar state differed from the
  /// independently walked one.
  std::uint64_t state_mismatches{0};
  /// Sequences aborted by an exception (a grammar violation in production
  /// code: an invalid accept or a pending draw the mask forbids).
  std::uint64_t exceptions{0};
};

/// VerifySampledChain with an injected bug (Bug::kNone uses production code).
l3::SampledVerification VerifyWithBug(
    Bug bug, const std::vector<std::vector<float>>& rows,
    std::span<const l3::MtpProposal> proposals, sampling::SamplerState& sampler,
    std::span<const std::int32_t> chain) {
  const auto row = [&](std::uint32_t r) {
    return std::span<const float>(rows[r]);
  };
  if (bug == Bug::kNone || bug == Bug::kDroppedCorrection ||
      bug == Bug::kDoubleAnchorAdvance || bug == Bug::kRejectedDraftAdvance) {
    return l3::VerifySampledChain(row, proposals, sampler, IsStop);
  }
  // The production loop with one deviation.
  auto shadow = sampler;  // kStaleRowState: the state after the anchor
  if (bug == Bug::kAdvanceOverDrafts) {
    for (std::size_t j = 1; j < chain.size(); ++j) {
      try {
        shadow.Accept(static_cast<sampling::TokenId>(chain[j]));
      } catch (const std::exception&) {
        break;  // the grammar cannot follow; keep the state reached
      }
    }
  }
  l3::SampledVerification out;
  while (out.keep <= proposals.size()) {
    const auto& proposal = proposals[out.keep - 1];
    const auto logits = row(out.keep - 1);
    const sampling::SamplerState& p_state =
        bug == Bug::kStaleRowState || bug == Bug::kAdvanceOverDrafts ? shadow
                                                                     : sampler;
    const auto masked = p_state.Distribution(logits);
    const auto unmasked = p_state.WithoutConstraint().Distribution(logits);
    const auto& accept_p = bug == Bug::kUnmaskedAcceptance ? unmasked : masked;
    const auto& residual_p = bug == Bug::kUnmaskedResidual ? unmasked : masked;
    sampling::TokenId token = proposal.token;
    bool accepted = true;
    if (!(sampler.Uniform() * proposal.probability <
          accept_p.probability(proposal.token))) {
      accepted = false;
      token = residual_p.SampleResidual(
          std::span(proposal.ids).first(proposal.size),
          std::span(proposal.probabilities).first(proposal.size),
          sampler.mutable_rng_state());
    }
    if (IsStop(token)) {
      out.stop = true;
      break;
    }
    if (!accepted) {
      out.correction = token;
      break;
    }
    sampler.Accept(token);
    ++out.keep;
  }
  return out;
}

/// Independent grammar walk: bytes through JsonConstraint::Advance, token
/// admission through a second vocabulary.
struct Walk {
  std::shared_ptr<const Constraint> grammar;
  std::shared_ptr<const sampling::ConstraintVocabulary> vocabulary;
  Constraint::State state;
  bool valid{true};

  void Step(sampling::TokenId token, const std::string& piece) {
    if (!valid) {
      return;
    }
    if (!vocabulary->Allows(*grammar, state, token)) {
      valid = false;
      return;
    }
    if (IsStop(token)) {
      valid = grammar->Complete(state);
      return;
    }
    for (const unsigned char byte : piece) {
      state = grammar->Advance(state, byte);
      if (state.empty()) {
        valid = false;
        return;
      }
    }
  }
};

struct Generated {
  Sequence sequence;
  bool valid{true};
  bool aborted{false};
};

/// One request as the server runs it: the pool's sampler accepts committed
/// tokens; each step works on a copy whose draw state is published afterwards.
Generated Generate(const Model& model, const sampling::SamplingConfig& config,
                   const GrammarCase& grammar,
                   const std::shared_ptr<const sampling::ConstraintVocabulary>&
                       independent_vocabulary,
                   std::size_t length, Mode mode, Bug bug, CycleStats* stats) {
  sampling::SamplerState pool(config);
  ModelState context;
  l3::MtpLengthController controller;
  Walk walk{grammar.grammar, independent_vocabulary, grammar.grammar->Start()};
  Generated out;
  std::uint64_t cycle = 0;
  try {
    while (out.sequence.tokens.size() < length) {
      const std::size_t budget = std::min<std::size_t>(
          l3::kMaxVerifyTokens, length - out.sequence.tokens.size());
      sampling::SamplerState working = pool;
      const auto anchor = working.Sample(model.Target(context));
      if (IsStop(anchor)) {
        walk.Step(anchor, {});
        out.sequence.stopped = true;
        break;
      }
      std::uint32_t drafts = 0;
      if (mode == Mode::kVaryingDrafts || mode == Mode::kUnrelatedDrafts ||
          mode == Mode::kCopies) {
        drafts = static_cast<std::uint32_t>(
            Mix(static_cast<std::uint64_t>(config.seed) * 131 + cycle) % 8);
      } else if (mode == Mode::kController) {
        drafts = controller.Choose(7, false, 0);
      }
      drafts = std::min<std::uint32_t>(drafts,
                                       static_cast<std::uint32_t>(budget - 1));
      ++cycle;
      std::vector<std::int32_t> committed;
      bool stop = false;
      std::optional<sampling::TokenId> rejected;
      if (drafts == 0) {
        working.Accept(anchor);
        committed = {static_cast<std::int32_t>(anchor)};
        controller.ObservePlainToken();
      } else {
        // Copies draw nothing from the request's stream; MTP proposals take
        // their own stream from it (MtpProposer).
        std::optional<l3::MtpProposer> proposer;
        std::vector<l3::MtpProposal> copies;
        if (mode != Mode::kCopies) {
          proposer.emplace(working, anchor);
        }
        std::vector<std::int32_t> chain{static_cast<std::int32_t>(anchor)};
        std::vector<ModelState> after{context};
        model.Advance(&after.back(), anchor);
        for (std::uint32_t j = 0; j < drafts; ++j) {
          sampling::TokenId token;
          if (mode == Mode::kCopies) {
            const auto draft = model.Draft(after.back(), false);
            token = static_cast<sampling::TokenId>(
                std::max_element(draft.begin(), draft.end()) - draft.begin());
            copies.push_back(l3::CopyProposal(token));
          } else {
            token = proposer->Propose(
                model.Draft(after.back(), mode == Mode::kUnrelatedDrafts));
          }
          chain.push_back(static_cast<std::int32_t>(token));
          after.push_back(after.back());
          model.Advance(&after.back(), token);
        }
        std::vector<std::vector<float>> rows;
        for (std::size_t j = 0; j < chain.size(); ++j) {
          rows.push_back(model.Target(after[j]));
        }
        working.Accept(anchor);
        const auto verified = VerifyWithBug(
            bug, rows,
            mode == Mode::kCopies ? std::span(copies) : proposer->proposals(),
            working, chain);
        committed.assign(chain.begin(), chain.begin() + verified.keep);
        stop = verified.stop;
        if (verified.keep < chain.size()) {
          rejected = static_cast<sampling::TokenId>(chain[verified.keep]);
        }
        // Coverage: the first unverified proposal is forbidden by the grammar.
        if (verified.keep < chain.size()) {
          auto probe = walk;
          for (const auto token : committed) {
            probe.Step(static_cast<sampling::TokenId>(token),
                       model.piece(static_cast<sampling::TokenId>(token)));
          }
          probe.Step(*rejected, model.piece(*rejected));
          stats->forbidden_drafts += probe.valid ? 0 : 1;
        }
        if (verified.correction && bug != Bug::kDroppedCorrection) {
          working.DeferSample(*verified.correction);
        }
        controller.Observe(verified.keep - 1, stop ? verified.keep - 1 : drafts,
                           false, 0);
        ++stats->cycles;
        stats->drafted += drafts;
        stats->accepted += verified.keep - 1;
      }
      pool.CopyDrawStateFrom(working);
      for (const auto token : committed) {
        pool.Accept(static_cast<sampling::TokenId>(token));
        model.Advance(&context, static_cast<sampling::TokenId>(token));
        out.sequence.tokens.push_back(token);
        walk.Step(static_cast<sampling::TokenId>(token),
                  model.piece(static_cast<sampling::TokenId>(token)));
      }
      if (bug == Bug::kDoubleAnchorAdvance) {
        pool.Accept(static_cast<sampling::TokenId>(committed.front()));
      }
      if (bug == Bug::kRejectedDraftAdvance && rejected && !stop &&
          !IsStop(*rejected)) {
        pool.Accept(*rejected);
      }
      if (walk.valid && pool.constraint_state() != walk.state) {
        ++stats->state_mismatches;
      }
      if (stop) {
        walk.Step(0, {});
        out.sequence.stopped = true;
        break;
      }
    }
  } catch (const std::exception&) {
    ++stats->exceptions;
    out.aborted = true;
  }
  out.valid = walk.valid && !out.aborted;
  return out;
}

struct Evidence {
  std::vector<double> pit;
  std::size_t impossible{0};
  Calibration stops;
  Calibration top;
};

/// Teacher-forces `s` through a plain constrained sampler.
void Score(const Model& model, const sampling::SamplingConfig& config,
           const Sequence& s, std::mt19937_64& rng, Evidence* e) {
  sampling::SamplerState plain(config);
  ModelState context;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (std::size_t i = 0; i <= s.tokens.size(); ++i) {
    if (i == s.tokens.size() && !s.stopped) {
      break;
    }
    const auto p = plain.Distribution(model.Target(context));
    const std::int64_t outcome =
        i < s.tokens.size() ? s.tokens[i] : kStopOutcome;
    const auto v = Pit(p, outcome, IsStop, unit(rng));
    e->pit.push_back(v.u);
    e->impossible += v.possible ? 0 : 1;
    e->stops.Add(outcome == kStopOutcome, v.p_stop);
    e->top.Add(v.top, v.p_top);
    if (!v.possible) {
      return;  // the rest of the sequence has no target distribution
    }
    if (i < s.tokens.size()) {
      const auto token = static_cast<sampling::TokenId>(s.tokens[i]);
      plain.Accept(token);
      model.Advance(&context, token);
    }
  }
}

/// Complete calls whose parsed `content` differs from the raw value text.
struct ArgumentCheck {
  std::size_t calls{0};
  std::size_t content_calls{0};
  std::size_t mismatches{0};
};

void CheckArguments(const Model& model, const Sequence& s, ArgumentCheck* out) {
  std::string text;
  for (const auto token : s.tokens) {
    text += model.piece(static_cast<sampling::TokenId>(token));
  }
  // The answer starts after the reasoning (the server parses content only):
  // at `</think>`, or at the first opener outside quoted reasoning text
  // (including an unfinished inline or fenced quote), as the grammar's
  // reasoning lexeme and the server's ParseGeneration read it.
  gufo::server::QuoteTracker quotes;
  quotes.Reset(text);
  auto opener = text.find("<tool_call>");
  while (opener != std::string::npos &&
         (quotes.QuotedAt(opener) || quotes.UnclosedAt(opener)))
    opener = text.find("<tool_call>", opener + 1);
  const auto think = text.find("</think>");
  if (think != std::string::npos &&
      (opener == std::string::npos || think < opener)) {
    text.erase(0, think + 8);
  } else if (opener != std::string::npos) {
    text.erase(0, opener);
  }
  std::vector<gufo::models::ling_3_flash::BailingTool> tools(2);
  tools[0].name = "write_file";
  tools[0].parameters_json = kWriteFileSchema;
  tools[1].name = "read_file";
  tools[1].parameters_json = kReadFileSchema;
  const auto scan = gufo::models::ling_3_flash::ScanBailingToolCalls(
      text, tools, false, false);
  for (const auto& call : scan.calls) {
    ++out->calls;
    for (const auto& argument : call.arguments) {
      if (argument.name != "content") {
        continue;
      }
      ++out->content_calls;
      // The raw value text by a sequential scan of the call's pairs: a raw
      // value ends at the first value closer (it cannot contain one), so a
      // key-like text inside an earlier value is skipped.
      std::optional<std::string> raw;
      for (std::size_t at = call.begin; at < call.end;) {
        const auto key = text.find("<arg_key>", at);
        const auto key_end = text.find("</arg_key>", key);
        const auto value = text.find("<arg_value>", key_end);
        const auto close = text.find("</arg_value>", value);
        if (key == std::string::npos || key_end == std::string::npos ||
            value == std::string::npos || close == std::string::npos ||
            close >= call.end) {
          break;
        }
        if (text.substr(key + 9, key_end - key - 9) == "content") {
          raw = text.substr(value + 11, close - value - 11);
        }
        at = close + 12;
      }
      const bool same = raw && argument.is_string && argument.value == *raw;
      out->mismatches += same ? 0 : 1;
      if (!same && out->mismatches <= 3) {
        std::printf("       argument mismatch: text=[%s] parsed=[%s]\n",
                    text.c_str(), argument.value.c_str());
      }
    }
  }
}

struct Outcome {
  std::size_t sequences{0};
  std::size_t invalid{0};
  std::size_t stopped{0};
  std::size_t tokens{0};
  Evidence evidence;
  CycleStats stats;
  ArgumentCheck arguments;
  gufo::tests::ling::KsResult ks;
};

Outcome Run(const Model& model, const sampling::SamplingConfig& base,
            const GrammarCase& grammar,
            const std::shared_ptr<const sampling::TokenConstraint>& binding,
            const std::shared_ptr<const sampling::ConstraintVocabulary>&
                independent_vocabulary,
            std::size_t sequences, std::size_t length, Mode mode, Bug bug,
            std::uint64_t seed_offset) {
  Outcome o;
  std::mt19937_64 rng(12345 + seed_offset);
  for (std::size_t n = 0; n < sequences; ++n) {
    auto config = base;
    config.constraint = binding;
    // Small consecutive seeds, as the agent harness sends.
    config.seed = static_cast<std::int64_t>(1000 + seed_offset + n);
    const auto g = Generate(model, config, grammar, independent_vocabulary,
                            length, mode, bug, &o.stats);
    ++o.sequences;
    o.invalid += g.valid ? 0 : 1;
    o.stopped += g.sequence.stopped ? 1 : 0;
    o.tokens += g.sequence.tokens.size();
    if (!g.aborted) {
      Score(model, config, g.sequence, rng, &o.evidence);
      CheckArguments(model, g.sequence, &o.arguments);
    }
  }
  o.ks = KsUniform(o.evidence.pit);
  return o;
}

constexpr double kAlpha = 1e-4;
constexpr double kMaxZ = 4.5;

/// The failed checks of an outcome (empty: all pass).
std::vector<std::string> Failures(const Outcome& o) {
  std::vector<std::string> failed;
  if (o.invalid != 0) {
    failed.push_back("grammar_revalidation");
  }
  if (o.stats.state_mismatches != 0) {
    failed.push_back("sampler_grammar_state");
  }
  if (o.stats.exceptions != 0) {
    failed.push_back("grammar_exception");
  }
  if (o.evidence.impossible != 0) {
    failed.push_back("impossible_token");
  }
  if (o.ks.p < kAlpha) {
    failed.push_back("pit_uniform");
  }
  if (std::fabs(o.evidence.stops.z()) > kMaxZ) {
    failed.push_back("stop_calibration");
  }
  if (std::fabs(o.evidence.top.z()) > kMaxZ) {
    failed.push_back("top1_calibration");
  }
  if (o.arguments.mismatches != 0) {
    failed.push_back("argument_roundtrip");
  }
  return failed;
}

std::string Join(const std::vector<std::string>& items) {
  std::string out;
  for (const auto& item : items) {
    out += (out.empty() ? "" : ",") + item;
  }
  return out.empty() ? "-" : out;
}

void Print(const std::string& label, const Outcome& o) {
  std::printf(
      "       %s: %zu seqs, %zu tokens (mean %.1f), stops %zu, invalid %zu, "
      "PIT n=%zu D=%.4f p=%.3g, stop z=%.2f, top-1 z=%.2f, impossible %zu, "
      "calls %zu (content %zu, mismatches %zu), drafted %llu accepted %llu "
      "forbidden %llu, state mismatches %llu, exceptions %llu\n",
      label.c_str(), o.sequences, o.tokens,
      static_cast<double>(o.tokens) /
          static_cast<double>(std::max<std::size_t>(1, o.sequences)),
      o.stopped, o.invalid, o.ks.n, o.ks.d, o.ks.p, o.evidence.stops.z(),
      o.evidence.top.z(), o.evidence.impossible, o.arguments.calls,
      o.arguments.content_calls, o.arguments.mismatches,
      static_cast<unsigned long long>(o.stats.drafted),
      static_cast<unsigned long long>(o.stats.accepted),
      static_cast<unsigned long long>(o.stats.forbidden_drafts),
      static_cast<unsigned long long>(o.stats.state_mismatches),
      static_cast<unsigned long long>(o.stats.exceptions));
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t sequences = argc > 1 ? std::stoul(argv[1]) : 300;
  const std::size_t bug_sequences = argc > 2 ? std::stoul(argv[2]) : 200;
  // Optional: only cases whose label contains argv[3]; seeds shifted by
  // argv[4] (a rerun with other draws).
  const std::string only = argc > 3 ? argv[3] : "";
  const std::uint64_t seed_shift = argc > 4 ? std::stoull(argv[4]) : 0;
  constexpr std::size_t kLength = 640;
  const auto pieces = Pieces();
  const auto vocabulary = MakeVocabulary(pieces);
  const auto independent = MakeVocabulary(pieces);
  struct Setting {
    const char* name;
    sampling::SamplingConfig config;
  };
  std::vector<Setting> settings;
  {
    sampling::SamplingConfig c;  // Ling preset
    c.temperature = 0.6F;
    c.top_p = 0.95F;
    c.top_k = 20;
    settings.push_back({"ling_preset", c});
    c.presence_penalty = 1.5F;
    c.frequency_penalty = 0.5F;
    c.repeat_penalty = 1.1F;
    c.repeat_last_n = 64;
    settings.push_back({"ling_preset_penalties", c});
    sampling::SamplingConfig wide;  // support wider than 64 candidates
    wide.temperature = 1.0F;
    wide.top_p = 0.98F;
    settings.push_back({"wide", wide});
  }
  const auto grammars = Grammars();
  for (const auto& grammar : grammars) {
    auto binding = std::make_shared<sampling::TokenConstraint>();
    binding->grammar = grammar.grammar;
    binding->vocabulary = vocabulary;
    const Model model(pieces, grammar.template_text);
    for (const auto& setting : settings) {
      for (const Mode mode :
           {Mode::kAutoregressive, Mode::kVaryingDrafts, Mode::kController,
            Mode::kUnrelatedDrafts, Mode::kCopies}) {
        const std::string label =
            grammar.name + "/" + setting.name + "/" + ModeName(mode);
        if (label.find(only) == std::string::npos) {
          continue;
        }
        const auto o =
            Run(model, setting.config, grammar, binding, independent, sequences,
                kLength, mode, Bug::kNone,
                100000 * static_cast<std::uint64_t>(mode) + seed_shift);
        Print(label, o);
        const auto failed = Failures(o);
        Expect(failed.empty(),
               label + ": all checks pass (failed: " + Join(failed) + ")");
        Expect(o.arguments.content_calls * 4 >= o.sequences,
               label + ": a quarter of the sequences complete the call");
        if (mode != Mode::kAutoregressive) {
          Expect(o.stats.accepted > 0 && o.stats.accepted < o.stats.drafted,
                 label + ": drafts are accepted and rejected");
          Expect(o.stats.forbidden_drafts > 0,
                 label + ": the grammar rejects drafts");
        }
      }
    }
  }

  // Power: every injected bug must fail at least one check.
  if (only.empty() || std::string("bug").find(only) != std::string::npos) {
    const auto& grammar = grammars.front();
    auto binding = std::make_shared<sampling::TokenConstraint>();
    binding->grammar = grammar.grammar;
    binding->vocabulary = vocabulary;
    const Model model(pieces, grammar.template_text);
    for (const Bug bug : {Bug::kAdvanceOverDrafts, Bug::kStaleRowState,
                          Bug::kUnmaskedAcceptance, Bug::kUnmaskedResidual,
                          Bug::kDroppedCorrection, Bug::kDoubleAnchorAdvance,
                          Bug::kRejectedDraftAdvance}) {
      const auto o =
          Run(model, settings.front().config, grammar, binding, independent,
              bug_sequences, kLength, Mode::kVaryingDrafts, bug, 900000);
      const std::string label = std::string("bug ") + BugName(bug);
      Print(label, o);
      const auto failed = Failures(o);
      Expect(!failed.empty(), label + ": detected by " + Join(failed));
    }
  }
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
