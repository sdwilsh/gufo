// Sampled MTP speculative decoding against plain autoregressive sampling on
// the GPU (Ling-3.0-flash): the output distribution must not change, for
// short and for long (hundreds to thousands of tokens) continuations, with and
// without the server's Bailing tool grammar. For fixed prompts and sampling
// settings it draws --samples continuations of --tokens tokens, with distinct
// seeds, in up to four modes from the same prompt snapshot:
//   * ar: plain sampling (SamplerState::Sample + Evaluate);
//   * mtp: Session::DecodeStep as the server drives it (working sampler copy,
//     draw state published, committed tokens accepted), adaptive drafts;
//   * mtp_fixed: the same with --fixed-drafts drafts every cycle, so the
//     multi-row verify, rejection and residual paths run even where the
//     controller would fall back (a truncated --layers trunk);
//   * mtp_fixed_restore: mtp_fixed with the seeds of mtp_fixed, but every
//     --restore-every committed tokens the context is snapshotted and restored
//     into the other of two sessions (with the draft policy reset), as the
//     server's prompt cache does between agent turns. Snapshots carry the MTP
//     state, so its sequences must equal mtp_fixed's token for token.
// Prompts (--prompt-ids, default the first --prompts): 0 prose, 1 a short
// chat answer that usually stops within a few tokens, 2 code, 3 a list, 4 an
// agent turn with the coding tools of the compare harness (thinking off, so it
// starts the write_file call at once and writes a long raw `content`
// argument), 5 the same with thinking on (long reasoning first).
// 6 and 7 are binding turns (thinking off / on): only a strict write_file
// (enum `mode`, integer `mode_bits`, raw string `content`) and a request for a
// short file. 8 and 9 are forcing turns (thinking off / on): only a strict
// `record` (const string `value`, enum `label`) that the prompt does not
// reveal ("call record with the required values").
// Settings (--configs): preset (the server's Ling preset), penalties (preset
// with presence/frequency/repeat penalties), wide, grammar (preset plus the
// server's lazy Bailing tool grammar with reasoning when thinking is on; tool
// prompts only), grammar_required (the grammar of tool_choice=required: the
// call is forced from the first token), grammar_bind (prompts 6 and 7: the
// strict write_file is the only tool and required, so the call opens at once
// and the grammar must remove mass at the key, enum and integer positions
// before the long raw `content`), grammar_force (prompts 8 and 9: the strict
// record is required; the model cannot infer the values, so the mask must
// remove mass of the filtered distribution inside the call and the parsed
// arguments must be the const and an enum member). Each grammar row reports the
// sequences that opened a call, those with a complete call, the tokens
// generated inside a call, and the steps where the mask removed > 1e-6 of the
// unconstrained sampler's mass (and the total removed mass), plus the same
// against the unfiltered softmax (temperature only) and the steps where the
// mask excluded a token of non-zero unfiltered probability; --require-binding
// fails a grammar row when its counters are zero (grammar_bind is gated on the
// unfiltered counter, the others on the filtered one); --require-calls also
// fails a grammar_bind row without complete calls whose `mode` and `mode_bits`
// parsed and a grammar_force row without complete calls of the right values.
// Checks per prompt, setting and mode (sampled_distribution.hpp): each MTP
// sequence is teacher-forced through the trunk and its randomized PIT values
// must be U(0, 1) (pooled and per position), the stop and top-1 counts must
// match their exact expectations, no committed token may have p = 0, and the
// MTP session's final logits must equal the teacher-forced ones bit for bit
// (after up to thousands of sampled tokens: KDA rollback, MTP rows and hidden
// carry over many cycles); the ar mode is the control of the same statistics.
// Two-sample permutation tests compare each MTP mode with ar: per-position
// token distributions over the first --position-limit positions (summed total
// variation), stop rate, length distribution, pooled token frequencies, the
// per-sequence mean log-probability of the committed tokens and the repeated
// 4-gram rate. On tool prompts every complete call is parsed by the server's
// Bailing parser and each string argument must equal the raw text between its
// value tags; under the grammar every sequence must re-validate from scratch
// (bytes walked through the grammar, a stop only where it is complete).
// Thresholds: KS p >= 1e-4, |z| <= 4.5, permutation p >= 2e-4.
// --inject deliberately breaks the MTP path to show the checks' power:
// drop-correction drops the deferred residual draw (biased where drafts are
// often accepted), skip-accept makes the request sampler skip the anchor of
// every step (penalty history and grammar state fall out of step with the
// committed text), teacher-prefill teacher-forces the MTP sequences in one
// prefill-family chunk (a trunk difference of about 1e-3 per layer, as a
// rollback or cache-row error would give).
//
//   mtp_distribution --model FIRST_SHARD.gguf [--layers N] [--samples 400]
//       [--tokens 12] [--fixed-drafts 3] [--configs preset,penalties]
//       [--modes ar,mtp,mtp_fixed] [--prompts 2 | --prompt-ids 2,4]
//       [--restore-every 48] [--position-limit 32] [--permutations 20000]
//       [--seed-base 1] [--inject none|drop-correction|skip-accept|
//       teacher-prefill] [--max-context 1024]
//       [--timeout-seconds S] [--require-binding]
//       [--require-calls]
// The JSON report is written once, on stdout, as one line that starts with
// `[report-json] `; progress goes to stderr. Exit 77 without --model; 0 pass,
// 1 a failed check.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/bailing_tool_calls.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/sampling.hpp"
#include "src/models/ling_3_flash/chat_template.hpp"
#include "tests/models/ling_3_flash/model_tool.hpp"
#include "tests/models/ling_3_flash/sampled_distribution.hpp"

namespace gufo::tests::ling::mtp_distribution {

namespace sampling = gufo::sampling;
using Constraint = sampling::JsonConstraint;

/// The compare harness's coding tools (tools/models/ling_3_flash/compare/
/// agent.py): name, description, parameter schema.
struct ToolSpec {
  const char* name;
  const char* description;
  const char* parameters;
};
constexpr ToolSpec kTools[] = {
    {"list_dir",
     "List a directory (one entry per line; directories end with '/').",
     R"({"type":"object","properties":{"path":{"type":"string","description":"Directory relative to the project root; '.' for the root"}},"required":["path"]})"},
    {"read_file", "Read a UTF-8 text file with 1-based line numbers.",
     R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})"},
    {"write_file",
     "Create or completely replace a file with the given contents.",
     R"({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]})"},
    {"run_tests",
     "Run the project's unittest suite (python3 -m unittest). Optionally pass "
     "a test module such as tests.test_foo.",
     R"({"type":"object","properties":{"target":{"type":"string","description":"Optional dotted test module, e.g. tests.test_foo"}},"required":[]})"},
};

/// The binding turn's only tool: write_file with a strict schema whose enum
/// `mode` and integer `mode_bits` make the grammar remove probability mass at
/// the key, enum and integer positions before the raw string `content`.
constexpr ToolSpec kBindTool = {
    "write_file",
    "Create or completely replace a file with the given contents.",
    R"({"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string","enum":["create","overwrite"]},"mode_bits":{"type":"integer"},"content":{"type":"string"}},"required":["path","mode","mode_bits","content"],"additionalProperties":false})"};

/// The forcing turn's only tool: a strict schema whose values the prompt does
/// not reveal (a const string and an enum), so a model that would return {}
/// without the grammar can only comply through the mask.
constexpr const char* kForceLiteral = "<think>literal</think></tool_call>";
constexpr ToolSpec kForceTool = {
    "record", "Record the required values.",
    R"({"type":"object","properties":{"value":{"type":"string","const":"<think>literal</think></tool_call>"},"label":{"type":"string","enum":["zeta-7","omega-3"]}},"required":["value","label"],"additionalProperties":false})"};

/// The single strict tool of a binding (`force` false) or forcing turn.
const ToolSpec* StrictTool(bool force) {
  return force ? &kForceTool : &kBindTool;
}

struct Prompt {
  std::string text;
  bool tools{false};
  bool thinking{false};
  /// The turn declares only kBindTool (strict) and may be forced to call it.
  bool bind{false};
  /// The turn declares only kForceTool (strict) and is forced to call it.
  bool force{false};
};

/// Prompts: prose, a short chat answer (thinking off, so most continuations
/// stop within a few tokens: stop handling, drafted stops), code, a list, and
/// an agent coding turn without and with thinking.
std::vector<Prompt> Prompts() {
  l3::ChatMessage user;
  user.role = "user";
  user.content =
      "Answer with one word only. What colour is the sky on a clear day?";
  l3::ChatTemplateOptions options;
  options.enable_thinking = false;
  std::vector<Prompt> out{
      {"The old lighthouse keeper climbed the spiral stairs every evening at "
       "dusk. On the night the storm arrived, he noticed something strange on "
       "the horizon:"},
      {l3::RenderChat(std::span(&user, 1), {}, options)},
      {"def parse_duration(text):\n    \"\"\"Parse strings such as '1h30m' or "
       "'45s' into a number of seconds.\"\"\"\n"},
      {"Here is a short list of practical tips for writing clear technical "
       "documentation:\n\n1."},
  };
  std::vector<l3::ChatTool> tools;
  for (const auto& tool : kTools) {
    auto function = json::Value::object();
    function["name"] = tool.name;
    function["description"] = tool.description;
    function["parameters"] = json::parse(tool.parameters);
    auto definition = json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function);
    tools.push_back({definition.dump()});
  }
  std::vector<l3::ChatMessage> messages(2);
  messages[0].role = "system";
  messages[0].content =
      "You are a coding agent working in a small Python project. Use the "
      "provided tools to inspect the code, make the change, and verify it with "
      "run_tests. Paths are relative to the project root. When the work is "
      "finished and verified, reply with a short summary and make no further "
      "tool calls.";
  messages[1].role = "user";
  messages[1].content =
      "Write ini.py from scratch with one function parse_ini(text) that parses "
      "INI text into a dict of sections, each a dict of key/value strings. "
      "Lines starting with '#' or ';' are comments; an inline ' ;' or ' #' "
      "starts a comment unless it is inside a double-quoted value, whose "
      "quotes are removed; keys and values are stripped; keys before the "
      "first section go to the section ''. Include docstrings and handle "
      "malformed lines by raising ValueError with the line number. Write the "
      "whole file with write_file now.";
  for (const bool thinking : {false, true}) {
    l3::ChatTemplateOptions o;
    o.enable_thinking = thinking;
    out.push_back({l3::RenderChat(messages, tools, o), true, thinking});
  }
  // Binding turns: a short file through the strict write_file above.
  std::vector<l3::ChatTool> bind_tools;
  {
    auto function = json::Value::object();
    function["name"] = kBindTool.name;
    function["description"] = kBindTool.description;
    function["parameters"] = json::parse(kBindTool.parameters);
    function["strict"] = true;
    auto definition = json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function);
    bind_tools.push_back({definition.dump()});
  }
  std::vector<l3::ChatMessage> bind_messages(2);
  bind_messages[0].role = "system";
  bind_messages[0].content =
      "You are a coding agent. Create files with the write_file tool: path is "
      "relative to the project root, mode is \"create\" for a new file or "
      "\"overwrite\" for an existing one, mode_bits is the Unix permission "
      "bits as a decimal integer (for example 420 for rw-r--r--).";
  bind_messages[1].role = "user";
  bind_messages[1].content =
      "Create the new file hello.py (permission bits 420) containing a short "
      "Python script of about ten lines that prints a greeting, with a "
      "docstring. Think for at most two sentences, then call write_file now.";
  for (const bool thinking : {false, true}) {
    l3::ChatTemplateOptions o;
    o.enable_thinking = thinking;
    out.push_back(
        {l3::RenderChat(bind_messages, bind_tools, o), true, thinking, true});
  }
  // Forcing turns: nothing in the prompt reveals the values.
  std::vector<l3::ChatTool> force_tools;
  {
    auto function = json::Value::object();
    function["name"] = kForceTool.name;
    function["description"] = kForceTool.description;
    function["parameters"] = json::parse(kForceTool.parameters);
    function["strict"] = true;
    auto definition = json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function);
    force_tools.push_back({definition.dump()});
  }
  std::vector<l3::ChatMessage> force_messages(1);
  force_messages[0].role = "user";
  force_messages[0].content =
      "Call record with the required values. No explanation.";
  for (const bool thinking : {false, true}) {
    l3::ChatTemplateOptions o;
    o.enable_thinking = thinking;
    out.push_back({l3::RenderChat(force_messages, force_tools, o), true,
                   thinking, false, true});
  }
  return out;
}

/// The server's grammar for these tools (ConstrainChatRequest: non-strict
/// schemas; `auto` choice with parallel calls, or a required single call;
/// reasoning first when thinking). `strict`: only that tool, strict, required.
std::shared_ptr<const sampling::TokenConstraint> BindToolGrammar(
    const l3::Model& model, bool thinking, bool required,
    const ToolSpec* strict) {
  std::vector<Constraint::Tool> tools;
  for (const auto& tool : strict ? std::span(strict, 1) : std::span(kTools)) {
    auto native = Constraint::ToolParameters(json::parse(tool.parameters),
                                             strict != nullptr,
                                             Constraint::ToolFormat::kBailing);
    if (!native) {
      throw std::runtime_error("the binding schema has no native form");
    }
    tools.emplace_back(tool.name, std::move(native));
  }
  auto grammar =
      Constraint::WithTools(nullptr, std::move(tools), required, !required,
                            Constraint::ToolFormat::kBailing);
  if (thinking) {
    grammar = Constraint::WithReasoning(grammar);
  }
  static std::shared_ptr<const sampling::ConstraintVocabulary> vocabulary;
  if (!vocabulary) {
    vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
        model.VocabSize(), [&](std::uint32_t id) {
          const auto token = static_cast<std::int32_t>(id);
          return sampling::ConstraintVocabulary::Piece{
              model.TokenText(token), model.IsStopToken(token)};
        });
  }
  auto binding = std::make_shared<sampling::TokenConstraint>();
  binding->grammar = std::move(grammar);
  binding->vocabulary = vocabulary;
  return binding;
}

enum class Mode { kAr, kMtp, kMtpFixed, kMtpFixedRestore };
const char* ModeName(Mode m) {
  return m == Mode::kAr         ? "ar"
         : m == Mode::kMtp      ? "mtp"
         : m == Mode::kMtpFixed ? "mtp_fixed"
                                : "mtp_fixed_restore";
}

enum class Inject { kNone, kDropCorrection, kSkipAccept, kTeacherPrefill };

struct Evidence {
  std::vector<double> pit;
  std::vector<std::vector<double>> pit_by_position;
  std::size_t impossible{0};
  Calibration stops;
  Calibration top;
};

void Record(Evidence* e, std::size_t position, const PitValue& v,
            bool stopped) {
  e->pit.push_back(v.u);
  if (e->pit_by_position.size() <= position) {
    e->pit_by_position.resize(position + 1);
  }
  e->pit_by_position[position].push_back(v.u);
  e->impossible += v.possible ? 0 : 1;
  e->stops.Add(stopped, v.p_stop);
  e->top.Add(v.top, v.p_top);
}

/// Tool-call checks of one mode (tool prompts).
struct CallChecks {
  std::size_t sequences_with_calls{0};
  std::size_t calls{0};
  std::size_t string_arguments{0};
  std::size_t argument_mismatches{0};
  std::size_t content_bytes{0};
  /// Sequences that do not re-validate under the grammar (grammar setting).
  std::size_t grammar_invalid{0};
  /// Binding turn: `mode` outside the enum, `mode_bits` not an integer.
  std::size_t schema_violations{0};
  std::size_t mode_arguments{0};
  std::size_t integer_arguments{0};
  /// Forcing turn: `value` equal to the const, `label` in the enum.
  std::size_t value_arguments{0};
  std::size_t label_arguments{0};
};

/// Whether the grammar mask bound: per row counters over all sequences.
struct Binding {
  std::size_t opened{0};          // sequences in which `<tool_call>` started
  std::size_t in_call_tokens{0};  // tokens generated inside a call
  std::size_t steps{0};           // sampling steps measured
  std::size_t masked_steps{0};    // steps where the mask removed mass > 1e-6
  std::size_t masked_in_call{0};  // the same, inside a call
  double removed_total{0.0};
  double removed_max{0.0};
  // Against the unfiltered softmax (temperature only, no penalties or
  // top-k/top-p/min-p): steps where the mask excluded a token of non-zero
  // probability (any step it restricts at all), in a call, steps above 1e-6
  // of mass, and that mass.
  std::size_t free_excluded_steps{0};
  std::size_t free_excluded_in_call{0};
  std::size_t free_masked_steps{0};
  double free_removed_total{0.0};
  double free_removed_max{0.0};
};

/// Mass the grammar mask removes at one step.
struct Removed {
  double filtered{0.0};    // of the preset-filtered distribution
  double unfiltered{0.0};  // of softmax(logits / temperature)
  bool excluded{false};    // a forbidden token has non-zero unfiltered mass
};

struct ModeResult {
  Mode mode{Mode::kAr};
  std::vector<Sequence> sequences;
  /// Per sequence: mean log p of the committed outcomes (teacher-forced).
  std::vector<double> mean_log_p;
  std::vector<double> repeat_rate;
  /// MTP sequences whose final logits differ from teacher forcing (bitwise).
  std::size_t final_logits_mismatch{0};
  double final_logits_max_abs{0.0};
  std::size_t restores{0};
  Evidence evidence;
  CallChecks calls;
  Binding binding;
  double seconds{0.0};
  std::uint64_t cycles{0};
  std::uint64_t drafted{0};
  std::uint64_t accepted{0};
};

/// Concatenated token pieces (the server's generated text).
std::string Text(const l3::Model& model, const Sequence& s) {
  std::string text;
  for (const auto token : s.tokens) {
    text += model.TokenText(static_cast<std::int32_t>(token));
  }
  return text;
}

/// Parses every complete call with the server's Bailing parser and checks
/// that each string argument equals the raw text between its value tags.
void CheckCalls(const std::string& generated, bool thinking,
                const ToolSpec* strict, CallChecks* out) {
  std::string text = generated;
  if (thinking) {
    // Reasoning ends at `</think>` or at a call opener (the Bailing parser).
    const auto think = text.find("</think>");
    const auto opener = text.find("<tool_call>");
    if (think != std::string::npos &&
        (opener == std::string::npos || think < opener)) {
      text.erase(0, think + 8);
    } else if (think == std::string::npos && opener == std::string::npos) {
      return;  // all reasoning
    }
  }
  std::vector<tokenization::ChatTool> tools;
  const bool bind = strict == &kBindTool;
  const bool force = strict == &kForceTool;
  for (const auto& tool : strict ? std::span(strict, 1) : std::span(kTools)) {
    tokenization::ChatTool t;
    t.name = tool.name;
    t.parameters_json = tool.parameters;
    tools.push_back(std::move(t));
  }
  const auto scan = server::ScanBailingToolCalls(text, tools, false, false);
  out->sequences_with_calls += scan.calls.empty() ? 0 : 1;
  for (const auto& call : scan.calls) {
    ++out->calls;
    // Raw values by a sequential scan of the call's pairs: a raw value ends
    // at the first value closer, so key-like text inside a value is skipped.
    std::map<std::string, std::string> raw;
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
      raw[text.substr(key + 9, key_end - key - 9)] =
          text.substr(value + 11, close - value - 11);
      at = close + 12;
    }
    for (const auto& argument : call.arguments) {
      if (bind && argument.name == "mode_bits") {
        ++out->integer_arguments;
        const bool digits =
            !argument.is_string && !argument.value.empty() &&
            std::ranges::all_of(argument.value, [](unsigned char c) {
              return std::isdigit(c) != 0;
            });
        out->schema_violations += digits ? 0 : 1;
      }
      if (!argument.is_string) {
        continue;
      }
      ++out->string_arguments;
      if (bind && argument.name == "mode") {
        ++out->mode_arguments;
        out->schema_violations +=
            argument.value == "create" || argument.value == "overwrite" ? 0 : 1;
      }
      if (force && argument.name == "value") {
        ++out->value_arguments;
        out->schema_violations += argument.value == kForceLiteral ? 0 : 1;
      }
      if (force && argument.name == "label") {
        ++out->label_arguments;
        out->schema_violations +=
            argument.value == "zeta-7" || argument.value == "omega-3" ? 0 : 1;
      }
      const auto found = raw.find(argument.name);
      const bool same = found != raw.end() && found->second == argument.value;
      out->argument_mismatches += same ? 0 : 1;
      if (argument.name == "content") {
        out->content_bytes += argument.value.size();
      }
      if (!same && out->argument_mismatches <= 2) {
        std::fprintf(stderr,
                     "[calls] argument %s differs from its raw text "
                     "(parsed %zu bytes)\n",
                     argument.name.c_str(), argument.value.size());
      }
    }
  }
}

/// Probability mass the unconstrained sampler (same settings, penalties and
/// history, no grammar) puts on tokens the grammar forbids at this position;
/// 0 while the mask makes no restriction.
Removed RemovedMass(const sampling::SamplerState& sampler,
                    std::span<const float> logits) {
  if (!sampler.NeedsConstraintMask()) {
    return {};
  }
  const auto mask =
      sampler.config().constraint->Allowed(sampler.constraint_state());
  Removed out;
  const auto free = sampler.WithoutConstraint().Distribution(logits);
  for (const auto& entry : free.entries()) {
    if (static_cast<std::size_t>(entry.token) < mask->size() &&
        (*mask)[static_cast<std::size_t>(entry.token)] == 0) {
      out.filtered += entry.value;
    }
  }
  const double inv_t = 1.0 / static_cast<double>(sampler.config().temperature);
  double top = -std::numeric_limits<double>::infinity();
  for (const float l : logits) {
    if (std::isfinite(l)) {
      top = std::max(top, static_cast<double>(l));
    }
  }
  double total = 0.0;
  double forbidden = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i])) {
      continue;
    }
    const double w = std::exp((static_cast<double>(logits[i]) - top) * inv_t);
    total += w;
    if (i < mask->size() && (*mask)[i] == 0) {
      forbidden += w;
    }
  }
  out.excluded = forbidden > 0.0;
  out.unfiltered = total > 0.0 ? forbidden / total : 0.0;
  return out;
}

/// Adds one sequence to the row's binding counters: `removed[i]` is the mass
/// the mask removed at step i (the last step may be the stop).
void AddBinding(const l3::Model& model, const Sequence& s,
                const std::vector<Removed>& removed, Binding* out) {
  constexpr std::string_view kOpen = "<tool_call>";
  constexpr std::string_view kClose = "</tool_call>";
  std::string text;
  bool in_call = false;
  bool opened = false;
  for (std::size_t i = 0; i < removed.size(); ++i) {
    // Step i chooses token i; `in_call` is the state before it.
    ++out->steps;
    if (removed[i].filtered > 1e-6) {
      ++out->masked_steps;
      out->masked_in_call += in_call ? 1 : 0;
      out->removed_total += removed[i].filtered;
      out->removed_max = std::max(out->removed_max, removed[i].filtered);
    }
    if (removed[i].excluded) {
      ++out->free_excluded_steps;
      out->free_excluded_in_call += in_call ? 1 : 0;
    }
    if (removed[i].unfiltered > 1e-6) {
      ++out->free_masked_steps;
      out->free_removed_total += removed[i].unfiltered;
      out->free_removed_max =
          std::max(out->free_removed_max, removed[i].unfiltered);
    }
    if (i >= s.tokens.size()) {
      continue;
    }
    const auto piece = model.TokenText(static_cast<std::int32_t>(s.tokens[i]));
    const std::size_t before = text.size();
    text += piece;
    const auto close = text.rfind(kClose);
    const auto open = text.rfind(kOpen);
    // A match is new when it ends inside the piece just appended.
    const bool fresh_close =
        close != std::string::npos && close + kClose.size() > before;
    const bool fresh_open =
        open != std::string::npos && open + kOpen.size() > before;
    if (fresh_open && (!fresh_close || open > close)) {
      in_call = true;
      opened = true;
    } else if (fresh_close) {
      out->in_call_tokens += 1;
      in_call = false;
      continue;
    }
    out->in_call_tokens += in_call ? 1 : 0;
  }
  out->opened += opened ? 1 : 0;
}

/// Re-validates a sequence from scratch: bytes walked through the grammar, a
/// stop only where the grammar is complete.
bool GrammarValid(const l3::Model& model, const Constraint& grammar,
                  const Sequence& s) {
  auto state = grammar.Start();
  for (const auto token : s.tokens) {
    for (const unsigned char byte :
         model.TokenText(static_cast<std::int32_t>(token))) {
      state = grammar.Advance(state, byte);
      if (state.empty()) {
        return false;
      }
    }
  }
  return !s.stopped || grammar.Complete(state);
}

int Execute(const Args& args, Report& report,
            const std::shared_ptr<l3::Model>& model) {
  const auto samples = static_cast<std::size_t>(args.Num("--samples", 400));
  const auto length = static_cast<std::size_t>(args.Num("--tokens", 12));
  const auto fixed = static_cast<std::uint32_t>(args.Num("--fixed-drafts", 3));
  const auto restore_every =
      static_cast<std::size_t>(args.Num("--restore-every", 48));
  const auto position_limit =
      static_cast<std::size_t>(args.Num("--position-limit", 32));
  const auto prompts = Prompts();
  std::vector<std::size_t> prompt_ids;
  if (args.Has("--prompt-ids")) {
    for (const auto id : ParseIds(args.Str("--prompt-ids"))) {
      if (id < 0 || static_cast<std::size_t>(id) >= prompts.size()) {
        report.Set("status", "error");
        report.Set("error", "unknown --prompt-ids entry " + std::to_string(id));
        return 1;
      }
      prompt_ids.push_back(static_cast<std::size_t>(id));
    }
  } else {
    const auto count = std::min<std::size_t>(
        static_cast<std::size_t>(args.Num("--prompts", 2)), prompts.size());
    for (std::size_t i = 0; i < count; ++i) {
      prompt_ids.push_back(i);
    }
  }
  const auto permutations =
      static_cast<std::size_t>(args.Num("--permutations", 20000));
  // Pooled-frequency permutations cost a pass over every token each.
  const auto unigram_permutations = std::min<std::size_t>(permutations, 2000);
  const bool require_binding = args.Has("--require-binding");
  const bool require_calls = args.Has("--require-calls");
  const auto seed_base = static_cast<std::int64_t>(args.Num("--seed-base", 1));
  const std::string config_list = args.Str("--configs", "preset,penalties");
  const std::string inject_name = args.Str("--inject", "none");
  const Inject inject =
      inject_name == "drop-correction"   ? Inject::kDropCorrection
      : inject_name == "skip-accept"     ? Inject::kSkipAccept
      : inject_name == "teacher-prefill" ? Inject::kTeacherPrefill
                                         : Inject::kNone;
  if (inject == Inject::kNone && inject_name != "none") {
    report.Set("status", "error");
    report.Set("error", "unknown --inject " + inject_name);
    return 1;
  }
  std::vector<Mode> modes;
  {
    std::stringstream list(args.Str("--modes", "ar,mtp,mtp_fixed"));
    std::string name;
    while (std::getline(list, name, ',')) {
      if (name == "ar" || name == "mtp" || name == "mtp_fixed" ||
          name == "mtp_fixed_restore") {
        modes.push_back(name == "ar"          ? Mode::kAr
                        : name == "mtp"       ? Mode::kMtp
                        : name == "mtp_fixed" ? Mode::kMtpFixed
                                              : Mode::kMtpFixedRestore);
      } else {
        report.Set("status", "error");
        report.Set("error", "unknown --modes entry " + name);
        return 1;
      }
    }
  }
  const bool restores =
      std::ranges::find(modes, Mode::kMtpFixedRestore) != modes.end();
  constexpr double kKsAlpha = 1e-4;
  constexpr double kMaxZ = 4.5;
  constexpr double kPermutationAlpha = 2e-4;
  if (!model->HasMtp()) {
    report.Set("status", "error");
    report.Set("error", "model was loaded without MTP");
    return 1;
  }

  struct Setting {
    std::string name;
    sampling::SamplingConfig config;
    bool grammar{false};
    bool bind{false};   // forced strict write_file (prompts with `bind`)
    bool force{false};  // forced strict record (prompts with `force`)
  };
  std::vector<Setting> settings;
  {
    std::stringstream list(config_list);
    std::string name;
    while (std::getline(list, name, ',')) {
      sampling::SamplingConfig c;  // the server's Ling preset
      c.temperature = 0.6F;
      c.top_p = 0.95F;
      c.top_k = 20;
      if (name == "penalties") {
        c.presence_penalty = 1.5F;
        c.frequency_penalty = 0.5F;
        c.repeat_penalty = 1.1F;
        c.repeat_last_n = 64;
      } else if (name == "wide") {
        c = {};
        c.temperature = 1.0F;
        c.top_p = 0.9F;
        c.min_p = 0.02F;
      } else if (name != "preset" && name != "grammar" &&
                 name != "grammar_required" && name != "grammar_bind" &&
                 name != "grammar_force") {
        report.Set("status", "error");
        report.Set("error", "unknown --configs entry " + name);
        return 1;
      }
      settings.push_back({name, c, name.starts_with("grammar"),
                          name == "grammar_bind", name == "grammar_force"});
    }
  }
  report.Set("inject", inject_name);

  std::string error;
  auto session = model->CreateSession(model->MaxContext(), &error);
  std::unique_ptr<l3::Session> spare;
  if (session && restores) {
    spare = model->CreateSession(model->MaxContext(), &error);
  }
  if (!session || (restores && !spare)) {
    report.Set("status", "error");
    report.Set("error", "session: " + error);
    return 1;
  }
  const auto is_stop = [&](sampling::TokenId t) {
    return model->IsStopToken(static_cast<std::int32_t>(t));
  };
  const auto fail = [&](const std::string& what) {
    std::fprintf(stderr, "mtp_distribution: %s: %s\n", what.c_str(),
                 error.c_str());
    report.Set("status", "error");
    report.Set("error", what + ": " + error);
    return 1;
  };

  auto checks = json::Value::array();
  auto results = json::Value::array();
  int failures = 0;
  const auto check = [&](const std::string& name, bool pass,
                         json::Value metrics) {
    metrics["name"] = name;
    metrics["pass"] = pass;
    checks.push_back(std::move(metrics));
    std::fprintf(stderr, "[check] %-64s %s\n", name.c_str(),
                 pass ? "PASS" : "FAIL");
    failures += pass ? 0 : 1;
  };

  for (const std::size_t pi : prompt_ids) {
    const auto& prompt = prompts[pi];
    const auto prompt_tokens = model->Tokenize(prompt.text);
    if (prompt_tokens.size() + length + 16 > model->MaxContext()) {
      error = std::to_string(prompt_tokens.size()) + " prompt tokens + " +
              std::to_string(length) + " exceed --max-context";
      return fail("context");
    }
    session->Reset();
    if (!session->Feed(prompt_tokens, {}, &error)) {
      return fail("prompt prefill");
    }
    const auto snapshot = session->SaveSnapshot(&error);
    if (!snapshot) {
      return fail("prompt snapshot");
    }
    const std::vector<sampling::TokenId> history(prompt_tokens.begin(),
                                                 prompt_tokens.end());
    const auto restore = [&]() {
      if (!session->RestoreSnapshot(*snapshot, &error)) {
        return false;
      }
      session->ResetDraftPolicy();
      return true;
    };

    for (const auto& setting : settings) {
      if (setting.grammar && !prompt.tools) {
        continue;  // the tool grammar needs a prompt that declares tools
      }
      if (setting.bind != prompt.bind || setting.force != prompt.force) {
        // a forced call needs its own turn's strict tool; the other rows
        // use the turns that declare the general tools
        if (setting.bind || setting.force || prompt.bind || prompt.force) {
          continue;
        }
      }
      auto base_config = setting.config;
      if (setting.grammar) {
        base_config.constraint = BindToolGrammar(
            *model, prompt.thinking,
            setting.bind || setting.force || setting.name == "grammar_required",
            setting.bind || setting.force ? StrictTool(setting.force)
                                          : nullptr);
      }
      std::mt19937_64 pit_rng(0x5eed + pi);
      std::uniform_real_distribution<double> unit(0.0, 1.0);
      std::vector<ModeResult> by_mode;
      for (const Mode mode : modes) {
        ModeResult r;
        r.mode = mode;
        const auto start = std::chrono::steady_clock::now();
        const l3::Session::DraftStats before = session->Statistics();
        const l3::Session::DraftStats spare_before =
            spare ? spare->Statistics() : l3::Session::DraftStats{};
        l3::Session::DraftHooks hooks;
        hooks.fixed_drafts =
            mode == Mode::kMtpFixed || mode == Mode::kMtpFixedRestore ? fixed
                                                                      : 0U;
        session->SetDraftHooks(hooks);
        if (spare) {
          spare->SetDraftHooks(hooks);
        }
        // The restore mode replays mtp_fixed's seeds: identical sequences.
        const Mode seed_mode =
            mode == Mode::kMtpFixedRestore ? Mode::kMtpFixed : mode;
        for (std::size_t n = 0; n < samples; ++n) {
          auto config = base_config;
          config.seed = seed_base +
                        1000000 * static_cast<std::int64_t>(seed_mode) +
                        static_cast<std::int64_t>(n);
          if (!restore()) {
            return fail("restore");
          }
          Sequence s;
          std::vector<Removed> removed;
          double log_p = 0.0;
          std::size_t outcomes = 0;
          if (mode == Mode::kAr) {
            sampling::SamplerState sampler(config, history);
            while (s.tokens.size() < length) {
              const auto logits = session->Logits();
              removed.push_back(RemovedMass(sampler, logits));
              const auto p = sampler.Distribution(logits);
              const auto token = sampler.Sample(logits);
              const bool stop = is_stop(token);
              const auto v =
                  Pit(p, stop ? kStopOutcome : token, is_stop, unit(pit_rng));
              Record(&r.evidence, s.tokens.size(), v, stop);
              log_p += std::log(std::max(v.p_outcome, 1e-300));
              ++outcomes;
              if (stop) {
                s.stopped = true;
                break;
              }
              sampler.Accept(token);
              s.tokens.push_back(token);
              if (!session->Evaluate(static_cast<std::int32_t>(token),
                                     &error)) {
                return fail("ar decode");
              }
            }
          } else {
            // LingTextRunner::DecodeStep and the pool's request loop.
            sampling::SamplerState sampler(config, history);
            l3::Session* active = session.get();
            std::size_t since_restore = 0;
            while (s.tokens.size() < length) {
              const std::size_t budget = std::min<std::size_t>(
                  l3::kMaxVerifyTokens, length - s.tokens.size());
              sampling::SamplerState working = sampler;
              l3::Session::DecodeResult step;
              if (!active->DecodeStep(budget, working, &step, &error, true)) {
                return fail("mtp decode");
              }
              sampler.CopyDrawStateFrom(working);
              if (inject == Inject::kDropCorrection) {
                sampler.RestoreDrawState({sampler.rng_state(), std::nullopt});
              }
              for (std::size_t k = 0; k < step.tokens.size(); ++k) {
                const auto token = step.tokens[k];
                if (!(inject == Inject::kSkipAccept && k == 0)) {
                  sampler.Accept(static_cast<sampling::TokenId>(token));
                }
                s.tokens.push_back(token);
              }
              if (step.stop) {
                s.stopped = s.tokens.size() < length;
                break;
              }
              since_restore += step.tokens.size();
              if (mode == Mode::kMtpFixedRestore &&
                  since_restore >= restore_every) {
                // A prompt-cache round trip: snapshot, restore into the other
                // session (which held another context), fresh draft policy.
                since_restore = 0;
                const auto saved = active->SaveSnapshot(&error);
                l3::Session* other =
                    active == session.get() ? spare.get() : session.get();
                if (!saved || !other->RestoreSnapshot(*saved, &error)) {
                  return fail("mid-sequence snapshot restore");
                }
                other->ResetDraftPolicy();
                active = other;
                ++r.restores;
              }
            }
            if (s.tokens.size() > length) {
              error = "decode exceeded its budget";
              return fail("mtp decode");
            }
            const std::vector<float> final_logits(active->Logits().begin(),
                                                  active->Logits().end());
            // Teacher forcing: p of every position from plain decoding rows
            // (chunks of at most 8 tokens: the small-batch family).
            if (!restore()) {
              return fail("restore");
            }
            sampling::SamplerState plain(base_config, history);
            const std::size_t vocab = session->Logits().size();
            const std::size_t positions = s.tokens.size() + (s.stopped ? 1 : 0);
            // Row i: the logits before position i (row 0 = the prompt's).
            std::vector<float> all(session->Logits().begin(),
                                   session->Logits().end());
            // Every committed token: the last row must equal the MTP
            // session's final logits.
            const std::size_t feed = s.tokens.size();
            std::vector<float> rows;
            const std::size_t chunk_tokens =
                inject == Inject::kTeacherPrefill ? feed : l3::kMaxVerifyTokens;
            for (std::size_t off = 0; off < feed; off += chunk_tokens) {
              const std::size_t n_rows =
                  std::min<std::size_t>(chunk_tokens, feed - off);
              std::vector<std::int32_t> chunk;
              for (std::size_t k = 0; k < n_rows; ++k) {
                chunk.push_back(static_cast<std::int32_t>(s.tokens[off + k]));
              }
              rows.clear();
              l3::FeedOptions options;
              options.logits_from = 0;
              options.all_logits = &rows;
              if (!session->Feed(chunk, options, &error) ||
                  rows.size() != n_rows * vocab) {
                return fail("teacher forcing");
              }
              all.insert(all.end(), rows.begin(), rows.end());
            }
            const std::span<const float> last =
                std::span<const float>(all).subspan(feed * vocab, vocab);
            if (!std::equal(last.begin(), last.end(), final_logits.begin(),
                            final_logits.end())) {
              ++r.final_logits_mismatch;
              for (std::size_t k = 0; k < vocab && k < final_logits.size();
                   ++k) {
                r.final_logits_max_abs = std::max(
                    r.final_logits_max_abs,
                    std::fabs(static_cast<double>(last[k]) - final_logits[k]));
              }
            }
            for (std::size_t i = 0; i < positions; ++i) {
              const auto row =
                  std::span<const float>(all).subspan(i * vocab, vocab);
              removed.push_back(RemovedMass(plain, row));
              const auto p = plain.Distribution(row);
              const bool stop = i == s.tokens.size();
              const std::int64_t outcome = stop ? kStopOutcome : s.tokens[i];
              const auto v = Pit(p, outcome, is_stop, unit(pit_rng));
              Record(&r.evidence, i, v, stop);
              log_p += std::log(std::max(v.p_outcome, 1e-300));
              ++outcomes;
              if (!v.possible) {
                break;  // a grammar violation: no target for the rest
              }
              if (!stop) {
                plain.Accept(static_cast<sampling::TokenId>(s.tokens[i]));
              }
            }
          }
          r.mean_log_p.push_back(
              log_p / static_cast<double>(std::max<std::size_t>(1, outcomes)));
          r.repeat_rate.push_back(RepeatedNgramRate(s));
          if (prompt.tools) {
            CheckCalls(Text(*model, s), prompt.thinking,
                       prompt.bind || prompt.force ? StrictTool(prompt.force)
                                                   : nullptr,
                       &r.calls);
          }
          if (setting.grammar) {
            AddBinding(*model, s, removed, &r.binding);
          }
          if (setting.grammar &&
              !GrammarValid(*model, *base_config.constraint->grammar, s)) {
            ++r.calls.grammar_invalid;
          }
          r.sequences.push_back(std::move(s));
          if ((n + 1) % 25 == 0) {
            std::fprintf(stderr, "[progress] p%zu %s %s %zu/%zu %.0fs\n", pi,
                         setting.name.c_str(), ModeName(mode), n + 1, samples,
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start)
                             .count());
          }
        }
        const auto after = session->Statistics();
        const auto spare_after =
            spare ? spare->Statistics() : l3::Session::DraftStats{};
        r.cycles = after.cycles - before.cycles + spare_after.cycles -
                   spare_before.cycles;
        r.drafted = after.drafted - before.drafted + spare_after.drafted -
                    spare_before.drafted;
        r.accepted = after.accepted - before.accepted + spare_after.accepted -
                     spare_before.accepted;
        r.seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
        by_mode.push_back(std::move(r));
      }
      session->SetDraftHooks({});
      if (spare) {
        spare->SetDraftHooks({});
      }

      const std::string base =
          "p" + std::to_string(pi) + "_" + setting.name + "_";
      const ModeResult* ar = nullptr;
      const ModeResult* fixed_result = nullptr;
      for (const auto& other : by_mode) {
        ar = other.mode == Mode::kAr ? &other : ar;
        fixed_result = other.mode == Mode::kMtpFixed ? &other : fixed_result;
      }
      for (std::size_t m = 0; m < by_mode.size(); ++m) {
        const auto& r = by_mode[m];
        const Mode mode = r.mode;
        const auto& e = r.evidence;
        const auto ks = KsUniform(e.pit);
        double worst_position_p = 1.0;
        std::size_t worst_position = 0;
        for (std::size_t i = 0; i < e.pit_by_position.size(); ++i) {
          const auto k = KsUniform(e.pit_by_position[i]);
          if (k.p < worst_position_p) {
            worst_position_p = k.p;
            worst_position = i;
          }
        }
        const double adjusted =
            std::min(1.0, worst_position_p *
                              static_cast<double>(e.pit_by_position.size()));
        std::size_t stopped = 0;
        std::size_t tokens = 0;
        for (const auto& s : r.sequences) {
          stopped += s.stopped ? 1 : 0;
          tokens += s.tokens.size();
        }
        double mean_log_p = 0.0;
        double repeat_rate = 0.0;
        for (std::size_t k = 0; k < r.sequences.size(); ++k) {
          mean_log_p += r.mean_log_p[k];
          repeat_rate += r.repeat_rate[k];
        }
        const double count =
            static_cast<double>(std::max<std::size_t>(1, r.sequences.size()));
        auto o = json::Value::object();
        o["prompt"] = U(pi);
        o["config"] = setting.name;
        o["mode"] = ModeName(mode);
        o["samples"] = U(r.sequences.size());
        o["pit_values"] = U(ks.n);
        o["ks_d"] = Round(ks.d);
        o["ks_p"] = Round(ks.p);
        o["worst_position"] = U(worst_position);
        o["worst_position_p_bonferroni"] = Round(adjusted);
        o["impossible_tokens"] = U(e.impossible);
        o["stops"] = U(stopped);
        o["stop_expected"] = Round(e.stops.expected, 6);
        o["stop_z"] = Round(e.stops.z());
        o["top1_observed"] = Round(e.top.observed, 6);
        o["top1_expected"] = Round(e.top.expected, 6);
        o["top1_z"] = Round(e.top.z());
        o["mean_length"] = Round(static_cast<double>(tokens) / count);
        o["mean_log_p"] = Round(mean_log_p / count, 6);
        o["repeated_4gram_rate"] = Round(repeat_rate / count, 6);
        o["seconds"] = Round(r.seconds);
        o["cycles"] = U(r.cycles);
        o["drafted"] = U(r.drafted);
        o["accepted"] = U(r.accepted);
        o["restores"] = U(r.restores);
        o["final_logits_mismatch"] = U(r.final_logits_mismatch);
        o["final_logits_max_abs"] = Round(r.final_logits_max_abs);
        o["acceptance"] = r.drafted ? Round(static_cast<double>(r.accepted) /
                                            static_cast<double>(r.drafted))
                                    : json::Value();
        if (prompt.tools) {
          auto c = json::Value::object();
          c["sequences_with_calls"] = U(r.calls.sequences_with_calls);
          c["calls"] = U(r.calls.calls);
          c["string_arguments"] = U(r.calls.string_arguments);
          c["argument_mismatches"] = U(r.calls.argument_mismatches);
          c["content_bytes"] = U(r.calls.content_bytes);
          c["grammar_invalid"] = U(r.calls.grammar_invalid);
          c["schema_violations"] = U(r.calls.schema_violations);
          c["mode_arguments"] = U(r.calls.mode_arguments);
          c["integer_arguments"] = U(r.calls.integer_arguments);
          c["value_arguments"] = U(r.calls.value_arguments);
          c["label_arguments"] = U(r.calls.label_arguments);
          o["tool_calls"] = std::move(c);
        }
        if (setting.grammar) {
          const auto& b = r.binding;
          auto j = json::Value::object();
          j["sequences_opened_call"] = U(b.opened);
          j["sequences_with_complete_call"] = U(r.calls.sequences_with_calls);
          j["tokens_in_call"] = U(b.in_call_tokens);
          j["steps"] = U(b.steps);
          j["masked_steps"] = U(b.masked_steps);
          j["masked_steps_in_call"] = U(b.masked_in_call);
          j["removed_mass_total"] = Round(b.removed_total, 6);
          j["removed_mass_max"] = Round(b.removed_max, 6);
          j["unfiltered_excluded_steps"] = U(b.free_excluded_steps);
          j["unfiltered_excluded_steps_in_call"] = U(b.free_excluded_in_call);
          j["unfiltered_masked_steps"] = U(b.free_masked_steps);
          j["unfiltered_removed_mass_total"] = Round(b.free_removed_total, 6);
          j["unfiltered_removed_mass_max"] = Round(b.free_removed_max, 6);
          o["grammar_binding"] = std::move(j);
          std::fprintf(
              stderr,
              "[binding] p%zu %s %s opened %zu/%zu complete %zu "
              "in-call tokens %zu masked steps %zu/%zu (in call %zu) "
              "removed mass total %.4g max %.4g; unfiltered excluded steps "
              "%zu (in call %zu) masked(>1e-6) %zu mass total %.4g "
              "max %.4g\n",
              pi, setting.name.c_str(), ModeName(mode), b.opened,
              r.sequences.size(), r.calls.sequences_with_calls,
              b.in_call_tokens, b.masked_steps, b.steps, b.masked_in_call,
              b.removed_total, b.removed_max, b.free_excluded_steps,
              b.free_excluded_in_call, b.free_masked_steps,
              b.free_removed_total, b.free_removed_max);
        }
        if (mode != Mode::kAr && ar != nullptr) {
          const std::size_t positions =
              std::min<std::size_t>(length + 1, position_limit);
          const auto two =
              CompareSequences(r.sequences, ar->sequences, positions,
                               permutations, 0xC0FFEE + pi * 16 + m);
          const auto unigram =
              UnigramTest(r.sequences, ar->sequences, unigram_permutations,
                          0xBEEF + pi * 16 + m);
          const auto log_p = MeanDifferenceTest(
              r.mean_log_p, ar->mean_log_p, permutations, 0xFACE + pi * 16 + m);
          const auto repeats =
              MeanDifferenceTest(r.repeat_rate, ar->repeat_rate, permutations,
                                 0xD00D + pi * 16 + m);
          o["vs_ar_position_tv_sum"] = Round(two.positions.statistic);
          o["vs_ar_position_p"] = Round(two.positions.p);
          o["vs_ar_stop_rate_p"] = Round(two.stop_rate.p);
          o["vs_ar_length_p"] = Round(two.length.p);
          o["vs_ar_unigram_tv"] = Round(unigram.statistic);
          o["vs_ar_unigram_p"] = Round(unigram.p);
          o["vs_ar_mean_log_p_diff"] = Round(log_p.statistic);
          o["vs_ar_mean_log_p_p"] = Round(log_p.p);
          o["vs_ar_repeat_rate_diff"] = Round(repeats.statistic);
          o["vs_ar_repeat_rate_p"] = Round(repeats.p);
          auto tv = json::Value::array();
          for (const double v : two.position_tv) {
            tv.push_back(Round(v));
          }
          o["vs_ar_position_tv"] = std::move(tv);
          // The pooled-frequency test has fewer permutations: its smallest
          // attainable p is 1 / (permutations + 1).
          const double unigram_alpha =
              std::max(kPermutationAlpha,
                       1.0 / static_cast<double>(unigram_permutations));
          check(base + ModeName(mode) + "_vs_ar_two_sample",
                two.positions.p >= kPermutationAlpha &&
                    two.stop_rate.p >= kPermutationAlpha &&
                    two.length.p >= kPermutationAlpha &&
                    unigram.p >= unigram_alpha &&
                    log_p.p >= kPermutationAlpha &&
                    repeats.p >= kPermutationAlpha,
                o);
        }
        if (mode == Mode::kMtpFixedRestore && fixed_result != nullptr) {
          std::size_t differ = 0;
          for (std::size_t k = 0;
               k < r.sequences.size() && k < fixed_result->sequences.size();
               ++k) {
            const auto& a = r.sequences[k];
            const auto& b = fixed_result->sequences[k];
            differ += a.tokens == b.tokens && a.stopped == b.stopped ? 0 : 1;
          }
          o["sequences_differing_from_mtp_fixed"] = U(differ);
          check(base + "mtp_fixed_restore_equals_mtp_fixed",
                differ == 0 && r.restores > 0, o);
        }
        if (prompt.tools) {
          check(base + ModeName(mode) + "_tool_arguments",
                r.calls.argument_mismatches == 0 &&
                    r.calls.grammar_invalid == 0 &&
                    r.calls.schema_violations == 0,
                o);
        }
        if (require_binding && setting.grammar) {
          // The grammar must have opened a call, generated inside it, and
          // removed probability mass, inside the call too (key, enum and
          // integer positions).
          const auto& b = r.binding;
          // grammar_bind: a realistic turn the model may already obey after the
          // preset filters, so it must only exclude tokens the unfiltered
          // softmax allows; grammar_force must remove filtered mass too.
          const bool binds = setting.bind
                                 ? b.free_excluded_in_call > 0
                                 : b.masked_steps > 0 && b.masked_in_call > 0;
          check(base + ModeName(mode) + "_grammar_binds",
                b.opened > 0 && b.in_call_tokens > 0 && binds, o);
        }
        if (require_calls && setting.force) {
          // Forced values: complete calls (every sequence unless thinking,
          // which may still reason at the token limit) whose `value` is the
          // const and `label` an enum member (no violations is checked above).
          check(base + ModeName(mode) + "_calls_complete",
                r.calls.sequences_with_calls > 0 &&
                    r.calls.value_arguments == r.calls.calls &&
                    r.calls.label_arguments == r.calls.calls &&
                    (prompt.thinking ||
                     r.calls.sequences_with_calls == r.sequences.size()),
                o);
        }
        if (require_calls && setting.bind) {
          // Forced binding: calls complete and `mode`, `mode_bits` parsed.
          check(base + ModeName(mode) + "_calls_complete",
                r.calls.sequences_with_calls > 0 &&
                    r.calls.mode_arguments > 0 && r.calls.integer_arguments > 0,
                o);
        }
        std::fprintf(
            stderr,
            "[result] p%zu %-9s %-17s KS D=%.4f p=%.3g worst-pos p=%.3g "
            "stops %zu (exp %.1f z=%.2f) top1 z=%.2f impossible %zu "
            "len %.2f logp %.4f rep %.4f acc %llu/%llu final-logits mismatch "
            "%zu (max %.3g) calls %zu arg-mismatch %zu grammar-invalid %zu "
            "%.0fs\n",
            pi, setting.name.c_str(), ModeName(mode), ks.d, ks.p, adjusted,
            stopped, e.stops.expected, e.stops.z(), e.top.z(), e.impossible,
            static_cast<double>(tokens) / count, mean_log_p / count,
            repeat_rate / count, static_cast<unsigned long long>(r.accepted),
            static_cast<unsigned long long>(r.drafted), r.final_logits_mismatch,
            r.final_logits_max_abs, r.calls.calls, r.calls.argument_mismatches,
            r.calls.grammar_invalid, r.seconds);
        check(base + ModeName(mode) + "_matches_ar_target",
              e.impossible == 0 && r.final_logits_mismatch == 0 &&
                  ks.p >= kKsAlpha && adjusted >= kKsAlpha &&
                  std::fabs(e.stops.z()) <= kMaxZ &&
                  std::fabs(e.top.z()) <= kMaxZ,
              o);
        if (mode == Mode::kMtpFixed || mode == Mode::kMtpFixedRestore) {
          check(base + ModeName(mode) + "_verified_drafts", r.drafted > 0, o);
        }
        results.push_back(std::move(o));
        report.Set("results", results);
        report.Set("checks", checks);
      }
    }
  }
  report.Set("checks", std::move(checks));
  report.Set("failures", failures);
  report.Set("status", failures == 0 ? "pass" : "fail");
  return failures == 0 ? 0 : 1;
}

}  // namespace gufo::tests::ling::mtp_distribution

int main(int argc, char** argv) {
  // One copy of the report, on stdout, behind a marker: pod logs merge stdout
  // and stderr, and a second copy on stderr interleaves with progress lines.
  gufo::tests::ling::Report::StdoutMarker() = "[report-json] ";
  return gufo::tests::ling::RunTool(
      argc, argv, "mtp_distribution",
      [](const gufo::tests::ling::Args& args,
         gufo::tests::ling::Report& report) {
        auto model =
            gufo::tests::ling::LoadModel(args, report, 256, 1024, true);
        return model ? gufo::tests::ling::mtp_distribution::Execute(
                           args, report, model)
                     : 1;
      });
}
