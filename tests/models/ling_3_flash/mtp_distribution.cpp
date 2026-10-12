// Sampled MTP speculative decoding against plain autoregressive sampling on
// the GPU (Ling-3.0-flash): the output distribution must not change. For
// fixed prompts (prose, a short chat answer that usually stops within a few
// tokens, code, a list) and sampling settings (the Ling preset, and the preset
// with presence/frequency/repeat penalties) it draws --samples continuations of
// --tokens tokens, with distinct seeds, in three modes from the same prompt
// snapshot:
//   * ar: plain sampling (SamplerState::Sample + Evaluate);
//   * mtp: Session::DecodeStep as the server drives it (working sampler copy,
//     draw state published, committed tokens accepted), adaptive drafts;
//   * mtp_fixed: the same with --fixed-drafts drafts every cycle, so the
//     multi-row verify, rejection and residual paths run even where the
//     controller would fall back (a truncated --layers trunk).
// Checks per prompt, setting and mode (sampled_distribution.hpp): each MTP
// sequence is teacher-forced through the trunk and its randomized PIT values
// must be U(0, 1) (pooled and per position), the stop and top-1 counts must
// match their exact expectations, no committed token may have p = 0, and the
// MTP session's final logits must equal the teacher-forced ones bit for bit;
// the ar mode is the control of the same statistics. Two-sample permutation
// tests compare each MTP mode with ar: per-position token distributions (summed
// total variation), stop rate and length distribution. Thresholds: KS
// p >= 1e-4, |z| <= 4.5, permutation p >= 2e-4.
//
//   mtp_distribution --model FIRST_SHARD.gguf [--layers N] [--samples 400]
//       [--tokens 12] [--fixed-drafts 3] [--configs preset,penalties]
//       [--modes ar,mtp,mtp_fixed]
//       [--prompts 2] [--permutations 20000] [--seed-base 1]
//       [--max-context 1024] [--timeout-seconds S]
// Exit 77 without --model; 0 pass, 1 a failed check.

#include <chrono>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/ling_3_flash/chat_template.hpp"
#include "tests/models/ling_3_flash/model_tool.hpp"
#include "tests/models/ling_3_flash/sampled_distribution.hpp"

namespace gufo::tests::ling::mtp_distribution {

namespace sampling = gufo::sampling;

/// Prompts: prose, a short chat answer (thinking off, so most continuations
/// stop within a few tokens: stop handling, drafted stops), code, a list.
std::vector<std::string> Prompts() {
  l3::ChatMessage user;
  user.role = "user";
  user.content =
      "Answer with one word only. What colour is the sky on a clear day?";
  l3::ChatTemplateOptions options;
  options.enable_thinking = false;
  return {
      "The old lighthouse keeper climbed the spiral stairs every evening at "
      "dusk. On the night the storm arrived, he noticed something strange on "
      "the horizon:",
      l3::RenderChat(std::span(&user, 1), {}, options),
      "def parse_duration(text):\n    \"\"\"Parse strings such as '1h30m' or "
      "'45s' into a number of seconds.\"\"\"\n",
      "Here is a short list of practical tips for writing clear technical "
      "documentation:\n\n1.",
  };
}

enum class Mode { kAr, kMtp, kMtpFixed };
const char* ModeName(Mode m) {
  return m == Mode::kAr ? "ar" : m == Mode::kMtp ? "mtp" : "mtp_fixed";
}

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

struct ModeResult {
  Mode mode{Mode::kAr};
  std::vector<Sequence> sequences;
  /// MTP sequences whose final logits differ from teacher forcing (bitwise).
  std::size_t final_logits_mismatch{0};
  double final_logits_max_abs{0.0};
  Evidence evidence;
  double seconds{0.0};
  std::uint64_t cycles{0};
  std::uint64_t drafted{0};
  std::uint64_t accepted{0};
};

int Execute(const Args& args, Report& report,
            const std::shared_ptr<l3::Model>& model) {
  const auto samples = static_cast<std::size_t>(args.Num("--samples", 400));
  const auto length = static_cast<std::size_t>(args.Num("--tokens", 12));
  const auto fixed = static_cast<std::uint32_t>(args.Num("--fixed-drafts", 3));
  const auto prompts = Prompts();
  const auto prompt_count = std::min<std::size_t>(
      static_cast<std::size_t>(args.Num("--prompts", 2)), prompts.size());
  const auto permutations =
      static_cast<std::size_t>(args.Num("--permutations", 20000));
  const auto seed_base = static_cast<std::int64_t>(args.Num("--seed-base", 1));
  const std::string config_list = args.Str("--configs", "preset,penalties");
  std::vector<Mode> modes;
  {
    std::stringstream list(args.Str("--modes", "ar,mtp,mtp_fixed"));
    std::string name;
    while (std::getline(list, name, ',')) {
      if (name == "ar" || name == "mtp" || name == "mtp_fixed") {
        modes.push_back(name == "ar"    ? Mode::kAr
                        : name == "mtp" ? Mode::kMtp
                                        : Mode::kMtpFixed);
      } else {
        report.Set("status", "error");
        report.Set("error", "unknown --modes entry " + name);
        return 1;
      }
    }
  }
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
      } else if (name != "preset") {
        report.Set("status", "error");
        report.Set("error", "unknown --configs entry " + name);
        return 1;
      }
      settings.push_back({name, c});
    }
  }

  std::string error;
  auto session = model->CreateSession(model->MaxContext(), &error);
  if (!session) {
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

  for (std::size_t pi = 0; pi < prompt_count; ++pi) {
    const auto prompt = model->Tokenize(prompts[pi]);
    session->Reset();
    if (!session->Feed(prompt, {}, &error)) {
      return fail("prompt prefill");
    }
    const auto snapshot = session->SaveSnapshot(&error);
    if (!snapshot) {
      return fail("prompt snapshot");
    }
    const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    const auto restore = [&]() {
      if (!session->RestoreSnapshot(*snapshot, &error)) {
        return false;
      }
      session->ResetDraftPolicy();
      return true;
    };

    for (const auto& setting : settings) {
      std::mt19937_64 pit_rng(0x5eed + pi);
      std::uniform_real_distribution<double> unit(0.0, 1.0);
      std::vector<ModeResult> by_mode;
      for (const Mode mode : modes) {
        ModeResult r;
        r.mode = mode;
        const auto start = std::chrono::steady_clock::now();
        const auto before = session->Statistics();
        l3::Session::DraftHooks hooks;
        hooks.fixed_drafts = mode == Mode::kMtpFixed ? fixed : 0U;
        session->SetDraftHooks(std::move(hooks));
        for (std::size_t n = 0; n < samples; ++n) {
          auto config = setting.config;
          config.seed = seed_base + 1000000 * static_cast<std::int64_t>(mode) +
                        static_cast<std::int64_t>(n);
          if (!restore()) {
            return fail("restore");
          }
          Sequence s;
          if (mode == Mode::kAr) {
            sampling::SamplerState sampler(config, history);
            while (s.tokens.size() < length) {
              const auto logits = session->Logits();
              const auto p = sampler.Distribution(logits);
              const auto token = sampler.Sample(logits);
              const bool stop = is_stop(token);
              Record(
                  &r.evidence, s.tokens.size(),
                  Pit(p, stop ? kStopOutcome : token, is_stop, unit(pit_rng)),
                  stop);
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
            while (s.tokens.size() < length) {
              const std::size_t budget = std::min<std::size_t>(
                  l3::kMaxVerifyTokens, length - s.tokens.size());
              sampling::SamplerState working = sampler;
              l3::Session::DecodeResult step;
              if (!session->DecodeStep(budget, working, &step, &error, true)) {
                return fail("mtp decode");
              }
              sampler.CopyDrawStateFrom(working);
              for (const auto token : step.tokens) {
                sampler.Accept(static_cast<sampling::TokenId>(token));
                s.tokens.push_back(token);
              }
              if (step.stop) {
                s.stopped = s.tokens.size() < length;
                break;
              }
            }
            if (s.tokens.size() > length) {
              error = "decode exceeded its budget";
              return fail("mtp decode");
            }
            const std::vector<float> final_logits(session->Logits().begin(),
                                                  session->Logits().end());
            // Teacher forcing: p of every position from plain decoding rows
            // (chunks of at most 8 tokens: the small-batch family).
            if (!restore()) {
              return fail("restore");
            }
            sampling::SamplerState plain(setting.config, history);
            const std::size_t vocab = session->Logits().size();
            const std::size_t positions = s.tokens.size() + (s.stopped ? 1 : 0);
            // Row i: the logits before position i (row 0 = the prompt's).
            std::vector<float> all(session->Logits().begin(),
                                   session->Logits().end());
            // Every committed token: the last row must equal the MTP
            // session's final logits.
            const std::size_t feed = s.tokens.size();
            std::vector<float> rows;
            for (std::size_t off = 0; off < feed; off += l3::kMaxVerifyTokens) {
              const std::size_t n =
                  std::min<std::size_t>(l3::kMaxVerifyTokens, feed - off);
              std::vector<std::int32_t> chunk;
              for (std::size_t k = 0; k < n; ++k) {
                chunk.push_back(static_cast<std::int32_t>(s.tokens[off + k]));
              }
              rows.clear();
              l3::FeedOptions options;
              options.logits_from = 0;
              options.all_logits = &rows;
              if (!session->Feed(chunk, options, &error) ||
                  rows.size() != n * vocab) {
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
              const auto p = plain.Distribution(
                  std::span<const float>(all).subspan(i * vocab, vocab));
              const bool stop = i == s.tokens.size();
              const std::int64_t outcome = stop ? kStopOutcome : s.tokens[i];
              Record(&r.evidence, i, Pit(p, outcome, is_stop, unit(pit_rng)),
                     stop);
              if (!stop) {
                plain.Accept(static_cast<sampling::TokenId>(s.tokens[i]));
              }
            }
          }
          r.sequences.push_back(std::move(s));
        }
        const auto after = session->Statistics();
        r.cycles = after.cycles - before.cycles;
        r.drafted = after.drafted - before.drafted;
        r.accepted = after.accepted - before.accepted;
        r.seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
        by_mode.push_back(std::move(r));
      }
      session->SetDraftHooks({});

      const std::string base =
          "p" + std::to_string(pi) + "_" + setting.name + "_";
      for (std::size_t m = 0; m < by_mode.size(); ++m) {
        const auto& r = by_mode[m];
        const Mode mode = r.mode;
        const ModeResult* ar = nullptr;
        for (const auto& other : by_mode) {
          ar = other.mode == Mode::kAr ? &other : ar;
        }
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
        o["mean_length"] =
            Round(static_cast<double>(tokens) /
                  static_cast<double>(std::max<std::size_t>(1, samples)));
        o["seconds"] = Round(r.seconds);
        o["cycles"] = U(r.cycles);
        o["drafted"] = U(r.drafted);
        o["accepted"] = U(r.accepted);
        o["final_logits_mismatch"] = U(r.final_logits_mismatch);
        o["final_logits_max_abs"] = Round(r.final_logits_max_abs);
        o["acceptance"] = r.drafted ? Round(static_cast<double>(r.accepted) /
                                            static_cast<double>(r.drafted))
                                    : json::Value();
        if (mode != Mode::kAr && ar != nullptr) {
          const auto two =
              CompareSequences(r.sequences, ar->sequences, length + 1,
                               permutations, 0xC0FFEE + pi * 16 + m);
          o["vs_ar_position_tv_sum"] = Round(two.positions.statistic);
          o["vs_ar_position_p"] = Round(two.positions.p);
          o["vs_ar_stop_rate_p"] = Round(two.stop_rate.p);
          o["vs_ar_length_p"] = Round(two.length.p);
          auto tv = json::Value::array();
          for (const double v : two.position_tv) {
            tv.push_back(Round(v));
          }
          o["vs_ar_position_tv"] = std::move(tv);
          check(base + ModeName(mode) + "_vs_ar_two_sample",
                two.positions.p >= kPermutationAlpha &&
                    two.stop_rate.p >= kPermutationAlpha &&
                    two.length.p >= kPermutationAlpha,
                o);
        }
        std::fprintf(
            stderr,
            "[result] p%zu %-9s %-9s KS D=%.4f p=%.3g worst-pos p=%.3g "
            "stops %zu (exp %.1f z=%.2f) top1 z=%.2f impossible %zu "
            "len %.2f acc %llu/%llu final-logits mismatch %zu (max %.3g) "
            "%.0fs\n",
            pi, setting.name.c_str(), ModeName(mode), ks.d, ks.p, adjusted,
            stopped, e.stops.expected, e.stops.z(), e.top.z(), e.impossible,
            static_cast<double>(tokens) / static_cast<double>(samples),
            static_cast<unsigned long long>(r.accepted),
            static_cast<unsigned long long>(r.drafted), r.final_logits_mismatch,
            r.final_logits_max_abs, r.seconds);
        check(base + ModeName(mode) + "_matches_ar_target",
              e.impossible == 0 && r.final_logits_mismatch == 0 &&
                  ks.p >= kKsAlpha && adjusted >= kKsAlpha &&
                  std::fabs(e.stops.z()) <= kMaxZ &&
                  std::fabs(e.top.z()) <= kMaxZ,
              o);
        if (mode == Mode::kMtpFixed) {
          check(base + "mtp_fixed_verified_drafts", r.drafted > 0, o);
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
