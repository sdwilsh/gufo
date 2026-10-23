// MTP acceptance and expert-union probe on the CPU reference (the trunk runs
// token by token, F32; about 4 s per token with 16 threads). It generates
// greedily from a real prompt and, at every cycle start, runs the MTP chain on
// the exact state a speculative step would have (trunk hidden of the last
// position, anchor token, MTP cache rows filled from trunk hiddens). A draft at
// depth j counts as accepted when it equals the trunk's own greedy token at
// that position, which is what greedy verification tests. One run therefore
// gives the acceptance that greedy speculative decoding would see on this
// text, for every start position and every depth.
//
// It also records the routed experts of every MoE layer for the generated
// positions. The union of experts that a verify chunk of n consecutive tokens
// reads dominates the growth of the verify cost on this MoE model.
//
//   mtp_acceptance_probe --model FIRST_SHARD.gguf --label NAME
//       (--prompt-file TEXT | --code-file TEXT --agent [--thinking])
//       [--prompt-tokens P] [--generate N] [--depth D] [--threads T]
//       [--out FILE.json]
//
// --prompt-file: raw text (control tokens recognised), first P tokens.
// --agent: renders a coding-agent conversation with the Bailing V3 chat
//   template. It has a system prompt, tools, a user request, an assistant
//   read_file call, and a tool result that holds the --code-file text cut to
//   P tokens. --thinking turns the template's thinking mode on (default off).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/chat_template.hpp"
#include "src/models/ling_3_flash/reference.hpp"
#include "src/models/ling_3_flash/tokenizer.hpp"
#include "src/models/ling_3_flash/weights.hpp"

namespace l3 = gufo::models::ling_3_flash;

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

class RouterSink final : public l3::DumpSink {
public:
  bool Wants(std::string_view name, int layer) const override {
    return name == "router_ids" && layer >= 2 && layer < 42;
  }
  void Float(std::string_view, int, std::uint32_t,
             std::span<const float>) override {}
  void Int(std::string_view, int layer, std::uint32_t,
           std::span<const std::int32_t> data) override {
    auto& row = current_[static_cast<std::size_t>(layer - 2)];
    row.assign(data.begin(), data.end());
  }
  /// Moves the layer rows of the token just run into the history.
  void Commit() { history_.push_back(current_); }
  [[nodiscard]] const std::vector<std::array<std::vector<std::int32_t>, 40>>&
  history() const {
    return history_;
  }

private:
  std::array<std::vector<std::int32_t>, 40> current_{};
  std::vector<std::array<std::vector<std::int32_t>, 40>> history_;
};

std::int32_t Argmax(const std::vector<float>& v) {
  return static_cast<std::int32_t>(std::max_element(v.begin(), v.end()) -
                                   v.begin());
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, label = "text", prompt_file, code_file, out_path;
  std::uint32_t prompt_tokens = 96, generate = 96, depth = 4, threads = 16;
  bool agent = false, thinking = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : "";
    };
    if (a == "--model")
      model_path = next();
    else if (a == "--label")
      label = next();
    else if (a == "--prompt-file")
      prompt_file = next();
    else if (a == "--code-file")
      code_file = next();
    else if (a == "--out")
      out_path = next();
    else if (a == "--prompt-tokens")
      prompt_tokens = std::atoi(next().c_str());
    else if (a == "--generate")
      generate = std::atoi(next().c_str());
    else if (a == "--depth")
      depth = std::atoi(next().c_str());
    else if (a == "--threads")
      threads = std::atoi(next().c_str());
    else if (a == "--agent")
      agent = true;
    else if (a == "--thinking")
      thinking = true;
  }
  if (model_path.empty() || (prompt_file.empty() && code_file.empty())) {
    std::fprintf(stderr, "usage: see the file header\n");
    return 77;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  auto weights = l3::ModelWeights::Bind(*reader, &error);
  auto tok = l3::LingTokenizer::CreateFromGguf(*reader, &error);
  if (!weights || !weights->mtp || !tok) {
    std::fprintf(stderr, "bind: %s\n", error.c_str());
    return 1;
  }

  // Prompt ids.
  std::vector<std::int32_t> prompt;
  if (agent) {
    auto ids = tok->Encode(ReadFile(code_file), false);
    ids.resize(std::min<std::size_t>(ids.size(), prompt_tokens));
    const std::string code =
        tok->Decode(std::vector<l3::TokenId>(ids.begin(), ids.end()));
    std::vector<l3::ChatMessage> messages(4);
    messages[0].role = "system";
    messages[0].content =
        "You are a coding agent working in a project directory. Use the tools "
        "to inspect, edit and verify. When the task is verified, answer DONE.";
    messages[1].role = "user";
    messages[1].content =
        "The unit test for the parser fails with an off-by-one error. Find "
        "the bug in the file I opened and fix it.";
    messages[2].role = "assistant";
    messages[2].tool_calls.push_back(
        {"read_file", {{"path", "src/parser.cpp", true}}});
    messages[3].role = "tool";
    messages[3].content = code;
    const l3::ChatTool tools[] = {
        {R"({"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}})"},
        {R"({"type":"function","function":{"name":"write_file","description":"Replace a file with new contents.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}})"},
        {R"({"type":"function","function":{"name":"terminal","description":"Run a shell command in the project directory.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}})"}};
    l3::ChatTemplateOptions options;
    options.enable_thinking = thinking;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    options.literal_spans = &spans;
    const std::string text = l3::RenderChat(messages, tools, options);
    const auto enc = tok->Encode(text, true, spans);
    prompt.assign(enc.begin(), enc.end());
  } else {
    const auto enc = tok->Encode(ReadFile(prompt_file), false);
    prompt.assign(enc.begin(), enc.end());
    prompt.resize(std::min<std::size_t>(prompt.size(), prompt_tokens));
  }
  std::fprintf(stderr, "[%s] prompt %zu tokens, generating %u, depth %u\n",
               label.c_str(), prompt.size(), generate, depth);

  l3::ReferenceModel::Options ropt;
  ropt.max_context =
      static_cast<std::uint32_t>(prompt.size()) + generate + depth + 8;
  ropt.threads = threads;
  ropt.check_mla = false;
  l3::ReferenceModel ref(*weights, ropt);
  RouterSink routes;
  std::vector<float> logits(l3::kVocab), mtp_logits(l3::kVocab);
  std::vector<float> h_next(l3::kHidden), h_prev(l3::kHidden);
  std::vector<std::vector<float>> hidden;  // trunk h_i per position
  const auto t_start = std::chrono::steady_clock::now();
  const auto seconds = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         t_start)
        .count();
  };

  // Prefill: trunk and MTP cache rows (row j-1 from (h_{j-1}, token j)).
  for (std::size_t j = 0; j < prompt.size(); ++j) {
    const bool last = j + 1 == prompt.size();
    if (!ref.Step(prompt[j],
                  last ? std::span<float>(logits) : std::span<float>{}, nullptr,
                  &error)) {
      std::fprintf(stderr, "step: %s\n", error.c_str());
      return 1;
    }
    hidden.emplace_back(ref.LastHidden().begin(), ref.LastHidden().end());
    if (j > 0 &&
        !ref.MtpStep(prompt[j], hidden[j - 1], {}, {}, nullptr, &error)) {
      std::fprintf(stderr, "mtp fill: %s\n", error.c_str());
      return 1;
    }
    if (j % 16 == 15 || last) {
      std::fprintf(stderr, "[%s] prefill %zu/%zu (%.0f s)\n", label.c_str(),
                   j + 1, prompt.size(), seconds());
    }
  }

  // Generation: truth[i] = trunk argmax at position i (the token at i + 1).
  struct Start {
    std::uint32_t position;
    std::vector<std::int32_t> drafts;
  };
  std::vector<std::int32_t> truth(prompt.size() - 1, -1);
  truth.push_back(Argmax(logits));
  std::vector<Start> starts;
  const std::uint32_t total = generate + depth;
  for (std::uint32_t g = 0; g < total; ++g) {
    const auto i = static_cast<std::uint32_t>(prompt.size() - 1 + g);
    const std::int32_t anchor = truth[i];
    if (g < generate) {
      // MTP chain on the state a speculative step would have.
      Start s{i, {}};
      std::int32_t token = anchor;
      std::span<const float> h = hidden[i];
      for (std::uint32_t j = 0; j < depth; ++j) {
        if (!ref.MtpStep(token, h, mtp_logits, h_next, nullptr, &error)) {
          std::fprintf(stderr, "mtp: %s\n", error.c_str());
          return 1;
        }
        token = Argmax(mtp_logits);
        s.drafts.push_back(token);
        h_prev = h_next;
        h = h_prev;
      }
      ref.MtpRewind(i);
      starts.push_back(std::move(s));
    }
    // Commit: the trunk consumes the anchor; the MTP row i is (h_i, anchor).
    if (!ref.MtpStep(anchor, hidden[i], {}, {}, nullptr, &error) ||
        !ref.Step(anchor, logits, &routes, &error)) {
      std::fprintf(stderr, "step: %s\n", error.c_str());
      return 1;
    }
    routes.Commit();
    hidden.emplace_back(ref.LastHidden().begin(), ref.LastHidden().end());
    truth.push_back(Argmax(logits));
    if (g % 8 == 7) {
      std::fprintf(stderr, "[%s] generated %u/%u (%.0f s)\n", label.c_str(),
                   g + 1, total, seconds());
    }
  }

  // Scores. d_j (j = 1..depth) must equal truth[i + j].
  std::vector<std::uint32_t> reach(depth + 1, 0);  // starts with >= j accepted
  for (const Start& s : starts) {
    std::uint32_t accepted = 0;
    for (std::uint32_t j = 0; j < depth; ++j) {
      if (s.drafts[j] != truth[s.position + 1 + j])
        break;
      ++accepted;
    }
    for (std::uint32_t j = 0; j <= accepted; ++j)
      ++reach[j];
  }
  const auto n = static_cast<double>(starts.size());
  std::ostringstream js;
  js.precision(5);
  js << "{\"label\": \"" << label << "\", \"prompt_tokens\": " << prompt.size()
     << ", \"generated\": " << generate << ", \"starts\": " << starts.size()
     << ", \"depth\": " << depth << ", \"conditional_acceptance\": [";
  for (std::uint32_t j = 1; j <= depth; ++j) {
    js << (j > 1 ? ", " : "")
       << (reach[j - 1] ? static_cast<double>(reach[j]) / reach[j - 1] : 0.0);
  }
  js << "], \"chain_survival\": [";
  for (std::uint32_t j = 1; j <= depth; ++j) {
    js << (j > 1 ? ", " : "") << reach[j] / n;
  }
  js << "], \"reach_counts\": [";
  for (std::uint32_t j = 0; j <= depth; ++j)
    js << (j ? ", " : "") << reach[j];
  js << "]";

  // Expert unions of n consecutive generated tokens, per MoE layer (mean).
  js << ", \"expert_union_mean\": {";
  const auto& hist = routes.history();
  for (std::uint32_t w = 1; w <= 8; ++w) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t s = 0; s + w <= hist.size(); ++s) {
      for (std::size_t layer = 0; layer < 40; ++layer) {
        std::set<std::int32_t> u;
        for (std::size_t t = s; t < s + w; ++t) {
          u.insert(hist[t][layer].begin(), hist[t][layer].end());
        }
        sum += static_cast<double>(u.size());
        ++count;
      }
    }
    js << (w > 1 ? ", " : "") << "\"" << w
       << "\": " << (count ? sum / count : 0.0);
  }
  js << "}, \"generated_text\": \"";
  {
    std::vector<l3::TokenId> gen;
    for (std::uint32_t g = 0; g < generate; ++g) {
      gen.push_back(static_cast<l3::TokenId>(truth[prompt.size() - 1 + g]));
    }
    for (const char c : tok->Decode(gen).substr(0, 600)) {
      if (c == '"' || c == '\\')
        js << '\\' << c;
      else if (c == '\n')
        js << "\\n";
      else if (static_cast<unsigned char>(c) < 0x20)
        js << ' ';
      else
        js << c;
    }
  }
  js << "\", \"seconds\": " << seconds() << "}\n";
  std::printf("%s", js.str().c_str());
  if (!out_path.empty()) {
    std::ofstream(out_path) << js.str();
  }
  return 0;
}
