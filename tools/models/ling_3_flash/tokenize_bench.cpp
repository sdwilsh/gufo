// CPU-only timing of Ling-3.0-Flash's host-side request path at agent
// context depths: JSON body parse, ToLingMessages, RenderChat, tokenization,
// and the prompt-cache boundary (src/cli/serve/ling_chat_messages.hpp), as
// LingTextRunner::PreparePrompt runs them. The tool synthesizes the
// conversations. They are tool-call heavy: code, JSON, logs, and optional
// prose from --doc.
//
//   ling_3_flash_tokenize_bench --vocab <ling-vocab.gguf|model.gguf>
//       [--doc document.txt] [--depths 16000,64000,128000,200000] [--reps 5]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "src/cli/serve/ling_chat_messages.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/ling_3_flash/chat_template.hpp"
#include "src/models/ling_3_flash/tokenizer.hpp"

namespace ling = gufo::models::ling_3_flash;
namespace tok = gufo::tokenization;
using gufo::server::LingRendering;
using Clock = std::chrono::steady_clock;

namespace {

double Ms(Clock::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

struct Rng {
  std::uint64_t s;
  std::uint64_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  std::size_t Below(std::size_t n) { return Next() % n; }
};

std::string CodeBlock(Rng& rng, std::size_t lines) {
  static const char* kFns[] = {"Parse", "Render", "Encode", "Flush", "Admit"};
  std::ostringstream out;
  for (std::size_t i = 0; i < lines; ++i) {
    const auto n = rng.Below(1000);
    switch (rng.Below(5)) {
      case 0:
        out << "    const auto value_" << n << " = " << kFns[rng.Below(5)]
            << "(request.messages[" << rng.Below(40) << "], options);\n";
        break;
      case 1:
        out << "    if (!result_" << n << ".has_value()) { return "
            << "std::nullopt; }  // handles caf\xC3\xA9 \xE2\x80\x94 case\n";
        break;
      case 2:
        out << "  {\"id\": " << n << ", \"path\": \"src/cli/serve/file_"
            << rng.Below(90) << ".cpp\", \"ok\": true, \"ms\": "
            << rng.Below(900) << ".5},\n";
        break;
      case 3:
        out << "2026-10-10T12:" << 10 + rng.Below(49) << ":" << 10 + rng.Below(49)
            << "Z INFO  worker-" << rng.Below(16) << " served /v1/chat "
            << rng.Below(5000) << " tokens in " << rng.Below(900) << " ms\n";
        break;
      default:
        out << "\t\t// " << n << ": <|role_end|> and <think> are data here;   "
            << "spaces   and\ttabs\r\n\n";
        break;
    }
  }
  return out.str();
}

struct Corpus {
  std::string doc;
  std::string Prose(Rng& rng, std::size_t bytes) const {
    if (doc.size() <= bytes)
      return doc;
    return doc.substr(rng.Below(doc.size() - bytes), bytes);
  }
};

std::vector<tok::ChatMessage> BuildConversation(
    const Corpus& corpus,
    std::size_t target_tokens, bool replaced_final_user) {
  Rng rng{0x9E3779B97F4A7C15ULL ^ target_tokens};
  std::vector<tok::ChatMessage> messages;
  messages.emplace_back(tok::ChatRole::kSystem,
                        "You are a coding agent. Use the tools provided.");
  messages.emplace_back(tok::ChatRole::kUser,
                        "Fix the flaky scheduler test and explain the cause.");
  std::size_t bytes = 0;
  // About 4 bytes per token for code and prose. The output reports the actual
  // token count.
  const auto budget = target_tokens * 4;
  std::size_t turn = 0;
  while (bytes < budget) {
    tok::ChatMessage assistant(tok::ChatRole::kAssistant,
                               "I will look at the file first.");
    assistant.thought = "Need to inspect the code before editing it.";
    tok::ChatMessage::ToolCall call;
    call.id = "call_" + std::to_string(turn);
    call.name = turn % 3 == 0 ? "read_file" : turn % 3 == 1 ? "bash" : "grep";
    call.arguments.push_back(
        {"path", "src/cli/serve/file_" + std::to_string(turn) + ".cpp", true});
    call.arguments.push_back({"limit", std::to_string(100 + turn), false});
    assistant.tool_calls.push_back(std::move(call));
    messages.push_back(std::move(assistant));

    const std::size_t size = 1500 + rng.Below(14000);
    std::string result = rng.Below(4) == 0 && !corpus.doc.empty()
                             ? corpus.Prose(rng, size)
                             : CodeBlock(rng, size / 60);
    bytes += result.size() + 300;
    tok::ChatMessage tool(tok::ChatRole::kTool, std::move(result));
    tool.tool_call_id = "call_" + std::to_string(turn);
    tool.name = "tool";
    messages.push_back(std::move(tool));
    ++turn;
  }
  if (replaced_final_user) {
    messages.emplace_back(tok::ChatRole::kUser,
                          "<system-reminder>Today is Saturday.</system-reminder>");
  }
  return messages;
}

std::vector<ling::ChatTool> Tools() {
  std::vector<ling::ChatTool> tools;
  for (const char* name : {"read_file", "bash", "grep", "edit", "glob",
                           "write_file", "web_fetch", "todo"}) {
    tools.push_back(
        {.definition_json =
             std::string("{\"type\":\"function\",\"function\":{\"name\":\"") +
             name +
             "\",\"description\":\"Does the " + name +
             " thing with many words of documentation.\",\"parameters\":"
             "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":"
             "\"string\"},\"limit\":{\"type\":\"integer\"}},\"required\":"
             "[\"path\"]}}}"});
  }
  return tools;
}

std::string OpenAiBody(const std::vector<tok::ChatMessage>& messages) {
  using gufo::json::Value;
  auto body = Value::object();
  body["model"] = "ling-3.0-flash";
  auto array = Value::array();
  for (const auto& m : messages) {
    auto item = Value::object();
    item["role"] = m.role == tok::ChatRole::kSystem      ? "system"
                   : m.role == tok::ChatRole::kAssistant ? "assistant"
                   : m.role == tok::ChatRole::kTool      ? "tool"
                                                         : "user";
    item["content"] = m.content;
    array.push_back(std::move(item));
  }
  body["messages"] = std::move(array);
  return body.dump();
}

}  // namespace

int main(int argc, char** argv) {
  std::string vocab, doc_path, depths = "16000,64000,128000,200000";
  int reps = 5;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--vocab")
      vocab = argv[i + 1];
    else if (k == "--doc")
      doc_path = argv[i + 1];
    else if (k == "--depths")
      depths = argv[i + 1];
    else if (k == "--reps")
      reps = std::atoi(argv[i + 1]);
  }
  if (vocab.empty()) {
    std::cerr << "usage: --vocab <gguf> [--doc f] [--depths a,b] [--reps n]\n";
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(vocab, &error);
  if (!reader) {
    std::cerr << error << '\n';
    return 1;
  }
  auto tokenizer = ling::LingTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::cerr << error << '\n';
    return 1;
  }
  Corpus corpus;
  if (!doc_path.empty()) {
    std::ifstream in(doc_path, std::ios::binary);
    corpus.doc.assign(std::istreambuf_iterator<char>(in), {});
  }
  const auto tools = Tools();

  constexpr std::size_t kDefaultBudget = std::size_t{64} << 20;
  const auto render = [&](const std::vector<tok::ChatMessage>& m) {
    LingRendering r;
    ling::ChatTemplateOptions options;
    options.literal_spans = &r.literal;
    options.stable_prefix_bytes = &r.stable_prefix_bytes;
    r.text = ling::RenderChat(gufo::server::ToLingMessages(m), tools, options);
    return r;
  };

  struct Timing {
    double json = 0, to_ling = 0, render = 0, encode = 0, boundary = 0,
           boundary_render = 0, count = 0;
    int boundary_renders = 0;
    std::size_t tokens = 0, bytes = 0, boundary_tokens = 0;
    std::uint64_t digest = 0xCBF29CE484222325ULL;  // FNV-1a of the ids
    double Prepare() const { return render + encode + boundary; }
  };
  // One request as LingTextRunner::PreparePrompt runs it (render, tokenize,
  // LingCacheBoundary) plus the OpenAI body parse and message conversion. The
  // tokenizer's cache state is whatever the caller left.
  const auto request = [&](const std::vector<tok::ChatMessage>& messages,
                           const std::string& body) {
    Timing t;
    auto start = Clock::now();
    const auto v = gufo::json::parse(body);
    if (!v.is_object())
      std::abort();
    t.json = Ms(Clock::now() - start);
    start = Clock::now();
    (void)gufo::server::ToLingMessages(messages);
    t.to_ling = Ms(Clock::now() - start);
    start = Clock::now();
    const auto rendering = render(messages);
    t.render = Ms(Clock::now() - start);
    start = Clock::now();
    const auto ids = tokenizer->Encode(rendering.text, true, rendering.literal);
    t.encode = Ms(Clock::now() - start);
    t.tokens = ids.size();
    for (const auto id : ids)
      t.digest = (t.digest ^ id) * 0x100000001B3ULL;
    t.bytes = rendering.text.size();
    const std::vector<gufo::server::TextRunnerToken> prompt(ids.begin(),
                                                            ids.end());
    start = Clock::now();
    const auto boundary = gufo::server::LingCacheBoundary(
        *tokenizer, messages,
        [&](const std::vector<tok::ChatMessage>& m) {
          const auto begin = Clock::now();
          auto r = render(m);
          t.boundary_render += Ms(Clock::now() - begin);
          ++t.boundary_renders;
          return r;
        },
        prompt);
    t.boundary = Ms(Clock::now() - start);
    t.boundary_tokens = boundary;
    return t;
  };
  const auto median = [&](auto&& run) {
    std::vector<Timing> all;
    for (int i = 0; i < reps; ++i)
      all.push_back(run());
    std::sort(all.begin(), all.end(), [](const Timing& a, const Timing& b) {
      return a.Prepare() < b.Prepare();
    });
    return all[all.size() / 2];
  };
  const auto reset = [&](std::size_t budget) {
    tokenizer->SetFragmentCacheBudget(0);  // drops every entry
    tokenizer->SetFragmentCacheBudget(budget);
  };

  std::cout << "scenario,variant,target,tokens,bytes,json,to_ling,render,"
               "encode,boundary,boundary_render,boundary_renders,prepare,boundary_tokens,digest\n";
  const auto print = [&](const char* scenario, bool replaced,
                         std::size_t target, const Timing& t) {
    std::cout << scenario << ',' << (replaced ? "replaced-user" : "tool-tail")
              << ',' << target << ',' << t.tokens << ',' << t.bytes << ','
              << t.json << ',' << t.to_ling << ',' << t.render << ','
              << t.encode << ',' << t.boundary << ',' << t.boundary_render
              << ',' << t.boundary_renders << ',' << t.Prepare() << ',' << t.boundary_tokens
              << ',' << std::hex << t.digest << std::dec << std::endl;
  };
  std::stringstream list(depths);
  std::string item;
  while (std::getline(list, item, ',')) {
    const auto target = static_cast<std::size_t>(std::stoull(item));
    for (const bool replaced : {false, true}) {
      const auto messages =
          BuildConversation(corpus, target, replaced);
      // The next turn: the same history plus one tool exchange (and the
      // per-turn trailing user message again when this one has it).
      auto next = messages;
      if (replaced)
        next.pop_back();
      Rng rng{target};
      tok::ChatMessage assistant(tok::ChatRole::kAssistant, "Next step.");
      tok::ChatMessage::ToolCall call;
      call.id = "call_next";
      call.name = "bash";
      call.arguments.push_back({"command", "ls -la", true});
      assistant.tool_calls.push_back(std::move(call));
      next.push_back(std::move(assistant));
      tok::ChatMessage tool(tok::ChatRole::kTool, CodeBlock(rng, 150));
      tool.tool_call_id = "call_next";
      next.push_back(std::move(tool));
      if (replaced)
        next.push_back(messages.back());
      const auto body = OpenAiBody(messages);
      const auto next_body = OpenAiBody(next);

      reset(0);  // the tokenizer as it was: nothing remembered
      print("uncached", replaced, target,
            median([&] { return request(messages, body); }));
      print("uncached-next-turn", replaced, target,
            median([&] { return request(next, next_body); }));
      print("cache-cold", replaced, target, median([&] {
              reset(kDefaultBudget);
              return request(messages, body);
            }));
      print("cache-next-turn", replaced, target, median([&] {
              reset(kDefaultBudget);
              (void)request(messages, body);  // the previous turn
              return request(next, next_body);
            }));
      reset(kDefaultBudget);
      (void)request(messages, body);
      print("cache-repeat", replaced, target,
            median([&] { return request(messages, body); }));
    }
  }
  return 0;
}
