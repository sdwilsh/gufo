#ifndef GUFO_SERVER_LING_CHAT_MESSAGES_HPP_
#define GUFO_SERVER_LING_CHAT_MESSAGES_HPP_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/ling_3_flash/chat_template.hpp"
#include "src/models/ling_3_flash/layout.hpp"
#include "src/models/ling_3_flash/tokenizer.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::server {

/// Converts server chat messages to Bailing V3 template messages. The Bailing
/// template has no developer role. Server-authored framing (response_format
/// instructions) follows the message content like the Qwen and DeepSeek
/// templates render it; as message data it is never promoted to control
/// tokens.
[[nodiscard]] inline std::vector<models::ling_3_flash::ChatMessage>
ToLingMessages(const std::vector<tokenization::ChatMessage>& messages) {
  namespace ling = models::ling_3_flash;
  std::vector<ling::ChatMessage> converted_messages;
  converted_messages.reserve(messages.size());
  for (const auto& message : messages) {
    std::string role;
    switch (message.role) {
      case tokenization::ChatRole::kDeveloper:
      case tokenization::ChatRole::kSystem:
        role = "system";
        break;
      case tokenization::ChatRole::kAssistant:
        role = "assistant";
        break;
      case tokenization::ChatRole::kTool:
        role = "tool";
        break;
      case tokenization::ChatRole::kUser:
        role = "user";
        break;
    }
    ling::ChatMessage converted{
        .role = std::move(role),
        .content = message.content + message.framing_suffix,
        .reasoning_content = message.thought,
        .tool_calls = {}};
    for (const auto& call : message.tool_calls) {
      ling::ChatMessage::ToolCall converted_call{.name = call.name,
                                                 .arguments = {}};
      for (const auto& argument : call.arguments) {
        converted_call.arguments.push_back({.name = argument.name,
                                            .value = argument.value,
                                            .is_string = argument.is_string});
      }
      converted.tool_calls.push_back(std::move(converted_call));
    }
    converted_messages.push_back(std::move(converted));
  }
  return converted_messages;
}

/// Whether the request ends with a text-only user message that an agent
/// client replaces in its next request instead of keeping it (per-turn
/// runtime context). As in the Qwen template (#407), such context follows a
/// tool result or another user message after an assistant reply; prompt
/// caching then keeps its checkpoint before that message.
[[nodiscard]] inline bool ReplacesFinalUserTurn(
    const std::vector<tokenization::ChatMessage>& messages) {
  using tokenization::ChatRole;
  if (messages.size() < 2 || messages.back().role != ChatRole::kUser ||
      !messages.back().images.empty())
    return false;
  const auto previous = messages[messages.size() - 2].role;
  return (previous == ChatRole::kTool || previous == ChatRole::kUser) &&
         std::any_of(messages.begin(), messages.end() - 1,
                     [](const tokenization::ChatMessage& message) {
                       return message.role == ChatRole::kAssistant;
                     });
}

/// One rendered conversation: the prompt text, the byte ranges that are
/// conversation data (never control tokens), and where its final message
/// begins.
struct LingRendering {
  std::string text;
  std::vector<models::ling_3_flash::LingTokenizer::LiteralRange> literal;
  std::size_t stable_prefix_bytes{0};
};

using LingRenderFn =
    std::function<LingRendering(const std::vector<tokenization::ChatMessage>&)>;

/// The prompt-cache checkpoint of a Ling prompt that was rendered and
/// tokenized from `messages` (`prompt`): the last token every continuation
/// shares with it, backed up past a replaceable final user turn (#407) and
/// far enough that a full chunk follows (the small-batch numeric family).
/// A replaced final turn costs a tokenization of the text before it only,
/// not of the whole conversation again.
[[nodiscard]] inline std::size_t LingCacheBoundary(
    const models::ling_3_flash::LingTokenizer& tokenizer,
    const std::vector<tokenization::ChatMessage>& messages,
    const LingRenderFn& render, std::span<const TextRunnerToken> prompt) {
  const auto shared = [&](std::span<const TextRunnerToken> other) {
    return static_cast<std::size_t>(
        std::mismatch(prompt.begin(), prompt.end(), other.begin(), other.end())
            .first -
        prompt.begin());
  };
  const auto encode = [&](const LingRendering& rendering,
                          std::size_t limit) -> std::vector<TextRunnerToken> {
    std::vector<models::ling_3_flash::LingTokenizer::LiteralRange> literal;
    for (const auto& [begin, end] : rendering.literal) {
      if (begin < limit)
        literal.emplace_back(begin, std::min(end, limit));
    }
    const auto ids = tokenizer.Encode(
        std::string_view(rendering.text).substr(0, limit), true, literal);
    return {ids.begin(), ids.end()};
  };

  // The generation prompt ("<role>ASSISTANT</role>" plus the thinking
  // opening) is mutable framing: a client that interrupts a request or drops
  // its assistant turn continues with another message instead. Keep a
  // checkpoint at the last token every continuation shares with the prompt.
  auto continuation = messages;
  continuation.emplace_back(tokenization::ChatRole::kUser, "");
  const auto continued = render(continuation);
  auto stable = shared(encode(continued, continued.text.size()));
  // As for Qwen (#407): agent clients may end every request with a text-only
  // user message (per-turn runtime context) that the next request replaces
  // rather than keeps. Keep the checkpoint before it, where the next
  // request's assistant turn begins. The same request without that message
  // differs from the prompt only from the final message's markup on, so the
  // tokens they share are those that end within the text before the markup
  // plus the opening "<role>" both turns spell.
  if (ReplacesFinalUserTurn(messages)) {
    constexpr std::string_view kRoleOpen = "<role>";
    const auto rendering = render(messages);
    const auto limit =
        std::min(rendering.stable_prefix_bytes + kRoleOpen.size(),
                 rendering.text.size());
    // The prompt's tokens spell its text, so the token boundary at `limit` is
    // found by walking back over the few tokens after it. That costs no
    // tokenization of the text before it; a boundary inside a token (or
    // tokens that do not spell the text) falls back to tokenizing the prefix.
    std::size_t count = prompt.size();
    std::size_t bytes = 0;
    const auto wanted = rendering.text.size() - limit;
    std::string spelled;
    while (bytes < wanted && count > 0) {
      --count;
      auto piece = tokenizer.DecodeToken(
          static_cast<models::ling_3_flash::TokenId>(prompt[count]));
      bytes += piece.size();
      spelled.insert(0, piece);
    }
    const bool exact =
        bytes == wanted &&
        std::string_view(rendering.text).substr(limit) == spelled;
    stable = std::min(stable, exact ? count : shared(encode(rendering, limit)));
  }
  // Numeric families (engine.hpp): the generation prompt is only a few
  // tokens, and a 1..8 token tail after the checkpoint would run in the
  // small-batch family, so the prompt state would differ from an uncached
  // prefill's. Back the checkpoint up until a full chunk follows it.
  const std::size_t tail = models::ling_3_flash::kSmallBatchMaxTokens + 1;
  const std::size_t size = prompt.size();
  if (size - stable < tail)
    stable = size > tail ? size - tail : 0;
  return stable;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_LING_CHAT_MESSAGES_HPP_
