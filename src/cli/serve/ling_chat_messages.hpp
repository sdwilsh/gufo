#ifndef GUFO_SERVER_LING_CHAT_MESSAGES_HPP_
#define GUFO_SERVER_LING_CHAT_MESSAGES_HPP_

#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/models/ling_3_flash/chat_template.hpp"
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

}  // namespace gufo::server

#endif  // GUFO_SERVER_LING_CHAT_MESSAGES_HPP_
