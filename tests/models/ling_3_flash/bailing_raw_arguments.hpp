// The raw text of a Bailing tool call's argument values, for the MTP tests
// that compare it with what the server's Bailing parser returns.
#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_BAILING_RAW_ARGUMENTS_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_BAILING_RAW_ARGUMENTS_HPP_

#include <cstddef>
#include <map>
#include <string>

namespace gufo::tests::ling {

/// Raw values of the call at [begin, end) of `text`, by a sequential scan of
/// its pairs: a raw value ends at the first value closer (it cannot contain
/// one), so key-like text inside an earlier value is skipped.
inline std::map<std::string, std::string> RawArgumentValues(
    const std::string& text, std::size_t begin, std::size_t end) {
  std::map<std::string, std::string> raw;
  for (std::size_t at = begin; at < end;) {
    const auto key = text.find("<arg_key>", at);
    const auto key_end = text.find("</arg_key>", key);
    const auto value = text.find("<arg_value>", key_end);
    const auto close = text.find("</arg_value>", value);
    if (key == std::string::npos || key_end == std::string::npos ||
        value == std::string::npos || close == std::string::npos ||
        close >= end) {
      break;
    }
    raw[text.substr(key + 9, key_end - key - 9)] =
        text.substr(value + 11, close - value - 11);
    at = close + 12;
  }
  return raw;
}

}  // namespace gufo::tests::ling

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_BAILING_RAW_ARGUMENTS_HPP_
