#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/deepseek_v4_flash/runtime/dspark_internal.h"
#include "src/models/deepseek_v4_flash/runtime/model_data_internal.h"
#include "src/models/deepseek_v4_flash/runtime/native_internal.h"

namespace {

using gufo::core::SessionMode;
using gufo::models::deepseek_v4_flash::Model;
using gufo::models::deepseek_v4_flash::ModelOptions;
using gufo::models::deepseek_v4_flash::Session;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

std::vector<std::uint8_t> SnapshotBytes(const Session& session) {
  std::string error;
  const auto snapshot = session.SaveSnapshot(&error);
  Expect(snapshot != nullptr, error);
  return {snapshot->bytes().begin(), snapshot->bytes().end()};
}

std::string Digest(std::span<const std::uint8_t> bytes) {
  return std::to_string(bytes.size()) + ":" +
         gufo::crypto::Sha256Hex(bytes).substr(0, 20);
}

struct PayloadSection {
  std::string name;
  std::size_t begin;
  std::size_t end;
};

std::vector<PayloadSection> PayloadSections(
    std::span<const std::uint8_t> payload) {
  const auto word = [&](std::size_t index) {
    std::uint32_t value = 0;
    std::memcpy(&value, payload.data() + index * sizeof(value), sizeof(value));
    return value;
  };
  std::vector<PayloadSection> sections;
  std::size_t at = 0;
  const auto add = [&](std::string name, std::size_t bytes) {
    sections.push_back({std::move(name), at, at + bytes});
    at += bytes;
  };
  constexpr std::size_t kHeaderWords = 19 + DS4_DSPARK_MAX_TARGET_LAYERS;
  constexpr std::size_t kRequestStateBytes = 12 * 8 + 8 * 4;
  const std::uint32_t tokens = word(7);
  const std::uint32_t raw_rows = word(12);
  const std::uint32_t draft_stages = word(13);
  const std::uint32_t draft_tokens = word(16);
  add("header", kHeaderWords * 4);
  add("dspark_request_state", kRequestStateBytes);
  add("tokens", std::size_t{tokens} * 4);
  add("logits", std::size_t{DS4_N_VOCAB} * 4);
  const std::size_t comp_counts = at / 4;
  add("n_comp", DS4_N_LAYER * 4);
  const std::size_t index_counts = at / 4;
  add("n_index_comp", DS4_N_LAYER * 4);
  for (std::uint32_t il = 0; il < DS4_N_LAYER; ++il) {
    const std::string layer = "L" + std::to_string(il) + ".";
    add(layer + "raw", std::size_t{raw_rows} * DS4_N_HEAD_DIM * 4);
    const std::uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio == 0)
      continue;
    const std::size_t coff = ratio == 4 ? 2 : 1;
    const std::size_t attn_state = coff * DS4_N_HEAD_DIM * coff * ratio * 4;
    add(layer + "attn_comp",
        std::size_t{word(comp_counts + il)} * DS4_N_HEAD_DIM * 4);
    add(layer + "attn_state_kv", attn_state);
    add(layer + "attn_state_score", attn_state);
    if (ratio != 4)
      continue;
    const std::size_t index_state =
        coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * 4;
    add(layer + "index_comp",
        std::size_t{word(index_counts + il)} * DS4_N_INDEXER_HEAD_DIM * 4);
    add(layer + "index_state_kv", index_state);
    add(layer + "index_state_score", index_state);
  }
  if (draft_tokens != 0) {
    for (std::uint32_t stage = 0; stage < draft_stages; ++stage) {
      add("draft" + std::to_string(stage) + ".raw",
          std::size_t{std::min(draft_tokens, DS4_N_SWA)} * DS4_N_HEAD_DIM * 4);
    }
  }
  if (at != payload.size())
    add("unparsed", payload.size() - at);
  return sections;
}

std::string StateDifference(std::span<const std::uint8_t> expected,
                            std::span<const std::uint8_t> actual) {
  if (expected.size() == actual.size() &&
      std::memcmp(expected.data(), actual.data(), expected.size()) == 0)
    return {};
  if (expected.size() != actual.size())
    return "size:" + std::to_string(expected.size()) + "/" +
           std::to_string(actual.size());
  std::string sections;
  for (const PayloadSection& section : PayloadSections(expected)) {
    std::size_t words = 0;
    for (std::size_t at = section.begin; at < section.end; at += 4)
      words += std::memcmp(expected.data() + at, actual.data() + at,
                           std::min<std::size_t>(4, section.end - at)) != 0;
    if (words != 0) {
      sections += (sections.empty() ? "" : ",") + section.name + ":" +
                  std::to_string(words) + "/" +
                  std::to_string((section.end - section.begin + 3) / 4);
    }
  }
  return sections;
}

float MaxLogitError(const Session& a, const Session& b) {
  std::string error;
  const auto x = a.CopyLogits(&error);
  const auto y = b.CopyLogits(&error);
  Expect(!x.empty() && x.size() == y.size(), "logit shape");
  if (std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0)
    return 0.0F;
  float maximum = 0.0F;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float delta = std::abs(x[i] - y[i]);
    maximum = std::max(maximum, std::isnan(delta) ? INFINITY : delta);
  }
  // -0.0F and 0.0F differ in bytes but subtract to zero.
  return maximum == 0.0F ? INFINITY : maximum;
}

std::size_t FreeDeviceBytes() {
  std::size_t free = 0;
  std::size_t total = 0;
  Expect(hipMemGetInfo(&free, &total) == hipSuccess, "hipMemGetInfo");
  return free;
}

constexpr int kGreedyTokens = 128;

std::optional<int> FirstDivergence(Session& reference, Session& candidate,
                                   bool speculative) {
  std::string error;
  int generated = 0;
  while (generated < kGreedyTokens) {
    std::vector<int> expected;
    std::vector<int> actual;
    const auto budget = static_cast<std::size_t>(kGreedyTokens - generated);
    if (speculative) {
      Expect(reference.DsparkStep(budget, 7, &expected, &error), error);
      Expect(candidate.DsparkStep(budget, 7, &actual, &error), error);
    } else {
      expected.push_back(reference.SelectNext(0.0F, nullptr));
      actual.push_back(candidate.SelectNext(0.0F, nullptr));
      Expect(reference.Evaluate(expected[0], &error), error);
      Expect(candidate.Evaluate(actual[0], &error), error);
    }
    Expect(!expected.empty(), "greedy progress");
    if (actual != expected || MaxLogitError(reference, candidate) != 0.0F ||
        !(reference.DsparkStatistics() == candidate.DsparkStatistics())) {
      return generated;
    }
    generated += static_cast<int>(expected.size());
  }
  return std::nullopt;
}

struct SyncedTail {
  float logit_error;
  long long device_bytes;
};

SyncedTail SyncTail(Session& decoded, Session& synced,
                    const std::vector<int>& target, std::size_t tail) {
  std::string error;
  const std::size_t free_before = FreeDeviceBytes();
  for (std::size_t i = target.size() - tail; i < target.size(); ++i)
    Expect(decoded.Evaluate(target[i], &error), error);
  Expect(synced.Sync(target, &error), error);
  const std::size_t free_after = FreeDeviceBytes();
  Expect(synced.Position() == static_cast<int>(target.size()) &&
             decoded.Position() == synced.Position(),
         "tail positions");
  return {
      MaxLogitError(decoded, synced),
      static_cast<long long>(free_before) - static_cast<long long>(free_after)};
}

std::vector<std::uint8_t> WithUnpooledCompressorRowFault(
    std::vector<std::uint8_t> payload, std::size_t position) {
  constexpr std::uint32_t kRatio = 128;
  Expect(position % kRatio != kRatio - 1, "fault row closes its window");
  std::uint32_t layer = 0;
  while (ds4_layer_compress_ratio(layer) != kRatio)
    ++layer;
  const std::string name = "L" + std::to_string(layer) + ".attn_state_kv";
  for (const PayloadSection& section : PayloadSections(payload)) {
    if (section.name != name)
      continue;
    std::uint8_t* row = payload.data() + section.begin +
                        (position % kRatio) * DS4_N_HEAD_DIM * sizeof(float);
    for (std::uint32_t i = 0; i < DS4_N_HEAD_DIM; ++i) {
      float value = 0.0F;
      std::memcpy(&value, row + i * sizeof(value), sizeof(value));
      value += 1.0F;
      std::memcpy(row + i * sizeof(value), &value, sizeof(value));
    }
    return payload;
  }
  Expect(false, "no " + name + " in the snapshot");
  return payload;
}

std::string Divergence(const std::optional<int>& step) {
  return step ? "diverged@" + std::to_string(*step) : "equal";
}

std::string StateLabel(const std::string& difference) {
  return difference.empty() ? "equal" : "DIFFERENT[" + difference + "]";
}

bool CheckHttpPromptTail(const std::shared_ptr<Model>& model, SessionMode mode,
                         const std::vector<int>& prompt) {
  const bool speculative = mode == SessionMode::kSpeculative;
  const std::size_t resume_minimum = ds4_rocm_graph_resume_prefill_min_tokens();
  const std::size_t longest_tail = resume_minimum - 1;
  const auto context =
      static_cast<std::uint32_t>((prompt.size() + 256 + 255) / 256 * 256);
  std::string error;
  auto decoded = model->CreateSession(mode, context, &error);
  auto synced = model->CreateSession(mode, context, &error);
  Expect(decoded && synced, error);
  const std::vector<int> prefix(prompt.begin(), prompt.end() - longest_tail);
  Expect(decoded->Sync(prefix, &error), error);
  Expect(synced->Sync(prefix, &error), error);
  const bool same_prefill = SnapshotBytes(*synced) == SnapshotBytes(*decoded);
  const auto snapshot = decoded->SaveSnapshot(&error);
  Expect(snapshot != nullptr, error);
  const auto restore = [&] {
    Expect(decoded->RestoreSnapshot(*snapshot, &error) &&
               synced->RestoreSnapshot(*snapshot, &error),
           error);
  };
  const auto label = [&](const std::string& tail, float logit_error) {
    std::cout << "mode=" << (speculative ? "dspark" : "ar")
              << " depth=" << prompt.size()
              << " prefill=" << (same_prefill ? "deterministic" : "DIFFERENT")
              << " tail=" << tail << " logits_max_error=" << logit_error;
  };

  bool ok = true;
  for (std::size_t tail = longest_tail; tail >= 2; --tail) {
    restore();
    for (std::size_t i = prefix.size(); i < prompt.size() - tail; ++i) {
      Expect(decoded->Evaluate(prompt[i], &error), error);
      Expect(synced->Evaluate(prompt[i], &error), error);
    }
    const SyncedTail result = SyncTail(*decoded, *synced, prompt, tail);
    label(std::to_string(tail), result.logit_error);
    const auto expected = SnapshotBytes(*decoded);
    const std::string state = StateDifference(expected, SnapshotBytes(*synced));
    const auto diverged = FirstDivergence(*decoded, *synced, speculative);
    const std::string final_state =
        StateDifference(SnapshotBytes(*decoded), SnapshotBytes(*synced));
    std::cout << " sync_device_bytes=" << result.device_bytes
              << " state=" << StateLabel(state)
              << " state_digest=" << Digest(expected) << " greedy"
              << kGreedyTokens << '=' << Divergence(diverged)
              << " final_state=" << StateLabel(final_state) << '\n';
    ok = ok && result.logit_error == 0.0F && state.empty() && !diverged &&
         final_state.empty();
  }

  restore();
  for (std::size_t i = prefix.size(); i < prompt.size(); ++i) {
    Expect(decoded->Evaluate(prompt[i], &error), error);
    Expect(synced->Evaluate(prompt[i], &error), error);
  }
  Expect(
      synced->RestoreSnapshot(WithUnpooledCompressorRowFault(
                                  SnapshotBytes(*decoded), prompt.size() - 1),
                              &error),
      error);
  const float fault_logit_error = MaxLogitError(*decoded, *synced);
  label("fault", fault_logit_error);
  const std::string fault_state =
      StateDifference(SnapshotBytes(*decoded), SnapshotBytes(*synced));
  const auto fault_diverged = FirstDivergence(*decoded, *synced, speculative);
  const bool fault_caught =
      fault_logit_error == 0.0F && !fault_state.empty() && fault_diverged;
  std::cout << " state=" << StateLabel(fault_state) << " greedy"
            << kGreedyTokens << '=' << Divergence(fault_diverged)
            << " state_fault=" << (fault_caught ? "separates" : "blind")
            << '\n';

  restore();
  auto extended = prompt;
  extended.push_back(prompt[prompt.size() - resume_minimum]);
  const SyncedTail control =
      SyncTail(*decoded, *synced, extended, resume_minimum);
  label(std::to_string(resume_minimum), control.logit_error);
  std::cout << " control="
            << (control.logit_error != 0.0F ? "separates" : "blind") << '\n';
  return ok && fault_caught;
}

}  // namespace

int main() try {
  std::cout << std::unitbuf;
  const char* model_path = std::getenv("GUFO_DEEPSEEK_V4_FLASH_MODEL");
  if (model_path == nullptr || model_path[0] == '\0') {
    std::cout << "SKIP: GUFO_DEEPSEEK_V4_FLASH_MODEL is not set\n";
    return 77;
  }
  const char* dspark_path = std::getenv("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
  const bool dspark = dspark_path != nullptr && dspark_path[0] != '\0';
  std::string error;
  const auto model =
      Model::Load(model_path,
                  ModelOptions{.max_context = 262144,
                               .dspark_model_path = dspark ? dspark_path : ""},
                  &error);
  Expect(model != nullptr, error);

  std::ifstream file(GUFO_DS4_BOUNDARY_PROMPT);
  Expect(file.good(), "open boundary prompt");
  const std::string text{std::istreambuf_iterator<char>(file), {}};
  for (const int depth : {2048, 32768, 131072}) {
    std::string body = text;
    for (int tokens = 2048; tokens < depth; tokens += 2048)
      body += "\n\n" + text;
    const std::vector<int> prompt = model->EncodeChat("", body);
    bool ok = CheckHttpPromptTail(model, SessionMode::kAutoregressive, prompt);
    if (dspark)
      ok = CheckHttpPromptTail(model, SessionMode::kSpeculative, prompt) && ok;
    Expect(ok, "prefill tail matches one-token decodes at depth " +
                   std::to_string(prompt.size()));
  }
  if (!dspark) {
    std::cout << "SKIP: GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL is not set; "
                 "the DSpark tail was not checked\n";
    return 77;
  }
  std::cout << "DeepSeek V4 Flash prefill tail test passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return EXIT_FAILURE;
}
