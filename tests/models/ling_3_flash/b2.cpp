// Runs all Ling-3.0-Flash gate evidence from one model load: layer_check,
// session_test, and logit_dump (greedy and KL) share the resident weights.
// Each sub-tool reports under its own key of the single stdout JSON.
//
//   ling_3_flash_b2 --model FIRST_SHARD.gguf
//       [--reference-dir DIR]            layer_check compares against the CPU
//                                        reference in DIR
//       [--greedy greedy.jsonl] [--kl-base kl_base.bin] [--kl-chunks N]
//       [--skip layer_check,session_test,logit_dump]
//       [--max-batch 512] [--max-context 1024] [--layers N] [--layer-first M]
//       [--timeout-seconds S] [--hold-seconds S]
//
// Exit 77 without --model; 1 when any sub-tool fails its gate.

#define LING_TOOL_NO_MAIN
#include "tests/models/ling_3_flash/layer_check.cpp"
#include "tests/models/ling_3_flash/logit_dump.cpp"
#include "tests/models/ling_3_flash/session_test.cpp"

namespace {

using namespace gufo::tests::ling;

int RunAll(const Args& args, Report& report) {
  auto model = LoadModel(args, report, 512, 1024);
  if (!model) {
    return 1;
  }
  const std::string skip = "," + args.Str("--skip") + ",";
  const auto wanted = [&](const char* name) {
    return skip.find(std::string(",") + name + ",") == std::string::npos;
  };
  int worst = 0;
  const auto run = [&](const char* name, auto&& fn) {
    if (!wanted(name)) {
      return;
    }
    std::fprintf(stderr, "[b2] ===== %s =====\n", name);
    const double t0 = Now();
    Report child(&report, name);
    int code = 1;
    try {
      code = fn(child);
    } catch (const std::exception& e) {
      child.Set("status", "error");
      child.Set("error", std::string(e.what()));
    }
    child.Set("seconds", Round(Now() - t0, 5));
    std::fprintf(stderr, "[b2] %s finished: exit %d in %.1f s\n", name, code,
                 Now() - t0);
    worst = std::max(worst, code);
  };
  // Run the cheapest evidence that localizes a failure first, and the long KL
  // pass last.
  run("layer_check",
      [&](Report& r) { return layer_tool::Run(args, r, model); });
  run("session_test",
      [&](Report& r) { return session_tool::Run(args, r, model); });
  if (args.Has("--greedy") || args.Has("--kl-base") ||
      args.Has("--tokens-file")) {
    run("logit_dump",
        [&](Report& r) { return logit_tool::Run(args, r, model); });
  }
  report.Set("status", worst == 0 ? "pass" : "fail");
  return worst;
}

}  // namespace

int main(int argc, char** argv) {
  return gufo::tests::ling::RunTool(argc, argv, "ling_3_flash_b2", RunAll);
}
