// Which host call on another thread breaks a thread-local HIP graph capture?
//
// The Ling-3.0-Flash executor captures small-batch decode steps into
// per-session graphs on its stream (hipStreamCaptureModeThreadLocal, with a
// shared-expert branch forked to a second stream). Snapshot workers copy
// session state on a private non-blocking transfer stream. This probe replays
// that pattern without the model. The main thread captures a graph of
// `--launches` kernel launches with a fork/join branch, `--captures` times,
// while a worker thread loops one of these host calls:
//
//   none      nothing (baseline)
//   copies    hipMemcpyAsync D2H into pinned memory on a non-blocking transfer
//             stream + hipStreamSynchronize(transfer), as SaveSnapshot copies
//   event     hipEventSynchronize on an event recorded on the capturing stream
//             (before the capture), as SaveSnapshot awaits forward_done_
//   event-locked  the same wait, under a mutex that the capture also holds
//             (Executor::capture_mutex_)
//   event-once    one unguarded wait in the middle of each capture and
//             nothing else, which models a single snapshot
//
// It prints one JSON line per mode: captures whose EndCapture failed (and with
// which error), and worker calls that returned an error. Results on the record
// (Radeon 860M, gfx1152 running gfx1151 code via HSA_OVERRIDE_GFX_VERSION=
// 11.5.1, ROCm 7.1, 3000 captures x 200 launches per mode):
//
//   mode          broken captures  worker calls  worker errors
//   none          0 / 3000         0             0
//   copies        0 / 3000         20,225        0
//   event         3000 / 3000      1,549,084     1,549,084
//   event-locked  0 / 3000         960,403       0
//
// Every broken capture ended with hipErrorStreamCaptureInvalidated; the
// unguarded waits failed with hipErrorCapturedEvent exactly once per broken
// capture (3000) and with hipErrorStreamCaptureUnsupported while the broken
// capture stayed open (1,546,084): the wait that breaks a capture fails itself,
// so in the server that snapshot is skipped as a capture failure.
//
// Build: tools/bench/build.sh tools/models/ling_3_flash/capture_probe.hip
// Run:   /tmp/capture_probe [--captures N] [--launches N] [--modes a,b,...]
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

__global__ void Touch(float* data, int n, float add) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    data[i] = data[i] * 0.999F + add;
  }
}

bool Ok(hipError_t e, const char* what) {
  if (e == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "capture_probe: %s: %s\n", what, hipGetErrorString(e));
  return false;
}

struct Result {
  int captures = 0;
  int broken = 0;
  std::map<std::string, int> end_errors;
  int stray_branch = 0;  ///< broken captures whose branch ran outside a graph
  int still_capturing = 0;   ///< `stream` still capturing after EndCapture
  int stream_unusable = 0;   ///< broken captures after which `stream` failed
  int recovered = 0;         ///< ... and worked after a second EndCapture
  int recovered_empty = 0;   ///< ... or after an empty Begin/EndCapture
  int branch_capturing = 0;  ///< branch stream capturing after all that
  long worker_calls = 0;
  long worker_errors = 0;
  std::map<std::string, long> worker_error_names;
};

Result RunMode(const std::string& mode, int captures, int launches) {
  constexpr int kN = 1 << 16;
  Result r;
  hipStream_t stream = nullptr;
  hipStream_t branch = nullptr;
  hipStream_t transfer = nullptr;
  hipEvent_t fork = nullptr;
  hipEvent_t join = nullptr;
  hipEvent_t forward_done = nullptr;
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
  void* pinned = nullptr;
  if (!Ok(hipStreamCreate(&stream), "stream") ||
      !Ok(hipStreamCreateWithFlags(&branch, hipStreamNonBlocking), "branch") ||
      !Ok(hipStreamCreateWithFlags(&transfer, hipStreamNonBlocking),
          "transfer") ||
      !Ok(hipEventCreateWithFlags(&fork, hipEventDisableTiming), "fork") ||
      !Ok(hipEventCreateWithFlags(&join, hipEventDisableTiming), "join") ||
      !Ok(hipEventCreateWithFlags(&forward_done, hipEventDisableTiming),
          "forward_done") ||
      !Ok(hipMalloc(&a, kN * sizeof(float)), "a") ||
      !Ok(hipMalloc(&b, kN * sizeof(float)), "b") ||
      !Ok(hipMalloc(&c, kN * sizeof(float)), "c") ||
      !Ok(hipHostMalloc(&pinned, kN * sizeof(float)), "pinned")) {
    std::exit(1);
  }
  (void)hipMemset(a, 0, kN * sizeof(float));
  (void)hipMemset(b, 0, kN * sizeof(float));
  (void)hipMemset(c, 0, kN * sizeof(float));
  (void)hipDeviceSynchronize();
  // An eager "forward" on the stream, its completion event recorded, as the
  // executor records forward_done_ at the end of every forward.
  Touch<<<kN / 256, 256, 0, stream>>>(a, kN, 1.0F);
  (void)hipEventRecord(forward_done, stream);
  (void)hipStreamSynchronize(stream);

  std::mutex capture_mutex;
  std::atomic<bool> stop{false};
  std::atomic<int> once{0};  // event-once: 1 = wait now, 2 = waited
  std::thread worker([&] {
    if (mode == "none") {
      return;
    }
    while (!stop.load(std::memory_order_relaxed)) {
      hipError_t e = hipSuccess;
      if (mode == "event-once") {
        if (once.load() != 1) {
          continue;
        }
        e = hipEventSynchronize(forward_done);
        once = 2;
      } else if (mode == "copies") {
        e = hipMemcpyAsync(pinned, c, kN * sizeof(float), hipMemcpyDeviceToHost,
                           transfer);
        if (e == hipSuccess) {
          e = hipStreamSynchronize(transfer);
        }
      } else if (mode == "event") {
        e = hipEventSynchronize(forward_done);
      } else if (mode == "event-locked") {
        const std::lock_guard<std::mutex> lock(capture_mutex);
        e = hipEventSynchronize(forward_done);
      }
      ++r.worker_calls;
      if (e != hipSuccess) {
        ++r.worker_errors;
        ++r.worker_error_names[hipGetErrorName(e)];
        (void)hipGetLastError();
      }
    }
  });

  for (int i = 0; i < captures; ++i) {
    std::unique_lock<std::mutex> lock(capture_mutex, std::defer_lock);
    if (mode == "event-locked") {
      lock.lock();
    }
    hipGraph_t graph = nullptr;
    // Branch output marker: nonzero after the capture only if branch launches
    // executed (a captured graph is never launched here).
    (void)hipMemsetAsync(b, 0, sizeof(float), branch);
    (void)hipStreamSynchronize(branch);
    if (!Ok(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
            "begin capture")) {
      std::exit(1);
    }
    for (int k = 0; k < launches; ++k) {
      if (mode == "event-once" && k == launches / 2) {
        // Exactly one wait from the worker, in the middle of the capture.
        once = 1;
        while (once.load() != 2) {
        }
        once = 0;
      }
      Touch<<<kN / 256, 256, 0, stream>>>(a, kN, 0.5F);
      if (k % 20 == 10) {
        // The shared-expert branch: fork, work on the branch, join.
        (void)hipEventRecord(fork, stream);
        (void)hipStreamWaitEvent(branch, fork, 0);
        Touch<<<kN / 256, 256, 0, branch>>>(b, kN, 0.25F);
        (void)hipEventRecord(join, branch);
        (void)hipStreamWaitEvent(stream, join, 0);
      }
    }
    const hipError_t end = hipStreamEndCapture(stream, &graph);
    if (lock.owns_lock()) {
      lock.unlock();
    }
    ++r.captures;
    if (end != hipSuccess || graph == nullptr) {
      ++r.broken;
      ++r.end_errors[hipGetErrorName(end)];
      // Drain anything a broken capture left running on the branch stream.
      (void)hipStreamSynchronize(branch);
      (void)hipGetLastError();
      float marker = 0.0F;
      if (hipMemcpyAsync(&marker, b, sizeof(float), hipMemcpyDeviceToHost,
                         branch) == hipSuccess &&
          hipStreamSynchronize(branch) == hipSuccess && marker != 0.0F) {
        ++r.stray_branch;
      }
      hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
      if (hipStreamIsCapturing(stream, &status) == hipSuccess &&
          status != hipStreamCaptureStatusNone) {
        ++r.still_capturing;
      }
      (void)hipGetLastError();
      // An eager step on the capture stream, as the executor's fallback.
      const auto eager = [&] {
        hipError_t e = hipMemsetAsync(c, 0, sizeof(float), stream);
        if (e == hipSuccess) {
          e = hipStreamSynchronize(stream);
        }
        (void)hipGetLastError();
        return e == hipSuccess;
      };
      if (!eager()) {
        ++r.stream_unusable;
        hipGraph_t left = nullptr;
        (void)hipStreamEndCapture(stream, &left);
        if (left != nullptr) {
          (void)hipGraphDestroy(left);
        }
        (void)hipGetLastError();
        if (eager()) {
          ++r.recovered;
        } else {
          hipGraph_t empty = nullptr;
          if (hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) ==
              hipSuccess) {
            (void)hipStreamEndCapture(stream, &empty);
          }
          if (empty != nullptr) {
            (void)hipGraphDestroy(empty);
          }
          (void)hipGetLastError();
          if (eager()) {
            ++r.recovered_empty;
          }
        }
      }
      if (hipStreamIsCapturing(branch, &status) == hipSuccess &&
          status != hipStreamCaptureStatusNone) {
        ++r.branch_capturing;
      }
    }
    if (graph != nullptr) {
      (void)hipGraphDestroy(graph);
    }
  }
  stop = true;
  worker.join();
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
  (void)hipHostFree(pinned);
  (void)hipFree(a);
  (void)hipFree(b);
  (void)hipFree(c);
  (void)hipEventDestroy(fork);
  (void)hipEventDestroy(join);
  (void)hipEventDestroy(forward_done);
  (void)hipStreamDestroy(stream);
  (void)hipStreamDestroy(branch);
  (void)hipStreamDestroy(transfer);
  return r;
}

std::string Names(const auto& counts) {
  std::string s = "{";
  bool first = true;
  for (const auto& [name, n] : counts) {
    s += (first ? "\"" : ",\"") + name + "\":" + std::to_string(n);
    first = false;
  }
  return s + "}";
}

}  // namespace

int main(int argc, char** argv) {
  int captures = 3000;
  int launches = 200;
  std::string modes = "none,copies,event,event-locked,event-once";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--captures") == 0) {
      captures = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--launches") == 0) {
      launches = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--modes") == 0) {
      modes = argv[i + 1];
    }
  }
  hipDeviceProp_t prop{};
  (void)hipGetDeviceProperties(&prop, 0);
  int status = 0;
  for (std::size_t at = 0; at <= modes.size();) {
    const std::size_t comma = std::min(modes.find(',', at), modes.size());
    const std::string mode = modes.substr(at, comma - at);
    at = comma + 1;
    if (mode.empty()) {
      continue;
    }
    const Result r = RunMode(mode, captures, launches);
    std::printf(
        "{\"tool\":\"capture_probe\",\"gpu\":\"%s\",\"mode\":\"%s\","
        "\"captures\":%d,\"launches\":%d,\"broken\":%d,\"end_errors\":%s,"
        "\"stray_branch\":%d,\"still_capturing\":%d,"
        "\"stream_unusable\":%d,\"recovered\":%d,\"recovered_empty\":%d,"
        "\"branch_capturing\":%d,"
        "\"worker_calls\":%ld,\"worker_errors\":%ld,\"worker_error_names\":%s}"
        "\n",
        prop.gcnArchName, mode.c_str(), r.captures, launches, r.broken,
        Names(r.end_errors).c_str(), r.stray_branch, r.still_capturing,
        r.stream_unusable, r.recovered, r.recovered_empty, r.branch_capturing,
        r.worker_calls, r.worker_errors, Names(r.worker_error_names).c_str());
    std::fflush(stdout);
    // The executor's contract: only an unguarded event wait may break it.
    if (mode.rfind("event", 0) != 0 && r.broken != 0) {
      status = 1;
    }
  }
  return status;
}
