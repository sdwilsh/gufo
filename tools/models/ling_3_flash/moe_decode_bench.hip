// Ling-3.0-flash routed-expert decode microbenchmark: the gate/up and down
// GEMVs of one MoE layer (ExpertGateUpDecodePlanned / ExpertDownDecodePlanned,
// the launches the decode graph replays) over full-size stacked expert tensors
// (512 x 768 x 2560, hipMalloc like the engine's weights) for 1..8 tokens.
// Every timed call uses its own routing (distinct experts drawn per call, so
// the 32 MiB MALL does not hold the working set), with the per-step expert
// union the real model shows for verify widths 1..8 (8, 13, 17, 21, 25, 28,
// 31, 34; `--routing disjoint` gives 8 n, `shared` 8).
//
// `route` adds one line per n for the router's top-8 selection (RouteDecode,
// with its expert plan) and one for the prefill Route of 512 tokens: device
// ms per call and a digest of ids, weights and plan.
// Prints one JSON line per (type pair, n): device ms per call for gate/up,
// down and the pair launched back to back (min / median over repetitions of
// a HIP graph of `--calls` calls, as the decode graphs replay them), the GB/s of the distinct experts' bytes, and FNV-1a
// digests of every call's outputs, so two builds of the kernel source can be
// compared bit for bit and for speed:
//
//   hipcc -O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151 \
//     -std=c++20 -I. tools/models/ling_3_flash/moe_decode_bench.hip -o bench
//   bench [--types route,iq4,q5] [--ns 1,2,...,8] [--calls 64] [--reps 15]
//         [--routing real|disjoint|shared]
//
// The kernel source is included directly (GUFO_LING_MOE_SOURCE overrides the
// path, e.g. a saved copy of the parent); compile with the production flags.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef GUFO_LING_MOE_SOURCE
#define GUFO_LING_MOE_SOURCE "src/models/ling_3_flash/kernels/rocm/moe.hip.cpp"
#endif
#include GUFO_LING_MOE_SOURCE

namespace {

namespace moe = gufo::models::ling_3_flash::rocm::moe;
using gufo::models::ling_3_flash::rocm::WeightType;

constexpr std::uint32_t kE = 512, kH = 2560, kF = 768, kK = 8;

void Ok(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

__device__ std::uint64_t Mix(std::uint64_t x) {
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 29;
  x *= 0x94D049BB133111EBULL;
  return x ^ (x >> 32);
}

__global__ void FillBytes(std::uint32_t* p, std::size_t n, std::uint32_t seed) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = static_cast<std::uint32_t>(Mix(i * 0x9E3779B97F4A7C15ULL + seed));
}

// Small finite F16 scales (d, dmin) at the given offsets of every block.
__global__ void FixScales(std::uint8_t* p, std::size_t blocks,
                          std::uint32_t block_bytes, int off0, int off1) {
  const std::size_t b = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (b >= blocks)
    return;
  std::uint8_t* blk = p + b * block_bytes;
  const std::uint64_t h = Mix(b + 7);
  const _Float16 d = static_cast<_Float16>(
      (0.5F + static_cast<float>(h & 0xFFFF) / 65536.0F) * 2e-3F);
  const _Float16 m = static_cast<_Float16>(
      (0.5F + static_cast<float>((h >> 16) & 0xFFFF) / 65536.0F) * 1e-3F);
  __builtin_memcpy(blk + off0, &d, 2);
  if (off1 >= 0)
    __builtin_memcpy(blk + off1, &m, 2);
}

__global__ void FillFloat(float* p, std::size_t n, std::uint32_t seed,
                          float s) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < n) {
    const std::uint64_t h = Mix(i * 4 + seed);
    float u = 0.0F;
    for (int k = 0; k < 4; ++k)
      u += static_cast<float>((h >> (16 * k)) & 0xFFFF) / 65536.0F;
    p[i] = (u - 2.0F) * 1.7320508F * s;
  }
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + 255) / 256);
}

std::uint64_t Fnv(const void* data, std::size_t bytes,
                  std::uint64_t h = 0xcbf29ce484222325ULL) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

std::vector<std::uint32_t> ParseList(const char* s) {
  std::vector<std::uint32_t> out;
  for (const char* p = s; *p != '\0';) {
    char* end = nullptr;
    out.push_back(static_cast<std::uint32_t>(std::strtoul(p, &end, 10)));
    p = *end == ',' ? end + 1 : end;
  }
  return out;
}

struct TypeInfo {
  WeightType type;
  std::uint32_t block_bytes;
  int d_off, m_off;
  const char* name;
};
constexpr TypeInfo kIq4{WeightType::kIQ4_XS, 136, 0, -1, "IQ4_XS"};
constexpr TypeInfo kQ4K{WeightType::kQ4_K, 144, 0, 2, "Q4_K"};
constexpr TypeInfo kQ5K{WeightType::kQ5_K, 176, 0, 2, "Q5_K"};
constexpr TypeInfo kQ6K{WeightType::kQ6_K, 210, 208, -1, "Q6_K"};

struct Tensor {
  TypeInfo t;
  std::size_t bytes{};
  std::size_t expert_bytes{};
  std::uint8_t* p{};
  Tensor(TypeInfo info, std::uint32_t rows, std::uint32_t cols,
         std::uint32_t seed)
      : t(info) {
    const std::size_t row = std::size_t{cols} / 256 * info.block_bytes;
    expert_bytes = row * rows;
    bytes = expert_bytes * kE;
    Ok(hipMalloc(&p, bytes), "weights");
    const std::size_t words = bytes / 4;
    FillBytes<<<Blocks(words), 256>>>(reinterpret_cast<std::uint32_t*>(p),
                                      words, seed);
    const std::size_t blocks = bytes / info.block_bytes;
    FixScales<<<Blocks(blocks), 256>>>(p, blocks, info.block_bytes, info.d_off,
                                       info.m_off);
  }
  ~Tensor() { (void)hipFree(p); }
};

constexpr std::uint32_t kUnion[9] = {0, 8, 13, 17, 21, 25, 28, 31, 34};

// ids[t][k] for one call: `u` distinct experts in total, each token's 8
// distinct, every one of the u used by some token.
std::vector<std::int32_t> MakeIds(std::uint32_t n, const std::string& mode,
                                  std::mt19937& rng) {
  const std::uint32_t u = mode == "shared"     ? 8
                          : mode == "disjoint" ? 8 * n
                                               : kUnion[n];
  std::vector<std::int32_t> pool(kE);
  for (std::uint32_t i = 0; i < kE; ++i)
    pool[i] = static_cast<std::int32_t>(i);
  std::shuffle(pool.begin(), pool.end(), rng);
  pool.resize(u);
  std::vector<std::int32_t> ids(std::size_t{n} * kK);
  std::uint32_t used = 0;
  for (std::uint32_t t = 0; t < n; ++t) {
    // New experts this token introduces: the rest spread over the tokens.
    const std::uint32_t left = n - t;
    std::uint32_t fresh =
        t == 0 ? std::min<std::uint32_t>(8, u) : (u - used + left - 1) / left;
    fresh = std::min<std::uint32_t>(fresh, 8);
    std::vector<std::int32_t> mine(pool.begin() + used,
                                   pool.begin() + used + fresh);
    used += fresh;
    std::vector<std::int32_t> old(pool.begin(), pool.begin() + used - fresh);
    std::shuffle(old.begin(), old.end(), rng);
    for (std::size_t i = 0; mine.size() < kK && i < old.size(); ++i)
      mine.push_back(old[i]);
    std::shuffle(mine.begin(), mine.end(), rng);
    for (std::uint32_t k = 0; k < kK; ++k)
      ids[std::size_t{t} * kK + k] = mine[k];
  }
  return ids;
}

std::uint32_t Distinct(const std::vector<std::int32_t>& ids) {
  std::vector<std::int32_t> s(ids);
  std::sort(s.begin(), s.end());
  return static_cast<std::uint32_t>(std::unique(s.begin(), s.end()) -
                                    s.begin());
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::uint32_t> ns{1, 2, 3, 4, 5, 6, 7, 8};
  std::string types = "route,iq4,q5";
  std::string routing = "real";
  std::uint32_t calls = 64;
  int reps = 15;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--ns") == 0)
      ns = ParseList(argv[i + 1]);
    else if (std::strcmp(argv[i], "--types") == 0)
      types = argv[i + 1];
    else if (std::strcmp(argv[i], "--routing") == 0)
      routing = argv[i + 1];
    else if (std::strcmp(argv[i], "--calls") == 0)
      calls = static_cast<std::uint32_t>(std::atoi(argv[i + 1]));
    else if (std::strcmp(argv[i], "--reps") == 0)
      reps = std::atoi(argv[i + 1]);
  }
  hipDeviceProp_t prop{};
  Ok(hipGetDeviceProperties(&prop, 0), "props");
  std::printf("{\"device\":\"%s\",\"arch\":\"%s\",\"cus\":%d,\"calls\":%u,"
              "\"reps\":%d,\"routing\":\"%s\",\"source\":\"%s\"}\n",
              prop.name, prop.gcnArchName, prop.multiProcessorCount, calls,
              reps, routing.c_str(), GUFO_LING_MOE_SOURCE);
  std::fflush(stdout);
  const std::uint32_t max_n = *std::max_element(ns.begin(), ns.end());
  float* x = nullptr;
  float* act = nullptr;
  float* out = nullptr;
  std::int32_t* ids = nullptr;
  std::uint8_t* scratch = nullptr;
  const std::size_t x_call = std::size_t{max_n} * kH;
  const std::size_t act_call = std::size_t{max_n} * kK * kF;
  const std::size_t out_call = std::size_t{max_n} * kK * kH;
  const std::size_t sb = (moe::DecodeScratchBytes() + 255) / 256 * 256;
  Ok(hipMalloc(&x, x_call * calls * sizeof(float)), "x");
  Ok(hipMalloc(&act, act_call * calls * sizeof(float)), "act");
  Ok(hipMalloc(&out, out_call * calls * sizeof(float)), "out");
  Ok(hipMalloc(&ids, std::size_t{max_n} * kK * calls * sizeof(std::int32_t)),
     "ids");
  Ok(hipMalloc(&scratch, sb * calls), "scratch");
  FillFloat<<<Blocks(x_call * calls), 256>>>(x, x_call * calls, 3, 1.0F);
  hipEvent_t ev[4];
  for (auto& e : ev)
    Ok(hipEventCreate(&e), "event");
  std::vector<float> host_act, host_out;
  // Routing (RouteDecode: top-8 selection, weights and the expert plan the
  // expert launches read) for 1..8 tokens and the prefill Route for 512: one
  // digest of ids, weights and plan per n over `calls` different logit rows
  // (some with exact ties), and the device ms per call (graph replay).
  if (types.find("route") != std::string::npos) {
    const std::uint32_t rows = 512;
    float* lg = nullptr;
    float* bias = nullptr;
    float* w = nullptr;
    std::int32_t* rid = nullptr;
    std::uint32_t* counter = nullptr;
    Ok(hipMalloc(&lg, std::size_t{rows} * kE * calls * sizeof(float)), "logits");
    Ok(hipMalloc(&bias, kE * sizeof(float)), "bias");
    Ok(hipMalloc(&w, std::size_t{rows} * kK * calls * sizeof(float)), "w");
    Ok(hipMalloc(&rid, std::size_t{rows} * kK * calls * sizeof(std::int32_t)),
       "ids");
    Ok(hipMalloc(&counter, sizeof(std::uint32_t)), "counter");
    Ok(hipMemset(counter, 0, sizeof(std::uint32_t)), "counter");
    FillFloat<<<Blocks(std::size_t{rows} * kE * calls), 256>>>(
        lg, std::size_t{rows} * kE * calls, 21, 2.0F);
    FillFloat<<<Blocks(kE), 256>>>(bias, kE, 22, 0.05F);
    {
      // Exact ties: every 5th row of logits from {-1, 0, 1}.
      std::vector<float> h(std::size_t{rows} * kE * calls);
      Ok(hipMemcpy(h.data(), lg, h.size() * sizeof(float), hipMemcpyDeviceToHost),
         "download");
      for (std::size_t r = 0; r < std::size_t{rows} * calls; r += 5)
        for (std::uint32_t i = 0; i < kE; ++i)
          h[r * kE + i] = static_cast<float>(static_cast<int>((r * 31 + i * 7) % 3) - 1);
      Ok(hipMemcpy(lg, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice),
         "upload");
    }
    for (const std::uint32_t n : ns) {
      const auto call = [&](std::uint32_t c, hipStream_t st) {
        const std::size_t r0 = std::size_t{c} * max_n;
        moe::RouteDecode(lg + r0 * kE, bias, rid + r0 * kK, w + r0 * kK, n,
                         scratch + sb * c, counter, st);
      };
      Ok(hipMemset(scratch, 0, sb * calls), "clear");
      for (std::uint32_t c = 0; c < calls; ++c)
        call(c, nullptr);
      Ok(hipDeviceSynchronize(), "route");
      std::vector<std::uint8_t> hs(sb * calls);
      std::vector<std::int32_t> hi(std::size_t{max_n} * kK * calls);
      std::vector<float> hw(hi.size());
      Ok(hipMemcpy(hs.data(), scratch, hs.size(), hipMemcpyDeviceToHost), "dl");
      Ok(hipMemcpy(hi.data(), rid, hi.size() * 4, hipMemcpyDeviceToHost), "dl");
      Ok(hipMemcpy(hw.data(), w, hw.size() * 4, hipMemcpyDeviceToHost), "dl");
      std::uint64_t d = 0xcbf29ce484222325ULL;
      for (std::uint32_t c = 0; c < calls; ++c) {
        const std::size_t r0 = std::size_t{c} * max_n * kK;
        d = Fnv(hi.data() + r0, std::size_t{n} * kK * 4, d);
        d = Fnv(hw.data() + r0, std::size_t{n} * kK * 4, d);
        // The plan: count, then the live entries (40 bytes each).
        const std::uint8_t* pl = hs.data() + sb * c;
        int count = 0;
        std::memcpy(&count, pl, 4);
        d = Fnv(pl, 4, d);
        d = Fnv(pl + 16, std::size_t(std::clamp(count, 0, 64)) * 40, d);
      }
      hipStream_t cs;
      Ok(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking), "stream");
      hipGraph_t g;
      hipGraphExec_t ge;
      Ok(hipStreamBeginCapture(cs, hipStreamCaptureModeThreadLocal), "capture");
      for (std::uint32_t c = 0; c < calls; ++c)
        call(c, cs);
      Ok(hipStreamEndCapture(cs, &g), "capture end");
      Ok(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
      Ok(hipGraphDestroy(g), "graph destroy");
      std::vector<float> t;
      for (int r = 0; r < reps + 1; ++r) {
        Ok(hipEventRecord(ev[0], cs), "record");
        Ok(hipGraphLaunch(ge, cs), "launch");
        Ok(hipEventRecord(ev[1], cs), "record");
        Ok(hipEventSynchronize(ev[1]), "sync");
        float ms = 0;
        Ok(hipEventElapsedTime(&ms, ev[0], ev[1]), "elapsed");
        if (r > 0)
          t.push_back(ms / calls);
      }
      std::sort(t.begin(), t.end());
      Ok(hipGraphExecDestroy(ge), "exec destroy");
      Ok(hipStreamDestroy(cs), "stream destroy");
      std::printf("{\"route\":\"RouteDecode\",\"n\":%u,\"ms\":%.5f,"
                  "\"ms_min\":%.5f,\"digest\":\"%016llx\"}\n",
                  n, t[t.size() / 2], t.front(),
                  static_cast<unsigned long long>(d));
      std::fflush(stdout);
    }
    {
      // Prefill routing of 512 tokens (Route, no plan).
      moe::Route(lg, bias, rid, w, rows, nullptr);
      Ok(hipDeviceSynchronize(), "route");
      std::vector<std::int32_t> hi(std::size_t{rows} * kK);
      std::vector<float> hw(hi.size());
      Ok(hipMemcpy(hi.data(), rid, hi.size() * 4, hipMemcpyDeviceToHost), "dl");
      Ok(hipMemcpy(hw.data(), w, hw.size() * 4, hipMemcpyDeviceToHost), "dl");
      const std::uint64_t d = Fnv(hw.data(), hw.size() * 4, Fnv(hi.data(), hi.size() * 4));
      std::printf("{\"route\":\"Route\",\"n\":%u,\"digest\":\"%016llx\"}\n", rows,
                  static_cast<unsigned long long>(d));
      std::fflush(stdout);
    }
    for (void* q : {static_cast<void*>(lg), static_cast<void*>(bias),
                    static_cast<void*>(w), static_cast<void*>(rid),
                    static_cast<void*>(counter)})
      Ok(hipFree(q), "free");
  }
  for (const char* pair : {"iq4", "q5"}) {
    if (types.find(pair) == std::string::npos)
      continue;
    const bool iq4 = std::strcmp(pair, "iq4") == 0;
    const TypeInfo gu = iq4 ? kIq4 : kQ5K;
    const TypeInfo dn = iq4 ? kQ4K : kQ6K;
    Tensor gate(gu, kF, kH, 11), up(gu, kF, kH, 12), down(dn, kH, kF, 13);
    Ok(hipDeviceSynchronize(), "fill");
    for (const std::uint32_t n : ns) {
      std::mt19937 rng(1000 + n);
      std::vector<std::uint32_t> distinct(calls);
      std::vector<std::int32_t> all_ids;
      for (std::uint32_t c = 0; c < calls; ++c) {
        auto v = MakeIds(n, routing, rng);
        distinct[c] = Distinct(v);
        v.resize(std::size_t{max_n} * kK, -1);
        all_ids.insert(all_ids.end(), v.begin(), v.end());
      }
      Ok(hipMemcpy(ids, all_ids.data(), all_ids.size() * sizeof(std::int32_t),
                   hipMemcpyHostToDevice),
         "ids");
      Ok(hipMemset(act, 0, act_call * calls * sizeof(float)), "clear");
      Ok(hipMemset(out, 0, out_call * calls * sizeof(float)), "clear");
      hipStream_t stream = nullptr;
      const auto gate_up = [&](std::uint32_t c, bool plan) {
        const auto* idc = ids + std::size_t{c} * max_n * kK;
        auto* sc = scratch + sb * c;
        if (plan)
          moe::ExpertGateUpDecode(gate.p, up.p, gu.type, x + x_call * c, idc, n,
                                  4.0F, act + act_call * c, sc, stream);
        else
          moe::ExpertGateUpDecodePlanned(gate.p, up.p, gu.type, x + x_call * c,
                                         idc, n, 4.0F, act + act_call * c, sc,
                                         stream);
      };
      const auto down_call = [&](std::uint32_t c) {
        moe::ExpertDownDecodePlanned(down.p, dn.type, act + act_call * c,
                                     ids + std::size_t{c} * max_n * kK, n,
                                     out + out_call * c, scratch + sb * c,
                                     stream);
      };
      // Untimed: build every call's expert plan, then the outputs to digest.
      for (std::uint32_t c = 0; c < calls; ++c)
        gate_up(c, true);
      for (std::uint32_t c = 0; c < calls; ++c)
        down_call(c);
      Ok(hipDeviceSynchronize(), "warm");
      host_act.resize(act_call * calls);
      host_out.resize(out_call * calls);
      Ok(hipMemcpy(host_act.data(), act, host_act.size() * sizeof(float),
                   hipMemcpyDeviceToHost),
         "download");
      Ok(hipMemcpy(host_out.data(), out, host_out.size() * sizeof(float),
                   hipMemcpyDeviceToHost),
         "download");
      const std::uint64_t d_act =
          Fnv(host_act.data(), host_act.size() * sizeof(float));
      const std::uint64_t d_out =
          Fnv(host_out.data(), host_out.size() * sizeof(float));
      // The timed sequences replay HIP graphs (as the engine's decode does),
      // so host launch rate does not enter the device time.
      hipStream_t cs;
      Ok(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking), "stream");
      const auto capture = [&](auto&& body) {
        hipGraph_t g;
        hipGraphExec_t ge;
        Ok(hipStreamBeginCapture(cs, hipStreamCaptureModeThreadLocal),
           "capture");
        body();
        Ok(hipStreamEndCapture(cs, &g), "capture end");
        Ok(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
        Ok(hipGraphDestroy(g), "graph destroy");
        return ge;
      };
      stream = cs;
      hipGraphExec_t g_gu = capture([&] {
        for (std::uint32_t c = 0; c < calls; ++c)
          gate_up(c, false);
      });
      hipGraphExec_t g_dn = capture([&] {
        for (std::uint32_t c = 0; c < calls; ++c)
          down_call(c);
      });
      hipGraphExec_t g_pair = capture([&] {
        for (std::uint32_t c = 0; c < calls; ++c) {
          gate_up(c, false);
          down_call(c);
        }
      });
      std::vector<float> t_gu, t_dn, t_pair;
      for (int r = 0; r < reps + 1; ++r) {
        Ok(hipEventRecord(ev[0], cs), "record");
        Ok(hipGraphLaunch(g_gu, cs), "launch");
        Ok(hipEventRecord(ev[1], cs), "record");
        Ok(hipGraphLaunch(g_dn, cs), "launch");
        Ok(hipEventRecord(ev[2], cs), "record");
        Ok(hipGraphLaunch(g_pair, cs), "launch");
        Ok(hipEventRecord(ev[3], cs), "record");
        Ok(hipEventSynchronize(ev[3]), "sync");
        float a = 0, b = 0, p = 0;
        Ok(hipEventElapsedTime(&a, ev[0], ev[1]), "elapsed");
        Ok(hipEventElapsedTime(&b, ev[1], ev[2]), "elapsed");
        Ok(hipEventElapsedTime(&p, ev[2], ev[3]), "elapsed");
        if (r == 0)
          continue;  // first replay: warm-up
        t_gu.push_back(a / calls);
        t_dn.push_back(b / calls);
        t_pair.push_back(p / calls);
      }
      for (auto* ge : {&g_gu, &g_dn, &g_pair})
        Ok(hipGraphExecDestroy(*ge), "exec destroy");
      Ok(hipStreamDestroy(cs), "stream destroy");
      stream = nullptr;
      for (auto* v : {&t_gu, &t_dn, &t_pair})
        std::sort(v->begin(), v->end());
      double experts = 0;
      for (const std::uint32_t d : distinct)
        experts += d;
      experts /= calls;
      const double gu_mb =
          experts * static_cast<double>(gate.expert_bytes + up.expert_bytes) /
          1e6;
      const double dn_mb =
          experts * static_cast<double>(down.expert_bytes) / 1e6;
      const auto med = [](const std::vector<float>& v) { return v[v.size() / 2]; };
      std::printf(
          "{\"types\":\"%s+%s\",\"n\":%u,\"experts\":%.2f,"
          "\"gate_up_ms\":%.5f,\"gate_up_ms_min\":%.5f,\"gate_up_gbps\":%.1f,"
          "\"down_ms\":%.5f,\"down_ms_min\":%.5f,\"down_gbps\":%.1f,"
          "\"pair_ms\":%.5f,\"pair_gbps\":%.1f,"
          "\"digest_act\":\"%016llx\",\"digest_out\":\"%016llx\"}\n",
          gu.name, dn.name, n, experts, med(t_gu), t_gu.front(),
          gu_mb / med(t_gu), med(t_dn), t_dn.front(), dn_mb / med(t_dn),
          med(t_pair), (gu_mb + dn_mb) / med(t_pair),
          static_cast<unsigned long long>(d_act),
          static_cast<unsigned long long>(d_out));
      std::fflush(stdout);
    }
  }
  return 0;
}
