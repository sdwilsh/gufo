// Golden output digests (FNV-1a over the F32 output bytes) of the prefill
// GEMM and MoE kernels as of the parent of the WS13 change (the original
// 64x64 GemmKernel / PrefillKernel). The WS13 kernels are bit-identical to
// them by design; dense_ops_test and moe_ops_test compare against these so a
// later change that alters prefill bits must regenerate them on purpose.
// Keys: dense (op, rows, k, n), inputs from a fixed per-case seed; MoE
// (label, expert types, n), inputs from the fixed seeds in moe_ops_test.cpp.
#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_PARENT_DIGESTS_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_PARENT_DIGESTS_HPP_

namespace parent_digests {

struct Dense {
  const char* op;
  unsigned rows, k, n;
  unsigned long long digest;
};
struct Moe {
  const char* key;
  unsigned long long act, out;
};

inline constexpr Dense kDense[] = {
    {"GemmQ8_0", 20512, 2560, 9, 0xe5b6a97824295c90ULL},
    {"GemmQ8_0", 2560, 4096, 9, 0x7883a2720e89b4efULL},
    {"GemmQ8_0", 768, 2560, 9, 0xdf1f28ec70924a74ULL},
    {"GemmQ8_0", 2560, 768, 9, 0x83d61d4b12b9be9eULL},
    {"GemmQ8_0", 6144, 2560, 9, 0xebc179d0a1c3b37bULL},
    {"GemmQ8_0", 2560, 6144, 9, 0x056d60e583a5a653ULL},
    {"GemmF16", 4097, 2560, 9, 0x8c834398b8354bfaULL},
    {"GemmQ8_0", 20512, 2560, 64, 0xb317a3fdf00c3e9bULL},
    {"GemmQ8_0", 2560, 4096, 64, 0xe6d9f63e7f1453ebULL},
    {"GemmQ8_0", 768, 2560, 64, 0x1a60d62492e473a1ULL},
    {"GemmQ8_0", 2560, 768, 64, 0x36c3327d987a82f3ULL},
    {"GemmQ8_0", 6144, 2560, 64, 0x7940e6a9a99db9b2ULL},
    {"GemmQ8_0", 2560, 6144, 64, 0x04ae1854638f7b55ULL},
    {"GemmF16", 4097, 2560, 64, 0x4f85cb4fac462556ULL},
    {"GemmQ8_0", 20512, 2560, 512, 0x81aa08b499ef6420ULL},
    {"GemmQ8_0", 2560, 4096, 512, 0xaf88cbd5906990c5ULL},
    {"GemmQ8_0", 768, 2560, 512, 0x620db8d96913b8b3ULL},
    {"GemmQ8_0", 2560, 768, 512, 0x81bc652dd397c161ULL},
    {"GemmQ8_0", 6144, 2560, 512, 0xc7bf031aa95308c8ULL},
    {"GemmQ8_0", 2560, 6144, 512, 0xd9041db9c8cbff6fULL},
    {"GemmF16", 4097, 2560, 512, 0x7af8535f1c65c05eULL},
    {"GemmQ8_0", 20512, 2560, 2048, 0x1aebe6727f5880f5ULL},
    {"GemmQ8_0", 2560, 4096, 2048, 0x843c80f03cf5fdfbULL},
    {"GemmQ8_0", 768, 2560, 2048, 0x0971dbbd028d5614ULL},
    {"GemmQ8_0", 2560, 768, 2048, 0x828573ca61e5ae65ULL},
    {"GemmQ8_0", 6144, 2560, 2048, 0xe1cf6a0d5184bb27ULL},
    {"GemmQ8_0", 2560, 6144, 2048, 0x727a7e307fb5a147ULL},
    {"GemmF16", 4097, 2560, 2048, 0xdf8b12185c8f5431ULL},
    {"", 0, 0, 0, 0}};
inline constexpr Moe kMoe[] = {
    {"pool16 IQ4_XS/Q4_K n=9", 0x1cc973f2530c2d03ULL, 0x69addf93106daa0bULL},
    {"pool16 IQ4_XS/Q4_K n=64", 0xa60dc7fa09d5775aULL, 0x955d623755d63f32ULL},
    {"pool16 IQ4_XS/Q4_K n=513", 0x3b9e5b9381d046a5ULL, 0xda4d6fbe2104e8e3ULL},
    {"pool16 Q5_K/Q6_K n=9", 0x3e6a03475e3fc3c4ULL, 0x0b26f05bab606791ULL},
    {"pool16 Q5_K/Q6_K n=64", 0x7c5f9131b22347dfULL, 0x84d0551b95eb80daULL},
    {"pool16 Q5_K/Q6_K n=513", 0xae1cc0c9ec4273e9ULL, 0x1fad28014455fda2ULL},
    {"real-shape IQ4_XS/Q4_K n=9", 0xc53ec427a3d1d07aULL,
     0xc7bc45ab3b9493b5ULL},
    {"real-shape IQ4_XS/Q4_K n=64", 0x48e7ca339dd6a6b3ULL,
     0x1af429693430eb7fULL},
    {"real-shape IQ4_XS/Q4_K n=512", 0x71e552ca6bfc1d41ULL,
     0xf4cd35f64d058d06ULL},
    {"real-shape IQ4_XS/Q4_K n=2048", 0x67a43b1e8a3b127bULL,
     0xa56e3be796d2fd6eULL},
    {"real-shape Q5_K/Q6_K n=9", 0x175f7a0a49877e30ULL, 0xc56022a8bd61e49aULL},
    {"real-shape Q5_K/Q6_K n=64", 0xa6ea13b990e3d29fULL, 0xaac794ac1f5431e2ULL},
    {"real-shape Q5_K/Q6_K n=512", 0xaa5b8fa0736a7feaULL,
     0xf82f02b16e670593ULL},
    {"real-shape Q5_K/Q6_K n=2048", 0xb37bb6b002c05cf5ULL,
     0x47bd34ec2cdc5038ULL},
    {"", 0, 0}};

}  // namespace parent_digests

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_PARENT_DIGESTS_HPP_
