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

// Non-absorbed MLA prefill (mla_ops_test TestNonAbsorbedParentDigest) as of
// the parent of WS31 (V^T stored row-strided, every tile of a partially live
// query wave on the pair-isolated P.V path). WS31 changes only where the
// expanded values live and which lanes vote for that path, so the output bits
// are the same. Keys: (first position, tokens, persistent store rows).
struct MlaPrefill {
  unsigned first, n, store_rows;
  unsigned long long digest;
};
inline constexpr MlaPrefill kMlaPrefill[] = {
    {0, 9, 0, 0xee5cdfa8018c746cULL},
    {0, 16, 2048, 0xebb8ab402439fb45ULL},
    {1000, 37, 2048, 0x66cec3a359f5cd7bULL},
    {2040, 16, 4096, 0xde954fd02d105dfcULL},
    {2047, 100, 0, 0xcf4b012635097ad5ULL},
    {4090, 100, 4096, 0xf630f8429e94a078ULL},
    {5000, 130, 8192, 0x6b60fa804f6d78baULL},
    {6100, 300, 4096, 0x5e186bea2e7dd599ULL},
    {16384, 16, 18432, 0x28c1b741f6d47c09ULL},
    {16384, 64, 0, 0xd2e01b4044c69407ULL},
    {0, 0, 0, 0}};

// Routed-expert decode GEMVs and routing (moe_ops_test TestDecodeDigest /
// TestRouteDigest) as of the parent of WS34 (block-wide top-8 extraction).
// WS34 rewrote the routing with wave reductions and kept the decode GEMVs
// (a register-staged rewrite was slower on gfx1151); every bit is the same.
// Keys: decode (label, types, n, routing, clamp): act, out digests; routing
// (n): ids ^ plan, weights digests.
struct Decode {
  const char* key;
  unsigned long long a, b;
};
inline constexpr Decode kDecode[] = {
    {"route-decode n=1", 0x46b8af560ff86360ULL, 0x3bdd9d688d2c1b90ULL},
    {"route-decode n=2", 0xc9bc1c3c9e6d4093ULL, 0x905a5a9245e5fbc1ULL},
    {"route-decode n=3", 0x19c3bb1e89432a74ULL, 0x122872206472c110ULL},
    {"route-decode n=5", 0x135d3cb2abe91146ULL, 0xe13ab69291d140d7ULL},
    {"route-decode n=8", 0xc631e6872c63bc16ULL, 0xe5e93705589228a1ULL},
    {"route n=513", 0xe07667a2f6b79780ULL, 0x0678ea1bf3cc2b10ULL},
    {"decode pool16 IQ4_XS/Q4_K n=1 L=0", 0xb1952cd32b04f745ULL,
     0x9911dde9ba91622fULL},
    {"decode pool16 IQ4_XS/Q4_K n=2 L=0", 0x29f6eb74553ea114ULL,
     0x7630a67b7bd38ff4ULL},
    {"decode pool16 IQ4_XS/Q4_K n=3 L=0", 0x459ce0f4b044fa63ULL,
     0xa738f7c684e8bfbeULL},
    {"decode pool16 IQ4_XS/Q4_K n=5 L=0", 0x903f4ae2f6cc1042ULL,
     0xb780f13dad049570ULL},
    {"decode pool16 IQ4_XS/Q4_K n=8 L=0", 0x1b45c88b0bf8c1c6ULL,
     0xca6b4f2bd2549d19ULL},
    {"decode pool16 IQ4_XS/Q4_K n=8 same-ids L=0", 0x5cca8251006f356cULL,
     0xbb32e5ce2e85b081ULL},
    {"decode pool16 IQ4_XS/Q4_K n=1 L=4", 0x7747f16677c780f1ULL,
     0x9911dde9ba91622fULL},
    {"decode pool16 IQ4_XS/Q4_K n=2 L=4", 0x284fc08cd2580d6aULL,
     0x7630a67b7bd38ff4ULL},
    {"decode pool16 IQ4_XS/Q4_K n=3 L=4", 0xf111d56fe93af29fULL,
     0xa738f7c684e8bfbeULL},
    {"decode pool16 IQ4_XS/Q4_K n=5 L=4", 0x829feec74d1147acULL,
     0xb780f13dad049570ULL},
    {"decode pool16 IQ4_XS/Q4_K n=8 L=4", 0xec4f16bbe27b80c2ULL,
     0xca6b4f2bd2549d19ULL},
    {"decode pool16 IQ4_XS/Q4_K n=8 same-ids L=4", 0x9a9b989f27e9553dULL,
     0xbb32e5ce2e85b081ULL},
    {"decode pool16 Q5_K/Q6_K n=1 L=0", 0x6ec5b1b2ad076349ULL,
     0x5721edc095b16e48ULL},
    {"decode pool16 Q5_K/Q6_K n=2 L=0", 0x2bf478056365556aULL,
     0x254281d62bd4ae03ULL},
    {"decode pool16 Q5_K/Q6_K n=3 L=0", 0xcb7a938807e7cd1cULL,
     0xda8b6f688950446eULL},
    {"decode pool16 Q5_K/Q6_K n=5 L=0", 0x4b303bd819e07dabULL,
     0xd03f5f86d2abc191ULL},
    {"decode pool16 Q5_K/Q6_K n=8 L=0", 0xdac6473f2abb9f64ULL,
     0x8eac46d3504d63dcULL},
    {"decode pool16 Q5_K/Q6_K n=8 same-ids L=0", 0x0cdd3204b90dcc43ULL,
     0x8a9be7bab27b1dbeULL},
    {"decode pool16 Q5_K/Q6_K n=1 L=4", 0xc8f49f22e28478aaULL,
     0x5721edc095b16e48ULL},
    {"decode pool16 Q5_K/Q6_K n=2 L=4", 0x2eac5310225be46eULL,
     0x254281d62bd4ae03ULL},
    {"decode pool16 Q5_K/Q6_K n=3 L=4", 0x04c04604db2765f2ULL,
     0xda8b6f688950446eULL},
    {"decode pool16 Q5_K/Q6_K n=5 L=4", 0x4cfad168ad112760ULL,
     0xd03f5f86d2abc191ULL},
    {"decode pool16 Q5_K/Q6_K n=8 L=4", 0x416c44d8f37b5befULL,
     0x8eac46d3504d63dcULL},
    {"decode pool16 Q5_K/Q6_K n=8 same-ids L=4", 0xe97a2adb53c635a1ULL,
     0x8a9be7bab27b1dbeULL},
    {"decode real-shape IQ4_XS/Q4_K n=1 L=4", 0x3c4450c90aace73aULL,
     0xd673fbd54ea6da13ULL},
    {"decode real-shape IQ4_XS/Q4_K n=2 L=4", 0xf398b5f2ad6a6e83ULL,
     0x08cc3737b08f948aULL},
    {"decode real-shape IQ4_XS/Q4_K n=8 L=4", 0x87ba200aadd042c0ULL,
     0xcc6601624d57912fULL},
    {"decode real-shape Q5_K/Q6_K n=1 L=4", 0x4418d223185d9b53ULL,
     0x2a9d4c088097e1fcULL},
    {"decode real-shape Q5_K/Q6_K n=2 L=4", 0x77d26a44a8c3a48eULL,
     0xb7deb1b2835cf092ULL},
    {"decode real-shape Q5_K/Q6_K n=8 L=4", 0xde6a5d1ff3840671ULL,
     0x69ea3ff32c182402ULL},
    {"", 0, 0}};

}  // namespace parent_digests

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_PARENT_DIGESTS_HPP_
