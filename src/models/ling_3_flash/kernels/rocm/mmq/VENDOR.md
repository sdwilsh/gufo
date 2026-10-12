# Quantised expert kernels

`quant_units.hip.hpp` carries the GGML block layouts of IQ4_XS, Q4_K, Q5_K and
Q6_K and the IQ4_NL codebook (`kvalues_iq4nl`). The layouts and the codebook
come from llama.cpp `ggml/src/ggml-common.h` (MIT, The ggml authors), the same
pin the other model packages vendor (commit
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`; see THIRD_PARTY_NOTICES.md). The
per-sub-block dequantisation follows `ggml-quants.c` `dequantize_row_*`.

No other upstream source is copied. llama.cpp's mmq/mmvq templates were not
used because they only instantiate Q8_1-activation integer dot products and
carry no MoE path for IQ4_XS or Q6_K. These kernels use F32 activations
(decode, exact) or F16 WMMA tiles (prefill) instead. The ggml notice shipped
as `share/licenses/gufo/third-party/llama.cpp.txt` (from `licenses/`) covers
this derivation, so no separate licence file is installed.
