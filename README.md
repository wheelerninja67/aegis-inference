# aegis

bare-metal inference engine for 1.58-bit ternary neural networks (bitnet). written in rust. 

we do not use gpus. aegis maps 2-bit quantized weights directly to cpu registers using branchless dual-bitmask separation. it leverages llvm auto-vectorization to dynamically target AVX2 (intel/amd) or NEON (arm/apple) intrinsics at compile time.

the goal is absolute low-latency, offline inference on consumer edge hardware.

## architecture

1. **aegis-core**: zero-copy memory mapped inference engine. sliding window kv-cache with constant-memory bounded context.
2. **aegis-quantizer**: absmean quantizer. crushes fp16 weights into absolute ternary states (-1, 0, 1) and packs 4 weights per u8 byte (16x compression).
3. **aegis-simd**: hardware bitmask router. calculates dot products purely through addition/subtraction. zero floating-point multiplication.
4. **aegis-router**: non-blocking tokio/axum async event loop for concurrent continuous batching.

## benchmarks

measured on intel i5-8265u (no dedicated gpu).

*   **parameters**: ~4.19M (1024x4096 test matrix)
*   **quantization**: 1.58-bit packed u8
*   **math kernel**: branchless avx2 lut separation
*   **latency**: 6.05 ms / token
*   **throughput**: 165.18 tokens / second

## build

requires rust nightly (`#![feature(portable_simd)]`).

```bash
git clone https://github.com/wheelerninja67/aegis-inference.git
cd aegis-inference

# compile with native hardware intrinsics (avx2/neon)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# boot the async router
cargo run --release --bin aegis_inference
```

## roadmap

*   avx-512 intrinsic expansions for 64-byte registers.
*   prefault memory mapping for zero-latency instantiation.
*   custom ARM SVE paths for tesla/spacex edge deployments.

license: MIT
