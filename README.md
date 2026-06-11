# Aegis Inference 🛡️

Aegis is a high-performance, CPU-bound LLM inference framework written in pure Rust. It is designed to shatter the GPU bottleneck by mapping 1.58-bit ternary quantization directly onto x86 AVX-512 and ARM SVE vector registers. 

By eliminating the need for unified memory and high-bandwidth VRAM, Aegis allows standard consumer CPUs and cloud compute nodes to run large language models at near-GPU parity.

## The Bottleneck
Modern LLM inference relies on 16-bit or 8-bit floating-point weights, requiring immense memory bandwidth. The GPU monopoly exists not because CPUs lack compute, but because CPUs lack the memory bandwidth to feed standard weights into their cores.

## The Aegis Solution
Aegis bypasses the von Neumann bottleneck through:
1. **1.58-bit Ternary States:** Weights are compressed to `-1, 0, 1`. This reduces the memory footprint by >85%, allowing entire neural network layers to reside strictly within the CPU's L3 cache.
2. **AVX-512 Forced Vectorization:** Aegis is written in bare-metal Rust, mapping ternary matrix multiplications directly to SIMD vector registers for pseudo-parallel execution.
3. **Zero-State Pruning:** Dynamic path routing skips compute cycles entirely if a weight vector registers as `0`.

## Architecture
- `aegis-core`: The base ternary tensor manipulation library.
- `aegis-simd`: Hardware-specific vectorization bindings (AVX-512, NEON).
- `aegis-router`: The sparse compute pruning engine.

## Getting Started (Nightly Rust Required)
```bash
cargo build --release --features="avx512"
```

## Contributing
Aegis is open-source. We welcome pull requests focused on SIMD optimization, cache-locality improvements, and ternary model conversion scripts.

*Project Aegis is currently under active development.*
