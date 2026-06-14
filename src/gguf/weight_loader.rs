// src/gguf/weight_loader.rs
//
// Pipeline:
//   GGUF file
//     |- GgufParser::read_tensor_into()  [raw Q8_0 bytes]
//          |- q8_blocks_to_ternary_bitmasks()  [threshold -> pack u64]
//               |- BitmaskTensor  [ready for AVX-512 kernel]

use crate::gguf::parser::GgufParser;
use std::alloc::{Layout, alloc};
use std::collections::HashMap;

// --- Q8_0 Block Layout --------------------------------------------------------
// 32 elements per block.
// [ f16 delta (2 bytes) | i8 x 32 (32 bytes) ]  = 34 bytes per block
// BitNet b1.58 stores weights as i8 in {-128..127} but values cluster at {-1, 0, 1}
// after the ternary training constraint. Delta is the per-block scale.

const Q8_BLOCK_SIZE: usize = 32; // elements per block
const Q8_BLOCK_BYTES: usize = 34; // 2 (f16) + 32 (i8s)

#[repr(C, packed)]
struct Q8Block {
    delta_bits: u16, // f16 little-endian -> use f16_to_f32() below
    quants: [i8; Q8_BLOCK_SIZE],
}

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    // Manual f16 -> f32 conversion without the `half` crate.
    // IEEE 754: sign(1) | exp(5) | mantissa(10)
    let sign = ((bits >> 15) as u32) << 31;
    let exp_raw = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as u32;

    let (exp_f32, mant_f32) = if exp_raw == 0 {
        // Subnormal: flush to zero (adequate for weight scales)
        (0u32, mantissa << 13)
    } else if exp_raw == 31 {
        // Inf / NaN passthrough
        (255u32, mantissa << 13)
    } else {
        (exp_raw as u32 + 127 - 15, mantissa << 13)
    };
    f32::from_bits(sign | (exp_f32 << 23) | mant_f32)
}

// --- Bitmask Tensor -----------------------------------------------------------

/// A fully packed, SIMD-ready ternary weight tensor.
/// Memory layout:
///   pos_mask: [ceil(rows x cols / 64) x u64]  - bit j=1 means weight j is +1
///   neg_mask: [ceil(rows x cols / 64) x u64]  - bit j=1 means weight j is -1
///   (zero weights: neither bit set)
pub struct BitmaskTensor {
    pub rows: usize,
    pub cols: usize,     // padded to multiple of 64
    pub cols_raw: usize, // original unpadded column count
    // Heap-allocated, 64-byte aligned (cache-line aligned, huge-page eligible)
    pub pos_mask: *mut u64,
    pub neg_mask: *mut u64,
    pub mask_words: usize, // = rows * (cols / 64)

    // Per-block scales preserved for dequantization during attention
    pub scales: Vec<f32>,
}

unsafe impl Send for BitmaskTensor {}
unsafe impl Sync for BitmaskTensor {}

impl Drop for BitmaskTensor {
    fn drop(&mut self) {
        if !self.pos_mask.is_null() {
            let layout = Layout::from_size_align(self.mask_words * 8, 64).unwrap();
            unsafe {
                std::alloc::dealloc(self.pos_mask as *mut u8, layout);
                std::alloc::dealloc(self.neg_mask as *mut u8, layout);
            }
        }
    }
}

impl BitmaskTensor {
    fn alloc_aligned_u64(n_words: usize) -> *mut u64 {
        let layout = Layout::from_size_align(n_words * 8, 64).expect("alignment layout failed");
        let ptr = unsafe { alloc(layout) as *mut u64 };
        assert!(!ptr.is_null(), "bitmask allocation failed");
        // Zero-initialize (zeroed = all weights are 0)
        unsafe {
            std::ptr::write_bytes(ptr, 0u8, n_words);
        }
        ptr
    }
}

// --- Core Conversion: Q8_0 blocks -> dual bitmasks ----------------------------

const TERNARY_THRESHOLD: f32 = 0.0;

pub fn q8_blocks_to_ternary_bitmasks(raw: &[u8], rows: usize, cols_raw: usize) -> BitmaskTensor {
    let cols = (cols_raw + 63) & !63;
    let mask_words = rows * (cols / 64);
    let n_blocks_total = (rows * cols_raw).div_ceil(Q8_BLOCK_SIZE);

    assert_eq!(
        raw.len(),
        n_blocks_total * Q8_BLOCK_BYTES,
        "raw buffer size mismatch: expected {} bytes for {} blocks, got {}",
        n_blocks_total * Q8_BLOCK_BYTES,
        n_blocks_total,
        raw.len()
    );

    let pos_mask = BitmaskTensor::alloc_aligned_u64(mask_words);
    let neg_mask = BitmaskTensor::alloc_aligned_u64(mask_words);
    let mut scales = Vec::with_capacity(n_blocks_total);

    let blocks: &[Q8Block] =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const Q8Block, n_blocks_total) };

    let pos_slice = unsafe { std::slice::from_raw_parts_mut(pos_mask, mask_words) };
    let neg_slice = unsafe { std::slice::from_raw_parts_mut(neg_mask, mask_words) };

    for (block_idx, block) in blocks.iter().enumerate() {
        let delta = f16_to_f32(block.delta_bits);
        scales.push(delta);

        let elem_base = block_idx * Q8_BLOCK_SIZE;

        for (local_idx, &q) in block.quants.iter().enumerate() {
            let elem_idx = elem_base + local_idx;
            if elem_idx >= rows * cols_raw {
                break;
            }

            let row = elem_idx / cols_raw;
            let col = elem_idx % cols_raw;

            let real_weight = delta * (q as f32);
            let threshold = delta.abs() * TERNARY_THRESHOLD;

            let bit_pos = row * cols + col;
            let word_idx = bit_pos / 64;
            let bit_shift = bit_pos % 64;

            if real_weight > threshold {
                pos_slice[word_idx] |= 1u64 << bit_shift;
            } else if real_weight < -threshold {
                neg_slice[word_idx] |= 1u64 << bit_shift;
            }
        }
    }

    BitmaskTensor {
        rows,
        cols,
        cols_raw,
        pos_mask,
        neg_mask,
        mask_words,
        scales,
    }
}

// --- Model Loader: wire GgufParser -> BitmaskTensor map -----------------------

use crate::tokenizer::AegisTokenizer;

pub struct AegisModel {
    pub tensors: HashMap<String, BitmaskTensor>,
    pub norm_tensors: HashMap<String, Vec<f32>>,
    pub tokenizer: AegisTokenizer,
    pub embed_table: Vec<f32>,
    pub n_ctx: u64,
    pub n_heads: u64,
    pub n_heads_kv: u64,
    pub n_embd: u64,
    pub n_layers: u64,
    pub head_dim: u64,
}

impl AegisModel {
    fn extract_int(val: &crate::gguf::parser::MetadataValue) -> Option<u64> {
        match val {
            crate::gguf::parser::MetadataValue::Uint64(x) => Some(*x),
            crate::gguf::parser::MetadataValue::UInt32(x) => Some(*x as u64),
            crate::gguf::parser::MetadataValue::Int32(x) => Some(*x as u64),
            _ => None,
        }
    }

    pub fn load_gguf(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut parser = GgufParser::open(path)?;

        let tokenizer = AegisTokenizer::from_gguf(&parser)?;
        eprintln!("[loader] tokenizer: {} vocab entries", tokenizer.vocab_size);

        let n_ctx = parser
            .header
            .metadata
            .get("llama.context_length")
            .or_else(|| parser.header.metadata.get("bitnet.context_length"))
            .and_then(Self::extract_int)
            .unwrap_or(2048);
        let n_embd = parser
            .header
            .metadata
            .get("llama.embedding_length")
            .or_else(|| parser.header.metadata.get("bitnet.embedding_length"))
            .and_then(Self::extract_int)
            .unwrap_or(4096);
        let n_heads = parser
            .header
            .metadata
            .get("llama.attention.head_count")
            .or_else(|| parser.header.metadata.get("bitnet.attention.head_count"))
            .and_then(Self::extract_int)
            .unwrap_or(32);
        let n_heads_kv = parser
            .header
            .metadata
            .get("llama.attention.head_count_kv")
            .or_else(|| parser.header.metadata.get("bitnet.attention.head_count_kv"))
            .and_then(Self::extract_int)
            .unwrap_or(n_heads);
        let n_layers = parser
            .header
            .metadata
            .get("llama.block_count")
            .or_else(|| parser.header.metadata.get("bitnet.block_count"))
            .and_then(Self::extract_int)
            .unwrap_or(32);
        let head_dim = n_embd / n_heads;

        let max_tensor_bytes = (0..parser.header.tensor_count as usize)
            .map(|i| parser.tensor_byte_size(i))
            .max()
            .unwrap_or(0);

        let mut scratch: Vec<u8> = vec![0u8; max_tensor_bytes];
        let mut tensors: HashMap<String, BitmaskTensor> = HashMap::new();
        let mut norm_tensors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut embed_table: Vec<f32> = Vec::new();

        let n_tensors = parser.header.tensor_count as usize;
        for idx in 0..n_tensors {
            let info = parser.header.tensors[idx].clone();
            let name = info.name.clone();

            if name == "token_embd.weight" {
                let byte_size = parser.tensor_byte_size(idx);
                unsafe {
                    parser.read_tensor_into(idx, scratch.as_mut_ptr(), byte_size)?;
                }

                let (rows, cols_raw) = match info.dimensions.as_slice() {
                    [c, r] => (*r as usize, *c as usize),
                    [c] => (1usize, *c as usize),
                    [c, r, _] => (*r as usize, *c as usize),
                    _ => continue,
                };

                if info.ggml_type == 8 {
                    let n_blocks_total = (rows * cols_raw).div_ceil(Q8_BLOCK_SIZE);
                    let blocks: &[Q8Block] = unsafe {
                        std::slice::from_raw_parts(
                            scratch.as_ptr() as *const Q8Block,
                            n_blocks_total,
                        )
                    };
                    embed_table.reserve(rows * cols_raw);
                    for block in blocks {
                        let delta = f16_to_f32(block.delta_bits);
                        for &q in &block.quants {
                            embed_table.push(q as f32 * delta);
                        }
                    }
                    embed_table.truncate(rows * cols_raw);
                } else if info.ggml_type == 0 {
                    let f32_slice: &[f32] = unsafe {
                        std::slice::from_raw_parts(scratch.as_ptr() as *const f32, rows * cols_raw)
                    };
                    embed_table.extend_from_slice(f32_slice);
                }
                eprintln!(
                    "[loader] loaded embedding table: {} elements",
                    embed_table.len()
                );
                continue;
            }

            if name.contains("norm") {
                let byte_size = parser.tensor_byte_size(idx);
                unsafe {
                    parser.read_tensor_into(idx, scratch.as_mut_ptr(), byte_size)?;
                }

                let num_elements = byte_size / 4;
                let mut norm_vec = Vec::with_capacity(num_elements);

                if info.ggml_type == 0 {
                    // FP32
                    let f32_slice: &[f32] = unsafe {
                        std::slice::from_raw_parts(scratch.as_ptr() as *const f32, num_elements)
                    };
                    norm_vec.extend_from_slice(f32_slice);
                } else {
                    eprintln!(
                        "[loader] unsupported norm tensor type {} for {}",
                        info.ggml_type, name
                    );
                    continue;
                }

                norm_tensors.insert(name.clone(), norm_vec);
                eprintln!("[loader] loaded norm tensor: {}", name);
                continue;
            }

            if !name.contains("weight") || name.contains("embed") {
                eprintln!("[loader] skipping non-ternary tensor: {}", name);
                continue;
            }

            if info.ggml_type != 8 {
                eprintln!(
                    "[loader] skipping non-Q8_0 tensor: {} (type {})",
                    name, info.ggml_type
                );
                continue;
            }

            let byte_size = parser.tensor_byte_size(idx);
            eprintln!(
                "[loader] loading tensor: {} ({} bytes, dims {:?})",
                name, byte_size, info.dimensions
            );

            unsafe {
                parser.read_tensor_into(idx, scratch.as_mut_ptr(), byte_size)?;
            }

            let (rows, cols_raw) = match info.dimensions.as_slice() {
                [c, r] => (*r as usize, *c as usize),
                [c] => (1usize, *c as usize),
                [c, r, _] => (*r as usize, *c as usize),
                _ => {
                    eprintln!(
                        "[loader] unexpected dims for {}: {:?} - skipping",
                        name, info.dimensions
                    );
                    continue;
                }
            };

            let bitmask_tensor =
                q8_blocks_to_ternary_bitmasks(&scratch[..byte_size], rows, cols_raw);

            tensors.insert(name, bitmask_tensor);
        }

        eprintln!(
            "[loader] loaded {} ternary tensors, {} norm tensors",
            tensors.len(),
            norm_tensors.len()
        );
        Ok(AegisModel {
            tensors,
            norm_tensors,
            tokenizer,
            embed_table,
            n_ctx,
            n_heads,
            n_heads_kv,
            n_embd,
            n_layers,
            head_dim,
        })
    }
}
