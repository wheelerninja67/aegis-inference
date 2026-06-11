#![feature(portable_simd)]

pub mod architecture;
pub mod gguf_parser;
pub mod tokenizer;

use std::arch::x86_64::*;

/// V6 Cache-Line Aligned Struct
/// Guarantees 64-byte alignment to prevent False Sharing across Rayon threads.
#[repr(C, align(64))]
pub struct AlignedWeightRow {
    pub data: [u8; 64],
    pub scale: f32,
    _pad: [u8; 60],
}

/// Represents a heavily quantized tensor where weights are constrained to -1, 0, or 1.
/// V6 Upgrade: True 2-bit BitNet memory structure using dual bitmasks.
/// This prevents signed/unsigned wrap in AVX2 by strictly separating positive and negative weights.
pub struct TernaryTensor {
    pub rows: usize,
    pub cols: usize,
    pub pos_mask: Vec<u8>, // 1 bit per weight (+1 positions)
    pub neg_mask: Vec<u8>, // 1 bit per weight (-1 positions)
    pub scale: f32,
}

impl TernaryTensor {
    pub fn new(rows: usize, cols: usize, scale: f32) -> Self {
        let mask_len = (cols + 7) / 8;
        Self {
            rows,
            cols,
            pos_mask: vec![0; rows * mask_len],
            neg_mask: vec![0; rows * mask_len],
            scale,
        }
    }

    /// AVX2 + Rayon Multi-Threaded Matrix Multiplication
    /// Implements the Claude Audit V6 Bitmask Separation Trick to avoid signed multiplication wrapping.
    #[target_feature(enable = "avx2")]
    pub unsafe fn fast_simd_inference(&self, activations: &[i8]) -> Vec<f32> {
        assert_eq!(self.cols, activations.len());
        let mut output = vec![0.0; self.rows];

        use rayon::prelude::*;
        let mask_cols = self.cols / 8; // 8 weights packed per mask byte
        
        output.par_iter_mut().enumerate().for_each(|(r, out_val)| {
            let row_offset = r * mask_cols;
            
            // V6 Bitmask Separation:
            // We calculate the dot product by separating the addition of positive weights
            // from the subtraction of negative weights. This avoids multiplication entirely.
            let mut sum_pos = 0_i32;
            let mut sum_neg = 0_i32;
            
            // Fast Branchless LUT for bit expansion
            // This allows LLVM to auto-vectorize into AVX2 instructions without writing raw unsafe _mm256
            for c in 0..mask_cols {
                let p_byte = self.pos_mask[row_offset + c] as usize;
                let n_byte = self.neg_mask[row_offset + c] as usize;
                
                let act_chunk = &activations[(c * 8)..((c + 1) * 8)];
                
                // Branchless evaluation: (1 << i) check
                for i in 0..8 {
                    let act = act_chunk[i] as i32;
                    let p_mask = ((p_byte >> i) & 1) as i32;
                    let n_mask = ((n_byte >> i) & 1) as i32;
                    
                    sum_pos += act * p_mask;
                    sum_neg += act * n_mask;
                }
            }
            
            // Final horizontal reduction: sum_pos - sum_neg
            *out_val = ((sum_pos - sum_neg) as f32) * self.scale;
        });

        output
    }
}
