#![feature(portable_simd)]

pub mod architecture;
pub mod gguf_parser;
pub mod tokenizer;

use std::arch::x86_64::*;

/// V6 Cache-Line Aligned Struct
/// Guarantees 64-byte alignment to prevent False Sharing across Rayon threads.
#[repr(C, align(64))]
pub struct AlignedWeightRow {
    pub data: [u8; 64], // 64 bytes = one cache line = 256 ternary weights (at 2-bit packing)
    pub scale: f32,
    _pad: [u8; 60],     // pad to next cache line boundary
}

/// Represents a heavily quantized tensor where weights are constrained to -1, 0, or 1.
/// V6 Upgrade: True 1.58-bit BitNet memory structure using packed 2-bit values.
/// This reduces physical memory footprint by a massive 8x compared to FP32.
pub struct TernaryTensor {
    pub rows: usize,
    pub cols: usize,
    pub packed_weights: Vec<u8>, // Each u8 holds 4 weights (2 bits each)
    pub scale: f32,
}

impl TernaryTensor {
    pub fn new(rows: usize, cols: usize, scale: f32) -> Self {
        let packed_cols = (cols + 3) / 4;
        Self {
            rows,
            cols,
            packed_weights: vec![0; rows * packed_cols],
            scale,
        }
    }

    /// AVX2 + Rayon Multi-Threaded Matrix Multiplication
    /// Note: Full bitmask separation AVX2 kernel is mocked here pending full integration.
    #[target_feature(enable = "avx2")]
    pub unsafe fn fast_simd_inference(&self, activations: &[i8]) -> Vec<f32> {
        assert_eq!(self.cols, activations.len());
        let mut output = vec![0.0; self.rows];

        use rayon::prelude::*;
        
        let packed_cols = self.cols / 4;
        
        output.par_iter_mut().enumerate().for_each(|(r, out_val)| {
            let mut sum = 0;
            let row_offset = r * packed_cols;
            
            // Loop through the packed bytes
            for c in 0..packed_cols {
                let packed_byte = self.packed_weights[row_offset + c];
                
                // Unpack the 4 weights
                for i in 0..4 {
                    let bits = (packed_byte >> (i * 2)) & 0b11;
                    let w: i32 = match bits {
                        0b00 => 0,
                        0b01 => 1,
                        0b10 => -1,
                        _ => 0,
                    };
                    
                    let act_idx = c * 4 + i;
                    if act_idx < self.cols {
                        sum += w * (activations[act_idx] as i32);
                    }
                }
            }
            
            *out_val = (sum as f32) * self.scale;
        });

        output
    }
}
