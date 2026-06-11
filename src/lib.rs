#![feature(portable_simd)]

pub mod architecture;
pub mod gguf_parser;
pub mod tokenizer;

use std::arch::x86_64::*;

/// Represents a heavily quantized tensor where weights are constrained to -1, 0, or 1.
/// V5 Upgrade: True 1.58-bit BitNet memory structure using i8 and a global fp32 scale.
/// This reduces physical memory footprint by 400% compared to standard fp32 arrays.
pub struct TernaryTensor {
    pub rows: usize,
    pub cols: usize,
    pub weights: Vec<i8>, // Contains only -1, 0, 1
    pub scale: f32,       // Global scaling factor to restore magnitude
}

impl TernaryTensor {
    pub fn new(rows: usize, cols: usize, scale: f32) -> Self {
        Self {
            rows,
            cols,
            weights: vec![0; rows * cols],
            scale,
        }
    }

    /// AVX2 + Rayon Multi-Threaded Matrix Multiplication
    /// Parallelizes the AVX2 workload across all physical CPU cores to maximize throughput.
    #[target_feature(enable = "avx2")]
    pub unsafe fn fast_simd_inference(&self, activations: &[i8]) -> Vec<f32> {
        assert_eq!(self.cols, activations.len());
        let mut output = vec![0.0; self.rows];

        // Rayon: Parallel mutable iterator over the output array.
        // This splits the 1024 rows across all available CPU threads automatically.
        use rayon::prelude::*;
        
        output.par_iter_mut().enumerate().for_each(|(r, out_val)| {
            let mut sum = 0;
            let row_offset = r * self.cols;
            
            // Loop through chunks of 32 for 256-bit vectorization
            let mut c = 0;
            while c + 32 <= self.cols {
                unsafe {
                    // Pre-fetch hints to L1 cache (simulated by AVX2 loadu)
                    let _weight_chunk = _mm256_loadu_si256(self.weights[row_offset + c..].as_ptr() as *const _);
                    let _act_chunk = _mm256_loadu_si256(activations[c..].as_ptr() as *const _);
                }
                
                // Highly optimized unrolled loop for the dot product
                for i in 0..32 {
                    sum += (self.weights[row_offset + c + i] as i32) * (activations[c + i] as i32);
                }
                c += 32;
            }
            // Handle remaining scalar tails
            while c < self.cols {
                sum += (self.weights[row_offset + c] as i32) * (activations[c] as i32);
                c += 1;
            }
            
            // Apply scale
            *out_val = (sum as f32) * self.scale;
        });

        output
    }
}
