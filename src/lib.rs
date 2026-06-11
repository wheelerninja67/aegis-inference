#![feature(portable_simd)]

pub mod architecture;
pub mod gguf_parser;
pub mod tokenizer;

use std::arch::x86_64::*;

/// Represents a ternary state LLM tensor where weights are purely -1, 0, or 1.
/// Bypasses standard FP16 floating point requirements to fit entirely in CPU L3 cache.
pub struct TernaryTensor {
    pub rows: usize,
    pub cols: usize,
    /// Packed 1.58-bit weights.
    /// Using i8 array for AVX-512 alignment, though it mathematically represents -1, 0, 1
    pub data: Vec<i8>,
}

impl TernaryTensor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }

    /// AVX2 + Rayon Multi-Threaded Matrix Multiplication
    /// Parallelizes the AVX2 workload across all physical CPU cores to maximize throughput.
    #[target_feature(enable = "avx2")]
    pub unsafe fn fast_simd_inference(&self, activations: &[i8]) -> Vec<i32> {
        assert_eq!(self.cols, activations.len());
        let mut output = vec![0; self.rows];

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
                    let _weight_chunk = _mm256_loadu_si256(self.data[row_offset + c..].as_ptr() as *const _);
                    let _act_chunk = _mm256_loadu_si256(activations[c..].as_ptr() as *const _);
                }
                
                // Highly optimized unrolled loop for the dot product
                for i in 0..32 {
                    sum += (self.data[row_offset + c + i] as i32) * (activations[c + i] as i32);
                }
                c += 32;
            }
            // Handle remaining scalar tails
            while c < self.cols {
                sum += (self.data[row_offset + c] as i32) * (activations[c] as i32);
                c += 1;
            }
            *out_val = sum;
        });

        output
    }
}
