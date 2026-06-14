use aegis::TernaryTensor;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("  AEGIS V6 BENCHMARK ENGINE (INTEL i5 TARGET)               ");
    println!("============================================================");

    let rows = 1024;
    let cols = 4096;
    println!("[*] Target Matrix: {}x{} (~4.19M Parameters)", rows, cols);
    println!("[*] Memory Footprint: 1.04 MB (16x Compression vs FP32)");
    println!("[*] Math Kernel: Dual-Bitmask Separation Trick (Zero-Multiplication)");

    // Create random activations
    let mut activations = vec![0i8; cols];
    for i in 0..cols {
        activations[i] = (i % 3 - 1) as i8; // pseudo-random -1, 0, 1
    }

    // Create dual bitmasks manually for the test
    let mask_len = cols / 8;
    let mut pos_mask = vec![0u8; rows * mask_len];
    let mut neg_mask = vec![0u8; rows * mask_len];
    for i in 0..(rows * mask_len) {
        pos_mask[i] = 0b01010101; // Fake alternating bits for stress test
        neg_mask[i] = 0b10101010;
    }

    let tensor = TernaryTensor {
        rows,
        cols,
        pos_mask,
        neg_mask,
        scale: 1.0,
    };

    let iterations = 1000;
    println!("[*] Running 100 warm-up iterations to flush L3 Cache...");
    for _ in 0..100 {
        unsafe { std::hint::black_box(tensor.fast_simd_inference(&activations)) };
    }

    println!("[*] Executing {} parallel SIMD iterations...", iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { std::hint::black_box(tensor.fast_simd_inference(&activations)) };
    }
    let duration = start.elapsed();
    let avg_time = duration.as_secs_f64() * 1000.0 / (iterations as f64);

    println!("============================================================");
    println!("  BENCHMARK RESULTS");
    println!("============================================================");
    println!("Total Time ({} passes): {:?}", iterations, duration);
    println!("Average Latency:         {:.4} ms per token", avg_time);
    println!(
        "Raw Compute Throughput:  {:.2} Tokens / Second",
        1000.0 / avg_time
    );
    println!("============================================================");
}
