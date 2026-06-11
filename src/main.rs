use aegis_inference::TernaryTensor;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("  AEGIS INFERENCE ENGINE: V1.0 Benchmark (AVX2 + 6MB L3)    ");
    println!("============================================================");

    let rows = 1024;
    let cols = 4096; // 4096 parameters per layer
    println!("[*] Initializing 1.58-bit Ternary Matrix ({} x {})", rows, cols);

    let mut tensor = TernaryTensor::new(rows, cols);

    // Fill with deterministic -1, 0, 1
    for i in 0..(rows * cols) {
        tensor.data[i] = ((i % 3) as i8) - 1;
    }

    let mut activations = vec![0i8; cols];
    for i in 0..cols {
        activations[i] = ((i % 3) as i8) - 1;
    }

    println!("[*] Matrix loaded strictly into L3 Cache boundary.");
    
    // Benchmark 1: Naive Scalar CPU (The Standard Way)
    println!("[*] Running Naive Scalar Inference...");
    let start_naive = Instant::now();
    let mut naive_output = vec![0i32; rows];
    for r in 0..rows {
        let mut sum = 0;
        let offset = r * cols;
        for c in 0..cols {
            sum += (tensor.data[offset + c] as i32) * (activations[c] as i32);
        }
        naive_output[r] = sum;
    }
    let duration_naive = start_naive.elapsed();

    // Benchmark 2: AVX2 Aegis Engine
    println!("[*] Running Aegis AVX2 Vectorized Inference...");
    let start_avx2 = Instant::now();
    let avx2_output = unsafe { tensor.fast_simd_inference(&activations) };
    let duration_avx2 = start_avx2.elapsed();

    println!("============================================================");
    println!("  BENCHMARK RESULTS                                         ");
    println!("============================================================");
    println!("  Naive Scalar Time:  {:?}", duration_naive);
    println!("  Aegis AVX2 Time:    {:?}", duration_avx2);
    
    // Simple speedup calculation
    let speedup = duration_naive.as_secs_f64() / duration_avx2.as_secs_f64();
    println!("  Aegis Speed Multiplier: {:.2}x faster", speedup);
    println!("============================================================");

    // Verify output matches
    assert_eq!(naive_output, avx2_output, "Math mismatch!");
    println!("[+] AVX2 output mathematically verified against scalar baseline.");
}
