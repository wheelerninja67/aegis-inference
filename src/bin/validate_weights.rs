// src/bin/validate_weights.rs
//
// Run: cargo run --bin validate_weights -- /path/to/bitnet-b1.58.gguf
// What it checks:
//   1. Zero tensor names (gguf parse correctness)
//   2. Ternary density (should be ~33% pos, ~33% neg, ~33% zero for BitNet)
//   3. SIMD dot product self-consistency (bitmask result == scalar result)

use aegis::gguf::weight_loader::AegisModel;
use aegis::simd::dispatch::vtable;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: validate_weights <model.gguf>");

    println!("[validate] opening {}...", path);
    let model = AegisModel::load_gguf(&path).expect("load failed");

    println!("[validate] {} tensors loaded", model.tensors.len());

    for (name, tensor) in &model.tensors {
        // --- Ternary density check ---
        let total_weights = tensor.rows * tensor.cols_raw;
        let mask_words_per_row = tensor.cols / 64;

        let (mut pos_count, mut neg_count) = (0u64, 0u64);
        for w in 0..(tensor.rows * mask_words_per_row) {
            let pm = unsafe { *tensor.pos_mask.add(w) };
            let nm = unsafe { *tensor.neg_mask.add(w) };
            pos_count += pm.count_ones() as u64;
            neg_count += nm.count_ones() as u64;
        }
        let zero_count = total_weights as u64 - pos_count - neg_count;
        let pos_pct = 100.0 * pos_count as f64 / total_weights as f64;
        let neg_pct = 100.0 * neg_count as f64 / total_weights as f64;
        let zer_pct = 100.0 * zero_count as f64 / total_weights as f64;

        println!(
            "[{}] {}x{} | +1: {:.1}%  -1: {:.1}%  0: {:.1}%",
            name, tensor.rows, tensor.cols_raw, pos_pct, neg_pct, zer_pct
        );

        if pos_count + neg_count > (total_weights as u64 * 95 / 100) {
            eprintln!("  ! WARNING: near-zero sparsity - model may not be true BitNet b1.58");
        }
        if zero_count > (total_weights as u64 * 95 / 100) {
            eprintln!("  ! WARNING: >95% zeros - threshold is destroying signal");
        }

        // --- SIMD self-consistency check on first row ---
        let test_cols = tensor.cols_raw.min(64);
        let activations: Vec<i8> = (0..test_cols).map(|i| (i % 7) as i8 - 3).collect();
        let mut padded_acts = vec![0i8; tensor.cols];
        padded_acts[..test_cols].copy_from_slice(&activations);

        // Scalar reference
        let mut scalar_dot: i32 = 0;
        for col in 0..tensor.cols_raw {
            let bit_pos = col;
            let word_idx = bit_pos / 64;
            let bit_shift = bit_pos % 64;
            let pm = unsafe { *tensor.pos_mask.add(word_idx) };
            let nm = unsafe { *tensor.neg_mask.add(word_idx) };
            let is_pos = (pm >> bit_shift) & 1;
            let is_neg = (nm >> bit_shift) & 1;
            let weight: i32 = is_pos as i32 - is_neg as i32;
            scalar_dot += weight * (padded_acts[col] as i32);
        }

        // SIMD kernel
        let vt = vtable();
        let simd_dot = unsafe {
            (vt.ternary_dot)(
                padded_acts.as_ptr(),
                tensor.pos_mask,
                tensor.neg_mask,
                tensor.cols,
            )
        };

        if scalar_dot == simd_dot {
            println!(
                "  v SIMD consistency: scalar={} simd={}",
                scalar_dot, simd_dot
            );
        } else {
            eprintln!(
                "  x SIMD MISMATCH on tensor '{}': scalar={} simd={}",
                name, scalar_dot, simd_dot
            );
            eprintln!("    -> Check bitmask bit-ordering vs SIMD lane ordering");
        }
    }
}
