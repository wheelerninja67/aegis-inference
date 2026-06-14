#![allow(non_snake_case)]
use std::arch::x86_64::*;

/// Processes 64 weights per loop iteration.
/// pos_mask and neg_mask are bit-packed: bit j of mask word i = weight (i*64 + j)
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_dot(
    acts: *const i8,
    pos_mask: *const u64,
    neg_mask: *const u64,
    n_weights: usize, // must be multiple of 64
) -> i32 {
    debug_assert_eq!(n_weights % 64, 0);
    let n_blocks = n_weights / 64;

    // Four 512-bit accumulators -> 16 x i32 each = 64 i32 totals
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();

    // Ones vector for madd trick (i16 -> i32 widening sum)
    let ones_i16 = _mm512_set1_epi16(1i16);

    let mut a_ptr = acts;
    let mut pm_ptr = pos_mask;
    let mut nm_ptr = neg_mask;

    for _ in 0..n_blocks {
        unsafe {
            // Load 64 activations as packed i8
            let a_vec: __m512i = _mm512_loadu_si512(a_ptr as *const __m512i);

            // THE V2.0 KILLER FEATURE: Load bitmasks DIRECTLY as AVX-512 k-registers
            let pm: __mmask64 = *pm_ptr;
            let nm: __mmask64 = *nm_ptr;

            // Zero-mask select: keep activation only where mask bit = 1
            let pos_contrib: __m512i = _mm512_maskz_mov_epi8(pm, a_vec);
            let neg_contrib: __m512i = _mm512_maskz_mov_epi8(nm, a_vec);

            // Lower 256 bits of pos and neg contributions
            let pos_lo_256: __m256i = _mm512_castsi512_si256(pos_contrib);
            let pos_hi_256: __m256i = _mm512_extracti64x4_epi64::<1>(pos_contrib);
            let neg_lo_256: __m256i = _mm512_castsi512_si256(neg_contrib);
            let neg_hi_256: __m256i = _mm512_extracti64x4_epi64::<1>(neg_contrib);

            // Widen i8 -> i16 (sign-extending)
            let pos_lo_16: __m512i = _mm512_cvtepi8_epi16(pos_lo_256);
            let pos_hi_16: __m512i = _mm512_cvtepi8_epi16(pos_hi_256);
            let neg_lo_16: __m512i = _mm512_cvtepi8_epi16(neg_lo_256);
            let neg_hi_16: __m512i = _mm512_cvtepi8_epi16(neg_hi_256);

            // diff = pos - neg in i16
            let diff_lo: __m512i = _mm512_sub_epi16(pos_lo_16, neg_lo_16);
            let diff_hi: __m512i = _mm512_sub_epi16(pos_hi_16, neg_hi_16);

            // Widening horizontal pair-sum: i16 x i16(ones) -> i32
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(diff_lo, ones_i16));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(diff_hi, ones_i16));

            a_ptr = a_ptr.add(64);
            pm_ptr = pm_ptr.add(1);
            nm_ptr = nm_ptr.add(1);
        }
    }

    // Horizontal reduction: acc0 + acc1 -> 16 i32 -> scalar
    let combined = _mm512_add_epi32(acc0, acc1);
    _mm512_reduce_add_epi32(combined)
}

// Temporary stub for batch processing
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_dot_batch(
    _acts: *const i8,
    _pos_masks: *const u64,
    _neg_masks: *const u64,
    _n_weights: usize,
    _m_rows: usize,
    _out: *mut i32,
) {
    // To be implemented in Phase 6: Continuous Batching
}
