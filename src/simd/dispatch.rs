use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdTier {
    Scalar       = 0,
    Avx2         = 1,
    Avx512Bw     = 2, // 512-bit without VNNI
    Avx512Vnni   = 3, // AVX-512 + VPDPBSSD: signed i8 dot product in hardware
    ArmNeon      = 4,
    ArmSve       = 5, // scalable: vl detected at runtime
}

/// All function pointers are `unsafe fn` — the caller guarantees alignment/length invariants.
pub struct KernelVtable {
    pub tier: SimdTier,
    /// ternary_dot: dot(activations[i8], pos_mask[bit-packed u64], neg_mask[bit-packed u64])
    /// `n_weights` MUST be a multiple of the tier's natural width (64 for AVX-512, 32 for AVX2)
    pub ternary_dot: unsafe fn(
        acts:     *const i8,
        pos_mask: *const u64,
        neg_mask: *const u64,
        n_weights: usize,
    ) -> i32,
    
    /// Batch version: compute M output rows simultaneously, reusing the activation vector
    pub ternary_dot_batch: unsafe fn(
        acts:      *const i8,
        pos_masks: *const u64, // row-major: [M x ceil(n_weights/64)] u64s
        neg_masks: *const u64,
        n_weights: usize,
        m_rows:    usize,
        out:       *mut i32,   // M output accumulators
    ),
}

static VTABLE: OnceLock<KernelVtable> = OnceLock::new();

pub fn vtable() -> &'static KernelVtable {
    VTABLE.get_or_init(|| detect_best_kernel())
}

fn detect_best_kernel() -> KernelVtable {
    #[cfg(target_arch = "x86_64")]
    {
        // For the Shadow Build, we will wire up the AVX512BW path first.
        // The other modules will be implemented progressively.
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return KernelVtable {
                tier: SimdTier::Avx512Bw,
                ternary_dot: super::avx512_bw::ternary_dot,
                ternary_dot_batch: super::avx512_bw::ternary_dot_batch,
            };
        }
        
        // Fallback or other architectures would be wired here.
    }
    
    // Default scalar fallback
    KernelVtable {
        tier: SimdTier::Scalar,
        ternary_dot: scalar_fallback_dot,
        ternary_dot_batch: scalar_fallback_batch,
    }
}

// Temporary scalar fallbacks until all SIMD modules are implemented
unsafe fn scalar_fallback_dot(
    _acts: *const i8,
    _pos_mask: *const u64,
    _neg_mask: *const u64,
    _n_weights: usize,
) -> i32 {
    0
}

unsafe fn scalar_fallback_batch(
    _acts: *const i8,
    _pos_masks: *const u64,
    _neg_masks: *const u64,
    _n_weights: usize,
    _m_rows: usize,
    _out: *mut i32,
) {}
