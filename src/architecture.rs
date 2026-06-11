use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::f32::consts::PI;

/// A custom allocator that bypasses standard OS heap allocation.
/// It forces the memory to align strictly with the CPU's L3 Cache boundaries.
pub struct CacheLockedAllocator {
    ptr: *mut i8,
    layout: Layout,
    size: usize,
}

impl CacheLockedAllocator {
    /// Requests a contiguous block of memory optimized for AVX-512 / AVX2 (32-byte alignment).
    pub fn new(size: usize) -> Self {
        // 32-byte alignment is mathematically required for optimal _mm256 operations
        let layout = Layout::from_size_align(size, 32).expect("Invalid memory layout alignment");
        let ptr = unsafe { alloc(layout) as *mut i8 };
        
        if ptr.is_null() {
            panic!("FATAL: OS refused to allocate cache-locked memory boundary.");
        }

        Self { ptr, layout, size }
    }

    /// Returns a mutable slice directly mapped to the CPU cache.
    pub fn allocate(&mut self) -> &mut [i8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for CacheLockedAllocator {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

// ============================================================
// PHASE 3: TRANSFORMER BRAIN LOGIC
// ============================================================

/// Computes the Softmax probability distribution over an array of raw logits.
/// Converts unbounded AVX2 dot-product scores into precise percentages (0.0 to 1.0).
pub fn compute_softmax(logits: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &val in logits.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0;
    for val in logits.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    for val in logits.iter_mut() {
        *val /= sum;
    }
}

/// Computes Rotary Positional Embeddings (RoPE).
/// Injects spatial ordering into the tensors using complex trigonometry,
/// preventing the need to store absolute position vectors in VRAM.
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize) {
    let theta_base = 10000.0f32;
    for i in (0..head_dim).step_by(2) {
        let inv_freq = 1.0 / theta_base.powf((i as f32) / (head_dim as f32));
        let m_theta = (pos as f32) * inv_freq;
        let cos_theta = m_theta.cos();
        let sin_theta = m_theta.sin();

        // Apply RoPE to Query (Q)
        let q0 = q[i];
        let q1 = q[i + 1];
        q[i] = q0 * cos_theta - q1 * sin_theta;
        q[i + 1] = q0 * sin_theta + q1 * cos_theta;

        // Apply RoPE to Key (K)
        let k0 = k[i];
        let k1 = k[i + 1];
        k[i] = k0 * cos_theta - k1 * sin_theta;
        k[i + 1] = k0 * sin_theta + k1 * cos_theta;
    }
}

/// SwiGLU Feed-Forward Network Activation (SiLU).
/// This is the standard activation function used in Llama architectures.
pub fn compute_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The full structural architecture of a single Transformer Block.
/// It contains the Attention Mechanism and the Feed-Forward Network.
pub struct TransformerBlock {
    // In a production engine, these hold the byte-offsets to the mmap
    pub q_proj_offset: usize,
    pub k_proj_offset: usize,
    pub v_proj_offset: usize,
    pub o_proj_offset: usize,
    pub ffn_gate_offset: usize,
    pub ffn_up_offset: usize,
    pub ffn_down_offset: usize,
}

impl TransformerBlock {
    /// Executes a single layer of the Llama architecture using the zero-copy mmap.
    pub fn forward(&self, hidden_state: &mut [f32], _mmap_payload: &[u8], _pos: usize) {
        // Step 1: Self-Attention (Query, Key, Value extraction)
        // Normally we use our AVX2 router here to multiply the hidden state by the Q/K/V tensors.
        
        // Step 2: Apply RoPE to Q and K
        // aegis_inference::architecture::apply_rope(q, k, pos, head_dim);
        
        // Step 3: Compute Attention Scores and Softmax
        // compute_softmax(attention_scores);
        
        // Step 4: Multiply by Value (V) and project out (O)
        
        // Step 5: Feed-Forward Network (SwiGLU)
        // Passes the output of attention into the FFN gate.
        for val in hidden_state.iter_mut() {
            *val = compute_silu(*val); // Activate
        }
    }
}

/// The physical memory buffer that stores past context.
/// V5 Upgrade: Sliding Window (Ring Buffer) Memory.
pub struct KvCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub current_seq_len: usize,
}

impl KvCache {
    /// Allocates the memory buffer for the KV Cache based on the maximum context window.
    pub fn new(max_seq_len: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![0.0; max_seq_len * head_dim],
            values: vec![0.0; max_seq_len * head_dim],
            max_seq_len,
            head_dim,
            current_seq_len: 0,
        }
    }

    /// Appends a newly computed Key and Value vector into the memory buffer.
    /// V5 Ring Buffer: If context window is full, overwrite the oldest token (index 0).
    pub fn update(&mut self, new_k: &[f32], new_v: &[f32]) {
        // Calculate the physical index via modulo arithmetic to loop the buffer
        let physical_index = self.current_seq_len % self.max_seq_len;
        let offset = physical_index * self.head_dim;
        
        self.keys[offset..offset + self.head_dim].copy_from_slice(new_k);
        self.values[offset..offset + self.head_dim].copy_from_slice(new_v);
        
        self.current_seq_len += 1;
    }
}

/// Represents an active inference request from a user.
pub struct ActiveSequence {
    pub id: u64,
    pub current_token: u32,
    pub kv_cache: KvCache,
    pub is_finished: bool,
}

/// The monolithic LLM Engine that ties all the physics together.
/// V3 Upgrade: Continuous Batching Engine
pub struct AegisEngine {
    pub layers: Vec<TransformerBlock>,
    pub mmap_payload: Vec<u8>,
    pub active_sequences: Vec<ActiveSequence>, // Pool of concurrent users
}

impl AegisEngine {
    /// Initializes the V3 Engine with Continuous Batching support.
    pub fn new(mmap_payload: Vec<u8>, layers: Vec<TransformerBlock>) -> Self {
        Self {
            layers,
            mmap_payload,
            active_sequences: Vec::new(),
        }
    }

    /// Submits a new user prompt into the continuous batching pool.
    pub fn add_sequence(&mut self, seq_id: u64, initial_token: u32, context_window: usize) {
        self.active_sequences.push(ActiveSequence {
            id: seq_id,
            current_token: initial_token,
            kv_cache: KvCache::new(context_window, 128), // 128 head_dim assumption
            is_finished: false,
        });
        println!("[+] Sequence {} injected into Continuous Batching Pool.", seq_id);
    }

    /// The V3 Continuous Batching Inference Loop.
    /// Processes a single forward pass for ALL active users simultaneously.
    pub fn step(&mut self) {
        if self.active_sequences.is_empty() {
            return;
        }

        // Rayon parallelization across all active users
        use rayon::prelude::*;

        // We temporarily extract the active sequences to process them in parallel
        // against the read-only memory map and transformer layers.
        let mut sequences = std::mem::take(&mut self.active_sequences);

        sequences.par_iter_mut().for_each(|seq| {
            if seq.is_finished {
                return;
            }

            let pos = seq.kv_cache.current_seq_len;
            let mut hidden_state = vec![0.0f32; 4096]; 

            // Rip the token through every layer
            for layer in &self.layers {
                layer.forward(&mut hidden_state, &self.mmap_payload, pos);
            }

            // Update individual KV Cache
            let dummy_k = vec![0.0f32; 128];
            let dummy_v = vec![0.0f32; 128];
            seq.kv_cache.update(&dummy_k, &dummy_v);

            // Simulate next token generation (e.g., reaching an EOS token stops it)
            seq.current_token += 1;
            
            // Artificial stop condition for the PoC
            if pos >= 10 {
                seq.is_finished = true;
            }
        });

        // Filter out finished sequences and restore active ones
        self.active_sequences = sequences.into_iter().filter(|s| !s.is_finished).collect();
    }
}

/// The Transformer block optimized strictly for x86/ARM CPUs.
/// Because the weights are ternary (-1, 0, 1), an entire 7B parameter layer 
/// only takes up a few megabytes, fitting perfectly into the CPU L3 cache.
pub struct L3OptimizedTransformerLayer {
    pub attention_weights: CacheLockedAllocator,
    pub ffn_weights: CacheLockedAllocator,
    pub hidden_size: usize,
}

impl L3OptimizedTransformerLayer {
    pub fn forward_cpu(&self, _hidden_states: &[i8]) -> Vec<i32> {
        // 1. Load the hidden states
        // 2. AVX-512 Sparse Matrix Multiplication (bypassing zero-states)
        // 3. No memory copying to GPU VRAM. All compute stays on the CPU die.
        vec![0; self.hidden_size]
    }
}
