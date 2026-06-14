use crate::kv_cache::page_pool::{PAGE_TOKENS, PagePool};

/// CPU Flash Attention with paged KV.
/// Processes one attention head at a time (call in parallel across heads via Rayon).
///
/// Algorithm: Online softmax (Milakov 2018) with tiled KV reads.
/// Tile size TILE_K chosen so that: TILE_K x head_dim x sizeof(i8) <= L2_SIZE / 4
///
/// For i5-8265U: 256KB L2 / 4 = 64KB -> TILE_K = 64KB / (128 bytes) = 512 tokens per tile.
pub const TILE_K: usize = 64; // tokens per KV tile (tune to target L2)

pub struct FlashAttnOutput {
    /// Output vector: head_dim x f32 (final weighted sum)
    pub out: Vec<f32>,
}

/// Single-query flash attention over paged KV cache.
/// q: query vector [head_dim] as f32
/// pool: reference to the physical page pool
/// block_table: sequence's logical->physical page map
/// n_kv_tokens: how many KV tokens are valid for this sequence
/// head_dim: dimension of each attention head
pub unsafe fn flash_attn_paged(
    q: &[f32],
    pool: &PagePool,
    block_table: &[u32],
    n_kv_tokens: usize,
    head_dim: usize,
    num_heads: usize,
    head_idx: usize,
) -> Vec<f32> {
    debug_assert_eq!(q.len(), head_dim);

    // Online softmax state
    let mut m: f32 = f32::NEG_INFINITY; // running max of QK scores
    let mut l: f32 = 0.0; // running sum of exp(score - m)
    let mut out = vec![0.0f32; head_dim]; // accumulator for weighted V sum

    // Process KV tokens in tiles of TILE_K
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut kv_processed = 0usize;

    while kv_processed < n_kv_tokens {
        let tile_end = (kv_processed + TILE_K).min(n_kv_tokens);
        let tile_size = tile_end - kv_processed;

        // Scratch buffers for this tile
        let mut scores = vec![0.0f32; tile_size];

        // Step 1: Compute Q x K^T for this tile
        for (tile_pos, kv_pos) in (kv_processed..tile_end).enumerate() {
            let page_idx = block_table[kv_pos / PAGE_TOKENS];
            let slot = kv_pos % PAGE_TOKENS;

            unsafe {
                // Get K pointer for this token, this head
                let k_base = pool.slab.add(
                    page_idx as usize * pool.page_stride
                        + slot * num_heads * head_dim
                        + head_idx * head_dim,
                );

                // Dot product: q (f32) . k (i8) with dequant scale
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[d] * (*k_base.add(d) as f32); // i8 -> f32 on the fly
                }
                scores[tile_pos] = dot * scale;
            }
        }

        // Step 2: Online softmax update over this tile's scores
        let m_tile = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let m_new = m.max(m_tile);

        // Rescale existing accumulator: out *= exp(m_old - m_new), l *= exp(m_old - m_new)
        let rescale = (m - m_new).exp();
        for o in out.iter_mut() {
            *o *= rescale;
        }
        l *= rescale;
        m = m_new;

        // Step 3: Weighted accumulation over V for this tile
        let mut l_tile = 0.0f32;
        for (tile_pos, kv_pos) in (kv_processed..tile_end).enumerate() {
            let page_idx = block_table[kv_pos / PAGE_TOKENS];
            let slot = kv_pos % PAGE_TOKENS;

            unsafe {
                let v_base = pool.slab.add(
                    page_idx as usize * pool.page_stride
                    + PAGE_TOKENS * num_heads * head_dim  // V section starts after K section
                    + slot * num_heads * head_dim
                    + head_idx * head_dim,
                );

                let alpha = (scores[tile_pos] - m).exp();
                l_tile += alpha;

                for d in 0..head_dim {
                    out[d] += alpha * (*v_base.add(d) as f32);
                }
            }
        }
        l += l_tile;
        kv_processed = tile_end;
    }

    // Normalize
    let l_inv = 1.0 / l;
    for o in out.iter_mut() {
        *o *= l_inv;
    }
    out
}
