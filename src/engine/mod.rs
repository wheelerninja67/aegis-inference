use std::sync::Arc;
use rayon::prelude::*;
use std::simd::num::SimdFloat;

use crate::gguf::weight_loader::AegisModel;
use crate::kv_cache::page_pool::{PagePool, PAGE_TOKENS};
use crate::attention::flash_cpu::flash_attn_paged;
use crate::simd::dispatch::vtable;
use crate::scheduler::batching::{Scheduler, SequenceRequest, SequenceState};

#[inline]
fn argmax_f32(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) -> u32 {
    let mut penalized_logits = logits.to_vec();
    // Aggressive Presence Penalty
    for &tok in generated_tokens {
        let idx = tok as usize;
        if idx < penalized_logits.len() {
            penalized_logits[idx] -= 50.0; // Flat massive logit reduction
        }
    }
    // Hard ban on the immediately preceding token to prevent consecutive stutter
    if let Some(&last_tok) = generated_tokens.last() {
        if (last_tok as usize) < penalized_logits.len() {
            penalized_logits[last_tok as usize] = f32::NEG_INFINITY;
        }
    }

    penalized_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

pub struct AegisEngine {
    pub model:      Arc<AegisModel>,
    pub pool:       Arc<PagePool>,
    pub scheduler:  Scheduler,
}

impl AegisEngine {
    pub fn new(model_path: &str, total_kv_pages: usize) -> Result<Self, Box<dyn std::error::Error>> {
        eprintln!("[engine] loading model from {}...", model_path);
        let model = Arc::new(AegisModel::load_gguf(model_path)?);
        eprintln!("[engine] model loaded: {} layers, {} heads, embd={}",
            model.n_layers, model.n_heads, model.n_embd);

        let pool = Arc::new(PagePool::new(
            model.n_heads_kv as usize,
            model.head_dim   as usize,
            total_kv_pages,
        ));
        eprintln!("[engine] PagePool: {} pages × {} bytes/page",
            total_kv_pages,
            2 * PAGE_TOKENS * model.n_heads_kv as usize * model.head_dim as usize,
        );

        let n_physical_cores = num_cpus::get_physical();
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n_physical_cores)
            .build_global();
        eprintln!("[engine] Rayon pool: {} threads (physical cores)", n_physical_cores);

        let scheduler = Scheduler::new(Arc::clone(&pool));

        Ok(AegisEngine {
            model,
            pool,
            scheduler,
        })
    }

    pub fn add_sequence(&mut self, seq_id: u32, prompt_tokens: Vec<u32>, max_new_tokens: u32) {
        self.scheduler.add_request(SequenceRequest {
            seq_id,
            prompt_tokens,
            max_new_tokens,
        });
    }

    /// Advances all running sequences by one token.
    /// Returns a list of (seq_id, generated_token_id).
    pub fn step_forward(&mut self) -> Vec<(u32, u32)> {
        self.scheduler.promote_waiting();

        let running = self.scheduler.running_sequences();
        if running.is_empty() {
            return Vec::new();
        }

        let outputs = self.step_batch(running);

        for (seq_id, token) in &outputs {
            if let Some(seq) = self.scheduler.active_seqs.iter_mut().find(|s| s.seq_id == *seq_id) {
                if (seq.num_tokens as usize) < seq.prompt_tokens.len() - 1 {
                    // Prefill phase: advance position, do not append generated token
                    seq.num_tokens += 1;
                } else {
                    // Generation phase: append new token
                    seq.prompt_tokens.push(*token);
                    seq.num_tokens += 1;
                    seq.generated_tokens += 1;
                }
            }
        }

        // Filter outputs to only return generated tokens, not prefill dummy tokens
        let filtered_outputs: Vec<(u32, u32)> = outputs.into_iter().filter(|(seq_id, _)| {
            if let Some(seq) = self.scheduler.active_seqs.iter().find(|s| s.seq_id == *seq_id) {
                seq.generated_tokens > 0
            } else {
                false
            }
        }).collect();

        self.scheduler.post_step_cleanup();
        filtered_outputs
    }

    fn step_batch(&self, sequences: &[SequenceState]) -> Vec<(u32, u32)> {
        let n_embd   = self.model.n_embd as usize;
        let n_layers = self.model.n_layers as usize;
        let n_heads  = self.model.n_heads as usize;
        let head_dim = self.model.head_dim as usize;
        let n_heads_kv = self.model.n_heads_kv as usize;

        let mut hidden_states: Vec<Vec<f32>> = sequences
            .iter()
            .map(|seq| self.embed_current_token(seq))
            .collect();

        for layer_idx in 0..n_layers {
            let layer_name_prefix = format!("blk.{}.", layer_idx);

            let wq = self.get_tensor(&format!("{}attn_q.weight",      layer_name_prefix));
            let wk = self.get_tensor(&format!("{}attn_k.weight",      layer_name_prefix));
            let wv = self.get_tensor(&format!("{}attn_v.weight",      layer_name_prefix));
            let wo = self.get_tensor(&format!("{}attn_output.weight", layer_name_prefix));
            let wgate = self.get_tensor(&format!("{}ffn_gate.weight", layer_name_prefix));
            let wup   = self.get_tensor(&format!("{}ffn_up.weight",   layer_name_prefix));
            let wdown = self.get_tensor(&format!("{}ffn_down.weight", layer_name_prefix));
            let w_attn_norm = self.get_norm_tensor(&format!("{}attn_norm.weight", layer_name_prefix));
            let w_ffn_norm  = self.get_norm_tensor(&format!("{}ffn_norm.weight", layer_name_prefix));

            let new_hidden: Vec<Vec<f32>> = hidden_states
                .par_iter()
                .zip(sequences.par_iter())
                .map(|(hidden, seq)| {
                    let mut normed_hidden = hidden.clone();
                    rms_norm_inplace(&mut normed_hidden, w_attn_norm);

                    let q_raw = self.ternary_proj(&normed_hidden, wq, n_embd);
                    let k_raw = self.ternary_proj(&normed_hidden, wk, n_heads_kv * head_dim);
                    let v_raw = self.ternary_proj(&normed_hidden, wv, n_heads_kv * head_dim);

                    let pos = seq.num_tokens as usize;
                    let q_rope = apply_rope(&q_raw, pos, n_heads, head_dim);
                    let k_rope = apply_rope(&k_raw, pos, n_heads_kv, head_dim);

                    let page_idx = seq.physical_page(pos);
                    let slot     = seq.slot_in_page(pos);

                    unsafe {
                        let k_scale = absmax_scale(&k_rope);
                        for (head_idx, k_head) in k_rope.chunks(head_dim).enumerate() {
                            let k_ptr = self.pool.slab.add(
                                page_idx as usize * self.pool.page_stride
                                + slot * n_heads_kv * head_dim
                                + head_idx * head_dim,
                            );
                            for (d, &kv) in k_head.iter().enumerate() {
                                *k_ptr.add(d) = (kv / k_scale * 127.0).clamp(-127.0, 127.0) as i8;
                            }
                        }

                        let v_scale = absmax_scale(&v_raw);
                        let v_section = self.pool.page_stride / 2;
                        for (head_idx, v_head) in v_raw.chunks(head_dim).enumerate() {
                            let v_ptr = self.pool.slab.add(
                                page_idx as usize * self.pool.page_stride
                                + v_section
                                + slot * n_heads_kv * head_dim
                                + head_idx * head_dim,
                            );
                            for (d, &vv) in v_head.iter().enumerate() {
                                *v_ptr.add(d) = (vv / v_scale * 127.0).clamp(-127.0, 127.0) as i8;
                            }
                        }
                    }

                    let attn_out: Vec<f32> = (0..n_heads)
                        .into_par_iter()
                        .flat_map(|head_idx| {
                            let q_head = &q_rope[head_idx * head_dim..(head_idx + 1) * head_dim];
                            unsafe {
                                flash_attn_paged(
                                    q_head,
                                    &self.pool,
                                    &seq.block_table,
                                    seq.num_tokens as usize + 1,
                                    head_dim,
                                    n_heads,
                                    head_idx,
                                )
                            }
                        })
                        .collect();

                    let attn_out_i8: Vec<i8> = f32_to_i8_absmax(&attn_out);
                    let proj_out = self.ternary_proj_i8(&attn_out_i8, wo, n_embd);

                    let mut post_attn: Vec<f32> = hidden
                        .iter()
                        .zip(proj_out.iter())
                        .map(|(h, p)| h + p)
                        .collect();

                    let mut normed_post_attn = post_attn.clone();
                    rms_norm_inplace(&mut normed_post_attn, w_ffn_norm);

                    let post_attn_i8 = f32_to_i8_absmax(&normed_post_attn);
                    let gate_raw = self.ternary_proj_i8(&post_attn_i8, wgate, wgate.rows);
                    let up_raw   = self.ternary_proj_i8(&post_attn_i8, wup,   wup.rows);

                    let gated: Vec<f32> = gate_raw
                        .iter()
                        .zip(up_raw.iter())
                        .map(|(g, u)| silu(*g) * u)
                        .collect();

                    let gated_i8 = f32_to_i8_absmax(&gated);
                    let ffn_out  = self.ternary_proj_i8(&gated_i8, wdown, n_embd);

                    post_attn
                        .iter()
                        .zip(ffn_out.iter())
                        .map(|(a, f)| a + f)
                        .collect()
                })
                .collect();

            hidden_states = new_hidden;
        }

        let w_out_norm = self.get_norm_tensor("output_norm.weight");

        hidden_states
            .iter()
            .zip(sequences.iter())
            .map(|(hidden, seq)| {
                let mut normed = hidden.clone();
                rms_norm_inplace(&mut normed, w_out_norm);

                let mut logits = if let Some(lm_head) = self.model.tensors.get("output.weight") {
                    let normed_i8 = f32_to_i8_absmax(&normed);
                    self.ternary_proj_i8(&normed_i8, lm_head, lm_head.rows)
                } else {
                    // Fallback to tied embeddings using native portable_simd f32x8 dot product
                    let vocab_size = self.model.embed_table.len() / self.model.n_embd as usize;
                    let mut fallback_logits = vec![0.0f32; vocab_size];
                    let n_embd = self.model.n_embd as usize;
                    
                    fallback_logits.par_iter_mut().enumerate().for_each(|(v, logit)| {
                        use std::simd::num::SimdFloat;
                        use std::simd::f32x8;
                        
                        let offset = v * n_embd;
                        let mut sum_vec = f32x8::splat(0.0);
                        let mut i = 0;
                        
                        // Process 8 floats at a time using AVX2/NEON intrinsics
                        while i + 8 <= n_embd {
                            let a = f32x8::from_slice(&normed[i..i+8]);
                            let b = f32x8::from_slice(&self.model.embed_table[offset + i .. offset + i + 8]);
                            sum_vec += a * b;
                            i += 8;
                        }
                        
                        let mut sum = sum_vec.reduce_sum();
                        // Handle remainder
                        while i < n_embd {
                            sum += normed[i] * self.model.embed_table[offset + i];
                            i += 1;
                        }
                        *logit = sum;
                    });
                    fallback_logits
                };
                
                debug_assert!(logits.iter().all(|x| x.is_finite()));

                // Apply repetition penalty (1.15 is standard)
                let token = argmax_f32(&mut logits, &seq.prompt_tokens, 1.25);
                (seq.seq_id, token)
            })
            .collect()
    }

    fn get_tensor(&self, name: &str) -> &crate::gguf::weight_loader::BitmaskTensor {
        self.model.tensors.get(name)
            .unwrap_or_else(|| panic!("[engine] missing tensor: {}", name))
    }

    fn get_norm_tensor(&self, name: &str) -> &[f32] {
        self.model.norm_tensors.get(name)
            .map(|v| v.as_slice())
            .unwrap_or_else(|| panic!("[engine] missing norm tensor: {}", name))
    }

    fn ternary_proj(
        &self,
        x:       &[f32],
        w:       &crate::gguf::weight_loader::BitmaskTensor,
        out_dim: usize,
    ) -> Vec<f32> {
        let x_i8 = f32_to_i8_absmax(x);
        self.ternary_proj_i8(&x_i8, w, out_dim)
    }

    fn ternary_proj_i8(
        &self,
        x:       &[i8],
        w:       &crate::gguf::weight_loader::BitmaskTensor,
        out_dim: usize,
    ) -> Vec<f32> {
        let vt = vtable();
        let words_per_row = w.cols / 64;

        (0..out_dim)
            .into_par_iter()
            .map(|row| {
                let pm = unsafe { w.pos_mask.add(row * words_per_row) };
                let nm = unsafe { w.neg_mask.add(row * words_per_row) };
                let raw_dot = unsafe {
                    (vt.ternary_dot)(x.as_ptr(), pm, nm, w.cols)
                };
                let block_offset = (row * w.cols_raw) / 32;
                let scale = w.scales.get(block_offset).copied().unwrap_or(1.0);
                raw_dot as f32 * scale
            })
            .collect()
    }

    fn embed_current_token(&self, seq: &SequenceState) -> Vec<f32> {
        let n_embd = self.model.n_embd as usize;
        let token_idx = (seq.num_tokens as usize).min(seq.prompt_tokens.len().saturating_sub(1));
        let token  = seq.prompt_tokens[token_idx] as usize;
            
        let start = token * n_embd;
        let end = start + n_embd;
        
        if end <= self.model.embed_table.len() {
            self.model.embed_table[start..end].to_vec()
        } else {
            vec![0.0f32; n_embd]
        }
    }
}

#[inline]
fn f32_to_i8_absmax(x: &[f32]) -> Vec<i8> {
    use std::simd::num::SimdFloat;
    use std::simd::{f32x8, i8x8, Simd};

    let absmax = absmax_scale(x);
    if absmax < 1e-9 { return vec![0i8; x.len()]; }
    
    let scale = 127.0 / absmax;
    let scale_vec = f32x8::splat(scale);
    let mut out = vec![0i8; x.len()];
    
    let mut i = 0;
    let min_vec = f32x8::splat(-127.0);
    let max_vec = f32x8::splat(127.0);

    while i + 8 <= x.len() {
        let v = f32x8::from_slice(&x[i..i+8]);
        let scaled = (v * scale_vec).simd_clamp(min_vec, max_vec);
        let int_vals: Simd<i32, 8> = scaled.cast();
        // Pack i32 to i8. Since values are bounded [-127, 127], safe to cast.
        let i8_arr: [i8; 8] = [
            int_vals[0] as i8, int_vals[1] as i8, int_vals[2] as i8, int_vals[3] as i8,
            int_vals[4] as i8, int_vals[5] as i8, int_vals[6] as i8, int_vals[7] as i8,
        ];
        out[i..i+8].copy_from_slice(&i8_arr);
        i += 8;
    }
    while i < x.len() {
        out[i] = (x[i] * scale).clamp(-127.0, 127.0) as i8;
        i += 1;
    }
    out
}

#[inline]
fn absmax_scale(x: &[f32]) -> f32 {
    use std::simd::num::SimdFloat;
    use std::simd::f32x8;

    let mut max_vec = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = f32x8::from_slice(&x[i..i+8]).abs();
        max_vec = max_vec.simd_max(v);
        i += 8;
    }
    let mut absmax = max_vec.reduce_max();
    while i < x.len() {
        if x[i].abs() > absmax { absmax = x[i].abs(); }
        i += 1;
    }
    absmax.max(1e-9)
}

#[inline]
fn rms_norm_inplace(x: &mut [f32], w: &[f32]) {
    use std::simd::num::SimdFloat;
    use std::simd::f32x8;

    let mut sum_vec = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = f32x8::from_slice(&x[i..i+8]);
        sum_vec += v * v;
        i += 8;
    }
    let mut sum = sum_vec.reduce_sum();
    while i < x.len() {
        sum += x[i] * x[i];
        i += 1;
    }

    let rms = (sum / x.len() as f32 + 1e-5).sqrt();
    let scale = 1.0 / rms;
    let scale_vec = f32x8::splat(scale);

    let mut j = 0;
    while j + 8 <= x.len() {
        let v = f32x8::from_slice(&x[j..j+8]);
        let weights = if j + 8 <= w.len() {
            f32x8::from_slice(&w[j..j+8])
        } else {
            f32x8::splat(1.0) // Fallback for weights if missing
        };
        let out = (v * scale_vec) * weights;
        out.copy_to_slice(&mut x[j..j+8]);
        j += 8;
    }
    while j < x.len() {
        x[j] = (x[j] * scale) * w.get(j).copied().unwrap_or(1.0);
        j += 1;
    }
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn apply_rope(x: &[f32], pos: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    let theta_base = 10000.0f32;

    for head in 0..n_heads {
        let offset = head * head_dim;
        for i in 0..(head_dim / 2) {
            let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = out[offset + i];
            let x1 = out[offset + i + head_dim / 2];
            out[offset + i]               = x0 * cos_a - x1 * sin_a;
            out[offset + i + head_dim / 2] = x0 * sin_a + x1 * cos_a;
        }
    }
    out
}
