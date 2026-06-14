use crate::kv_cache::page_pool::{PagePool, PAGE_TOKENS};
use std::sync::Arc;

pub struct SequenceRequest {
    pub seq_id: u32,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: u32,
}

pub struct SequenceState {
    pub seq_id: u32,
    pub prompt_tokens: Vec<u32>,
    pub num_tokens: u32,
    pub max_new_tokens: u32,
    pub physical_pages: Vec<u32>,
    // The physical KV block table
    pub block_table: Vec<u32>,
    pub status: SequenceStatus,
    pub generated_tokens: u32,
}

impl SequenceState {
    pub fn virtual_len(&self) -> usize {
        self.num_tokens as usize
    }

    pub fn physical_page(&self, pos: usize) -> u32 {
        self.block_table[pos / PAGE_TOKENS]
    }

    pub fn slot_in_page(&self, pos: usize) -> usize {
        pos % PAGE_TOKENS
    }
}

#[derive(PartialEq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

pub struct Scheduler {
    pub active_seqs: Vec<SequenceState>,
    pub waiting_seqs: Vec<SequenceState>,
    pub page_pool: Arc<PagePool>,
    pub max_batch_size: usize,
}

impl Scheduler {
    pub fn new(pool: Arc<PagePool>) -> Self {
        Self {
            active_seqs: Vec::new(),
            waiting_seqs: Vec::new(),
            page_pool: pool,
            max_batch_size: 16,
        }
    }

    pub fn add_request(&mut self, req: SequenceRequest) {
        self.waiting_seqs.push(SequenceState {
            seq_id: req.seq_id,
            num_tokens: req.prompt_tokens.len() as u32,
            prompt_tokens: req.prompt_tokens,
            max_new_tokens: req.max_new_tokens,
            physical_pages: Vec::new(),
            block_table: Vec::new(),
            status: SequenceStatus::Waiting,
            generated_tokens: 0,
        });
    }

    pub fn promote_waiting(&mut self) {
        while self.active_seqs.len() < self.max_batch_size && !self.waiting_seqs.is_empty() {
            let seq = &self.waiting_seqs[0];
            let required_pages = (seq.num_tokens as usize).div_ceil(PAGE_TOKENS);
            let prefetch_pages = required_pages + 1;

            let mut mapped_pages = Vec::with_capacity(prefetch_pages);
            let mut oom = false;
            for _ in 0..prefetch_pages {
                if let Some(p) = self.page_pool.alloc_page() {
                    mapped_pages.push(p);
                } else {
                    oom = true;
                    break;
                }
            }
            if oom {
                for p in mapped_pages {
                    self.page_pool.free_page(p);
                }
                break;
            }

            let mut seq = self.waiting_seqs.remove(0);
            seq.physical_pages = mapped_pages.clone();
            seq.block_table = mapped_pages;
            seq.status = SequenceStatus::Running;
            self.active_seqs.push(seq);
        }
    }

    pub fn running_sequences(&self) -> &[SequenceState] {
        &self.active_seqs
    }

    pub fn post_step_cleanup(&mut self) -> Vec<u32> {
        let mut finished = Vec::new();
        let mut i = 0;
        while i < self.active_seqs.len() {
            let seq = &mut self.active_seqs[i];
            
            if seq.generated_tokens >= seq.max_new_tokens {
                seq.status = SequenceStatus::Finished;
            }

            if seq.status == SequenceStatus::Finished {
                let seq = self.active_seqs.remove(i);
                for p in seq.physical_pages {
                    self.page_pool.free_page(p);
                }
                finished.push(seq.seq_id);
            } else {
                let required_pages = (seq.num_tokens as usize + 1).div_ceil(PAGE_TOKENS);
                if required_pages > seq.physical_pages.len() {
                    if let Some(p) = self.page_pool.alloc_page() {
                        seq.physical_pages.push(p);
                        seq.block_table.push(p);
                    } else {
                        // Preempt
                        let mut seq = self.active_seqs.remove(i);
                        for p in seq.physical_pages.drain(..) {
                            self.page_pool.free_page(p);
                        }
                        seq.block_table.clear();
                        seq.status = SequenceStatus::Waiting;
                        self.waiting_seqs.insert(0, seq);
                        continue;
                    }
                }
                i += 1;
            }
        }
        finished
    }
}
