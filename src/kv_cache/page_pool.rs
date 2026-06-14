use std::alloc::{alloc_zeroed, Layout};
use std::sync::atomic::{AtomicU64, Ordering};

/// 16 tokens per page.
/// Chosen so one page = 16 x num_heads x head_dim x 2 (K+V) x sizeof(i8)
/// For a typical 32-head, 128-dim model: 16 x 32 x 128 x 2 = 131,072 bytes = 128 KB per page.
/// Adjust PAGE_TOKENS to fit within your L2 size.
pub const PAGE_TOKENS: usize = 16;

/// A single physical KV page.
/// Layout: [K data for PAGE_TOKENS tokens][V data for PAGE_TOKENS tokens]
/// K data: PAGE_TOKENS x num_heads x head_dim x sizeof(i8)
/// Both K and V stored as quantized i8 to halve bandwidth.
#[repr(C)]
pub struct KVPage {
    /// Raw byte slab. Size: 2 x PAGE_TOKENS x num_heads x head_dim
    pub data: *mut i8,
    /// How many token slots are currently filled (0..PAGE_TOKENS)
    pub used_slots: u8,
    _pad: [u8; 7],
}

unsafe impl Send for KVPage {}
unsafe impl Sync for KVPage {}

/// The physical page pool. One global instance per inference server.
pub struct PagePool {
    /// Flat slab: all pages laid out contiguously.
    /// Allocated once at startup with huge-page alignment (2MB).
    pub slab: *mut i8,
    pub slab_len: usize,
    /// bytes per page
    pub page_stride: usize,
    pub total_pages: usize,
    /// Free-list as a bit-packed array. Bit i = 1 -> page i is free.
    /// AtomicU64 allows lock-free alloc/free from concurrent request handlers.
    free_bitmap: Box<[AtomicU64]>,
}

unsafe impl Send for PagePool {}
unsafe impl Sync for PagePool {}

impl PagePool {
    /// Allocate the pool. Call once at server startup.
    pub fn new(num_heads: usize, head_dim: usize, total_pages: usize) -> Self {
        let page_stride = {
            let raw = 2 * PAGE_TOKENS * num_heads * head_dim;
            // Pad to 64-byte cache-line boundary
            (raw + 63) & !63
        };
        let slab_len = page_stride * total_pages;

        // Align to 2MB for transparent huge-page eligibility (THP)
        let layout = Layout::from_size_align(slab_len, 2 * 1024 * 1024)
            .expect("layout failure");

        let slab = unsafe { alloc_zeroed(layout) as *mut i8 };
        assert!(!slab.is_null(), "page pool allocation failed");

        let bitmap_words = total_pages.div_ceil(64);
        let mut free_bitmap: Vec<AtomicU64> = Vec::with_capacity(bitmap_words);
        
        // Mark all pages as free (bit = 1)
        for i in 0..bitmap_words {
            let remaining = total_pages.saturating_sub(i * 64);
            let bits = if remaining >= 64 { u64::MAX } else { (1u64 << remaining) - 1 };
            free_bitmap.push(AtomicU64::new(bits));
        }

        PagePool {
            slab,
            slab_len,
            page_stride,
            total_pages,
            free_bitmap: free_bitmap.into_boxed_slice(),
        }
    }

    /// Allocate one physical page. Returns physical page index or None if pool exhausted.
    /// Lock-free: uses CAS loop on the bitmap.
    #[inline]
    pub fn alloc_page(&self) -> Option<u32> {
        for (word_idx, atomic_word) in self.free_bitmap.iter().enumerate() {
            let mut word = atomic_word.load(Ordering::Relaxed);
            loop {
                if word == 0 { break; } // no free pages in this word
                let bit_idx = word.trailing_zeros() as usize; // index of first free page
                let new_word = word & !(1u64 << bit_idx); // clear the bit (mark as used)
                match atomic_word.compare_exchange_weak(
                    word, new_word, Ordering::AcqRel, Ordering::Relaxed
                ) {
                    Ok(_) => {
                        let page_idx = word_idx * 64 + bit_idx;
                        if page_idx < self.total_pages {
                            return Some(page_idx as u32);
                        }
                        return None;
                    }
                    Err(actual) => word = actual, // retry with updated word
                }
            }
        }
        None // pool exhausted
    }

    /// Free a physical page back to the pool.
    #[inline]
    pub fn free_page(&self, page_idx: u32) {
        let page_idx = page_idx as usize;
        debug_assert!(page_idx < self.total_pages);
        let word_idx = page_idx / 64;
        let bit_idx = page_idx % 64;
        self.free_bitmap[word_idx].fetch_or(1u64 << bit_idx, Ordering::Release);
    }
}

/// Per-sequence state tracked by the scheduler
pub struct SequenceState {
    pub seq_id: u32,
    /// Logical -> physical page map. logical_block[i] = physical page index.
    pub block_table: Vec<u32>,
    /// Total tokens generated so far in this sequence
    pub num_tokens: u32,
    /// Ring cursor: for sliding-window attention, which logical block is "oldest"
    pub ring_head: u32,
}

impl SequenceState {
    /// Get the physical page index for logical token position `pos`
    #[inline]
    pub fn physical_page(&self, pos: usize) -> u32 {
        let logical_block = pos / PAGE_TOKENS;
        self.block_table[logical_block]
    }

    /// Get the slot within a page for token position `pos`
    #[inline]
    pub fn slot_in_page(&self, pos: usize) -> usize {
        pos % PAGE_TOKENS
    }
}
