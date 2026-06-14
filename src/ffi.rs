use crate::engine::AegisEngine;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aegis_init(
    model_path: *const c_char,
    total_kv_pages: usize,
) -> *mut c_void {
    let c_str = unsafe {
        assert!(!model_path.is_null());
        CStr::from_ptr(model_path)
    };
    let path = c_str.to_str().expect("Invalid UTF-8 in model path");

    match AegisEngine::new(path, total_kv_pages) {
        Ok(engine) => Box::into_raw(Box::new(engine)) as *mut c_void,
        Err(e) => {
            eprintln!("[ffi] aegis init failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aegis_add_sequence(
    engine_ptr: *mut c_void,
    seq_id: u32,
    prompt: *const c_char,
    max_new_tokens: u32,
) -> i32 {
    let engine = unsafe { &mut *(engine_ptr as *mut AegisEngine) };

    let c_str = unsafe {
        assert!(!prompt.is_null());
        CStr::from_ptr(prompt)
    };
    let text = c_str.to_str().expect("Invalid UTF-8 in prompt");

    match engine.model.tokenizer.encode(text) {
        Ok(tokens) => {
            engine.add_sequence(seq_id, tokens, max_new_tokens);
            0 // Success
        }
        Err(e) => {
            eprintln!("[ffi] tokenization failed: {}", e);
            -1 // Error
        }
    }
}

#[repr(C)]
pub struct StepResult {
    pub seq_id: u32,
    pub token_text: *mut c_char,
}

#[repr(C)]
pub struct StepBatchResult {
    pub results: *mut StepResult,
    pub count: usize,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aegis_step_forward(engine_ptr: *mut c_void) -> StepBatchResult {
    let engine = unsafe { &mut *(engine_ptr as *mut AegisEngine) };
    let outputs = engine.step_forward();

    let mut ffi_results = Vec::with_capacity(outputs.len());
    for (seq_id, token_id) in outputs {
        let text = engine
            .model
            .tokenizer
            .decode_token(token_id)
            .trim_start_matches('▁');
        let c_str = CString::new(text).unwrap_or_else(|_| CString::new("").unwrap());
        ffi_results.push(StepResult {
            seq_id,
            token_text: c_str.into_raw(), // caller must free!
        });
    }

    let count = ffi_results.len();
    let ptr = ffi_results.as_mut_ptr();
    std::mem::forget(ffi_results); // give ownership to C

    StepBatchResult {
        results: ptr,
        count,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aegis_free_batch_result(batch: StepBatchResult) {
    if batch.count == 0 || batch.results.is_null() {
        return;
    }
    let results = unsafe { Vec::from_raw_parts(batch.results, batch.count, batch.count) };
    for res in results {
        if !res.token_text.is_null() {
            unsafe {
                let _ = CString::from_raw(res.token_text);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aegis_free(engine_ptr: *mut c_void) {
    if !engine_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(engine_ptr as *mut AegisEngine);
        }
    }
}
