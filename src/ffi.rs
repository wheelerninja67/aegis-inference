use crate::TernaryTensor;
use std::slice;

/// Exposes the Aegis 1.58-bit Ternary inference core to C, C++, Python, and Zig.
/// 
/// `aegis_create_tensor` instantiates the tensor inside Rust's memory manager.
#[no_mangle]
pub extern "C" fn aegis_create_tensor(rows: usize, cols: usize, scale: f32) -> *mut TernaryTensor {
    let tensor = Box::new(TernaryTensor::new(rows, cols, scale));
    Box::into_raw(tensor)
}

/// Executes the AVX2 / NEON branchless bitmask kernel from a foreign language.
/// This bypasses Python's GIL and Zig's single-thread constraints, executing
/// the math natively in Rust's Rayon thread pool.
#[no_mangle]
pub unsafe extern "C" fn aegis_fast_inference(
    tensor_ptr: *const TernaryTensor,
    activations_ptr: *const i8,
    activations_len: usize,
    output_ptr: *mut f32,
) {
    if tensor_ptr.is_null() || activations_ptr.is_null() || output_ptr.is_null() {
        return; // Guard against null pointer panics
    }
    
    let tensor = &*tensor_ptr;
    let activations = slice::from_raw_parts(activations_ptr, activations_len);
    
    // Execute the zero-multiplication math kernel
    let result = tensor.fast_simd_inference(activations);
    
    // Copy the resulting f32 buffer back into the caller's memory space
    let output_slice = slice::from_raw_parts_mut(output_ptr, tensor.rows);
    output_slice.copy_from_slice(&result);
}

/// Safely drops the Rust tensor memory from a foreign caller to prevent memory leaks.
#[no_mangle]
pub unsafe extern "C" fn aegis_free_tensor(tensor_ptr: *mut TernaryTensor) {
    if !tensor_ptr.is_null() {
        let _ = Box::from_raw(tensor_ptr);
    }
}
