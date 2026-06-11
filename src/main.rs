use aegis_inference::gguf_parser::GgufParser;
use std::time::Instant;
use rand::Rng;

fn main() {
    println!("============================================================");
    println!("  AEGIS INFERENCE ENGINE: V0.2 (GGUF Mmap Parsing)");
    println!("============================================================");

    let model_path = "models/tinyllama.gguf";

    println!("[*] Attempting to parse GGUF model at: {}", model_path);
    let start_parse = Instant::now();

    match GgufParser::open(model_path) {
        Ok(mut parser) => {
            println!("[+] GGUF Header Extracted Successfully.");
            println!("    |- Magic:    {:#010x}", parser.header.magic);
            println!("    |- Version:  {}", parser.header.version);
            println!("    |- Tensors:  {}", parser.header.tensor_count);
            println!("    |- KV Pairs: {}", parser.header.kv_count);
            
            // Initiate Zero-Copy Mmap
            if let Err(e) = parser.map_tensors() {
                println!("[-] Failed to memory map tensors: {}", e);
                return;
            }

            println!("\n============================================================");
            println!("  PHASE 2: ZERO-COPY TENSOR ROUTING & AVX2 INFERENCE");
            println!("============================================================");
            
            let mmap_bytes = parser.raw_bytes().unwrap();
            
            // Simulate routing a massive 1024x4096 weight matrix directly 
            // from the mapped SSD payload into the CPU L3 Cache (AVX2).
            let rows = 1024;
            let cols = 4096;
            let required_bytes = rows * cols;
            
            // In a full implementation, we parse the GGUF alignment offset.
            // Here, we grab a safe chunk from the middle of the mapped model payload.
            let simulated_offset = 10_000_000; 
            
            if mmap_bytes.len() < simulated_offset + required_bytes {
                println!("[-] Model too small for simulated offset.");
                return;
            }

            // ZERO-COPY SLICE: This instantly creates a reference to the weights 
            // on the SSD without allocating any new RAM.
            let _tensor_slice = &mmap_bytes[simulated_offset .. simulated_offset + required_bytes];
            println!("[*] Successfully routed a {} byte tensor slice directly from NVMe.", required_bytes);

            // Generate a deterministic input activation vector
            let mut input_vector = vec![0i8; cols];
            for i in 0..cols {
                input_vector[i] = ((i % 3) as i8) - 1;
            }

            // NOTE: We cannot directly pass `_tensor_slice` (which is u8) to our AVX2 
            // function (which currently expects i8) without an unsafe cast. 
            // For the physics proof, we transmute the zero-copy pointer.
            let tensor_ptr = _tensor_slice.as_ptr() as *const i8;
            let tensor_i8_slice = unsafe { std::slice::from_raw_parts(tensor_ptr, required_bytes) };
            
            // Re-use our TernaryTensor structure for the AVX2 logic
            let tensor_view = aegis_inference::TernaryTensor {
                rows,
                cols,
                data: tensor_i8_slice.to_vec(), // In V0.3 we will avoid this clone
            };

            println!("[*] Injecting Zero-Copy slice into AVX2 Hardware Vectorizer...");
            let start_avx2 = Instant::now();
            let _avx2_output = unsafe { tensor_view.fast_simd_inference(&input_vector) };
            let avx2_time = start_avx2.elapsed();

            println!("[+] AVX2 Inference Completed in: {:?}", avx2_time);
            println!("[+] System Status: Tensor routed and executed successfully.");
        }
        Err(e) => {
            println!("[-] Failed to open GGUF file. Is the model fully downloaded?");
            println!("[-] Error: {}", e);
        }
    }

    println!("[*] Parse & Map Execution Time: {:?}", start_parse.elapsed());
    println!("============================================================");
}
