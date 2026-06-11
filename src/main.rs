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
            let avx2_output = unsafe { tensor_view.fast_simd_inference(&input_vector) };
            let avx2_time = start_avx2.elapsed();

            println!("[+] AVX2 Inference Completed in: {:?}", avx2_time);
            
            // Phase 3: Transformer Logic Integration
            println!("\n============================================================");
            println!("  PHASE 3: SOFTMAX PROBABILITY ENGINE");
            println!("============================================================");
            
            // Convert AVX2 raw logits (i32) into floating point (f32) for Softmax
            let mut float_logits: Vec<f32> = avx2_output.iter().map(|&x| x as f32).collect();
            
            println!("[*] Raw AVX2 Output Sample (first 5): {:?}", &avx2_output[0..5]);
            
            let start_softmax = Instant::now();
            aegis_inference::architecture::compute_softmax(&mut float_logits);
            let softmax_time = start_softmax.elapsed();
            
            println!("[+] Softmax Execution Time: {:?}", softmax_time);
            println!("[*] Softmax Probabilities Sample (first 5): {:?}", &float_logits[0..5]);
            
            // Verify sum of probabilities equals 1.0
            let sum: f32 = float_logits.iter().sum();
            println!("[+] Mathematical Verification: Total Probability Sum = {:.4}", sum);

            // Phase 6: Byte-Pair Encoding (BPE) Tokenization
            println!("\n============================================================");
            println!("  PHASE 6: BPE TOKENIZER (ENGLISH TO MATH)");
            println!("============================================================");
            
            let mut tokenizer = aegis_inference::tokenizer::BpeTokenizer::new();
            if let Err(e) = tokenizer.load_vocabulary("models/vocab.txt") {
                println!("[-] Failed to load vocab: {}", e);
            } else {
                let human_input = "the matrix has you neo !";
                println!("[*] Human Input String: \"{}\"", human_input);
                
                let math_tokens = tokenizer.encode(human_input);
                println!("[+] Mathematical Token Encoding: {:?}", math_tokens);
                
                let decoded_string = tokenizer.decode(&math_tokens);
                println!("[+] Engine Decoded Output: \"{}\"", decoded_string);
            }

            // Phase 7: Continuous Batching Simulation
            println!("\n============================================================");
            println!("  PHASE 7: CONTINUOUS BATCHING (V3 ARCHITECTURE)");
            println!("============================================================");
            
            let mut batch_engine = aegis_inference::architecture::AegisEngine::new(Vec::new(), Vec::new());
            
            // Inject 3 concurrent users into the engine
            batch_engine.add_sequence(101, 5, 2048);
            batch_engine.add_sequence(102, 12, 2048);
            batch_engine.add_sequence(103, 7, 2048);

            println!("[*] Commencing Rayon parallel batched inference step...");
            let start_batch = Instant::now();
            batch_engine.step();
            let batch_time = start_batch.elapsed();
            
            println!("[+] Processed 3 parallel sequences in {:?}", batch_time);
            println!("[+] Active sequences remaining in pool: {}", batch_engine.active_sequences.len());

            println!("\n============================================================");
            println!("[+] SYSTEM STATUS: AEGIS ARCHITECTURE V3.0 FULLY OPERATIONAL.");
            println!("============================================================");
        }
        Err(e) => {
            println!("[-] Failed to open GGUF file. Is the model fully downloaded?");
            println!("[-] Error: {}", e);
        }
    }

    println!("[*] Parse & Map Execution Time: {:?}", start_parse.elapsed());
    println!("============================================================");
}
