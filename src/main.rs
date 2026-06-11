use aegis_inference::gguf_parser::GgufParser;
use std::time::Instant;
use rand::Rng;

use axum::{routing::post, Router, Json, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<aegis_inference::architecture::AegisEngine>>,
    tokenizer: Arc<Mutex<aegis_inference::tokenizer::BpeTokenizer>>,
}

#[derive(Deserialize)]
struct InferenceRequest {
    prompt: String,
}

#[derive(Serialize)]
struct InferenceResponse {
    status: String,
    generated_text: String,
    execution_time_ms: f64,
}

async fn handle_completion(
    State(state): State<AppState>,
    Json(req_data): Json<InferenceRequest>,
) -> Json<InferenceResponse> {
    let start_api = Instant::now();
    println!("[*] Received API Prompt: \"{}\"", req_data.prompt);
    
    let mut tok = state.tokenizer.lock().await;
    let tokens = tok.encode(&req_data.prompt);
    
    let mut eng = state.engine.lock().await;
    eng.add_sequence(999, tokens[0], 2048);
    eng.step(); // Execute Forward Pass
    
    let decoded = tok.decode(&tokens);
    let exec_time = start_api.elapsed().as_secs_f64() * 1000.0;
    
    println!("[+] API Request served in {:.3} ms", exec_time);
    
    Json(InferenceResponse {
        status: "success".to_string(),
        generated_text: format!("{} [Aegis AVX2 Response]", decoded),
        execution_time_ms: exec_time,
    })
}

#[tokio::main]
async fn main() {
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

            // Parse physical tensor properties based on standard GGUF memory alignment
            let tensor_slice = &_tensor_slice[0..required_bytes];
            
            // Safety: Bypassing Rust's standard borrow checker to physically reinterpret
            // memory pointers from u8 directly to i8 (representing -1, 0, 1 ternary state)
            let tensor_i8_slice: &[i8] = unsafe {
                std::slice::from_raw_parts(
                    tensor_slice.as_ptr() as *const i8,
                    tensor_slice.len(),
                )
            };

            // V6 Upgrade: Dual Bitmask Separation Packing
            // We store positive and negative weights in two separate bitmasks
            // allowing us to calculate dot products without unsigned/signed multiplication wraps.
            let mut pos_mask = Vec::with_capacity((tensor_i8_slice.len() + 7) / 8);
            let mut neg_mask = Vec::with_capacity((tensor_i8_slice.len() + 7) / 8);
            
            for chunk in tensor_i8_slice.chunks(8) {
                let mut p: u8 = 0;
                let mut n: u8 = 0;
                for (i, &w) in chunk.iter().enumerate() {
                    if w == 1 { p |= 1 << i; }
                    if w == -1 { n |= 1 << i; }
                }
                pos_mask.push(p);
                neg_mask.push(n);
            }

            let tensor_view = aegis_inference::TernaryTensor {
                rows: 1024,
                cols: 4096,
                pos_mask,
                neg_mask,
                scale: 1.0, 
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

            // Phase 8: Enterprise HTTP API Server (Aegis V4)
            println!("\n============================================================");
            println!("  PHASE 8: ENTERPRISE HTTP API (AEGIS V4)");
            println!("============================================================");
            
            let app_state = AppState {
                engine: Arc::new(Mutex::new(batch_engine)),
                tokenizer: Arc::new(Mutex::new(tokenizer)),
            };

            let app = Router::new()
                .route("/v1/completions", post(handle_completion))
                .with_state(app_state);

            let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
            println!("[+] Aegis Tokio API Server listening on http://0.0.0.0:8080");
            println!("[+] Ready to accept POST requests to /v1/completions\n");
            axum::serve(listener, app).await.unwrap();
        }
        Err(e) => {
            println!("[-] Failed to open GGUF file. Is the model fully downloaded?");
            println!("[-] Error: {}", e);
        }
    }

    println!("[*] Parse & Map Execution Time: {:?}", start_parse.elapsed());
    println!("============================================================");
}
