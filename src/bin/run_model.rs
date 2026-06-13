use aegis::engine::AegisEngine;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_demo = args.iter().any(|a| a == "--demo" || a == "run");

    let mut model_path = "models/tinyllama-1.1b-chat.Q8_0.gguf".to_string();

    if is_demo {
        println!("[*] Aegis Demo Mode Activated");
        std::fs::create_dir_all("models").unwrap_or(());
        let demo_path = "models/demo-1.58b.gguf";
        
        if !std::path::Path::new(demo_path).exists() {
            println!("[*] Demo model not found locally. Auto-downloading from Hugging Face...");
            println!("[*] (Note: Downloading 800MB via system curl...)");
            
            let status = std::process::Command::new("curl")
                .arg("-L")
                .arg("https://huggingface.co/wheelerninja67/aegis-demo/resolve/main/demo-1.58b.gguf")
                .arg("-o")
                .arg(demo_path)
                .status()
                .expect("Failed to execute curl command. Is curl installed?");
                
            if !status.success() {
                eprintln!("[-] Failed to download the demo model.");
                std::process::exit(1);
            }
        }
        model_path = demo_path.to_string();
    }

    println!("[*] Initializing Aegis Engine with model: {}", model_path);
    
    match AegisEngine::new(&model_path, 1024) {
        Ok(mut engine) => {
            println!("[+] Engine loaded successfully!");
            
            // Real ChatML prompt for TinyLlama
            let prompt_text = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";
            println!("\n[USER] What is the capital of France?");
            
            let prompt_tokens = engine.model.tokenizer.encode(prompt_text).expect("Tokenizer encoding failed");
            
            // Queue the sequence in the scheduler (Request ID 1, 128 max new tokens)
            engine.add_sequence(1, prompt_tokens, 128);
            
            print!("[AEGIS] ");
            std::io::stdout().flush().unwrap();
            
            let mut generated = 0;
            while generated < 128 {
                let outputs = engine.step_forward();
                
                if engine.scheduler.running_sequences().is_empty() {
                    break; // All sequences finished or stopped
                }
                
                for (_seq_id, token_id) in outputs {
                    if engine.model.tokenizer.is_eos(token_id) {
                        generated = 128; // force break
                        break;
                    }
                    let token_str = engine.model.tokenizer.decode_token(token_id);
                    // Replace the BPE Metaspace character ' ' (U+2581) with a normal space
                    let clean_str = token_str.replace(" ", " ").replace("<0x0A>", "\n");
                    print!("{}", clean_str);
                    std::io::stdout().flush().unwrap();
                }
                generated += 1;
            }
            println!("\n\n[+] Generation complete.");
        }
        Err(e) => {
            eprintln!("[-] Failed to load engine: {}", e);
        }
    }
}
