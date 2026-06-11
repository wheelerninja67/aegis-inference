use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Bare-metal Byte-Pair Encoding (BPE) Tokenizer for Llama 3 architectures.
/// This translates human language into mathematical integers the AVX2 engine can compute,
/// and decodes the AVX2 probabilities back into human language.
pub struct BpeTokenizer {
    /// Maps a token ID to its string representation (Decoding)
    id_to_token: HashMap<u32, String>,
    /// Maps a string piece to its token ID (Encoding)
    token_to_id: HashMap<String, u32>,
}

impl BpeTokenizer {
    /// Initializes an empty Tokenizer.
    pub fn new() -> Self {
        Self {
            id_to_token: HashMap::new(),
            token_to_id: HashMap::new(),
        }
    }

    /// Loads the tokenizer vocabulary from a file (e.g., tokenizer.model or raw text).
    /// In a production system, this reads the GGUF KV metadata to extract the vocab.
    pub fn load_vocabulary<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut current_id = 0;
        for line in reader.lines() {
            let token = line?;
            // Llama 3 BPE uses specific byte representations for spaces, 
            // but for this proof of concept, we map strings directly.
            self.id_to_token.insert(current_id, token.clone());
            self.token_to_id.insert(token, current_id);
            current_id += 1;
        }

        println!("[+] Tokenizer Vocabulary Loaded: {} tokens mapped.", current_id);
        Ok(())
    }

    /// Translates a raw English string into a sequence of mathematical token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        // A full BPE implementation uses greedy longest-match or merge rules.
        // For the physics PoC, we split by whitespace as a naive fallback.
        for word in text.split_whitespace() {
            if let Some(&id) = self.token_to_id.get(word) {
                tokens.push(id);
            } else {
                // Return an Unknown <UNK> token if not found
                tokens.push(0); 
            }
        }
        tokens
    }

    /// Translates the mathematical output from the Softmax engine back into English.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut decoded_text = String::new();
        for &id in token_ids {
            if let Some(token) = self.id_to_token.get(&id) {
                decoded_text.push_str(token);
                decoded_text.push(' '); // Add spacing
            }
        }
        decoded_text.trim_end().to_string()
    }
}
