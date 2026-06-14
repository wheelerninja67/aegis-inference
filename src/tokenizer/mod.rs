use crate::gguf::parser::{GgufParser, MetadataValue};
use tokenizers::{AddedToken, Tokenizer};

pub const EOS_TOKEN_ID: u32 = 2; // LLaMA-family universal EOS
pub const BOS_TOKEN_ID: u32 = 1;
pub const UNK_TOKEN_ID: u32 = 0;

pub struct AegisTokenizer {
    inner: Tokenizer,
    // Reverse map: token_id → string (for decoding SSE output to text)
    id_to_token: Vec<String>,
    pub vocab_size: u32,
}

impl AegisTokenizer {
    /// Load tokenizer directly from an open GgufParser.
    /// Call this inside AegisModel::load_gguf — no second file open needed.
    pub fn from_gguf(parser: &GgufParser) -> Result<Self, Box<dyn std::error::Error>> {
        let meta = &parser.header.metadata;

        // Extract vocab arrays from GGUF metadata
        let tokens: Vec<String> = match meta.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => return Err("missing tokenizer.ggml.tokens in GGUF metadata".into()),
        };

        let scores: Vec<f32> = match meta.get("tokenizer.ggml.scores") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::Float32(f) => *f,
                    _ => 0.0f32,
                })
                .collect(),
            _ => vec![0.0f32; tokens.len()],
        };

        let token_types: Vec<i32> = match meta.get("tokenizer.ggml.token_type") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::Int32(i) => *i,
                    _ => 1i32,
                })
                .collect(),
            _ => vec![1i32; tokens.len()],
        };

        let vocab_size = tokens.len() as u32;
        assert_eq!(tokens.len(), scores.len(), "token/score count mismatch");

        use std::collections::HashMap;
        use tokenizers::models::bpe::BpeBuilder;

        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_size as usize);
        let mut byte_fallback_map: HashMap<u8, String> = HashMap::new();

        for (id, token) in tokens.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);

            if token_types[id] == 6
                && let Some(byte_val) = parse_byte_token(token)
            {
                byte_fallback_map.insert(byte_val, token.clone());
            }
        }

        let merges = reconstruct_merges_from_scores(&tokens, &scores);

        let bpe = BpeBuilder::new()
            .vocab_and_merges(vocab.clone(), merges)
            .unk_token(tokens[UNK_TOKEN_ID as usize].clone())
            .byte_fallback(true)
            .build()
            .map_err(|e| format!("BPE build failed: {}", e))?;

        use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
        use tokenizers::pre_tokenizers::metaspace::Metaspace as MetaspacePreTokenizer;
        use tokenizers::pre_tokenizers::metaspace::PrependScheme;

        let mut tokenizer = Tokenizer::new(bpe);

        tokenizer.with_pre_tokenizer(tokenizers::PreTokenizerWrapper::Metaspace(
            MetaspacePreTokenizer::new('▁', PrependScheme::First, true),
        ));
        tokenizer.with_decoder(tokenizers::DecoderWrapper::Metaspace(
            MetaspaceDecoder::new('▁', PrependScheme::First, true),
        ));

        tokenizer.add_special_tokens(&[
            AddedToken::from(tokens[BOS_TOKEN_ID as usize].clone(), true),
            AddedToken::from(tokens[EOS_TOKEN_ID as usize].clone(), true),
        ]);

        let id_to_token = tokens;

        Ok(AegisTokenizer {
            inner: tokenizer,
            id_to_token,
            vocab_size,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("tokenize failed: {}", e))?;

        let mut ids = vec![BOS_TOKEN_ID]; // always prepend BOS
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    #[inline]
    pub fn decode_token(&self, token_id: u32) -> &str {
        self.id_to_token
            .get(token_id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    pub fn decode_sequence(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        self.inner
            .decode(ids, true)
            .map_err(|e| format!("decode failed: {}", e).into())
    }

    #[inline(always)]
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == EOS_TOKEN_ID
    }
}

fn parse_byte_token(s: &str) -> Option<u8> {
    let inner = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(inner, 16).ok()
}

fn reconstruct_merges_from_scores(tokens: &[String], scores: &[f32]) -> Vec<(String, String)> {
    let token_set: std::collections::HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();

    let mut merges: Vec<(String, String, f32)> = Vec::new();

    for (id, token) in tokens.iter().enumerate() {
        if token.len() < 2 {
            continue;
        }
        if token.starts_with('<') && token.ends_with('>') {
            continue;
        }

        for split_pos in 1..token.len() {
            if !token.is_char_boundary(split_pos) {
                continue;
            }
            let left = &token[..split_pos];
            let right = &token[split_pos..];
            if token_set.contains(left) && token_set.contains(right) {
                merges.push((left.to_string(), right.to_string(), scores[id]));
                break;
            }
        }
    }

    merges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    merges.into_iter().map(|(l, r, _)| (l, r)).collect()
}
