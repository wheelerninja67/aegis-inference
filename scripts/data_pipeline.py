import os
import json
import asyncio
import aiohttp
from typing import List, Dict

# ==============================================================================
# AEGIS SYNTHETIC DATA INGESTION PIPELINE
# ==============================================================================
# This script blasts asynchronous requests to the Anthropic API to synthetically 
# generate high-quality, institutional-grade training data for your 1.58-bit model.

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# We force the model to output at the highest possible intellectual level.
SYSTEM_PROMPT = """You are an elite systems architect, quantitative researcher, and mathematician. 
Generate complex, step-by-step logical reasoning traces, bare-metal C/Rust systems programming challenges, 
and highly advanced mathematical proofs. The output must be perfectly formatted, highly analytical, 
and strictly factual for training a next-generation local LLM."""

# These are the seed prompts. Once you get the API keys, we will expand this to 10,000+ prompts.
SEED_PROMPTS = [
    "Explain the mathematical mechanics behind 1.58-bit ternary quantization (BitNet) and why it eliminates multiplication.",
    "Write a high-performance C program using AVX2 intrinsic functions (_mm256_load_si256) to calculate the dot product of two arrays.",
    "Derive the Kelly Criterion mathematically for sizing trades in a highly volatile Nasdaq 100 regime.",
    "Explain false sharing in CPU L3 cache and how to prevent it using memory struct alignment in Rust."
]

async def generate_synthetic_data(session: aiohttp.ClientSession, prompt: str) -> Dict:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": "claude-3-opus-20240229", # Can swap to sonnet-3.5 for speed
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                content = data["content"][0]["text"]
                return {
                    "instruction": prompt,
                    "output": content
                }
            else:
                error_msg = await response.text()
                print(f"[-] API Error: {response.status} - {error_msg}")
                return None
    except Exception as e:
        print(f"[-] Network Exception: {e}")
        return None

async def main():
    if not ANTHROPIC_API_KEY:
        print("[-] FATAL: ANTHROPIC_API_KEY environment variable not found.")
        print("[*] Waiting for Anthropic to approve the Claude For Good grant.")
        print("[*] Once approved, run: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    print("[*] Initiating Aegis Synthetic Data Ingestion Pipeline...")
    print(f"[*] Firing {len(SEED_PROMPTS)} asynchronous tasks to Anthropic API...")
    
    results = []
    # We use aiohttp to make concurrent non-blocking requests. 
    # This generates data 10x faster than a standard python loop.
    async with aiohttp.ClientSession() as session:
        tasks = [generate_synthetic_data(session, prompt) for prompt in SEED_PROMPTS]
        responses = await asyncio.gather(*tasks)
        
        for res in responses:
            if res:
                results.append(res)
                
    # Save to JSONL format. This is the exact format required by HuggingFace 
    # and llama.cpp when you actually train the model weights.
    os.makedirs("data", exist_ok=True)
    with open("data/aegis_synthetic_training_set.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"[+] Pipeline complete. Saved {len(results)} high-quality samples to data/aegis_synthetic_training_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
