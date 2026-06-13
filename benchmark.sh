#!/usr/bin/env bash
set -e

echo "============================================="
echo " AEGIS vs LLAMA.CPP : BENCHMARK MATRIX TOOL  "
echo "============================================="

MODEL="models/demo-1.58b.gguf"

if [ ! -f "$MODEL" ]; then
    echo "[-] Model $MODEL not found. Please run 'aegis run --demo' first to download it."
    exit 1
fi

echo "[*] Benchmarking Aegis Inference Engine..."
# We run aegis and time how long it takes to generate 128 tokens
# (Assuming aegis prints a performance metric at the end, or we use 'time')
time ./target/release/aegis_inference --demo > aegis_bench.log 2>&1

echo "[*] Benchmarking llama.cpp (for comparison)..."
if [ ! -f "./llama-cli" ]; then
    echo "[*] Downloading llama.cpp binary for benchmark..."
    # Note: In a real environment we would download the exact matching llama-cli binary
    echo "[-] llama-cli not found. Please compile llama.cpp and place 'llama-cli' in this directory."
else
    time ./llama-cli -m "$MODEL" -n 128 -p "What is the capital of France?" > llama_bench.log 2>&1
fi

echo ""
echo "============================================="
echo "               BENCHMARK RESULTS             "
echo "============================================="
echo "Analyze the logs in aegis_bench.log and llama_bench.log"
echo "Populate the Markdown table in README.md with the Tokens/Second and RAM usage."
