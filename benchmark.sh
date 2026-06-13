#!/usr/bin/env bash
set -e

echo "============================================="
echo " AEGIS vs LLAMA.CPP : BENCHMARK MATRIX TOOL  "
echo "============================================="

MODEL="models/demo-1.58b.gguf"
WEBHOOK_URL="REPLACE_WITH_YOUR_DISCORD_OR_API_WEBHOOK_URL"

if [ ! -f "$MODEL" ]; then
    echo "[-] Model $MODEL not found. Please run 'aegis run --demo' first to download it."
    exit 1
fi

echo "[*] Detecting Hardware Specs..."
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" = "Darwin" ]; then
    CPU=$(sysctl -n machdep.cpu.brand_string)
    RAM=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))"GB"
else
    CPU=$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
    RAM=$(free -h | awk '/^Mem:/ {print $2}')
fi

echo "[*] Benchmarking Aegis Inference Engine..."
# Time the engine run
start_time=$(date +%s%3N)
./target/release/aegis_inference --demo > aegis_bench.log 2>&1
end_time=$(date +%s%3N)
aegis_time=$((end_time - start_time))

echo "[*] Benchmarking llama.cpp (for comparison)..."
llama_time="0"
if [ ! -f "./llama-cli" ]; then
    echo "[-] llama-cli not found in current directory. Skipping llama.cpp baseline."
else
    start_time=$(date +%s%3N)
    ./llama-cli -m "$MODEL" -n 128 -p "What is the capital of France?" > llama_bench.log 2>&1
    end_time=$(date +%s%3N)
    llama_time=$((end_time - start_time))
fi

echo ""
echo "============================================="
echo "               BENCHMARK RESULTS             "
echo "============================================="
echo "Hardware: $CPU ($RAM, $OS $ARCH)"
echo "Aegis Total Time: ${aegis_time}ms"
echo "Llama Total Time: ${llama_time}ms"

if [ "$WEBHOOK_URL" != "REPLACE_WITH_YOUR_DISCORD_OR_API_WEBHOOK_URL" ]; then
    echo "[*] Transmitting anonymous benchmark data back to Aegis HQ..."
    curl -s -X POST -H "Content-Type: application/json" \
        -d "{\"content\": \"**New Aegis Benchmark**\\n**CPU:** $CPU\\n**OS:** $OS $ARCH\\n**RAM:** $RAM\\n**Aegis Time:** ${aegis_time}ms\\n**Llama Time:** ${llama_time}ms\"}" \
        $WEBHOOK_URL
    echo "[+] Data sent! Thank you for contributing."
else
    echo "[-] Webhook URL not configured. Results saved locally."
fi
