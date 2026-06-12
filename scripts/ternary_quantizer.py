import os
import torch
import numpy as np

# ==============================================================================
# AEGIS TERNARY QUANTIZER (1.58-BIT BITNET B1.58)
# ==============================================================================
# This script takes standard FP16/FP32 neural network weights and crushes them 
# into absolute ternary states (-1, 0, 1) using the AbsMean quantization formula.
# It then packs 4 ternary weights into a single 8-bit byte (u8) to map directly
# into the Aegis AVX2 branchless bitmask separation engine.

def absmean_quantize(weight_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the official BitNet b1.58 quantization formula.
    1. Calculate the scaling factor (gamma) as the mean of absolute values.
    2. Divide weights by gamma.
    3. Clip between -1 and 1.
    4. Round to nearest integer (-1, 0, 1).
    """
    print(f"[*] Original FP16 Tensor Shape: {weight_tensor.shape}")
    
    # 1. Calculate scaling factor (gamma)
    gamma = weight_tensor.abs().mean().clamp(min=1e-5)
    
    # 2. Scale, Clip, and Round to -1, 0, 1
    quantized_weights = torch.round(torch.clamp(weight_tensor / gamma, -1.0, 1.0))
    
    print(f"[+] Quantization Complete. Unique values: {torch.unique(quantized_weights).tolist()}")
    return quantized_weights.to(torch.int8)

def pack_ternary_to_u8(ternary_tensor: torch.Tensor) -> np.ndarray:
    """
    Packs 4 ternary weights (-1, 0, 1) into a single unsigned 8-bit integer (u8).
    This reduces the memory footprint by 16x compared to FP32, allowing the 
    Aegis Rust engine to execute bitmask separation perfectly.
    
    Encoding map (2-bits per weight):
     0 (00) -> 0
     1 (01) -> +1
     2 (10) -> -1
    """
    flattened = ternary_tensor.flatten().numpy()
    
    # Pad array if not divisible by 4
    if len(flattened) % 4 != 0:
        padding = 4 - (len(flattened) % 4)
        flattened = np.pad(flattened, (0, padding), mode='constant', constant_values=0)
        
    print(f"[*] Packing {len(flattened)} ternary weights into {len(flattened)//4} bytes...")
    
    # Map to 2-bit unsigned logic: 0 -> 0, 1 -> 1, -1 -> 2
    mapped = np.zeros_like(flattened, dtype=np.uint8)
    mapped[flattened == 1] = 1
    mapped[flattened == -1] = 2
    
    # Pack 4 weights per byte using bit shifts
    # [W0, W1, W2, W3] -> (W0) | (W1 << 2) | (W2 << 4) | (W3 << 6)
    packed_u8 = np.zeros(len(mapped) // 4, dtype=np.uint8)
    
    for i in range(4):
        packed_u8 |= (mapped[i::4] << (i * 2))
        
    print(f"[+] Successfully packed. Compression Ratio: 16:1 vs FP32")
    return packed_u8

def main():
    print("============================================================")
    print("  AEGIS 1.58-BIT TERNARY QUANTIZER")
    print("============================================================")
    
    # For demonstration, we simulate a small linear layer weight matrix
    # Once the model is trained via HuggingFace, we pass the real state_dict here.
    vocab_size = 4096
    hidden_dim = 1024
    
    print(f"[*] Generating synthetic {hidden_dim}x{vocab_size} FP16 weight matrix for quantization test...")
    simulated_weights = torch.randn((hidden_dim, vocab_size), dtype=torch.float16)
    
    # Step 1: Quantize to Ternary
    ternary_weights = absmean_quantize(simulated_weights)
    
    # Step 2: Pack to 2-bit u8 for Aegis Rust Engine
    packed_binary = pack_ternary_to_u8(ternary_weights)
    
    # Step 3: Save to disk
    os.makedirs("models", exist_ok=True)
    output_path = "models/quantized_aegis_test.bin"
    packed_binary.tofile(output_path)
    
    print(f"[+] Saved Aegis-compatible binary to: {output_path}")
    print(f"[+] Total Disk Size: {len(packed_binary) / 1024 / 1024:.2f} MB")
    print("============================================================")

if __name__ == "__main__":
    main()
