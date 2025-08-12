#!/usr/bin/env python3
"""
Benchmark comparison of different attention implementations:
1. Flash Attention (standalone)
2. PyTorch's Flash Attention
3. PyTorch regular attention
4. cuDNN attention (via PyTorch)
"""

import torch
import torch.nn.functional as F
import time
from typing import Tuple, Dict, Optional
import warnings

# Check for flash_attn availability
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not installed. Install with: pip install flash-attn")

from typing import Callable

def benchmark_function(func: Callable, *args, warmup: int = 10, iterations: int = 100, **kwargs) -> float:
    """Benchmark a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Actual benchmark
    start = time.time()
    for _ in range(iterations):
        func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    return (time.time() - start) / iterations


def regular_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                     dropout_p: float = 0.0, causal: bool = True) -> torch.Tensor:
    """Standard attention implementation."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Reshape for attention computation
    q = q.transpose(1, 2)  # [B, H, S, D]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)  # [B, S, H, D]


def run_benchmarks(batch_size: int = 2, seq_len: int = 2048, 
                  num_heads: int = 12, head_dim: int = 64,
                  dtype: torch.dtype = torch.float16) -> Dict[str, float]:
    """Run all attention benchmarks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    dtype=dtype, device=device)
    
    results = {}
    
    # 1. Flash Attention (standalone)
    if FLASH_ATTN_AVAILABLE and device == 'cuda':
        try:
            time_flash = benchmark_function(
                flash_attn_func, q, k, v, causal=True
            )
            results['Flash Attention (standalone)'] = time_flash
        except Exception as e:
            results['Flash Attention (standalone)'] = f"Error: {str(e)}"
    else:
        results['Flash Attention (standalone)'] = "Not available"
    
    # 2. PyTorch Flash Attention
    if device == 'cuda':
        q_t = q.transpose(1, 2)  # PyTorch expects [B, H, S, D]
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        try:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                time_pytorch_flash = benchmark_function(
                    F.scaled_dot_product_attention, q_t, k_t, v_t, is_causal=True
                )
                results['PyTorch Flash Attention'] = time_pytorch_flash
        except Exception as e:
            results['PyTorch Flash Attention'] = f"Error: {str(e)}"
    else:
        results['PyTorch Flash Attention'] = "CUDA required"
    
    # 3. PyTorch Regular Attention
    time_regular = benchmark_function(
        regular_attention, q, k, v, causal=True
    )
    results['PyTorch Regular'] = time_regular
    
    # 4. cuDNN (via PyTorch with math kernel)
    if device == 'cuda':
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,  # This typically uses cuDNN
                enable_mem_efficient=False
            ):
                time_cudnn = benchmark_function(
                    F.scaled_dot_product_attention, q_t, k_t, v_t, is_causal=True
                )
                results['cuDNN (via PyTorch)'] = time_cudnn
        except Exception as e:
            results['cuDNN (via PyTorch)'] = f"Error: {str(e)}"
    else:
        results['cuDNN (via PyTorch)'] = "CUDA required"
    
    return results


def main() -> None:
    """Run benchmarks with different configurations."""
    print("Attention Implementation Benchmarks")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    else:
        print("Warning: CUDA not available, running on CPU")
    
    print()
    
    # Test configurations
    configs = [
        {"seq_len": 512, "batch_size": 8, "dtype": torch.float16},
        {"seq_len": 1024, "batch_size": 4, "dtype": torch.float16},
        {"seq_len": 2048, "batch_size": 2, "dtype": torch.float16},
        {"seq_len": 4096, "batch_size": 1, "dtype": torch.float16},
        {"seq_len": 512, "batch_size": 8, "dtype": torch.bfloat16},
        {"seq_len": 1024, "batch_size": 4, "dtype": torch.bfloat16},
        {"seq_len": 2048, "batch_size": 2, "dtype": torch.bfloat16},
        {"seq_len": 4096, "batch_size": 1, "dtype": torch.bfloat16},
    ]
    
    for config in configs:
        print(f"\nConfig: Batch={config['batch_size']}, Seq={config['seq_len']}")
        print("-" * 50)
        
        results = run_benchmarks(**config)
        
        # Find the fastest implementation
        valid_times = {k: v for k, v in results.items() 
                      if isinstance(v, float)}
        
        if valid_times:
            fastest = min(valid_times.items(), key=lambda x: x[1])
            
            for impl, time_val in results.items():
                if isinstance(time_val, float):
                    speedup = time_val / fastest[1]
                    print(f"{impl:30s}: {time_val*1000:8.3f} ms ({speedup:.2f}x)")
                else:
                    print(f"{impl:30s}: {time_val}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()
