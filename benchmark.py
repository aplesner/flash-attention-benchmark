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
import torch.utils.benchmark as benchmark
from torch.nn.attention import SDPBackend
import time
from typing import Tuple, Dict, Optional
import warnings

# Check for flash_attn availability
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = Callable
    print("Warning: flash-attn not installed. Install with: pip install flash-attn")

from typing import Callable


def benchmark_forward(
    fn: Callable, *inputs, desc="", config_name: str="", verbose=False, autocast=True, amp_dtype=torch.float16, **kwinputs
) -> Tuple[benchmark.Timer, benchmark.Measurement]:
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
        label=config_name,
        sub_label=desc,
        description="Forward pass",
    )
    m = t.blocked_autorange(
        min_run_time=.2,  # Ensure at least 0.2 seconds of measurement
    )
    if verbose:
        print(m)
    return t, m


def regular_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                     dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
    """Standard attention implementation."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
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


def flash_attention_benchmark(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, device: str = 'cuda', config_name: str = ""
) -> Optional[Tuple[benchmark.Timer, benchmark.Measurement]]:
    """Benchmark Flash Attention if available."""
    if not FLASH_ATTN_AVAILABLE:
        print("Flash Attention not available. Please install flash-attn.")
        return None

    elif device not in ['cuda', 'cpu']:
        print(f"Unsupported device: {device}. Only 'cuda' and 'cpu' are supported.")
        return None

    else:
        try:
            return benchmark_forward(
                flash_attn_func, q, k, v, causal=causal, desc="Flash Attention Benchmark", config_name=config_name,
            )   
        except Exception as e:
            print(f"Error in Flash Attention benchmark:\n{str(e)}")
            return None


def pytorch_attention_benchmark(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, device: str = 'cuda', backends = (SDPBackend.FLASH_ATTENTION,), config_name: str = ""
) -> Optional[Tuple[benchmark.Timer, benchmark.Measurement]]:
    # Check backend compatibility
    for backend in backends:
        if backend not in [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]:
            print(f"Unsupported backend: {backend}.")
            return None

    if device == 'cuda':
        try:
            backend_names = ", ".join([backend.name for backend in backends])
            # print(f"Using PyTorch Attention with backends: {backend_names}")
            with torch.nn.attention.sdpa_kernel(*backends):
                return benchmark_forward(
                    F.scaled_dot_product_attention, q, k, v, is_causal=causal, desc=f"PyTorch ({backend_names}) Benchmark", config_name=config_name
                )
        except Exception as e:
            print(f"Error in PyTorch Flash Attention benchmark:\n{str(e)}")
            return None
    else:
        print("Error: CUDA required")
        return None


def regular_attention_benchmark(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, config_name: str = ""
) -> Optional[Tuple[benchmark.Timer, benchmark.Measurement]]:
    """Benchmark PyTorch's regular attention."""
    try:
        return benchmark_forward(
            regular_attention, q, k, v, causal=causal, desc="PyTorch Regular Attention Benchmark", config_name=config_name
        )
    except Exception as e:
        print(f"Error in PyTorch Regular Attention benchmark:\n{str(e)}")
        return None


def pytorch_attention_backends_benchmarks(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, device: str = 'cuda', config_name: str = ""
) -> Dict[str, Optional[Tuple[benchmark.Timer, benchmark.Measurement]]]:
    """Run benchmarks for all supported PyTorch attention backends."""
    results = {}

    for backend in [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]:
        results["PyTorch " + backend.name] = pytorch_attention_benchmark(
            q, k, v, causal=causal, device=device, backends=(backend,), config_name=config_name
        )
    
    return results


def run_benchmarks(batch_size: int = 2, seq_len: int = 2048, 
                  num_heads: int = 12, head_dim: int = 64,
                  dtype: torch.dtype = torch.float16, causal: bool = False) -> dict[str, Optional[Tuple[benchmark.Timer, benchmark.Measurement]]]:
    """Run all attention benchmarks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=dtype, device=device)
    # Copy to transposed for Flash Attention
    q_t = q.transpose(1, 2)  # [B, H, S, D] -> [B, S, H, D]
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    config_name = f"Batch={batch_size}, Seq={seq_len}, Heads={num_heads}, HeadDim={head_dim}, Dtype={dtype}, Causal={causal}"
    
    results = {}
    
    # 1. Flash Attention (standalone)
    results['Flash Attention'] = flash_attention_benchmark(q_t, k_t, v_t, causal=causal, device=device, config_name=config_name)
    
    # 2. PyTorch Regular Attention
    results['PyTorch Regular Attention'] = regular_attention_benchmark(q, k, v, causal=causal, config_name=config_name)

    # 3. PyTorch Attention with Different Backends
    results.update(pytorch_attention_backends_benchmarks(q, k, v, causal=causal, device=device, config_name=config_name))

    return results


def main() -> None:
    """Run benchmarks with different configurations."""
    print("Attention Benchmarks")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    else:
        print("Warning: CUDA not available, running on CPU")
    
    print()
    
    import itertools
    sequence_lengths = [512, 1024, 2048]
    batch_sizes = [2, 4]
    num_heads = [8, 12]
    head_dims = [64, 128]
    causal = [False, True]

    # Test configurations
    configs = itertools.product(
        batch_sizes, sequence_lengths, num_heads, head_dims, causal
    )
    # Convert to list of dicts for easier handling
    configs = [
        {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'dtype': torch.float16,  # Default dtype
            'causal': causal,  # Default causal setting
        }
        for batch_size, seq_len, num_heads, head_dim, causal in configs
    ]
    print(f"Running benchmarks for {len(configs)} configurations...")
    
    for config in configs[:5]:
        # print(f"\nConfig: Batch={config['batch_size']}, Seq={config['seq_len']}")
        # print("-" * 50)
        
        results = run_benchmarks(**config) # type: ignore
        print("Benchmark Results:")
        for name, measurement_timer in results.items():
            if measurement_timer is not None:
                measurement = measurement_timer[1]
            else:
                print(f"{name}: Not available")
                continue

            # print(f"{name}: {measurement.mean:.4f} seconds")

        # Collect measurements for comparison
        measurements = [measurement_timer[1] for measurement_timer in results.values() if measurement_timer is not None]
        # Only compare if multiple measurements are available
        if len(measurements) > 1:
            try:
                benchmark_results = benchmark.Compare(measurements)
                print(type(benchmark_results))
                print(benchmark_results)
            except Exception as e:
                print(f"Error during comparison: {e}")
                # Fallback: print individual measurement means
                for m in measurements:
                    print(f"{m.label}: {m.mean:.4f} seconds")
        else:
            print("Not enough measurements for comparison.")
        # Continue to next configuration

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()
