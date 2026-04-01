"""
GPU profiling harness.

Measures:
  - Per-token latency (mean, std, p50, p99) via repeated generation
  - Peak GPU memory allocation and KV-cache footprint
  - PyTorch Profiler trace (Chrome JSON)
  - Roofline model: arithmetic intensity, memory/compute bound classification

Usage:
    from src.profiling.harness import profile_model
    result = profile_model(model, tokenizer, "Llama-2-7B", "GPTQ_INT4", bits=4)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU hardware specs (for Roofline)
# ---------------------------------------------------------------------------

GPU_SPECS: dict[str, dict] = {
    "A100": {
        "peak_flops_fp16_tflops": 312.0,
        "memory_bw_tbps"        : 2.0,
    },
    "A6000": {
        "peak_flops_fp16_tflops": 154.0,
        "memory_bw_tbps"        : 0.768,
    },
    "H100": {
        "peak_flops_fp16_tflops": 989.0,
        "memory_bw_tbps"        : 3.35,
    },
    "V100": {
        "peak_flops_fp16_tflops": 125.0,
        "memory_bw_tbps"        : 0.9,
    },
    "T4": {
        "peak_flops_fp16_tflops": 65.0,
        "memory_bw_tbps"        : 0.32,
    },
}


def detect_gpu_specs() -> dict:
    if not torch.cuda.is_available():
        return {"name": "CPU", "peak_flops_fp16_tflops": 1.0, "memory_bw_tbps": 0.05}
    name = torch.cuda.get_device_name(0)
    for key, specs in GPU_SPECS.items():
        if key in name:
            return {"name": name, **specs}
    # Unknown GPU — return conservative defaults
    logger.warning(f"Unknown GPU: {name}. Using conservative defaults.")
    return {"name": name, "peak_flops_fp16_tflops": 100.0, "memory_bw_tbps": 0.5}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProfilingResult:
    model_name : str
    precision  : str
    gpu_name   : str

    # Throughput
    tokens_per_sec    : float = 0.0
    latency_ms_mean   : float = 0.0
    latency_ms_std    : float = 0.0
    latency_ms_p50    : float = 0.0
    latency_ms_p99    : float = 0.0

    # Memory (GB)
    gpu_mem_model_gb  : float = 0.0
    gpu_mem_peak_gb   : float = 0.0
    kv_cache_gb       : float = 0.0

    # Roofline
    arithmetic_intensity : float = 0.0   # FLOP/Byte
    ridge_point          : float = 0.0   # FLOP/Byte
    is_memory_bound      : bool  = True
    achieved_tflops      : float = 0.0
    peak_tflops          : float = 0.0
    memory_bw_tbps       : float = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ProfilingResult":
        with open(path) as f:
            return cls(**json.load(f))

    def summary(self) -> str:
        bound = "MEMORY" if self.is_memory_bound else "COMPUTE"
        return (
            f"{self.model_name} [{self.precision}] on {self.gpu_name}\n"
            f"  Throughput : {self.tokens_per_sec:.1f} tok/s\n"
            f"  Latency    : {self.latency_ms_mean:.2f} ms/tok "
            f"(p99={self.latency_ms_p99:.2f})\n"
            f"  GPU Mem    : {self.gpu_mem_peak_gb:.2f} GB peak\n"
            f"  Bound      : {bound} "
            f"(AI={self.arithmetic_intensity:.2f} vs ridge={self.ridge_point:.2f})"
        )


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_latency(
    model,
    tokenizer,
    prompt: str = "The future of artificial intelligence is",
    n_new_tokens: int = 128,
    n_runs: int = 20,
    n_warmup: int = 5,
    device: str = "cuda",
) -> dict:
    """
    Measure per-token generation latency over n_runs runs.
    Warmup runs are excluded from statistics.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    latencies_ms = []

    for i in range(n_warmup + n_runs):
        torch.cuda.synchronize()
        t0  = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=n_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        t1  = time.perf_counter()

        n_gen = out.shape[1] - inputs["input_ids"].shape[1]
        if i >= n_warmup and n_gen > 0:
            latencies_ms.append((t1 - t0) * 1000.0 / n_gen)

    arr = np.array(latencies_ms)
    return {
        "tokens_per_sec"  : float(1000.0 / arr.mean()),
        "latency_ms_mean" : float(arr.mean()),
        "latency_ms_std"  : float(arr.std()),
        "latency_ms_p50"  : float(np.percentile(arr, 50)),
        "latency_ms_p99"  : float(np.percentile(arr, 99)),
    }


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_memory(
    model,
    tokenizer,
    n_new_tokens: int = 128,
    device: str = "cuda",
) -> dict:
    """Measure peak GPU memory and KV-cache footprint."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    before_bytes = torch.cuda.memory_allocated(device)
    inputs = tokenizer("Memory benchmark prompt.", return_tensors="pt").to(device)

    _ = model.generate(**inputs, max_new_tokens=n_new_tokens,
                        do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    after_bytes = torch.cuda.memory_allocated(device)
    peak_bytes  = torch.cuda.max_memory_allocated(device)

    return {
        "gpu_mem_model_gb" : before_bytes / 1e9,
        "gpu_mem_peak_gb"  : peak_bytes   / 1e9,
        "kv_cache_gb"      : (after_bytes - before_bytes) / 1e9,
    }


# ---------------------------------------------------------------------------
# PyTorch Profiler
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_profiler_trace(
    model,
    tokenizer,
    output_dir: str | Path,
    tag: str,
    n_new_tokens: int = 32,
    device: str = "cuda",
) -> None:
    """Export PyTorch Profiler Chrome trace and print top CUDA kernels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(output_dir / f"{tag}_trace.json")

    inputs = tokenizer("Profiling inference run.", return_tensors="pt").to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("model_generate"):
            _ = model.generate(**inputs, max_new_tokens=n_new_tokens,
                                do_sample=False, use_cache=True)

    prof.export_chrome_trace(trace_path)
    logger.info(f"Profiler trace saved: {trace_path}")
    logger.info(
        "\nTop 15 CUDA ops:\n"
        + prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    )


# ---------------------------------------------------------------------------
# Roofline model
# ---------------------------------------------------------------------------

def compute_roofline(
    flops_per_token: float,
    bytes_per_token: float,
    hw_specs: dict,
) -> dict:
    """
    Roofline model: determine if workload is compute or memory bound.

    Ridge point = peak_flops / memory_bw  [FLOP/Byte]
    If AI < ridge_point → memory bound (moving data is the bottleneck)
    If AI >= ridge_point → compute bound (ALU is the bottleneck)
    """
    peak_flops  = hw_specs["peak_flops_fp16_tflops"] * 1e12   # FLOP/s
    memory_bw   = hw_specs["memory_bw_tbps"]         * 1e12   # Byte/s
    ridge_point = peak_flops / memory_bw                       # FLOP/Byte

    ai              = flops_per_token / bytes_per_token
    is_memory_bound = ai < ridge_point

    # Performance ceiling for this AI
    achieved = min(memory_bw * ai, peak_flops)   # FLOP/s

    return {
        "arithmetic_intensity": ai,
        "ridge_point"         : ridge_point,
        "is_memory_bound"     : is_memory_bound,
        "achieved_tflops"     : achieved / 1e12,
        "peak_tflops"         : peak_flops / 1e12,
        "memory_bw_tbps"      : memory_bw / 1e12,
    }


def estimate_flops_per_token(
    hidden_size : int,
    n_layers    : int,
    vocab_size  : int,
    seq_len     : int,
) -> float:
    """
    Approximate FLOPs for one generated token (decode step).

    Per layer:
      Attention Q,K,V,O projections: 4 * 2 * seq_len * d^2   (matmul FLOPs = 2*m*n*k)
      Attention scores + softmax:    2 * seq_len^2 * d
      FFN (2 linear layers, d→4d→d): 2 * 2 * d * 4d
    Output projection: 2 * d * vocab_size
    """
    d = hidden_size
    attn_proj  = 4 * 2 * seq_len * d * d
    attn_score = 2 * seq_len * seq_len * d
    ffn        = 2 * 2 * d * 4 * d

    per_layer  = attn_proj + attn_score + ffn
    total      = per_layer * n_layers + 2 * d * vocab_size
    return float(total)


def estimate_bytes_per_token(
    hidden_size: int,
    n_layers   : int,
    bits       : int,
    seq_len    : int,
) -> float:
    """
    Bytes read from DRAM per generated token (batch=1, memory-bound regime).

    Weight bytes (quantized):
      Each layer: (4 * d^2 + 2 * d * 4d) weights
    KV cache (FP16):
      2 * n_layers * seq_len * d * 2 bytes
    """
    bytes_per_weight = bits / 8.0
    d   = hidden_size
    w   = (4 * d * d + 2 * d * 4 * d) * bytes_per_weight * n_layers
    kv  = 2 * n_layers * seq_len * d * 2   # 2 bytes = FP16
    return float(w + kv)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

# Default transformer dimensions by model family
MODEL_DIMS: dict[str, dict] = {
    "Llama-2-7B" : {"hidden_size": 4096, "n_layers": 32, "vocab_size": 32000},
    "Llama-2-13B": {"hidden_size": 5120, "n_layers": 40, "vocab_size": 32000},
    "Mistral-7B" : {"hidden_size": 4096, "n_layers": 32, "vocab_size": 32000},
}

PRECISION_BITS: dict[str, int] = {
    "FP16"      : 16,
    "GPTQ_INT8" : 8,
    "GPTQ_INT4" : 4,
    "AWQ_INT4"  : 4,
    "NF4"       : 4,
}


def profile_model(
    model,
    tokenizer,
    model_name : str,
    precision  : str,
    output_dir : str | Path = "results/profiling",
    device     : str = "cuda",
    seq_len    : int = 128,
    run_profiler: bool = True,
) -> ProfilingResult:
    """
    Full profiling pipeline for one model/precision config.
    Saves JSON result and profiler trace.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}_{precision}".replace(" ", "_").replace("/", "_")

    hw = detect_gpu_specs()
    logger.info(f"Profiling {tag} on {hw['name']}")

    lat  = benchmark_latency(model, tokenizer, device=device)
    mem  = measure_memory(model, tokenizer, device=device)

    if run_profiler:
        try:
            run_profiler_trace(model, tokenizer, output_dir, tag, device=device)
        except Exception as e:
            logger.warning(f"Profiler trace failed: {e}")

    # Roofline
    dims    = MODEL_DIMS.get(model_name, MODEL_DIMS["Llama-2-7B"])
    bits    = PRECISION_BITS.get(precision, 16)
    flops   = estimate_flops_per_token(**dims, seq_len=seq_len)
    bytesm  = estimate_bytes_per_token(
        dims["hidden_size"], dims["n_layers"], bits, seq_len
    )
    roofline = compute_roofline(flops, bytesm, hw)

    result = ProfilingResult(
        model_name  = model_name,
        precision   = precision,
        gpu_name    = hw["name"],
        **lat,
        **mem,
        **roofline,
    )
    result.save(output_dir / f"{tag}_profiling.json")
    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# Roofline plot
# ---------------------------------------------------------------------------

def plot_roofline(
    results   : list[ProfilingResult],
    save_path : str | Path,
) -> plt.Figure:
    """Plot all configs on a single roofline diagram."""
    if not results:
        raise ValueError("No results to plot")

    hw = detect_gpu_specs()
    peak_flops = hw["peak_flops_fp16_tflops"]    # TFLOP/s
    memory_bw  = hw["memory_bw_tbps"]            # TB/s
    ridge      = results[0].ridge_point

    ai_range = np.logspace(-3, 4, 400)
    mem_roof = memory_bw * ai_range       # TB/s * FLOP/Byte = TFLOP/s
    cmp_roof = np.full_like(ai_range, peak_flops)
    roof     = np.minimum(mem_roof, cmp_roof)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.loglog(ai_range, roof,      "k-",  lw=2.0, label="Roofline")
    ax.loglog(ai_range, mem_roof,  "b--", lw=1.0, alpha=0.5, label="Memory BW limit")
    ax.axvline(ridge, color="gray", ls=":", lw=1.0)
    ax.text(ridge * 1.1, peak_flops * 0.6,
            f"Ridge ≈ {ridge:.0f} FLOP/B", fontsize=9, color="gray")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for r, c in zip(results, colors):
        ax.scatter(r.arithmetic_intensity, r.achieved_tflops,
                   color=c, s=140, zorder=5,
                   label=f"{r.precision} ({'Mem' if r.is_memory_bound else 'Cmp'})")
        ax.annotate(
            r.precision,
            (r.arithmetic_intensity, r.achieved_tflops),
            textcoords="offset points", xytext=(8, 4), fontsize=8,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (TFLOP/s)",   fontsize=12)
    ax.set_title(
        f"Roofline Model — {results[0].model_name} on {results[0].gpu_name}",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Roofline plot saved: {save_path}")
    return fig
