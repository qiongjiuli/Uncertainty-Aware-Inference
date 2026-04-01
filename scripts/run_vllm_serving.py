#!/usr/bin/env python
"""
scripts/run_vllm_serving.py

Benchmarks each quantization config via vLLM's OpenAI-compatible API.
Measures throughput (tok/s), latency, and requests/sec at multiple
concurrency levels (concurrency sweep).

PREREQUISITES:
  pip install vllm aiohttp
  # Start vLLM servers in separate terminals first (see --print_commands)

Usage:
    # Step 1: print launch commands and start servers
    python scripts/run_vllm_serving.py --print_commands \
        --model_id meta-llama/Llama-2-7b-hf --model_name Llama-2-7B

    # Step 2: once servers are running, benchmark them
    python scripts/run_vllm_serving.py \
        --model_id meta-llama/Llama-2-7b-hf --model_name Llama-2-7B \
        --output_dir results/vllm \
        --precisions FP16 GPTQ_INT4 AWQ_INT4 NF4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging import init_wandb, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_PORT_MAP: dict[str, int] = {
    "FP16"      : 8000,
    "GPTQ_INT8" : 8001,
    "GPTQ_INT4" : 8002,
    "AWQ_INT4"  : 8003,
    "NF4"       : 8004,
}

VLLM_QUANT_MAP: dict[str, str | None] = {
    "FP16"      : None,
    "GPTQ_INT8" : "bitsandbytes",
    "GPTQ_INT4" : "gptq",
    "AWQ_INT4"  : "awq",
    "NF4"       : "bitsandbytes",
}

PREQUANTIZED_IDS: dict[str, dict[str, str]] = {
    "meta-llama/Llama-2-7b-hf": {
        "GPTQ_INT4": "TheBloke/Llama-2-7B-GPTQ",
        "AWQ_INT4" : "TheBloke/Llama-2-7B-AWQ",
    },
    "mistralai/Mistral-7B-v0.1": {
        "GPTQ_INT4": "TheBloke/Mistral-7B-v0.1-GPTQ",
        "AWQ_INT4" : "TheBloke/Mistral-7B-v0.1-AWQ",
    },
    "meta-llama/Llama-2-13b-hf": {
        "GPTQ_INT4": "TheBloke/Llama-2-13B-GPTQ",
        "AWQ_INT4" : "TheBloke/Llama-2-13B-AWQ",
    },
}

SAMPLE_PROMPTS = [
    "Explain the Pythagorean theorem in simple terms.",
    "What are the main causes of climate change?",
    "Write a short story about a robot learning to paint.",
    "Summarize the key ideas behind transformer attention.",
    "What is the difference between supervised and unsupervised learning?",
    "How does photosynthesis produce oxygen?",
    "Describe three applications of large language models in healthcare.",
    "What are the trade-offs between model size and inference speed?",
    "Explain gradient descent in one paragraph.",
    "How does quantization reduce GPU memory usage?",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ServingResult:
    model_name       : str
    precision        : str
    n_concurrent     : int
    n_requests       : int
    mean_latency_ms  : float
    p50_latency_ms   : float
    p99_latency_ms   : float
    tokens_per_sec   : float
    requests_per_sec : float
    success_rate     : float

    def summary(self) -> str:
        return (
            f"{self.precision} | n_concurrent={self.n_concurrent} | "
            f"tps={self.tokens_per_sec:.1f} | "
            f"lat_mean={self.mean_latency_ms:.0f}ms | "
            f"p99={self.p99_latency_ms:.0f}ms | "
            f"success={self.success_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# Launch commands
# ---------------------------------------------------------------------------

def build_launch_command(
    model_id  : str,
    precision : str,
    port      : int,
    gpu_memory_utilization: float = 0.90,
    max_model_len : int = 2048,
    tensor_parallel: int = 1,
) -> str:
    model = PREQUANTIZED_IDS.get(model_id, {}).get(precision, model_id)
    quant = VLLM_QUANT_MAP.get(precision)
    quant_flag = f"--quantization {quant}" if quant else ""

    cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model} "
        f"--dtype float16 "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"--max-model-len {max_model_len} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--port {port} "
        f"--served-model-name default "
        f"{quant_flag}"
    ).strip()
    return cmd


def print_launch_commands(model_id: str, precisions: list[str]) -> None:
    print("\n" + "="*65)
    print("  vLLM Server Launch Commands")
    print("  Run each in a SEPARATE terminal before benchmarking.")
    print("="*65 + "\n")
    for p in precisions:
        port = VLLM_PORT_MAP[p]
        cmd  = build_launch_command(model_id, p, port)
        print(f"# {p} (port {port})")
        print(cmd)
        print()
    print("="*65)
    print("When all servers show 'Application startup complete', press Enter.")
    print("="*65 + "\n")


# ---------------------------------------------------------------------------
# Async request sending
# ---------------------------------------------------------------------------

async def _send_one(
    session   : aiohttp.ClientSession,
    url       : str,
    prompt    : str,
    max_tokens: int = 100,
) -> dict | None:
    payload = {
        "model"      : "default",
        "prompt"     : prompt,
        "max_tokens" : max_tokens,
        "temperature": 0.0,
    }
    try:
        t0 = time.perf_counter()
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
        t1 = time.perf_counter()
        n_tok = data.get("usage", {}).get("completion_tokens", max_tokens)
        return {"latency_ms": (t1 - t0) * 1000.0, "n_tokens": n_tok, "ok": True}
    except Exception as e:
        logger.debug(f"Request failed: {e}")
        return {"latency_ms": 0.0, "n_tokens": 0, "ok": False}


async def _run_concurrent(
    url            : str,
    prompts        : list[str],
    n_concurrent   : int,
    total_requests : int,
    max_tokens     : int = 100,
) -> list[dict]:
    semaphore = asyncio.Semaphore(n_concurrent)
    results   = []

    async def throttled(prompt: str) -> dict:
        async with semaphore:
            return await _send_one(session, url, prompt, max_tokens)

    async with aiohttp.ClientSession() as session:
        tasks = [
            throttled(prompts[i % len(prompts)])
            for i in range(total_requests)
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


# ---------------------------------------------------------------------------
# Single benchmark point
# ---------------------------------------------------------------------------

def benchmark_at_concurrency(
    model_name    : str,
    precision     : str,
    port          : int,
    n_concurrent  : int,
    total_requests: int = 100,
    max_tokens    : int = 100,
) -> ServingResult | None:
    url = f"http://localhost:{port}/v1/completions"
    logger.info(f"  [{precision}] port={port} concurrent={n_concurrent} requests={total_requests}")

    t_start = time.perf_counter()
    raw     = asyncio.run(_run_concurrent(
        url, SAMPLE_PROMPTS, n_concurrent, total_requests, max_tokens
    ))
    elapsed = time.perf_counter() - t_start

    ok_results = [r for r in raw if r["ok"]]
    if not ok_results:
        logger.warning(f"  All requests failed for {precision}")
        return None

    latencies   = np.array([r["latency_ms"] for r in ok_results])
    total_toks  = sum(r["n_tokens"] for r in ok_results)
    success_rate= len(ok_results) / len(raw)

    return ServingResult(
        model_name       = model_name,
        precision        = precision,
        n_concurrent     = n_concurrent,
        n_requests       = len(raw),
        mean_latency_ms  = float(latencies.mean()),
        p50_latency_ms   = float(np.percentile(latencies, 50)),
        p99_latency_ms   = float(np.percentile(latencies, 99)),
        tokens_per_sec   = total_toks / elapsed,
        requests_per_sec = len(ok_results) / elapsed,
        success_rate     = success_rate,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_throughput_vs_concurrency(
    all_sweeps: dict[str, list[ServingResult]],
    save_path : Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors    = plt.cm.tab10(np.linspace(0, 1, len(all_sweeps)))

    for (precision, sweep), color in zip(all_sweeps.items(), colors):
        if not sweep:
            continue
        conc = [r.n_concurrent   for r in sweep]
        tps  = [r.tokens_per_sec for r in sweep]
        lat  = [r.mean_latency_ms for r in sweep]
        p99  = [r.p99_latency_ms  for r in sweep]

        axes[0].plot(conc, tps,  "o-", color=color, label=precision)
        axes[1].plot(conc, lat,  "o-", color=color, label=precision)
        axes[2].plot(conc, p99,  "o-", color=color, label=precision)

    for ax, (title, ylabel) in zip(axes, [
        ("Throughput vs Concurrency",    "Tokens / Second"),
        ("Mean Latency vs Concurrency",  "Latency (ms)"),
        ("P99 Latency vs Concurrency",   "P99 Latency (ms)"),
    ]):
        ax.set_xlabel("Concurrent Requests", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"vLLM Serving Benchmark — {list(all_sweeps.values())[0][0].model_name}",
        fontsize=13, fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    logger.info(f"Throughput plot saved: {save_path}")


def plot_serving_summary(
    results   : list[ServingResult],
    save_path : Path,
) -> None:
    """Bar chart of throughput and mean latency for peak-concurrency results."""
    import matplotlib.pyplot as plt

    labels = [r.precision       for r in results]
    tps    = [r.tokens_per_sec  for r in results]
    lat    = [r.mean_latency_ms for r in results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(labels, tps, color=colors)
    for bar, v in zip(bars, tps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.0f}", ha="center", fontsize=10)
    ax1.set_ylabel("Tokens / Second", fontsize=12)
    ax1.set_title("Throughput", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(labels, lat, color=colors)
    for bar, v in zip(bars2, lat):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.0f}ms", ha="center", fontsize=10)
    ax2.set_ylabel("Mean Latency (ms)", fontsize=12)
    ax2.set_title("Latency", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"vLLM Serving Summary — {results[0].model_name}",
        fontsize=13, fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM serving benchmark")
    p.add_argument("--model_id",      required=True)
    p.add_argument("--model_name",    required=True)
    p.add_argument("--output_dir",    default="results/vllm")
    p.add_argument("--precisions",    nargs="+",
                   default=["FP16", "GPTQ_INT4", "AWQ_INT4", "NF4"])
    p.add_argument("--print_commands", action="store_true",
                   help="Print vLLM launch commands and exit")
    p.add_argument("--concurrency_levels", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16],
                   help="Concurrency levels to sweep")
    p.add_argument("--total_requests",  type=int, default=100)
    p.add_argument("--max_tokens",      type=int, default=100)
    p.add_argument("--wandb_project",   default="uncertainty-aware-inference")
    p.add_argument("--no_wandb",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_commands:
        print_launch_commands(args.model_id, args.precisions)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs")

    run = None
    if not args.no_wandb:
        run = init_wandb(
            project = args.wandb_project,
            name    = f"{args.model_name}-vllm-serving",
            tags    = [args.model_name, "vllm", "serving"],
            config  = vars(args),
        )

    all_sweeps     : dict[str, list[ServingResult]] = {}
    peak_results   : list[ServingResult] = []

    for precision in args.precisions:
        port  = VLLM_PORT_MAP.get(precision)
        if port is None:
            logger.warning(f"No port configured for {precision}, skipping")
            continue

        logger.info(f"\n{'='*55}\n  Benchmarking {precision} (port {port})\n{'='*55}")
        sweep : list[ServingResult] = []

        for n_conc in args.concurrency_levels:
            r = benchmark_at_concurrency(
                args.model_name, precision, port,
                n_concurrent   = n_conc,
                total_requests = args.total_requests,
                max_tokens     = args.max_tokens,
            )
            if r:
                sweep.append(r)
                logger.info(f"  {r.summary()}")
                if run:
                    run.log({
                        f"serving/{precision}/tps_c{n_conc}": r.tokens_per_sec,
                        f"serving/{precision}/lat_c{n_conc}": r.mean_latency_ms,
                    })

        if sweep:
            all_sweeps[precision] = sweep
            # Best throughput point
            best = max(sweep, key=lambda r: r.tokens_per_sec)
            peak_results.append(best)

    # ── Save raw results ──
    raw_out = {
        prec: [asdict(r) for r in sweep]
        for prec, sweep in all_sweeps.items()
    }
    with open(output_dir / f"{args.model_name}_vllm_results.json", "w") as f:
        json.dump(raw_out, f, indent=2)

    # ── Plots ──
    if len(all_sweeps) > 0:
        plot_throughput_vs_concurrency(
            all_sweeps,
            save_path=output_dir / "plots" / f"{args.model_name}_concurrency_sweep.png",
        )
    if peak_results:
        plot_serving_summary(
            peak_results,
            save_path=output_dir / "plots" / f"{args.model_name}_serving_summary.png",
        )

    # ── Summary ──
    logger.info("\n=== Serving Benchmark Summary (best concurrency) ===")
    for r in sorted(peak_results, key=lambda r: r.tokens_per_sec, reverse=True):
        logger.info(f"  {r.summary()}")

    if run:
        run.finish()
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
