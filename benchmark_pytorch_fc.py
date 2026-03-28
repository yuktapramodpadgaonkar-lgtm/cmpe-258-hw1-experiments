#!/usr/bin/env python3
"""
PyTorch fully-connected (FC) layer benchmark:
  Mode 1 — CUDA-core-style FP32 path: TF32 tensor math disabled for matmul.
  Mode 2 — Tensor Core path: TF32 allowed for FP32 matmul (Ampere+ GPUs).

Uses CUDA events + synchronize for GPU-side timing.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def gemm_flops(batch: int, in_f: int, out_f: int) -> int:
    """FLOPs for y = x @ W^T with x:[B,K], W:[N,K] → one GEMM of shape (B,K) × (K,N)."""
    m, k, n = batch, in_f, out_f
    return 2 * m * n * k


def set_tf32_matmul(enabled: bool) -> None:
    """Match assignment: FP32 I/O; TF32 compute only when enabled (Ampere+)."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    # Conv layers use cuDNN; harmless for pure Linear but keeps settings consistent.
    torch.backends.cudnn.allow_tf32 = enabled


@torch.inference_mode()
def bench_linear_once(
    batch: int,
    in_features: int,
    out_features: int,
    use_tf32: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns (avg_latency_ms, throughput_gflops) for one configuration.
    """
    set_tf32_matmul(use_tf32)

    x = torch.randn(batch, in_features, device=device, dtype=torch.float32)
    layer = nn.Linear(in_features, out_features, bias=True, device=device, dtype=torch.float32)

    # Warm-up: amortize kernel JIT, memory, clock boost
    for _ in range(warmup):
        _ = layer(x)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    for _ in range(iters):
        _ = layer(x)
    ender.record()
    torch.cuda.synchronize()

    total_ms = starter.elapsed_time(ender)
    avg_ms = total_ms / float(iters)
    flops = gemm_flops(batch, in_features, out_features)
    # bias adds negligible work vs GEMM; throughput uses GEMM FLOPs as in assignment FC focus
    gflops = (flops / (avg_ms / 1000.0)) / 1e9
    return avg_ms, gflops


@dataclass
class SweepResult:
    batch: int
    in_f: int
    out_f: int
    mode: str
    avg_ms: float
    gflops: float


def parse_size_triples(specs: Iterable[str]) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for s in specs:
        parts = s.lower().replace("x", ",").split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected B,K,N as B,K,N or BxKxN, got: {s}")
        b, k, n = (int(p.strip()) for p in parts)
        out.append((b, k, n))
    return out


def default_sweep() -> List[Tuple[int, int, int]]:
    """Reasonable grid for FC GEMM (batch, in, out)."""
    sizes = [256, 512, 1024, 2048, 4096]
    triples: List[Tuple[int, int, int]] = []
    for b in (64, 128, 256):
        for dim in sizes:
            triples.append((b, dim, dim))
    return triples


def plot_results(rows: List[SweepResult], out_prefix: str) -> None:
    cuda_rows = [r for r in rows if r.mode == "pytorch_cuda_fp32_no_tf32"]
    tc_rows = [r for r in rows if r.mode == "pytorch_tf32_tensor"]

    def label(r: SweepResult) -> str:
        return f"B{r.batch}_K{r.in_f}_N{r.out_f}"

    # Sort by problem "size" (FLOPs)
    def sort_key(r: SweepResult):
        return gemm_flops(r.batch, r.in_f, r.out_f)

    cuda_rows = sorted(cuda_rows, key=sort_key)
    tc_rows = sorted(tc_rows, key=sort_key)
    labels = [label(r) for r in cuda_rows]

    lat_cuda = [r.avg_ms for r in cuda_rows]
    lat_tc = [r.avg_ms for r in tc_rows]
    g_cuda = [r.gflops for r in cuda_rows]
    g_tc = [r.gflops for r in tc_rows]

    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, lat_cuda, marker="o", label="CUDA-core path (TF32 off)")
    axes[0].plot(x, lat_tc, marker="s", label="Tensor Core path (TF32 on)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Avg latency per forward (ms)")
    axes[0].set_title("PyTorch FC — latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, g_cuda, marker="o", label="CUDA-core path (TF32 off)")
    axes[1].plot(x, g_tc, marker="s", label="Tensor Core path (TF32 on)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Throughput (GFLOPS, GEMM FLOPs)")
    axes[1].set_title("PyTorch FC — throughput")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    png = out_prefix + "_pytorch_fc.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {png}")


def main() -> int:
    p = argparse.ArgumentParser(description="FC layer: CUDA-core FP32 vs TF32 Tensor Core (PyTorch)")
    p.add_argument("--warmup", type=int, default=20, help="Warm-up forwards (not timed)")
    p.add_argument("--iters", type=int, default=100, help="Timed iterations per config")
    p.add_argument(
        "--sizes",
        nargs="*",
        default=None,
        help='List of "B,K,N" (batch, in_features, out_features). Default: built-in sweep.',
    )
    p.add_argument("--csv", type=str, default="results_pytorch_fc.csv")
    p.add_argument("--plot-prefix", type=str, default="comparison")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    torch.cuda.set_device(0)

    triples = parse_size_triples(args.sizes) if args.sizes else default_sweep()

    rows: List[SweepResult] = []
    for b, k, n in triples:
        for use_tf32, mode_name in (
            (False, "pytorch_cuda_fp32_no_tf32"),
            (True, "pytorch_tf32_tensor"),
        ):
            avg_ms, gflops = bench_linear_once(
                batch=b,
                in_features=k,
                out_features=n,
                use_tf32=use_tf32,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            rows.append(SweepResult(b, k, n, mode_name, avg_ms, gflops))
            print(
                f"{mode_name:28s}  B={b:5d} K={k:5d} N={n:5d}  "
                f"lat_ms={avg_ms:.4f}  GEMM_GFLOPS={gflops:.2f}"
            )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "batch", "in_features", "out_features", "avg_latency_ms", "gemm_gflops"])
        for r in rows:
            w.writerow([r.mode, r.batch, r.in_f, r.out_f, f"{r.avg_ms:.6f}", f"{r.gflops:.4f}"])

    print(f"Wrote CSV: {args.csv}")
    plot_results(rows, args.plot_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
