#!/usr/bin/env python3
"""Plot latency and throughput from benchmark CSV (PyTorch or cuBLAS output)."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Row:
    mode: str
    batch: int
    in_f: int
    out_f: int
    avg_ms: float
    gflops: float


def gemm_flops(batch: int, in_f: int, out_f: int) -> int:
    m, k, n = batch, in_f, out_f
    return 2 * m * n * k


def load_csv(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                Row(
                    mode=d["mode"].strip(),
                    batch=int(d["batch"]),
                    in_f=int(d["in_features"]),
                    out_f=int(d["out_features"]),
                    avg_ms=float(d["avg_latency_ms"]),
                    gflops=float(d["gemm_gflops"]),
                )
            )
    return rows


def plot(rows: List[Row], title_prefix: str, out_png: str) -> None:
    modes = sorted({r.mode for r in rows})
    if len(modes) < 2:
        print("Need at least two distinct modes in CSV for comparison plot.", file=sys.stderr)
        return

    def triplet(r: Row):
        return (r.batch, r.in_f, r.out_f)

    def label(t):
        b, k, n = t
        return f"B{b}_K{k}_N{n}"

    by_mode: dict[str, dict[tuple[int, int, int], Row]] = {m: {} for m in modes}
    for r in rows:
        by_mode[r.mode][triplet(r)] = r

    common_keys = set.intersection(*(set(by_mode[m].keys()) for m in modes))
    if not common_keys:
        print("No overlapping (B,K,N) across modes.", file=sys.stderr)
        return

    sorted_keys = sorted(common_keys, key=lambda t: gemm_flops(*t))
    labels = [label(t) for t in sorted_keys]

    series_ms = {}
    series_g = {}
    for m in modes:
        series_ms[m] = [by_mode[m][t].avg_ms for t in sorted_keys]
        series_g[m] = [by_mode[m][t].gflops for t in sorted_keys]

    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for m in modes:
        axes[0].plot(x, series_ms[m], marker="o", label=m)
        axes[1].plot(x, series_g[m], marker="o", label=m)

    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Avg latency per forward (ms)")
    axes[0].set_title(f"{title_prefix} — latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Throughput (GFLOPS, GEMM FLOPs)")
    axes[1].set_title(f"{title_prefix} — throughput")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--title", default="FC GEMM")
    p.add_argument("--out", default="plot_from_csv.png")
    args = p.parse_args()
    rows = load_csv(args.csv)
    plot(rows, args.title, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
