#!/usr/bin/env python
"""Render the cross-model residency figure used in the paper.

Reads ``residency_sweep_<target>.csv`` for each model in the sweep and
produces a 1 x N panel figure plotting DRAM bytes per token versus GBuf
size at log scale, with solid lines for the precision-aware scheduler
and dashed lines for the precision-oblivious INT8-budget baseline. A
vertical guide marks each model's INT4 total decoder footprint.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELS = [
    ("qwen25_15b_instruct", "Qwen-1.5B"),
    ("gemma2_2b_it",        "Gemma-2-2B"),
    ("qwen25_7b_instruct",  "Qwen-7B"),
    ("llama31_8b_instruct", "Llama-8B"),
]

POLICY_ORDER = ["INT4-uniform", "Mixed-4.5", "Mixed-5.0", "INT8-uniform"]
POLICY_COLOR = {
    "INT4-uniform": "tab:red",
    "Mixed-4.5":    "tab:orange",
    "Mixed-5.0":    "tab:green",
    "INT8-uniform": "tab:blue",
}


def load_sweep(target: str):
    """Returns (rows_by_policy_strategy, int4_decoder_MB)."""
    path = RESULTS / f"residency_sweep_{target}.csv"
    if not path.exists() and target == "qwen25_15b_instruct":
        # First-target sweep was written without the target suffix.
        path = RESULTS / "residency_sweep.csv"
    by_key = defaultdict(list)
    int4_decoder = None
    with path.open() as f:
        for r in csv.DictReader(f):
            gbuf_mb = int(r["gbuf_bytes"]) // (1 << 20)
            dram_mb = float(r["dram_bytes_per_token"]) / 1e6
            by_key[(r["label"], r["strategy"])].append((gbuf_mb, dram_mb))
            if r["label"] == "INT4-uniform":
                int4_decoder = float(r["total_decoder_bytes_aware"]) / 1e6
    for k in by_key:
        by_key[k].sort()
    return by_key, int4_decoder


def render(out_path: Path) -> Path:
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (target, label) in zip(axes, MODELS):
        sweep, int4_mb = load_sweep(target)
        for policy in POLICY_ORDER:
            color = POLICY_COLOR[policy]
            aware = sweep.get((policy, "precision_aware"), [])
            obliv = sweep.get((policy, "precision_oblivious_int8"), [])
            if aware:
                xs, ys = zip(*aware)
                ax.plot(xs, ys, color=color, marker="o", markersize=4,
                        linewidth=1.5, label=f"{policy} aware")
            if obliv:
                xs, ys = zip(*obliv)
                ax.plot(xs, ys, color=color, linestyle="--", marker="x",
                        markersize=4, linewidth=1.0, alpha=0.75,
                        label=f"{policy} oblivious")
        if int4_mb is not None:
            ax.axvline(int4_mb, color="gray", linestyle=":", linewidth=1.0,
                       label=f"INT4 = {int4_mb:.0f} MB")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("GBuf size (MB, log scale)")
        ax.set_title(f"{label}  (decoder INT4 = {int4_mb:.0f} MB)")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    axes[0].set_ylabel("DRAM bytes / token  (MB/token)")
    axes[0].legend(loc="lower left", fontsize=7, frameon=True)
    fig.suptitle(
        "Cross-model sweep — precision-aware residency win persists "
        "across scale (1.5B / 2B / 7B / 8B)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out = RESULTS / "figs" / "sweep_cross_model.png"
    print(f"wrote {render(out)}")
