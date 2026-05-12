"""Phase 6 — plots and Pareto curves.

matplotlib is a heavy import and lives in requirements-extra.txt, so we
import it lazily — the package can still be imported on a fresh env.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_sensitivity(sensitivity_json: Path | str, out_path: Path | str) -> Path:
    plt = _plt()
    data = json.loads(Path(sensitivity_json).read_text())
    rows = data["layers"]
    xs = [r["idx"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    for b in (4, 8):
        ys = [r.get(f"delta_{b}bit", 0.0) for r in rows]
        ax.plot(xs, ys, marker="o", label=f"{b}-bit Δppl")
    ax.set_xlabel("layer index"); ax.set_ylabel("Δppl vs fp16")
    ax.set_title(f"Per-layer sensitivity — {data['target']}")
    ax.legend(); ax.grid(alpha=0.3)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_policy(policy_yaml: Path | str, out_path: Path | str) -> Path:
    plt = _plt()
    pol = yaml.safe_load(Path(policy_yaml).read_text())
    bits = pol["per_layer_bits"]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.bar(range(len(bits)), bits, color=["#5b9bd5" if b == min(bits) else "#ed7d31" for b in bits])
    ax.set_xlabel("layer index"); ax.set_ylabel("bits")
    ax.set_title(f"{pol['target']} policy (target {pol['target_avg_bits']} bpw, "
                 f"achieved {pol['achieved_avg_bits']:.2f})")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pareto(points: Iterable[dict], x: str, y: str, out_path: Path | str,
                label_key: str = "name") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    for p in points:
        ax.scatter(p[x], p[y])
        ax.annotate(p.get(label_key, ""), (p[x], p[y]), fontsize=8)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=0.3)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
