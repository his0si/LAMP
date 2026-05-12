"""E3 aggregator — combines accuracy (E1 GPTQ ppl) with system efficiency
(E2-residency DRAM/token) into the paper's main result.

Produces the paper's main result table and Pareto figure:

    x = WikiText-2 perplexity (E1)
    y = DRAM bytes / token under a chosen GBuf size (E2-residency)
    points: { uniform + aware, mixed + aware (ours), mixed + oblivious_int8 }
            × { policy budgets we tested }

Two complementary plotting helpers:

  * `plot_residency_pareto_single_gbuf` — one operating point (e.g. 512 MB).
    Compact main figure: 4 policies × 2 schedulers, arrows showing the
    aware → oblivious gap.
  * `plot_residency_pareto_multi_gbuf` — sweep across several GBuf sizes
    on the same axes. Shows how the operating regime shifts.

Both consume `results/residency_sweep.csv` (produced by `pipeline.residency`)
plus the per-tag eval JSON files at `results/eval/<tag>/ppl_wikitext2.json`.

The legacy mapping-based aggregator (Timeloop cache → 4-strategy mapping
comparison) is retained as `build_points` / `plot_pareto` for the E2 mapping
pilot retrospectives — it is *not* the main figure under the residency
contribution.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from . import config, mapping
from .mapping import MappingCache, Strategy, aggregate_model, unique_shapes_qwen


# ---- Residency-based codesign (current main contribution) ---------------

@dataclass
class ResidencyPoint:
    policy_tag: str         # e.g. "Mixed-5.0"
    scheduler: str          # "aware" | "oblivious"
    gbuf_MB: int
    avg_bpw: float
    ppl: float | None
    dram_bytes_per_token: int
    pinned_count: int
    pinned_bytes_aware: int

    @property
    def label(self) -> str:
        return f"{self.policy_tag} / {self.scheduler} @ {self.gbuf_MB}MB"


def _try_ppl(eval_dir: Path) -> float | None:
    f = eval_dir / "ppl_wikitext2.json"
    if not f.exists():
        return None
    return float(json.loads(f.read_text())["ppl"])


def build_residency_codesign(
    *,
    residency_csv: Path | str,
    policy_to_eval_tag: dict[str, str],   # "Mixed-5.0" → "mixed_5p0"
    policy_to_yaml: dict[str, str | Path],
    eval_dir_root: Path | str = config.DEFAULT_RESULTS / "eval",
) -> list[ResidencyPoint]:
    """Join residency rows × policy + eval ppl into codesign points."""
    rows = list(csv.DictReader(open(residency_csv)))
    eval_dir_root = Path(eval_dir_root)
    points: list[ResidencyPoint] = []
    for r in rows:
        tag = r.get("label", "")
        if tag not in policy_to_eval_tag:
            continue
        eval_tag = policy_to_eval_tag[tag]
        ppl = _try_ppl(eval_dir_root / eval_tag)
        sched = "aware" if r["strategy"] == "precision_aware" else "oblivious"
        pol = yaml.safe_load(Path(policy_to_yaml[tag]).read_text())
        points.append(ResidencyPoint(
            policy_tag=tag,
            scheduler=sched,
            gbuf_MB=int(r["gbuf_bytes"]) >> 20,
            avg_bpw=float(pol["achieved_avg_bits"]),
            ppl=ppl,
            dram_bytes_per_token=int(r["dram_bytes_per_token"]),
            pinned_count=int(r["pinned_count"]),
            # pinned_bytes_aware is optional (some sweep CSVs trimmed it).
            pinned_bytes_aware=int(r.get("pinned_bytes_aware") or 0),
        ))
    return points


def save_residency_csv(points: Iterable[ResidencyPoint], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["policy_tag", "scheduler", "gbuf_MB", "avg_bpw", "ppl",
                    "dram_MB_per_token", "pinned_count", "pinned_MB"])
        for p in points:
            w.writerow([
                p.policy_tag, p.scheduler, p.gbuf_MB, f"{p.avg_bpw:.3f}",
                "" if p.ppl is None else f"{p.ppl:.4f}",
                f"{p.dram_bytes_per_token / 1e6:.3f}",
                p.pinned_count,
                f"{p.pinned_bytes_aware / 1e6:.3f}",
            ])
    return path


_POLICY_COLORS = {
    "INT4-uniform": "#1f77b4",
    "Mixed-4.5":    "#2ca02c",
    "Mixed-5.0":    "#ff7f0e",
    "INT8-uniform": "#d62728",
}


def plot_residency_pareto_single_gbuf(
    points: Iterable[ResidencyPoint], *, gbuf_MB: int, out_path: Path | str,
    fp16_ppl: float | None = None,
) -> Path:
    """Compact main figure at one operating point. `fp16_ppl` draws a vertical
    reference line; pass None to suppress (e.g. when target FP16 baseline
    is not measured yet)."""
    import matplotlib.pyplot as plt

    pts = [p for p in points if p.gbuf_MB == gbuf_MB and p.ppl is not None]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    by_pol: dict[str, dict[str, ResidencyPoint]] = {}
    for p in pts:
        by_pol.setdefault(p.policy_tag, {})[p.scheduler] = p

    for tag, both in by_pol.items():
        c = _POLICY_COLORS.get(tag, "#7f7f7f")
        a = both.get("aware"); o = both.get("oblivious")
        if a is None or o is None:
            continue
        ax.scatter(a.ppl, a.dram_bytes_per_token / 1e6, marker="o", s=110,
                   color=c, edgecolor="black", linewidth=0.5, label=f"{tag} (aware)")
        ax.scatter(o.ppl, o.dram_bytes_per_token / 1e6, marker="x", s=110,
                   color=c, label=None)
        if a.dram_bytes_per_token != o.dram_bytes_per_token:
            ax.annotate(
                "", xy=(a.ppl, a.dram_bytes_per_token / 1e6),
                xytext=(o.ppl, o.dram_bytes_per_token / 1e6),
                arrowprops=dict(arrowstyle="->", color=c, alpha=0.55, lw=1.5),
            )
            gain = o.dram_bytes_per_token / max(1, a.dram_bytes_per_token)
            ax.annotate(
                f"{gain:.2f}× ↓",
                ((a.ppl + o.ppl) / 2,
                 (a.dram_bytes_per_token + o.dram_bytes_per_token) / 2e6),
                xytext=(8, 0), textcoords="offset points", fontsize=9, color=c,
            )

    if fp16_ppl is not None:
        ax.axvline(fp16_ppl, color="gray", lw=1, ls=":",
                   label=f"FP16 ppl = {fp16_ppl:.2f}")
    ax.set_xlabel("WikiText-2 perplexity (real GPTQ, E1)")
    ax.set_ylabel(f"DRAM bytes / token at GBuf = {gbuf_MB} MB (MB)")
    ax.set_title(f"Accuracy × per-token DRAM @ GBuf = {gbuf_MB} MB\n"
                 f"○ aware  vs  × oblivious  (arrows = aware reduction)")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    return out_path


def plot_residency_pareto_multi_gbuf(
    points: Iterable[ResidencyPoint], *, gbuf_MB_list: list[int],
    out_path: Path | str,
) -> Path:
    """Faceted: one panel per GBuf size."""
    import matplotlib.pyplot as plt

    n = len(gbuf_MB_list)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    pts_by_g: dict[int, list[ResidencyPoint]] = {g: [] for g in gbuf_MB_list}
    for p in points:
        if p.gbuf_MB in pts_by_g and p.ppl is not None:
            pts_by_g[p.gbuf_MB].append(p)

    for ax, g in zip(axes, gbuf_MB_list):
        by_pol: dict[str, dict[str, ResidencyPoint]] = {}
        for p in pts_by_g[g]:
            by_pol.setdefault(p.policy_tag, {})[p.scheduler] = p
        for tag, both in by_pol.items():
            c = _POLICY_COLORS.get(tag, "#7f7f7f")
            for sched, marker in (("aware", "o"), ("oblivious", "x")):
                pt = both.get(sched)
                if pt is None:
                    continue
                # 'x' marker uses linewidth for the strokes; only the 'o' face needs edgecolor.
                kwargs = dict(marker=marker, s=90, color=c)
                if marker == "o":
                    kwargs.update(edgecolor="black", linewidth=0.5)
                else:
                    kwargs.update(linewidth=2.0)
                ax.scatter(pt.ppl, pt.dram_bytes_per_token / 1e6, **kwargs,
                           label=tag if (sched == "aware" and ax is axes[0]) else None)
        ax.axvline(8.89, color="gray", lw=0.8, ls=":")
        ax.set_title(f"GBuf = {g} MB")
        ax.set_xlabel("WikiText-2 ppl")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("DRAM bytes / token (MB)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("E3: accuracy × DRAM/token across GBuf operating points  "
                 "(○ aware ours, × precision-oblivious)")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    return out_path


# ---- Legacy mapping-based aggregator (retained for E2 mapping retros) ----

@dataclass
class CodesignPoint:
    label: str            # e.g. "Mixed-5.0 / precision_aware"
    strategy: Strategy
    avg_bpw: float
    ppl: float | None     # may be None if accuracy not yet measured
    energy_pJ_per_token: float
    dram_bytes_per_token: int
    mean_pe_utilization: float
    cycles_per_token: int


def build_points(
    *,
    target_key: str,
    policy_paths: dict[str, str | Path],     # tag -> policy yaml
    mapping_cache_path: Path | str,
    eval_dir_root: Path | str = config.DEFAULT_RESULTS / "eval",
) -> list[CodesignPoint]:
    """[Legacy] For each policy, compute 4 codesign points (one per strategy)."""
    cfg = config.load_targets()
    target = cfg["targets"][target_key]
    shapes = unique_shapes_qwen(target)
    cache = MappingCache(mapping_cache_path)

    strategies: list[Strategy] = [
        "uniform_int4",
        "uniform_int8",
        "oblivious_int8",
        "precision_aware",
    ]

    points: list[CodesignPoint] = []
    for tag, policy_yaml in policy_paths.items():
        pol = yaml.safe_load(Path(policy_yaml).read_text())
        avg_bpw = pol["achieved_avg_bits"]
        ppl = _try_ppl(Path(eval_dir_root) / tag)

        for strat in strategies:
            hw = aggregate_model(
                policy_yaml=policy_yaml, cache=cache,
                shapes_per_layer=shapes, strategy=strat,
            )
            label = f"{tag} / {strat}"
            points.append(CodesignPoint(
                label=label, strategy=strat, avg_bpw=avg_bpw, ppl=ppl,
                energy_pJ_per_token=hw["energy_pJ_per_token"],
                dram_bytes_per_token=hw["dram_bytes_per_token"],
                mean_pe_utilization=hw["mean_pe_utilization"],
                cycles_per_token=hw["cycles_per_token"],
            ))
    return points


def save_csv(points: Iterable[CodesignPoint], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["label","strategy","avg_bpw","ppl","energy_pJ_per_token",
                    "dram_bytes_per_token","mean_pe_utilization","cycles_per_token"])
        for p in points:
            w.writerow([p.label, p.strategy, p.avg_bpw,
                        "" if p.ppl is None else f"{p.ppl:.4f}",
                        f"{p.energy_pJ_per_token:.3e}",
                        p.dram_bytes_per_token,
                        f"{p.mean_pe_utilization:.4f}",
                        p.cycles_per_token])
    return path


def plot_pareto(points: Iterable[CodesignPoint], *, y: str = "energy_pJ_per_token",
                out_path: Path | str) -> Path:
    """[Legacy] E2 mapping-cache Pareto."""
    import matplotlib.pyplot as plt
    pts = [p for p in points if p.ppl is not None]
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {
        "uniform_int4":    ("#1f77b4", "o"),
        "uniform_int8":    ("#2ca02c", "s"),
        "oblivious_int8":  ("#d62728", "^"),
        "precision_aware": ("#ff7f0e", "D"),
    }
    seen = set()
    for p in pts:
        c, m = palette[p.strategy]
        ax.scatter(p.ppl, getattr(p, y), c=c, marker=m, s=70,
                   label=(p.strategy if p.strategy not in seen else None))
        seen.add(p.strategy)
        ax.annotate(p.label.split(" / ")[0], (p.ppl, getattr(p, y)),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("WikiText-2 perplexity (lower = better)")
    ax.set_ylabel({
        "energy_pJ_per_token": "energy / token (pJ)",
        "dram_bytes_per_token": "DRAM bytes / token",
        "cycles_per_token": "cycles / token",
    }.get(y, y))
    ax.set_title("[Legacy] mapping-cache Pareto, four strategies")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    return out_path
