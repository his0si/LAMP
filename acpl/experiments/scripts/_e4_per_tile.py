#!/usr/bin/env python
"""E4 driver — per-tile hwnorm policy + residency comparison vs E3 baseline.

Runs after per-projection sensitivity has been saved to
`results/sensitivity/qwen25_15b_instruct_per_projection.json`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, policy, residency  # noqa: E402


TARGET = "qwen25_15b_instruct"
PER_PROJ_PATH = "experiments/results/sensitivity/qwen25_15b_instruct_per_projection.json"
BUDGETS = (4.5, 5.0, 5.5, 6.0)
GBUF_SIZES_MB = (128, 256, 512, 1024)


def main() -> int:
    if not Path(PER_PROJ_PATH).exists():
        print(f"missing {PER_PROJ_PATH} — wait for sensitivity to finish")
        return 1

    cfg = config.load_targets()
    target_cfg = cfg["targets"][TARGET]

    # 1) Generate hwnorm per-tile policies at 4 budgets.
    print("=== generating hwnorm per-tile policies ===")
    hwnorm_paths: dict[float, Path] = {}
    for B in BUDGETS:
        pol = policy.make_per_tile_policy(
            PER_PROJ_PATH, target_cfg=target_cfg,
            allowed_widths=[4, 8], target_avg_bits=B, scorer="hwnorm",
        )
        path = policy.save_per_tile(pol, "experiments/results/policies")
        hwnorm_paths[B] = path
        n_8 = sum(1 for v in pol["per_tile_bits"].values() if v == 8)
        n_4 = sum(1 for v in pol["per_tile_bits"].values() if v == 4)
        print(f"  B={B} achieved={pol['achieved_avg_bits']:.3f}  "
              f"8-bit tiles={n_8}/196  4-bit tiles={n_4}/196  → {path.name}")

    # 2) Run residency sweep on each hwnorm policy.
    print("\n=== residency sweep (hwnorm per-tile policies) ===")
    rows = []
    for B, p in hwnorm_paths.items():
        for r in residency.compare(TARGET, p, [m << 20 for m in GBUF_SIZES_MB]):
            r["label"] = f"hwnorm-{B}"
            r["scorer"] = "hwnorm"
            rows.append(r)

    # 3) Existing E3 baselines (per-layer greedy + INT4/INT8 uniform) — re-load.
    baselines = {
        "INT4-uniform":  "experiments/results/policies/qwen25_15b_instruct_4.0bit.yaml",
        "Mixed-4.5":     "experiments/results/policies/qwen25_15b_instruct_4.5bit.yaml",
        "Mixed-5.0":     "experiments/results/policies/qwen25_15b_instruct_5.0bit.yaml",
        "INT8-uniform":  "experiments/results/policies/qwen25_15b_instruct_8.0bit.yaml",
    }
    for tag, p in baselines.items():
        for r in residency.compare(TARGET, p, [m << 20 for m in GBUF_SIZES_MB]):
            r["label"] = tag
            r["scorer"] = "greedy" if "Mixed" in tag else "uniform"
            rows.append(r)

    out_csv = Path("experiments/results/e4_residency_compare.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        fields = ["label", "scorer", "strategy", "gbuf_bytes",
                  "pinned_count", "dram_bytes_per_token", "total_decoder_bytes_aware"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\n=== wrote {out_csv} ({len(rows)} rows) ===")

    # 4) Print compact summary table: GBuf × policy variant × scheduler.
    print("\n=== summary @ GBuf = 512 MB (aware only) ===")
    print(f"{'policy':<18} {'scorer':<10} {'avg_bpw':>8} {'DRAM/tok MB':>13}")
    for tag in ("INT4-uniform", "Mixed-4.5", "hwnorm-4.5",
                "Mixed-5.0", "hwnorm-5.0", "hwnorm-5.5", "hwnorm-6.0", "INT8-uniform"):
        for r in rows:
            if (r["label"] == tag and r["strategy"] == "precision_aware"
                    and (int(r["gbuf_bytes"]) >> 20) == 512):
                # avg_bpw: compute from total/(196 tiles) — coarse
                total = int(r["total_decoder_bytes_aware"])
                bpw = total / 1310.20e6 * 8.0   # 1.31 GB at INT8 = max
                print(f"  {tag:<16} {r['scorer']:<10} {bpw:>8.3f} {int(r['dram_bytes_per_token'])/1e6:>12.1f}")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
