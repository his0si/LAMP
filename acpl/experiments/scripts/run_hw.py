#!/usr/bin/env python
"""Phase 3 driver — Timeloop+Accelergy per (layer × precision).

Reads the policy YAML, expands into LayerWorkload entries with the
correct bit-width per layer, and dispatches to pipeline.hw_timeloop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, hw_timeloop  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--arch", default=None,
                    help="override arch yaml under configs/arch/")
    args = ap.parse_args()

    targets = config.load_targets()
    tspec = targets["targets"][args.target]
    pol = yaml.safe_load(Path(args.policy).read_text())

    workloads = hw_timeloop.layer_workloads_from_model(tspec)
    # Map block-level bits → per-linear bits via the layer index in the name.
    for wl in workloads:
        l = int(wl.name.split(".")[0].replace("layer", ""))
        wl.bits_w = pol["per_layer_bits"][l]

    arch = Path(args.arch) if args.arch else (
        config.EXPERIMENTS_ROOT / "configs" / "arch" / "eyeriss_like.yaml"
    )
    results = []
    for wl in workloads:
        try:
            r = hw_timeloop.run_layer(wl, arch_template=arch)
            results.append(r.__dict__)
        except NotImplementedError as exc:
            print(f"[run_hw] stub: {exc}", file=sys.stderr)
            return 3

    out = config.DEFAULT_RESULTS / "hw" / f"{args.target}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[run_hw] {len(results)} workloads → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
