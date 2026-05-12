#!/usr/bin/env python
"""Phase 4a driver — build a GPTQ-quantized model from a policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, quant_runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--policy", required=True,
                    help="path to a policy YAML produced by run_policy.py")
    ap.add_argument("--base-bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    cfg = config.load_targets()
    spec = config.get_target(cfg, args.target)
    model_path = spec.weights_path(cfg["defaults"]["models_root"])

    out_dir = config.DEFAULT_RESULTS / "quantized" / f"{args.target}_from_{Path(args.policy).stem}"
    saved = quant_runner.quantize(
        model_path=model_path,
        policy_yaml_path=args.policy,
        out_dir=out_dir,
        base_bits=args.base_bits,
        group_size=args.group_size,
        calib_dataset=cfg["defaults"]["calib_dataset"],
        calib_num_samples=cfg["defaults"]["calib_num_samples"],
        calib_seqlen=cfg["defaults"]["calib_seqlen"],
    )
    print(f"[run_quant] saved → {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
