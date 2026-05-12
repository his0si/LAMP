#!/usr/bin/env python
"""Phase 1 driver — per-layer sensitivity analysis.

Usage:
  conda activate LAMP_acpl
  python experiments/scripts/run_sensitivity.py qwen25_15b_instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, sensitivity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="key into experiments/configs/targets.yaml")
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--out", default=None, help="override output dir")
    args = ap.parse_args()

    cfg = config.load_targets()
    spec = config.get_target(cfg, args.target)
    model_path = spec.weights_path(cfg["defaults"]["models_root"])
    if not model_path.exists():
        print(f"[run_sensitivity] missing weights: {model_path}", file=sys.stderr)
        return 2

    print(f"[run_sensitivity] loading {model_path}")
    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16, device_map="auto"
    ).eval()

    result = sensitivity.run_sensitivity(
        model, tok, target=args.target, bits_to_try=args.bits
    )
    out_dir = Path(args.out) if args.out else config.DEFAULT_RESULTS / "sensitivity"
    saved = sensitivity.save(result, out_dir)
    print(f"[run_sensitivity] wrote {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
