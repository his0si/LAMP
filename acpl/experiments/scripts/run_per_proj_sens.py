#!/usr/bin/env python
"""Per-projection sensitivity runner for an arbitrary target.

Generates the ``{target}_per_projection.json`` input that the E4
hardware-normalized scorer consumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import config
from pipeline.sensitivity import (
    run_per_projection_sensitivity,
    save_per_projection,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--calib", type=int, default=50,
                    help="number of wikitext-2 test texts for the ppl signal")
    args = ap.parse_args()

    cfg = config.load_targets()
    spec = config.get_target(cfg, args.target)
    model_path = spec.weights_path(cfg["defaults"]["models_root"])

    print(f"[per-proj] loading {args.target} from {model_path}")
    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16, device_map="auto",
    )

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t.strip()][: args.calib]

    print(f"[per-proj] running 196 ablations × 2 bits with {len(texts)} calib texts")
    result = run_per_projection_sensitivity(
        model, tok, target=args.target, calib_texts=texts,
    )
    out = save_per_projection(result, "experiments/results/sensitivity")
    print(f"[per-proj] wrote {out}  fp16_ppl={result['fp16_ppl']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
