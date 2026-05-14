#!/usr/bin/env python
"""Minimal HellaSwag accuracy evaluation for FP16 or GPTQ checkpoints.

Loads a model, picks N samples from the HellaSwag validation split, and
scores the four candidate endings by length-normalized log-likelihood
(the standard zero-shot HellaSwag protocol). Reports top-1 accuracy.

Usage:
    python eval_hellaswag.py /path/to/model_dir --tag <tag> [--samples N]
                             [--quantized]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def score_hellaswag(model, tok, n_samples: int) -> dict:
    """Return accuracy + raw stats on a HellaSwag validation subset."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if n_samples and n_samples < len(ds):
        ds = ds.select(range(n_samples))
    n = len(ds)
    correct = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for ex in ds:
            ctx = ex["activity_label"] + ": " + ex["ctx"]
            gold = int(ex["label"])
            scores = []
            ctx_ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            ctx_len = ctx_ids.size(1)
            for end in ex["endings"]:
                full_ids = tok(ctx + " " + end, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
                if full_ids.size(1) <= ctx_len:
                    scores.append(float("-inf"))
                    continue
                logits = model(full_ids).logits
                cont_ids = full_ids[0, ctx_len:]
                cont_logits = logits[0, ctx_len - 1: full_ids.size(1) - 1]
                logprobs = torch.log_softmax(cont_logits.float(), dim=-1)
                chosen = logprobs.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
                scores.append(float(chosen.sum().item()) / max(1, cont_ids.numel()))
            pred = int(torch.tensor(scores).argmax().item())
            correct += int(pred == gold)
    return {"n_samples": n, "correct": correct, "acc": correct / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--tag", required=True, help="output dir name under results/eval/")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--quantized", action="store_true",
                    help="Load via GPTQModel (for E1/E4 quantized checkpoints).")
    args = ap.parse_args()

    if args.quantized:
        from gptqmodel import GPTQModel
        model = GPTQModel.load(args.model_dir, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.float16, device_map="auto",
        )
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model.eval()

    t0 = time.time()
    stats = score_hellaswag(model, tok, args.samples)
    dt = time.time() - t0

    out_dir = Path("experiments/results/eval") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "model_dir": str(args.model_dir),
        "task": "hellaswag",
        "wall_seconds": round(dt, 1),
        **stats,
    }
    (out_dir / "hellaswag_acc.json").write_text(json.dumps(payload, indent=2))
    print(f"[hellaswag] {args.tag}: acc={stats['acc']:.4f}  "
          f"n={stats['n_samples']}  wall={dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
