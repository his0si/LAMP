"""Phase 4b — accuracy evaluation.

Two modes:
  perplexity(model_path)          → wikitext-2 + c4 perplexity (fast, ~min)
  downstream(model_path, tasks)   → lm-evaluation-harness (slow, ~hours)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sensitivity import compute_perplexity


def _is_gptq_checkpoint(model_path: Path) -> bool:
    """Detect a gptqmodel-produced checkpoint by the presence of
    quantize_config.json — this avoids optimum's broken EXLLAMA_V1 path."""
    return (model_path / "quantize_config.json").exists()


def load_model_for_eval(model_path: Path | str, device_map: str = "auto"):
    p = Path(model_path)
    tok = AutoTokenizer.from_pretrained(str(p))
    if _is_gptq_checkpoint(p):
        from gptqmodel import GPTQModel
        model = GPTQModel.load(str(p), device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(p), dtype=torch.float16, device_map=device_map
        )
    model.eval()
    return model, tok


def eval_perplexity(model_path: Path | str, dataset: str = "wikitext-2") -> dict:
    from datasets import load_dataset

    model, tok = load_model_for_eval(model_path)
    if dataset == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if t.strip()]
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = [next(iter(ds))["text"] for _ in range(256)]
    else:
        raise ValueError(dataset)
    ppl = compute_perplexity(model, tok, texts, device="cuda")
    return {"dataset": dataset, "ppl": ppl}


def eval_downstream(
    model_path: Path | str,
    tasks: Iterable[str] = ("hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu"),
    batch_size: int = 8,
) -> dict:
    """lm-evaluation-harness wrapper. Returns the raw `results` dict."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=str(model_path), dtype="float16", batch_size=batch_size)
    out = simple_evaluate(model=lm, tasks=list(tasks))
    return {"results": out["results"], "configs": out["configs"]}


def write_json(path: Path | str, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
