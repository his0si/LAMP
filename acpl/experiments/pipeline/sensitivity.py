"""Phase 1 — per-layer sensitivity analysis.

Approach: simulated-quantization ablation.
For each transformer block l and candidate bit-width b ∈ {4, 8}:
  1. Wrap every nn.Linear in block l with a fake-quantizer at b bits.
  2. Compute perplexity on a held-out slice of WikiText-2.
  3. Record ppl(l, b) - ppl(fp16) as the sensitivity score.

Fake quantization is symmetric per-channel min/max, applied only to the
weight tensor at forward time (activations stay fp16). This keeps the
analysis fast (~minutes for a 1.5B model on one RTX 2000 Ada) while still
ranking layers consistently with what GPTQ will see in Phase 4.

Output: results/sensitivity/{target}.json with shape
    { "fp16_ppl": float,
      "layers": [ {"idx": int, "ppl_4bit": float, "ppl_8bit": float, ...}, ... ] }
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class SensitivityResult:
    target: str
    fp16_ppl: float
    per_layer: list[dict]

    def to_json(self) -> str:
        return json.dumps(
            {"target": self.target, "fp16_ppl": self.fp16_ppl, "layers": self.per_layer},
            indent=2,
        )


def fake_quantize_(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric uniform quantization, returned as fp16."""
    qmax = 2 ** (bits - 1) - 1
    w = weight.detach()
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(w / scale).clamp_(-qmax - 1, qmax)
    return (q * scale).to(weight.dtype)


@contextmanager
def quantize_block(block: nn.Module, bits: int):
    """Temporarily replace every Linear weight inside `block` with its
    fake-quantized version. Restores originals on exit."""
    backups: list[tuple[nn.Linear, torch.Tensor]] = []
    try:
        for mod in block.modules():
            if isinstance(mod, nn.Linear):
                backups.append((mod, mod.weight.data.clone()))
                mod.weight.data.copy_(fake_quantize_(mod.weight, bits))
        yield
    finally:
        for mod, w in backups:
            mod.weight.data.copy_(w)


@contextmanager
def quantize_projection(block: nn.Module, projection_role: str, bits: int):
    """Quantize ONE projection (q/k/v/o/gate/up/down_proj) within a block."""
    if projection_role in ("q_proj", "k_proj", "v_proj", "o_proj"):
        parent = block.self_attn
    elif projection_role in ("gate_proj", "up_proj", "down_proj"):
        parent = block.mlp
    else:
        raise ValueError(f"unknown projection role: {projection_role}")
    mod = getattr(parent, projection_role)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{projection_role} is not nn.Linear")
    backup = mod.weight.data.clone()
    try:
        mod.weight.data.copy_(fake_quantize_(mod.weight, bits))
        yield
    finally:
        mod.weight.data.copy_(backup)


@torch.no_grad()
def compute_perplexity(
    model, tokenizer, texts: Iterable[str], seq_len: int = 2048, device: str = "cuda"
) -> float:
    """Standard sliding-window perplexity over a list of strings."""
    model.eval()
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    total_nll, total_tokens = 0.0, 0
    for start in range(0, encodings.size(1) - 1, seq_len):
        end = min(start + seq_len, encodings.size(1))
        chunk = encodings[:, start:end]
        if chunk.size(1) < 2:
            break
        out = model(chunk, labels=chunk)
        n_tok = chunk.size(1) - 1
        total_nll += out.loss.item() * n_tok
        total_tokens += n_tok
    return float(torch.tensor(total_nll / total_tokens).exp())


def find_transformer_blocks(model) -> list[nn.Module]:
    """Return the list of decoder blocks for common architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        cur = model
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and isinstance(cur, (list, nn.ModuleList)):
            return list(cur)
    raise RuntimeError("Could not locate transformer block list on model.")


def run_sensitivity(
    model,
    tokenizer,
    *,
    target: str,
    bits_to_try: list[int] = (4, 8),
    calib_texts: list[str] | None = None,
    device: str = "cuda",
) -> SensitivityResult:
    if calib_texts is None:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        calib_texts = [t for t in ds["text"] if t.strip()][:200]

    fp16_ppl = compute_perplexity(model, tokenizer, calib_texts, device=device)
    blocks = find_transformer_blocks(model)

    rows: list[dict] = []
    for idx, block in enumerate(blocks):
        row = {"idx": idx}
        for b in bits_to_try:
            with quantize_block(block, b):
                ppl = compute_perplexity(model, tokenizer, calib_texts, device=device)
            row[f"ppl_{b}bit"] = ppl
            row[f"delta_{b}bit"] = ppl - fp16_ppl
        rows.append(row)

    return SensitivityResult(target=target, fp16_ppl=fp16_ppl, per_layer=rows)


def save(result: SensitivityResult, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.target}.json"
    out_path.write_text(result.to_json())
    return out_path


# ---- Per-projection sensitivity (E4) ----

PROJECTION_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")


def run_per_projection_sensitivity(
    model,
    tokenizer,
    *,
    target: str,
    bits_to_try: list[int] = (4, 8),
    calib_texts: list[str] | None = None,
    device: str = "cuda",
) -> dict:
    """Per-(layer, projection) ablation. ~5× longer wall-time than the
    per-block version (28 × 7 = 196 ablations × len(bits_to_try))."""
    if calib_texts is None:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        calib_texts = [t for t in ds["text"] if t.strip()][:200]

    fp16_ppl = compute_perplexity(model, tokenizer, calib_texts, device=device)
    blocks = find_transformer_blocks(model)

    tiles: list[dict] = []
    for layer_idx, block in enumerate(blocks):
        for role in PROJECTION_ROLES:
            row = {"layer_idx": layer_idx, "role": role}
            for b in bits_to_try:
                with quantize_projection(block, role, b):
                    ppl = compute_perplexity(model, tokenizer, calib_texts, device=device)
                row[f"ppl_{b}bit"] = ppl
                row[f"delta_{b}bit"] = ppl - fp16_ppl
            tiles.append(row)

    return {"target": target, "fp16_ppl": fp16_ppl, "tiles": tiles}


def save_per_projection(result: dict, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result['target']}_per_projection.json"
    out_path.write_text(json.dumps(result, indent=2))
    return out_path
