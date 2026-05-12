"""Phase 4a — build a mixed-precision quantized model from a policy.

GPTQ via gptqmodel/Optimum supports per-module bit-width overrides through
the QuantizeConfig.dynamic argument (regex → override). We map each
transformer block to its policy width and produce a quantized checkpoint
that the eval/profile phases can consume.

Reference: gptqmodel.QuantizeConfig docs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


def build_dynamic_overrides(policy_yaml_path: Path | str) -> dict[str, dict]:
    """Map per-layer bit choices into the regex-keyed dict GPTQModel expects.

    For HF Qwen2/Llama style models, transformer blocks live under
    `model.layers.<idx>.{self_attn.q_proj, ..., mlp.down_proj}`. We promote
    the layer-level decision to every linear inside that layer.
    """
    pol = yaml.safe_load(Path(policy_yaml_path).read_text())
    base_width = min(pol["allowed_widths"])
    overrides: dict[str, dict] = {}
    for idx, bits in enumerate(pol["per_layer_bits"]):
        if bits == base_width:
            continue
        pattern = rf"model\.layers\.{idx}\..*"
        overrides[pattern] = {"bits": bits}
    return overrides


def quantize(
    *,
    model_path: Path | str,
    policy_yaml_path: Path | str,
    out_dir: Path | str,
    base_bits: int = 4,
    group_size: int = 128,
    calib_dataset: str = "wikitext",
    calib_num_samples: int = 128,
    calib_seqlen: int = 2048,
) -> Path:
    """Run GPTQ with per-layer bit overrides. Returns the saved-model dir.

    Heavy lift — only import gptqmodel inside this function to keep the
    package importable on machines that haven't installed it yet.
    """
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    overrides = build_dynamic_overrides(policy_yaml_path)
    qcfg = QuantizeConfig(
        bits=base_bits,
        group_size=group_size,
        desc_act=True,
        dynamic=overrides or None,
    )

    if calib_dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 200][:calib_num_samples]
    else:
        raise NotImplementedError(f"calib_dataset={calib_dataset}")

    tok = AutoTokenizer.from_pretrained(str(model_path))
    # device_map="auto" lets GPTQModel shard across visible GPUs — required
    # for 7B+ on 16 GB cards.
    model = GPTQModel.load(str(model_path), qcfg, device_map="auto")
    model.quantize(texts, batch_size=1, tokenizer=tok)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    return out_dir
