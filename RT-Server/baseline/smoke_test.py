#!/usr/bin/env python3
"""Run a short load/generation smoke test for downloaded baseline models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "baseline" / "models.yaml"
DEFAULT_PROMPT = "Summarize layer-wise mixed precision in one sentence."
console = Console()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    models = raw.get("models", {})
    if not models:
        raise ValueError(f"No models found in {path}")
    return models


def resolve_selection(models: dict[str, dict[str, Any]], requested: list[str]) -> list[str]:
    if not requested or requested == ["all"]:
        return [name for name, spec in models.items() if spec["role"] != "optional_small_gemma"]

    missing = sorted(set(requested) - set(models))
    if missing:
        available = ", ".join(sorted(models))
        raise SystemExit(f"Unknown model(s): {', '.join(missing)}. Available: {available}")
    return requested


def model_source(name: str, spec: dict[str, Any], allow_remote: bool) -> Path | str:
    local_dir = ROOT / spec["local_dir"]
    if (local_dir / "config.json").exists():
        return local_dir
    if allow_remote:
        return spec["repo_id"]
    raise FileNotFoundError(
        f"{name} is not downloaded at {local_dir}. Run baseline/download_models.py first "
        "or pass --allow-remote."
    )


def run_one(name: str, spec: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    source = model_source(name, spec, args.allow_remote)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_kwargs: dict[str, Any] = {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        source,
        **load_kwargs,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = text[len(args.prompt) :].strip() if text.startswith(args.prompt) else text.strip()
    location = "local" if isinstance(source, Path) else "remote"
    return location, completion.replace("\n", " ")[:160]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Model keys from models.yaml, or 'all'.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use bitsandbytes 8-bit loading.")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Load from Hugging Face if the local snapshot is not present.",
    )
    args = parser.parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose only one of --load-in-4bit or --load-in-8bit.")

    models = load_manifest(args.manifest)
    names = resolve_selection(models, args.models)

    table = Table(title="Baseline smoke test")
    table.add_column("name")
    table.add_column("source")
    table.add_column("status")
    table.add_column("sample")

    failed = False
    for name in names:
        try:
            source, sample = run_one(name, models[name], args)
            table.add_row(name, source, "ok", sample)
        except Exception as exc:  # noqa: BLE001
            failed = True
            table.add_row(name, "-", f"failed: {type(exc).__name__}", str(exc)[:160])

    console.print(table)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
