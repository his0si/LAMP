#!/usr/bin/env python3
"""Sanity-check the local experiment environment and model manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baseline" / "models.yaml"
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Read configs from Hugging Face when local snapshots are missing.",
    )
    args = parser.parse_args()

    console.print(f"Python/Torch environment")
    console.print(f"- torch: {torch.__version__}")
    console.print(f"- cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            console.print(f"- cuda:{i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")

    with MANIFEST.open("r", encoding="utf-8") as f:
        models = yaml.safe_load(f)["models"]

    table = Table(title="Manifest/config check")
    table.add_column("name")
    table.add_column("repo_id")
    table.add_column("config")
    table.add_column("weights")
    table.add_column("status")

    for name, spec in models.items():
        local_dir = ROOT / spec["local_dir"]
        has_config = (local_dir / "config.json").exists()
        has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
        if (local_dir / "config.json").exists():
            try:
                cfg = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)
                status = f"ok: {getattr(cfg, 'model_type', 'unknown')}"
            except Exception as exc:  # noqa: BLE001
                status = f"failed: {type(exc).__name__}"
        elif args.allow_remote:
            try:
                cfg = AutoConfig.from_pretrained(spec["repo_id"], trust_remote_code=False)
                status = f"remote ok: {getattr(cfg, 'model_type', 'unknown')}"
            except Exception as exc:  # noqa: BLE001
                status = f"remote failed: {type(exc).__name__}"
        else:
            status = "missing local snapshot"
        table.add_row(
            name,
            spec["repo_id"],
            str(has_config).lower(),
            str(has_weights).lower(),
            status,
        )

    console.print(table)


if __name__ == "__main__":
    main()
