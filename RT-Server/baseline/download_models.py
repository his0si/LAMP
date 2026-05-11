#!/usr/bin/env python3
"""Download baseline model snapshots declared in baseline/models.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "baseline" / "models.yaml"
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


def print_plan(models: dict[str, dict[str, Any]], names: list[str]) -> None:
    table = Table(title="Baseline download plan")
    table.add_column("name")
    table.add_column("repo_id")
    table.add_column("role")
    table.add_column("gated")
    table.add_column("local_dir")

    for name in names:
        spec = models[name]
        table.add_row(
            name,
            spec["repo_id"],
            spec["role"],
            str(spec.get("gated", False)).lower(),
            spec["local_dir"],
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        help="Model keys from models.yaml, or 'all'. Default: all non-optional baselines.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--revision", default=None, help="Optional HF revision/commit/tag.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded.")
    parser.add_argument(
        "--local-dir-use-symlinks",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Forwarded to huggingface_hub.snapshot_download.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue downloading the remaining models if one model fails.",
    )
    args = parser.parse_args()

    models = load_manifest(args.manifest)
    names = resolve_selection(models, args.models)
    print_plan(models, names)

    if args.dry_run:
        console.print("[yellow]Dry run only; no files were downloaded.[/yellow]")
        return

    failed: list[tuple[str, str]] = []
    for name in names:
        spec = models[name]
        local_dir = ROOT / spec["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Downloading {name}[/bold] from {spec['repo_id']} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=args.revision,
                local_dir=local_dir,
                local_dir_use_symlinks=args.local_dir_use_symlinks,
                allow_patterns=[
                    "*.json",
                    "*.model",
                    "*.safetensors",
                    "*.txt",
                    "tokenizer.*",
                    "generation_config.json",
                    "special_tokens_map.json",
                ],
                ignore_patterns=[
                    "*.bin",
                    "*.h5",
                    "*.msgpack",
                    "*.onnx",
                    "*.tflite",
                ],
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            failed.append((name, message))
            console.print(f"[red]Failed to download {name}: {message}[/red]")
            if not args.keep_going:
                raise

    if failed:
        console.print("[red]Completed with failures:[/red]")
        for name, message in failed:
            console.print(f"- {name}: {message}")
        raise SystemExit(1)

    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
