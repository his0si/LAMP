#!/usr/bin/env python3
"""Verify that PyTorch can run a small CUDA operation on the selected GPU."""

from __future__ import annotations

import torch
from rich.console import Console


console = Console()


def main() -> None:
    console.print(f"torch: {torch.__version__}")
    console.print(f"torch cuda build: {torch.version.cuda}")
    console.print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch in this environment.")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    console.print(f"gpu: {props.name}")
    console.print(f"total memory: {props.total_memory / 1024**3:.1f} GiB")

    x = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    y = x @ x.T
    torch.cuda.synchronize()
    console.print(f"matmul ok: shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}")


if __name__ == "__main__":
    main()
