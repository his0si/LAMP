#!/usr/bin/env python
"""Phase 5 driver — GPU latency / throughput / energy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, profile_gpu  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--prompt", default="Explain in detail how transformer self-attention works.")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    res = profile_gpu.profile_generation(
        args.model_path, prompt=args.prompt,
        new_tokens=args.new_tokens, gpu_index=args.gpu,
    )
    tag = args.tag or Path(args.model_path).name
    out_path = config.DEFAULT_RESULTS / "profile" / f"{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res.__dict__, indent=2))
    print(f"[run_profile] tok/s={res.tokens_per_s:.2f}  "
          f"mJ/tok={res.energy_per_token_mJ:.2f}  → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
