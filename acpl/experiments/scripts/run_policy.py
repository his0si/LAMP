#!/usr/bin/env python
"""Phase 2 driver — mixed-precision policy generation.

Usage:
  python experiments/scripts/run_policy.py qwen25_15b_instruct --avg-bits 5.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--avg-bits", type=float, default=None)
    ap.add_argument("--policy-cfg", default=None,
                    help="override path to policy_search.yaml")
    args = ap.parse_args()

    pcfg_path = Path(args.policy_cfg) if args.policy_cfg else (
        config.EXPERIMENTS_ROOT / "configs" / "policy_search.yaml"
    )
    pcfg = config.load_yaml(pcfg_path)
    if args.avg_bits is not None:
        pcfg["target_avg_bits"] = args.avg_bits

    sens_path = config.DEFAULT_RESULTS / "sensitivity" / f"{args.target}.json"
    if not sens_path.exists():
        print(f"[run_policy] run Phase 1 first; missing {sens_path}", file=sys.stderr)
        return 2

    pol = policy.make_policy(sens_path, pcfg)
    out_dir = config.DEFAULT_RESULTS / "policies"
    saved = policy.save(pol, out_dir)
    print(f"[run_policy] target_avg={pol.target_avg_bits}  "
          f"achieved_avg={pol.achieved_avg_bits:.3f}  → {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
