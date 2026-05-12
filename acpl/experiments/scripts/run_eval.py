#!/usr/bin/env python
"""Phase 4b driver — perplexity and downstream eval."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config, eval as pipe_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path",
                    help="HF dir — either a baseline or a results/quantized/* dir")
    ap.add_argument("--tag", default=None, help="label for the output filename")
    ap.add_argument("--skip-downstream", action="store_true",
                    help="run only wikitext perplexity (fast)")
    ap.add_argument("--tasks", nargs="+",
                    default=["hellaswag", "arc_easy", "arc_challenge", "winogrande"])
    args = ap.parse_args()

    tag = args.tag or Path(args.model_path).name
    out_root = config.DEFAULT_RESULTS / "eval" / tag

    ppl = pipe_eval.eval_perplexity(args.model_path, dataset="wikitext-2")
    pipe_eval.write_json(out_root / "ppl_wikitext2.json", ppl)
    print(f"[run_eval] wikitext2 ppl={ppl['ppl']:.3f}")

    if not args.skip_downstream:
        res = pipe_eval.eval_downstream(args.model_path, tasks=args.tasks)
        pipe_eval.write_json(out_root / "lm_eval.json", res)
        print(f"[run_eval] downstream tasks → {out_root/'lm_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
