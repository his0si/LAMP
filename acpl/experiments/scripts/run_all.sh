#!/usr/bin/env bash
# End-to-end pipeline. Override TARGET / AVG_BITS via env, e.g.:
#   TARGET=qwen25_15b_instruct AVG_BITS=5.0 ./scripts/run_all.sh
set -euo pipefail

TARGET="${TARGET:-qwen25_15b_instruct}"
AVG_BITS="${AVG_BITS:-5.0}"

if [ "${CONDA_DEFAULT_ENV:-}" != "LAMP_acpl" ]; then
  echo "Run 'conda activate LAMP_acpl' first." >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE/.."   # we expect commands to run from acpl/

echo "[run_all] 1/5 sensitivity"
python experiments/scripts/run_sensitivity.py "$TARGET"

echo "[run_all] 2/5 policy"
python experiments/scripts/run_policy.py "$TARGET" --avg-bits "$AVG_BITS"

POLICY_FILE="experiments/results/policies/${TARGET}_${AVG_BITS}bit.yaml"

echo "[run_all] 3/5 quantize"
python experiments/scripts/run_quant.py "$TARGET" --policy "$POLICY_FILE"

QUANT_DIR="experiments/results/quantized/${TARGET}_from_$(basename "$POLICY_FILE" .yaml)"

echo "[run_all] 4/5 eval (perplexity only — pass --skip-downstream off to run lm-eval)"
python experiments/scripts/run_eval.py "$QUANT_DIR" --skip-downstream

echo "[run_all] 5/5 GPU profile"
python experiments/scripts/run_profile.py "$QUANT_DIR"

echo "[run_all] HW sim (will fail until Timeloop arch is wired up — Phase 3)"
python experiments/scripts/run_hw.py "$TARGET" --policy "$POLICY_FILE" || true

echo "[run_all] done. See experiments/results/"
