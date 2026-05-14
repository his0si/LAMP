#!/usr/bin/env bash
# Run HellaSwag (1000 samples) on every model: FP16 baseline + 4 GPTQ policies × 4 models.
# Phase 1: small models on dedicated single GPUs in parallel with Qwen-7B on (2,3).
# Phase 2: Llama-8B across all 4 GPUs once the small ones finish.
set -uo pipefail
cd /home/heeseo/LAMP/acpl
source /usr/local/conda/etc/profile.d/conda.sh
conda activate LAMP_acpl

N=1000

# args: GPUS BASELINE_MODELDIR FP_TAG TARGET_KEY GPTQ_POLICIES...
# But simpler: spell out per call.

# Wrapper that records to per-tag json under results/eval/.
hs() {
  local GPUS=$1 MODEL_DIR=$2 TAG=$3 EXTRA=$4
  CUDA_VISIBLE_DEVICES=$GPUS python experiments/scripts/eval_hellaswag.py \
      "$MODEL_DIR" --tag "$TAG" --samples $N $EXTRA \
      >> experiments/results/_hellaswag.log 2>&1
  echo "[hs] $TAG done on GPUs $GPUS"
}

: > experiments/results/_hellaswag.log

echo "[hs] Phase 1: small models (1.5B on GPU 0, 2B on GPU 1) + 7B on GPUs 2,3"

# --- GPU 0: Qwen-1.5B FP16 + 4 policies (sequential, ~5 min total)
{
  hs 0 baseline/models/qwen25_15b_instruct qwen25_15b_fp16 ""
  hs 0 experiments/results/quantized/qwen25_15b_instruct_from_qwen25_15b_instruct_4.0bit qwen25_15b_int4_uniform "--quantized"
  hs 0 experiments/results/quantized/qwen25_15b_instruct_from_qwen25_15b_instruct_4.5bit qwen25_15b_mixed_4p5 "--quantized"
  hs 0 experiments/results/quantized/qwen25_15b_instruct_from_qwen25_15b_instruct_5.0bit qwen25_15b_mixed_5p0 "--quantized"
  hs 0 experiments/results/quantized/qwen25_15b_instruct_from_qwen25_15b_instruct_8.0bit qwen25_15b_int8_uniform "--quantized"
} &
PID_15B=$!

# --- GPU 1: Gemma-2B FP16 + 4 policies
{
  hs 1 baseline/models/gemma2_2b_it gemma2_2b_fp16 ""
  hs 1 experiments/results/quantized/gemma2_2b_it_from_gemma2_2b_it_4.0bit gemma2_2b_int4_uniform "--quantized"
  hs 1 experiments/results/quantized/gemma2_2b_it_from_gemma2_2b_it_4.5bit gemma2_2b_mixed_4p5 "--quantized"
  hs 1 experiments/results/quantized/gemma2_2b_it_from_gemma2_2b_it_5.0bit gemma2_2b_mixed_5p0 "--quantized"
  hs 1 experiments/results/quantized/gemma2_2b_it_from_gemma2_2b_it_8.0bit gemma2_2b_int8_uniform "--quantized"
} &
PID_2B=$!

# --- GPUs 2,3: Qwen-7B FP16 + 4 policies
{
  hs 2,3 baseline/models/qwen25_7b_instruct qwen25_7b_fp16 ""
  hs 2,3 experiments/results/quantized/qwen25_7b_instruct_from_qwen25_7b_instruct_4.0bit qwen25_7b_int4_uniform "--quantized"
  hs 2,3 experiments/results/quantized/qwen25_7b_instruct_from_qwen25_7b_instruct_4.5bit qwen25_7b_mixed_4p5 "--quantized"
  hs 2,3 experiments/results/quantized/qwen25_7b_instruct_from_qwen25_7b_instruct_5.0bit qwen25_7b_mixed_5p0 "--quantized"
  hs 2,3 experiments/results/quantized/qwen25_7b_instruct_from_qwen25_7b_instruct_8.0bit qwen25_7b_int8_uniform "--quantized"
} &
PID_7B=$!

wait $PID_15B $PID_2B $PID_7B
echo "[hs] Phase 1 done"

echo "[hs] Phase 2: Llama-8B across all 4 GPUs (sequential)"
hs 0,1,2,3 baseline/models/llama31_8b_instruct llama31_8b_fp16 ""
hs 0,1,2,3 experiments/results/quantized/llama31_8b_instruct_from_llama31_8b_instruct_4.0bit llama31_8b_int4_uniform "--quantized"
hs 0,1,2,3 experiments/results/quantized/llama31_8b_instruct_from_llama31_8b_instruct_4.5bit llama31_8b_mixed_4p5 "--quantized"
hs 0,1,2,3 experiments/results/quantized/llama31_8b_instruct_from_llama31_8b_instruct_5.0bit llama31_8b_mixed_5p0 "--quantized"
hs 0,1,2,3 experiments/results/quantized/llama31_8b_instruct_from_llama31_8b_instruct_8.0bit llama31_8b_int8_uniform "--quantized"

echo "[hs] Phase 2 done"

# --- Summary table
echo "[hs] === HellaSwag accuracy summary (N=$N) ==="
python - <<'PY'
import json
from pathlib import Path
results = []
for d in sorted(Path("experiments/results/eval").glob("*/hellaswag_acc.json")):
    j = json.loads(d.read_text())
    results.append((j["tag"], j["acc"], j["n_samples"]))
print(f"{'tag':<32} {'acc':>8} {'n':>5}")
for tag, acc, n in results:
    print(f"{tag:<32} {acc:>8.4f} {n:>5}")
PY

echo "ALL HELLASWAG DONE  ($(date +%H:%M:%S))"
