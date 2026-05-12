#!/usr/bin/env bash
# E1: GPTQ + perplexity for 4 policies on Qwen2.5-1.5B-Instruct.
# Calibration: 32 wikitext-2 samples (sufficient for ranking validation).
# Output: results/quantized/<tag>/ and results/eval/<tag>/ppl_wikitext2.json
set -uo pipefail   # no -e: we want to keep going if one policy fails

cd /home/heeseo/LAMP/acpl
LOG=experiments/results/_e1_runlog.txt
: > "$LOG"

run_one() {
  local POLICY=$1 BASE_BITS=$2 TAG=$3
  echo "[E1] === quant: $TAG (policy=$POLICY, base_bits=$BASE_BITS) ===" | tee -a "$LOG"
  python experiments/scripts/run_quant.py qwen25_15b_instruct \
    --policy "$POLICY" --base-bits "$BASE_BITS" --group-size 128 2>&1 \
    | tee -a "$LOG" | tail -2
  echo "[E1] === eval: $TAG ===" | tee -a "$LOG"
  python experiments/scripts/run_eval.py \
    "experiments/results/quantized/qwen25_15b_instruct_from_$(basename "$POLICY" .yaml)" \
    --tag "$TAG" --skip-downstream 2>&1 \
    | tee -a "$LOG" | tail -2
  echo "[E1] done $TAG  ($(date +%H:%M:%S))" | tee -a "$LOG"
}

# Reduce calibration to 32 samples (env passed via PY snippet)
export LAMP_CALIB_N=32

# We need to override calib_num_samples. Cleaner: pass --calib-num env into run_quant.
# For now: edit-in-place via Python helper.
python - <<'PY'
import yaml
p = 'experiments/configs/targets.yaml'
d = yaml.safe_load(open(p))
d['defaults']['calib_num_samples'] = 32
with open(p,'w') as f: yaml.safe_dump(d,f,sort_keys=False)
print('calib_num_samples ->', d['defaults']['calib_num_samples'])
PY

source /usr/local/conda/etc/profile.d/conda.sh
conda activate LAMP_acpl

T0=$(date +%s)
run_one experiments/results/policies/qwen25_15b_instruct_4.0bit.yaml 4 int4_uniform
run_one experiments/results/policies/qwen25_15b_instruct_4.5bit.yaml 4 mixed_4p5
run_one experiments/results/policies/qwen25_15b_instruct_5.0bit.yaml 4 mixed_5p0
run_one experiments/results/policies/qwen25_15b_instruct_8.0bit.yaml 8 int8_uniform
T1=$(date +%s)

echo "[E1] ALL DONE in $((T1-T0))s" | tee -a "$LOG"
echo "[E1] PPL SUMMARY:" | tee -a "$LOG"
for TAG in int4_uniform mixed_4p5 mixed_5p0 int8_uniform; do
  F=experiments/results/eval/$TAG/ppl_wikitext2.json
  if [ -f "$F" ]; then
    PPL=$(python -c "import json; print(json.load(open('$F'))['ppl'])")
    echo "  $TAG  ppl=$PPL" | tee -a "$LOG"
  else
    echo "  $TAG  (missing $F)" | tee -a "$LOG"
  fi
done
