#!/usr/bin/env bash
# Generate hwnorm policies + GPTQ + eval for the 3 non-Qwen-1.5B targets.
# Assumes per-projection sensitivity JSONs already exist for each target.
set -uo pipefail

cd /home/heeseo/LAMP/acpl
source /usr/local/conda/etc/profile.d/conda.sh
conda activate LAMP_acpl

# ---- 1. Generate hwnorm per-tile policies (instant) ---------------------
echo "[hwnorm-3m] 1/3 generate per-tile policies"
python - <<'PY'
import sys
sys.path.insert(0, "experiments")
from pipeline import config, policy

cfg = config.load_targets()
TARGETS = {
    "gemma2_2b_it":        "experiments/results/sensitivity/gemma2_2b_it_per_projection.json",
    "qwen25_7b_instruct":  "experiments/results/sensitivity/qwen25_7b_instruct_per_projection.json",
    "llama31_8b_instruct": "experiments/results/sensitivity/llama31_8b_instruct_per_projection.json",
}
for tgt, per_proj in TARGETS.items():
    tcfg = cfg["targets"][tgt]
    for B in (4.5, 5.0):
        pol = policy.make_per_tile_policy(
            per_proj, target_cfg=tcfg,
            allowed_widths=[4, 8], target_avg_bits=B, scorer="hwnorm",
        )
        out = policy.save_per_tile(pol, "experiments/results/policies")
        n8 = sum(1 for v in pol["per_tile_bits"].values() if v == 8)
        print(f"  {tgt} B={B}  achieved={pol['achieved_avg_bits']:.3f}  8-bit={n8}  → {out.name}")
PY

# ---- 2. GPTQ + eval, phased to fit 4 GPUs -------------------------------
run_qe() {
  local GPUS=$1 TGT=$2 BPW=$3
  local POL=experiments/results/policies/${TGT}_${BPW}bit_hwnorm.yaml
  local TAG=${TGT}_hwnorm_${BPW//./p}
  local QDIR=experiments/results/quantized/${TGT}_from_${TGT}_${BPW}bit_hwnorm
  local LOG=experiments/results/_hwnorm_${TGT}_${BPW}.log
  : > "$LOG"
  CUDA_VISIBLE_DEVICES=$GPUS python experiments/scripts/run_quant.py "$TGT" \
      --policy "$POL" --base-bits 4 --group-size 128 >> "$LOG" 2>&1
  CUDA_VISIBLE_DEVICES=$GPUS python experiments/scripts/run_eval.py "$QDIR" \
      --tag "$TAG" --skip-downstream >> "$LOG" 2>&1
  echo "[hwnorm-3m] $TGT B=$BPW done on GPUs $GPUS"
}

echo "[hwnorm-3m] 2/3 Phase A — Gemma×2 (single-GPU each) + Qwen-7B B=4.5 (2 GPUs)"
run_qe 0   gemma2_2b_it      4.5 &
run_qe 1   gemma2_2b_it      5.0 &
run_qe 2,3 qwen25_7b_instruct 4.5 &
wait

echo "[hwnorm-3m] 2/3 Phase B — Qwen-7B B=5.0 + Llama-8B B=4.5 (2 GPUs each)"
run_qe 0,1 qwen25_7b_instruct 5.0 &
run_qe 2,3 llama31_8b_instruct 4.5 &
wait

echo "[hwnorm-3m] 2/3 Phase C — Llama-8B B=5.0 (4 GPUs)"
run_qe 0,1,2,3 llama31_8b_instruct 5.0

# ---- 3. Summarize ppl ----------------------------------------------------
echo "[hwnorm-3m] 3/3 ppl summary"
for TGT in gemma2_2b_it qwen25_7b_instruct llama31_8b_instruct; do
  for BPW in 4.5 5.0; do
    F=experiments/results/eval/${TGT}_hwnorm_${BPW//./p}/ppl_wikitext2.json
    if [[ -f "$F" ]]; then
      PPL=$(python -c "import json; print(json.load(open('$F')).get('ppl'))")
      echo "  $TGT  hwnorm-$BPW   ppl=$PPL"
    else
      echo "  $TGT  hwnorm-$BPW   (eval missing: $F)"
    fi
  done
done

echo "ALL HWNORM-3M DONE  ($(date +%H:%M:%S))"
