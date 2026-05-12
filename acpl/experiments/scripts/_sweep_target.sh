#!/usr/bin/env bash
# End-to-end sweep for a single target: sensitivity → policy → GPTQ × 4 → eval × 4 → residency.
#
# Usage:
#   conda activate LAMP_acpl
#   CUDA_VISIBLE_DEVICES=1 bash scripts/_sweep_target.sh gemma2_2b_it 2>&1 | tee results/_sweep_gemma2_2b.log
#
# Inherits CUDA_VISIBLE_DEVICES so the caller picks the GPU.
set -uo pipefail

if [ $# -lt 1 ]; then echo "usage: $0 <target_key>"; exit 2; fi
TARGET=$1
cd /home/heeseo/LAMP/acpl

LOG=experiments/results/_sweep_${TARGET}.log
: > "$LOG"
trap 'echo "[sweep:$TARGET] interrupted at line $LINENO"' INT

src() { source /usr/local/conda/etc/profile.d/conda.sh; conda activate LAMP_acpl; }
src

# 1. Per-block sensitivity (use 200 calib texts — same default as Qwen-1.5B baseline).
echo "[sweep:$TARGET] 1/6 sensitivity" | tee -a "$LOG"
python experiments/scripts/run_sensitivity.py "$TARGET" 2>&1 | tee -a "$LOG" | tail -2

# 2. Greedy policies at 4 budgets.
echo "[sweep:$TARGET] 2/6 policies (greedy)" | tee -a "$LOG"
for B in 4.5 5.0 5.5 6.0; do
  python experiments/scripts/run_policy.py "$TARGET" --avg-bits "$B" 2>&1 | tee -a "$LOG" | tail -1
done

# 3. INT4-uniform / INT8-uniform reference policies (hand-author per num_hidden_layers).
echo "[sweep:$TARGET] 3/6 uniform reference policies" | tee -a "$LOG"
python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(open('experiments/configs/targets.yaml'))
target = cfg['targets']['$TARGET']
L = target['num_hidden_layers']
for bits, label in ((4, '4.0bit'), (8, '8.0bit')):
    out = Path(f'experiments/results/policies/${TARGET}_{label}.yaml')
    pol = dict(target='$TARGET', allowed_widths=[bits], target_avg_bits=float(bits),
               achieved_avg_bits=float(bits), per_layer_bits=[bits]*L,
               notes=f'INT{bits}-uniform reference (auto)', scorer='uniform')
    out.write_text(yaml.safe_dump(pol, sort_keys=False))
    print(f'wrote {out}')
PY

# 4-5. GPTQ + eval for each of 4 policies (32-sample calib for tractability).
echo "[sweep:$TARGET] 4-5/6 GPTQ + eval (4 policies × ~10-30 min)" | tee -a "$LOG"
run_qe() {
  local POLICY=$1 BASE=$2 TAG=$3
  echo "    ── quant $TAG (base=${BASE}b) ──" | tee -a "$LOG"
  python experiments/scripts/run_quant.py "$TARGET" --policy "$POLICY" --base-bits "$BASE" --group-size 128 2>&1 \
      | tee -a "$LOG" | tail -1
  local QDIR="experiments/results/quantized/${TARGET}_from_$(basename "$POLICY" .yaml)"
  echo "    ── eval $TAG ──" | tee -a "$LOG"
  python experiments/scripts/run_eval.py "$QDIR" --tag "${TARGET}_${TAG}" --skip-downstream 2>&1 \
      | tee -a "$LOG" | grep -E "ppl|done|missing" | tail -2
}

run_qe "experiments/results/policies/${TARGET}_4.0bit.yaml" 4 "int4_uniform"
run_qe "experiments/results/policies/${TARGET}_4.5bit.yaml" 4 "mixed_4p5"
run_qe "experiments/results/policies/${TARGET}_5.0bit.yaml" 4 "mixed_5p0"
run_qe "experiments/results/policies/${TARGET}_8.0bit.yaml" 8 "int8_uniform"

# 6. Residency sweep + codesign CSV.
echo "[sweep:$TARGET] 6/6 residency + codesign" | tee -a "$LOG"
python - <<PY 2>&1 | tee -a "$LOG"
import sys, csv; sys.path.insert(0, 'experiments')
from pipeline import residency, codesign

target = '$TARGET'
policies = {
    'INT4-uniform':  f'experiments/results/policies/{target}_4.0bit.yaml',
    'Mixed-4.5':     f'experiments/results/policies/{target}_4.5bit.yaml',
    'Mixed-5.0':     f'experiments/results/policies/{target}_5.0bit.yaml',
    'INT8-uniform':  f'experiments/results/policies/{target}_8.0bit.yaml',
}
policy_to_eval = {
    'INT4-uniform': f'{target}_int4_uniform',
    'Mixed-4.5':    f'{target}_mixed_4p5',
    'Mixed-5.0':    f'{target}_mixed_5p0',
    'INT8-uniform': f'{target}_int8_uniform',
}
gbufs = [(m << 20) for m in (128, 256, 512, 1024)]
rows = []
for tag, p in policies.items():
    for r in residency.compare(target, p, gbufs):
        r['label'] = tag; rows.append(r)
with open(f'experiments/results/residency_sweep_{target}.csv', 'w') as f:
    fn = ['label','strategy','gbuf_bytes','pinned_count','pinned_bytes_aware','dram_bytes_per_token','total_decoder_bytes_aware']
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
    for r in rows: w.writerow({k:r.get(k) for k in fn})
print(f'wrote experiments/results/residency_sweep_{target}.csv ({len(rows)} rows)')

pts = codesign.build_residency_codesign(
    residency_csv=f'experiments/results/residency_sweep_{target}.csv',
    policy_to_eval_tag=policy_to_eval,
    policy_to_yaml=policies)
codesign.save_residency_csv(pts, f'experiments/results/codesign_{target}.csv')
codesign.plot_residency_pareto_single_gbuf(pts, gbuf_MB=512,
    out_path=f'experiments/results/figs/codesign_{target}_512.png')
codesign.plot_residency_pareto_multi_gbuf(pts, gbuf_MB_list=[128,512,1024],
    out_path=f'experiments/results/figs/codesign_{target}_multi.png')
print(f'wrote figs/codesign_{target}_{{512,multi}}.png')

# Quick summary
print('=== summary @ GBuf=512 MB, aware sched ===')
for p in sorted(pts, key=lambda x: x.avg_bpw):
    if p.gbuf_MB == 512 and p.scheduler == 'aware' and p.ppl is not None:
        print(f'  {p.policy_tag:<14}  ppl={p.ppl:.3f}  DRAM/tok={p.dram_bytes_per_token/1e6:.1f}MB')
PY

echo "[sweep:$TARGET] ALL DONE  ($(date +%H:%M:%S))" | tee -a "$LOG"
