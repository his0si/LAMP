#!/usr/bin/env bash
# Qwen-7B specific sweep: multi-GPU sharded sensitivity + GPTQ, 50-text calib.
set -uo pipefail
TARGET=qwen25_7b_instruct
cd /home/heeseo/LAMP/acpl
LOG=experiments/results/_sweep_${TARGET}.log
: > "$LOG"

source /usr/local/conda/etc/profile.d/conda.sh
conda activate LAMP_acpl
export CUDA_VISIBLE_DEVICES=2,3   # caller can override if needed

# 1. Multi-GPU sharded sensitivity with 50 calib texts.
echo "[sweep:$TARGET] 1/6 sensitivity (50 calib, multi-GPU)" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG" | tail -3
import sys, time, torch
sys.path.insert(0, 'experiments')
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pipeline import config, sensitivity

cfg = config.load_targets()
spec = config.get_target(cfg, 'qwen25_7b_instruct')
mp = spec.weights_path(cfg['defaults']['models_root'])
tok = AutoTokenizer.from_pretrained(str(mp))
print('loading 7B with device_map=auto (multi-GPU sharded)...')
model = AutoModelForCausalLM.from_pretrained(str(mp), dtype=torch.float16, device_map='auto').eval()

ds = load_dataset('wikitext','wikitext-2-raw-v1', split='test')
texts = [t for t in ds['text'] if t.strip()][:50]
print(f'running per-block sensitivity ({spec.num_hidden_layers} layers × 2 bits, 50 calib)...')
t0=time.time()
res = sensitivity.run_sensitivity(model, tok, target='qwen25_7b_instruct', bits_to_try=[4,8], calib_texts=texts)
print(f'done in {time.time()-t0:.1f}s; fp16_ppl={res.fp16_ppl:.3f}')
sensitivity.save(res, 'experiments/results/sensitivity')
PY

# 2. Greedy policies.
echo "[sweep:$TARGET] 2/6 policies" | tee -a "$LOG"
for B in 4.5 5.0 5.5 6.0; do
  python experiments/scripts/run_policy.py "$TARGET" --avg-bits "$B" 2>&1 | tee -a "$LOG" | tail -1
done

# 3. Uniform reference policies.
echo "[sweep:$TARGET] 3/6 uniform refs" | tee -a "$LOG"
python - <<PY 2>&1 | tee -a "$LOG"
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

# 4-5. GPTQ + eval × 4. GPTQModel.load now uses device_map="auto" (multi-GPU sharded for 7B).
echo "[sweep:$TARGET] 4-5/6 GPTQ + eval (4 policies × ~30-60 min)" | tee -a "$LOG"
run_qe() {
  local POLICY=$1 BASE=$2 TAG=$3
  echo "  ── quant $TAG ──" | tee -a "$LOG"
  python experiments/scripts/run_quant.py "$TARGET" --policy "$POLICY" --base-bits "$BASE" --group-size 128 2>&1 \
      | tee -a "$LOG" | tail -1
  local QDIR="experiments/results/quantized/${TARGET}_from_$(basename "$POLICY" .yaml)"
  echo "  ── eval $TAG ──" | tee -a "$LOG"
  python experiments/scripts/run_eval.py "$QDIR" --tag "${TARGET}_${TAG}" --skip-downstream 2>&1 \
      | tee -a "$LOG" | grep -E "ppl|done|missing" | tail -1
}
run_qe "experiments/results/policies/${TARGET}_4.0bit.yaml" 4 "int4_uniform"
run_qe "experiments/results/policies/${TARGET}_4.5bit.yaml" 4 "mixed_4p5"
run_qe "experiments/results/policies/${TARGET}_5.0bit.yaml" 4 "mixed_5p0"
run_qe "experiments/results/policies/${TARGET}_8.0bit.yaml" 8 "int8_uniform"

# 6. Residency sweep + codesign.
echo "[sweep:$TARGET] 6/6 residency + codesign" | tee -a "$LOG"
python - <<PY 2>&1 | tee -a "$LOG"
import sys, csv; sys.path.insert(0, 'experiments')
from pipeline import residency, codesign
target = '$TARGET'
policies = {n: f'experiments/results/policies/{target}_{b}bit.yaml'
            for n,b in (('INT4-uniform','4.0'),('Mixed-4.5','4.5'),('Mixed-5.0','5.0'),('INT8-uniform','8.0'))}
policy_to_eval = {n: f'{target}_{s}' for n,s in
                  (('INT4-uniform','int4_uniform'),('Mixed-4.5','mixed_4p5'),
                   ('Mixed-5.0','mixed_5p0'),('INT8-uniform','int8_uniform'))}
gbufs = [(m << 20) for m in (128, 256, 512, 1024, 2048, 4096)]   # 7B is bigger, sweep more
rows = []
for tag, p in policies.items():
    for r in residency.compare(target, p, gbufs):
        r['label'] = tag; rows.append(r)
with open(f'experiments/results/residency_sweep_{target}.csv', 'w') as f:
    fn=['label','strategy','gbuf_bytes','pinned_count','pinned_bytes_aware','dram_bytes_per_token','total_decoder_bytes_aware']
    w=csv.DictWriter(f, fieldnames=fn); w.writeheader()
    for r in rows: w.writerow({k:r.get(k) for k in fn})
print(f'wrote residency_sweep_{target}.csv ({len(rows)} rows)')
pts = codesign.build_residency_codesign(
    residency_csv=f'experiments/results/residency_sweep_{target}.csv',
    policy_to_eval_tag=policy_to_eval, policy_to_yaml=policies)
codesign.save_residency_csv(pts, f'experiments/results/codesign_{target}.csv')
codesign.plot_residency_pareto_single_gbuf(pts, gbuf_MB=1024,
    out_path=f'experiments/results/figs/codesign_{target}_1024.png')
codesign.plot_residency_pareto_multi_gbuf(pts, gbuf_MB_list=[512,1024,2048],
    out_path=f'experiments/results/figs/codesign_{target}_multi.png')
print('=== summary @ GBuf=1024 MB ===')
for p in sorted(pts, key=lambda x:x.avg_bpw):
    if p.gbuf_MB==1024 and p.scheduler=='aware' and p.ppl is not None:
        print(f'  {p.policy_tag:<14}  ppl={p.ppl:.3f}  DRAM/tok={p.dram_bytes_per_token/1e6:.1f}MB')
PY

echo "[sweep:$TARGET] ALL DONE  ($(date +%H:%M:%S))" | tee -a "$LOG"
