# LAMP 실험 환경 / Experiments

이 폴더는 **Layer-wise Mixed-Precision LLM 가속기 스케줄링** 연구를 위한
실험 코드와 설정의 단일 진입점입니다. `../baseline/`은 손대지 않습니다 —
거기 있는 모델/스크립트는 말 그대로 baseline 스냅샷이고, 모든 실험적
변경은 여기 안에서만 일어납니다.

> **선행 조건.** 이 폴더의 모든 명령은 `conda activate LAMP_acpl` 상태에서
> 실행되어야 합니다. `conda` 명령이 없다고 나오면 README 루트(`../README.md`)
> §3-1을 따라 `source /usr/local/conda/etc/profile.d/conda.sh`를 실행하거나,
> 이미 `~/.bashrc`에 추가되어 있으니 새 SSH 세션을 다시 여세요.

---

## 1. 연구 주제 / Topic

### Paper claim (2026-05-12 — E2-residency pivot 후 확정)

> **"Given a fixed mixed-precision policy and a finite on-chip buffer,
> *precision-aware residency scheduling* pins more layers' weights on-chip
> and reduces per-token DRAM traffic versus a precision-oblivious
> scheduler that budgets every tile at the worst-case bit-width."**

한 줄 모티베이션: *"Layer-wise mixed precision shrinks the model so more
fits on-chip; ignoring per-layer bit-widths at scheduling time leaves that
fit advantage on the table."*

> **Pivot 이력.** 원 가설은 "precision-aware mapping이 energy/token을 줄인다"
> 였음. E2 mapping pilot (§8b) 에서 우리 16×16 Eyeriss arch + N=1 decode regime
> 에서는 mapping geometry가 cycle/energy 차이를 거의 만들지 않고 byte-width로
> 다 흡수됨을 확인. mapping → residency scheduling으로 contribution surface
> 이동. negative mapping result는 motivation 사다리로 보존 (§8b → §8c).

### Contribution layering

| 층위 | 역할 | 본 논문 기여? |
| --- | --- | --- |
| Layer-wise quantization sensitivity | **입력 (known)** — 후반부 layer 민감도는 motivating figure (E1) | No (재현) |
| Mixed-precision **policy** (greedy bit-width) | **입력 (known)** — HAWQ/BRECQ 계열의 알고리즘 | No (재현) |
| GPTQ 실측 + ranking 검증 (E1) | **viability gate** — Kendall τ = +1.0, isolated sensitivity가 GPTQ ranking 예측 OK | (재현이지만 본 paper의 토대) |
| Mapping spread 분석 (E2 pilot) | **negative-but-useful** — decode에서 mapping은 byte effect에 잠식 | **Yes** (motivation) |
| Precision-aware **residency scheduling** | **본 contribution** — knapsack-based on-chip pinning이 mixed-precision policy를 인식해 DRAM/tok 1.78~3.16× 감소 (§8c) | **Yes (main)** |
| System-level closure | accuracy(ppl) × DRAM bytes/token Pareto를 정책 4종 × GBuf 8 종에서 일관 분석 | **Yes** |

MAGNETO(power-aware mapping)와 GATHER(LLM accelerator) 라인을 잇는 위치.
LLM-decoder 특화·weight-dominant traffic·layer-wise heterogeneity의 세
specificity로 일반 DNN quantization-mapping 연구와 분리됩니다.

---

## 2. 실험 5종 / Experiments

| ID | 상태 | 역할 | 가설 / 측정 |
| --- | :---: | --- | --- |
| **E1** | done | Viability gate | greedy isolated-layer sensitivity가 만든 정책의 ppl ranking이 실제 GPTQ 양자화 후에도 보존되는가. (§8a) |
| **E2-mapping** | done | Motivation (negative) | 같은 layer shape에서 4b vs 8b의 best mapping이 의미 있게 다른가. → cycle/energy는 byte 효과로 흡수, mapping 기여 작음. (§8b) |
| **E2-residency** | done | **Main result** | 같은 mixed-precision policy 위에서 precision-aware vs oblivious 스케줄러의 per-token DRAM 격차. (§8c) |
| **E3** | pending | 통합 Pareto + 표 | residency × accuracy × 정책 4종을 codesign.py로 묶고 main figure 마무리 |
| **E4** | pending | HW-normalized policy | sensitivity score를 Δppl/Δbyte로 바꾸면 정책이 달라지고 residency Pareto에서 추가 이득이 나오는가 |
| **Sweep** | pending | 일반화 | Qwen-7B, Gemma-2B에서 같은 곡선이 재현되는가 |

**Critical path (확정)**: E1 → E2-mapping → E2-residency → E3 → E4 → Sweep.

### Pipeline 모듈 → Experiment 매핑

| 모듈 / 스크립트 | E1 | E2-mapping | E2-residency | E3 | E4 |
| --- | :---: | :---: | :---: | :---: | :---: |
| `pipeline/sensitivity.py`, `run_sensitivity.py` | input | | | | input |
| `pipeline/policy.py`, `run_policy.py` | input | | input | input | HW-norm 확장 |
| `pipeline/quant_runner.py`, `pipeline/eval.py` | **main** | | proxy | accuracy 점 | |
| `pipeline/hw_timeloop.py`, `pipeline/mapping.py` | | **main** | motivation | optional | |
| `pipeline/residency.py` | | | **main** | **main** | |
| `pipeline/codesign.py` | | | | **aggregator** | |
| `pipeline/profile_gpu.py` | proxy | | | proxy | |

---

## 3. 환경 셋업 / Setup

### 3-1. LAMP_acpl 활성화 확인

```bash
conda activate LAMP_acpl
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

기대 출력: `2.9.1+cu128 5.8.0` (acpl 서버 기준, `../README.md` §1 참고).

### 3-2. 실험 전용 dep 설치

`../requirements.txt`는 baseline용입니다. 실험에 필요한 추가 패키지는
`requirements-extra.txt`에 따로 모아 두었습니다.

```bash
conda activate LAMP_acpl
bash experiments/scripts/install_extras.sh
```

내용: `optimum`, `gptqmodel`, `lm-eval`, `scipy`, `matplotlib`,
`seaborn`, `typer`, `pynvml`.

### 3-3. Hardware simulator (Timeloop + Accelergy)

**acpl 서버에서 실제로 동작한 방법** — sudo / docker 권한이 없어도 conda env
안에서 source build로 설치 가능. 한 번에 끝나도록 스크립트화해 두었습니다.

```bash
conda activate LAMP_acpl
bash experiments/scripts/install_timeloop.sh
# Re-source so LD_LIBRARY_PATH activate hook이 잡힘:
conda deactivate && conda activate LAMP_acpl
which timeloop-mapper accelergy   # 둘 다 경로가 떠야 함
```

스크립트가 하는 일:

1. **build deps** (`scons`, `libconfig`, `yaml-cpp`, `ncurses`, `cmake`,
   conda-managed `gcc_linux-64` / `gxx_linux-64`)를 conda-forge에서 설치.
2. **Accelergy**(pure Python)를 git clone → `pip install .`.
3. **Timeloop v3.0.3**(NVlabs)를 git clone → `--with-isl`를 건드리지 않는
   tag로 checkout. (더 최신 main / Accelergy-Project fork는 ISL/barvinok이
   hardcoded 되어 있어 sudo 없이는 곤란.)
4. **gcc-13 호환 패치**: `std::uint64_t`를 쓰는 92개 헤더에 `<cstdint>`를
   자동 삽입. 패치 없이는 gcc-13에서 `'uint64_t' in namespace 'std' does not
   name a type` 컴파일 에러가 남.
5. `scons -j$(nproc)`로 빌드 → bin/lib을 conda env의 `bin/`·`lib/`에 symlink.
6. `etc/conda/activate.d/timeloop_ld.sh`에 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
   영구 export — `libconfig++.so.11`이 런타임에 찾아짐.

다른 서버에서 재현할 때 막힐 만한 지점:
- conda-forge gcc가 system glibc보다 너무 오래되면 ABI 충돌. 안 되면
  `conda install -c conda-forge 'gcc_linux-64>=13'`로 강제.
- `libtinfo`를 conda env 안에서 못 찾으면 `conda install -c conda-forge
  ncurses` 다시 실행.
- 이 스크립트는 `pat-public/src/pat → src/pat` symlink를 만듭니다. Timeloop
  설계상 필수 단계지만 새 clone 직후라면 빠뜨리기 쉬움.

> **상태 (2026-05-12).** `timeloop-{mapper,model,metrics,simple-mapper,
> design-space}` 5개 바이너리 + `accelergy` CLI 모두 PATH에 노출됨.
> `pipeline.hw_timeloop._materialize_arch()` 구현 완료 — E2 mapping pilot에서
> q_proj/o_proj/up_proj × {INT4, INT8} 6 매퍼 실행 통과 (§8b). 이후 main
> contribution이 residency scheduling으로 이동했기 때문에 Timeloop은 motivation
> 단계까지만 사용되고 main figure는 `pipeline/residency.py`로 생성.

### 3-4. 폴더 구조

```text
experiments/
├── README.md                  # 이 파일 — 셋업, 실행법, 측정 방법론
├── ROADMAP.md                 # E1→끝까지 실험 계획 1-pager
├── requirements-extra.txt     # 실험 전용 pip dep
├── pipeline/                  # python 패키지 (importable) — 알고리즘 구현
│   ├── README.md              # 모듈별 1줄 설명 + 데이터 흐름도
│   ├── __init__.py
│   ├── config.py              # YAML 로더
│   ├── sensitivity.py         # Phase 1
│   ├── policy.py              # Phase 2 (greedy bit-width)
│   ├── hw_timeloop.py         # Phase 3 (Timeloop wrapper — _materialize_arch stub)
│   ├── mapping.py             # E2/E3 (LayerShape, MappingCache, aggregate_model)
│   ├── codesign.py            # E3 (policy + mapping + eval → Pareto)
│   ├── quant_runner.py        # Phase 4a (GPTQ)
│   ├── eval.py                # Phase 4b (ppl + lm-eval; GPTQModel.load 우회)
│   ├── profile_gpu.py         # Phase 5 (threaded NVML)
│   └── viz.py                 # Phase 6
├── configs/
│   ├── targets.yaml           # 모델 manifest (baseline 가중치를 참조)
│   ├── policy_search.yaml     # 비트 배정 탐색 설정
│   └── arch/
│       └── eyeriss_like.yaml  # Timeloop 아키텍처 템플릿
├── scripts/
│   ├── install_extras.sh
│   ├── install_timeloop.sh    # Timeloop + Accelergy source build (sudo 불필요)
│   ├── run_sensitivity.py     # Phase 1 entry
│   ├── run_policy.py          # Phase 2
│   ├── run_hw.py              # Phase 3
│   ├── run_quant.py           # Phase 4a
│   ├── run_eval.py            # Phase 4b
│   ├── run_profile.py         # Phase 5
│   ├── _e1_quant_and_eval.sh  # E1 sweep (4 policy 양자화+평가)
│   └── run_all.sh             # 1→5 일괄 실행
└── results/                   # gitignored: 모든 출력물
```

**불변 규칙**: 이 폴더 안의 코드는 `../baseline/`의 파일을 읽기만 합니다.
가중치는 `../baseline/models/<key>/` 에서 그대로 가져오고 복제하지
않습니다.

**어떤 문서를 봐야 하나**:
- 처음 셋업·실행 → 이 파일 (`README.md`)
- 코드 모듈이 무엇을 하는지 → `pipeline/README.md`
- 실험 전체 흐름·다음 단계 → `ROADMAP.md`

---

## 4. End-to-end 실행 / Quick run

```bash
conda activate LAMP_acpl
cd /home/heeseo/LAMP/acpl

# 한 번에 (1.5B, 5 bpw 예산):
TARGET=qwen25_15b_instruct AVG_BITS=5.0 bash experiments/scripts/run_all.sh
```

단계별로 따로 돌리려면:

```bash
python experiments/scripts/run_sensitivity.py qwen25_15b_instruct
python experiments/scripts/run_policy.py      qwen25_15b_instruct --avg-bits 5.0
python experiments/scripts/run_quant.py       qwen25_15b_instruct \
    --policy experiments/results/policies/qwen25_15b_instruct_5.0bit.yaml
python experiments/scripts/run_eval.py        experiments/results/quantized/qwen25_15b_instruct_from_qwen25_15b_instruct_5.0bit --skip-downstream
python experiments/scripts/run_profile.py     experiments/results/quantized/qwen25_15b_instruct_from_qwen25_15b_instruct_5.0bit
python experiments/scripts/run_hw.py          qwen25_15b_instruct \
    --policy experiments/results/policies/qwen25_15b_instruct_5.0bit.yaml
```

---

## 5. 모듈 API / Module API

각 단계는 두 진입점을 갖습니다.

- **CLI**: `scripts/run_*.py` — 인자 파싱, 경로 정리, 출력 파일 저장만 담당.
- **library**: `pipeline/<phase>.py` — 실제 알고리즘. 노트북·다른 스크립트에서
  바로 import해 호출 가능.

예) Phase 1을 노트북에서 직접:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline import sensitivity, config

cfg = config.load_targets()
spec = config.get_target(cfg, "qwen25_15b_instruct")
mp = spec.weights_path(cfg["defaults"]["models_root"])

tok = AutoTokenizer.from_pretrained(str(mp))
model = AutoModelForCausalLM.from_pretrained(str(mp), torch_dtype="float16", device_map="auto")

res = sensitivity.run_sensitivity(model, tok, target=spec.key, bits_to_try=[4, 8])
sensitivity.save(res, "experiments/results/sensitivity")
```

---

## 6. 측정 방법 / How each metric is measured

> 모든 절대값은 **상대 비교**에만 의미가 있습니다. 동일한 코드 경로 안에서
> 베이스라인(FP16, uniform-INT4, uniform-INT8, mixed)을 같이 측정해서
> 차이를 보고합니다.

### 6-1. Accuracy

| 메트릭 | 구현 위치 | 데이터셋 | 절차 |
| --- | --- | --- | --- |
| Perplexity (wikitext-2) | `pipeline.sensitivity.compute_perplexity`, `pipeline.eval.eval_perplexity` | WikiText-2 raw `test` split | 전체 split을 `"\n\n".join` 으로 이어붙이고 2048-token sliding window NLL → `exp(mean)`. |
| Perplexity (C4) | `pipeline.eval.eval_perplexity("c4")` | C4 `validation`, streaming 256 docs | 동일 sliding window. Streaming이라 캐시 부담 없음. |
| HellaSwag / ARC-{Easy,Challenge} / Winogrande | `pipeline.eval.eval_downstream` | lm-evaluation-harness 기본 split | `lm_eval.simple_evaluate` 호출, fp16 inference. 정확도는 `acc`(또는 `acc_norm`)로 보고. |
| MMLU | `eval_downstream(..., tasks=["mmlu"])` | lm-eval `mmlu` (5-shot 기본) | 카테고리별 acc 평균. 시간이 오래 걸리니 sweep에서는 옵션. |

**주의.** GPTQ로 양자화한 체크포인트는 `transformers` 측에서 자동으로
디퀀트 커널을 골라 inference합니다. 양자화 직후 `evaluate`가 GPU 메모리를
재활용하므로, `run_eval.py`는 fresh process로 돌리는 것을 권장합니다.

### 6-2. Efficiency (GPU 실측 — proxy)

`pipeline.profile_gpu.profile_generation` (E1·E3 proxy 측정용) —
- **Latency / throughput**: `model.generate(..., max_new_tokens=N, do_sample=False)`을
  `torch.cuda.synchronize()`로 감싸 wall time 측정. 2회 warmup 후 1회 본 측정.
  보고값은 `new_tokens / elapsed_s` (tokens/s).
- **Energy per token**: NVML `nvmlDeviceGetPowerUsage`를 별도 thread에서 20ms
  간격으로 sample (`_PowerSampler` 클래스). idle 전력 (직전 10회 중앙값) 빼서
  net power 추정, `energy_J = avg_W × elapsed_s`. **caveat**: idle 측정 직전
  단계의 GPU가 이미 elevated 상태일 수 있어서 mJ/token 절대값은 신뢰가 낮음.
  상대 비교(정책끼리)에 사용 권장. 정밀 측정은 cold-start idle 추가 시 개선됨.

**Reproducibility 체크리스트**:
1. 동일 GPU index (`CUDA_VISIBLE_DEVICES=0`) 고정.
2. 동일 prompt, 동일 `max_new_tokens`, `do_sample=False`.
3. 측정 사이에 `nvidia-smi --gpu-reset` 또는 1초 sleep으로 thermal drift 완화.

### 6-3. Hardware metrics — Timeloop (E2 mapping pilot)

`pipeline.hw_timeloop.run_layer` — 레이어 × 비트폭 조합마다 timeloop-mapper를
한 번 돌리고 `stats.txt` 의 headline summary를 파싱:

| 메트릭 | 정의 | 비고 |
| --- | --- | --- |
| **Cycles** | mapper가 보고하는 총 사이클 (memory stall 포함) | latency proxy |
| **Energy (pJ)** | Timeloop PAT 모델이 component-wise 적분한 총 에너지 (Accelergy 미사용 v1) | bit-width-aware: storage `word-bits`와 arithmetic energy가 bits_w에 종속 |
| **PE Utilization** | mapper가 reportar하는 0~1 점유율 | decode N=1에서는 memory bound라 ~6% 수준, prefill N=128에서는 ~100% |
| **DRAM bytes** | Total scalar accesses × word-bits / 8 (DRAM section) | weight stream bytes; 본 paper 의 핵심 메트릭 |
| **GBuf hit rate** | (TBD) — 현재 0.0 placeholder | follow-up 작업 |

E2 pilot 결과(§8b)에서 cycle은 byte effect에 잠식되는 것을 확인했고, 본
paper는 **DRAM bytes/token**을 main 효율 메트릭으로 채택. main figure는
Timeloop이 아닌 analytic residency model로 생성 (§6-4 참고).

### 6-4. Memory-residency metrics (E2-residency main)

`pipeline.residency.schedule_dram_per_token` —

| 메트릭 | 정의 | 비고 |
| --- | --- | --- |
| **DRAM bytes / token** | `Σ(unpinned tile bytes)` | autoregressive decode에서 토큰당 weight read 비용. 본 paper main efficiency 메트릭. |
| **Pinned tile count** | GBuf에 잡힌 (layer, projection) tile 수 (총 196개 중) | 스케줄링 capacity 가시화 |
| **Pinned bytes (aware)** | 실제 점유한 bytes (정밀 budget) | aware vs oblivious 차이의 직접 원인 |

스케줄러는 0/1 knapsack greedy (smallest-first), 매 token마다 weight를 한
번씩만 touch 하므로 access frequency 동일 → smallest-first가 byte 회수
관점에서 최적. 두 시나리오 비교:
- `precision_aware`: 실 bits_aware로 budgeting
- `precision_oblivious_int8`: 모든 tile을 INT8로 가정해 budgeting

§8c에 결과 표·그림·해석.

### 6-5. 비교 베이스라인

모든 그래프는 최소 다음 4개 컬럼을 함께 표시합니다.

| 라벨 | 의미 |
| --- | --- |
| FP16 | 양자화 없음. accuracy 상한선. |
| INT8-uniform | 모든 레이어 8-bit. mid budget 비교용. |
| INT4-uniform | 모든 레이어 4-bit. accuracy 하한선. |
| Mixed-N bpw | LAMP 정책. 평균 비트 폭 N (e.g. 5.0). |

---

## 7. 결과 파일 명세 / Result file conventions

```text
results/
├── sensitivity/{target}.json
│   {"target": str, "fp16_ppl": float,
│    "layers": [{"idx": int, "ppl_4bit": float, "ppl_8bit": float,
│                "delta_4bit": float, "delta_8bit": float}, ...]}
├── policies/{target}_{avg_bits}bit.yaml
│   target, allowed_widths, target_avg_bits, achieved_avg_bits,
│   per_layer_bits (list[int]), notes
├── quantized/{target}_from_{policy_stem}/
│   gptqmodel가 저장한 HF 호환 디렉토리 (config.json, model.safetensors, ...)
├── eval/{tag}/ppl_wikitext2.json
│   {"dataset": "wikitext-2", "ppl": float}
├── eval/{tag}/lm_eval.json
│   lm-evaluation-harness raw output (results, configs)
├── profile/{tag}.json
│   ProfileResult fields (tokens_per_s, energy_per_token_mJ, ...)
├── hw/{target}_summary.json
│   [{"layer": str, "bits_w": int, "cycles": int, "energy_pJ": float,
│     "pe_utilization": float, "dram_bytes": int, "gbuf_hit_rate": float}, ...]
└── figs/*.png  (생성 후)
```

`results/` 전체는 git ignore 대상으로 두는 것을 권장합니다(이미 baseline의
`.gitignore` 정책과 동일한 방향).

---

## 8. 한 번에 비교 실험 만들기 / Sweep recipe

같은 모델에 대해 평균 비트 4.5 / 5.0 / 5.5 / 6.0을 비교하는 sweep 예시:

```bash
for B in 4.5 5.0 5.5 6.0; do
  python experiments/scripts/run_policy.py qwen25_15b_instruct --avg-bits "$B"
  POL="experiments/results/policies/qwen25_15b_instruct_${B}bit.yaml"
  python experiments/scripts/run_quant.py   qwen25_15b_instruct --policy "$POL"
  TAG="qwen25_15b_instruct_from_qwen25_15b_instruct_${B}bit"
  python experiments/scripts/run_eval.py    "experiments/results/quantized/${TAG}" --skip-downstream
  python experiments/scripts/run_profile.py "experiments/results/quantized/${TAG}"
done
```

각 단계는 cached input(sensitivity, baseline weights)을 재활용하므로 sweep
한 번에 GPU 1장 기준 1.5B는 ~1–2시간, 7B는 ~반나절 안에 끝납니다.

---

## 8a. E1 결과 / Ranking Validation (Qwen2.5-1.5B-Instruct, 2026-05-12)

**E1은 paper viability gate** — sensitivity-기반 정책의 ranking이 실제 GPTQ
양자화 후에도 보존되는지 확인. GPTQ 32-sample calibration, group_size=128,
desc_act=True. 전체 sweep wall-time = 3023s (RTX 2000 Ada × 1).

| Policy | avg bpw | Real GPTQ ppl | Δ vs FP16 | Sensitivity upper | Δ upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 baseline | 16.00 | **8.890** | — | — | — |
| INT4-uniform | 4.00 | 10.452 | +1.56 | 12.887 | +4.00 |
| Mixed-4.5 | 4.57 | 10.279 | +1.39 | 11.196 | +2.31 |
| Mixed-5.0 | 5.00 | 10.091 | +1.20 | 10.789 | +1.90 |
| INT8-uniform | 8.00 | 9.604 | +0.71 | 8.915 | +0.03 |

**Statistical 판정 (`results/e1_ranking.csv`, `figs/e1_ranking_validation.png`):**

- `Kendall τ(real_ppl vs sensitivity_upper) = +1.000` — 완벽 monotonic 일치
- `Kendall τ(avg_bpw vs real_ppl) = −1.000` — bit 늘면 ppl 단조 감소
- 실측 순서: INT4 > Mixed-4.5 > Mixed-5.0 > INT8 (sensitivity 예측 그대로)

**→ E1 통과.** Greedy isolated-layer sensitivity가 만든 정책 순위가 GPTQ 후에도
유지됨. 즉 paper에서 mixed-precision policy를 *known input*으로 받아들이고
mapping/scheduling을 contribution surface로 가는 framing의 전제 OK.

### 보조 발견

1. **Sensitivity 상한은 정량적으로 매우 보수적** — INT4에서 GPTQ가 예측 손실의
   **61%를 흡수**, Mixed-5.0에서 **37%**. Ranking은 보존되지만 magnitude는
   isolated-layer model이 못 잡음. Paper 본문에서 "ordering oracle이지
   magnitude oracle은 아니다"로 정리.

2. **"GPTQ floor" ≈ +0.71 ppl** — Sensitivity는 INT8을 lossless로 보지만,
   group_size=128 groupwise + 32-sample calibration이 일정 floor 형성. 모든
   정책 비교에서 공통 비용 깔림. *양자화-특이* 손실은 `(real_Δ − 0.71)`:

   | Policy | real Δ | quant-specific Δ | INT4 대비 회복률 |
   | --- | ---: | ---: | ---: |
   | INT4-uniform | +1.56 | +0.85 | 0.0% |
   | Mixed-4.5 | +1.39 | +0.68 | 20.0% |
   | Mixed-5.0 | +1.20 | +0.49 | 42.4% |
   | INT8-uniform | +0.71 | 0.00 | 100.0% |

3. **Memory–accuracy trade-off** — Mixed-5.0은 INT4 대비 메모리 1.25×에
   INT4→INT8 ppl 갭의 42.6%를 회복. 1.5B 스케일에서는 부드러운 Pareto이지
   knee point는 아님. 7B/Gemma-2B에서 곡선 형태가 달라지는지가 sweep 포인트.

---

## 8b. E2 Pilot 결과 / Mapping Spread (2026-05-12)

**E2는 paper의 두 번째 viability gate** — "같은 layer shape에서 INT4와 INT8이
*의미 있게 다른 최적 mapping*을 선호하는가" 검증. Pilot 3 shape × 2 bits = 6
Timeloop mapper run, 16×16 PE Eyeriss-like arch + 256 KB GBuf + 32 GB/s DRAM.

### Mapper가 고른 mapping (`timeloop-mapper.map.txt`)

| shape | INT4 best (DRAM loop) | INT8 best (DRAM loop) |
| --- | --- | --- |
| q_proj (M=1536, K=1536) | `for C in [0:6)` (K 통째) | `for K in [0:2); for C in [0:6)` (K split) |
| o_proj (M=1536, K=1536) | 동일 (K 통째) | 동일 (K split) |
| up_proj (M=8960, K=1536) | K split | K split (더 큰 shape이라 양쪽 다 분할) |

**Mapping shape 자체는 다릅니다** — INT4는 q_proj·o_proj에서 K 전체를 GBuf
안에 잡아두는 mapping을 찾고, INT8은 buffer 용량 초과로 K를 2번 분할.
이건 precision-aware mapping spread의 직접 증거.

### Cycles / Energy / DRAM (N=1, autoregressive decode)

| shape | bits | cycles | energy (pJ) | PE util | DRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| q_proj | 4 | 147,552 | 1.347e8 | 0.06 | 1.18 |
| q_proj | 8 | 147,648 | 2.698e8 | 0.06 | 2.36 |
| o_proj | 4 | 147,552 | 1.347e8 | 0.06 | 1.18 |
| o_proj | 8 | 147,648 | 2.698e8 | 0.06 | 2.36 |
| up_proj | 4 | 860,256 | 7.855e8 | 0.06 | 6.89 |
| up_proj | 8 | 860,256 | 1.572e9 | 0.06 | 13.77 |

### 핵심 발견 *

- **cycles INT4 ≈ INT8** (0% 차이) — mapping shape는 달랐는데 cycle 결과는 같음.
- **energy / DRAM은 정확히 2× 차이** — byte-width 효과로 완전히 설명.
- PE utilization = 0.06 (decode N=1은 memory bound, 100% memory-stall).
- 64 KB GBuf로 압박해도, N=128 (prefill 모드, util=1.0)에서도 cycles는 두 bit-width가 동일.

**해석.** Autoregressive decode regime에서는 weight를 토큰당 한 번씩만
touch하므로 mapping geometry가 cycle 수에 거의 영향을 못 줍니다. cycle 수는
`max(arith_cycles, dram_bw_cycles)`로 결정되는데, 두 mapping 모두 동일한
*total weight bytes streamed* 이므로 같은 DRAM 시간을 씁니다. 결국 cycle/energy
스프레드는 byte-width로 environment-collapsed.

### Paper에서의 위치

이 negative result는 § §8c (residency)로 가는 **motivation 사다리**로 사용:

> "16×16 Eyeriss decode regime에서 mapping geometry는 byte effect에 잠식된다
> → 같은 byte 예산 안에서 어디에 byte를 들이느냐(residency)가 더 큰 ROI이다."

대안으로 검토됐지만 채택하지 않은 방향 (history):
- *prefill + decode joint scheduling*: 두 regime에 다른 mapping을 적용하면
  spread 회복 가능. paper scope 확장이 필요해서 부수 작업으로 분리.
- *arch 자체 교체*: LLM-specific systolic + layer-pipeline 등. architecture
  novelty까지 가는 대형 작업이라 보류.
- *negative-result paper*: "byte traffic is the dominant lever" 메시지로
  재구성. residency가 positive로 나와서 불필요해짐.

### Pilot artifacts

- `results/hw/runs/<shape>_w{4,8}/timeloop-mapper.{map,stats}.txt`
- `results/hw/mapping_cache.json` (3 shape × 2 bits = 6 entries)

---

## 8c. E2-Residency — Memory-Residency Scheduling (2026-05-12)

E2 pilot의 "decode regime에서 byte traffic이 dominant lever" 발견을 받아
contribution 축을 **mapping search**에서 **memory-residency scheduling**으로
이동. 새 claim:

> **"Given a fixed mixed-precision policy and a finite on-chip buffer,
> precision-aware residency scheduling pins more layers' weights on-chip
> and reduces per-token DRAM traffic versus a precision-oblivious
> scheduler that budgets every tile at the worst-case bit-width."**

### Setup

- Decoder weight tiles per layer × projection: 28 layers × 7 projections =
  **196 tiles**. Total bytes 정책별로 다름 (INT4 655 MB ↔ INT8 1310 MB).
- Greedy 0/1 knapsack (smallest-first); 매 token마다 weight를 한 번씩만
  touch 하므로 access frequency가 동일 → smallest-first가 최적.
- 두 스케줄러:
  - `precision_aware`: actual `bytes = bits_aware × n_weights / 8`로 packing
  - `precision_oblivious_int8`: 모든 tile을 INT8 budget(`= n_weights`)으로
    가정. mixed-precision 모델 위에서 INT8-기준 deployment scheduler를
    돌리는 시나리오.

### 결과 — Per-token DRAM traffic (`results/residency_sweep.csv`, `figs/e2_residency_pareto.png`)

**Pareto operating point: GBuf = 512 MB** (대표 figure)

| Policy | ppl (E1) | aware DRAM/tok | oblivious DRAM/tok | **aware win** |
| --- | ---: | ---: | ---: | ---: |
| INT4-uniform | 10.452 | **124 MB** | 392 MB | **3.16×** |
| Mixed-4.5 | 10.279 | **213 MB** | 454 MB | **2.13×** |
| Mixed-5.0 | 10.091 | **289 MB** | 516 MB | **1.78×** |
| INT8-uniform | 9.604 | 784 MB | 784 MB | 1.00× (no headroom) |

INT8-uniform은 정책 자체가 worst-case bit-width이므로 oblivious budget이
정확 → 스케줄링으로 이득 없음. **Mixed-precision일수록 aware vs oblivious
gap이 큼.**

**Bimodal regime — GBuf = 1024 MB**

| Policy | aware DRAM/tok | oblivious DRAM/tok |
| --- | ---: | ---: |
| INT4-uniform | **0 MB (100% on-chip)** | 124 MB |
| Mixed-4.5 | **0 MB (100% on-chip)** | 186 MB |
| Mixed-5.0 | **0 MB (100% on-chip)** | 248 MB |
| INT8-uniform | 248 MB | 248 MB |

1 GB GBuf에서는 **Mixed-5.0까지의 모든 정책이 precision-aware scheduling 하에
완전히 on-chip resident**. INT8-uniform 모델은 1.31 GB이라 절대 안 들어감.
이게 mixed-precision + aware 스케줄링의 가장 강한 임팩트 영역.

### Sweep figure (`figs/e2_residency_pareto.png`)

좌측 패널: DRAM bytes/token vs GBuf size (4 MB - 1 GB log scale), 4 정책 × 2
스케줄러. 실선=aware, 점선=oblivious.
우측 패널: GBuf=512 MB 에서의 accuracy × DRAM/token Pareto. 각 정책에 aware
(원)·oblivious (X) 표시, 화살표가 aware reduction을 보여줌.

### 핵심 발견

1. **Precision-oblivious scheduling은 mixed-precision policy의 byte 이득을
   1.7~3.2× 만큼 흘려보냄** (GBuf=512 MB 기준). 즉 mixed-precision 모델을
   배포할 때 deployment scheduler가 policy를 모르면 GBuf에 보수적으로
   budget해서 fewer tile을 pin함.

2. **GBuf가 model size에 가까워질수록 win이 커짐**. INT4-uniform/Mixed-4.5는
   1 GB GBuf에서 100% on-chip → DRAM/tok = 0. INT8-uniform은 그 자체로 너무
   커서 어떤 스케줄링도 도와줄 수 없음.

3. **Mapping spread (E2 pilot)의 negative result는 motivation으로 그대로 사용
   가능** — "mapping geometry는 byte effect에 잠식되니까, 같은 byte 예산 안에서
   residency 결정을 잘 하는 게 더 큰 ROI"라는 논리적 사다리.

### 한계 & follow-up

- 현재는 0/1 knapsack (전체 tile pin or stream). prefetch-based scheduling은
  미반영. layer pipeline + prefetch를 모델링하면 oblivious도 partial pin이
  가능해 gap이 줄 수 있음 — 후속 실험에서 검토 필요.
- T (생성 길이) 미반영. T가 작으면 (e.g. T=1) pinning 자체가 의미 없음.
  paper에서 T ≥ 32 같은 가정 명시 필요.
- Energy estimation은 아직 byte cost × DRAM pJ/byte (PAT)로만 추정. Accelergy
  full power-area-timing 분석은 후속.

### Artifacts

- `results/residency_sweep.csv` — 4 정책 × 8 GBuf × 2 스케줄러 = 64 rows
- `figs/e2_residency_pareto.png` — 2-panel 메인 figure
- `pipeline/residency.py` — 모듈 (knapsack, sweep, compare 함수)

---

## 8d. E3 — 통합 Pareto Figure (2026-05-12)

E1 ppl(실측 GPTQ)과 E2-residency DRAM/token을 하나의 figure로 묶어 paper main
result로 폴리시. `pipeline.codesign.build_residency_codesign()`이 두 결과를
join하고, `plot_residency_pareto_*` 두 함수가 figure 생성.

### Main figure (`figs/codesign_main.png` — GBuf = 512 MB operating point)

x축: WikiText-2 ppl (E1 실측). y축: DRAM bytes/token at GBuf = 512 MB.
○ = precision-aware (본 contribution), × = precision-oblivious_int8 baseline,
arrow = aware → oblivious gap.

| Policy | ppl | aware (MB) | oblivious (MB) | **aware ↓** |
| --- | ---: | ---: | ---: | ---: |
| INT4-uniform | 10.452 | **124** | 392 | **3.17×** |
| Mixed-4.5 | 10.279 | **213** | 454 | **2.13×** |
| Mixed-5.0 | 10.091 | **289** | 516 | **1.79×** |
| INT8-uniform | 9.604 | 784 | 784 | 1.00× |

INT8-uniform은 aware == oblivious — uniform policy면 정밀도가 worst-case
budget과 같아서 scheduling으로 이득 없음. mixed-precision일수록 aware 우위 ↑.

### Supplementary figure (`figs/codesign_multi_gbuf.png`)

3-panel: GBuf ∈ {128, 512, 1024} MB. 각 패널마다 4 정책 × 2 스케줄러 8점.

- **GBuf = 128 MB** (작은 buffer): aware-oblivious gap 10-15% — 어차피 pin 가능량이 작아서 스케줄링 ROI ↓.
- **GBuf = 512 MB** (sweet spot): aware 우위가 가장 큼. paper main operating point.
- **GBuf = 1024 MB** (bimodal): mixed-precision aware 모두 DRAM/tok = 0 (100% on-chip). oblivious는 여전히 stream 필요. INT8-uniform은 모델이 1.31 GB라 어떤 스케줄링으로도 fit 불가.

### Paper main message (이 두 figure로 전달 가능)

1. **정확도 끝점**: INT8-uniform (ppl=9.60) ↔ INT4-uniform (10.45). Mixed-5.0
   (10.09)·Mixed-4.5 (10.28)는 그 사이의 interpolation.
2. **효율 끝점**: 모든 mixed-precision 정책의 *aware* DRAM/tok이 INT8-uniform
   대비 큰 폭으로 낮음 — 단순히 byte 수가 적기 때문이 아니라 스케줄링이 그
   bytes를 GBuf에 효과적으로 매핑하기 때문.
3. **본 contribution**: aware × oblivious 비교가 "scheduling이 byte 이득을
   얼마나 실현하는가"를 직접 정량화. mixed-precision 정책의 byte 절감이
   precision-oblivious deployment scheduler 하에서는 1.7~3.2× 만큼 손실됨.

### Artifacts

- `results/codesign_qwen15b.csv` — 64 codesign 점 (4 정책 × 8 GBuf × 2 스케줄러), policy_tag·scheduler·gbuf_MB·avg_bpw·ppl·DRAM_MB_per_token·pinned_count·pinned_MB
- `figs/codesign_main.png` — single-GBuf main figure
- `figs/codesign_multi_gbuf.png` — 3-panel supplementary
- `pipeline/codesign.py` — `build_residency_codesign`, `plot_residency_pareto_{single,multi}_gbuf`

---

## 8e. E4 — HW-normalized per-tile policy (2026-05-12)

E1/E2-residency/E3에서 입력으로 받은 mixed-precision 정책은 **per-layer**
greedy로 만들어졌고, "Δppl / Δbits"를 점수로 사용. E4는 두 가지 확장:

1. **Per-projection sensitivity**: 28 layer × 7 projection = 196 ablations로
   각 (layer, role) 타일의 Δppl을 측정 (`results/sensitivity/qwen25_15b_instruct_per_projection.json`,
   wall-time 3.8 min on free GPU 1, 50-text calib).
2. **HW-normalized scorer**: greedy 점수를 "Δppl / Δbytes" 로 바꿈 — 작은
   projection (k/v_proj)이 promote 비용이 싸기 때문에 우선 승격됨.

### Policy shape (B = 4.5 비교)

| Projection role | Mixed-4.5 (per-layer greedy) | **hwnorm-4.5 (per-tile)** |
| --- | ---: | ---: |
| q_proj (28 tiles, 1.18 MB / INT4) | 4 → 8-bit | **7 → 8-bit** |
| k_proj (28 tiles, 0.20 MB) | 4 → 8-bit | **17 → 8-bit** |
| v_proj (28 tiles, 0.20 MB) | 4 → 8-bit | **25 → 8-bit (89%)** |
| o_proj (28 tiles, 1.18 MB) | 4 → 8-bit | 15 → 8-bit |
| gate_proj (28 tiles, 6.89 MB) | 4 → 8-bit | **0 (모두 4-bit)** |
| up_proj (28 tiles, 6.89 MB) | 4 → 8-bit | 2 → 8-bit |
| down_proj (28 tiles, 6.89 MB) | 4 → 8-bit | 5 → 8-bit |
| **Total 8-bit tiles** | **28** | **71** |
| achieved avg bpw | 4.571 | **4.503** (smaller!) |

hwnorm은 **k_proj/v_proj 위주로 8-bit 승격**(작아서 byte 비용 ↓, 그러나 sensitivity 충분)을
선택하고 MLP의 큰 projection(gate/up/down)은 모두 4-bit로 둠. 결과적으로 같은
정책 "예산" 안에서 *더 적은 평균 bpw*에 *더 많은 sensitivity-critical 타일*을 보호.

### Residency 결과 — `figs/e4_hwnorm_vs_greedy.png`

bpw vs DRAM/token, 4-panel (GBuf 128/256/512/1024 MB), 3 series (uniform / greedy per-layer / hwnorm per-tile).
| Budget | greedy bpw | hwnorm bpw | greedy DRAM/tok | hwnorm DRAM/tok | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| B=4.5 | 4.571 | 4.503 | 213.3 MB | 206.4 MB | **+3.2%** ↓ |
| B=5.0 | 5.000 | 5.025 | 289.0 MB | 289.0 MB | 0.0% |
| B=5.5 | (없음) | 5.509 | — | 371.6 MB | — |
| B=6.0 | (없음) | 6.022 | — | 454.2 MB | — |

**핵심**: iso-budget에서 DRAM/token 곡선은 거의 겹침 (Pareto 측 gain 작음).
이건 residency 스케줄러가 *총 byte량* 위주로 결정하기 때문 — policy 안에서
어느 projection을 promote했는지는 GBuf packing 단계에서 거의 안 보임.

### 정직한 해석

E4의 **실제 win은 residency가 아니라 정책 자체의 질적 변화**:
- hwnorm은 attention의 v_proj/k_proj (KV projection, attention quality에 직결)를
  거의 모두 8-bit로 보호. MLP의 큰 projection은 모두 4-bit.
- 같은 평균 bpw에서 더 sensitivity-critical 타일을 보호 → **accuracy 측 가설은
  hwnorm < Mixed-4.5 ppl**.
- 단, accuracy 비교에는 GPTQ를 hwnorm 정책으로 다시 돌려야 함 — 본 turn에서는
  비용 사유로 미진행 (정책당 ~10 min × 4 = ~40 min). 이건 sweep 단계에서 7B와
  함께 일괄 진행 권장.

### Artifacts

- `results/sensitivity/qwen25_15b_instruct_per_projection.json` — 196 tiles ×
  {4, 8} bits ablation Δppl
- `results/policies/qwen25_15b_instruct_{4.5,5.0,5.5,6.0}bit_hwnorm.yaml`
- `results/e4_residency_compare.csv` — 64 rows (4 hwnorm + 4 baseline) × 4 GBuf × 2 sched
- `figs/e4_hwnorm_vs_greedy.png` — 4-panel bpw vs DRAM
- `pipeline/policy.py` — `make_per_tile_policy`, `save_per_tile`
- `pipeline/sensitivity.py` — `quantize_projection`, `run_per_projection_sensitivity`
- `pipeline/residency.py` — `enumerate_tiles` dispatches between `per_layer_bits` and `per_tile_bits`

---

## 8f. Sweep — Cross-model generalization (2026-05-12)

Qwen2.5-1.5B에서 확립한 pipeline을 **Gemma-2-2B-IT**와 **Qwen2.5-7B-Instruct**에
재현. 각 모델에서 sensitivity → policy(4 budget) → GPTQ × 4 → eval × 4 →
residency sweep을 전체로 다시 수행. 7B는 multi-GPU sharded (CUDA_VISIBLE_DEVICES=2,3).

### 모델별 절대값

| Model | params | FP16 ppl (wikitext-2) | Decoder bytes @ INT4 | @ INT8 | GPTQ wall-time |
| --- | --- | --- | ---: | ---: | --- |
| Qwen-1.5B-Instruct | 1.54 B | 8.890 (200 calib) | 655 MB | 1310 MB | 12 min/policy |
| Gemma-2-2B-IT | 2.61 B | (eval 측 미산정) | 968 MB | 1937 MB | 8 min/policy |
| Qwen-7B-Instruct | 7.61 B | 6.121 (50 calib, multi-GPU) | 3250 MB | 6500 MB | ~25 min/policy |

### E1 ranking — 3 모델 모두 보존

| Policy | Qwen-1.5B ppl | Gemma-2-2B ppl | Qwen-7B ppl |
| --- | ---: | ---: | ---: |
| INT4-uniform | 10.452 | 14.800 | 7.785 |
| Mixed-4.5 | **10.279** | **14.723** | **7.724** |
| Mixed-5.0 | **10.091** | **14.387** | **7.661** |
| INT8-uniform | 9.604 | 13.773 | 7.376 |

세 모델 모두 `Kendall τ(bpw vs ppl) = −1.0`. greedy isolated-layer
sensitivity가 만든 정책 순위가 모델 family·scale에 무관하게 GPTQ에서
유지됨.

### E2-residency win — 3 모델 모두 재현 (`figs/sweep_cross_model.png`)

GBuf ≈ ½ × INT4 decoder bytes operating point (각 모델의 "sweet spot"):

| Model | INT4 decoder | "sweet" GBuf | INT4 win | Mixed-4.5 win | Mixed-5.0 win | INT8 win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-1.5B | 655 MB | 256 MB | **1.33×** | 1.21× | 1.18× | 1.00× |
| Gemma-2-2B | 968 MB | 512 MB | **1.55×** | 1.37× | 1.27× | 1.00× |
| Qwen-7B | 3250 MB | 2048 MB | **1.97×** | 1.45× | 1.30× | 1.00× |

GBuf ≈ INT4 decoder size operating point (가장 극적 영역):

| Model | GBuf | INT4 win | Mixed-4.5 | Mixed-5.0 |
| --- | ---: | ---: | ---: | ---: |
| Qwen-1.5B | 512 MB | **3.17×** | 2.13× | 1.79× |
| Gemma-2-2B | 1024 MB | **∞ (100% on-chip)** | 4.67× | 2.58× |
| Qwen-7B | 4096 MB | **∞** | ∞ | ∞ |

### 핵심 발견 (paper에서 강조할 부분)

1. **Pattern universal — 모델 family와 scale에 무관**. Gemma(이종 family)와
   Qwen 7B(4× scale)에서 모두 같은 곡선 형태. paper "generalization"
   evidence 확보.

2. **Win factor가 모델 크기와 함께 증가** — iso-relative-GBuf 비교 시:
   1.5B 1.33× → 2B 1.55× → 7B 1.97×. 큰 모델일수록 per-tile byte 정밀
   회계의 이득이 더 큼.

3. **GBuf ≈ INT4 total 영역에서 모든 mixed-precision policy가 100% on-chip
   resident** — INT8-uniform은 절대 fit 불가. paper main figure는 이 영역.

4. **INT8-uniform은 aware == oblivious** (모든 모델에서 win=1.0×). 정책
   자체가 worst-case bit-width라 스케줄링 ROI 없음. mixed-precision이
   존재해야 residency contribution이 발현.

### Sweep methodology 주의사항

- Qwen-7B sensitivity는 16 GiB single GPU에서 OOM 발생 (7B FP16 = 14 GB +
  CE-loss output ~1 GB). multi-GPU sharded (CUDA_VISIBLE_DEVICES=2,3,
  `device_map="auto"`)로 우회. 그래도 fp16_ppl=6.12 (50 calib)는 sensitivity
  ranking에는 충분하나 GPTQ eval (200 calib)과 직접 비교는 부적합.
- Gemma의 codesign 첫 실행에서 `pinned_bytes_aware` 컬럼 누락 KeyError —
  `pipeline.codesign.build_residency_codesign`에 fallback (`r.get(...) or 0`)
  추가, `_sweep_target.sh`·`_sweep_qwen7b.sh`의 CSV writer 보강.
- GPTQ wall-time 모두 32-sample calibration 기준. paper-quality 결과는
  128 calib 필요 (~4× wall-time).

### Sweep artifacts

- **Sensitivity**: `results/sensitivity/{target}.json` (qwen25_15b/2_2b/7b)
- **Policies**: `results/policies/{target}_{B}bit.yaml` (4-uniform, 4.5/5.0/5.5/6.0 greedy, 8-uniform)
- **Quantized models**: `results/quantized/{target}_from_{policy}/`
- **Eval**: `results/eval/{target}_{tag}/ppl_wikitext2.json`
- **Residency sweep**: `results/residency_sweep_{target}.csv`
- **Codesign**: `results/codesign_{target}.csv`, `figs/codesign_{target}_*.png`
- **Cross-model**: `figs/sweep_cross_model.png` (3-panel main)
- **Sweep logs**: `results/_sweep_{target}.log`
- **Driver scripts**: `experiments/scripts/_sweep_target.sh` (generic),
  `experiments/scripts/_sweep_qwen7b.sh` (multi-GPU for 7B)

---

## 9. 트러블슈팅 / Troubleshooting

| 증상 | 해결 |
| --- | --- |
| `ModuleNotFoundError: pipeline` | `experiments/`에서 실행하지 말고 `acpl/`에서 실행하거나, 스크립트는 자체적으로 `sys.path` 보정함. |
| `gptqmodel` import 오류 | `bash experiments/scripts/install_extras.sh` 다시 실행. |
| `pynvml` import 오류 | 같은 install_extras 스크립트가 설치. driver 590에서 정상 동작 확인. |
| `timeloop-mapper: command not found` | §3-3 설치 확인. PATH가 비-interactive shell에 빠져 있을 수 있음. |
| HW 단계가 `NotImplementedError` | 의도된 stub. `pipeline/hw_timeloop._materialize_arch`를 채워야 함 (Phase 3 작업). |
| OOM (7B에서 sensitivity) | `sensitivity.compute_perplexity` 호출 시 `seq_len=1024`로 낮추거나 `CUDA_VISIBLE_DEVICES`로 GPU 분산. |

---

## 10. 다음 단계 / Next steps

### 완료된 viability gates

- [x] `profile_gpu.profile_generation` thread-sampled NVML
- [x] Timeloop + Accelergy source build (`scripts/install_timeloop.sh`)
- [x] **E1**: GPTQ 4-policy ranking validation — Kendall τ +1.000 (§8a)
- [x] **E2-mapping pilot**: Timeloop으로 3 shape × 2 bits 매핑 비교 — byte effect가 mapping 차이를 흡수, motivation으로 활용 (§8b)
- [x] **E2-residency**: Knapsack-based on-chip pinning sweep — aware 대 oblivious 1.78~3.16× DRAM reduction at GBuf=512 MB (§8c)

### 다음 (critical path)

- [ ] **E3**: residency × accuracy × 정책 4종을 `pipeline/codesign.py`로 묶고 main figure 마무리. 입력: `results/residency_sweep.csv`, `results/eval/<tag>/ppl_wikitext2.json`. 산출: `results/codesign_qwen15b.csv`, `figs/codesign_main.png`.
- [ ] **E4**: `pipeline/policy.py`에 HW-normalized score 옵션 추가 (`weight_metric: ppl_per_byte` 등). 새 정책 4종 생성 → 같은 residency sweep 재실행 → main figure에 추가 라인.
- [ ] **Sweep**: `qwen25_7b_instruct`, `gemma2_2b_it`로 확장. sensitivity (~30분/모델) → policy (즉시) → GPTQ (~3시간/policy/7B) → residency sweep (즉시).

### 후속 / 부수 작업

- [ ] residency 모델의 한계 보강 (§8c "한계 & follow-up"):
  - 0/1 knapsack → prefetch + partial-pin 모델
  - 생성 길이 T 변수화
  - Accelergy full PAT energy 분석
- [ ] `policy.py` ILP 모드(PuLP) — Pareto edge case 검증
- [ ] activation 정밀도(현재 fp16 고정) 변수화로 search space 확장
- [ ] `pipeline/hw_timeloop._materialize_arch()` 의 Accelergy 통합 (현재는 PAT 내장 모델만)
