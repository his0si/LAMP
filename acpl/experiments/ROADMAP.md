# LAMP Experiment Roadmap — E1 → 끝까지

이 문서는 paper의 실험 계획 전체를 한눈에 보이게 정리한 1-pager입니다.
세부 측정 방법론은 `README.md` §6, 코드 인터페이스는 `pipeline/README.md`.

## Paper claim (2026-05-12 — E2 pilot 후 보정)

> **"Given a fixed mixed-precision policy and a finite on-chip buffer,
> precision-aware *residency scheduling* pins more layers' weights on-chip
> and reduces per-token DRAM traffic versus a precision-oblivious
> scheduler that budgets every tile at the worst-case bit-width."**

Layer-wise quantization sensitivity와 mixed-precision policy는 **입력 (known)**.
본 논문의 contribution은 그 policy 위에서 어느 layer의 weight를 on-chip에
pin할지 결정하는 *scheduling*.

> **Pivot note.** 원래 claim은 "precision-aware mapping reduces energy/token".
> E2 pilot (README §8b)에서 16×16 Eyeriss arch + N=1 decode에서는 mapping
> geometry가 cycle/energy 차이를 거의 만들지 않고 byte-width로 다 흡수됨을
> 확인. mapping → residency로 contribution surface를 이동. 자세한 결과는
> README §8c.

## 실험 전체 흐름

```text
[Phase 1] sensitivity      ──┐
[Phase 2] policy            ──┼──► [Phase 4a] GPTQ ──► [Phase 4b] eval ──┐
                              │                                          │
                              └──► [Phase 3] Timeloop mapping cache ──► [E3] codesign aggregate
                                                                          │
                                                                          ▼
                                                              accuracy × efficiency Pareto
```

순서: **E1 → E2 → E3 → E4 → 확장 sweep.**
E1과 E2는 viability gate(여기서 무너지면 contribution이 깨짐).

---

## Experiment 진행 상황 / Status

| ID | 단계 | 상태 | 산출물 | 다음 액션 |
| --- | --- | :---: | --- | --- |
| **E1** | Sensitivity ranking validation | **done** | `results/eval/*/ppl_wikitext2.json`, `results/e1_ranking.csv`, `figs/e1_ranking_validation.png` | (없음) |
| **E2-mapping pilot** | 3 shape × 2 bits Timeloop mapping spread | **done (negative result)** | `results/hw/mapping_cache.json` (6), `results/hw/runs/*`. 결론: mapping이 byte effect에 흡수, contribution surface 이동. | (없음, motivation으로 활용) |
| **E2-residency** | Memory-residency scheduling (knapsack over 4 정책 × 8 GBuf × 2 스케줄러) | **done (main result)** | `results/residency_sweep.csv`, `figs/e2_residency_pareto.png`. Aware vs oblivious 1.78×~3.16× DRAM reduction at GBuf=512 MB. | E3 진행 |
| **E3** | (재정의) Pareto figure + 정책별 codesign 표 | **done** | `results/codesign_qwen15b.csv`, `figs/codesign_main.png` (512 MB), `figs/codesign_multi_gbuf.png` | (없음) |
| **E4** | HW-normalized per-tile policy | **done** | `results/sensitivity/*_per_projection.json`, `results/policies/*_hwnorm.yaml`, `results/e4_residency_compare.csv`, `figs/e4_hwnorm_vs_greedy.png`. Residency 측 gain은 작지만(3.2%) policy shape 질적 변화 — k/v_proj 위주 보호. README §8e. | accuracy 검증은 sweep 단계 |
| **Sweep** | 7B, Gemma-2B 일반화 | **done** | 3 모델 모두 ranking 보존, residency win 1.33~1.97× @ iso-relative-GBuf. `figs/sweep_cross_model.png`, README §8f. | (없음, paper draft 진행) |

---

## E1 — Sensitivity Ranking Validation (완료)

**목적**: greedy isolated-layer sensitivity가 만든 정책 ranking이 실제 GPTQ
양자화 후에도 보존되는지 확인. 깨지면 inter-layer-aware policy search로
contribution 축이 옮겨가야 함.

**입력**: `pipeline.sensitivity` 결과 + `pipeline.policy` 결과 4개 정책
(INT4-uniform, Mixed-4.5, Mixed-5.0, INT8-uniform).

**측정 방법**: GPTQ 32-sample calibration, group_size=128, desc_act=True →
WikiText-2 test split sliding-window perplexity (seq_len=2048).

**결과 (Qwen2.5-1.5B-Instruct, 2026-05-12, wall-time 3023s):**

| Policy | avg bpw | real ppl | Δ vs FP16 | upper Δ |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 16.0 | 8.890 | — | — |
| INT4-uniform | 4.0 | 10.452 | +1.56 | +4.00 |
| Mixed-4.5 | 4.57 | 10.279 | +1.39 | +2.31 |
| Mixed-5.0 | 5.0 | 10.091 | +1.20 | +1.90 |
| INT8-uniform | 8.0 | 9.604 | +0.71 | +0.03 |

- `Kendall τ(real vs upper) = +1.000`, `Kendall τ(bpw vs real) = −1.000`.
- **결론: PASS.** 추가 발견(GPTQ가 상한의 ~60% 흡수, INT8 GPTQ floor ≈ 0.71)은
  `README.md` §8a에 정리.

**재현 명령**:
```bash
conda activate LAMP_acpl
cd /home/heeseo/LAMP/acpl
bash experiments/scripts/_e1_quant_and_eval.sh
```

---

## E2-mapping — Mapping Spread Pilot (DONE 2026-05-12, motivation only)

**Pilot 결과 한 줄**: 16×16 Eyeriss + N=1 decode에서 mapping geometry는 실제로
4b vs 8b가 다르게 잡지만(K split 여부), cycle/energy 차이는 byte-width로 다
흡수돼서 cross-app loss는 ~1.00×. 결과 + 해석은 `README.md §8b`.

**시사점**: mapping search를 main contribution에 두지 않고 byte traffic에
집중한 residency scheduling으로 이동. 아래 E2-residency 참조.

---

## E2-residency — Memory-Residency Scheduling (DONE 2026-05-12 — main contribution)

**결과 한 줄**: GBuf=512 MB에서 precision-aware residency scheduling이
precision-oblivious 대비 DRAM bytes/token을 **1.78×~3.16× 감소** (mixed-precision
정책 기준). GBuf=1024 MB에서는 mixed-precision까지 100% on-chip resident.
INT8-uniform은 모델 자체가 1.31 GB로 어떤 스케줄링도 도와줄 수 없음.

**자세한 내용**: `README.md §8c` (표·플롯·해석·한계).

**원래 mapping 가설** (Mapping Spread): 같은 layer shape에서 4-bit vs 8-bit이

**측정**:
- 7 unique linear shape (`q/k/v/o_proj`, `gate/up/down_proj`) × 2 bits = 14 mapper run.
- 각 run에서 top-K mapping의 `(cycles, energy_pJ, pe_utilization, dram_bytes, gbuf_hit_rate)` 추출.
- **Cross-application loss**: `pipeline.mapping.cross_application_loss()` —
  4-bit best mapping을 8-bit에 강제 적용했을 때 energy/cycles 비율, 역도 동일.

**완료된 pilot**: `q_proj`, `o_proj`, `up_proj` × {INT4, INT8} = 6 매퍼 실행.
결과: mapping geometry는 다르지만 cycle/energy 결과는 byte 효과로 흡수.
자세한 분석 `README.md` §8b.

---

## E2-residency — Main Contribution (DONE 2026-05-12)

**한 줄**: GBuf=512 MB에서 precision-aware residency scheduling이
precision-oblivious 대비 DRAM bytes/token을 **1.78~3.16× 감소**.
GBuf=1024 MB에서는 Mixed-5.0까지의 모든 정책이 100% on-chip resident
(DRAM/tok = 0). INT8-uniform은 모델 자체가 1.31 GB라 어떤 스케줄링도 안 됨.

**코드**: `pipeline/residency.py` (knapsack pinning).

**산출물**:
- `results/residency_sweep.csv` (4 정책 × 8 GBuf × 2 스케줄러 = 64 rows)
- `figs/e2_residency_pareto.png` (2-panel main figure)
- `README.md` §8c (표·해석·한계)

---

## E3 — Pareto integration (DONE 2026-05-12)

**결과 한 줄**: GBuf=512 MB에서 INT4-uniform/Mixed-4.5/Mixed-5.0 에서
precision-aware × precision-oblivious DRAM 격차가 1.79~3.17×. INT8-uniform은
aware==oblivious (자기 자신이 worst-case 정밀도). 자세한 내용 `README.md §8d`.

**산출물**:
- `results/codesign_qwen15b.csv` (64 codesign 점)
- `figs/codesign_main.png` (single-GBuf 512 MB main figure)
- `figs/codesign_multi_gbuf.png` (3-panel 128/512/1024 MB)

**원래 설계** (참고용):

**비교 4-strategy × policy 4-budget**:

| Tag | Strategy | 정밀도 정책 | Scheduler | 입력 메트릭 |
| --- | --- | --- | --- | --- |
| A | `uniform_int4_aware` | 전 layer INT4 | aware | ppl=10.45, DRAM/tok = best |
| B | `uniform_int8_aware` | 전 layer INT8 | aware = oblivious | ppl=9.60 |
| **C** | `mixed_oblivious_int8` | Mixed-N | INT8 budget 강제 | ppl=mixed-N |
| **D** | `mixed_aware` | Mixed-N | bit-aware budget | ppl=mixed-N |

x = ppl(실측 E1), y = DRAM bytes/token (residency sweep). GBuf size를
parameter로 표시 (예: solid lines = 512 MB, dashed = 1024 MB).

**필요 코드 작업**: `pipeline/codesign.py`를 residency 결과를 받도록 확장:
- `build_points()`에 mapping cache 대신 residency rows 입력
- `plot_pareto()`에 GBuf parameter 추가

**산출물**: `results/codesign_qwen15b.csv`, `figs/codesign_main.png`.

**예상 시간**: 1-2시간.

---

## E4 — HW-normalized Policy (확장)

**가설**: greedy 점수를 `Δppl(l) / Δbyte(l)`로 바꾸면 (= per-byte accuracy
gain) 정책이 달라지고 E3 Pareto에서 추가 이득이 나오는가.

**현재 greedy**: `argmax Δppl(l, 4→8) / 1bit` (모든 layer에 동일 byte cost).
**제안**: `argmax Δppl(l, 4→8) / (n_weights(l) × 4 / 8)` — byte-normalized
score. 작은 layer (k/v_proj)일수록 4b→8b 승격이 cheaper, 우선순위 ↑.

E2-residency가 끝났으니 분모(byte cost)는 즉시 계산 가능. method novelty
보강용.

**필요 코드 작업** (`pipeline/policy.py`):
- `greedy_assign()`에 `cost_fn: Callable[[int, int], float]` 옵션 추가.
- 기본 cost = 1 (현재), HW-aware cost = `n_weights × (target_bits - current_bits) / 8`.

**산출물**: `results/policies/<target>_hwnorm_<B>bit.yaml` + 같은 residency
sweep을 새 정책으로 다시 → E3 figure에 추가 라인.

**예상 시간**: 코딩 반일, 실행+분석 반나절.

---

## 확장 Sweep — 7B / Gemma-2B 일반화

1.5B에서 E3까지 통과한 뒤에만 의미 있음. 같은 파이프라인을 다른 target으로 다시
돌려서 효과가 모델 family·scale에 invariant한지 확인.

**필요 작업**:
- `configs/targets.yaml`에서 `qwen25_7b_instruct`, `gemma2_2b_it` 활성화
  (이미 정의 있음).
- Sensitivity는 모델당 ~30분-1시간 (7B), ~10분 (2B).
- Policy는 즉시.
- GPTQ는 7B에서 ~3-4시간/policy. 4 policy = 반나절. multi-GPU sharding 가능.
- E2 mapping cache는 hidden_size/intermediate_size가 다르므로 모델당 별도 캐시.

**산출물**: 모델별 같은 형식의 결과 디렉터리, table에 컬럼 추가.

**예상 시간**: 모델당 1-2일.

---

## 전체 일정 (2026-05-12 업데이트)

| 단계 | 상태 | 실제 / 남은 시간 |
| --- | --- | --- |
| E1 | done | wall-time 50분 (4 policies × GPTQ 32-sample) |
| E2 mapping pilot | done | 코딩 반일 + 매퍼 ~5분, negative motivation으로 보존 |
| E2-residency main | done | 코딩 반일 + sweep 즉시, **main contribution** |
| E3 codesign integrate + main figure | done | 1시간, `figs/codesign_main.png` polished main figure |
| E4 per-tile hwnorm policy + comparison | done | 코딩 1시간, per-proj sensitivity 4분, sweep 즉시 |
| Qwen-7B sweep (sensitivity → policy → GPTQ → residency) | done | sensitivity 2분, GPTQ × 4 ~75분, total ~80 min wall-time |
| Gemma-2B sweep | done | sensitivity 2분, GPTQ × 4 ~20분 |
| Paper draft + figure polishing | next | 1주 |
| **남은 총량** | | **~1주 (paper draft only)** |

ISCA/MICRO/HPCA 마감이면 paper draft까지 3-4주 안전 마진. 현재 critical-path
4 viability gate(E1 + E2-mapping + E2-residency + infra install) 모두 통과.

---

## 자주 함께 보는 파일

| 보고 싶은 것 | 파일 |
| --- | --- |
| 셋업·실행법 | `README.md` |
| 측정 방법론 (어떻게 ppl 재고, energy 재는지) | `README.md` §6 |
| E1 결과 표 | `README.md` §8a, `results/e1_ranking.csv` |
| 코드 모듈 한눈에 | `pipeline/README.md` |
| (이 문서) 실험 전체 흐름 | `ROADMAP.md` |
