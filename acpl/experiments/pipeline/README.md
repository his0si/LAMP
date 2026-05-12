# `pipeline/` — implementation modules

이 폴더는 LAMP 실험의 모든 **알고리즘 구현체**를 담은 Python 패키지입니다.
폴더만 보고도 무엇이 들어있는지 알게 하기 위해 모듈 이름은 *실험의 phase*를
그대로 따라갑니다. 각 모듈은 노트북에서 단독 import해 호출할 수 있고, 같은
이름의 `scripts/run_<phase>.py`가 CLI 래퍼입니다.

This folder contains the **algorithm implementations** for every phase
of the LAMP experiment as one importable Python package. Each module is
independently usable; the matching `scripts/run_<phase>.py` is the CLI
front-end.

## 모듈 빠른 참조 / Module quick reference

| 모듈 | Phase | 입력 | 출력 | CLI |
| --- | --- | --- | --- | --- |
| `config.py` | (utility) | YAML 경로 | dataclass / dict | — |
| `sensitivity.py` | **1** — sensitivity scoring | HF model + tokenizer | `{layer_idx: Δppl_4bit, Δppl_8bit}` JSON | `run_sensitivity.py` |
| `policy.py` | **2** — bit-width policy | sensitivity JSON + budget | `per_layer_bits` YAML | `run_policy.py` |
| `hw_timeloop.py` | **3** — Timeloop+Accelergy | arch YAML + (shape, bits) | `MappingResult` dataclass | `run_hw.py` |
| `quant_runner.py` | **4a** — GPTQ quantize | policy YAML | quantized HF checkpoint | `run_quant.py` |
| `eval.py` | **4b** — perplexity + downstream | quantized checkpoint | ppl JSON / lm-eval JSON | `run_eval.py` |
| `profile_gpu.py` | **5** — GPU profile | quantized checkpoint | tokens/s, mJ/token | `run_profile.py` |
| `viz.py` | **6** — plots | result JSONs | PNG | — |
| `mapping.py` | E2 (mapping pilot) | mapping cache | aggregated metrics | — |
| `residency.py` | **E2-residency (main)** | policy + GBuf size | knapsack-pinned tile set, DRAM bytes/token | (driver inline) |
| `codesign.py` | E3 — final aggregator | policy + mapping + eval | Pareto plot, codesign CSV | — |

## 데이터 흐름 / Data flow

```text
sensitivity.run_sensitivity()
  → results/sensitivity/{target}.json
     │
     ▼
policy.make_policy()  ─────►  results/policies/{target}_{B}bit.yaml
     │                                                  │
     │              ┌───────────────────────────────────┘
     │              ▼
     │       quant_runner.quantize()
     │              │
     │              ▼
     │       results/quantized/<tag>/   ◄─── HF format checkpoint
     │              │
     │       ┌──────┴───────────┐
     │       ▼                  ▼
     │   eval.eval_perplexity   profile_gpu.profile_generation
     │       │                  │
     │       ▼                  ▼
     │   results/eval/<tag>/    results/profile/<tag>.json
     │
     ▼
hw_timeloop.run_layer()         ◄─── Timeloop+Accelergy 호출
     │
     ▼
results/hw/mapping_cache.json
     │
     ▼
mapping.aggregate_model() + codesign.build_points()
     │
     ▼
results/figs/codesign_pareto.png  (E3 main figure)
```

## 모듈 상호의존 / Internal imports

```text
config        ◄── 모든 모듈
sensitivity   → config (없음, 자체)
policy        → (없음)
hw_timeloop   → config
mapping       → config
codesign      → config, mapping
quant_runner  → (없음)
eval          → sensitivity (compute_perplexity 재사용)
profile_gpu   → (없음)
viz           → (matplotlib lazy import)
```

`__init__.py`의 docstring과 `__all__`이 진실의 단일 출처입니다.
