# LAMP

Layer-wise Mixed-precision experiments for causal language models.

이 저장소는 인과 언어 모델(causal LM)에 대한 layer-wise mixed-precision 실험
환경을 **두 개의 서로 다른 서버**에서 평행하게 운영합니다. 각 서버 폴더는
self-contained입니다 — 자체 conda 환경 spec, 의존성 목록, 설치 스크립트,
baseline manifest, 다운로드/검증/smoke 스크립트, 그리고 (gitignored인)
로컬 모델 스냅샷을 모두 폴더 안에 가지고 있습니다. 두 폴더는 conda
환경/스크립트/모델 스냅샷을 서로 공유하지 않습니다.

This repository runs layer-wise mixed-precision experiments on causal language
models in parallel on **two different servers**. Each server folder is
self-contained — its own conda env spec, requirements, setup script, baseline
manifest, download/check/smoke scripts, and (gitignored) local model snapshots.
The two folders share no conda env, script, or model snapshot.

---

## Servers / 서버 구성

| Folder | Host / 머신 | GPU | Conda env | 비고 |
| --- | --- | --- | --- | --- |
| [`acpl/`](acpl/) | `ewha-acpl6` (acpl 서버) | 4 × NVIDIA RTX 2000 Ada, 16 GiB each (총 64 GiB) | **`LAMP_acpl`** | 9B는 multi-GPU FP16 sharding 가능 |
| [`RT-Server/`](RT-Server/) | RT-Server | 1 × NVIDIA RTX 5060 Ti, 16 GiB | **`LAMP_RT`** | 단일 GPU; 9B는 `--load-in-4bit` 기본 |

각 폴더의 README가 그 서버의 사양, 환경 설치 절차, baseline 다운로드 방법,
검증·smoke test, multi-/single-GPU 운영 팁, 트러블슈팅, 현재 로컬 상태를
모두 설명합니다.

Each folder's README documents server specs, environment install, baseline
download, validation, smoke tests, multi-/single-GPU tips, troubleshooting, and
current local status.

- [`acpl/README.md`](acpl/README.md) — acpl 서버 전체 가이드
- [`acpl/baseline/README.md`](acpl/baseline/README.md) — acpl baseline 모델 상세 reproduction 가이드
- [`RT-Server/README.md`](RT-Server/README.md) — RT-Server 전체 가이드

---

## Shared Baseline Models / 공통 baseline 모델

두 서버 모두 동일한 baseline 모델 셋을 사용합니다 (`<server>/baseline/models.yaml`).

Both servers use the identical baseline set (`<server>/baseline/models.yaml`):

| Key | Hugging Face repo | Role / 역할 | Access |
| --- | --- | --- | --- |
| `llama31_8b_instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 메인 8B 기준선 | Gated (Meta Llama) |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | 다른 계열(Qwen)에서의 일반화 검증 | Public |
| `gemma2_9b_it` | `google/gemma-2-9b-it` | 중형 Gemma 비교 (acpl: multi-GPU FP16, RT-Server: 4-bit) | Gated (Gemma) |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | 소형 Gemma 비교 (small ↔ medium scale 효과) | Gated (Gemma) |
| `qwen25_15b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 빠른 sweep · ablation | Public |

manifest는 두 폴더에 동일하게 들어 있습니다. 모델 셋을 바꿀 때는 양쪽을 함께
업데이트해서 결과 비교 가능성을 유지하세요.

The `models.yaml` is mirrored in both folders. When changing the model set,
update both copies so results stay comparable.

---

## Repository Layout / 저장소 구조

```text
LAMP/
├── README.md                # this file (두 서버 overview)
├── .gitignore
├── acpl/                    # acpl 서버, conda env: LAMP_acpl
│   ├── README.md
│   ├── environment.yml
│   ├── requirements.txt
│   ├── scripts/setup_env.sh
│   └── baseline/
│       ├── README.md
│       ├── models.yaml
│       ├── download_models.py
│       ├── check_env.py
│       ├── verify_gpu.py
│       ├── smoke_test.py
│       └── models/          # gitignored, ~40 GB downloaded on acpl
└── RT-Server/               # RT-Server, conda env: LAMP_RT
    ├── README.md
    ├── environment.yml
    ├── requirements.txt
    ├── scripts/setup_env.sh
    └── baseline/
        ├── models.yaml
        ├── download_models.py
        ├── check_env.py
        ├── verify_gpu.py
        ├── smoke_test.py
        └── models/          # gitignored
```

---

## Quick Start

각 서버에 SSH로 들어가 **해당 서버의 폴더 안에서만** 작업합니다. 두 폴더를 같은
머신에 둬도 conda 환경이 별개(`LAMP_acpl`, `LAMP_RT`)라 충돌하지 않습니다.

SSH into each server and work **only inside that server's folder**. The two
folders can coexist on the same host because the conda envs (`LAMP_acpl`,
`LAMP_RT`) are independent.

```bash
# On the acpl server:
cd /home/heeseo/LAMP/acpl
source /usr/local/conda/etc/profile.d/conda.sh   # acpl 서버는 conda init 필요
conda activate LAMP_acpl
# 자세한 설치/다운로드/검증 절차는 acpl/README.md 참고

# On RT-Server:
cd /home/heeseo/LAMP/RT-Server
conda activate LAMP_RT
# 자세한 절차는 RT-Server/README.md 참고
```

---

## Why Folders Instead of Branches / 왜 폴더 분리인가

같은 코드를 두 서버에서 돌리고 결과를 비교해야 하므로, 브랜치를 가르기보다
**한 브랜치에 server별 폴더를 두는 방식**을 택했습니다. 이유:

We keep both servers in the same branch as sibling folders rather than separate
branches because:

- 실험 코드 (`download_models.py`, `check_env.py`, `smoke_test.py` 등)는 두
  서버에서 동일해야 결과 비교가 의미 있음. 브랜치를 나누면 cherry-pick/merge
  부담이 큼.
- manifest와 평가 스크립트 변경을 한 PR에서 동시에 반영할 수 있음.
- 결과·로그를 같은 트리에 두면 비교 노트 작성이 쉬움.
- 환경/스크립트만 다르므로 폴더 분리만으로 충분히 격리됨.

— Code (download/check/smoke) must match across servers for the comparison to
  be meaningful; branching makes that painful.
— Manifest and script changes can land in a single PR.
— Keeping results/logs in one tree makes cross-server comparison notes easier.
— Only env and scripts differ, so folder separation is sufficient isolation.

---

## Git Policy / Git 정책

- `*/baseline/models/` 아래의 모델 weight는 **절대 커밋하지 않습니다**.
  `.gitignore`가 `**/baseline/models/`로 두 서버 폴더 모두를 잡습니다.
- `node_modules/`, `package*.json`도 자동으로 무시됩니다 (npm 산출물).
- 저장소에는 environment 파일, 스크립트, manifest, 문서만 commit합니다.
- baseline manifest나 평가 스크립트를 바꿀 때는 두 서버 폴더에 동일하게
  반영하세요.

- Never commit weights under `*/baseline/models/`. The `.gitignore` rule
  `**/baseline/models/` covers both server folders.
- `node_modules/` and `package*.json` are also ignored.
- Commit only environment files, scripts, manifest, and documentation.
- Mirror manifest / script changes across both server folders.
