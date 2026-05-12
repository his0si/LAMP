# LAMP — acpl 서버 환경

이 폴더는 **acpl 서버**(`ewha-acpl6`)에서 LAMP (layer-wise mixed-precision)
실험을 돌리기 위한 self-contained 설정입니다. 환경 만들기부터 baseline 모델
다운로드, 검증, smoke test까지의 모든 절차를 이 README 하나로 처리할 수
있도록 정리했습니다. 다른 서버 설정은 `../RT-Server/`에 따로 있고, 두 폴더는
conda 환경/스크립트/모델 스냅샷을 서로 공유하지 않습니다.

This folder is a self-contained configuration for running LAMP (layer-wise
mixed-precision) experiments on the **acpl server** (`ewha-acpl6`). Everything
from creating the conda environment to downloading baselines, validating the
setup, and running a generation smoke test is described in this README. The
other server lives in `../RT-Server/`; the two folders share no conda env,
script, or model snapshot.

---

## 1. 서버 사양 / Server specs

2026-05-11 acpl 서버(`ewha-acpl6`)에서 확인한 사양입니다.

Observed on `ewha-acpl6` on 2026-05-11:

| 항목 | 값 |
| --- | --- |
| Hostname | `ewha-acpl6` |
| OS | Ubuntu 24.04.4 LTS (kernel 6.8.0-100-generic, x86_64) |
| CPU | AMD Ryzen Threadripper PRO 5965WX, 24C / 48T |
| System RAM | 251 GiB (8 GiB swap) |
| Disk | 1.8 TB root (`/dev/nvme0n1p2`), ~600 GiB free |
| GPU | **NVIDIA RTX 2000 Ada Generation × 4**, 16 GiB VRAM each (총 64 GiB) |
| NVIDIA driver | 590.48.01 |
| CUDA driver capability | CUDA 13.1 |
| Conda location | `/usr/local/conda` (system-wide install) |
| Conda env (이 폴더) | **`LAMP_acpl`** |
| Python | 3.11 (Conda 관리) |
| PyTorch | `2.9.1+cu128` (CUDA 12.8 wheel, forward-compat with driver 590) |
| Transformers | `5.8.0` |
| Hugging Face Hub | `1.14.0` |

GPU 한 장은 16 GiB지만 4장이라 7B/8B는 단일 GPU FP16으로, 9B는 multi-GPU
sharding으로 4-bit quantization 없이 돌릴 수 있습니다. 1.5B–3B는 어느 GPU에서나
여유롭게 실행됩니다.

Each GPU has 16 GiB. With four cards (64 GiB total), 7B/8B fit on one card in
FP16, 9B can be sharded across cards in FP16, and 1.5B–3B fit comfortably on any
single card.

---

## 2. 폴더 구조 / Folder layout

```text
acpl/
├── README.md              # this file
├── environment.yml        # Conda env spec (name: LAMP_acpl)
├── requirements.txt       # pip requirements (after torch is installed)
├── scripts/
│   └── setup_env.sh       # one-shot env setup
└── baseline/
    ├── models.yaml        # baseline model manifest
    ├── download_models.py # Hugging Face snapshot downloader
    ├── check_env.py       # env + manifest sanity check
    ├── verify_gpu.py      # CUDA / FP16 matmul check
    ├── smoke_test.py      # generation smoke test
    └── models/            # gitignored; downloaded weights live here
```

모든 명령은 이 `acpl/` 폴더에서 실행하는 것을 기준으로 작성했습니다.

All commands below assume the current directory is this `acpl/` folder.

---

## 3. 초기 1회 설정 / One-time setup

### 3-1. Conda 초기화 (acpl 서버 특이사항)

acpl 서버는 conda가 `/usr/local/conda`에 설치되어 있고, 새 셸이 자동으로
초기화하지 않습니다. **매 새 SSH 세션마다 한 번** 실행하거나, `~/.bashrc`에
영구적으로 추가하세요.

The acpl server has Conda at `/usr/local/conda` and new shells do not source it
automatically. Run this **once per new SSH session**, or append it to
`~/.bashrc` to make it permanent:

```bash
source /usr/local/conda/etc/profile.d/conda.sh
```

영구 등록:

To persist:

```bash
echo 'source /usr/local/conda/etc/profile.d/conda.sh' >> ~/.bashrc
```

### 3-2. Conda Terms of Service 수락 (최초 1회만)

기본 채널 ToS를 수락하지 않은 새 계정이라면 다음을 한 번만 실행합니다.
acpl 서버는 이미 수락된 상태입니다.

If your account hasn't accepted the default-channel ToS yet, run this once.
On the acpl server, it is already accepted:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 3-3. LAMP_acpl 환경 생성 + 의존성 설치

#### 옵션 A — 스크립트로 한 번에

```bash
cd /home/heeseo/LAMP/acpl
bash scripts/setup_env.sh
conda activate LAMP_acpl
```

#### 옵션 B — 명령을 직접 입력

```bash
cd /home/heeseo/LAMP/acpl
conda env create -f environment.yml
conda activate LAMP_acpl
python -m pip install torch==2.9.1+cu128 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`cu128`은 CUDA 12.8 runtime wheel입니다. driver 590(CUDA 13.1)은
forward-compatible이라 추가 작업 없이 동작합니다.

`cu128` ships the CUDA 12.8 runtime. Driver 590 (CUDA 13.1) is
forward-compatible, so no further action is needed.

### 3-4. 설치 결과 검증

```bash
conda activate LAMP_acpl
python -c "import torch, transformers, huggingface_hub, accelerate, bitsandbytes; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count()); \
print('transformers', transformers.__version__); \
print('huggingface_hub', huggingface_hub.__version__); \
print('accelerate', accelerate.__version__); \
print('bitsandbytes', bitsandbytes.__version__)"
```

기대 출력 (acpl 서버 기준):

Expected output on the acpl server:

```text
torch 2.9.1+cu128 cuda True gpus 4
transformers 5.8.0
huggingface_hub 1.14.0
accelerate 1.13.0
bitsandbytes 0.49.2
```

---

## 4. Hugging Face 로그인

`baseline/download_models.py`는 gated 모델(Llama, Gemma)을 받기 위해 HF 토큰이
필요합니다.

`baseline/download_models.py` needs an HF token for gated models (Llama,
Gemma).

```bash
# 현재 로그인 상태 확인
hf auth whoami

# 로그인이 안 되어 있으면
huggingface-cli login
```

또는 환경변수만 잠깐 쓰려면 `export HF_TOKEN=hf_...` 한 줄로도 됩니다.

You can also set `HF_TOKEN` for a one-off session:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx
```

Meta Llama와 Google Gemma는 각각 Hugging Face 모델 페이지에서 라이선스 동의를
받아야 합니다. 동의가 review 대기 중이면 `GatedRepoError: 403`이 발생합니다.

Meta Llama and Google Gemma each require accepting the license on the model's
Hugging Face page. While the request is pending, you'll see `GatedRepoError:
403`.

---

## 5. Baseline 모델 다운로드

baseline 모델 manifest(`baseline/models.yaml`)에 정의된 5개 모델은 다음과
같습니다.

The baseline manifest (`baseline/models.yaml`) defines five models:

| Key | Hugging Face repo | Role / 역할 | 접근 |
| --- | --- | --- | --- |
| `llama31_8b_instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 메인 8B 기준선 | Gated |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | 다른 계열(Qwen)에서의 일반화 검증 | Public |
| `gemma2_9b_it` | `google/gemma-2-9b-it` | 중형 Gemma 비교 | Gated |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | 소형 Gemma 비교 (small ↔ medium scale 효과) | Gated |
| `qwen25_15b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 빠른 sweep · ablation | Public |

### 5-1. 다운로드 계획만 확인 (dry-run)

```bash
python baseline/download_models.py --dry-run
```

### 5-2. 기본 baseline 전체 다운로드

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py all --keep-going
```

- `--keep-going`: 한 모델이 실패해도(예: gated 승인 대기) 나머지를 계속 받음.
- `HF_HUB_DISABLE_XET=1`: HF의 Xet 전송 backend가 멈출 때 우회. 대용량
  다운로드에서 안정적이라 기본으로 켜는 것을 권장.

### 5-3. 일부만 받기

```bash
# 작은 모델만 먼저 받아서 파이프라인 검증
HF_HUB_DISABLE_XET=1 python baseline/download_models.py qwen25_15b_instruct
```

```bash
# Llama 승인이 늦게 나면 나중에 별도로
HF_HUB_DISABLE_XET=1 python baseline/download_models.py llama31_8b_instruct
```

다운로드된 스냅샷은 `acpl/baseline/models/<key>/` 아래에 저장됩니다. 이
디렉터리는 git ignore 대상입니다.

Downloaded snapshots live under `acpl/baseline/models/<key>/`. The directory is
gitignored.

---

## 6. 환경 / GPU 검증

```bash
python baseline/check_env.py    # torch/cuda 정보 + 각 모델의 config·weight 존재 여부
python baseline/verify_gpu.py   # CUDA device 확인 + FP16 matmul 실행
```

`verify_gpu.py`는 기본적으로 `cuda:0`만 검사합니다. 4장을 모두 확인하려면:

`verify_gpu.py` checks only `cuda:0` by default. To validate every card:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python baseline/verify_gpu.py
done
```

---

## 7. Smoke test (generation 확인)

다운로드된 모델을 실제로 로드해서 짧은 generation까지 돌립니다.

Loads each downloaded model and runs a short deterministic generation.

```bash
# 작은 모델 한 개만
python baseline/smoke_test.py qwen25_15b_instruct

# 다운로드된 baseline 전체
python baseline/smoke_test.py all
```

`smoke_test.py`는 `device_map="auto"`를 사용합니다. acpl 서버처럼 GPU가 4장
있으면 9B 모델도 자동으로 sharding되어 FP16 그대로 로드됩니다. 단일 GPU만
보이도록 강제하면 9B는 `--load-in-4bit`이 필요합니다.

`smoke_test.py` uses `device_map="auto"`. With four GPUs on the acpl server, 9B
shards across cards in FP16. If you constrain it to a single card, pair 9B with
`--load-in-4bit`:

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/smoke_test.py gemma2_9b_it --load-in-4bit
```

---

## 8. Multi-GPU 운영 팁 / Multi-GPU tips

- `device_map="auto"`는 Hugging Face Accelerate가 weight를 GPU에 분산
  배치합니다. 9B는 보통 GPU 2장에 걸쳐 로드됩니다.
- 1.5B / 2B / 7B / 8B는 단일 GPU로 충분합니다. 여러 모델을 동시에 sweep할
  때는 `CUDA_VISIBLE_DEVICES=0`, `=1`, `=2`, `=3`으로 분리해서 4개 job을
  병렬로 띄울 수 있습니다.
- 9B를 단일 GPU에서 돌려야 한다면 `--load-in-4bit`을 사용합니다. 단, 4-bit
  결과는 layer-wise precision 평가의 기준선과 동일하지 않으므로 비교 보고에
  명시합니다.

- `device_map="auto"` lets Accelerate spread weights across cards. 9B usually
  lands on two cards.
- 1.5B / 2B / 7B / 8B fit on a single card. Use `CUDA_VISIBLE_DEVICES=0..3` to
  run four jobs in parallel.
- Use `--load-in-4bit` for 9B only when limiting to a single card; note that
  the 4-bit result is not equivalent to the FP16 baseline.

---

## 9. 트러블슈팅 / Troubleshooting

| 증상 | 원인 / 해결 |
| --- | --- |
| `conda: command not found` | Conda init이 안 됨. `source /usr/local/conda/etc/profile.d/conda.sh` 실행. |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | 위 3-2 ToS 수락 명령 1회 실행. |
| `GatedRepoError: 403` for Llama | Meta Llama 라이선스 승인 대기. HF 페이지에서 access 신청 후 승인되면 같은 명령 다시 실행. |
| `GatedRepoError: 403` for Gemma | Google Gemma 라이선스 동의 안 됨. HF Gemma 페이지에서 동의 후 재시도. |
| Xet 전송이 멈춤 | `HF_HUB_DISABLE_XET=1`을 앞에 붙여 다시 실행. |
| 9B가 단일 GPU에서 OOM | `--load-in-4bit` 사용, 또는 multi-GPU에서 `device_map="auto"`로 sharding. |
| `bitsandbytes` import 실패 | Linux x86_64에서만 동작. 시스템 패키지 문제이면 `pip install --upgrade bitsandbytes` 재시도. |

---

## 10. 현재 로컬 상태 / Current local status (2026-05-12)

| Key | 로컬 상태 |
| --- | --- |
| `qwen25_7b_instruct` | done 15 GB 다운로드 완료 |
| `gemma2_9b_it` | done 18 GB 다운로드 완료 |
| `gemma2_2b_it` | done 4.9 GB 다운로드 완료 |
| `qwen25_15b_instruct` | done 2.9 GB 다운로드 완료, smoke test 통과 |
| `llama31_8b_instruct` | not yet Meta Llama 라이선스 review 대기 (`GatedRepoError: 403`) |

승인된 뒤 Llama만 별도로 받으려면:

Once Llama access is approved, fetch only that one:

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py llama31_8b_instruct
```

---

## 11. Git 정책 / Git policy

- `acpl/baseline/models/` 아래 weight는 절대 커밋하지 않습니다 (`.gitignore`).
- 환경 파일, 스크립트, manifest, 문서만 커밋합니다.
- manifest나 평가 스크립트를 바꿀 때는 `../RT-Server/baseline/`에도 동일하게
  반영해 두 서버 결과의 비교 가능성을 유지합니다.

- Never commit weights under `acpl/baseline/models/` (handled by `.gitignore`).
- Commit only env files, scripts, manifest, and documentation.
- Mirror manifest and script changes to `../RT-Server/baseline/` so results
  stay comparable.
