# LAMP — RT-Server 환경

이 폴더는 **RT-Server**(NVIDIA RTX 5060 Ti 16 GiB 단일 GPU 머신)에서 LAMP
(layer-wise mixed-precision) 실험을 돌리기 위한 self-contained 설정입니다. 환경
생성부터 baseline 다운로드, 검증, smoke test까지의 절차를 이 README 하나로
처리할 수 있습니다. acpl 서버 설정은 `../acpl/`에 따로 있고, 두 폴더는 conda
환경/스크립트/모델 스냅샷을 서로 공유하지 않습니다.

This folder is a self-contained configuration for running LAMP experiments on
**RT-Server** (a single-GPU host with an NVIDIA RTX 5060 Ti 16 GiB). The
acpl-server setup lives in `../acpl/`; the two folders share no conda env,
script, or model snapshot.

---

## 1. 서버 사양 / Server specs

이전에 확인한 RT-Server 사양입니다 (acpl 서버와 비교용으로 함께 보세요).

Previously observed RT-Server specs (compare against the acpl server):

| 항목 | 값 |
| --- | --- |
| OS | Ubuntu / Linux |
| GPU | **NVIDIA GeForce RTX 5060 Ti**, 단일 GPU, 16 GiB VRAM (Torch에서 ~15.5 GiB 보고) |
| NVIDIA driver | 570.211.01 |
| CUDA driver capability | CUDA 12.8 |
| Conda env (이 폴더) | **`LAMP_RT`** |
| Python | 3.11 (Conda 관리) |
| PyTorch | `2.9.1+cu128` |
| Transformers | `5.8.0` |
| Hugging Face Hub | `1.14.0` |

서버 사양이 위와 다르면 conda 환경을 만들기 전에 이 표를 갱신하세요. 가장
중요한 값은 GPU 종류·VRAM·driver 버전·CUDA capability입니다.

If your machine's specs drift from the table, update it before creating the
env. The most load-bearing fields are GPU model, VRAM, driver version, and CUDA
capability.

---

## 2. 폴더 구조 / Folder layout

```text
RT-Server/
├── README.md              # this file
├── environment.yml        # Conda env spec (name: LAMP_RT)
├── requirements.txt       # pip requirements (after torch is installed)
├── scripts/
│   └── setup_env.sh       # one-shot env setup
└── baseline/
    ├── models.yaml        # baseline model manifest (identical to acpl)
    ├── download_models.py
    ├── check_env.py
    ├── verify_gpu.py
    ├── smoke_test.py
    └── models/            # gitignored; downloaded weights live here
```

모든 명령은 이 `RT-Server/` 폴더에서 실행하는 것을 기준으로 작성했습니다.

All commands below assume the current directory is this `RT-Server/` folder.

---

## 3. 초기 1회 설정 / One-time setup

### 3-1. Conda 초기화

RT-Server에 conda가 사용자 홈(`~/miniconda3` 등)에 설치되어 있고 `~/.bashrc`로
PATH가 잡혀있다면 별도 init이 필요하지 않습니다. 그렇지 않은 경우 conda
설치 위치의 `etc/profile.d/conda.sh`를 한 번 source 해주세요.

If conda is installed in your home dir and already on PATH via `~/.bashrc`,
nothing to do. Otherwise:

```bash
source /path/to/conda/etc/profile.d/conda.sh
```

### 3-2. Conda Terms of Service 수락 (최초 1회)

기본 채널 ToS를 수락하지 않은 새 계정이라면 한 번만:

If your account hasn't accepted the default-channel ToS yet:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 3-3. LAMP_RT 환경 생성 + 의존성 설치

#### 옵션 A — 스크립트로 한 번에

```bash
cd /home/heeseo/LAMP/RT-Server
bash scripts/setup_env.sh
conda activate LAMP_RT
```

#### 옵션 B — 명령을 직접 입력

```bash
cd /home/heeseo/LAMP/RT-Server
conda env create -f environment.yml
conda activate LAMP_RT
python -m pip install torch==2.9.1+cu128 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

### 3-4. 설치 결과 검증

```bash
conda activate LAMP_RT
python -c "import torch, transformers, huggingface_hub, accelerate, bitsandbytes; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count()); \
print('transformers', transformers.__version__); \
print('huggingface_hub', huggingface_hub.__version__); \
print('accelerate', accelerate.__version__); \
print('bitsandbytes', bitsandbytes.__version__)"
```

RT-Server에서는 `cuda True`, `gpus 1` 이 나와야 정상입니다.

On RT-Server you should see `cuda True` and `gpus 1`.

---

## 4. Hugging Face 로그인

```bash
hf auth whoami
# 또는
huggingface-cli login
```

또는 환경변수만 잠깐 쓰려면:

Or for a one-off session:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx
```

Meta Llama와 Google Gemma는 모델 페이지에서 라이선스 동의가 필요합니다. acpl
서버와 같은 HF 계정을 쓰면 한 번만 승인받으면 됩니다.

Meta Llama and Google Gemma require accepting the license on their HF pages.
Using the same HF account as the acpl server avoids duplicate approvals.

---

## 5. Baseline 모델 다운로드

manifest는 acpl 서버와 동일합니다. 두 서버에서 같은 모델 셋을 받아야 결과
비교가 가능합니다.

The manifest matches the acpl server. Both servers must download the same set
for results to be comparable.

| Key | Hugging Face repo | Role / 역할 | 접근 |
| --- | --- | --- | --- |
| `llama31_8b_instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 메인 8B 기준선 | Gated |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | 다른 계열 일반화 검증 | Public |
| `gemma2_9b_it` | `google/gemma-2-9b-it` | 중형 Gemma 비교 (RT-Server에선 4-bit 권장) | Gated |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | 소형 Gemma 비교 | Gated |
| `qwen25_15b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 빠른 sweep · ablation | Public |

```bash
# 계획 확인
python baseline/download_models.py --dry-run

# 전체 다운로드 (Xet 우회 + gated 실패 시 진행)
HF_HUB_DISABLE_XET=1 python baseline/download_models.py all --keep-going

# 일부만
HF_HUB_DISABLE_XET=1 python baseline/download_models.py qwen25_15b_instruct
```

다운로드된 스냅샷은 `RT-Server/baseline/models/<key>/`에 저장됩니다.
gitignored.

Downloaded snapshots live under `RT-Server/baseline/models/<key>/` and are
gitignored.

---

## 6. 환경 / GPU 검증

```bash
python baseline/check_env.py
python baseline/verify_gpu.py
```

RT-Server는 GPU 1장이므로 `check_env.py` 출력에서 `cuda:0` 한 줄만 나옵니다.

RT-Server has a single GPU, so `check_env.py` lists only `cuda:0`.

---

## 7. Smoke test (generation 확인)

```bash
python baseline/smoke_test.py qwen25_15b_instruct
python baseline/smoke_test.py all --load-in-4bit
```

`smoke_test.py`는 `device_map="auto"`를 사용하는데, GPU가 한 장뿐이라 사실상
단일 GPU 배치가 됩니다. 큰 모델이 GPU에 안 들어가면 Accelerate가 CPU/disk로
일부 layer를 offload하며, 그 경우 속도가 크게 떨어집니다.

`smoke_test.py` uses `device_map="auto"`. Here it resolves to single-GPU
placement. If a model is too large, Accelerate offloads some layers to CPU/disk,
which is much slower than fitting fully on GPU.

---

## 8. 단일 GPU 운영 팁 / Single-GPU tips

RT-Server는 16 GiB VRAM 한 장이라 acpl처럼 multi-GPU sharding으로 9B를 FP16
전체 정밀도로 로드할 수 없습니다. 권장 전략:

RT-Server has a single 16 GiB card, so it cannot shard 9B to FP16 like acpl.
Recommended strategy:

- **1.5B / 2B**: FP16 그대로. 여유 메모리 큼.
  *FP16 fits easily.*
- **7B / 8B**: FP16에서 ~14–15 GiB를 차지. 짧은 generation은 그대로,
  배치/길이가 큰 평가는 `--load-in-4bit`이나 CPU offload 사용.
  *FP16 uses ~14–15 GiB; OK for short generation. For batched/long runs, use
  `--load-in-4bit` or CPU offload.*
- **9B**: 16 GiB에 FP16으로는 안 들어감. 기본적으로 `--load-in-4bit` 사용.
  비교 평가는 acpl의 FP16 결과와 함께 보고, 차이는 명시.
  *Does not fit at FP16 in 16 GiB. Default to `--load-in-4bit`; cross-check
  against the acpl FP16 run when comparing.*

```bash
python baseline/smoke_test.py llama31_8b_instruct
python baseline/smoke_test.py gemma2_9b_it --load-in-4bit
```

---

## 9. 트러블슈팅 / Troubleshooting

| 증상 | 원인 / 해결 |
| --- | --- |
| `conda: command not found` | Conda init 누락. `source /path/to/conda/etc/profile.d/conda.sh`. |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | 위 3-2 ToS 수락 명령 1회 실행. |
| `GatedRepoError: 403` for Llama / Gemma | 해당 모델의 HF 페이지에서 라이선스 동의 후 재시도. |
| Xet 전송이 멈춤 | `HF_HUB_DISABLE_XET=1`을 앞에 붙여 재실행. |
| 7B/8B/9B OOM | `--load-in-4bit` 또는 generation 길이 축소. |
| 결과가 acpl과 다름 | RT-Server의 4-bit 경로와 acpl의 FP16 경로 차이일 수 있음. quantization 여부를 비교 보고에 반드시 명시. |

---

## 10. Git 정책 / Git policy

- `RT-Server/baseline/models/` 아래 weight는 절대 커밋하지 않습니다.
- 환경 파일, 스크립트, manifest, 문서만 커밋합니다.
- manifest나 평가 스크립트를 바꿀 때는 `../acpl/baseline/`에도 동일하게
  반영해 두 서버 결과의 비교 가능성을 유지합니다.

- Never commit weights under `RT-Server/baseline/models/`.
- Commit only env files, scripts, manifest, and documentation.
- Mirror manifest and script changes to `../acpl/baseline/`.
