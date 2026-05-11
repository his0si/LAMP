# LAMP

Layer-wise Mixed-precision experiments for causal language models.

인과 언어 모델(causal language model)을 대상으로 layer-wise mixed-precision
실험을 하기 위한 저장소입니다. 현재는 실험 전에 필요한 baseline 모델
환경, 다운로드 스크립트, 로컬 검증 스크립트를 정리해두었습니다.

This repository prepares the baseline model environment for layer-wise
mixed-precision experiments on causal language models. It keeps model selection,
Hugging Face downloads, environment checks, and smoke tests reproducible.

## Baseline Documentation / Baseline 문서

자세한 모델 목록, Hugging Face 접근 권한, 다운로드 방법, 검증 방법, 현재
로컬 모델 상태는 [baseline/README.md](baseline/README.md)에 정리되어
있습니다.

Detailed baseline model choices, Hugging Face access requirements, download
commands, validation steps, and the current local snapshot status are documented
in [baseline/README.md](baseline/README.md).

## Repository Layout / 저장소 구조

- `baseline/models.yaml`: baseline 모델 manifest입니다. Hugging Face repo ID,
  모델 역할, 로컬 저장 경로가 들어 있습니다.
- `baseline/download_models.py`: Hugging Face snapshot을 `baseline/models/`
  아래로 다운로드합니다.
- `baseline/check_env.py`: Python, Torch, CUDA, 로컬 모델 config/weight 존재
  여부를 확인합니다.
- `baseline/verify_gpu.py`: CUDA 인식과 FP16 matmul 실행을 확인합니다.
- `baseline/smoke_test.py`: 다운로드된 모델을 로드하고 짧은 generation을
  실행합니다.
- `environment.yml`, `requirements.txt`, `scripts/setup_env.sh`: 환경 설치
  파일입니다.

- `baseline/models.yaml`: baseline model manifest with Hugging Face repo IDs,
  model roles, and local snapshot paths.
- `baseline/download_models.py`: downloads Hugging Face snapshots into
  `baseline/models/`.
- `baseline/check_env.py`: checks Python, Torch, CUDA, and local model
  config/weight availability.
- `baseline/verify_gpu.py`: verifies CUDA discovery and a small FP16 matmul.
- `baseline/smoke_test.py`: loads a downloaded model and runs a short generation
  test.
- `environment.yml`, `requirements.txt`, `scripts/setup_env.sh`: environment
  setup files.

모델 weight는 Git에 커밋하지 않습니다. 로컬 weight는 `baseline/models/`에
저장되며, 이 디렉터리는 Git ignore 대상입니다.

Model weights are not committed. Local weights are stored under
`baseline/models/`, which is ignored by Git.

## Tested Environment / 테스트한 환경

2026-05-11 기준 로컬에서 확인한 환경입니다.

The local environment tested on 2026-05-11 was:

- OS: Ubuntu/Linux
- GPU: NVIDIA GeForce RTX 5060 Ti
- VRAM reported by Torch: 15.5 GiB
- NVIDIA driver: 570.211.01
- CUDA runtime/build used by PyTorch: CUDA 12.8
- Python: 3.11 in the `LAMP` Conda environment
- PyTorch: `2.9.1+cu128`
- Transformers: `5.8.0`
- Hugging Face Hub: `1.14.0`

CUDA/GPU 환경 검사는 통과했습니다. 또한
`Qwen/Qwen2.5-1.5B-Instruct`는 로컬 weight로 smoke test를 통과해 실제
generation까지 확인했습니다.

The CUDA/GPU checks passed. `Qwen/Qwen2.5-1.5B-Instruct` was also downloaded
locally and passed the smoke test with real text generation.

## Baseline Models / Baseline 모델

모델 목록은 `baseline/models.yaml`에 정의되어 있습니다.

The model set is defined in `baseline/models.yaml`.

| Key | Hugging Face repo | Role / 역할 | Downloaded by `all` | Access |
| --- | --- | --- | --- | --- |
| `llama31_8b_instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Main 8B reference baseline / 메인 8B 기준선 | Yes | Gated |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | Cross-family generalization / 다른 계열 일반화 검증 | Yes | Public |
| `gemma2_9b_it` | `google/gemma-2-9b-it` | Medium Gemma comparison / 중형 Gemma 비교 | Yes | Gated |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | Optional small Gemma / 선택 소형 Gemma | No | Gated |
| `qwen25_15b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | Fast sweep and ablation / 빠른 sweep 및 ablation | Yes | Public |

`download_models.py all`은 기본 baseline 세트인 `llama31_8b_instruct`,
`qwen25_7b_instruct`, `gemma2_9b_it`, `qwen25_15b_instruct`를 받습니다.
`gemma2_2b_it`는 명시적으로 선택할 때만 받는 optional 모델입니다.

`download_models.py all` downloads the default baseline set:
`llama31_8b_instruct`, `qwen25_7b_instruct`, `gemma2_9b_it`, and
`qwen25_15b_instruct`. `gemma2_2b_it` is optional and downloaded only when
explicitly requested.

다운로드되는 weight는 각 Hugging Face repository의 공식 `safetensors`
snapshot입니다. 특정 revision을 고정해야 하는 실험이 아니라면 별도로
checkpoint를 고를 필요가 없습니다.

The downloaded weights are the official `safetensors` snapshots from each
Hugging Face repository. No separate checkpoint choice is required unless an
experiment intentionally pins a specific revision.

## Quick Start / 빠른 시작

환경을 만들고 dependency를 설치합니다.

Create the environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate LAMP
python -m pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

또는 setup script를 사용할 수 있습니다.

Or use the setup script:

```bash
bash scripts/setup_env.sh
conda activate LAMP
```

Hugging Face 로그인 상태를 확인합니다.

Check Hugging Face authentication:

```bash
hf auth whoami
```

로그인되어 있지 않으면 CLI로 로그인합니다.

If not logged in, authenticate with the CLI:

```bash
huggingface-cli login
```

기본 baseline 다운로드 계획을 미리 봅니다.

Preview the default baseline download plan:

```bash
python baseline/download_models.py --dry-run
```

기본 baseline 세트를 다운로드합니다.

Download the default baseline set:

```bash
python baseline/download_models.py all --keep-going
```

Hugging Face Xet 전송이 멈추면 Xet을 끄고 다시 실행합니다.

If the Hugging Face Xet transfer backend stalls, disable Xet and retry:

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py all --keep-going
```

작은 Qwen 1.5B 모델만 먼저 받고 검증하려면:

To download and validate only the small Qwen 1.5B model first:

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py qwen25_15b_instruct
python baseline/smoke_test.py qwen25_15b_instruct
```

## Validation / 검증

환경과 로컬 모델 파일 상태를 확인합니다.

Check the environment and local model files:

```bash
python baseline/check_env.py
python baseline/verify_gpu.py
```

전체 기본 baseline이 모두 다운로드된 뒤에는 다음으로 generation smoke test를
실행합니다.

After the full default baseline set is downloaded, run the generation smoke
test:

```bash
python baseline/smoke_test.py all --load-in-4bit
```

15.5 GiB GPU에서는 7B, 8B, 9B 모델에 `--load-in-4bit`를 사용하는 것을
권장합니다. 1.5B 모델은 테스트한 머신에서 quantization 없이 로드와 실행이
가능했습니다.

On the tested 15.5 GiB GPU, use `--load-in-4bit` for 7B, 8B, and 9B models.
The 1.5B model can load without quantization on the tested machine.

## Current Local Status / 현재 로컬 상태

2026-05-11 마지막 확인 기준:

As of the latest local check on 2026-05-11:

- `qwen25_15b_instruct`: Hugging Face에서 다운로드 완료, smoke test 통과.
- `llama31_8b_instruct`: Meta Llama gated access가 review 대기 상태라
  `GatedRepoError: 403`으로 다운로드 미완료.
- `qwen25_7b_instruct`: partial download는 삭제했고 로컬에는 보관하지 않음.
- `gemma2_9b_it`: 접근 확인은 통과했지만 full snapshot은 아직 다운로드하지 않음.
- `gemma2_2b_it`: optional 모델이며 접근 확인은 통과, 다운로드하지 않음.

- `qwen25_15b_instruct`: downloaded from Hugging Face and passed smoke test.
- `llama31_8b_instruct`: not downloaded because Meta Llama gated access was
  still pending review and returned `GatedRepoError: 403`.
- `qwen25_7b_instruct`: partial downloads were removed and no local snapshot is
  kept.
- `gemma2_9b_it`: access check passed, but the full snapshot has not been
  downloaded.
- `gemma2_2b_it`: optional model; access check passed, not downloaded.

## Git Policy / Git 정책

`baseline/models/` 아래의 모델 weight는 커밋하지 않습니다. 이 파일들은 매우
크고 일부는 license-gated 모델입니다. 저장소에는 manifest, script, 환경 파일,
문서만 커밋합니다.

Do not commit files under `baseline/models/`. Model snapshots are large and some
are license-gated. Commit only manifests, scripts, environment files, and
documentation.
