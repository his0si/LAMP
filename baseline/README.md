# Baseline Models / Baseline 모델

이 디렉터리는 layer-wise mixed-precision 실험에 사용할 baseline 모델
manifest, Hugging Face 다운로드 스크립트, 환경 검사 스크립트, smoke test를
담고 있습니다.

This directory contains the baseline model manifest, Hugging Face download
script, environment checks, and smoke tests for the layer-wise mixed-precision
experiments.

## Files / 파일

- `models.yaml`: 모델 key, Hugging Face repo ID, 역할, 로컬 저장 경로.
- `download_models.py`: `models.yaml`에 선언된 모델 snapshot을 다운로드.
- `check_env.py`: Torch/CUDA와 로컬 snapshot의 config/weight 존재 여부 확인.
- `verify_gpu.py`: CUDA device와 FP16 matmul 실행 확인.
- `smoke_test.py`: 로컬 모델을 로드하고 짧은 deterministic generation 실행.

- `models.yaml`: model keys, Hugging Face repo IDs, roles, and local paths.
- `download_models.py`: downloads model snapshots declared in `models.yaml`.
- `check_env.py`: checks Torch/CUDA and local config/weight availability.
- `verify_gpu.py`: verifies CUDA device discovery and FP16 matmul execution.
- `smoke_test.py`: loads a local model and runs a short deterministic generation.

## Model Set / 모델 구성

| Key | Hugging Face repo | Role / 역할 | Default `all` | Access |
| --- | --- | --- | --- | --- |
| `llama31_8b_instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Main 8B reference baseline / 메인 8B 기준선 | Yes | Gated |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | Cross-family validation / 다른 모델 계열 일반화 검증 | Yes | Public |
| `gemma2_9b_it` | `google/gemma-2-9b-it` | Medium Gemma comparison / 중형 Gemma 비교 | Yes | Gated |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | Optional smaller Gemma / 선택 소형 Gemma | No | Gated |
| `qwen25_15b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | Fast sweep and ablation / 빠른 sweep 및 ablation | Yes | Public |

`download_models.py all`은 기본 세트인 `llama31_8b_instruct`,
`qwen25_7b_instruct`, `gemma2_9b_it`, `qwen25_15b_instruct`를 다운로드합니다.
`gemma2_2b_it`는 disk/VRAM/시간 제약이 있을 때 명시적으로 선택하는 optional
모델입니다.

`download_models.py all` downloads the default set:
`llama31_8b_instruct`, `qwen25_7b_instruct`, `gemma2_9b_it`, and
`qwen25_15b_instruct`. `gemma2_2b_it` is an optional model for disk, VRAM, or
iteration-time constrained runs.

다운로드되는 weight는 각 Hugging Face repository의 공식 `safetensors`
snapshot입니다. 보통은 사용자가 별도로 weight/checkpoint를 고를 필요가
없습니다. 특정 실험에서만 `--revision`으로 Hugging Face revision을 고정합니다.

The scripts download the official Hugging Face `safetensors` snapshots. In the
normal workflow, users do not need to manually choose a separate weight or
checkpoint. Pin a Hugging Face revision with `--revision` only when an experiment
requires it.

## Why These Models / 모델 선정 이유

`llama31_8b_instruct`는 메인 기준선입니다. Llama 3.1 8B 계열에서
layer-wise precision 정책이 얼마나 잘 작동하는지 보는 중심 비교점입니다.

`llama31_8b_instruct` is the main reference baseline. It is the primary point for
checking whether the layer-wise precision policy works in the Llama 3.1 8B
setting.

`qwen25_7b_instruct`는 다른 계열 일반화 검증용입니다. Llama와 tokenizer 및
architecture가 다른 Qwen 계열에서 같은 정책이 유지되는지 확인합니다.

`qwen25_7b_instruct` is the cross-family validation model. It checks whether the
same policy transfers to Qwen, whose tokenizer and architecture differ from
Llama.

`gemma2_9b_it`는 중형 Gemma 비교용입니다. 7B/8B 모델과 비슷한 규모지만 다른
계열이므로 architecture와 scale 차이에 따른 동작을 확인하는 데 사용합니다.

`gemma2_9b_it` is the medium Gemma comparison model. It is close to the 7B/8B
scale but from a different family, so it helps expose architecture- and
scale-specific behavior.

`gemma2_2b_it`는 optional 소형 Gemma입니다. 9B가 너무 무겁거나 소형/중형
차이를 비교하고 싶을 때 사용합니다.

`gemma2_2b_it` is the optional small Gemma model. Use it when 9B is too expensive
or when comparing small and medium model behavior.

`qwen25_15b_instruct`는 빠른 sweep와 ablation용입니다. 큰 모델을 돌리기 전에
설정, 코드 경로, 실험 아이디어를 빠르게 확인하는 데 적합합니다.

`qwen25_15b_instruct` is the fast sweep and ablation model. It is suitable for
quickly checking settings, code paths, and experiment ideas before spending time
on larger models.

## Hugging Face Access / Hugging Face 접근 권한

먼저 로그인 상태를 확인합니다.

Check authentication first:

```bash
hf auth whoami
```

로그인되어 있지 않으면 CLI로 로그인합니다.

If not logged in, authenticate with the CLI:

```bash
huggingface-cli login
```

권한 요구사항은 다음과 같습니다.

Access requirements:

| Model | Requirement / 요구사항 |
| --- | --- |
| `llama31_8b_instruct` | Meta Llama gated license approval. Pending review still fails with `GatedRepoError: 403`. / Meta Llama 라이선스 승인 필요. review 대기 중이면 `GatedRepoError: 403` 발생. |
| `qwen25_7b_instruct` | Public / 공개 모델 |
| `gemma2_9b_it` | Google Gemma gated license approval / Google Gemma 라이선스 승인 필요 |
| `gemma2_2b_it` | Google Gemma gated license approval / Google Gemma 라이선스 승인 필요 |
| `qwen25_15b_instruct` | Public / 공개 모델 |

Qwen은 public이지만 로그인 상태로 다운로드하면 anonymous rate limit을 피하는 데
도움이 됩니다.

Qwen models are public, but authenticated downloads usually help avoid anonymous
rate limits.

## Setup / 환경 설정

Conda 환경을 만들고 dependency를 설치합니다.

Create the Conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate LAMP
python -m pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

이 프로젝트는 Python/package 격리를 위해 Conda를 사용하고, PyTorch는 CUDA
12.8 wheel index에서 pip로 설치합니다. 테스트한 RTX 5060 Ti 환경에서는 이
조합이 CUDA 사용 가능한 Torch를 설치했습니다.

This project uses Conda for Python/package isolation and installs PyTorch from
the CUDA 12.8 wheel index with pip. On the tested RTX 5060 Ti machine, this
installed a CUDA-enabled Torch build.

## Download / 다운로드

다운로드 계획만 확인합니다.

Preview the download plan:

```bash
python baseline/download_models.py --dry-run
```

기본 baseline 전체를 다운로드합니다.

Download the default baselines:

```bash
python baseline/download_models.py all
```

gated 모델 하나가 실패해도 나머지를 계속 받고 싶으면:

Continue past inaccessible gated models:

```bash
python baseline/download_models.py all --keep-going
```

Hugging Face Xet 전송 backend가 멈추거나 너무 느리면:

Use this if the Hugging Face Xet transfer backend stalls:

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py all --keep-going
```

작은 모델로 먼저 빠르게 검증하려면:

Download a smaller first pass:

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py qwen25_15b_instruct
```

optional Gemma 2B까지 같이 받으려면:

Download the optional Gemma 2B model as well:

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py qwen25_15b_instruct gemma2_2b_it
```

다운로드된 snapshot은 `baseline/models/` 아래에 저장됩니다.

Downloaded snapshots are placed under `baseline/models/`:

```text
baseline/models/llama31_8b_instruct/
baseline/models/qwen25_7b_instruct/
baseline/models/gemma2_9b_it/
baseline/models/gemma2_2b_it/
baseline/models/qwen25_15b_instruct/
```

다운로드가 중간에 끊기면 같은 명령을 다시 실행하면 됩니다. `config.json`과
tokenizer 파일만 있고 `.safetensors` weight가 없으면 실행 가능한 로컬 모델이
아닙니다.

If a download is interrupted, rerun the same command. A directory with only
`config.json` and tokenizer files but no `.safetensors` weights is not a runnable
local model.

## Environment Check / 환경 확인

환경과 로컬 모델 파일 상태를 확인합니다.

Check the environment and local model file status:

```bash
python baseline/check_env.py
python baseline/verify_gpu.py
```

`check_env.py`는 Torch/CUDA 정보와 각 모델의 local config/weight 존재 여부를
출력합니다. `verify_gpu.py`는 CUDA device 확인과 FP16 matmul을 실행합니다.

`check_env.py` reports Torch/CUDA information and local config/weight
availability for each model. `verify_gpu.py` checks CUDA device discovery and
runs an FP16 matmul.

## Smoke Test / 실행 검증

Qwen 1.5B 하나만 검증하려면:

To test only Qwen 1.5B:

```bash
python baseline/smoke_test.py qwen25_15b_instruct
```

기본 baseline 전체가 다운로드된 뒤 모두 검증하려면:

To test every default baseline after download:

```bash
python baseline/smoke_test.py all --load-in-4bit
```

smoke test는 tokenizer와 model을 로드하고, CUDA가 있으면 CUDA에 올린 뒤, 짧은
deterministic generation을 실행합니다. 로컬 snapshot이 없으면 기본적으로
실패합니다. 이렇게 해서 실험 도중 의도치 않은 remote download를 피합니다.

The smoke test loads the tokenizer and model, places the model on CUDA when
available, and runs a short deterministic generation. It fails by default if a
local snapshot is missing, which prevents accidental remote downloads during
experiments.

15.5 GiB GPU에서는 7B/8B/9B 모델에 `--load-in-4bit`를 사용하는 것을
권장합니다. 1.5B 모델은 테스트한 머신에서 quantization 없이 로드 가능했습니다.

On a 15.5 GiB GPU, use `--load-in-4bit` for 7B/8B/9B models. The 1.5B model can
load without quantization on the tested machine.

## Tested Local Status / 테스트한 로컬 상태

2026-05-11에 확인한 환경:

Environment checked on 2026-05-11:

- GPU: NVIDIA GeForce RTX 5060 Ti, 15.5 GiB visible to Torch.
- PyTorch: `2.9.1+cu128`.
- CUDA: available and verified with an FP16 matmul.
- Transformers: `5.8.0`.
- Hugging Face Hub: `1.14.0`.

마지막 로컬 snapshot 상태:

Latest local snapshot status:

| Key | Local status / 로컬 상태 |
| --- | --- |
| `qwen25_15b_instruct` | Downloaded from Hugging Face and passed `smoke_test.py`. / 다운로드 완료 및 smoke test 통과. |
| `llama31_8b_instruct` | Not downloaded because access was still pending review and returned `GatedRepoError: 403`. / Meta Llama 승인 대기 상태라 다운로드 미완료. |
| `qwen25_7b_instruct` | Not kept locally; partial downloads were removed. / partial 파일 삭제, 로컬 snapshot 없음. |
| `gemma2_9b_it` | Access check passed, but full snapshot was not downloaded. / 접근 확인은 통과, full snapshot 미다운로드. |
| `gemma2_2b_it` | Optional; access check passed, not downloaded. / optional 모델, 접근 확인 통과, 미다운로드. |

## GitHub Policy / GitHub 정책

`baseline/models/` 아래 파일은 GitHub에 올리지 않습니다. 모델 snapshot은 크고,
일부는 license-gated입니다. 저장소에는 manifest, script, 환경 파일, 문서만
포함합니다.

Do not push files under `baseline/models/` to GitHub. Model snapshots are large,
and some are license-gated. The repository should contain only the manifest,
scripts, environment files, and documentation needed to recreate the local model
directory.
