# acpl/baseline — Baseline 모델 재현 가이드

이 문서는 **acpl 서버에서 baseline 모델 셋을 지금과 똑같이 갖춰 놓는 방법**을
단계별로 적어둔 reproduction 가이드입니다. 환경(`LAMP_acpl`)이 이미 만들어져
있고 Hugging Face 로그인이 되어 있다고 가정합니다. 그 전 단계는
[`../README.md`](../README.md) 1~4 절을 따라 하세요.

This is a step-by-step reproduction guide for the **acpl server**'s baseline
model set. It assumes the `LAMP_acpl` conda environment is already created and
Hugging Face authentication is set up — see [`../README.md`](../README.md)
sections 1–4 for those steps.

---

## 1. Baseline 모델 셋 / Baseline model set

manifest는 `models.yaml`에 정의되어 있습니다.

The manifest is defined in `models.yaml`:

| Key | Hugging Face repo | Family | Size | Role / 역할 | Access |
| --- | --- | --- | --- | --- | --- |
| `llama31_8b_instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Llama | 8B | 메인 8B 기준선 | Gated |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | Qwen | 7B | 다른 계열 일반화 검증 | Public |
| `gemma2_9b_it` | `google/gemma-2-9b-it` | Gemma | 9B | 중형 Gemma 비교 (multi-GPU FP16) | Gated |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | Gemma | 2B | 소형 Gemma 비교 (scale 효과) | Gated |
| `qwen25_15b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | Qwen | 1.5B | 빠른 sweep · ablation | Public |

`download_models.py all`은 위 5개 전체를 받습니다. role 이름 어느 것도
"optional_*"로 시작하지 않기 때문에 default 다운로드 대상에서 빠지는 모델은
없습니다.

`download_models.py all` downloads all five entries. None of the roles begin
with `"optional_*"`, so no model is excluded from the default set.

각 모델 디렉터리는 `acpl/baseline/models/<key>/`에 저장됩니다. 이 경로는
gitignore 대상이며 GitHub에 올라가지 않습니다.

Each model directory lands under `acpl/baseline/models/<key>/`. The path is
gitignored and never reaches GitHub.

---

## 2. 재현 명령 / Reproduction commands

### 2-1. 사전 준비

```bash
cd /home/heeseo/LAMP/acpl
source /usr/local/conda/etc/profile.d/conda.sh   # acpl 서버는 conda init 필요
conda activate LAMP_acpl
hf auth whoami                                    # 또는 huggingface-cli whoami
```

`hf auth whoami`가 user 이름을 출력하지 않으면 `huggingface-cli login`으로
로그인하거나 `export HF_TOKEN=hf_xxx`를 설정합니다. acpl 서버는 토큰이 환경
변수에 미리 잡혀 있을 수 있으니 `env | grep -i HF_TOKEN`으로 확인하세요.

If `hf auth whoami` does not print a user, run `huggingface-cli login` or set
`export HF_TOKEN=hf_xxx`. Check `env | grep -i HF_TOKEN` first — on the acpl
server it may already be exported.

### 2-2. 다운로드 계획 미리보기 (dry-run)

```bash
python baseline/download_models.py --dry-run
```

manifest의 5개 모델이 표로 출력되면 정상입니다.

You should see all five entries in a Rich table.

### 2-3. 전체 baseline 다운로드 (실제 실행)

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py all --keep-going
```

옵션 의미:

- `all`: manifest의 default 모델 전체.
- `--keep-going`: 한 모델이 실패해도(예: Llama gated 승인 대기) 나머지를 계속
  받음. 끝에 실패 목록을 요약하고 exit code 1로 종료.
- `HF_HUB_DISABLE_XET=1`: HF의 Xet 전송 backend를 끄고 일반 HTTP로 받음. 대용량
  스냅샷에서 Xet이 멈추는 사례를 회피. 기본으로 켜 두는 것을 권장.

### 2-4. 일부만 받기

```bash
# 작은 모델 하나만으로 파이프라인 검증
HF_HUB_DISABLE_XET=1 python baseline/download_models.py qwen25_15b_instruct

# Llama 승인이 늦게 떨어졌을 때 따로
HF_HUB_DISABLE_XET=1 python baseline/download_models.py llama31_8b_instruct

# 두 개 이상을 지정해도 됨
HF_HUB_DISABLE_XET=1 python baseline/download_models.py qwen25_15b_instruct gemma2_2b_it
```

---

## 3. 예상 다운로드 크기와 소요 시간 / Expected size & duration

2026-05-11 acpl 서버에서 측정한 값입니다.

Measured on the acpl server on 2026-05-11:

| Key | 파일 수 | 디스크 점유 | 소요 시간 |
| --- | --- | --- | --- |
| `qwen25_7b_instruct` | 11 | 15 GB | ~21 분 |
| `gemma2_9b_it` | 11 | 18 GB | ~26 분 |
| `gemma2_2b_it` | 9 | 4.9 GB | ~7.5 분 |
| `qwen25_15b_instruct` | 7 | 2.9 GB | ~4.5 분 |
| `llama31_8b_instruct` | 12 | (16 GB 예상) | 라이선스 승인 후 |
| **합계 (Llama 제외)** | 38 | **~40 GB** | **~60 분** |

전체 baseline(Llama 포함)이 받아지면 약 56 GB가 필요합니다. acpl 서버의 루트
파티션(`/dev/nvme0n1p2`)은 약 600 GiB가 비어 있으므로 여유 충분.

The full baseline including Llama needs about 56 GB. The acpl server's root
partition has ~600 GiB free, plenty of headroom.

---

## 4. 다운로드 결과 검증 / Verifying the download

```bash
python baseline/check_env.py
```

`check_env.py`는 (1) torch/CUDA 정보와 (2) 각 모델의 `config.json`/weight 존재
여부를 표로 출력합니다. 다운로드가 성공한 모델은 `status` 칼럼에 `ok:
<model_type>` (예: `qwen2`, `gemma2`)로 표시되고, 다운로드 안 된 모델은
`missing local snapshot`으로 표시됩니다.

`check_env.py` prints (1) torch/CUDA info and (2) a per-model table of config
and weight presence. Successful models show `status = ok: <model_type>`;
missing snapshots show `missing local snapshot`.

기대 출력 (현재 acpl 상태):

Expected output (current acpl state):

```
- torch: 2.9.1+cu128
- cuda available: True
- cuda:0..3: NVIDIA RTX 2000 Ada Generation, 15.8 GiB

llama31_8b_instruct  ...  missing local snapshot
qwen25_7b_instruct   ...  ok: qwen2
gemma2_9b_it         ...  ok: gemma2
gemma2_2b_it         ...  ok: gemma2
qwen25_15b_instruct  ...  ok: qwen2
```

GPU 4장이 모두 인식되는지도 확인:

Verify all four GPUs:

```bash
python baseline/verify_gpu.py
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python baseline/verify_gpu.py
done
```

---

## 5. Smoke test (실제 로드 + generation)

다운로드된 모델을 하나씩 GPU에 올리고 짧은 deterministic generation까지
돌립니다. 기본 prompt는 "Summarize layer-wise mixed precision in one sentence."
입니다.

Loads each downloaded model onto GPUs and runs a short deterministic
generation. The default prompt is `"Summarize layer-wise mixed precision in one
sentence."`.

```bash
# 작은 모델 한 개로 빠르게 확인
python baseline/smoke_test.py qwen25_15b_instruct

# 받은 baseline 전체 (acpl: device_map=auto 로 9B 자동 sharding)
python baseline/smoke_test.py all
```

기대 출력 (Qwen 1.5B 결과 예):

Expected output (Qwen 1.5B sample):

```
┃ qwen25_15b_instruct │ local  │ ok     │ Layer-wise Mixed Precision (LMP)…
```

acpl 서버는 GPU가 4장이라 9B 모델도 4-bit 없이 sharding으로 그대로 로드됩니다.
단일 GPU 강제 시(예: `CUDA_VISIBLE_DEVICES=0`) 9B는 OOM이 나므로
`--load-in-4bit`을 함께 씁니다.

The acpl server has four GPUs, so 9B loads in FP16 via sharding. When
constrained to one GPU, pair 9B with `--load-in-4bit`:

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/smoke_test.py gemma2_9b_it --load-in-4bit
```

---

## 6. Llama 3.1 라이선스 처리 / Handling Llama gated access

`download_models.py all`이 Llama에서 `GatedRepoError: 403`을 내면 다음 중 한
경우입니다.

If `download_models.py all` fails on Llama with `GatedRepoError: 403`, one of
the following is true:

1. **라이선스 미신청**: Hugging Face의
   `meta-llama/Meta-Llama-3.1-8B-Instruct` 페이지에서 "Request access"를 누르고
   양식을 제출합니다.
2. **라이선스 review 대기**: 메일이 올 때까지 기다립니다. 보통 몇 시간 ~ 1일.
3. **다른 계정으로 로그인 됨**: `hf auth whoami`로 토큰의 소유자를 확인.
   승인된 계정과 다르면 그 계정으로 다시 `huggingface-cli login`.

승인이 떨어지면 같은 명령으로 Llama만 받습니다.

Once access is granted, fetch only Llama:

```bash
HF_HUB_DISABLE_XET=1 python baseline/download_models.py llama31_8b_instruct
```

다운로드가 끝나면 `check_env.py`에서 `llama31_8b_instruct` 행이 `ok: llama`로
바뀌어야 합니다.

After completion, the `llama31_8b_instruct` row in `check_env.py` should
become `ok: llama`.

---

## 7. 현재 acpl 로컬 상태 / Current acpl local status (2026-05-12)

| Key | 상태 |
| --- | --- |
| `qwen25_7b_instruct` | done 15 GB 완료 |
| `gemma2_9b_it` | done 18 GB 완료 |
| `gemma2_2b_it` | done 4.9 GB 완료 |
| `qwen25_15b_instruct` | done 2.9 GB 완료 · smoke test 통과 |
| `llama31_8b_instruct` | not yet Meta Llama 라이선스 review 대기 |

`acpl/baseline/models/` 총 용량 ~40 GB. Llama 승인 시 ~56 GB로 늘어남.

Total under `acpl/baseline/models/`: ~40 GB. Will grow to ~56 GB once Llama is
approved.

---

## 8. 트러블슈팅 / Troubleshooting

| 증상 | 원인 / 해결 |
| --- | --- |
| Xet 전송이 멈춤 | `HF_HUB_DISABLE_XET=1` 앞에 붙이고 재실행. |
| `GatedRepoError: 403` (Llama) | 위 6 절 참고. |
| `GatedRepoError: 403` (Gemma) | Google Gemma 모델 페이지에서 라이선스 동의 후 재시도. |
| `Read timed out`, 일부 파일만 받힘 | 같은 명령을 다시 실행하면 이어받기. `download_models.py`가 같은 `local_dir`를 사용. |
| `local_dir_use_symlinks` 경고 | huggingface_hub의 deprecation 경고로 무시 가능. |
| `Completed with failures: ...` + exit 1 | `--keep-going`이 부분 실패를 허용한 결과. 실패한 모델만 다시 받으면 됨. |

---

## 9. Git 정책 / Git policy

`acpl/baseline/models/` 안의 weight는 절대 commit하지 않습니다. 저장소에 들어가
야 할 것은 `models.yaml`, `download_models.py`, 검증 스크립트, 그리고 이
README뿐입니다. `.gitignore`가 모델 디렉터리를 무시하도록 다음 룰이
포함되어야 합니다:

Never commit weights under `acpl/baseline/models/`. The repo should contain
only `models.yaml`, `download_models.py`, validation scripts, and this README.
The repo's `.gitignore` must contain a rule that ignores the model directory:

```gitignore
# 두 가지 중 한 가지면 됨:
**/baseline/models/
# 또는 명시적으로:
acpl/baseline/models/
RT-Server/baseline/models/
```

`git check-ignore -v acpl/baseline/models/x` 실행 결과에 위 룰이 매치되어야
안전합니다.

`git check-ignore -v acpl/baseline/models/x` should report a hit against the
rule above.
