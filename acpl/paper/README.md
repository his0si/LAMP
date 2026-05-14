# Paper folder

투고용 IEEE Access 원고. 두 가지 형태로 보관:

| 파일 | 용도 |
| --- | --- |
| `access.tex` | IEEE에서 받은 **원본 빈 템플릿** (참조용, 수정 금지) |
| `main.tex` | **투고 원고 (이 파일을 컴파일)** |
| `draft.md` | 초안 (영문 + 한글). 사고 흐름·근거 정리용. 투고 전 무관. |

## 컴파일

`main.tex`는 IEEE Access 클래스(`ieeeaccess.cls`)를 요구합니다.
acpl 서버에는 LaTeX이 설치돼 있지 않으므로 다음 방법 중 하나를 사용하세요.

### Overleaf
1. 새 프로젝트 → IEEE Access 템플릿 선택
2. 이 폴더의 `main.tex` 업로드 (또는 내용 복붙)
3. `experiments/results/figs/` 아래의 PNG 3개도 같이 업로드:
   - `e1_ranking_validation.png`
   - `codesign_main.png`
   - `sweep_cross_model.png`
4. `main.tex`의 `\includegraphics{...}` 경로를 Overleaf 폴더 구조에 맞춰
   조정 (예: `figs/e1_ranking_validation.png`)
5. PDF 생성 → 다운로드

### 로컬 TeX Live
```bash
# 필수 패키지: ieeeaccess class, IEEEtran 계열
# Ubuntu: sudo apt install texlive-publishers texlive-latex-extra texlive-fonts-recommended
cd paper/
pdflatex main.tex
pdflatex main.tex   # 참고문헌 정렬 위해 2회
```

## 투고 시 체크리스트

IEEE Access 가이드라인 기준.

- [ ] **저자 정보 채우기**
  - `\author{}` — 실제 이름 + ORCID
  - `\address[1]{}` — 소속 + 이메일
  - `\corresp{}` — 교신 저자
  - `\tfootnote{}` — 펀딩 정보 (또는 삭제)
  - 세 개의 `\begin{IEEEbiography}{Author Name}` 본문도 채우기
  - 모든 저자 ORCID 공개 상태인지 확인
- [ ] **DOI 자리**: `\doi{10.1109/ACCESS.XXXX.DOI}` — IEEE가 채워줌
- [ ] **History**: `\history{Submitted on XX XX, 2026.}` 날짜 채우기
- [ ] **PDF와 tex 파일 모두 제출**, 내용 일치 확인
- [ ] **각 파일 40 MB 이하** (figure 압축 충분)
- [ ] **20 페이지 이하** (현재 추정: 약 12-15페이지)
- [ ] **3-10 keyword** ✓ (현재 9)
- [ ] **약어 본문 첫 등장 시 풀이름 정의** ✓
- [ ] **Lena 이미지 미사용** ✓ (해당 없음)
- [ ] **AI 사용 명시 + 인용** ✓ (Acknowledgment 섹션,
      `\bibitem{anthropic2024claude}`)
- [ ] **Article type** 선택 시 *Research Article*

## 본 원고의 근거

모든 수치는 `experiments/results/`의 산출물에 추적 가능:

| Paper 수치 | 산출물 |
| --- | --- |
| Table~II (mapping pilot) | `results/hw/runs/{q_proj,up_proj}_w{4,8}/timeloop-mapper.stats.txt` |
| Table~III (E1 GPTQ ppl, 4 models) | `results/eval/<tag>/ppl_wikitext2.json` 16 개 (4 모델 × 4 정책) |
| Table~IV (residency 512 MB) | `results/codesign_qwen15b.csv` |
| Table~V (GPU profile) | `results/profile/baseline_fp16_qwen25_15b.json` |
| Table~VI (cross-model win, 4 models) | `results/codesign_{gemma2_2b_it, qwen25_7b_instruct, llama31_8b_instruct}.csv` |
| Table~VII-VIII (E4 hwnorm) | `results/policies/*_hwnorm.yaml`, `results/e4_residency_compare.csv` |
| Fig.~2 (`e1_ranking_validation.png`) | 동일 디렉토리, `codesign.py` 생성 |
| Fig.~3 (`codesign_main.png`) | 동일 |
| Fig.~4 (`sweep_cross_model.png`) | 동일 |

`pipeline/`을 다시 실행하면 모든 산출물이 재생성됩니다 (`experiments/README.md`
§4 참조).
