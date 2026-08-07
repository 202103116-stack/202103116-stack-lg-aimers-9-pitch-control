# LG Aimers 9기 — Phase 2: 투구 제구 성공 확률 예측 AI 온라인 해커톤

LG Aimers 9기 Phase 2 온라인 해커톤 프로젝트 저장소입니다.

- 대회 페이지: https://dacon.io/competitions/official/236743
- 주최: LG AI연구원 · 주관: DACON

## 문제 개요

각 행은 KBO 경기에서 한 번의 투구 직전 상태를 나타냅니다. 투구 직전까지 알 수 있는 경기 상황·선수 정보·과거 이력 피처만으로 해당 투구의 **제구 성공 확률(`control_success`, 1=성공/0=실패)** 을 예측합니다.

- **평가지표**: Brier Skill Score — `score = max(0, 100000 * (1 - brier / baseline_brier))`, `baseline_brier = r(1-r)` (`r`은 평균 성공률)
- **제출 방식**: CSV가 아닌 **코드 제출**(zip) — `model/`(학습된 모델) + `script.py`(추론 스크립트) + `requirements.txt`
- **추론 제약**: 6 vCPU / 28GB RAM / L4 GPU(22.4GiB), 추론 10분 이내, 패키지 설치 10분 이내, 오프라인 실행, zip 10GB 이하

### 핵심 규칙 (리키지 금지)

- 평가 데이터의 각 행은 **독립적으로** 예측해야 함 — `test.csv` 내 다른 행을 이용한 집계/빈도/타겟인코딩/롤링 피처 금지
- 현재 투구의 사후 정보(실제 위치, 판정, 결과, 구종, Trackman 측정값), 2025년 Trackman 데이터 사용 금지
- 외부 API(OpenAI, Gemini 등) 및 비공개 가중치 모델, 외부 데이터셋 사용 금지 (공개 가중치 + 최소 비상업 라이선스 모델만 허용)
- 제공된 `asof_*` 피처(투구 직전 시점까지의 과거 기록 기반)는 공식 입력으로 사용 가능

### 일정

| 날짜 | 내용 |
| --- | --- |
| 08.05 | 대회 시작 |
| 08.26 | 팀 병합 마감 |
| 09.01 | 리더보드 제출 마감 |
| 09.02 | 대회 종료 |
| 09.07 | 코드/PPT 제출 마감 |
| 09.11 | 코드 검증 마감 |
| 09.14 | Phase 3(오프라인 해커톤, 약 100팀) 참가자 발표 |

## 데이터

컬럼별 상세 설명은 [`docs/data_description.md`](docs/data_description.md) 참고. `data/` 폴더는 `.gitignore`에 의해 버전관리되지 않으므로, 팀원 각자 원본 `open.zip`을 받아 아래 구조로 배치해야 합니다.

```
data/
  train.csv               # 학습 입력 + 정답 (1,475,092행 x 49컬럼, 2019~2023 시즌)
  test.csv                # 배포용 샘플 5행 (실제 평가 245,789행은 제출 시 서버가 교체)
  sample_submission.csv   # 제출 양식 샘플
  trackman_history.csv    # 2019~2024 Trackman 로그 (1,793,078행 x 30컬럼, 보조 피처용)
```

## 폴더 구조

```
.
├── data/                        # 대회 데이터 (gitignore, 로컬에만 존재)
├── docs/
│   └── data_description.md      # 공식 데이터 설명서
├── notebooks/
│   └── baseline/                # 공식 베이스라인 노트북 (RandomForest)
│       ├── baseline_train.ipynb
│       └── baseline_inference.ipynb
├── references/
│   └── baseline_example/        # 공식 제출 예시 (script.py, requirements.txt, rf.pkl)
├── src/                         # 피처엔지니어링 / 학습 / 제출 패키징 스크립트
├── experiments/                 # 실험 기록 (가설/변경점/검증 스코어 로그)
├── model/                       # 학습된 모델 아티팩트 출력 (gitignore)
├── output/                      # 예측/제출 파일 출력 (gitignore)
├── submission/                  # 제출 패키징
│   ├── script.py                # 추론 스크립트 (버전관리됨)
│   ├── requirements.txt         # 제출용 최소 requirements (버전관리됨)
│   ├── model/                   # 패키징용 모델 사본 (gitignore)
│   ├── submit.zip               # 항상 최신 빌드 — Dacon에 이 파일만 업로드 (gitignore)
│   └── archive/                 # 실험별 제출 이력 사본, 예: exp001_lgbm_submit.zip (gitignore)
├── CLAUDE.md                    # Claude Code 작업 규칙/컨텍스트
├── requirements.txt
└── .gitignore
```

## 환경 설정

평가 서버가 **Python 3.11.15**를 쓰므로(CLAUDE.md "Submission Constraints" 참고), 로컬도 Python 3.11로 맞출 것. 시스템 기본 `python`이 3.11이 아니면 `py -3.11` 등으로 버전을 지정할 것.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Windows에서 `jupyter` 설치가 "Long Path" 관련 오류로 실패할 수 있음 — 이 경우 나머지 패키지(pandas/scikit-learn/lightgbm/xgboost 등)는 정상 설치되니 우선 스크립트로 작업을 진행하고, jupyter가 필요하면 [Windows 긴 경로 지원 활성화](https://pip.pypa.io/warnings/enable-long-paths) 후 재설치할 것.

## 베이스라인 실행

1. `data/` 폴더에 원본 CSV 배치
2. `notebooks/baseline/baseline_train.ipynb` 실행 → `model/rf.pkl` 생성 (2019~2023 학습, 2024 검증)
3. `notebooks/baseline/baseline_inference.ipynb` 실행 → `output/submission.csv` 생성

베이스라인은 `trackman_history.csv`를 활용하지 않는 얕은 RandomForest 모델입니다. 개선 방향은 주로 (1) `trackman_history.csv` 기반 피처 엔지니어링, (2) `asof_*` cold-start 결측치 처리/smoothing, (3) 더 강력한 모델(LightGBM/XGBoost 등)로의 전환입니다.

exp_001(LightGBM + 파생 피처)은 `python src/train_lgbm.py`로 재현할 수 있습니다.

## 제출 패키징

`model/` 안의 학습된 아티팩트가 준비되면, 아래 명령으로 `submission/submit.zip`(Dacon 업로드용, 항상 최신)과 `submission/archive/<label>_submit.zip`(이력 보관용 사본)을 함께 생성합니다.

```bash
python src/package_submission.py exp001_lgbm
```

Dacon에는 **항상 `submission/submit.zip`만** 업로드하면 됩니다 — 실험이 바뀌어도 파일명이 고정돼 있어 헷갈릴 일이 없습니다. zip 내부 구조(`model/`, `script.py`, `requirements.txt`가 최상위)는 Dacon 코드 제출 가이드와 정확히 일치하도록 스크립트가 보장합니다.

## 작업 워크플로우 & 실험 기록

이 프로젝트는 [`docs/WORKFLOW.md`](docs/WORKFLOW.md)에 정리된 9단계 ML 워크플로우(문제정의 → 데이터이해 → 누수검증 → 검증전략 → 전처리/FE → 모델링 → 평가/오류분석 → 최종학습/제출 → 회고, 7→5/6 피드백 루프 포함)를 따릅니다.

- 새 실험(피처/모델 변경)을 시작할 때: [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md)를 복사해서 `experiments/exp_XXX_설명.md`로 기록
- 리더보드에 제출할 때마다: [`experiments/SUBMISSION_LOG.md`](experiments/SUBMISSION_LOG.md)에 한 줄 기록 → 팀 회의 때 이 파일 기준으로 진행상황 공유

## 팀

5인 팀 (Phase 2 진행 중)
