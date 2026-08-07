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
├── src/                         # 팀 공용 피처엔지니어링 / 학습 / 추론 코드 (작업 예정)
├── model/                       # 학습된 모델 아티팩트 출력 (gitignore)
├── output/                      # 예측/제출 파일 출력 (gitignore)
├── requirements.txt
└── .gitignore
```

## 환경 설정

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 베이스라인 실행

1. `data/` 폴더에 원본 CSV 배치
2. `notebooks/baseline/baseline_train.ipynb` 실행 → `model/rf.pkl` 생성 (2019~2023 학습, 2024 검증)
3. `notebooks/baseline/baseline_inference.ipynb` 실행 → `output/submission.csv` 생성

베이스라인은 `trackman_history.csv`를 활용하지 않는 얕은 RandomForest 모델입니다. 개선 방향은 주로 (1) `trackman_history.csv` 기반 피처 엔지니어링, (2) `asof_*` cold-start 결측치 처리/smoothing, (3) 더 강력한 모델(LightGBM/XGBoost 등)로의 전환입니다.

## 팀

5인 팀 (Phase 2 진행 중)
