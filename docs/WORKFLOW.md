# ML 워크플로우 (LG Aimers 9기 Phase2)

이 프로젝트에서 따르는 표준 ML 워크플로우입니다. 각 단계를 이 저장소의 실제
파일/폴더와 연결해뒀습니다. 회의 때 "지금 몇 단계에 있고, 다음에 뭘 할지"
이야기할 때 이 문서를 기준으로 삼으면 됩니다.

## 전체 흐름

```
1. 문제 정의 (Problem Definition)
        ↓
2. 데이터 확보 및 이해 (Data Acquisition & Understanding)
        ↓
3. 데이터 품질 / 누수 검증 (Data Quality & Leakage Audit)
        ↓
4. 검증 전략 설계 (Validation Design)
        ↓
5. 베이스라인 재현 → 전처리 / 피처 엔지니어링 (Baseline → Preprocessing & FE)
        ↓
6. 모델링 / 학습 (Modeling & Training)
        ↓
7. 평가 / 오류 분석 (Evaluation & Error Analysis)
        ↓  ─────────────────────────────┐
        │   (가설이 틀렸거나 개선 여지가 있으면) │
        │   5~6단계로 돌아가 피처·모델 개선     │
        └─────────────────────────────↺
        ↓ (더 개선할 가설이 없거나 마감이 임박하면)
8. 최종 학습 / 제출 패키징 (Final Training & Deployment)
        ↓
9. 리더보드 결과 기록 / 회고 (Monitoring & Iteration)
        ↓
   (다음 회의에서 1~7단계 중 필요한 지점으로 재진입)
```

7→5/6 루프가 이 워크플로우의 핵심입니다. 대부분의 실제 작업 시간은 이
루프 안에서 쓰이고, 8~9단계는 "이번 제출 시점에 루프를 잠깐 멈추고
스냅샷을 남기는" 단계입니다.

---

## 1. 문제 정의 (Problem Definition)

- **목표**: 투구 직전까지 알 수 있는 정보만으로 `control_success`(제구 성공 확률) 예측
- **평가지표**: Brier Skill Score (`docs/data_description.md`, `CLAUDE.md` 참고)
- 이미 완료된 단계입니다. README.md / CLAUDE.md에 정리되어 있습니다.

## 2. 데이터 확보 및 이해 (Data Acquisition & Understanding)

- `data/` 폴더에 `train.csv`, `test.csv`, `sample_submission.csv`, `trackman_history.csv` 배치 (팀원 각자 `open.zip` 원본에서 압축 해제, `.gitignore`로 버전관리 제외됨)
- `docs/data_description.md` 정독 — 컬럼별 의미, `asof_*` 피처 정의, 금지 정보 목록
- EDA에서 확인할 것:
  - `control_success` 비율 (클래스 불균형 정도)
  - 시즌별(`season`) 분포 — 2019~2024, 2024가 내부 검증용
  - `asof_*` 피처들의 cold-start 결측 비율 (표본 수 `asof_*_n`이 작은 경우)
  - `trackman_history.csv`가 `train.csv`와 어떻게 연결될 수 있는지 (직접 조인 키는 없음 — `pitcher_id`/`batter_id` 등 간접 키 탐색 필요)
- 산출물: EDA 노트북 (예: `notebooks/eda/`, 아직 미생성 — 필요할 때 만들면 됨)

## 3. 데이터 품질 / 누수 검증 (Data Quality & Leakage Audit)

- `CLAUDE.md`의 "Competition Safety Rules" 12개 항목을 새 피처마다 재확인
- 체크 질문: "이 값이 실제 투구 시점 이전에 알 수 있는 정보인가?"
- `trackman_history.csv`를 쓸 경우, `game_date` 기준으로 **해당 투구 시점 이전** 로그만 사용하는지 반드시 검증
- test.csv의 다른 행을 이용한 집계/타겟인코딩/롤링 피처는 절대 금지 (규칙 위반 = 실격 사유)

## 4. 검증 전략 설계 (Validation Design)

- 기본 전략: `season 2019-2023` 학습 / `season 2024` 검증 (공식 베이스라인과 동일)
- 실제 평가 대상은 **2025 시즌**이므로, 검증 전략이 "미래 시즌 예측" 상황을 얼마나 잘 흉내 내는지가 핵심 기준
- 필요시 확장: 월별/롤링 검증, 여러 시즌 겹쳐서 안정성 확인 등 (단, 시간 순서를 깨는 랜덤 분할은 기본으로 쓰지 않음)

## 5. 베이스라인 재현 → 전처리 / 피처 엔지니어링

- **먼저** `notebooks/baseline/`을 그대로 실행해 동일한 검증 점수가 나오는지 확인 (기준점 확보)
- 이후 개선 방향:
  - `trackman_history.csv` 기반 피처 (구종별 rolling 구속/무브먼트, 투수 단위 요약 등) — 베이스라인은 이 파일을 아예 안 씀
  - `asof_*` cold-start 결측치 처리/smoothing 전략
  - 범주형 인코딩, 파생 변수 등
- 재사용 가능한 피처 함수는 `src/`에 작성 (노트북에만 인라인으로 두지 않기 — 팀원 간 재사용/리뷰가 어려워짐)

## 6. 모델링 / 학습 (Modeling & Training)

- 베이스라인: `RandomForestClassifier` (얕은 트리, `CLAUDE.md` "Official Baseline" 참고)
- 확장 후보: LightGBM, XGBoost (`requirements.txt`에 이미 추가됨)
- 시드 고정, 하이퍼파라미터 명시적으로 기록 (아래 실험 기록 참고)

## 7. 평가 / 오류 분석 (Evaluation & Error Analysis)

- Brier Score + 공식 스코어(`max(0, 100000*(1-brier/r(1-r)))`)로 검증셋 평가
- 오류 분석 예시: 어떤 상황(주자 상태, 카운트, cold-start 표본 부족 등)에서 예측이 특히 부정확한지
- **이 단계 결과는 반드시 `experiments/TEMPLATE.md` 양식으로 기록** → 그 관찰을 바탕으로 5~6단계로 돌아가 다음 가설 실행

## 8. 최종 학습 / 제출 패키징 (Final Training & Deployment)

이 대회에서 "배포"는 실서비스 배포가 아니라 **리더보드 제출**을 의미합니다.

1. 검증으로 방향을 확인한 설정으로 전체 데이터(2019~2024)로 재학습
2. `model/`에 모델 아티팩트 저장
3. `script.py` 작성/갱신 (추론 로직)
4. `submission/`에 `model/` + `script.py` + `requirements.txt`로 zip 패키징
5. 제출 전 체크리스트 (`CLAUDE.md` "Submission Constraints" 참고): NaN/inf 없음, 확률이 [0,1] 범위, `row_id` 순서/개수가 `sample_submission.csv`와 일치, 오프라인 상태에서 실행 확인, 추론 10분 이내

## 9. 리더보드 결과 기록 / 회고 (Monitoring & Iteration)

- 제출 직후 **`experiments/SUBMISSION_LOG.md`에 한 줄 기록** (날짜, 로컬 점수, 리더보드 점수, 무엇을 바꿨는지)
- 로컬 검증 점수와 리더보드 점수의 차이를 확인 — 차이가 크면 4단계(검증 전략) 자체를 재검토해야 한다는 신호
- 팀 회의에서 `SUBMISSION_LOG.md` + 관련 `experiments/` 기록을 보며 다음에 어떤 가설(피처/모델)을 우선순위로 시도할지 결정 → 1~7단계 중 필요한 지점으로 재진입
