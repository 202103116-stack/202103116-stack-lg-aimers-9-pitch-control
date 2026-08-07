# LG Aimers 9기 Phase 2 - 투구 제구 성공 확률 예측

## Project Objective

This repository is for the LG Aimers 9기 Phase 2 competition on DACON:
"투구 제구 성공 확률 예측 AI 온라인 해커톤"
(https://dacon.io/competitions/official/236743)

The task is to predict, for each pitch in the test set:

P(control_success = 1)

using only information available before the pitch is thrown. The
prediction must be a probability between 0 and 1.

Per the official baseline notebook (`notebooks/baseline/baseline_train.ipynb`):
`train.csv` covers seasons 2019-2024, and the real (hidden) evaluation
data is the **2025 season**.

---

## Ground Truth Sources

When making decisions, use the following priority:

1. DACON official competition description, evaluation, rules, notices
2. `docs/data_description.md` (official data dictionary)
3. Official distributed dataset (`data/`)
4. Official baseline notebooks (`notebooks/baseline/`) and
   `references/baseline_example/` (script.py, requirements.txt, rf.pkl)
5. LG Twins / KBO domain knowledge
6. General machine-learning knowledge

Do not invent assumptions that are not supported by official material.
If something is not confirmed anywhere in sources 1-4, say so explicitly
instead of presenting it as fact.

Clearly distinguish:
- verified fact
- experimental result
- hypothesis
- suggestion

---

## Competition Safety Rules

NEVER violate these rules (see `docs/data_description.md` §5-6 for the
official source).

1. Do not use external datasets.
2. Do not use external APIs (e.g. OpenAI, Gemini) for the submitted model or inference.
3. Only use models with publicly available weights under a license that
   permits at least non-commercial use (MIT, Apache 2.0, etc).
4. Only information available before the pitch may be used.
5. Each test row is an independent prediction target.
6. Never use another test row to construct features for a test row
   (no cross-row aggregation, frequency stats, target encoding, or
   rolling/expanding features computed over `test.csv`).
7. Never compute statistics from the whole test distribution, and never
   calibrate predictions using test-set statistics.
8. Fit all preprocessing (imputers, encoders, scalers) only on training data.
9. `asof_*` columns provided in the official data ARE allowed — they are
   already computed strictly from information available before each pitch.
10. `trackman_history.csv` may be used for feature engineering, but never
    in a way that leaks information from after the evaluation timestamp,
    and 2025 Trackman data does not exist / must not be assumed.
11. The final submission must run fully offline.
12. Do not modify official raw data in `data/`.

If any feature or approach may violate these rules, STOP and explicitly
warn the user before implementing it.

---

## Official Evaluation

Target: `control_success`

Brier Score:
`mean((prediction - target) ** 2)`

Official score (Brier Skill Score):
```
max(0, 100000 * (1 - BrierScore / (r * (1 - r))))
```
where `r` is the mean `control_success` rate of the evaluated dataset.
(Verified against both the DACON rules page and
`notebooks/baseline/baseline_train.ipynb` §5.)

Always report, for every experiment:
- Brier Score
- official local score
- validation split used
- prediction mean
- prediction min/max

---

## Validation Policy

The hidden evaluation set is the **2025 season** (confirmed in the
official baseline notebook). The default validation must therefore be
season-based, not random:

- Train: 2019-2023
- Validation: 2024 (held out)

This matches the official baseline exactly. Do not use a random split as
the primary validation. Later experiments may use rolling/expanding
time-based validation (e.g. walk-forward by season or month), but must
still respect chronological order.

---

## Official Baseline

Source: `notebooks/baseline/baseline_train.ipynb` +
`notebooks/baseline/baseline_inference.ipynb` (verified line-by-line).

- Features = all columns in `test.csv` except `row_id` (47 features: 3
  categorical + 44 numeric)
- Target = `control_success`
- Categorical columns: `top_bottom`, `game_type`, `base_state`
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`
  for categorical features
- `SimpleImputer(strategy="median")` for numeric features
- Both wrapped in a single `ColumnTransformer` inside an sklearn `Pipeline`
- `RandomForestClassifier(n_estimators=100, max_depth=10,
  min_samples_leaf=200, n_jobs=-1, random_state=42)`
- Validation: train on seasons 2019-2023, validate on season 2024
- Final model retrained on the full 2019-2024 `train.csv`
- Model serialized with `joblib.dump(model, "./model/rf.pkl", compress=3)`
- `trackman_history.csv` is explicitly NOT used by the baseline — it is
  left as an opportunity for feature engineering

Do not silently change the baseline during baseline reproduction.

---

## Required ML Workflow

Full description with per-stage detail and the file/folder each stage
maps to: `docs/WORKFLOW.md`. Short version:

1. Problem definition
2. Data acquisition & understanding
3. Data quality / leakage audit
4. Validation design
5. Baseline reproduction, then preprocessing / feature engineering
6. Modeling & training
7. Evaluation & error analysis
   → loop back to 5/6 with a new hypothesis (this is where most iteration happens)
8. Final training & deployment (= full retrain + submission packaging,
   not production deployment)
9. Monitoring & iteration (= log the leaderboard result, compare to
   local validation, decide the next hypothesis)

Do not jump directly to complex models. Reproduce the official baseline
first (step 5) before changing anything.

---

## Coding Rules

Python only.

Prefer:
- vectorized pandas / numpy
- reusable functions
- deterministic seeds
- explicit feature lists
- logging
- assertions

Avoid:
- unnecessary `.apply`
- repeated large dataframe copies
- repeated merges
- test-time fitting
- hidden notebook state
- hard-coded user-specific absolute paths (e.g. no `C:\Users\...`)

All executable code should use repository-relative paths (see
`DATA_DIR = "./data"` pattern in the baseline notebooks).

---

## Repository Map

```
.
├── data/                         # competition data (gitignored, populate locally)
├── docs/
│   ├── data_description.md       # official data dictionary
│   └── WORKFLOW.md               # full 9-stage ML workflow, mapped to this repo
├── notebooks/
│   └── baseline/                 # official baseline notebooks (do not modify)
│       ├── baseline_train.ipynb
│       └── baseline_inference.ipynb
├── references/
│   └── baseline_example/         # official working submission example (do not modify)
│       ├── script.py
│       ├── requirements.txt
│       └── rf.pkl                # gitignored (binary)
├── src/                          # reusable feature engineering / training / packaging scripts
├── experiments/                  # TEMPLATE.md, SUBMISSION_LOG.md, and exp_*.md records (see "Experiment Discipline")
├── model/                        # trained model artifacts (gitignored)
├── output/                       # local prediction/submission csv output (gitignored)
├── submission/
│   ├── script.py                 # inference entry point (tracked)
│   ├── requirements.txt          # minimal deps beyond pre-installed (tracked)
│   ├── model/                    # copy of model artifacts for packaging (gitignored)
│   ├── submit.zip                # always the latest build, upload this to DACON (gitignored)
│   └── archive/                  # per-experiment copies, e.g. exp001_lgbm_submit.zip (gitignored)
├── README.md
└── requirements.txt
```

Build/rebuild `submission/submit.zip` with `python src/package_submission.py <label>`
— do not hand-assemble the zip. The DACON code guide confirms the
uploaded zip's filename is free-form (no spaces/Korean), only the
internal structure (`model/`, `script.py`, `requirements.txt` at the
zip root) must match exactly — that's what the script guarantees.

Do not modify files in `notebooks/baseline/` or `references/`.
Do not commit raw competition data or model binaries (already enforced
via `.gitignore`).

---

## Experiment Discipline

Use `experiments/TEMPLATE.md` for every experiment (copy it to
`experiments/exp_XXX_short-name.md`). Before every experiment, state:
- experiment ID
- hypothesis
- baseline
- single intended change
- validation method
- leakage risk

After every experiment, record:
- features
- model
- hyperparameters
- seed
- Brier Score
- official local score
- runtime
- prediction statistics
- observations
- next hypothesis

Never claim an improvement without an actual executed comparison.

Every leaderboard submission also gets one row in
`experiments/SUBMISSION_LOG.md` (date, local score, leaderboard
score/rank, what changed, next plan) — this is what the team uses in
meetings to see progress at a glance.

---

## Submission Constraints

평가 서버 사양 — 출처: https://dacon.io/competitions/official/236743/overview/evaluation (직접 확인 완료, 2026-08-07):
- OS: Ubuntu 22.04.5 LTS
- Python: 3.11.15
- CUDA: 12.8 이상
- GPU: NVIDIA L4 (VRAM 22.4GiB)
- CPU: 6 vCPU
- RAM: 28GB
- 인터넷 접속: 비활성화 (패키지 설치 외 외부 서버 연결/다운로드 불가)
- Inference time <= 10 minutes (evaluation set: 245,789 rows)
- Package installation <= 10 minutes
- Submission zip <= 10GB, <= 32GB uncompressed
- Language: Python only (`.py` or `.ipynb`, UTF-8)

사전 설치된 주요 Python 라이브러리 (버전 호환이 이미 맞춰져 있으므로,
`requirements.txt`에는 가급적 이 버전과 다르게 재지정하지 않는 것을 권장):
```
torch==2.7.1+cu128       pandas==2.0.3           numpy==1.26.4
scipy==1.15.3            scikit-learn==1.8.0     joblib==1.5.3
threadpoolctl==3.6.0     narwhals==2.21.2        transformers==4.46.3
accelerate==1.9.0        sentencepiece==0.1.99   regex==2023.12.25
tqdm==4.66.4              loguru==0.7.2           pyyaml==6.0.1
rich==13.7.1
```

사전 설치된 시스템 패키지:
```
git, build-essential, python3.11, python3.11-dev, python3.11-venv,
python3-pip, libffi-dev, libblas3, liblapack3, libomp-dev, tzdata,
unzip, p7zip-full, gfortran, libatlas-base-dev, default-jre-headless,
cmake, pkg-config, ninja-build, libgl1, libglib2.0-0
```

LightGBM/XGBoost는 사전 설치 목록에 없으므로 `requirements.txt`로
설치해야 함 (10분 제한 내 설치되는지 확인 필요, `build-essential`/`cmake`/
`gfortran`은 이미 있어 컴파일 필요한 패키지도 대체로 문제없음).

로컬 개발 환경: `.venv`는 Python 3.11.9(시스템에 병행 설치된 버전)로
구성되어 있고, pandas/numpy/scikit-learn/joblib은 위 사전 설치 버전과
동일하게 맞춰 놓았음 (`requirements.txt` 참고). LightGBM/XGBoost는
사전 설치 목록에 없어 별도 설치.

LightGBM 사용 시 주의: `LGBMClassifier`(sklearn 래퍼)를 joblib/pickle로
저장했다가 다른 프로세스에서 로드하면 이 환경에서 네이티브 크래시
(access violation)가 재현됨 (`experiments/exp_001_*.md` 참고). 반드시
`booster_.save_model(path)` / `lgb.Booster(model_file=path)`(LightGBM
자체 텍스트 포맷)으로 저장/로드할 것 — `src/train_lgbm.py`,
`submission/script.py`가 이 패턴을 따름.

Expected submitted archive layout — `src/package_submission.py`가
이 구조로 `submission/submit.zip`을 생성함:
```
submit.zip
├── model/           # trained model artifact(s)
├── script.py        # inference entry point, executed by the server
└── requirements.txt
```

The script must create: `output/submission.csv`

Before packaging, verify:
- no NaN
- no inf
- probabilities in [0, 1]
- row count matches `sample_submission.csv`
- `row_id` order matches `sample_submission.csv`
- test data is never modified
- inference runs from a clean process

---

## Working Style With Claude

Before changing code:
1. inspect relevant files
2. explain what will change
3. identify leakage / rule risks
4. make the smallest necessary change
5. run verification
6. summarize results

Do not rewrite large portions of the repository unnecessarily.

When unsure about a competition rule: do not assume. Ask the user to
verify against official DACON material.
