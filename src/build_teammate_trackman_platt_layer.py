"""Adds our own recency-focused Platt calibration layer on top of the
just-built teammate+trackman20 package, per user request to make sure
this final candidate includes every technique that showed a real effect
this session: Platt regression calibration + season-drift adaptation
(our exp_022 lineage) ON TOP OF teammate's own count-state correction
(group_offsets + trend_offsets, kept exactly as their framework fit it
-- "허찬의 count-state별 보정").

Data source for the held-out 2024 predictions: the earlier `study` run
(experiments/SCRATCH_trackman20) trained a model on season<2024 with the
IDENTICAL model spec, features, and fixed random seeds as builder.py's
internal `calibration_components` (same protocol=holdout_2024, same
triple_count_trend_ols_cpuonly spec) -- so its held-out predictions are
a valid, numerically equivalent proxy for what builder.py computed
internally (and never persisted) to fit group_offsets/trend_offsets.
We reconstruct the FULLY teammate-corrected held-out prediction (raw
blend + static per-member affine + group_offset + trend_offset, exactly
mirroring their generated script.py's main()) and fit a Platt regressor
on that vs the real 2024 labels -- this is a proper closed-form fit
(unlike their static affine constants, which were never fit, just
manually chosen), and specifically targets the most recent labeled
season, which is what "시즌 드리프트 대응" means in this project's own
history (exp_022 lineage) -- NOT the exp_023 extrapolation-to-2025 trick,
which this project already proved harmful (LB -45.61) and is
deliberately NOT reproduced here.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import gzip
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "./src")
from metrics import official_score

sys.path.insert(0, r"c:\Users\yeob0\lgAimers9_pitch-control\_teammate_repo\experiment_workspace\src")
from aimers_exp.runner import _learn_centered_group_trend_offsets  # noqa: E402

STUDY_PRED_PATH = (
    r"c:\Users\yeob0\lgAimers9_pitch-control\_teammate_repo\experiment_workspace\runs"
    r"\scratch_trackman20\20260823T034807626142Z_f6419ad9e8a7\trackman20\predictions_2024.csv.gz"
)
TRAIN_TRACKMAN_PATH = r"c:\Users\yeob0\lgAimers9_pitch-control\_teammate_repo\experiment_workspace\data_trackman\train.csv"
MODEL_JOBLIB_PATH = r"c:\Users\yeob0\lgAimers9_pitch-control\_teammate_repo\experiment_workspace\output\final_trackman20\package\model\model.joblib"
OUT_CALIBRATOR_PATH = "./output/teammate_trackman20_platt_calibrator.pkl"


def map_offset(frame, columns, values):
    if not columns:
        return np.zeros(len(frame), dtype=float)
    if len(columns) == 1:
        mapped = frame[columns[0]].map(values)
    else:
        keys = pd.MultiIndex.from_frame(frame.loc[:, columns])
        mapped = pd.Series(values.reindex(keys).to_numpy(), index=frame.index)
    return mapped.fillna(0.0).to_numpy(dtype=float)


def main():
    print("=" * 80)
    print("1. study 단계의 2024 진짜 held-out 예측(raw+static affine 반영됨) 로드")
    print("=" * 80)
    with gzip.open(STUDY_PRED_PATH, "rt") as f:
        pred_df = pd.read_csv(f)
    print(f"held-out 2024 n={len(pred_df)}")

    bundle = joblib.load(MODEL_JOBLIB_PATH)
    spec = bundle["model_spec"]
    group_offsets = bundle["group_offsets"]
    trend_offsets = bundle["trend_offsets"]
    group_columns = bundle["group_columns"]
    trend_group_columns = bundle["trend_group_columns"]
    trend_strength = float(spec.get("trend_strength", 0.0))
    print(f"group_columns={group_columns}, trend_group_columns={trend_group_columns}, trend_strength={trend_strength}")

    print()
    print("=" * 80)
    print("2. group_offset(2024 진짜 OOT) + trend_offset(주의: 프로덕션용은 2019-2024 전체로")
    print("   적합돼 2024 자체가 이미 포함됨 -- 2024 검증엔 못 씀. 2019-2023만으로 별도")
    print("   재적합해서 진짜 held-out 검증용 trend를 따로 만든다, 우리 exp_029/037/038의")
    print("   trend_2319_only 관례와 동일)")
    print("=" * 80)
    train_cols = ["row_id", "season"] + sorted(set(group_columns) | set(trend_group_columns))
    train_all = pd.read_csv(TRAIN_TRACKMAN_PATH, encoding="utf-8-sig", usecols=train_cols)
    train_pre2024 = train_all.loc[train_all["season"] < 2024]
    train2024 = train_all.loc[train_all["season"] == 2024]

    # 팀원 프로덕션 trend_offsets(model.joblib에 저장된 것)는 2019-2024 전체로 적합돼
    # 2025 외삽용이다 -- 2024를 그 적합에 이미 썼으므로 2024 검증에 그대로 쓰면 순환
    # 검증이 된다(실측 결과 score가 923.19로 오히려 하락, 이 순환성이 원인으로 추정).
    # 우리 프로젝트의 exp_029 관례 그대로, 2019-2023만으로 별도 재적합해서 진짜
    # held-out(2024)을 검증한다 -- 이 clean 버전만 검증에 쓰고, 프로덕션 패키지 자체는
    # 원래 팀원 코드가 만든 trend_offsets(2019-2024 전체, 2025 외삽용)를 그대로 둔다.
    target_col = pd.read_csv(TRAIN_TRACKMAN_PATH, encoding="utf-8-sig", usecols=["row_id", "control_success"])
    train_pre2024 = train_pre2024.merge(target_col, on="row_id", how="left")
    trend_offsets_clean = _learn_centered_group_trend_offsets(
        train_pre2024,
        train_pre2024["control_success"],
        trend_group_columns,
        shrinkage=float(spec.get("trend_shrinkage", 10_000.0)),
        method=str(spec.get("trend_method", "wls")),
    )
    print(f"clean trend_offsets(2019-2023만): {trend_offsets_clean.to_dict()}")
    print(f"프로덕션 trend_offsets(2019-2024, 2025외삽용, 참고용): {trend_offsets.to_dict()}")

    merged = pred_df.merge(train2024, on="row_id", how="left", validate="one_to_one")
    assert merged[group_columns + trend_group_columns].isna().sum().sum() == 0

    group_correction = map_offset(merged, group_columns, group_offsets)
    trend_correction_clean = trend_strength * map_offset(merged, trend_group_columns, trend_offsets_clean)
    y = merged["target"].to_numpy()

    raw_pred = merged["probability"].to_numpy()
    with_group = np.clip(raw_pred + group_correction, 0.0, 1.0)
    fully_corrected = np.clip(raw_pred + group_correction + trend_correction_clean, 0.0, 1.0)

    brier_raw, score_raw = official_score(raw_pred, y)
    brier_group, score_group = official_score(with_group, y)
    brier_full, score_full = official_score(fully_corrected, y)
    print(f"[study raw(affine만)] score={score_raw:.2f}")
    print(f"[+ group_offset(진짜 2024 OOT)] score={score_group:.2f}")
    print(f"[+ trend_offset(2019-2023만으로 적합한 clean 버전)] score={score_full:.2f}")

    print()
    print("=" * 80)
    print("3. 우리 방식 Platt 회귀보정 추가 (2024 진짜 held-out, 시즌드리프트 대응)")
    print("=" * 80)
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(fully_corrected.reshape(-1, 1), y)
    platt_calibrated = calibrator.predict_proba(fully_corrected.reshape(-1, 1))[:, 1]
    brier_platt, score_platt = official_score(platt_calibrated, y)
    print(f"coef={calibrator.coef_}, intercept={calibrator.intercept_}")
    print(f"[+ Platt 회귀보정] score={score_platt:.2f}")

    print()
    print("=" * 80)
    print("결과 요약")
    print("=" * 80)
    print(f"  raw(affine만)                              = {score_raw:.2f}")
    print(f"  + 팀원 count-state 보정(group+trend)        = {score_full:.2f}")
    print(f"  + 우리 Platt 회귀보정(최종)                 = {score_platt:.2f}")
    print(f"  Δ(Platt 보정 단독 기여)                      = {score_platt - score_full:+.2f}")

    os.makedirs("./output", exist_ok=True)
    joblib.dump(calibrator, OUT_CALIBRATOR_PATH)
    print(f"\n저장: {OUT_CALIBRATOR_PATH}")
    print("완료.")


if __name__ == "__main__":
    main()
