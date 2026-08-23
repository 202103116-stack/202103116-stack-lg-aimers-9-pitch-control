"""Final, self-consistent calibration recipe for the teammate+trackman20
model, after discovering two things via careful validation (see
experiments/exp_039_teammate_trackman_graft.md):

1. bundle["group_offsets"]/bundle["trend_offsets"] (saved inside
   output/final_trackman20/package/model/model.joblib) were fit against
   builder.py's own internal `calibration_components` model -- a
   DIFFERENT raw-prediction baseline than this script's `study` run
   predictions, even though both are nominally "trained on season<2024,
   same spec". Reusing bundle's offsets against study's raw predictions
   produced an invalid, mismatched comparison (looked like the
   correction HURT trackman20 specifically, -19.68) -- not a genuine
   trackman-interaction problem.
2. Recomputing group_offsets SELF-CONSISTENTLY (fit directly from the
   SAME raw predictions they'll be applied to) restores the expected
   positive effect, and is methodologically equivalent to teammate's own
   "OOT residual from a <2024-trained proxy model" design -- just using
   our own proxy (the `study` run's calibration-split model) instead of
   builder.py's internal (inaccessible) one.
3. trend_offset consistently HURT in every clean, self-consistent test
   (both for baseline_78 and trackman20) -- dropped entirely.
4. Our own Platt regression (fit on the group-corrected 2024 held-out
   predictions) adds a further, real improvement on top.

Final validated local scores (2024 genuine held-out, self-consistent):
  baseline_78 best (raw+group+Platt)   = 947.10
  trackman20  best (raw+group+Platt)   = 950.19  <- winner, +3.09

This script computes and saves the PRODUCTION versions of group_offsets
and the Platt calibrator (fit on the full study-2024 held-out set) for
use when patching the final script.py.
"""

import gzip
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "./src")
from metrics import official_score

STUDY_PRED_PATH = (
    r"c:\Users\yeob0\lgAimers9_pitch-control\_teammate_repo\experiment_workspace\runs"
    r"\scratch_trackman20\20260823T034807626142Z_f6419ad9e8a7\trackman20\predictions_2024.csv.gz"
)
TRAIN_TRACKMAN_PATH = r"c:\Users\yeob0\lgAimers9_pitch-control\_teammate_repo\experiment_workspace\data_trackman\train.csv"
GROUP_COLS = ["balls_before", "strikes_before"]
GROUP_SHRINKAGE = 500.0
OUT_DIR = "./output"


def map_offset(frame, columns, values):
    keys = pd.MultiIndex.from_frame(frame.loc[:, columns])
    return pd.Series(values.reindex(keys).to_numpy(), index=frame.index).fillna(0.0).to_numpy(dtype=float)


def main():
    with gzip.open(STUDY_PRED_PATH, "rt") as f:
        pred_df = pd.read_csv(f)
    train_all = pd.read_csv(TRAIN_TRACKMAN_PATH, encoding="utf-8-sig", usecols=["row_id", "season"] + GROUP_COLS)
    train2024 = train_all.loc[train_all["season"] == 2024]
    merged = pred_df.merge(train2024, on="row_id", how="left", validate="one_to_one")
    y = merged["target"].to_numpy()
    raw = merged["probability"].to_numpy()

    residual = y - raw
    residual = residual - residual.mean()
    tmp = merged[GROUP_COLS].copy()
    tmp["__r"] = residual
    grp = tmp.groupby(GROUP_COLS, observed=True)["__r"].agg(["sum", "count"])
    group_offsets = grp["sum"] / (grp["count"] + GROUP_SHRINKAGE)
    group_corr = map_offset(merged, GROUP_COLS, group_offsets)
    with_group = np.clip(raw + group_corr, 0.0, 1.0)

    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(with_group.reshape(-1, 1), y)
    final_pred = calibrator.predict_proba(with_group.reshape(-1, 1))[:, 1]

    brier_raw, score_raw = official_score(raw, y)
    brier_final, score_final = official_score(final_pred, y)
    print(f"raw = {score_raw:.2f}")
    print(f"raw + group_offset + Platt (최종) = {score_final:.2f}")
    print(f"coef={calibrator.coef_}, intercept={calibrator.intercept_}")
    print(f"group_offsets: {group_offsets.to_dict()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(
        {"group_columns": GROUP_COLS, "group_offsets": group_offsets, "calibrator": calibrator},
        os.path.join(OUT_DIR, "teammate_trackman20_final_calibration.pkl"),
    )
    print(f"\n저장: {OUT_DIR}/teammate_trackman20_final_calibration.pkl")
    print("완료.")


if __name__ == "__main__":
    main()
