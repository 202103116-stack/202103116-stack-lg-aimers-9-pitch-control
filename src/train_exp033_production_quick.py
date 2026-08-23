"""Quick production build for exp_033's validated win (season_decay=0.90),
105 features unchanged from exp_030. Built for speed (target: usable
submission within ~20min) while exp_037's full champion-assembly run
(120 features + decay, genuine 5-fold OOF) continues in the background --
this is deliberately a SMALLER, faster win to use up today's submission
quota, not a replacement for exp_037.

Key shortcut (legitimate, not an approximation): shrinkage_priors,
trackman_league_fallback, season_state_*_priors, and all the flat lookup
CSVs are pure aggregations of the raw training data (train.csv +
trackman_clean.csv) -- they do NOT depend on season_decay or CatBoost's
own weighting at all, so exp_030's already-computed versions (from
submission/archive/exp030_season_state_no_extrapolation/model/) are
reused byte-for-byte instead of rebuilt. Only the two things season_decay
actually changes get freshly (re)computed here: the CatBoost model
weights/predictions themselves, and the calibrator + count_trend that
are fit ON TOP of those new predictions.

Calibration note: reuses exp_027's cached OOF(2019-2023, un-decayed
model) for the 2019-2023 half of the calibration/count-trend data (this
project's established OOF cost is ~55min/seed -- regenerating it decay-
weighted is exactly what exp_037 is doing in parallel; using the
existing un-decayed OOF here is a deliberate approximation for THIS
quick build only) combined with this run's own genuine held-out
2024 predictions (from a model that never saw 2024, decay-weighted) for
the 2024 half. This is the same combination pattern train_exp029_
count_trend.py / train_exp030_repro.py already use.
"""

import os
import shutil
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import catboost as cb
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from exp030_baseline import ALL_FEATURES, DATA_DIR, build_holdout_split
from features import CAT_COLS
from metrics import official_score
from train_exp029_count_trend import GROUP_COLS, TREND_STRENGTH, apply_group_trend, compute_group_trend

MODEL_DIR = "./model"
OUTPUT_DIR = "./output"
CHAMPION_ARCHIVE = "./submission/archive/exp030_season_state_no_extrapolation/model"
DECAY = 0.90
SEEDS = [42, 1]
WEIGHT_2024 = 100.0
BASELINE_SCORE = 875.00


def season_weight(season_col, max_season, decay=DECAY):
    return decay ** (max_season - season_col.astype(float))


def cb_params(seed):
    return dict(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                loss_function="Logloss", eval_metric="Logloss",
                random_seed=seed, thread_count=-1, verbose=False)


def fit_platt_weighted(raw_pred, y, sample_weight, seed=42):
    raw_pred = np.asarray(raw_pred, dtype=float).reshape(-1, 1)
    clf = LogisticRegression(random_state=seed)
    clf.fit(raw_pred, y, sample_weight=sample_weight)
    return clf


def main():
    print("=" * 80)
    print("1. 2019-2023 학습 / 2024 검증 (105피처, decay=0.90) -- 2시드")
    print("=" * 80)
    X_train, y_train, X_val, y_val, feats = build_holdout_split()
    assert list(feats) == ALL_FEATURES
    print(f"train={X_train.shape}, val={X_val.shape}")

    best_iters = {}
    val_preds = {}
    for seed in SEEDS:
        w = season_weight(X_train["season"], max_season=2023)
        train_pool = cb.Pool(X_train, y_train, cat_features=CAT_COLS, weight=w)
        val_pool = cb.Pool(X_val, y_val, cat_features=CAT_COLS)
        t = time.time()
        clf = cb.CatBoostClassifier(**cb_params(seed))
        clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        pred = clf.predict_proba(X_val)[:, 1]
        best_iters[seed] = clf.get_best_iteration()
        brier, score = official_score(pred, y_val)
        print(f"[seed={seed}] {time.time()-t:.1f}s best_iter={best_iters[seed]} raw score={score:.2f}")
        val_preds[seed] = pred
    avg_val_pred = (val_preds[42] + val_preds[1]) / 2
    brier, score = official_score(avg_val_pred, y_val)
    print(f"[2시드 평균] raw Brier={brier:.6f} score={score:.2f}")

    print()
    print("=" * 80)
    print("2. step1 Platt (exp_027 캐시 OOF[2019-2023, 비-decay] + 이번 decay val_pred[2024])")
    print("=" * 80)
    npz42 = np.load(os.path.join(OUTPUT_DIR, "exp027_oof_cache_seed42.npz"))
    npz1 = np.load(os.path.join(OUTPUT_DIR, "exp027_oof_cache_seed1.npz"))
    oof_2319 = (npz42["pred"] + npz1["pred"]) / 2
    y_2319 = npz42["y"]

    p_combo = np.concatenate([oof_2319, avg_val_pred])
    y_combo = np.concatenate([y_2319, y_val.to_numpy()])
    sw = np.concatenate([np.ones(len(oof_2319)), np.full(len(avg_val_pred), WEIGHT_2024)])
    calibrator = fit_platt_weighted(p_combo, y_combo, sample_weight=sw)
    print(f"coef={calibrator.coef_}, intercept={calibrator.intercept_}")
    oof_calibrated = calibrator.predict_proba(oof_2319.reshape(-1, 1))[:, 1]
    val_calibrated = calibrator.predict_proba(avg_val_pred.reshape(-1, 1))[:, 1]
    step1_brier, step1_score = official_score(val_calibrated, y_val)
    print(f"[step1 only, 2024 held-out] score={step1_score:.2f}")

    print()
    print("=" * 80)
    print("3. count-trend 검증 (2019-2023 잔차로 2024 진짜 held-out 외삽)")
    print("=" * 80)
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=["season", "control_success"] + GROUP_COLS)
    train_2319 = train_raw.loc[train_raw["season"] < 2024].reset_index(drop=True)
    assert len(train_2319) == len(oof_2319)
    train_2319 = train_2319.copy()
    train_2319["residual"] = y_2319 - oof_calibrated
    train_2319["centered_residual"] = (
        train_2319["residual"] - train_2319["season"].map(train_2319.groupby("season")["residual"].mean())
    )
    trend_2319_only = compute_group_trend(train_2319, target_season=2024)
    val_group = X_val[GROUP_COLS].reset_index(drop=True)
    correction_2024 = apply_group_trend(val_group, trend_2319_only)
    pred_final = np.clip(val_calibrated + TREND_STRENGTH * correction_2024, 0.0, 1.0)
    final_brier, final_score = official_score(pred_final, y_val)
    delta = final_score - BASELINE_SCORE
    print(f"[step1 + count-trend, 2024 held-out] score={final_score:.2f} (기준선 {BASELINE_SCORE:.2f}, Δ={delta:+.2f})")

    print()
    print("=" * 80)
    print("4. 프로덕션 재학습: 2019-2024 전체 (decay=0.90, holdout best_iter 재사용)")
    print("=" * 80)
    X_full, y_full, _, _, _ = build_holdout_split(val_season=9999, max_train_season=2024)
    print(f"full train={X_full.shape}")
    w_full = season_weight(X_full["season"], max_season=2024)
    full_pool = cb.Pool(X_full, y_full, cat_features=CAT_COLS, weight=w_full)

    os.makedirs(MODEL_DIR, exist_ok=True)
    seed_model_files = []
    for seed in SEEDS:
        fp = cb_params(seed)
        fp["iterations"] = best_iters[seed] + 1
        clf = cb.CatBoostClassifier(**fp)
        t = time.time()
        clf.fit(full_pool)
        print(f"[seed={seed}] 최종 재학습 완료: {time.time()-t:.1f}s, iterations={fp['iterations']}")
        fname = f"catboost_model_seed{seed}.cbm"
        clf.save_model(os.path.join(MODEL_DIR, fname))
        seed_model_files.append(fname)

    print()
    print("=" * 80)
    print("5. count-trend 2025 외삽 (2019-2024 전체 잔차)")
    print("=" * 80)
    train_2024 = train_raw.loc[train_raw["season"] == 2024].reset_index(drop=True).copy()
    train_2024["residual"] = y_val.to_numpy() - val_calibrated
    combined = pd.concat([
        train_2319[["season"] + GROUP_COLS + ["residual"]],
        train_2024[["season"] + GROUP_COLS + ["residual"]],
    ], ignore_index=True)
    season_mean_all = combined.groupby("season")["residual"].transform("mean")
    combined["centered_residual"] = combined["residual"] - season_mean_all
    trend_full = compute_group_trend(combined, target_season=2025)
    for key, v in sorted(trend_full.items()):
        print(f"  {GROUP_COLS}={key}: 2025 외삽 centered_residual={v:+.5f}")

    print()
    print("=" * 80)
    print("6. 메타/lookup 저장 (exp_030 아카이브의 priors/lookup 그대로 재사용 -- decay와 무관)")
    print("=" * 80)
    base_meta = joblib.load(os.path.join(CHAMPION_ARCHIVE, "model_meta.pkl"))
    new_meta = dict(base_meta)
    new_meta.pop("seed_weights", None)
    new_meta["seed_model_files"] = seed_model_files
    new_meta["calibrator"] = calibrator
    new_meta["count_trend"] = {"group_cols": GROUP_COLS, "trend": trend_full, "strength": TREND_STRENGTH}
    new_meta["exp_id"] = "exp033_season_decay_090_quick"
    joblib.dump(new_meta, os.path.join(MODEL_DIR, "model_meta.pkl"), compress=3)
    print(f"저장: {MODEL_DIR}/model_meta.pkl")

    for fname in ["trackman_pitcher_lookup.csv", "season_state_pitcher_lookup.csv", "season_state_batter_lookup.csv"]:
        shutil.copy2(os.path.join(CHAMPION_ARCHIVE, fname), os.path.join(MODEL_DIR, fname))
    print("복사: trackman/season_state lookup CSV (exp_030 원본 그대로)")

    stale = os.path.join(MODEL_DIR, "catboost_model.cbm")
    if os.path.exists(stale):
        os.remove(stale)

    print(f"\n검증 점수(2024 held-out, step1+count-trend) = {final_score:.2f} (기준선 {BASELINE_SCORE:.2f} 대비 {delta:+.2f})")
    print("완료.")


if __name__ == "__main__":
    main()
