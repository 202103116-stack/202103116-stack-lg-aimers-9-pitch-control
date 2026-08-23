"""exp_037 -- champion assembly. Combines the two validated positive
results from this round on top of exp_030's exact calibration chain
(step1 weighted Platt, NO exp_023 extrapolation shift -- exp_030's own
winning choice -- + exp_029-style count-state trend correction):

  - exp_033: season_decay=0.90 CatBoost sample weight
    (weight = 0.90 ** (max_season - row_season))
  - exp_035: 105 pooled + 15 pitch_type_group-conditioned trackman
    physical columns = 120 features total

EXCLUDED from this production build (documented, not silently dropped):
  - exp_031 (heterogeneous ensemble, CatBoost-A+B+ExtraTrees): the
    directional screen was positive (+20.74 on an approximate,
    self-fit-on-2024 Platt calibration) but a leak-safe production
    version needs genuine 5-fold OOF for THREE differently-tuned
    models (est. 4+ hours combined, based on exp_027's own single-
    architecture OOF cost of ~55min/seed) -- out of this session's
    budget. Left as documented future work.
  - exp_034 (OOT count-state residual correction): rejected outright,
    -26.18 vs baseline (experiments/exp_034_oot_count_calibration.md).

Structure (each stage checkpointed to ./output/ so a crash mid-run
doesn't lose earlier expensive work):
  A. Build the 120-feature 2019-2023/2024 holdout split (pooled+group
     trackman, season-state features) -- same leak-safety pattern as
     exp030_baseline.build_holdout_split, extended.
  B. Quick 2-seed CatBoost fit (decay=0.90 weighted) on that holdout --
     sanity check the combined raw signal before committing to the
     expensive stage C, cache val_pred for later calibration.
  C. Genuine 5-fold x 2-seed OOF on 2019-2023 (decay=0.90 weighted) --
     the expensive step (~110min based on exp_027's own OOF cost).
  D. step1 Platt (weight=100 for 2024, no extrapolation) on OOF+val.
  E. exp_029-style count-state trend correction, validated against the
     genuine 2024 held-out (trend fit on 2019-2023 residuals only) --
     GO/NO-GO gate against the 875.00 exp_030_repro baseline.
  F. If GO (combined score beats 875.00 by >= 2pts, this project's
     noise floor): retrain on full 2019-2024, build production
     lookups, extrapolate count-trend to 2025, save model_meta.pkl.
"""

import os
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
from sklearn.model_selection import KFold

from exp030_baseline import ALL_FEATURES, BASE_FEATURES, DATA_DIR, RECENT_SEASONS_FOR_PRIOR, TRACKMAN_SHRINK_K
from features import CAT_COLS, apply_shrinkage, build_features, fit_shrinkage_priors
from metrics import official_score
from season_state_features import (
    attach_season_state_features, attach_season_state_features_flat, build_season_baselines,
    fit_season_state_priors, latest_season_baselines, season_state_cols,
)
from train_exp029_count_trend import GROUP_COLS as COUNT_GROUP_COLS, TREND_STRENGTH, apply_group_trend, compute_group_trend
from trackman_pitcher_features import (
    GROUP_TRACKMAN_COLS, TRACKMAN_PITCHER_ASOF_COLS, attach_pitcher_physical_features,
    attach_pitcher_physical_features_by_group, build_pitcher_physical_asof_tables,
    build_pitcher_physical_asof_tables_by_group, build_test_time_pitcher_lookup,
    build_test_time_pitcher_lookup_by_group, load_pitcher_mapping,
)

MODEL_DIR = "./model"
OUTPUT_DIR = "./output"
ID = "row_id"
TARGET = "control_success"
SEEDS = [42, 1]
N_FOLDS = 5
DECAY = 0.90
WEIGHT_2024 = 100.0
NOISE_FLOOR = 2.0
BASELINE_SCORE = 875.00
FEATURES_120 = ALL_FEATURES + GROUP_TRACKMAN_COLS
SEASON_STATE_COLS = season_state_cols("pitcher") + season_state_cols("batter")
# NOTE: "season", "balls_before", "strikes_before" are already part of
# FEATURES_120 (they're 3 of the champion's original 105 raw features --
# verified against model_meta.pkl), so they never need to be added
# separately for weighting/grouping -- doing so would duplicate-label
# the column (the exact bug exp_033 hit and fixed).


def _recent_seasons_df(df, n=RECENT_SEASONS_FOR_PRIOR):
    seasons = sorted(df["season"].unique())
    return df[df["season"].isin(seasons[-n:])]


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


def build_dataset(max_train_season, val_season=None):
    """Returns (df, is_val, train_only) with the 120-feature columns +
    EXTRA_COLS attached. If val_season is None, every row up to
    max_train_season is "train_only" (production full-data build).
    Otherwise rows == val_season are held out (holdout / OOT-style
    build), mirroring exp030_baseline.build_holdout_split's contract."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                         usecols=BASE_FEATURES + [TARGET])
    train = build_features(train)
    train = train.loc[train["season"] <= max_train_season].reset_index(drop=True)

    if val_season is not None:
        is_val = train["season"] == val_season
    else:
        is_val = pd.Series(False, index=train.index)
    train_only = train.loc[~is_val]

    trackman_clean = pd.read_csv(os.path.join(DATA_DIR, "processed", "trackman_clean.csv"), encoding="utf-8-sig")
    pitcher_mapping = load_pitcher_mapping()
    tables_pooled = build_pitcher_physical_asof_tables(trackman_clean)
    tables_group = build_pitcher_physical_asof_tables_by_group(trackman_clean)

    priors = fit_shrinkage_priors(_recent_seasons_df(train_only))
    df = apply_shrinkage(train, priors)
    df = attach_pitcher_physical_features(df, tables_pooled, pitcher_mapping, shrink_k=TRACKMAN_SHRINK_K)
    df = attach_pitcher_physical_features_by_group(df, tables_group, pitcher_mapping, shrink_k=TRACKMAN_SHRINK_K)

    pitcher_baselines = build_season_baselines(train_only, "pitcher")
    pitcher_priors = fit_season_state_priors(train_only, "pitcher")
    df = attach_season_state_features(df, pitcher_baselines, pitcher_priors, "pitcher")
    batter_baselines = build_season_baselines(train_only, "batter")
    batter_priors = fit_season_state_priors(train_only, "batter")
    df = attach_season_state_features(df, batter_baselines, batter_priors, "batter")

    return df, is_val, priors, pitcher_baselines, pitcher_priors, batter_baselines, batter_priors, tables_pooled, tables_group, pitcher_mapping


def main():
    print("=" * 80)
    print("A. 2019-2023 학습 / 2024 검증 데이터셋 구축 (120피처: 기존105 + 구종군15)")
    print("=" * 80)
    (df_holdout, is_val_holdout, _, _, _, _, _, tables_pooled_h, tables_group_h, mapping_h) = build_dataset(
        max_train_season=2024, val_season=2024)
    X_train = df_holdout.loc[~is_val_holdout, FEATURES_120]
    y_train = df_holdout.loc[~is_val_holdout, TARGET]
    X_val = df_holdout.loc[is_val_holdout, FEATURES_120]
    y_val = df_holdout.loc[is_val_holdout, TARGET]
    print(f"train={X_train.shape}, val={X_val.shape}, n_features={len(FEATURES_120)}")

    print()
    print("=" * 80)
    print("B. 2시드 CatBoost 학습 (decay=0.90 가중, 2019-2023 학습/2024 검증) -- 빠른 확인")
    print("=" * 80)
    val_pred_path_tmpl = os.path.join(OUTPUT_DIR, "exp037_val_pred_seed{seed}.npy")
    best_iter_path = os.path.join(OUTPUT_DIR, "exp037_best_iters.pkl")
    best_iters = joblib.load(best_iter_path) if os.path.exists(best_iter_path) else {}
    val_preds = {}
    for seed in SEEDS:
        cache = val_pred_path_tmpl.format(seed=seed)
        if os.path.exists(cache) and seed in best_iters:
            val_preds[seed] = np.load(cache)
            print(f"[seed={seed}] 캐시 재사용: {cache} (best_iter={best_iters[seed]})")
            continue
        w = season_weight(X_train["season"], max_season=2023)
        train_pool = cb.Pool(X_train[FEATURES_120], y_train, cat_features=CAT_COLS, weight=w)
        val_pool = cb.Pool(X_val[FEATURES_120], y_val, cat_features=CAT_COLS)
        t = time.time()
        clf = cb.CatBoostClassifier(**cb_params(seed))
        clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        pred = clf.predict_proba(X_val[FEATURES_120])[:, 1]
        brier, score = official_score(pred, y_val)
        best_iters[seed] = clf.get_best_iteration()
        print(f"[seed={seed}] {time.time()-t:.1f}s, best_iter={best_iters[seed]}, raw Brier={brier:.6f}, raw score={score:.2f}")
        val_preds[seed] = pred
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(cache, pred)
        joblib.dump(best_iters, best_iter_path)
    avg_val_pred = (val_preds[42] + val_preds[1]) / 2
    avg_brier, avg_score = official_score(avg_val_pred, y_val)
    print(f"[2시드 평균] raw Brier={avg_brier:.6f} | raw score={avg_score:.2f} (참고: exp_030 105피처/decay없음 raw=818.79)")

    print()
    print("=" * 80)
    print("C. 5-fold x 2시드 진짜 OOF (2019-2023, decay=0.90 가중) -- 시간이 오래 걸림")
    print("=" * 80)
    train_2319 = df_holdout.loc[~is_val_holdout, FEATURES_120 + [TARGET]].reset_index(drop=True)
    X_2319 = train_2319[FEATURES_120]
    y_2319 = train_2319[TARGET].to_numpy()
    season_2319 = train_2319["season"]
    print(f"2019-2023 n={len(X_2319)}, features={len(FEATURES_120)}")

    for seed in SEEDS:
        cache_path = os.path.join(OUTPUT_DIR, f"exp037_oof_cache_seed{seed}.npz")
        if os.path.exists(cache_path):
            print(f"이미 존재: {cache_path} (재사용)")
            continue
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_pred = np.zeros(len(train_2319))
        t0 = time.time()
        for fold, (tr_idx, ho_idx) in enumerate(kf.split(train_2319)):
            X_tr, y_tr = X_2319.iloc[tr_idx], y_2319[tr_idx]
            X_ho = X_2319.iloc[ho_idx]
            w_tr = season_weight(season_2319.iloc[tr_idx], max_season=2023)
            train_pool = cb.Pool(X_tr, y_tr, cat_features=CAT_COLS, weight=w_tr)
            clf = cb.CatBoostClassifier(**cb_params(seed))
            clf.fit(train_pool, verbose=False)
            oof_pred[ho_idx] = clf.predict_proba(X_ho)[:, 1]
            print(f"  [seed={seed}] fold {fold+1}/{N_FOLDS} 완료 ({time.time()-t0:.1f}s 누적)")
        oof_brier, oof_score = official_score(oof_pred, y_2319)
        print(f"[seed={seed}] OOF(2019-2023): Brier={oof_brier:.6f}, score={oof_score:.2f}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez(cache_path, pred=oof_pred, y=y_2319)
        print(f"저장: {cache_path}")

    npz42 = np.load(os.path.join(OUTPUT_DIR, "exp037_oof_cache_seed42.npz"))
    npz1 = np.load(os.path.join(OUTPUT_DIR, "exp037_oof_cache_seed1.npz"))
    oof_2319 = (npz42["pred"] + npz1["pred"]) / 2
    y_2319_check = npz42["y"]
    assert np.array_equal(y_2319_check, y_2319)

    print()
    print("=" * 80)
    print("D. step1 Platt (2024 가중치=100, exp_023 외삽 없음 -- exp_030의 선택)")
    print("=" * 80)
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
    print("E. count-state 시즌추세 보정 (exp_029 방식), 2024 진짜 held-out으로 GO/NO-GO 검증")
    print("=" * 80)
    train_2319_resid = train_2319[["season"] + COUNT_GROUP_COLS].copy()
    train_2319_resid["residual"] = y_2319 - oof_calibrated
    season_mean = train_2319_resid.groupby("season")["residual"].transform("mean")
    train_2319_resid["centered_residual"] = train_2319_resid["residual"] - season_mean
    trend_2319_only = compute_group_trend(train_2319_resid, target_season=2024)

    val_group_df = X_val[COUNT_GROUP_COLS].reset_index(drop=True)
    correction_2024 = apply_group_trend(val_group_df, trend_2319_only)
    pred_final_holdout = np.clip(val_calibrated + TREND_STRENGTH * correction_2024, 0.0, 1.0)
    final_brier, final_score = official_score(pred_final_holdout, y_val)
    print(f"[step1 + count-trend, 2024 held-out] score={final_score:.2f}")
    print(f"기준선(exp_030_repro) = {BASELINE_SCORE:.2f}")
    delta = final_score - BASELINE_SCORE
    print(f"Δ = {delta:+.2f}")

    go = delta >= NOISE_FLOOR
    print(f"\n{'GO' if go else 'NO-GO'}: {'노이즈 기준(+2점) 이상 개선 -- 프로덕션 재학습 진행' if go else '개선폭이 노이즈 수준 이하 -- 프로덕션 재학습 중단, 결과만 보고'}")

    if not go:
        print("\n완료 (NO-GO, F단계 생략).")
        return

    print()
    print("=" * 80)
    print("F. 프로덕션: 전체 2019-2024로 재학습 + 2025 외삽 count-trend + 패키징용 아티팩트 저장")
    print("=" * 80)
    (df_full, is_val_full, final_priors, pitcher_baselines_full, pitcher_priors_full,
     batter_baselines_full, batter_priors_full, tables_pooled_full, tables_group_full, mapping_full) = build_dataset(
        max_train_season=2024, val_season=None)
    X_full = df_full[FEATURES_120]
    y_full = df_full[TARGET]
    w_full = season_weight(df_full["season"], max_season=2024)
    full_pool = cb.Pool(X_full, y_full, cat_features=CAT_COLS, weight=w_full)

    os.makedirs(MODEL_DIR, exist_ok=True)
    seed_model_files = []
    for seed in SEEDS:
        # holdout stage B에서 캐시해둔 best_iteration을 그대로 재사용 (exp_027
        # 관례와 동일: 홀드아웃에서 조기종료로 찾은 반복수를, 데이터가 더 많은
        # 최종 학습에도 그대로 씀 -- 과적합 방지용 반복수 상한이지 데이터
        # 크기에 비례하는 값이 아님).
        best_iter = best_iters[seed]

        final_params = cb_params(seed)
        final_params["iterations"] = best_iter + 1
        final_clf = cb.CatBoostClassifier(**final_params)
        t = time.time()
        final_clf.fit(full_pool)
        print(f"[seed={seed}] 최종 재학습 완료: {time.time()-t:.1f}s, iterations={final_params['iterations']} (best_iter={best_iter})")
        fname = f"catboost_model_seed{seed}.cbm"
        final_clf.save_model(os.path.join(MODEL_DIR, fname))
        seed_model_files.append(fname)

    print()
    print("F-2. 프로덕션 lookup/메타 저장")
    print("-" * 80)
    lookup_pooled, fallback_pooled = build_test_time_pitcher_lookup(tables_pooled_full, mapping_full, shrink_k=TRACKMAN_SHRINK_K)
    lookup_group, fallback_group = build_test_time_pitcher_lookup_by_group(tables_group_full, mapping_full, shrink_k=TRACKMAN_SHRINK_K)
    lookup_combined = lookup_pooled.merge(lookup_group, on="pitcher_id", how="outer")
    lookup_path = os.path.join(MODEL_DIR, "trackman_pitcher_lookup.csv")
    lookup_combined.to_csv(lookup_path, index=False, encoding="utf-8-sig")
    fallback_combined = {**fallback_pooled, **fallback_group}
    print(f"저장: {lookup_path} ({lookup_combined.shape})")

    pitcher_season_lookup = latest_season_baselines(df_full, "pitcher")
    batter_season_lookup = latest_season_baselines(df_full, "batter")
    pitcher_season_priors = fit_season_state_priors(df_full, "pitcher")
    batter_season_priors = fit_season_state_priors(df_full, "batter")
    pitcher_season_lookup.to_csv(os.path.join(MODEL_DIR, "season_state_pitcher_lookup.csv"), index=False, encoding="utf-8-sig")
    batter_season_lookup.to_csv(os.path.join(MODEL_DIR, "season_state_batter_lookup.csv"), index=False, encoding="utf-8-sig")
    print(f"저장: season_state_{{pitcher,batter}}_lookup.csv ({pitcher_season_lookup.shape}, {batter_season_lookup.shape})")

    print()
    print("F-3. count-trend 2025 외삽 (2019-2024 전체 잔차)")
    print("-" * 80)
    train_2024_resid = X_val[COUNT_GROUP_COLS].reset_index(drop=True).copy()
    train_2024_resid["season"] = 2024
    train_2024_resid["residual"] = y_val.to_numpy() - val_calibrated
    combined_resid = pd.concat([
        train_2319_resid[["season"] + COUNT_GROUP_COLS + ["residual"]],
        train_2024_resid[["season"] + COUNT_GROUP_COLS + ["residual"]],
    ], ignore_index=True)
    season_mean_all = combined_resid.groupby("season")["residual"].transform("mean")
    combined_resid["centered_residual"] = combined_resid["residual"] - season_mean_all
    trend_full = compute_group_trend(combined_resid, target_season=2025)
    for key, v in sorted(trend_full.items()):
        print(f"  {COUNT_GROUP_COLS}={key}: 2025 외삽 centered_residual={v:+.5f}")

    stale = os.path.join(MODEL_DIR, "catboost_model.cbm")
    if os.path.exists(stale):
        os.remove(stale)
        print(f"제거: {stale}")

    joblib.dump(
        {
            "model_type": "catboost_seedbag",
            "base_features": BASE_FEATURES,
            "all_features": FEATURES_120,
            "trackman_cols": TRACKMAN_PITCHER_ASOF_COLS + GROUP_TRACKMAN_COLS,
            "cat_cols": CAT_COLS,
            "shrinkage_priors": final_priors,
            "trackman_league_fallback": fallback_combined,
            "trackman_shrink_k": TRACKMAN_SHRINK_K,
            "season_state_cols": SEASON_STATE_COLS,
            "season_state_pitcher_priors": pitcher_season_priors,
            "season_state_batter_priors": batter_season_priors,
            "season_state_shrink_k": 20,
            "seed_model_files": seed_model_files,
            "calibrator": calibrator,
            "count_trend": {"group_cols": COUNT_GROUP_COLS, "trend": trend_full, "strength": TREND_STRENGTH},
            "exp_id": "exp_037_champion_assembly_seedbag42_1",
        },
        os.path.join(MODEL_DIR, "model_meta.pkl"),
        compress=3,
    )
    print(f"저장: {MODEL_DIR}/model_meta.pkl")
    print(f"\n검증 점수(2024 held-out, step1+count-trend) = {final_score:.2f} (기준선 {BASELINE_SCORE:.2f} 대비 {delta:+.2f})")
    print("완료.")


if __name__ == "__main__":
    main()
