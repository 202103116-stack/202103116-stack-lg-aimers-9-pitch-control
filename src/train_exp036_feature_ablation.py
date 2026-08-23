"""exp_036 -- feature simplification search, deferred earlier in this
session for time cost, now run overnight per user request. Two-phase
greedy backward elimination over the 58 DERIVED feature columns of
exp_030's 105-feature set (the official 47 raw columns are protected --
exp_004 already established that removing "redundant-looking" raw
columns was pure loss, so that ground isn't re-tested here).

Phase 1: screen 10 concept groups individually (single-seed quick fit,
2019-2023/2024 holdout) -- for each, remove that group and measure the
raw score delta vs the full-105 baseline.
Phase 2: greedy sequential cumulative removal, safest group first (by
Phase 1 delta), stopping the moment cumulative removal drops more than
NOISE_THRESHOLD below the full baseline -- catches negative interactions
between groups that looked individually safe.
Final: genuine 5-fold x 2-seed OOF + exp_030's exact calibration chain
(step1 Platt, no extrapolation, + count-trend) on the winning reduced
feature set, GO/NO-GO vs 875.00 baseline (matching exp_037's gate),
production build + save if GO.

NOTE ON SHARED ./model/ DIRECTORY: writes the same filenames as every
other production script in this project (model_meta.pkl,
catboost_model_seed{42,1}.cbm, lookup CSVs). Do not run concurrently
with train_exp037_champion_assembly.py or train_exp031_production.py.
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

from exp030_baseline import ALL_FEATURES, DATA_DIR, build_holdout_split
from features import CAT_COLS
from metrics import official_score
from train_exp029_count_trend import GROUP_COLS, TREND_STRENGTH, apply_group_trend, compute_group_trend

MODEL_DIR = "./model"
OUTPUT_DIR = "./output"
CHAMPION_ARCHIVE = "./submission/archive/exp030_season_state_no_extrapolation/model"
NOISE_THRESHOLD = 2.0
WEIGHT_2024 = 100.0
BASELINE_SCORE = 875.00
N_FOLDS = 5
SEEDS = [42, 1]

FEATURE_GROUPS = {
    "count_pressure_state": [
        "count_diff", "count_total", "two_strike", "three_ball", "full_count",
        "late_inning", "score_margin_abs", "is_close_game", "runners_scoring_position",
        "three_ball_x_risp",
    ],
    "pitcher_batter_interaction": [
        "pitcher_minus_batter_success", "pitcher_middle_minus_success", "batter_middle_minus_success",
        "same_hand_matchup", "same_hand_x_breaking_rate", "same_hand_x_offspeed_rate",
    ],
    "pitcher_form_trend": [
        "pitcher_recent_form_delta", "pitcher_form_trend_isolated",
        "pitcher_middle_trend_isolated", "pitcher_form_volatility",
    ],
    "experience_log": ["pitcher_experience_log", "batter_experience_log"],
    "pitchmix_diversity": ["pitchmix_diversity"],
    "pressure_interaction": [
        "pitcher_success_under_pressure", "pitcher_reverse_under_pressure", "pitcher_success_x_risp",
    ],
    "shrinkage": [
        "shrunk_pitcher_success_rate", "shrunk_pitcher_reverse_rate", "shrunk_pitcher_middle_rate",
        "shrunk_pitcher_ball_rate", "shrunk_pitcher_strike_rate", "shrunk_batter_success_rate",
        "shrunk_batter_middle_rate", "shrunk_pitcher_fastball_rate", "shrunk_pitcher_breaking_rate",
        "shrunk_pitcher_offspeed_rate", "shrunk_pitcher_x_batter_success",
    ],
    "trackman_pooled": [
        "trackman_rel_speed_asof", "trackman_spin_rate_asof", "trackman_induced_vert_break_asof",
        "trackman_horz_break_asof", "trackman_extension_asof",
    ],
    "season_state_pitcher": [
        "pitcher_season_n", "pitcher_season_success_rate_k20", "pitcher_season_success_delta_career",
        "pitcher_season_reverse_rate_k20", "pitcher_season_reverse_delta_career",
        "pitcher_season_middle_rate_k20", "pitcher_season_middle_delta_career",
        "pitcher_season_ball_rate_k20", "pitcher_season_ball_delta_career",
        "pitcher_season_strike_rate_k20", "pitcher_season_strike_delta_career",
    ],
    "season_state_batter": [
        "batter_season_n", "batter_season_success_rate_k20", "batter_season_success_delta_career",
        "batter_season_middle_rate_k20", "batter_season_middle_delta_career",
    ],
}

_all_group_cols = sorted(c for cols in FEATURE_GROUPS.values() for c in cols)
_protected = sorted(c for c in ALL_FEATURES if c not in _all_group_cols)
assert len(_all_group_cols) == 58, len(_all_group_cols)
assert len(_protected) == 47, len(_protected)
assert set(_all_group_cols) | set(_protected) == set(ALL_FEATURES)


def cb_params(seed=42):
    return dict(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                loss_function="Logloss", eval_metric="Logloss",
                random_seed=seed, thread_count=-1, verbose=False)


def quick_score(X_train, y_train, X_val, y_val, feature_list, seed=42):
    cat_cols_present = [c for c in CAT_COLS if c in feature_list]
    train_pool = cb.Pool(X_train[feature_list], y_train, cat_features=cat_cols_present)
    val_pool = cb.Pool(X_val[feature_list], y_val, cat_features=cat_cols_present)
    clf = cb.CatBoostClassifier(**cb_params(seed))
    clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    pred = clf.predict_proba(X_val[feature_list])[:, 1]
    return official_score(pred, y_val)


def fit_platt_weighted(raw_pred, y, sample_weight, seed=42):
    raw_pred = np.asarray(raw_pred, dtype=float).reshape(-1, 1)
    clf = LogisticRegression(random_state=seed)
    clf.fit(raw_pred, y, sample_weight=sample_weight)
    return clf


def main():
    print("=" * 80)
    print("0. 데이터 로드 (2019-2023/2024 홀드아웃, exp_030의 105피처)")
    print("=" * 80)
    X_train, y_train, X_val, y_val, feats = build_holdout_split()
    assert list(feats) == ALL_FEATURES
    print(f"train={X_train.shape} val={X_val.shape}")
    print(f"피처 그룹: {[(k, len(v)) for k, v in FEATURE_GROUPS.items()]}")

    print()
    print("=" * 80)
    print("Phase 0. 전체 105피처 기준선 (single seed=42)")
    print("=" * 80)
    t = time.time()
    base_brier, base_score = quick_score(X_train, y_train, X_val, y_val, ALL_FEATURES)
    print(f"[105피처 전체] {time.time()-t:.1f}s raw Brier={base_brier:.6f} raw score={base_score:.2f}")

    print()
    print("=" * 80)
    print("Phase 1. 그룹단위 조(粗) 제거 스크리닝 (10회)")
    print("=" * 80)
    phase1_deltas = {}
    for name, cols in FEATURE_GROUPS.items():
        reduced = [c for c in ALL_FEATURES if c not in cols]
        t = time.time()
        brier, score = quick_score(X_train, y_train, X_val, y_val, reduced)
        delta = score - base_score
        phase1_deltas[name] = delta
        print(f"[- {name} ({len(cols)}개)] {time.time()-t:.1f}s score={score:.2f} Δ={delta:+.2f}"
              f" {'(안전, 제거 후보)' if delta >= -NOISE_THRESHOLD else '(유의미하게 손해, 보존)'}")

    print()
    print("=" * 80)
    print("Phase 2. 그리디 누적 제거 (안전한 것부터, 손해 나기 직전까지)")
    print("=" * 80)
    safe_order = sorted(
        [n for n, d in phase1_deltas.items() if d >= -NOISE_THRESHOLD],
        key=lambda n: -phase1_deltas[n],
    )
    print(f"Phase 1에서 안전 판정된 그룹 (안전한 순): {safe_order}")

    removed_so_far = []
    best_removed = []
    best_delta = 0.0
    for name in safe_order:
        trial_removed = removed_so_far + [name]
        drop_cols = [c for g in trial_removed for c in FEATURE_GROUPS[g]]
        reduced = [c for c in ALL_FEATURES if c not in drop_cols]
        t = time.time()
        brier, score = quick_score(X_train, y_train, X_val, y_val, reduced)
        delta = score - base_score
        print(f"[누적 제거 {trial_removed} ({len(drop_cols)}개 컬럼)] {time.time()-t:.1f}s score={score:.2f} Δ={delta:+.2f}")
        if delta >= -NOISE_THRESHOLD:
            removed_so_far = trial_removed
            best_removed = trial_removed
            best_delta = delta
        else:
            print(f"  -> {name} 추가 시 누적 손해가 노이즈 기준을 넘음, 여기서 중단")
            break

    final_drop_cols = [c for g in best_removed for c in FEATURE_GROUPS[g]]
    final_features = [c for c in ALL_FEATURES if c not in final_drop_cols]
    print()
    print(f"최종 간소화 피처셋: {len(final_features)}개 (제거: {best_removed}, {len(final_drop_cols)}개 컬럼, 스크리닝 Δ={best_delta:+.2f})")

    if not best_removed:
        print("\n안전하게 뺄 수 있는 그룹이 하나도 없음 -- 105개가 실제로는 과하지 않다는 결론.")
        print("완료 (제거 없음, 최종 확인/프로덕션 단계 생략).")
        return

    print()
    print("=" * 80)
    print("Final. 최종 확인: 진짜 5-fold x 2시드 OOF + exp_030 보정체인, GO/NO-GO")
    print("=" * 80)
    y_2319_arr = y_train.to_numpy()

    oof_cache = os.path.join(OUTPUT_DIR, "exp036_oof_cache_{seed}.npz")
    val_cache = os.path.join(OUTPUT_DIR, "exp036_val_pred_seed{seed}.npy")
    best_iter_path = os.path.join(OUTPUT_DIR, "exp036_best_iters.pkl")
    best_iters = joblib.load(best_iter_path) if os.path.exists(best_iter_path) else {}
    val_preds = {}
    for seed in SEEDS:
        vc = val_cache.format(seed=seed)
        if os.path.exists(vc) and seed in best_iters:
            val_preds[seed] = np.load(vc)
            print(f"[seed={seed}] val_pred 캐시 재사용 (best_iter={best_iters[seed]})")
        else:
            cat_cols_present = [c for c in CAT_COLS if c in final_features]
            train_pool = cb.Pool(X_train[final_features], y_train, cat_features=cat_cols_present)
            val_pool = cb.Pool(X_val[final_features], y_val, cat_features=cat_cols_present)
            t = time.time()
            clf = cb.CatBoostClassifier(**cb_params(seed))
            clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
            val_preds[seed] = clf.predict_proba(X_val[final_features])[:, 1]
            best_iters[seed] = clf.get_best_iteration()
            print(f"[seed={seed}] 홀드아웃 {time.time()-t:.1f}s best_iter={best_iters[seed]}")
            np.save(vc, val_preds[seed])
            joblib.dump(best_iters, best_iter_path)
    avg_val_pred = (val_preds[42] + val_preds[1]) / 2

    for seed in SEEDS:
        oc = oof_cache.format(seed=seed)
        if os.path.exists(oc):
            print(f"[seed={seed}] OOF 캐시 재사용")
            continue
        cat_cols_present = [c for c in CAT_COLS if c in final_features]
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_pred = np.zeros(len(X_train))
        t0 = time.time()
        for fold, (tr_idx, ho_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr = X_train[final_features].iloc[tr_idx], y_train.iloc[tr_idx]
            X_ho = X_train[final_features].iloc[ho_idx]
            pool = cb.Pool(X_tr, y_tr, cat_features=cat_cols_present)
            clf = cb.CatBoostClassifier(**cb_params(seed))
            clf.fit(pool, verbose=False)
            oof_pred[ho_idx] = clf.predict_proba(X_ho)[:, 1]
            print(f"  [seed={seed}] fold {fold+1}/{N_FOLDS} ({time.time()-t0:.1f}s)")
        np.savez(oc, pred=oof_pred, y=y_2319_arr)

    npz42 = np.load(oof_cache.format(seed=42))
    npz1 = np.load(oof_cache.format(seed=1))
    oof_2319 = (npz42["pred"] + npz1["pred"]) / 2
    y_2319 = npz42["y"]

    p_combo = np.concatenate([oof_2319, avg_val_pred])
    y_combo = np.concatenate([y_2319, y_val.to_numpy()])
    sw = np.concatenate([np.ones(len(oof_2319)), np.full(len(avg_val_pred), WEIGHT_2024)])
    calibrator = fit_platt_weighted(p_combo, y_combo, sample_weight=sw)
    oof_calibrated = calibrator.predict_proba(oof_2319.reshape(-1, 1))[:, 1]
    val_calibrated = calibrator.predict_proba(avg_val_pred.reshape(-1, 1))[:, 1]
    step1_brier, step1_score = official_score(val_calibrated, y_val)
    print(f"[step1 only] score={step1_score:.2f}")

    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=["season", "control_success"] + GROUP_COLS)
    train_2319_resid = train_raw.loc[train_raw["season"] < 2024].reset_index(drop=True).copy()
    assert len(train_2319_resid) == len(oof_2319)
    train_2319_resid["residual"] = y_2319 - oof_calibrated
    train_2319_resid["centered_residual"] = (
        train_2319_resid["residual"] - train_2319_resid["season"].map(train_2319_resid.groupby("season")["residual"].mean())
    )
    trend_2319_only = compute_group_trend(train_2319_resid, target_season=2024)
    val_group = X_val[GROUP_COLS].reset_index(drop=True)
    correction_2024 = apply_group_trend(val_group, trend_2319_only)
    pred_final = np.clip(val_calibrated + TREND_STRENGTH * correction_2024, 0.0, 1.0)
    final_brier, final_score = official_score(pred_final, y_val)
    delta_vs_baseline = final_score - BASELINE_SCORE
    print(f"[step1 + count-trend, 2024 held-out] score={final_score:.2f} (기준선 {BASELINE_SCORE:.2f}, Δ={delta_vs_baseline:+.2f})")

    go = delta_vs_baseline >= -NOISE_THRESHOLD
    print(f"\n{'GO' if go else 'NO-GO'}: {'노이즈 기준 내 -- 간소화 피처셋 채택, 프로덕션 진행' if go else '기준선보다 유의미하게 손해 -- 프로덕션 중단'}")
    if not go:
        print("완료 (NO-GO).")
        return

    print()
    print("=" * 80)
    print("프로덕션: 전체 2019-2024 재학습 + 저장")
    print("=" * 80)
    X_full, y_full, _, _, _ = build_holdout_split(val_season=9999, max_train_season=2024)
    os.makedirs(MODEL_DIR, exist_ok=True)
    cat_cols_present = [c for c in CAT_COLS if c in final_features]
    full_pool = cb.Pool(X_full[final_features], y_full, cat_features=cat_cols_present)

    seed_model_files = []
    for seed in SEEDS:
        fp = cb_params(seed)
        fp["iterations"] = best_iters[seed] + 1
        clf = cb.CatBoostClassifier(**fp)
        t = time.time()
        clf.fit(full_pool)
        print(f"[seed={seed}] 최종 재학습: {time.time()-t:.1f}s iterations={fp['iterations']}")
        fname = f"catboost_model_seed{seed}.cbm"
        clf.save_model(os.path.join(MODEL_DIR, fname))
        seed_model_files.append(fname)

    train_2024_resid = X_val[GROUP_COLS].reset_index(drop=True).copy()
    train_2024_resid["season"] = 2024
    train_2024_resid["residual"] = y_val.to_numpy() - val_calibrated
    combined = pd.concat([
        train_2319_resid[["season"] + GROUP_COLS + ["residual"]],
        train_2024_resid[["season"] + GROUP_COLS + ["residual"]],
    ], ignore_index=True)
    season_mean_all = combined.groupby("season")["residual"].transform("mean")
    combined["centered_residual"] = combined["residual"] - season_mean_all
    trend_full = compute_group_trend(combined, target_season=2025)

    base_meta = joblib.load(os.path.join(CHAMPION_ARCHIVE, "model_meta.pkl"))
    new_meta = dict(base_meta)
    new_meta.pop("seed_weights", None)
    new_meta["all_features"] = final_features
    new_meta["seed_model_files"] = seed_model_files
    new_meta["calibrator"] = calibrator
    new_meta["count_trend"] = {"group_cols": GROUP_COLS, "trend": trend_full, "strength": TREND_STRENGTH}
    new_meta["exp_id"] = "exp036_feature_ablation"
    joblib.dump(new_meta, os.path.join(MODEL_DIR, "model_meta.pkl"), compress=3)
    print(f"저장: {MODEL_DIR}/model_meta.pkl (n_features={len(final_features)}, 제거={best_removed})")

    import shutil
    for fname in ["trackman_pitcher_lookup.csv", "season_state_pitcher_lookup.csv", "season_state_batter_lookup.csv"]:
        shutil.copy2(os.path.join(CHAMPION_ARCHIVE, fname), os.path.join(MODEL_DIR, fname))

    stale = os.path.join(MODEL_DIR, "catboost_model.cbm")
    if os.path.exists(stale):
        os.remove(stale)

    print(f"\n최종 검증 점수 = {final_score:.2f} (기준선 {BASELINE_SCORE:.2f} 대비 {delta_vs_baseline:+.2f}, 피처 105->{len(final_features)})")
    print("완료.")


if __name__ == "__main__":
    main()
