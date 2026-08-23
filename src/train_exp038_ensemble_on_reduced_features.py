"""exp_038 -- combine this overnight round's two strongest NON-decay
local wins: exp_031's heterogeneous ensemble (CatBoost-A + CatBoost-B +
ExtraTrees, structure only) applied on top of exp_036's simplified
98-feature set (105 minus pitchmix_diversity/pitcher_form_trend/
experience_log -- confirmed noise-level in exp_036's screening).
Deliberately excludes exp_033's season_decay=0.90: the only submitted
candidate carrying that component (exp_033_season_decay_090_quick)
regressed on the real leaderboard despite a positive local holdout
signal (see experiments/SUBMISSION_LOG.md, 2026-08-22 entry) -- both
exp_031 and exp_036 individually used NEITHER decay NOR extrapolation,
so this combination avoids the one mechanism with an actual observed
LB failure this session.

Unlike train_exp031_production.py, CatBoost-A's OOF cannot be reused
from the exp_027 cache here (that cache is for the 105-feature
architecture; this run uses 98 features) -- all three members need
genuine fresh 5-fold OOF. This is the most expensive remaining
experiment in this round (~3.5-4h estimated).

NOTE ON SHARED ./model/ DIRECTORY: same filename contract as every
other production script here. Run sequentially only.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ensemble import blend_predict, coarse_fine_blend_search
from exp030_baseline import DATA_DIR, build_holdout_split
from features import CAT_COLS
from metrics import official_score
from model_factory import fit_cat_ordinal_encoder
from train_exp029_count_trend import GROUP_COLS, TREND_STRENGTH, apply_group_trend, compute_group_trend

MODEL_DIR = "./model"
OUTPUT_DIR = "./output"
CHAMPION_ARCHIVE = "./submission/archive/exp030_season_state_no_extrapolation/model"
REDUCED_FEATURES_SOURCE = "./submission/archive/exp036_feature_ablation/model/model_meta.pkl"
SEEDS_A = [42, 1]
SEED_B = 42
SEED_ET = 42
N_FOLDS = 5
WEIGHT_2024 = 100.0
BASELINE_SCORE = 875.00
NOISE_FLOOR = 2.0

CATBOOST_B_PARAMS = dict(iterations=600, learning_rate=0.025, depth=7, l2_leaf_reg=3.0,
                          bootstrap_type="Bayesian", bagging_temperature=1.0, border_count=128)
ET_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=20, max_features=0.7, n_jobs=-1)


def cb_params_a(seed):
    return dict(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                loss_function="Logloss", eval_metric="Logloss",
                random_seed=seed, thread_count=-1, verbose=False)


def cb_params_b(seed):
    p = dict(CATBOOST_B_PARAMS)
    p.update(loss_function="Logloss", eval_metric="Logloss", random_seed=seed, thread_count=-1, verbose=False)
    return p


def fit_platt_weighted(raw_pred, y, sample_weight, seed=42):
    raw_pred = np.asarray(raw_pred, dtype=float).reshape(-1, 1)
    clf = LogisticRegression(random_state=seed)
    clf.fit(raw_pred, y, sample_weight=sample_weight)
    return clf


def oof_catboost(X, y, cat_cols, params_fn, seed, cache_path):
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"OOF 캐시 재사용: {cache_path}")
        return d["pred"]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(X))
    t0 = time.time()
    for fold, (tr_idx, ho_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_ho = X.iloc[ho_idx]
        pool = cb.Pool(X_tr, y_tr, cat_features=cat_cols)
        clf = cb.CatBoostClassifier(**params_fn(seed))
        clf.fit(pool, verbose=False)
        oof_pred[ho_idx] = clf.predict_proba(X_ho)[:, 1]
        print(f"  fold {fold+1}/{N_FOLDS} ({time.time()-t0:.1f}s)")
    np.savez(cache_path, pred=oof_pred, y=y.to_numpy())
    return oof_pred


def main():
    print("=" * 80)
    print("A. 축소 피처셋(exp_036) 로드 + 2019-2023/2024 홀드아웃")
    print("=" * 80)
    reduced_meta = joblib.load(REDUCED_FEATURES_SOURCE)
    features = reduced_meta["all_features"]
    print(f"피처 수: {len(features)} (exp_036에서 105->98로 간소화된 셋)")

    X_train_full, y_train, X_val_full, y_val, all105 = build_holdout_split()
    X_train = X_train_full[features]
    X_val = X_val_full[features]
    cat_cols_present = [c for c in CAT_COLS if c in features]
    print(f"train={X_train.shape} val={X_val.shape}")

    print()
    print("=" * 80)
    print("B. CatBoost-A: 98피처 신규 5-fold x 2시드 OOF + 홀드아웃 val_pred")
    print("=" * 80)
    oof_a_by_seed = {}
    val_a_by_seed = {}
    best_iters_a = {}
    best_iter_a_path = os.path.join(OUTPUT_DIR, "exp038_best_iters_A.pkl")
    if os.path.exists(best_iter_a_path):
        best_iters_a = joblib.load(best_iter_a_path)
    for seed in SEEDS_A:
        oof_a_by_seed[seed] = oof_catboost(
            X_train, y_train, cat_cols_present, cb_params_a, seed,
            os.path.join(OUTPUT_DIR, f"exp038_oof_cache_catboostA_seed{seed}.npz"))
        vc = os.path.join(OUTPUT_DIR, f"exp038_val_pred_catboostA_seed{seed}.npy")
        if os.path.exists(vc) and seed in best_iters_a:
            val_a_by_seed[seed] = np.load(vc)
            print(f"[CatBoost-A seed={seed}] val_pred 캐시 재사용 (best_iter={best_iters_a[seed]})")
        else:
            train_pool = cb.Pool(X_train, y_train, cat_features=cat_cols_present)
            val_pool = cb.Pool(X_val, y_val, cat_features=cat_cols_present)
            t = time.time()
            clf = cb.CatBoostClassifier(**cb_params_a(seed))
            clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
            val_a_by_seed[seed] = clf.predict_proba(X_val)[:, 1]
            best_iters_a[seed] = clf.get_best_iteration()
            print(f"[CatBoost-A seed={seed}] 홀드아웃 {time.time()-t:.1f}s best_iter={best_iters_a[seed]}")
            np.save(vc, val_a_by_seed[seed])
            joblib.dump(best_iters_a, best_iter_a_path)
    oof_a = (oof_a_by_seed[42] + oof_a_by_seed[1]) / 2
    val_a = (val_a_by_seed[42] + val_a_by_seed[1]) / 2
    npz_a42 = np.load(os.path.join(OUTPUT_DIR, "exp038_oof_cache_catboostA_seed42.npz"))
    y_2319 = npz_a42["y"]
    brier, score = official_score(val_a, y_val)
    print(f"[CatBoost-A raw val, 2시드평균] score={score:.2f}")

    print()
    print("=" * 80)
    print("C. CatBoost-B: 5-fold OOF(seed=42) + 홀드아웃 val_pred")
    print("=" * 80)
    oof_b = oof_catboost(X_train, y_train, cat_cols_present, cb_params_b, SEED_B,
                          os.path.join(OUTPUT_DIR, f"exp038_oof_cache_catboostB_seed{SEED_B}.npz"))
    val_b_cache = os.path.join(OUTPUT_DIR, f"exp038_val_pred_catboostB_seed{SEED_B}.npy")
    best_iter_b_path = os.path.join(OUTPUT_DIR, "exp038_best_iter_catboostB.pkl")
    if os.path.exists(val_b_cache) and os.path.exists(best_iter_b_path):
        val_b = np.load(val_b_cache)
        best_iter_b = joblib.load(best_iter_b_path)
        print(f"캐시 재사용 (best_iter={best_iter_b})")
    else:
        pool = cb.Pool(X_train, y_train, cat_features=cat_cols_present)
        val_pool = cb.Pool(X_val, y_val, cat_features=cat_cols_present)
        t = time.time()
        clf = cb.CatBoostClassifier(**cb_params_b(SEED_B))
        clf.fit(pool, eval_set=val_pool, early_stopping_rounds=100)
        val_b = clf.predict_proba(X_val)[:, 1]
        best_iter_b = clf.get_best_iteration()
        print(f"[CatBoost-B holdout] {time.time()-t:.1f}s best_iter={best_iter_b}")
        np.save(val_b_cache, val_b)
        joblib.dump(best_iter_b, best_iter_b_path)
    brier, score = official_score(val_b, y_val)
    print(f"[CatBoost-B raw val] score={score:.2f}")

    print()
    print("=" * 80)
    print("D. ExtraTrees: 5-fold OOF(seed=42) + 홀드아웃 val_pred")
    print("=" * 80)
    encoder = fit_cat_ordinal_encoder(X_train, cat_cols_present)
    X_train_enc = X_train.copy()
    X_train_enc[cat_cols_present] = encoder.transform(X_train[cat_cols_present])
    X_val_enc = X_val.copy()
    X_val_enc[cat_cols_present] = encoder.transform(X_val[cat_cols_present])

    oof_cache_et = os.path.join(OUTPUT_DIR, f"exp038_oof_cache_extratrees_seed{SEED_ET}.npz")
    if os.path.exists(oof_cache_et):
        oof_et = np.load(oof_cache_et)["pred"]
        print("캐시 재사용")
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_ET)
        oof_et = np.zeros(len(X_train_enc))
        t0 = time.time()
        for fold, (tr_idx, ho_idx) in enumerate(kf.split(X_train_enc)):
            X_tr, y_tr = X_train_enc.iloc[tr_idx], y_train.iloc[tr_idx]
            X_ho = X_train_enc.iloc[ho_idx]
            imputer = SimpleImputer(strategy="median")
            X_tr_imp = imputer.fit_transform(X_tr)
            clf = ExtraTreesClassifier(random_state=SEED_ET, **ET_PARAMS)
            clf.fit(X_tr_imp, y_tr)
            X_ho_imp = imputer.transform(X_ho)
            oof_et[ho_idx] = clf.predict_proba(X_ho_imp)[:, 1]
            print(f"  [ExtraTrees] fold {fold+1}/{N_FOLDS} ({time.time()-t0:.1f}s)")
        np.savez(oof_cache_et, pred=oof_et, y=y_train.to_numpy())
    oof_brier, oof_score = official_score(oof_et, y_train)
    print(f"[ExtraTrees OOF] score={oof_score:.2f}")

    val_et_cache = os.path.join(OUTPUT_DIR, f"exp038_val_pred_extratrees_seed{SEED_ET}.npy")
    if os.path.exists(val_et_cache):
        val_et = np.load(val_et_cache)
        print("캐시 재사용")
    else:
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_enc)
        t = time.time()
        clf = ExtraTreesClassifier(random_state=SEED_ET, **ET_PARAMS)
        clf.fit(X_train_imp, y_train)
        X_val_imp = imputer.transform(X_val_enc)
        val_et = clf.predict_proba(X_val_imp)[:, 1]
        print(f"[ExtraTrees holdout] {time.time()-t:.1f}s")
        np.save(val_et_cache, val_et)
    brier, score = official_score(val_et, y_val)
    print(f"[ExtraTrees raw val] score={score:.2f}")

    print()
    print("=" * 80)
    print("E. 블렌드 가중치 탐색 (2024 val raw)")
    print("=" * 80)
    pred_dict_val = {"CatBoost-A": val_a, "CatBoost-B": val_b, "ExtraTrees": val_et}
    candidates = list(pred_dict_val.keys())
    best_weights, best_brier, best_score = coarse_fine_blend_search(pred_dict_val, y_val.to_numpy(), candidates)
    print(f"최적 가중치: {best_weights}  [블렌드 raw] score={best_score:.2f}")

    print()
    print("=" * 80)
    print("F. step1 Platt (OOF 블렌드 + val 블렌드, 동일 가중치)")
    print("=" * 80)
    pred_dict_oof = {"CatBoost-A": oof_a, "CatBoost-B": oof_b, "ExtraTrees": oof_et}
    oof_blend = blend_predict(pred_dict_oof, best_weights)
    val_blend = blend_predict(pred_dict_val, best_weights)

    p_combo = np.concatenate([oof_blend, val_blend])
    y_combo = np.concatenate([y_2319, y_val.to_numpy()])
    sw = np.concatenate([np.ones(len(oof_blend)), np.full(len(val_blend), WEIGHT_2024)])
    calibrator = fit_platt_weighted(p_combo, y_combo, sample_weight=sw)
    oof_calibrated = calibrator.predict_proba(oof_blend.reshape(-1, 1))[:, 1]
    val_calibrated = calibrator.predict_proba(val_blend.reshape(-1, 1))[:, 1]
    step1_brier, step1_score = official_score(val_calibrated, y_val)
    print(f"[step1 only] score={step1_score:.2f}")

    print()
    print("=" * 80)
    print("G. count-trend 검증 (진짜 2024 held-out) -- GO/NO-GO")
    print("=" * 80)
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                             usecols=["season", "control_success"] + GROUP_COLS)
    train_2319 = train_raw.loc[train_raw["season"] < 2024].reset_index(drop=True)
    assert len(train_2319) == len(oof_blend)
    train_2319 = train_2319.copy()
    train_2319["residual"] = y_2319 - oof_calibrated
    train_2319["centered_residual"] = (
        train_2319["residual"] - train_2319["season"].map(train_2319.groupby("season")["residual"].mean())
    )
    trend_2319_only = compute_group_trend(train_2319, target_season=2024)
    val_group = X_val_full[GROUP_COLS].reset_index(drop=True)
    correction_2024 = apply_group_trend(val_group, trend_2319_only)
    pred_final = np.clip(val_calibrated + TREND_STRENGTH * correction_2024, 0.0, 1.0)
    final_brier, final_score = official_score(pred_final, y_val)
    delta = final_score - BASELINE_SCORE
    print(f"[step1 + count-trend, 2024 held-out] score={final_score:.2f} (기준선 {BASELINE_SCORE:.2f}, Δ={delta:+.2f})")

    go = delta >= NOISE_FLOOR
    print(f"\n{'GO' if go else 'NO-GO'}")
    if not go:
        print("완료 (NO-GO, H단계 생략).")
        return

    print()
    print("=" * 80)
    print("H. 프로덕션: 전체 2019-2024 재학습 + 저장")
    print("=" * 80)
    X_full_full, y_full, _, _, _ = build_holdout_split(val_season=9999, max_train_season=2024)
    X_full = X_full_full[features]
    os.makedirs(MODEL_DIR, exist_ok=True)

    catboost_a_files = []
    for seed in SEEDS_A:
        fp = cb_params_a(seed)
        fp["iterations"] = best_iters_a[seed] + 1
        full_pool = cb.Pool(X_full, y_full, cat_features=cat_cols_present)
        clf = cb.CatBoostClassifier(**fp)
        t = time.time()
        clf.fit(full_pool)
        print(f"[CatBoost-A seed={seed} 최종] {time.time()-t:.1f}s iterations={fp['iterations']}")
        fname = f"catboost_A_seed{seed}.cbm"
        clf.save_model(os.path.join(MODEL_DIR, fname))
        catboost_a_files.append(fname)

    fp = cb_params_b(SEED_B)
    fp["iterations"] = best_iter_b + 1
    full_pool = cb.Pool(X_full, y_full, cat_features=cat_cols_present)
    clf_b = cb.CatBoostClassifier(**fp)
    t = time.time()
    clf_b.fit(full_pool)
    print(f"[CatBoost-B 최종] {time.time()-t:.1f}s iterations={fp['iterations']}")
    catboost_b_file = "catboost_B_final.cbm"
    clf_b.save_model(os.path.join(MODEL_DIR, catboost_b_file))

    encoder_full = fit_cat_ordinal_encoder(X_full, cat_cols_present)
    X_full_enc = X_full.copy()
    X_full_enc[cat_cols_present] = encoder_full.transform(X_full[cat_cols_present])
    imputer_full = SimpleImputer(strategy="median")
    X_full_imp = imputer_full.fit_transform(X_full_enc)
    t = time.time()
    clf_et = ExtraTreesClassifier(random_state=SEED_ET, **ET_PARAMS)
    clf_et.fit(X_full_imp, y_full)
    print(f"[ExtraTrees 최종] {time.time()-t:.1f}s")
    et_pipe = Pipeline([("impute", imputer_full), ("clf", clf_et)])
    et_bundle_file = "extratrees_bundle.pkl"
    joblib.dump({"encoder": encoder_full, "pipe": et_pipe}, os.path.join(MODEL_DIR, et_bundle_file), compress=3)

    print()
    print("count-trend 2025 외삽")
    train_2024 = train_raw.loc[train_raw["season"] == 2024].reset_index(drop=True).copy()
    train_2024["residual"] = y_val.to_numpy() - val_calibrated
    combined = pd.concat([
        train_2319[["season"] + GROUP_COLS + ["residual"]],
        train_2024[["season"] + GROUP_COLS + ["residual"]],
    ], ignore_index=True)
    season_mean_all = combined.groupby("season")["residual"].transform("mean")
    combined["centered_residual"] = combined["residual"] - season_mean_all
    trend_full = compute_group_trend(combined, target_season=2025)

    base_meta = joblib.load(os.path.join(CHAMPION_ARCHIVE, "model_meta.pkl"))
    weight_a = best_weights["CatBoost-A"]
    members = (
        [{"type": "catboost", "file": f, "weight": weight_a / len(SEEDS_A)} for f in catboost_a_files]
        + [{"type": "catboost", "file": catboost_b_file, "weight": best_weights["CatBoost-B"]}]
        + [{"type": "extra_trees", "file": et_bundle_file, "weight": best_weights["ExtraTrees"]}]
    )
    new_meta = {
        "model_type": "ensemble",
        "base_features": base_meta["base_features"],
        "all_features": features,
        "trackman_cols": base_meta["trackman_cols"],
        "cat_cols": cat_cols_present,
        "shrinkage_priors": base_meta["shrinkage_priors"],
        "trackman_league_fallback": base_meta["trackman_league_fallback"],
        "trackman_shrink_k": base_meta["trackman_shrink_k"],
        "season_state_cols": base_meta["season_state_cols"],
        "season_state_pitcher_priors": base_meta["season_state_pitcher_priors"],
        "season_state_batter_priors": base_meta["season_state_batter_priors"],
        "season_state_shrink_k": base_meta["season_state_shrink_k"],
        "members": members,
        "calibrator": calibrator,
        "count_trend": {"group_cols": GROUP_COLS, "trend": trend_full, "strength": TREND_STRENGTH},
        "exp_id": "exp038_ensemble_on_reduced_features",
    }
    joblib.dump(new_meta, os.path.join(MODEL_DIR, "model_meta.pkl"), compress=3)
    print(f"저장: {MODEL_DIR}/model_meta.pkl (n_features={len(features)}, members={[m['type']+':'+m['file'] for m in members]})")

    for fname in ["trackman_pitcher_lookup.csv", "season_state_pitcher_lookup.csv", "season_state_batter_lookup.csv"]:
        shutil.copy2(os.path.join(CHAMPION_ARCHIVE, fname), os.path.join(MODEL_DIR, fname))

    stale = os.path.join(MODEL_DIR, "catboost_model.cbm")
    if os.path.exists(stale):
        os.remove(stale)
    for stale_pair in ["catboost_model_seed42.cbm", "catboost_model_seed1.cbm"]:
        p = os.path.join(MODEL_DIR, stale_pair)
        if os.path.exists(p):
            os.remove(p)
            print(f"제거(다른 model_type의 잔여 파일): {p}")

    print(f"\n검증 점수(2024 held-out, step1+count-trend) = {final_score:.2f} (기준선 {BASELINE_SCORE:.2f} 대비 {delta:+.2f})")
    print("완료.")


if __name__ == "__main__":
    main()
