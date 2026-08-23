"""Prepares trackman-augmented copies of the teammate's (EXP_130,
github.com/seokjin-wq/Aimers-9th) train.csv/test.csv/sample_submission.csv
so their `aimers_exp` framework can train on 78+20=98 features without any
code changes to that framework (their `resolve_feature_names` does raw
column pass-through from test.csv's header -- see the two new columns
just need to physically exist with matching names in both files).

Their model.joblib inspection confirmed EXP_130 uses ZERO trackman
columns -- this grafts our validated pooled (5) + pitch_type_group-
conditioned (15) trackman physical features onto their stronger base
(LB 1043.66 vs our own 1006.62).

train.csv gets the TRUE as-of (leak-safe, per-row season/game_month)
values via merge_asof (same mechanism as our own production pipeline).
test.csv (their small local 5-row sample, used only for their own local
build/package testing) gets the flat test-time lookup + league fallback
-- exactly mirroring how our submission/script.py resolves trackman
features for real 2025 rows. The SAME flat lookup CSV is saved
separately so a later step can bundle it into the final submission
package for the real 2025 evaluation set (which won't have these columns
baked in).
"""

import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

from trackman_pitcher_features import (
    GROUP_TRACKMAN_COLS, TRACKMAN_PITCHER_ASOF_COLS, attach_pitcher_physical_features,
    attach_pitcher_physical_features_by_group, build_pitcher_physical_asof_tables,
    build_pitcher_physical_asof_tables_by_group, build_test_time_pitcher_lookup,
    build_test_time_pitcher_lookup_by_group, load_pitcher_mapping,
)

OUR_DATA_DIR = "./data"
TEAMMATE_DATA_DIR = r"../_teammate_repo/experiment_workspace/data"
OUT_DIR = r"../_teammate_repo/experiment_workspace/data_trackman"
TRACKMAN_SHRINK_K = 50
ALL_TRACKMAN_COLS = TRACKMAN_PITCHER_ASOF_COLS + GROUP_TRACKMAN_COLS


def main():
    print("=" * 80)
    print("0. 트랙맨 as-of 테이블 구축 (풀링 5 + 구종군 15)")
    print("=" * 80)
    trackman_clean = pd.read_csv(os.path.join(OUR_DATA_DIR, "processed", "trackman_clean.csv"), encoding="utf-8-sig")
    pitcher_mapping = load_pitcher_mapping()
    tables_pooled = build_pitcher_physical_asof_tables(trackman_clean)
    tables_group = build_pitcher_physical_asof_tables_by_group(trackman_clean)
    print(f"트랙맨 컬럼 {len(ALL_TRACKMAN_COLS)}개: {ALL_TRACKMAN_COLS}")

    print()
    print("=" * 80)
    print("1. 팀원 train.csv에 진짜 as-of 값 부착 (2019-2024 전체, leak-safe)")
    print("=" * 80)
    train = pd.read_csv(os.path.join(TEAMMATE_DATA_DIR, "train.csv"), encoding="utf-8-sig")
    print(f"원본 train: {train.shape}")
    train = attach_pitcher_physical_features(train, tables_pooled, pitcher_mapping, shrink_k=TRACKMAN_SHRINK_K)
    train = attach_pitcher_physical_features_by_group(train, tables_group, pitcher_mapping, shrink_k=TRACKMAN_SHRINK_K)
    print(f"부착 후 train: {train.shape}")
    assert all(c in train.columns for c in ALL_TRACKMAN_COLS)

    print()
    print("=" * 80)
    print("2. 팀원 test.csv(로컬 5행 샘플)에 flat lookup + league fallback 부착")
    print("=" * 80)
    lookup_pooled, fallback_pooled = build_test_time_pitcher_lookup(tables_pooled, pitcher_mapping, shrink_k=TRACKMAN_SHRINK_K)
    lookup_group, fallback_group = build_test_time_pitcher_lookup_by_group(tables_group, pitcher_mapping, shrink_k=TRACKMAN_SHRINK_K)
    lookup_combined = lookup_pooled.merge(lookup_group, on="pitcher_id", how="outer")
    fallback_combined = {**fallback_pooled, **fallback_group}

    test = pd.read_csv(os.path.join(TEAMMATE_DATA_DIR, "test.csv"), encoding="utf-8-sig")
    print(f"원본 test: {test.shape}")
    test = test.merge(lookup_combined, on="pitcher_id", how="left")
    for col in ALL_TRACKMAN_COLS:
        test[col] = test[col].fillna(fallback_combined[col])
    print(f"부착 후 test: {test.shape}")
    assert all(c in test.columns for c in ALL_TRACKMAN_COLS)
    assert test[ALL_TRACKMAN_COLS].isna().sum().sum() == 0

    print()
    print("=" * 80)
    print("3. 저장")
    print("=" * 80)
    os.makedirs(OUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8-sig")
    shutil.copy2(os.path.join(TEAMMATE_DATA_DIR, "sample_submission.csv"), os.path.join(OUT_DIR, "sample_submission.csv"))
    print(f"저장: {OUT_DIR}/{{train.csv, test.csv, sample_submission.csv}}")

    lookup_combined.to_csv("./output/teammate_trackman_lookup.csv", index=False, encoding="utf-8-sig")
    import joblib
    joblib.dump(fallback_combined, "./output/teammate_trackman_fallback.pkl")
    print("저장: ./output/teammate_trackman_lookup.csv, teammate_trackman_fallback.pkl (실제 2025 제출 패키징용)")
    print("완료.")


if __name__ == "__main__":
    main()
