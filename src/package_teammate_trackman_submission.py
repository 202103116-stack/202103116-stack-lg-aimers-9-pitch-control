"""Assembles the final submittable package for the teammate(EXP_130)+
trackman20 model: takes the already-built triple-ensemble model.joblib
(cpu/gpu/extra members, trained on full 2019-2024, unmodified) and
teammate's generated script.py, then:
  1. Patches script.py to merge our trackman lookup CSV onto the real
     test.csv before feature engineering (the real 2025 evaluation set
     never has these columns baked in -- unlike our own local
     data_trackman/test.csv used only for teammate's own build/test).
  2. Replaces teammate's static per-member affine's downstream group+
     trend correction with OUR validated recipe (self-consistent
     group_offset, no trend, + our Platt regression -- see
     experiments/exp_039_teammate_trackman_graft.md for why: naively
     reusing their group_offsets/trend_offsets caused a large,
     misleading regression due to a model-mismatch artifact, and even
     once fixed, trend_offset still measurably hurt in every clean test;
     group_offset + Platt was the actual local-validated winner, 950.19
     vs baseline_78's own best of 947.10).
Produces a self-contained submission/archive/<label>/ folder in THIS
repo (not teammate's), following this project's own packaging
convention (model/, script.py, requirements.txt + submit.zip).
"""

import os
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

TEAMMATE_PACKAGE_DIR = r"c:\Users\yeob0\lgAimers9_pitch-control\_teammate_repo\experiment_workspace\output\final_trackman20\package"
OUR_OUTPUT_DIR = "./output"
LABEL = "exp039_teammate_trackman20_graft"
ARCHIVE_DIR = f"./submission/archive/{LABEL}"

PATCHED_MAIN_TAIL = '''def main() -> None:
    data_dir = ROOT / "data"
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    bundle = joblib.load(ROOT / "model" / "model.joblib")

    # exp_039: 실제 2025 평가셋에는 우리 커스텀 트랙맨 컬럼이 없으므로,
    # pitcher_id 기준 flat lookup(as-of 계산은 학습 시점에 이미 끝남)을
    # 병합하고 매핑 안 되는 투수(리그 전체 폴백)까지 채운다.
    trackman_lookup = pd.read_csv(ROOT / "model" / "trackman_lookup.csv")
    trackman_fallback = joblib.load(ROOT / "model" / "trackman_fallback.pkl")
    trackman_cols = [c for c in trackman_lookup.columns if c != "pitcher_id"]
    test = test.merge(trackman_lookup, on="pitcher_id", how="left")
    for col in trackman_cols:
        test[col] = test[col].fillna(trackman_fallback[col])

    frame = engineer_features(
        test,
        success_prior=float(bundle["success_prior"]),
        shrinkage_k=float(bundle["shrinkage_k"]),
        requested_custom=bundle["custom_features"],
    )
    frame = apply_official_state_reference(
        frame,
        bundle["state_reference"],
        requested=bundle["custom_features"],
        shrinkage_k=float(bundle["shrinkage_k"]),
    )
    selected = frame.loc[:, bundle["features"]]
    cat_matrix = selected.copy()
    for column in bundle["categorical"]:
        cat_matrix[column] = cat_matrix[column].fillna("__MISSING__").astype(str)
    numeric = bundle["cat_numeric"]
    cat_matrix.loc[:, numeric] = (
        cat_matrix.loc[:, numeric]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(bundle["cat_medians"])
        .astype("float32")
    )
    extra_matrix = bundle["extra_preprocessor"].transform(selected)
    spec = bundle["model_spec"]

    def adjusted_cat(model) -> np.ndarray:
        raw = model.predict_proba(cat_matrix)[:, 1]
        return np.clip(
            0.5 + float(spec.get("cat_scale", 1.0)) * (raw - 0.5)
            + float(spec.get("cat_shift", 0.0)),
            0.0, 1.0,
        )

    probability = (
        float(spec["cpu_weight"]) * adjusted_cat(bundle["cpu_model"])
        + float(spec["gpu_weight"]) * adjusted_cat(bundle["gpu_model"])
        + float(spec["extra_weight"]) * np.clip(
            bundle["extra_model"].predict_proba(extra_matrix)[:, 1]
            + float(spec.get("extra_shift", 0.0)),
            0.0, 1.0,
        )
    )

    # exp_039: 팀원 원본의 group_offset+trend_offset 조합을 우리 자체
    # 검증(2024 진짜 held-out, 자기일관적 재계산)으로 교체 -- trend는
    # 일관되게 손해였고(-3.15~-19.68), group_offset 단독 + 우리 Platt
    # 회귀보정이 최선이었음(950.19, exp_039 문서 참고).
    our_calibration = joblib.load(ROOT / "model" / "our_calibration.pkl")
    probability = probability + map_offset(
        frame, our_calibration["group_columns"], our_calibration["group_offsets"]
    )
    probability = np.clip(probability, 0.0, 1.0)
    probability = our_calibration["calibrator"].predict_proba(
        probability.reshape(-1, 1)
    )[:, 1]
    probability = np.clip(probability, 0.0, 1.0)

    prediction = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: probability})
    submission = sample[[ID_COL]].merge(
        prediction, on=ID_COL, how="left", validate="one_to_one"
    )
    if submission[ID_COL].tolist() != sample[ID_COL].tolist():
        raise RuntimeError("sample_submission의 row_id 순서를 보존하지 못했습니다.")
    if submission[TARGET_COL].isna().any():
        raise RuntimeError("예측값에 결측값이 있습니다.")
    if not submission[TARGET_COL].between(0.0, 1.0).all():
        raise RuntimeError("예측 확률 범위를 벗어났습니다.")
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)


if __name__ == "__main__":
    main()
'''


def main():
    print("=" * 80)
    print("1. 원본 script.py 로드 + main() 교체")
    print("=" * 80)
    with open(os.path.join(TEAMMATE_PACKAGE_DIR, "script.py"), "r", encoding="utf-8") as f:
        original = f.read()
    marker = "def main() -> None:"
    idx = original.index(marker)
    prelude = original[:idx]
    patched = prelude + PATCHED_MAIN_TAIL
    print(f"원본 {len(original)}자 -> 패치본 {len(patched)}자")

    print()
    print("=" * 80)
    print("2. 아카이브 폴더 구성")
    print("=" * 80)
    if os.path.exists(ARCHIVE_DIR):
        shutil.rmtree(ARCHIVE_DIR)
    model_dir = os.path.join(ARCHIVE_DIR, "model")
    os.makedirs(model_dir)

    shutil.copy2(os.path.join(TEAMMATE_PACKAGE_DIR, "model", "model.joblib"), os.path.join(model_dir, "model.joblib"))
    shutil.copy2(os.path.join(OUR_OUTPUT_DIR, "teammate_trackman_lookup.csv"), os.path.join(model_dir, "trackman_lookup.csv"))
    shutil.copy2(os.path.join(OUR_OUTPUT_DIR, "teammate_trackman_fallback.pkl"), os.path.join(model_dir, "trackman_fallback.pkl"))
    shutil.copy2(os.path.join(OUR_OUTPUT_DIR, "teammate_trackman20_final_calibration.pkl"), os.path.join(model_dir, "our_calibration.pkl"))
    print(f"복사: model.joblib, trackman_lookup.csv, trackman_fallback.pkl, our_calibration.pkl -> {model_dir}")

    with open(os.path.join(ARCHIVE_DIR, "script.py"), "w", encoding="utf-8") as f:
        f.write(patched)
    shutil.copy2(os.path.join(TEAMMATE_PACKAGE_DIR, "requirements.txt"), os.path.join(ARCHIVE_DIR, "requirements.txt"))
    print(f"저장: {ARCHIVE_DIR}/script.py, requirements.txt")

    print()
    print("=" * 80)
    print("3. submit.zip 생성")
    print("=" * 80)
    zip_path = os.path.join(ARCHIVE_DIR, "submit.zip")
    entries = {
        "script.py": os.path.join(ARCHIVE_DIR, "script.py"),
        "requirements.txt": os.path.join(ARCHIVE_DIR, "requirements.txt"),
    }
    for fname in os.listdir(model_dir):
        entries[f"model/{fname}"] = os.path.join(model_dir, fname)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in entries.items():
            zf.write(path, arcname)
    print(f"저장: {zip_path}")
    print("\n완료.")


if __name__ == "__main__":
    main()
