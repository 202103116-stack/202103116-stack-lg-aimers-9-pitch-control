from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

CUSTOM_FEATURES = frozenset(('pitcher_gap_prev1_career', 'pitcher_gap_prev3_career', 'pitcher_gap_prev5_career', 'win_expectancy_dist50', 'count_diff', 'count_total', 'same_hand_matchup', 'pressure_x_recent_form', 'runners_x_li', 'batter_success_rate_shrunk', 'reverse_rate_x_li', 'middle_rate_x_count_diff', 'late_inning_x_recent_form', 'offspeed_x_li', 'count_state', 'is_full_count', 'has_two_strikes', 'has_three_balls', 'has_two_outs', 'pitcher_season_n', 'pitcher_season_success_rate_k20', 'pitcher_season_reverse_rate_k20', 'pitcher_season_middle_rate_k20', 'pitcher_season_ball_rate_k20', 'pitcher_season_strike_rate_k20', 'batter_season_n', 'batter_season_success_rate_k20', 'batter_season_middle_rate_k20', 'same_hand_x_fastball', 'same_hand_x_breaking', 'same_hand_x_offspeed', 'pitcher_history_reliability_k100', 'batter_history_reliability_k100', 'pitcher_success_x_reliability', 'batter_success_x_reliability', 'pitcher_reverse_x_reliability', 'pitcher_middle_x_reliability'))
ROW_LOCAL_CUSTOM_FEATURES = ('pitcher_gap_prev1_career', 'pitcher_gap_prev3_career', 'pitcher_gap_prev5_career', 'win_expectancy_dist50', 'count_diff', 'count_total', 'same_hand_matchup', 'pressure_x_recent_form', 'runners_x_li', 'batter_success_rate_shrunk', 'reverse_rate_x_li', 'middle_rate_x_count_diff', 'late_inning_x_recent_form', 'offspeed_x_li', 'count_state', 'is_full_count', 'has_two_strikes', 'has_three_balls', 'has_two_outs', 'pitcher_season_n', 'pitcher_season_success_rate_k20', 'pitcher_season_reverse_rate_k20', 'pitcher_season_middle_rate_k20', 'pitcher_season_ball_rate_k20', 'pitcher_season_strike_rate_k20', 'batter_season_n', 'batter_season_success_rate_k20', 'batter_season_middle_rate_k20', 'same_hand_x_fastball', 'same_hand_x_breaking', 'same_hand_x_offspeed', 'pitcher_history_reliability_k100', 'batter_history_reliability_k100', 'pitcher_success_x_reliability', 'batter_success_x_reliability', 'pitcher_reverse_x_reliability', 'pitcher_middle_x_reliability')
SEQUENTIAL_FEATURES = frozenset()
REFERENCE_STATE_FEATURES = frozenset(('pitcher_gap_prev1_career', 'pitcher_gap_prev3_career', 'pitcher_gap_prev5_career', 'win_expectancy_dist50', 'count_diff', 'count_total', 'same_hand_matchup', 'pressure_x_recent_form', 'runners_x_li', 'batter_success_rate_shrunk', 'reverse_rate_x_li', 'middle_rate_x_count_diff', 'late_inning_x_recent_form', 'offspeed_x_li', 'count_state', 'is_full_count', 'has_two_strikes', 'has_three_balls', 'has_two_outs', 'pitcher_season_n', 'pitcher_season_success_rate_k20', 'pitcher_season_reverse_rate_k20', 'pitcher_season_middle_rate_k20', 'pitcher_season_ball_rate_k20', 'pitcher_season_strike_rate_k20', 'batter_season_n', 'batter_season_success_rate_k20', 'batter_season_middle_rate_k20', 'same_hand_x_fastball', 'same_hand_x_breaking', 'same_hand_x_offspeed', 'pitcher_history_reliability_k100', 'batter_history_reliability_k100', 'pitcher_success_x_reliability', 'batter_success_x_reliability', 'pitcher_reverse_x_reliability', 'pitcher_middle_x_reliability'))

def _safe_rate(series: pd.Series, prior: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(prior)


def _safe_count(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)


def engineer_features(
    frame: pd.DataFrame,
    *,
    success_prior: float,
    shrinkage_k: float = 50.0,
    requested_custom: Collection[str] | None = None,
) -> pd.DataFrame:
    """Create only rule-compliant, row-local features.

    ``requested_custom`` is deliberately checked before any sequence operation.
    Evaluation rows must never affect one another, even through an unused
    intermediate column.
    """
    requested = frozenset(
        ROW_LOCAL_CUSTOM_FEATURES
        if requested_custom is None
        else requested_custom
    )
    unknown = requested.difference(CUSTOM_FEATURES)
    if unknown:
        raise ValueError(f"알 수 없는 custom feature: {sorted(unknown)}")
    forbidden = requested.intersection(SEQUENTIAL_FEATURES)
    if forbidden:
        raise ValueError(
            "대회 규칙상 평가 행 간 연산이 필요한 feature는 사용할 수 없습니다: "
            f"{sorted(forbidden)}"
        )
    out = frame.copy()
    for window in (1, 3, 5):
        out[f"pitcher_gap_prev{window}_career"] = (
            out[f"asof_pitcher_prev{window}_game_success_rate"]
            - out["asof_pitcher_success_rate"]
        ).astype("float32")

    out["win_expectancy_dist50"] = (
        out["home_win_expectancy"] - 50.0
    ).abs().astype("float32")
    out["count_diff"] = (
        out["balls_before"] - out["strikes_before"]
    ).astype("int8")
    out["count_total"] = (
        out["balls_before"] + out["strikes_before"]
    ).astype("int8")
    out["same_hand_matchup"] = (
        out["pitcher_hand"] == out["batter_hand"]
    ).astype("int8")
    out["pressure_x_recent_form"] = (
        out["count_diff"] * out["asof_pitcher_prev3_game_success_rate"]
    ).astype("float32")
    out["runners_x_li"] = (out["num_runners_on"] * out["li"]).astype(
        "float32"
    )

    batter_n = _safe_count(out["asof_batter_n"])
    batter_rate = _safe_rate(out["asof_batter_success_rate"], success_prior)
    out["batter_success_rate_shrunk"] = (
        (batter_rate * batter_n + success_prior * shrinkage_k)
        / (batter_n + shrinkage_k)
    ).astype("float32")
    out["reverse_rate_x_li"] = (
        out["asof_pitcher_reverse_rate"] * out["li"]
    ).astype("float32")
    out["middle_rate_x_count_diff"] = (
        out["asof_pitcher_middle_rate"] * out["count_diff"]
    ).astype("float32")
    out["late_inning_x_recent_form"] = (
        (out["inning"] >= 7).astype("int8")
        * out["asof_pitcher_prev3_game_success_rate"]
    ).astype("float32")
    out["offspeed_x_li"] = (
        out["asof_pitcher_offspeed_rate"] * out["li"]
    ).astype("float32")

    out["recent_history_missing"] = out[
        "asof_pitcher_prev1_game_success_rate"
    ].isna().astype("int8")
    out["pitcher_history_missing"] = out[
        "asof_pitcher_success_rate"
    ].isna().astype("int8")
    out["batter_history_missing"] = out[
        "asof_batter_success_rate"
    ].isna().astype("int8")

    pitcher_n = _safe_count(out["asof_pitcher_n"])
    pitcher_success = _safe_rate(
        out["asof_pitcher_success_rate"], success_prior
    )
    pitcher_reverse = _safe_rate(
        out["asof_pitcher_reverse_rate"], 1.0 - success_prior
    )
    out["pitcher_success_rate_shrunk"] = (
        (pitcher_success * pitcher_n + success_prior * shrinkage_k)
        / (pitcher_n + shrinkage_k)
    ).astype("float32")
    out["pitcher_reverse_rate_shrunk"] = (
        (pitcher_reverse * pitcher_n + (1.0 - success_prior) * shrinkage_k)
        / (pitcher_n + shrinkage_k)
    ).astype("float32")
    out["success_minus_reverse"] = (
        pitcher_success - pitcher_reverse
    ).astype("float32")
    out["log1p_pitcher_n"] = np.log1p(pitcher_n).astype("float32")
    out["log1p_batter_n"] = np.log1p(batter_n).astype("float32")

    out["pitcher_team_win_expectancy"] = np.where(
        out["top_bottom"].eq("T"),
        out["home_win_expectancy"],
        out["away_win_expectancy"],
    ).astype("float32")
    out["log1p_li"] = np.log1p(out["li"].clip(lower=0)).astype("float32")
    out["scoring_position"] = (
        out["runner_on_2b"].astype(bool) | out["runner_on_3b"].astype(bool)
    ).astype("int8")
    out["scoring_position_x_li"] = (
        out["scoring_position"] * out["li"]
    ).astype("float32")

    out["count_state"] = (
        out["balls_before"] * 3 + out["strikes_before"]
    ).astype("int8")
    out["is_full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype("int8")
    out["has_two_strikes"] = (out["strikes_before"] == 2).astype("int8")
    out["has_three_balls"] = (out["balls_before"] == 3).astype("int8")
    out["has_two_outs"] = (out["outs_before"] == 2).astype("int8")
    out["count_hands_state"] = (
        out["count_state"] * 9
        + (out["pitcher_hand"] - 1) * 3
        + (out["batter_hand"] - 1)
    ).astype("int16")
    out["count_matchup_state"] = (
        out["count_state"] * 2 + out["same_hand_matchup"]
    ).astype("int16")
    out["count_out_state"] = (
        out["count_state"] * 3 + out["outs_before"]
    ).astype("int16")
    out["count_hands_out_state"] = (
        out["count_hands_state"] * 3 + out["outs_before"]
    ).astype("int16")
    out["recent_success_blend_135"] = (
        0.2 * out["asof_pitcher_prev1_game_success_rate"]
        + 0.5 * out["asof_pitcher_prev3_game_success_rate"]
        + 0.3 * out["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")
    out["recent_middle_blend_135"] = (
        0.2 * out["asof_pitcher_prev1_game_middle_rate"]
        + 0.5 * out["asof_pitcher_prev3_game_middle_rate"]
        + 0.3 * out["asof_pitcher_prev5_game_middle_rate"]
    ).astype("float32")
    out["recent_quality_blend_135"] = (
        out["recent_success_blend_135"] - out["recent_middle_blend_135"]
    ).astype("float32")
    out["recent_success_span_15"] = (
        out["asof_pitcher_prev1_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    ).abs().astype("float32")
    out["recent_middle_span_15"] = (
        out["asof_pitcher_prev1_game_middle_rate"]
        - out["asof_pitcher_prev5_game_middle_rate"]
    ).abs().astype("float32")
    out["hand_matchup_state"] = (
        (out["pitcher_hand"] - 1) * 2 + (out["batter_hand"] - 1)
    ).astype("int8")
    pitchmix = {
        "fastball": out["asof_pitcher_fastball_rate"],
        "breaking": out["asof_pitcher_breaking_rate"],
        "offspeed": out["asof_pitcher_offspeed_rate"],
    }
    same_hand = out["same_hand_matchup"].astype("float32")
    pitcher_hand_centered = (out["pitcher_hand"] - 1.5).astype("float32")
    batter_hand_centered = (out["batter_hand"] - 1.5).astype("float32")
    for mix_name, mix_rate in pitchmix.items():
        out[f"same_hand_x_{mix_name}"] = (same_hand * mix_rate).astype("float32")
        out[f"pitcher_hand_x_{mix_name}"] = (
            pitcher_hand_centered * mix_rate
        ).astype("float32")
        out[f"batter_hand_x_{mix_name}"] = (
            batter_hand_centered * mix_rate
        ).astype("float32")
        for matchup in range(4):
            out[f"matchup{matchup}_x_{mix_name}"] = (
                out["hand_matchup_state"].eq(matchup).astype("float32") * mix_rate
            ).astype("float32")
    # Row-local reliability features.  They use only the organizer-provided
    # as-of history in the current row, so evaluation rows remain independent.
    history_bins = [-np.inf, 0, 10, 50, 200, 1000, np.inf]
    out["pitcher_history_bucket"] = pd.cut(
        pitcher_n, bins=history_bins, labels=False, include_lowest=True
    ).fillna(0).astype("int8")
    out["batter_history_bucket"] = pd.cut(
        batter_n, bins=history_bins, labels=False, include_lowest=True
    ).fillna(0).astype("int8")
    pitcher_reliability = (pitcher_n / (pitcher_n + 100.0)).astype("float32")
    batter_reliability = (batter_n / (batter_n + 100.0)).astype("float32")
    out["pitcher_history_reliability_k100"] = pitcher_reliability
    out["batter_history_reliability_k100"] = batter_reliability
    out["pitcher_success_x_reliability"] = (
        pitcher_success * pitcher_reliability
    ).astype("float32")
    out["batter_success_x_reliability"] = (
        batter_rate * batter_reliability
    ).astype("float32")
    out["pitcher_reverse_x_reliability"] = (
        pitcher_reverse * pitcher_reliability
    ).astype("float32")
    out["pitcher_middle_x_reliability"] = (
        _safe_rate(out["asof_pitcher_middle_rate"], 0.0) * pitcher_reliability
    ).astype("float32")
    for metric in ("ball", "strike", "fastball", "breaking", "offspeed"):
        out[f"pitcher_{metric}_x_reliability"] = (
            _safe_rate(out[f"asof_pitcher_{metric}_rate"], 0.0)
            * pitcher_reliability
        ).astype("float32")
    pitcher_middle = _safe_rate(out["asof_pitcher_middle_rate"], 0.0)
    for reliability_k in (50, 200):
        pitcher_rel = (pitcher_n / (pitcher_n + reliability_k)).astype("float32")
        batter_rel = (batter_n / (batter_n + reliability_k)).astype("float32")
        suffix = f"_k{reliability_k}"
        out[f"pitcher_history_reliability{suffix}"] = pitcher_rel
        out[f"batter_history_reliability{suffix}"] = batter_rel
        out[f"pitcher_success_x_reliability{suffix}"] = (
            pitcher_success * pitcher_rel
        ).astype("float32")
        out[f"batter_success_x_reliability{suffix}"] = (
            batter_rate * batter_rel
        ).astype("float32")
        out[f"pitcher_reverse_x_reliability{suffix}"] = (
            pitcher_reverse * pitcher_rel
        ).astype("float32")
        out[f"pitcher_middle_x_reliability{suffix}"] = (
            pitcher_middle * pitcher_rel
        ).astype("float32")
    out["inning_phase_state"] = np.select(
        [out["inning"].le(3), out["inning"].le(6), out["inning"].le(9)],
        [0, 1, 2],
        default=3,
    ).astype("int8")
    for phase, phase_name in enumerate(("early", "middle", "late", "extra")):
        out[f"{phase_name}_inning_x_li"] = (
            out["inning_phase_state"].eq(phase).astype("float32") * out["li"]
        ).astype("float32")
    late = out["inning"].ge(7).astype("float32")
    out["late_inning_x_pitcher_success"] = (
        late * out["asof_pitcher_success_rate"]
    ).astype("float32")
    out["late_inning_x_pitcher_reverse"] = (
        late * out["asof_pitcher_reverse_rate"]
    ).astype("float32")
    out["late_inning_x_pitcher_middle"] = (
        late * out["asof_pitcher_middle_rate"]
    ).astype("float32")
    pressure_masks = {
        "two_strike": out["strikes_before"].eq(2).astype("float32"),
        "three_ball": out["balls_before"].eq(3).astype("float32"),
        "full_count": (
            out["balls_before"].eq(3) & out["strikes_before"].eq(2)
        ).astype("float32"),
    }
    for pressure_name, pressure_mask in pressure_masks.items():
        for metric in ("success", "reverse", "middle"):
            out[f"{pressure_name}_x_pitcher_{metric}"] = (
                pressure_mask * out[f"asof_pitcher_{metric}_rate"]
            ).astype("float32")
    # Everything below this point in the historical implementation derives
    # one evaluation row from neighboring evaluation rows. It is intentionally
    # unreachable in the compliant path and retained only so old quarantined
    # experiment records remain interpretable.
    return out


def apply_official_state_reference(
    frame: pd.DataFrame,
    reference: dict[str, Any],
    *,
    requested: Collection[str],
    shrinkage_k: float = 20.0,
) -> pd.DataFrame:
    """Apply frozen official-train state to rows without aggregating those rows."""
    selected = frozenset(requested).intersection(REFERENCE_STATE_FEATURES)
    out = frame.copy()
    for entity, entity_reference in reference.get("entities", {}).items():
        metrics = tuple(entity_reference["baseline_counts"])
        key = f"{entity}_id"
        count_column = f"asof_{entity}_n"
        baseline_count = out[key].map(entity_reference["baseline_count"]).fillna(0.0)
        current_n = pd.to_numeric(out[count_column], errors="coerce").fillna(0.0)
        season_n = (current_n - baseline_count).clip(lower=0.0)
        n_name = f"{entity}_season_n"
        if n_name in selected:
            out[n_name] = np.log1p(season_n).astype("float32")
        for metric in metrics:
            candidates = [(f"{entity}_season_{metric}_rate_k20", shrinkage_k)]
            if metric == "success" and entity in {"pitcher", "batter"}:
                candidates.extend(
                    (f"{entity}_season_success_rate_k{k_value}", float(k_value))
                    for k_value in (5, 10, 50, 100)
                )
            if entity == "pitcher" and metric == "ball":
                candidates.append(("pitcher_season_ball_rate_k500", 500.0))
            if entity == "pitcher" and metric == "strike":
                candidates.append(("pitcher_season_strike_rate_k200", 200.0))
            if entity == "batter" and metric == "middle":
                candidates.append(("batter_season_middle_rate_k200", 200.0))
            if not any(name in selected for name, _ in candidates):
                continue
            rate_column = f"asof_{entity}_{metric}_rate"
            current_count = current_n * pd.to_numeric(
                out[rate_column], errors="coerce"
            )
            baseline_metric = out[key].map(
                entity_reference["baseline_counts"][metric]
            ).fillna(0.0)
            delta_count = (current_count - baseline_metric).clip(
                lower=0.0, upper=season_n
            )
            count_name = f"{entity}_season_{metric}_count_log"
            if count_name in selected:
                out[count_name] = np.log1p(delta_count.fillna(0.0)).astype("float32")
            failure_name = f"{entity}_season_failure_count_log"
            if metric == "success" and failure_name in selected:
                out[failure_name] = np.log1p(
                    (season_n - delta_count.fillna(0.0)).clip(lower=0.0)
                ).astype("float32")
            prior = float(entity_reference["priors"][metric])
            for name, k_value in candidates:
                if name in selected:
                    out[name] = (
                        (delta_count.fillna(0.0) + k_value * prior)
                        / (season_n + k_value)
                    ).astype("float32")
        for metric in metrics:
            delta_name = f"{entity}_season_{metric}_delta_career"
            season_name = f"{entity}_season_{metric}_rate_k20"
            if delta_name in selected:
                out[delta_name] = (
                    out[season_name] - out[f"asof_{entity}_{metric}_rate"]
                ).astype("float32")
    return out




from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ID_COL = "row_id"
TARGET_COL = "control_success"


def map_offset(frame: pd.DataFrame, columns: list[str], values: pd.Series) -> np.ndarray:
    if not columns:
        return np.zeros(len(frame), dtype=float)
    if len(columns) == 1:
        mapped = frame[columns[0]].map(values)
    else:
        keys = pd.MultiIndex.from_frame(frame.loc[:, columns])
        mapped = pd.Series(values.reindex(keys).to_numpy(), index=frame.index)
    return mapped.fillna(0.0).to_numpy(dtype=float)


def main() -> None:
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
