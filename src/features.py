"""Row-local feature engineering.

Every feature here is computed independently per row from columns that
are already part of the official train/test schema. No cross-row
aggregation, no external data — see CLAUDE.md "Competition Safety
Rules". Safe to apply identically to train and test.

exp_002 adds two things on top of exp_001's 15 features (see
experiments/exp_002_asof_refinement.md for the full per-feature
rationale table):

1. 8 new row-local derived columns (`DERIVED_COLS`) — situational
   interactions, a cleaner recency-trend decomposition, batter-side
   mirroring, and a handedness-matchup flag.
2. A cold-start empirical-Bayes shrinkage mechanism (`SHRINKAGE_SPECS`
   / `fit_shrinkage_priors` / `apply_shrinkage`) for the asof_* rate
   columns. This is a *fit/transform* pair, unlike everything else in
   this module: `fit_shrinkage_priors` MUST be called only on the
   training split (never on validation or test rows), because the
   prior it computes is a global mean that would leak validation/test
   statistics if fit on them. `apply_shrinkage` itself is row-local and
   safe to call on any split once the priors are fit.
"""

import numpy as np

CAT_COLS = ["top_bottom", "game_type", "base_state"]

DERIVED_COLS = [
    "count_diff",
    "count_total",
    "two_strike",
    "three_ball",
    "full_count",
    "late_inning",
    "score_margin_abs",
    "is_close_game",
    "runners_scoring_position",
    "pitcher_minus_batter_success",
    "pitcher_middle_minus_success",
    "pitcher_recent_form_delta",
    "pitcher_experience_log",
    "batter_experience_log",
    "pitchmix_diversity",
    # --- exp_002 additions below ---
    "pitcher_success_under_pressure",
    "pitcher_reverse_under_pressure",
    "pitcher_success_x_risp",
    "pitcher_form_trend_isolated",
    "pitcher_middle_trend_isolated",
    "pitcher_form_volatility",
    "batter_middle_minus_success",
    "same_hand_matchup",
]

# (raw rate col, sample-size col, shrinkage strength k, output col name).
# shrunk = (n*rate + k*prior) / (n+k) — prior must come from
# fit_shrinkage_priors(train_only_df). k=50 is chosen so that only the
# genuine cold-start tail is pulled toward the prior: the 1st-5th
# percentiles of asof_pitcher_n / asof_batter_n in train.csv are
# ~18-107, so k=50 meaningfully shrinks that tail while leaving the
# bulk of rows close to their raw rate (at n=0 the formula reduces
# exactly to the prior).
SHRINKAGE_SPECS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n", 50, "shrunk_pitcher_success_rate"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n", 50, "shrunk_pitcher_reverse_rate"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n", 50, "shrunk_pitcher_middle_rate"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n", 50, "shrunk_pitcher_ball_rate"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n", 50, "shrunk_pitcher_strike_rate"),
    ("asof_batter_success_rate", "asof_batter_n", 50, "shrunk_batter_success_rate"),
    ("asof_batter_middle_rate", "asof_batter_n", 50, "shrunk_batter_middle_rate"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", 50, "shrunk_pitcher_fastball_rate"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", 50, "shrunk_pitcher_breaking_rate"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", 50, "shrunk_pitcher_offspeed_rate"),
]
SHRUNK_COLS = [spec[3] for spec in SHRINKAGE_SPECS]
ALL_DERIVED_COLS = DERIVED_COLS + SHRUNK_COLS


def build_features(df):
    """Return a copy of df with derived columns added (no columns dropped)."""
    df = df.copy()

    df["count_diff"] = df["strikes_before"] - df["balls_before"]
    df["count_total"] = df["strikes_before"] + df["balls_before"]
    df["two_strike"] = (df["strikes_before"] == 2).astype(int)
    df["three_ball"] = (df["balls_before"] == 3).astype(int)
    df["full_count"] = df["two_strike"] & df["three_ball"]

    df["late_inning"] = (df["inning"] >= 7).astype(int)

    df["score_margin_abs"] = df["score_diff_pitcher_team"].abs()
    df["is_close_game"] = (df["score_margin_abs"] <= 1).astype(int)
    df["runners_scoring_position"] = (
        (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)
    ).astype(int)

    df["pitcher_minus_batter_success"] = (
        df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    )
    df["pitcher_middle_minus_success"] = (
        df["asof_pitcher_middle_rate"] - df["asof_pitcher_success_rate"]
    )
    df["pitcher_recent_form_delta"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )

    df["pitcher_experience_log"] = np.log1p(df["asof_pitcher_n"])
    df["batter_experience_log"] = np.log1p(df["asof_batter_n"])

    mix_cols = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    mix_sq_sum = sum(df[c].fillna(0) ** 2 for c in mix_cols)
    df["pitchmix_diversity"] = 1 - mix_sq_sum

    # --- exp_002 additions ---

    # A. 압박 상황별 투수 경향: li(leverage index)·득점권 주자로 투수의
    # 성공률/반대성 비율을 조건화. 트리 모델은 두 피처를 각각 분할할 순
    # 있어도 곱셈적 상호작용은 명시적으로 줘야 num_leaves=63 안에서
    # 찾기 쉬움.
    df["pitcher_success_under_pressure"] = df["asof_pitcher_success_rate"] * df["li"]
    df["pitcher_reverse_under_pressure"] = df["asof_pitcher_reverse_rate"] * df["li"]
    df["pitcher_success_x_risp"] = (
        df["asof_pitcher_success_rate"] * df["runners_scoring_position"]
    )

    # B. 최근 폼 추세/변동성: prev1/3/5는 중첩 누적 평균이라 단순
    # prev1-prev5는 prev5에 최근 경기가 섞여 추세가 흐려짐. 대수적으로
    # "4~5경기째만의 평균" = (5*prev5 - 3*prev3) / 2 로 분리해서, 가장
    # 최근 경기 대 진짜 예전 경기를 비교하는 추세 피처를 만든다.
    def _isolated_trend(prev1, prev3, prev5):
        early_only = (5 * prev5 - 3 * prev3) / 2
        return prev1 - early_only

    df["pitcher_form_trend_isolated"] = _isolated_trend(
        df["asof_pitcher_prev1_game_success_rate"],
        df["asof_pitcher_prev3_game_success_rate"],
        df["asof_pitcher_prev5_game_success_rate"],
    )
    df["pitcher_middle_trend_isolated"] = _isolated_trend(
        df["asof_pitcher_prev1_game_middle_rate"],
        df["asof_pitcher_prev3_game_middle_rate"],
        df["asof_pitcher_prev5_game_middle_rate"],
    )
    form_cols = df[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ]
    df["pitcher_form_volatility"] = form_cols.max(axis=1) - form_cols.min(axis=1)

    # C. 타자 지표 보완: 기존엔 pitcher_middle_minus_success만 있고
    # 타자 쪽 동일 지표가 없었음 — 대칭적으로 추가.
    df["batter_middle_minus_success"] = (
        df["asof_batter_middle_rate"] - df["asof_batter_success_rate"]
    )

    # D. 손잡이 매치업: 전형적인 platoon 신호. pitcher_hand/batter_hand는
    # 이미 원본 컬럼으로 모델에 들어가지만 둘을 직접 비교한 피처는 없었음.
    df["same_hand_matchup"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    return df


def fit_shrinkage_priors(train_df, specs=SHRINKAGE_SPECS):
    """Compute the global prior rate for each shrinkage spec.

    MUST be called on the training split only (e.g. seasons 2019-2023
    for the held-out validation run, or the full 2019-2024 train.csv
    for the final refit) — never on validation or test rows, since the
    prior is a dataset-wide mean and would otherwise leak those splits'
    statistics into a "training-derived" feature.
    """
    return {rate_col: train_df[rate_col].mean() for rate_col, _, _, _ in specs}


def apply_shrinkage(df, priors, specs=SHRINKAGE_SPECS):
    """Row-local empirical-Bayes shrinkage: shrunk = (n*r + k*prior)/(n+k).

    Safe to call on any split (train/val/test) once `priors` has been
    fit on the training split via `fit_shrinkage_priors`. Always
    NaN/inf-free: raw rate and sample-size are filled with 0 first, so
    n=0 (cold start) reduces exactly to the prior.
    """
    df = df.copy()
    for rate_col, n_col, k, shrunk_col in specs:
        r = df[rate_col].fillna(0)
        n = df[n_col].fillna(0)
        df[shrunk_col] = (n * r + k * priors[rate_col]) / (n + k)
    return df
