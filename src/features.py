"""Row-local feature engineering.

Every feature here is computed independently per row from columns that
are already part of the official train/test schema. No cross-row
aggregation, no external data — see CLAUDE.md "Competition Safety
Rules". Safe to apply identically to train and test.
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
]


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

    return df
