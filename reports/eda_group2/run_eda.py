"""Group 2 EDA — 투구 직전 카운트 및 점수 상황 (8개 컬럼).

담당 컬럼:
    balls_before, strikes_before, outs_before,
    run_top_before, run_bot_before, run_total_before,
    score_diff_home, score_diff_pitcher_team

이 스크립트는 train.csv만 읽고(test.csv는 사용하지 않음), 행 단위 통계와
group-by 집계만 수행한다 (CLAUDE.md 누수 규칙과 충돌 없음 — 여기서 나온
어떤 값도 test.csv 피처 생성에 직접 대입하지 않고, 대회 코드 쪽에서 재현할
때는 train만으로 다시 fit해야 한다).

실행:
    python reports/eda_group2/run_eda.py

산출물:
    reports/eda_group2/figures/*.png
    reports/eda_group2/tables/*.csv
    (표준출력을 reports/eda_group2/eda_run_log.txt 로 리다이렉트해서 보관 —
     README.md의 모든 수치는 이 로그에서만 가져온다)
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 라벨이 깨지지 않도록 폰트 설정 (Windows 기본 탑재 폰트)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = "./data"
OUT_DIR = "./reports/eda_group2"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

COUNT_COLS = ["balls_before", "strikes_before", "outs_before"]
SCORE_COLS = [
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
]
ALL_COLS = COUNT_COLS + SCORE_COLS
TARGET = "control_success"

FLAGGED_COUNTS = [(0, 0), (0, 2), (3, 0), (3, 1), (3, 2)]

SCORE_BIN_EDGES = [-np.inf, -4, -1, 0, 3, np.inf]
SCORE_BIN_LABELS = ["큰 열세(<=-4)", "열세(-3~-1)", "동점(0)", "리드(1~3)", "큰 리드(>=4)"]


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data():
    usecols = ["season", "top_bottom"] + ALL_COLS + [TARGET]
    df = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=usecols
    )
    return df


def step1_sanity_check(df):
    section("1. 프로젝트/데이터 구조 확인")
    print(f"train.csv shape: {df.shape}")
    missing = [c for c in ALL_COLS + [TARGET, "season", "top_bottom"] if c not in df.columns]
    print(f"누락된 컬럼: {missing if missing else '없음 (전부 존재)'}")
    print("\ndtype:")
    print(df[ALL_COLS + [TARGET]].dtypes)

    rows = []
    for c in ALL_COLS:
        s = df[c]
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_null": int(s.isna().sum()),
                "min": s.min(),
                "max": s.max(),
                "n_unique": int(s.nunique()),
            }
        )
    quality = pd.DataFrame(rows)
    print("\n컬럼별 결측/범위/unique 개수:")
    print(quality.to_string(index=False))
    quality.to_csv(os.path.join(TAB_DIR, "00_quality_summary.csv"), index=False)
    return quality


def step2_derived_relationships(df):
    section("2. 파생 관계 검증 (Data Quality Audit)")
    n = len(df)
    results = []

    # (a) run_total_before == run_top_before + run_bot_before
    lhs = df["run_total_before"]
    rhs = df["run_top_before"] + df["run_bot_before"]
    match = (lhs == rhs)
    results.append(
        {
            "relationship": "run_total_before == run_top_before + run_bot_before",
            "matches": int(match.sum()),
            "mismatches": int((~match).sum()),
            "match_rate": match.mean(),
        }
    )

    # (b) score_diff_home == run_bot_before - run_top_before
    lhs2 = df["score_diff_home"]
    rhs2 = df["run_bot_before"] - df["run_top_before"]
    match2 = (lhs2 == rhs2)
    results.append(
        {
            "relationship": "score_diff_home == run_bot_before - run_top_before",
            "matches": int(match2.sum()),
            "mismatches": int((~match2).sum()),
            "match_rate": match2.mean(),
        }
    )

    # (c) score_diff_pitcher_team 재구성 — 두 가지 부호 가설을 모두 테스트하고
    # 실제로 어느 쪽이 맞는지 데이터로 확인한다 (추측으로 단정하지 않음).
    is_top = df["top_bottom"] == "T"
    hyp_a = np.where(is_top, df["score_diff_home"], -df["score_diff_home"])
    hyp_b = np.where(is_top, -df["score_diff_home"], df["score_diff_home"])
    match_a = (df["score_diff_pitcher_team"] == hyp_a)
    match_b = (df["score_diff_pitcher_team"] == hyp_b)
    results.append(
        {
            "relationship": "score_diff_pitcher_team == (top_bottom=='T' ? score_diff_home : -score_diff_home)",
            "matches": int(match_a.sum()),
            "mismatches": int((~match_a).sum()),
            "match_rate": match_a.mean(),
        }
    )
    results.append(
        {
            "relationship": "score_diff_pitcher_team == (top_bottom=='T' ? -score_diff_home : score_diff_home)",
            "matches": int(match_b.sum()),
            "mismatches": int((~match_b).sum()),
            "match_rate": match_b.mean(),
        }
    )

    rel_df = pd.DataFrame(results)
    print(rel_df.to_string(index=False))
    print(f"\n(n={n})")
    print("top_bottom 고유값:", sorted(df["top_bottom"].unique().tolist()))
    rel_df.to_csv(os.path.join(TAB_DIR, "01_derived_relationships.csv"), index=False)
    return rel_df


def step2b_legal_range_check(df):
    section("2b. 야구 규칙상 비정상적인 값 확인")
    checks = {
        "balls_before": (0, 3),
        "strikes_before": (0, 2),
        "outs_before": (0, 2),
    }
    rows = []
    for col, (lo, hi) in checks.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        rows.append(
            {
                "column": col,
                "expected_range": f"[{lo}, {hi}]",
                "n_out_of_range": len(bad),
                "out_of_range_values": sorted(bad[col].unique().tolist()) if len(bad) else [],
            }
        )
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    result.to_csv(os.path.join(TAB_DIR, "02_legal_range_check.csv"), index=False)
    return result


def step3_univariate(df):
    section("3. 단변량 통계")
    desc = df[ALL_COLS].describe().T
    desc["nunique"] = df[ALL_COLS].nunique()
    print(desc.to_string())
    desc.to_csv(os.path.join(TAB_DIR, "03_univariate_stats.csv"))

    # 카운트형: bar plot (value_counts)
    for col in COUNT_COLS:
        vc = df[col].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(vc.index.astype(str), vc.values, color="#4C72B0")
        ax.set_title(f"{col} distribution (n={len(df):,})")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"01_bar_{col}.png"), dpi=120)
        plt.close(fig)

    # 점수형: histogram + zoom histogram + boxplot
    for col in SCORE_COLS:
        s = df[col]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s, bins=40, color="#55A868")
        ax.set_title(f"{col} histogram (full range)")
        ax.set_xlabel(col)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"02_hist_{col}.png"), dpi=120)
        plt.close(fig)

        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s_zoom = s[(s >= lo) & (s <= hi)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s_zoom, bins=30, color="#C44E52")
        ax.set_title(f"{col} histogram (1%-99% zoom: [{lo:.0f}, {hi:.0f}])")
        ax.set_xlabel(col)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"03_hist_zoom_{col}.png"), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 5))
        ax.boxplot(s.dropna(), orientation="vertical")
        ax.set_title(f"{col} boxplot")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"04_box_{col}.png"), dpi=120)
        plt.close(fig)

    print(f"\n그래프 저장 완료: {FIG_DIR}")
    return desc


def step4_target_relation(df):
    section("4. Target(control_success) 관계 분석 — 값별 group-by")
    overall_rate = df[TARGET].mean()
    print(f"전체 control_success 평균: {overall_rate:.4f}")

    for col in ALL_COLS:
        g = df.groupby(col)[TARGET].agg(n="count", success_rate="mean").reset_index()
        g["diff_vs_overall"] = g["success_rate"] - overall_rate
        g = g.sort_values(col)
        out_path = os.path.join(TAB_DIR, f"04_target_by_{col}.csv")
        g.to_csv(out_path, index=False)
        print(f"\n[{col}] (저장: {out_path})")
        print(g.to_string(index=False))
    return overall_rate


def step5_count_analysis(df, overall_rate):
    section("5. Count 분석 (balls_before x strikes_before)")
    pivot_n = df.pivot_table(
        index="strikes_before", columns="balls_before", values=TARGET, aggfunc="count"
    )
    pivot_rate = df.pivot_table(
        index="strikes_before", columns="balls_before", values=TARGET, aggfunc="mean"
    )
    print("표본 수:")
    print(pivot_n.to_string())
    print("\nsuccess rate:")
    print(pivot_rate.to_string())
    pivot_n.to_csv(os.path.join(TAB_DIR, "05_count_bxs_n.csv"))
    pivot_rate.to_csv(os.path.join(TAB_DIR, "05_count_bxs_success_rate.csv"))

    # heatmaps
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot_rate.values, cmap="RdBu_r", vmin=overall_rate - 0.05, vmax=overall_rate + 0.05)
    ax.set_xticks(range(len(pivot_rate.columns)))
    ax.set_xticklabels(pivot_rate.columns)
    ax.set_yticks(range(len(pivot_rate.index)))
    ax.set_yticklabels(pivot_rate.index)
    ax.set_xlabel("balls_before")
    ax.set_ylabel("strikes_before")
    ax.set_title(f"success rate by count (overall={overall_rate:.3f})")
    for i in range(pivot_rate.shape[0]):
        for j in range(pivot_rate.shape[1]):
            v = pivot_rate.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_heatmap_count_success_rate.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot_n.values, cmap="viridis")
    ax.set_xticks(range(len(pivot_n.columns)))
    ax.set_xticklabels(pivot_n.columns)
    ax.set_yticks(range(len(pivot_n.index)))
    ax.set_yticklabels(pivot_n.index)
    ax.set_xlabel("balls_before")
    ax.set_ylabel("strikes_before")
    ax.set_title("sample size by count")
    for i in range(pivot_n.shape[0]):
        for j in range(pivot_n.shape[1]):
            v = pivot_n.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="n")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_heatmap_count_sample_size.png"), dpi=120)
    plt.close(fig)

    section("5b. 특정 카운트 상황 (0-0, 0-2, 3-0, 3-1, 3-2)")
    rows = []
    for b, s in FLAGGED_COUNTS:
        mask = (df["balls_before"] == b) & (df["strikes_before"] == s)
        n = mask.sum()
        rate = df.loc[mask, TARGET].mean() if n else np.nan
        rows.append(
            {
                "count": f"{b}-{s}",
                "n": int(n),
                "success_rate": rate,
                "diff_vs_overall": (rate - overall_rate) if n else np.nan,
            }
        )
    flagged = pd.DataFrame(rows)
    print(flagged.to_string(index=False))
    flagged.to_csv(os.path.join(TAB_DIR, "05_flagged_counts.csv"), index=False)

    section("5c. 시즌별 count success-rate (전체 balls x strikes)")
    season_count = (
        df.groupby(["season", "balls_before", "strikes_before"])[TARGET]
        .agg(n="count", success_rate="mean")
        .reset_index()
    )
    season_count.to_csv(os.path.join(TAB_DIR, "05_count_by_season_full.csv"), index=False)
    print(f"(전체 시즌x카운트 표는 05_count_by_season_full.csv 로 저장, {len(season_count)}행)")

    section("5d. 시즌별 특정 카운트 success-rate 비교표")
    rows = []
    for b, s in FLAGGED_COUNTS:
        mask = (df["balls_before"] == b) & (df["strikes_before"] == s)
        sub = df.loc[mask]
        g = sub.groupby("season")[TARGET].agg(n="count", success_rate="mean")
        for season, r in g.iterrows():
            rows.append(
                {"count": f"{b}-{s}", "season": season, "n": int(r["n"]), "success_rate": r["success_rate"]}
            )
    flagged_season = pd.DataFrame(rows)
    pivot_flagged_season = flagged_season.pivot(index="season", columns="count", values="success_rate")
    print(pivot_flagged_season.to_string())
    pivot_flagged_season.to_csv(os.path.join(TAB_DIR, "05_flagged_counts_by_season.csv"))
    return pivot_rate, pivot_n, flagged


def step6_outs_analysis(df, overall_rate):
    section("6. Outs 분석")
    g = df.groupby("outs_before")[TARGET].agg(n="count", success_rate="mean").reset_index()
    g["diff_vs_overall"] = g["success_rate"] - overall_rate
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "06_outs_summary.csv"), index=False)

    section("6b. balls x strikes x outs 3-way")
    g3 = (
        df.groupby(["balls_before", "strikes_before", "outs_before"])[TARGET]
        .agg(n="count", success_rate="mean")
        .reset_index()
    )
    g3["diff_vs_overall"] = g3["success_rate"] - overall_rate
    SMALL_N_THRESHOLD = 500
    g3["small_sample"] = g3["n"] < SMALL_N_THRESHOLD
    g3 = g3.sort_values(["balls_before", "strikes_before", "outs_before"])
    print(g3.to_string(index=False))
    print(f"\n표본 {SMALL_N_THRESHOLD}건 미만 조합 수: {int(g3['small_sample'].sum())} / {len(g3)}")
    g3.to_csv(os.path.join(TAB_DIR, "06_count_outs_3way.csv"), index=False)
    return g, g3


def step7_score_analysis(df, overall_rate):
    section("7. Score 분석")

    for col in ["score_diff_pitcher_team", "score_diff_home"]:
        g = df.groupby(col)[TARGET].agg(n="count", success_rate="mean").reset_index()
        g["diff_vs_overall"] = g["success_rate"] - overall_rate
        g = g.sort_values(col)
        print(f"\n[{col}] raw value group-by (unique={df[col].nunique()})")
        print(g.to_string(index=False))
        g.to_csv(os.path.join(TAB_DIR, f"07_{col}_raw.csv"), index=False)

    section("7b. 점수차 구간화 (단순/해석 가능한 기준, 최적화 아님)")
    df = df.copy()
    df["score_bin_pitcher"] = pd.cut(
        df["score_diff_pitcher_team"], bins=SCORE_BIN_EDGES, labels=SCORE_BIN_LABELS
    )
    gb = (
        df.groupby("score_bin_pitcher", observed=True)[TARGET]
        .agg(n="count", success_rate="mean")
        .reset_index()
    )
    gb["diff_vs_overall"] = gb["success_rate"] - overall_rate
    print(gb.to_string(index=False))
    gb.to_csv(os.path.join(TAB_DIR, "07_score_diff_pitcher_bins.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(gb["score_bin_pitcher"].astype(str), gb["success_rate"], color="#8172B2")
    ax.axhline(overall_rate, color="red", linestyle="--", label=f"overall={overall_rate:.3f}")
    ax.set_ylabel("success rate")
    ax.set_title("score_diff_pitcher_team bins vs success rate")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_score_bins_success_rate.png"), dpi=120)
    plt.close(fig)

    section("7c. EDA 전용 파생 후보 (모델 피처로 확정 아님)")
    df["abs_score_diff"] = df["score_diff_pitcher_team"].abs()
    df["score_state"] = np.select(
        [df["score_diff_pitcher_team"] > 0, df["score_diff_pitcher_team"] == 0, df["score_diff_pitcher_team"] < 0],
        ["leading", "tied", "trailing"],
    )
    df["is_lopsided"] = (df["abs_score_diff"] >= 4).astype(int)

    for col in ["score_state", "is_lopsided"]:
        g = df.groupby(col)[TARGET].agg(n="count", success_rate="mean").reset_index()
        g["diff_vs_overall"] = g["success_rate"] - overall_rate
        print(f"\n[{col}]")
        print(g.to_string(index=False))
        g.to_csv(os.path.join(TAB_DIR, f"07_{col}.csv"), index=False)

    return df


def step8_season_drift(df, overall_rate):
    section("8. 시즌별 안정성 (Season Drift)")
    g = df.groupby("season")[TARGET].agg(n="count", success_rate="mean").reset_index()
    g["diff_vs_overall"] = g["success_rate"] - overall_rate
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "08_season_overall.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["season"], g["success_rate"], marker="o", color="#4C72B0")
    ax.axhline(overall_rate, color="red", linestyle="--", label=f"overall={overall_rate:.3f}")
    ax.set_xlabel("season")
    ax.set_ylabel("success rate")
    ax.set_title("season별 전체 control_success 평균")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_season_overall_trend.png"), dpi=120)
    plt.close(fig)

    section("8b. 시즌별 score-bin success rate")
    gb = (
        df.groupby(["season", "score_bin_pitcher"], observed=True)[TARGET]
        .agg(n="count", success_rate="mean")
        .reset_index()
    )
    pivot = gb.pivot(index="season", columns="score_bin_pitcher", values="success_rate")
    print(pivot.to_string())
    pivot.to_csv(os.path.join(TAB_DIR, "08_season_score_bin.csv"))

    fig, ax = plt.subplots(figsize=(7, 4))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=str(col))
    ax.set_xlabel("season")
    ax.set_ylabel("success rate")
    ax.set_title("시즌별 score_diff_pitcher_team 구간 success rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "09_season_score_bin_trend.png"), dpi=120)
    plt.close(fig)

    return g, pivot


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_data()
    step1_sanity_check(df)
    step2_derived_relationships(df)
    step2b_legal_range_check(df)
    step3_univariate(df)
    overall_rate = step4_target_relation(df)
    step5_count_analysis(df, overall_rate)
    step6_outs_analysis(df, overall_rate)
    df = step7_score_analysis(df, overall_rate)
    step8_season_drift(df, overall_rate)

    section("완료")
    print(f"그래프: {FIG_DIR}")
    print(f"표: {TAB_DIR}")


if __name__ == "__main__":
    main()
