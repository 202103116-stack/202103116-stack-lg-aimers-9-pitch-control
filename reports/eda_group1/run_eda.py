"""Group 1 EDA — 시간·경기 정보, 손 유형·팀 ID, 선수 ID (12개 컬럼).

담당 컬럼 (원 담당자: 허원준 — 이 스크립트는 원본 노트를 재검증/재구성한 버전):
    season, game_month, game_dayofweek, inning, top_bottom, game_type,
    pitcher_id, batter_id, pitcher_hand, batter_hand,
    pitcher_team_id, batter_team_id

이 스크립트는 train.csv만 읽고(test.csv는 사용하지 않음), 행 단위 통계와
group-by 집계만 수행한다. "선수 ID/팀 ID별 누적 성공률" 같은 값은 train.csv
전체를 groupby한 EDA 전용 집계이며, 이 자체를 test.csv 피처로 옮기면
CLAUDE.md의 cross-row 누수 규칙을 위반한다 — 실제 피처로 쓰려면 반드시
공식 asof_* 컬럼(투구 시점까지만 누적)을 써야 한다. 이 구분은 README에도
명시한다.

다른 그룹 컬럼(li, score_diff_pitcher_team, asof_pitcher_* 등)과의 상호작용은
여기서 다루지 않는다 — 그런 분석은 그룹 4(투수 asof_* 지표)와 최종 통합 EDA
리포트(reports/eda_final/)에서 다룬다. 이 리포트는 그룹 1의 12개 컬럼 자체와
target의 관계, 그리고 그룹 1 컬럼끼리의 상호작용(season x game_type,
pitcher_hand x batter_hand 등)에 집중한다.

실행:
    python reports/eda_group1/run_eda.py

산출물:
    reports/eda_group1/figures/*.png
    reports/eda_group1/tables/*.csv
    (표준출력을 reports/eda_group1/eda_run_log.txt 로 리다이렉트해서 보관 —
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

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = "./data"
OUT_DIR = "./reports/eda_group1"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

TIME_COLS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type"]
HAND_TEAM_COLS = ["pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]
ID_COLS = ["pitcher_id", "batter_id"]
ALL_COLS = TIME_COLS + HAND_TEAM_COLS + ID_COLS
TARGET = "control_success"

SAMPLE_SIZE_THRESHOLD = 200  # ID 안정성 판단 기준 (원본 노트와 동일 기준 사용)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data():
    usecols = ALL_COLS + [TARGET, "asof_pitcher_n"]
    df = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=usecols
    )
    return df


def step1_quality(df):
    section("1. 데이터 품질 확인")
    print(f"train.csv shape: {df.shape}")
    missing = [c for c in ALL_COLS + [TARGET] if c not in df.columns]
    print(f"누락된 컬럼: {missing if missing else '없음 (전부 존재)'}")

    rows = []
    for c in ALL_COLS:
        s = df[c]
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_null": int(s.isna().sum()),
                "n_unique": int(s.nunique()),
                "min": s.min() if pd.api.types.is_numeric_dtype(s) else "",
                "max": s.max() if pd.api.types.is_numeric_dtype(s) else "",
            }
        )
    quality = pd.DataFrame(rows)
    print(quality.to_string(index=False))
    quality.to_csv(os.path.join(TAB_DIR, "00_quality_summary.csv"), index=False)

    dup = int(df.duplicated().sum())
    print(f"\n완전 중복행: {dup}건")
    return quality


def step2_univariate_grids(df, overall_rate):
    section("2. 단변량 EDA — 그룹별 그리드")

    # --- Grid A: 시간·경기 정보 6개 (2x3) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    tables = {}
    for ax, col in zip(axes.flat, TIME_COLS):
        g = df.groupby(col)[TARGET].agg(n="count", success_rate="mean").reset_index()
        g = g.sort_values(col)
        tables[col] = g
        ax.plot(g[col].astype(str), g["success_rate"] * 100, marker="o", color="#4C72B0")
        ax.axhline(overall_rate * 100, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(col)
        ax.set_ylabel("success rate (%)")
        ax.tick_params(axis="x", rotation=45)
        g.to_csv(os.path.join(TAB_DIR, f"02_target_by_{col}.csv"), index=False)
    fig.suptitle(f"Group 1 — 시간·경기 정보 6개 (전체 평균={overall_rate * 100:.2f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_grid_time_info.png"), dpi=120)
    plt.close(fig)

    for col in TIME_COLS:
        g = tables[col]
        vmax = g["success_rate"].max()
        vmin = g["success_rate"].min()
        print(f"\n[{col}] n_levels={len(g)}, 값간 최대차이={100*(vmax-vmin):.4f}%p")
        print(g.to_string(index=False))

    # --- Grid B: 손 유형·팀 ID 4개 (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.flat, HAND_TEAM_COLS):
        g = df.groupby(col)[TARGET].agg(n="count", success_rate="mean").reset_index()
        g = g.sort_values(col)
        ax.plot(g[col].astype(str), g["success_rate"] * 100, marker="o", color="#55A868")
        ax.axhline(overall_rate * 100, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(col)
        ax.set_ylabel("success rate (%)")
        ax.tick_params(axis="x", rotation=90 if col.endswith("team_id") else 0)
        g.to_csv(os.path.join(TAB_DIR, f"02_target_by_{col}.csv"), index=False)
        print(f"\n[{col}] n_levels={len(g)}")
        print(g.to_string(index=False))
    fig.suptitle(f"Group 1 — 손 유형·팀 ID 4개 (전체 평균={overall_rate * 100:.2f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_grid_hand_team.png"), dpi=120)
    plt.close(fig)

    # --- 선수 ID 2개: n vs rate 산점도 (1x2) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    id_summary = {}
    for ax, col in zip(axes, ID_COLS):
        g = df.groupby(col)[TARGET].agg(n="count", success_rate="mean")
        id_summary[col] = g
        ax.scatter(g["n"], g["success_rate"] * 100, alpha=0.4, s=12, color="#4C72B0")
        ax.set_xscale("log")
        ax.axvline(SAMPLE_SIZE_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(overall_rate * 100, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(f"{col}: pitches per ID (log scale)")
        ax.set_ylabel("success rate (%)")
        ax.set_title(f"{col} (n_unique={df[col].nunique()})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_scatter_player_id.png"), dpi=120)
    plt.close(fig)

    section("2b. 선수 ID 표본 크기 요약")
    rows = []
    for col in ID_COLS:
        g = id_summary[col]
        small = g[g["n"] < 100]
        big = g[g["n"] >= SAMPLE_SIZE_THRESHOLD]
        rows.append(
            {
                "id_col": col,
                "n_unique_id": len(g),
                "median_rows_per_id": g["n"].median(),
                "n_id_under_100": len(small),
                f"n_id_over_{SAMPLE_SIZE_THRESHOLD}": len(big),
                f"success_rate_p10_over_{SAMPLE_SIZE_THRESHOLD}": big["success_rate"].quantile(0.1),
                f"success_rate_median_over_{SAMPLE_SIZE_THRESHOLD}": big["success_rate"].median(),
                f"success_rate_p90_over_{SAMPLE_SIZE_THRESHOLD}": big["success_rate"].quantile(0.9),
            }
        )
    id_stat = pd.DataFrame(rows)
    print(id_stat.to_string(index=False))
    id_stat.to_csv(os.path.join(TAB_DIR, "02_id_sample_summary.csv"), index=False)
    return id_summary


def step3_season_game_type(df, overall_rate):
    section("3. season x game_type 상호작용")
    g = df.groupby(["season", "game_type"])[TARGET].agg(n="count", success_rate="mean").reset_index()
    pivot_rate = g.pivot(index="season", columns="game_type", values="success_rate")
    pivot_n = g.pivot(index="season", columns="game_type", values="n")
    print("success rate:")
    print((pivot_rate * 100).round(4).to_string())
    print("\nsample size:")
    print(pivot_n.to_string())
    pivot_rate.to_csv(os.path.join(TAB_DIR, "03_season_gametype_rate.csv"))
    pivot_n.to_csv(os.path.join(TAB_DIR, "03_season_gametype_n.csv"))

    diff = (pivot_rate.get("F") - pivot_rate.get("R")) * 100 if "F" in pivot_rate.columns and "R" in pivot_rate.columns else None
    if diff is not None:
        print("\nF - R (%p):")
        print(diff.round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in pivot_rate.columns:
        ax.plot(pivot_rate.index, pivot_rate[col] * 100, marker="o", label=str(col))
    ax.set_xlabel("season")
    ax.set_ylabel("success rate (%)")
    ax.set_title("game_type effect changes across seasons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_season_gametype_trend.png"), dpi=120)
    plt.close(fig)
    return pivot_rate, pivot_n


def step4_season_month_heatmap(df):
    section("4. season x game_month heatmap")
    g = df.groupby(["season", "game_month"])[TARGET].agg(n="count", success_rate="mean").reset_index()
    pivot_rate = g.pivot(index="season", columns="game_month", values="success_rate")
    pivot_n = g.pivot(index="season", columns="game_month", values="n")
    print("success rate (%):")
    print((pivot_rate * 100).round(2).to_string())
    print("\nsample size:")
    print(pivot_n.to_string())
    pivot_rate.to_csv(os.path.join(TAB_DIR, "04_season_month_rate.csv"))
    pivot_n.to_csv(os.path.join(TAB_DIR, "04_season_month_n.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axes[0].imshow(pivot_rate.values * 100, cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(range(len(pivot_rate.columns)))
    axes[0].set_xticklabels(pivot_rate.columns)
    axes[0].set_yticks(range(len(pivot_rate.index)))
    axes[0].set_yticklabels(pivot_rate.index)
    axes[0].set_title("season x month: success rate (%)")
    for i in range(pivot_rate.shape[0]):
        for j in range(pivot_rate.shape[1]):
            v = pivot_rate.values[i, j]
            if not np.isnan(v):
                axes[0].text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pivot_n.values, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(pivot_n.columns)))
    axes[1].set_xticklabels(pivot_n.columns)
    axes[1].set_yticks(range(len(pivot_n.index)))
    axes[1].set_yticklabels(pivot_n.index)
    axes[1].set_title("season x month: sample size")
    for i in range(pivot_n.shape[0]):
        for j in range(pivot_n.shape[1]):
            v = pivot_n.values[i, j]
            if not np.isnan(v):
                axes[1].text(j, i, f"{int(v/1000)}k", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_heatmap_season_month.png"), dpi=120)
    plt.close(fig)
    return pivot_rate, pivot_n


def step5_inning_analysis(df, overall_rate):
    section("5. 이닝 구간별 성공률")
    bins = [0, 3, 6, 9, 99]
    labels = ["1-3회", "4-6회", "7-9회", "연장"]
    df = df.copy()
    df["inning_bin"] = pd.cut(df["inning"], bins=bins, labels=labels)
    g = df.groupby("inning_bin", observed=True)[TARGET].agg(n="count", success_rate="mean").reset_index()
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "05_inning_bin_summary.csv"), index=False)

    section("5b. 동일 투수 1-3회 vs 7-9회 비교 (각 구간 200구 이상)")
    early = df[df["inning_bin"] == "1-3회"].groupby("pitcher_id")[TARGET].agg(n="count", rate="mean")
    late = df[df["inning_bin"] == "7-9회"].groupby("pitcher_id")[TARGET].agg(n="count", rate="mean")
    both = early[early["n"] >= SAMPLE_SIZE_THRESHOLD].join(
        late[late["n"] >= SAMPLE_SIZE_THRESHOLD], lsuffix="_early", rsuffix="_late", how="inner"
    )
    both["diff_late_minus_early"] = both["rate_late"] - both["rate_early"]
    n_pitchers = len(both)
    pct_lower_late = (both["diff_late_minus_early"] < 0).mean() * 100 if n_pitchers else np.nan
    mean_diff = both["diff_late_minus_early"].mean() if n_pitchers else np.nan
    print(f"비교 대상 투수 수: {n_pitchers}")
    print(f"7-9회 성공률이 1-3회보다 낮은 투수 비율: {pct_lower_late:.2f}%")
    print(f"평균 차이 (late - early): {mean_diff*100:.4f}%p")
    both.to_csv(os.path.join(TAB_DIR, "05_inning_same_pitcher_compare.csv"))
    return g, both


def step6_hand_matchup(df, overall_rate):
    section("6. pitcher_hand x batter_hand 매치업")
    g = df.groupby(["pitcher_hand", "batter_hand"])[TARGET].agg(n="count", success_rate="mean").reset_index()
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "06_hand_matchup.csv"), index=False)

    same = g[g["pitcher_hand"] == g["batter_hand"]]
    diff = g[g["pitcher_hand"] != g["batter_hand"]]
    same_rate = (same["n"] * same["success_rate"]).sum() / same["n"].sum()
    diff_rate = (diff["n"] * diff["success_rate"]).sum() / diff["n"].sum()
    print(f"\n같은 손 유형 조합 가중평균 성공률: {same_rate*100:.4f}%")
    print(f"다른 손 유형 조합 가중평균 성공률: {diff_rate*100:.4f}%")
    print(f"조합 간 최대-최소 차이: {(g['success_rate'].max()-g['success_rate'].min())*100:.4f}%p")

    fig, ax = plt.subplots(figsize=(5, 4))
    pivot = g.pivot(index="pitcher_hand", columns="batter_hand", values="success_rate")
    im = ax.imshow(pivot.values * 100, cmap="RdBu_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("batter_hand")
    ax.set_ylabel("pitcher_hand")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]*100:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="success rate (%)")
    ax.set_title("pitcher_hand x batter_hand success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_heatmap_hand_matchup.png"), dpi=120)
    plt.close(fig)
    return g


def step7_team_analysis(df, overall_rate):
    section("7. 팀 ID 분석")
    g_p = df.groupby("pitcher_team_id")[TARGET].agg(n="count", success_rate="mean").reset_index()
    g_b = df.groupby("batter_team_id")[TARGET].agg(n="count", success_rate="mean").reset_index()
    print("[pitcher_team_id]")
    print(g_p.to_string(index=False))
    print("\n[batter_team_id]")
    print(g_b.to_string(index=False))
    g_p.to_csv(os.path.join(TAB_DIR, "07_pitcher_team_summary.csv"), index=False)
    g_b.to_csv(os.path.join(TAB_DIR, "07_batter_team_summary.csv"), index=False)

    section("7b. pitcher_team_id x batter_team_id 조합")
    g2 = df.groupby(["pitcher_team_id", "batter_team_id"])[TARGET].agg(n="count", success_rate="mean").reset_index()
    n_possible = df["pitcher_team_id"].nunique() * df["batter_team_id"].nunique()
    same_team = (g2["pitcher_team_id"] == g2["batter_team_id"]).sum()
    print(f"관측된 조합 수: {len(g2)} / 가능한 최대 조합 수: {n_possible}")
    print(f"투수팀==타자팀인 관측 조합 수: {same_team}")
    g2.to_csv(os.path.join(TAB_DIR, "07_team_matchup.csv"), index=False)

    pivot_rate = g2.pivot(index="pitcher_team_id", columns="batter_team_id", values="success_rate")
    pivot_n = g2.pivot(index="pitcher_team_id", columns="batter_team_id", values="n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = axes[0].imshow(pivot_rate.values * 100, cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(range(len(pivot_rate.columns))); axes[0].set_xticklabels(pivot_rate.columns, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(pivot_rate.index))); axes[0].set_yticklabels(pivot_rate.index, fontsize=7)
    axes[0].set_title("pitcher x batter team: success rate (%)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(pivot_n.values, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(pivot_n.columns))); axes[1].set_xticklabels(pivot_n.columns, rotation=90, fontsize=7)
    axes[1].set_yticks(range(len(pivot_n.index))); axes[1].set_yticklabels(pivot_n.index, fontsize=7)
    axes[1].set_title("pitcher x batter team: sample size")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_heatmap_team_matchup.png"), dpi=120)
    plt.close(fig)

    section("7c. 선수 이동 (팀 변경)")
    per_pitcher_teams = df.groupby("pitcher_id")["pitcher_team_id"].nunique()
    n_multi_team = int((per_pitcher_teams > 1).sum())
    n_total_pitchers = len(per_pitcher_teams)
    per_pitcher_season_teams = df.groupby(["pitcher_id", "season"])["pitcher_team_id"].nunique()
    n_multi_team_season = int((per_pitcher_season_teams > 1).sum())
    print(f"여러 팀에서 관측된 투수 수: {n_multi_team} / {n_total_pitchers} ({n_multi_team/n_total_pitchers*100:.2f}%)")
    print(f"같은 시즌 안에서 여러 팀으로 관측된 투수-시즌 조합: {n_multi_team_season}건")
    return g_p, g_b, g2


def step8_player_season_novelty(df):
    section("8. 시즌별 신규 선수 비율")
    rows = []
    seen_pitchers, seen_batters = set(), set()
    for season in sorted(df["season"].unique()):
        sub = df[df["season"] == season]
        cur_pitchers = set(sub["pitcher_id"].unique())
        cur_batters = set(sub["batter_id"].unique())
        new_pitchers = cur_pitchers - seen_pitchers
        new_batters = cur_batters - seen_batters
        new_pitcher_rows = sub["pitcher_id"].isin(new_pitchers).sum()
        new_batter_rows = sub["batter_id"].isin(new_batters).sum()
        rows.append(
            {
                "season": season,
                "n_rows": len(sub),
                "new_pitcher_row_pct": new_pitcher_rows / len(sub) * 100,
                "new_batter_row_pct": new_batter_rows / len(sub) * 100,
            }
        )
        seen_pitchers |= cur_pitchers
        seen_batters |= cur_batters
    novelty = pd.DataFrame(rows)
    print(novelty.to_string(index=False))
    novelty.to_csv(os.path.join(TAB_DIR, "08_player_season_novelty.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(novelty["season"], novelty["new_pitcher_row_pct"], marker="o", label="신규 투수 투구 비율")
    ax.plot(novelty["season"], novelty["new_batter_row_pct"], marker="o", label="신규 타자 상대 투구 비율")
    ax.set_xlabel("season")
    ax.set_ylabel("share of pitches (%)")
    ax.set_title("시즌별 신규 선수 비중")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_new_player_share.png"), dpi=120)
    plt.close(fig)
    return novelty


def step9_weekday_gametype(df):
    section("9. game_dayofweek x game_type (우선순위 낮은 조합 검증)")
    g = df.groupby(["game_dayofweek", "game_type"])[TARGET].agg(n="count", success_rate="mean").reset_index()
    pivot_rate = g.pivot(index="game_dayofweek", columns="game_type", values="success_rate")
    pivot_n = g.pivot(index="game_dayofweek", columns="game_type", values="n")
    print("success rate (%):")
    print((pivot_rate * 100).round(2).to_string())
    print("\nsample size:")
    print(pivot_n.to_string())
    pivot_rate.to_csv(os.path.join(TAB_DIR, "09_weekday_gametype_rate.csv"))
    pivot_n.to_csv(os.path.join(TAB_DIR, "09_weekday_gametype_n.csv"))
    return pivot_rate, pivot_n


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_data()
    overall_rate = df[TARGET].mean()
    print(f"전체 행 수: {len(df):,}  |  전체 control_success 평균: {overall_rate*100:.4f}%")
    print(f"고유 pitcher_id 수: {df['pitcher_id'].nunique()}  |  고유 batter_id 수: {df['batter_id'].nunique()}")

    step1_quality(df)
    step2_univariate_grids(df, overall_rate)
    step3_season_game_type(df, overall_rate)
    step4_season_month_heatmap(df)
    step5_inning_analysis(df, overall_rate)
    step6_hand_matchup(df, overall_rate)
    step7_team_analysis(df, overall_rate)
    step8_player_season_novelty(df)
    step9_weekday_gametype(df)

    section("완료")
    print(f"그래프: {FIG_DIR}")
    print(f"표: {TAB_DIR}")


if __name__ == "__main__":
    main()
