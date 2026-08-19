"""최종 통합 EDA — 그룹 1~5 컬럼 간 교차 분석.

각 그룹 리포트(reports/eda_group1~5)는 자기 그룹 컬럼 + target 관계에
집중했고, 다른 그룹 컬럼과의 상호작용은 여기로 미뤘다. 이 스크립트는
그 교차 분석만 다룬다 — 그룹별 리포트에서 이미 검증된 단변량 수치는
반복하지 않는다.

1라운드(A~E)는 그룹 1~4만으로 작성됐고, 2라운드(F~H)에서 그룹 5
(asof_batter_*, asof_pitcher_prev*_rate — 담당자 문서 없이 직접 EDA함,
reports/eda_group5/ 참고)를 추가했다.

실행:
    python reports/eda_final/run_eda.py
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
OUT_DIR = "./reports/eda_final"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
TARGET = "control_success"

USECOLS = [
    # group1
    "season", "inning", "pitcher_hand", "batter_hand", "pitcher_team_id",
    # group2
    "balls_before", "strikes_before", "outs_before", "score_diff_pitcher_team",
    # group3
    "runner_on_2b", "runner_on_3b", "li",
    # group4
    "asof_pitcher_success_rate", "asof_pitcher_n",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    # group5
    "asof_batter_success_rate", "asof_batter_n",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    TARGET,
]


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=USECOLS)
    df["inning_bin"] = pd.cut(df["inning"], bins=[0, 3, 6, 9, 99], labels=["1-3회", "4-6회", "7-9회", "연장"])
    df["runner_in_scoring_position"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(int)
    df["score_state"] = np.select(
        [df["score_diff_pitcher_team"] > 0, df["score_diff_pitcher_team"] == 0, df["score_diff_pitcher_team"] < 0],
        ["리드", "동점", "열세"],
        default="",
    )
    df["full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["three_ball"] = (df["balls_before"] == 3).astype(int)

    p1 = df["asof_pitcher_prev1_game_success_rate"]
    p3 = df["asof_pitcher_prev3_game_success_rate"]
    p5 = df["asof_pitcher_prev5_game_success_rate"]
    early_only = (5 * p5 - 3 * p3) / 2
    df["pitcher_form_trend_isolated"] = p1 - early_only
    return df


def A_inning_li_score_runner(df, overall_rate):
    section("A. inning_bin x li분위 x (score_state x 득점권주자) 4-way")
    d = df.copy()
    d["li_q"] = d.groupby("inning_bin", observed=True)["li"].transform(
        lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    )
    d["col_label"] = d["score_state"] + "/득점권" + d["runner_in_scoring_position"].astype(str)
    g = d.groupby(["inning_bin", "li_q", "col_label"], observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    g.to_csv(os.path.join(TAB_DIR, "A_inning_li_score_runner.csv"), index=False)

    g["row_label"] = g["inning_bin"].astype(str) + "/" + g["li_q"].astype(str)
    pivot_rate = g.pivot(index="row_label", columns="col_label", values="rate")
    pivot_n = g.pivot(index="row_label", columns="col_label", values="n")
    row_order = [f"{ib}/{q}" for ib in ["1-3회", "4-6회", "7-9회", "연장"] for q in ["Q1", "Q2", "Q3", "Q4"]]
    pivot_rate = pivot_rate.reindex(row_order)
    pivot_n = pivot_n.reindex(row_order)
    print(pivot_rate.round(4).to_string())
    pivot_rate.to_csv(os.path.join(TAB_DIR, "A_pivot_rate.csv"))
    pivot_n.to_csv(os.path.join(TAB_DIR, "A_pivot_n.csv"))

    # 이닝 구간별 평균(li/score/runner 무관) -- LI가 통제돼도 이닝 효과가 남는지 확인
    by_inning = g.groupby("inning_bin", observed=True).apply(
        lambda x: (x["n"] * x["rate"]).sum() / x["n"].sum(), include_groups=False
    )
    print("\n이닝 구간별 (li분위x상황 통제 후) 가중평균 성공률:")
    print(by_inning.round(4).to_string())

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(pivot_rate.values * 100, cmap="RdBu_r", aspect="auto",
                    vmin=overall_rate * 100 - 4, vmax=overall_rate * 100 + 4)
    ax.set_xticks(range(len(pivot_rate.columns))); ax.set_xticklabels(pivot_rate.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot_rate.index))); ax.set_yticklabels(pivot_rate.index, fontsize=7)
    ax.set_title("inning x li quartile x (score_state x scoring position): success rate (%)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "A_heatmap_inning_li_score_runner.png"), dpi=120)
    plt.close(fig)
    return pivot_rate, pivot_n, by_inning


def B_hand_matchup_pitchmix(df):
    section("B. pitcher_hand x batter_hand x 과거 구종군 비율")
    d = df.copy()
    d["hand_matchup"] = d["pitcher_hand"].astype(str) + "-" + d["batter_hand"].astype(str)
    mix_cols = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    all_rows = []
    for ax, col in zip(axes, mix_cols):
        sub = d[[col, "hand_matchup", TARGET]].dropna().copy()
        sub["q"] = sub.groupby("hand_matchup")[col].transform(
            lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        )
        g = sub.groupby(["hand_matchup", "q"], observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
        g["mix_col"] = col
        all_rows.append(g)
        for hm in sorted(g["hand_matchup"].unique()):
            gg = g[g["hand_matchup"] == hm]
            ax.plot(gg["q"].astype(str), gg["rate"] * 100, marker="o", label=hm)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("success rate (%)")
    axes[0].legend(fontsize=7, title="hand_matchup")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "B_hand_pitchmix_trend.png"), dpi=120)
    plt.close(fig)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(os.path.join(TAB_DIR, "B_hand_pitchmix.csv"), index=False)
    print(f"관측된 (hand_matchup x mix_col x quartile) 조합 수: {len(full)}, 표본 최소값: {full['n'].min()}")
    print(f"성공률 범위: {full['rate'].min()*100:.2f}% ~ {full['rate'].max()*100:.2f}%")
    return full


def C_count_li(df, overall_rate):
    section("C. balls x strikes x li분위")
    d = df.copy()
    d["li_q"] = pd.qcut(d["li"], 4, labels=["Q1(낮음)", "Q2", "Q3", "Q4(높음)"], duplicates="drop")
    g = d.groupby(["strikes_before", "balls_before", "li_q"], observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    g.to_csv(os.path.join(TAB_DIR, "C_count_li.csv"), index=False)

    fc = d[d["full_count"] == 1].groupby("li_q", observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    print("풀카운트(3-2) 한정, li 분위별:")
    print(fc.to_string(index=False))
    fc.to_csv(os.path.join(TAB_DIR, "C_fullcount_by_li.csv"), index=False)

    non_fc = d[d["full_count"] == 0].groupby("li_q", observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    print("\n비-풀카운트, li 분위별:")
    print(non_fc.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(fc["li_q"].astype(str), fc["rate"] * 100, marker="o", label="풀카운트(3-2)")
    ax.plot(non_fc["li_q"].astype(str), non_fc["rate"] * 100, marker="o", label="그 외 카운트")
    ax.axhline(overall_rate * 100, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("success rate (%)")
    ax.set_title("풀카운트 페널티가 li(중요도)에 따라 커지는지")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "C_fullcount_li_trend.png"), dpi=120)
    plt.close(fig)
    return g, fc, non_fc


def D_outs_runner(df, overall_rate):
    section("D. outs_before x runner_in_scoring_position x three_ball")
    g = df.groupby(["outs_before", "runner_in_scoring_position", "three_ball"])[TARGET].agg(n="count", rate="mean").reset_index()
    g["diff_vs_overall"] = g["rate"] - overall_rate
    g = g.sort_values(["outs_before", "runner_in_scoring_position", "three_ball"])
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "D_outs_runner_threeball.csv"), index=False)

    pivot = g[g["three_ball"] == 1].pivot(index="outs_before", columns="runner_in_scoring_position", values="rate")
    print("\n3볼 상황 한정, outs x 득점권주자:")
    print((pivot * 100).round(2).to_string())
    return g


def E_season_pitcher_quality(df):
    section("E. season x asof_pitcher_success_rate 분위 — 시즌 하락이 투수 구성 변화 때문인지")
    d = df[df["asof_pitcher_success_rate"].notna()].copy()
    d["quality_q"] = pd.qcut(d["asof_pitcher_success_rate"], 5, labels=["Q1(낮음)", "Q2", "Q3", "Q4", "Q5(높음)"], duplicates="drop")
    g = d.groupby(["season", "quality_q"], observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    pivot = g.pivot(index="season", columns="quality_q", values="rate")
    print((pivot * 100).round(2).to_string())
    pivot.to_csv(os.path.join(TAB_DIR, "E_season_quality_pivot.csv"))

    # 투수 품질 분위 구성비가 시즌별로 바뀌는지(=투수 구성 자체가 바뀌는지)
    comp = d.groupby(["season", "quality_q"], observed=True).size().unstack()
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    print("\n시즌별 분위 구성비(%):")
    print(comp_pct.round(2).to_string())
    comp_pct.to_csv(os.path.join(TAB_DIR, "E_season_quality_composition.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for q in pivot.columns:
        axes[0].plot(pivot.index, pivot[q] * 100, marker="o", label=str(q))
    axes[0].set_title("품질 분위 통제 후에도 시즌 하락이 남는지")
    axes[0].set_ylabel("success rate (%)")
    axes[0].legend(fontsize=7)
    for q in comp_pct.columns:
        axes[1].plot(comp_pct.index, comp_pct[q], marker="o", label=str(q))
    axes[1].set_title("시즌별 투수 품질 분위 구성비(%)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E_season_quality.png"), dpi=120)
    plt.close(fig)
    return pivot, comp_pct


def F_season_batter_quality(df):
    section("F. season x asof_batter_success_rate 분위 — 그룹5 타자 쪽에서도 §E 재현되는지")
    d = df[df["asof_batter_success_rate"].notna()].copy()
    d["quality_q"] = pd.qcut(d["asof_batter_success_rate"], 5, labels=["Q1(낮음)", "Q2", "Q3", "Q4", "Q5(높음)"], duplicates="drop")
    g = d.groupby(["season", "quality_q"], observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    pivot = g.pivot(index="season", columns="quality_q", values="rate")
    print((pivot * 100).round(2).to_string())
    pivot.to_csv(os.path.join(TAB_DIR, "F_season_batter_quality_pivot.csv"))

    comp = d.groupby(["season", "quality_q"], observed=True).size().unstack()
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    print("\n시즌별 분위 구성비(%):")
    print(comp_pct.round(2).to_string())
    comp_pct.to_csv(os.path.join(TAB_DIR, "F_season_batter_quality_composition.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for q in pivot.columns:
        axes[0].plot(pivot.index, pivot[q] * 100, marker="o", label=str(q))
    axes[0].set_title("타자 품질 분위 통제 후에도 시즌 하락이 남는지")
    axes[0].set_ylabel("success rate (%)")
    axes[0].legend(fontsize=7)
    for q in comp_pct.columns:
        axes[1].plot(comp_pct.index, comp_pct[q], marker="o", label=str(q))
    axes[1].set_title("시즌별 asof_batter_success_rate 분위 구성비(%)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "F_season_batter_quality.png"), dpi=120)
    plt.close(fig)
    return pivot, comp_pct


def G_form_trend_li(df, overall_rate):
    section("G. pitcher_form_trend_isolated x li — 최근 폼 추세가 압박 상황에서 더 크게 작동하는지")
    d = df[df["pitcher_form_trend_isolated"].notna()].copy()
    d["trend_q"] = pd.qcut(d["pitcher_form_trend_isolated"], 4, labels=["Q1(하락세)", "Q2", "Q3", "Q4(상승세)"], duplicates="drop")
    d["li_q"] = pd.qcut(d["li"], 3, labels=["li 낮음", "li 중간", "li 높음"], duplicates="drop")
    g = d.groupby(["li_q", "trend_q"], observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    pivot = g.pivot(index="li_q", columns="trend_q", values="rate")
    print((pivot * 100).round(2).to_string())
    pivot.to_csv(os.path.join(TAB_DIR, "G_trend_li_pivot.csv"))

    spread = pivot["Q4(상승세)"] - pivot["Q1(하락세)"]
    print("\nQ4(상승세)-Q1(하락세) 격차, li 구간별:")
    print((spread * 100).round(2).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for li_level in pivot.index:
        ax.plot(pivot.columns.astype(str), pivot.loc[li_level] * 100, marker="o", label=str(li_level))
    ax.axhline(overall_rate * 100, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("success rate (%)")
    ax.set_title("li 구간별 pitcher_form_trend_isolated 효과")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "G_trend_li.png"), dpi=120)
    plt.close(fig)
    return pivot, spread


def H_pitcher_batter_matchup(df):
    section("H. asof_pitcher_success_rate x asof_batter_success_rate 매치업 (그룹4 x 그룹5)")
    d = df[df["asof_pitcher_success_rate"].notna() & df["asof_batter_success_rate"].notna()].copy()
    d["pitcher_q"] = pd.qcut(d["asof_pitcher_success_rate"], 5, labels=["P1", "P2", "P3", "P4", "P5"], duplicates="drop")
    d["batter_q"] = pd.qcut(d["asof_batter_success_rate"], 5, labels=["B1", "B2", "B3", "B4", "B5"], duplicates="drop")
    g = d.groupby(["pitcher_q", "batter_q"], observed=True)[TARGET].agg(n="count", rate="mean").reset_index()
    pivot = g.pivot(index="pitcher_q", columns="batter_q", values="rate")
    print((pivot * 100).round(2).to_string())
    pivot.to_csv(os.path.join(TAB_DIR, "H_pitcher_batter_pivot.csv"))

    diag_gap = pivot.loc["P5", "B5"] - pivot.loc["P1", "B1"]
    print(f"\nP5xB5 - P1xB1: {diag_gap*100:.2f}%p")
    additive_check = (pivot.loc["P5"] - pivot.loc["P1"]).mean()
    print(f"P5 - P1 평균 격차(batter_q 무관, 가법적이면 batter축에서 거의 일정해야 함): {additive_check*100:.2f}%p")
    print((pivot.loc["P5"] - pivot.loc["P1"]).round(4).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values * 100, cmap="RdBu_r")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("asof_batter_success_rate quintile")
    ax.set_ylabel("asof_pitcher_success_rate quintile")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]*100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="success rate (%)")
    ax.set_title("투수 x 타자 품질 분위 매치업")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "H_heatmap_pitcher_batter.png"), dpi=120)
    plt.close(fig)
    return pivot


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_data()
    overall_rate = df[TARGET].mean()
    print(f"전체 행 수: {len(df):,}  |  전체 control_success 평균: {overall_rate*100:.4f}%")

    A_inning_li_score_runner(df, overall_rate)
    B_hand_matchup_pitchmix(df)
    C_count_li(df, overall_rate)
    D_outs_runner(df, overall_rate)
    E_season_pitcher_quality(df)
    F_season_batter_quality(df)
    G_form_trend_li(df, overall_rate)
    H_pitcher_batter_matchup(df)

    section("완료")
    print(f"그래프: {FIG_DIR}")
    print(f"표: {TAB_DIR}")


if __name__ == "__main__":
    main()
