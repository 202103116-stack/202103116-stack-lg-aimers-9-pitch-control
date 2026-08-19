"""Trackman EDA — trackman_history.csv 탐색 (exp_005 준비 단계).

목적: `data/trackman_history.csv`(2019~2024, 1,793,078행 x 30컬럼)를
train/test와 연계할 수 있는 방법을 찾는다. 사전 검증 결과 train/test와
trackman 사이에는 선수(pitcher_id vs pitcher_trackman_id)/팀(pitcher_team_id
vs pitcher_team) 단위 공유 키가 전혀 없고(값 겹침 0건), 실제 평가 대상은
season=2025인데 trackman은 2019~2024만 존재한다(2025 트랙맨 사용 금지,
CLAUDE.md Safety Rule 10). 조인은 train/test에도 있는 상황 컬럼
(season, game_month, balls/strikes/outs_before, pitcher_hand,
batter_hand 등)으로만 가능하고, 정보 누수 방지를 위해 실제 피처
엔지니어링(exp_005) 단계에서는 `(season, game_month)` 기준 누적
(as-of) 집계 — 그 행의 달보다 엄격히 이전 달까지의 트랙맨만 사용 —
를 쓴다(§E가 이 설계의 콜드스타트 시뮬레이션). 이 스크립트 자체는
피처를 만들지 않는 순수 설명적(descriptive) EDA라 이 제약과 무관하게
전체 데이터를 봐도 무방하다. 자세한 배경은 이 EDA를 위해 승인받은 plan
(`C:\\Users\\yeob0\\.claude\\plans\\lg-aimers-9-groovy-glacier.md`) 참고.

이 스크립트는 trackman_history.csv와 train.csv를 읽되, 행 단위로 직접
조인하지 않는다(공유 키가 없어 애초에 불가능) — 각각을 공통 축(season,
game_month, game_dayofweek, balls/strikes/outs_before, pitcher_hand,
batter_hand)으로 독립적으로 집계한 뒤 나란히 비교하는 방식만 사용한다.
test.csv는 사용하지 않는다(CLAUDE.md 누수 규칙 6번).

실행:
    python reports/eda_trackman/run_eda.py

산출물:
    reports/eda_trackman/figures/*.png
    reports/eda_trackman/tables/*.csv
    (표준출력을 reports/eda_trackman/eda_run_log.txt 로 리다이렉트해서 보관 —
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
OUT_DIR = "./reports/eda_trackman"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

PHYS_COLS = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]
TARGET = "control_success"


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def main():
    section("0. 데이터 로드")
    tm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    print("trackman shape:", tm.shape)

    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    print("train shape:", tr.shape)

    # Trackman hand encoding -> train과 동일한 형태로 맞춰서 이후 비교에 사용
    # (train: pitcher_hand/batter_hand는 1/2 코드, trackman: 'Right'/'Left' 문자열)
    hand_map = {"Right": "R", "Left": "L"}
    tm["pitcher_hand_c"] = tm["pitcher_hand"].map(hand_map)
    tm["batter_hand_c"] = tm["batter_hand"].map(hand_map)
    tm["same_hand"] = (tm["pitcher_hand_c"] == tm["batter_hand_c"]).astype(int)

    # ------------------------------------------------------------------
    section("A. 데이터 품질")
    # ------------------------------------------------------------------
    print("\n-- A1. 컬럼별 결측률 --")
    null_rate = tm.isnull().mean().sort_values(ascending=False)
    print(null_rate[null_rate > 0].round(4))
    null_rate.to_frame("null_rate").to_csv(os.path.join(TAB_DIR, "A1_null_rate.csv"))

    print("\n-- A2. pitch_type_group 분포 --")
    ptg_counts = tm["pitch_type_group"].value_counts(dropna=False)
    ptg_pct = (ptg_counts / len(tm) * 100).round(2)
    print(pd.DataFrame({"count": ptg_counts, "pct": ptg_pct}))
    pd.DataFrame({"count": ptg_counts, "pct": ptg_pct}).to_csv(
        os.path.join(TAB_DIR, "A2_pitch_type_group_dist.csv")
    )

    print("\n-- A3. pitch_type_group별 물리량 결측률 --")
    null_by_ptg = tm.groupby("pitch_type_group")[PHYS_COLS].apply(
        lambda g: g.isnull().mean()
    )
    print(null_by_ptg.round(4))
    null_by_ptg.to_csv(os.path.join(TAB_DIR, "A3_null_rate_by_pitch_type_group.csv"))

    print("\n-- A4. balls_before / outs_before 야구 규칙 범위 이탈 --")
    legal_balls = tm["balls_before"].between(0, 3)
    legal_strikes = tm["strikes_before"].between(0, 2)
    legal_outs = tm["outs_before"].between(0, 2)
    range_check = pd.DataFrame(
        {
            "column": ["balls_before", "strikes_before", "outs_before"],
            "min": [tm["balls_before"].min(), tm["strikes_before"].min(), tm["outs_before"].min()],
            "max": [tm["balls_before"].max(), tm["strikes_before"].max(), tm["outs_before"].max()],
            "n_out_of_range": [
                (~legal_balls).sum(),
                (~legal_strikes).sum(),
                (~legal_outs).sum(),
            ],
            "pct_out_of_range": [
                round((~legal_balls).mean() * 100, 4),
                round((~legal_strikes).mean() * 100, 4),
                round((~legal_outs).mean() * 100, 4),
            ],
        }
    )
    print(range_check.to_string(index=False))
    range_check.to_csv(os.path.join(TAB_DIR, "A4_legal_range_check.csv"), index=False)

    print("\n-- A4b. 범위 이탈 행의 season/pitch_type_group 분포 (있다면) --")
    bad_rows = tm[~(legal_balls & legal_strikes & legal_outs)]
    if len(bad_rows) > 0:
        print(bad_rows["season"].value_counts().sort_index())
        print(bad_rows["pitch_type_group"].value_counts())
    else:
        print("범위 이탈 행 없음")

    print("\n-- A5. tagged_pitch_type vs auto_pitch_type vs pitch_type_group --")
    print("tagged_pitch_type unique:", tm["tagged_pitch_type"].nunique())
    print("auto_pitch_type unique:", tm["auto_pitch_type"].nunique())
    tagged_x_group = pd.crosstab(tm["tagged_pitch_type"], tm["pitch_type_group"])
    print(tagged_x_group)
    tagged_x_group.to_csv(os.path.join(TAB_DIR, "A5_tagged_pitch_type_x_group.csv"))

    # ------------------------------------------------------------------
    section("B. 반복축(매년 되풀이) 신호 탐색")
    # ------------------------------------------------------------------
    print("\n-- B1. pitch_type_group별 물리량 평균/표준편차 (전 시즌 통합) --")
    ptg_phys = tm.groupby("pitch_type_group")[PHYS_COLS].agg(["mean", "std", "count"])
    print(ptg_phys.round(2))
    ptg_phys.to_csv(os.path.join(TAB_DIR, "B1_pitch_type_group_physical_stats.csv"))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, col in zip(axes.flat, PHYS_COLS):
        means = tm.groupby("pitch_type_group")[col].mean().reindex(
            ["fastball", "breaking", "offspeed", "other"]
        )
        stds = tm.groupby("pitch_type_group")[col].std().reindex(
            ["fastball", "breaking", "offspeed", "other"]
        )
        ax.bar(means.index, means.values, yerr=stds.values, capsize=4, color="steelblue")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("구종군(pitch_type_group)별 물리량 평균 ± 표준편차")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "B1_pitch_type_group_physical.png"), dpi=110)
    plt.close(fig)

    print("\n-- B2. 좌우 매치업(same_hand) x pitch_type_group 물리량 --")
    hand_ptg = tm.groupby(["same_hand", "pitch_type_group"])[PHYS_COLS].mean()
    print(hand_ptg.round(2))
    hand_ptg.to_csv(os.path.join(TAB_DIR, "B2_same_hand_x_pitch_type_group_physical.csv"))

    print("\n-- B2b. same_hand별 구종군 사용 비율 (breaking 비율이 실제로 다른가) --")
    hand_ptg_rate = pd.crosstab(tm["same_hand"], tm["pitch_type_group"], normalize="index") * 100
    print(hand_ptg_rate.round(2))
    hand_ptg_rate.to_csv(os.path.join(TAB_DIR, "B2b_same_hand_pitch_type_group_rate.csv"))

    print("\n-- B2c. (참고, 별도 데이터셋) train.csv same_hand_x_breaking_rate 관련 success_rate --")
    tr_same_hand = (tr["pitcher_hand"] == tr["batter_hand"]).astype(int)
    tr_success_by_hand = tr.groupby(tr_same_hand)[TARGET].mean() * 100
    tr_success_by_hand.index.name = "same_hand"
    print(tr_success_by_hand.round(3))
    tr_success_by_hand.to_frame("control_success_rate_pct").to_csv(
        os.path.join(TAB_DIR, "B2c_train_success_rate_by_same_hand.csv")
    )

    print("\n-- B3. 카운트 상황(balls_before/strikes_before/outs_before)별 구속/구종 선택 --")
    count_speed = tm.groupby(["balls_before", "strikes_before"])["rel_speed"].mean().unstack()
    print(count_speed.round(2))
    count_speed.to_csv(os.path.join(TAB_DIR, "B3_rel_speed_by_count.csv"))

    print("\n-- B3b. 3볼 상황 vs 그 외, 구종군 사용 비율 --")
    three_ball = (tm["balls_before"] == 3).astype(int)
    three_ball_ptg = pd.crosstab(three_ball, tm["pitch_type_group"], normalize="index") * 100
    three_ball_ptg.index.name = "three_ball"
    print(three_ball_ptg.round(2))
    three_ball_ptg.to_csv(os.path.join(TAB_DIR, "B3b_three_ball_pitch_type_group_rate.csv"))

    print("\n-- B3c. 3볼 상황 vs 그 외, 평균 구속 --")
    three_ball_speed = tm.groupby(three_ball)["rel_speed"].agg(["mean", "std", "count"])
    three_ball_speed.index.name = "three_ball"
    print(three_ball_speed.round(3))
    three_ball_speed.to_csv(os.path.join(TAB_DIR, "B3c_three_ball_rel_speed.csv"))

    print("\n-- B4. game_month별 물리량 (계절 효과, 전 시즌 통합) --")
    month_phys = tm.groupby("game_month")[["rel_speed", "spin_rate", "induced_vert_break"]].mean()
    print(month_phys.round(2))
    month_phys.to_csv(os.path.join(TAB_DIR, "B4_month_physical_stats.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(month_phys.index, month_phys["rel_speed"], marker="o", label="rel_speed")
    ax2 = ax.twinx()
    ax2.plot(
        month_phys.index, month_phys["spin_rate"], marker="s", color="orange", label="spin_rate"
    )
    ax.set_xlabel("game_month")
    ax.set_ylabel("rel_speed (mph)")
    ax2.set_ylabel("spin_rate (rpm)")
    fig.legend(loc="upper center", ncol=2)
    fig.suptitle("월별 평균 구속 / 회전수 (2019~2024 통합)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "B4_month_physical.png"), dpi=110)
    plt.close(fig)

    print("\n-- B5. game_dayofweek별 물리량 --")
    dow_phys = tm.groupby("game_dayofweek")[["rel_speed", "spin_rate"]].mean()
    print(dow_phys.round(2))
    dow_phys.to_csv(os.path.join(TAB_DIR, "B5_dayofweek_physical_stats.csv"))

    # ------------------------------------------------------------------
    section("C. 시즌 추세 — eda_final §E 교차검증")
    # ------------------------------------------------------------------
    print("\n-- C1. 시즌별 물리량 평균 (트랙맨) --")
    season_phys = tm.groupby("season")[PHYS_COLS].mean()
    print(season_phys.round(3))
    season_phys.to_csv(os.path.join(TAB_DIR, "C1_season_physical_stats.csv"))

    print("\n-- C2. 시즌별 구종군 사용 비율 (트랙맨) --")
    season_ptg_rate = pd.crosstab(tm["season"], tm["pitch_type_group"], normalize="index") * 100
    print(season_ptg_rate.round(2))
    season_ptg_rate.to_csv(os.path.join(TAB_DIR, "C2_season_pitch_type_group_rate.csv"))

    print("\n-- C3. (참고) train.csv 시즌별 control_success_rate — eda_final §E 재확인 --")
    season_success = tr.groupby("season")[TARGET].mean() * 100
    print(season_success.round(2))
    season_success.to_frame("control_success_rate_pct").to_csv(
        os.path.join(TAB_DIR, "C3_train_season_success_rate.csv")
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(season_phys.index, season_phys["rel_speed"], marker="o", color="steelblue")
    axes[0].set_title("시즌별 평균 rel_speed (트랙맨)")
    axes[1].plot(season_phys.index, season_phys["induced_vert_break"], marker="o", color="darkorange")
    axes[1].set_title("시즌별 평균 induced_vert_break (트랙맨)")
    axes[2].plot(season_success.index, season_success.values, marker="o", color="crimson")
    axes[2].set_title("시즌별 control_success_rate (train, %)")
    for ax in axes:
        ax.set_xlabel("season")
        ax.grid(alpha=0.3)
    fig.suptitle("시즌 추세 비교 — 트랙맨 물리량 vs train 성공률 (같은 축 season으로만 병렬 비교, 행 단위 조인 아님)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "C_season_trend_comparison.png"), dpi=110)
    plt.close(fig)

    print("\n-- C4. 상관계수 (시즌 단위 6개 점, 참고용) --")
    merged_season = season_phys.copy()
    merged_season["control_success_rate_pct"] = season_success
    corr = merged_season.corr()["control_success_rate_pct"].drop("control_success_rate_pct")
    print(corr.round(3))
    corr.to_frame("corr_with_season_success_rate").to_csv(
        os.path.join(TAB_DIR, "C4_season_level_correlation.csv")
    )

    # ------------------------------------------------------------------
    section("D. 조인 실현 가능성 — 축별 표본 크기")
    # ------------------------------------------------------------------
    print("\n-- D1. (game_month, game_dayofweek) 조합별 표본 수 --")
    d1 = tm.groupby(["game_month", "game_dayofweek"]).size()
    print("조합 수:", d1.shape[0], "/ 최소:", d1.min(), "/ 최대:", d1.max(), "/ 중앙값:", d1.median())
    d1.to_csv(os.path.join(TAB_DIR, "D1_month_dayofweek_counts.csv"))

    print("\n-- D2. (balls_before, strikes_before, outs_before) 조합별 표본 수 --")
    d2 = tm.groupby(["balls_before", "strikes_before", "outs_before"]).size().sort_values()
    print(d2)
    d2.to_csv(os.path.join(TAB_DIR, "D2_count_situation_counts.csv"))

    print("\n-- D3. (pitch_type_group, same_hand, balls_before) 조합별 표본 수 --")
    d3 = tm.groupby(["pitch_type_group", "same_hand", "balls_before"]).size().sort_values()
    print("조합 수:", d3.shape[0], "/ 최소 표본:", d3.min(), "/ 최소 표본 5개 이하 조합 수:", (d3 <= 5).sum())
    d3.to_csv(os.path.join(TAB_DIR, "D3_ptg_hand_balls_counts.csv"))

    print("\n-- D4. (season, game_month, pitch_type_group) 조합별 표본 수 (비반복축 포함, 참고용) --")
    d4 = tm.groupby(["season", "game_month", "pitch_type_group"]).size().sort_values()
    print("조합 수:", d4.shape[0], "/ 최소 표본:", d4.min(), "/ 최소 표본 30개 이하 조합 수:", (d4 <= 30).sum())
    d4.to_csv(os.path.join(TAB_DIR, "D4_season_month_ptg_counts.csv"))

    # ------------------------------------------------------------------
    section("E. asof 누적(as-of, (season, game_month) 기준) 콜드스타트 시뮬레이션")
    # ------------------------------------------------------------------
    # plan §1c 설계 검증용: train/test 행에 트랙맨 피처를 붙일 때, 그 행의
    # (season, game_month)보다 "엄격히 이전" 달까지의 트랙맨 데이터만
    # 누적해서 쓴다. 여기서는 그 누적 표본 수가 시간이 지나며 어떻게
    # 늘어나는지, 특히 2019년 초반 콜드스타트 구간이 얼마나 심각한지
    # (전체 기준과, same_hand x balls_before로 잘게 쪼갠 기준 둘 다) 확인한다.
    tm["ym"] = tm["season"] * 100 + tm["game_month"]
    tr["ym"] = tr["season"] * 100 + tr["game_month"]

    print("\n-- E1. 전체 기준 (season,game_month) 별 누적 표본 수 (그 달 이전까지) --")
    counts_by_ym = tm.groupby("ym").size().sort_index()
    cum_before = counts_by_ym.cumsum().shift(1, fill_value=0)
    e1 = pd.DataFrame({"n_this_month": counts_by_ym, "cum_before_this_month": cum_before})
    print(e1.head(12))
    print("...")
    print(e1.tail(6))
    e1.to_csv(os.path.join(TAB_DIR, "E1_overall_cumulative_by_month.csv"))

    n_thin_1000 = (e1["cum_before_this_month"] < 1000).sum()
    print(f"\n누적 표본 < 1000인 (season,month) 개수: {n_thin_1000} / 전체 {len(e1)}개")

    print("\n-- E2. train.csv 행이 실제로 콜드스타트 구간(누적<1000)에 얼마나 걸리는지 --")
    thin_yms = set(e1.index[e1["cum_before_this_month"] < 1000])
    tr_in_thin = tr["ym"].isin(thin_yms)
    print(f"콜드스타트 구간에 속하는 train 행 수: {tr_in_thin.sum()} ({tr_in_thin.mean()*100:.3f}%)")
    if tr_in_thin.sum() > 0:
        print(tr.loc[tr_in_thin, "ym"].value_counts().sort_index())

    print("\n-- E3. 세분화 기준 (same_hand x balls_before, 8개 조합)의 누적 표본 수 --")
    grid = pd.MultiIndex.from_product(
        [sorted(tm["ym"].unique()), [0, 1], [0, 1, 2, 3]],
        names=["ym", "same_hand", "balls_before"],
    ).to_frame(index=False)
    combo_counts = (
        tm[tm["balls_before"] <= 3]
        .groupby(["ym", "same_hand", "balls_before"])
        .size()
        .reset_index(name="n")
    )
    combo_full = grid.merge(combo_counts, on=["ym", "same_hand", "balls_before"], how="left")
    combo_full["n"] = combo_full["n"].fillna(0)
    combo_full = combo_full.sort_values(["same_hand", "balls_before", "ym"])
    combo_full["cum_before"] = combo_full.groupby(["same_hand", "balls_before"])["n"].cumsum() - combo_full["n"]

    worst_per_month = combo_full.groupby("ym")["cum_before"].min()
    print("월별 8개 조합 중 최소 누적 표본 수 (앞 12개월):")
    print(worst_per_month.head(12))
    print("월별 8개 조합 중 최소 누적 표본 수 (뒤 6개월):")
    print(worst_per_month.tail(6))
    worst_per_month.to_frame("min_cum_before_across_combos").to_csv(
        os.path.join(TAB_DIR, "E3_finegrained_cumulative_worst_case.csv")
    )

    n_thin_100 = (worst_per_month < 100).sum()
    print(f"\n8개 조합 중 최소 누적 표본이 100 미만인 (season,month) 개수: {n_thin_100} / 전체 {len(worst_per_month)}개")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(len(e1)), e1["cum_before_this_month"].values, marker=".")
    axes[0].axhline(1000, color="red", linestyle="--", label="1000")
    axes[0].set_title("전체 기준 누적 표본 수 (시간순 (season,month) 인덱스)")
    axes[0].set_xlabel("(season,month) 순번")
    axes[0].legend()
    axes[1].plot(range(len(worst_per_month)), worst_per_month.values, marker=".", color="darkorange")
    axes[1].axhline(100, color="red", linestyle="--", label="100")
    axes[1].set_title("same_hand x balls_before(8개 조합) 중 최소 누적 표본 수")
    axes[1].set_xlabel("(season,month) 순번")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E_coldstart_cumulative.png"), dpi=110)
    plt.close(fig)

    print("\n완료.")


if __name__ == "__main__":
    main()
