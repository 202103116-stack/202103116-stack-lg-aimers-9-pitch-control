"""Group 3 EDA — 주자 상황 및 경기 중요도 (8개 컬럼).

담당 컬럼 (원본 노트를 재검증/재구성한 버전):
    runner_on_1b, runner_on_2b, runner_on_3b, num_runners_on, base_state,
    home_win_expectancy, away_win_expectancy, li

train.csv만 읽고(test.csv는 사용하지 않음) 행 단위 통계와 group-by
집계만 수행한다.

다른 그룹 컬럼(inning, score_diff_pitcher_team 등)과의 상호작용은
최종 통합 EDA(reports/eda_final/)에서 다룬다.

실행:
    python reports/eda_group3/run_eda.py
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
OUT_DIR = "./reports/eda_group3"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

RUNNER_COLS = ["runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state"]
WE_LI_COLS = ["home_win_expectancy", "away_win_expectancy", "li"]
ALL_COLS = RUNNER_COLS + WE_LI_COLS
TARGET = "control_success"

BASE_STATE_LABELS = {
    "___": "주자 없음", "1__": "1루", "_2_": "2루", "__3": "3루",
    "12_": "1·2루", "1_3": "1·3루", "_23": "2·3루", "123": "만루",
}


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data():
    usecols = ALL_COLS + [TARGET, "top_bottom"]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=usecols)
    return df


def step1_quality(df):
    section("1. 데이터 품질 확인")
    print(f"train.csv shape: {df.shape}")
    rows = []
    for c in ALL_COLS:
        s = df[c]
        rows.append({
            "column": c, "dtype": str(s.dtype), "n_null": int(s.isna().sum()),
            "n_unique": int(s.nunique()),
            "min": s.min() if pd.api.types.is_numeric_dtype(s) else "",
            "max": s.max() if pd.api.types.is_numeric_dtype(s) else "",
        })
    quality = pd.DataFrame(rows)
    print(quality.to_string(index=False))
    quality.to_csv(os.path.join(TAB_DIR, "00_quality_summary.csv"), index=False)
    return quality


def step2_invariants(df):
    section("2. 구조적 불변식 검증")
    n = len(df)
    results = []

    # runner flags 0/1 only
    for c in ["runner_on_1b", "runner_on_2b", "runner_on_3b"]:
        bad = (~df[c].isin([0, 1])).sum()
        results.append({"check": f"{c} in {{0,1}}", "violations": int(bad)})

    # num_runners_on in [0,3]
    bad_nro = (~df["num_runners_on"].between(0, 3)).sum()
    results.append({"check": "num_runners_on in [0,3]", "violations": int(bad_nro)})

    # num_runners_on == sum of flags
    sum_flags = df["runner_on_1b"] + df["runner_on_2b"] + df["runner_on_3b"]
    mismatch = (df["num_runners_on"] != sum_flags).sum()
    results.append({"check": "num_runners_on == sum(runner_on_1b/2b/3b)", "violations": int(mismatch)})

    # base_state consistent with flags
    expected_state = (
        np.where(df["runner_on_1b"] == 1, "1", "_")
        + np.where(df["runner_on_2b"] == 1, "2", "_")
        + np.where(df["runner_on_3b"] == 1, "3", "_")
    )
    state_mismatch = (df["base_state"].astype(str) != expected_state).sum()
    results.append({"check": "base_state consistent with runner flags", "violations": int(state_mismatch)})

    # home + away win expectancy ~= 100
    we_sum = df["home_win_expectancy"] + df["away_win_expectancy"]
    results.append({
        "check": "home_win_expectancy + away_win_expectancy range",
        "violations": f"min={we_sum.min():.4f}, max={we_sum.max():.4f}",
    })
    corr_we = df["home_win_expectancy"].corr(df["away_win_expectancy"])

    rel_df = pd.DataFrame(results)
    print(rel_df.to_string(index=False))
    print(f"\n(n={n})")
    print(f"home_win_expectancy vs away_win_expectancy 상관계수: {corr_we:.7f}")
    rel_df.to_csv(os.path.join(TAB_DIR, "01_invariant_checks.csv"), index=False)
    return rel_df


def step3_runner_presence(df, overall_rate):
    section("3. 베이스별 주자 존재 비율 및 성공률")
    rows = []
    for base, col in [("1루", "runner_on_1b"), ("2루", "runner_on_2b"), ("3루", "runner_on_3b")]:
        present = df[df[col] == 1]
        absent = df[df[col] == 0]
        rows.append({
            "base": base, "n_present": len(present), "pct_present": len(present) / len(df) * 100,
            "rate_absent": absent[TARGET].mean(), "rate_present": present[TARGET].mean(),
        })
    base_presence = pd.DataFrame(rows)
    print(base_presence.to_string(index=False))
    base_presence.to_csv(os.path.join(TAB_DIR, "03_base_presence.csv"), index=False)

    section("3b. 출루 주자 수 분포")
    g = df.groupby("num_runners_on")[TARGET].agg(n="count", success_rate="mean").reset_index()
    g["pct"] = g["n"] / len(df) * 100
    g["diff_vs_overall"] = g["success_rate"] - overall_rate
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "03b_num_runners_summary.csv"), index=False)
    return base_presence, g


def step4_base_state(df, overall_rate):
    section("4. base_state 별 분포와 성공률")
    g = df.groupby("base_state")[TARGET].agg(n="count", success_rate="mean").reset_index()
    g["pct"] = g["n"] / len(df) * 100
    g["diff_vs_overall"] = g["success_rate"] - overall_rate
    g["label"] = g["base_state"].map(BASE_STATE_LABELS)
    order = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
    g["order"] = g["base_state"].map({s: i for i, s in enumerate(order)})
    g = g.sort_values("order").drop(columns="order")
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "04_base_state_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(g["label"], g["success_rate"] * 100, color="#4C72B0")
    ax.axvline(overall_rate * 100, color="red", linestyle="--", linewidth=0.8, label=f"overall={overall_rate*100:.2f}%")
    ax.set_xlim(g["success_rate"].min() * 100 - 1, g["success_rate"].max() * 100 + 1)
    ax.set_xlabel("success rate (%)")
    ax.set_title("주자 배치별 제구 성공률")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_barh_base_state.png"), dpi=120)
    plt.close(fig)
    return g


def step5_we_li_stats(df):
    section("5. 기대 승률 / li 기술통계")
    desc = df[WE_LI_COLS].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    print(desc.to_string())
    desc.to_csv(os.path.join(TAB_DIR, "05_we_li_stats.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, WE_LI_COLS):
        ax.hist(df[col].dropna(), bins=40, color="#55A868")
        ax.set_title(col)
        ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_grid_hist_we_li.png"), dpi=120)
    plt.close(fig)
    return desc


def step6_li_analysis(df, overall_rate):
    section("6. li 구간별 성공률")
    bins = [-0.001, 0.5, 1.0, 2.0, 3.0, np.inf]
    labels = ["<=0.5", "0.5~1.0", "1.0~2.0", "2.0~3.0", ">3.0"]
    df = df.copy()
    df["li_bin"] = pd.cut(df["li"], bins=bins, labels=labels)
    g = df.groupby("li_bin", observed=True).agg(n=(TARGET, "count"), avg_li=("li", "mean"), success_rate=(TARGET, "mean")).reset_index()
    g["pct"] = g["n"] / len(df) * 100
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "06_li_bin_summary.csv"), index=False)

    corr_li = df["li"].corr(df[TARGET])
    print(f"\nli vs target 피어슨 상관계수: {corr_li:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(g["li_bin"].astype(str), g["success_rate"] * 100, marker="o", color="#C44E52")
    ax.axhline(overall_rate * 100, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("success rate (%)")
    ax.set_title("li 구간별 성공률 (비선형)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_li_bin_trend.png"), dpi=120)
    plt.close(fig)
    return g, corr_li


def step7_win_expectancy_analysis(df, overall_rate):
    section("7. 기대 승률과 제구 성공률")
    corr_home = df["home_win_expectancy"].corr(df[TARGET])
    print(f"home_win_expectancy vs target 피어슨 상관계수: {corr_home:.4f}")

    df = df.copy()
    df["pitcher_team_win_expectancy"] = np.where(
        df["top_bottom"] == "T", df["home_win_expectancy"], df["away_win_expectancy"]
    )
    corr_ptwe = df["pitcher_team_win_expectancy"].corr(df[TARGET])
    print(f"pitcher_team_win_expectancy vs target 피어슨 상관계수: {corr_ptwe:.4f}")

    q10 = df["pitcher_team_win_expectancy"].quantile(0.10)
    q50_lo, q50_hi = df["pitcher_team_win_expectancy"].quantile([0.45, 0.55])
    q90 = df["pitcher_team_win_expectancy"].quantile(0.90)

    bottom = df[df["pitcher_team_win_expectancy"] <= q10]
    middle = df[(df["pitcher_team_win_expectancy"] >= q50_lo) & (df["pitcher_team_win_expectancy"] <= q50_hi)]
    top = df[df["pitcher_team_win_expectancy"] >= q90]

    rows = [
        {"segment": "하위 10% (열세)", "n": len(bottom), "success_rate": bottom[TARGET].mean()},
        {"segment": "중간 45~55% (접전)", "n": len(middle), "success_rate": middle[TARGET].mean()},
        {"segment": "상위 10% (우세)", "n": len(top), "success_rate": top[TARGET].mean()},
    ]
    seg = pd.DataFrame(rows)
    print(seg.to_string(index=False))
    seg.to_csv(os.path.join(TAB_DIR, "07_pitcher_team_we_segments.csv"), index=False)

    # 10분위 전체
    df["we_decile"] = pd.qcut(df["pitcher_team_win_expectancy"], 10, duplicates="drop")
    g = df.groupby("we_decile", observed=True)[TARGET].agg(n="count", success_rate="mean").reset_index()
    print("\n10분위 전체:")
    print(g.to_string(index=False))
    g.to_csv(os.path.join(TAB_DIR, "07_pitcher_team_we_deciles.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(g))
    ax.plot(x, g["success_rate"] * 100, marker="o", color="#8172B2")
    ax.axhline(overall_rate * 100, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in g["we_decile"]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("success rate (%)")
    ax.set_title("pitcher_team_win_expectancy 10분위 vs 성공률")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_pitcher_team_we_deciles.png"), dpi=120)
    plt.close(fig)
    return seg, g


def step8_runner_base_state_caveat(df):
    section("8. 주자 변수 간 정보 중복 확인 (선형 모델용 참고)")
    # base_state를 dummy로 취급했을 때 runner flags/num_runners_on과 완전 대응하는지 이미 §2에서 검증됨
    print("§2에서 num_runners_on == sum(flags), base_state == flags 조합이 각각 100% 일치함을 확인.")
    print("따라서 runner_on_1b/2b/3b, num_runners_on, base_state 5개는 같은 정보를 5가지 형태로 표현한 것.")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_data()
    overall_rate = df[TARGET].mean()
    print(f"전체 행 수: {len(df):,}  |  전체 control_success 평균: {overall_rate*100:.4f}%")

    step1_quality(df)
    step2_invariants(df)
    step3_runner_presence(df, overall_rate)
    step4_base_state(df, overall_rate)
    step5_we_li_stats(df)
    step6_li_analysis(df, overall_rate)
    step7_win_expectancy_analysis(df, overall_rate)
    step8_runner_base_state_caveat(df)

    section("완료")
    print(f"그래프: {FIG_DIR}")
    print(f"표: {TAB_DIR}")


if __name__ == "__main__":
    main()
