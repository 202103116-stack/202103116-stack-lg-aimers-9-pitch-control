"""Group 4 EDA — 투수의 과거 제구/구종 이력 (asof_pitcher_* 10개 컬럼).

담당 컬럼 (원본 노트 2건 "4번 EDA", "4번 EDA-허찬"을 통합/재검증):
    asof_pitcher_n, asof_pitcher_success_rate, asof_pitcher_reverse_rate,
    asof_pitcher_middle_rate, asof_pitcher_ball_rate, asof_pitcher_strike_rate,
    asof_pitcher_pitchmix_n, asof_pitcher_fastball_rate,
    asof_pitcher_breaking_rate, asof_pitcher_offspeed_rate

이 컬럼들은 전부 공식 asof_* (투구 시점까지만 누적) 값이라 row-local하고
CLAUDE.md 누수 규칙과 충돌하지 않는다.

실행:
    python reports/eda_group4/run_eda.py
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
OUT_DIR = "./reports/eda_group4"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

N_COLS = ["asof_pitcher_n", "asof_pitcher_pitchmix_n"]
RATE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
ALL_COLS = N_COLS + RATE_COLS
TARGET = "control_success"

N_BINS = [(-1, 0, "0"), (0, 1, "1"), (1, 10, "2-10"), (10, 30, "11-30"), (30, 100, "31-100"),
          (100, 300, "101-300"), (300, 1000, "301-1000"), (1000, np.inf, "1001+")]


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data():
    usecols = ["season", "pitcher_id"] + ALL_COLS + [TARGET]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=usecols)
    return df


def step1_quality(df):
    section("1. 데이터 품질 확인")
    print(f"train.csv shape: {df.shape}")
    print(f"고유 pitcher_id 수: {df['pitcher_id'].nunique()}")
    rows = []
    for c in ALL_COLS:
        s = df[c]
        rows.append({
            "column": c, "dtype": str(s.dtype), "n_null": int(s.isna().sum()),
            "null_pct": s.isna().mean() * 100,
            "min": s.min(skipna=True), "max": s.max(skipna=True),
        })
    quality = pd.DataFrame(rows)
    print(quality.to_string(index=False))
    quality.to_csv(os.path.join(TAB_DIR, "00_quality_summary.csv"), index=False)
    return quality


def step2_coldstart(df):
    section("2. 결측치 = cold-start 검증")
    n_zero = int((df["asof_pitcher_n"] == 0).sum())
    print(f"asof_pitcher_n == 0 인 행: {n_zero}")
    for c in RATE_COLS:
        n_null = int(df[c].isna().sum())
        n_null_and_zero = int((df[c].isna() & (df["asof_pitcher_n"] == 0)).sum())
        n_null_and_positive = int((df[c].isna() & (df["asof_pitcher_n"] > 0)).sum())
        print(f"  [{c}] 결측={n_null}, n==0에서 결측={n_null_and_zero}, n>0인데 결측={n_null_and_positive}")
    n_unique_pitchers_at_zero = df.loc[df["asof_pitcher_n"] == 0, "pitcher_id"].nunique()
    print(f"n==0 행의 고유 pitcher_id 수: {n_unique_pitchers_at_zero} (전체 투수 수: {df['pitcher_id'].nunique()})")


def step3_dedup_and_composition(df):
    section("3. 변수 간 중복/구성 관계")
    n_diff = int((df["asof_pitcher_n"] != df["asof_pitcher_pitchmix_n"]).sum())
    print(f"asof_pitcher_n != asof_pitcher_pitchmix_n 인 행: {n_diff} / {len(df)}")

    ball_strike_sum = df["asof_pitcher_ball_rate"] + df["asof_pitcher_strike_rate"]
    mae_bs = (1 - ball_strike_sum).abs().mean()
    print(f"ball_rate + strike_rate: mean={ball_strike_sum.mean():.4f}, |1-sum| 평균절대오차={mae_bs:.4f}")

    mix_sum = df["asof_pitcher_fastball_rate"] + df["asof_pitcher_breaking_rate"] + df["asof_pitcher_offspeed_rate"]
    mae_mix = (1 - mix_sum).dropna().abs().mean()
    print(f"fastball+breaking+offspeed: mean={mix_sum.mean():.6f}, |1-sum| 평균절대오차={mae_mix:.6f}")

    corr = df[["asof_pitcher_n"] + RATE_COLS + ["asof_pitcher_pitchmix_n"]].corr()
    print("\n상관행렬 (일부):")
    print(corr.loc[["asof_pitcher_success_rate", "asof_pitcher_reverse_rate"],
                    ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]].to_string())
    corr.to_csv(os.path.join(TAB_DIR, "03_correlation_matrix.csv"))

    fig, ax = plt.subplots(figsize=(8, 7))
    cols_for_heatmap = ["asof_pitcher_n"] + RATE_COLS
    cm = df[cols_for_heatmap].corr()
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm.columns))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cm.index))); ax.set_yticklabels(cm.index, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax)
    ax.set_title("Pearson correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_corr_heatmap.png"), dpi=120)
    plt.close(fig)
    return corr


def step4_univariate(df):
    section("4. 단변량 분포 (그리드)")
    desc = df[ALL_COLS].describe().T
    print(desc.to_string())
    desc.to_csv(os.path.join(TAB_DIR, "04_univariate_stats.csv"))

    sample = df.sample(min(300_000, len(df)), random_state=42)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    plot_cols = ["asof_pitcher_n"] + RATE_COLS
    for ax, col in zip(axes.flat, plot_cols):
        if col == "asof_pitcher_n":
            ax.hist(np.log1p(sample[col]), bins=40, color="#4C72B0")
            ax.set_title("log1p(asof_pitcher_n)")
        else:
            ax.hist(sample[col].dropna(), bins=40, color="#C44E52", range=(0, 1))
            ax.set_title(col)
    for ax in axes.flat[len(plot_cols):]:
        ax.axis("off")
    fig.suptitle(f"Group 4 — 분포 (300k 샘플, full row count={len(df):,})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_grid_distributions.png"), dpi=120)
    plt.close(fig)
    return desc


def step5_target_correlation(df):
    section("5. Target과의 단순 상관관계")
    rows = []
    for c in ALL_COLS:
        corr = df[c].corr(df[TARGET])
        rows.append({"column": c, "pearson_corr": corr})
    corr_df = pd.DataFrame(rows).sort_values("pearson_corr", key=lambda s: s.abs(), ascending=False)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(os.path.join(TAB_DIR, "05_target_correlation.csv"), index=False)

    q10_lo = df["asof_pitcher_success_rate"].quantile(0.1)
    q10_hi = df["asof_pitcher_success_rate"].quantile(0.9)
    low = df[df["asof_pitcher_success_rate"] <= q10_lo]
    high = df[df["asof_pitcher_success_rate"] >= q10_hi]
    print(f"\nasof_pitcher_success_rate 하위10% 집단 실제 성공률: {low[TARGET].mean()*100:.2f}%")
    print(f"asof_pitcher_success_rate 상위10% 집단 실제 성공률: {high[TARGET].mean()*100:.2f}%")
    return corr_df


def step6_target_by_decile(df, overall_rate):
    section("6. rate 변수 10분위별 target 관계")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, RATE_COLS):
        sub = df[[col, TARGET]].dropna().copy()
        sub["decile"] = pd.qcut(sub[col], 10, duplicates="drop")
        g = sub.groupby("decile", observed=True).agg(mean_feat=(col, "mean"), rate=(TARGET, "mean")).reset_index()
        ax.plot(g["mean_feat"], g["rate"] * 100, marker="o", color="#55A868")
        ax.axhline(overall_rate * 100, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("mean feature value in decile", fontsize=7)
        ax.set_ylabel("target rate (%)", fontsize=7)
        g.to_csv(os.path.join(TAB_DIR, f"06_decile_{col}.csv"), index=False)
    for ax in axes.flat[len(RATE_COLS):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_grid_decile_target.png"), dpi=120)
    plt.close(fig)


def step7_reliability_by_n(df):
    section("7. 표본 크기(n)에 따른 신뢰도 — success_rate 예시")
    rows = []
    for lo, hi, label in N_BINS:
        mask = (df["asof_pitcher_n"] > lo) & (df["asof_pitcher_n"] <= hi)
        sub = df.loc[mask, "asof_pitcher_success_rate"]
        rows.append({
            "n_bin": label, "rows": int(mask.sum()), "non_missing": int(sub.notna().sum()),
            "mean": sub.mean(), "std": sub.std(),
        })
    rel = pd.DataFrame(rows)
    print(rel.to_string(index=False))
    rel.to_csv(os.path.join(TAB_DIR, "07_reliability_by_n_success_rate.csv"), index=False)
    return rel


def step8_season_drift(df, overall_rate):
    section("8. 시즌별 추세")
    g_target = df.groupby("season")[TARGET].mean()
    g_success = df.groupby("season")["asof_pitcher_success_rate"].mean()
    g_reverse = df.groupby("season")["asof_pitcher_reverse_rate"].mean()
    g_middle = df.groupby("season")["asof_pitcher_middle_rate"].mean()
    g_fastball = df.groupby("season")["asof_pitcher_fastball_rate"].mean()
    g_breaking = df.groupby("season")["asof_pitcher_breaking_rate"].mean()
    g_offspeed = df.groupby("season")["asof_pitcher_offspeed_rate"].mean()
    g_missing = df.groupby("season")["asof_pitcher_success_rate"].apply(lambda s: s.isna().mean())

    season_tbl = pd.DataFrame({
        "target_rate": g_target, "success_rate_mean": g_success, "reverse_rate_mean": g_reverse,
        "middle_rate_mean": g_middle, "fastball_rate_mean": g_fastball, "breaking_rate_mean": g_breaking,
        "offspeed_rate_mean": g_offspeed, "success_rate_missing_pct": g_missing * 100,
    })
    print(season_tbl.to_string())
    season_tbl.to_csv(os.path.join(TAB_DIR, "08_season_drift.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(g_target.index, g_target.values * 100, marker="o", color="#C44E52")
    axes[0, 0].set_title("target rate by season")

    axes[0, 1].plot(g_success.index, g_success.values * 100, marker="o", label="success_rate")
    axes[0, 1].plot(g_reverse.index, g_reverse.values * 100, marker="o", label="reverse_rate")
    axes[0, 1].plot(g_middle.index, g_middle.values * 100, marker="o", label="middle_rate")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].set_title("control profile means by season")

    axes[1, 0].plot(g_fastball.index, g_fastball.values * 100, marker="o", label="fastball")
    axes[1, 0].plot(g_breaking.index, g_breaking.values * 100, marker="o", label="breaking")
    axes[1, 0].plot(g_offspeed.index, g_offspeed.values * 100, marker="o", label="offspeed")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("pitch-mix means by season")

    axes[1, 1].plot(g_missing.index, g_missing.values * 100, marker="o", color="#8172B2")
    axes[1, 1].set_title("success_rate missing rate by season (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_grid_season_drift.png"), dpi=120)
    plt.close(fig)
    return season_tbl


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_data()
    overall_rate = df[TARGET].mean()
    print(f"전체 행 수: {len(df):,}  |  전체 control_success 평균: {overall_rate*100:.4f}%")

    step1_quality(df)
    step2_coldstart(df)
    step3_dedup_and_composition(df)
    step4_univariate(df)
    step5_target_correlation(df)
    step6_target_by_decile(df, overall_rate)
    step7_reliability_by_n(df)
    step8_season_drift(df, overall_rate)

    section("완료")
    print(f"그래프: {FIG_DIR}")
    print(f"표: {TAB_DIR}")


if __name__ == "__main__":
    main()
