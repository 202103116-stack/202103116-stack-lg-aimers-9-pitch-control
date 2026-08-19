"""Group 5 EDA — 투수 최근 경기 폼 추세 + 타자 상대 과거 기록 (9개 컬럼).

담당자 문서가 아직 없어 직접 수행한다 (그룹 2 방식과 동일: 실제 코드
실행으로만 수치를 만들고, 문서가 도착하면 대조/보강한다).

담당 컬럼:
    asof_pitcher_prev1_game_success_rate, asof_pitcher_prev3_game_success_rate,
    asof_pitcher_prev5_game_success_rate, asof_pitcher_prev1_game_middle_rate,
    asof_pitcher_prev3_game_middle_rate, asof_pitcher_prev5_game_middle_rate,
    asof_batter_n, asof_batter_success_rate, asof_batter_middle_rate

전부 공식 asof_* 값(투구 시점까지만 누적)이라 row-local, CLAUDE.md
누수 규칙과 충돌하지 않는다.

실행:
    python reports/eda_group5/run_eda.py
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
OUT_DIR = "./reports/eda_group5"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

PREV_SUCCESS_COLS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]
PREV_MIDDLE_COLS = [
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]
BATTER_COLS = ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"]
ALL_COLS = PREV_SUCCESS_COLS + PREV_MIDDLE_COLS + BATTER_COLS
TARGET = "control_success"

N_BINS = [(-1, 0, "0"), (0, 1, "1"), (1, 10, "2-10"), (10, 30, "11-30"), (30, 100, "31-100"),
          (100, 300, "101-300"), (300, 1000, "301-1000"), (1000, np.inf, "1001+")]


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data():
    usecols = ["season", "pitcher_id", "batter_id", "asof_pitcher_n", "asof_pitcher_success_rate"] + ALL_COLS + [TARGET]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=usecols)
    return df


def step1_quality(df):
    section("1. 데이터 품질 확인")
    print(f"train.csv shape: {df.shape}")
    print(f"고유 pitcher_id: {df['pitcher_id'].nunique()}, 고유 batter_id: {df['batter_id'].nunique()}")
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


def step2_batter_coldstart(df):
    section("2. asof_batter_* 결측 = cold-start 검증 (그룹4와 동일 패턴)")
    n_zero = int((df["asof_batter_n"] == 0).sum())
    print(f"asof_batter_n == 0 인 행: {n_zero}")
    for c in ["asof_batter_success_rate", "asof_batter_middle_rate"]:
        n_null = int(df[c].isna().sum())
        n_null_and_zero = int((df[c].isna() & (df["asof_batter_n"] == 0)).sum())
        n_null_and_positive = int((df[c].isna() & (df["asof_batter_n"] > 0)).sum())
        print(f"  [{c}] 결측={n_null}, n==0에서 결측={n_null_and_zero}, n>0인데 결측={n_null_and_positive}")
    n_unique_batters_at_zero = df.loc[df["asof_batter_n"] == 0, "batter_id"].nunique()
    print(f"n==0 행의 고유 batter_id 수: {n_unique_batters_at_zero} (전체 타자 수: {df['batter_id'].nunique()})")


def step3_prev_game_missing_hierarchy(df):
    section("3. prev1/3/5 결측 계층 구조 — '경기 수 부족'형 cold-start")
    rows = []
    for c in PREV_SUCCESS_COLS + PREV_MIDDLE_COLS:
        n_null = int(df[c].isna().sum())
        rows.append({"column": c, "n_null": n_null, "null_pct": n_null / len(df) * 100})
    miss = pd.DataFrame(rows)
    print(miss.to_string(index=False))
    miss.to_csv(os.path.join(TAB_DIR, "03_prev_missing_hierarchy.csv"), index=False)

    # prev1 결측인데 prev3/prev5는 결측 아님 -> 있으면 이상
    p1, p3, p5 = "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate"
    anomaly_13 = int((df[p1].isna() & df[p3].notna()).sum())
    anomaly_35 = int((df[p3].isna() & df[p5].notna()).sum())
    print(f"\nprev1 결측인데 prev3는 있음(이상 후보): {anomaly_13}건")
    print(f"prev3 결측인데 prev5는 있음(이상 후보): {anomaly_35}건")
    print("(prev5>=5경기, prev3>=3경기, prev1>=1경기 필요 — 계층 구조상 prev1 결측률 <= prev3 <= prev5 이어야 함)")

    # asof_pitcher_n==0(첫 투구)일 때 prev1도 결측인지
    n_zero = df["asof_pitcher_n"] == 0
    p1_null_at_n0 = int((df.loc[n_zero, p1].isna()).sum())
    print(f"\nasof_pitcher_n==0인 행 {int(n_zero.sum())}건 중 prev1도 결측: {p1_null_at_n0}건")
    return miss


def step4_univariate_grid(df):
    section("4. 단변량 분포 (3x3 그리드)")
    desc = df[ALL_COLS].describe().T
    print(desc.to_string())
    desc.to_csv(os.path.join(TAB_DIR, "04_univariate_stats.csv"))

    sample = df.sample(min(300_000, len(df)), random_state=42)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    plot_cols = PREV_SUCCESS_COLS + PREV_MIDDLE_COLS + BATTER_COLS
    for ax, col in zip(axes.flat, plot_cols):
        if col == "asof_batter_n":
            ax.hist(np.log1p(sample[col]), bins=40, color="#4C72B0")
            ax.set_title("log1p(asof_batter_n)", fontsize=9)
        else:
            ax.hist(sample[col].dropna(), bins=40, color="#55A868", range=(0, 1))
            ax.set_title(col, fontsize=8)
    fig.suptitle(f"Group 5 — 분포 (300k 샘플, full row count={len(df):,})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_grid_distributions.png"), dpi=120)
    plt.close(fig)
    return desc


def step5_nested_correlation(df):
    section("5. prev1/prev3/prev5 상관관계 (중첩 윈도우 구조 확인)")
    corr_success = df[PREV_SUCCESS_COLS].corr()
    corr_middle = df[PREV_MIDDLE_COLS].corr()
    print("[success_rate 계열]")
    print(corr_success.to_string())
    print("\n[middle_rate 계열]")
    print(corr_middle.to_string())
    corr_success.to_csv(os.path.join(TAB_DIR, "05_corr_prev_success.csv"))
    corr_middle.to_csv(os.path.join(TAB_DIR, "05_corr_prev_middle.csv"))

    corr_with_longrun = df[["asof_pitcher_success_rate"] + PREV_SUCCESS_COLS].corr()["asof_pitcher_success_rate"]
    print("\n장기 누적(asof_pitcher_success_rate)과의 상관:")
    print(corr_with_longrun.to_string())
    return corr_success, corr_middle


def step6_isolated_trend_check(df):
    section("6. exp_002 pitcher_form_trend_isolated 공식 검증")
    p1, p3, p5 = df["asof_pitcher_prev1_game_success_rate"], df["asof_pitcher_prev3_game_success_rate"], df["asof_pitcher_prev5_game_success_rate"]
    early_only = (5 * p5 - 3 * p3) / 2
    trend = p1 - early_only
    valid = early_only.between(-0.5, 1.5)  # 합리적 범위(비율이므로 대략 0~1 근처여야 함, 약간의 마진)
    print(f"early_only 계산 가능 행(3개 모두 non-null): {trend.notna().sum():,} / {len(df):,}")
    print(f"early_only 기술통계:\n{early_only.describe().to_string()}")
    print(f"\nearly_only가 [-0.5, 1.5] 범위를 벗어나는 행: {(~valid & trend.notna()).sum():,} 건 "
          f"({(~valid & trend.notna()).mean()*100:.4f}%) — 음수/1 초과는 대수적 외삽이 실제 비율 범위를 벗어난 경우")
    print(f"\ntrend(=pitcher_form_trend_isolated) 기술통계:\n{trend.describe().to_string()}")

    corr_trend_target = trend.corr(df["control_success"])
    print(f"\ntrend vs target 피어슨 상관계수: {corr_trend_target:.4f}")

    out = pd.DataFrame({"early_only": early_only, "trend": trend, "target": df["control_success"]}).dropna()
    out.sample(min(5000, len(out)), random_state=42).to_csv(os.path.join(TAB_DIR, "06_isolated_trend_sample.csv"), index=False)
    return early_only, trend


def step7_target_correlation_and_decile(df, overall_rate):
    section("7. Target과의 단순 상관관계")
    rows = []
    for c in ALL_COLS:
        corr = df[c].corr(df[TARGET])
        rows.append({"column": c, "pearson_corr": corr})
    corr_df = pd.DataFrame(rows).sort_values("pearson_corr", key=lambda s: s.abs(), ascending=False)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(os.path.join(TAB_DIR, "07_target_correlation.csv"), index=False)

    section("7b. 10분위별 target 관계 (3x3 그리드)")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plot_cols = PREV_SUCCESS_COLS + PREV_MIDDLE_COLS + BATTER_COLS
    for ax, col in zip(axes.flat, plot_cols):
        sub = df[[col, TARGET]].dropna().copy()
        try:
            sub["decile"] = pd.qcut(sub[col], 10, duplicates="drop")
        except ValueError:
            sub["decile"] = pd.cut(sub[col], 10)
        g = sub.groupby("decile", observed=True).agg(mean_feat=(col, "mean"), rate=(TARGET, "mean")).reset_index()
        ax.plot(g["mean_feat"], g["rate"] * 100, marker="o", color="#C44E52")
        ax.axhline(overall_rate * 100, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(col, fontsize=8)
        ax.set_xlabel("mean feature value in decile", fontsize=6)
        ax.set_ylabel("target rate (%)", fontsize=6)
        g.to_csv(os.path.join(TAB_DIR, f"07_decile_{col}.csv"), index=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_grid_decile_target.png"), dpi=120)
    plt.close(fig)
    return corr_df


def step8_reliability_by_batter_n(df):
    section("8. asof_batter_n 구간별 신뢰도 (그룹4 §7과 동일 방식)")
    rows = []
    for lo, hi, label in N_BINS:
        mask = (df["asof_batter_n"] > lo) & (df["asof_batter_n"] <= hi)
        sub = df.loc[mask, "asof_batter_success_rate"]
        rows.append({
            "n_bin": label, "rows": int(mask.sum()), "non_missing": int(sub.notna().sum()),
            "mean": sub.mean(), "std": sub.std(),
        })
    rel = pd.DataFrame(rows)
    print(rel.to_string(index=False))
    rel.to_csv(os.path.join(TAB_DIR, "08_reliability_by_batter_n.csv"), index=False)
    return rel


def step9_season_drift(df):
    section("9. 시즌별 추세")
    g = df.groupby("season")[["asof_batter_success_rate", "asof_batter_middle_rate"] + PREV_SUCCESS_COLS].mean()
    g["target_rate"] = df.groupby("season")[TARGET].mean()
    g["batter_missing_pct"] = df.groupby("season")["asof_batter_success_rate"].apply(lambda s: s.isna().mean() * 100)
    print(g.to_string())
    g.to_csv(os.path.join(TAB_DIR, "09_season_drift.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(g.index, g["target_rate"] * 100, marker="o", label="target", color="black")
    axes[0].plot(g.index, g["asof_batter_success_rate"] * 100, marker="o", label="asof_batter_success_rate")
    axes[0].legend(fontsize=8)
    axes[0].set_title("target vs asof_batter_success_rate by season")

    for c in PREV_SUCCESS_COLS:
        axes[1].plot(g.index, g[c] * 100, marker="o", label=c.replace("asof_pitcher_", ""))
    axes[1].legend(fontsize=7)
    axes[1].set_title("prev1/3/5 success rate means by season")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_season_drift.png"), dpi=120)
    plt.close(fig)
    return g


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_data()
    overall_rate = df[TARGET].mean()
    print(f"전체 행 수: {len(df):,}  |  전체 control_success 평균: {overall_rate*100:.4f}%")

    step1_quality(df)
    step2_batter_coldstart(df)
    step3_prev_game_missing_hierarchy(df)
    step4_univariate_grid(df)
    step5_nested_correlation(df)
    step6_isolated_trend_check(df)
    step7_target_correlation_and_decile(df, overall_rate)
    step8_reliability_by_batter_n(df)
    step9_season_drift(df)

    section("완료")
    print(f"그래프: {FIG_DIR}")
    print(f"표: {TAB_DIR}")


if __name__ == "__main__":
    main()
