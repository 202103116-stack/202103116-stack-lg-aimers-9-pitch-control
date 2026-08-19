# Group 4 EDA — 투수의 과거 제구/구종 이력 (asof_pitcher_*)

이 그룹은 팀원 두 명이 각각 독립적으로 분석("4번 EDA", "4번 EDA-허찬")한
문서가 있었다. 두 문서 모두 같은 10개 컬럼을 다뤘고 결론도 서로
일치했다 — 이 리포트는 둘을 하나로 통합하고, 모든 수치를 실제 코드
재실행으로 재검증한 버전이다. 실행 코드는 [`run_eda.py`](run_eda.py)
(재실행: `python reports/eda_group4/run_eda.py`, 실행 로그 원본은
[`eda_run_log.txt`](eda_run_log.txt)). 이 문서의 모든 수치는 그 로그와
`tables/*.csv`에서 그대로 가져왔다 — 추정치는 없음.

데이터: `data/train.csv` (2019~2024, 1,475,092행). `test.csv`는 사용하지
않았다.

**검증 결과**: 두 원본 문서의 수치는 재검증 결과 **전부 정확히
일치**했다(상관계수 소수 넷째 자리까지, cold-start 792건, n-bin별
평균/표준편차까지 그대로 재현됨) — 고칠 부분은 없었고, 이 리포트는
두 문서의 좋은 내용을 하나로 합치고 그룹 1·2·3 리포트와 형식을
맞추는 데 집중했다. 두 문서 사이에 있던 유일한 차이는 관점 차이였다:
"4번 EDA"는 컬럼별 설명표에 좀 더 힘을 실었고, "4번 EDA-허찬"은
n구간별 신뢰도 표(§7)와 season drift 4분할 그림(§8)에 좀 더 힘을
실었다 — 이 리포트는 둘 다 포함한다.

---

## 0. 담당 컬럼 10개

| 구분 | 컬럼 | 설명 |
|---|---|---|
| 누적량 | `asof_pitcher_n` | 투수의 누적 투구 수 |
| 누적량 | `asof_pitcher_pitchmix_n` | 구종 비율 계산에 사용된 표본 수 |
| 제구 | `asof_pitcher_success_rate` | 과거 누적 제구 성공률 |
| 제구 | `asof_pitcher_reverse_rate` | 의도 반대성(포수 요구 반대 방향) 투구 비율 |
| 제구 | `asof_pitcher_middle_rate` | 가운데·위험 코스 투구 비율 |
| 결과 | `asof_pitcher_ball_rate` | 볼성 결과 비율 |
| 결과 | `asof_pitcher_strike_rate` | 스트라이크성 결과 비율 |
| 구종 | `asof_pitcher_fastball_rate` | fastball 계열 사용 비율 |
| 구종 | `asof_pitcher_breaking_rate` | breaking 계열 사용 비율 |
| 구종 | `asof_pitcher_offspeed_rate` | offspeed 계열 사용 비율 |

**CLAUDE.md와의 연결**: `asof_pitcher_middle_rate`, `asof_pitcher_ball_rate`,
`asof_pitcher_reverse_rate`는 공식 `control_success` 판정 기준(3가지
실패 조건 — 가운데 부근/존 크게 벗어남/포수 요구 반대)과 정확히
1:1 대응한다(CLAUDE.md "Target Definition" 섹션 참고). 즉 이 그룹의
컬럼들은 타깃 정의 자체를 요약한 값에 가깝다 — 그룹 4가 상관계수
기준으로 target과 가장 강하게 연결되는 이유이기도 하다(§5).

`success_rate` + `reverse_rate` + `middle_rate`의 합은 1이 아니다(서로
완전히 배타적인 분류가 아님) — 셋 다 서로 다른 관점의 비율이다.

---

## 1. 데이터 품질

- Shape `(1,475,092, 13)`, 고유 `pitcher_id` 792명.
- `asof_pitcher_n`, `asof_pitcher_pitchmix_n`: **결측 0건**, 범위 [0, 15,449].
- 나머지 8개 비율 컬럼: **정확히 792건 결측**(=0.054%), 범위 [0, 1] —
  범위 이탈 없음 ([`00_quality_summary.csv`](tables/00_quality_summary.csv)).

---

## 2. 결측치 = cold-start (버그 아님)

([콘솔 로그](eda_run_log.txt) §2)

- `asof_pitcher_n == 0`인 행 **정확히 792건**
- 8개 비율 컬럼 모두 이 792건에서만 결측(`n>0`인데 결측인 행은 **0건**)
- 이 792건의 고유 `pitcher_id` 수 = **792명** = 전체 투수 수와 정확히 일치

**결론: 결측은 데이터 오류가 아니라 각 투수가 데이터에 처음 등장하는
cold-start 상태를 뜻한다** — 투수 792명 전원이 정확히 한 번씩(첫
투구에서) 이 상태를 거친다. 모델에서는 0으로 단순 대체하기보다
그룹/전체 사전확률로 대체하거나(exp_002의 `SHRINKAGE_SPECS`가 이미
이 방식을 구현 중), cold-start 여부 자체를 별도 이진 변수로 만드는
것이 적절하다.

---

## 3. 변수 간 중복 및 구성 관계

([`03_correlation_matrix.csv`](tables/03_correlation_matrix.csv), 그림
`figures/01_corr_heatmap.png`)

### 3a. `asof_pitcher_n` == `asof_pitcher_pitchmix_n`

전체 1,475,092행에서 **완전히 동일**(불일치 0건) — 이 데이터셋에서는
두 표본 수 컬럼이 항상 같다. 다중공선성/중복을 줄이려면 하나만 사용.

### 3b. `ball_rate` + `strike_rate` ≠ 1

합의 평균은 0.8141, `|1-합|`의 평균절대오차는 **0.1859** — 평균적으로
약 18~19%는 "볼성도 스트라이크성도 아닌 결과"로 분류된다는 뜻이다.
**하나를 `1-나머지`로 대체하면 안 된다** — 두 컬럼은 서로 배타적인
이분류가 아니다.

### 3c. 구종 3개는 합이 정확히 1 (구성비 데이터)

`fastball_rate + breaking_rate + offspeed_rate`의 평균절대오차는
**0.000000** — 반올림 오차 수준을 넘어 사실상 정확히 1이다. 선형/
로지스틱 모델에서는 셋 중 하나를 반드시 제외해야 한다(완전
다중공선성). 트리 모델도 정보 중복을 줄이려면 하나를 제거할 수 있다.

### 3d. 주요 상관관계

| 변수 조합 | 상관계수 | 해석 |
|---|---|---|
| `success_rate` ↔ `reverse_rate` | **-0.8099** | 제구 성공률이 높을수록 의도 반대성 비율이 낮음 |
| `asof_pitcher_n` ↔ `asof_pitcher_pitchmix_n` | 1.0000 | 완전 중복 (§3a) |

다만 `success_rate`, `reverse_rate`, `middle_rate`는 합이 1이 아니므로
(§0) 서로 완전히 배타적인 결과로 해석하면 안 된다.

---

## 4. 단변량 분포

([`04_univariate_stats.csv`](tables/04_univariate_stats.csv), 그림
`figures/02_grid_distributions.png` — 300k 샘플)

| 컬럼 | mean | std | 25% | 50% | 75% |
|---|---|---|---|---|---|
| `asof_pitcher_n` | 2,661.3 | 2,690.0 | 684 | 1,776 | 3,725 |
| `success_rate` | 53.52% | 6.18%p | 49.91% | 53.20% | 56.90% |
| `reverse_rate` | 21.52% | 6.07%p | 17.89% | 21.80% | 25.11% |
| `middle_rate` | 14.19% | 3.18%p | 12.56% | 14.19% | 15.70% |
| `ball_rate` | 37.22% | 4.18%p | 34.81% | 36.94% | 39.22% |
| `strike_rate` | 44.19% | 3.96%p | 42.31% | 44.23% | 46.24% |
| `fastball_rate` | 55.44% | 9.91%p | 48.65% | 54.40% | 61.36% |
| `breaking_rate` | 28.93% | 10.83%p | 21.59% | 28.93% | 36.18% |
| `offspeed_rate` | 15.63% | 10.85%p | 7.50% | 14.06% | 21.70% |

`asof_pitcher_n`은 평균(2,661)이 중앙값(1,776)보다 커서 오른쪽으로
길게 늘어진 분포 — 일부 투수의 누적 투구 수가 매우 많다. 모델에서는
원본값과 함께 `log1p(asof_pitcher_n)` 변환을 고려할 수 있다
(`src/features.py`의 `pitcher_experience_log`로 이미 반영됨).

비율 변수들에서 관측되는 0%/100% 극단값은 누적 투구 수가 1~2개인
초기 cold-start 시점에 자연스럽게 발생하는 값이다 — 단순 이상치로
제거하지 않고, `asof_pitcher_n`(표본 크기)과 함께 해석하거나
smoothing(exp_002의 shrinkage)을 적용해야 한다.

---

## 5. Target과의 단순 상관관계

([`05_target_correlation.csv`](tables/05_target_correlation.csv))

| 컬럼 | 피어슨 상관계수 |
|---|---|
| `asof_pitcher_success_rate` | **+0.0843** |
| `asof_pitcher_reverse_rate` | **-0.0795** |
| `asof_pitcher_middle_rate` | -0.0361 |
| `asof_pitcher_ball_rate` | -0.0177 |
| `asof_pitcher_offspeed_rate` | +0.0138 |
| `asof_pitcher_breaking_rate` | -0.0136 |
| `asof_pitcher_n` / `pitchmix_n` | -0.0116 |
| `asof_pitcher_strike_rate` | +0.0039 |
| `asof_pitcher_fastball_rate` | -0.0002 |

`success_rate`가 담당 10개 중 target과 가장 강한 양의 관계,
`reverse_rate`가 가장 강한 음의 관계를 보인다 — CLAUDE.md의 타깃 정의
(가운데 투구/존 이탈/반대 방향 = 실패)와 정확히 부합하는 결과다.

**분위수로 보면 절대 상관계수 숫자보다 신호가 훨씬 크다**: `success_rate`
하위 10% 집단의 실제 제구 성공률은 **45.68%**, 상위 10% 집단은
**61.23%** — 그 차이는 15.5%p로 상당히 크다. target이 0/1 이진값이라
점 단위 선형 상관은 작게 나오지만, 구간별 그룹 평균으로 보면 신호가
분명하다(§6). 이는 그룹 3(`li`)에서도 동일하게 관측된 패턴이다 —
피어슨 상관이 작다고 신호가 없다고 단정하면 안 된다.

---

## 6. rate 변수 10분위별 target 관계

(각 컬럼별 표: `tables/06_decile_*.csv`, 그림 `figures/03_grid_decile_target.png`)

| 피처 | 패턴 |
|---|---|
| `success_rate` | 분위 낮음→높음에서 target이 대략 46%→61%로 단조 상승 |
| `reverse_rate` | 분위 낮음→높음에서 target이 단조 하락 |
| `middle_rate`/`ball_rate` | 대체로 하락 방향, success/reverse보다 완만 |
| `strike_rate`/구종 3개 | 변화가 작거나 뚜렷한 방향 없이 들쭉날쭉 |

**담당 10개 중 target과 가장 분명한 신호는 `success_rate`(높을수록
target↑)와 `reverse_rate`(높을수록 target↓)다.** 나머지는 단독보다
다른 그룹 컬럼(카운트·주자 상황 등)과 결합했을 때 유용할 가능성이
높다.

---

## 7. 표본 크기(n)에 따른 신뢰도

([`07_reliability_by_n_success_rate.csv`](tables/07_reliability_by_n_success_rate.csv))

`asof_pitcher_success_rate`를 `asof_pitcher_n` 구간별로 나눠 평균과
표준편차를 본 결과:

| n 구간 | 행 수 | 유효값 | 평균 | 표준편차 |
|---|---|---|---|---|
| 0 | 792 | 0 | (결측) | (결측) |
| 1 | 792 | 792 | 0.5518 | **0.4976** |
| 2-10 | 7,104 | 7,104 | 0.5407 | 0.2516 |
| 11-30 | 15,291 | 15,291 | 0.5461 | 0.1529 |
| 31-100 | 48,229 | 48,229 | 0.5438 | 0.1090 |
| 101-300 | 115,971 | 115,971 | 0.5387 | 0.0776 |
| 301-1000 | 307,236 | 307,236 | 0.5361 | 0.0609 |
| 1001+ | 979,677 | 979,677 | 0.5339 | **0.0474** |

`n=1`일 때는 값이 무조건 0 또는 1이라 표준편차가 0.50에 가깝다(동전
던지기와 동일한 분산 구조). `n`이 커질수록 표준편차가 꾸준히
줄어든다(0.50 → 0.05) — **비율 변수는 반드시 표본 수(`asof_pitcher_n`)와
함께 해석해야 하며, 표본이 적은 값을 그대로 믿으면 안 된다.**
이것이 exp_002의 empirical-Bayes shrinkage(`k=50`)가 정확히 겨냥하는
문제다 — `src/features.py`의 `SHRINKAGE_SPECS` docstring도 "1st-5th
percentile of `asof_pitcher_n`이 18~107" 라고 명시하는데, 이 표는 그
근거를 더 세밀하게 뒷받침한다(n<100 구간에서 표준편차가 0.11~0.50으로
매우 큼).

---

## 8. 시즌별 추세

([`08_season_drift.csv`](tables/08_season_drift.csv), 그림
`figures/04_grid_season_drift.png`)

| season | target_rate | success_rate 평균 | reverse_rate 평균 | middle_rate 평균 | success_rate 결측률 |
|---|---|---|---|---|---|
| 2019 | 56.47% | 55.98% | 18.23% | 12.49% | 0.1495% |
| 2020 | 53.27% | 54.72% | 19.19% | 13.73% | 0.0451% |
| 2021 | 53.28% | 53.55% | 22.13% | 14.10% | 0.0384% |
| 2022 | 52.89% | 53.54% | 22.34% | 14.38% | 0.0356% |
| 2023 | 50.00% | 52.37% | 23.30% | 14.75% | 0.0257% |
| 2024 | 48.61% | 51.14% | 23.72% | 15.57% | 0.0320% |

**중요 발견**: `target_rate`(당해 시즌 실제 성공률)뿐 아니라
`asof_pitcher_success_rate`의 시즌 평균(=투수들이 그 시즌에 들고 온
"과거 누적" 성공률)도 함께 하락한다(55.98%→51.14%) — 즉 시즌 하락
추세가 target에서만 보이는 게 아니라 투수들의 과거 누적 지표 자체에도
이미 반영되어 있다. `reverse_rate`/`middle_rate` 평균은 반대로
상승한다(제구 실패 신호가 누적 지표에도 그대로 나타남).

구종 비율(`fastball`/`breaking`/`offspeed`)은 시즌별로 비교적 안정적
(fastball 54.5~56.4% 범위)이고, 결측률(=cold-start 비율)은 전 시즌
0.03~0.15%로 매우 낮게 유지된다.

**→ random split보다 2019-2023 학습/2024 검증 같은 시간순 검증이
적절하다는 CLAUDE.md의 Validation Policy를 다시 한번 뒷받침한다.**

---

## 9. 변수 중복 및 모델링 주의사항 (요약)

| 컬럼 | 문제 | 처리 방침 |
|---|---|---|
| `asof_pitcher_pitchmix_n` | `asof_pitcher_n`과 100% 동일 (§3a) | 하나만 사용 |
| `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate` | 합이 1이 아님(§3b), 서로 `1-x`로 대체 불가 | 둘 다 독립 정보로 유지 |
| `asof_pitcher_fastball/breaking/offspeed_rate` | 합이 정확히 1(구성비, §3c) | 선형 모델은 하나 제외, 트리 모델은 선택적 |
| 8개 비율 컬럼 전체 | 표본 작을수록 신뢰도 낮음(§7, 표준편차 0.50→0.05) | 원본값 그대로 쓰지 않고 `asof_pitcher_n`과 결합하거나 shrinkage 적용 |

---

## 10. Feature Engineering 후보

| 후보 | 근거 | 기존 피처와의 관계 |
|---|---|---|
| `pitcher_cold_start` (`asof_pitcher_n==0`) | §2 — 결측 792건이 정확히 cold-start | 신규 제안, `pitcher_experience_log`(log1p)와는 다른 정보(이진 플래그) |
| empirical-Bayes shrinkage (`shrunk = (n·r+k·prior)/(n+k)`) | §7 — n<100에서 표준편차가 0.11~0.50으로 매우 큼 | **이미 `src/features.py`의 `SHRINKAGE_SPECS`/`apply_shrinkage`로 존재**(exp_002), k=50이 §7의 n 분포와 일치함을 재확인 |
| `pitcher_middle_minus_success`, `pitcher_reverse_under_pressure` 등 | §5, §6 — success/reverse가 target과 가장 강하게 연결 | **이미 `src/features.py`에 exp_002 피처로 존재** |
| `success_rate × season` calibration | §8 — 시즌 하락이 target뿐 아니라 누적 지표 자체에도 반영됨 | exp_002 shrinkage prior가 2019-2024 전체 평균 기반이라 최근 시즌 가중을 반영 못할 가능성 — validation에서 최근 시즌 가중 방식과 비교 필요(그룹 2 리포트 §13-1과 동일 결론) |

---

## 11. 삭제 후보

| 컬럼 | 근거 | 처리 방침 |
|---|---|---|
| `asof_pitcher_pitchmix_n` | `asof_pitcher_n`과 100% 동일 (§3a) | 삭제 전 baseline vs 제거 ablation 필요 — 단, 완전 중복이므로 이 그룹 안에서는 가장 안전한 삭제 후보 |

`ball_rate`/`strike_rate`/`middle_rate`/구종 3개는 합 제약이 있어도
서로 독립적인 정보를 담고 있어(§3b, §3c) 삭제 후보로 보지 않는다.

---

## 12. 추가 교차 분석 후보 (담당 범위 밖 — 최종 통합 EDA에서 다룸)

| 후보 | 왜 필요해 보이는지 |
|---|---|
| `pitcher_hand × batter_hand × 과거 구종군 비율` | 그룹 1 원본 노트 §3.2 — 손 유형 조합 안에서도 구종 성향에 따라 46.88~56.80%까지 벌어짐 |
| `success_rate × li` / `success_rate × runner_in_scoring_position` | exp_002에 이미 `pitcher_success_under_pressure`, `pitcher_success_x_risp`로 존재 — 그룹 3/4 교차 검증을 최종 리포트에서 재확인 |
| `success_rate × count(balls×strikes)` | 그룹 2 리포트 §12에서 이미 후보로 제시 |
| `asof_batter_*`(그룹 5)와의 상호작용 | `pitcher_minus_batter_success` 등 이미 존재 — 그룹 5 EDA 확보 후 재검증 |

---

## 13. 팀 회의용 핵심 결론

1. **관찰**: 결측 792건은 데이터 오류가 아니라 투수 792명 전원이
   정확히 한 번씩 거치는 cold-start(첫 등장) 상태다.
   **근거**: §2 — `asof_pitcher_n==0`인 행 792건 = 결측 792건의
   pitcher_id 수 792명 = 전체 투수 수.
   **해석**: 오류 처리(제거/평균 대체)가 아니라 cold-start 전용 처리가
   필요.
   **다음 action**: 신규 없음 — exp_002 shrinkage가 이미 이를 반영.

2. **관찰**: `asof_pitcher_n`과 `asof_pitcher_pitchmix_n`은 완전
   중복이고, 구종 3개(fastball/breaking/offspeed)는 합이 정확히 1이다.
   **근거**: §3a(불일치 0건), §3c(MAE=0.000000).
   **해석**: `pitchmix_n`은 정보 손실 없이 제거 가능한 유일한 완전
   중복 컬럼.
   **다음 action**: baseline vs `pitchmix_n` 제거 ablation.

3. **관찰**: `success_rate`/`reverse_rate`가 담당 10개 중 target과
   가장 강하게 연결되며, 이는 공식 타깃 정의(3가지 실패 조건)와
   정확히 대응한다.
   **근거**: §5 — 상관계수 +0.0843/-0.0795(10개 중 최대), 분위수
   기준 하위10% 45.68% vs 상위10% 61.23%.
   **해석**: 그룹 4가 이번 5개 그룹 중 target 예측에 가장 직접적으로
   기여할 그룹일 가능성이 높음.
   **다음 action**: 신규 없음 — 이미 exp_002 feature importance 상위권.

4. **관찰**: 비율 변수는 표본이 적을수록(`asof_pitcher_n` 작을수록)
   표준편차가 매우 크다(n=1일 때 0.4976 → n=1001+일 때 0.0474).
   **근거**: §7 표.
   **해석**: 소표본 구간의 극단값(0%/100%)을 그대로 믿으면 안 됨.
   **다음 action**: 신규 없음 — exp_002 shrinkage(`k=50`)가 이미
   이 문제를 겨냥해 설계됨, k값이 §7 분포와 일치하는지만 재확인.

5. **관찰**: 시즌 하락 추세가 target뿐 아니라 `success_rate`/
   `reverse_rate`의 시즌 평균 자체에도 반영되어 있다.
   **근거**: §8 — success_rate 시즌평균 55.98%(2019)→51.14%(2024).
   **해석**: 투수들이 시즌 초에 "들고 오는" 과거 이력 자체가 이미
   그 시즌의 하락한 기저율을 어느 정도 반영 — 완전히 독립적인 신호는
   아님.
   **다음 action**: exp_002 shrinkage prior가 전체 기간 평균 기반이라
   최근 시즌 가중이 필요한지 validation에서 비교(그룹 2 리포트와
   동일한 다음 action).

---

## 14. 중요 원칙 (본 EDA 전체에 적용)

- raw data(`data/train.csv`) 수정 없음, test data 기반 fitting/분포
  사용 없음, 미래 정보 사용 없음.
- `asof_*` 컬럼은 전부 투구 시점까지만 누적된 공식 제공값이라 이
  자체로 row-local, 누수 위험 없음(CLAUDE.md 규칙 9번).
- EDA 결과만으로 컬럼을 채택·제거하지 않음 — 모든 후보는 validation
  ablation이 필요하다고 명시.
- 상관관계가 낮다는 이유만으로 제거 결정을 내리지 않음(§5 — 피어슨은
  작아도 분위수 그룹 평균 차이는 15.5%p로 큼).
- 이 문서의 모든 수치는 `eda_run_log.txt`와 `tables/*.csv`에서 실제
  실행된 값만 사용 — 추정치 없음.
- 작은 차이는 과도하게 해석하지 않음, 표현은 "연관"·"패턴"·"가설"로
  한정 — 인과관계로 서술하지 않음.
