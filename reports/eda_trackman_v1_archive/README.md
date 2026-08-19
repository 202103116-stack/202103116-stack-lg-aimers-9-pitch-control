# Trackman EDA — `trackman_history.csv` 탐색 (exp_005 준비 단계)

담당: 이번 세션(Claude). 실행 코드는 [`run_eda.py`](run_eda.py) (재실행:
`python reports/eda_trackman/run_eda.py`, 실행 로그 원본은
[`eda_run_log.txt`](eda_run_log.txt)). 이 문서의 모든 수치는 그 로그와
`tables/*.csv`에서 그대로 가져왔다 — 추정치는 없음.

데이터: `data/trackman_history.csv`(2019~2024, 1,793,078행 x 30컬럼),
`data/train.csv`(1,475,092행, season/success_rate 교차검증용). `test.csv`
는 사용하지 않았다. 배경/설계 근거는 이 EDA를 위해 승인받은 plan
(`C:\Users\yeob0\.claude\plans\lg-aimers-9-groovy-glacier.md`) §1 참고 —
요약하면: **train/test와 trackman 사이엔 선수/팀 단위 공유 키가 전혀
없고**(§D), **실제 평가 시즌(2025)엔 trackman이 아예 없어서**, 조인은
`season`+`game_month` 기준 **누적(as-of) 집계**로만 안전하게 할 수 있다
(§E가 이 설계를 검증).

---

## A. 데이터 품질

- 8개 물리량 컬럼(`rel_speed`, `spin_rate`, `induced_vert_break`,
  `horz_break`, `extension`, `rel_height`, `rel_side`, `zone_speed`) 결측률
  전부 **1% 미만** (최고 `spin_rate` 0.70%) — 양호.
- `pitch_type_group` 분포: `fastball` 51.93%, `breaking` 28.60%,
  `offspeed` 18.23%, `other` 1.24%. **`other`만 물리량 결측률이 34%**로
  확 뛴다(다른 3개 그룹은 0.03~0.74%) — `other`로 분류된 투구는 트래킹
  자체가 실패한 경우가 많다는 뜻으로 보인다. 이후 집계에서 `other`는
  가급적 제외하거나 별도 취급하는 게 안전.
- `balls_before`/`strikes_before`/`outs_before` 야구 규칙 범위(각각
  0~3/0~2/0~2) 이탈: **balls_before 1건, strikes_before 1건,
  outs_before 95건** — 전부 합쳐도 0.005% 수준으로 무시 가능. 이탈 행은
  2022년(86건)·2023년(11건)에만 나타남 — 특정 시즌의 소수 기록 오류로
  보임. exp_005에서 트랙맨 집계 시 이 행들만 필터링하면 충분.
- `tagged_pitch_type`(17종)과 `pitch_type_group`(4종) 교차표
  ([`A5`](tables/A5_tagged_pitch_type_x_group.csv)) 확인 결과 분류
  기준이 대체로 상식과 일치(`Fastball`→fastball 96%, `Slider`/`Curveball`
  →breaking, `ChangeUp`/`Splitter`→offspeed). 소수 오탈자성 라벨
  (`SInker`, `Undefind`, `Slurve` 등)이 있지만 표본이 1~200건 수준이라
  무시 가능.

---

## B. train에도 있는 축 기준 신호

### B1. 구종군별 물리량 (트랙맨 내부 비교, 조인 키로는 못 씀)

| pitch_type_group | rel_speed 평균 | 표본 수 |
|---|---|---|
| fastball | 142.70 | 931,100 |
| offspeed | 130.47 | 326,803 |
| breaking | 127.53 | 512,845 |
| other | 125.55 | 14,713 |

구종군별 물리량 차이가 뚜렷하다(패스트볼이 변화구보다 평균 15km/h 가까이
빠름). 하지만 §1c에서 확인했듯 **`pitch_type_group`은 train/test에 없는
컬럼**이라 이 자체로는 조인 불가 — 아래 B2/B3처럼 train에도 있는 축
(좌우 매치업, 카운트 상황)으로 조건화해서 "이 조건에서 변화구 비율이
몇 %인가" 형태로 우회해야 실제로 쓸 수 있다.

### B2. 좌우 매치업(same_hand) — **exp_003 `same_hand_x_breaking_rate`의 물리적 근거 확인**

| same_hand | breaking 비율 | fastball 비율 | offspeed 비율 |
|---|---|---|---|
| 0 (다른 손) | 21.59% | 52.37% | 24.75% |
| 1 (같은 손) | **35.38%** | 51.50% | 11.93% |

같은 손 매치업일 때 변화구 비율이 **13.8%p 더 높다**(21.59%→35.38%,
상대적으로 +64%) — 야구 상식(같은 손 상대에겐 변화구가 더 잘 먹힘)과
정확히 일치. 그리고 (참고 데이터셋인) train.csv에서 같은 손 매치업의
`control_success_rate`는 51.49%로, 다른 손(53.26%)보다 **1.77%p 낮다**
([`B2c`](tables/B2c_train_success_rate_by_same_hand.csv)). 두 사실을
합치면: 같은 손 매치업 → 변화구를 더 많이 던짐 → 변화구는 원래 제구가
더 어려움(B1: 패스트볼보다 15km/h 느리고 무브먼트가 큼) → 관측된
성공률이 더 낮음, 이라는 일관된 인과 후보가 만들어진다(다만 데이터로
인과까지 확정할 순 없음 — 가설로 표기). exp_003에서 이미 채택된
`same_hand_x_breaking_rate` 피처가 왜 먹히는지에 대한 물리적 근거를
이번에 처음 확인한 셈이다.

### B3. 카운트 상황 — 3볼일 때 구종/구속이 실제로 바뀐다

| | breaking | fastball | offspeed | 평균 rel_speed |
|---|---|---|---|---|
| 3볼 아님 | 29.40% | 50.75% | 18.63% | 135.74 |
| **3볼** | **20.01%** | **64.60%** | 13.94% | **138.36** |

3볼 상황에서 투수들은 변화구를 확 줄이고(29.40%→20.01%) 패스트볼
비중을 크게 늘리며(50.75%→64.60%), 평균 구속도 2.6km/h 더 빠르다 —
"볼넷을 피하려고 스트라이크존에 강하게 꽂아 넣는다"는 통설과 일치.
그룹2 EDA가 찾은 "3볼 상황 성공률 저하" 패턴에 물리적 설명을 더해준다
— 다만 exp_003의 `three_ball_x_risp` 피처는 importance 0.007로 거의
안 쓰였다(`experiments/exp_003_eda_features_and_model.md`) — 이 트랙맨
신호(패스트볼 비중 증가, 구속 증가)를 직접 반영한 새 피처가 그 대체재로
더 나을 가능성이 있다(§F에서 제안).

### B4. 월별 물리량 — **11월(포스트시즌)이 확연히 다르다**

| game_month | rel_speed | spin_rate | induced_vert_break | 비고 |
|---|---|---|---|---|
| 3~10월 | 135.83~136.20 | 2155~2189 | 24.88~27.32 | 정규시즌, 안정적 |
| **11월** | **138.90** | **2199.76** | **29.31** | 포스트시즌 |

11월은 다른 달보다 구속·회전수·무브먼트가 전부 눈에 띄게 높다 —
[`D1`](tables/D1_month_dayofweek_counts.csv) 확인 결과 11월 표본은
시즌당 수천 건 수준(포스트시즌 경기 수 자체가 적음)으로, 다른 달
(2~6만 건대)보다 훨씬 적다. 포스트시즌에 강한 투수가 집중 등판하고
총력전 성향(불펜 조기 투입, 전력투구)이라 이런 차이가 난다는 해석이
자연스럽다(가설). `game_month`는 train/test에도 있고 매년 반복되는
축이라, exp_005에서 "포스트시즌 여부"(`game_month==11`) 같은 파생
피처를 만들 근거가 된다.

### B5. 요일 효과

요일별 구속/회전수 차이는 월요일(rel_speed 137.26)이 소폭 높은 정도로,
B4의 월별 차이(11월 vs 나머지)에 비하면 훨씬 미미하다
([`B5`](tables/B5_dayofweek_physical_stats.csv)) — 우선순위 낮음.

---

## C. 시즌 추세 — `eda_final` §E 교차검증

`eda_final` §E는 "2019→2024 성공률 7.85%p 하락이 (1) 투수 품질 구성
변화와 (2) 같은 품질 내 하락이 동시에 일어난 결과"라고 결론지었다.
트랙맨의 시즌별 물리량 추세를 보면 새로운 힌트가 있다.

| season | rel_speed | induced_vert_break | breaking 비율 | (참고)train success_rate |
|---|---|---|---|---|
| 2019 | 135.47 | 27.68 | 25.83% | 56.47% |
| 2020 | 135.57 | 27.93 | 25.56% | 53.27% |
| 2021 | 135.62 | 27.50 | 25.81% | 53.28% |
| 2022 | 136.43 | 25.20 | 29.69% | 52.89% |
| 2023 | 136.19 | 24.42 | 31.04% | 50.00% |
| 2024 | 136.35 | **22.80** | **32.48%** | 48.61% |

두 가지가 눈에 띈다:

1. **`induced_vert_break`(수직 무브먼트)가 27.68 → 22.80으로 6년간
   ~18% 줄었다** — success_rate와의 상관계수 **+0.870**
   ([`C4`](tables/C4_season_level_correlation.csv), 시즌 단위 6개 점
   기준이라 참고용). 무브먼트가 준다는 건 공이 더 "밋밋하게" 온다는
   뜻인데, 이게 왜 제구 **성공률** 하락과 같이 가는지는 데이터만으론
   설명 안 됨 — 다만 KBO가 2019년 반발계수를 낮춘 공인구를 도입했다는
   것은 잘 알려진 사실이라, 공 자체의 물리 특성 변화가 무브먼트에 영향
   줬을 가능성은 있다(가설, 확인 안 됨).
2. **변화구 비율이 25.83% → 32.48%로 6.65%p 늘었다.** B2/B3에서 확인한
   "변화구는 패스트볼보다 원래 제구가 어렵다"는 관계를 그대로 적용하면,
   **투수들이 해마다 변화구를 더 많이 던지게 된 것 자체가 리그 전체
   성공률을 끌어내리는 방향으로 작용했을 것**이라는 새로운 가설이
   생긴다 — `eda_final` §E가 쓴 "품질 분위"(타자 상대 성공/실패
   이력 기반)와는 독립적인 축이라, §E의 "같은 품질 안에서도 하락"
   현상을 부분적으로 설명할 후보가 될 수 있다(같은 asof 성공률 분위에
   속한 투수라도, 그 해엔 그 투수 자신이 변화구를 더 많이 던졌을 수
   있으므로). **이 역시 상관 수준의 관찰이며 인과를 확정하는 건
   아니다.**

`rel_speed`(구속)는 오히려 소폭 **증가**(135.47→136.35, 상관 -0.739)
하는데, 흔히 생각하는 "투수 구위 저하" 가설과는 반대 방향이다 — 최소한
"공이 약해져서 성공률이 떨어졌다"는 단순한 설명은 트랙맨 데이터로는
뒷받침되지 않는다.

---

## D. 조인 실현 가능성

- `(game_month, game_dayofweek)` 60개 조합 전부 최소 631건 이상 —
  세분화해도 표본 부족 문제 없음.
- `(balls_before, strikes_before, outs_before)`: 정상 범위 조합(balls
  0~3 x strikes 0~2 x outs 0~2, 총 36개)은 전부 수천~16만 건대로 충분.
  A4에서 확인한 범위 이탈 조합(예: balls=4)은 1~19건뿐이라 자연스럽게
  걸러짐 ([`D2`](tables/D2_count_situation_counts.csv)).
- `(pitch_type_group, same_hand, balls_before)` 33개 조합 중 표본
  5개 이하는 1개뿐(역시 범위 이탈 관련).
- `(season, game_month, pitch_type_group)` 188개 조합 중 표본 30개
  이하는 3개뿐(2020년 시즌 시작 지연 등 특수 상황으로 추정).

**결론: 세분화해도 표본 부족으로 못 쓰는 조합은 거의 없다** — 유일한
예외는 아래 E에서 다루는 "시간축 초반"(2019년 3월) 콜드스타트.

---

## E. asof 누적(as-of) 콜드스타트 시뮬레이션

plan §1c에서 설계한 `(season, game_month)` 기준 누적 집계(그 달보다
엄격히 이전 달까지의 트랙맨만 사용)가 실제로 얼마나 이른 시점부터
믿을 만한지 확인했다.

- **전체 기준**: 누적 표본이 1,000 미만인 (season,month) 버킷은 **딱
  1개**(2019년 3월, 정의상 누적 0) — 그다음 달인 2019년 4월부터는
  이미 누적 14,965건으로 충분. [`E1`](tables/E1_overall_cumulative_by_month.csv)
- **train.csv 행 기준 실제 영향**: 콜드스타트 구간(누적<1,000)에
  걸리는 train 행은 **13,395건, 전체의 0.908%**뿐이고, 전부 2019년
  3월 한 달에 몰려 있다.
- **세분화 기준(`same_hand`×`balls_before`, 8개 조합)**: 최악의 조합
  기준으로도 2019년 4월이면 이미 609건, 5월엔 2,477건까지 누적된다.
  100건 미만인 버킷도 2019년 3월(정의상 0건) 딱 1개뿐 —
  [`E3`](tables/E3_finegrained_cumulative_worst_case.csv).

**결론: 콜드스타트 문제는 사실상 2019년 3월 한 달(train 행 기준
0.9%)에 국한된다.** 그 한 달만 더 굵은 fallback(예: 전체 이력 평균,
혹은 그냥 결측 처리 후 모델이 median/기본값으로 처리)을 쓰면 되고,
plan에서 걱정했던 정교한 다단계 shrinkage backoff는 이 프로젝트
규모에서는 필요 이상일 수 있다 — 심플하게 "누적 표본이 0이면 전체
이력 평균으로 채운다" 정도의 1단계 fallback으로 충분해 보인다.

---

## F. exp_005 제안 — 어떤 트랙맨 피처를 넣을 것인가

위 결과를 종합해 신호가 뚜렷하고 조인이 실현 가능한 것만 추린다.
모두 `(season, game_month)` asof 누적 설계(§1c/E)를 따르고, 콜드스타트
(2019-03)는 전체 이력 평균으로 채운다.

### 채택 후보

| 피처 | 계산 | 근거 |
|---|---|---|
| `trackman_same_hand_breaking_rate_asof` | `(season,month)` 누적 x `same_hand` 별 breaking 비율, train 행의 `pitcher_hand==batter_hand` 로 조회 | B2 — same_hand=1일 때 35.38% vs 0일 때 21.59%, 격차가 크고 안정적. 기존 `same_hand_x_breaking_rate`(train 자체 asof 통계)와 상관될 수 있으므로 **모델에 함께 넣고 importance로 중복 여부 확인 필요** |
| `trackman_three_ball_fastball_rate_asof` / `trackman_three_ball_rel_speed_asof` | `(season,month)` 누적 x `three_ball` 별 fastball 비율/평균 구속, train 행의 `balls_before==3` 여부로 조회 | B3 — 3볼일 때 fastball 64.60% vs 50.75%, 구속 +2.6km/h. exp_003에서 실패한 `three_ball_x_risp`(importance 0.007)의 물리적 근거 기반 대체 후보 |
| `is_postseason` (`game_month == 11`) | train 컬럼만으로 계산(트랙맨 불필요, 트랙맨은 이 축이 유의미하다는 근거만 제공) | B4 — 11월 물리량이 확연히 다름(구속/회전수/무브먼트 전부 상승) → 포스트시즌이라는 별도 상황 축이 존재함을 시사 |
| `trackman_league_breaking_rate_asof` | `(season,month)` 누적, train 전체에 동일하게 붙는 시간축 매크로 피처(변화구 비율) | C2 — 시즌이 갈수록 변화구 비율 상승(25.83%→32.48%), success_rate 하락과 같은 방향. 개별 투수 매치업이 아니라 "리그 전체 트렌드"를 모델에 알려주는 용도 — `season` 컬럼과 정보가 겹칠 수 있어 **importance로 `season` 대비 부가가치 확인 필요** |

### 기각/보류

| 후보 | 이유 |
|---|---|
| 구종군(`pitch_type_group`) 자체를 조인 키로 사용 | §1c — train/test에 없는 컬럼, 현재 투구의 실측값을 흘리는 셈이라 원천적으로 불가 |
| `trackman_league_rel_speed_asof`(구속 매크로 트렌드) | C — 구속은 시즌이 갈수록 오히려 소폭 **증가**해 success_rate 하락과 반대 방향 상관(-0.739). 위 `induced_vert_break` 기반 피처보다 신호가 약함 — 1차로는 보류, 여유 있으면 나중에 ablation으로 확인 |
| 요일(`game_dayofweek`) 기반 피처 | B5 — 차이가 미미해 우선순위 낮음 |
| 좌우 매치업 물리량(무브먼트) 직접 사용 | B2 표에서 same_hand별 무브먼트 차이 자체는 크지 않음(비율 차이가 핵심 신호) — breaking rate로 충분, 별도 무브먼트 피처는 중복 위험 |

### 다음 단계 (plan §3-2~5)

1. `src/features.py`에 `(season, game_month)` asof 누적 집계 함수 추가
   (콜드스타트는 전체 이력 평균 1단계 fallback).
2. 위 "채택 후보" 4개를 더해 `src/train_exp005.py`로 CatBoost 검증 —
   exp_003(723.17) 대비 비교.
3. `same_hand_x_breaking_rate`/`season`과의 중복 여부를 feature
   importance로 확인(exp_004에서 배운 교훈: importance 0이라고 무조건
   빼면 안 되지만, 완전히 죽은 피처면 §F의 "보류" 후보부터 먼저
   제거 검토).
4. 결과를 `experiments/exp_005_trackman.md`에 기록, 채택 기준은
   기존과 동일(로컬 점수가 exp_003 이상).
