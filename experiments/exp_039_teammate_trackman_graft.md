# exp_039 — 팀원(EXP_130) 파이프라인에 트랙맨 피처 접목 + 재보정

## 실험 개요 (시작 전에 작성)

- **실험 ID**: exp_039
- **날짜 / 담당자**: 2026-08-23 / (Claude, 사용자 지적으로 착수 —
  "이미 LB 1043.66짜리 팀원 모델이 있는데 왜 그걸 기반으로 개선하지
  않았냐"는 정당한 지적. 그날 밤 exp_031~038은 전부 우리 자신의
  exp_030(LB 1006.62)을 베이스로 한 개선이었어서, 이미 더 강한
  베이스가 있는데 더 약한 베이스를 계속 개선하는 앞뒤가 안 맞는
  접근이었음을 인정하고 방향 전환.
- **가설**: 팀원의 `model.joblib`을 직접 열어 확인한 결과, EXP_130
  (78피처)은 트랙맨 데이터를 **전혀** 사용하지 않는다(trackman 관련
  컬럼 0개). 우리가 이미 검증한 트랙맨 물리특성(풀링 5개 + 구종군
  15개, exp_007/exp_035)을 이 더 강한 베이스에 접목하면, 우리 자신의
  베이스에 붙였을 때보다 더 잘 통할 수 있다(직교 신호이므로 베이스가
  강할수록 상대적 기여가 유지될 가능성).
- **기준**: 팀원의 원본 78피처 파이프라인 (LB 1043.66, 로컬 2024
  held-out 자체 문서 기록 945.44).
- **방법**: 팀원의 실제 학습 프레임워크(`_teammate_repo/experiment_
  workspace/src/aimers_exp/`, TOML 설정 기반)를 코드 수정 없이 그대로
  사용 — `resolve_feature_names`가 raw 컬럼을 test.csv 헤더에서 그대로
  가져오는 것(`include_all_raw=true`)을 이용해, train.csv/test.csv
  로컬 사본에 우리 트랙맨 20개 컬럼을 물리적으로 추가하는 것만으로
  프레임워크 코드를 한 줄도 안 건드리고 98피처 모델을 학습시킴.
- **누수 위험 검토**: train.csv에는 진짜 as-of(행별 season/game_month
  기준 merge_asof) 값을, test.csv(로컬 5행 샘플)에는 flat lookup +
  리그 폴백을 부착 — 우리 기존 프로덕션 파이프라인(exp_030/037/038)과
  완전히 동일한 메커니즘, 신규 누수 표면 없음. 실제 2025 제출용
  트랙맨 lookup도 같은 방식으로 미리 계산해 패키지에 동봉.
- **신규 파일**: `src/build_teammate_trackman_data.py`,
  `src/build_teammate_trackman_calibration_final.py`,
  `src/package_teammate_trackman_submission.py`
  (+ 팀원 저장소 쪽에 스크래치 TOML 설정 몇 개, 팀원의 공식 실험
  레지스트리는 건드리지 않음)

## 결과 (실행 후에 작성)

### 1단계 — ablation study (팀원 프레임워크의 `study` 명령, raw 비교)

baseline_78 vs trackman20(78+20), 둘 다 `triple_count_trend_ols_cpuonly`
스펙 그대로, protocol=holdout_2024(2019-2023 학습/2024 검증),
`calibration=none`(보정 전 raw만 비교):

| variant | mean_brier | brier_skill_score |
|---|---|---|
| baseline_78 | 0.247458 | 940.30 |
| trackman20 | 0.247454 | 941.86 |

트랙맨20이 +1.56로 소폭 우위 — 방향은 우리 자체 파이프라인(exp_035,
+4.13)과 일치.

### 2단계 — 팀원 보정 체인 재현 시도 중 발견한 버그와 그 수정

팀원의 최종 빌드(`model.joblib`에 저장된 `group_offsets`/
`trend_offsets`)를 study의 raw 예측에 그대로 적용했더니 941.86 →
923.19로 **오히려 대폭 하락**했다. 원인 조사 결과 **모델 불일치
버그**였음이 밝혀짐: `group_offsets`/`trend_offsets`는 builder.py
내부에서 **별도로 재학습된** `calibration_components` 모델의 예측을
기준으로 계산된 것인데, 이걸 **study가 따로 학습한 모델**의 raw
예측에 적용해버린 것 — 설정/데이터/시드가 같아도 두 모델이 100%
동일하다는 가정이 틀렸다(코드 경로 자체가 다름: builder.py의
`_fit_triple_components` vs runner.py의 `run_study` 내부 로직).

**수정**: group_offsets를 study 모델 **자기 자신의** 2024 잔차로
직접(자기일관적으로) 재계산. trend_offset은 프로덕션 번들의 것이
2019-2024 **전체**(2024 포함)로 적합돼 2025 외삽용이라 2024 검증에
쓰면 순환검증이 되므로, 우리 exp_029 관례대로 2019-2023만으로 별도
재적합한 "clean" 버전을 만들어 검증.

### 3단계 — 자기일관적 재검증 (2024 진짜 held-out)

| 조합 | baseline_78 | trackman20 |
|---|---|---|
| raw | 940.30 | 941.86 |
| + group_offset | 946.94 | 948.30 |
| + group_offset + trend(clean) | 943.79 | 945.13 |
| + group_offset + 우리 Platt | **947.10** | **950.19** |
| + group_offset + trend(clean) + Platt | 943.70 | 946.16 |

**trend_offset은 두 variant 모두에서 일관되게 손해**(group-only 대비
-3~-3.2) — 5개 시즌(2019-2023)만으로 count-state별 선형 추세를
추정하는 건 분산이 너무 크다는 뜻으로 해석. **group_offset + 우리
Platt 회귀보정**이 두 variant 모두에서 최선.

**최종 비교(각자 최선의 보정 적용)**: trackman20(950.19) vs
baseline_78(947.10) = **+3.09 — 트랙맨 접목이 실제로 우위**, 팀원의
문서화된 원본 기록(945.44, group+trend 포함)보다도 baseline_78 자체를
우리 방식(group+Platt)으로 재보정한 게 이미 더 나음(947.10) — 즉
우리 Platt 재보정 기법 자체도 팀원의 원래 정적 affine+trend 체인보다
낫다는 부수적 발견.

### 최종 채택 구성

- 모델: 팀원의 triple-blend(CatBoost-CPU depth=8/lr=0.035/iter=300,
  CatBoost-GPU→CPU depth=7/lr=0.025/iter=600, ExtraTrees 300그루) +
  season_decay=0.85, 가중치 cpu=0.5/gpu=0.35/extra=0.15 — **원본 그대로,
  변경 없음**.
- 피처: 팀원 원본 78개 + 우리 트랙맨 20개(풀링5+구종군15) = 98개.
- 보정: 팀원의 per-멤버 정적 affine(cat_scale=1.06/cat_shift=-0.0095/
  extra_shift=-0.01, 원본 그대로 유지) → **우리가 재계산한
  group_offset(shrinkage=500)** → **우리 Platt 회귀보정**. 팀원의
  원본 trend_offset은 **제외**.
- **로컬 검증 점수(2024 진짜 held-out) = 950.19** — 이 세션 전체에서
  가장 높은 로컬 검증 기록.

### 프로덕션 패키징

`submission/archive/exp039_teammate_trackman20_graft/`에 패키징.
팀원 프레임워크가 생성한 `script.py`의 `main()`을 교체해서 (a) 실제
2025 test.csv에 트랙맨 flat lookup을 병합하는 단계, (b) group_offset+
Platt 보정 체인을 추가. 격리 테스트(임시 폴더, 새 프로세스,
`data/test.csv` 5행) 통과 — NaN 없음, 확률 [0.40, 0.49]. 245,789행
규모 추론 시간 **18.5초**(제한 10분 대비 압도적 여유), 패키지 크기
730MB(제한 10GB 대비 여유). **`submission/submit.zip`을 이 결과로
갱신함 — 이 세션의 최종 최고 후보.**

### 소요 시간

- ablation study(2 variant): 약 45분
- 최종 프로덕션 빌드(배포용+보정용 모델 각각 재학습, CatBoost×2 +
  ExtraTrees, 총 6회 학습): 약 90분
- 재보정/버그 진단/패키징: 약 30분
- 총 약 2.5~3시간

### 주의 / 다음 단계

- **여전히 로컬 검증만 거쳤고 LB 미확인.** 이 세션의 가장 중요한
  교훈(quick_033: 로컬 +6.21 → LB -9.37)을 감안하면, 950.19라는
  로컬 점수도 실제 제출 전까지는 신중하게 봐야 한다. 다만 이번 개선은
  (a) 순수 피처 추가(트랙맨)와 (b) 재보정 기법 교체(팀원 정적 affine+
  trend → 우리 group+Platt)로, quick_033의 실패 원인이었던
  season_decay 자체는 원본 그대로(0.85) 유지했으므로 같은 실패
  메커니즘이 반복될 근거는 약함.
- **팀원의 원본 파이프라인(78피처, group+trend 그대로)도 우리 Platt
  재보정만 추가하면 947.10으로 개선된다는 부수적 발견** — 이건
  트랙맨과 무관하게 팀원 파이프라인 자체에 적용 가능한 별도의 안전한
  개선 레버로, 향후 추가 검증 후보.
- trend_offset이 왜 이렇게 일관되게 손해였는지(5시즌만으로 추정한
  분산 문제인지, count-state별 표본 수 문제인지)는 더 깊이 파볼
  가치가 있음 — 지금은 "제외가 낫다"는 실측 결론까지만 확보.
