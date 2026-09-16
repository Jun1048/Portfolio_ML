# 📁 Portfolio_ML
 
데이터 수집부터 전처리, 분석, 시각화, 모델링까지 진행한 프로젝트를 정리한 저장소입니다.
 
---
 
## 📑 목차

| # | 프로젝트명 | 한 줄 설명 | 사용 기술 | 링크 |
|---|---|---|---|---|
| 1 | **Boston Housing(보스턴 주택 시장 예측)** | 1970년대 보스턴 506개 지역 데이터로 주택가격 결정요인을 회귀분석으로 규명하고, 후진소거·가정검정까지 거친 해석 중심 모형 구축 | `Python`, `Pandas`, `Statsmodels`, `Scikit-learn` | [🔗 바로가기](#1-boston-housing보스턴-주택-시장-예측) |
| 2 | **Diamonds Prices(다이아몬드 가격 예측)** | 4C(캐럿·컷·컬러·투명도) 등급 데이터로 가격 결정요인을 규명하고 예측모델(SHAP 해석 포함) 구축 | `Python`, `Pandas`, `Scikit-learn`, `CatBoost`, `SHAP` | [🔗 바로가기](#2-diamonds-prices다이아몬드-가격-예측) |
| 3 | **Apple Quality(사과 품질 예측)** | 사과 4,000개체 데이터로 품질(good/bad) 판정요인을 로지스틱 회귀로 규명, 로짓 비선형성·억제변수 진단까지 수행한 분류모델(SHAP 해석 포함) 구축 | `Python`, `Pandas`, `Scikit-learn`, `Statsmodels`, `CatBoost`, `SHAP` | [🔗 바로가기](#3-apple-quality사과-품질-예측) |
---
 
## 1. Boston Housing(보스턴 주택 시장 예측)

- Background

주제 : 1970년대 보스턴 지역 주택가격 결정요인 규명 및 해석 중심 회귀모형 구축

> 1970년 미국 연방정부의 자동차 배기가스 규제 시행을 계기로, 대기오염(NOX)에 대한 지불의사(WTP)를 주택가격 차이로 간접 추정하기 위해 만들어진 데이터 <br/>
> 원 설계 목적(환경정책 편익 추정)과 현재 분석 목적(가격예측)이 다르다는 점을 인지하고, 6단계(개요-EDA-전처리-모델링-결과-결론) 방법론으로 완성형 포트폴리오로 구성

- Summary

(1) Data Collection
- 수집대상 : 1970년 보스턴 SMSA 내 506개 census tract, 변수 14개(CRIM~MEDV)
- 수집 출처 : Harrison & Rubinfeld(1978) 원논문 → UCI ML Repository → Kaggle(altavish/boston-housing-dataset)

(2) Data Preprocessing
- 결측치·중복행 0건 확인
- 라벨링·더미변수 → 로그변환(CRIM·DIS·RAD·LSTAT·MEDV, ZN, B 총 7종) → 이상치 IQR 경계값 대체(행 삭제 없이 506개 유지) → 다중공선성 제거(VIF 반복검사, 임계값 10.0) → 정규화, 5단계 체크포인트로 비교
- TAX-RAD 다중공선성(VIF 8.88·7.40)이 우려되었으나, 로그변환+이상치대체만으로 최대 VIF 7.57까지 자연히 해소되어 변수를 강제로 제거하지 않음

(3) Model & Algorithms
- OLS 회귀 + 후진소거(backward elimination) 기반 해석 모형

> 모델링 과정<br/>
> 프로세스 : 체크포인트 5종(기준선→로그변환→이상치대체→공선성제거→스케일링) 각각에 OLS+후진소거 적용 → 원본 척도 RMSE·간명성 기준으로 최종 체크포인트 선정<br/>
> 최종 채택 : 체크포인트 "로그변환" 단계, 변수 10개(전체 13개 중 B·INDUS·AGE 제거)<br/>
> 성능 : RMSE $4,336, MAE $3,075, R² 0.777(원본 척도)<br/>
> 표준오차는 등분산성 위배에 대응해 HC3(이분산 강건표준오차)로 산출

- 회귀계수 분석

> 저소득층비율(LSTAT)의 표준화계수(|β|=0.584)가 압도적 1위<br/>
> 고속도로접근성(RAD)은 단순 상관관계(음)와 다변량 회귀계수(양)의 부호가 반전 — 다른 변수를 통제하면 순수효과 방향이 달라짐을 확인<br/>
> 찰스강인접(CHAS)은 회귀계수상 유의(p=0.004)하나 별도 시도한 ML 변수중요도 분석에서는 기여도가 낮게 나와, "통계적 유의성"과 "예측 기여도"가 다른 질문임을 확인

(4) Review
- 회귀분석 4대 가정(선형성·정규성·등분산성·독립성)이 전부 위배됨 — Ramsey RESET, Kolmogorov-Smirnov, Breusch-Pagan, Durbin-Watson 검정으로 확인
- 종속변수가 $50,000에서 상한 절단되어 있어, 고가 주택 구간 예측 신뢰도가 낮음
- 별도로 시도한 ML 기반 접근(CatBoost 등)은 예측 성능은 더 높았으나(R²=0.888) 형식적 가정검정·계수 해석을 제공하지 않아, 본 리포트는 해석 가능성을 우선한 OLS 결과를 공식 채택함
- 원 데이터의 수집 목적(환경정책 편익 추정)과 현재 분석 목적(가격예측)이 달라, 실무 적용 시 이 괴리를 반드시 고지해야 함

보러가기: [Boston Housing 포트폴리오](https://github.com/Jun1048/Portfolio_ML/tree/main/Boston%20Housing)
 
---

## 2. Diamonds Prices(다이아몬드 가격 예측)

- Background

주제 : 다이아몬드 4C(캐럿·컷·컬러·투명도) 등급 기반 가격 결정요인 규명 및 예측모델 구축
> 주제 선정 배경 : 앞서 완성한 Boston Housing 포트폴리오와 동일한 6단계 방법론(개요-EDA-전처리-모델링-결과-결론)을 새로운 도메인(다이아몬드 소매시장)에 적용해 두 번째 완성형 포트폴리오로 구성

- Summary

(1) Data Collection

- 수집대상 : 원형 브릴리언트 컷 다이아몬드 53,940건, 변수 10개(price~z)
- 수집 출처 : 온라인 다이아몬드 판매 플랫폼 원자료 → 공개 데이터셋 등재 → Kaggle 재배포본

(2) Data Preprocessing

- 결측치 0건, 완전 중복행 146건 삭제(53,940→53,794건), 물리적으로 불가능한 0값(x·y·z) 19건 추가 삭제(→53,775건)
- 왜도 기준 로그변환 2종(price·carat) — 왜도 각각 1.618→0.116, 1.116→0.581로 개선
- depth는 가격과의 상관이 사실상 0(ρ≈0.01)으로 확인되어 최종 변수에서 제외
- carat·x·y·z 다중공선성(VIF 최대 63) 확인 → 반복적 VIF 제거로 x·y·z 제거, carat 단독 채택(VIF 1점대로 해소)
- 캐럿 "매직사이즈"(0.5·0.7·1.0·1.5·2.0) 이산점 실측 확인 → 근접 플래그 파생변수 생성

(3) Model & Algorithms

- 회귀 모델링 및 SHAP 기반 해석

> 모델링 과정
> 프로세스 : 로그변환·순서형 인코딩·플래그 생성 → 학습/검증 분할(8:2) → 계열별 전처리(스케일링) → 11종 베이스라인 비교 → 하이퍼파라미터 튜닝 → 최종모형 선정
> 주요 투입 변수 : 캐럿(carat), 컷·컬러·투명도 등급, 테이블비율(table), 매직사이즈 근접 플래그 등 6개
> 모델링 : 1. 선형계열 - LinearRegression·Ridge·Lasso·ElasticNet
> 2. 비선형/거리기반 - KNN·SVR
> 3. 트리/앙상블 계열 - DecisionTree·RandomForest·XGBoost·LightGBM·CatBoost
> 4. 최종 선정 - RMSE 기준 근소격차 그룹핑 후 보조지표(MAE·R²)로 결함 점검하여 CatBoost 채택(Test RMSE 0.1009, R² 0.9900)

- SHAP 분석

> 캐럿(carat) 단 하나가 가격 변동의 약 79~84% 설명
> 4C 등급(컷·컬러·투명도)은 원시 데이터에서는 등급이 낮을수록 평균가가 오히려 높아 보이는 역설이 있었으나, 캐럿을 통제한 SHAP 순수효과에서는 등급이 높을수록 가격 기여도가 정상적으로 증가함을 확인(교란관계 해소)

(4) Review

- 변수 축소(6→3개) 시도는 성능을 10.01% 악화시켜 사전 규칙에 따라 기각 — 개별 중요도가 낮다고 항상 제거해도 되는 건 아님을 확인
- 삭제한 중복행 146건이 실제로는 서로 다른 실물 다이아몬드였을 가능성을 완전히 배제하지는 못함
- 데이터 수집 시점·화폐 기준연도가 명시돼 있지 않아 확인 불가로 남음

보러가기: [Diamonds Prices 포트폴리오](https://github.com/Jun1048/Portfolio_ML/tree/main/Diamonds%20Prices)

---

## 3. Apple Quality(사과 품질 예측)

- Background

주제 : 사과 품질(good/bad) 판정요인 규명 및 분류모델 구축

> 주제 선정 배경 : 앞서 완성한 Boston Housing·Diamonds Prices 포트폴리오와 동일한 6단계 방법론(개요-EDA-전처리-모델링-결과-결론)을 새로운 도메인(농산물 품질 분류)에 적용한 세 번째 완성형 포트폴리오로 구성 <br/>
> 앞선 두 프로젝트가 연속형 목표변수(가격)를 예측하는 회귀 문제였다면, 이번엔 이진 목표변수(품질 등급)를 분류하는 문제라 평가지표·검정방법을 회귀와 다르게 설계함

- Summary

(1) Data Collection
- 수집대상 : 사과 4,000개체, 변수 9개(식별자·크기·무게·당도·아삭함·과즙함량·숙성도·산도·품질)
- 수집 출처 : 원자료 제공처 → Kaggle 재배포본
- 원 물리 단위(cm, g 등)가 아닌 표준화된 값으로 제공되어, 계수 해석은 "표준화 척도 1단위당"으로 한정됨(원 데이터 자체의 한계)

(2) Data Preprocessing
- 결측치·중복행 0건 확인, 두 품질 등급(good/bad) 거의 정확히 균형(50.1%/49.9%)
- 종속변수 라벨링(good=1/bad=0), 학습:검증 8:2 stratify 분할
- 다중공선성 없음(VIF 전부 1.5 미만) 확인 → 변수 제거·차원축소 불필요
- 로지스틱 회귀 가정 검정(Box-Tidwell)에서 7개 변수 중 6개가 로짓 비선형성 위배 확인(위배 변수를 한 번에 처방하지 않고 하나씩 처방·재진단하는 반복 절차로 검증) → 제곱항 처방으로 정확도 74.4%→79.0% 개선

(3) Model & Algorithms
- 이진 분류 모델링 및 SHAP 기반 해석

> 모델링 과정<br/>
> 프로세스 : 라벨링·스케일링 → 학습/검증 stratify 분할(8:2) → 로지스틱 회귀 기준선 및 가정검정 → 처방 → 9종 베이스라인 비교 → 최종모형 선정<br/>
> 주요 투입 변수 : 크기(Size), 무게(Weight), 당도(Sweetness), 아삭함(Crunchiness), 과즙함량(Juiciness), 숙성도(Ripeness), 산도(Acidity) 7개<br/>
> 모델링 : 1. 선형계열 - LogisticRegression(기준선 및 처방 후)<br/>
> 2. 거리·확률 기반 - KNN·SVC·GaussianNB<br/>
> 3. 트리/앙상블 계열 - DecisionTree·RandomForest·XGBoost·LightGBM·CatBoost<br/>
> 4. 최종 선정 - 정확도 기준 근소격차 그룹핑 후 CatBoost 채택(Accuracy 0.8825, ROC-AUC 0.9565), 설명 가능성이 필요한 경우를 위해 처방된 로지스틱 회귀를 별도 트랙으로 병행 채택

- SHAP 분석

> 크기·당도·과즙함량 3개 변수가 판정 영향력의 약 56% 설명<br/>
> 무게·산도는 단변량 검정에서는 품질과 무관해 보였으나 다변량 모형에서 강하게 유의해지는 억제변수(suppressor variable) 현상 확인 — SHAP 방향과 로지스틱 오즈비 방향이 7개 변수 전부 일치해 결과를 교차검증함

(4) Review
- 원본 물리 단위·데이터 수집 시점 및 지역이 공개돼 있지 않아 확인 불가로 남음
- good/bad 라벨을 누가 어떤 기준으로 매겼는지 공개돼 있지 않아, 모형이 학습한 것이 객관적 품질인지 데이터 제작자의 판정 기준인지 확인 불가로 남음
- 제곱항 6개가 추가되며 모형이 복잡해져 오즈비를 한 줄로 해석하기 어려워졌고, 과적합 위험도 함께 커짐
- Train-CV 격차(8.2%)가 소폭 잔존하나 임계값 이내로, 새 데이터에서도 유사한 성능이 재현될 것으로 판단됨

보러가기: [Apple Quality 포트폴리오](https://github.com/Jun1048/Portfolio_ML/tree/main/Apple%20Quality)

---

## 🛠 Skills & Tools
 
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-1E90FF?style=flat-square)

---
 
## 📬 Contact
 
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:eett308@gmail.com)
 
대표 프로필 페이지: [Jun1048/Jun1048 →](https://github.com/Jun1048/Jun1048)
