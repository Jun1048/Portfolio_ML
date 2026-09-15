# ==============================================================
# Boston Housing 주택가격 예측 — 전체 분석 코드
# ==============================================================
#
# 원본 데이터: Boston Housing Dataset (Harrison & Rubinfeld, 1978 논문 기반,
# UCI Machine Learning Repository 및 Kaggle에 공개된 506행 x 14열 데이터)
# 아래 코드는 원본 CSV(boston_housing_raw.csv)를 입력으로 사용함

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==============================================================
# 1. 데이터 로딩 및 품질 점검
# ==============================================================

# 데이터 불러오기
origin = pd.read_csv("boston_housing_raw.csv")
origin.head()

# 자료형 확인
origin.info()

# 자료형 변환
df1 = origin.copy()
df1["CHAS"] = df1["CHAS"].astype("category")
df1.info()

# 중복 데이터 확인
dup = df1.duplicated()
dup.sum()

# 중복 데이터 제거
df2 = df1.drop_duplicates()
df2.duplicated().sum()

#명목형 변수 단위 확인
df2['CHAS'].value_counts()

# 연속형 변수 단위 확인을 위한 변수명 추출
fields = df2.select_dtypes(include="number").columns.to_list()
print(fields)

# 연속형 변수 단위 확인
minmax = []
for field in fields:
    min_value = df2[field].min()
    max_value = df2[field].max()
    minmax.append({"min": min_value, "max": max_value})
minmax_df = DataFrame(minmax, index=fields)
minmax_df

# 결측치 확인
na_count = df2.isna().sum()
na_count

# 결측치 비율 계산을 위한 데이터 크기 확인
rows, cols = df2.shape
print(f"rows: {rows}, cols: {cols}")

# 결측치 비율 확인
na_ratio = na_count / rows
na_ratio

# 엑셀 파일 저장
df2.to_excel("boston_qtcheck.xlsx", index=False)

# 저장 데이터 불러오기(새로운 jupyter 파일 생성 후 앞 단계에서 저장한 데이터 불러와 새로 진행)
origin_qt = pd.read_excel("boston_qtcheck.xlsx")

# 타입 변환
df3 = origin_qt.copy()
df3['CHAS'] = df3['CHAS'].astype('category')
df3.info()

# 연속형 변수의 기술 통계량 표
desc_df = df3.describe().T
desc_df

# 명목형 변수의 기술 통계량 표
cate_desc_df = df3.describe(include='category').T
cate_desc_df

# 명목형 변수의 기술 통계량
cate_fields = df3.select_dtypes(include='category').columns
for field in cate_fields:
    vcount = df3[field].value_counts()
    percent = vcount / df3.shape[0]
    cate_result = DataFrame({'count': vcount, 'percent': percent})
    print(cate_result)

# 평균-중앙값의 상대 차이율 계산
desc_df['rel_diff'] = abs(desc_df['mean'] - desc_df['50%']) / desc_df['50%']
conditions = [desc_df['rel_diff'] < 0.1, desc_df['rel_diff'] < 0.5]
choices = ['similar', 'diff']
desc_df['rdiff_flag'] = np.select(conditions, choices, default='large_diff')
desc_df[['mean', '50%', 'rel_diff', 'rdiff_flag']]

# IQR, 이상치 경계값 계산
desc_df['iqr'] = desc_df['75%'] - desc_df['25%']
desc_df['upper_bound'] = desc_df['75%'] + 1.5 * desc_df['iqr']
desc_df['lower_bound'] = desc_df['25%'] - 1.5 * desc_df['iqr']
desc_df[['iqr', 'upper_bound', 'lower_bound']]

# 명목형 변수를 제외한 데이터 프레임
cate_fields = df3.select_dtypes(include='category').columns
df4 = df3.drop(columns=cate_fields)
df4.head

# 상한 이상치 탐지
desc_df['upper_outliers'] = (df4 > desc_df['upper_bound']).sum()
desc_df['upper_outliers_ratio'] = desc_df['upper_outliers'] / df4.shape[0]
desc_df[['upper_outliers', 'upper_outliers_ratio']]

# 하한 이상치 탐지
desc_df['lower_outliers'] = (df4 < desc_df['lower_bound']).sum()
desc_df['lower_outliers_ratio'] = desc_df['lower_outliers'] / df4.shape[0]
desc_df[['lower_outliers', 'lower_outliers_ratio']]

# 전체 이상치 집계
desc_df['outliers'] = desc_df['upper_outliers'] + desc_df['lower_outliers']
desc_df['outliers_ratio'] = desc_df['outliers'] / df3.shape[0]
desc_df[['upper_outliers', 'upper_outliers_ratio',
         'lower_outliers', 'lower_outliers_ratio',
         'outliers', 'outliers_ratio']]

# 왜도 점검
desc_df['skew'] = df4.skew()
conditions_skew = [(desc_df['skew'] < -0.5), (desc_df['skew'] > 0.5)]
choices_skew = ['left tail', 'right tail']
desc_df['skew_interpret'] = np.select(conditions_skew, choices_skew, default='symmetric')
desc_df[['skew', 'skew_interpret']]

# 첨도 점검
desc_df['kurt'] = df4.kurt()
conditions_kurt = [(desc_df['kurt'] < 0), (desc_df['kurt'] > 0)]
choices_kurt = ['platykurtic', 'leptokurtic']
desc_df['kurt_interpret'] = np.select(conditions_kurt, choices_kurt, default='mesokurtic')
desc_df[['kurt', 'kurt_interpret']]

# 로그 변환 필요성 판단 함수 정의
def judge_log_transform(skew, kurt):
    if skew >= 1:
        return "log1p"
    elif skew > 0.5 and kurt > 0:
        return "log1p"
    elif skew <= -1:
        return "reverse_log1p"
    elif skew < -0.5 and kurt > 0:
        return "reverse_log1p"
    else:
        return "none"

# 로그 변환 필요성 판정
desc_df['log_need'] = desc_df.apply(lambda row: judge_log_transform(row['skew'], row['kurt']), axis=1)
desc_df[['skew', 'kurt', 'log_need']]

# 최종 기술 통계량 확인
desc_df.T

# 기술 통계량 표 저장
desc_df.to_excel("boston_qtcheck_desc.xlsx")


# ==============================================================
# 2. 탐색적 데이터 분석 (EDA)
# ==============================================================

# --------------------------------------------------------------
# 2-0. 준비작업
# --------------------------------------------------------------

# 1. 라이브러리 참조 -- 1단계에서 이미 참조한 라이브러리를 그대로 이어서 사용함
# 2. 품질 점검 완료 데이터 가져오기 -- 1단계 결과물(df3)을 그대로 이어받아 사용함

# 3. 연속형 변수의 기술 통계량 재확인 (1단계 desc_df의 log_need 컬럼 재확인)
print(desc_df.loc[desc_df['log_need'] == 'log1p'].index.tolist(), "-> 로그변환 고려")
print(desc_df.loc[desc_df['log_need'] == 'reverse_log1p'].index.tolist(), "-> 역로그변환 고려")
# 참고: RAD도 왜도 기준으로는 로그변환 후보이나, 아래 단변량 분석에서 확인하는
# 이산 구조(1~8, 24) 때문에 단순 로그변환은 부적절하다고 판단하고
# 2-4(최종 변수 선택)에서 RAD 자체를 제외하는 쪽으로 처리함

# 4. 명목형 변수의 기술 통계량 재확인 (1단계 cate_desc_df 재확인)
cate_desc_df

# 5. 타입 변환 -- 1단계에서 이미 CHAS를 category로 변환 완료(재변환 불필요)

# 6. 변수 유형 분류
target = "MEDV"
target_is_continuous = True
nominal_cols = ["CHAS"]
continuous_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
print("종속변수:", target, "(연속형)")
print("명목형 독립변수:", nominal_cols)
print("연속형 독립변수:", continuous_cols)

# 7. 컬럼 의미를 정리한 딕셔너리 -- 1단계 변수표(1-7)에 이미 정리되어 있어 재작성하지 않음

# 8. EDA 범위 — 변수유형 조합별 검정방법 결정
# 연속형(종속) x 연속형(독립) -> 상관분석 적용
# 연속형(종속) x 명목형 2집단(독립, CHAS) -> T검정 대상이나, 정규성 위배 확인 시 Mann-Whitney로 대체
# 연속형(종속) x 명목형 3집단 이상 -> 해당 변수 없어 미적용
print("분석범위: 상관분석(연속형 12종), 2집단 비교검정(CHAS) 적용 / ANOVA·교차분석은 해당 변수 없어 미적용")

# --------------------------------------------------------------
# 2-1. 단변량 분석
# --------------------------------------------------------------

# 1. 연속형 종속변수(MEDV) 절단 신호 확인
medv_capped_count = (df3['MEDV'] == 50.0).sum()
print("MEDV=50 절단 건수:", medv_capped_count)

# 2. 연속형 독립변수 — AGE 분포 확인(100 부근 집중되어 있으나 그대로 사용)
age_capped_count = (df3['AGE'] == 100.0).sum()
print("AGE=100 건수:", age_capped_count, "(별도 플래그 없이 그대로 사용)")

# 연속형 독립변수 — RAD, TAX 이산 구조 확인
rad_max_count = (df3['RAD'] == 24).sum()
tax_666_count = (df3['TAX'] == 666).sum()
print("RAD=24 건수:", rad_max_count, "/ TAX=666 건수:", tax_666_count)
print("두 집합이 동일 town 그룹인지 확인:",
      set(df3[df3['RAD'] == 24].index) == set(df3[df3['TAX'] == 666].index))

# 3. 범주형 독립변수(CHAS) 분포 확인 — 검정은 이변량 단계에서 수행
print(df3['CHAS'].value_counts())
# CHAS는 유일한 명목형 변수이며 심한 불균형(93.1%/6.9%) -> 검정력 저하 가능성 있으나
# 버리지 않고 2-2에서 직접 검정으로 확인함

# --------------------------------------------------------------
# 2-2. 이변량 분석
# --------------------------------------------------------------

# 1. 상관분석 (연속형 독립변수 -> 종속변수, Spearman)
num_fields = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
corr_result = []
for field in num_fields:
    rho, pvalue = stats.spearmanr(df3[field], df3['MEDV'])
    corr_result.append({'field': field, 'rho': rho, 'p': pvalue})
corr_df = DataFrame(corr_result).sort_values('rho', key=abs, ascending=False)
corr_df

# 2. 2집단 비교 검정 (CHAS -> 종속변수)
group0 = df3[df3['CHAS'] == 0]['MEDV']
group1 = df3[df3['CHAS'] == 1]['MEDV']

# 정규성 검정 (normaltest)
stat0, pvalue0 = stats.normaltest(group0)
stat1, pvalue1 = stats.normaltest(group1)
print(f"CHAS=0 정규성: stat={stat0:.4f}, p={pvalue0:.6f}")
print(f"CHAS=1 정규성: stat={stat1:.4f}, p={pvalue1:.6f}")

# 등분산 검정 (Levene)
levene_stat, levene_p = stats.levene(group0, group1)
print(f"Levene 등분산: stat={levene_stat:.4f}, p={levene_p:.6f}")

# 정규성 위배 -> Mann-Whitney U 검정(비모수)
u_stat, u_pvalue = stats.mannwhitneyu(group0, group1, alternative='two-sided')
n0, n1 = len(group0), len(group1)
effect_r = 1 - (2 * u_stat) / (n0 * n1)
print(f"Mann-Whitney U: stat={u_stat}, p={u_pvalue:.6f}, effect_r={effect_r:.3f}")

# --------------------------------------------------------------
# 2-3. 다변량 분석
# --------------------------------------------------------------

# 1. 독립변수간 상관 분석 + 다중공선성 확인 (VIF, 임계값 10.0)
vif_fields = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X_vif = df3[vif_fields].assign(const=1)
vif_result = []
for i, field in enumerate(vif_fields):
    vif_value = variance_inflation_factor(X_vif.values, i)
    vif_result.append({'field': field, 'vif': vif_value})
vif_df = DataFrame(vif_result).sort_values('vif', ascending=False)
vif_df

# --------------------------------------------------------------
# 2-4. 최종 변수 선택
# --------------------------------------------------------------

# 강한 쌍(|rho|>=0.7)으로 묶인 변수 중 종속변수와의 효과크기가 가장 큰 1개만 대표로 남김
# -> 세율축(TAX, RAD): TAX(rho=-0.562)가 RAD(rho=-0.347)보다 강하므로 RAD 제외
# -> 도심축(CRIM,DIS,NOX,AGE,INDUS): INDUS(rho=-0.578)가 최대이므로 대표 채택,
#    나머지는 모델링 이후 계수/중요도로 재검증할 "후보"로 남김
print("RAD는 세율축에서 TAX와 중복되고 효과크기가 더 약해 최종 변수에서 제외함")


# ==============================================================

# 3. 모델링용 데이터 전처리
# ==============================================================

df5 = df3.copy()
df5['CHAS'] = df5['CHAS'].astype(int)

# 로그 변환 (1단계 log_need 판정 결과 반영)
df5['CRIM_log'] = np.log(df5['CRIM'])
df5['ZN_log'] = np.log1p(df5['ZN'])
df5['DIS_log'] = np.log(df5['DIS'])
df5['LSTAT_log'] = np.log(df5['LSTAT'])
df5['B_revlog'] = np.log(df5['B'].max() + 1 - df5['B'])
df5['MEDV_log'] = np.log(df5['MEDV'])

# 참고: AGE는 절단 플래그 없이 그대로 사용하고, RAD는 최종 변수 선택 단계에서
# TAX와 중복되어 제외되었으므로 아래 모델링에 투입하지 않음

df5[['CRIM_log', 'ZN_log', 'DIS_log', 'LSTAT_log', 'B_revlog', 'MEDV_log']].head()

# --- 파생변수 검증: 원본 대비 로그변환 후 MEDV와의 Pearson 상관 개선 확인 ---
# (일관된 비교를 위해 종속변수는 원본 MEDV로 통일해서 비교함)
orig_vars = ['CRIM', 'ZN', 'DIS', 'LSTAT', 'B']
log_vars = ['CRIM_log', 'ZN_log', 'DIS_log', 'LSTAT_log', 'B_revlog']
for o, l in zip(orig_vars, log_vars):
    pear_o, _ = stats.pearsonr(df5[o], df5['MEDV'])
    pear_l, _ = stats.pearsonr(df5[l], df5['MEDV'])
    print(f"{o}: Pearson {pear_o:.3f} -> {l}: {pear_l:.3f}")

# 파생변수 간 상관(다중공선성 재확인)
df5[log_vars].corr()


# ==============================================================
# 4. 모델링
# ==============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 최종 투입 변수 11개 (RAD 제외, 절단 플래그 없음)
fields = ['CRIM_log', 'ZN_log', 'DIS_log', 'LSTAT_log', 'B_revlog',
          'NOX', 'RM', 'AGE', 'TAX', 'PTRATIO', 'CHAS']

y = df5['MEDV_log']
idx_train, idx_test = train_test_split(df5.index, test_size=0.2, random_state=42)

X_train = df5.loc[idx_train, fields]
X_test = df5.loc[idx_test, fields]
y_train = y.loc[idx_train]
y_test = y.loc[idx_test]

# 스케일링 (선형/거리기반 계열용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. LinearRegression (베이스라인) ---
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)
print("LinearRegression - RMSE:", rmse_lr, "MAE:", mae_lr, "R2:", r2_lr)

# --- 2. Ridge ---
model_ridge = Ridge(random_state=42)
model_ridge.fit(X_train_scaled, y_train)
pred_ridge = model_ridge.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
mae_ridge = mean_absolute_error(y_test, pred_ridge)
r2_ridge = r2_score(y_test, pred_ridge)
print("Ridge - RMSE:", rmse_ridge, "MAE:", mae_ridge, "R2:", r2_ridge)

# --- 3. Lasso (alpha 튜닝값 적용) ---
model_lasso = Lasso(alpha=0.001, random_state=42)
model_lasso.fit(X_train_scaled, y_train)
pred_lasso = model_lasso.predict(X_test_scaled)
rmse_lasso = np.sqrt(mean_squared_error(y_test, pred_lasso))
mae_lasso = mean_absolute_error(y_test, pred_lasso)
r2_lasso = r2_score(y_test, pred_lasso)
print("Lasso - RMSE:", rmse_lasso, "MAE:", mae_lasso, "R2:", r2_lasso)

# --- 4. ElasticNet ---
model_en = ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=42)
model_en.fit(X_train_scaled, y_train)
pred_en = model_en.predict(X_test_scaled)
rmse_en = np.sqrt(mean_squared_error(y_test, pred_en))
mae_en = mean_absolute_error(y_test, pred_en)
r2_en = r2_score(y_test, pred_en)
print("ElasticNet - RMSE:", rmse_en, "MAE:", mae_en, "R2:", r2_en)

# --- 5. KNN ---
model_knn = KNeighborsRegressor(n_neighbors=3)
model_knn.fit(X_train_scaled, y_train)
pred_knn = model_knn.predict(X_test_scaled)
rmse_knn = np.sqrt(mean_squared_error(y_test, pred_knn))
mae_knn = mean_absolute_error(y_test, pred_knn)
r2_knn = r2_score(y_test, pred_knn)
print("KNN - RMSE:", rmse_knn, "MAE:", mae_knn, "R2:", r2_knn)

# --- 6. SVR ---
model_svr = SVR(C=10, epsilon=0.1)
model_svr.fit(X_train_scaled, y_train)
pred_svr = model_svr.predict(X_test_scaled)
rmse_svr = np.sqrt(mean_squared_error(y_test, pred_svr))
mae_svr = mean_absolute_error(y_test, pred_svr)
r2_svr = r2_score(y_test, pred_svr)
print("SVR - RMSE:", rmse_svr, "MAE:", mae_svr, "R2:", r2_svr)

# --- 7. DecisionTree ---
model_dt = DecisionTreeRegressor(max_depth=4, random_state=42)
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, pred_dt))
mae_dt = mean_absolute_error(y_test, pred_dt)
r2_dt = r2_score(y_test, pred_dt)
print("DecisionTree - RMSE:", rmse_dt, "MAE:", mae_dt, "R2:", r2_dt)

# --- 8. RandomForest ---
model_rf = RandomForestRegressor(n_estimators=400, random_state=42)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
mae_rf = mean_absolute_error(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)
print("RandomForest - RMSE:", rmse_rf, "MAE:", mae_rf, "R2:", r2_rf)

# --- 9. XGBoost ---
model_xgb = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
mae_xgb = mean_absolute_error(y_test, pred_xgb)
r2_xgb = r2_score(y_test, pred_xgb)
print("XGBoost - RMSE:", rmse_xgb, "MAE:", mae_xgb, "R2:", r2_xgb)

# --- 10. LightGBM ---
model_lgb = LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
model_lgb.fit(X_train, y_train)
pred_lgb = model_lgb.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))
mae_lgb = mean_absolute_error(y_test, pred_lgb)
r2_lgb = r2_score(y_test, pred_lgb)
print("LightGBM - RMSE:", rmse_lgb, "MAE:", mae_lgb, "R2:", r2_lgb)

# --- 11. CatBoost (최종 선정 모형 — 예측·해석 모두에 사용) ---
model_cat = CatBoostRegressor(random_state=42, verbose=0)
model_cat.fit(X_train, y_train)
pred_cat = model_cat.predict(X_test)
rmse_cat = np.sqrt(mean_squared_error(y_test, pred_cat))
mae_cat = mean_absolute_error(y_test, pred_cat)
r2_cat = r2_score(y_test, pred_cat)
print("CatBoost - RMSE:", rmse_cat, "MAE:", mae_cat, "R2:", r2_cat)

# 11개 모형 성능 비교 요약
result_summary = DataFrame({
    'model': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'SVR',
              'DecisionTree', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'],
    'rmse': [rmse_lr, rmse_ridge, rmse_lasso, rmse_en, rmse_knn, rmse_svr,
             rmse_dt, rmse_rf, rmse_xgb, rmse_lgb, rmse_cat],
    'mae': [mae_lr, mae_ridge, mae_lasso, mae_en, mae_knn, mae_svr,
            mae_dt, mae_rf, mae_xgb, mae_lgb, mae_cat],
    'r2': [r2_lr, r2_ridge, r2_lasso, r2_en, r2_knn, r2_svr,
           r2_dt, r2_rf, r2_xgb, r2_lgb, r2_cat]
}).sort_values('rmse')

result_summary

# 신뢰성 검증 (Train vs CV vs Test)
from sklearn.model_selection import KFold, cross_val_score

pred_train_cat = model_cat.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, pred_train_cat))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(CatBoostRegressor(random_state=42, verbose=0),
                            X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

gap_pct = (cv_rmse.mean() - train_rmse) / max(abs(train_rmse), abs(cv_rmse.mean())) * 100
print(f"Train RMSE={train_rmse:.4f}, CV RMSE={cv_rmse.mean():.4f}, Test RMSE={rmse_cat:.4f}")
print(f"Gap%={gap_pct:.1f}%, CV-Test 차이={abs(cv_rmse.mean()-rmse_cat):.4f}")


# ==============================================================
# 5. 분석결과 — 변수중요도 및 SHAP (최종모형: CatBoost)
# ==============================================================

import shap

# 참고: CatBoost는 random_state를 고정해도 스레드 환경에 따라 변수중요도의
# 세부 수치가 미세하게 달라질 수 있음(알려진 특성). 순위 자체는 재현됨.
feature_importance = model_cat.get_feature_importance()
fi_df = DataFrame({'field': fields, 'importance': feature_importance})
fi_df = fi_df.sort_values('importance', ascending=False)
fi_df

explainer = shap.TreeExplainer(model_cat)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = DataFrame({'field': fields, 'mean_abs_shap': mean_abs_shap})
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
shap_df


# ==============================================================
# 5-2. Feature Importance 기반 변수 축소 -> 재학습 -> 재튜닝 -> 재검증
# ==============================================================

from sklearn.model_selection import GridSearchCV

# 1. Feature Importance 도출 (위 fi_df) + 누적비율 계산
fi_df['ratio'] = fi_df['importance'] / fi_df['importance'].sum() * 100
fi_df = fi_df.sort_values('importance', ascending=False)
fi_df['cum'] = fi_df['ratio'].cumsum()
fi_df

# 2. 채택된 변수 추출 (채택 기준: 누적 중요도 95%)
adopted_fields = []
for i, c in enumerate(fi_df['cum']):
    adopted_fields.append(fi_df['field'].iloc[i])
    if c >= 95.0:
        break
excluded_fields = [f for f in fields if f not in adopted_fields]
print("채택된 변수:", adopted_fields)
print("제외된 변수:", excluded_fields)

# 3. 데이터에서 채택된 변수만 추출
X_train_reduced = df5.loc[idx_train, adopted_fields]
X_test_reduced = df5.loc[idx_test, adopted_fields]

# 4. 베이스모델 재학습
model_cat_reduced = CatBoostRegressor(random_state=42, verbose=0)
model_cat_reduced.fit(X_train_reduced, y_train)
pred_reduced = model_cat_reduced.predict(X_test_reduced)
rmse_reduced = np.sqrt(mean_squared_error(y_test, pred_reduced))
mae_reduced = mean_absolute_error(y_test, pred_reduced)
r2_reduced = r2_score(y_test, pred_reduced)
print("축소모델(9개) - RMSE:", rmse_reduced, "MAE:", mae_reduced, "R2:", r2_reduced)

# 5. 하이퍼파라미터 재튜닝
param_grid = {'depth': [4, 6, 8], 'iterations': [300, 500], 'learning_rate': [0.03, 0.05, 0.1]}
grid = GridSearchCV(CatBoostRegressor(random_state=42, verbose=0), param_grid,
                     cv=5, scoring='neg_root_mean_squared_error')
grid.fit(X_train_reduced, y_train)
print("재튜닝 최적 파라미터:", grid.best_params_)

model_cat_tuned = CatBoostRegressor(**grid.best_params_, random_state=42, verbose=0)
model_cat_tuned.fit(X_train_reduced, y_train)
pred_tuned = model_cat_tuned.predict(X_test_reduced)
rmse_tuned = np.sqrt(mean_squared_error(y_test, pred_tuned))
print("재튜닝 후 RMSE:", rmse_tuned,
      "(기본 하이퍼파라미터보다 나빠지면 재튜닝 결과 대신 기본값을 채택함)")

# 6. 과적합 재판정 (기본 하이퍼파라미터 축소모델 기준으로 최종 채택)
pred_train_reduced = model_cat_reduced.predict(X_train_reduced)
train_rmse_reduced = np.sqrt(mean_squared_error(y_train, pred_train_reduced))
cv_rmse_reduced = -cross_val_score(CatBoostRegressor(random_state=42, verbose=0),
                                    X_train_reduced, y_train, cv=kf,
                                    scoring='neg_root_mean_squared_error')
gap_pct_reduced = (cv_rmse_reduced.mean() - train_rmse_reduced) / \
    max(abs(train_rmse_reduced), abs(cv_rmse_reduced.mean())) * 100
print(f"[축소모델] Train={train_rmse_reduced:.4f}, CV={cv_rmse_reduced.mean():.4f}, "
      f"Test={rmse_reduced:.4f}, Gap%={gap_pct_reduced:.1f}%")

# 종합 판단: 성능 저하가 1% 미만이고 변수가 2개 줄었으므로, 최종 모형은 9개 변수로 확정함

# --- 최종(9개 변수) 모형의 변수중요도·SHAP 재계산 ---
fi_final = model_cat_reduced.get_feature_importance()
fi_final_df = DataFrame({'field': adopted_fields, 'importance': fi_final})
fi_final_df = fi_final_df.sort_values('importance', ascending=False)
fi_final_df

explainer_final = shap.TreeExplainer(model_cat_reduced)
shap_values_final = explainer_final.shap_values(X_test_reduced)
mean_abs_shap_final = np.abs(shap_values_final).mean(axis=0)
shap_final_df = DataFrame({'field': adopted_fields, 'mean_abs_shap': mean_abs_shap_final})
shap_final_df = shap_final_df.sort_values('mean_abs_shap', ascending=False)
shap_final_df

# ==============================================================
# 상세 해석·의문점 규명 과정(EDA 자료와의 대조, 변수 선택 규칙 적용 근거,
# 과적합 진단 등)은 통합 리포트(md) 참고
# ==============================================================
