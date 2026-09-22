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
import statsmodels.api as sm


# ==============================================================
# 1. 데이터 로딩 및 품질 점검
# ==============================================================

origin = pd.read_csv("boston_housing_raw.csv")
origin.head()
origin.info()

df1 = origin.copy()
df1["CHAS"] = df1["CHAS"].astype("category")
df1.info()

dup = df1.duplicated()
dup.sum()

df2 = df1.drop_duplicates()
df2.duplicated().sum()

df2['CHAS'].value_counts()

fields = df2.select_dtypes(include="number").columns.to_list()
print(fields)

minmax = []
for field in fields:
    min_value = df2[field].min()
    max_value = df2[field].max()
    minmax.append({"min": min_value, "max": max_value})
minmax_df = DataFrame(minmax, index=fields)
minmax_df

na_count = df2.isna().sum()
na_count

rows, cols = df2.shape
print(f"rows: {rows}, cols: {cols}")

na_ratio = na_count / rows
na_ratio

df2.to_excel("boston_qtcheck.xlsx", index=False)
origin_qt = pd.read_excel("boston_qtcheck.xlsx")

df3 = origin_qt.copy()
df3['CHAS'] = df3['CHAS'].astype('category')
df3.info()

desc_df = df3.describe().T
desc_df

cate_desc_df = df3.describe(include='category').T
cate_desc_df

cate_fields = df3.select_dtypes(include='category').columns
for field in cate_fields:
    vcount = df3[field].value_counts()
    percent = vcount / df3.shape[0]
    cate_result = DataFrame({'count': vcount, 'percent': percent})
    print(cate_result)

desc_df['rel_diff'] = abs(desc_df['mean'] - desc_df['50%']) / desc_df['50%']
conditions = [desc_df['rel_diff'] < 0.1, desc_df['rel_diff'] < 0.5]
choices = ['similar', 'diff']
desc_df['rdiff_flag'] = np.select(conditions, choices, default='large_diff')

desc_df['iqr'] = desc_df['75%'] - desc_df['25%']
desc_df['upper_bound'] = desc_df['75%'] + 1.5 * desc_df['iqr']
desc_df['lower_bound'] = desc_df['25%'] - 1.5 * desc_df['iqr']

cate_fields = df3.select_dtypes(include='category').columns
df4 = df3.drop(columns=cate_fields)

desc_df['upper_outliers'] = (df4 > desc_df['upper_bound']).sum()
desc_df['upper_outliers_ratio'] = desc_df['upper_outliers'] / df4.shape[0]
desc_df['lower_outliers'] = (df4 < desc_df['lower_bound']).sum()
desc_df['lower_outliers_ratio'] = desc_df['lower_outliers'] / df4.shape[0]
desc_df['outliers'] = desc_df['upper_outliers'] + desc_df['lower_outliers']
desc_df['outliers_ratio'] = desc_df['outliers'] / df3.shape[0]

desc_df['skew'] = df4.skew()
desc_df['kurt'] = df4.kurt()


def judge_log_transform(skew, kurt):
    if skew >= 1:
        return "log"
    elif skew > 0.5 and kurt > 0:
        return "log"
    elif skew <= -1:
        return "reverse_log1p"
    elif skew < -0.5 and kurt > 0:
        return "reverse_log1p"
    else:
        return "none"


desc_df['log_need'] = desc_df.apply(lambda row: judge_log_transform(row['skew'], row['kurt']), axis=1)
# ZN은 최솟값이 0이라 순수 log 대신 log1p를 별도 지정함
desc_df.loc['ZN', 'log_need'] = 'log1p'
print(desc_df[['skew', 'kurt', 'log_need']])

desc_df.to_excel("boston_qtcheck_desc.xlsx")


# ==============================================================
# 2. 탐색적 데이터 분석 (EDA)
# ==============================================================

# --------------------------------------------------------------
# 2-0. 준비작업
# --------------------------------------------------------------

print(desc_df.loc[desc_df['log_need'] == 'log'].index.tolist(), "-> 로그변환 대상")
print(desc_df.loc[desc_df['log_need'] == 'log1p'].index.tolist(), "-> log1p 변환 대상")
print(desc_df.loc[desc_df['log_need'] == 'reverse_log1p'].index.tolist(), "-> 역로그변환 대상")

cate_desc_df

target = "MEDV"
target_is_continuous = True
nominal_cols = ["CHAS"]
continuous_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
print("종속변수:", target, "(연속형)")
print("명목형 독립변수:", nominal_cols)
print("연속형 독립변수:", continuous_cols)

print("분석범위: 상관분석(연속형 12종), 2집단 비교검정(CHAS) 적용 / ANOVA·교차분석은 해당 변수 없어 미적용")

# --------------------------------------------------------------
# 2-1. 단변량 분석
# --------------------------------------------------------------

medv_capped_count = (df3['MEDV'] == 50.0).sum()
print("MEDV=50 절단 건수:", medv_capped_count)

age_capped_count = (df3['AGE'] == 100.0).sum()
print("AGE=100 건수:", age_capped_count, "(별도 플래그 없이 그대로 사용)")

rad_max_count = (df3['RAD'] == 24).sum()
tax_666_count = (df3['TAX'] == 666).sum()
print("RAD=24 건수:", rad_max_count, "/ TAX=666 건수:", tax_666_count)
print("두 집합이 동일 town 그룹인지 확인:",
      set(df3[df3['RAD'] == 24].index) == set(df3[df3['TAX'] == 666].index))

print(df3['CHAS'].value_counts())
# CHAS는 유일한 명목형 변수이며 심한 불균형(93.1%/6.9%) -> 검정력 저하 가능성 있으나
# 버리지 않고 2-2에서 직접 검정으로 확인함

# --------------------------------------------------------------
# 2-2. 이변량 분석
# --------------------------------------------------------------

# 1. 상관분석 (연속형 독립변수 -> 종속변수, Spearman)
# MEDV 자체가 비정규(왜도 +1.108)라서 12개 변수 전체에 Spearman을 일괄 적용함
num_fields = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
corr_result = []
for field in num_fields:
    rho, pvalue = stats.spearmanr(df3[field], df3['MEDV'])
    corr_result.append({'field': field, 'rho': rho, 'p': pvalue})
corr_df = DataFrame(corr_result).sort_values('rho', key=abs, ascending=False)
print(corr_df)

# 2. 2집단 비교 검정 (CHAS -> 종속변수)
group0 = df3[df3['CHAS'] == 0]['MEDV']
group1 = df3[df3['CHAS'] == 1]['MEDV']

stat0, pvalue0 = stats.normaltest(group0)
stat1, pvalue1 = stats.normaltest(group1)
print(f"CHAS=0 정규성: stat={stat0:.4f}, p={pvalue0:.6f}")
print(f"CHAS=1 정규성: stat={stat1:.4f}, p={pvalue1:.6f}")

levene_stat, levene_p = stats.levene(group0, group1)
print(f"Levene 등분산: stat={levene_stat:.4f}, p={levene_p:.6f}")

u_stat, u_pvalue = stats.mannwhitneyu(group0, group1, alternative='two-sided')
u_less, p_less = stats.mannwhitneyu(group0, group1, alternative='less')
u_greater, p_greater = stats.mannwhitneyu(group0, group1, alternative='greater')
n0, n1 = len(group0), len(group1)
effect_r = 1 - (2 * u_stat) / (n0 * n1)
print(f"Mann-Whitney U(양측): stat={u_stat}, p={u_pvalue:.6f}, effect_r={effect_r:.3f}")
print(f"단측(0<1): p={p_less:.6f} / 단측(0>1): p={p_greater:.6f}")

# --------------------------------------------------------------
# 2-3. 다변량 분석
# --------------------------------------------------------------

vif_fields = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X_vif = df3[vif_fields].assign(const=1)
vif_result = []
for i, field in enumerate(vif_fields):
    vif_value = variance_inflation_factor(X_vif.values, i)
    vif_result.append({'field': field, 'vif': vif_value})
vif_df = DataFrame(vif_result).sort_values('vif', ascending=False)
print(vif_df)

# --------------------------------------------------------------
# 2-4. 최종 변수 선택 (EDA 단계의 예비 판단 — 3~4단계에서 재검정함)
# --------------------------------------------------------------

print("EDA 예비 판단: RAD는 TAX와 중복(세율축)되어 보류 후보로 분류함")
print("-> 실제 최종 채택 여부는 3단계 로그변환·이상치처리, 4단계 VIF·후진소거로 재검정함")


# ==============================================================
# 3. 모델링용 데이터 전처리
# ==============================================================

# --------------------------------------------------------------
# 3-1~3-2. 명목형 라벨링 및 더미변수 인코딩
# --------------------------------------------------------------

# CHAS는 이미 0/1 이진값이라 라벨링·더미인코딩 모두 실질적 변화 없음
df_ck0 = df3.copy()
df_ck0['CHAS'] = df_ck0['CHAS'].astype(int)
df_ck0.to_excel("boston_checkpoint_0.xlsx", index=False)

# --------------------------------------------------------------
# 3-3. 로그 변환
# --------------------------------------------------------------

log_cols = ['CRIM', 'DIS', 'RAD', 'LSTAT', 'MEDV']
log1p_cols = ['ZN']
reflect_cols = ['B']

df_ck1 = df_ck0.copy()
for c in log_cols:
    df_ck1[c] = np.log(df_ck1[c])
for c in log1p_cols:
    df_ck1[c] = np.log1p(df_ck1[c])
for c in reflect_cols:
    df_ck1[c] = np.log(df_ck1[c].max() + 1 - df_ck1[c])

print("로그변환 후 왜도:")
for c in log_cols + log1p_cols + reflect_cols:
    print(c, round(df_ck1[c].skew(), 3))

df_ck1.to_excel("boston_checkpoint_1.xlsx", index=False)

# --------------------------------------------------------------
# 3-4. 이상치 대체 (IQR 경계값 클리핑)
# --------------------------------------------------------------

df_ck2 = df_ck1.copy()
outlier_report = []
for c in continuous_cols:
    q1, q3 = df_ck1[c].quantile(0.25), df_ck1[c].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df_ck1[c] < lower) | (df_ck1[c] > upper)).sum()
    df_ck2[c] = df_ck1[c].clip(lower, upper)
    outlier_report.append({'field': c, 'lower': lower, 'upper': upper,
                            'n_outliers': n_out, 'ratio': n_out / len(df_ck1)})
outlier_df = DataFrame(outlier_report)
print(outlier_df)

df_ck2.to_excel("boston_checkpoint_2.xlsx", index=False)

# --------------------------------------------------------------
# 3-5. 다중공선성 제거 (VIF 반복 제거, 임계값 10.0)
# --------------------------------------------------------------


def reduce_vif(data, columns, threshold=10.0):
    cols = list(columns)
    while True:
        X = data[cols].assign(const=1)
        vifs = [variance_inflation_factor(X.values, i) for i in range(len(cols))]
        max_vif = max(vifs)
        if max_vif < threshold:
            break
        worst = cols[vifs.index(max_vif)]
        print(f"VIF 초과로 제거 -> {worst} (VIF={max_vif:.2f})")
        cols.remove(worst)
    return cols, max_vif


remain_cols, max_vif = reduce_vif(df_ck2, continuous_cols, threshold=10.0)
print("남은 변수:", remain_cols)
print("최대 VIF:", round(max_vif, 2))

df_ck3 = df_ck2.copy()  # 제거된 변수가 없어 체크포인트2와 동일
df_ck3.to_excel("boston_checkpoint_3.xlsx", index=False)

# --------------------------------------------------------------
# 3-6. 정규화 (StandardScaler)
# --------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

df_ck4 = df_ck3.copy()
scaler = StandardScaler()
df_ck4[continuous_cols] = scaler.fit_transform(df_ck3[continuous_cols])
df_ck4.to_excel("boston_checkpoint_4.xlsx", index=False)


# ==============================================================
# 4. 모델링
# ==============================================================

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

checkpoints = {"0_기준선": df_ck0, "1_로그변환": df_ck1, "2_이상치대체": df_ck2,
               "3_공선성제거": df_ck3, "4_스케일링": df_ck4}
log1p_y = {"0_기준선": False, "1_로그변환": True, "2_이상치대체": True,
           "3_공선성제거": True, "4_스케일링": True}
all_x = continuous_cols + ["CHAS"]


def backward_ols(data, target_col, xcols, use_hc3=False):
    """유의하지 않은 변수(p>0.05)를 하나씩 제거하는 후진소거 OLS"""
    y = data[target_col]
    cols = list(xcols)
    while True:
        X = sm.add_constant(data[cols])
        cov_type = 'HC3' if use_hc3 else 'nonrobust'
        model = sm.OLS(y, X).fit(cov_type=cov_type)
        pvals = model.pvalues.drop('const')
        maxp = pvals.max()
        if maxp <= 0.05:
            break
        worst = pvals.idxmax()
        print(f"제거 -> {worst} (p={maxp:.4f})")
        cols.remove(worst)
    return model, cols


fits = {}
result = []
for name, data in checkpoints.items():
    log = log1p_y[name]
    print(f"=== {name} ===")
    # 체크포인트0은 등분산성이 아직 확인되지 않았으므로 일반 표준오차,
    # 로그변환 이후 체크포인트는 HC3(강건표준오차)를 사용함
    fit, cols = backward_ols(data, "MEDV", all_x, use_hc3=log)
    fits[name] = (fit, cols)

    X = sm.add_constant(data[cols])
    pred = fit.predict(X)
    if log:
        y_true = np.exp(data["MEDV"]) if name != "0_기준선" else data["MEDV"]
        y_pred = np.exp(pred)
    else:
        y_true = data["MEDV"]
        y_pred = pred

    result.append({
        "모델": name, "변수수": len(cols),
        "R2(모델척도)": r2_score(data["MEDV"], pred),
        "R2(원본척도)": r2_score(origin["MEDV"], y_pred),
        "RMSE(원본척도)": root_mean_squared_error(origin["MEDV"], y_pred),
        "MAE(원본척도)": mean_absolute_error(origin["MEDV"], y_pred),
        "AIC": fit.aic, "BIC": fit.bic
    })

result_df = DataFrame(result).set_index("모델")
print(result_df.round(3))

# 최종 모델 채택: 1순위 원본척도 RMSE, 2순위 간명성 -> "1_로그변환"
final_fit, final_cols = fits["1_로그변환"]
final_data = checkpoints["1_로그변환"]
print("최종 채택 모델: 1_로그변환")
print("최종 변수:", final_cols)


# ==============================================================
# 5. 분석결과
# ==============================================================

# --------------------------------------------------------------
# 5-1. 모형 적합도
# --------------------------------------------------------------

print(final_fit.summary())

# --------------------------------------------------------------
# 5-2. 회귀계수 보고표 및 표준화계수(영향력 순위)
# --------------------------------------------------------------

coef_df = DataFrame({'B': final_fit.params, 't': final_fit.tvalues,
                      'p': final_fit.pvalues}).round(4)
print(coef_df)

from scipy.stats import zscore

Xz = final_data[final_cols].apply(zscore)
yz = zscore(final_data['MEDV'])
fit_std = sm.OLS(yz, Xz).fit()
beta = fit_std.params.sort_values(key=abs, ascending=False)
print("표준화계수(영향력 순위):")
print(beta.round(3))

# --------------------------------------------------------------
# 5-4. 회귀분석 가정 검정
# --------------------------------------------------------------

from statsmodels.stats.diagnostic import linear_reset, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

X_final = sm.add_constant(final_data[final_cols])

# 1. 선형성 - Ramsey RESET
reset = linear_reset(final_fit, power=2, use_f=True)
print("Ramsey RESET F:", round(reset.fvalue, 3), "p:", round(reset.pvalue, 4))

# 2. 정규성 - Kolmogorov-Smirnov
resid_std = final_fit.resid / final_fit.resid.std()
ks_stat, ks_p = stats.kstest(resid_std, 'norm')
print("KS stat:", round(ks_stat, 3), "p:", round(ks_p, 4))

# 3. 등분산성 - Breusch-Pagan
bp = het_breuschpagan(final_fit.resid, X_final)
print("Breusch-Pagan LM:", round(bp[0], 3), "p:", round(bp[1], 4))

# 4. 독립성 - Durbin-Watson
dw = durbin_watson(final_fit.resid)
print("Durbin-Watson:", round(dw, 3))


# ==============================================================
# 6-1 질문3 보완 — 홀드아웃(학습:검증=8:2) 검증 성능
# ==============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

idx_tr, idx_te = train_test_split(final_data.index, test_size=0.2, random_state=42)
Xtr_h = sm.add_constant(final_data.loc[idx_tr, final_cols])
Xte_h = sm.add_constant(final_data.loc[idx_te, final_cols])
ytr_h = final_data.loc[idx_tr, 'MEDV']
yte_h_log = final_data.loc[idx_te, 'MEDV']
yte_h_orig = origin.loc[idx_te, 'MEDV']

fit_holdout = sm.OLS(ytr_h, Xtr_h).fit(cov_type='HC3')
pred_te_log = fit_holdout.predict(Xte_h)
pred_te_orig = np.exp(pred_te_log)

print("[홀드아웃 검증] R2(원본척도):", round(r2_score(yte_h_orig, pred_te_orig), 3))
print("[홀드아웃 검증] RMSE(원본척도):", round(root_mean_squared_error(yte_h_orig, pred_te_orig), 3))
print("[홀드아웃 검증] MAE(원본척도):", round(mean_absolute_error(yte_h_orig, pred_te_orig), 3))


# ==============================================================
# 3-1-1 보완 — AGE=100 절단 플래그 유의성 검증
# ==============================================================

final_data['is_age_capped'] = (origin['AGE'] == 100).astype(int)
X_age = sm.add_constant(final_data[final_cols + ['is_age_capped']])
fit_age = sm.OLS(final_data['MEDV'], X_age).fit(cov_type='HC3')
print("is_age_capped 계수:", round(fit_age.params['is_age_capped'], 4),
      "p:", round(fit_age.pvalues['is_age_capped'], 4))


# ==============================================================
# 3-8 보완 — MEDV 절단 16건 제외 후 490개 재분석 (Pace & Gilley 1997 재현)
# ==============================================================

mask_490 = origin['MEDV'] < 50.0
data_490 = final_data.loc[mask_490]
orig_490 = origin.loc[mask_490]

fit_490, cols_490 = backward_ols(data_490, 'MEDV', final_cols, use_hc3=True)
print("490개 재분석 최종 변수:", cols_490)

pred_490_log = fit_490.predict(sm.add_constant(data_490[cols_490]))
pred_490_orig = np.exp(pred_490_log)
print("[490개] R2(원본척도):", round(r2_score(orig_490['MEDV'], pred_490_orig), 3))
print("[490개] RMSE(원본척도):", round(root_mean_squared_error(orig_490['MEDV'], pred_490_orig), 3))
print("[490개] MAE(원본척도):", round(mean_absolute_error(orig_490['MEDV'], pred_490_orig), 3))
