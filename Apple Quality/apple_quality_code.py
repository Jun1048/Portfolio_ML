# ======================================================================
# Apple Quality 사과 품질 예측
# ======================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

RANDOM_STATE = 3217

# ======================================================================
# 1. 데이터 로딩 및 구조 점검
# ======================================================================

df = pd.read_csv("apple_quality.csv")

num_cols = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]

print("데이터 크기:", df.shape)
print(df.dtypes)
print("A_id 고유값 수:", df["A_id"].nunique(), "/", len(df))
print("중복행:", df.duplicated().sum())
print("결측치 합계:", df.isnull().sum().sum())
print(df["Quality"].value_counts())

# ======================================================================
# 2. 기술 통계량 (연속형 + 범주형)
# ======================================================================

desc = df[num_cols].describe().T
desc["skew"] = df[num_cols].skew()
desc["kurtosis"] = df[num_cols].kurt()
print(desc)

for c in num_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = ((df[c] < lower) | (df[c] > upper)).sum()
    print(f"{c}: IQR={iqr:.3f}, 이상치 {out}건 ({out/len(df)*100:.2f}%)")

# ======================================================================
# 3. 탐색적 데이터 분석 - 정규성, 그룹 차이, 다중공선성
# ======================================================================

df["Quality_label"] = (df["Quality"] == "good").astype(int)  # good=1, bad=0 (알파벳순)

for c in num_cols:
    stat, p = stats.normaltest(df[c])
    print(f"{c}: D'Agostino p={p:.4g}")

good = df[df["Quality"] == "good"]
bad = df[df["Quality"] == "bad"]

# 두 집단 모두 정규성을 만족하면 Bartlett 등분산 검정 후 Student/Welch t-test,
# 하나라도 정규성을 만족하지 않으면 Mann-Whitney U 검정으로 분기
for c in num_cols:
    _, p_good = stats.normaltest(good[c])
    _, p_bad = stats.normaltest(bad[c])
    if p_good >= 0.05 and p_bad >= 0.05:
        _, bart_p = stats.bartlett(good[c], bad[c])
        equal_var = bart_p >= 0.05
        stat, p = stats.ttest_ind(good[c], bad[c], equal_var=equal_var)
        method = "Student t-test" if equal_var else "Welch t-test"
    else:
        stat, p = stats.mannwhitneyu(good[c], bad[c], alternative="two-sided")
        method = "Mann-Whitney U"
    print(f"{c}: {method}, statistic={stat:.3f}, p={p:.4g}")

corr = df[num_cols].corr(method="pearson")
print(corr.round(3))

# ======================================================================
# 4. 모델링용 데이터 전처리 - 분할, 스케일링, VIF
# ======================================================================

X = df[num_cols]
y = df["Quality_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 거리·커널 기반 비교 모형(KNN/SVC/GaussianNB)에만 쓸 표준화 버전 (7번에서 사용)
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=num_cols, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=num_cols, index=X_test.index)

# 로지스틱 회귀는 오즈비 해석 일관성을 위해 원본 제공 척도를 그대로 사용
X_train_const = sm.add_constant(X_train)
vif_data = pd.DataFrame()
vif_data["변수"] = X_train_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_const.values, i) for i in range(X_train_const.shape[1])]
print(vif_data)

# ======================================================================
# 5. 로지스틱 회귀 기준선 모형 및 가정 검정 (Box-Tidwell, 반복 진단-처방)
# ======================================================================

logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit(disp=0)
print(result.summary())
print("오즈비:\n", np.exp(result.params))


def boxtidwell_diagnose(data_lin, source_for_ln, target_vars, y):
    """target_vars 각각에 대해 x*ln(x) 보조항을 넣어 로짓 선형성을 검정."""
    rows = []
    for col in target_vars:
        bt = data_lin.copy()
        base = source_for_ln[col]
        if base.min() <= 0:
            base = base - base.min() + 1
        bt[f"{col}_aux"] = base * np.log(base)
        bt_fit = sm.Logit(y, sm.add_constant(bt)).fit(disp=0)
        rows.append((col, bt_fit.tvalues[f"{col}_aux"], bt_fit.pvalues[f"{col}_aux"]))
    return rows


# 위배 변수를 가장 심한 것부터 하나씩 처방하고, 매 라운드 나머지 변수를 재진단
data_lin = X_train.copy()
active_vars = num_cols.copy()
recipe = []

fit_prev = sm.Logit(y_train, sm.add_constant(data_lin)).fit(disp=0)
aic_prev = fit_prev.aic

round_num = 0
while True:
    round_num += 1
    diag = boxtidwell_diagnose(data_lin, X_train, active_vars, y_train)
    violating = sorted([d for d in diag if d[2] < 0.05], key=lambda d: -abs(d[1]))
    if not violating:
        print(f"[라운드 {round_num}] 위배 0종 -> 종료")
        break

    worst_col, worst_z, worst_p = violating[0]
    data_lin[f"{worst_col}_sq"] = data_lin[worst_col] ** 2
    fit_new = sm.Logit(y_train, sm.add_constant(data_lin)).fit(disp=0)
    lr_p = 1 - stats.chi2.cdf(2 * (fit_new.llf - fit_prev.llf), 1)
    print(f"[라운드 {round_num}] 위배 {len(violating)}종 -> '{worst_col}' 처방 "
          f"(z={worst_z:.3f}, 우도비 p={lr_p:.4g}, AIC {aic_prev:.2f}->{fit_new.aic:.2f})")

    recipe.append(worst_col)
    active_vars.remove(worst_col)
    fit_prev, aic_prev = fit_new, fit_new.aic
    if not active_vars:
        break

print("\n제곱항 처방 변수:", recipe)
print("처방 제외(선형성 충족):", [c for c in num_cols if c not in recipe])
print("최종 Pseudo R²:", fit_prev.prsquared)

# 독립성 검정 (Durbin-Watson)
from statsmodels.stats.stattools import durbin_watson
print("Durbin-Watson:", durbin_watson(fit_prev.resid_pearson))

# ======================================================================
# 6. 처방 - 계층 원칙에 따른 제곱항 추가 및 성능 비교
# ======================================================================

X_train_fix = data_lin.copy()  # 반복 처방이 모두 반영된 학습 데이터
X_test_fix = X_test.copy()
for v in recipe:
    X_test_fix[f"{v}_sq"] = X_test[v] ** 2

model_base = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model_base.fit(X_train, y_train)
pred_base = model_base.predict(X_test)
proba_base = model_base.predict_proba(X_test)[:, 1]
print("기준선 Accuracy:", accuracy_score(y_test, pred_base))

model_fix = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model_fix.fit(X_train_fix, y_train)
pred_fix = model_fix.predict(X_test_fix[X_train_fix.columns])
proba_fix = model_fix.predict_proba(X_test_fix[X_train_fix.columns])[:, 1]
print("처방 후 Accuracy:", accuracy_score(y_test, pred_fix))
print("처방 후 F1:", f1_score(y_test, pred_fix))
print("처방 후 ROC-AUC:", roc_auc_score(y_test, proba_fix))

param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), param_grid, cv=5, scoring="accuracy")
gs.fit(X_train_fix, y_train)
print("최적 C:", gs.best_params_)

# ======================================================================
# 7. 베이스라인 다중 모델 비교
# ======================================================================

models = {
    "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=RANDOM_STATE),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    "CatBoost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
}

results = []
for name, model in models.items():
    if name in ["LogisticRegression", "KNN", "SVC", "GaussianNB"]:
        Xtr, Xte = X_train_s, X_test_s
    else:
        Xtr, Xte = X_train, X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    results.append([
        name,
        accuracy_score(y_test, pred),
        f1_score(y_test, pred),
        roc_auc_score(y_test, proba),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
    ])

res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "ROC_AUC", "Precision", "Recall"])
res_df = res_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
print(res_df)

# ======================================================================
# 8. 최종 모형(CatBoost) 신뢰성 검증 - Train / CV / Test
# ======================================================================

final_model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
final_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, final_model.predict(X_train))
test_acc = accuracy_score(y_test, final_model.predict(X_test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    CatBoostClassifier(random_state=RANDOM_STATE, verbose=0), X_train, y_train, cv=skf, scoring="accuracy"
)
cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

gap = train_acc - cv_mean
gap_pct = gap / max(abs(train_acc), abs(cv_mean)) * 100

print(f"Train Accuracy: {train_acc:.4f}")
print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Gap%: {gap_pct:.2f}%")

train_sizes, train_scores, val_scores = learning_curve(
    CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
    X_train, y_train, cv=skf, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
)
for ts, tr, va in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"{ts:5d}  train={tr:.4f}  val={va:.4f}  gap={tr-va:.4f}")

# ======================================================================
# 9. 변수 중요도 및 SHAP 분석
# ======================================================================

fi = final_model.get_feature_importance(prettified=True)
fi["ratio"] = fi["Importances"] / fi["Importances"].sum() * 100
print(fi[["Feature Id", "ratio"]])

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_summary = pd.DataFrame({"변수": num_cols, "mean|SHAP|": mean_abs_shap}).sort_values(
    "mean|SHAP|", ascending=False
)
shap_summary["ratio"] = shap_summary["mean|SHAP|"] / shap_summary["mean|SHAP|"].sum() * 100
print(shap_summary)

X_test_arr = X_test.reset_index(drop=True)
for i, c in enumerate(num_cols):
    corr_dir = np.corrcoef(X_test_arr[c], shap_values[:, i])[0, 1]
    print(f"{c}: SHAP-값 상관 = {corr_dir:.3f}")

weight_idx = num_cols.index("Weight")
weight_shap = shap_values[:, weight_idx]
for c in num_cols:
    corr_int = np.corrcoef(X_test_arr[c], weight_shap)[0, 1]
    print(f"Weight SHAP vs {c}: {corr_int:.3f}")

print("분석 파이프라인 실행 완료")
