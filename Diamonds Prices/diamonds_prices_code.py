# ==============================================================
# Diamond Price Prediction 다이아몬드 가격 예측 — 전체 분석 코드
# ==============================================================
#
# 원본 데이터: diamonds 데이터셋(Kaggle 재배포본, 53,940행 x 10열)
# 주 모델링 파이프라인은 강의 교재의 PBT(선형회귀 파이프라인) 노트북을 그대로 재현함
# (OLS 회귀 + 후진소거법, 도메인 지식 트랙과 기계적 트랙을 비교)
#
# 아래 코드는 원본 CSV(diamonds_price_raw.csv)를 입력으로 사용함

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

RANDOM_STATE = 3217  # 교재 커스텀 모듈(my_stats, my_ols 등)의 기본값과 통일

# ----------------------------------------------------------------
# 1. 프로젝트 개요 — 데이터 구조/품질 점검
# ----------------------------------------------------------------
df = pd.read_csv("diamonds_price_raw.csv")
print("원본 데이터 크기:", df.shape)
print("중복행:", df.duplicated().sum())
print("결측치:", df.isnull().sum().sum())

# 완전 중복행 삭제 (53,940 -> 53,794)
df = df.drop_duplicates().reset_index(drop=True)
print("중복 삭제 후:", df.shape)

# 기술통계량 (연속형: 평균·중앙값·왜도·첨도·이상치 / 범주형: count·unique·top·freq)
num_cols = ["price", "carat", "x", "y", "z", "depth", "table"]
cat_cols = ["cut", "color", "clarity"]

for c in cat_cols:
    vc = df[c].value_counts()
    print(f"\n{c}: count={df[c].count()}, unique={df[c].nunique()}, "
          f"top={vc.idxmax()}, freq={vc.max()}")

# 물리적 이상치(0값) 확인 — x/y/z 는 EDA 결과에 따라 최종 모델에서 아예 제외되므로
# (또는 IQR 이상치 대체 단계에서 처리되므로) 행 자체를 삭제하지는 않음
print("\nx=0:", (df["x"] == 0).sum(), "건")
print("y=0:", (df["y"] == 0).sum(), "건")
print("z=0:", (df["z"] == 0).sum(), "건")
print("y 최댓값:", df["y"].max(), " / z 최댓값:", df["z"].max())

# 캐럿 매직사이즈 이산점 확인 (참고용 — 이번 최종모델에는 채택되지 않음)
for t in [0.5, 0.7, 1.0, 1.5, 2.0]:
    shy = ((df["carat"] >= t - 0.05) & (df["carat"] < t)).sum()
    at = (df["carat"] == t).sum()
    print(f"임계값 {t}: 직전(shy) {shy}건 / 정확히 {at}건")

# ----------------------------------------------------------------
# 2. 탐색적 데이터 분석 — 이변량/다변량 (교재 EDA 결과 재확인)
# ----------------------------------------------------------------

# 연속형 변수-price Spearman 상관
cont_vars = ["carat", "x", "y", "z", "depth", "table"]
spearman_results = []
for c in cont_vars:
    rho, p = stats.spearmanr(df[c], df["price"])
    spearman_results.append({"변수": c, "rho": round(rho, 3), "p": round(p, 4)})
print("\n", pd.DataFrame(spearman_results).sort_values("rho", key=abs, ascending=False))

# 범주형(cut/color/clarity) - price Welch ANOVA
for c in cat_cols:
    groups = [g["price"].values for _, g in df.groupby(c)]
    lev_stat, lev_p = stats.levene(*groups)
    f_stat, f_p = stats.f_oneway(*groups)
    print(f"\n{c} - price ANOVA: F={f_stat:.2f}, p={f_p:.4g} "
          f"(Levene p={lev_p:.4g})")

# depth는 price와 무관함을 재확인 (교재 EDA 결과표: "효과 사실상 0")
rho_depth, p_depth = stats.spearmanr(df["depth"], df["price"])
print(f"\ndepth-price rho={rho_depth:.3f}, p={p_depth:.4g} -> 대표본 효과로 인한 형식적 유의성, 제외 대상")

# ----------------------------------------------------------------
# 3~4. 모델링용 데이터 전처리 + 모델링 — 교재 PBT 파이프라인 재현
#      (OLS + 후진소거법, 체크포인트별 비교)
# ----------------------------------------------------------------

# EDA 결과 반영: depth 제외 (모든 체크포인트에 공통 적용)
nominal_cols = ["cut", "color", "clarity"]
d_base = df[["price", "carat", "x", "y", "z", "table", "cut", "color", "clarity"]].copy()


def iqr_cap(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def backward_elim(y, X, alpha=0.05):
    """후진소거법: p-value가 가장 큰 변수를 유의수준 이하가 될 때까지 하나씩 제거"""
    X = X.copy()
    while True:
        model = sm.OLS(y, X).fit(cov_type="HC3")
        pvals = model.pvalues.drop("const", errors="ignore")
        if len(pvals) == 0 or pvals.max() <= alpha:
            return model
        X = X.drop(columns=[pvals.idxmax()])


def fit_pipeline(data, log=False, outlier=False, vif_remove=False, backward=True):
    """전처리 체크포인트를 순서대로 적용하고 OLS로 적합함"""
    d = data.copy()
    cont_cols = [c for c in ["carat", "x", "y", "z", "table"] if c in d.columns]

    if log:
        d["price"] = np.log1p(d["price"])
        d["carat"] = np.log(d["carat"])  # 탄력성 해석을 위해 순수 로그 사용
        for c in cont_cols:
            if c != "carat":
                d[c] = np.log1p(d[c])

    if outlier:
        for c in cont_cols + ["price"]:
            d[c] = iqr_cap(d[c])

    if vif_remove:
        work = cont_cols.copy()
        while len(work) > 1:
            vif = pd.Series(
                [variance_inflation_factor(d[work].values, i) for i in range(len(work))],
                index=work,
            )
            if vif.max() < 10.0:
                break
            d = d.drop(columns=[vif.idxmax()])
            work.remove(vif.idxmax())

    dd = pd.get_dummies(d, columns=nominal_cols, drop_first=True)
    y = dd["price"]
    X = sm.add_constant(dd.drop(columns=["price"]).astype(float))
    model = backward_elim(y, X) if backward else sm.OLS(y, X).fit(cov_type="HC3")

    pred = model.predict(X[model.params.index])
    if log:
        pred_o, y_o = np.expm1(pred), np.expm1(y)
    else:
        pred_o, y_o = pred, y
    rmse = np.sqrt(np.mean((y_o - pred_o) ** 2))
    return dict(model=model, rmse=rmse, r2=model.rsquared, nvar=len(model.params) - 1)


print("\n=== 기계적 트랙 (EDA 채택 변수 그대로 투입) ===")
checkpoints = {
    "0_기준선": dict(),
    "1_로그변환": dict(log=True),
    "2_이상치대체": dict(log=True, outlier=True),
    "3_공선성제거": dict(log=True, outlier=True, vif_remove=True),
}
fits = {}
for name, kw in checkpoints.items():
    fits[name] = fit_pipeline(d_base, **kw)
    r = fits[name]
    print(f"{name}: RMSE={r['rmse']:.3f}  R2(모델척도)={r['r2']:.4f}  변수 {r['nvar']}개")

print("\n=== 도메인 지식 트랙 (x, y, z를 carat으로 대체) ===")
d_domain = d_base.drop(columns=["x", "y", "z"])
domain_checkpoints = {
    "5_도메인_기준선": dict(),
    "6_도메인_로그변환": dict(log=True),
    "7_도메인_이상치대체": dict(log=True, outlier=True),
    "8_도메인_공선성제거": dict(log=True, outlier=True, vif_remove=True),
}
for name, kw in domain_checkpoints.items():
    fits[name] = fit_pipeline(d_domain, **kw)
    r = fits[name]
    print(f"{name}: RMSE={r['rmse']:.3f}  R2(모델척도)={r['r2']:.4f}  변수 {r['nvar']}개")

best_name = min(fits, key=lambda k: fits[k]["rmse"])
final_fit = fits[best_name]["model"]
print(f"\n최종 채택 모델: {best_name}  (RMSE={fits[best_name]['rmse']:.3f}, "
      f"독립변수 {fits[best_name]['nvar']}개)")

# ----------------------------------------------------------------
# 5. 분석결과 — 최종 모델 계수 해석 및 회귀 진단
# ----------------------------------------------------------------
print("\n=== 최종 모델 요약 ===")
print(final_fit.summary())

print("\ncarat 계수(탄력성):", final_fit.params["carat"])
print("-> 캐럿(중량)이 1% 증가하면 가격은 약", round(final_fit.params["carat"], 3), "% 상승함")

# ----------------------------------------------------------------
# 5-1-1. 홀드아웃 검증 — 처음 보는 데이터에 대한 예측 정확도
#        (5-1의 계수는 전체데이터 적합이라 계수 해석용, 예측 정확도는 별도 검증 필요)
# ----------------------------------------------------------------
from sklearn.model_selection import train_test_split as _tts

_train_idx, _test_idx = _tts(d_domain.index, test_size=0.2, random_state=RANDOM_STATE)
_train, _test = d_domain.loc[_train_idx].copy(), d_domain.loc[_test_idx].copy()

for c in ["price", "carat", "table"]:
    if c == "carat":
        _train[c] = np.log(_train[c]); _test[c] = np.log(_test[c])
    else:
        _train[c] = np.log1p(_train[c]); _test[c] = np.log1p(_test[c])

for c in ["price", "carat", "table"]:
    q1, q3 = _train[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    _train[c] = _train[c].clip(lo, hi)
    _test[c] = _test[c].clip(lo, hi)

_train_d = pd.get_dummies(_train, columns=nominal_cols, drop_first=True)
_test_d = pd.get_dummies(_test, columns=nominal_cols, drop_first=True)
_test_d = _test_d.reindex(columns=_train_d.columns, fill_value=0)

_y_tr = _train_d["price"]
_X_tr = sm.add_constant(_train_d.drop(columns=["price"]).astype(float))
_holdout_model = backward_elim(_y_tr, _X_tr)

_X_te = sm.add_constant(_test_d[_holdout_model.params.index.drop("const")].astype(float), has_constant="add")
_pred_log = _holdout_model.predict(_X_te)
_y_te_log = _test_d["price"]
_pred_dollar, _y_te_dollar = np.expm1(_pred_log), np.expm1(_y_te_log)

_holdout_rmse = np.sqrt(np.mean((_y_te_dollar - _pred_dollar) ** 2))
_holdout_mae = np.mean(np.abs(_y_te_dollar - _pred_dollar))
_holdout_r2_dollar = 1 - np.sum((_y_te_dollar - _pred_dollar) ** 2) / np.sum((_y_te_dollar - _y_te_dollar.mean()) ** 2)
_holdout_r2_log = 1 - np.sum((_y_te_log - _pred_log) ** 2) / np.sum((_y_te_log - _y_te_log.mean()) ** 2)

print(f"\n=== 홀드아웃 검증(n={len(_test)}, 처음 보는 데이터 기준) ===")
print(f"RMSE(달러)={_holdout_rmse:.2f}  MAE(달러)={_holdout_mae:.2f}  "
      f"R2(달러척도)={_holdout_r2_dollar:.4f}  R2(로그척도)={_holdout_r2_log:.4f}")
print("-> 전체데이터 적합 RMSE($796.68)와 비슷한 수준이면 과대적합 신호 없음")

resid = final_fit.resid
X_final = sm.add_constant(final_fit.model.exog, has_constant="skip")

reset = linear_reset(final_fit, power=2, use_f=True)
ks_stat, ks_p = stats.kstest(resid, "norm", args=(resid.mean(), resid.std()))
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(resid, final_fit.model.exog)
dw = durbin_watson(resid)

print("\n=== 회귀 가정 검정 ===")
print(f"선형성(RESET)   F={reset.fvalue:.2f}  p={reset.pvalue:.4g}")
print(f"정규성(K-S)     stat={ks_stat:.4f}  p={ks_p:.4g}")
print(f"등분산성(BP)    F={bp_f:.2f}  p={bp_fp:.4g}")
print(f"독립성(DW)      {dw:.3f}")

within1sd = (np.abs(resid) <= resid.std()).mean()
print(f"±1 표준편차 구간 비율: {within1sd*100:.1f}% (정규분포 기대값 68%)")

# ----------------------------------------------------------------
# [확장 트랙 — 교재 범위 밖, 참고용] CatBoost + SHAP 비선형 모형
# ----------------------------------------------------------------
# 아래는 교재 PBT 파이프라인에는 없는, 본 분석에서 추가로 시도한 트리 기반
# 비선형 모형 트랙임. 해석 가능한 선형모형(위 최종 채택 모델)을 대체하는 것이
# 아니라, 참고용 비교 대상으로만 남겨둠.

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import shap

cut_order = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_order = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_order = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
df["cut_encoded"] = df["cut"].map(cut_order)
df["color_encoded"] = df["color"].map(color_order)
df["clarity_encoded"] = df["clarity"].map(clarity_order)
df["log_price"] = np.log1p(df["price"])
df["log_carat"] = np.log(df["carat"])
thresholds = [0.5, 0.7, 1.0, 1.5, 2.0]
df["is_near_magic_size"] = df["carat"].apply(
    lambda c: int(any((t - 0.05 <= c < t) for t in thresholds))
)

FEATURE_COLS = ["log_carat", "cut_encoded", "color_encoded", "clarity_encoded",
                 "table", "is_near_magic_size"]
Xc = df[FEATURE_COLS]
yc = df["log_price"]
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=RANDOM_STATE)

cb_params = dict(iterations=600, depth=8, learning_rate=0.1, random_state=RANDOM_STATE, verbose=0)
cb_model = CatBoostRegressor(**cb_params)
cb_model.fit(Xtr, ytr)
pred_te = cb_model.predict(Xte)
print("\n=== [확장 트랙] CatBoost 결과(로그스케일) ===")
print(f"Test RMSE={np.sqrt(mean_squared_error(yte, pred_te)):.4f}  "
      f"MAE={mean_absolute_error(yte, pred_te):.4f}  R2={r2_score(yte, pred_te):.4f}")

explainer = shap.TreeExplainer(cb_model)
rng = np.random.RandomState(RANDOM_STATE)
idx = rng.choice(len(Xte), size=min(800, len(Xte)), replace=False)
X_sample = Xte.iloc[idx]
shap_values = explainer.shap_values(X_sample)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
print(pd.DataFrame({"변수": FEATURE_COLS, "mean|SHAP|": mean_abs_shap})
      .sort_values("mean|SHAP|", ascending=False))
