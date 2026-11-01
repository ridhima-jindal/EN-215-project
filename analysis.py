# EN 215 mini-project: Lucknow NASA POWER hourly data, 2023-2025
# Everything for Stage 1 and Stage 2 is computed here and saved to results.json + figs/.
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

F = "POWER_Point_Hourly_20230101_20251231_026d85N_080d95E_LST.csv"   # put the CSV in the same folder as this script
OUT = "figs/"
os.makedirs("figs", exist_ok=True)
R = {}                                  # results collected for the write-up
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

# ---------- load and Q2 quality checks ----------
raw = pd.read_csv(F, skiprows=14)
raw["dt"] = pd.to_datetime(dict(year=raw.YEAR, month=raw.MO, day=raw.DY, hour=raw.HR))
vars_ = ["ALLSKY_SFC_SW_DWN","CLRSKY_SFC_SW_DWN","T2M","RH2M","WS10M","PRECTOTCORR"]
R["rows"] = len(raw)
R["n_minus999"] = int((raw[vars_] == -999).sum().sum())
R["n_nan"] = int(raw[vars_].isna().sum().sum())
R["n_dup"] = int(raw.duplicated("dt").sum())
R["expected_rows"] = len(pd.date_range(raw.dt.min(), raw.dt.max(), freq="h"))
R["night_zero_frac"] = float((raw.ALLSKY_SFC_SW_DWN == 0).mean())
R["clr_zero_frac"] = float((raw.CLRSKY_SFC_SW_DWN == 0).mean())
R["allsky_gt_clr"] = int((raw.ALLSKY_SFC_SW_DWN > raw.CLRSKY_SFC_SW_DWN + 1).sum())
R["allsky_zero_while_clr_pos"] = int(((raw.ALLSKY_SFC_SW_DWN == 0) & (raw.CLRSKY_SFC_SW_DWN > 0)).sum())
R["rh_gt100"] = int((raw.RH2M > 100).sum())
R["rh_min_max"] = [float(raw.RH2M.min()), float(raw.RH2M.max())]
R["precip_zero_frac"] = float((raw.PRECTOTCORR == 0).mean())
R["precip_max"] = float(raw.PRECTOTCORR.max())
R["T_min_max"] = [float(raw.T2M.min()), float(raw.T2M.max())]
R["naive_mean_all"] = float(raw.ALLSKY_SFC_SW_DWN.mean())
day = raw[raw.CLRSKY_SFC_SW_DWN > 0].copy()
R["day_rows"] = len(day); R["day_mean_irr"] = float(day.ALLSKY_SFC_SW_DWN.mean())
# what a naive -999 treatment would have done (illustration only: file has none)
# clearness index; keep hours with clear-sky >= 50 Wh/m2 so the ratio is not unstable at dawn/dusk
d = raw[raw.CLRSKY_SFC_SW_DWN >= 50].copy()
d["kt"] = d.ALLSKY_SFC_SW_DWN / d.CLRSKY_SFC_SW_DWN
R["kt_rows"] = len(d); R["kt_gt1"] = int((d.kt > 1).sum()); R["kt_max"] = float(d.kt.max())
d["kt"] = d.kt.clip(upper=0.999)       # cap physically impossible >1 values (beta needs <1)
d["kt"] = d.kt.clip(lower=0.001)       # beta needs strictly positive support
d["month"] = d.dt.dt.month

# ---------- Q3: first variable = hourly clearness index kt ----------
kt = d.kt.values
R["kt_mean"] = float(kt.mean()); R["kt_var"] = float(kt.var(ddof=1)); R["kt_sd"] = float(kt.std(ddof=1))
cands = {}
a, b, loc, sc = stats.beta.fit(kt, floc=0, fscale=1)
cands["Beta"] = (stats.beta, (a, b, 0, 1), 2)
p = stats.truncnorm.fit(kt, 0, 1) if False else None
mu, sd = kt.mean(), kt.std()
cands["Normal"] = (stats.norm, (mu, sd), 2)
g = stats.gamma.fit(kt, floc=0); cands["Gamma"] = (stats.gamma, g, 2)
w = stats.weibull_min.fit(kt, floc=0); cands["Weibull"] = (stats.weibull_min, w, 2)
R["kt_fits"] = {}
for n, (dist, par, k) in cands.items():
    ll = dist.logpdf(kt, *par).sum()
    R["kt_fits"][n] = dict(AIC=float(2*k-2*ll), KS=float(stats.kstest(kt, dist.cdf, args=par).statistic),
                          params=[float(x) for x in par])
R["kt_beta_theory_mean"] = float(a/(a+b)); R["kt_beta_theory_var"] = float(a*b/((a+b)**2*(a+b+1)))
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
xs = np.linspace(0.001, 1, 400)
ax[0].hist(kt, bins=50, density=True, color="#bbb", edgecolor="w")
for n, c in zip(["Beta","Gamma","Normal"], ["C3","C0","C2"]):
    dist, par, _ = cands[n]; ax[0].plot(xs, dist.pdf(xs, *par), c, label=n)
ax[0].set_xlabel("hourly clearness index kt (ratio)"); ax[0].set_ylabel("density"); ax[0].legend(); ax[0].set_title("Histogram with fitted densities")
srt = np.sort(kt); ecdf = np.arange(1, len(srt)+1)/len(srt)
ax[1].plot(srt, ecdf, "k", lw=2, label="empirical")
for n, c in zip(["Beta","Gamma","Normal"], ["C3","C0","C2"]):
    dist, par, _ = cands[n]; ax[1].plot(xs, dist.cdf(xs, *par), c, ls="--", label=n)
ax[1].set_xlabel("kt"); ax[1].set_ylabel("CDF"); ax[1].legend(); ax[1].set_title("Empirical vs fitted CDF")
plt.tight_layout(); plt.savefig(OUT+"q3_kt_fit.png"); plt.close()

# ---------- Q4: second variable = daily temperature range (DTR) ----------
raw["date"] = raw.dt.dt.floor("D")
daily = raw.groupby("date").agg(T_max=("T2M","max"), T_min=("T2M","min"), T_mean=("T2M","mean"),
        RH=("RH2M","mean"), RH_min=("RH2M","min"), WS=("WS10M","mean"), rain=("PRECTOTCORR","sum"),
        H_all=("ALLSKY_SFC_SW_DWN","sum"), H_clr=("CLRSKY_SFC_SW_DWN","sum"), n=("T2M","size"))
R["days"] = len(daily); R["days_not24"] = int((daily.n != 24).sum())
daily["DTR"] = daily.T_max - daily.T_min
daily["H_kwh"] = daily.H_all/1000.0                       # kWh/m2/day
daily["KT"] = daily.H_all/daily.H_clr                     # daily clearness index
daily["month"] = daily.index.month; daily["year"] = daily.index.year
dtr = daily.DTR.values
R["dtr_mean"] = float(dtr.mean()); R["dtr_var"] = float(dtr.var(ddof=1)); R["dtr_sd"] = float(dtr.std(ddof=1))
R["dtr_min_max"] = [float(dtr.min()), float(dtr.max())]
c2 = {"Gamma": (stats.gamma, stats.gamma.fit(dtr, floc=0), 2),
      "Weibull": (stats.weibull_min, stats.weibull_min.fit(dtr, floc=0), 2),
      "Normal": (stats.norm, (dtr.mean(), dtr.std()), 2),
      "Lognormal": (stats.lognorm, stats.lognorm.fit(dtr, floc=0), 2)}
R["dtr_fits"] = {}
for n, (dist, par, k) in c2.items():
    ll = dist.logpdf(dtr, *par).sum()
    R["dtr_fits"][n] = dict(AIC=float(2*k-2*ll), KS=float(stats.kstest(dtr, dist.cdf, args=par).statistic), params=[float(x) for x in par])
R["dtr_skew"] = float(stats.skew(dtr)); R["kt_skew"] = float(stats.skew(kt))
best_dtr = min(R["dtr_fits"], key=lambda n: R["dtr_fits"][n]["AIC"]); R["dtr_best"] = best_dtr
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
xs2 = np.linspace(dtr.min(), dtr.max(), 300)
ax[0].hist(dtr, bins=30, density=True, color="#bbb", edgecolor="w")
for n, c in zip(["Gamma","Normal","Weibull"], ["C0","C2","C1"]):
    dist, par, _ = c2[n]; ax[0].plot(xs2, dist.pdf(xs2, *par), c, label=n)
ax[0].set_xlabel("daily temperature range (deg C)"); ax[0].set_ylabel("density"); ax[0].legend(); ax[0].set_title("Histogram with fitted densities")
for n, c in zip(["Gamma","Normal","Weibull"], ["C0","C2","C1"]):
    dist, par, _ = c2[n]
    q = np.linspace(0.01, 0.99, 99); ax[1].plot(dist.ppf(q, *par), np.quantile(dtr, q), c, label=n)
lim = [dtr.min(), dtr.max()]; ax[1].plot(lim, lim, "k:", lw=1)
ax[1].set_xlabel("theoretical quantile"); ax[1].set_ylabel("sample quantile"); ax[1].legend(); ax[1].set_title("QQ plot")
plt.tight_layout(); plt.savefig(OUT+"q4_dtr_fit.png"); plt.close()

# ---------- Q5: conditional probability / expectation ----------
d2 = d.merge(daily[["RH"]], left_on=d.dt.dt.floor("D"), right_index=True)   # daily mean RH attached to hours
hour_rh = d.RH2M
A = d.kt > 0.7
R["p_kt07"] = float(A.mean())
for name, mask in [("RHlt40", d.RH2M < 40), ("RH40_70", (d.RH2M >= 40) & (d.RH2M < 70)), ("RHge70", d.RH2M >= 70)]:
    R["p_kt07_"+name] = float(A[mask].mean()); R["n_"+name] = int(mask.sum())
mon = d.month.isin([6,7,8,9])
R["p_kt07_monsoon"] = float(A[mon].mean()); R["p_kt07_nonmonsoon"] = float(A[~mon].mean())
R["E_kt_RHlt40"] = float(d.kt[d.RH2M < 40].mean()); R["E_kt_RHge70"] = float(d.kt[d.RH2M >= 70].mean())
R["E_H_by_month"] = {int(k): float(v) for k, v in daily.groupby("month").H_kwh.mean().items()}
# independence check: is P(A|B) = P(A)?  two-proportion z test, RH<40 vs RH>=70
x1, n1 = int(A[d.RH2M < 40].sum()), int((d.RH2M < 40).sum()); x2, n2 = int(A[d.RH2M >= 70].sum()), int((d.RH2M >= 70).sum())
pp = (x1+x2)/(n1+n2); z = (x1/n1-x2/n2)/np.sqrt(pp*(1-pp)*(1/n1+1/n2)); R["z_cond"] = float(z)

# ---------- Q6: one-sample t-test ----------
mj = daily[daily.month.isin([5,6])].H_kwh
R["mj_n"] = len(mj); R["mj_mean"] = float(mj.mean()); R["mj_sd"] = float(mj.std(ddof=1))
t = stats.ttest_1samp(mj, 5.5); R["mj_t"] = float(t.statistic); R["mj_p"] = float(t.pvalue)
ci = stats.t.interval(0.95, len(mj)-1, loc=mj.mean(), scale=mj.std(ddof=1)/np.sqrt(len(mj)))
R["mj_ci"] = [float(ci[0]), float(ci[1])]
R["mj_shapiro_p"] = float(stats.shapiro(mj).pvalue); R["mj_skew"] = float(stats.skew(mj))
R["mj_wilcoxon_p"] = float(stats.wilcoxon(mj - 5.5).pvalue)
R["mj_lag1_autocorr"] = float(mj.autocorr(1)) if hasattr(mj, "autocorr") else None
# power: chance of detecting a true 0.3 kWh shortfall at alpha 0.05 (normal approx, one-sample)
from statsmodels.stats.power import TTestPower
R["power_0p3"] = float(TTestPower().power(effect_size=0.3/mj.std(ddof=1), nobs=len(mj), alpha=0.05))
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.hist(mj, bins=20, color="#bbb", edgecolor="w"); ax.axvline(5.5, c="C3", label="benchmark 5.5"); ax.axvline(mj.mean(), c="C0", label=f"sample mean {mj.mean():.2f}")
ax.set_xlabel("daily irradiation, May-Jun (kWh/m2/day)"); ax.set_ylabel("days"); ax.legend(); plt.tight_layout(); plt.savefig(OUT+"q6_test.png"); plt.close()

# ---------- Stage 2: regression on daily clearness index ----------
daily["rain_log"] = np.log1p(daily.rain)
daily["rainy"] = (daily.rain >= 1.0).astype(int)
daily["sin_doy"] = np.sin(2*np.pi*daily.index.dayofyear/365.25); daily["cos_doy"] = np.cos(2*np.pi*daily.index.dayofyear/365.25)
tr = daily[daily.year == 2023]; va = daily[daily.year == 2024]; te = daily[daily.year == 2025]
R["split"] = dict(train=len(tr), val=len(va), test=len(te))
forms = {"M1: RH": "KT ~ RH",
         "M2: RH+DTR": "KT ~ RH + DTR",
         "M3: RH+DTR+wind+rain": "KT ~ RH + DTR + WS + rain_log",
         "M4: M3+season": "KT ~ RH + DTR + WS + rain_log + sin_doy + cos_doy"}
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
R["models"] = {}; fits = {}
for n, f in forms.items():
    m = smf.ols(f, tr).fit(); fits[n] = m
    R["models"][n] = dict(formula=f, train_rmse=rmse(tr.KT, m.predict(tr)), val_rmse=rmse(va.KT, m.predict(va)),
                          train_r2=float(m.rsquared), adj_r2=float(m.rsquared_adj), val_r2=float(1-((va.KT-m.predict(va))**2).sum()/((va.KT-va.KT.mean())**2).sum()),
                          aic=float(m.aic))
best = min(R["models"], key=lambda n: R["models"][n]["val_rmse"]); R["best_by_val"] = best
# baseline: predict training mean
R["baseline_val_rmse"] = rmse(va.KT, np.full(len(va), tr.KT.mean()))
# nested-model ANOVA F tests (training data)
nm = list(fits)
R["nested_anova"] = []
for i in range(1, len(nm)):
    a_ = anova_lm(fits[nm[i-1]], fits[nm[i]])
    R["nested_anova"].append(dict(comp=f"{nm[i-1]} vs {nm[i]}", F=float(a_["F"].iloc[1]), p=float(a_["Pr(>F)"].iloc[1])))
# final model: refit chosen specification on train+val, evaluate once on test
trva = pd.concat([tr, va])
fin_f = forms[best]
fin = smf.ols(fin_f, trva).fit()
R["final_formula"] = fin_f
R["final_params"] = {k: dict(coef=float(fin.params[k]), se=float(fin.bse[k]), t=float(fin.tvalues[k]), p=float(fin.pvalues[k]),
                             lo=float(fin.conf_int().loc[k,0]), hi=float(fin.conf_int().loc[k,1])) for k in fin.params.index}
R["final_F"] = float(fin.fvalue); R["final_Fp"] = float(fin.f_pvalue); R["final_r2"] = float(fin.rsquared); R["final_adjr2"] = float(fin.rsquared_adj)
R["final_df"] = [int(fin.df_model), int(fin.df_resid)]
an = anova_lm(fin, typ=2); R["final_anova_typ2"] = {k: dict(sumsq=float(an.loc[k,"sum_sq"]), df=float(an.loc[k,"df"]),
                       F=(None if np.isnan(an.loc[k,"F"]) else float(an.loc[k,"F"])), p=(None if np.isnan(an.loc[k,"PR(>F)"]) else float(an.loc[k,"PR(>F)"]))) for k in an.index}
pte = fin.predict(te)
R["test_rmse"] = rmse(te.KT, pte); R["test_mae"] = float(np.mean(np.abs(te.KT-pte)))
R["test_r2"] = float(1-((te.KT-pte)**2).sum()/((te.KT-te.KT.mean())**2).sum())
R["test_baseline_rmse"] = rmse(te.KT, np.full(len(te), trva.KT.mean()))
R["kt_sd_daily"] = float(daily.KT.std())
# VIF and residual diagnostics for final model
Xv = sm.add_constant(trva[["RH","DTR","WS","rain_log"] + (["sin_doy","cos_doy"] if "season" in best else [])])
R["vif"] = {c: float(variance_inflation_factor(Xv.values, i)) for i, c in enumerate(Xv.columns) if c != "const"}
R["dw"] = float(sm.stats.durbin_watson(fin.resid)); R["bp_p"] = float(sm.stats.het_breuschpagan(fin.resid, fin.model.exog)[1])
R["corr_RH_DTR"] = float(daily[["RH","DTR"]].corr().iloc[0,1]); R["corr_RH_KT"] = float(daily[["RH","KT"]].corr().iloc[0,1])
R["corr_DTR_KT"] = float(daily[["DTR","KT"]].corr().iloc[0,1])
# under/overfitting demonstration: polynomial expansion of M3 predictors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
cols = ["RH","DTR","WS","rain_log"]
sweep = []
for deg in [0, 1, 2, 3, 4, 5, 6]:
    if deg == 0:
        pr_tr, pr_va = np.full(len(tr), tr.KT.mean()), np.full(len(va), tr.KT.mean())
    else:
        mdl = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False), LinearRegression()).fit(tr[cols], tr.KT)
        pr_tr, pr_va = mdl.predict(tr[cols]), mdl.predict(va[cols])
    sweep.append((deg, rmse(tr.KT, pr_tr), rmse(va.KT, pr_va)))
R["poly_sweep"] = sweep
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].plot([s[0] for s in sweep], [s[1] for s in sweep], "o-", label="train"); ax[0].plot([s[0] for s in sweep], [s[2] for s in sweep], "s-", label="validation")
ax[0].set_ylim(0, 0.8); ax[0].set_xlabel("polynomial degree (0 = mean only; y-axis clipped)"); ax[0].set_ylabel("RMSE of daily KT"); ax[0].legend(); ax[0].set_title("Under/overfitting check")
ax[1].scatter(te.KT, pte, s=8, alpha=.6); ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set_xlabel("observed daily KT (2025)"); ax[1].set_ylabel("predicted"); ax[1].set_title("Test set 2025")
ax[2].scatter(fin.fittedvalues, fin.resid, s=6, alpha=.5); ax[2].axhline(0, c="k", lw=.8); ax[2].set_xlabel("fitted"); ax[2].set_ylabel("residual"); ax[2].set_title("Residuals (train+val)")
plt.tight_layout(); plt.savefig(OUT+"s2_model.png"); plt.close()
# time series of the split
fig, ax = plt.subplots(figsize=(8, 2.4)); 
for dat, c, l in [(tr,"C0","train 2023"),(va,"C1","validation 2024"),(te,"C3","test 2025")]: ax.plot(dat.index, dat.KT, c, lw=.7, label=l)
ax.set_ylabel("daily KT"); ax.legend(ncol=3, loc="lower center"); plt.tight_layout(); plt.savefig(OUT+"s2_split.png"); plt.close()
# robustness: residuals are autocorrelated (Durbin-Watson ~1.1), so also report HAC (Newey-West) p-values
hac = fin.get_robustcov_results(cov_type="HAC", maxlags=7)
R["hac_p"] = {k: float(v) for k, v in zip(fin.params.index, hac.pvalues)}
# effective sample size for the Q6 t-test given lag-1 autocorrelation r:  n_eff = n (1-r)/(1+r)
r1 = R["mj_lag1_autocorr"]; neff = R["mj_n"]*(1-r1)/(1+r1)
R["mj_neff"] = float(neff); R["mj_t_adj"] = float(R["mj_t"]*np.sqrt(neff/R["mj_n"]))
R["mj_p_adj"] = float(2*stats.t.sf(abs(R["mj_t_adj"]), neff-1))
daily.to_csv("daily.csv")
json.dump(R, open("results.json","w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
