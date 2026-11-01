# EN 215 mini-project | Stage 2 linear regression of daily clearness index
# Data: NASA POWER hourly, Lucknow (26.85 N, 80.95 E), 2023-2025, local standard time.
# Put the CSV in the same folder as this script, then run:  python stage2_regression.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CSV = "POWER_Point_Hourly_20230101_20251231_026d85N_080d95E_LST.csv"
os.makedirs("figs", exist_ok=True)

# The file has a 14-line text header, so the table starts at line 15
raw = pd.read_csv(CSV, skiprows=14)
raw["dt"] = pd.to_datetime(dict(year=raw.YEAR, month=raw.MO, day=raw.DY, hour=raw.HR))

# Daily table: one row per day
raw["date"] = raw.dt.dt.floor("D")
daily = raw.groupby("date").agg(T_max=("T2M", "max"), T_min=("T2M", "min"), RH=("RH2M", "mean"),
                                WS=("WS10M", "mean"), rain=("PRECTOTCORR", "sum"),
                                H_all=("ALLSKY_SFC_SW_DWN", "sum"), H_clr=("CLRSKY_SFC_SW_DWN", "sum"))
daily["DTR"] = daily.T_max - daily.T_min          # daily temperature range, deg C
daily["H_kwh"] = daily.H_all / 1000.0             # daily irradiation, kWh/m2/day
daily["KT"] = daily.H_all / daily.H_clr           # daily clearness index
daily["month"] = daily.index.month
daily["year"] = daily.index.year

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

daily["rain_log"] = np.log1p(daily.rain)                       # log(1 + rain) tames the heavy right tail
daily["sin_doy"] = np.sin(2 * np.pi * daily.index.dayofyear / 365.25)   # season as smooth sin/cos terms
daily["cos_doy"] = np.cos(2 * np.pi * daily.index.dayofyear / 365.25)
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

# ---- Q1: split by YEAR, never at random (daily values are autocorrelated) ----
tr, va, te = daily[daily.year == 2023], daily[daily.year == 2024], daily[daily.year == 2025]
print("train/val/test days:", len(tr), len(va), len(te))

# ---- Q3: compare nested models on the validation year ----
forms = {"M1: RH": "KT ~ RH", "M2: RH+DTR": "KT ~ RH + DTR",
         "M3: +wind+rain": "KT ~ RH + DTR + WS + rain_log",
         "M4: +season": "KT ~ RH + DTR + WS + rain_log + sin_doy + cos_doy"}
print("baseline (training mean) validation RMSE: %.3f" % rmse(va.KT, np.full(len(va), tr.KT.mean())))
fits = {}
for name, f in forms.items():
    m = smf.ols(f, tr).fit(); fits[name] = m
    print("%-16s train RMSE %.3f | val RMSE %.3f | adj R2 (train) %.3f"
          % (name, rmse(tr.KT, m.predict(tr)), rmse(va.KT, m.predict(va)), m.rsquared_adj))
names = list(fits)
for i in range(1, len(names)):                                  # F-test for each added block of terms
    a = anova_lm(fits[names[i - 1]], fits[names[i]])
    print("F-test %s vs %s: F = %.1f, p = %.1e" % (names[i - 1], names[i], a["F"].iloc[1], a["Pr(>F)"].iloc[1]))

# ---- Q2: final model refitted on train+val, coefficient tests and ANOVA ----
best = "M4: +season"
trva = pd.concat([tr, va])
fin = smf.ols(forms[best], trva).fit()
print(fin.summary().tables[1])                                  # coefficients, SE, t, p, 95% CI
print("overall F = %.1f (df %d, %d), p = %.1e | R2 = %.3f | adj R2 = %.3f"
      % (fin.fvalue, fin.df_model, fin.df_resid, fin.f_pvalue, fin.rsquared, fin.rsquared_adj))
print(anova_lm(fin, typ=2))                                     # F-test for each term added last
hac = fin.get_robustcov_results(cov_type="HAC", maxlags=7)       # autocorrelation-robust p-values
print("HAC p-values:", dict(zip(fin.params.index, np.round(hac.pvalues, 4))))
X = sm.add_constant(trva[["RH", "DTR", "WS", "rain_log", "sin_doy", "cos_doy"]])
print("VIF:", {c: round(variance_inflation_factor(X.values, i), 1) for i, c in enumerate(X.columns) if c != "const"})
print("corr(RH, DTR) = %.2f | Durbin-Watson = %.2f | Breusch-Pagan p = %.1e"
      % (daily[["RH", "DTR"]].corr().iloc[0, 1], sm.stats.durbin_watson(fin.resid), sm.stats.het_breuschpagan(fin.resid, fin.model.exog)[1]))

# ---- Test year scored ONCE, after the model was fixed ----
pte = fin.predict(te)
r2 = 1 - ((te.KT - pte) ** 2).sum() / ((te.KT - te.KT.mean()) ** 2).sum()
print("TEST 2025: RMSE %.3f | MAE %.3f | R2 %.2f | baseline RMSE %.3f"
      % (rmse(te.KT, pte), np.mean(np.abs(te.KT - pte)), r2, rmse(te.KT, np.full(len(te), trva.KT.mean()))))

# ---- Under/overfitting: polynomial degree sweep (0 = predict the mean) ----
cols = ["RH", "DTR", "WS", "rain_log"]; sweep = []
for deg in range(0, 7):
    if deg == 0:
        ptr, pva = np.full(len(tr), tr.KT.mean()), np.full(len(va), tr.KT.mean())
    else:
        mdl = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False), LinearRegression()).fit(tr[cols], tr.KT)
        ptr, pva = mdl.predict(tr[cols]), mdl.predict(va[cols])
    sweep.append((deg, rmse(tr.KT, ptr), rmse(va.KT, pva))); print("degree %d: train %.3f | val %.3f" % sweep[-1])

# ---- Figures ----
fig, ax = plt.subplots(figsize=(8, 2.4))
for dat, c, l in [(tr, "C0", "train 2023"), (va, "C1", "validation 2024"), (te, "C3", "test 2025")]:
    ax.plot(dat.index, dat.KT, c, lw=.7, label=l)
ax.set_ylabel("daily KT"); ax.legend(ncol=3, loc="lower center"); plt.tight_layout(); plt.savefig("figs/s2_split.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].plot([s[0] for s in sweep], [s[1] for s in sweep], "o-", label="train")
ax[0].plot([s[0] for s in sweep], [s[2] for s in sweep], "s-", label="validation")
ax[0].set_ylim(0, 0.8); ax[0].set_xlabel("polynomial degree (0 = mean only; y-axis clipped)"); ax[0].set_ylabel("RMSE of daily KT"); ax[0].legend()
ax[0].set_title("Under/overfitting check")
ax[1].scatter(te.KT, pte, s=8, alpha=.6); ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set_xlabel("observed daily KT (2025)"); ax[1].set_ylabel("predicted"); ax[1].set_title("Test set 2025")
ax[2].scatter(fin.fittedvalues, fin.resid, s=6, alpha=.5); ax[2].axhline(0, c="k", lw=.8); ax[2].set_xlabel("fitted"); ax[2].set_ylabel("residual"); ax[2].set_title("Residuals (train+val)")
plt.tight_layout(); plt.savefig("figs/s2_model.png", dpi=130)
