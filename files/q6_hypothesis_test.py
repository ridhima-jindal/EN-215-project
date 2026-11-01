# EN 215 mini-project | Q6 one-sample t-test
# Data: NASA POWER hourly, Lucknow (26.85 N, 80.95 E), 2023-2025, local standard time.
# Put the CSV in the same folder as this script, then run:  python q6_hypothesis_test.py
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

# H0: mean daily irradiation in May-June = 5.5 kWh/m2/day (assumed benchmark); H1: it differs
BENCH = 5.5
mj = daily[daily.month.isin([5, 6])].H_kwh
n = len(mj)
t = stats.ttest_1samp(mj, BENCH)
ci = stats.t.interval(0.95, n - 1, loc=mj.mean(), scale=mj.std(ddof=1) / np.sqrt(n))
print("n = %d | mean = %.2f | sd = %.2f | t = %.2f | p = %.1e" % (n, mj.mean(), mj.std(ddof=1), t.statistic, t.pvalue))
print("95%% CI for the mean: %.2f to %.2f" % ci)

# Checks on the assumptions
print("Shapiro (normality) p = %.1e | skewness = %.2f" % (stats.shapiro(mj).pvalue, stats.skew(mj)))
print("Wilcoxon signed-rank p = %.1e (does not assume normality)" % stats.wilcoxon(mj - BENCH).pvalue)
r1 = mj.autocorr(1)                       # neighbouring days are correlated, so n overstates the evidence
neff = n * (1 - r1) / (1 + r1)
t_adj = t.statistic * np.sqrt(neff / n)
print("lag-1 autocorrelation = %.2f | effective n = %.0f | adjusted t = %.2f | adjusted p = %.4f"
      % (r1, neff, t_adj, 2 * stats.t.sf(abs(t_adj), neff - 1)))
# Power: chance of detecting a true 0.3 kWh/m2/day difference at alpha = 0.05
from statsmodels.stats.power import TTestPower
print("power to detect a 0.3 difference: %.2f" % TTestPower().power(effect_size=0.3 / mj.std(ddof=1), nobs=n, alpha=0.05))

fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.hist(mj, bins=20, color="#bbb", edgecolor="w")
ax.axvline(BENCH, c="C3", label="benchmark 5.5"); ax.axvline(mj.mean(), c="C0", label="sample mean %.2f" % mj.mean())
ax.set_xlabel("daily irradiation, May-Jun (kWh/m2/day)"); ax.set_ylabel("days"); ax.legend()
plt.tight_layout(); plt.savefig("figs/q6_test.png", dpi=130)
