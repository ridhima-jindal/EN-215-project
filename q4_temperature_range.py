# EN 215 mini-project | Q4 second variable: daily temperature range
# Data: NASA POWER hourly, Lucknow (26.85 N, 80.95 E), 2023-2025, local standard time.
# Put the CSV in the same folder as this script, then run:  python q4_temperature_range.py
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

dtr = daily.DTR.values
print("days = %d | mean = %.2f C | variance = %.2f C^2 | sd = %.2f | skewness = %.2f"
      % (len(dtr), dtr.mean(), dtr.var(ddof=1), dtr.std(ddof=1), stats.skew(dtr)))
print("min = %.1f, max = %.1f deg C" % (dtr.min(), dtr.max()))

fits = {"Weibull": (stats.weibull_min, stats.weibull_min.fit(dtr, floc=0)),
        "Normal": (stats.norm, (dtr.mean(), dtr.std())),
        "Gamma": (stats.gamma, stats.gamma.fit(dtr, floc=0)),
        "Lognormal": (stats.lognorm, stats.lognorm.fit(dtr, floc=0))}
for name, (dist, par) in fits.items():
    aic = 2 * 2 - 2 * dist.logpdf(dtr, *par).sum()
    ks = stats.kstest(dtr, dist.cdf, args=par).statistic
    print("%-10s AIC = %7.0f   KS = %.3f" % (name, aic, ks))
print("Weibull shape = %.2f (close to 3 means nearly bell-shaped)" % fits["Weibull"][1][0])

# Plot: histogram with fitted densities, and QQ plot against each candidate
xs = np.linspace(dtr.min(), dtr.max(), 300)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
ax[0].hist(dtr, bins=30, density=True, color="#bbb", edgecolor="w")
for n, c in [("Gamma", "C0"), ("Normal", "C2"), ("Weibull", "C1")]:
    dist, par = fits[n]; ax[0].plot(xs, dist.pdf(xs, *par), c, label=n)
ax[0].set_xlabel("daily temperature range (deg C)"); ax[0].set_ylabel("density"); ax[0].legend()
ax[0].set_title("Histogram with fitted densities")
q = np.linspace(0.01, 0.99, 99)
for n, c in [("Gamma", "C0"), ("Normal", "C2"), ("Weibull", "C1")]:
    dist, par = fits[n]; ax[1].plot(dist.ppf(q, *par), np.quantile(dtr, q), c, label=n)
ax[1].plot([dtr.min(), dtr.max()], [dtr.min(), dtr.max()], "k:", lw=1)
ax[1].set_xlabel("theoretical quantile"); ax[1].set_ylabel("sample quantile"); ax[1].legend(); ax[1].set_title("QQ plot")
plt.tight_layout(); plt.savefig("figs/q4_dtr_fit.png", dpi=130)
