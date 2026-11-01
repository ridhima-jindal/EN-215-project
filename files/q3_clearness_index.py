# EN 215 mini-project | Q3 first variable: hourly clearness index
# Data: NASA POWER hourly, Lucknow (26.85 N, 80.95 E), 2023-2025, local standard time.
# Put the CSV in the same folder as this script, then run:  python q3_clearness_index.py
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

# Hourly clearness index kt = all-sky / clear-sky irradiance.
# Keep only hours with clear-sky >= 50 Wh/m2: near sunrise/sunset the ratio is unstable.
d = raw[raw.CLRSKY_SFC_SW_DWN >= 50].copy()
d["kt"] = d.ALLSKY_SFC_SW_DWN / d.CLRSKY_SFC_SW_DWN
d["kt"] = d.kt.clip(lower=0.001, upper=0.999)   # kt above 1 is impossible for Beta; cap it
d["month"] = d.dt.dt.month

kt = d.kt.values
print("n = %d | mean = %.3f | variance = %.4f | sd = %.3f | skewness = %.2f"
      % (len(kt), kt.mean(), kt.var(ddof=1), kt.std(ddof=1), stats.skew(kt)))

# Fit four candidate distributions by maximum likelihood; compare with AIC and the KS distance
fits = {"Beta": (stats.beta, stats.beta.fit(kt, floc=0, fscale=1)),
        "Normal": (stats.norm, (kt.mean(), kt.std())),
        "Gamma": (stats.gamma, stats.gamma.fit(kt, floc=0)),
        "Weibull": (stats.weibull_min, stats.weibull_min.fit(kt, floc=0))}
for name, (dist, par) in fits.items():
    k = 2                                           # two free parameters each
    aic = 2 * k - 2 * dist.logpdf(kt, *par).sum()
    ks = stats.kstest(kt, dist.cdf, args=par).statistic
    print("%-8s AIC = %10.0f   KS = %.3f" % (name, aic, ks))
a, b = fits["Beta"][1][:2]
print("Beta a = %.2f, b = %.2f | theory mean = %.3f, theory variance = %.4f"
      % (a, b, a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1))))

# Plot: histogram with fitted densities, and empirical CDF with fitted CDFs
xs = np.linspace(0.001, 1, 400)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
ax[0].hist(kt, bins=50, density=True, color="#bbb", edgecolor="w")
for n, c in [("Beta", "C3"), ("Gamma", "C0"), ("Normal", "C2")]:
    dist, par = fits[n]; ax[0].plot(xs, dist.pdf(xs, *par), c, label=n)
ax[0].set_xlabel("hourly clearness index kt (ratio)"); ax[0].set_ylabel("density"); ax[0].legend()
ax[0].set_title("Histogram with fitted densities")
srt = np.sort(kt); ax[1].plot(srt, np.arange(1, len(srt) + 1) / len(srt), "k", lw=2, label="empirical")
for n, c in [("Beta", "C3"), ("Gamma", "C0"), ("Normal", "C2")]:
    dist, par = fits[n]; ax[1].plot(xs, dist.cdf(xs, *par), c, ls="--", label=n)
ax[1].set_xlabel("kt"); ax[1].set_ylabel("CDF"); ax[1].legend(); ax[1].set_title("Empirical vs fitted CDF")
plt.tight_layout(); plt.savefig("figs/q3_kt_fit.png", dpi=130)
