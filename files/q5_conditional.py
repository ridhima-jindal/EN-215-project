# EN 215 mini-project | Q5 conditional probability and expectation
# Data: NASA POWER hourly, Lucknow (26.85 N, 80.95 E), 2023-2025, local standard time.
# Put the CSV in the same folder as this script, then run:  python q5_conditional.py
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

# Event A = a very clear hour (kt > 0.7). Condition B = humidity level (or season).
A = d.kt > 0.7
print("P(A) = %.3f" % A.mean())
groups = {"RH < 40 %": d.RH2M < 40, "RH 40-70 %": (d.RH2M >= 40) & (d.RH2M < 70), "RH >= 70 %": d.RH2M >= 70,
          "Monsoon (Jun-Sep)": d.month.isin([6, 7, 8, 9]), "Non-monsoon": ~d.month.isin([6, 7, 8, 9])}
for name, mask in groups.items():
    # P(A | B) = P(A and B) / P(B) = share of hours in B that are also in A
    print("P(A | %-18s) = %.3f   hours = %5d   E[kt | B] = %.3f" % (name, A[mask].mean(), mask.sum(), d.kt[mask].mean()))
# Are A and "dry" independent? Two-proportion z-test, RH<40 vs RH>=70
m1, m2 = d.RH2M < 40, d.RH2M >= 70
x1, n1, x2, n2 = A[m1].sum(), m1.sum(), A[m2].sum(), m2.sum()
p = (x1 + x2) / (n1 + n2)
z = (x1 / n1 - x2 / n2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
print("two-proportion z = %.1f" % z)

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

print("E[daily irradiation | month], kWh/m2/day:")
print(daily.groupby("month").H_kwh.mean().round(2).to_string())
