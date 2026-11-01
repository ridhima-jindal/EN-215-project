# EN 215 mini-project | Q2 data quality
# Data: NASA POWER hourly, Lucknow (26.85 N, 80.95 E), 2023-2025, local standard time.
# Put the CSV in the same folder as this script, then run:  python q2_data_quality.py
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

vars_ = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M", "PRECTOTCORR"]
print("rows:", len(raw))
# 1. NASA POWER marks missing values as -999
print("-999 values:", int((raw[vars_] == -999).sum().sum()), "| NaN:", int(raw[vars_].isna().sum().sum()))
# 2. duplicates and gaps in the hourly time index
expected = len(pd.date_range(raw.dt.min(), raw.dt.max(), freq="h"))
print("duplicate timestamps:", int(raw.duplicated("dt").sum()), "| rows:", len(raw), "| expected hours:", expected)
# 3. night zeros: real (sun below horizon), not missing
print("share of irradiance exactly 0: %.3f" % (raw.ALLSKY_SFC_SW_DWN == 0).mean())
day = raw[raw.CLRSKY_SFC_SW_DWN > 0]
print("mean over ALL hours: %.0f Wh/m2 | mean over DAYTIME hours: %.0f Wh/m2 (n=%d)"
      % (raw.ALLSKY_SFC_SW_DWN.mean(), day.ALLSKY_SFC_SW_DWN.mean(), len(day)))
# 4. clearness index above 1 and the dawn/dusk filter
d = raw[raw.CLRSKY_SFC_SW_DWN >= 50].copy()
ratio = d.ALLSKY_SFC_SW_DWN / d.CLRSKY_SFC_SW_DWN
print("hours kept for kt (clear-sky >= 50):", len(d))
print("kt above 1: %d hours, max %.3f" % ((ratio > 1).sum(), ratio.max()))
# 5. precipitation: mostly zeros with a heavy tail (real storms, not errors)
print("share of hours with zero rain: %.2f | max: %.1f mm/h" % ((raw.PRECTOTCORR == 0).mean(), raw.PRECTOTCORR.max()))
print("RH range: %.1f to %.1f %% | T range: %.1f to %.1f deg C" % (raw.RH2M.min(), raw.RH2M.max(), raw.T2M.min(), raw.T2M.max()))
