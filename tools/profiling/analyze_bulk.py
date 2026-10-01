"""Break down the float32 bulk differences. Usage: run.sh analyze_bulk.py CASE_KIND"""
import json
import sys

import numpy as np

from _build import OUT

stem = f"{OUT}/stage2_precision/{sys.argv[1]}"
d = json.load(open(stem + ".json"))
z = np.load(stem + ".npz")
for k in ["dlogL_abs", "dlog_mu", "dn_eff_rel", "dselection_ll", "dsum_event_ll", "dpe_var_sum",
          "dtotal_lnL_variance"]:
    s = d["bulk_1x"][k]
    print(f"{k:22s} median {s.get('median_abs', float('nan')):.2e} p95 {s.get('p95_abs', float('nan')):.2e} "
          f"max {s.get('max_abs', float('nan')):.2e} mism {s['n_finite_mismatch']}")
print("map logL64", d["map"]["logL64"], "floored dirs", d["map"]["n_floored_directions"],
      "bulk logL64 range", d["bulk_logL64_range"])
print("estimates", json.dumps(d["estimates"])[:600])
l64, l32 = z["bulk64_log_likelihood"], z["bulk32_log_likelihood"]
s64, s32 = z["bulk64_selection_log_correction"], z["bulk32_selection_log_correction"]
tag = z["scale_tag"]
n = d["n_events"]
dl = l32 - l64
var64 = z["bulk64_pe_var_sum"] + n**2 / z["bulk64_n_eff"]
o = np.argsort(-np.abs(np.nan_to_num(dl)))[:10]
print("largest |dlogL| in the bulk draws:")
for i in o:
    print(f"  dl {dl[i]:+.3e} l64 {l64[i]:.2f} sel64 {s64[i]:.2f} dsel {s32[i]-s64[i]:+.3e} "
          f"devsum {z['bulk32_sum_event_ll'][i]-z['bulk64_sum_event_ll'][i]:+.3e} neff {z['bulk64_n_eff'][i]:.0f} "
          f"var64 {var64[i]:.2f} dlogmu {z['bulk32_log_mu'][i]-z['bulk64_log_mu'][i]:+.2e} tag {tag[i]:.0f}")
m = (tag == 1) & np.isfinite(dl)
thr = 1e-2
big = np.abs(dl[m]) > thr
print(f"bulk1x: fraction |dl|>{thr}: {big.mean():.3f}; their var64 range "
      f"{var64[m][big].min() if big.any() else 0:.2f}-{var64[m][big].max() if big.any() else 0:.2f};"
      f" var64 of the rest: median {np.median(var64[m][~big]):.2f}")
near = m & (var64 < 0.8 * 20.0)
print(f"bulk1x draws with total lnL variance < 16 (soft-guard gate shut): n={near.sum()} "
      f"max|dl| {np.abs(dl[near]).max() if near.any() else float('nan'):.2e} "
      f"p99 {np.quantile(np.abs(dl[near]), 0.99) if near.any() else float('nan'):.2e}")
