"""Where do the large float32 prior-draw differences sit? Usage: run.sh analyze_prior_tail.py CASE_KIND"""
import sys

import numpy as np

from _build import OUT

d = np.load(f"{OUT}/stage2_precision/{sys.argv[1]}.npz")
l64, l32 = d["prior64_log_likelihood"], d["prior32_log_likelihood"]
fin = np.isfinite(l64) & np.isfinite(l32)
dl = np.abs(l32 - l64)
peak = np.nanmax(np.where(np.isfinite(l64), l64, np.nan))
print(f"peak prior-draw logL64 {peak:.2f}")
for lab, m in [("logL64 within 50 of best prior draw", fin & (l64 > peak - 50)),
               ("logL64 within 500", fin & (l64 > peak - 500)),
               ("soft-guard wall (sel corr < -1000)", fin & (d["prior64_selection_log_correction"] < -1000)),
               ("all finite", fin)]:
    if m.sum():
        print(f"{lab:40s} n={m.sum():4d} median {np.median(dl[m]):.2e} p99 {np.quantile(dl[m],0.99):.2e} max {dl[m].max():.2e}"
              f"  rel max {np.max(dl[m]/np.abs(l64[m])):.2e}")
mm = np.isfinite(l64) != np.isfinite(l32)
print("finite mismatches:", int(mm.sum()))
for i in np.where(mm)[0][:10]:
    print(f"  l64 {l64[i]:.3f} l32 {l32[i]:.3f} ninf64 {d['prior64_n_inf_events'][i]} ninf32 {d['prior32_n_inf_events'][i]}"
          f" neff64 {d['prior64_n_eff'][i]:.3g} neff32 {d['prior32_n_eff'][i]:.3g}")
o = np.argsort(-np.where(fin, dl, -1))[:8]
print("largest |dlogL|:")
for i in o:
    print(f"  dl {dl[i]:.3e} l64 {l64[i]:.2f} sel64 {d['prior64_selection_log_correction'][i]:.2f}"
          f" dsel {d['prior32_selection_log_correction'][i]-d['prior64_selection_log_correction'][i]:.3e}"
          f" devsum {d['prior32_sum_event_ll'][i]-d['prior64_sum_event_ll'][i]:.3e} neff64 {d['prior64_n_eff'][i]:.3g}")
