"""N_eff, selection variance N^2/N_eff and PE variance at the posterior-relevant bulk draws."""
import json

import numpy as np

from _build import OUT

for c in ["real_spectral", "T_spectral", "T_dark", "R1_spectral", "R1_dark"]:
    z = np.load(f"{OUT}/stage2_precision/{c}.npz")
    j = json.load(open(f"{OUT}/stage2_precision/{c}.json"))
    m = z["bulk64_log_likelihood"] > j["map"]["logL64"] - 30
    ne = z["bulk64_n_eff"][m]
    n = j["n_events"]
    print(f"{c:14s} N={n:4d} N_inj={j['n_sel']:8d} n_eff median {np.median(ne):.4g} [{ne.min():.3g}, {ne.max():.3g}] "
          f"n_eff/N_inj {np.median(ne)/j['n_sel']:.3g}  N^2/N_eff median {np.median(n*n/ne):.3g}  "
          f"sum PE var median {np.median(z['bulk64_pe_var_sum'][m]):.3g}  "
          f"H0 MAP {j['map']['theta'][0]:.1f} sd {j['map']['laplace_sd'][0]:.1f}")
