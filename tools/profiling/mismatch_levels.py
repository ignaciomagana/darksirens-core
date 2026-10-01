"""How far below the MAP are the float32/float64 finite mismatches?"""
import json

import numpy as np

from _build import OUT

for c in ["real_spectral", "T_spectral", "T_dark", "R1_spectral", "R1_dark"]:
    z = np.load(f"{OUT}/stage2_precision/{c}.npz")
    lm = json.load(open(f"{OUT}/stage2_precision/{c}.json"))["map"]["logL64"]
    for side in ("prior", "bulk"):
        a, b = z[f"{side}64_log_likelihood"], z[f"{side}32_log_likelihood"]
        mm = np.isfinite(a) != np.isfinite(b)
        f64fin = mm & np.isfinite(a)
        top = a[f64fin].max() if f64fin.any() else np.nan
        print(f"{c:14s} {side:5s} mismatches {mm.sum():3d} (f64 finite, f32 -inf: {f64fin.sum():3d}; reverse {(mm & np.isfinite(b)).sum()}) "
              f"highest such logL64 = MAP - {lm - top:.0f}")
