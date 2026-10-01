"""Precision table over all stage-2 cases. Usage: run.sh summarize_precision.py [--md OUT.md]

Subsets:
  prior/all        every prior draw finite in both precisions
  prior/top50      prior draws within 50 nats of the best prior draw
  bulk/all         every Laplace draw (1x and 2x), including those in the soft-guard wall
  bulk/post30      Laplace draws with logL64 > logL64(MAP) - 30 (posterior-relevant)
"""
import argparse
import json
import os

import numpy as np

from _build import OUT

CASES = ["real_spectral", "T_spectral", "T_dark", "R1_spectral", "R1_dark"]


def st(a, b, m):
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb & m
    d = np.abs(b[both] - a[both])
    mism = int(((fa != fb) & m).sum())
    if d.size == 0:
        return dict(n=int(both.sum()), mism=mism)
    return dict(n=int(both.sum()), mism=mism, med=float(np.median(d)), p95=float(np.quantile(d, .95)),
                p99=float(np.quantile(d, .99)), max=float(d.max()), mean_signed=float(np.mean(b[both] - a[both])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=f"{OUT}/stage2_precision/precision_table.md")
    args = ap.parse_args()
    lines = ["| case | subset | n | finite mismatch | median abs dlogL | p95 | p99 | max | mean signed | "
             "N_eff rel max | log_mu abs max | total-variance abs max |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    est_lines = ["| case | MAP logL64 | ESS of Laplace proposal | dlogZ estimate | posterior-weighted dlogL mean +- sd | max param-mean shift (sigma) |",
                 "|---|---|---|---|---|---|"]
    out = {}
    for c in CASES:
        stem = f"{OUT}/stage2_precision/{c}"
        if not os.path.exists(stem + ".npz"):
            continue
        z = np.load(stem + ".npz")
        j = json.load(open(stem + ".json"))
        n_ev = j["n_events"]
        res = {}
        for side in ("prior", "bulk"):
            l64, l32 = z[f"{side}64_log_likelihood"], z[f"{side}32_log_likelihood"]
            if side == "prior":
                best = np.nanmax(np.where(np.isfinite(l64), l64, np.nan))
                subsets = {"all": np.ones_like(l64, bool), "top50": l64 > best - 50}
            else:
                lm = j["map"]["logL64"]
                subsets = {"all": np.ones_like(l64, bool), "post30": l64 > lm - 30}
            for name, m in subsets.items():
                s = st(l64, l32, m)
                ne64, ne32 = z[f"{side}64_n_eff"], z[f"{side}32_n_eff"]
                ok = m & np.isfinite(ne64) & np.isfinite(ne32) & (ne64 > 0)
                s["neff_rel_max"] = float(np.max(np.abs(ne32[ok] / ne64[ok] - 1))) if ok.any() else None
                mu64, mu32 = z[f"{side}64_log_mu"], z[f"{side}32_log_mu"]
                ok2 = m & np.isfinite(mu64) & np.isfinite(mu32)
                s["logmu_abs_max"] = float(np.max(np.abs(mu32[ok2] - mu64[ok2]))) if ok2.any() else None
                v64 = z[f"{side}64_pe_var_sum"] + n_ev**2 / z[f"{side}64_n_eff"]
                v32 = z[f"{side}32_pe_var_sum"] + n_ev**2 / z[f"{side}32_n_eff"]
                ok3 = ok & np.isfinite(v64) & np.isfinite(v32)
                s["var_abs_max"] = float(np.max(np.abs(v32[ok3] - v64[ok3]))) if ok3.any() else None
                res[f"{side}/{name}"] = s
                f = lambda k: (f"{s[k]:.2e}" if s.get(k) is not None else "-")
                lines.append(f"| {c} | {side}/{name} | {s['n']} | {s['mism']} | {f('med')} | {f('p95')} | "
                             f"{f('p99')} | {f('max')} | {f('mean_signed')} | {f('neff_rel_max')} | "
                             f"{f('logmu_abs_max')} | {f('var_abs_max')} |")
        e = j["estimates"]
        sh = max(abs(v) for v in e["param_mean_shift_in_sigma"].values())
        est_lines.append(f"| {c} | {j['map']['logL64']:.2f} | {e['ess_float64_weights']:.1f} | "
                         f"{e['dlogZ_float32_minus_float64']:.2e} | {e['posterior_weighted_dlogL_mean']:.2e} +- "
                         f"{e['posterior_weighted_dlogL_sd']:.2e} | {sh:.2e} |")
        out[c] = res
    txt = "\n".join(lines) + "\n\nEstimates (importance reweighting over the 1x Laplace draws):\n\n" + "\n".join(est_lines) + "\n"
    print(txt)
    with open(args.md, "w") as fh:
        fh.write(txt)
    with open(args.md.replace(".md", ".json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
