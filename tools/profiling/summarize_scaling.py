"""Stage 3 table plus linear fits t = a + b*x for each swept axis.
Usage: run.sh summarize_scaling.py [--md OUT.md]"""
import argparse
import json

import numpy as np

from _build import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=f"{OUT}/stage3_scaling/scaling_table.md")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{OUT}/stage3_scaling/scaling.jsonl") if l.startswith("{")]
    L = ["| case | kind | dtype | n_events | nsamp | N_pe | N_inj | catalog rows x N_max | galaxies | ms/call (median) | XLA temp MB | GB accessed (XLA est.) | peak RSS MB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cs = "-" if r["catalog_shape"] is None else f"{r['catalog_shape'][0]}x{r['catalog_shape'][1]}"
        L.append(f"| {r['case']} | {r['kind']} | {r['compute_dtype'] or 'f64'} | {r['n_events']} | {r['nsamp']} | {r['n_pe']} | "
                 f"{r['n_sel']} | {cs} | {r['n_galaxies'] or '-'} | {1e3*r['t_call_median_s']:.1f} | {r['xla_temp_mb']:.0f} | "
                 f"{(r['xla_bytes_accessed'] or 0)/1e9:.2f} | {r['peak_rss_mb']:.0f} |")
    fits = []

    def fit(sel, xkey, label):
        rs = [r for r in rows if sel(r)]
        if len(rs) < 3:
            return
        x = np.array([float(xkey(r)) for r in rs])
        for ykey, yl in (("t_call_median_s", "s/call"), ("xla_temp_mb", "temp MB")):
            y = np.array([r[ykey] for r in rs])
            b, a = np.polyfit(x, y, 1)
            k = np.polyfit(np.log(x), np.log(np.maximum(y, 1e-9)), 1)[0]
            fits.append(f"| {label} | {yl} | {a:.4g} | {b:.4g} | {k:.2f} | {x.min():.3g}-{x.max():.3g} |")

    real = lambda r: r["case"] == "real" and r["compute_dtype"] is None
    fit(lambda r: real(r) and r["nsamp_arg"] is None and r["inj_arg"] == 1.0, lambda r: r["n_pe"], "real spectral: N_pe (events axis)")
    fit(lambda r: real(r) and r["events_arg"] == 1.0 and r["inj_arg"] == 1.0, lambda r: r["n_pe"], "real spectral: N_pe (nsamp axis)")
    fit(lambda r: real(r) and r["events_arg"] == 1.0 and r["nsamp_arg"] is None, lambda r: r["n_sel"], "real spectral: N_inj")
    d = lambda r: r["case"] == "R1" and r["kind"] == "dark" and r["compute_dtype"] is None
    fit(lambda r: d(r) and r["inj_arg"] == 1.0 and r["events_arg"] == 1.0, lambda r: r["catalog_shape"][1], "R1 dark: N_max (galaxies per row)")
    fit(lambda r: d(r) and r["gal_arg"] == 1 and r["events_arg"] == 1.0, lambda r: r["n_sel"], "R1 dark: N_inj")
    fit(lambda r: d(r) and r["gal_arg"] == 1 and r["inj_arg"] == 1.0, lambda r: r["n_pe"], "R1 dark: N_pe")
    txt = "\n".join(L) + "\n\nFits y = a + b*x (and log-log slope k):\n\n| axis | y | a | b | k | x range |\n|---|---|---|---|---|---|\n" + "\n".join(fits) + "\n"
    print(txt)
    open(args.md, "w").write(txt)


if __name__ == "__main__":
    main()
