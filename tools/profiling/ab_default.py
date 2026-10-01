"""Default-path identity probe: optimized-HLO hash (metadata stripped) and logL bits.

Run once with this branch's src and once with main's src on PYTHONPATH, then
compare the two JSON files (compare mode).
Usage: run.sh ab_default.py OUT.json            (probe)
       run.sh ab_default.py --compare A.json B.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys


def probe(out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import darksirens
    from _build import build, prior_draws

    res = {"darksirens_file": darksirens.__file__}
    for case, kind in [("T", "spectral"), ("T", "dark"), ("R1", "spectral"), ("R1", "dark"),
                       ("real", "spectral")]:
        analysis, ll, ptform, _, _ = build(case, kind)
        f = ll.as_pytree_callable()
        th = prior_draws(ptform, len(analysis.parameters.labels), 16, seed=7)
        g = jax.jit(jax.vmap(lambda g, t: g(t), in_axes=(None, 0)))
        comp = g.lower(f, jnp.asarray(th)).compile()
        txt = comp.as_text()
        txt = re.sub(r", metadata=\{[^}]*\}", "", txt)
        txt = re.sub(r"metadata=\{[^}]*\}", "", txt)
        vals_vmap = np.asarray(comp(f, jnp.asarray(th)))
        vals_call = np.asarray([np.asarray(ll(jnp.asarray(t))) for t in th[:6]])
        res[f"{case}_{kind}"] = dict(
            hlo_sha256=hashlib.sha256(txt.encode()).hexdigest(),
            hlo_lines=txt.count("\n"),
            logL_vmap_hex=[float(v).hex() for v in vals_vmap],
            logL_call_hex=[float(v).hex() for v in vals_call],
        )
        print(case, kind, res[f"{case}_{kind}"]["hlo_sha256"][:16], vals_vmap[:4], flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    ok = True
    for k in A:
        if k == "darksirens_file":
            print("A:", A[k], "\nB:", B[k])
            continue
        same_hlo = A[k]["hlo_sha256"] == B[k]["hlo_sha256"]
        same_v = A[k]["logL_vmap_hex"] == B[k]["logL_vmap_hex"]
        same_c = A[k]["logL_call_hex"] == B[k]["logL_call_hex"]
        ok &= same_hlo and same_v and same_c
        print(f"{k:16s} optimized-HLO identical: {same_hlo}   vmap logL bitwise: {same_v} "
              f"({len(A[k]['logL_vmap_hex'])} pts)   call logL bitwise: {same_c}")
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        probe(sys.argv[1])
