"""Opt-in DARKSIRENS_EXP_PAIRING_ANALYTIC_SCALE=1 against the default.

Compares logL bits and gradient finiteness at the stage-2 prior and bulk draws,
plus the timing and XLA temp memory at batch 1.
Usage: run.sh validate_analytic_scale.py CASE KIND [--n-prior 100]
Appends to $OUT/stage1_proto/analytic_scale_validation.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from _build import OUT, build, load_stores, make_analysis

FLAG = "DARKSIRENS_EXP_PAIRING_ANALYTIC_SCALE"


def arm(ll, th_all, grad_pts):
    import jax
    import jax.numpy as jnp
    f = ll.as_pytree_callable()
    g = jax.jit(jax.vmap(lambda g, t: g(t), in_axes=(None, 0)))
    vals = np.concatenate([np.asarray(g(f, jnp.asarray(th_all[i:i + 8]))) for i in range(0, len(th_all), 8)
                           ]) if len(th_all) % 8 == 0 else None
    vg = jax.jit(jax.value_and_grad(lambda g, t: g(t), argnums=1))
    grads = np.stack([np.asarray(vg(f, jnp.asarray(t))[1]) for t in grad_pts])
    t1 = jnp.asarray(th_all[:1])
    comp = g.lower(f, t1).compile()
    comp(f, t1).block_until_ready()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        comp(f, t1).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return vals, grads, float(np.median(ts)), comp.memory_analysis().temp_size_in_bytes / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("kind")
    ap.add_argument("--n-prior", type=int, default=96)
    ap.add_argument("--n-grad", type=int, default=24)
    args = ap.parse_args()
    z = np.load(f"{OUT}/stage2_precision/{args.case}_{args.kind}.npz")
    th = np.concatenate([z["th_prior"][: args.n_prior], z["th_bulk"][:200]])
    th = th[: (len(th) // 8) * 8]
    l64 = z["bulk64_log_likelihood"][:200]
    grad_pts = np.concatenate([z["th_bulk"][: args.n_grad // 2],
                               z["th_prior"][np.isfinite(z["prior64_log_likelihood"])][: args.n_grad // 2]])
    analysis = make_analysis(args.case, args.kind)
    ev, inj = load_stores(args.case, analysis)
    os.environ.pop(FLAG, None)
    _, ll0, _, _, _ = build(args.case, args.kind, events=ev, injections=inj, analysis=analysis)
    v0, g0, t0, m0 = arm(ll0, th, grad_pts)
    os.environ[FLAG] = "1"
    _, ll1, _, _, _ = build(args.case, args.kind, events=ev, injections=inj, analysis=analysis)
    v1, g1, t1, m1 = arm(ll1, th, grad_pts)
    os.environ.pop(FLAG, None)
    fin = np.isfinite(v0) & np.isfinite(v1)
    same = (v0.view(np.int64) == v1.view(np.int64))
    d = np.abs(v1[fin] - v0[fin])
    gf0, gf1 = np.isfinite(g0).all(axis=1), np.isfinite(g1).all(axis=1)
    both = gf0 & gf1
    grel = np.abs(g1[both] - g0[both]) / np.maximum(np.abs(g0[both]), 1e-12)
    rec = dict(case=args.case, kind=args.kind, n_points=int(len(th)), n_finite_both=int(fin.sum()),
               n_bitwise_equal=int(same.sum()), finite_mismatch=int((np.isfinite(v0) != np.isfinite(v1)).sum()),
               dlogL_max_abs=float(d.max()) if d.size else None,
               n_grad_points=int(len(grad_pts)), grad_finite_default=int(gf0.sum()),
               grad_finite_flag=int(gf1.sum()),
               grad_rel_diff_max=float(grel.max()) if grel.size else None,
               grad_rel_diff_median=float(np.median(grel)) if grel.size else None,
               t_call_default_s=t0, t_call_flag_s=t1, speedup=t0 / t1,
               xla_temp_default_mb=m0, xla_temp_flag_mb=m1, loadavg=os.getloadavg())
    with open(f"{OUT}/stage1_proto/analytic_scale_validation.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))
    del l64


if __name__ == "__main__":
    main()
