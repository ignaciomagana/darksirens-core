"""Capture a jax.profiler trace of the compiled likelihood and aggregate per-HLO-op time.

Usage: run.sh trace_profile.py CASE KIND BATCH [--reps N]
Writes $OUT/stage1_trace/<case>_<kind>_b<B>/ (raw trace) and ops.csv (aggregated).
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import time

from _build import OUT, build, prior_draws


def aggregate(trace_dir):
    files = glob.glob(f"{trace_dir}/**/*.trace.json.gz", recursive=True)
    if not files:
        return None
    with gzip.open(sorted(files)[-1]) as fh:
        tr = json.load(fh)
    ev = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X"]
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        name = e.get("name", "")
        agg[name][0] += float(e.get("dur", 0.0))
        agg[name][1] += 1
    return sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda r: -r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("kind")
    ap.add_argument("batch", type=int)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp

    analysis, ll, ptform, _, _ = build(args.case, args.kind)
    f = ll.as_pytree_callable()
    ndim = len(analysis.parameters.labels)
    th = jnp.asarray(prior_draws(ptform, ndim, args.batch, seed=5))
    g = jax.jit(jax.vmap(lambda g, t: g(t), in_axes=(None, 0)))
    g(f, th).block_until_ready()
    g(f, th).block_until_ready()
    d = args.outdir or f"{OUT}/stage1_trace/{args.case}_{args.kind}_b{args.batch}"
    os.makedirs(d, exist_ok=True)
    t0 = time.perf_counter()
    with jax.profiler.trace(d):
        for _ in range(args.reps):
            g(f, th).block_until_ready()
    wall = (time.perf_counter() - t0) / args.reps
    rows = aggregate(d)
    with open(f"{d}/ops.csv", "w") as fh:
        fh.write("name,total_us_per_call,count_per_call\n")
        for name, dur, cnt in rows or []:
            fh.write(f"\"{name}\",{dur/args.reps:.1f},{cnt/args.reps:.1f}\n")
    print("wall per call (s)", wall)
    for r in (rows or [])[:40]:
        print(f"{r[1]/args.reps/1e3:10.2f} ms  x{r[2]/args.reps:6.1f}  {r[0][:120]}")


if __name__ == "__main__":
    main()
