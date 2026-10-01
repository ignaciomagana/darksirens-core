"""Stage 1 timing: per-point time vs batch, compile time, peak RSS, XLA memory/cost.

One (case, kind, batch) per process so the peak RSS is that configuration's.
Usage: run.sh bench_timing.py CASE KIND BATCH [--tag TAG] [--bind k=v ...]
Appends one JSON line to $OUT/stage1_timing/timing.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from _build import OUT, build, cur_rss_mb, prior_draws, rss_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("kind")
    ap.add_argument("batch", type=int)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--min-time", type=float, default=8.0)
    ap.add_argument("--min-reps", type=int, default=3)
    ap.add_argument("--max-reps", type=int, default=50)
    ap.add_argument("--bind", nargs="*", default=[])
    ap.add_argument("--max-temp-gb", type=float, default=200.0)
    ap.add_argument("--out", default=f"{OUT}/stage1_timing/timing.jsonl")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    bind_kw = {}
    for kv in args.bind:
        k, v = kv.split("=", 1)
        bind_kw[k] = None if v == "None" else (int(v) if v.isdigit() else v)

    t0 = time.perf_counter()
    analysis, ll, ptform, _, _ = build(args.case, args.kind, **bind_kw)
    t_build = time.perf_counter() - t0
    rss_after_build = cur_rss_mb()
    ndim = len(analysis.parameters.labels)
    f = ll.as_pytree_callable()
    B = args.batch
    thetas = prior_draws(ptform, ndim, B * 4, seed=123).reshape(4, B, ndim)

    batched = jax.jit(jax.vmap(lambda g, t: g(t), in_axes=(None, 0)))
    t0 = time.perf_counter()
    lowered = batched.lower(f, jnp.asarray(thetas[0]))
    t_lower = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0
    mem = compiled.memory_analysis()
    cost = compiled.cost_analysis()
    if isinstance(cost, list):
        cost = cost[0] if cost else {}
    cost = dict(cost or {})

    temp_gb = (mem.temp_size_in_bytes / 2**30) if mem else 0.0
    if temp_gb > args.max_temp_gb:
        rec = dict(case=args.case, kind=args.kind, batch=B, tag=args.tag, bind=bind_kw,
                   skipped=f"xla temp {temp_gb:.1f} GiB > --max-temp-gb {args.max_temp_gb}",
                   t_compile_s=t_compile, xla_temp_mb=temp_gb * 1024,
                   xla_flops=cost.get("flops"), xla_bytes_accessed=cost.get("bytes accessed"))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        return

    # warm-up
    t0 = time.perf_counter()
    out = compiled(f, jnp.asarray(thetas[0]))
    out.block_until_ready()
    t_first = time.perf_counter() - t0

    times = []
    k = 0
    tstart = time.perf_counter()
    while True:
        th = jnp.asarray(thetas[k % 4])
        t0 = time.perf_counter()
        compiled(f, th).block_until_ready()
        times.append(time.perf_counter() - t0)
        k += 1
        if k >= args.max_reps:
            break
        if k >= args.min_reps and time.perf_counter() - tstart > args.min_time:
            break
    times = np.asarray(times)
    rec = dict(
        case=args.case, kind=args.kind, batch=B, tag=args.tag, bind=bind_kw,
        ndim=ndim, n_events=ll.n_events, nsamp=ll.nsamp,
        n_sel=int(ll.gw_selection.dL.shape[0]),
        catalog_shape=None if ll.catalog is None else list(ll.catalog.zgals.shape),
        t_build_s=t_build, t_lower_s=t_lower, t_compile_s=t_compile, t_first_call_s=t_first,
        reps=int(times.size), t_call_median_s=float(np.median(times)),
        t_call_min_s=float(times.min()), t_call_max_s=float(times.max()),
        t_per_point_median_ms=1e3 * float(np.median(times)) / B,
        t_per_point_min_ms=1e3 * float(times.min()) / B,
        peak_rss_mb=rss_mb(), rss_after_build_mb=rss_after_build,
        xla_temp_mb=mem.temp_size_in_bytes / 2**20 if mem else None,
        xla_arg_mb=mem.argument_size_in_bytes / 2**20 if mem else None,
        xla_out_mb=mem.output_size_in_bytes / 2**20 if mem else None,
        xla_flops=cost.get("flops"), xla_transcendentals=cost.get("transcendentals"),
        xla_bytes_accessed=cost.get("bytes accessed"),
        loadavg=os.getloadavg(), ncpu=os.cpu_count(),
        n_finite=int(np.isfinite(np.asarray(out)).sum()),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
