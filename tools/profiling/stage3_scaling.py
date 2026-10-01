"""Stage 3: cost/memory scaling with n_events, nsamp, n_injections, galaxies per row.

Inputs are subsampled or tiled IN MEMORY (the files are read-only).
Usage: run.sh stage3_scaling.py CASE KIND [--events F] [--nsamp K] [--inj F] [--gal K]
         [--batch B] [--compute-dtype float32]
  --events F : keep the first round(F*n_events) events (F<=1) or tile events F times (F>1, int)
  --nsamp K  : keep the first K PE samples of every event
  --inj F    : random subsample (F<1) or tile (F>1, int) of the detected injections; ndraw scaled by F
  --gal K    : replicate every catalog galaxy K times inside its row (z jittered by 1e-5*j)
Appends one JSON line to $OUT/stage3_scaling/scaling.jsonl.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np

from _build import OUT, build, cur_rss_mb, load_stores, prior_draws, rss_mb


def scale_events(store, frac, nsamp_keep):
    n, s = store.n_events, store.nsamp
    if frac <= 1.0:
        idx_ev = np.arange(max(1, int(round(frac * n))))
    else:
        idx_ev = np.tile(np.arange(n), int(frac))
    k = s if nsamp_keep is None else min(int(nsamp_keep), s)
    rows = (idx_ev[:, None] * s + np.arange(k)[None, :]).reshape(-1)
    cols = {name: np.asarray(v)[rows] for name, v in store.columns.items()}
    names = None
    if store.event_names is not None:
        names = tuple(f"{store.event_names[i]}#{j}" for j, i in enumerate(idx_ev))
    return dataclasses.replace(store, columns=cols, prior_wt=np.asarray(store.prior_wt)[rows],
                               n_events=int(idx_ev.size), nsamp=int(k), event_names=names)


def scale_injections(store, frac, seed=0):
    n = int(np.asarray(store.prior_wt).shape[0])
    if frac < 1.0:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n, size=int(round(frac * n)), replace=False))
    elif frac > 1.0:
        rows = np.tile(np.arange(n), int(frac))
    else:
        return store
    cols = {name: np.asarray(v)[rows] for name, v in store.columns.items()}
    return dataclasses.replace(store, columns=cols, prior_wt=np.asarray(store.prior_wt)[rows],
                               n_injections=int(rows.size),
                               ndraw=int(round(store.ndraw * rows.size / n)))


def scale_catalog(cat_store, k):
    if k == 1:
        return cat_store
    c = cat_store.catalog
    z, dz, w, ng = (np.asarray(a) for a in (c.zgals, c.dzgals, c.wgals, c.ngals))
    R, M = z.shape
    real = np.arange(M)[None, :] < ng[:, None]
    zn = np.zeros((R, M * k)); dzn = np.zeros((R, M * k)); wn = np.zeros((R, M * k))
    for r in range(R):
        m = int(ng[r])
        if m == 0:
            continue
        zz = np.concatenate([z[r, :m] + 1e-5 * j for j in range(k)])
        order = np.argsort(zz, kind="stable")
        zn[r, : m * k] = zz[order]
        dzn[r, : m * k] = np.tile(dz[r, :m], k)[order]
        wn[r, : m * k] = np.tile(w[r, :m], k)[order]
    new = c._replace(zgals=zn, dzgals=dzn, wgals=wn, ngals=(ng * k).astype(ng.dtype))
    del real
    return dataclasses.replace(cat_store, catalog=new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("kind")
    ap.add_argument("--events", type=float, default=1.0)
    ap.add_argument("--nsamp", type=int, default=None)
    ap.add_argument("--inj", type=float, default=1.0)
    ap.add_argument("--gal", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--compute-dtype", default=None)
    ap.add_argument("--min-time", type=float, default=6.0)
    ap.add_argument("--out", default=f"{OUT}/stage3_scaling/scaling.jsonl")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import darksirens as ds
    from _build import make_analysis

    analysis = make_analysis(args.case, args.kind)
    if args.gal != 1:
        cat = scale_catalog(analysis.catalog, args.gal)
        cosmology = ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075, w0=-1.0, wa=0.0)
        population = ds.Population("powerlaw+peak", shared_beta=True, shared_spin=True,
                                   shared_gamma=True)
        analysis = ds.model(cosmology=cosmology, population=population, catalog=cat)
    events, injections = load_stores(args.case, analysis)
    events = scale_events(events, args.events, args.nsamp)
    injections = scale_injections(injections, args.inj)
    kw = {} if args.compute_dtype is None else dict(compute_dtype=args.compute_dtype)
    t0 = time.perf_counter()
    analysis, ll, ptform, _, _ = build(args.case, args.kind, events=events,
                                       injections=injections, analysis=analysis, **kw)
    t_bind = time.perf_counter() - t0
    f = ll.as_pytree_callable()
    ndim = len(analysis.parameters.labels)
    th = jnp.asarray(prior_draws(ptform, ndim, args.batch, seed=99))
    g = jax.jit(jax.vmap(lambda g, t: g(t), in_axes=(None, 0)))
    t0 = time.perf_counter()
    comp = g.lower(f, th).compile()
    t_compile = time.perf_counter() - t0
    mem = comp.memory_analysis()
    cost = comp.cost_analysis()
    cost = (cost[0] if isinstance(cost, list) else cost) or {}
    comp(f, th).block_until_ready()
    times = []
    t_start = time.perf_counter()
    while len(times) < 3 or time.perf_counter() - t_start < args.min_time:
        t0 = time.perf_counter()
        comp(f, th).block_until_ready()
        times.append(time.perf_counter() - t0)
        if len(times) >= 30:
            break
    cat_shape = None if ll.catalog is None else list(ll.catalog.zgals.shape)
    rec = dict(case=args.case, kind=args.kind, events_arg=args.events, nsamp_arg=args.nsamp,
               inj_arg=args.inj, gal_arg=args.gal, batch=args.batch,
               compute_dtype=args.compute_dtype,
               n_events=ll.n_events, nsamp=ll.nsamp, n_pe=ll.n_events * ll.nsamp,
               n_sel=int(ll.gw_selection.dL.shape[0]), catalog_shape=cat_shape,
               n_galaxies=None if ll.catalog is None else int(np.asarray(ll.catalog.ngals).sum()),
               t_bind_s=t_bind, t_compile_s=t_compile, reps=len(times),
               t_call_median_s=float(np.median(times)), t_call_min_s=float(np.min(times)),
               xla_temp_mb=mem.temp_size_in_bytes / 2**20, xla_arg_mb=mem.argument_size_in_bytes / 2**20,
               xla_flops=cost.get("flops"), xla_bytes_accessed=cost.get("bytes accessed"),
               peak_rss_mb=rss_mb(), loadavg=os.getloadavg())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
