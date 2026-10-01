"""Prototype (monkeypatch, NOT library code): two pairing-normaliser variants.

  analytic_scale : PowerLawPairing taper panel factored by an ANALYTIC upper bound
                   max(q_cut**beta, 1) instead of the node maximum, so the node sum is
                   one reduction (XLA can fuse the producer; no (N x 32) buffer).
                   Same quadrature rule; differs from default only by rounding.
  shared_nodes   : TIMING BOUND ONLY. Every sample uses the m1-independent m2-node
                   integral m1**(-beta-1) * sum_k w_k m2_k**beta S(m2_k) dm, which is the
                   exact same GL rule for samples with m1 >= m_min + dm_min but WRONG for
                   samples inside the primary taper. Measures what an O(1)-per-sample
                   normaliser would cost; its logL is not a candidate.

Usage: run.sh proto_pairing.py CASE KIND VARIANT [--n 48] [--batch 1]
Appends to $OUT/stage1_proto/proto.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from _build import OUT, build, load_stores, make_analysis, prior_draws, rss_mb


def patch(variant):
    import jax.numpy as jnp
    from darksirens.population.parametric import PowerLawPairing
    from darksirens.population.utils import sfilter_low

    orig = PowerLawPairing._panel_norm

    def analytic_scale(self, m1, m_min, dm_min, theta, t, w):
        edges = self._panel_boundaries(m1, m_min, dm_min, theta)
        closed = self._plateau_integral(m1, edges[-2], m_min, dm_min, theta)
        assert closed is not None and len(edges) == 3
        beta = theta[0]
        q_cut = edges[0]
        safe_q = jnp.where(q_cut > 0.0, q_cut, 1.0)
        bound = jnp.maximum(jnp.where(q_cut > 0.0, safe_q**beta, 1.0), 1.0)
        scale = jnp.maximum(bound, closed[1])
        scale_s = jnp.where(scale > 0, scale, 1.0)
        nodes, width = self._panel_from_edges(edges[0], edges[1], t)
        p0 = self._eval_unnorm(m1[..., None], nodes, m_min, dm_min, theta)
        n_sc = jnp.sum(w * (p0 / scale_s[..., None]), axis=-1) * width
        n_sc = n_sc + closed[0] / scale_s
        return n_sc, scale_s

    def shared_nodes(self, m1, m_min, dm_min, theta, t, w):
        edges = self._panel_boundaries(m1, m_min, dm_min, theta)
        closed = self._plateau_integral(m1, edges[-2], m_min, dm_min, theta)
        beta = theta[0]
        m_edge, m_sh = self._taper_shoulder(m_min, dm_min, theta)
        m2 = m_edge + t * (m_sh - m_edge)                 # (K,) proposal-level nodes
        G = jnp.sum(w * m2**beta * sfilter_low(m2, m_min, dm_min)) * (m_sh - m_edge)
        taper = G * m1 ** (-beta - 1.0)
        n = taper + closed[0]
        return n, jnp.ones_like(n)

    PowerLawPairing._panel_norm = {"analytic_scale": analytic_scale,
                                   "shared_nodes": shared_nodes}[variant]
    return orig


def run(ll, th, batch, label):
    import jax
    import jax.numpy as jnp
    f = ll.as_pytree_callable()
    g = jax.jit(jax.vmap(lambda g, t: g(t), in_axes=(None, 0)))
    tb = jnp.asarray(th[:batch])
    comp = g.lower(f, tb).compile()
    mem = comp.memory_analysis()
    cost = comp.cost_analysis()
    cost = (cost[0] if isinstance(cost, list) else cost) or {}
    comp(f, tb).block_until_ready()
    times = []
    t0 = time.perf_counter()
    while len(times) < 3 or time.perf_counter() - t0 < 6.0:
        t1 = time.perf_counter()
        comp(f, tb).block_until_ready()
        times.append(time.perf_counter() - t1)
        if len(times) >= 20:
            break
    vals = np.concatenate([np.asarray(g(f, jnp.asarray(th[i:i + batch])))
                           for i in range(0, len(th), batch)])
    return dict(label=label, t_call_median_s=float(np.median(times)),
                xla_temp_mb=mem.temp_size_in_bytes / 2**20,
                xla_flops=cost.get("flops"), xla_bytes_accessed=cost.get("bytes accessed")), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("kind")
    ap.add_argument("variant", choices=["analytic_scale", "shared_nodes"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    analysis = make_analysis(args.case, args.kind)
    ev, inj = load_stores(args.case, analysis)
    _, ll0, ptform, _, _ = build(args.case, args.kind, events=ev, injections=inj, analysis=analysis)
    th = prior_draws(ptform, len(analysis.parameters.labels), args.n, seed=11)
    r0, v0 = run(ll0, th, args.batch, "default")
    patch(args.variant)
    _, ll1, _, _, _ = build(args.case, args.kind, events=ev, injections=inj, analysis=analysis)
    r1, v1 = run(ll1, th, args.batch, args.variant)
    both = np.isfinite(v0) & np.isfinite(v1)
    d = np.abs(v1[both] - v0[both])
    rec = dict(case=args.case, kind=args.kind, variant=args.variant, batch=args.batch,
               default=r0, patched=r1, n=int(v0.size), n_both_finite=int(both.sum()),
               n_finite_mismatch=int((np.isfinite(v0) != np.isfinite(v1)).sum()),
               dlogL_median_abs=float(np.median(d)) if d.size else None,
               dlogL_max_abs=float(d.max()) if d.size else None,
               speedup=r0["t_call_median_s"] / r1["t_call_median_s"],
               peak_rss_mb=rss_mb(), loadavg=os.getloadavg())
    os.makedirs(f"{OUT}/stage1_proto", exist_ok=True)
    with open(f"{OUT}/stage1_proto/proto.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
