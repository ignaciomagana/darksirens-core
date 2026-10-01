"""Stage 2: float32 per-sample weights vs the float64 default.

Usage: run.sh stage2_precision.py CASE KIND [--n-prior 500] [--n-bulk 200]
Writes $OUT/stage2_precision/<case>_<kind>.{npz,json}.

Prior draws: the sampler's prior transform on uniform cubes.
Bulk: float64 maximum likelihood (L-BFGS-B from the best prior draws, jax
gradients), Laplace covariance from the float64 Hessian there, then draws
N(MAP, cov) at 1x (n_bulk) and 2x (n_bulk/2) scale, clipped to the prior box.
The Laplace draws stand in for posterior samples (no 13-D/16-D posterior of
these exact analyses exists on disk); the logZ / posterior-shift numbers are
importance-reweighting ESTIMATES over those draws.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from _build import OUT, build, diagnostics_fn, load_stores, make_analysis, prior_draws

FIELDS = ("log_likelihood", "log_mu", "n_eff", "selection_log_correction")


def evaluate(diag, thetas, batch):
    import jax.numpy as jnp
    out = {k: [] for k in FIELDS + ("sum_event_ll", "pe_var_sum", "n_inf_events")}
    for i in range(0, len(thetas), batch):
        chunk = thetas[i:i + batch]
        pad = batch - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
        r = diag(jnp.asarray(chunk))
        n = batch - pad
        for k in FIELDS:
            out[k].append(np.asarray(getattr(r, k))[:n])
        ev = np.asarray(r.event_log_evidence)[:n]
        out["sum_event_ll"].append(ev.sum(axis=1))
        out["n_inf_events"].append((~np.isfinite(ev)).sum(axis=1))
        out["pe_var_sum"].append(np.asarray(r.event_mc_variance)[:n].sum(axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}


def stats(a, b, rel=False):
    """Differences b - a over entries finite in both."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = b[both] - a[both]
    if rel:
        d = d / np.abs(a[both])
    ad = np.abs(d)
    res = dict(n=int(a.size), n_both_finite=int(both.sum()),
               n_finite_mismatch=int((fa != fb).sum()))
    if ad.size:
        res.update(mean=float(d.mean()), std=float(d.std()), median_abs=float(np.median(ad)),
                   p95_abs=float(np.quantile(ad, 0.95)), p99_abs=float(np.quantile(ad, 0.99)),
                   max_abs=float(ad.max()))
    return res


def compare(r64, r32, n_events):
    var64 = r64["pe_var_sum"] + n_events**2 / r64["n_eff"]
    var32 = r32["pe_var_sum"] + n_events**2 / r32["n_eff"]
    return dict(
        dlogL_abs=stats(r64["log_likelihood"], r32["log_likelihood"]),
        dlogL_rel=stats(r64["log_likelihood"], r32["log_likelihood"], rel=True),
        dlog_mu=stats(r64["log_mu"], r32["log_mu"]),
        dn_eff_rel=stats(r64["n_eff"], r32["n_eff"], rel=True),
        dselection_ll=stats(r64["selection_log_correction"], r32["selection_log_correction"]),
        dsum_event_ll=stats(r64["sum_event_ll"], r32["sum_event_ll"]),
        dpe_var_sum=stats(r64["pe_var_sum"], r32["pe_var_sum"]),
        dtotal_lnL_variance=stats(var64, var32),
        n_inf_events_differ=int((r64["n_inf_events"] != r32["n_inf_events"]).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("kind")
    ap.add_argument("--n-prior", type=int, default=500)
    ap.add_argument("--n-bulk", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--n-starts", type=int, default=3)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    t_start = time.time()
    analysis = make_analysis(args.case, args.kind)
    events, injections = load_stores(args.case, analysis)
    _, b64, ptform, _, _ = build(args.case, args.kind, events=events, injections=injections,
                                 analysis=analysis)
    _, b32, _, _, _ = build(args.case, args.kind, events=events, injections=injections,
                            analysis=analysis, compute_dtype="float32")
    p = analysis.parameters
    ndim = len(p.labels)
    lo, hi = np.asarray(p.lower, float), np.asarray(p.upper, float)
    d64 = diagnostics_fn(b64, batched=True)
    d32 = diagnostics_fn(b32, batched=True)

    # --- prior draws
    th_prior = prior_draws(ptform, ndim, args.n_prior, seed=2026)
    t0 = time.time()
    rp64 = evaluate(d64, th_prior, args.batch)
    t_p64 = time.time() - t0
    t0 = time.time()
    rp32 = evaluate(d32, th_prior, args.batch)
    t_p32 = time.time() - t0
    prior_cmp = compare(rp64, rp32, b64.n_events)
    print("prior:", json.dumps(prior_cmp["dlogL_abs"]), flush=True)

    # --- float64 maximum likelihood
    f64 = b64.as_pytree_callable()
    vg = jax.jit(jax.value_and_grad(lambda g, t: g(t), argnums=1))

    span = hi - lo
    state = {"f": None, "g": None}

    def negll_u(u):
        x = lo + span * np.clip(u, 0.0, 1.0)
        v, g = vg(f64, jnp.asarray(x))
        v, g = float(v), np.asarray(g, float) * span
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            # push the line search back without poisoning the L-BFGS memory
            f0 = state["f"] if state["f"] is not None else 1e30
            return f0 + 1e6, (state["g"] if state["g"] is not None else np.zeros_like(u))
        state["f"], state["g"] = -v, -g
        return -v, -g

    ll_prior = rp64["log_likelihood"]
    order = np.argsort(-np.where(np.isfinite(ll_prior), ll_prior, -np.inf))
    best = None
    opt_log = []
    for j in order[: args.n_starts]:
        u0 = (th_prior[j] - lo) / span
        state["f"] = state["g"] = None
        res = minimize(negll_u, u0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * ndim, options=dict(maxiter=args.maxiter))
        opt_log.append(dict(start_logL=float(ll_prior[j]), end_logL=float(-res.fun),
                            nit=int(res.nit), nfev=int(res.nfev), message=str(res.message)))
        print("opt:", opt_log[-1], flush=True)
        if best is None or res.fun < best.fun:
            best = res
    u_map = np.clip(np.asarray(best.x), 0.0, 1.0)
    x_map = lo + span * u_map
    ll_map = float(vg(f64, jnp.asarray(x_map))[0])

    # --- Laplace covariance: central finite differences of the (finite) jax gradient
    # in unit coordinates (forward-over-reverse jax.hessian is NaN for this likelihood).
    def grad_u(u):
        x = lo + span * u
        return np.asarray(vg(f64, jnp.asarray(x))[1], float) * span

    h = 1e-4
    H = np.zeros((ndim, ndim))
    for k in range(ndim):
        e = np.zeros(ndim)
        up = min(u_map[k] + h, 1.0)
        dn = max(u_map[k] - h, 0.0)
        e[k] = up - u_map[k]
        gp = grad_u(u_map + e)
        e[k] = dn - u_map[k]
        gm = grad_u(u_map + e)
        H[:, k] = (gp - gm) / (up - dn)
    H = -0.5 * (H + H.T)                     # curvature of -logL (unit coords)
    H[~np.isfinite(H)] = 0.0
    w, V = np.linalg.eigh(H)
    # non-positive or tiny curvature directions (a parameter at a bound or weakly
    # constrained): cap the width at 10% of the prior span in that direction
    w_floor = 1.0 / 0.1**2
    n_bad = int((w < w_floor).sum())
    w_c = np.maximum(w, w_floor)
    cov_u = (V / w_c) @ V.T
    cov = cov_u * np.outer(span, span)
    rng = np.random.default_rng(31)
    n1, n2 = args.n_bulk, args.n_bulk // 2
    L = np.linalg.cholesky(cov + 1e-300 * np.eye(ndim))
    z1 = rng.standard_normal((n1, ndim)) @ L.T
    z2 = 2.0 * rng.standard_normal((n2, ndim)) @ L.T
    th_bulk = np.clip(np.concatenate([x_map + z1, x_map + z2]), lo, hi)
    scale_tag = np.concatenate([np.ones(n1), 2 * np.ones(n2)])
    rb64 = evaluate(d64, th_bulk, args.batch)
    rb32 = evaluate(d32, th_bulk, args.batch)
    bulk1 = scale_tag == 1
    cmp_b1 = compare({k: v[bulk1] for k, v in rb64.items()},
                     {k: v[bulk1] for k, v in rb32.items()}, b64.n_events)
    cmp_b2 = compare({k: v[~bulk1] for k, v in rb64.items()},
                     {k: v[~bulk1] for k, v in rb32.items()}, b64.n_events)
    print("bulk 1x:", json.dumps(cmp_b1["dlogL_abs"]), flush=True)

    # --- logZ / posterior shift ESTIMATES: importance weights over the 1x Laplace draws,
    # themselves re-weighted to the float64 likelihood (Laplace proposal -> float64 posterior).
    l64 = rb64["log_likelihood"][bulk1]
    l32 = rb32["log_likelihood"][bulk1]
    zz = z1
    log_q = -0.5 * np.einsum("ij,jk,ik->i", zz, np.linalg.inv(cov), zz)
    ok = np.isfinite(l64) & np.isfinite(l32)
    lw64 = np.where(ok, l64 - log_q, -np.inf)
    lw32 = np.where(ok, l32 - log_q, -np.inf)

    def lse(a):
        m = np.max(a)
        return m + np.log(np.sum(np.exp(a - m)))

    dlogZ = float(lse(lw32) - lse(lw64))
    W64 = np.exp(lw64 - lse(lw64))
    W32 = np.exp(lw32 - lse(lw32))
    ess64 = float(1.0 / np.sum(W64**2))
    xb = th_bulk[bulk1]
    mu64 = W64 @ xb
    sd64 = np.sqrt(np.maximum(W64 @ (xb - mu64) ** 2, 1e-300))
    mu32 = W32 @ xb
    shift_sigma = ((mu32 - mu64) / sd64).tolist()
    # direct: posterior-weighted mean of dlogL and its spread (what moves the posterior)
    d = (l32 - l64)[ok]
    wpost = W64[ok] / W64[ok].sum()
    post_dlogL_mean = float(wpost @ d)
    post_dlogL_sd = float(np.sqrt(wpost @ (d - post_dlogL_mean) ** 2))

    summary = dict(
        case=args.case, kind=args.kind, labels=list(p.labels), n_events=b64.n_events,
        nsamp=b64.nsamp, n_sel=int(b64.gw_selection.dL.shape[0]),
        prior=prior_cmp,
        prior_eval_s=dict(float64=t_p64, float32=t_p32, n=args.n_prior, batch=args.batch),
        map=dict(theta=x_map.tolist(), logL64=ll_map, optimizer=opt_log,
                 laplace_sd=np.sqrt(np.diag(cov)).tolist(), n_floored_directions=n_bad),
        bulk_1x=cmp_b1, bulk_2x=cmp_b2,
        bulk_logL64_range=[float(np.nanmin(np.where(np.isfinite(rb64['log_likelihood']), rb64['log_likelihood'], np.nan))),
                           float(np.nanmax(rb64["log_likelihood"]))],
        estimates=dict(
            note="importance-reweighting estimates over Laplace draws (proxy posterior)",
            dlogZ_float32_minus_float64=dlogZ, ess_float64_weights=ess64,
            posterior_weighted_dlogL_mean=post_dlogL_mean,
            posterior_weighted_dlogL_sd=post_dlogL_sd,
            param_mean_shift_in_sigma=dict(zip(p.labels, shift_sigma)),
        ),
        wall_s=time.time() - t_start,
    )
    d = f"{OUT}/stage2_precision"
    os.makedirs(d, exist_ok=True)
    stem = f"{d}/{args.case}_{args.kind}"
    with open(stem + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    np.savez_compressed(stem + ".npz", th_prior=th_prior, th_bulk=th_bulk, scale_tag=scale_tag,
                        x_map=x_map, cov=cov,
                        **{f"prior64_{k}": v for k, v in rp64.items()},
                        **{f"prior32_{k}": v for k, v in rp32.items()},
                        **{f"bulk64_{k}": v for k, v in rb64.items()},
                        **{f"bulk32_{k}": v for k, v in rb32.items()})
    print(json.dumps(summary["bulk_1x"], indent=1))
    print(json.dumps(summary["estimates"], indent=1))


if __name__ == "__main__":
    main()
