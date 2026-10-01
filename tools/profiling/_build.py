"""Shared builders for the likelihood profiling experiment (CPU only).

Mirrors exactly what ds.infer hands a sampler. Inputs are read-only.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

DATA = "/hildafs/projects/phy220048p/magana/darksirens-core-data"
BENCH = f"{DATA}/darksirens_benchmark_local"
OUT = f"{DATA}/likelihood_profiling_2026-09-30"
REAL_PE = f"{BENCH}/gwcat/exports/A_pe_chieff_bbh259_n4096_v20.h5"
REAL_SEL = f"{BENCH}/gwcat/exports/A_sel_chieffref_o3o4ab_v20.h5"
FIX = f"{BENCH}/mock/fixtures"
CATALOGS = {"T": "catalog_pixelated_nside_16.h5", "S": "catalog_pixelated_nside_32.h5",
            "R1": "catalog_pixelated_nside_64.h5"}


def paths(case: str):
    """case: 'real', or '<fixture>' (T, R1, ...)."""
    if case == "real":
        return REAL_PE, REAL_SEL, None
    d = f"{FIX}/{case}"
    cat = f"{d}/{CATALOGS[case]}" if case in CATALOGS else None
    return f"{d}/mock_gw_events.h5", f"{d}/mock_gw_selection.h5", cat


def make_analysis(case: str, kind: str = "spectral"):
    import darksirens as ds

    _, _, catpath = paths(case)
    cosmology = ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075, w0=-1.0, wa=0.0)
    population = ds.Population("powerlaw+peak", shared_beta=True, shared_spin=True,
                               shared_gamma=True)
    if kind == "spectral":
        return ds.model(cosmology=cosmology, population=population)
    if kind == "dark":
        catalog = ds.load_catalog(catpath)
        return ds.model(cosmology=cosmology, population=population, catalog=catalog)
    raise ValueError(kind)


def load_stores(case: str, analysis):
    import darksirens as ds
    from darksirens.runtime_binding import required_fit_columns

    pe, sel, _ = paths(case)
    cols = required_fit_columns(analysis)
    return ds.load_events(pe, fit_columns=cols), ds.load_injections(sel, fit_columns=cols)


def build(case: str, kind: str = "spectral", *, events=None, injections=None,
          analysis=None, **bind_kw):
    """Return (analysis, loglike, ptform, events, injections)."""
    from darksirens.inference.prior import make_prior_transform
    from darksirens.runtime_binding import bind_analysis

    if analysis is None:
        analysis = make_analysis(case, kind)
    if events is None or injections is None:
        ev, inj = load_stores(case, analysis)
        events = ev if events is None else events
        injections = inj if injections is None else injections
    kw = dict(selection_neff_soft_guard=True, max_likelihood_variance=20.0)
    kw.update(bind_kw)
    loglike = bind_analysis(analysis, events=events, injections=injections, **kw)
    p = analysis.parameters
    ptform = make_prior_transform(p.lower, p.upper, prior_kinds=p.prior_kinds,
                                  joint_constraints=p.joint_constraints)
    return analysis, loglike, ptform, events, injections


def prior_draws(ptform, ndim, n, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n, ndim))
    return np.stack([np.asarray(ptform(ui)) for ui in u])


def rss_mb():
    """Peak RSS of this process so far (MB)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cur_rss_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def diagnostics_fn(bound, batched=False):
    """jit(theta -> Diagnostics NamedTuple) replicating BoundAnalysis._evaluate
    with return_diagnostics=True (spectral and incomplete-catalog only)."""
    import jax
    from darksirens.analysis import IncompleteCatalogRedshift, SpectralRedshift
    from darksirens.likelihood.hierarchical import (
        dark_siren_log_likelihood, spectral_siren_log_likelihood)
    from darksirens.runtime_binding import _decode_theta

    a = bound.analysis
    pop = a.population
    common = dict(pop_model=pop.model_name, shared_beta=pop.shared_beta,
                  shared_spin=pop.shared_spin, shared_gamma=pop.shared_gamma,
                  sel_batch_size=bound.sel_batch_size,
                  selection_neff_soft_guard=bound.selection_neff_soft_guard,
                  max_likelihood_variance=bound.max_likelihood_variance,
                  pe_event_block=bound.pe_event_block,
                  angular_model=a.angular_model, return_diagnostics=True)
    extra = {}
    if getattr(bound, "compute_dtype", None) is not None:
        extra["compute_dtype"] = bound.compute_dtype

    def f(theta, gw_pe, gw_sel, catalog, cache, pin):
        cosmo, popp, cat, ang = _decode_theta(a, theta, z_depth=bound.z_depth)
        if isinstance(a.redshift, SpectralRedshift):
            return spectral_siren_log_likelihood(
                cosmo, popp, gw_pe, gw_sel, bound.n_events, bound.nsamp, bound.n_draw,
                angular_params=ang, **common, **extra)
        if isinstance(a.redshift, IncompleteCatalogRedshift):
            return dark_siren_log_likelihood(
                cosmo, cat, popp, gw_pe, catalog, cache, gw_sel, catalog, cache,
                bound.n_events, bound.nsamp, bound.n_draw,
                pinned_kernel_pe=pin, pinned_kernel_sel=pin, angular_params=ang,
                **common, **extra)
        raise TypeError(type(a.redshift))

    jf = jax.jit(jax.vmap(f, in_axes=(0, None, None, None, None, None)) if batched else f)
    return lambda theta: jf(theta, bound.gw_pe, bound.gw_selection, bound.catalog,
                            bound.observed_density_cache, bound.kernel_pin)
