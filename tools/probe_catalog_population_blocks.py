#!/usr/bin/env python3
"""Fixed-theta parity probe for per-catalog population blocks of the field mixture.

``ds.model(catalog=[A, B, ...], catalog_sky_weighting="field",
per_catalog_population={k: [...]})`` against the frozen reference's
tracer-dependent population blocks: legacy commit af896ca ("likelihood:
support tracer-dependent population blocks", branch
``analysis8-marked-populations`` of the legacy repository, parent 2b86a2d),
the code the gws-agn Analyses 8-13 ran on. That commit is not on the pinned
reference c042527, so this probe is not part of a CI job: run its legacy arm
with an af896ca checkout first on ``PYTHONPATH``.

Run the two arms in separate Python processes: both expose the package name
``darksirens`` and must never share an interpreter.

The candidate arm goes through the public surface: ``ds.model(...,
per_catalog_population=...)`` and ``bind_analysis`` on in-memory full-sky
stores. The legacy arm calls ``darksiren_log_likelihood`` with
``catalog_sky_weighting="field"``, ``n_catalogs=K``, ``mixture_*`` and
``mixture_pop_params`` (the population vectors of catalogs 2..K, built
here from the base vector with the catalog's own entries replaced, as its
``ParameterDecoder.decode_mixture`` does), on catalogs carrying the field
inputs its loader builds (``build_field_normalization_inputs``,
``build_field_depth_inputs``). The two catalogs have different HEALPix
resolutions (nside 2 and 4). The legacy arm also reports its
``build_parameter_space`` labels for each cell, which ``--compare`` checks
against the candidate's.

The population is ``powerlaw+peak`` with every parameter not sampled fixed
at the registry fiducial and the merger-rate slope gamma pinned at
``GAMMA = 0`` in both arms (the gws-agn mocks use 0; core's
``Population(fixed=True)`` would use 2.5). The completeness is the per-row
count ratio (af896ca has no selection completeness). Legacy stores the
normaliser's observed density in float32 and core in float64;
``--legacy-f64-obs`` replaces legacy's table by the float64 one, as
``tools/probe_catalog_mixture.py`` does.

``--compare LEGACY CANDIDATE --rtol R`` checks two outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

LEGACY_SHA = "af896cae6f3f3dd1f87dec50046e3a8228f59b39"
POP_MODEL = "powerlaw+peak"
GAMMA = 0.0
N_EVENTS = 6
NSAMP = 16
N_SEL = 512
N_DRAW = 2.0 * N_SEL
MAX_VARIANCE = 1.0e3
OM0 = 0.3075
N_RANDOM = 24

#: (name, K, z_depth, base sampled population names, {catalog: names}).
#: "a11": Analysis 11 (base mu_G, mu_chi and catalog 2's copies sampled);
#: "a13": plus the shared widths sigma_G, sigma_chi (Analyses 12-13);
#: "a8": catalog 2's spin mean only, the whole base population fixed;
#: "shared": no blocks (the control: legacy's shared-population path).
CELLS = (
    ("shared", 2, None, ("G.mu", "mu_chi"), {}),
    ("a8", 2, None, (), {2: ("mu_chi",)}),
    ("a11", 2, None, ("G.mu", "mu_chi"), {2: ("G.mu", "mu_chi")}),
    ("a13", 2, None, ("G.mu", "G.sigma", "mu_chi", "sigma_chi"), {2: ("G.mu", "mu_chi")}),
    ("a11_depth", 2, 0.22, ("G.mu", "mu_chi"), {2: ("G.mu", "mu_chi")}),
    ("k3", 3, None, ("mu_chi",), {2: ("G.mu",), 3: ("G.mu", "mu_chi")}),
)

#: Draw ranges of the random points, by plain name (survey names unsuffixed).
RANGES = {
    "H0": (55.0, 85.0),
    "log10n0": (-2.8, -1.6),
    "delta": (-0.5, 0.5),
    "sigma_kde": (0.0, 0.008),
    "G.mu": (28.0, 42.0),
    "G.sigma": (2.0, 8.0),
    "mu_chi": (-0.15, 0.15),
    "sigma_chi": (0.05, 0.3),
}
#: The fiducial point's per-catalog values (an AGN-like branch).
FIDUCIAL_COPY = {"G.mu": 40.0, "mu_chi": 0.1}


def _catalog(seed, nside, z_depth):
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    n_max = 5
    ngals = rng.integers(1, n_max + 1, npix).astype(np.int32)
    ngals[rng.choice(npix, size=max(2, npix // 12), replace=False)] = 0
    zgals = np.full((npix, n_max), 100.0)
    dzgals = np.ones((npix, n_max))
    wgals = np.zeros((npix, n_max))
    z_hi = 0.28 if z_depth is None else 0.3
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.02, z_hi, n))
        dzgals[row, :n] = 0.01 * (1.0 + zgals[row, :n])
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    return dict(zgals=zgals, dzgals=dzgals, wgals=wgals, ngals=ngals,
                apix=np.pi / (3.0 * nside * nside), nside=nside)


def _samples():
    rng = np.random.default_rng(20261002)

    def draw(n, dl_lo, dl_hi, per_event):
        k = n // per_event
        rep = lambda a: np.repeat(a, per_event)  # noqa: E731
        m1 = rep(rng.uniform(25.0, 60.0, k)) * np.exp(rng.normal(0.0, 0.05, n))
        q = np.clip(rep(rng.uniform(0.6, 0.95, k)) + rng.normal(0.0, 0.05, n), 0.1, 1.0)
        ra = np.mod(rep(rng.uniform(0.0, 2.0 * np.pi, k)) + rng.normal(0.0, 0.08, n), 2.0 * np.pi)
        dec = np.clip(rep(np.arcsin(rng.uniform(-1.0, 1.0, k))) + rng.normal(0.0, 0.08, n), -1.5, 1.5)
        dL = rep(rng.uniform(dl_lo, dl_hi, k)) * np.exp(rng.normal(0.0, 0.12, n))
        chi = rep(rng.uniform(-0.1, 0.2, k)) + rng.normal(0.0, 0.05, n)
        return dict(m1det=m1, m2det=q * m1, dL=dL, chieff=chi, ra=ra, dec=dec)

    pe = draw(N_EVENTS * NSAMP, 300.0, 1100.0, NSAMP)
    pe["prior_wt"] = np.full(N_EVENTS * NSAMP, 1.0 / NSAMP)
    sel = draw(N_SEL, 200.0, 1400.0, 1)
    sel["prior_wt"] = np.ones(N_SEL)
    return pe, sel


def _cell_catalogs(K, z_depth):
    nsides = (2, 4, 2)[:K]
    return [_catalog(100 + k, nsides[k], z_depth) for k in range(K)]


def _suffix(k):
    return "" if k == 1 else f"_c{k}"


def _points(cell):
    """The cell's points: plain-name dicts (survey and copies suffixed ``_c{k}``)."""
    name, K, _depth, base_sampled, blocks = cell
    rng = np.random.default_rng(sum(map(ord, name)))
    fid = {"H0": 67.74, "G.mu": 35.0, "G.sigma": 5.0, "mu_chi": 0.0, "sigma_chi": 0.1}
    points = []
    fiducial = {"H0": fid["H0"]}
    for k in range(1, K + 1):
        s = _suffix(k)
        fiducial.update({f"log10n0{s}": -2.0 - 0.3 * (k - 1), f"delta{s}": 0.0, f"sigma_kde{s}": 0.0})
    for p in base_sampled:
        fiducial[p] = fid[p]
    for k, names in blocks.items():
        for p in names:
            fiducial[f"{p}_c{k}"] = FIDUCIAL_COPY[p]
    for m in range(2, K + 1):
        fiducial[f"fcat_{m}"] = 0.3
    points.append(fiducial)
    for i in range(N_RANDOM):
        point = {"H0": rng.uniform(*RANGES["H0"])}
        for k in range(1, K + 1):
            s = _suffix(k)
            for p in ("log10n0", "delta", "sigma_kde"):
                point[p + s] = rng.uniform(*RANGES[p])
        for p in base_sampled:
            point[p] = rng.uniform(*RANGES[p])
        for k, names in blocks.items():
            for p in names:
                point[f"{p}_c{k}"] = rng.uniform(*RANGES[p])
        for m in range(2, K + 1):
            # The boundary sticks 0 and 1 at two points; uniform otherwise.
            point[f"fcat_{m}"] = 0.0 if i == 0 else 1.0 if i == 1 else rng.uniform(0.0, 1.0)
        points.append(point)
    return points


def _base_population(get_fixed_population_params):
    theta = np.asarray(get_fixed_population_params(POP_MODEL), dtype=np.float64).copy()
    theta[-1] = GAMMA
    return theta


def _vectors(point, K, blocks, base_sampled, base, names):
    """Catalog 1..K population vectors at ``point`` (catalog k's copies, else catalog 1)."""
    v1 = base.copy()
    for p in base_sampled:
        v1[names.index(p)] = point[p]
    out = [v1]
    for k in range(2, K + 1):
        v = v1.copy()
        for p in blocks.get(k, ()):
            v[names.index(p)] = point[f"{p}_c{k}"]
        out.append(v)
    return out


def _candidate():
    import jax

    jax.config.update("jax_enable_x64", True)

    import darksirens as ds
    from darksirens.catalog.io import CatalogStore
    from darksirens.catalog.types import GalaxyCatalog
    from darksirens.gw.types import GWStore, SelectionStore
    from darksirens.population import get_fixed_population_params, get_model, pop_model_prior_parser
    from darksirens.runtime_binding import bind_analysis

    print("darksirens:", ds.__file__)
    pe, sel = _samples()
    fit = ("m1det", "q", "dL", "chieff")
    events = GWStore(
        format_version="probe", path="probe-pe", fit_columns=fit,
        columns={k: v for k, v in pe.items() if k != "prior_wt"}, attrs={},
        n_events=N_EVENTS, nsamp=NSAMP, prior_wt=pe["prior_wt"],
        event_names=tuple(f"e{i}" for i in range(N_EVENTS)),
    )
    injections = SelectionStore(
        format_version="probe", path="probe-sel", fit_columns=fit,
        columns={k: v for k, v in sel.items() if k != "prior_wt"}, attrs={},
        n_injections=N_SEL, ndraw=N_DRAW, prior_wt=sel["prior_wt"],
    )
    base = _base_population(get_fixed_population_params)
    labels = [str(x) for x in pop_model_prior_parser(POP_MODEL)[2]]
    names = [s.name for s in get_model(POP_MODEL).param_specs]
    plain_to_label = dict(zip(names, labels))
    rows, label_rows = {}, {}
    for cell in CELLS:
        name, K, z_depth, base_sampled, blocks = cell
        stores = [
            CatalogStore(
                path=f"probe-catalog-{k}", nside=c["nside"], z_depth=z_depth,
                catalog=GalaxyCatalog(
                    apix=c["apix"], zgals=c["zgals"], dzgals=c["dzgals"],
                    wgals=c["wgals"], ngals=c["ngals"], unique_pixels=None,
                ),
            )
            for k, c in enumerate(_cell_catalogs(K, z_depth))
        ]
        fixed = {lab: float(v) for lab, nm, v in zip(labels, names, base) if nm not in base_sampled}
        analysis = ds.model(
            cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=OM0),
            population=ds.Population(POP_MODEL, fixed=fixed),
            catalog=stores,
            catalog_sky_weighting="field",
            per_catalog_population={k: list(v) for k, v in blocks.items()} or None,
        )
        bound = bind_analysis(analysis, events=events, injections=injections,
                              max_likelihood_variance=MAX_VARIANCE)
        plan_labels = list(analysis.parameters.labels)
        label_rows[name] = plan_labels

        def plain(label):
            for p, lab in plain_to_label.items():
                if label == lab:
                    return p
                if label.startswith(lab + "_c"):
                    return p + label[len(lab):]
            return label

        values = []
        for point in _points(cell):
            theta = np.asarray([point[plain(lab)] for lab in plan_labels], dtype=np.float64)
            values.append(float(bound(theta)))
        rows[name] = values
    return rows, label_rows, base


def _legacy(f64_obs, only=None):
    import healpy as hp
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import darksirens
    from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
    from darksirens.gw.populations import get_model
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.parameters import _sticks_to_log_weights
    from darksirens.inference.prior import build_parameter_space, pop_model_prior_parser
    from darksirens.likelihood.core import darksiren_log_likelihood
    from darksirens.likelihood.events import make_gw_event
    from darksirens.redshift import zgrid
    from darksirens.redshift.completion import (
        _kde_dndz_obs,
        build_field_depth_inputs,
        build_field_normalization_inputs,
    )

    print("darksirens:", darksirens.__file__)
    pe, sel = _samples()
    base = _base_population(get_fixed_population_params)
    names = [s.name for s in get_model(POP_MODEL).param_specs]
    labels = [str(x) for x in pop_model_prior_parser(POP_MODEL)[2]]
    rows, label_rows = {}, {}
    for cell in CELLS:
        name, K, z_depth, base_sampled, blocks = cell
        if only and name not in only:
            continue
        cats = _cell_catalogs(K, z_depth)
        # The legacy labels of this configuration (Om0 and the unsampled
        # population fixed), for the --compare label check.
        fpv = {"Om0": OM0}
        fpv.update({lab: float(v) for lab, nm, v in zip(labels, names, base) if nm not in base_sampled})
        space = build_parameter_space(
            POP_MODEL, False, False, False, prior_overrides={}, fixed_parameter_values=fpv,
            fix_de=True, universe_model="dark_sirens", n_catalogs=K,
            lss_completion_active=[False] * K, use_lss=False,
            **(
                {"per_catalog_pop_params": tuple(f"{p}_c{k}" for k, v in blocks.items() for p in v)}
                if blocks else {}
            ),
        )
        label_rows[name] = [str(x) for x in space[0]]

        def event(cols):
            pix = [
                np.asarray(hp.ang2pix(c["nside"], np.pi / 2.0 - cols["dec"], cols["ra"]), dtype=np.int32)
                for c in cats
            ]
            return make_gw_event(
                m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"], chieff=cols["chieff"],
                prior_wt=cols["prior_wt"], pixels=np.stack(pix, axis=1),
            )

        gw_pe, gw_sel = event(pe), event(sel)
        em = []
        for c in cats:
            field = build_field_normalization_inputs(c["zgals"], c["wgals"], c["ngals"])
            obs = field.dN_obs_s
            if f64_obs:
                batch = jax.jit(jax.vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
                obs = jnp.asarray(batch(
                    jnp.asarray(field.occupied_pixels, dtype=jnp.int32), jnp.asarray(c["zgals"]),
                    jnp.asarray(c["wgals"]), jnp.asarray(c["ngals"]),
                ), dtype=jnp.float64)
            extra = dict(
                field_dN_obs_s=obs, field_n_empty=field.n_empty,
                field_N_obs_total=field.N_obs_total,
                field_occupied_pixels=field.occupied_pixels,
            )
            if z_depth is not None:
                depth = build_field_depth_inputs(c["zgals"], c["dzgals"], c["wgals"], c["ngals"])
                extra.update(field_depth_z=depth.z, field_depth_dz=depth.dz, field_depth_c=depth.c)
            em.append(EMCatalog(
                apix=c["apix"], zgals=jnp.asarray(c["zgals"]), dzgals=jnp.asarray(c["dzgals"]),
                wgals=jnp.asarray(c["wgals"]), ngals=jnp.asarray(c["ngals"]),
                delta_g_pix_z=jnp.zeros((1, int(zgrid.shape[0]))), dN_obs_kde=None,
                pixel_to_cache_idx=None, unique_pixels=None, **extra,
            ))
        values = []
        for point in _points(cell):
            surveys = []
            for k in range(1, K + 1):
                s = _suffix(k)
                surveys.append(SurveyParams(
                    n0=10.0 ** point[f"log10n0{s}"], z50=1.0, w=0.5, delta=point[f"delta{s}"],
                    b_miss=1.0, alpha_miss=1.0, sigma_kde=point[f"sigma_kde{s}"], z_depth=z_depth,
                ))
            vectors = _vectors(point, K, blocks, base_sampled, base, names)
            sticks = jnp.asarray([point[f"fcat_{m}"] for m in range(2, K + 1)])
            # Shared-population cells pass no mixture_pop_params at all, so
            # they also run on a reference without the per-catalog blocks.
            pop_kw = (
                dict(mixture_pop_params=tuple(jnp.asarray(v) for v in vectors[1:]))
                if blocks else {}
            )
            values.append(float(darksiren_log_likelihood(
                CosmoParams(H0=point["H0"], Om0=OM0, w0=-1.0, wa=0.0),
                surveys[0], jnp.asarray(vectors[0]), gw_pe, em[0], gw_sel, em[0],
                N_EVENTS, NSAMP, N_DRAW, POP_MODEL, "dark_sirens",
                max_likelihood_variance=MAX_VARIANCE,
                catalog_sky_weighting="field",
                n_catalogs=K,
                mixture_surveys=tuple(surveys[1:]),
                mixture_em_catalogs_pe=tuple(em[1:]),
                mixture_em_catalogs_sel=tuple(em[1:]),
                mixture_log_weights=_sticks_to_log_weights(sticks),
                **pop_kw,
            )))
        rows[name] = values
    return rows, label_rows, base


def _compare(legacy_path, candidate_path, rtol):
    legacy = json.loads(Path(legacy_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    assert legacy["population"] == candidate["population"], "population vectors differ"
    worst = 0.0
    n_points = 0
    for name, values in legacy["results"].items():
        if name not in candidate["results"]:
            continue
        if legacy["labels"][name] != candidate["labels"][name]:
            raise AssertionError(
                f"{name}: labels differ: legacy {legacy['labels'][name]} "
                f"candidate {candidate['labels'][name]}"
            )
        got = candidate["results"][name]
        cell_worst = 0.0
        for a, b in zip(values, got):
            if not (math.isfinite(a) and math.isfinite(b)):
                if a == b:
                    continue
                raise AssertionError(f"{name}: non-finite {a} vs {b}")
            diff = abs(a - b)
            cell_worst = max(cell_worst, diff)
            if diff > rtol * max(1.0, abs(a)):
                raise AssertionError(f"{name}: legacy={a!r} candidate={b!r} |d|={diff:.3e}")
        n_points += len(values)
        worst = max(worst, cell_worst)
        print(f"{name:10s} {len(values):3d} points  labels equal  max |dlogL| = {cell_worst:.2e}")
    print(f"{n_points} points: max |dlogL| = {worst:.3e} (rtol {rtol:.1e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"))
    parser.add_argument("--out")
    parser.add_argument("--legacy-f64-obs", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("LEGACY", "CANDIDATE"))
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--only", nargs="*", default=None,
                        help="legacy arm: run only these cells (e.g. 'shared' on a reference "
                        "without per-catalog blocks)")
    args = parser.parse_args()
    if args.compare:
        _compare(*args.compare, args.rtol)
        return
    if args.implementation == "legacy":
        rows, label_rows, population = _legacy(args.legacy_f64_obs, args.only)
    else:
        rows, label_rows, population = _candidate()
    payload = {
        "implementation": args.implementation,
        "legacy_sha": LEGACY_SHA,
        "pop_model": POP_MODEL,
        "gamma": GAMMA,
        "legacy_f64_obs": bool(args.legacy_f64_obs),
        "population": [float(x) for x in population],
        "labels": label_rows,
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
