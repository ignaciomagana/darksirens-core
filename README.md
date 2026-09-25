# darksirens-core

`darksirens-core` is the reusable hierarchical population and cosmological
inference kernel for gravitational-wave sirens. The Python distribution and
import package are both named `darksirens`.

The reconstruction is parity-gated against the frozen legacy reference
`ignaciomagana/darksirens@c042527238bd71421b792936bc48c3b815b90d6d`.
Architectural decisions and exact acceptance records live in
`ignaciomagana/darksirens-rebuild`.

## Install

Python 3.11 is the validated interpreter. The base install pins the numerical
stack used for reconstruction parity and includes TinyNS, the public default
sampler:

```bash
pip install .
```

Optional integrations are explicit:

```bash
pip install ".[gp]"       # tinygp/equinox population models
pip install ".[dynesty]"  # Dynesty backend
pip install ".[numpyro]"  # NumPyro backend
pip install ".[gwcat]"    # canonical chi_eff prior support when required by a store
pip install ".[test]"     # reconstruction test dependencies
```

The base package does not install survey, LSS or lensing companions.

## Ordinary public API

A standard catalog dark-siren analysis is deliberately small:

```python
import darksirens as ds

events = ds.load_events("pe.h5")
injections = ds.load_injections("selection.h5")
catalog = ds.load_catalog("catalog.h5")

cosmology = ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075)
population = ds.Population("brokenpowerlaw+2peaks", fixed="gwtc5")

analysis = ds.model(
    cosmology=cosmology,
    population=population,
    catalog=catalog,
)

result = ds.infer(
    analysis,
    events=events,
    injections=injections,
    sampler="tinyns",
)
```

To sample only part of the population or of the survey block, fix the rest
at chosen values; they leave the sampled coordinates and enter the likelihood
as constants:

```python
population = ds.Population(
    "powerlaw+peak", fixed={"PL.m_max": 80.0, "G.mu": 35.0, "G.sigma": 5.0}
)
analysis = ds.model(
    cosmology=cosmology,
    population=population,
    catalog=catalog,
    fixed_survey={"delta": 0.0, "sigma_kde": 0.0},
)
```

The same model/inference stack supports catalog-free spectral sirens,
complete-catalog analyses, bright/counterpart sirens, reusable angular source
models and sampled population models. The standardized PE/injection and catalog
loaders consume reconstructed core file contracts; raw survey construction is
not a core responsibility.

Runnable public-API templates are under `examples/`.

## Specialized analyses

Core exposes four explicit one-way extension boundaries instead of a plugin
framework: a sampler-facing `InferenceTarget`, a host-density redshift seam,
a completion-curve composition seam and footprint-aware field estimators.

`InferenceTarget` lets a specialized package provide a log likelihood and a
`ParameterPlan` while reusing the accepted prior/sampler stack:

```python
import darksirens as ds

plan = ds.ParameterPlan(
    labels=("x",),
    lower=(-1.0,),
    upper=(1.0,),
    prior_kinds=(("uniform", None, None),),
    joint_constraints=(),
)

target = ds.InferenceTarget(
    log_likelihood=lambda theta: -0.5 * theta[0] ** 2,
    parameters=plan,
)
result = ds.infer(target)
```

For field/LSS-style redshift information, `darksirens.likelihood.host_density`
provides the explicit `RedshiftModel` protocol and target builder. Companion
packages own their field/count/tracer state; core owns the GW weighting,
selection reduction, prior assembly and sampler execution.

Two narrower seams let a survey owner reuse the ordinary catalog machinery with
its own completeness prescription. `build_incomplete_catalog_prior_state_from_curves`
in `darksirens.catalog.models` builds the accepted conditional prior from an
already-constructed `CompletionCurves` object, so count-derived,
magnitude-selection or external curves all enter through one function.
`darksirens.selection.footprint` composes one coverage fraction per catalog row
into the magnitude-selection curves, and `darksirens.catalog.field` evaluates the
complementary field-weighted numerator. Raw map parsing and the construction of
those inputs stay in survey packages.

## Guards worth knowing

Core refuses inputs that the frozen reference also refused, at the place the
user typed them rather than as a silent `-inf`:

- `ds.Cosmology` rejects `Om0`, `w0` or `wa` values or bounds outside the
  tabulated distance-grid support (`Om0` in [0.1575, 0.4575], `w0` in
  [-2.25, 0.25], `wa` in [-2.5, 2.5]); `H0` is unrestricted.
- `ds.model(..., completeness="complete")` gives a galaxy-free catalog row zero
  host mass by default (`empty_policy="zero"`, the frozen behavior); the
  comoving-volume fallback is the explicit opt-in `empty_policy="volume"`.
- `ds.infer` refuses a PE store and an injection store that declare different
  gwcat pairing contracts, and warns when their declared cosmologies differ.
- `ds.infer(..., selection_neff_guard="auto"|"hard"|"soft")` selects the
  sparse-selection guard; `auto` uses the soft penalized wall for NumPyro and
  the hard `-inf` wall otherwise. `max_likelihood_variance`, `sel_batch_size`
  and `pe_event_block` are likelihood options on the same call.
- Ordinary `ds.infer` results carry `log_prior_volume_fraction` and
  `logZ_corrected` next to the raw `logZ`, so evidences from angular models
  that reject part of their prior box are comparable.
- `ParameterPlan` prior kinds and joint-constraint kinds are closed
  vocabularies; anything else raises instead of being reinterpreted. An
  `InferenceTarget` plan must spell out `loc` and `scale` for every
  non-uniform prior (core's own `ParamSpec` may leave them `None`, meaning the
  standard `(0, 1)` defaults).
- The opt-in `DARKSIRENS_GW_PAIRING_M1_GRID` normaliser grid is sized to the
  bound model's mass support at bind time and refused if it cannot cover it.

## Ownership boundary

Core owns reusable siren inference machinery: cosmology, GW data contracts,
population models, angular population models, ordinary galaxy-catalog runtime,
redshift kernels/completeness, counterpart objects, GW selection integrals,
hierarchical likelihoods, priors, sampler adapters, checkpoints, results and
provenance.

Core does **not** own:

- raw DESI/KIBO/Legacy/GLADE schemas, masks, depth construction or survey
  selection fitting;
- Q/LSS ensembles, latent density fields, count models or multitracer state;
- lensing-specific physical state or likelihoods;
- campaign-specific mega CLIs.

Companion packages may depend on `darksirens`; `darksirens` never imports them.
There is no `universe_model` switchboard and no generic plugin registry.

## Validation

Existing mature behavior is accepted only after fixed-coordinate parity against
the pinned legacy implementation. The final Phase-8 branch also runs the full
reconstructed regression suite, a companion-import firewall, frozen bright-siren
parity, the extension-seam replays, and a fresh-wheel installation smoke test.
The suite additionally anchors the cosmology kernels, the sample weights and the
HEALPix pixelisation against astropy, hand-computed expectations and healpy, and
exercises the real TinyNS, Dynesty and NumPyro backends, so a physics change
cannot pass on self-consistency alone.
See `VALIDATION.md` and the phase records in `darksirens-rebuild` for exact
heads, workflow runs and tolerances.
