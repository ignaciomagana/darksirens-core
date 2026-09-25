# darksirens core contract

## Role

This repository builds the Python distribution and import package `darksirens`.
It owns the reusable hierarchical population and siren-inference kernel.

Frozen reconstruction reference:

```text
ignaciomagana/darksirens@c042527238bd71421b792936bc48c3b815b90d6d
```

Existing mature behavior is moved only after fixed-coordinate parity against
that reference. Architecture and science are not changed simultaneously during
migration.

## Dependency direction

Future companion packages may import core:

```text
darksirens-surveys  -> darksirens
darksirens-lss      -> darksirens
darksirens-lensing  -> darksirens
```

Core must never import a companion package. LSS and lensing companions must not
depend on the survey companion merely to use core inference primitives.

## Core ownership

Core owns:

- JAX cosmology, distance interpolation and volume factors;
- standardized GW posterior/injection contracts and loaders;
- population models, including optional GP models and component-spin bases;
- reusable angular source-population models;
- standardized ordinary galaxy-catalog runtime;
- ordinary redshift kernels, completeness and host weights;
- bright-siren counterpart objects;
- GW selection integrals and reliability diagnostics;
- spectral, dark, complete-catalog and bright hierarchical likelihoods;
- parameter/prior assembly and joint-prior transforms;
- TinyNS, Dynesty and NumPyro execution adapters;
- checkpoint/result/provenance primitives;
- explicit specialized-analysis seams described below.

Core does not own:

- native DESI/KIBO/Legacy/GLADE schemas, masks, depth-map construction, raw
  survey ingestion/weights or survey selection fitting;
- Q/LSS ensembles, latent density fields, galaxy-count likelihoods or
  multitracer state;
- lensing-specific physical state, partitions, marks or likelihoods;
- campaign-specific command glue;
- a generic plugin registry or `universe_model` switchboard.

## Frozen package-root API

The ordinary package-root surface (`darksirens.__all__`) is:

```python
__version__
configure_jax_runtime
Cosmology
Population
Counterpart
ParameterPlan
InferenceTarget
load_events
load_injections
load_catalog
model
infer
```

Typical ordinary usage is:

```python
import darksirens as ds

events = ds.load_events("pe.h5")
injections = ds.load_injections("selection.h5")
catalog = ds.load_catalog("catalog.h5")

analysis = ds.model(
    cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
    population=ds.Population("brokenpowerlaw+2peaks", fixed="gwtc5"),
    catalog=catalog,
)

result = ds.infer(
    analysis,
    events=events,
    injections=injections,
)
```

Ordinary users should not need low-level GW event structs, catalog internals,
parameter decoders or sampler adapters.

### Public option contract

- `Cosmology` accepts `Om0`, `w0` and `wa` only inside the tabulated
  distance-grid support (`Om0` [0.1575, 0.4575], `w0` [-2.25, 0.25], `wa`
  [-2.5, 2.5], closed intervals); `H0` is unrestricted. Out-of-grid values or
  bounds raise at construction.
- `Population(name, fixed={parameter: value, ...})` fixes only the named
  population parameters, at the given values, and samples the rest. A key is
  the model's label (`analysis.parameters.population_labels`) or its ASCII name
  where the model declares one (`"PL.alpha"`). `model(..., fixed_survey={name:
  value})` does the same for the survey parameters of a catalog analysis
  (`log10n0`, `delta`, `sigma_kde`; no `log10n0` for `completeness="complete"`)
  and is refused for spectral and bright sirens. Unknown names, and values
  outside the parameter's prior bounds, raise `ValueError`. Fixed parameters
  leave `ParameterPlan.labels`; `ParameterPlan.fixed_population_values` and
  `ParameterPlan.fixed_survey` record them, the bound likelihood reads them as
  constants, and `parameter_plan_semantic(plan)` in
  `darksirens.inference.run_fingerprint` puts them in a run fingerprint.
  Fixing one member of a joint prior pair warns: the other member keeps the
  likelihood-side rejection.
- `model(..., completeness="complete", empty_policy="zero"|"volume")`: the
  default `"zero"` is the frozen behavior; `"volume"` is the explicit opt-in
  robustness approximation. `empty_policy` is illegal with any other
  composition.
- `infer(..., selection_neff_guard="auto"|"hard"|"soft",
  max_likelihood_variance=None, sel_batch_size=None, pe_event_block=None)` are
  likelihood options, never sampler options. `auto` resolves to `soft` for
  NumPyro and `hard` otherwise. They are refused for an `InferenceTarget`.
- `infer(..., sampler_preflight="on"|"off")` is a sampler option (default
  `"on"`). On a fresh TinyNS or Dynesty run it draws a few prior samples and
  raises when none has a finite likelihood (and warns when very few do);
  `"off"` skips it. When the check fails or warns, its advice names `infer`
  keywords, not command-line flags.
- Ordinary `infer` results carry `log_prior_volume_fraction` and, for a finite
  `logZ`, `logZ_corrected = logZ - log_prior_volume_fraction`. The raw `logZ`
  is exactly the sampler's number.
- `infer` refuses a PE store and an injection store whose `contract_hash`
  attributes differ (stores without one are exempt) and warns when their
  declared cosmologies differ.
- `ParameterPlan.prior_kinds` are `(kind, loc, scale)` triples with `kind` in
  `{uniform, normal, lognormal, beta}`; joint-constraint kinds are
  `{ordered_le, conditional_upper, simplex, ball3}` with arity 2/2/2/3. The
  prior transform raises on any other kind. Inside core, `None` for `loc` or
  `scale` keeps the `ParamSpec` meaning: the standard `(0, 1)` defaults, which
  core's own registries never rely on. At the `InferenceTarget` seam a
  non-uniform kind must state `loc` and `scale` explicitly (finite, `scale`
  positive); `beta` consumes only its shape (`scale`), so its `loc` may be
  omitted.

## Specialized extension seams

### `InferenceTarget`

A specialized package may provide an explicit likelihood plus `ParameterPlan`
and reuse core prior transforms and sampler execution:

```python
target = ds.InferenceTarget(log_likelihood=callable, parameters=plan)
result = ds.infer(target)
```

This seam knows nothing about specialized physics.

### Host-density/redshift seam

`darksirens.likelihood.host_density` exposes an explicit `RedshiftModel`
protocol, parameter-plan composition and `make_host_density_target`. A companion
owns its opaque redshift/field/count state; core continues to own GW population
weighting, Jacobians, PE reduction, selection reduction, priors and samplers.

### Completion-curve composition seam

`darksirens.catalog.models.build_incomplete_catalog_prior_state_from_curves`
builds the accepted ordinary conditional prior from an already-constructed
`darksirens.catalog.completeness.CompletionCurves` object. Core owns the kernel,
the finite-depth observed-count factor, the additive missing density and the row
normalization; the caller owns the completeness prescription that produced the
curves.

### Footprint and field seam

`darksirens.selection.footprint` composes one externally supplied coverage
fraction per standardized catalog row into the magnitude-selection completeness
curves, and `darksirens.catalog.field` evaluates the complementary field-weighted
(unnormalized additive) host-density numerator. Raw map parsing, HEALPix
degradation and the construction of the row fractions belong to survey packages.

There is no model-name discovery, entry-point registration or callback registry.

## Import contract

`import darksirens` must remain light. Package-root import must not initialize or
import JAX, h5py, TinyNS, Dynesty, NumPyro, tinygp, gwcat or companion packages.
Scientific work configures/imports those modules lazily when the corresponding
public function is called.

## Numerical invariants

- validated reconstruction numerics use CPython 3.11, JAX/JAXLIB 0.4.34,
  NumPy 1.26.4, SciPy 1.12.0 and h5py 3.12.1;
- x64 is enabled before float64 scientific grids are constructed;
- `DARKSIRENS_ZMAX` retains the frozen grid meaning;
- standardized posterior and injection proposals use the canonical
  `(m1det, q, dL)` sample-coordinate convention;
- the source-to-detector Jacobian is owned by one core weighting path;
- padded GW samples are rejected by explicit structural masks;
- selection effective-sample-size and likelihood-variance guards are part of
  the scientific contract;
- ordinary isotropic behavior stays on the already validated compute path;
- sampler zero-free-parameter exact evidence occurs before backend validation or
  optional-backend import;
- cosmology priors lie inside the tabulated distance-grid support, so no sample
  is silently `-inf` from an unrepresentable `(Om0, w0, wa)`;
- a complete-catalog row without galaxies contributes zero host mass unless the
  volume fallback is requested explicitly;
- GP z-conditional and m1-conditional normalisers are tabulated in the GP
  coordinate (`log1p(z)`) and on nodes that follow the sampled mass support, and
  the mass-ratio normaliser at fixed primary mass integrates on nodes spanning
  the allowed mass-ratio range and tabulates the normaliser divided by the
  low-mass taper, so the conditional density integrates to one inside the prior
  box, including just above `m_min`;
- `ang2pix_ring` reproduces `healpy.ang2pix(..., nest=False)` exactly,
  including ring-boundary and polar-cap rounding;
- marked-host PE and selection views share one catalog view and one mark table,
  or one survey-wide reference table, and mark tables are z-centered.

## Packaging contract

The base distribution includes the validated numerical stack plus the pinned
TinyNS commit used by the legacy campaign, because TinyNS is the public default
sampler. SciPy is a base dependency because ordinary completeness evaluation
uses `scipy.special` at runtime.

GP models, Dynesty, NumPyro and gwcat prior support are explicit extras; the
Dynesty extra also carries Matplotlib because its optional run diagnostics plot.
Raw survey/LSS/lensing packages are never package dependencies of core.
