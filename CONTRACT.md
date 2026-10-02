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
  multitracer state (core owns the generic field-weighted mixture of
  standardized catalogs, `catalog_sky_weighting="field"`, and its
  missing-host extension point; tracer likelihoods, Q tables and member
  provenance stay in the LSS companion);
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
decode_parameters
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
parameter decoders or sampler adapters. `decode_parameters` is the one decoder
on the root surface: it is for companions and diagnostics that need the
physical parameters behind a coordinate vector (see "Parameter decoding"
below).

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
  and is refused for spectral and bright sirens. Unknown names raise
  `ValueError`. A value outside the parameter's prior bounds (inclusive)
  raises `ValueError` naming the parameter, the value and the bounds, unless
  `model(..., allow_out_of_prior=True)`: that accepts it, in either block, and
  warns (`UserWarning`) once per such value. Fixed parameters leave
  `ParameterPlan.labels`; `ParameterPlan.fixed_population_values`,
  `ParameterPlan.fixed_survey` and `ParameterPlan.allow_out_of_prior` record
  them, the bound likelihood reads them as constants, and
  `parameter_plan_semantic(plan)` in `darksirens.inference.run_fingerprint`
  puts them in a run fingerprint. Fixing one member of a joint prior pair
  warns: the other member keeps the likelihood-side rejection. Fixed values
  that violate a model's joint prior constraint (for example
  `lambda_0 + lambda_1 <= 1`), or leave the pair's sampled member no prior
  support, raise `ValueError` with or without `allow_out_of_prior`.
- `model(..., kernel_pin="auto"|"off")` (default `"auto"`): an
  incomplete-catalog analysis that samples none of `Om0`, `w0`, `wa`, `delta`,
  `sigma_kde` (`H0` may be sampled) evaluates the per-galaxy catalog kernel
  quadrature once, at bind time, at `H0_ref = 67.74`, and each call adds the
  exact scalar `3 ln(H0 / H0_ref)` to it, as the frozen reference's H0 kernel
  pin does. The likelihood agrees with the per-call quadrature to rounding,
  not bit for bit (on the benchmark fixtures, within 1e-15 relative on the
  total and on each event's log evidence, and 1.2e-13 on the Monte Carlo
  variance diagnostics). `"off"` keeps the per-call
  quadrature, and its bound program is unchanged. `ParameterPlan.kernel_pin`
  and `ParameterPlan.kernel_pin_active` record the setting and whether it
  applies, and `parameter_plan_semantic(plan)` puts both in a run fingerprint.
  The setting has no effect on spectral, bright-siren or complete-catalog
  analyses. Each call re-derives eight catalog rows from the live parameters
  and returns `-inf` if they disagree with the pin by more than 1e-9. A
  `BoundAnalysis` refuses a pin its plan does not admit, or one built for
  another catalog shape (for example after `dataclasses.replace`). The pin
  belongs to the catalog it was built from, and the eight-row check need not
  notice a same-shape catalog that differs elsewhere, so the pin carries a
  digest (`PinnedCatalogKernel.catalog_digest`, `BoundAnalysis.kernel_pin_digest`)
  of the catalog arrays the kernel reads (`zgals`, `dzgals`, `wgals`,
  `ngals`, every byte), the fixed `Om0`, `w0`, `wa`, `delta`, `sigma_kde` and
  `z_depth`, its `H0_ref` and probe rows, and the kernel's static settings.
  A `BoundAnalysis` recomputes it from its own catalog and plan when it is
  made (and unpickled) and refuses a pin whose digest differs, naming both;
  a new catalog needs a new `bind_analysis`. The digest is computed on the
  host and is pytree metadata, never a traced value, and it does not enter
  the tree structure's equality, so a compiled program serves a pin built
  from other data without retracing; the per-call program is unchanged.
- `model(..., completeness="complete", empty_policy="zero"|"volume")`: the
  default `"zero"` is the frozen behavior; `"volume"` is the explicit opt-in
  robustness approximation. `empty_policy` is illegal with any other
  composition.
- `model(..., completeness="selection", selection=..., row_fraction=None)`
  (opt-in) completes an incomplete catalog with an explicit
  magnitude-selection model instead of the per-row count ratio. `selection`
  is a `GaussianMagnitudeSelection`, a `SchechterMagnitudeSelection`
  (`darksirens.selection.catalog`) or its runtime payload mapping
  (`darksirens-catalog-selection-1.0`). Every row's completeness is the radial
  curve `C_sel(z)` (`selection_completion_curves`), times the row's coverage
  fraction when `row_fraction` (one value in [0, 1] per row of the catalog
  store) is given (`selection_completion_curves_with_row_fraction`); the
  kernel, the finite-depth convention and the missing-host budget are the
  incomplete catalog's, and no observed-density cache is built. The survey
  block is `log10n0`, `delta`, `sigma_kde` (with `n0_units`, `fixed_survey`
  and the kernel pin as for `completeness="incomplete"`). The selection
  nuisances (`M0hat`, `sigma_M` for the Gaussian family; `Mstar_hat`, `alpha`
  for the Schechter family) are fixed at the model's values unless
  `survey_priors` names them, which samples them; `m_lim`, `M_faint_offset`
  and the K-correction are never sampled. `ParameterPlan.catalog_model`
  records the selection payload and the row fraction's sha256, and
  `parameter_plan_semantic(plan)` adds it to a run fingerprint (only when it
  is set, so default fingerprints are unchanged). `decode_parameters`
  returns the selection model, with its sampled nuisances in place, as
  `catalog.selection` (`CatalogParameters.selection`, `None` otherwise). On
  the frozen reference's `c_mode="selection"` fixture the likelihood agrees
  with it to 3e-14 (`tools/probe_selection_completeness.py`).
- `model(..., catalog_sky_weighting="conditional"|"field",
  field_normalizer=None)`: the default `"conditional"` is the frozen
  per-row normalization (one catalog). `"field"` (opt-in) keeps each row's
  host mass and divides by the catalog's survey-global total
  `Z_k = sum_p [N_obs,p m_p + N_miss,p]` over every row of its sky
  (`darksirens.catalog.mixture`). `catalog` may then be a list of `K >= 1`
  catalog stores, each holding every row of its own HEALPix sky (different
  `nside` are allowed): a GW sample's host density is
  `sum_k w_k n_k(z | p_k) / Z_k` with `n_k` the field numerator at its row
  `p_k` of catalog k, for PE samples and injections alike. With one catalog
  `Z_1` cancels and is not evaluated (the value is the field host-density
  seam's, to rounding). `completeness` is `"incomplete"` or `"selection"`
  for every catalog (`"complete"` is refused); `selection` and
  `row_fraction` are then lists with one entry per catalog. Catalog 1's survey
  labels are `log10n0`, `delta`, `sigma_kde` (plus selection nuisances named
  in `survey_priors`), catalog `k >= 2`'s carry the suffix `_c{k}`, and the
  weights are the stick-breaking coordinates `fcat_2 .. fcat_K` after the
  survey blocks, `fcat_m ~ Beta(1, K - m + 1)` on [0, 1] (uniform on the
  simplex; `w = (1 - fcat_2, fcat_2)` at `K = 2`), the frozen reference's
  labels, priors and order; `fixed_survey` and `survey_priors` take these
  labels (`fixed_survey` may fix a weight). `field_normalizer="auto"` (the
  default) evaluates `Z_k` in the `"moments"` form for the selection
  completeness (one curve, exact) and in the `"direct"` form (every full-sky
  row's curves, row blocked) for the count ratio, the only exact form there;
  `"moments"` with the count ratio is refused. The kernel pin applies per
  catalog, to its compact view and (with a survey depth) its full sky, when
  `Om0`, `w0`, `wa` and its own `delta`, `sigma_kde` are fixed; each pin's
  catalog digest is checked against its view when the binding is made. A
  conditional analysis given `K >= 2` catalogs is refused. `compute_dtype="float32"`,
  the `missing_density`, `kernel_layout` and `kernel_window` settings (the
  window on each compact view), `n0_units`, the
  soft and hard guards and the layout options apply as for the ordinary
  incomplete catalog. `decode_parameters` returns
  `CatalogMixtureParameters(components, log_weights)` as `catalog` (one
  `CatalogParameters` per catalog with its store's `z_depth`) and refuses an
  explicit `z_depth`. `ParameterPlan.catalog_model` records the weighting,
  completeness, number of catalogs, normaliser, per-catalog pin activity,
  selection payloads and row-fraction digests. On the frozen reference's
  `darksiren_log_likelihood(..., catalog_sky_weighting="field",
  n_catalogs=K)` the likelihood agrees to 3e-14
  (`tools/probe_catalog_mixture.py`; on split mock T, 72 points, 3e-14).
- `model(..., per_catalog_population={k: [parameter, ...]})` (opt-in, a
  field-weighted mixture of `K >= 2` catalogs only): catalog `k` (`2 <= k <=
  K`) carries its own copy of the named population parameters (labels or
  ASCII names), labelled `"<label>_c{k}"`, with the base parameter's prior
  bounds and kind; every entry not named is catalog 1's (the analysis's
  population), so a sampled base width is shared. The copies sit after the
  survey blocks and before the weights, catalog by catalog in model order. A
  joint constraint of the model whose members are all copied is applied to
  the copies. Each catalog's population multiplies its own branch, for PE
  samples and injections alike: `log w = logsumexp_k [log w_k + log p_pop(theta
  | L_k) + log n_k - log Z_k] - log J - log pi`, so `mu = sum_k w_k
  alpha_k(L_k)`. `decode_parameters` returns the K vectors as
  `catalog.populations` (`CatalogMixtureParameters.populations`, `None`
  otherwise; catalog 1's is `population`). `ParameterPlan.catalog_population`
  and `n_catalog_population` record the blocks, and `parameter_plan_semantic`
  adds them only when set. `compute_dtype="float32"` is supported. Against the
  reference's tracer-dependent blocks (legacy af896ca, replayed onto c042527)
  the likelihood agrees to 3e-14 (`tools/probe_catalog_population_blocks.py`).
- `model(..., survey_priors={label: prior})` (opt-in) sets the prior of named
  survey parameters of a catalog analysis: `(lower, upper)` or
  `("uniform", lower, upper)`, a normal `("normal", loc, scale)` truncated to
  the parameter's default bounds, or `("normal", loc, scale, lower, upper)`.
  The defaults are uniform: `log10n0` [-4, -1], `delta` [-3, 3], `sigma_kde`
  [0, 0.05], and for the selection nuisances `M0hat` [-23, -18], `sigma_M`
  [0.05, 3], `Mstar_hat` [-23, -18], `alpha` [-1.9, 0]. Lower bounds may not
  reach `sigma_kde < 0`, `sigma_M <= 0` or `alpha <= -2`. A parameter cannot
  be both fixed and given a prior; unknown labels raise. The bounds and prior
  kinds are part of the plan, so a run fingerprint changes with them.
- `infer(..., selection_neff_guard="auto"|"hard"|"soft",
  max_likelihood_variance=None, sel_batch_size=None, pe_event_block=None,
  compute_dtype=None)` are likelihood options, never sampler options. `auto`
  resolves to `soft` for NumPyro and `hard` otherwise. They are refused for an
  `InferenceTarget`.
- `bind_analysis(..., compute_dtype=None|"float64"|"float32")`: `None` and
  `"float64"` are the default float64 program, unchanged bit for bit (the
  option is not even passed to the likelihood). `"float32"` rounds the
  per-sample columns to float32 at bind time and evaluates the per-sample PE
  and selection log weights in float32. Each weight is returned as float64, so
  the log-sum-exp reductions, Monte-Carlo variances, `N_eff`, soft guard and
  final sum stay float64. Per-proposal grids (distance table, comoving volume,
  catalog kernel normalisers and completeness curves) are computed in float64.
  The float32 likelihood is value-only: differentiating it raises `TypeError`
  (float32 population densities underflow in the tails, where reverse-mode
  derivatives are NaN), and `infer(..., sampler="numpyro")` refuses it.
  It is refused with a `ValueError` for bright and complete-catalog analyses,
  anisotropic angular models, component-spin populations and every
  Gaussian-process population (`population/gp.py`): the GP covariance algebra
  stays float64 (`cond(K)` reaches 5e5-2e7), so float32 gains nothing there,
  and binned GP models showed the largest float32 shifts.
- Run fingerprints record the likelihood options that can change the
  likelihood value: `core_numerics_semantic(bound)` (in
  `darksirens.inference.run_fingerprint`) adds a `"likelihood_options"`
  entry with a non-default `compute_dtype`, `max_likelihood_variance` and
  `selection_neff_soft_guard=True`, and no entry for a default binding, so
  `core_numerics_semantic(bound)` then equals `core_numerics_semantic()` and
  every existing digest is unchanged. `sel_batch_size` and `pe_event_block`
  are never recorded: they split the same log-sum-exp sums into blocks
  (padding rows carry `-inf` weight and `Ndraw` is unchanged) and move the
  value by floating-point reassociation only (within the pinned 1e-12
  relative contract; 4e-16 on the DESI P12.4 target).
- Evaluation settings (`darksirens.catalog.settings` and the pairing
  settings of `darksirens.population.utils`) enter
  `core_numerics_semantic()` when they are not at their historical value.
  Since 2026-10-02 the defaults are `pairing_norm="auto"`,
  `kernel_layout="galaxy_list"`, `missing_density="auto"` and
  `kernel_window="auto"` (each the faster evaluation where it applies, and
  the historical one otherwise, without raising); they are recorded, with a
  binding's kernel window recorded as the tolerance it was bound with (none
  when `"auto"` attached no window). The historical values
  (`"per_sample"`, `"padded"`, `"grid"`, `"off"`) are left out, so a
  fingerprint made before the change matches when they are set explicitly,
  and they reproduce the pre-change program bit for bit. Resuming a
  pre-change checkpoint under the new defaults is refused as a settings
  change, and the message names those settings.
- `infer(..., sampler_preflight="on"|"off")` is a sampler option (default
  `"on"`). On a fresh TinyNS or Dynesty run it draws a few prior samples and
  raises when none has a finite likelihood (and warns when very few do);
  `"off"` skips it. When the check fails or warns, its advice names `infer`
  keywords, not command-line flags.
- Ordinary `infer` results carry `log_prior_volume_fraction` and, for a finite
  `logZ`, `logZ_corrected = logZ - log_prior_volume_fraction`. The raw `logZ`
  is exactly the sampler's number.
- Every `infer` result carries the sampler-termination fields `dlogz_final`,
  `stop_reason`, `ncall` and `niter`
  (`darksirens.inference.nested_output.TERMINATION_FIELDS`); a backend that
  does not report one sets it to `None`. For Dynesty, `dlogz_final` is the
  remaining-evidence estimate at the check that ended the run, bitwise the
  number dynesty compared with `dlogz`; `stop_reason` is `"convergence"` (the
  `dlogz` criterion held, including when a cap was reached at the same
  check), `"maxcall"` (more than `max_samples` calls in this sampling call),
  `"plateau"` or `"unknown"`; `ncall` and `niter` are dynesty's call counter
  and iteration count. TinyNS reports its own final dlogz, call and iteration
  counts with `"convergence"`, `"maxiter"`, `"callback"`,
  `"replacement_failure"` or `"unknown"`. NumPyro and a target with no free
  parameter set all four to `None`. The names and the value `"convergence"`
  are a downstream contract: convergence checks read them as they are.
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

A target run that writes or resumes a dynesty or TinyNS checkpoint is
fingerprinted by `ds.infer`
(`run_fingerprint.inference_target_semantic`): the target's parameter plan
(`parameter_plan_semantic`), `core_numerics_semantic()`, the sampler and its
target-setting options (`sampler_semantic`: everything except checkpoint and
resume machinery, progress printing and the preflight), and the target's
optional `identity` and `provenance`. `run_fingerprint.json` is written beside
every checkpoint the run writes; a resume whose checkpoint directory holds a
different or no fingerprint raises `ResumeFingerprintError`, naming each
differing entry, unless the sampler option `resume_force=True` is given (the
reference's forced-resume rules apply). The result then carries
`run_fingerprint_digest`. A run without a checkpoint builds and writes nothing,
and its result is unchanged.

`InferenceTarget(..., identity=None, provenance=None)` are keyword-only:
`identity` is a JSON-like value naming the target, and `provenance` is a
mapping, or a zero-argument callable returning one, that a companion fills
with the state its likelihood was built from (artifact content hashes, e.g.
`run_fingerprint.file_identity(path)`, fixed values, its likelihood options
through `likelihood_options_semantic`). Core hashes both as given and
interprets neither; the callable runs only when a fingerprint is built. A
checkpointing target that sets neither warns, because its fingerprint cannot
tell its likelihood from another target's with the same plan. Ordinary
analyses are not fingerprinted by `ds.infer`.

### Parameter decoding

`decode_parameters(analysis, theta, *, z_depth=BINDING_DEPTH)` returns the
physical parameters the bound likelihood evaluates at `theta`, as an immutable
named record `darksirens.runtime_binding.DecodedParameters`:

- `cosmology`: `CosmologyParameters(H0, Om0, w0, wa)`; sampled entries come
  from `theta`, fixed ones are the plan's values (`fixed_cosmology`);
- `population`: the full population vector in `population_labels` order,
  fixed entries (`Population(fixed=...)`) included;
- `catalog`: `CatalogParameters(n0, delta, sigma_kde, z_depth)` for a catalog
  analysis, `None` for spectral and bright sirens. `n0` is physical
  (Mpc^-3): `10**log10n0`, times `(H0 / 100)**3` under `n0_units="h_scaled"`,
  and `1.0` for `completeness="complete"`. Survey parameters fixed with
  `fixed_survey` enter as 0-d arrays of `theta`'s dtype. `z_depth` is
  structural (a Python float or `None`, never traced). `selection` is
  `None`, or for `completeness="selection"` the runtime selection model with
  any sampled nuisance (`survey_priors`) in place of its value;
- `angular`: the angular-model coordinates (`angular_labels`), empty for the
  isotropic model.

`analysis` is a `ds.model` analysis or a `BoundAnalysis`; `theta` has one
value per `analysis.parameters.labels` entry, and any other shape raises
`ValueError` naming the labels. `z_depth` defaults to the depth
`bind_analysis` uses (the catalog store's `z_depth`, or the binding's); an
explicit `z_depth` is accepted with an `Analysis` and refused with a
`BoundAnalysis`. The bound likelihood decodes through the same function
(`_decode_theta`), so the operations are the same, and a likelihood that
decodes through `decode_parameters` inside its jitted program compiles to the
bound program (same optimized HLO, same value). It works eagerly and under
`jax.jit`, `jax.vmap` and `jax.grad`. Every field is a copy of `theta`, a plan
value or `10**log10n0`, identical in every context, except the h-scaled `n0`,
which XLA may round differently in differently compiled programs (eager, jit,
vmap; within 1e-15 relative).

This is public API, and companion packages (lss, surveys, the DESI consumer)
decode through it instead of copying the private decoder. Its signature, its
record's fields and their meaning change only through this contract, like any
other package-root name.

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

The field seam takes the catalog kernel pin too. A field target whose sampled
labels include none of `Om0`, `w0`, `wa`, `delta`, `sigma_kde`
(`field_kernel_pin_applies(labels, setting)`) builds the pin once with
`darksirens.catalog.field.build_pinned_field_kernel`, the ordinary path's
builder (`H0_ref = 67.74`, eight probe rows, tolerance 1e-9), passes it to its
jitted evaluation as an argument, and hands it to
`build_field_incomplete_catalog_prior_state_from_curves(..., pinned_kernel=pin)`
on every call. The completion curves are still evaluated per call; each call
re-derives the probe rows from the live parameters and a disagreement makes the
host-density likelihood `-inf`. The seam refuses a pin built for another
catalog shape, and a concrete pin under a traced catalog (a pin closed over by
a jit would be a constant of the compiled program). Like the ordinary path's
pin, it belongs to the catalog it was built from, and the eight-row check need
not notice a same-shape catalog that differs elsewhere, so it carries the same
catalog digest. `check_field_kernel_pin(pin, cosmo, params, catalog)`
recomputes it on the host and refuses a pin built from another catalog view or
premise, naming both digests; a target calls it where it attaches the pin to
the catalog view it evaluates, outside its jit (inside, the catalog is traced
and the seam cannot read it), and an eager call of the seam runs it itself. A
new catalog view needs a new pin.
`field_kernel_pin_plan(plan, setting)` records `kernel_pin` and
`kernel_pin_active` on the target's plan, so `parameter_plan_semantic` puts
them in the target's run fingerprint. Without a pin the seam's program is
unchanged.

### Catalog mixture seam

`ds.model(catalog=[...], catalog_sky_weighting="field")` is the ordinary path's
field-weighted host density of `K >= 1` standardized catalogs
(`darksirens.catalog.mixture`, `darksirens.likelihood.mixture`). A companion
that modulates a catalog's missing-host density (for example a Q table)
passes a `MissingHostExtension` to
`darksirens.likelihood.mixture.make_catalog_mixture_target(analysis, events=,
injections=, extension=)`, which returns an `InferenceTarget` whose
coordinates are the analysis's labels followed by the extension's
(`parameter_spec()`). The extension is an explicit argument, never
registered. Core calls `missing_density(context, dN_miss)` on each catalog's
compact rows and, for `K >= 2`, on its full sky, recomputes every row's
`N_miss` from the result by the redshift-grid trapezoid, and builds the
numerator's row masses and `Z_k` from it; `missing_total(context)` may
instead return the modulated full-sky missing mass (for example from moments
the extension precomputed), and `None` makes core sum the modulated full-sky
curves. `n_members = M >= 1` makes the likelihood `logsumexp_m logL_m -
log M` over one member index shared by every catalog, with the
member-independent work (kernels, base curves, observed totals) done once.
The `MissingHostContext` gives the catalog index, the view and its global
rows, the member, the extension's parameters and `data` (a jit operand, never
a constant), the proposal's cosmology and catalog parameters, `dN_exp`, the
depth mask, the selection curve `Cbar(z)`, the view's row fraction and the
full-sky observed total. An explicit `missing_density="gather"` is refused
with an extension; the `"auto"` default keeps the grid there. The target's fingerprint provenance records the analysis plan's
semantic, the non-default likelihood options and the extension's
`provenance()`.

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
