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
stack used for reconstruction parity and includes TinyNS. `ds.infer`'s default
sampler is Dynesty, an optional extra, so a typical install is:

```bash
pip install ".[dynesty]"
```

Without the extra, `ds.infer` with the default sampler raises an `ImportError`
that names the extra; `ds.infer(..., sampler="tinyns")` runs on the base
install. Optional integrations are explicit:

```bash
pip install ".[gp]"       # tinygp/equinox population models
pip install ".[dynesty]"  # Dynesty backend (the default sampler)
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
)  # sampler="dynesty" (default), "tinyns" or "numpyro"

ds.save_result("result.h5", result, labels=analysis.parameters.labels)
```

`ds.save_result` writes the result to one HDF5 file, published atomically:
the posterior `samples` with their `labels`, the nested sampler's dead points
(`logl_dead`, `logwt_dead`), one attribute per scalar entry (`logZ`,
`logZerr`, `stop_reason`, ...) and every other entry, `guard_report`
included, as JSON in the `result_json` attribute
(`darksirens.io.results.save_result` documents the layout).

The likelihood `ds.infer` samples is available on its own, for a grid or a
profile, bound exactly as `ds.infer` binds it:

```python
log_likelihood = ds.log_likelihood(analysis, events=events, injections=injections)
log_likelihood(np.array([70.0]))  # one value per analysis.parameters.labels entry
```

It takes the same likelihood options as `ds.infer`
(`selection_neff_guard`, `max_likelihood_variance`, `sel_batch_size`,
`pe_event_block`, `compute_dtype`); with no sampler to resolve against,
`selection_neff_guard="auto"` is the hard guard.

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

A fixed value must lie inside its parameter's prior bounds;
`ds.model(..., allow_out_of_prior=True)` accepts one outside them, such as
`log10n0 = log10(5e-5)` below the `[-4, -1]` prior, with a warning.

With `delta` and `sigma_kde` fixed, and `Om0`, `w0`, `wa` fixed (the default
cosmology), the catalog kernel's redshift dependence is fixed up to a scalar
`H0` factor, so the per-galaxy kernel quadrature is evaluated once, at bind
time, instead of on every likelihood call (`kernel_pin="auto"`, the default).
The values agree with the per-call quadrature to rounding;
`ds.model(..., kernel_pin="off")` keeps the per-call quadrature.

An incomplete catalog can instead be completed with an explicit
magnitude-selection model (for example the runtime payload
`darksirens-surveys` writes for a fitted survey). Its nuisances are fixed at
the model's values unless `survey_priors` samples them:

```python
from darksirens.selection.catalog import SchechterMagnitudeSelection

analysis = ds.model(
    cosmology=cosmology,
    population=population,
    catalog=catalog,
    completeness="selection",
    selection=SchechterMagnitudeSelection(
        m_lim=20.0, Mstar_hat=-19.63, alpha=-1.07, M_faint_offset=-0.09
    ),
    survey_priors={"log10n0": (-5.0, -1.0), "Mstar_hat": ("normal", -19.63, 0.01)},
)
```

The completeness can also be given directly as a table in redshift, for a
catalog whose completeness was measured rather than fitted with a luminosity
function:

```python
from darksirens.selection.catalog import TabulatedSelection

analysis = ds.model(
    cosmology=cosmology,
    population=population,
    catalog=catalog,  # z_depth = 0.3
    completeness="selection",
    selection=TabulatedSelection(
        z=[0.0, 0.05, 0.1, 0.2, 0.3],
        completeness=[1.0, 0.92, 0.61, 0.18, 0.04],
    ),
)
```

The nodes must be strictly increasing and need not be evenly spaced; the
values lie in [0, 1] and the curve is linear between nodes. It is a function
of the catalog redshift only: `H0` and the other cosmological parameters do
not enter it, and it has no parameter to sample. The table is never
extrapolated. It must start at redshift 0 and reach the catalog's `z_depth`,
or the top of the model's redshift grid (5 by default) for a catalog without
one, and `ds.model` raises otherwise. The same table can be passed as the
payload `{"format_version": "darksirens-catalog-selection-1.0", "family":
"tabulated", "z": [...], "completeness": [...]}`, so the catalog file is the
same for analyses with different curves. `row_fraction` multiplies it row by
row, as for the other families.

A catalog with galaxy weights needs care here. In this mode a row's host
density is `N_obs p_cat(z) + (1 - C(z)) n0 dV/dz`: the catalogued part is the
row's galaxy count times a weight-normalized kernel, which is the row's
weighted density divided by the row's mean weight. For hosts weighted by, say,
stellar mass, the missing part must be in the same unit. The completeness
supplied must then be the weight-fraction completeness (the fraction of the
total weight that is in the catalog at each redshift), and `n0` must be the
reference weight density divided by the catalog's mean weight, with `n0_units`
stated. Supplying weights together with a count-based `n0` and a count-based
completeness biases `H0`; a consumer reported -1.5 km/s/Mpc on a toy (not
verified here). Core divides by each row's own mean weight, so one `n0` is
exact only where the rows' mean weights equal the catalog's.

The default count-ratio completeness (`completeness="incomplete"`) is biased
where the completeness falls. It divides each row's observed galaxy density,
smoothed with a Gaussian of width 0.05 in redshift, by the expected density
smoothed the same way. Both densities rise steeply with redshift, so the ratio
at `z` is dominated by the far side of the kernel, where fewer galaxies are
catalogued: to first order it is the completeness near `z + 2 w^2 / z`, and
the missing-host density is too large where the completeness drops. Measured
H0 shifts: +0.68 +- 0.14 km/s/Mpc against the exact model on an unclustered
toy with a known completeness (the darksirens-desi-gwtc5 clean-room check),
and +0.78 +- 0.18 km/s/Mpc against the magnitude-selection completeness on
100 seeds of the clustered examples mock. The default is kept for
reproducibility. `count_ratio="pooled"` (opt-in) smooths the ratio itself,
each galaxy weighted by one over the expected density at its own redshift,
pools all the rows of the catalog before the clip at 1, and gives row `p` the
completeness `f_p C(z)` with `f_p` its `row_fraction` (1 without one):

```python
analysis = ds.model(
    cosmology=cosmology,
    population=population,
    catalog=catalog,
    completeness="incomplete",
    count_ratio="pooled",
    count_ratio_window=0.02,  # Gaussian width in redshift (the default)
    row_fraction=coverage,  # optional, one value in [0, 1] per catalog row
)
```

What is left is the kernel's own `w^2 C''/2`, so the window should be small:
on the toy the curve is within 0.006 of the truth at 0.01 and 0.016 at 0.02
(the default estimator is off by 0.14 at z = 0.2). The whole catalog is one
completeness cell, so a survey whose depth varies over the sky should be
given as one catalog per depth (`catalog_sky_weighting="field"`, each with
its `row_fraction`). Binding refuses a window that holds fewer than 100
effective galaxies at the catalog's median redshift and warns below 1000,
and warns when more than a tenth of the rows are empty without a
`row_fraction` (they count as surveyed). Within about three windows of
`z = 0` the curve is clipped to 1 (galaxies scattered to the lowest redshifts
carry weights that grow as `z^-2`), so the catalog is taken to be complete
there.

Several catalogs combine in a field-weighted mixture: each sample's host
density is `sum_k w_k n_k(z | p_k) / Z_k`, with each catalog's host mass
normalized by its survey-global total and stick-breaking weights `fcat_k`:

```python
analysis = ds.model(
    cosmology=cosmology,
    population=population,
    catalog=[galaxies, agn],
    catalog_sky_weighting="field",
    completeness="selection",
    selection=[galaxy_selection, agn_selection],
    fixed_survey={"delta": 0.0, "sigma_kde": 0.0, "delta_c2": 0.0, "sigma_kde_c2": 0.0},
    survey_priors={"log10n0_c2": (-6.0, -4.0)},
)
analysis.parameters.labels  # ('H0', 'log10n0', 'log10n0_c2', 'fcat_2')
```

Each catalog may have its own completeness, for example
`completeness=["incomplete", "selection"]` with `selection=[None,
agn_selection]`, or `"complete"` for a catalog that holds every host.

Galaxy weights (`wgals`) are normalized inside each sky row. Under the field
weighting the host mass of a row is by default its galaxy count, so the
weights only decide which galaxy of a row is the host: a row with one heavy
galaxy and a row with one light galaxy carry the same mass. For a complete
catalog, `host_mass="weight"` makes the row's host mass the sum of its galaxy
weights instead, in the row's density and in the catalog's survey-global
total:

```python
analysis = ds.model(
    cosmology=cosmology,
    population=population,
    catalog=catalog,
    catalog_sky_weighting="field",
    completeness="complete",
    host_mass="weight",
)
```

Every galaxy then hosts in proportion to its own weight (a stellar mass or a
luminosity, for example) whatever row it is in. The weights must be finite
and strictly positive, and their unit does not matter. The default
`host_mass="count"` changes nothing. The option is refused for an incomplete
catalog (`completeness="incomplete"` or `"selection"`, for any catalog of a
mixture), because the hosts missing from the catalog would then have to be
counted by weight too, and for the conditional weighting, which divides each
row's host mass out.

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

A companion that builds its own likelihood on an ordinary plan decodes the
coordinate vector with `ds.decode_parameters`, the decoder the bound
likelihood itself uses:

```python
decoded = ds.decode_parameters(analysis, theta)  # or a BoundAnalysis
decoded.cosmology       # CosmologyParameters(H0, Om0, w0, wa)
decoded.population      # every population parameter, fixed ones included
decoded.catalog         # CatalogParameters(n0, delta, sigma_kde, z_depth, selection), or None
decoded.angular         # the angular-model coordinates
```

It applies the plan's fixed values and the `n0_units` conversion, works
eagerly and under `jax.jit`, `jax.vmap` and `jax.grad`, and refuses a `theta`
of the wrong length. It is public API: changes go through `CONTRACT.md`.

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
those inputs stay in survey packages. A field target that fixes `Om0`, `w0`,
`wa`, `delta` and `sigma_kde` can build the catalog kernel once
(`darksirens.catalog.field.build_pinned_field_kernel`) and pass it to the seam
on every call as a jit argument (`pinned_kernel=`), as `ds.model` does with
`kernel_pin="auto"`.

## Guards worth knowing

Core refuses inputs that the frozen reference also refused, at the place the
user typed them rather than as a silent `-inf`:

- A fixed population preset (`Population(name, fixed=True)`, `'in_prior_v2'`,
  `'gwtc5'`) pins a curated fiducial vector; `Population(...).fiducial_values`
  shows it by parameter label. The legacy powerlaw+peak vector has the
  merger-rate slope `gamma = 2.5` (the measured kappa_z; legacy used 0 before
  commit 0befab7), so data simulated with another slope need it pinned:
  `fixed={r"$\gamma$": 0.0, ...}`. Check `fiducial_values` against how a mock
  was generated before comparing results.
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
  the hard `-inf` wall otherwise. `max_likelihood_variance`, `sel_batch_size`,
  `pe_event_block` and `compute_dtype` are likelihood options on the same call.
- Where the guard cuts the prior is reported. Before sampling an ordinary
  analysis, `ds.infer` evaluates the likelihood's diagnostics at 32 prior
  draws and records in `result["guard_report"]` which guard fired
  (`"selection_neff"`: too few effective injections; `"pe_mc_variance"`: the
  per-event PE Monte-Carlo variance used up the `max_likelihood_variance`
  budget) and the range of each sampled parameter where it fired. It warns
  once when more than 5% of the draws are guarded. The likelihood itself is
  unchanged; `guard_report=False` skips the check and an integer sets the
  number of draws.
- `ds.infer(..., compute_dtype="float32")` (or `bind_analysis(...,
  compute_dtype="float32")`) is an opt-in speed option: the per-sample PE and
  selection weights are evaluated in float32, while every reduction
  (log-sum-exp, Monte-Carlo variances, `N_eff`, soft guard, final sum) and every
  per-proposal grid stays float64. The default (`None` or `"float64"`) is the
  float64 program, bit for bit. The float32 likelihood is value-only, for the
  nested samplers (dynesty, TinyNS): differentiating it raises, and
  `sampler="numpyro"` refuses it, because float32 population densities
  underflow in the tails, where their gradients are NaN. It covers spectral and
  incomplete-catalog analyses with an isotropic angular model and a chi_eff
  population. Bright
  and complete-catalog analyses, angular models, component spins and
  Gaussian-process populations are refused. The GP code keeps its covariance
  algebra in float64, so float32 gains nothing there.
- Ordinary `ds.infer` results carry `log_prior_volume_fraction` and
  `logZ_corrected` next to the raw `logZ`, so evidences from angular models
  that reject part of their prior box are comparable.
- Every `ds.infer` result says how the sampler stopped: `stop_reason`
  (`"convergence"` when the `dlogz` criterion held, `"maxcall"`/`"maxiter"`
  for a budget cap), the final remaining-evidence estimate `dlogz_final`, and
  the `ncall`/`niter` counters (`None` where a backend has no such notion).
- `ParameterPlan` prior kinds and joint-constraint kinds are closed
  vocabularies; anything else raises instead of being reinterpreted. An
  `InferenceTarget` plan must spell out `loc` and `scale` for every
  non-uniform prior (core's own `ParamSpec` may leave them `None`, meaning the
  standard `(0, 1)` defaults).
- The opt-in `DARKSIRENS_GW_PAIRING_M1_GRID` normaliser grid is sized to the
  bound model's mass support at bind time and refused if it cannot cover it.
- **Speed defaults (since 2026-10-02).** Four evaluation settings default to
  the faster evaluation. Each keeps its historical value, which reproduces a
  run made before the change bit for bit (optimized program and likelihood)
  and resumes its checkpoint:

  | setting (env) | default | historical value |
  |---|---|---|
  | `configure_normalization_grids(pairing_norm=...)` (`DARKSIRENS_GW_PAIRING_NORM`) | `"auto"` | `"per_sample"` |
  | `configure_catalog_evaluation(kernel_layout=...)` (`DARKSIRENS_CATALOG_KERNEL_LAYOUT`) | `"galaxy_list"` | `"padded"` |
  | `configure_catalog_evaluation(missing_density=...)` (`DARKSIRENS_CATALOG_MISSING_DENSITY`) | `"auto"` | `"grid"` |
  | `configure_catalog_evaluation(kernel_window=...)` (`DARKSIRENS_CATALOG_KERNEL_WINDOW`) | `"auto"` | `"off"` |

  (`configure_catalog_evaluation` is in `darksirens.catalog.settings`, and
  `configure_normalization_grids` in `darksirens.population.utils`; set them,
  or the environment variables, before binding.) `"auto"` means "the faster
  evaluation wherever it applies, the historical one otherwise", without
  raising; the explicit faster value keeps refusing what it cannot serve.
  - `pairing_norm="per_point"` integrates the pairing normaliser's
    secondary-mass taper once per likelihood point instead of once per PE
    sample and injection, for `PowerLawPairing` and
    `GWTC5FiducialBPL2PeaksPairing`. Above the taper shoulder it is the
    per-sample quadrature rule evaluated once (equal to rounding); inside the
    taper window it reads a small per-point table that is closer to a
    converged reference than the per-sample rule. `"auto"` takes it wherever
    it applies, and keeps the per-sample rule for other pairings and when the
    opt-in pairing m1 grid is set (an explicit `"per_point"` refuses the m1
    grid).
  - `kernel_layout="galaxy_list"` evaluates the per-galaxy kernel normaliser
    on the real galaxies only, not on the padded catalog; the values are the
    same bit for bit. It refuses nothing, so it has no `"auto"`.
  - `missing_density="gather"` evaluates the missing-host density at each
    sample's two bracketing redshift nodes instead of materialising
    `(N_rows, 1000)` grids per proposal, with the same arithmetic. `"auto"`
    gathers wherever `"gather"` would, and keeps the grid with a missing-host
    extension, which `"gather"` refuses.
  - `kernel_window=eps` sums each GW sample's catalog kernel over a
    fixed-length window of its pixel's redshift-sorted galaxies instead of
    all of them. The window is sized at bind time (at the fixed `sigma_kde`,
    or at its prior's upper edge when it is sampled) so that, at every
    redshift, the part of the sum it leaves out is at most `eps` times the
    pixel's largest single-galaxy peak term; a traced check makes the
    likelihood `-inf` (never a truncated value) if the window's premises fail
    at a proposal. It applies to incomplete, complete and field-weighted
    catalog analyses, with the kernel pin and `compute_dtype="float32"`. See
    `darksirens.catalog.redshift.kernel_window` for the bound. `"auto"` is
    `eps = 1e-10` on every catalog view whose rows are sorted by redshift
    (the `load_catalog` default) and no window on one that is not, where an
    explicit tolerance refuses; the marked host kernel drops a window
    attached under `"auto"` and refuses an explicit one.

  Against the historical values the defaults move the float64 log-likelihood
  by at most 1.4e-7 (relative 2.3e-13) on the real 259-event spectral
  likelihood and 4.0e-8 on the mock dark-siren cases, all of it from the
  per-point pairing normaliser. The defaults are recorded in the run
  fingerprint and the historical values are not, so resuming a checkpoint
  written before 2026-10-02 under the new defaults is refused as a settings
  change; the refusal names the settings that resume it (see `MIGRATION.md`).

## Modelling assumptions

### Photometric redshifts: the smooth prior on true redshift

**The kernel.** Every catalog analysis (complete, incomplete, field-weighted
mixture, marked hosts) places each galaxy `i` in redshift with the unit-mass
kernel of `darksirens.catalog.redshift`:

```text
p(z | galaxy i) = N(z; z_i, sigma_i) g(z) / Z_i,    g(z) = dV_c/dz (1 + z)^delta,
sigma_i = max(sqrt(dz_i^2 + sigma_kde^2), 1e-4),    Z_i = integral of N(z; z_i, sigma_i) g(z) dz,
```

with `z_i` and `dz_i` the catalog's stored redshift and its uncertainty. Read
as Bayes' theorem for the galaxy's true redshift `z`, this is a Gaussian
redshift likelihood (width taken at the stored redshift) times a prior
`g(z)`: the smooth comoving volume, the same along every line of sight. The
host density of a sky pixel is the sum of its galaxies' kernels. This is the
frozen legacy kernel, and core keeps it bit for bit.

**What it neglects.** The true redshifts of the galaxies along a line of sight
`p` are clustered, `pi_p(z) = g(z) [1 + delta_p(z)]`, with structure on
redshift scales of 0.01 or less. Expanding the prior across the kernel, a
prior `pi` displaces the kernel's mean from the stored redshift by

```text
Delta z ~ sigma^2 d ln(pi)/dz.
```

With `pi = g` every galaxy moves behind its stored redshift by
`sigma^2 d ln(dV_c/dz)/dz` (about `2 sigma^2 / z` at low redshift), uniformly
over the sky; the exact posterior instead pulls each galaxy toward the
structure it belongs to. The per-galaxy error is
`sigma^2 [d ln g/dz - d ln pi_p/dz]`, so the bias it causes in `H0` scales as
`sigma^2`: halving the redshift error quarters it. Its scatter averages down
over the many galaxies an event sees; its sky-averaged mean does not, and it
does not shrink with more events, because every event carries it. Its sign
and size for `H0` depend on how each event's distance posterior samples the
structures, and have to be measured.

**When the approximation is safe.**

- The structure along each line of sight is smoother than the kernel width
  `sigma_z (1 + z)`, so `d ln(1 + delta_p)/dz` is small across a kernel (weak
  clustering, or redshift errors much narrower than the structures).
- The galaxies have spectroscopic redshifts (`dz` of order `1e-4` to `1e-3`):
  at `dz = 1e-3` the prior's shift is about 300 times smaller than for a
  photo-z of `0.015 (1 + z)`. See the guard interaction below before using
  them.
- `sigma_kde` adds in quadrature to every `sigma_i`, so a smoothing width
  enlarges the shift exactly as a wider photo-z would.

**Measured size on clustered mocks.** The `darksirens-examples` quick campaign
(150 events, a complete catalog with field sky weighting at `nside = 16`, a
lognormal universe in redshift shells of 0.01 with an AR(1) correlation length
of 0.02, photo-z errors `sigma_z (1 + z)` with `sigma_z = 0.015`, 41 seeds)
gives these offsets of the `H0` posterior mean from the truth, in km/s/Mpc
(mean over seeds and its standard error; a posterior standard deviation is
about 5.5):

| likelihood | catalog depth | `H0` offset |
|---|---|---|
| core kernel, `sigma_z = 0.015` | 0.85 | `-1.65 +- 0.74` |
| exact kernel (the mock's photo-z likelihood times its true clustered prior) | 0.85 | `-0.39 +- 0.62` |
| core kernel, same noise draws at `sigma_z = 0.0075` | 0.85 | about `-0.25` |
| core kernel, `sigma_z = 0.015` | 0.95 | `-1.34 +- 0.96` (P-P KS p = 0.01) |
| core kernel, same galaxies and draws at `sigma_z = 0.005` | 0.95 | `+0.39 +- 0.61` (KS p = 0.58) |
| paired difference, 0.005 minus 0.015, same seed | 0.95 | `+1.73 +- 0.64` |

The shift is about a third of a posterior standard deviation per analysis; the
exact clustered prior removes most of it, and halving the width shrinks it by
at least the factor four `sigma^2` predicts. Two other ingredients were
measured on the same seeds and are small: taking the width at the stored
redshift instead of the true one (`dz = sigma_z (1 + z_obs)`, as real data
must) contributes `+0.10`, and the `kernel_window` speed setting contributes
nothing. Every analysis that puts a photometric galaxy on a smooth
comoving-volume prior makes this approximation; core has no catalog-informed
alternative today.

**Narrow kernels and the likelihood-variance guard.** The guard that bounds the
Monte Carlo variance of the total log-likelihood
(`max_likelihood_variance`, 1 nat squared by default;
`darksirens.selection.gw`) includes each event's PE reweighting variance.
When the kernels are narrow, few of an event's PE samples land on a galaxy's
kernel, that variance grows, and the guard returns `-inf`, first at low `H0`,
where the nearest events map to the lowest redshifts and a pixel holds few
galaxies. On the same campaign at depth 0.95:

- `sigma_z = 0.005` loses low-`H0` points on 24 of 41 seeds: on 21 only at
  `H0 <= 50.5`, where the `sigma_z = 0.015` posterior has at most 0.2% of its
  mass; on 3 up to `H0 = 66.5`, which truncates the posterior;
- the narrowest width accepted at `H0 = 40` on every seed is `0.0145`, and on
  one seed the guard removes `H0 < 56` even at `0.015`;
- spectroscopic widths (`1e-4`) make the likelihood `-inf` at every `H0` in
  `[40, 100]` (depth 0.85, all five seeds tried).

The cut does not raise: it shows only as `-inf` values on a likelihood grid or
as a truncated posterior. Before trusting a narrow-kernel catalog analysis,
evaluate the likelihood across the `H0` prior; more PE samples per event
lower the variance, while widening the kernels (`sigma_kde`) brings back the
smooth-prior shift above.

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
