# darksirens-core migration record

Frozen legacy reference:

```text
ignaciomagana/darksirens@c042527238bd71421b792936bc48c3b815b90d6d
```

The reconstruction rule is parity first: freeze behavior, port the smallest
coherent unit, prove fixed-coordinate parity, then simplify architecture only
behind the same gates. The frozen legacy repository is read-only.

## Accepted core phases

| Phase | Scope | Integrated/accepted state |
|---|---|---|
| 1 | immutable cross-domain reference bank and replay harness | frozen reference assets |
| 2 | JAX runtime, cosmology, redshift grid, standardized GW stores/loaders | `450b9bdb66d2dc2d6e7f927143f9b4b4b9f9cec6` |
| 3 | parametric/GP population models and registry | `e0b40fef65261a27b67aa9657a97216df3e8444f` |
| 4 | spectral hierarchical likelihood and GW selection | `0f97feff7eb283a1f541bef9a776c9347084e70e` |
| 5 | ordinary catalog runtime, completeness, dark/complete/bright likelihoods and generic hosts | `86e0c88a51482d17fac70f111057d277df9387fd` |
| 6 | inference I/O, priors, samplers, checkpoints, results and provenance | `d82becaf76bf62c0f72a71b32ebbf9b238ba4f13` |
| 7 | public loaders/specs/model/runtime binding/infer plus reusable angular models | `6ed3dc74aa0fcde4da128036cc3d56c29250d370` |
| 8A | public counterpart/bright-siren composition | `5256141e00a2d96a72b9cbf2182a77174c8c269e` |
| 8B | explicit sampler-facing `InferenceTarget` | `2e98ca1f1a67cf688f8da8444c3747d446a4e91d` |
| 8C | explicit host-density/redshift extension seam | `875a949d5a9f5ffb89f3a64cad030dcfc6daf6a2` |
| 12A | generic `CompletionCurves` composition seam | PR #8, squash merge `8b9dc64` |
| 12B | footprint-aware field catalog estimator | PR #9, squash merge `bb4812d` |
| review follow-up | guards, independent anchors, inherited-defect fixes | PRs #10 (`4c0e960`), #16 (`8bb1fc2`), then #12, #13, #14, #15 in order |

Exact per-slice heads, trees, workflow runs and parity probes are recorded in
`ignaciomagana/darksirens-rebuild/phases/`.

## Final ownership after reconstruction

### Core

The reconstructed package owns reusable scientific/runtime machinery:

```text
cosmology
standardized GW PE/injection data
population models
angular population models
standardized ordinary galaxy-catalog runtime
ordinary redshift kernels/completeness/host weights
counterpart objects
GW selection
hierarchical likelihoods
parameter/prior assembly
sampler adapters
checkpoints/results/provenance
explicit extension seams
```

### Surveys companion

The future survey package owns native survey products and their construction:

```text
DESI/KIBO/Legacy/GLADE native schemas
raw catalog ingestion
masks/depth maps
survey weights
survey selection/completeness fitting inputs
survey-specific preprocessing
```

### LSS companion

The future LSS package owns specialized density-field/count state:

```text
Q_LSS / Q ensembles
latent fields
counts and tracer likelihoods
multitracer state
field construction and specialized provenance
```

It may use the core host-density seam but core never imports it.

### Lensing companion

The future lensing package owns lensing-specific physical state and likelihoods.
It may expose its likelihood to core sampling through `InferenceTarget`; core
never learns lensing classes or dispatch names.

## Deliberately not reconstructed

The old monolithic ownership pattern is not a migration target. In particular,
core does not recreate:

- a giant `universe_model`/parameter-decoder switchboard;
- generic plugin discovery;
- raw survey construction inside the inference package;
- LSS or lensing state hidden inside generic core containers;
- campaign-specific mega CLIs.

## Phase 8 final freeze

Phase 8D is packaging/API/install/example closure only. It must make no
scientific arithmetic change relative to accepted 8C. Its acceptance is recorded
in the rebuild control repository; after merge, the resulting `main` tree is the
core baseline from which companion repositories may begin.

The release workflow's check that no science source changed since accepted 8C
was retired once Phases 12A and 12B changed that source on purpose; the other
release checks now run on every pull request (see `VALIDATION.md`).

## Post-freeze work

Phases 12A and 12B added two narrow composition seams (completion curves;
footprint fractions and the field numerator) without touching the accepted
evaluators. Both were accepted through PRs #8 and #9 with control records in
`ignaciomagana/darksirens-rebuild` (`phases/12A_*`, `phases/12B_*`). Neither has
a dedicated workflow; both are covered by the broad regression suite.

The review follow-up implemented the confirmed findings of an adversarial parity
review against the frozen reference, as a stack of PRs merged in order. It is
deliberately not parity-neutral in four places, each chosen over the frozen
behavior for a stated reason:

- the complete-catalog empty-row default returns to the frozen `zero`; the
  reconstruction had flipped it to `volume`, the one true parity drift found;
- the GP z-conditional normaliser is tabulated in `log1p(z)` on 145 nodes
  instead of uniformly in z on 40, and the m1-conditional q-normaliser follows
  the sampled taper toe, and the coarse (q, chi) lattice on the m1-conditional
  branch tabulates q on 128 nodes instead of 24. All three defects are
  inherited from the reference; the z-node change costs roughly 3x per
  proposal on z-conditioned GP models, the other two nothing measurable. The
  128-node lattice fixed the recorded point (m1 = m_min + 0.25 dm_min at the
  fiducial taper) but not the narrow allowed mass-ratio range just above
  `m_min` in general; the second follow-up below fixes that;
- `ang2pix_ring` now matches healpy at ring boundaries, the polar caps and
  right ascensions just below a multiple of 2 pi;
- inputs the reference refused and the reconstruction had silently accepted
  (out-of-grid cosmology priors, unknown prior or constraint kinds, mismatched
  PE/injection contracts, uncentered or view-dependent host marks, a short
  redshift-parameter vector, a non-divisible selection batch) now raise.

Everything else in those PRs is a test or a probe: independent anchors against
astropy, healpy, scipy quadrature and hand-computed expectations; real TinyNS,
Dynesty and NumPyro runs; a Phase 4 spectral probe that now covers the
out-of-table distance mask; and coverage for the `@md` rate evolution and the
`in_prior_v2` fiducial set.

### Second review follow-up

A second stack of PRs (#17 to #23, merged in order) finished the items the
first follow-up had left open. Only one of them changes numerics:

- **GP mass-ratio normaliser just above the minimum mass (changes numerics).**
  For `gp1d_q`, `gp2d_q_chi`, `gp2d_q_z` and `gp3d_q_chi_z` the density over
  mass ratio (and spin) at fixed primary mass must integrate to 1. The 128-node
  mass-ratio lattice gave 1.0005 at the recorded point, but closer to `m_min`
  (0.81, and 0.58 for `gp3d_q_chi_z`, at m_min + 0.05 dm_min) and for a narrow
  taper inside the priors (m_min = 10, dm_min = 0.05: 0.055 to 0.085 at
  m_min + 0.25 dm_min) all four models were wrong. Two errors of opposite sign
  were at work: a fixed mass-ratio grid cannot follow the narrow allowed range
  just above `m_min`, and interpolating the log-normaliser across the sharp
  low-mass taper was inaccurate. The normaliser now integrates mass ratio on
  nodes that span the allowed range itself and tabulates the normaliser divided
  by the taper, which is smooth. Every probe at m_min + 0.1 dm_min or above is
  now within 0.4% of 1 (m_min + 0.05 dm_min: within 1.1% at the fiducial taper,
  3.9% at dm_min = 0.05). Values away from the taper move at the 1e-4 level.
  `gp1d_q` is about 2.5 times slower on the value alone (a few milliseconds for
  20,000 queries on CPU) and about 1.3 times on value plus gradient; the other
  three models cost about the same or less.
- **Run fingerprints (no likelihood change).** Fingerprints now hash a canonical
  form of the semantic block: arrays as dtype, shape and every value at full
  precision, tuples as lists, numpy scalars as Python scalars, NaN and infinity
  written out, and anything else refused instead of hashed as text. The old
  rule hashed arrays at print precision, so the resume gate could accept a
  different target. The schema moves from 3 to 4 and older checkpoints are
  refused with a message saying why. Plain-JSON semantic blocks hash exactly as
  before. Semantic blocks with non-string keys, sets, complex numbers,
  datetimes or bytes now raise. `core_numerics_semantic()` and
  `core_environment_advisory()` record the settings core resolved at import
  from `DARKSIRENS_*` variables; `infer()` calls neither, nor the resume gate
  (for `InferenceTarget` runs it now does: see "Run-fingerprint coverage"
  below).
- **Missing-galaxy budget count check (no likelihood change).**
  `darksirens.selection.catalog.selection_budget_audit` ports the frozen
  reference's diagnostic: under parametric magnitude selection it compares the
  catalogued count the model predicts with the real count. It is not exported
  from the package root and nothing calls it.
- **Nested-sampler startup check message (no numerics change).** Its advice now
  names `infer(..., selection_neff_guard="soft")`,
  `infer(..., max_likelihood_variance=<cap>)` and
  `infer(..., sampler_preflight="off")` instead of the old command-line flags,
  and no longer points to a diagnostic script core does not ship.
- **Release checks and parity probes (no numerics change).** The release
  workflow runs again, on every pull request, against the installed wheel. The
  angular-wiring and sampler-dispatch parity probes now run the frozen
  reference code on the reference side instead of a copy written in the probe.

None of the items was found already fixed: the mass-ratio normaliser was
fixed only at the recorded point, and the other items were open.

### Partial fixing (additive API, no likelihood change)

The frozen reference fixed any subset of parameters with
`--fixed_parameter_values`; core could only fix the whole population (a preset)
and always sampled `log10n0`, `delta` and `sigma_kde`. Two keywords close that
gap without changing any scientific definition:
`Population(name, fixed={parameter: value})` and
`model(..., fixed_survey={name: value})`. The fixed parameters leave the
sampled plan, their values are constants of the bound likelihood, and the
decoded parameters equal the all-sampled decode at the same point bit for bit.
The jitted likelihood agrees with the all-sampled one to a few ulp (at most
4.5e-15 relative on the test fixtures), because XLA may fold the constants;
evaluated op by op it is bit-identical. Keys are the plan labels, as in the
reference, or a parameter's ASCII name. Unlike the reference, a fixed value
outside the parameter's prior bounds raises by default;
`model(..., allow_out_of_prior=True)` gives the reference's behavior for
default bounds (accept and warn), for ablations such as
`log10n0 = log10(5e-5)`. With or without it, fixed values that leave a joint
prior no support raise.
`ParameterPlan` gains `fixed_population_values`, `fixed_survey` and
`allow_out_of_prior`, and `run_fingerprint.parameter_plan_semantic(plan)`
records them for the resume gate.

### Sampler termination fields (additive result fields, no sampling change)

The frozen reference returned a Dynesty result without saying whether the run
stopped on its `dlogz` criterion or on its `max_samples` call cap, so a caller
could not assert convergence. Every sampler result now carries
`dlogz_final`, `stop_reason`, `ncall` and `niter`
(`nested_output.termination_record`). For Dynesty the adapter rebuilds the
stop check of dynesty 2.1.4's sampling loop from the sampler's state after
`run_nested`, using the loop's own running evidence, so `dlogz_final` is
bitwise the number dynesty compared with `dlogz`. It adds no attribute to the
sampler and calls `run_nested` exactly as before: samples, `logZ`, `logZerr`
and dead points are bitwise unchanged at the same seed. TinyNS reads the same
fields from its diagnostics; NumPyro and the zero-free result set them to
`None`. The Phase 6N/6O/6S/6T parity probes require these four fields on
every candidate result and then drop them (`tools/_termination_fields.py`),
so the comparison with the frozen reference stays exact on everything else.

### Catalog kernel pin (numerics change on pinned plans only)

With `Om0`, `w0`, `wa`, `delta` and `sigma_kde` fixed, the galaxy measure
`g(z) = dV_c/dz (1+z)^delta` depends on `H0` only through the factor
`(H0_ref / H0)^3`, and the kernel widths not at all, so every galaxy's kernel
normalisation moves by the same scalar. The frozen reference uses this: when
none of the five is sampled it evaluates the per-galaxy 24-node quadrature once
per run, at `H0_ref = 67.74`, and adds `3 ln(H0 / H0_ref)` per call (its H0
kernel pin). Core now does the same, by default:
`model(..., kernel_pin="auto")` pins an incomplete-catalog analysis whose plan
samples none of the five (`H0` may be sampled). The quadrature, kernel,
masks and completeness are unchanged; only the kernel state moves to bind
time, and the completeness curves are still evaluated per call. Like the
reference, each call rebuilds eight catalog rows from the live parameters and
turns the likelihood into `-inf` if they disagree with the pin by more than
1e-9 (they agree to about 1e-14).

The per-call probe re-derives eight rows, so a pin served with another
catalog of the same shape that differs only elsewhere gives a finite, wrong
likelihood (in `tests/test_kernel_pin.py`, with one galaxy moved by -0.01 in
redshift on a row the probe does not rebuild, the stale pin gives a value
1.5e-5 from the correct one at H0 = 30). The pin therefore carries
`catalog_digest`, a blake2b digest of the catalog arrays the kernel
reads (`zgals`, `dzgals`, `wgals`, `ngals`: dtype, shape and every byte,
padding included), the fixed `Om0`, `w0`, `wa`, `delta`, `sigma_kde` and
`z_depth`, the pin's `H0_ref` and probe rows, and the kernel's static
settings (redshift grid, quadrature nodes, width floor, padding sentinel,
distance-table grid, interpolation switches; not the distance table's
values). `build_pinned_catalog_kernel` computes it on the host
(`catalog_kernel_pin_digest`), and `BoundAnalysis` recomputes it from its own
catalog and plan with `check_pinned_catalog_kernel` whenever it is made,
replaced or unpickled, and refuses a mismatch with a `ValueError` naming both
digests. `BoundAnalysis.kernel_pin_digest` exposes it; the plan does not carry
the pin, so `parameter_plan_semantic` does not record it. The digest is
pytree metadata, not a leaf, and pins that differ only in it have equal tree
structures, so a compiled program serves a pin built from other data without
retracing (and a pin returned by a jitted function keeps the digest that
function was first traced with: build pins outside a jit). The jit operands,
the lowered program and every pinned value are unchanged (on fixture T,
three pinned plans, single pass and 4096/6: 66 values, 24 gradients, the 42
pin arrays and the 6 lowered programs bit for bit). It costs 7 ms on T and
0.34 s on a 196,608 x 70 catalog on CPU, twice per `bind_analysis` (at the
build and at the check).

The pinned likelihood is not bit-identical to the per-call quadrature: the two
round differently. On the harness fixtures T and S (plans `dark_H0`,
`dark_pop`, `dark_joint_cosmo_pop`, single pass and blocked) the total log
likelihood agrees to 7.6e-16 relative, each event's log evidence to 4.4e-16,
the Monte Carlo variance diagnostics to 1.2e-13 relative and every
per-sample catalog log-density to 5.7e-14 absolute; against the
reference, which pins, the total agrees to 2.9e-16 relative (bit for bit in 8
of 12 cells, against 4 of 12 unpinned). On CPU a call is 1.4 to 6.7 times
faster (T `dark_H0`: 15.8 ms against 105.7 ms), for about 1 s more at bind.
The pin keeps two `(n_rows, n_max)` float64 arrays (the fused kernel
log-weights and the inverse widths), three per-row vectors (the row offsets,
the empty-row flags and the depth masses) and eight probe-row indices (224 MB
on a 196,608 x 70 catalog); the reference keeps five such arrays. The pin is built
by the per-call kernel builder, run once at bind, so binding needs the
transient memory of one unpinned kernel build: on that catalog the CPU
compiler gives the build 14.9 GB of scratch, while the per-call catalog terms
need 3.2 GB pinned against 15.4 GB unpinned. Complete-catalog analyses are
not pinned, as in the reference. `kernel_pin="off"` keeps the per-call
quadrature, and the bound program is then byte-identical to the previous
release's. `ParameterPlan` gains `kernel_pin` and `kernel_pin_active`, and
`run_fingerprint.parameter_plan_semantic(plan)` records both.

### Catalog kernel pin on the field seam (opt-in, per target)

A field (host-density) target builds its prior state per proposal through
`darksirens.catalog.field.build_field_incomplete_catalog_prior_state_from_curves`,
which now takes `pinned_kernel=` as the conditional seam does. The DESI P12.4
target of `desi_darksirens_selection` is one: it samples `H0`, `M0hat` and
`sigma_M` and fixes `Om0`, `w0`, `wa`, `delta`, `sigma_kde` and `z_depth`, so
its catalog kernel moves only through `3 ln(H0 / H0_ref)` while its completion
curves still move with all three. Such a target can build the kernel once with
`build_pinned_field_kernel` (the ordinary path's builder: `H0_ref = 67.74`,
eight probe rows, tolerance 1e-9), pass it to its jitted evaluation as an
argument, and hand it to the seam on every call; `field_kernel_pin_applies`
is the activation rule (sampled labels, never values) and
`field_kernel_pin_plan(plan, setting)` records `kernel_pin` and
`kernel_pin_active` on the target's plan for `parameter_plan_semantic`
(`combine_parameter_plans` keeps returning neutral metadata). The seam refuses
a pin built for another catalog shape, and a concrete pin under a traced
catalog.

Nothing changes for a target that passes no pin: its lowered program is
byte-identical to the previous release's. With a pin the values move by
rounding only. On the consumer's synthetic P12.4 fixture and on the campaign
fixtures T and S through a field-style target (single pass and the P12.4
blocks 131072/32, hard and soft guards, with and without a depth), against the
per-call quadrature: the total log likelihood within 8.8e-16 relative, each
event's log evidence within 7.1e-15 absolute, `n_eff` within 1.6e-14, the
Monte Carlo variance diagnostics within 2.8e-13 relative, and every
per-sample field log density within 1.2e-13 absolute. On CPU a call on T and
S is 2.4 to 13 times faster (T, single pass: 15 ms against 134 ms), for 0.7
to 1.9 s more at build; on the small consumer fixture at the P12.4 blocks the
padded 131072-injection selection pass dominates and the time is unchanged. A
pin belongs to the catalog view and the distance table it was built from: a
target that evaluates other data must rebuild it. The probe returns `-inf`
when the difference reaches its eight rows (a pin built under another
premise, or a catalog or table rescaled as a whole), but a catalog of the
same shape that differs only outside the probe rows keeps a stale pin that
the probe need not detect, as on the ordinary path. The field pin therefore
carries the ordinary path's `catalog_digest` (the same builder, the same
digest), and `darksirens.catalog.field.check_field_kernel_pin(pin, cosmo,
params, catalog)` recomputes it on the host from the catalog view and fixed
premise the target serves the pin with, and refuses a mismatch with a
`ValueError` naming both digests. Inside the target's jit the catalog is
traced and cannot be hashed, so a target calls the check where it attaches
the pin to its catalog view (its jit operands), outside the jit, and again
whenever it replaces either; an eager call of the seam (nothing traced) runs
the check itself. The digest is not a jit operand and splits no jit cache: a
target's compiled program is unchanged and still serves a rebuilt pin without
retracing.

### Run-fingerprint coverage (no likelihood change)

Two provenance gaps are closed; no likelihood value changes, and the
fingerprint schema stays 4 (the canonical hashing is unchanged; only new,
optional entries are added).

- **Likelihood options.** `core_numerics_semantic(likelihood=None)` takes the
  `BoundAnalysis` (or a mapping of `bind_analysis` likelihood options) and adds
  `"likelihood_options"` with each value-changing option that differs from its
  default: `compute_dtype` (`"float32"`), `max_likelihood_variance` (a cap
  other than 1.0) and `selection_neff_soft_guard` (`True`).
  `likelihood_options_semantic` returns that entry alone. `sel_batch_size` and
  `pe_event_block` are layout: the blocked passes compute the same log-sum-exp
  sums (padding rows carry `-inf`, `Ndraw` is unchanged) and differ from the
  single pass by reassociation only (the pinned 1e-12 relative contract; 4e-16
  on the DESI P12.4 target, consumer Phase 12J), so recording them would only
  make a requeue with another memory budget refuse its own checkpoint, as the
  frozen reference's exclusion of both says. A default binding adds no entry:
  default digests equal those of the previous release (pinned in
  `tests/test_run_fingerprint_coverage.py`).
- **InferenceTarget runs.** Before, no core module called the resume gate, so
  `ds.infer` on a target wrote no fingerprint and resumed any checkpoint. A
  target run that writes or resumes a dynesty or TinyNS checkpoint is now
  fingerprinted (parameter plan, core numerics, sampler settings, and the
  target's new keyword-only `identity` and `provenance` hook), the fingerprint
  is written beside each checkpoint the run writes, and a resume is refused on
  a mismatch unless `resume_force=True`. A checkpoint written before this
  release has no fingerprint and is refused once; `resume_force=True` resumes
  it and stamps the current fingerprint, after which resumes are gated
  normally. A target run without a checkpoint is unchanged. This is a
  behaviour change for resuming existing target checkpoints, for example the
  DESI P12.4 target of `desi_darksirens_selection`, which checkpoints dynesty
  through `ds.infer`: after it adopts this release, its first resume of a
  checkpoint written before must pass `resume_force=True` once. Its likelihood
  options (`selection_neff_soft_guard=True`, `max_likelihood_variance=20`)
  belong to its target and enter the fingerprint only if it puts them in its
  `provenance`; `core_numerics_semantic()` and `parameter_plan_semantic` are
  unchanged for it.
- **Not covered.** `ds.infer` still writes no fingerprint for an ordinary
  analysis: core does not hash the event, injection or catalog files, so a
  caller that checkpoints an ordinary analysis composes its own fingerprint
  (`parameter_plan_semantic`, `core_numerics_semantic(bound)`, its data
  identity, e.g. `file_identity`).

### Public parameter decoding (additive API, no likelihood change)

Companion packages (`darksirens-lss`, `darksirens-surveys`, the DESI
consumer) decoded coordinate vectors with copies of core's private
`runtime_binding._decode_theta`, so any change to it would silently diverge
from them. `ds.decode_parameters(analysis, theta, *, z_depth=...)` is now the
public form: it takes a `ds.model` analysis or a `BoundAnalysis` and returns
the immutable record `runtime_binding.DecodedParameters(cosmology,
population, catalog, angular)`, with the plan's fixed values, the
`fixed_survey` constants and the `n0_units` conversion applied, and `z_depth`
defaulting to the depth `bind_analysis` uses. The private decoder now returns
the same record (it unpacks as the old 4-tuple) and the bound likelihood
still calls it, so there is one implementation: on a clean `main`
comparison the default likelihood's optimized HLO and logL are bitwise
unchanged. The package-root surface gains `decode_parameters`; see
"Parameter decoding" in `CONTRACT.md` for the stability promise.

### Selection completeness in `ds.model` (opt-in, default unchanged)

`ds.model(..., completeness="selection", selection=<model or payload>,
row_fraction=None)` is the ordinary path's form of the frozen reference's
`c_mode="selection"`: every catalog row's completeness is the radial
magnitude-selection curve `C_sel(z)` (Gaussian or Schechter luminosity
function, `darksirens.selection.catalog`), optionally times a coverage
fraction per row (`darksirens.selection.footprint`), in place of the per-row
count ratio. It reuses the accepted curve builders and the incomplete
catalog's conditional prior state, so the kernel, depth convention, kernel
pin, `n0_units`, `missing_density="gather"` and float32 weights all work as
for `completeness="incomplete"`. The selection nuisances (`M0hat`/`sigma_M`
or `Mstar_hat`/`alpha`) are fixed at the selection model's values unless the
new `survey_priors={label: prior}` samples them (the reference sampled them
by default, flat or under the fit's normal prior; core makes that explicit).
`survey_priors` also overrides the bounds or sets a truncated-normal prior of
`log10n0`, `delta`, `sigma_kde` for any catalog analysis.
`CatalogParameters` gains a `selection` field (default `None`), which
`decode_parameters` fills for a selection analysis; `ParameterPlan` gains
`catalog_model`, the canonical JSON of the selection payload and row-fraction
digest, which `parameter_plan_semantic` records only when set. Parity with the
reference's `darksiren_log_likelihood(..., c_mode=selection)` on a synthetic
fixture: 3e-14 (`tools/probe_selection_completeness.py`, in the phase-5
workflow). Every default plan, fingerprint and bound program is unchanged
(optimized HLO and logL bit for bit against a clean `main`).

### Field-weighted multi-catalog mixture (opt-in, default unchanged)

`ds.model(catalog=[A, B, ...], catalog_sky_weighting="field")` is the
ordinary path's form of the frozen reference's `universe_model="dark_sirens"`
with `n_catalogs=K`, `catalog_sky_weighting="field"` and stick weights
`fcat_k`. Each catalog's host density in a sky row is the field numerator
`N_obs p_cat + dN_miss` of `darksirens.catalog.field`, divided by its
survey-global total `Z_k` (the full-sky sum of the same numerator), and the
catalogs combine as `logsumexp_k [log w_k + log n_k - log Z_k]` per sample,
for the PE and selection terms alike. Labels, priors and order follow the
reference (`log10n0`, ..., `log10n0_c2`, ..., `fcat_2 .. fcat_K` with
`Beta(1, K - m + 1)`), and `completeness="selection"` gives each catalog its
own selection model. Differences from the reference, all deliberate: each
catalog keeps its own HEALPix resolution and compact view (the reference's
loader does the same); `Z_k` is computed in float64 throughout (the
reference stores the count-ratio normaliser's observed density in float32;
on the probes and mock T the two agree to 3e-14 anyway); with one catalog
`Z` is not evaluated, since it cancels (the reference computes it); and
the reference's aggregate completeness, strata, LSS `delta_g`, marks, Q tables
and latent fields are not in core: a companion modulates the missing-host
density through `MissingHostExtension` and
`darksirens.likelihood.mixture.make_catalog_mixture_target`. New public
pieces: `darksirens.catalog.mixture` (normaliser forms, stick weights,
extension protocol and context), `darksirens.likelihood.mixture`,
`CatalogMixtureParameters`, `ParameterPlan.catalog_model` for the mixture,
`models.assemble_incomplete_catalog_prior_state` (the second half of the
accepted state constructor, factored out unchanged). Every default plan,
fingerprint and bound program is unchanged (optimized HLO and logL bit for
bit against a clean `main`).

### Per-catalog completeness in the field mixture (opt-in, default unchanged)

`ds.model(catalog=[A, B], catalog_sky_weighting="field",
completeness=["incomplete", "selection"], selection=[None, fit_B])` gives
each catalog its own completeness: the per-row count ratio, the
magnitude-selection curve or `"complete"`. The frozen reference's kernel takes
one `SurveyParams` per catalog with its own `c_mode`, so a count-ratio plus
selection mixture is its `darksiren_log_likelihood(..., mixture_surveys=...)`
with per-catalog `c_mode` (its CLI broadcasts one `--c_mode`); the two agree
to 4e-14 on the probe and on split mock T. `"complete"` is the reference's
field-weighted `universe_model="dark_sirens_complete"` (`n_k = N_obs,p
p_cat`, `Z_k = sum_p N_obs,p`, no survey depth), which the reference allows
only for every catalog at once; core also mixes it with incomplete catalogs.
`field_normalizer` takes one value or one per catalog, and `"auto"` is chosen
per catalog. One value, or a list of equal entries, gives the plan,
fingerprint and bound program of today (optimized HLO and logL bit for bit
against a clean `main`).

### Per-catalog population blocks (opt-in, default unchanged)

`ds.model(catalog=[A, B, ...], catalog_sky_weighting="field",
per_catalog_population={2: ["G.mu", "mu_chi"]})` is the form of the
reference's tracer-dependent population blocks (legacy commit af896ca,
`per_catalog_pop_params=("G.mu_c2", "mu_chi_c2")` with `mixture_pop_params`,
the code of the gws-agn Analyses 8 to 13). Catalog `k` carries its own copy
of the named population parameters, labelled `"<label>_c{k}"` (for example
`"$\mu_{\rm G}$_c2"`), with the base parameter's prior; every other entry is
catalog 1's and so shared. Each branch's population multiplies its own
host-density term inside the branch sum, `logsumexp_k [log w_k + log
p_pop(theta | L_k) + log n_k - log Z_k]`, for the PE samples and the
injections alike, as the reference does. Labels and order follow the
reference (the copies after the survey blocks, before `fcat_2 .. fcat_K`).
Spelling differs: core takes a mapping `{catalog number: [names]}`, names
being labels or ASCII names (`"mu_chi"`), where the reference took a list of
suffixed plain names (`"mu_chi_c2"`) or broadcast an unsuffixed one.
`ds.decode_parameters` returns the K vectors as `catalog.populations`. Parity
with af896ca's kernel replayed onto the pinned reference c042527: 2.8e-14
(`tools/probe_catalog_population_blocks.py`, not in CI: af896ca is not on the
public reference). Run directly on af896ca, the values differ by up to 0.14 on
the probe's fixture, the shared-population control included: af896ca branches
from 2b86a2d, and the reference's powerlaw+peak density changed after that
commit (pairing normaliser and support fixes), which core inherits from
c042527.


### Speed defaults (numerics change by default; historical values reproduce)

The owner decided on 2026-10-02 to make four evaluation settings default to
their faster form. The old behaviour stays available by setting each
historical value explicitly, which reproduces the program of `main` before
the change (b47e41c) bit for bit: optimized HLO and log-likelihood bits,
checked on 24 mock and real cases.

| setting (env) | new default | historical value |
|---|---|---|
| `configure_normalization_grids(pairing_norm=...)` (`DARKSIRENS_GW_PAIRING_NORM`) | `"auto"` | `"per_sample"` |
| `configure_catalog_evaluation(kernel_layout=...)` (`DARKSIRENS_CATALOG_KERNEL_LAYOUT`) | `"galaxy_list"` | `"padded"` |
| `configure_catalog_evaluation(missing_density=...)` (`DARKSIRENS_CATALOG_MISSING_DENSITY`) | `"auto"` | `"grid"` |
| `configure_catalog_evaluation(kernel_window=...)` (`DARKSIRENS_CATALOG_KERNEL_WINDOW`) | `"auto"` | `"off"` |

`"auto"` is the faster evaluation wherever it applies and the historical one
where the explicit faster value refuses, without raising. An explicit
`"per_point"`, `"gather"` or tolerance keeps refusing as before.

- `pairing_norm="auto"`: the per-point normaliser for the pairings that
  declare its structure (`PowerLawPairing`, `GWTC5FiducialBPL2PeaksPairing`);
  the per-sample rule for any other pairing, and for every pairing when the
  opt-in `pairing_m1_grid` is set (then it is the m1 grid, exactly as under
  `"per_sample"`).
- `kernel_layout="galaxy_list"`: refuses no case, so it is the default value
  itself; its values equal the padded layout's bit for bit.
- `missing_density="auto"`: gathered, except with a missing-host extension
  (`make_catalog_mixture_target`), where the grid is kept.
- `kernel_window="auto"`: a window of tolerance `1e-10` on every catalog view
  whose rows are sorted by redshift, with finite redshifts and a finite
  `sigma_kde` bound (`darksirens.catalog.redshift.kernel_window_applies`);
  no window (the full-row sum) on a view that is not. Such a window is marked
  `strict=False`, and the marked host kernel drops it and sums every galaxy,
  where it refuses an explicit one.

**Fingerprints and resume (behaviour change).** As for `pairing_scale` on
2026-10-01, the historical values are left out of
`core_numerics_semantic()`, so a fingerprint made before the change matches
when they are set explicitly. The new defaults are recorded
(`normalization_grids.pairing_norm = "auto"`, `catalog_evaluation =
{kernel_layout: "galaxy_list", missing_density: "auto", kernel_window:
"auto"}`; with a catalog binding, `kernel_window` is the tolerance it was
bound with, `1e-10`, or absent when `"auto"` attached no window). Resuming a
checkpoint written before the change under the new defaults is therefore
refused as a settings change, and the refusal now names the fix: set
`pairing_norm="per_sample"`, `kernel_layout="padded"`,
`missing_density="grid"` and `kernel_window="off"` (or the environment
variables `DARKSIRENS_GW_PAIRING_NORM=per_sample`,
`DARKSIRENS_CATALOG_KERNEL_LAYOUT=padded`,
`DARKSIRENS_CATALOG_MISSING_DENSITY=grid`,
`DARKSIRENS_CATALOG_KERNEL_WINDOW=off`) to resume it. The catalog entries are
recorded for spectral runs too, as explicit values always were.

**Values.** Against the historical values (CPU, float64, 16 prior draws plus
the stage-2 MAP and 8 Laplace draws per case): |dlogL| at most 1.4e-7
(relative 2.3e-13) on the real 259-event spectral likelihood with
powerlaw+peak, 2.7e-12 with GWTC-5, 4.0e-8 on the K = 2 field mixture (T and
R1), 2.3e-8 on the mock T complete catalog and at most 3.6e-9 on the other
mock T and R1 dark, field, selection-completeness and z_depth cases. All of
it comes from the per-point pairing normaliser: run alone, the galaxy list is
bitwise, the gathered density is within 2.8e-14 and the window within
4.6e-11. In float32 the per-point normaliser shifts the likelihood by up to
0.09 (mock T) and 0.21 (real) far from the maximum, where float32 is already
0.03 to 50 nats off float64; within 30 nats of the maximum float32 is as
close to float64 under the new defaults as under the historical ones (1.3e-3
on T, 1.7e-5 on R1, 5.5e-6 against 5.0e-6 on real).

**Parity with the frozen reference.** The legacy probes in CI
(`tools/probe_*.py` against c042527 at rtol 1e-12) pass unchanged under the
new defaults, so they keep running at the defaults; the population and
spectral probes evaluate the per-point normaliser, and the mixture and
selection-completeness probes bind through `bind_analysis` with the galaxy
list, the gathered density and the window.


### Public API gaps found by the examples (default sampler changes; no likelihood change)

The owner approved on 2026-10-05 three changes found by the internal examples
(`darksirens-examples/GAPS.md`).

**Default sampler (behaviour change).** `ds.infer` without `sampler=` now
runs Dynesty; it ran TinyNS before, which is not used for production. Dynesty
stays the optional extra `darksirens[dynesty]`, and TinyNS stays in the base
install. Without the extra, the default raises an `ImportError` that names
`sampler=` and the extra. It never falls back to TinyNS. A plan with no free
parameters needs no sampler and still returns its exact evidence on the base
install. To get the old default, pass `sampler="tinyns"`. The signature
default is now `sampler=None`, meaning `"dynesty"`. `selection_neff_guard="auto"`
resolves to the hard guard for both samplers, so the likelihood is the same.
A target run that relied on the default and writes a checkpoint now has
`sampler: "dynesty"` in its run fingerprint, so a TinyNS checkpoint made
through the old default resumes only with `sampler="tinyns"` (and that
checkpoint's TinyNS options).

**Two package-root names.** Both are additive.
- `ds.log_likelihood(analysis, *, events, injections, ...)` returns the
  likelihood `ds.infer` samples (the `BoundAnalysis` from `bind_analysis`),
  bound through the same code with the same five likelihood options. With no
  sampler to resolve against, `selection_neff_guard="auto"` is the hard guard,
  as for both nested samplers. Its values equal `bind_analysis(...)`'s bit for
  bit.
- `ds.save_result(path, result, *, labels=None)` writes a result through
  `atomic_result_hdf5` and `write_dead_point_datasets`, in the layout
  `darksirens.io.results.save_result` documents.

The examples-only rule (`tests/test_examples.py`: examples call only
`ds.__all__` names) covers both: `examples/likelihood_grid.py` evaluates
`ds.log_likelihood` on an H0 grid, and `examples/ordinary_catalog.py --out`
writes its result with `ds.save_result`.

**Guard report (additive result key).** When the selection guard makes the
likelihood `-inf` (or, with the soft guard, penalises it) over part of the
prior, the sampler leaves that part out of the posterior, and before this
change nothing reported it. `ds.infer(..., guard_report=True)` (the default)
evaluates the bound likelihood's diagnostics at 32 prior draws before
sampling an ordinary analysis. It records in `result["guard_report"]` which
guard fired and the range of each sampled parameter where it fired, and warns
once when more than 5% of the draws are guarded. The guard is
`"selection_neff"` when the injection `N_eff` fails `max(5 N_obs, N_obs^2 /
max_likelihood_variance)` with no PE variance, and `"pe_mc_variance"` when it
passes that bound but the summed per-event PE variance took the budget.
`guard_report=False` skips it; an integer sets the number of draws. The cost
is one extra compilation and the draws' likelihood evaluations.

**Numerics.** None of this changes a likelihood value. The diagnostics are a
separately jitted program (`BoundAnalysis.diagnostics`), and the bound
likelihood's call and program are unchanged; the default path passes the
likelihood the same keywords as before. The draws use an RNG of their own
seeded from the sampler `seed`, so the sampler's streams are untouched. The
legacy parity probes (`tools/probe_*.py`) and the parity tests pass
unchanged.


### Galaxy-list build on GPU (memory fix; no likelihood change)

The galaxy-list kernel normaliser (`kernel_layout="galaxy_list"`, the default
since 2026-10-02) ran its galaxies in at most 32 unrolled chunks and relied on
a data dependence to keep one chunk live at a time. On GPU, XLA kept them live
together. On a two-catalog field mixture with 151 million galaxies (nside 32,
no survey depth), building the kernel pin needed 32.6 GB of temporaries,
against 1.1 GB for the padded layout. It ran out of memory at JAX's default
75% limit on an 80 GB A100 (reported by the gws-agn analysis).

The chunks now run as a `lax.map` loop over fixed chunks of 2^20 galaxies on
every backend except CPU. CPU keeps the unrolled chunks, which XLA runs
multi-threaded. On GPU the same build needs 0.56 GB. Each galaxy executes the
same arithmetic in any chunk, and the last chunk is padded with placeholder
galaxies whose outputs are dropped. The schedule is
`darksirens.catalog.redshift._GALAXY_MAP` (`"auto"`, `"unrolled"` or
`"loop"`). The tests run the loop on CPU against the padded state, its
gradients and vmap.


### Per-point pairing table built once per likelihood call (speed fix; no likelihood change)

The per-point pairing normaliser (`pairing_norm="auto"`, the default since
2026-10-02) normalises `p(q | m1)` from a small table of the integrated
secondary-mass taper. The table depends on the population hyperparameters
alone, but the population density is evaluated inside the likelihood's
per-block loops (the PE-event blocks of `pe_event_block` and the injection
batches of `sel_batch_size`), so the table was rebuilt in every block. On a
GPU profile of a two-catalog field mixture (442 blocks per call), that
rebuild was about 48 ms of small kernel launches per call, the whole of the
15% by which the default was slower than `pairing_norm="per_sample"`.
On an A100 with a 130-block mock, the hoisted table takes the per-point
default from 52.4 to 42.4 ms per call (spectral siren) and from 30.4 to
19.5 ms (two catalogs with their own populations), level with the
per-sample rule (42.8 and 19.9 ms).

The likelihood now builds the table once per call, before the loops, and
every block reads it. `PairingModel._per_point_state` builds the table and
`_per_point_density(..., state=)` reads it. `prepare(theta, dtype)` on
`PopulationModel`, `MixtureModel`, the GWTC-3 and GWTC-5 fiducial models and
the pairing returns the table for the same parameter slices the density
uses. The weight functions in `darksirens.likelihood.weights` and the
float32 weights take it as `prepared=`. Every argument is optional, and
without it the density builds its own table as before. Models without the
per-point normaliser, such as the Gaussian-process populations, are called
exactly as before.

**Numerics.** No likelihood value changes. The table is built with the same
operations in the same dtypes (the kernel in the per-sample dtype, the
accumulation in float64), so the density and the log-likelihood are the same
bit for bit. The tests check the density for both production pairings in
float64 and float32 at the prior edges, and the population models. Through
`bind_analysis` with several PE blocks and injection batches they check the
log-likelihood and its diagnostics against the program without the hoist
(`darksirens.likelihood.weights._HOIST_POPULATION_STATE = False`), in float64
and float32. The cases are a spectral siren with the population sampled or
fixed, a catalog, and per-catalog population branches. The gradient agrees
to rounding, not bit for bit: the table's share of the gradient is now
summed over the blocks before it is carried back through the table, where
before each block carried its own back. The difference is at most 7.8e-16 of
the gradient's largest component on the test likelihoods (3.3e-15 on the
density alone). A table
built for another dtype is not used; the density then builds its own.

### GWTC-5 fixed population: effective-spin values (fixed-population results change; declared divergence from legacy)

The fixed vector of `gwtc5_fiducial_bpl2peaks` (and its aliases
`gwtc5_fiducial_brokenpowerlaw+2peaks` and `gwtc5_brokenpowerlaw+2peaks`)
changes in two entries: `mu_chi` 0.0633 to 0.04 and `sigma_chi` 0.3654 to
0.10.

- **Why.** The old pair are the GWTC-5 release medians for the spin
  magnitude. The model's spin component is a truncated Gaussian in the
  effective spin, so the distribution was about 3.7 times too wide. On the
  259 GWTC-5 BBH events the spectral H0 likelihood peaks 107 to 110 lower in
  log with the old pair than with a width near 0.10, and the
  fixed-population H0 is about 1.7 km/s/Mpc lower.
- **The new pair is a fit, not a published number:** the best point of a
  30-point spectral grid over the pair with every other parameter at the
  release medians, to two decimals.
- **What changes.** Every result that fixes this population at its stored
  vector (`fixed=True`, `fixed="gwtc5"`). Sampled populations and the model
  itself do not change.
- **Reproducing old results.** Pass the full vector as an explicit
  `fixed={...}` mapping with the old pair.
- **Legacy parity.** Legacy keeps the old pair. `tools/probe_population.py`
  declares the two entries (`DECLARED_FIDUCIAL_DIVERGENCE`): both codes are
  evaluated at legacy's vector, so the model's numerics are still compared
  at 1e-12, and each code's stored pair is checked against its declared
  value.

### Host mass of a sky row from the galaxy weights (opt-in, default unchanged)

`ds.model(catalog=..., catalog_sky_weighting="field", completeness="complete",
host_mass="weight")` makes the host mass of a sky row the sum of its galaxy
weights. By default (`host_mass="count"`) the row's host mass is its galaxy
count and the weights only share that mass among the galaxies of the row.
That is the frozen reference's convention, and it means a row with one heavy
galaxy weighs the same as a row with one light galaxy. With
`host_mass="weight"` the row's density is `W_p p_cat(z | p) = sum_i w_i
K_i(z)` and the catalog's total is `Z_k = sum_p W_p`, with `W_p` the sum of
the weights of the row's real galaxies. Each galaxy then hosts in proportion
to its own weight in every row. Inside a row nothing changes.

- **Default.** Nothing changes without the option. The plan record, the run
  fingerprint and the bound program are the same, and the log-likelihood and
  its gradient are the same bit for bit against a clean `main` on the test
  catalogs. The legacy parity probes run the defaults and are not touched.
- **Scope.** Complete catalogs under the field weighting, one catalog or a
  mixture in which every catalog is complete. In a mixture the option is one
  value for all catalogs, and each catalog is normalized by its own total
  weight, so the catalogs' weights may be in different units.
- **Refused.** `completeness="incomplete"` and `"selection"`, for any
  catalog. The hosts that are missing from an incomplete catalog are counted
  by number (`n0` and the completeness are number quantities), so weighting
  only the galaxies in the catalog would compare weights with counts. The
  conditional weighting is refused as well, because it divides each row's
  host mass out and the option would have no effect. The likelihood also
  refuses a binding whose row sums do not match the option.
- **Weights.** They must be finite and strictly positive, as before.
  Multiplying every weight of a catalog by one constant changes nothing.
- **Cost.** One pass over the stored weights when the analysis is bound,
  in blocks of rows, and one float64 number per row kept on the binding.
  Padding slots beyond a row's galaxy count are not read. No per-galaxy array
  is added and the per-call work is unchanged. The sums are taken before the
  full-sky view of a complete catalog drops its galaxy slots.
- **Fingerprint.** `ParameterPlan.catalog_model` records `host_mass` when it
  is `"weight"`, so a checkpoint of one mode is not resumed in the other.
- **Legacy parity.** The reference has no such mode. The tests compare the
  option on a catalog with integer weights against the same catalog with each
  galaxy written weight-many times at unit weight, evaluated by count.

### Input checks for five silent wrong answers (new errors and warnings; no likelihood change for valid float64 inputs)

Five kinds of input gave a finite, plausible likelihood that was wrong, with
no error and a clean guard report. Each was reproduced first, on small
synthetic inputs and on the quick mock (150 events, 4,096 samples each, an
nside-16 catalog). The numbers below are log-likelihood differences.

- **A float32 catalog in the count-ratio completeness is now computed
  correctly.** The observed-count density floors each galaxy's kernel mass at
  `1e-300`, which is zero in float32. A padding redshift beyond the redshift
  grid (the surveys writer pads with 100) has zero mass, so the masked slot
  contributed 0/0 and the density was NaN in every row with padding (3,071 of
  3,072 on the mock). The likelihood stayed finite and was wrong by +7.2,
  +8.1 and +8.6 at H0 = 55, 67.74 and 80. The density now reads the redshifts
  in float64 whatever the catalog stores
  (`darksirens.catalog.completeness._observed_density_row`). A float32
  catalog then gives the likelihood of the same values stored as float64, to
  rounding (2e-12 on the test fixture). Float32 catalogs are not refused: the
  complete and selection modes were already correct for them. For a float64
  catalog the cast is a no-op and the density is the same bit for bit.
- **A non-finite redshift in the padding of `zgals` now raises
  `ValueError`** (`validate_catalog`, so at load and at bind). The kernel sum
  multiplies every slot's redshift offset by a reciprocal width that is zero
  on padding, and `nan * 0` and `inf * 0` are NaN for the whole row. The
  result was `-inf` at every H0 (complete catalog; count ratio with NaN) or a
  small finite shift (count ratio with infinity, selection). Finite padding
  of any value is ignored exactly, as before. Non-finite padding in `dzgals`
  and `wgals` is ignored exactly and still accepted.
- **`ds.load_catalog` now checks the file.** It raises `ValueError` when the
  file has more rows than `12 * nside**2`. Row `r` is HEALPix RING pixel `r`,
  so more rows mean a wrong `nside` attribute or rows nothing reads: a file
  written at nside 2 and declared as nside 1 bound, read its first 12 rows as
  the sky, and moved a complete-catalog likelihood by -9 to -40. A file with
  fewer rows than pixels loads with a `UserWarning`, because it binds only
  when no GW sample falls in a missing pixel (binding already raises
  otherwise). A negative redshift of a real galaxy also loads with a
  `UserWarning` that gives the number of such galaxies and the minimum: the
  kernel is defined for it (the galaxy's redshift distribution is truncated
  at zero) and blueshifted nearby galaxies are real. The checks binding makes (array shapes, integer `ngals`, finite redshifts,
  finite non-negative widths and positive finite weights of the real
  galaxies) now also run at load, and the message names the file. A NaN
  redshift of a real galaxy therefore raises `ValueError` at load, where the
  row sort raised `AssertionError` before. The checks read the arrays in row
  blocks of about four million slots, at load and at bind, so their own
  memory does not grow with the catalog.
- **`ds.load_events` now checks that the samples of an event are stored
  together.** The store has no per-sample event index: event `i` is rows
  `[i * nsamp, (i + 1) * nsamp)` by contract, so a file in another order had
  the right layout and was accepted (+135 to +163 on the mock). The loader
  now compares, for `m1det`, `dL`, `ra` and `dec`, the scatter between the
  means of the declared blocks with the scatter inside them
  (`darksirens.gw.store.event_block_problems`). If no column separates the
  blocks (ratio below 3, where interchangeable rows give about 1), the
  samples are not grouped by event. The loader raises `RuntimeError` when
  the sample-major reading (event `i` is rows `i, i + nobs, ...`) does
  separate them beyond chance (probability below 1e-6), and emits a
  `RuntimeWarning` otherwise, since a store of near-identical events is
  legitimate. The check is statistical: it cannot see a few misplaced
  samples, or two events exchanged in part. Stores with one event or one
  sample per event are not checked. In-memory `GWStore` objects are not
  checked.
- **Fixing `log10n0` without `n0_units` now emits a `UserWarning`.** The
  default stays `"physical"` and no value changes. A fitted density is
  usually h-scaled (`darksirens-surveys` reports `n0_units: "h_scaled"`),
  and the two readings differ by `(H0 / 100)**3`: on the mock the same fixed
  value moves the likelihood by -2.6 to -6.0 across H0 = 45 to 90. Pass
  `n0_units="physical"` or `n0_units="h_scaled"` to `ds.model` to state the
  unit; the warning then does not appear. A sampled `log10n0` does not warn.

**Numerics.** For valid float64 inputs nothing changes. The log-likelihood
was compared before and after on 16 configurations (spectral; complete, count
ratio and selection, each conditional and field-weighted, with and without a
survey depth; two catalogs; a fixed density in both units) at three H0
values: all 48 values are the same bit for bit. The tests compare the
observed-count density against the previous implementation bit for bit on a
float64 catalog.


### Kernel pin and row chunks at bind (memory fix; no likelihood change)

Binding a large catalog needed far more memory than the bound likelihood
keeps. The bound likelihood keeps 40 bytes per padded slot of the compact
view (rows touched by an event or an injection times the longest row): 24 for
the catalog's redshifts, widths and weights, 16 for the kernel pin's two
per-slot leaves. A consumer measured the bind peak on GPU (complete catalog,
field weighting, nside 64, 150 to 400 million galaxies) at 65 bytes per slot
when the number of touched rows was a multiple of 512 and 105 when it was not,
and a 400-million-galaxy bind failed on a single 24.9 GiB request. Those GPU
numbers are the consumer's. The numbers below were measured for this change
on CPU, where the device is the host (peak resident memory of the process,
49,152 or 48,185 touched rows, 50, 100 and 200 million galaxies).

Two things held the extra memory, and both are gone.

- **The row-chunked build copied the catalog to pad it.** Above 2^25 slots
  the kernel is built in chunks of 512 rows. When the row count was not a
  multiple of 512, the three per-slot arrays were concatenated with zero rows
  first (24 bytes per slot) and the padded outputs were sliced afterwards (up
  to 16 more). Each chunk is now sliced from the catalog inside the loop.
  The last chunk is moved back to end at the last row, so the rows it shares
  with the chunk before it are computed twice and dropped. Nothing is padded
  and no argument is copied (`darksirens.catalog.redshift._map_rows`).
- **The kernel pin built the whole per-proposal state.** The pin keeps two of
  the state's five per-slot leaves. The other three (24 bytes per slot) were
  outputs of the same program and stayed live until it returned. Above the
  same threshold the pin now builds only its own leaves, chunk by chunk, and
  writes each chunk into them in place
  (`darksirens.catalog.redshift._pinned_kernel_leaves`). It no longer
  evaluates the kernel window's check, which the pin never kept (every
  likelihood call still evaluates it).

| bytes per slot at bind, CPU | before | after |
| --- | --- | --- |
| peak, touched rows a multiple of 512 | 76.0 to 76.9 | 52.1 to 53.0 |
| peak, touched rows not a multiple of 512 | 106.5 to 107.4 | 52.3 to 53.4 |
| resident after bind | 40.3 to 41.3 | 40.3 to 41.4 |

The remaining 12 bytes per slot above the resident 40 are one chunk's
temporaries, not a per-slot cost: on CPU a chunk needs about 1,130 bytes for
each of its 512 x N_max slots (the 24 quadrature nodes of every slot), which
at nside 64 is 1/96 of the catalog. On GPU the consumer measured the chunk's
share at 0.1 to 0.2 GiB. The pin alone took 49.7 s before and 33.7 s after at
50 million galaxies, 172 and 119 s at 100 million, 195 and 125 s at 200
million (each pair on one node, 32 cores). The per-proposal state, which a
run without the pin builds on every call, peaks 24 bytes per slot lower when
the rows are not a multiple of 512 (86 to 62 at 50 million galaxies, 84 to 60
at 100 million; its outputs are still reassembled once) and is unchanged
otherwise, in memory and in time.

Not changed: a view of at most 2^25 slots is still built in one pass, whose
temporaries are the same 1,130 bytes per slot on CPU (37 GB just under the
threshold); and with `kernel_layout="galaxy_list"` (the default for an
incomplete catalog) the normaliser of the real galaxies is still evaluated
whole before the chunks. With the list attached, the pin build alone peaked
at 80 bytes per slot above the catalog before and 66 after (rows a multiple
of 512), 92 and 68 otherwise (50 million galaxies, CPU, no survey depth).

**Numerics.** No likelihood value changes. Every row runs the same
arithmetic in any chunk. Measured on CPU (AMD EPYC 7542), before and after:

- The pin's leaves, by digest, on the 50, 100 and 200 million galaxy
  catalogs with 49,152 and 49,000 rows, and on the bindings with 49,152 and
  48,185 touched rows: identical. The log-likelihood of those bindings at
  eight H0 values: identical. With a galaxy list attached (50 million
  galaxies): identical.
- The log-likelihood at three H0 values and its H0 derivative, as hex
  floats, on 20 configurations of the mock (complete, count ratio and
  selection completion; conditional and field weighting; one and two
  catalogs; `host_mass="weight"`; pin on and off; kernel window on and off;
  padded and galaxy-list layouts; every eighth PE sample of each event, soft
  selection guard), with the library's chunk constants and with the chunked
  schedule forced at 500, 512, 3,072 and 4,096 rows per chunk: the five
  output files are byte-identical.

The tests compare the chunks sliced in the loop with the schedule they
replace (kept in the test as the reference), state and pin, bit for bit, at
three chunk sizes in both layouts with and without a survey depth. They also
compare the chunked state and pin with the unchunked ones at five chunk sizes
(tails of four rows and of one row, an exact multiple, exactly one chunk,
fewer rows than a chunk): bit for bit, except with a galaxy list and a survey
depth, where the comparison is to 1e-14. There a row's mass below the depth
can come out one ulp apart when the row function is compiled as a loop body
instead of one vmap (6.7e-16 relative, measured). The chunked schedule had
that property before this change, and it does not separate before from
after.

### Tabulated completeness for `completeness="selection"` (opt-in, default unchanged)

`ds.model(..., completeness="selection", selection=TabulatedSelection(z,
completeness))` takes the catalog's completeness directly as a table in
redshift, where the selection mode so far took a Gaussian or Schechter
magnitude-selection model. It is for a completeness that was measured (for
example against a deeper survey) and not fitted with a luminosity function.

- **Payload.** `{"format_version": "darksirens-catalog-selection-1.0",
  "family": "tabulated", "z": [...], "completeness": [...]}`: two lists of
  floats of equal length, the nodes and the values. It is the selection
  payload the other families use, so the catalog file does not change between
  analyses with different curves. `TabulatedSelection` stores both as tuples
  of Python floats and compares by value.
- **Meaning.** `C(z)` in [0, 1], linear between nodes (which need not be
  evenly spaced) and clipped to [0, 1]. It enters at the one place the other
  families' curve does (`selection_curve`), so `C(z, row) = row_fraction[row]
  * C(z)`, hosts above the catalog's `z_depth` are all missing, and the
  missing-host budget `(1 - C) n0 dV_c/dz (1 + z)^delta`, the field weighting
  and the mixture (one table per catalog) are unchanged. It is a function of
  the catalog redshift alone. The magnitude-selection curves do not depend on
  `H0` either (the h-scaling of their magnitude zero point cancels it), but
  they do depend on `Om0`, `w0` and `wa` through the distance modulus; the
  table depends on none of them.
- **No sampled parameter.** The family is fixed: the survey block is
  `log10n0`, `delta`, `sigma_kde`, and a selection nuisance named in
  `survey_priors` is refused as unknown.
- **Refused** (`ValueError`, from the payload and in `ds.model`): non-finite
  nodes or values, values outside [0, 1], nodes not strictly increasing,
  fewer than 2 nodes, unequal lengths, arrays that are not one-dimensional.
- **Coverage, no extrapolation.** The curve is read on the model's redshift
  grid, whose lowest point is 0. The first node must be at or below 0 and the
  last at or above the catalog's `z_depth`. For a catalog without a
  `z_depth` the last node must reach the top of the grid (5 by default):
  the magnitude-selection curves are read on the whole grid in that case, and
  so is the table. `ds.model` raises otherwise, for each catalog of a mixture
  against its own depth. The low-level `selection_curve` and
  `selection_completion_curves` do not make this check; outside the nodes
  they return 0 and never the end value.
- **Fingerprint.** `ParameterPlan.catalog_model` records `family`,
  `n_nodes`, `z_min`, `z_max` and `table_sha256` (sha256 of the float64 `z`
  bytes followed by the `completeness` bytes) in place of the arrays. Two
  tables give two run fingerprints.
- **Default and existing modes.** Nothing changes without the family. On 21
  configurations (count ratio, complete, Gaussian and Schechter selection,
  each conditional and field-weighted, with and without a depth; a row
  fraction; a sampled nuisance; a payload mapping; two two-catalog mixtures)
  the log-likelihood at three points, its gradient and the plan record are
  the same bit for bit as on `main` before the change.
- **Checks.** A Schechter curve tabulated on the model grid gives the
  Schechter mode's log-likelihood to 3e-14, at H0 values other than the one
  the table was built at. On uniform tables over [0, 0.25] the difference is
  2.8e-2, 2.1e-3, 8.8e-5, 1.1e-5 and 5.6e-7 for 9, 33, 129, 513 and 2049
  nodes (second order in the spacing). A table of ones on a catalog without a
  depth gives the complete catalog's log-likelihood to rounding, for both
  weightings; with a depth it does not, because every host above the depth is
  missing in the selection mode. A non-uniform table gives the value of the
  uniform table of the same piecewise-linear function.
- **Legacy parity.** The frozen reference has no such family.

**Catalogs with galaxy weights.** In the selection mode a row's host density
is `N_obs,p p_cat(z | p) + (1 - C(z)) n0 dV_c/dz (1 + z)^delta` in
row `p`, with `N_obs,p` the row's galaxy count and `p_cat` the kernel
with the weights normalized inside the row. The catalogued term is therefore
the row's weighted density divided by the row's mean weight. If the hosts are
meant to be weighted (by stellar mass, say), the missing term has to be in
the same unit: the completeness supplied must be the weight-fraction
completeness (the fraction of the total weight that is in the catalog at each
redshift), and `n0` must be the reference weight density divided by the
catalog's mean weight, with `n0_units` stated. Supplying weights together with
a count-based `n0` and a count-based completeness biases `H0`: a consumer
reported -1.5 km/s/Mpc on a toy, which was not reproduced here. The
prescription is exact only where a row's mean weight equals the catalog's,
because core divides by each row's own mean weight and takes one `n0`. Core
does not check any of this, and `host_mass="weight"` remains refused with an
incomplete catalog.

### Real galaxies outside the redshift range of the kernel (new errors and a warning at bind; no likelihood change for accepted catalogs)

Each galaxy's redshift distribution is its Gaussian truncated to the grid
`[0, DARKSIRENS_ZMAX]` (default 5) and renormalised. A real galaxy stored far
outside that range, in units of its effective width
`max(sqrt(dz**2 + sigma_kde**2), 1e-4)`, made its whole sky row lose its
hosts, with no error. Reproduced on an nside-2 catalog with one such galaxy
added to one row (15 of 48 PE samples), `sigma_kde = 0.01`, against the same
galaxy placed at z = 4:

- **Above the grid.** With a redshift error of 1e-4 (50 widths beyond the
  top) a galaxy at z = 5.5 gave `-inf` for `completeness="complete"` and a
  finite likelihood lower by 0.97, 3.05 and 3.00 at H0 = 55, 70 and 80 for
  `"selection"`, in the conditional and the field weighting; the count ratio
  moved by 1e-5. With a photometric error of 0.01 (1 + z), the same galaxy
  at z = 5.5 (7.6 widths) and at z = 6 were evaluated correctly, to rounding,
  and z = 10 failed in the same way. The galaxy's normaliser stays accurate
  (1e-3 or better out to 40 widths); what fails is the row's kernel sum,
  which is taken relative to the row's largest galaxy weight
  (`log_kw_eff_rowmax`): the galaxy's weight is the reciprocal of its
  vanishing in-grid mass, and once it exceeds the others by about e^745 they
  underflow to zero.
- **Below zero.** A galaxy at z = -0.1 with width 0.01 gave `-inf` with
  both completeness models; at z = -0.3 to -1 it gave `-inf` for a complete
  catalog and `-inf` or the same finite shift as above with the selection
  completeness. Here the normaliser itself fails: its
  quadrature nodes are clipped to z = 0, where the galaxy measure vanishes.
  Against a dense quadrature of the same integrand it is right to its usual
  7e-3 down to 6.0 widths below zero, off by 0.03 at 6.25, 0.5 at 6.75 and
  3.6 at 7.0, and by about 700 from 7.25 on, at every width and `delta`
  tried. z = -0.002 and -0.02 (0.2 and 2 widths) were evaluated correctly.

`bind_analysis` now checks the real galaxies of the catalog (the first
`ngals` slots of each row; padding, such as the surveys writer's 100.0, is
not read):

- **A real redshift above the top of the grid raises `ValueError`**, naming
  the number of such galaxies, the first one's row and slot, the largest
  redshift and the grid's upper edge (`validate_catalog(..., z_max=...)`,
  run inside the pass binding already makes). The rule does not depend on
  the width: a galaxy at or below the top keeps at least half of its kernel
  on the grid and is evaluated correctly, so kernels that straddle the top
  are accepted as before (the same likelihood as with the galaxy at z = 4,
  to 1e-9). A galaxy above a survey depth and inside the grid is accepted as
  before.
- **A real redshift more than 5 effective widths below zero raises
  `ValueError`** when that holds at the largest `sigma_kde` the analysis
  evaluates (its fixed value or the upper edge of its prior), so at every
  proposal (`darksirens.catalog.redshift.check_kernels_below_zero`; the
  limit is `KERNEL_WIDTHS_BELOW_ZERO_MAX`, one width inside the last
  correct value). When it holds only below some `sigma_kde` inside a sampled
  prior, binding emits a `UserWarning` that names that value: the catalog is
  accepted, and the row is wrong for proposals below it. With the default
  prior `[0, 0.05]` and a spectroscopic error, a galaxy at z = -0.002 is in
  this case for `sigma_kde` below 4e-4. Negative redshifts within reach
  (peculiar velocities) are accepted silently at bind; `ds.load_catalog`
  keeps its warning for any negative redshift of a real galaxy.
- `ds.load_catalog` does not make these checks: the catalog package imports
  without the cosmology tables, and the width depends on the analysis.
  Callers that compact a catalog themselves (`compact_catalog`,
  `compact_pe_selection_catalog` without `z_max`) are not checked either.
- **`host_mass="weight"` without a catalog** now raises a `ValueError` that
  says a catalog is required, instead of the refusal about the conditional
  sky weighting.

**Cost.** The upper check adds one comparison to the row blocks binding
already reads. The lower check is one more pass in row blocks that takes the
minimum of each block and looks further only where it is negative.

**Numerics.** Accepted catalogs evaluate exactly as before: 150
log-likelihoods are the same bit for bit before and after (complete, count
ratio and selection, each conditional and field-weighted, at three H0
values, on eight catalogs: plain, with a survey depth, with a galaxy at
z = 4.99, at the grid top, at z = 3 above a survey depth, at z = -0.002,
-0.02 and -0.1 with a wide error; and two catalogs together).
