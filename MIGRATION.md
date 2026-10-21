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
need 3.2 GB pinned against 15.4 GB unpinned. A conditional complete catalog
is not pinned, as in the reference; under `catalog_sky_weighting="field"`
the pin applies to every catalog, complete ones included. `kernel_pin="off"`
keeps the per-call quadrature, and the bound program is then byte-identical
to the previous release's. `ParameterPlan` gains `kernel_pin` and `kernel_pin_active`, and
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
  frozen reference's exclusion of both says. A default binding adds no
  `"likelihood_options"` entry: default digests equal those of the previous
  release (pinned in `tests/test_run_fingerprint_coverage.py`). That does not
  make `core_numerics_semantic(bound)` equal to `core_numerics_semantic()`
  for a catalog binding: since 2026-10-02 a catalog binding records the
  kernel window it was bound with in place of the setting `"auto"`, so a
  fingerprint and its resume check must be built the same way.
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
  itself; its values equal the padded layout's to rounding, and bit for bit
  under the unrolled schedule `_GALAXY_MAP="auto"` picks on CPU. Off CPU it
  runs a loop (see "Galaxy-list build on GPU" below), which with a survey
  depth differs from the padded state by a few ulp (`log_kw_eff` 3.6e-15 at
  a loop chunk of 4, 2.3e-13 at 1, measured with `z_depth = 0.8`).
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
bound with, `1e-10`, or absent when `"auto"` attached no window). A
fingerprinted resume of a checkpoint written before the change under the new
defaults is therefore refused as a settings change: an `InferenceTarget` run,
which `ds.infer` fingerprints, or a run whose caller writes and checks its
own fingerprint. An ordinary `ds.infer` resume is not fingerprinted; it warns
when no `run_fingerprint.json` lies beside the checkpoint (see "Review fixes
(2026-10-20)"). The refusal names the fix: set
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
list and the gathered density. They do not exercise the window: in every
bind-level probe fixture the window is as long as the rows (`W = N_max = 5`
at the pinned `sigma_kde` and at the prior edge), so the windowed sum never
runs there, and no CI probe binds the default conditional count-ratio
analysis with a kernel pin against the reference. The window is checked
core against core (`tests/test_catalog_kernel_window.py`); a local
bind-level comparison with an active window (`W = 9 < 60`, pin active)
matched the reference to 1.3e-15 (review of 2026-10-20). A probe cell with
an active window is still to be added.


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
count and the weights only share that mass among the galaxies of the row,
which sets the row's redshift shape (see "Catalogs with galaxy weights"
below for what that does with an incomplete catalog).
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
  is `"weight"`, so a fingerprinted checkpoint (an `InferenceTarget` run, or
  a fingerprint its caller gates) of one mode is not resumed in the other;
  an ordinary `ds.infer` resume is not fingerprinted and only warns.
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
  fewer than 2 nodes, unequal lengths, arrays that are not one-dimensional;
  with `TypeError`, entries that are not integers or floats (strings such as
  `"0.1"` and booleans, which NumPy would convert).
- **Coverage, no extrapolation.** The curve is read on the model's redshift
  grid, whose lowest point is 0. The first node must be at or below 0 and the
  last at or above the catalog's `z_depth`. For a catalog without a
  `z_depth` the last node must reach the top of the grid (5 by default):
  the magnitude-selection curves are read on the whole grid in that case, and
  so is the table. `ds.model` raises otherwise, for each catalog of a mixture
  against its own depth, and the error names the catalog (`catalog 2
  (<path>): ...`). The low-level `selection_curve` and
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
because core divides by each row's own mean weight and takes one `n0`. It is
also exact where the row has no catalogued density, since no weight enters
there: in empty rows (`N_obs,p = 0`) and above the depth (`p_cat` is zero
there and the missing term is the full `n0 dV_c/dz (1 + z)^delta`). Core
does not check any of this, and `host_mass="weight"` remains refused with an
incomplete catalog.

The count-ratio completeness (`completeness="incomplete"`, the default) has
no such remedy. Its estimator counts raw galaxies, whatever the weights,
while the observed term still carries them: the row's kernels are weighted,
so the weights set the row's redshift shape. Weights that vary with redshift
inside a row (a union catalog that down-weights its photometric galaxies,
for example) tilt the observed hosts against the missing ones and bias
`H0`, and no weight-fraction completeness can be supplied to the count
ratio. Use unit weights there, or weights that do not vary with redshift
inside a row.

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
- **The whole likelihood, with `sigma_kde` fixed.** A consumer observed
  that the same galaxy at z = 5.5 with width 1e-4 and `sigma_kde` fixed at
  0 (5000 widths beyond the top) made the whole likelihood `-inf` at H0 = 55
  and 70 but not 80, with the count ratio and the selection completeness as
  well, while with `sigma_kde` sampled it stayed finite. Reproduced on the
  code before this change (`delta` and `sigma_kde` fixed at 0, so the
  kernel pin applies): the galaxy's pinned log weight is 1.25e7, where the
  float64 spacing (1.9e-9) exceeds the per-call probe's absolute tolerance
  (1e-9), so the probe that compares the live kernel with the pin shifted by
  `3 ln(H0 / H0_ref)` fails by rounding alone. It failed at H0 = 55, 60 and
  70 and passed at 67.74 (the pin's reference), 75 and 80, and a failed
  probe sets the likelihood to `-inf`. This happens only when the galaxy's
  row is one of the pin's 8 probe rows: with the galaxy in each of the 36
  rows of the fixture that had room, 7 gave `-inf` at H0 = 55 and 70 in both
  modes, and the others gave the pinned value equal to the unpinned one. In
  the probe rows `kernel_pin="off"` or a sampled `sigma_kde` (evaluated at 0
  or 0.01) gave finite values, which are still wrong where the row holds PE
  samples (the shift above). The measurements above used a sampled
  `sigma_kde` at 0.01 and the row with most PE samples, which is not a
  probe row, so they did not show this. The refusal below covers both: the
  galaxy is refused at bind whatever `sigma_kde` and the pin setting.

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

### Pooled count-ratio completeness (opt-in, default unchanged)

(The default changed to this estimator on 2026-10-10; see the next section.)

`ds.model(catalog=..., completeness="incomplete", count_ratio="pooled",
count_ratio_window=0.02, row_fraction=None)` replaces the estimator of the
count-ratio completeness. The default estimator (`count_ratio="row"`)
smooths each row's observed galaxy density and the expected density with the
same Gaussian of width 0.05 and divides. Both densities rise steeply with
redshift, so the ratio at `z` leans on the far side of the kernel, where the
completeness is lower. To first order it is `C + w^2 (C' dlnE/dz + C''/2)`,
`E = dN_exp/dz`, which is the completeness near `z + 2 w^2 / z` at low
redshift. The missing-host density is then too large where the completeness
falls, and H0 comes out high:

- the darksirens-desi-gwtc5 clean-room check (unclustered toy, exact
  reference model, 16 realisations of 200 events): +0.68 +- 0.14 km/s/Mpc,
  0.17 of the posterior width, the same without the clip, without Poisson
  noise and pooled over the sky, and scaling as `w^2` (+1.94, +0.67, +0.25,
  +0.12, +0.04 at w = 0.1, 0.05, 0.03, 0.02, 0.01);
- our examples mock (clustered, magnitude limit 21, photo-z 0.015(1+z), 100
  seeds of about 150 events): +0.78 +- 0.18 km/s/Mpc against the
  magnitude-selection completeness;
- core's own curve on a toy with a known completeness falling from 0.92 to
  0.23 between z = 0.15 and 0.3: 0.14 low at z = 0.2 with the default width,
  0.027 at 0.02 and 0.007 at 0.01, and on the examples mock 0.15 to 0.20 low
  at z = 0.08 to 0.13.

The default is kept, bit for bit, for reproducibility and for the legacy
parity tests. `count_ratio="pooled"` smooths the ratio itself (each galaxy
weighted by one over the expected density at its own redshift, normalised
by the kernel mass inside the surveyed range), pools all the rows of the
catalog before the clip and gives row `p` the completeness `f_p C(z)`, with
`f_p` the optional `row_fraction`.

- **Default.** Nothing changes without the option: the plan record, the run
  fingerprint, the binding (the observed-density cache) and the
  log-likelihood bit for bit, for one catalog under either sky weighting and
  for a two-catalog mixture. The legacy parity probes run the defaults.
- **Accuracy.** Its expectation is the kernel mean of the completeness, so
  the `w^2 C''/2` of any kernel remains: on the toy the pooled curve is
  within 0.006 of the truth at w = 0.01 and 0.016 at 0.02 (tests). The
  clean-room toy recovered the exact model to +0.00 +- 0.01 km/s/Mpc at
  w = 0.01 with this construction. On the examples mock (100 seeds, same
  grids as above) its H0 minus the magnitude-selection completeness's is
  -0.03 +- 0.05 km/s/Mpc at w = 0.02 and -0.10 +- 0.04 at 0.01, against
  +0.78 +- 0.18 for the default. Narrowing the default's per-row window does
  not help there: with about 180 galaxies per row the clip of each row's
  noisy ratio takes over (+0.59 +- 0.28 at 0.02, +1.28 +- 0.37 at 0.01).
- **Pooling.** The catalog is one completeness cell: every row shares the
  same radial curve, scaled by its coverage. It does not model a depth or
  magnitude limit that varies over the sky. Splitting such a catalog into one
  catalog per stratum under `catalog_sky_weighting="field"` is a different
  model: in each stratum's rows the other catalogs count every host as
  missing, so each stratum's galaxy contrast is diluted by its mixture weight
  (`ds.model` warns when two catalogs' row fractions cover disjoint rows;
  this advice, given here before 2026-10-20, is withdrawn). Galaxy-free rows
  count as surveyed unless `row_fraction` says otherwise (a warning above
  10% of the rows). Clustering along the line of sight is averaged over the
  whole sky rather than per row.
- **Low redshift.** Each galaxy's weight `1/dN_exp(z_i)` grows as `z_i^-2`.
  With photometric redshifts some galaxies scatter to `z_obs` near 0, where
  the expected density vanishes, so within about three windows of `z = 0`
  the pooled ratio is far above 1 (6 at z = 0.05 with w = 0.02 on the
  examples mock) and the clip makes it 1: the estimator takes the catalog
  to be complete there, whatever its completeness, and removes its missing
  hosts there. For a catalog that carries no host information this gives a
  spurious preference in H0; binding now warns, and the opt-in
  `count_ratio="pooled_kernel"` avoids it (see "Review fixes (2026-10-20)").
- **Noise.** Binding refuses a window with fewer than 100 effective galaxies
  at the catalog's median redshift and warns below 1000: the clip at 1 then
  biases the curve low.
- **Cost.** At bind, one pass over the catalog in row blocks (a histogram of
  the real galaxies' redshifts in bins of 1e-4) and one `(N_z, N_z)`
  operator (8 MB), in place of the `(N_rows, N_z)` observed-density cache
  (its kernel over every padded slot). Per call, one matrix-vector product
  over the grid, then the arithmetic of the row-fraction selection
  completeness.
- **Fingerprint.** `ParameterPlan.catalog_model` records `count_ratio`, the
  window and the row fraction's digest when `"pooled"`.
- **Not covered.** `darksirens-lss` passes the bound cache to
  `completion_curves`, which accepts the pooled cache, but it has not been
  tested with it. The reference has no such mode.

### Pooled count ratio by default (numerics change by default; declared divergence from legacy)

`ds.model` now uses the pooled count-ratio estimator of the previous section
by default. The signature default is `count_ratio=None`, which resolves to
`"pooled"` with `count_ratio_window=0.02` wherever a catalog runs the
count-ratio completeness, and to the inert `"row"` everywhere else. The
legacy per-row estimator stays available, unchanged, as `count_ratio="row"`.

- **Why.** The per-row estimator is biased where the completeness falls
  (to first order `C + w^2 (C' dlnE/dz + C''/2)` at `w = 0.05`): H0 high by
  +0.68 +- 0.14 km/s/Mpc on the clean-room toy and +0.78 +- 0.18 on the
  examples mock (100 seeds). The pooled estimator recovers the toy at
  -0.020 +- 0.030 km/s/Mpc at w = 0.02 and is -0.03 +- 0.05 km/s/Mpc from
  the magnitude-selection completeness on the mock (100 seeds).
- **What changes.** For every `ds.model` call with a catalog and
  `completeness=None` or `"incomplete"` (either sky weighting; in a field
  mixture, every catalog that runs `"incomplete"`) and no `count_ratio`:
  - the likelihood (the pooled curve in place of the per-row one), and so
    the posterior;
  - the plan record and the run fingerprint: `ParameterPlan.catalog_model`
    now records `count_ratio="pooled"` and the window. A fingerprinted
    resume of a checkpoint written before the change (an `InferenceTarget`
    run, or a fingerprint its caller gates) is refused as a settings
    change, and the refusal names the fix (`ds.model(...,
    count_ratio="row")`); an ordinary `ds.infer` resume is not
    fingerprinted and only warns;
  - binding: it builds one `(N_z, N_z)` pooled operator in place of the
    per-row observed-density cache, and it now **refuses** a catalog with
    fewer than 100 effective galaxies under the kernel at its median
    redshift (small or toy catalogs), and warns below 1000, for more than
    10% galaxy-free rows without a `row_fraction`, and for real galaxies at
    `z <= 0`;
  - `row_fraction` is now accepted with the count-ratio completeness (the
    coverage of each row, as for `"pooled"` before); it is still refused
    with `count_ratio="row"`;
  - a `count_ratio_window` alone now asks for `"pooled"` (it was refused
    without `count_ratio="pooled"`).
  - The catalog is one completeness cell: every row shares the radial
    curve, scaled by its coverage. It does not model a depth or magnitude
    limit that varies over the sky, and splitting such a catalog into one
    catalog per stratum under `catalog_sky_weighting="field"` is a different
    model (each stratum's galaxy contrast is diluted by its mixture weight;
    see "Pooling" in the previous section).
- **Who is affected.** Anyone calling `ds.model` with a catalog and
  `completeness=None` or `"incomplete"` without `count_ratio`.
  Unchanged: spectral and bright-siren analyses, `completeness="complete"`
  and `"selection"`, an explicit `count_ratio`, and the low-level functions
  (`completion_curves`, `build_completion_state`,
  `build_observed_density_cache`, and the likelihoods given no cache or an
  `ObservedDensityCache`), which keep the per-row estimator and select the
  pooled one only when given a `PooledCountRatioCache`. A record built
  directly (`IncompleteCatalogRedshift`, `FieldCatalogMixtureRedshift`)
  keeps the field default `"row"`. `darksirens-lss` builds the
  observed-density cache itself (`fixed_table.py`, `gridding.py`), so its
  fixed-table default is still the per-row estimator; moving it to the
  pooled one is a separate change in that repository.
- **Getting the old behaviour.** Pass `count_ratio="row"`. It reproduces
  the program of `main` before the change (427e53a) bit for bit: the plan
  record (the per-row estimator is not recorded), the run fingerprint, the
  binding (the observed-density cache) and the log-likelihood. The probes
  below check the bit-for-bit part on their cells.
- **Legacy parity (declared divergence).** The frozen reference has only the
  per-row estimator, so the new default has no reference counterpart. The
  parity probes that go through `ds.model` run the per-row estimator
  explicitly and declare the default (`DECLARED_COUNT_RATIO_DEFAULT =
  "pooled"`, checked against `ds.model`'s resolved default):
  `tools/probe_catalog_mixture.py` (CI, phase 5, at rtol 1e-12) and
  `tools/probe_catalog_population_blocks.py` (manual, against af896ca).
  Their candidate outputs are byte-identical to those of 427e53a. The other
  catalog probes (`probe_catalog_likelihood`, `probe_catalog_completeness`,
  `probe_catalog_kernel`, `probe_catalog_marks`, `probe_catalog_selection`,
  `probe_selection_completeness`) call the low-level functions or run the
  selection completeness and are unaffected. The tests that compare a bound
  count-ratio likelihood with a direct per-row evaluation, or that run
  small toy catalogs too sparse for the pooled estimator, pass
  `count_ratio="row"` (`test_kernel_pin`, `test_partial_fixing`,
  `test_runtime_binding`, `test_runtime_binding_jit`, `test_compute_dtype`,
  `test_n0_units`, `test_pytree_callable`, `test_catalog_dark_memory`,
  `test_decode_parameters`, `test_catalog_mixture`,
  `test_catalog_mixture_completeness`, `test_catalog_population_blocks`,
  `test_host_mass_weight`, `test_input_checks`,
  `test_pairing_per_point_hoist`, and the plan-record checks of
  `test_selection_completeness` and `test_tabulated_completeness`). The
  public example `examples/ordinary_catalog.py` keeps the default; its test
  catalog grew from 3 to 400 galaxies per row so that the pooled estimator
  binds without a warning.
  `test_count_ratio_pooled` pins the default (pooled, window 0.02, bit for
  bit the explicit setting), its inert resolution without a count-ratio
  catalog, and that `"row"` keeps the previous record.

## Review fixes (2026-10-20)

A multi-agent review on 2026-10-20 confirmed 130 findings against this tree
and the frozen reference. They were fixed in twelve packages, merged in this
order on one branch: completeness and selection budget, GW loaders, field
mixture, marked hosts, catalog, parametric populations, GP models and
registry, runtime binding, public API, samplers, likelihood, and cosmology and
packaging. Each finding was fixed, guarded (now refused or warned), put behind
an opt-in option, deferred with a reason, or found not reproducible.

**Default numerics.** Every default log-likelihood of a valid input is
unchanged bit for bit, except in these places, each a crash, a silent wrong
answer outside the frozen reference's code, or a declared divergence:

- a selection-variance budget already exhausted by the PE variances is now
  rejected whatever `N_eff` is (before, `N_eff = +inf` or `N_eff > N_obs^2
  1e12` passed with a finite value);
- a one-catalog field analysis under the soft Neff guard
  (`ds.infer(sampler="numpyro")` turns it on by default) now has the
  reference's density `n_1 / Z_1`, so its soft wall has its calibrated height;
- the kernel pin's probe tolerance is relative where the kernel's log weights
  are huge, so pinned likelihoods that were `-inf` by rounding alone are now
  finite (a declared divergence: the reference has the same holes);
- `ang2pix_ring` matches healpy for right ascensions within about 4.4e-16
  below a multiple of 2 pi;
- an integer or float32 `theta` is converted to float64 before decoding;
- the mass tapers `sfilter_low` and `sfilter_high` return 1, not 3.5e-57, at
  a mass one float inside the shoulder where the rounded distance to the edge
  equals the taper width (see "Populations").

The parity probes the CI workflows run, run on this tree and on `main` at
1022429, write byte-identical output files (85 of 88) except: the
field-mixture probe's two records, which gain the key `soft_guard_cap` (null);
and the nested-sampler startup check's remedy text (see "Public API"). The
three sampler probes listed under "Samplers and prior transforms" each differ
from the frozen reference in one item on purpose; each probe declares its
item, checks it and reports the reference's form, so those CI gates stay exact
on everything else (see `VALIDATION.md`). Some results gain keys or change
keys that are not likelihood values: `log_prior_volume_fraction` and
`logZ_corrected` for partially fixed joint prior pairs and for core-built
targets, `guard_report["selection_coverage"]`, and the fingerprints of
host-density targets, of one-catalog field analyses under the soft Neff guard,
and of plans that record a new non-default option.

Where two packages fixed the same thing, the branch keeps one behaviour,
stated in the subsection it belongs to: the pooled cache's depth (both checks
are kept), unknown `ds.infer` keywords (refused, not warned), a target's
prior-volume offset (the public API package's), the partial-fixing warning
(the public API package's, with the samplers package's wording), and the
float32 column check of a binding (the likelihood package's).

### Completeness and the selection budget (default numerics unchanged)

Every default log-likelihood is unchanged bit for bit. This was checked on
nine bound analyses: the pooled and the row count ratio, conditional, field
and two-catalog, with a depth and a row fraction, and completeness="selection"
with a row fraction. The completeness, selection, likelihood and mixture
parity probes write the same output as before and pass against the frozen
reference. What changes is warnings, the refusal of inputs that gave a
silently wrong answer, and three opt-in options.

- **The pooled count ratio near z = 0 (warning).** The default counts each
  galaxy at its point redshift, with the weight 1/dN_exp, which grows as z^-2.
  With photometric redshifts some galaxies scatter to z_obs near 0, and the
  ratio is far above 1 below z ≈ 0.07: about 300 at z = 0.01 on a mock catalog
  that is 50% complete with errors 0.03(1+z). The clip then makes the catalog
  complete there and removes every missing host there, whatever the
  completeness. For a catalog that carries no host information, ln L_cat − ln
  L_spec was +7.3 nats at H0 = 20 and fell by 1.4 nats from H0 = 50 to 100 (64
  events). The marginal H0 moved by −1.0 km/s/Mpc (−0.15 sd); scaled to four
  times the events it moved by −0.33 sd. Building the cache now warns where
  this ratio exceeds the one of the catalog's own host density (below) by more
  than a factor 1.5, and names the redshifts. Spectroscopic catalogs,
  including ones with a 450 km/s peculiar-velocity error, do not warn.
- **count_ratio="pooled_kernel" (opt-in).** The pooled count ratio with each
  galaxy counted through its redshift kernel instead of its point redshift.
  The kernel width is dzgals; sigma_kde is not included, because it may be
  sampled. This is the pooled ratio of the catalog's own host density, so the
  observed and the missing hosts add up to the expected density for a catalog
  that carries no host information, with or without redshift errors. Galaxies
  at z <= 0 count through the part of their kernel above 0. On the mock above,
  ln L_cat − ln L_spec is flat to 0.22 nats over H0 = 20 to 140 at the true n0
  and to 0.15 nats when marginalised over n0. A falling completeness gives at
  most 0.33 nats. With spectroscopic redshifts it gives the default's curve to
  1e-3 in the ratio. The plan records it with its window; at the low level it
  is build_pooled_count_ratio_cache(..., redshift_errors=True). The default
  stays "pooled".
- **A row fraction of 0 on rows that hold galaxies.** Under the pooled count
  ratio this is now refused (ds.model and build_pooled_count_ratio_cache).
  Those galaxies entered every row's ratio but none of the coverage (+20% with
  8 of 48 rows), and were counted twice in their own row. Under
  completeness="selection", which keeps the frozen reference's values (the
  reference accepts such rows), it is a warning.
  build_pooled_count_ratio_cache also refuses fractions that are not finite or
  lie outside [0, 1].
- **The depth a pooled cache was built for.** A cache built for the grid top
  and evaluated at a depth of 0.35 gave C = 0.29 instead of about 0.55 at z =
  0.349, silently. PooledCountRatioCache gains two records, written by
  build_pooled_count_ratio_cache: `z_top`, the 0-d depth its operator was
  normalised to (z_depth, or the top of the grid), and `depth`, a static
  PooledCacheDepth holding its z_depth. completion_curves and
  gathered_completion_curves raise ValueError where the comparison can be
  decided when they are traced: in the bound likelihood, whose z_depth is a
  Python value, exactly; inside their own jit, where z_depth is traced, a
  depth against none. A traced depth of another value makes the curves NaN
  (z_top), so the likelihood is -inf. BoundAnalysis also refuses a cache of
  another depth than its own. A cache constructed without these records is not
  checked. Pass the catalog's z_depth when building a cache yourself.
- **A depth that varies over the sky.** The advice to split such a catalog
  into one catalog per depth under catalog_sky_weighting="field" is withdrawn
  (README, CONTRACT and the pooled sections above now say so). Each catalog
  has one global weight, so in the rows of one region the other catalog counts
  every host as missing. Each region's galaxy contrast is then diluted by its
  mixture weight, which pulls the result toward the catalog-free one. ds.model
  now warns when two catalogs' row fractions are never positive in the same
  row. The pooled estimator models one completeness for the whole catalog.
  Only count_ratio="row" lets the completeness vary by row, and it is biased
  where the completeness falls and where rows are sparse.
- **count_ratio="row" (legacy; values unchanged).**
  - build_observed_density_cache refuses real galaxies more than 7 widths of
    its 0.05 kernel below z = 0 (z < −0.35). There the kernel's mass rounds to
    0, and a galaxy at z = −0.5 made its row complete out to z ≈ 1.3.
  - It warns when a typical row holds fewer than 1000 effective galaxies under
    the kernel at the catalog's median redshift. The clip of each row's noisy
    ratio then biases the completeness low: a complete catalog with 13
    galaxies per row (nside 128) reads 0.59 at z = 0.02. The missing hosts
    this adds grow with nside.
  - Without a batch_size, the rows per batch keep the (rows, N_z, N_max)
    kernel within 2^30 bytes, at most the previous 512. The values are the
    same bit for bit; peak memory fell from 13 GB to 3.5 GB for 2.5 million
    galaxies at nside 8.
- **selection_budget_audit(..., budget="likelihood", row_fraction=None)
  (opt-in).** The completeness is 0 above z_depth and the footprint is the
  summed row fraction. The default N_obs_total is the likelihood's observed
  count: each row's galaxies times its catalog kernel's mass below the depth.
  The default budget="frozen" keeps the legacy definitions and values. With a
  depth, the frozen budget can be many times the count the likelihood
  predicts: 18.6 times for GaussianMagnitudeSelection(22, −20.2, 1) at z_depth
  = 0.25.
- **Selection payloads.** The Gaussian and Schechter decoders and
  validate_catalog_selection refuse booleans and strings, as the tabulated
  family does. Before, m_lim=True decoded to a first-magnitude limit (every
  host missing) and k_corr_coeffs="01" to the coefficients (0, 1). A NumPy
  array of K-correction coefficients is now accepted; it raised "truth value
  of an array ... is ambiguous".
- **Tests.** tests/test_count_ratio_null.py pins the uninformative-catalog
  null: spectroscopic redshifts flat under the default and photometric ones
  flat under "pooled_kernel". The photometric null under the default is an
  expected failure (`xfail`, not strict) that records its spread as
  `default_null_spread_nats` (7.19 nats at the test's seeds, against the 1.5
  nats of a flat null), so the test measures the known defect and passes
  without failing the suite the day the default is fixed.

### GW loaders and the PE/selection pair gate (stricter input checks; default likelihood values unchanged except where noted)

- Pair gate: the detection cut and redshift truncation are now compared. gwcat
  records z_max, the detection statistic, the FAR threshold and snr_min, but
  leaves them out of contract_hash. bind_analysis now compares them on every
  pair that records them (top-level attrs first, then the 2.1 contract). Two
  stated values that differ are refused. A value stated on one side only
  raises a RuntimeWarning. Two stated detection statistics must agree: a
  FAR-only event list paired with a "far_or_snr" injection mask is refused,
  where gwcat only warns. Expect the one-sided warning on every bind of the
  current production pairs ("FAR: PE far_max=nan vs selection
  far_threshold=1.0").
- chieff_reference selections: spin_reference_amax must equal the PE chi_eff
  ceiling (chi_eff_amax_1/2_per_event, or chi_eff_amax) to rtol 1e-9,
  otherwise the pair is refused, for hashless 2.0 pairs too.
  spin_reference_coverage_ok=False raises a RuntimeWarning. Core now ACCEPTS a
  2.1 (chieff PE, chieff_reference selection) pair when every other pairing
  field agrees and the advisory columns differ only by a1, a2, cost1, cost2,
  chip. gwcat's own validate refuses such a pair on the hash (its 2.1 hash
  does not yet recognise it, and gwcat says to export it at 2.0).
- chi_eff fold (chi_eff_in_p_pe=False / chi_eff_swap_applied=False): needs a
  gwcat that provides chi_eff_prior_logprob_in_support (the gwcat extra now
  pins one; see "Packaging" below). Folded values inside the support are
  unchanged. Refused now: PE samples or injections beyond the ceiling; a NaN
  or non-positive ceiling; per-event (amax_1, amax_2) ceilings that differ
  from the scalar chi_eff_amax; a store that records only one of
  chi_eff_amax_1/2_per_event, or per-event arrays that are not one value per
  event.
- PE events with p_pe == 0 for every sample are refused. Zero-weighted samples
  in other events raise a RuntimeWarning.
- Bootstrapped PE events: a RuntimeWarning at every load names the events
  gwcat resampled with replacement (n_unique_samples_per_event < nsamp). Their
  per-event MC variance is too small by about 1 + nsamp/n (bounded by 1 +
  nsamp/n_unique). The current production stores warn for GW150914_095045
  (2383 unique rows) and GW200129_065458 (1732), bound up to 3.36. The
  variance used by the guard is not rescaled.
- Event-block layout check: a contiguous store of alike events sorted inside
  each event now warns instead of being refused.
- Injection count: bind_analysis refuses a selection ndraw smaller than the
  number of injections with positive pdraw. The likelihood seams (spectral,
  ordinary, bright, host-density) refuse n_draw below the live injections when
  their inputs are concrete.
- Selection variance guard: an exhausted budget (sum of per-event PE variances
  >= max_likelihood_variance - 1e-12) is now rejected whatever N_eff is. The
  hard guard returns -inf, and the soft guard keeps its wall at full depth.
  Before, N_eff = +inf or N_eff > N_obs^2*1e12 passed with a finite value.
  Default likelihood values change only in those cases; everywhere else the
  jitted bound likelihood is bitwise equal to the previous release.
  guard_report's classify_guard follows the same rule.
- compute_dtype="float32" (opt-in): selection padding keeps float32 columns
  and the bound q. With a sel_batch_size that does not divide the injections,
  float32 ln L values move at the ~1e-8 to 2e-7 nats level. Float64 results
  are bit-identical.

### Field mixture

- New ds.model(..., mixture_density=...) for a field mixture of two or more
  catalogs.
  - "pixel" (the default) is the frozen reference's convention. Each branch
    n_k / Z_k is the host probability of the sample's row, a mass per pixel.
    When the catalogs' nsides differ, branch k carries its pixel area A_k, so
    the weights act as w_k A_k / sum_j w_j A_j and fcat_k are not host
    fractions.
  - "steradian" (opt-in) multiplies each branch by A_min / A_k, its pixel area
    relative to the smallest in the mixture. Each branch is then a host
    density per steradian on the finest catalog's pixel scale, and the weights
    are host fractions whatever the nsides.
  - With equal nsides the factor is exactly 1, so "steradian" gives the
    default's values bit for bit under the hard and the soft Neff guard alike,
    log mu included. The common scale A_min keeps log mu, which the soft wall
    reads, on a per-pixel scale.
  - ds.model now warns when the nsides differ and mixture_density is not
    given. Pass mixture_density="pixel" to keep the reference convention
    without the warning.
  - mixture_density is refused with one catalog and with
    catalog_sky_weighting="conditional".
  - The plan records "steradian" in catalog_model. "pixel" is not recorded, so
    existing fingerprints are unchanged.
- One-catalog field analysis under the soft Neff guard
  (selection_neff_soft_guard=True; ds.infer(sampler="numpyro") turns it on by
  default):
  - The density is now n_1 / Z_1, as in the frozen reference. With
    tools/probe_catalog_mixture.py --soft-guard 0.08, the one-catalog cells
    now match the reference to a relative 3.7e-14; before, they differed by up
    to 2.7e4 nats.
  - Changed outputs:
    - log L where the soft wall is active: the wall now has its calibrated
      height. Before, log mu was offset by +log Z_1 and the wall was much
      shallower: on the test fixture at max_likelihood_variance=0.065, log L
      moves from -19016.6 to -67614.7.
    - log L where the wall is inactive: changes only by rounding (about
      4e-15).
    - Diagnostics: log_mu and every event_log_evidence are now normalised,
      shifted by -log Z_1.
  - Extra cost: bind_analysis now also binds the catalog's full sky (and a
    full-sky kernel pin when the catalog has a survey depth). The likelihood
    evaluates Z_1 on every proposal, by the direct full-sky sum for the count
    ratio (the pooled default has no moments form). That is the same
    per-catalog Z_k work a two-catalog mixture does: about 12 nside^2 rows
    times the redshift grid (196,608 rows at nside 128). Expect more run time
    and device memory for every one-catalog field run under numpyro.
  - The run fingerprint records it: under the soft guard, a binding of a
    one-catalog field analysis adds `"one_catalog_field_density":
    "normalised"` to `likelihood_options` (`core_numerics_semantic(bound)`,
    and the provenance of `make_catalog_mixture_target`). A soft-guard
    checkpoint of a one-catalog field run written before this change is
    therefore refused on resume (`ResumeFingerprintError`, naming the entry);
    `resume_force=True` would continue it on the new likelihood, so restart
    such runs instead. Hard-guard runs, two-catalog mixtures and the options
    mapping a companion passes (which cannot tell the analysis) are
    unchanged.
  - A one-catalog binding without the full sky (for example one made without
    the soft guard and then switched on with dataclasses.replace) is refused
    under the soft guard. Rebind with bind_analysis(...,
    selection_neff_soft_guard=True).
  - The hard-guard program (the dynesty default) is unchanged. Its diagnostics
    log_mu and event_log_evidence for a one-catalog field analysis stay
    unnormalised, offset by +log Z_1, which cancels from the likelihood
    (documented on BoundAnalysis.diagnostics). The frozen reference normalises
    them under either guard.
- Field mixtures of two or more catalogs: a NaN in any branch (an extension, a
  selection model, a curve, a missing-host budget) now makes the likelihood
  -inf under every kernel layout. Before, this held only when some catalog
  carried a pin, galaxy list or kernel window. Finite values are unchanged.
- MissingHostExtension:
  - A negative modulated density is clipped at 0 before Z_k, as the numerator
    already treated it (the reference's max(1 + b delta_g, 0)).
  - missing_total must return a 0-d scalar (ValueError otherwise), and a
    non-finite or negative total gives -inf.
  - MissingHostContext gains completeness_curve: the clipped curve of the
    selection completeness or of the pooled count ratio, and None for the
    per-row count ratio.
  - For the pooled count ratio, row_fraction is now the pooled cache's row
    coverage instead of None.
- catalog.mixture.normalized_state and eval_normalized_field_prior_vmap now
  carry the row masses' pin, galaxy-list and window poison into log Z.
- The field_normalizer="moments" refusal now reads "is implemented only for
  completeness='selection'" and gives the reason for each completeness. Which
  inputs are refused is unchanged.
- tools/probe_catalog_mixture.py gains --soft-guard CAP, which runs both arms
  under the soft Neff guard. CI runs it without the flag, as before.

### Marked hosts: non-finite marks, selection completeness, auto window, base-weight mu_miss (new errors; opt-in estimator; default unchanged)

- **A non-finite host mark of a real galaxy now raises `ValueError`.** The
  same applies to a non-finite reference redshift or reference mark
  (`check_centered_marks`, so in `build_marked_incomplete_catalog_prior_state`
  and `marked_dark_siren_log_likelihood`). The error names the first row, slot
  and mark. Before, one NaN mark, or NaN padding, made `mu_miss` NaN over most
  of the grid. The row normaliser then read as `log_Z = 0` in every row, and
  the marked prior came back finite and about 13 nats off. Padded slots are
  now ignored whatever they hold: they are read as zero marks before the
  product with eta, so NaN padding gives the zero-padded prior and gradient
  bit for bit. Legacy refused non-finite marks in pixelate and zeroed the
  padding. A NaN normaliser now poisons its row (`-inf` likelihood) instead of
  reading as 0. Two equal mark tables that contain NaN now count as the same
  table in the shared-view check.
- **The marked-host prior refuses `CatalogParameters.selection`.** It raises
  `ValueError` instead of silently running the count ratio under
  `completeness="selection"`, which was up to 3 nats per sample off the
  selection prior.
- **The marked likelihood drops a `kernel_window="auto"` window on the
  catalogs the per-sample evaluator is handed**, not only on the ones the
  states are built from. An explicit window is still refused. Before, the
  marked prior was summed over the window with no `window_ok` verdict, which
  gave a finite wrong value past the sizing `sigma_kde`. The marked kernel now
  sums every galaxy, as documented. A marked state built directly must be
  evaluated against the catalog without its window.
- **Opt-in `mu_miss_estimator="weighted"`** on `missing_host_efficiency_grid`,
  `build_marked_incomplete_catalog_prior_state` and
  `marked_dark_siren_log_likelihood`. The default `"count"` is the legacy
  count mean of `h` per redshift bin. The observed branch carries `N_obs
  <h>_w`, the mean weighted by the catalog base weights. So when the weights
  correlate with the mark (for example luminosity weights with a stellar-mass
  mark), a redshift-independent mark still moves the catalog:missing odds by
  `<h>_w / <h>`: x0.6 to x1.7 over eta in [-2, 2] in the review. `"weighted"`
  averages `sum w h / sum w` per bin over the aligned rows, so the two
  branches cancel. It refuses a survey-wide reference table, which carries no
  base weights.
- **`CenteredHostMarks` survives standard JAX tree operations.** `tree_map` to
  non-array leaves, `eval_shape`, `device_put` and `vmap` prefix trees now
  work: `tree_unflatten` no longer re-runs the validating constructor.
- **`ds.model(counterparts=...)` validates each counterpart.** `z` and `dz`
  must be finite, positive, real scalars. Unless the counterpart is
  `sky_marginalized`, `pixel` must be an integral RING id in `[0, 12 *
  counterpart_nside**2)`. Before, a zero or NaN `dz`, a NaN `z` or an
  out-of-frame pixel bound and gave `-inf` at every theta, with nothing naming
  the counterpart. A negative `dz`, which was silently read as `|dz|`, is now
  refused, as legacy did.

**Numerics.** Unchanged for finite marks, `selection=None`, the default
`"count"` estimator, catalogs without a window, and valid counterparts.
`tools/probe_catalog_marks.py` (21 values) and
`tools/probe_catalog_likelihood.py` (105 values) are the same bit for bit
before and after. So is every leaf of the marked prior state on a fixture with
non-uniform weights, a survey depth, an empty row, garbage padding and a
reference table. The one public value that changes is `log_h`, the third
output of `build_marked_catalog_kernel_state`: it is now 0 at padded slots
that held non-zero padding, and nothing reads those slots.

### Catalog (new errors, an in-graph poison, and fixes for -inf holes and NaN gradients; no likelihood change for valid inputs)

- **Kernel pin probe tolerance.** The pin's in-graph probe compares each
  probed slot to max(`KERNEL_PIN_TOL` = 1e-9, `KERNEL_PIN_TOL_EPS` (64) * eps
  * |value|) instead of an absolute 1e-9. With a survey depth, rows whose
  galaxies all lie thousands of widths above it have log_kw_eff near 2.5e7.
  There float64 resolves 3.7e-9, so the probe failed on rounding alone and the
  pinned likelihood was -inf at about half the H0 values (12 of 25 in the
  review scenario; the unpinned program was finite). Those values are now
  finite and equal the unpinned program to 1e-12. Wherever |log_kw_eff| < ~7e4
  the tolerance is still 1e-9, so every finite value is unchanged bit for bit.
  This is a declared divergence from legacy, which has the same absolute 1e-9
  and the same holes.
- **Survey depth checks.** `load_catalog` refuses a `z_depth` attribute that
  is not a finite redshift above 0. `bind_analysis` refuses a store `z_depth`
  outside (0, grid end] for every completeness: count ratio row and pooled,
  selection, complete, and each catalog of a field mixture, whose error names
  the catalog. Before, only the pooled count ratio checked the range. A NaN or
  non-positive depth used to drop the catalog from the observed branch, and a
  depth beyond the grid gave rows more observed hosts than galaxies, both with
  no error. A catalog without a depth has `z_depth=None` (no attribute in the
  file). Direct kernel callers are not checked at bind; for them the depth
  normaliser reads a depth beyond the grid end as the grid end (mass below it
  1, not 1.058). New host helper:
  `darksirens.catalog.types.check_survey_depth`.
- **NaN missing-host budget.** A NaN in any row's `N_obs + N_miss` (for
  example from a companion's curves through
  `build_incomplete_catalog_prior_state_from_curves`, or from a missing-host
  extension's density) now makes every row's `log_Z` NaN, and so the
  likelihood -inf. Before, the row read as log Z = 0 and got a finite,
  inflated host density. In a field mixture the branch poison is added to
  every term (see "Field mixture"). Finite budgets are unchanged bit for bit.
  Inherited from legacy and guarded here.
- **Direct catalog prior.** `log_catalog_prior` and `log_catalog_prior_vmap`
  raise `ValueError` when `params.z_depth` is not None, instead of silently
  returning the untruncated kernel. Use `build_catalog_kernel_state` with
  `eval_log_catalog_prior_state` for a depth-limited catalog. Without a depth
  they are unchanged. Inherited from legacy and guarded here.
- **Non-finite padded widths.** A NaN or inf `dzgals` padding slot is read as
  1.0 when the kernel forms its widths (padded and galaxy-list layouts, and
  the marked kernel). Values are unchanged. The `sigma_kde` and `H0`
  gradients, which were NaN on such catalogs, are now finite and equal to
  those of finite padding. `validate_catalog` still accepts any padded width;
  padded redshifts must still be finite.
- **Kernel window sizing memory.** `kernel_window` and
  `kernel_window_applies`, which run under the `kernel_window="auto"` default,
  read the catalog in row blocks. A device catalog is sliced on the device
  first, so the binding no longer keeps host copies of `zgals` and `dzgals`.
  Before, peak host memory was 2.1x one padded float64 array, and a GPU
  binding kept two such arrays for its lifetime. The window is unchanged bit
  for bit.
- **ang2pix_ring.** Right ascension is folded into [0, 2*pi) in radians before
  the HEALPix azimuth reduction, as healpy does. Pixel ids change only for ra
  within about 4.4e-16 below a multiple of 2*pi, where core now matches healpy
  (one ring off at even nside, and one pixel off in the polar caps, before).
- **Text.** The catalog docstrings now name the current defaults: pooled count
  ratio, galaxy-list layout, `kernel_window="auto"`, gathered missing density.
  The `load_catalog` docstring lists the checks it makes and those left to
  bind. The negative-redshift warning now says that a galaxy far below zero is
  refused at bind, rather than kept as a host.
- **Parity.** All catalog parity probes (`probe_catalog_kernel`,
  `_likelihood`, `_completeness`, `_selection`, `_mixture`, `_marks`,
  `_population_blocks`, `probe_selection_completeness`,
  `probe_healpix_geometry`, `probe_public_catalog_loader`) give candidate
  outputs byte-identical to `main` at 1022429.
- **Deferred.** The missing-host density is still interpolated linearly
  between grid nodes, while the spectral prior is interpolated in log. This is
  a legacy-parity default. The fix needs an owner decision on the formulation:
  an opt-in factorised (1-C) x log-volume interpolant, or an empty-catalog
  spectral baseline.

### Populations: the taper shoulder, float32 fixes, new opt-in normalisations (default unchanged except at the shoulder)

**Taper shoulder at a rounded edge sum (a silent wrong answer; values change
only there).** `sfilter_low` decided its shoulder from `m >= fl(m_min + dm)`
but its exponent from the rounded distance `fl(m - m_min)`. Where the two
disagree, `fl(m - m_min) == dm` while `m < fl(m_min + dm)`, the exponent's
second denominator was 0, the exponent clipped to its cap and the taper
returned its floor, 3.5e-57, where its limit is 1: `sfilter_low(10.1, 2.05,
8.05)`. `sfilter_high` had the mirror case, `sfilter_high(7.95, 20.0,
12.05)`. A sample or injection stored at such a mass lost about 130 nats per
taper: for `powerlaw` with m_min = 2.05 and dm_min = 8.05, log p at m1 = m2 =
10.1 was 260 nats below its neighbours. Each shoulder is now also decided
from the rounded distance the exponent uses (`delta >= dm`). Measured on
`main` (the frozen reference has the same code) and on this tree:
- two-decimal (m_min, dm) pairs on [2, 10] x [0.01, 10], at the parsed
  decimal sum: 6008 of 801,000 returned the floor; now none;
- one float below the float sum of random edges: 1.6% (3112 of 200,000) in
  float64, and 12,480 of the 801,000 decimal pairs in float32; now none;
- `sfilter_high` one float above the float difference of random edges: 1.2%
  (24,798 of 2,000,000); now none;
- every other input gives the same value bit for bit: 400,000 continuous
  masses through each taper, the population registry goldens, and the CI
  parity probes (probe_population and the spectral, catalog and bright-siren
  probes write the same files as on `main`). The gradient at the shoulder is
  0, the taper's own.

This is a declared divergence from the frozen reference at those floats
only; no CI fixture lands on one. tests/test_population_model_support.py
holds the regression tests (they fail on `main`).

**Float32 pairing normaliser in the taper toe (only `compute_dtype="float32"`
changes).** Two float32 paths returned wrong densities for m1 within about
1.2% of the secondary-mass taper window above m_min:
- With the opt-in `pairing_m1_grid`, the grid and edge normalisers were built
  from float64 nodes. A float32-floored density was divided by the integral of
  a float64-floored kernel and came out exp(130 - 79.85) = exp(50.15) too
  large. One such selection injection took the powerlaw+peak likelihood from
  26.7005 to -inf. The nodes now use the per-sample dtype, the products scale
  * n_sc are formed in float64, and the density is returned in the per-sample
  dtype (it used to come back float64).
- With `pairing_norm="per_sample"`, under jit XLA rewrites (p / scale) / n_sc
  into p / (scale * n_sc). That product flushes to zero in float32, and the
  density came out +inf, a dropped weight, for every m1 below m_min + 1.9e-3
  at m_min = 5, dm_min = 3. The float32 branch now multiplies by the
  reciprocal.

Float64 is unchanged bit for bit: pairing values and gradients under
per_sample, the m1 grid and per_point; probe_population and
probe_spectral_likelihood.

**Golomb 1G+tail gradients (values unchanged).** For dm_tail <~ 0.6 Msun
(prior floor 0.1), the logistic tail taper overflowed exp at the low end of
the mass grid. Values were exact, but d log p_pop / d(M_tr, Delta, dm_tail)
was NaN for every sample. The gradients are now finite and equal central
differences. Values are bit for bit the same.

**`golomb_quadrature="split"` (opt-in, default unchanged).** Set with
`configure_normalization_grids(golomb_quadrature="split")` or env
`DARKSIRENS_GW_GOLOMB_QUADRATURE=split`. The default (`"grid"`, the
reference's rule) integrates the Golomb progenitor mass by trapezoid on the
fixed 0.4-Msun mass grid. It has two errors:
- log p_pop jumps by 1.5-2.4 nats (30 samples near the bump) every time the
  sampled disruption edge M_tr + 2 Delta crosses a node, and is nearly flat in
  between (d/dM_tr 0.18 against a true 4.47).
- A low-mass kernel narrower than the spacing is seen at one node, so phi_1G
  at 1 Msun is 3.4x too large at sigma = 0.05 (2.2x at 0.08).

`"split"` integrates [m_lo, 20] and [20, M_tr] by trapezoid uniform in ln M_I
(384 + 128 nodes) and [M_tr, M_tr + 2 Delta] by 64-node Gauss-Legendre.
Measured with it: smooth in M_tr and Delta (largest step 0.035 nats on a
0.005-Msun scan), d/dM_tr equal to central differences, and phi_1G at the grid
nodes within 2e-4 of a dense reference for sigma in {0.05, 0.08, 0.2}. The
cost per call is the same. Not covered: phi is still evaluated on the mass
grid and interpolated linearly, so between nodes at low mass a steep IMF keeps
an interpolation error (about 10% near 1.2 Msun for a = 2.35).

**`population_norm="exact"` (opt-in, default unchanged).** Set with
`configure_normalization_grids(population_norm="exact")` or env
`DARKSIRENS_GW_POPULATION_NORM=exact`. By default two families integrate to a
hyperparameter-dependent constant. That constant cancels in a likelihood with
one population, but it reweights catalog branches that carry different
populations (`per_catalog_population`). Use this option there.
- **Grammar mixtures.** A component without its own low-mass edge (the
  Gaussian peak) is normalised over the whole mass grid, but its pairing
  removes all its mass below the mixture's m_min. powerlaw+peak integrates to
  0.880 at G.mu = 20, G.sigma = 10, PL.m_min = 10, and brokenpowerlaw+2peaks
  to 0.9977 at its fiducial. `"exact"` normalises such a component over
  [m_min, m_hi], giving 1.0000 to the reference quadrature's accuracy, and the
  mixture weights become population fractions.
- **GWTC-3 and GWTC-5 fiducial masses.** The kept fraction was formed as 1 -
  deficit. For GWTC-3 at alpha = 10.8, m_min = 3.24, delta_m = 9.11 (inside
  Table VI) the normaliser is 0.38% low. `"exact"` integrates the kept part
  directly: within 3e-9 of scipy.quad there. For GWTC-5 the worst of 300 prior
  draws goes from 3.0e-4 to 6.8e-6.

**Component-spin tilt mixture (opt-in, default unchanged).**
`ComponentSpinModel(..., tilt_mixture="binary")` or
`default_component_spin(tilt_mixture="binary")` evaluates the GWTC-3 Default
tilt model, one aligned/isotropic mixture over the binary: zeta N(c1) N(c2) +
(1 - zeta)/4. The default `"per_spin"` multiplies two one-tilt mixtures. It is
what the `gwtc3_plpeak_component_spin` preset uses. It has the same one-tilt
marginal but a different joint density: at zeta = 0.75, sigma_t = 0.9, two
anti-aligned spins get 0.50 of the Default model's density, and the tilts are
uncorrelated against 0.072. So that preset's zeta_spin and sigma_t are not the
published model's.

The docs also stopped calling the [1, 10]^2 Beta-shape box the LVK magnitude
prior. The LVK prior is uniform in the mean and variance, and it allows mean
magnitudes below 1/11.

**Golomb tail taper (opt-in, default unchanged).**
`GolombRemnantMass1GPlusTail(..., tail_taper="planck")` is a keyword-only
option. It makes the tail exactly zero below M_BH,max - dm_tail. The default
logistic never reaches zero there. Against a steep (m/M_BH,max)^-c, the
"high-mass" tail can then dominate low mass: at c = 8, dm_tail = 20 (inside
the prior box) it is 47% of all mass, almost all of it below 10 Msun. The
registered `golomb_1g+tail` keeps the logistic.

The Golomb docstrings now also state that the models have no minimum
black-hole mass: the population's low-mass edge is the mass grid's M_LO = 1
Msun.

**Fingerprints.** `golomb_quadrature` and `population_norm` enter
`normalization_grids` only when set to their opt-in values, so every existing
fingerprint still matches. `tilt_mixture` and `tail_taper` are constructor
options of a model the user registers under its own name.

**pairing_scale default of 2026-10-01 (documentation).** The default of
`pairing_scale` became `"analytic"` on 2026-10-01 (37aeda1). It equals the
historical `"node_max"` arithmetic to rounding (|dlogL| <= 2.2e-11 on the real
259-event likelihood), not bit for bit. To reproduce a run made before
2026-10-01 bit for bit, or to resume its checkpoint, also set
`pairing_scale="node_max"` (env `DARKSIRENS_GW_PAIRING_SCALE=node_max`), in
addition to the four 2026-10-02 historical values. CONTRACT.md's list of
historical values names it now, and so do the catalog/settings.py and
population/utils.py docstrings.

### GP models, sky x z models and the population registry (default numerics unchanged)

Every default output is bit-identical to the previous tree. The candidate
outputs of probe_population, probe_angular_advanced, probe_angular_wiring,
probe_spectral_likelihood and probe_run_fingerprint_gate are byte-identical,
and so is the default log_p_pop of all 19 GP models. Default run fingerprints
are unchanged, because each new setting enters core_numerics_semantic only
when it is not its default. (The core-built targets' new
`log_prior_volume_fraction` is described under "Public API" below.)

New opt-in options. Each is latched at import from the environment and read
when a likelihood is traced.
- `DARKSIRENS_GP_NORM_QUADRATURE=resolved` (default `lattice`, the frozen
  rule). It covers the GP normalisers that do not cancel against the selection
  term: every z-conditional N(z) and every m1-conditional N(m1).
  - The normalisers are integrated on nodes that follow the sampled tapers and
    the q-support, at 4 nodes per RBF and 8 per Matern-3/2 length-scale floor.
    The kernel is contracted one axis at a time.
  - log N is interpolated in z with a cubic stencil; the additive model uses
    linear interpolation on twice the nodes.
  - The q-free models' baseline pairing is normalised with PowerLawPairing's
    own rule.
  - The lattice rule is inherited from the reference. At the prior corner
    (log_amp ceiling, ls_chi and ls_z at their floors), int p_gp(m1, chi | z)
    of gp3d_m1_chi_z runs from 0.98 at z = 0.1 to 1.61 at z = 1.2 on the
    lattice, and the baseline p(q | m1) integrates to 0.02-0.46 just above
    m_min. The resolved rule gives 1.000-1.001, and 1 within 1e-3 for the
    pairing.
  - Cost on CPU with 512 queries, value plus gradient: from about the same to
    5x (gp4d_additive 8.5 s -> 44 s); gp4d resolved 53 s at 5.8 GB peak.
- `DARKSIRENS_SKY_ZNORM_PER_LS` (default 3, the frozen value) sets the density
  of the per-redshift sky normaliser table of sphere_gp_z and overdensity_gp,
  in intervals per sky_log_ls_z floor. At 12, the interpolation error in the
  log normaliser at the prior corner falls from up to 0.10 to below 0.007. The
  option resolves the interpolation part only: on an independent sphere, the
  sky mean of g between nodes is still off by up to 0.023 at 12 per floor
  (0.125 at 3). Up to 0.016 of that comes from the sphere quadrature at the
  nodes, which this option does not change. So it does not make the sky field
  mean-one to 1e-2.
- `DARKSIRENS_SKY_Z_NODE_HI` (default 3.0, the frozen value) sets the top of
  those models' z inducing nodes. Setting it to zMax keeps the sky anisotropic
  above z ~ 3, where the field otherwise decays to isotropy. The node count is
  kept, so the spacing widens.

Behaviour changes outside the defaults:
- shared_beta / shared_spin / shared_gamma = False on a bespoke registered
  model (the GWTC-5 and GWTC-3 fiducials, Golomb, every GP model) now raises
  ValueError. The flags used to be ignored while population_latex said
  "Per-component". For a one-component grammar model the flags are still
  accepted, and the label no longer claims a per-component structure.
- '<one-component>@md' with shared_gamma=False now has a fixed preset, the
  same vector as shared_gamma=True, as get_model already allowed.
- JointGPPopulation and build_gp_model refuse an unknown mean_mode, axis name
  or '@' suffix.
- With the opt-in pairing m1 grid (`DARKSIRENS_GW_PAIRING_M1_GRID`) on:
  - A sample outside the grid takes the per-sample normaliser instead of
    jnp.interp's end-node value. A grid narrower than the model therefore
    costs no accuracy. Before, every 200 < m1 <= 300 sample of
    gwtc5_fiducial_bpl2peaks on a 200 Msun grid was clamped to the m1 = 200
    normaliser: 0.09 / 0.21 / 0.38 nats high at m1 = 220 / 250 / 299. That
    happened to a grid nobody sized, and to a density held from get_model or
    pop_model_parser and traced after a narrower model was resolved. With the
    grid sized for its model, values and gradients are bit-identical to
    before.
  - Every likelihood primitive now sizes the grid for its own model when it is
    traced, so a direct spectral_siren_log_likelihood call on
    gwtc5_fiducial_bpl2peaks is covered.
  - Sizing starts from the configured grid, so it no longer depends on which
    models were bound earlier in the process. A powerlaw+peak likelihood
    traced after a GWTC-5 binding moves back to its fresh-process values, by
    up to 2.2e-5 per sample.
  - A binding's fingerprint records the grid sized for its own model, from the
    settings the binding was made with.
  - configure_normalization_grids applies to the configured grid, not to a
    grown one; `configured_normalization_grid_settings()` returns it.
  - A binding's check of its trace-time settings (see "Runtime binding")
    compares the configured grid, so a binding traced after a wider model grew
    the grid is not refused.
  - assert_pairing_grid_covers_support still raises for an undersized grid, as
    the check that the grid was sized.
- Messages: the out-of-prior fiducial warning and the pairing-grid error name
  the public controls instead of command-line flags.

### Runtime binding (default likelihood values unchanged)

The binding-level parity probes give byte-identical output on main and the
branch.

- Cosmology: a fixed H0 must be > 0, and so must the lower bound of an H0
  prior. H0 <= 0 used to be accepted, which silently cut the prior to H0 > 0.
  Values must be real numbers: NumPy scalars (float32, int64) and 0-d arrays
  are now accepted; bools and strings (for example H0=('20', '140')) raise
  TypeError. CONTRACT.md and README.md no longer call H0 "unrestricted".
- Population('brokenpowerlaw+2peaks') is the GWTC-5 model only with
  fixed='gwtc5'. Sampled, at another preset, or partly fixed, it is the
  curated grammar composition BPL+2G, a different 17-parameter model. That
  still holds, but constructing it now raises a UserWarning. To sample or
  partly fix the GWTC-5 model, use Population('gwtc5_fiducial_bpl2peaks',
  ...).
- Bound likelihood and decode_parameters: an integer or float32 theta is
  converted to float64 first. Before, it truncated or rounded the plan's fixed
  values, giving a wrong finite log L or -inf under the kernel pin. A complex
  or boolean theta raises TypeError.
- BoundAnalysis checks the operands derived from its analysis whenever it is
  built: on dataclasses.replace and on unpickling as well as on bind. It
  refuses a count-ratio cache of the wrong estimator (either pooled ratio, or
  the row estimator) or depth, a missing or extra completeness='selection' row
  fraction, and float32 columns with compute_dtype=None (the message is the
  likelihood package's; see "Likelihood").
- BoundAnalysis stores the trace-time settings at bind: the normalisation
  grids as configured (before the opt-in pairing m1 grid is sized for a model)
  and the missing_density of configure_catalog_evaluation. If they change
  before a program is first traced, that trace raises RuntimeError;
  already-compiled programs keep working. core_numerics_semantic(bound)
  reports the binding's stored settings.
- bind_analysis(selection_neff_soft_guard=...) must be a bool. Mode strings
  raise TypeError; use selection_neff_guard in ds.infer and ds.log_likelihood.
- A redshift record built directly with a pooled count_ratio and no window
  uses the default window 0.02.
- The pooled cache records its depth; see "Completeness and the selection
  budget".
- Bright sirens: global pixel ids are int64 when nside > 13377. Before, ids
  wrapped in int32 and every event was -inf.
- ds.infer guard report: new key result['guard_report']['selection_coverage']
  (shell, dL_edge, dL_max, points, max_edge_share, max_edge_nats, at). It
  gives the share of the selection integral carried by the outermost 10% of
  the detected injections' distance range, at the prior draws and at the edges
  of the cosmology prior. A UserWarning is raised when N_obs times that share
  exceeds 0.1 nat, which indicates a campaign that stops inside the detectable
  volume and biases H0 high. The guard report now makes about twice as many
  likelihood-diagnostics evaluations. For a likelihood used without ds.infer,
  call darksirens.inference.guard_report.selection_coverage(bound, thetas).
- ds.infer refuses selection_neff_soft_guard passed as a sampler option, and
  every other keyword no core backend reads (see "Public API").
- Documentation: README.md now gives the float32 compute_dtype scope with
  field-weighted analyses without a complete catalog, as CONTRACT.md does.
  CONTRACT.md's import contract now lists the process-wide state a scientific
  submodule sets: jax_enable_x64; 'highest' matmul precision
  (cosmology.distances); the XLA_PYTHON_CLIENT_PREALLOCATE/ALLOCATOR defaults
  (gw.samples).

### Public API: ds.infer and ds.model option checks, prior-volume bookkeeping, code identity

No likelihood value, sample or default fingerprint digest changes. The only
changed numbers are ds.infer's log_prior_volume_fraction and logZ_corrected
for analyses that partially fix a joint prior pair, and the same two keys on
the results of core-built targets.

- **Unknown infer keywords are refused.** `ds.infer(..., **options)` used to
  pass every keyword it does not name to the sampler backends. They read
  options with `getattr` and a default, so a misspelt or foreign name was
  dropped without a word. Examples: `selection_neff_soft_guard=True`
  (bind_analysis's name) ran the hard wall; `max_likelihood_varience` left the
  cap at 1 nat; `n_live` left nlive at 1000; `resume_from` started a fresh
  run. For an InferenceTarget such keys were also hashed into the fingerprint.
  infer now raises `TypeError` before anything is bound, naming infer's
  spelling (`selection_neff_guard='soft'`, `resume_from_resolved`,
  `checkpoint_interval_seconds`) or the closest match. The runtime binding
  package had made the same keywords a `UserWarning`; the stricter refusal is
  kept. The command-line names `resume` and `checkpoint_interval`, which
  ds.infer never reads, are refused with a pointer to infer's names.
- **Unverified ordinary resumes warn.** `ds.infer` fingerprints
  InferenceTarget runs only. An ordinary analysis that resumes a dynesty or
  TinyNS checkpoint with no `run_fingerprint.json` beside it now warns that
  nothing checked the checkpoint's settings. The warning shows how to gate the
  run: `parameter_plan_semantic`, `core_numerics_semantic(bound)` and the data
  identity, passed to `gate_and_stamp_checkpoint_fingerprint`. The resume
  itself goes ahead. README.md, CONTRACT.md and this file's sections on the
  2026-10-02 speed defaults, host_mass and the pooled count-ratio default
  promised that a resume under changed defaults is refused; they now say that
  this holds for InferenceTarget runs and for fingerprints a caller gates
  itself, while an ordinary `ds.infer` resume only warns.
- **Code identity.** `core_environment_advisory()['code']` now carries
  `darksirens_version` and, when they can be read, the `git_sha` and
  `git_dirty` of the imported darksirens (from its checkout, or from
  `direct_url.json` for a VCS install) and `tinyns_commit`. The code-identity
  drift warning therefore fires on a resume under another core commit.
  `save_result` writes the same JSON as the `code_identity` HDF5 attribute.
  The advisory block is not hashed, so digests are unchanged. `__version__`
  still reads 0.1.0.dev0.
- **Plan records.** `ParameterPlan.catalog_model` records a conditional
  complete catalog's opt-in `empty_policy='volume'`; `'zero'` records nothing.
  `ParameterPlan.angular_model` records `angular`. `parameter_plan_semantic`
  adds the name only for `sphere_gp_z` and `overdensity_gp`
  (`ANGULAR_MODELS_SHARING_COORDINATES`), whose coordinates are identical.
  Only those plans get new digests. A `make_host_density_target` target now
  records the base analysis's plan semantic, its population and angular models
  in its own fingerprint provenance (see "Likelihood").
- **Prior-volume correction.** `log_prior_volume_fraction` used to hold the
  angular model's fraction only. Fixing one member of a joint prior pair
  samples the other member over its own bounds, with likelihood-side
  rejection. `ParameterPlan.fixed_constraint_log_fraction` now records the log
  fraction kept, and `ds.infer` adds it. Examples: `lambda_0 = 0.6` gives ln
  0.4 = -0.916; `m1_low = 5` gives ln(2/7) = -1.253; `m2_low = 5` gives
  ln(5/7) = -0.336. The fraction is not added (0.0, and the warning says so)
  when a member of the pair has a per-catalog copy (`per_catalog_population`):
  the likelihood is then zero wherever any catalog's population vector leaves
  the support (see "Likelihood"), and the fraction that keeps depends on which
  members are copied; it is not computed. `InferenceTarget` gains an optional
  `log_prior_volume_fraction` (a finite number <= 0; booleans and strings
  raise `TypeError`); `make_catalog_mixture_target` and
  `make_host_density_target` set it from their analysis. Their `ds.infer`
  results now carry `log_prior_volume_fraction` and `logZ_corrected` (0.0 and
  an unchanged logZ for isotropic and dipole skies, -0.505 nat for
  `multipole`, -3.381 for `multipole_l3`, plus a partially fixed pair's
  fraction). A target that declares nothing gets the result unchanged. The raw
  `logZ` is untouched everywhere. (The GP and registry package had added the
  same field and the angular part; the two are one feature here.)
- **Fixing m2_low.** Fixing the lower member of a `conditional_upper` pair
  gives the upper member a uniform prior over the interval left, not the
  declared conditional (proportional to 1/(m1_low - 3)). The warning now says
  that fixing it also changes the prior on the upper member and that the
  posterior differs, not only logZ, and that fixing the upper member gives the
  declared conditional.
- **ds.model option checks.** `count_ratio_window` accepts any real number
  (`np.float32`, `np.int64`). It refuses a window below the pooled histogram's
  bin width (1e-4), at which about half the redshift nodes saw no galaxy.
  `n0_units` is refused for a field mixture whose catalogs are all complete,
  as it already was for one complete catalog. The row_fraction error and the
  ds.model docstring now say that `row_fraction` also applies to the pooled
  count ratio (the default).
- **Kernel pin docs.** Under `catalog_sky_weighting='field'` the kernel pin
  applies to every catalog, complete ones included, and only a conditional
  complete catalog is unpinned. The code is unchanged; CONTRACT.md and the
  kernel pin section above now say so.
- **Startup-check remedies.** For an InferenceTarget, the nested-sampler
  startup check now says to build the target's likelihood with
  `selection_neff_soft_guard=True` or a larger `max_likelihood_variance`,
  since infer refuses both keywords for a target.

### Samplers and prior transforms

Default outputs do not change. probe_prior_transform, probe_dynesty_adapter,
probe_numpyro_init and probe_sampler_dispatch are byte-identical to legacy.
probe_joint_prior_resolver, probe_numpyro_static and probe_numpyro_adapter
each differ only in one item listed below: a warning text, an error text, and
the lognormal NUTS site. Each probe declares its item as a divergence from
the reference (`DECLARED_WARNING_SUFFIX`, `DECLARED_ERROR_TEXT`,
`DECLARED_LOGNORMAL_TRANSFORM`), as the population probe declares the GWTC-5
effective-spin values: each arm checks that its own output carries its
declared form exactly (the candidate's lognormal domain must be the site's own
truncation interval), and both arms report the reference's form, so the
phase6q, phase6s and phase7c1 CI gates compare everything else exactly. An
undeclared change, or the declared change missing, still fails them.

- **InferenceTarget joint constraints.** `InferenceTarget` and
  `combine_parameter_plans` now refuse a joint constraint whose cube map is
  not its declared density. Each member must be uniform (Beta(1, 1) on [0, 1]
  also counts). `ordered_le` and `conditional_upper` need equal bounds on both
  members, `simplex` needs [0, 1], and `ball3` needs [-1, 1]. These are the
  bounds the ordinary-path resolver already requires. Before, such plans were
  accepted and dynesty or TinyNS sampled another prior with no warning.
- **Maps that share a coordinate.** `make_prior_transform`, and so every
  nested-sampler run, refuses joint maps that share a coordinate unless they
  are `conditional_upper` chains or trees declared parent first. A chain
  declared child first, x0|x1 before x1|x2, used to break x0 <= x1 on half the
  draws; declare the conditioning pair first. NumPyro's chained-constraint
  error now says this.
- **Truncated normal and lognormal windows.** `make_prior_transform` refuses a
  `normal` or `lognormal` prior whose window the inverse CDF cannot resolve:
  the bounds hold less than 1e-10 of the normal's mass, or more than 1e-6 of
  the window's mass lies beyond 7.03 sigma of loc. Such priors used to give
  one constant draw (loc ± 7.03 scale), often outside the bounds. This can be
  reached through `survey_priors={name: ("normal", loc, scale)}`, which
  truncates to the default bounds. The `InferenceTarget` seam runs the same
  check and names the parameter.
- **NumPyro rejection.** NumPyro now applies the `ordered_le`, `simplex` and
  `ball3` constraints that its startup message announces as likelihood-side
  rejection: a `joint_prior` factor of 0 or -inf in the NUTS model, and an
  initial-point search that skips starts outside them. Ordinary analyses
  already reject there through their valid masks, so their NUTS chains are
  bitwise unchanged. An `InferenceTarget`'s NUTS posterior now respects its
  declared constraints.
- **NumPyro lognormal.** A NumPyro `lognormal` site now has support [lower,
  upper]. It used to be (0, inf), so NUTS sampled outside the prior box. This
  changes NumPyro runs that use a lognormal prior kind; no core registry
  declares one.
- **Dynesty finished checkpoints.** Resuming a checkpoint saved at the end of
  a finished run still samples nothing (dynesty's behaviour). It now prints
  that it does so and keeps the run's stop reason: a capped run reads back
  `"maxcall"`, not `"unknown"`.
- **conditional_upper warnings.** The rejection-fallback warning for
  `conditional_upper`, and the partial-fixing warning when its bounded member
  (for example m2_low) is fixed, now say that the prior density changes, not
  only logZ.

### Likelihood

Default analyses are unchanged: the candidate arms of the spectral, catalog,
catalog-mixture and per-catalog-population probes are byte-identical to
before.

- **Per-catalog population blocks** (`ds.model(...,
  per_catalog_population=...)`, opt-in). When any branch's population vector
  breaks one of the population model's joint prior constraints, the likelihood
  is now `-inf`, as it is with one shared population. For GWTC-5 those
  constraints are `lambda_0 + lambda_1 <= 1` and `m_2,low <= m_1,low`.
  - Before, the mixture's logsumexp dropped that branch and returned the other
    branches' finite value. Over the whole invalid region of a partially
    copied constraint group, that value was the `fcat_2 = 0` likelihood.
  - Catalog 1's own rejection under partial fixing was lost the same way.
  - Values at valid points are unchanged.
  - Population models that declare no joint constraint keep their program.
  - The warning "keeps the population model's likelihood-side rejection (the
    invalid region has zero likelihood)" is now true for these copies.
- **`make_host_density_target`**:
  - The target's likelihood is now one `jax.jit`, built once, over the
    coordinate, the GW samples and the distance table. Values are unchanged.
    With `sel_batch_size`, a call went from about 2.5 s to about 1 ms on the
    test toy.
  - `redshift_model.log_density` and `log_auxiliary_likelihood` are therefore
    always traced; before, only `sel_batch_size` or `pe_event_block` traced
    them. The opaque states are constants of the program.
  - The target now sets a run-fingerprint provenance block: the base
    analysis's plan semantic (fixed values included), its population model and
    sharing flags, its angular model, `n_events`, `nsamp` and `n_draw`, and
    non-default `max_likelihood_variance` and `selection_neff_soft_guard`.
  - It takes new `identity=` and `provenance=` keywords. The caller's
    provenance is recorded under `"companion"`.
  - Its jit traces the GW samples, where the reducer cannot count the live
    injections, so the target refuses an `n_draw` below them when it is
    built (the GW package's check, see "GW loaders" above).
  - The fingerprint of every host-density target changes, so a checkpoint
    written before this change resumes only with `resume_force=True`.
- **`compute_dtype="float32"`**:
  - `dataclasses.replace(bound, compute_dtype="float32")` now rounds and
    checks the per-sample columns as `bind_analysis` does, and refuses values
    below the float32 normal range.
  - The likelihood functions called directly with `compute_dtype` check their
    concrete columns the same way.
  - `replace(bound32, compute_dtype=None)` is refused ("compute_dtype=None
    runs the default float64 program, but gw_pe columns [...] are float32");
    bind again for the default program.
- **`bright_siren_log_likelihood`** now refuses a `gw_pe` whose length is not
  `n_events * nsamp`, as the other hierarchies do. Before, a too-small `nsamp`
  regrouped samples across events without an error.

### Packaging: the gwcat extra, the Python range, save_result, opt-in low-z interpolation, the TinyNS pin

- **TinyNS pin (kept; documented and guarded).** The base install still
  requires TinyNS 3f9e1b2 (version 0.1.0). It is the validated
  reconstruction stack (`VALIDATION.md`), the frozen reference's pin, and the
  TinyNS the full-suite regression job installs; moving it would change what
  every base install and that job run, which is a release decision, not a
  review fix. TinyNS 0.1.0 has no live-cov proposal, so the presets
  `livecov`, `bounded_single`, `bounded_multi` and `fused_bounded_multi` are
  refused on it before sampling, as before (`ValueError: ... needs tinyns >=
  0.2.0`). The refusal now also names those presets and the remedy: the
  TinyNS the real-backends job installs and runs `livecov` against
  (5b6da641c4d72b309778afdb0aea133b5de8c9e5, version 0.2.3). A test checks
  that commit against `real-backends.yml`, so the remedy cannot go stale. The
  README says which presets run on the base install. Still open, for the
  owner: moving the base pin to a 0.2.x release, or moving TinyNS into an
  optional extra like the other backends (which would also lift the
  direct-URL requirement that blocks a PyPI upload).
- **gwcat extra pin.** `pip install ".[gwcat]"` now installs gwcat
  3ba1dbf4116bb18f76c9103f79a914564d778182 (gwcat master) instead of 3184bdb.
  The old commit had no `ChiEffPrior.support`, so every load that folds the
  chi_eff prior in raised `ImportError: the installed gwcat predates the -inf
  out-of-support chi_eff convention`. That means a PE store with
  `chi_eff_in_p_pe=False`, or a selection store with
  `chi_eff_swap_applied=False`. Those stores now load. The prior folded in is
  gwcat's current default, the closed-form (`"exact"`) implementation. An
  environment with an older local gwcat that uses the grid table gives
  in-support values about 3e-3 apart in ln p. Stores that already carry the
  prior are unaffected. The frozen legacy reference pinned the same unusable
  commit.
- **Python range.** `requires-python` is now `>=3.11,<3.13`. The pinned numpy
  1.26.4 and scipy 1.12.0 have no CPython 3.13 wheels, so 3.13+ is refused at
  resolution instead of failing in a source build. 3.11 remains the validated
  interpreter.
- **`ds.save_result`.** A 0-d NumPy or JAX array entry is now written as an
  HDF5 attribute, as other scalars are; before, it was silently skipped. A JAX
  array is written to `result_json` as its full list of values; before, it was
  its `str()`, which is a `"..."` summary for a long array. Any other non-JSON
  object still falls back to `str()`. Results made only of Python types, which
  covers everything `ds.infer` returns, are written exactly as before.
- **Opt-in `DARKSIRENS_LOW_Z_INTERP=corrected`.** The default, `legacy`, is
  unchanged and bit-identical. The variable is read at import; an unknown
  value raises `ValueError`. It corrects two low-redshift chords inherited
  from the frozen reference:
  - `log_interp_zgrid` interpolates `log_grid - 2 ln z` and adds `2 ln z`
    back, so every cell follows the z^2 law, not only z < 0.0018. This covers
    the spectral volume prior, the catalog galaxy measure, the empty-catalog
    value and the bright-siren volume factor. Against quadrature, legacy log p
    is 5.8e-2 low at z = 0.002, 0.12 low near 0.0026 and 8.3e-3 low at 0.01;
    corrected is within 1.2e-4 at the fiducial and 3.4e-4 at the widest CPL
    corner.
  - `z_of_dL_precomputed` inverts the first distance cell (z < 0.0036)
    exactly, so `z_of_dL(dL_of_z(z))` returns z; legacy was up to 5.6e-3 low.
  - The float32 likelihood kernels follow the switch. The run fingerprint
    records `low_z_interpolation: "corrected"` only when set, so default
    fingerprints are unchanged and a checkpoint is not resumed across modes.
  - On the spectral-likelihood fixture the opt-in moves ln L by at most 7e-5.
- **New `H0` keyword.** `darksirens.cosmology.z_of_dL_precomputed(dL, dL_grid,
  H0=None)` takes an optional `H0`, the grid's Hubble constant. Only the
  opt-in reads it; under the opt-in, a call without it raises `ValueError`.
  Every core caller passes it.

### Documentation corrected with these fixes

- The README's `ds.log_likelihood` example passes one value per label of its
  catalog analysis (four), not one.
- The README counts five extension boundaries, the catalog-mixture
  missing-host seam included.
- The `Om0` support is given with its float64 endpoints (0.15749999999999997,
  0.45749999999999996): the literal 0.4575 is outside it.
- `kernel_layout="galaxy_list"` equals the padded layout to rounding, bit for
  bit only under the unrolled CPU schedule (README, the speed-defaults section
  above and `catalog/settings.py`).
- `core_numerics_semantic(bound)` equals `core_numerics_semantic()` only for a
  binding without a catalog view (CONTRACT.md and the run-fingerprint section
  above).
- The galaxy weights' effect under the count ratio is stated in README.md, the
  ds.model docstring and the selection-weights paragraph above: the count
  ratio counts raw galaxies while the weights set each row's redshift shape,
  and the selection mode's remedy does not exist there.
- The speed-defaults section above no longer says that the legacy probes bind
  with the kernel window: no CI probe fixture activates it.
- VALIDATION.md no longer lists `real-backends.yml` without `pipefail` as a
  known gap.
- CONTRACT.md's GP normaliser invariant states the accuracy of the default
  lattice rule at the prior corner.

