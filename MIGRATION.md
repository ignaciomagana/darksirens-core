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
  from `DARKSIRENS_*` variables; `infer()` calls neither, nor the resume gate.
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

The pinned likelihood is not bit-identical to the per-call quadrature: the two
round differently. On the harness fixtures T and S (plans `dark_H0`,
`dark_pop`, `dark_joint_cosmo_pop`, single pass and blocked) the total log
likelihood agrees to 7.6e-16 relative, every gate field to 6.7e-14 relative
and every per-sample catalog log-density to 5.7e-14 absolute; against the
reference, which pins, the total agrees to 2.9e-16 relative (bit for bit in 8
of 12 cells, against 4 of 12 unpinned). On CPU a call is 1.4 to 6.7 times
faster (T `dark_H0`: 15.8 ms against 105.7 ms), for about 1 s more at bind.
The pin keeps two `(n_rows, n_max)` float64 arrays (the fused kernel
log-weights and the inverse widths) and four per-row vectors (224 MB on a
196,608 x 70 catalog); the reference keeps five such arrays. The pin is built
by the per-call kernel builder, run once at bind, so binding needs the
transient memory of one unpinned kernel build: on that catalog the CPU
compiler gives the build 14.9 GB of scratch, while the per-call catalog terms
need 3.2 GB pinned against 15.4 GB unpinned. Complete-catalog analyses are
not pinned, as in the reference. `kernel_pin="off"` keeps the per-call
quadrature, and the bound program is then byte-identical to the previous
release's. `ParameterPlan` gains `kernel_pin` and `kernel_pin_active`, and
`run_fingerprint.parameter_plan_semantic(plan)` records both.
