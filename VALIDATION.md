# Validation contract and accepted reconstruction state

`darksirens-core` is reconstructed against a pinned numerical reference rather
than by copying the legacy package wholesale.

## Frozen legacy reference

```text
repository: ignaciomagana/darksirens
commit:     c042527238bd71421b792936bc48c3b815b90d6d
```

The immutable cross-domain reference bank under `tests/reference/legacy/` may
not be regenerated from reconstructed code.

## Validated reconstruction stack

```text
CPython:     3.11
JAX/JAXLIB:  0.4.34
NumPy:       1.26.4
SciPy:       1.12.0
h5py:        3.12.1
Astropy:     6.1.4 (validation/legacy probe dependency)
TinyNS:      3f9e1b2537f32b59f17ee9ce68b2d725681a024c
backend:     CPU for parity gates
x64:         enabled for scientific evaluation
```

The package metadata freezes the core numerical runtime to this validated stack.
Optional sampler/GP integrations remain separately declared.

## Acceptance rule

A stochastic posterior agreement is never a substitute for fixed-coordinate
likelihood/selection parity. A migrated slice is accepted only after its
specific scientific/behavioral gate and the relevant broad/historical gates pass
on the same exact head.

Harness-only failures are repaired in the harness. Production scientific code is
changed only for a demonstrated semantic failure.

## Major accepted milestones

### Phase 2 — foundation

```text
workflow: 34431185197 / 102726848203 SUCCESS
```

Cosmology and standardized GW-store probes were exact on the deterministic
fixtures, including exact GW serialized outputs.

### Phase 3 — population

Accepted population parity covered the frozen parametric registry and GP model
interfaces. Final broad acceptance was 125 passed with one regeneration-only
skip; fixed-point population comparison was exact at the frozen tolerance.

### Phase 4 — spectral likelihood

```text
accepted head: cfdb138d40d66614bf9b1264c2574d0b497b812d
workflow:      34445525661 / 102769369532 SUCCESS
merge:         0f97feff7eb283a1f541bef9a776c9347084e70e
post-merge:    34447108888 / 102774202182 SUCCESS
```

Separate-process legacy/candidate parity was exact for per-event evidence and
variance, selection `log_mu`, effective sample size, selection correction and
full spectral likelihood at the frozen fixed points.

### Phase 5 — ordinary catalog, dark/complete/bright sirens

```text
merge:      86e0c88a51482d17fac70f111057d277df9387fd
post-merge: 34534097673 / 103061536070 SUCCESS
```

The accepted phase covers ordinary catalog compression/redshift kernels,
non-LSS completeness, incomplete and complete catalog likelihoods, bright
counterparts and generic host marks.

### Phase 6 — inference/runtime I/O

```text
accepted integration head: d3e8dcdbf107d881405d1f14badab7bd0ea4d74f
merge:                     d82becaf76bf62c0f72a71b32ebbf9b238ba4f13
post-merge integrity:      34592289964 / 103240104097 SUCCESS
```

The final sampler-dispatch slice passed dedicated, runtime and historical gates
at the same exact head. This phase freezes zero-dimensional exact evidence,
prior transforms, TinyNS/Dynesty/NumPyro adapters, checkpoints, result assembly
and provenance primitives.

### Phase 7 — public ordinary API + angular models

```text
accepted head: be95e95bdf77144880cba5752ed5feafd40a697f
merge:         6ed3dc74aa0fcde4da128036cc3d56c29250d370
post-merge:    34633321021 / 103375170965 SUCCESS
```

The Phase-7 PR historical/scientific matrix completed 30/30 green. The accepted
public surface includes loaders, declarative cosmology/population specs,
`ds.model`, runtime binding, `ds.infer`, portable HEALPix RING geometry and the
full reusable angular model family.

### Phase 8A — public counterpart/bright composition

```text
accepted head: 5256141e00a2d96a72b9cbf2182a77174c8c269e
8A dedicated:  34634374241 / 103378585284 SUCCESS
broad:         34634374571 / 103378586398 SUCCESS
broad result:  522 passed, 1 skipped
```

Frozen Phase-5 bright-likelihood parity replayed green.

### Phase 8B — `InferenceTarget`

```text
accepted head: 2e98ca1f1a67cf688f8da8444c3747d446a4e91d
8B dedicated:  34635288617 / 103381567625 SUCCESS
8A replay:     34635288475 / 103381567028 SUCCESS
broad:         34635288518 / 103381567351 SUCCESS
broad result:  528 passed, 1 skipped
```

The specialized-target path reuses the accepted prior/sampler stack and keeps the
zero-dimensional exact-evidence short circuit before backend validation/import.

### Phase 8C — host-density extension seam

```text
accepted head: 875a949d5a9f5ffb89f3a64cad030dcfc6daf6a2
accepted tree: 1219bba07d3327c83da0164602e60abe8083ba0b
8C dedicated:  34636321077 / 103384995159 SUCCESS
8B replay:     34636321027 / 103384994787 SUCCESS
8A replay:     34636321033 / 103384994066 SUCCESS
broad:         34636321017 / 103384994129 SUCCESS
broad result:  534 passed, 1 skipped
```

The dedicated test proves that an external-style volume redshift model reproduces
the accepted spectral likelihood through the generic host-density wrapper,
auxiliary likelihood is added exactly once, parameter-block offsets are correct,
and the seam imports no companion package.

## Final Phase 8D gate

The final core-freeze workflow builds a wheel and installs it into a clean Python
3.11 environment. It checks package metadata, light root import, base runtime
dependencies, the pinned TinyNS default backend, public examples, default
zero-dimensional target dispatch, and, at the time of 8D, that no scientific
source file changed relative to accepted 8C.

That last check was retired in the second review follow-up: Phases 12A and 12B
changed the science source on purpose, so it failed on every later tree and the
checks behind it stopped running. It was not moved to a newer pinned commit,
which would go stale at the next accepted science change. The workflow
(`phase8d-final-freeze.yml`) now runs on pull requests, pushes to `main` and
manual dispatch, and runs `tests/test_packaging_contract.py` and
`tests/test_examples.py` against the installed wheel, failing on any failure or
skip. Both public examples execute: `custom_target.py` against its analytic
evidence and `ordinary_catalog.py` on tiny stores with a reduced sampler.

Phase 8D acceptance and the final post-merge integrity run are recorded in
`ignaciomagana/darksirens-rebuild`; this document intentionally does not predict
run IDs before those exact-head gates exist.

## Phases 12A and 12B

PR #8 (squash merge `8b9dc64`) and PR #9 (squash merge `bb4812d`), each with a
control record in `ignaciomagana/darksirens-rebuild` (`phases/12A_*`,
`phases/12B_*`), adding `build_incomplete_catalog_prior_state_from_curves`,
`catalog/field.py` and `selection/footprint.py`. Accepted by
`tests/test_completion_curve_composition.py` (the curve seam reproduces the
accepted incomplete-catalog state exactly) and
`tests/test_phase12b_field_footprint.py`. No dedicated workflow; the broad
regression suite covers both.

## Review follow-up

A 130-agent adversarial review against the frozen reference confirmed 40
findings; the stacked PRs #10, #16, #12, #13, #14 and #15 implement them, merged
in that order. Acceptance is mutation-based: every fix ships with a test that
fails on the review's own mutant, verified in the PR history. The load-bearing
ones:

```text
dV_of_z without 1/E(z)             astropy anchor fails, ratio 4.95
ddL_of_z one-off in (1+z)          finite-difference anchor fails, ratio 0.91
m1src = m1det (no source frame)    pinned weight fails (0.58 rel), dlnL/dH0 0.070 -> 0.009
distance mask removed              Phase 4 probe fails, max rel 3.1; unit anchor fails
GP z nodes uniform in z            all 9 z-normalisation checks fail (up to 1.19)
fixed 64-node m1 table             toe probes fail (1.25 at m1 = m_min + 0.25 dm_min)
24-node q lattice, m1-conditional  independent-grid check fails (0.876 at the same probe)
ang2pix_ring pre-fix               12 of 49 healpy parity cases fail
soft guard without double-where    Neff = inf gradient is NaN
```

Parity was re-established after the changes: the Phase 2 cosmology probe
(`compare_cosmology_probe.py`, rtol 0) and the widened Phase 4 spectral probe
(`compare_spectral_probe.py`, rtol 1e-12) both pass legacy against candidate,
every other `tools/probe_*.py` candidate output is byte-identical to the
pre-branch tree, and `tests/data/population_registry_golden.json` gained two
`@md` entries with every pre-existing entry unchanged.

Local measurement on the validated stack with every optional backend installed
(TinyNS, Dynesty, NumPyro, healpy, Matplotlib), on the final tree of the stack:
714 passed, 1 skipped (the golden-regeneration guard), against 543 passed,
1 skipped before the stack. The reproducible record is the GitHub
Actions matrix on the PR heads. `phase8-regression` installs the same optional
stack, and the dedicated `real-backends` workflow runs the end-to-end TinyNS,
Dynesty and NumPyro tests, the real Dynesty checkpoint round-trip and the
healpy parity suite and fails on any skip. Every PR-triggered workflow passed
on every head below; the load-bearing runs:

```text
main after #10 (4c0e960)  reference-integrity          35326494368  SUCCESS
main after #16 (8bb1fc2)  reference-integrity          35326699909  SUCCESS

#12 head b7304ad          phase4-spectral-likelihood   35327465851  SUCCESS
                          phase5-catalog-dark-bright   35327465922  SUCCESS
                          phase8-regression            35327466013  SUCCESS
#13 head 2c8e646          phase3-population            35327467822  SUCCESS
                          phase4-spectral-likelihood   35327467787  SUCCESS
                          phase5-catalog-dark-bright   35327467821  SUCCESS
                          phase8-regression            35327467785  SUCCESS
#14 head f5e070c          phase7c3-healpix-geometry    35327467109  SUCCESS
                          phase8-regression            35327467040  SUCCESS  714 passed, 1 skipped
                          real-backends                35327467082  SUCCESS
#15 head 9a87ffb          phase2-foundation            35327952962  SUCCESS
                          phase4-spectral-likelihood   35327952991  SUCCESS
                          phase5-catalog-dark-bright   35327952955  SUCCESS
                          phase7-regression            35327952966  SUCCESS
                          phase8-regression            35327952967  SUCCESS  714 passed, 1 skipped
                          real-backends                35327953066  SUCCESS  75 passed, 0 skipped
```

The commit that records this table changes no code, so the `9a87ffb` runs
stand for the final tree of the stack.

Deliberate numerics changes and their reach are recorded in `MIGRATION.md`.

### Second review follow-up

The stacked PRs #17 to #23 finish the items the first follow-up left open. Only
the GP mass-ratio normaliser changes numerics; the rest are a diagnostic, a
provenance fix, a message fix and stronger gates. As before, each ships with a
check that fails on a mutant:

```text
package-root name renamed                 5 wheel-mode packaging/example tests fail;
                                          the release job now fails with them
PE sample mask removed from core          angular-wiring probe fails; old probe passed
NumPyro plan/initialisation dropped       sampler-dispatch probe fails; old probe passed
preflight text reworded outside the map   nested-preflight comparison fails
fingerprint hashed via json default=str   12 of 34 fingerprint tests fail
budget audit without completeness         scipy anchor fails, 1041.25 against 2.656
previous GP mass-ratio normaliser         8 of 8 near-minimum-mass cases fail
                                          (0.81 / 0.58 at m_min + 0.05 dm_min)
```

What each gate now checks:

- The angular-wiring probe (`tools/probe_angular_wiring.py`) runs the frozen
  likelihood module's own per-event PE reduction on the reference side, and its
  data include an invalid PE sample, a zero-prior-weight sample and `-inf`
  population weights, so the PE sample mask matters.
- The sampler-dispatch probe (`tools/probe_sampler_dispatch.py`) runs the frozen
  `run_sampler` end to end on the reference side and core's real dispatcher,
  adapters and NumPyro preparation on the other. Only the external backends are
  replaced, by the same recording fakes on both sides; the checkpoint plan and
  nested preflight are the same stubs on both sides, so the checkpointing
  branches are not covered by this probe.
- The nested-preflight comparison is word for word except for a documented
  table (`LEGACY_REMEDY_MAP` in `tools/probe_nested_sampler_preflight.py`) that
  maps the old command-line advice to the `infer` keywords.
- The fingerprint parity check in `phase6-inference-io.yml` allows exactly the
  schema change from 3 to 4 and the reworded schema-mismatch line; every other
  part of the probe output, including the plain-JSON digests, must match.
- `selection_budget_audit` matches the frozen function to 1e-12 and an
  independent scipy quadrature to 1e-5.
- The GP mass-ratio check integrates on a grid spanning only the allowed
  mass-ratio range, for the fiducial and a narrow taper; the registry golden
  file holds only parametric models and does not cover this change.

Known gaps kept on purpose: `real-backends.yml` still pipes pytest into `tee`
without `pipefail`, so a failing test there does not by itself fail that job; only the names of the package-root API are frozen, not their
signatures; the example checks are smoke checks, not numerical anchors.

Local measurement on the final code tree of the stack (`519100d`, the head of
#22), with the same optional backends as above: 759 passed, 1 skipped (the
golden-regeneration guard), against 714 passed, 1 skipped before the stack. The
45 new tests are 11 packaging and example tests, 2 preflight-message tests, 19
fingerprint tests, 5 budget-audit tests and 8 GP mass-ratio cases.

### Catalog kernel pin

`model(..., kernel_pin="auto")` evaluates the catalog kernel state once, at
bind time, when `Om0`, `w0`, `wa`, `delta` and `sigma_kde` are fixed (see
MIGRATION.md). `tests/test_kernel_pin.py` (44 tests) checks the activation
rule through `ds.model`, the pinned likelihood against `kernel_pin="off"` at
rtol 1e-12 for H0 from 20 to 140 (three plans, single pass and scanned blocks),
the pinned state leaf by leaf against the per-call state (padding and empty
rows exact, occupied-row offsets and per-sample densities within 1e-12
absolute), gradients, the program without the pin operand, the probe against
pins built under six violated premises, the fingerprint and resume gate, and
the bound jit's no-retrace, data-as-argument and pickling properties. Each
mutant fails the new tests:

```text
shift sign flipped                        16 of 44 fail
row offset not shifted                     2 fail (occupied-row offsets)
probe tolerance 1e30                       6 fail (the six violated premises)
probe verdict not spent on log_Z           6 fail
activation ignores sigma_kde               3 fail
activation ignores kernel_pin="off"       11 fail
binding does not serve the pin             6 fail
fingerprint without the kernel_pin block   2 fail
pin built at H0 = 70                       2 fail
```

With `kernel_pin="off"`, and for every plan the pin does not apply to, the
lowered StableHLO of the bound likelihood is byte-identical to the previous
release's (spectral, incomplete and complete catalogs, blocks single and
32/1, each lowered in a fresh process).

## Permanent firewall

The broad Phase-8 regression statically parses `src/darksirens/**/*.py` and
rejects imports of survey, LSS or lensing companion packages. That firewall is a
core release invariant, not a temporary reconstruction diagnostic.
