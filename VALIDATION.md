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
zero-dimensional target dispatch, and that no scientific source file changed
relative to accepted 8C.

Phase 8D acceptance and the final post-merge integrity run are recorded in
`ignaciomagana/darksirens-rebuild`; this document intentionally does not predict
run IDs before those exact-head gates exist.

## Permanent firewall

The broad Phase-8 regression statically parses `src/darksirens/**/*.py` and
rejects imports of survey, LSS or lensing companion packages. That firewall is a
core release invariant, not a temporary reconstruction diagnostic.
