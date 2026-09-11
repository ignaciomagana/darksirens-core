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
