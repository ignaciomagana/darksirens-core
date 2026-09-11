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

The ordinary package-root surface is:

```python
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
  optional-backend import.

## Packaging contract

The base distribution includes the validated numerical stack plus the pinned
TinyNS commit used by the legacy campaign, because TinyNS is the public default
sampler. SciPy is a base dependency because ordinary completeness evaluation
uses `scipy.special` at runtime.

GP models, Dynesty, NumPyro and gwcat prior support are explicit extras. Raw
survey/LSS/lensing packages are never package dependencies of core.
