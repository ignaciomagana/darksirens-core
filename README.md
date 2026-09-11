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
stack used for reconstruction parity and includes TinyNS, the public default
sampler:

```bash
pip install .
```

Optional integrations are explicit:

```bash
pip install ".[gp]"       # tinygp/equinox population models
pip install ".[dynesty]"  # Dynesty backend
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
    sampler="tinyns",
)
```

The same model/inference stack supports catalog-free spectral sirens,
complete-catalog analyses, bright/counterpart sirens, reusable angular source
models and sampled population models. The standardized PE/injection and catalog
loaders consume reconstructed core file contracts; raw survey construction is
not a core responsibility.

Runnable public-API templates are under `examples/`.

## Specialized analyses

Core exposes two explicit one-way extension boundaries instead of a plugin
framework.

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

For field/LSS-style redshift information, `darksirens.likelihood.host_density`
provides the explicit `RedshiftModel` protocol and target builder. Companion
packages own their field/count/tracer state; core owns the GW weighting,
selection reduction, prior assembly and sampler execution.

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
See `VALIDATION.md` and the phase records in `darksirens-rebuild` for exact
heads, workflow runs and tolerances.
