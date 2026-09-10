# darksirens core contract

## Role

This repository builds the Python distribution and import package `darksirens`.
It owns the reusable hierarchical siren/population inference kernel.

Reference behavior during reconstruction is
`ignaciomagana/darksirens@c042527238bd71421b792936bc48c3b815b90d6d`.

## Dependency direction

Companion packages may import `darksirens`:

- `darksirens-surveys`
- `darksirens-lss`
- `darksirens-lensing`

Core must never import those packages.

## Current implemented surface

Phase 2 currently owns:

- explicit, side-effect-free JAX runtime configuration;
- flat-CPL cosmological distance/volume tables;
- the shared logarithmic redshift grid;
- standardized gwcat PE and detected-injection store contracts;
- named GW/selection store records;
- validated PE/injection loaders.

Population, catalog, likelihood, selection, inference and public high-level
analysis construction are deliberately not implemented in this phase.

## Numerical invariants

### Cosmology

- x64 is enabled before float64 cosmology grids are constructed;
- the interpolation grids, node counts, CPL expansion law and low-z correction
  reproduce the pinned legacy implementation;
- the large distance table is threaded through JIT boundaries as data rather
  than captured as a repeated HLO literal;
- `DARKSIRENS_ZMAX` retains its legacy meaning.

### GW stores

- PE `p_pe` is non-negative; zero is legal and becomes zero importance weight;
- detected-injection `pdraw` is strictly positive and retains physical scale;
- PE proposal density is normalized independently per event;
- every required store column is 1-D with a contract-consistent length;
- detector- and source-frame masses obey `m2 <= m1`;
- sky coordinates are finite radians in their declared ranges;
- spin-basis negotiation is density-aware, including advisory columns;
- `chieff_reference` selection stores require `spin_reference_amax` and an
  applied reference-prior reweighting;
- component-basis stores may be consumed only by a model fitting the component
  spin coordinates;
- `gwcat` is imported lazily only when the loader must evaluate a chi_eff prior.

## Import contract

`import darksirens` must remain light and must not import JAX. Calling
`darksirens.configure_jax_runtime()` performs the process-level JAX
configuration explicitly.

Importing scientific subpackages such as `darksirens.cosmology` or
`darksirens.gw` may import JAX.

## Future stable user API

The intended high-level surface remains:

```python
import darksirens as ds

events = ds.load_events(...)
injections = ds.load_injections(...)
catalog = ds.load_catalog(...)
cosmo = ds.Cosmology(...)
pop = ds.Population(...)
analysis = ds.model(...)
result = ds.infer(...)
```

That surface is not frozen until the ordinary core likelihood reaches parity.
