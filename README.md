# darksirens-core

Clean reconstruction of the core `darksirens` hierarchical siren inference
package.

The legacy numerical reference is
`ignaciomagana/darksirens@c042527238bd71421b792936bc48c3b815b90d6d`.
Reconstruction state and architectural decisions are tracked in
`ignaciomagana/darksirens-rebuild`.

## Reconstruction status

Phase 1 froze the cross-domain legacy likelihood reference bank.

Phase 2 establishes the first scientific core:

- explicit JAX runtime configuration;
- JAX flat-CPL cosmological distances and comoving volume;
- the shared logarithmic redshift grid;
- standardized gwcat PE/injection store contracts;
- density-aware `chieff`, `component`, and `chieff_reference` basis handling;
- named GW and selection store loaders.

The Phase-2 branch is parity-gated against the pinned legacy implementation in
separate processes. Cosmology currently agrees exactly on the deterministic
probe, and standardized GW-loader outputs agree exactly on the shared HDF5
fixtures.

Population models, catalog/redshift models, the hierarchical likelihood,
selection effects, and samplers are migrated in later parity-gated phases.
