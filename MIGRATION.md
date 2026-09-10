# darksirens-core migration record

Reference: `ignaciomagana/darksirens@c042527238bd71421b792936bc48c3b815b90d6d`.

## Phase 1

Reference/parity infrastructure was established on `main` before scientific
implementation migration began.

## Phase 2 — foundation

| Legacy source | New owner | Migration mode | Status |
|---|---|---|---|
| `darksirens/core/jax_config.py` | `src/darksirens/_jax.py` | port/simplify interface | parity gated |
| cosmology constants from `core/constants.py` | `cosmology/parameters.py`, `distances.py` | split | parity gated |
| `darksirens/utils/interp2d.py` | `cosmology/_interpolation.py` | move/private | parity gated |
| `darksirens/utils/cosmology.py` | `cosmology/distances.py`, `volume.py` | split | parity gated |
| `darksirens/redshift/grid.py` | `cosmology/_grid.py` | move/private | parity gated |
| `CosmoParams` from `core/types.py` | `cosmology/parameters.py` | move/rename | parity gated |
| `darksirens/gw/store_contract.py` | `gw/store.py` | move/simplify docs | exact parity |
| `GWStore`, `SelectionStore` from `gw/utils.py` | `gw/types.py` | move | exact parity |
| standard loaders from `gw/utils.py` | `gw/samples.py` | split/clean dependencies | exact parity |
| `gw/samples.py` facade | `gw/__init__.py` + `gw/samples.py` | fold | exact parity |

The deterministic separate-process probes at the Phase-2 validation gate report
zero numerical difference for cosmology and zero difference for the serialized
GW store outputs on the shared fixtures. The store comparison is exact
(`rtol=0`).

## Deliberately not migrated in Phase 2

- `gw/populations/*` — next population milestone;
- `gw/flows.py` — deferred until the ordinary sample path is stable;
- `gw/selection.py` — pdet-flow surrogate, deferred;
- catalog/redshift prior/completeness code;
- hierarchical likelihood and selection-integral code;
- inference/sampler code;
- any LSS or lensing implementation.

## Structural cleanup already achieved

The GW store layer no longer imports `healpy`, progress-bar machinery, or
population code. It owns only standardized posterior/injection records and
validation/processing of those records.

The package root no longer initializes JAX or constructs cosmology tables.
