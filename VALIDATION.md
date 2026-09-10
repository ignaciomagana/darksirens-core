# Validation contract and current results

This repository is reconstructed against a pinned numerical reference rather
than by copying the legacy package wholesale.

## Legacy reference

```text
repository: ignaciomagana/darksirens
commit:     c042527238bd71421b792936bc48c3b815b90d6d
```

The cross-domain likelihood reference bank is stored under
`tests/reference/legacy/` and may never be regenerated from the reconstructed
implementation.

## Validated CPU stack for reconstruction parity

```text
CPython: 3.11
JAX/JAXLIB: 0.4.34
NumPy: 1.26.4
SciPy: 1.12.0
h5py: 3.12.1
Astropy: 6.1.4
pytest: 8.3.4
ruff: 0.15.20
backend: CPU
x64: enabled for scientific parity probes
```

## Phase 1 — frozen likelihood reference

The immutable reference and pinned-legacy replay workflows are green. The
canonical reconstructed target remains `rtol=1e-12, atol=0`; the separately
documented ~2.3e-12 drift of three legacy Q/LSS cells is confined to the
legacy-replay profile.

## Phase 2 — foundation and GW input contracts

Final scientific validation on branch `rebuild/phase2-foundation` used workflow
run `34431185197`, job `102726848203`.

Results:

```text
ruff definite-error gate: PASS
candidate unit tests: 14 passed
cosmology legacy/new probe: PASS
cosmology max_abs: 0.000e+00
cosmology max_rel: 0.000e+00
GW store legacy/new probe: PASS
GW store max_abs: 0.000e+00
GW store max_rel: 0.000e+00
GW comparison mode: exact (rtol=0)
light package-root import: PASS
```

The cosmology probe covers the distance table shape/samples, the shared redshift
grid, three background cosmologies, low/intermediate/high redshift distances,
`dV/dz`, `E(z)`, `ddL/dz`, inverse `dL -> z`, distance modulus, and redshift-grid
index/interpolation behavior.

The GW probe uses the same generated HDF5 fixtures in independent legacy and
candidate processes and covers `gwcat-*-2.1` stores in:

```text
chieff
component
chieff_reference (selection)
```

It compares validated raw columns, processed PE proposal weights, physical
selection `pdraw`, fit-column metadata, event names and counts exactly.

Candidate unit tests additionally pin malformed length rejection, invalid sky
and mass ordering, `p_pe == 0` vs `pdraw == 0`, `ndraw` consistency,
component-model negotiation, required `spin_reference_amax`, and unswapped
chi_eff draw-density folding.

The final code audit also ensures XLA allocator defaults are applied before JAX
is imported by the GW loader module; explicit user environment values still win.

## Acceptance rule

Stochastic posterior agreement never substitutes for fixed-coordinate
likelihood/selection parity. As higher-level core code is migrated, it must
also pass the frozen 15-cell likelihood bank before being marked complete.
