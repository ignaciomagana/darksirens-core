# Public decode_parameters PR: progress (2026-10-02)

Branch `feat/public-decode-parameters` from origin/main f527b94. This file is a
WIP checkpoint and must be deleted before the PR is opened/merged.

## Done
- `src/darksirens/runtime_binding.py`: `DecodedParameters(cosmology, population,
  catalog, angular)` NamedTuple; `_decode_theta` returns it (unpacks as the old
  4-tuple; bound likelihood still calls it); public `decode_parameters(analysis
  | BoundAnalysis, theta, *, z_depth=BINDING_DEPTH)` validates shape (ValueError
  naming labels), resolves z_depth (catalog store's / binding's / None;
  explicit z_depth refused with a BoundAnalysis) and calls `_decode_theta`.
- `src/darksirens/_binding_depth.py`: JAX-free sentinel; root `ds.decode_parameters`
  in `__init__.py` + `__all__`.
- Docs: CONTRACT.md (frozen root list + "Parameter decoding" section with
  stability promise), README, MIGRATION.md; packaging test and phase8d freeze
  list updated.
- `tests/test_decode_parameters.py` (20 tests, ~50 s loaded node): jaxpr identity
  with the bound decode for 8 cases; HLO+logL bitwise of the bound rebuilt on the
  public decode for 3 cases; fields/fixed values/h-scaled n0; jit/vmap/grad;
  wrong-shape rejection; type errors.

## Verified
- Identity vs clean clone of origin/main (scratchpad/decodepr/identity.py,
  id_*.json, id2_*.json): 13 cases (mock T spectral hard/soft, T dark sampled,
  fixed+pin, pin off, h-scaled+z_depth, complete, dipole; real 259 spectral
  gwtc5 and powerlaw+peak, hard and soft guard): optimized HLO (vmap and single)
  and logL bitwise ALL IDENTICAL.
- lss coverage (scratchpad/decodepr/lss_cover.py): lss's `_decode_base` +
  `_decode_survey` equal the public decode bitwise (eager, jit, jit+vmap) for two
  tracers, h-scaled and physical, when tracer k's own theta is gathered from the
  combined theta.
- Finding: h-scaled n0 differs by 1-2 ulp between eager / jit / vmap programs
  (XLA rewrites (H0/100)**3); documented.

## Left
- Full suite (running: scratchpad/decodepr/full_suite.log).
- CI replay (scratchpad/decodepr/ci_local.py): first pass failed only because
  ruff is not on PATH in --system-site-packages venvs (rc=127); fix the runner
  (symlink conda ruff into the venv bin) and rerun; check phase8d wheel install
  failure; also run legacy-reference-replay.yml explicitly.
- Remove this file, push, open PR with gh -R ignaciomagana/darksirens-core
  (body file; do not merge), report back.
