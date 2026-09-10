# Frozen legacy numerical reference

This directory is immutable reference data for reconstructing `darksirens`.

Source:

```text
repository: ignaciomagana/darksirens
commit:     c042527238bd71421b792936bc48c3b815b90d6d
source test: tests/test_unified_k1_golden.py
source golden: tests/golden/unified_k1_golden.json
```

`unified_k1_golden.json` is byte-identical to legacy Git blob `560e44adb712763893111a3607f6d7ca8b168b7a`. The source test defining the fixture and its feature-liveness probes is Git blob `0f4e3301db2b4a73c574791a871d53fd56f0cad2`.

The bank contains 15 likelihood cells evaluated at parameter-space fractions `[0.5, 0.35, 0.65]`. The recorded backends are `cpu` and `gpu:NVIDIA H100 NVL`. The legacy comparison contract is `rtol=1e-12`, `atol=0`; exact equality is expected on the same backend when the operation ordering is preserved.

## Candidate output

A reconstructed evaluator writes JSON with the same shape:

```json
{
  "cpu": {
    "plain_full": [0.0, 0.0, 0.0],
    "spectral": [0.0, 0.0, 0.0]
  }
}
```

It may contain all cells or only the cells owned by one package. Compare with:

```bash
python tools/compare_reference.py candidate.json --backend cpu --owner core
```

Use `--exact` only when exact same-backend operation-order parity is the intended gate.

## Regeneration rule

Never regenerate this bank from reconstructed code. A reference update requires:

1. an explicit new pinned legacy commit;
2. regeneration on that legacy implementation;
3. an updated source/golden blob SHA in the manifest;
4. a documented reason in the reconstruction control repo.

A missing backend is preferable to borrowing values from a different GPU model.
