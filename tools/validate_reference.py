#!/usr/bin/env python3
"""Validate the frozen legacy reference bundle without importing darksirens."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFDIR = ROOT / "tests" / "reference" / "legacy"
GOLDEN = REFDIR / "unified_k1_golden.json"
MANIFEST = REFDIR / "unified_k1_manifest.json"


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"reference validation failed: {message}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    golden = json.loads(GOLDEN.read_text())

    if manifest.get("schema_version") != 2:
        fail(f"unexpected manifest schema {manifest.get('schema_version')!r}")

    ref = manifest["reference"]
    if ref["repository"] != "ignaciomagana/darksirens":
        fail("legacy repository changed")
    if ref["commit"] != "c042527238bd71421b792936bc48c3b815b90d6d":
        fail("legacy commit changed")
    if git_blob_sha(GOLDEN) != ref["golden_blob_sha"]:
        fail("frozen golden file is not byte-identical to the recorded legacy blob")

    stack = manifest["validated_cpu_stack"]
    required_stack = {
        "recorded_fast_gate_python": "3.11.10",
        "jax": "0.4.34",
        "numpy": "1.26.4",
        "scipy": "1.12.0",
        "pytest": "8.3.4",
    }
    for key, expected in required_stack.items():
        if stack.get(key) != expected:
            fail(f"validated stack {key} changed: {stack.get(key)!r}")

    expected_backends = set(manifest["recorded_backends"])
    if set(golden) != expected_backends:
        fail(f"backend set changed: {sorted(golden)}")

    cells = manifest["cells"]
    expected_cells = set(cells)
    if len(expected_cells) != 15:
        fail(f"expected 15 unified K=1 cells, found {len(expected_cells)}")

    valid_owners = {"core", "lss", "lensing"}
    for name, spec in cells.items():
        if spec.get("owner") not in valid_owners:
            fail(f"cell {name!r} has invalid owner {spec.get('owner')!r}")
        if not spec.get("meaning"):
            fail(f"cell {name!r} has no scientific meaning recorded")

    for backend, bank in golden.items():
        if set(bank) != expected_cells:
            missing = sorted(expected_cells - set(bank))
            extra = sorted(set(bank) - expected_cells)
            fail(f"{backend}: cell mismatch; missing={missing}, extra={extra}")
        for name, values in bank.items():
            if not isinstance(values, list) or len(values) != 3:
                fail(f"{backend}/{name}: expected exactly three coordinate values")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
                fail(f"{backend}/{name}: non-finite/non-numeric value")

    comparison = manifest["comparison"]
    if comparison["coordinate_fractions"] != [0.5, 0.35, 0.65]:
        fail(f"coordinate fractions changed: {comparison['coordinate_fractions']}")
    if comparison["canonical_rtol"] != 1e-12:
        fail("canonical relative tolerance changed")
    if comparison["atol"] != 0.0:
        fail("canonical absolute tolerance changed")

    drift = manifest["known_pinned_checkout_drift"]
    if drift["backend"] != "cpu":
        fail("known pinned-checkout drift backend changed")
    if set(drift["cells"]) != {"qdet", "use_lss", "ensemble_marg"}:
        fail(f"known drift cell set changed: {drift['cells']}")
    if drift["legacy_replay_rtol"] != 3e-12:
        fail("legacy replay tolerance changed")
    if drift["legacy_replay_rtol"] <= comparison["canonical_rtol"]:
        fail("legacy replay tolerance must be looser than canonical tolerance")

    # The two ordinary catalog representations are a direct parity pair.
    for backend, bank in golden.items():
        if bank["plain_full"] != bank["plain_compact"]:
            fail(f"{backend}: plain_full and plain_compact no longer agree exactly")

    print(
        "reference bundle valid: "
        f"{len(golden)} backends, {len(expected_cells)} cells, 3 coordinates/cell; "
        f"golden blob {ref['golden_blob_sha']}"
    )


if __name__ == "__main__":
    main()
