#!/usr/bin/env python3
"""Evaluate the frozen K=1 fixture using a separate pinned legacy checkout.

This script deliberately does not import reconstructed ``darksirens`` code. The
legacy checkout path is inserted at the front of ``sys.path`` and the legacy
source test is loaded directly from that checkout. The result is serialized for
the neutral comparator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

PINNED_SHA = "c042527238bd71421b792936bc48c3b815b90d6d"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.legacy_root.resolve()
    source = root / "tests" / "test_unified_k1_golden.py"
    if not source.exists():
        raise SystemExit(f"legacy golden source test not found: {source}")

    # Keep JAX on CPU for the canonical replay bank. The workflow sets this too,
    # but fail early if somebody invokes the script with a conflicting backend.
    requested = os.environ.get("JAX_PLATFORMS")
    if requested not in (None, "", "cpu"):
        raise SystemExit(f"legacy CPU replay requires JAX_PLATFORMS=cpu, got {requested!r}")
    os.environ["JAX_PLATFORMS"] = "cpu"

    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("legacy_unified_k1_golden", source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load legacy source test: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    backend = module._golden_backend_key()
    if backend != "cpu":
        raise SystemExit(f"legacy replay resolved unexpected backend {backend!r}")

    values = {name: module._evaluate_cell(name) for name in sorted(module.CELLS)}
    payload = {
        "cpu": values,
        "_provenance": {
            "legacy_repository": "ignaciomagana/darksirens",
            "legacy_commit": PINNED_SHA,
            "source_test": "tests/test_unified_k1_golden.py",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(values)} legacy cells to {args.out}")


if __name__ == "__main__":
    main()
