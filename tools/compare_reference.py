#!/usr/bin/env python3
"""Compare fixed-coordinate outputs against the frozen legacy golden bank.

Candidate JSON uses the same backend -> cell -> [three values] shape as the
reference. The default ``canonical`` profile is the acceptance target for the
reconstructed implementation. ``legacy-replay`` exists only to verify the
known numerical state of the pinned legacy checkout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFDIR = ROOT / "tests" / "reference" / "legacy"
GOLDEN = REFDIR / "unified_k1_golden.json"
MANIFEST = REFDIR / "unified_k1_manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path, help="candidate JSON file")
    p.add_argument("--backend", default="cpu", help="reference backend key")
    p.add_argument(
        "--owner",
        choices=("all", "core", "lss", "lensing"),
        default="all",
        help="compare only cells assigned to one reconstructed package",
    )
    p.add_argument(
        "--profile",
        choices=("canonical", "legacy-replay"),
        default="canonical",
        help="canonical reconstructed target or pinned-checkout replay envelope",
    )
    p.add_argument(
        "--exact",
        action="store_true",
        help="require exact float equality; valid only with the canonical profile",
    )
    return p.parse_args()


def relerr(got: float, expected: float) -> float:
    if got == expected:
        return 0.0
    if expected == 0.0:
        return math.inf
    return abs(got - expected) / abs(expected)


def tolerance_for(manifest: dict, backend: str, cell: str, profile: str) -> float:
    canonical = float(manifest["comparison"]["canonical_rtol"])
    if profile == "canonical":
        return canonical

    drift = manifest["known_pinned_checkout_drift"]
    if backend == drift["backend"] and cell in set(drift["cells"]):
        return float(drift["legacy_replay_rtol"])
    return canonical


def main() -> None:
    args = parse_args()
    if args.exact and args.profile != "canonical":
        raise SystemExit("--exact is only valid with --profile canonical")

    manifest = json.loads(MANIFEST.read_text())
    reference = json.loads(GOLDEN.read_text())
    candidate = json.loads(args.candidate.read_text())

    if args.backend not in reference:
        raise SystemExit(
            f"reference backend {args.backend!r} is not recorded; "
            f"available={sorted(reference)}"
        )
    if args.backend not in candidate:
        raise SystemExit(
            f"candidate backend {args.backend!r} is missing; "
            f"available={sorted(candidate)}"
        )

    cells = manifest["cells"]
    selected = [
        name
        for name in sorted(cells)
        if args.owner == "all" or cells[name]["owner"] == args.owner
    ]
    expected_bank = reference[args.backend]
    got_bank = candidate[args.backend]
    atol = float(manifest["comparison"]["atol"])

    failures: list[str] = []
    max_rel = 0.0
    max_allowed = 0.0

    for name in selected:
        if name not in got_bank:
            failures.append(f"{name}: missing")
            continue
        got_values = got_bank[name]
        exp_values = expected_bank[name]
        if not isinstance(got_values, list) or len(got_values) != 3:
            failures.append(f"{name}: expected three values, got {got_values!r}")
            continue

        rtol = tolerance_for(manifest, args.backend, name, args.profile)
        max_allowed = max(max_allowed, rtol)
        for i, (got, exp) in enumerate(zip(got_values, exp_values)):
            if not isinstance(got, (int, float)) or not math.isfinite(got):
                failures.append(f"{name}[{i}]: non-finite/non-numeric {got!r}")
                continue
            got = float(got)
            exp = float(exp)
            err = relerr(got, exp)
            max_rel = max(max_rel, err)
            if args.exact:
                ok = got == exp
            else:
                ok = abs(got - exp) <= atol + rtol * abs(exp)
            if not ok:
                failures.append(
                    f"{name}[{i}]: got={got:.17g} expected={exp:.17g} "
                    f"relerr={err:.3e} allowed_rtol={rtol:.3e}"
                )

    if failures:
        print(
            f"FAIL backend={args.backend} owner={args.owner} profile={args.profile} "
            f"cells={len(selected)} max_relerr={max_rel:.3e}"
        )
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    if args.exact:
        mode = "exact"
    else:
        mode = f"profile={args.profile}, max_rtol={max_allowed:g}, atol={atol:g}"
    print(
        f"PASS backend={args.backend} owner={args.owner} cells={len(selected)} "
        f"mode={mode} max_relerr={max_rel:.3e}"
    )


if __name__ == "__main__":
    main()
