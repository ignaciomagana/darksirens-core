#!/usr/bin/env python3
"""Exact parity probe for the core host-side HEALPix RING mapper.

The frozen ordinary inference path used ``healpy.ang2pix(nside, pi/2-dec, ra)``.
This probe compares the dependency-free reconstruction against the validated
campaign environment's healpy 1.17.3 over randomized skies and boundary cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np

from darksirens.catalog.geometry import ang2pix_ring

REFERENCE_HEALPY = "1.17.3"
NSIDES = (1, 2, 3, 4, 8, 16, 64, 128, 1024)


def _boundary_coordinates():
    half_pi = 0.5 * np.pi
    two_pi = 2.0 * np.pi
    transition = np.arcsin(2.0 / 3.0)

    ra_base = np.array(
        [
            0.0,
            np.nextafter(0.0, 1.0),
            0.5 * np.pi,
            np.nextafter(0.5 * np.pi, 0.0),
            np.nextafter(0.5 * np.pi, np.inf),
            np.pi,
            np.nextafter(np.pi, 0.0),
            np.nextafter(np.pi, np.inf),
            1.5 * np.pi,
            np.nextafter(1.5 * np.pi, 0.0),
            np.nextafter(1.5 * np.pi, np.inf),
            np.nextafter(two_pi, 0.0),
            two_pi,
        ],
        dtype=np.float64,
    )
    dec_base = np.array(
        [
            -half_pi,
            np.nextafter(-half_pi, 0.0),
            -transition,
            np.nextafter(-transition, -half_pi),
            np.nextafter(-transition, 0.0),
            0.0,
            transition,
            np.nextafter(transition, 0.0),
            np.nextafter(transition, half_pi),
            np.nextafter(half_pi, 0.0),
            half_pi,
        ],
        dtype=np.float64,
    )
    ra, dec = np.meshgrid(ra_base, dec_base, indexing="ij")
    return ra.ravel(), dec.ravel()


def _random_coordinates(seed=260911, n=20000):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float64)
    # Uniform in sin(dec), so both polar caps and equatorial belt are sampled.
    dec = np.arcsin(rng.uniform(-1.0, 1.0, n)).astype(np.float64)
    return ra, dec


def _compare(nside, ra, dec, label):
    theta = 0.5 * np.pi - dec
    expected = np.asarray(hp.ang2pix(nside, theta, ra, nest=False), dtype=np.int64)
    actual = np.asarray(ang2pix_ring(nside, ra, dec), dtype=np.int64)
    if not np.array_equal(actual, expected):
        bad = np.flatnonzero(actual != expected)
        i = int(bad[0])
        raise AssertionError(
            f"HEALPix RING parity failure ({label}, nside={nside}): "
            f"{bad.size}/{actual.size} mismatches; first index={i}, "
            f"ra={ra[i]:.17g}, dec={dec[i]:.17g}, "
            f"candidate={int(actual[i])}, healpy={int(expected[i])}"
        )
    return int(actual.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if hp.__version__ != REFERENCE_HEALPY:
        raise RuntimeError(
            f"reference probe requires healpy {REFERENCE_HEALPY}, got {hp.__version__}"
        )

    random_ra, random_dec = _random_coordinates()
    boundary_ra, boundary_dec = _boundary_coordinates()
    counts = {}
    for nside in NSIDES:
        n_random = _compare(nside, random_ra, random_dec, "random")
        n_boundary = _compare(nside, boundary_ra, boundary_dec, "boundary")
        counts[str(nside)] = {"random": n_random, "boundary": n_boundary}

    # Candidate-only periodicity extension. Legacy stores RA in the canonical
    # interval, but accepting wrapped values is useful and must not change the
    # canonical result.
    base = ang2pix_ring(64, random_ra[:1024], random_dec[:1024])
    np.testing.assert_array_equal(
        base, ang2pix_ring(64, random_ra[:1024] + 2.0 * np.pi, random_dec[:1024])
    )
    np.testing.assert_array_equal(
        base, ang2pix_ring(64, random_ra[:1024] - 4.0 * np.pi, random_dec[:1024])
    )

    payload = {
        "reference_healpy": hp.__version__,
        "nsides": list(NSIDES),
        "counts": counts,
        "status": "exact",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
