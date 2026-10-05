"""Evaluate the likelihood ds.infer samples on an H0 grid, without a sampler."""

from __future__ import annotations

import argparse
import math

import numpy as np

import darksirens as ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", help="standardized PE HDF5 store")
    parser.add_argument("injections", help="standardized selection HDF5 store")
    parser.add_argument("--points", type=int, default=25, help="number of H0 grid points")
    args = parser.parse_args()

    events = ds.load_events(args.events)
    injections = ds.load_injections(args.injections)

    # A catalog-free (spectral-siren) analysis with only H0 sampled.
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("brokenpowerlaw+2peaks", fixed="gwtc5"),
    )
    log_likelihood = ds.log_likelihood(analysis, events=events, injections=injections)

    grid = np.linspace(20.0, 140.0, args.points)
    values = np.array([float(log_likelihood(np.array([h0]))) for h0 in grid])
    # -inf marks a point where a likelihood guard refused the Monte-Carlo estimate.
    for h0, value in zip(grid, values):
        print(f"H0 = {h0:6.2f}  logL = {value:.6g}")
    finite = values[np.isfinite(values)]
    best = grid[np.argmax(np.where(np.isfinite(values), values, -np.inf))]
    print(f"finite points: {finite.size}/{grid.size}")
    print(f"logL max = {finite.max() if finite.size else -math.inf} at H0 = {best:.2f}")


if __name__ == "__main__":
    main()
