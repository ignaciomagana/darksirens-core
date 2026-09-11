"""Minimal ordinary catalog dark-siren analysis using only the public API."""

from __future__ import annotations

import argparse

import darksirens as ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", help="standardized PE HDF5 store")
    parser.add_argument("injections", help="standardized selection HDF5 store")
    parser.add_argument("catalog", help="standardized pixelated catalog HDF5 store")
    parser.add_argument("--sampler", default="tinyns", choices=("tinyns", "dynesty", "numpyro"))
    args = parser.parse_args()

    events = ds.load_events(args.events)
    injections = ds.load_injections(args.injections)
    catalog = ds.load_catalog(args.catalog)

    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("brokenpowerlaw+2peaks", fixed="gwtc5"),
        catalog=catalog,
    )

    result = ds.infer(
        analysis,
        events=events,
        injections=injections,
        sampler=args.sampler,
    )
    print(f"logZ = {result['logZ']}")


if __name__ == "__main__":
    main()
