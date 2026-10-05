"""Minimal ordinary catalog dark-siren analysis using only the public API."""

from __future__ import annotations

import argparse

import darksirens as ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", help="standardized PE HDF5 store")
    parser.add_argument("injections", help="standardized selection HDF5 store")
    parser.add_argument("catalog", help="standardized pixelated catalog HDF5 store")
    parser.add_argument("--sampler", default="dynesty", choices=("dynesty", "tinyns", "numpyro"))
    parser.add_argument("--out", help="write the result to this HDF5 file")
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
    # Where the likelihood guards cut the prior, from the draws made before sampling.
    report = result["guard_report"]
    print(f"guarded prior draws: {report['guarded']}/{report['n_draws']}")
    for guard, entry in report["fired"].items():
        print(f"  {guard}: {entry['ranges']}")
    if args.out:
        ds.save_result(args.out, result, labels=analysis.parameters.labels)
    print(f"logZ = {result['logZ']}")


if __name__ == "__main__":
    main()
