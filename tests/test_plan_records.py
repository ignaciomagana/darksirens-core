"""What ``ds.model`` records on its plan, and which option combinations it refuses.

A run fingerprint built the documented way
(:func:`~darksirens.inference.run_fingerprint.parameter_plan_semantic`) can
only tell two analyses apart by what their plans record, so every option that
changes the likelihood value must change the plan's semantic block, and an
option that changes nothing must not.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

import darksirens as ds
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import GalaxyCatalog
from darksirens.inference.run_fingerprint import (
    fingerprint_from_semantic,
    parameter_plan_semantic,
)

COSMOLOGY = ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075, w0=-1.0, wa=0.0)
FIXED_POP = ds.Population("powerlaw+peak", fixed=True)


def _catalog(path="catalog-fixture.h5"):
    """One galaxy pair in each of the 12 rows of an nside-1 sky."""
    npix = 12
    z0 = np.linspace(0.075, 0.115, npix)
    zgals = np.column_stack([z0, z0 + 0.035])
    return CatalogStore(
        path=path,
        nside=1,
        z_depth=0.30,
        catalog=GalaxyCatalog(
            apix=np.pi / 3.0,
            zgals=zgals,
            dzgals=np.full_like(zgals, 0.008),
            wgals=np.ones_like(zgals),
            ngals=np.full(npix, 2, dtype=np.int32),
            unique_pixels=None,
        ),
    )


def _digest(analysis):
    return fingerprint_from_semantic(
        {"parameters": parameter_plan_semantic(analysis.parameters)}
    )["digest"]


# ---------------------------------------------------------------------------
# Value-changing options that the plan left out


def test_the_volume_empty_policy_is_recorded_and_zero_is_not():
    # The review's scenario: log L 16.11622 (zero) against 16.11465 (volume)
    # at the same theta, with equal plans and equal digests.
    kwargs = dict(cosmology=COSMOLOGY, population=FIXED_POP, catalog=_catalog(),
                  completeness="complete")
    default = ds.model(**kwargs)
    zero = ds.model(**kwargs, empty_policy="zero")
    volume = ds.model(**kwargs, empty_policy="volume")

    assert default.parameters == zero.parameters
    assert zero.parameters.catalog_model == ""
    assert "catalog_model" not in parameter_plan_semantic(zero.parameters)
    assert volume.parameters != zero.parameters
    assert volume.parameters.catalog_model_settings == {
        "completeness": "complete", "empty_policy": "volume",
    }
    assert _digest(volume) != _digest(zero)


def test_angular_models_with_the_same_coordinates_have_different_plans():
    a = ds.model(cosmology=COSMOLOGY, population=FIXED_POP, angular="sphere_gp_z")
    b = ds.model(cosmology=COSMOLOGY, population=FIXED_POP, angular="overdensity_gp")
    # The same 195 coordinates, bounds and prior kinds ...
    assert a.parameters.labels == b.parameters.labels
    assert a.parameters.lower == b.parameters.lower
    assert a.parameters.prior_kinds == b.parameters.prior_kinds
    # ... but not the same plan or fingerprint.
    assert a.parameters != b.parameters
    assert parameter_plan_semantic(a.parameters)["angular_model"] == "sphere_gp_z"
    assert parameter_plan_semantic(b.parameters)["angular_model"] == "overdensity_gp"
    assert _digest(a) != _digest(b)


@pytest.mark.parametrize("angular", ["isotropic", "dipole", "multipole"])
def test_angular_models_told_apart_by_their_coordinates_record_no_name(angular):
    analysis = ds.model(cosmology=COSMOLOGY, population=FIXED_POP, angular=angular)
    assert analysis.parameters.angular_model == angular
    assert "angular_model" not in parameter_plan_semantic(analysis.parameters)


def test_the_recorded_set_is_every_angular_model_whose_coordinates_are_shared():
    from darksirens.inference.run_fingerprint import ANGULAR_MODELS_SHARING_COORDINATES
    from darksirens.population.angular import (
        ANGULAR_MODEL_NAMES,
        angular_model_prior_parser,
        get_angular_model,
    )

    by_coordinates = collections.defaultdict(set)
    for name in ANGULAR_MODEL_NAMES:
        lower, upper, labels, kinds, _ = angular_model_prior_parser(name)
        groups = getattr(get_angular_model(name), "constraint_groups", None) or ()
        key = (
            tuple(str(label) for label in labels),
            tuple(float(x) for x in lower),
            tuple(float(x) for x in upper),
            tuple(tuple(kind) for kind in kinds),
            repr(tuple(groups)),
        )
        by_coordinates[key].add(name)
    shared = set().union(*(names for names in by_coordinates.values() if len(names) > 1))
    assert shared == set(ANGULAR_MODELS_SHARING_COORDINATES)
