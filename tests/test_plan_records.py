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


# ---------------------------------------------------------------------------
# The pooled count ratio's window


@pytest.mark.parametrize(
    "window, expected",
    [
        (np.float32(0.02), float(np.float32(0.02))),
        (np.float64(0.02), 0.02),
        (np.int64(1), 1.0),
        (0.02, 0.02),
    ],
)
def test_count_ratio_window_accepts_any_real_number(window, expected):
    # The review's scenario: a window read from a config array or an HDF5
    # attribute (np.float32, np.int64) was refused as "not a number".
    analysis = ds.model(cosmology=COSMOLOGY, population=FIXED_POP, catalog=_catalog(),
                        count_ratio_window=window)
    assert analysis.redshift.count_ratio == "pooled"
    assert analysis.redshift.count_ratio_window == expected
    assert type(analysis.redshift.count_ratio_window) is float


@pytest.mark.parametrize("window", [True, np.bool_(True), "0.02", 1j])
def test_count_ratio_window_refuses_what_is_not_a_number(window):
    with pytest.raises(TypeError, match="count_ratio_window must be a number"):
        ds.model(cosmology=COSMOLOGY, population=FIXED_POP, catalog=_catalog(),
                 count_ratio_window=window)


def test_count_ratio_window_below_the_histogram_bin_is_refused():
    # The review's scenario: window=1e-5 passed every check, and about half
    # the redshift nodes fell between bin centres and saw no galaxy (R ~ 0).
    from darksirens.catalog.completeness import POOLED_BIN_WIDTH

    for window in (1e-5, 0.5 * POOLED_BIN_WIDTH):
        with pytest.raises(ValueError, match="at least the pooled histogram's bin width"):
            ds.model(cosmology=COSMOLOGY, population=FIXED_POP, catalog=_catalog(),
                     count_ratio_window=window)
    at_the_bin = ds.model(cosmology=COSMOLOGY, population=FIXED_POP, catalog=_catalog(),
                          count_ratio_window=POOLED_BIN_WIDTH)
    assert at_the_bin.redshift.count_ratio_window == POOLED_BIN_WIDTH


# ---------------------------------------------------------------------------
# Field-weighted options of complete catalogs


def test_n0_units_is_refused_for_an_all_complete_field_mixture():
    # The review's scenario: accepted and fingerprinted although no catalog
    # has a log10n0, while the conditional complete catalog refuses it.
    kwargs = dict(cosmology=COSMOLOGY, population=FIXED_POP, catalog_sky_weighting="field")
    for catalog in ([_catalog("a.h5"), _catalog("b.h5")], _catalog()):
        with pytest.raises(ValueError, match="n0_units applies only to an incomplete-catalog"):
            ds.model(**kwargs, catalog=catalog, completeness="complete", n0_units="h_scaled")
    with pytest.raises(ValueError, match="n0_units applies only to an incomplete-catalog"):
        ds.model(cosmology=COSMOLOGY, population=FIXED_POP, catalog=_catalog(),
                 completeness="complete", n0_units="h_scaled")
    mixed = ds.model(**kwargs, catalog=[_catalog("a.h5"), _catalog("b.h5")],
                     completeness=["incomplete", "complete"], count_ratio="row",
                     n0_units="h_scaled")
    assert mixed.parameters.n0_units == "h_scaled"
    assert "log10n0" in mixed.parameters.labels


def test_the_kernel_pin_applies_to_a_field_weighted_complete_catalog_only():
    fixed = dict(cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075, w0=-1.0, wa=0.0),
                 population=FIXED_POP, catalog=_catalog(), completeness="complete",
                 fixed_survey={"delta": 0.0, "sigma_kde": 0.01})
    conditional = ds.model(**fixed)
    field = ds.model(**fixed, catalog_sky_weighting="field")
    assert conditional.parameters.kernel_pin_active is False
    assert field.parameters.kernel_pin_active is True
    assert field.parameters.catalog_model_settings["kernel_pin_active"] == [True]
    off = ds.model(**fixed, catalog_sky_weighting="field", kernel_pin="off")
    assert off.parameters.kernel_pin_active is False


# ---------------------------------------------------------------------------
# Partially fixed joint prior pairs

_GWTC5 = "gwtc5_fiducial_bpl2peaks"
_L0 = r"$\lambda_0$"
_M1L, _M2L = r"$m_{1,{\rm low}}$", r"$m_{2,{\rm low}}$"


def _partially_fixed(fixed):
    with pytest.warns(RuntimeWarning) as record:
        analysis = ds.model(cosmology=COSMOLOGY, population=ds.Population(_GWTC5, fixed=fixed))
    texts = [str(w.message) for w in record if "joint prior constraint" in str(w.message)]
    assert len(texts) == 1
    return analysis, texts[0]


def test_fixing_the_lower_member_of_a_conditional_pair_says_the_prior_changes():
    # The review's scenario: m2_low = 5 leaves m1_low ~ U(5, 10) (prior mean
    # 7.5), while the declared prior conditioned on m2_low = 5 falls as
    # 1 / (m1_low - 3) (prior mean 6.99). The warning said only logZ changed.
    analysis, text = _partially_fixed({_M2L: 5.0})
    assert _M1L in analysis.parameters.labels
    assert "has fixed member(s)" in text
    assert "not the declared joint prior conditioned on" in text
    assert "1 / ($m_{1,{\\rm low}}$ - 3.0)" in text
    assert "the posterior differs from that conditional, not only logZ" in text
    assert "ds.infer adds to log_prior_volume_fraction" in text


@pytest.mark.parametrize("fixed", [{_M1L: 5.0}, {_L0: 0.6}])
def test_fixing_a_member_whose_partner_keeps_its_conditional_prior_says_only_logZ(fixed):
    _, text = _partially_fixed(fixed)
    assert "not the declared joint prior" not in text
    assert "logZ carries the log prior-fraction offset, which ds.infer adds" in text
