from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

import darksirens as ds


def test_default_cosmology_samples_only_h0():
    cosmo = ds.Cosmology()
    assert cosmo.free_parameters == (("H0", 20.0, 140.0),)
    assert cosmo.fixed_parameters == {"Om0": 0.3075, "w0": -1.0, "wa": 0.0}


def test_cosmology_scalar_or_bounds_contract_and_order():
    cosmo = ds.Cosmology(
        H0=67.74,
        Om0=(0.2, 0.4),
        w0=(-2.0, -0.3),
        wa=0,
    )
    assert cosmo.free_parameters == (
        ("Om0", 0.2, 0.4),
        ("w0", -2.0, -0.3),
    )
    assert cosmo.fixed_parameters == {"H0": 67.74, "wa": 0.0}


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"H0": (70.0, 70.0)}, ValueError),
        ({"Om0": (0.5, 0.1)}, ValueError),
        ({"w0": (float("nan"), -0.2)}, ValueError),
        ({"wa": True}, TypeError),
        ({"H0": (20.0,)}, ValueError),
        ({"H0": "wide"}, TypeError),
    ],
)
def test_cosmology_rejects_invalid_parameter_declarations(kwargs, exc):
    with pytest.raises(exc):
        ds.Cosmology(**kwargs)


def test_cosmology_rejects_prior_outside_the_tabulated_distance_grid():
    # Outside the table r_of_z is NaN and the likelihood is -inf, so an
    # accepted out-of-grid bound truncates the prior with no diagnostic.
    with pytest.raises(
        ValueError,
        match=r"Om0 prior .*\[0\.15749999999999997, 0\.45749999999999996\]",
    ):
        ds.Cosmology(H0=(20.0, 140.0), Om0=(0.10, 0.50))

    with pytest.raises(ValueError, match=r"wa prior .*\[-2\.5, 2\.5\]"):
        ds.Cosmology(wa=(-3.0, 3.0))


def test_cosmology_rejects_fixed_value_outside_the_tabulated_distance_grid():
    with pytest.raises(
        ValueError,
        match=r"Om0 fixed at 0\.55 .*\[0\.15749999999999997, 0\.45749999999999996\]",
    ):
        ds.Cosmology(Om0=0.55)


def test_cosmology_accepts_grid_endpoints_and_any_positive_h0():
    # The support is closed, and H0 is not an interpolation axis.
    assert ds.Cosmology(w0=(-2.25, 0.25)).free_parameters == (
        ("H0", 20.0, 140.0),
        ("w0", -2.25, 0.25),
    )
    assert ds.Cosmology(H0=(1.0, 1.0e6)).free_parameters == (("H0", 1.0, 1.0e6),)
    assert ds.Cosmology(H0=1.0e-3).fixed_parameters["H0"] == 1.0e-3


@pytest.mark.parametrize(
    "h0, match",
    [
        ((-20.0, 140.0), r"H0 prior \[-20\.0, 140\.0\] reaches H0 <= 0"),
        ((0.0, 140.0), r"H0 prior \[0\.0, 140\.0\] reaches H0 <= 0"),
        (0.0, r"H0 fixed at 0\.0 must be > 0"),
        (-67.0, r"H0 fixed at -67\.0 must be > 0"),
    ],
)
def test_cosmology_rejects_a_non_positive_h0(h0, match):
    """Distances scale as 1/H0: infinite at 0 and negative below, so the
    likelihood is -inf there and a prior reaching it was silently truncated
    to H0 > 0 while the plan recorded the declared bounds."""
    with pytest.raises(ValueError, match=match):
        ds.Cosmology(H0=h0)


@pytest.mark.parametrize(
    "value", [np.float32(70.0), np.int64(70), np.float64(70.0), np.array(70.0), np.array(70)]
)
def test_cosmology_takes_numpy_real_scalars(value):
    assert ds.Cosmology(H0=value).fixed_parameters["H0"] == 70.0
    assert type(ds.Cosmology(H0=value).H0) is float
    assert ds.Cosmology(H0=(np.float32(20), value)).free_parameters == (("H0", 20.0, 70.0),)


@pytest.mark.parametrize(
    "value", [("20", "140"), (20.0, "140"), np.True_, (np.True_, 140.0), np.array([70.0]), "70"]
)
def test_cosmology_refuses_values_that_are_not_real_numbers(value):
    with pytest.raises(TypeError):
        ds.Cosmology(H0=value)


def _curated_warning():
    return pytest.warns(UserWarning, match=r"curated grammar composition BPL\+2G .*not the GWTC-5")


def test_population_sampled_and_fiducial_modes():
    with _curated_warning():
        sampled = ds.Population("brokenpowerlaw+2peaks")
    assert not sampled.is_fixed
    assert sampled.model_name == "brokenpowerlaw+2peaks"
    assert sampled.fiducial_set is None

    with _curated_warning():
        legacy = ds.Population("brokenpowerlaw+2peaks", fixed=True)
    assert legacy.is_fixed
    assert legacy.model_name == "brokenpowerlaw+2peaks"
    assert legacy.fiducial_set == "legacy"

    corrected = ds.Population("2powerlaws+peak", fixed="in_prior_v2")
    assert corrected.is_fixed
    assert corrected.model_name == "2powerlaws+peak"
    assert corrected.fiducial_set == "in_prior_v2"


def test_architecture_gwtc5_public_spelling_resolves_existing_preset():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pop = ds.Population("brokenpowerlaw+2peaks", fixed="gwtc5")
    assert pop.is_fixed
    assert pop.model_name == "gwtc5_fiducial_bpl2peaks"
    assert pop.fiducial_set == "legacy"

    from darksirens.population import get_fixed_population_params

    theta = get_fixed_population_params(pop.model_name, fiducials=pop.fiducial_set)
    assert theta.shape == (17,)


def test_gwtc5_preset_refuses_unrelated_model():
    with pytest.raises(ValueError, match="only compatible.*'gwtc5_fiducial_bpl2peaks'"):
        ds.Population("powerlaw+peak", fixed="gwtc5")


@pytest.mark.parametrize("fixed", [None, True, "in_prior_v2", {"$\\gamma$": 2.5439}])
def test_the_curated_bpl2g_name_warns_that_it_is_not_the_gwtc5_model(fixed):
    """'brokenpowerlaw+2peaks' is the GWTC-5 model only with fixed='gwtc5'.
    Sampled, at a preset or partly fixed (gamma is a label of both models) it
    is the curated grammar composition, a different 17-parameter model, which
    used to be selected without a word."""
    with _curated_warning():
        pop = ds.Population("brokenpowerlaw+2peaks", fixed=fixed)
    assert pop.model_name == "brokenpowerlaw+2peaks"
    # The GWTC-5 model is sampled or partly fixed under its own name, quietly.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gwtc5 = ds.Population("gwtc5_fiducial_bpl2peaks", fixed=None if fixed is True else fixed)
    labels = ds.model(population=gwtc5).parameters.population_labels
    assert "$\\alpha_1$" in labels or any("alpha_1" in label for label in labels)
    assert not any("v_1" in label for label in labels)


def test_population_options_are_declarative():
    with _curated_warning():
        pop = ds.Population(
            "brokenpowerlaw+2peaks",
            shared_beta=False,
            shared_spin=False,
            shared_gamma=False,
        )
    assert not pop.shared_beta
    assert not pop.shared_spin
    assert not pop.shared_gamma


@pytest.mark.parametrize("fixed", ["nope", 3.0, object()])
def test_population_rejects_unknown_fixed_mode(fixed):
    with pytest.raises((TypeError, ValueError)):
        ds.Population("brokenpowerlaw+2peaks", fixed=fixed)


def test_package_root_specs_are_dependency_light():
    assert ds.Cosmology.__module__ == "darksirens._specs"
    assert ds.Population.__module__ == "darksirens._specs"
    # Checked in a fresh interpreter: other test modules legitimately import
    # the real sampler backends into this process, so an in-process check of
    # sys.modules would depend on collection order.
    code = (
        "import sys, darksirens as ds; ds.Cosmology(Om0=(0.2, 0.4));"
        " ds.Population('brokenpowerlaw+2peaks', fixed='gwtc5');"
        " heavy = {'jax', 'numpyro', 'dynesty', 'tinyns', 'h5py'} & set(sys.modules);"
        " assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
