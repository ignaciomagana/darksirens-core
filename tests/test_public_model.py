from __future__ import annotations

import numpy as np
import pytest

import darksirens as ds
from darksirens.analysis import (
    CompleteCatalogRedshift,
    IncompleteCatalogRedshift,
    SpectralRedshift,
)
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import GalaxyCatalog
from darksirens.inference.prior import make_prior_transform
from darksirens.population import pop_model_prior_parser


def _catalog(z_depth=0.3):
    cat = GalaxyCatalog(
        apix=np.pi / 3.0,
        zgals=np.array([[0.1, 100.0]]),
        dzgals=np.array([[0.001, 1.0]]),
        wgals=np.array([[1.0, 0.0]]),
        ngals=np.array([1], dtype=np.int32),
        unique_pixels=None,
    )
    return CatalogStore(path="memory.h5", nside=1, z_depth=z_depth, catalog=cat)


def _fixed_cosmology():
    return ds.Cosmology(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def test_spectral_fixed_population_has_no_catalog_coordinates():
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("brokenpowerlaw+2peaks", fixed="gwtc5"),
    )
    assert isinstance(analysis.redshift, SpectralRedshift)
    assert analysis.catalog is None
    assert analysis.parameters.labels == ("H0",)
    assert analysis.parameters.lower == (20.0,)
    assert analysis.parameters.upper == (140.0,)
    assert analysis.parameters.n_cosmology == 1
    assert analysis.parameters.n_population == 0
    assert analysis.parameters.n_catalog == 0
    assert len(analysis.parameters.fixed_population) == 17
    assert analysis.parameters.joint_constraints == ()


def test_incomplete_catalog_uses_frozen_no_lss_survey_block():
    store = _catalog()
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(40.0, 100.0), Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
        catalog=store,
    )
    assert isinstance(analysis.redshift, IncompleteCatalogRedshift)
    assert analysis.catalog is store
    assert analysis.parameters.labels == (
        "H0", "log10n0", "delta", "sigma_kde"
    )
    assert analysis.parameters.lower == (40.0, -4.0, -3.0, 0.0)
    assert analysis.parameters.upper == (100.0, -1.0, 3.0, 0.05)
    assert analysis.parameters.n_catalog == 3
    assert "b_miss" not in analysis.parameters.labels


def test_complete_catalog_uses_only_delta_and_kernel_width():
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
        catalog=_catalog(),
        completeness="complete",
    )
    assert isinstance(analysis.redshift, CompleteCatalogRedshift)
    assert analysis.parameters.labels == ("H0", "delta", "sigma_kde")
    assert analysis.parameters.lower == (20.0, -3.0, 0.0)
    assert analysis.parameters.upper == (140.0, 3.0, 0.05)
    assert analysis.parameters.n_catalog == 2


def test_sampled_parameter_order_is_cosmology_population_catalog():
    population = ds.Population("powerlaw+peak")
    analysis = ds.model(
        cosmology=ds.Cosmology(
            H0=(20.0, 140.0),
            Om0=0.3075,
            w0=(-1.5, -0.5),
            wa=0.0,
        ),
        population=population,
        catalog=_catalog(),
    )
    _, _, pop_labels, pop_kinds, _ = pop_model_prior_parser(
        population.model_name,
        shared_beta=True,
        shared_spin=True,
        shared_gamma=True,
    )
    plan = analysis.parameters
    assert plan.labels[:2] == ("H0", "w0")
    assert plan.labels[2:2 + len(pop_labels)] == tuple(pop_labels)
    assert plan.labels[-3:] == ("log10n0", "delta", "sigma_kde")
    assert plan.n_cosmology == 2
    assert plan.n_population == len(pop_labels)
    assert plan.prior_kinds[:2] == (("uniform", None, None),) * 2
    assert plan.prior_kinds[2:2 + len(pop_labels)] == tuple(tuple(k) for k in pop_kinds)


def test_sampled_gwtc5_plan_carries_normalized_joint_maps():
    analysis = ds.model(
        cosmology=_fixed_cosmology(),
        population=ds.Population("gwtc5_fiducial_bpl2peaks"),
    )
    plan = analysis.parameters
    assert {kind for kind, _ in plan.joint_constraints} == {
        "simplex", "conditional_upper"
    }
    transform = make_prior_transform(
        plan.lower,
        plan.upper,
        plan.prior_kinds,
        joint_constraints=plan.joint_constraints,
    )
    theta = np.asarray(
        transform(np.random.default_rng(7).uniform(size=(256, len(plan.labels))))
    )
    by_kind = dict(plan.joint_constraints)
    i, j = by_kind["simplex"]
    assert np.all(theta[:, i] + theta[:, j] <= 1.0 + 1e-12)
    i, j = by_kind["conditional_upper"]
    assert np.all(theta[:, i] <= theta[:, j] + 1e-12)


def test_fixed_cosmology_is_carried_outside_sampled_coordinates():
    analysis = ds.model(
        cosmology=_fixed_cosmology(),
        population=ds.Population("powerlaw+peak", fixed=True),
    )
    assert analysis.parameters.labels == ()
    assert analysis.parameters.fixed_cosmology == (
        ("H0", 67.74),
        ("Om0", 0.3075),
        ("w0", -1.0),
        ("wa", 0.0),
    )


def test_model_defaults_cosmology_but_requires_population():
    analysis = ds.model(population=ds.Population("powerlaw+peak", fixed=True))
    assert analysis.parameters.labels == ("H0",)
    with pytest.raises(TypeError):
        ds.model(population="powerlaw+peak")


def test_catalog_completeness_validation_is_compositional():
    with pytest.raises(ValueError, match="requires a catalog"):
        ds.model(
            population=ds.Population("powerlaw+peak", fixed=True),
            completeness="complete",
        )
    with pytest.raises(ValueError, match="completeness must"):
        ds.model(
            population=ds.Population("powerlaw+peak", fixed=True),
            catalog=_catalog(),
            completeness="mystery",
        )
    with pytest.raises(TypeError, match="CatalogStore"):
        ds.model(
            population=ds.Population("powerlaw+peak", fixed=True),
            catalog=object(),
        )


def test_analysis_does_not_reintroduce_universe_model_switchboard():
    analysis = ds.model(
        population=ds.Population("powerlaw+peak", fixed=True),
        catalog=_catalog(),
    )
    assert not hasattr(analysis, "universe_model")
    assert "universe_model" not in analysis.__dataclass_fields__
