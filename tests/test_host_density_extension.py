from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.analysis import IncompleteCatalogRedshift, ParameterPlan
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import log_comoving_volume_prior
from darksirens.gw import make_gw_event
from darksirens.inference.target import combine_parameter_plans
from darksirens.likelihood.hierarchical import spectral_siren_log_likelihood
from darksirens.likelihood.host_density import (
    HostDensityLikelihoodDiagnostics,
    host_density_log_likelihood,
    make_host_density_target,
)


UNIFORM = ("uniform", None, None)
MAX_VAR = 1.0e6


def _plan(labels=(), *, constraints=()):
    n = len(labels)
    return ParameterPlan(
        labels=tuple(labels),
        lower=tuple(-1.0 for _ in range(n)),
        upper=tuple(1.0 for _ in range(n)),
        prior_kinds=tuple(UNIFORM for _ in range(n)),
        joint_constraints=tuple(constraints),
    )


class VolumeRedshiftModel:
    def __init__(self, plan=None):
        self.plan = _plan() if plan is None else plan
        self.density_states = []
        self.aux_calls = 0
        self.aux_state = None

    def parameter_spec(self):
        return self.plan

    def log_density(self, z, pixel, cosmology, parameters, state):
        self.density_states.append(state)
        return log_comoving_volume_prior(z, cosmology)

    def log_auxiliary_likelihood(self, parameters, state):
        self.aux_calls += 1
        self.aux_state = state
        if len(self.plan.labels) == 0:
            return 0.0
        return parameters[0]


def _runtime_events():
    nsamp = 4
    nsel = 256

    pe_m1 = np.array([34.0, 36.0, 38.0, 40.0])
    pe_ra = np.array([0.2, 1.3, 2.4, 4.7])
    pe_dec = np.array([-0.35, -0.10, 0.20, 0.45])
    pe_cos = np.cos(pe_dec)
    pe = make_gw_event(
        m1det=pe_m1,
        m2det=0.8 * pe_m1,
        dL=np.array([430.0, 465.0, 500.0, 535.0]),
        chieff=np.array([-0.02, 0.00, 0.01, 0.03]),
        prior_wt=np.ones(nsamp),
        pixels=np.array([0, 1, 2, 3], dtype=np.int32),
        nx=pe_cos * np.cos(pe_ra),
        ny=pe_cos * np.sin(pe_ra),
        nz=np.sin(pe_dec),
    )

    sel_m1 = np.linspace(33.0, 41.0, nsel)
    sel_ra = np.linspace(0.05, 2.0 * np.pi - 0.05, nsel)
    sel_dec = np.linspace(-0.6, 0.6, nsel)
    sel_cos = np.cos(sel_dec)
    selection = make_gw_event(
        m1det=sel_m1,
        m2det=0.8 * sel_m1,
        dL=np.linspace(410.0, 555.0, nsel),
        chieff=np.linspace(-0.04, 0.04, nsel),
        prior_wt=np.ones(nsel),
        pixels=np.mod(np.arange(nsel), 12).astype(np.int32),
        nx=sel_cos * np.cos(sel_ra),
        ny=sel_cos * np.sin(sel_ra),
        nz=np.sin(sel_dec),
    )
    return pe, selection, nsamp, nsel


def _base_analysis(*, angular="isotropic"):
    return model(
        cosmology=Cosmology(H0=67.74, Om0=0.3075),
        population=Population("powerlaw+peak", fixed=True),
        angular=angular,
    )


def _direct_spectral(base, pe, selection, nsamp, nsel, *, angular_params=None):
    return spectral_siren_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        np.asarray(base.parameters.fixed_population),
        pe,
        selection,
        1,
        nsamp,
        float(nsel),
        pop_model=base.population.model_name,
        shared_beta=base.population.shared_beta,
        shared_spin=base.population.shared_spin,
        shared_gamma=base.population.shared_gamma,
        angular_model=base.angular_model,
        angular_params=angular_params,
        selection_neff_soft_guard=True,
        max_likelihood_variance=MAX_VAR,
    )


def test_volume_model_is_exact_spectral_wrapper_and_states_are_separate():
    pe, selection, nsamp, nsel = _runtime_events()
    base = _base_analysis()
    redshift = VolumeRedshiftModel()
    pe_state = object()
    selection_state = object()
    auxiliary_state = object()

    expected = _direct_spectral(base, pe, selection, nsamp, nsel)
    actual = host_density_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        np.asarray(base.parameters.fixed_population),
        np.empty(0),
        pe,
        pe_state,
        selection,
        selection_state,
        1,
        nsamp,
        float(nsel),
        redshift_model=redshift,
        pop_model=base.population.model_name,
        auxiliary_state=auxiliary_state,
        shared_beta=base.population.shared_beta,
        shared_spin=base.population.shared_spin,
        shared_gamma=base.population.shared_gamma,
        selection_neff_soft_guard=True,
        max_likelihood_variance=MAX_VAR,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.isfinite(float(actual))
    assert any(state is pe_state for state in redshift.density_states)
    assert any(state is selection_state for state in redshift.density_states)
    assert redshift.aux_calls == 1
    assert redshift.aux_state is auxiliary_state


def test_auxiliary_term_is_added_exactly_once_and_reported():
    pe, selection, nsamp, nsel = _runtime_events()
    base = _base_analysis()
    redshift = VolumeRedshiftModel(_plan(("eta",)))
    eta = 0.25

    expected = _direct_spectral(base, pe, selection, nsamp, nsel)
    diagnostics = host_density_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        np.asarray(base.parameters.fixed_population),
        np.array([eta]),
        pe,
        object(),
        selection,
        object(),
        1,
        nsamp,
        float(nsel),
        redshift_model=redshift,
        pop_model=base.population.model_name,
        auxiliary_state="counts",
        selection_neff_soft_guard=True,
        max_likelihood_variance=MAX_VAR,
        return_diagnostics=True,
    )

    assert isinstance(diagnostics, HostDensityLikelihoodDiagnostics)
    np.testing.assert_array_equal(
        np.asarray(diagnostics.log_likelihood),
        np.asarray(expected + eta),
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.log_auxiliary_likelihood),
        np.asarray(eta),
    )
    assert redshift.aux_calls == 1
    assert redshift.aux_state == "counts"


def test_combine_parameter_plans_offsets_constraints_and_rejects_duplicates():
    first = _plan(("a", "b"), constraints=(("ordered_le", (0, 1)),))
    second = _plan(
        ("c", "d", "e"),
        constraints=(("ball3", (0, 1, 2)),),
    )
    combined = combine_parameter_plans(first, second)
    assert combined.labels == ("a", "b", "c", "d", "e")
    assert combined.joint_constraints == (
        ("ordered_le", (0, 1)),
        ("ball3", (2, 3, 4)),
    )
    assert combined.n_cosmology == 0
    assert combined.n_population == 0
    assert combined.n_catalog == 0
    assert combined.n_angular == 0

    with pytest.raises(ValueError, match="combined ParameterPlan labels must be unique"):
        combine_parameter_plans(first, _plan(("b",)))


def test_target_builder_preserves_base_angular_model_and_extension_order():
    pe, selection, nsamp, nsel = _runtime_events()
    base = _base_analysis(angular="dipole")
    redshift = VolumeRedshiftModel(_plan(("eta",)))

    target = make_host_density_target(
        base,
        redshift_model=redshift,
        gw_pe=pe,
        gw_selection=selection,
        pe_state="pe",
        selection_state="selection",
        auxiliary_state="aux",
        n_events=1,
        nsamp=nsamp,
        n_draw=float(nsel),
        selection_neff_soft_guard=True,
        max_likelihood_variance=MAX_VAR,
    )

    assert target.parameters.labels == base.parameters.labels + ("eta",)
    angular = np.zeros(base.parameters.n_angular)
    angular[0] = 0.2
    eta = 0.1
    theta = np.concatenate([angular, np.array([eta])])

    expected = host_density_log_likelihood(
        CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
        np.asarray(base.parameters.fixed_population),
        np.array([eta]),
        pe,
        "pe",
        selection,
        "selection",
        1,
        nsamp,
        float(nsel),
        redshift_model=redshift,
        pop_model=base.population.model_name,
        auxiliary_state="aux",
        shared_beta=base.population.shared_beta,
        shared_spin=base.population.shared_spin,
        shared_gamma=base.population.shared_gamma,
        angular_model="dipole",
        angular_params=angular,
        selection_neff_soft_guard=True,
        max_likelihood_variance=MAX_VAR,
    )
    actual = target.log_likelihood(theta)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_target_builder_rejects_catalog_bearing_base_analysis():
    pe, selection, nsamp, nsel = _runtime_events()
    base = _base_analysis()
    bad = replace(base, redshift=IncompleteCatalogRedshift(object()))

    with pytest.raises(ValueError, match="catalog-free SpectralRedshift"):
        make_host_density_target(
            bad,
            redshift_model=VolumeRedshiftModel(),
            gw_pe=pe,
            gw_selection=selection,
            pe_state=None,
            selection_state=None,
            n_events=1,
            nsamp=nsamp,
            n_draw=float(nsel),
        )


def test_redshift_model_contract_requires_parameter_plan_and_scalar_auxiliary():
    pe, selection, nsamp, nsel = _runtime_events()
    base = _base_analysis()

    class BadSpec(VolumeRedshiftModel):
        def parameter_spec(self):
            return object()

    with pytest.raises(TypeError, match="must return ParameterPlan"):
        make_host_density_target(
            base,
            redshift_model=BadSpec(),
            gw_pe=pe,
            gw_selection=selection,
            pe_state=None,
            selection_state=None,
            n_events=1,
            nsamp=nsamp,
            n_draw=float(nsel),
        )

    class VectorAux(VolumeRedshiftModel):
        def log_auxiliary_likelihood(self, parameters, state):
            return np.array([0.0, 0.0])

    with pytest.raises(ValueError, match="must return a scalar"):
        host_density_log_likelihood(
            CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
            np.asarray(base.parameters.fixed_population),
            np.empty(0),
            pe,
            None,
            selection,
            None,
            1,
            nsamp,
            float(nsel),
            redshift_model=VectorAux(),
            pop_model=base.population.model_name,
            selection_neff_soft_guard=True,
            max_likelihood_variance=MAX_VAR,
        )
