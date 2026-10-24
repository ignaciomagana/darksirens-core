"""Options the population registry used to drop or refuse inconsistently.

* Sharing flags on a model with nothing to share: a bespoke registered model
  (GWTC-5/GWTC-3 fiducials, Golomb, every GP model) silently ignored
  ``shared_*=False`` while the recorded label said "Per-component"; it is now
  refused.  A one-component grammar mixture is the same model either way, so it
  is accepted and its label no longer claims per-component structure.
* ``'<one-component>@md'`` with ``shared_gamma=False`` sampled fine but its
  fixed preset was refused.
* GP constructors accepted any ``mean_mode`` and axis name (``gp1d_m1@md`` built
  a zero-mean model; an unknown axis vanished).
* The out-of-prior and pairing-grid messages named command-line flags core
  does not have.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import darksirens as ds
from darksirens.population import get_fixed_population_params, get_model, pop_model_prior_parser

_COSMOLOGY = ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075)
_OFF = dict(shared_beta=False, shared_spin=False, shared_gamma=False)


@pytest.mark.parametrize("name", ["gwtc5_fiducial_bpl2peaks", "gwtc3_fiducial_plpeak",
                                  "golomb_1g", "gp1d_m1"])
@pytest.mark.parametrize("flag", ["shared_beta", "shared_spin", "shared_gamma"])
def test_sharing_flags_are_refused_for_a_bespoke_model(name, flag):
    with pytest.raises(ValueError, match=f"{flag}=False cannot be applied.*{name}"):
        get_model(name, **{flag: False})
    with pytest.raises(ValueError, match="bespoke registered population model"):
        pop_model_prior_parser(name, **{flag: False})
    if name != "gp1d_m1":
        with pytest.raises(ValueError, match="bespoke registered population model"):
            get_fixed_population_params(name, **{flag: False})


def test_gwtc5_preset_with_per_component_flags_is_refused():
    """The finding's scenario: the run fit the shared model while the record
    read 'GWTC-5 Fiducial BPL+2G (Per-component beta, ...)'."""
    pop = ds.Population("brokenpowerlaw+2peaks", fixed="gwtc5", shared_beta=False)
    with pytest.raises(ValueError, match="gwtc5_fiducial_bpl2peaks"):
        ds.model(cosmology=_COSMOLOGY, population=pop)
    with pytest.raises(ValueError, match="gwtc5_fiducial_bpl2peaks"):
        pop.fiducial_values


@pytest.mark.parametrize("name", ["powerlaw", "peak", "brokenpowerlaw", "powerlaw@md"])
def test_one_component_label_does_not_claim_per_component_structure(name):
    shared = pop_model_prior_parser(name)
    off = pop_model_prior_parser(name, **(dict(_OFF, shared_gamma=True)
                                           if name.endswith("@md") else _OFF))
    assert off[2] == shared[2]                       # same labels: the same model
    assert off[4] == shared[4]
    assert "Per-component" not in off[4]
    analysis = ds.model(cosmology=_COSMOLOGY,
                        population=ds.Population(name, fixed=True, **_OFF))
    assert "Per-component" not in analysis.population_latex


def test_multi_component_label_still_names_per_component_structure():
    latex = pop_model_prior_parser("powerlaw+peak", **_OFF)[4]
    assert latex.endswith(
        r"(Per-component $\beta$, Per-component Spin, Per-component $\gamma$)")


@pytest.mark.parametrize("name", ["powerlaw@md", "peak@md", "brokenpowerlaw@md"])
def test_one_component_md_preset_accepts_unshared_gamma(name):
    """get_model built this model; get_fixed_population_params refused it."""
    model = get_model(name, shared_gamma=False)
    shared = np.asarray(get_fixed_population_params(name))
    got = np.asarray(get_fixed_population_params(name, shared_gamma=False))
    np.testing.assert_array_equal(got, shared)
    assert got.shape == (len(model.param_specs),)
    values = ds.Population(name, fixed=True, shared_gamma=False).fiducial_values
    assert list(values.values()) == list(got)


def test_multi_component_md_preset_still_requires_shared_gamma():
    with pytest.raises(ValueError, match="shared_gamma=True"):
        get_fixed_population_params("powerlaw+peak@md", shared_gamma=False)


def test_gp_constructors_refuse_unknown_mean_modes_and_axes():
    pytest.importorskip("jax")
    from darksirens.population.gp import JointGPPopulation, build_gp_model

    with pytest.raises(ValueError, match="mean-mode suffix 'md'"):
        build_gp_model("gp1d_m1@md")
    with pytest.raises(ValueError, match="mean-mode suffix"):
        build_gp_model("gp_separable@shpae")
    with pytest.raises(ValueError, match="mean_mode 'Shape'"):
        JointGPPopulation(("m1",), mean_mode="Shape")
    with pytest.raises(ValueError, match="unknown GP axis name.*chieff"):
        JointGPPopulation(("m1", "chieff"))
    # The two documented modes still build, and differ by the shape-mean slope.
    names = {s.name for s in build_gp_model("gp1d_m1@zero").param_specs}
    assert "alpha_gp" not in names
    assert "alpha_gp" in {s.name for s in build_gp_model("gp1d_m1@shape").param_specs}
    # Through the registry an "@" suffix is a rate decoration, not a mean mode.
    with pytest.raises(ValueError, match="Unknown rate-evolution decoration"):
        get_model("gp1d_m1@zero")


def test_out_of_prior_hint_names_the_public_escape_hatch(caplog):
    with caplog.at_level(logging.WARNING, logger="darksirens.population.grammar"):
        ds.Population("2powerlaws+peak", fixed=True).fiducial_values
    text = caplog.text
    assert "outside prior bounds" in text
    assert "fixed='in_prior_v2'" in text
    assert "--population_fiducials" not in text
    ds.Population("2powerlaws+peak", fixed="in_prior_v2").fiducial_values


def test_pairing_grid_error_names_the_public_controls(monkeypatch):
    from dataclasses import replace

    import darksirens.population.utils as U

    monkeypatch.setattr(U, "_NORMALIZATION_GRID_SETTINGS",
                        replace(U._NORMALIZATION_GRID_SETTINGS, pairing_m1_grid=1024,
                                pairing_m_hi=200.0))
    with pytest.raises(ValueError) as info:
        U.assert_pairing_grid_covers_support(300.0, model_name="gwtc5_fiducial_bpl2peaks")
    message = str(info.value)
    assert "--pairing_norm_grid" not in message and "CLI" not in message
    assert "ensure_pairing_grid_covers" in message
    assert "DARKSIRENS_GW_PAIRING_M1_GRID=none" in message
