"""``model(..., n0_units=...)``: the unit of the incomplete catalog's ``log10n0``.

``"physical"`` (the frozen default) reads ``log10n0`` as Mpc^-3 at the sampled
H0, so the expected galaxy count scales as ``n0 * H0^-3`` against fixed
catalog counts. ``"h_scaled"`` reads it as h^3 Mpc^-3 and the bound likelihood
uses ``n0 = 10**log10n0 * (H0 / 100)**3``. The default must leave every
decode bit for bit as before; the h-scaled likelihood at ``(H0, log10n0)``
must equal the physical one at ``(H0, log10n0 + 3 log10(H0 / 100))``; and the
setting is part of the plan's fingerprint block only when it is not the
default, so existing fingerprints are unchanged.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from darksirens import Population, model
from darksirens.catalog import H0_N0_REF, physical_n0
from darksirens.inference.run_fingerprint import parameter_plan_semantic
from darksirens.runtime_binding import _decode_theta, bind_analysis

from test_partial_fixing import COSMOLOGY, MODEL, TAIL, _catalog, _stores


def _analysis(n0_units=None, **kwargs):
    return model(
        cosmology=COSMOLOGY,
        population=Population(MODEL, fixed=TAIL),
        catalog=_catalog(),
        n0_units=n0_units,
        **kwargs,
    )


def _theta(analysis, H0, log10n0, delta=0.5, sigma_kde=0.01):
    values = {"H0": H0, "log10n0": log10n0, "delta": delta, "sigma_kde": sigma_kde}
    labels = analysis.parameters.labels
    rest = {
        label: 0.5 * (lo + hi)
        for label, lo, hi in zip(labels, analysis.parameters.lower, analysis.parameters.upper)
        if label not in values
    }
    values.update(rest)
    return np.array([values[label] for label in labels])


def test_default_is_physical_and_the_plan_records_it():
    default = _analysis()
    explicit = _analysis("physical")
    assert default.redshift.n0_units == "physical"
    assert default.parameters.n0_units == "physical"
    assert default.parameters == explicit.parameters
    assert _analysis("h_scaled").parameters.n0_units == "h_scaled"


def test_physical_decode_is_bitwise_ten_to_the_log10n0():
    analysis = _analysis()
    theta = _theta(analysis, 70.0, -2.5)
    _, _, params, _ = _decode_theta(analysis, theta, z_depth=0.3)
    want = 10.0 ** jax.numpy.asarray(theta[analysis.parameters.labels.index("log10n0")])
    assert np.asarray(params.n0).tobytes() == np.asarray(want).tobytes()


@pytest.mark.parametrize("H0", (60.0, 67.74, 80.0))
def test_h_scaled_decode_scales_n0_by_h_cubed(H0):
    analysis = _analysis("h_scaled")
    theta = _theta(analysis, H0, -2.0)
    _, _, params, _ = _decode_theta(analysis, theta, z_depth=0.3)
    assert math.isclose(float(params.n0), 10.0 ** -2.0 * (H0 / H0_N0_REF) ** 3, rel_tol=1e-14)


@pytest.mark.parametrize("H0", (60.0, 72.5))
def test_h_scaled_likelihood_is_the_physical_one_at_the_shifted_log10n0(H0):
    events, injections = _stores()
    h_bound = bind_analysis(_analysis("h_scaled"), events=events, injections=injections)
    p_bound = bind_analysis(_analysis("physical"), events=events, injections=injections)
    log10n0_hat = -2.0
    got = float(h_bound(_theta(h_bound.analysis, H0, log10n0_hat)))
    shifted = log10n0_hat + 3.0 * math.log10(H0 / H0_N0_REF)
    want = float(p_bound(_theta(p_bound.analysis, H0, shifted)))
    assert np.isfinite(want)
    assert abs(got - want) <= 1e-12 * abs(want), (H0, got, want)


def test_h_scaled_moves_the_likelihood_with_H0_differently():
    # Same coordinate, two H0 values: the physical and h-scaled likelihoods
    # differ in their H0 dependence because the expected count differs.
    events, injections = _stores()
    h_bound = bind_analysis(_analysis("h_scaled"), events=events, injections=injections)
    p_bound = bind_analysis(_analysis("physical"), events=events, injections=injections)
    dh = float(h_bound(_theta(h_bound.analysis, 75.0, -2.0))) - float(h_bound(_theta(h_bound.analysis, 65.0, -2.0)))
    dp = float(p_bound(_theta(p_bound.analysis, 75.0, -2.0))) - float(p_bound(_theta(p_bound.analysis, 65.0, -2.0)))
    assert dh != dp


def test_physical_n0_helper():
    assert physical_n0(-2.0, 70.0) == 10.0 ** -2.0
    assert math.isclose(physical_n0(-2.0, 70.0, "h_scaled"), 1e-2 * 0.7 ** 3, rel_tol=1e-15)
    with pytest.raises(ValueError, match="n0_units"):
        physical_n0(-2.0, 70.0, "comoving")


def test_fingerprint_block_is_unchanged_for_the_default_and_changes_for_h_scaled():
    physical = parameter_plan_semantic(_analysis().parameters)
    assert "n0_units" not in physical
    h_scaled = parameter_plan_semantic(_analysis("h_scaled").parameters)
    assert h_scaled["n0_units"] == "h_scaled"
    assert {k: v for k, v in h_scaled.items() if k != "n0_units"} == physical


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(catalog=None), "incomplete-catalog"),
        (dict(completeness="complete"), "incomplete-catalog"),
        (dict(n0_units="comoving"), "n0_units must be one of"),
    ],
)
def test_n0_units_is_refused_outside_its_domain(kwargs, match):
    base = dict(cosmology=COSMOLOGY, population=Population(MODEL, fixed=TAIL), catalog=_catalog(), n0_units="h_scaled")
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        model(**base)
