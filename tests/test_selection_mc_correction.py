"""``selection_mc_correction``: the three forms of the selection Monte-Carlo term.

The likelihood carries ``-N log mu`` plus one of

* ``"farr"`` (the default): ``+N(N+3)/(2 N_eff)``, arXiv:1904.10879 Eq. 15;
* ``"none"``: nothing, the point estimate;
* ``"heinzel_vitale"``: ``-N(N+1)/(2 N_eff)``, arXiv:2509.07221 Eq. 24.

The default must stay the program it has always been (the file's own warning: a
second use of an operand moves admitted values by an ulp), so the forms are
chosen at trace time. These tests pin that, the exact value of each form under
both guards, the sign of each by Monte Carlo, that the guards are the same in all
three, the gradients at the edges of the estimator, the fingerprint, and that the
option reaches every likelihood, a host-density target included.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln

import darksirens as ds
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.inference.public import infer
from darksirens.inference.run_fingerprint import (
    core_numerics_semantic,
    fingerprint_from_semantic,
    inference_target_semantic,
    likelihood_options_semantic,
)
from darksirens.likelihood.host_density import host_density_log_likelihood
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.gw import (
    DEFAULT_SELECTION_MC_CORRECTION,
    SELECTION_MC_CORRECTIONS,
    check_selection_mc_correction,
    selection_log_correction,
    selection_reduce_from_ldw_provider,
)

from test_count_ratio_pooled import A, _bind, _pooled, _points
from test_host_density_extension import (
    MAX_VAR,
    VolumeRedshiftModel,
    _base_analysis,
    _runtime_events,
    _volume_target,
)
from test_run_fingerprint_coverage import _analysis, _stores

MODES = SELECTION_MC_CORRECTIONS
OPT_IN = tuple(mode for mode in MODES if mode != DEFAULT_SELECTION_MC_CORRECTION)


def _term(mode, n, neff):
    """The Monte-Carlo term by hand."""
    return {
        "farr": n * (n + 3.0) / (2.0 * neff),
        "none": 0.0,
        "heinzel_vitale": -n * (n + 1.0) / (2.0 * neff),
    }[mode]


def _frozen_selection_log_correction(
    log_mu, Neff, nEvents, soft_guard=False, max_likelihood_variance=1.0, pe_variance_sum=0.0
):
    """``selection_log_correction`` as it stood on main at 7457025, comments removed."""
    n = float(nEvents)
    budget = jnp.maximum(max_likelihood_variance - pe_variance_sum, 1e-12)
    threshold = jnp.maximum(5.0 * n, (n * n) / budget)
    exhausted = budget <= 1e-12
    if soft_guard:
        finite_neff = jnp.isfinite(Neff)
        neff_safe = jnp.where(finite_neff, Neff, threshold)
        x = jnp.where(finite_neff, neff_safe / threshold, jnp.inf)
        x = jnp.where(exhausted, jnp.minimum(x, 1e-3), x)
        gate = jax.nn.softplus(200.0 * (1.0 - x) - 10.0)
        reward_mag = n * jax.nn.softplus(-log_mu)
        wall = -gate * (100.0 + 2.0 * reward_mag)
        Neff_taylor = jnp.maximum(Neff, threshold)
        correction = (
            -n * log_mu
            + n * (3.0 + n) / (2.0 * Neff_taylor)
        )
        total = correction + wall
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)
    too_sparse = ~(Neff > threshold) | exhausted
    correction = (
        -n * log_mu
        + n * (3.0 + n) / (2.0 * Neff)
    )
    return jnp.where(too_sparse, -jnp.inf, correction)


# ---------------------------------------------------------------------------
# The default is the program it was


@pytest.mark.parametrize("soft_guard", [False, True])
@pytest.mark.parametrize("default", [dict(), dict(selection_mc_correction="farr")])
def test_the_default_is_bit_identical_to_main(soft_guard, default):
    def current(log_mu, neff, pe):
        return selection_log_correction(
            log_mu, neff, 259, soft_guard=soft_guard, max_likelihood_variance=4.0,
            pe_variance_sum=pe, **default,
        )

    def frozen(log_mu, neff, pe):
        return _frozen_selection_log_correction(
            log_mu, neff, 259, soft_guard=soft_guard, max_likelihood_variance=4.0,
            pe_variance_sum=pe,
        )

    args = (jnp.asarray(-1.5), jnp.asarray(3.0e4), jnp.asarray(1.2))
    assert str(jax.make_jaxpr(current)(*args)) == str(jax.make_jaxpr(frozen)(*args))

    # And the values, bit for bit, jitted, across the guard and its edge cases.
    grid = np.array(
        [[lm, neff, pe]
         for lm in (-30.0, -3.3, 0.0, 0.7)
         for neff in (0.0, 1.0, 1295.0, 2.2e4, 2.24e4, 2.3e4, 3.0e4, 4.0e6, np.inf, np.nan)
         for pe in (0.0, 1.2, 3.9, 3.999999999999, 4.0, 5.0, np.nan)]
    )
    run_current = jax.jit(jax.vmap(current))
    run_frozen = jax.jit(jax.vmap(frozen))
    a = np.asarray(run_current(*grid.T))
    b = np.asarray(run_frozen(*grid.T))
    assert a.tobytes() == b.tobytes()


def test_the_forms_and_their_default():
    assert DEFAULT_SELECTION_MC_CORRECTION == "farr"
    assert MODES == ("farr", "none", "heinzel_vitale")


# ---------------------------------------------------------------------------
# Exact values


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("soft_guard", [False, True])
def test_exact_value_above_the_threshold(mode, soft_guard):
    n, neff, log_mu = 40, 4.0e6, np.log(0.5)
    got = float(selection_log_correction(
        jnp.asarray(log_mu), jnp.asarray(neff), n, soft_guard=soft_guard,
        selection_mc_correction=mode,
    ))
    want = -n * log_mu + _term(mode, n, neff)
    assert got == pytest.approx(want, rel=1e-14)


def test_the_three_forms_have_the_signs_they_should():
    n, neff = 40, 4.0e6
    value = {
        mode: float(selection_log_correction(
            jnp.asarray(0.0), jnp.asarray(neff), n, selection_mc_correction=mode,
        ))
        for mode in MODES
    }
    assert value["heinzel_vitale"] < value["none"] == 0.0 < value["farr"]


@pytest.mark.parametrize("mode", MODES)
def test_soft_value_below_the_threshold(mode):
    """The wall by hand, plus each form's term: only ``"farr"`` is frozen."""
    n, log_mu, cap, pe = 50, -1.0, 4.0, 1.5
    threshold = max(5.0 * n, n * n / (cap - pe))
    neff = 0.97 * threshold
    got = float(selection_log_correction(
        jnp.asarray(log_mu), jnp.asarray(neff), n, soft_guard=True,
        max_likelihood_variance=cap, pe_variance_sum=pe, selection_mc_correction=mode,
    ))
    gate = float(jax.nn.softplus(200.0 * (1.0 - neff / threshold) - 10.0))
    wall = -gate * (100.0 + 2.0 * n * float(jax.nn.softplus(-log_mu)))
    term = _term(mode, n, threshold if mode == "farr" else neff)
    assert got == pytest.approx(-n * log_mu + term + wall, rel=1e-12)


@pytest.mark.parametrize("n, neff", [(10, 2000.0), (259, 3.0e4)])
def test_the_opt_in_forms_agree_with_the_same_arithmetic_on_arrays(n, neff):
    """Vectorised and jitted as the likelihood is, not only as Python scalars."""
    log_mu = jnp.linspace(-2.0, 1.0, 7)

    @jax.jit
    def run(x):
        return selection_log_correction(
            x, jnp.asarray(neff), n, max_likelihood_variance=10.0,
            selection_mc_correction="heinzel_vitale",
        )

    np.testing.assert_allclose(
        np.asarray(run(log_mu)),
        -n * np.asarray(log_mu) - n * (n + 1.0) / (2.0 * neff),
        rtol=1e-14,
        atol=0,
    )


# ---------------------------------------------------------------------------
# The sign, by Monte Carlo


def test_monte_carlo_each_form_on_a_gamma_estimate():
    """``mu_hat`` Gamma with mean 1 and variance ``1/k``, so ``N_eff = k`` exactly.

    The likelihood estimate ``mu_hat^-N`` is then high by a factor known in
    closed form, about ``exp(N(N+1)/(2k))``. ``"heinzel_vitale"`` removes that
    factor, so the mean of its exponential is 1 within its error; ``"none"``
    stays high by the factor (a negative control), and ``"farr"`` adds a term of
    the same sign and about twice the exponent (the other). A wrong sign fails.
    """
    n, k, draws = 10, 2000.0, 4_000_000
    assert n * n / k < 0.06  # small: the leading order is the whole story here
    rng = np.random.default_rng(20261031)
    log_mu_hat = np.log(rng.gamma(k, 1.0 / k, size=draws))
    exact_bias = n * np.log(k) + gammaln(k - n) - gammaln(k)
    assert exact_bias == pytest.approx(n * (n + 1.0) / (2.0 * k), rel=0.01)

    def mean_and_error(mode):
        value = selection_log_correction(
            jnp.asarray(log_mu_hat), jnp.asarray(k), n, selection_mc_correction=mode,
        )
        weights = np.exp(np.asarray(value))
        return weights.mean(), weights.std(ddof=1) / np.sqrt(draws)

    unbiased, error = mean_and_error("heinzel_vitale")
    assert abs(unbiased - 1.0) < 5.0 * error, (unbiased, error)
    # The size of each control, and far enough from 1 to fail a swapped sign.
    point, error = mean_and_error("none")
    assert point == pytest.approx(np.exp(exact_bias), abs=5.0 * error)
    assert point - 1.0 > 100.0 * error
    farr, error = mean_and_error("farr")
    assert farr == pytest.approx(
        np.exp(exact_bias + n * (n + 3.0) / (2.0 * k)), abs=5.0 * error
    )
    assert np.log(farr) / np.log(point) == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# The guards are the same in all three


@pytest.mark.parametrize("soft_guard", [False, True])
def test_heinzel_vitale_reads_the_selection_neff_only(soft_guard):
    """Above the threshold the term does not move with the PE variance: it uses
    the selection N_eff, never the total variance (arXiv:2509.07221 Eq. 24-25)."""
    n, neff, cap = 10, 1.0e4, 4.0
    values = {
        pe: float(selection_log_correction(
            jnp.asarray(-1.0), jnp.asarray(neff), n, soft_guard=soft_guard,
            max_likelihood_variance=cap, pe_variance_sum=pe,
            selection_mc_correction="heinzel_vitale",
        ))
        for pe in (0.0, 1.0, 2.5, 3.9)
    }
    assert len(set(values.values())) == 1, values
    assert values[0.0] == pytest.approx(n - n * (n + 1.0) / (2.0 * neff), rel=1e-14)


@pytest.mark.parametrize("mode", MODES)
def test_the_hard_wall_is_the_same_in_every_form(mode):
    n, cap, pe = 50, 4.0, 1.5
    threshold = max(5.0 * n, n * n / (cap - pe))

    def value(neff, pe=pe):
        return float(selection_log_correction(
            jnp.asarray(-1.0), jnp.asarray(neff), n,
            max_likelihood_variance=cap, pe_variance_sum=pe, selection_mc_correction=mode,
        ))

    assert np.isfinite(value(threshold * (1.0 + 1e-9)))
    for x in (1.0, 0.99, 0.5, 1e-6):
        assert value(x * threshold) == -np.inf
    # An exhausted budget, whatever N_eff is.
    for neff in (1.0e6, np.inf):
        assert value(neff, pe=cap) == -np.inf


@pytest.mark.parametrize("x", [0.999, 0.97, 0.9, 0.5, 0.05])
def test_the_soft_wall_is_the_same_in_every_form(x):
    """Below the threshold the forms differ by their term and nothing else."""
    n, log_mu, cap, pe = 50, -1.0, 4.0, 1.5
    threshold = max(5.0 * n, n * n / (cap - pe))
    neff = x * threshold
    walls = {}
    for mode in MODES:
        total = float(selection_log_correction(
            jnp.asarray(log_mu), jnp.asarray(neff), n, soft_guard=True,
            max_likelihood_variance=cap, pe_variance_sum=pe, selection_mc_correction=mode,
        ))
        term = _term(mode, n, threshold if mode == "farr" else neff)
        walls[mode] = total - (-n * log_mu + term)
    assert walls["none"] == pytest.approx(walls["farr"], rel=1e-10)
    assert walls["heinzel_vitale"] == pytest.approx(walls["farr"], rel=1e-10)


# ---------------------------------------------------------------------------
# The edges of the estimator


def _reduce(log_weights):
    return selection_reduce_from_ldw_provider(
        lambda start, size: jax.lax.dynamic_slice_in_dim(log_weights, start, size),
        log_weights.shape[0],
        4.0,
        None,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("soft_guard", [False, True])
@pytest.mark.parametrize(
    "weights",
    [
        pytest.param(jnp.zeros(4), id="neff-inf"),
        pytest.param(jnp.full(4, -jnp.inf), id="mu-zero-neff-zero"),
        pytest.param(jnp.array([0.0, 1.0, -0.5, 2.0]), id="ordinary"),
    ],
)
def test_gradients_through_the_reduction_at_the_edges(mode, soft_guard, weights):
    def correction(x):
        log_mu, neff, _ = _reduce(x)
        return selection_log_correction(
            log_mu, neff, nEvents=1, soft_guard=soft_guard, selection_mc_correction=mode,
        )

    value = correction(weights)
    grad = jax.grad(correction)(weights)
    assert not bool(jnp.any(jnp.isnan(grad))), grad
    if bool(jnp.isfinite(value)):
        assert bool(jnp.all(jnp.isfinite(grad))), grad


@pytest.mark.parametrize("mode", OPT_IN)
@pytest.mark.parametrize("soft_guard", [False, True])
@pytest.mark.parametrize(
    "log_mu, neff",
    [(-1.0, np.inf), (-1.0, 0.0), (-np.inf, 0.0)],
    ids=["neff-inf", "neff-zero", "mu-zero"],
)
def test_direct_gradients_of_the_opt_in_forms_at_the_edges(mode, soft_guard, log_mu, neff):
    """The double-where: no NaN cotangent in any operand, ``pe_variance_sum`` and
    the cap included, whatever the value (the term divides by ``N_eff``)."""

    def correction(lm, ne, pe, cap):
        return selection_log_correction(
            lm, ne, nEvents=10, soft_guard=soft_guard, max_likelihood_variance=cap,
            pe_variance_sum=pe, selection_mc_correction=mode,
        )

    args = tuple(jnp.asarray(v) for v in (log_mu, neff, 0.3, 1.0))
    grads = jax.grad(correction, argnums=(0, 1, 2, 3))(*args)
    if soft_guard and np.isneginf(log_mu):
        # At mu = 0 the wall's product -gate * (100 + inf) sends a NaN into N_eff,
        # the PE variance and the cap in every form, "farr" as on main; through
        # the reduction above those cotangents stop at the weights' selects.
        grads = grads[:1]
    assert not any(bool(jnp.isnan(g)) for g in grads), grads


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("soft_guard", [False, True])
def test_neff_inf_is_the_pure_poisson_term(mode, soft_guard):
    value = selection_log_correction(
        jnp.asarray(-1.0), jnp.asarray(jnp.inf), 10, soft_guard=soft_guard,
        selection_mc_correction=mode,
    )
    assert float(value) == pytest.approx(10.0, rel=1e-14)


@pytest.mark.parametrize("mode", MODES)
def test_a_nan_threshold_guards_at_infinite_neff(mode):
    """The wall is off at ``N_eff = +inf``, so only the term can carry a NaN
    budget to the soft path's collapse; the opt-in forms must carry it too."""
    value = selection_log_correction(
        jnp.asarray(-1.0), jnp.asarray(jnp.inf), 10, soft_guard=True,
        pe_variance_sum=jnp.asarray(jnp.nan), selection_mc_correction=mode,
    )
    assert bool(jnp.isneginf(value))


# ---------------------------------------------------------------------------
# Unknown values


@pytest.mark.parametrize("value", ["Farr", "off", "", None, 1, True])
def test_an_unknown_form_is_refused_everywhere(value):
    with pytest.raises(ValueError, match="selection_mc_correction must be one of"):
        check_selection_mc_correction(value)
    with pytest.raises(ValueError, match="selection_mc_correction must be one of"):
        selection_log_correction(
            jnp.asarray(0.0), jnp.asarray(1.0e6), 10, selection_mc_correction=value,
        )
    events, injections = _stores()
    with pytest.raises(ValueError, match="selection_mc_correction must be one of"):
        bind_analysis(
            _analysis(), events=events, injections=injections, selection_mc_correction=value,
        )
    with pytest.raises(ValueError, match="selection_mc_correction must be one of"):
        _volume_target(selection_mc_correction=value)
    with pytest.raises(ValueError, match="selection_mc_correction must be one of"):
        likelihood_options_semantic({"selection_mc_correction": value})


def test_replace_is_checked_too():
    events, injections = _stores()
    bound = bind_analysis(_analysis(), events=events, injections=injections)
    assert bound.selection_mc_correction == "farr"
    with pytest.raises(ValueError, match="selection_mc_correction must be one of"):
        dataclasses.replace(bound, selection_mc_correction="Farr")


# ---------------------------------------------------------------------------
# The run fingerprint


def _digest(semantic):
    return fingerprint_from_semantic(semantic)["digest"]


def test_the_fingerprint_is_unchanged_at_the_default_and_differs_otherwise():
    events, injections = _stores()
    analysis = _analysis()
    plain = bind_analysis(analysis, events=events, injections=injections)
    explicit = bind_analysis(
        analysis, events=events, injections=injections, selection_mc_correction="farr",
    )
    assert likelihood_options_semantic(explicit) == {}
    assert likelihood_options_semantic({"selection_mc_correction": "farr"}) == {}
    assert _digest(core_numerics_semantic(explicit)) == _digest(core_numerics_semantic(plain))
    digests = {_digest(core_numerics_semantic(plain))}
    for mode in OPT_IN:
        bound = bind_analysis(
            analysis, events=events, injections=injections, selection_mc_correction=mode,
        )
        assert likelihood_options_semantic(bound) == {"selection_mc_correction": mode}
        digests.add(_digest(core_numerics_semantic(bound)))
    assert len(digests) == 3


def test_a_target_fingerprint_records_the_form():
    def block(target):
        return inference_target_semantic(target, sampler="dynesty", options={})

    default = block(_volume_target())
    assert block(_volume_target(selection_mc_correction="farr")) == default
    changed = block(_volume_target(selection_mc_correction="none"))
    assert changed != default
    assert changed["inference_target"]["provenance"]["likelihood_options"] == {
        "selection_mc_correction": "none"
    }


# ---------------------------------------------------------------------------
# The option reaches every likelihood


def _expected_shift(mode, n, neff):
    """``ln L`` under ``mode`` minus ``ln L`` under ``"farr"``, away from the wall."""
    return _term(mode, n, neff) - _term("farr", n, neff)


def _check_shifts(make_bound, theta):
    farr = make_bound("farr")
    record = farr.diagnostics(jnp.asarray(theta))
    base = float(farr(jnp.asarray(theta)))
    assert np.isfinite(base)
    neff = float(record.n_eff)
    assert np.isfinite(neff) and neff > 0.0
    for mode in OPT_IN:
        bound = make_bound(mode)
        assert bound.selection_mc_correction == mode
        assert float(bound(jnp.asarray(theta))) - base == pytest.approx(
            _expected_shift(mode, farr.n_events, neff), rel=1e-8, abs=1e-10
        )
        # The diagnostics record is the same program's.
        selection_term = float(bound.diagnostics(jnp.asarray(theta)).selection_log_correction)
        assert selection_term == pytest.approx(
            float(record.selection_log_correction) + _expected_shift(mode, farr.n_events, neff),
            rel=1e-8,
            abs=1e-10,
        )


def test_the_option_reaches_a_spectral_binding():
    events, injections = _stores()
    analysis = _analysis()
    theta = np.array([70.0])

    def make(mode):
        return bind_analysis(
            analysis, events=events, injections=injections, max_likelihood_variance=1.0e6,
            selection_mc_correction=mode,
        )

    _check_shifts(make, theta)


@pytest.mark.parametrize("weighting", ["conditional", "field"])
def test_the_option_reaches_a_catalog_binding(weighting):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="count_ratio='pooled'")
        analysis = _pooled(A, catalog_sky_weighting=weighting)

        def make(mode):
            return _bind(analysis, selection_mc_correction=mode)

        _check_shifts(make, _points(analysis, 1)[0])


def test_the_option_reaches_a_host_density_target():
    """The DESI target calls ``host_density_log_likelihood`` directly, so without
    this the option would never reach it."""
    pe, selection, nsamp, nsel = _runtime_events()
    base = _base_analysis()

    def direct(mode):
        return host_density_log_likelihood(
            CosmologyParameters(67.74, 0.3075, -1.0, 0.0),
            np.asarray(base.parameters.fixed_population),
            np.array([]),
            pe,
            None,
            selection,
            None,
            1,
            nsamp,
            float(nsel),
            redshift_model=VolumeRedshiftModel(),
            pop_model=base.population.model_name,
            shared_beta=base.population.shared_beta,
            shared_spin=base.population.shared_spin,
            shared_gamma=base.population.shared_gamma,
            max_likelihood_variance=MAX_VAR,
            return_diagnostics=True,
            selection_mc_correction=mode,
        )

    farr = direct("farr")
    neff = float(farr.n_eff)
    assert np.isfinite(float(farr.log_likelihood))
    theta = jnp.asarray([0.0])
    target_farr = _volume_target(max_likelihood_variance=MAX_VAR)
    for mode in OPT_IN:
        other = direct(mode)
        shift = _expected_shift(mode, 1, neff)
        assert float(other.log_likelihood) - float(farr.log_likelihood) == pytest.approx(
            shift, rel=1e-8, abs=1e-12
        )
        target = _volume_target(max_likelihood_variance=MAX_VAR, selection_mc_correction=mode)
        assert float(target.log_likelihood(theta)) - float(
            target_farr.log_likelihood(theta)
        ) == pytest.approx(shift, rel=1e-8, abs=1e-12)


def test_ds_log_likelihood_takes_the_option_and_none_means_the_default():
    events, injections = _stores()
    analysis = _analysis()
    bound = ds.log_likelihood(
        analysis, events=events, injections=injections, max_likelihood_variance=1.0e6,
    )
    assert bound.selection_mc_correction == "farr"
    bound = ds.log_likelihood(
        analysis, events=events, injections=injections, max_likelihood_variance=1.0e6,
        selection_mc_correction="heinzel_vitale",
    )
    assert bound.selection_mc_correction == "heinzel_vitale"


def test_infer_refuses_the_option_for_a_target():
    with pytest.raises(TypeError, match="selection_mc_correction"):
        infer(_volume_target(), sampler="dynesty", selection_mc_correction="none")
    with pytest.raises(TypeError, match="selection_mc_correction"):
        infer(_volume_target(), sampler="dynesty", selection_mc_correction="farr")


def test_the_guard_report_records_a_form_other_than_the_default():
    from darksirens.inference.guard_report import guard_report

    events, injections = _stores()
    analysis = _analysis()

    def report(mode):
        bound = bind_analysis(
            analysis, events=events, injections=injections, max_likelihood_variance=1.0e6,
            selection_mc_correction=mode,
        )
        return guard_report(
            bound, lambda u: 20.0 + 120.0 * u, analysis.parameters.labels, n_draws=2, seed=1,
        )

    assert "selection_mc_correction" not in report("farr")
    assert report("none")["selection_mc_correction"] == "none"
