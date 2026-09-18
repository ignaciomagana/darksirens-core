"""Independent anchors for the cosmology -> Jacobian -> mass-frame chain.

Every other numeric test in the likelihood layer is a restatement of the
implementation's own expression or a core-vs-core wiring comparison, so a
changed physics convention ships green.  The anchors here are written from
first principles instead:

* the sample weight is compared against a numpy expectation built from
  ``astropy.cosmology.FlatLambdaCDM`` rather than from core's own ``z_of_dL``
  and ``ddL_of_z``;
* the H0 slope of the spectral likelihood is pinned in sign and magnitude, the
  channel that the detector-to-source mass conversion dominates;
* the out-of-table distance mask is pinned against an explicitly masked
  fixture and against the clamp-only alternative.

MEASURED: dropping ``/(1 + z)`` at ``likelihood/weights.py:154`` leaves the
rest of the suite green while moving ``dlnL/dH0`` on the fixture below from
+0.0701 to +0.0092.
"""

import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, z_at_value

from darksirens.cosmology.distances import dL_grid_bounds
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.runtime import make_gw_event
from darksirens.likelihood import spectral_siren_log_likelihood
from darksirens.likelihood.weights import log_sample_weight
from darksirens.population import get_fixed_population_params


H0 = 67.74
OM0 = 0.3075
COSMO = CosmologyParameters(H0=H0, Om0=OM0, w0=-1.0, wa=0.0)
POP_MODEL = "powerlaw+peak"


# ------------------------------------------------------------------
# 1. log_sample_weight against an astropy-built numpy expectation.
# ------------------------------------------------------------------

M1DET = 52.0
Q = 0.7
CHIEFF = 0.05
DL = (250.0, 900.0, 2500.0)


def _log_pop_m1src_only(m1src, q, z, chieff, pop_params):
    """Analytic population density that depends on the SOURCE-frame mass only."""
    del q, z, chieff, pop_params
    return -0.5 * ((m1src - 30.0) / 8.0) ** 2


def _flat_log_prior_z(z, pix, catalog):
    del z, pix, catalog
    return 0.0


def _astropy_expected_log_weight(dL_value):
    """log w built from astropy alone: no core cosmology on this side.

    ``log_sample_weight`` returns

        log p_pop(m1src, q, z) + log p_z(z) - log|d(dL)/dz| - log(1+z)
        - log(prior_wt),

    with the Jacobian convention of
    ``log_jacobian_m1src_q_z_to_m1det_q_dL``.  Here p_z is flat and
    prior_wt = 1, so only the mass-frame conversion and the Jacobian remain.
    """
    astro = FlatLambdaCDM(H0=H0, Om0=OM0)
    z = float(z_at_value(astro.luminosity_distance, dL_value * u.Mpc, zmax=10.0))
    step = 1e-5
    ddL_dz = float(
        (
            astro.luminosity_distance(z + step).value
            - astro.luminosity_distance(z - step).value
        )
        / (2.0 * step)
    )
    m1src = M1DET / (1.0 + z)
    return -0.5 * ((m1src - 30.0) / 8.0) ** 2 - (np.log(ddL_dz) + np.log1p(z))


def test_log_sample_weight_matches_an_astropy_built_expectation():
    dL = jnp.asarray(DL)
    ones = jnp.ones_like(dL)
    actual = np.asarray(
        log_sample_weight(
            M1DET * ones,
            Q * ones,
            dL,
            CHIEFF * ones,
            jnp.zeros_like(dL, dtype=jnp.int32),
            ones,
            COSMO,
            None,
            jnp.asarray([]),
            None,
            _log_pop_m1src_only,
            _flat_log_prior_z,
        )
    )
    expected = np.asarray([_astropy_expected_log_weight(v) for v in DL])
    # Core tabulates dL(z) and d(dL)/dz on a fixed grid; the measured floor of
    # that interpolation against astropy is 2.2e-6 relative on these points.
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=0.0)

    # The anchor has to bite: evaluating the population at the DETECTOR-frame
    # mass instead moves every point by far more than that tolerance.
    astro = FlatLambdaCDM(H0=H0, Om0=OM0)
    z = np.asarray(
        [
            float(z_at_value(astro.luminosity_distance, v * u.Mpc, zmax=10.0))
            for v in DL
        ]
    )
    detector_frame_shift = -0.5 * ((M1DET - 30.0) / 8.0) ** 2 + 0.5 * (
        (M1DET / (1.0 + z) - 30.0) / 8.0
    ) ** 2
    assert np.all(np.abs(detector_frame_shift) > 1e-2 * np.abs(expected))


def test_sample_weight_moves_by_the_source_frame_shift_between_two_distances():
    """Same detector-frame mass at two distances differs by the mass-frame term."""
    astro = FlatLambdaCDM(H0=H0, Om0=OM0)
    dL = jnp.asarray([400.0, 2400.0])
    ones = jnp.ones_like(dL)
    got = np.asarray(
        log_sample_weight(
            M1DET * ones,
            Q * ones,
            dL,
            CHIEFF * ones,
            jnp.zeros_like(dL, dtype=jnp.int32),
            ones,
            COSMO,
            None,
            jnp.asarray([]),
            None,
            _log_pop_m1src_only,
            _flat_log_prior_z,
        )
    )
    z = np.asarray(
        [
            float(z_at_value(astro.luminosity_distance, float(v) * u.Mpc, zmax=10.0))
            for v in dL
        ]
    )
    step = 1e-5
    ddL = np.asarray(
        [
            (
                astro.luminosity_distance(zi + step).value
                - astro.luminosity_distance(zi - step).value
            )
            / (2.0 * step)
            for zi in z
        ]
    )
    pop = -0.5 * ((M1DET / (1.0 + z) - 30.0) / 8.0) ** 2
    expected_delta = (pop[1] - pop[0]) - (
        np.log(ddL[1]) - np.log(ddL[0]) + np.log1p(z[1]) - np.log1p(z[0])
    )
    np.testing.assert_allclose(got[1] - got[0], expected_delta, rtol=1e-4, atol=0.0)


# ------------------------------------------------------------------
# 2. dlnL/dH0 on a spectral fixture.
# ------------------------------------------------------------------

# The helpers from tests/test_spectral_likelihood.py, with the distance range
# widened: at its shipped dL in [420, 1050] (PE) and [300, 1500] (injections)
# the H0 channel is numerically degenerate (dlnL/dH0 = -1.3e-3), which cannot
# anchor a sign.  Everything else is that fixture.
PE_DL = (420.0, 4000.0)
SEL_DL = (300.0, 6000.0)
N_EVENTS, NSAMP, N_SEL, N_DRAW = 5, 64, 256, 1000.0
# Measured on this tree at the fiducial; the review's own fixture gave +0.1383.
DLNL_DH0 = 0.0700937739767669


def _spectral_samples(n_events=N_EVENTS, nsamp=NSAMP, dL=PE_DL, valid=None):
    n = n_events * nsamp
    m1 = jnp.linspace(32.0, 44.0, n)
    q = jnp.linspace(0.68, 0.92, n)
    return make_gw_event(
        m1det=m1,
        m2det=m1 * q,
        dL=jnp.linspace(dL[0], dL[1], n) if not isinstance(dL, jnp.ndarray) else dL,
        chieff=jnp.linspace(-0.12, 0.18, n),
        prior_wt=jnp.linspace(0.7, 1.3, n),
        pixels=jnp.zeros(n, dtype=jnp.int32),
        valid=jnp.ones(n, dtype=bool) if valid is None else valid,
    )


def _spectral_selection(n=N_SEL, dL=SEL_DL, valid=None):
    m1 = jnp.linspace(28.0, 50.0, n)
    q = jnp.linspace(0.60, 0.96, n)
    return make_gw_event(
        m1det=m1,
        m2det=m1 * q,
        dL=jnp.linspace(dL[0], dL[1], n) if not isinstance(dL, jnp.ndarray) else dL,
        chieff=jnp.linspace(-0.2, 0.25, n),
        prior_wt=jnp.linspace(0.8, 1.2, n),
        pixels=jnp.zeros(n, dtype=jnp.int32),
        valid=jnp.ones(n, dtype=bool) if valid is None else valid,
    )


def _spectral_of_H0(pe, sel, n_events=N_EVENTS, nsamp=NSAMP):
    pop = get_fixed_population_params(POP_MODEL)

    def log_likelihood(h0):
        return spectral_siren_log_likelihood(
            CosmologyParameters(H0=h0, Om0=OM0, w0=-1.0, wa=0.0),
            pop,
            pe,
            sel,
            n_events,
            nsamp,
            N_DRAW,
            pop_model=POP_MODEL,
            max_likelihood_variance=1e6,
        )

    return log_likelihood


def test_h0_slope_of_the_spectral_likelihood_is_positive_and_pinned():
    """m1src = m1det/(1+z(H0)) is the dominant H0-sensitivity channel."""
    log_likelihood = _spectral_of_H0(_spectral_samples(), _spectral_selection())
    h0 = jnp.asarray(H0)
    slope = float(jax.grad(log_likelihood)(h0))

    assert slope > 0.05, slope
    assert slope == pytest.approx(DLNL_DH0, rel=1e-6)

    # A two-point finite difference anchors the autodiff value independently.
    step = 1e-3
    up = log_likelihood(jnp.asarray(H0 + step))
    down = log_likelihood(jnp.asarray(H0 - step))
    fd = float((up - down) / (2.0 * step))
    assert fd == pytest.approx(slope, rel=1e-5)


# ------------------------------------------------------------------
# 3. Out-of-table distance mask.
# ------------------------------------------------------------------

PE_OUT_OF_TABLE = (1, 11)
SEL_OUT_OF_TABLE = (3, 40, 61)
_OOB_EVENTS, _OOB_NSAMP, _OOB_NSEL = 2, 8, 64


def _out_of_table_fixture(mode):
    """``mode`` in {inside, out_of_table, masked, clamped}."""
    lo, hi = (float(v) for v in dL_grid_bounds(H0, OM0, -1.0, 0.0))
    n_pe = _OOB_EVENTS * _OOB_NSAMP
    pe_dL = np.linspace(420.0, 1050.0, n_pe)
    sel_dL = np.linspace(300.0, 1500.0, _OOB_NSEL)
    pe_valid = np.ones(n_pe, dtype=bool)
    sel_valid = np.ones(_OOB_NSEL, dtype=bool)

    if mode == "out_of_table":
        # dL_lo is exactly 0, so the lower rail is only reachable from a
        # nonpositive distance -- precisely the corrupted input the mask must
        # discard rather than clamp onto the grid edge.
        pe_dL[PE_OUT_OF_TABLE[0]] = lo - 1.0
        pe_dL[PE_OUT_OF_TABLE[1]] = 1.3 * hi
        sel_dL[SEL_OUT_OF_TABLE[0]] = lo - 5.0
        sel_dL[SEL_OUT_OF_TABLE[1]] = 1.1 * hi
        sel_dL[SEL_OUT_OF_TABLE[2]] = 1.5 * hi
    elif mode == "masked":
        for i in PE_OUT_OF_TABLE:
            pe_dL[i] = 700.0
            pe_valid[i] = False
        for i in SEL_OUT_OF_TABLE:
            sel_dL[i] = 700.0
            sel_valid[i] = False
    elif mode == "clamped":
        pe_dL[PE_OUT_OF_TABLE[0]] = lo
        pe_dL[PE_OUT_OF_TABLE[1]] = hi
        sel_dL[SEL_OUT_OF_TABLE[0]] = lo
        sel_dL[SEL_OUT_OF_TABLE[1]] = hi
        sel_dL[SEL_OUT_OF_TABLE[2]] = hi
    elif mode != "inside":
        raise AssertionError(mode)

    pe = _spectral_samples(
        _OOB_EVENTS, _OOB_NSAMP, jnp.asarray(pe_dL), jnp.asarray(pe_valid)
    )
    sel = _spectral_selection(_OOB_NSEL, jnp.asarray(sel_dL), jnp.asarray(sel_valid))
    return pe, sel


def _out_of_table_diagnostics(mode):
    pe, sel = _out_of_table_fixture(mode)
    return spectral_siren_log_likelihood(
        COSMO,
        get_fixed_population_params(POP_MODEL),
        pe,
        sel,
        _OOB_EVENTS,
        _OOB_NSAMP,
        N_DRAW,
        pop_model=POP_MODEL,
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )


def test_out_of_table_distances_are_dropped_not_clamped():
    """hierarchical.py masks ``supported`` samples out; the clamp alone does not."""
    inside = _out_of_table_diagnostics("inside")
    out_of_table = _out_of_table_diagnostics("out_of_table")
    masked = _out_of_table_diagnostics("masked")
    clamped = _out_of_table_diagnostics("clamped")

    for name in (
        "log_likelihood",
        "log_mu",
        "n_eff",
        "selection_log_correction",
        "event_log_evidence",
        "event_mc_variance",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(out_of_table, name)),
            np.asarray(getattr(masked, name)),
            err_msg=name,
        )

    # The out-of-table samples really are load-bearing on this fixture, and the
    # clamp-only alternative is a different answer.
    assert float(out_of_table.log_likelihood) != float(inside.log_likelihood)
    assert float(out_of_table.log_mu) != float(clamped.log_mu)
    assert float(out_of_table.log_likelihood) != float(clamped.log_likelihood)
    assert abs(float(out_of_table.log_mu) - float(clamped.log_mu)) > 1e-3
