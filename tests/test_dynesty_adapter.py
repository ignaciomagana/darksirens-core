"""Phase 6O/6P tests for the lazy Dynesty execution adapter."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from darksirens.inference.dynesty_adapter import (
    _dynesty_running_logz,
    _dynesty_termination,
    _normalized_dynesty_weights,
)
from darksirens.inference.nested_output import TERMINATION_FIELDS


def test_weight_normalization_matches_frozen_formula():
    logw, weights = _normalized_dynesty_weights([-np.inf, -2.0, -3.0])
    expected = np.exp(logw - (-2.0))
    expected /= expected.sum()
    np.testing.assert_array_equal(logw, np.array([-np.inf, -2.0, -3.0]))
    np.testing.assert_allclose(weights, expected, rtol=0, atol=0)


def test_no_finite_weights_fail_eagerly():
    with pytest.raises(RuntimeError, match="no finite posterior weights"):
        _normalized_dynesty_weights([-np.inf, np.nan])


def test_non_normalizable_weights_fail_eagerly():
    with pytest.raises(RuntimeError, match="could not be normalized"):
        _normalized_dynesty_weights([0.0, np.inf])


# --- final dlogz and stop reason -------------------------------------------
#
# A fake sampler carries the post-run state dynesty 2.1.4 leaves behind: the
# live likelihoods, ``it`` (dead points + 1), the call counter, and a
# ``saved_run`` whose ``logz`` list run_nested has already replaced with the
# recomputed evidence of the whole run (dead points followed by live points).


def _post_run_sampler(*, running_logz, recomputed_logz, logvol, live_logl, ncall,
                      nlive=4, logvol_init=0.0):
    n_dead = len(logvol)
    return SimpleNamespace(
        it=n_dead + 1,
        ncall=ncall,
        nlive=nlive,
        logvol_init=logvol_init,
        live_logl=np.asarray(live_logl, dtype=float),
        saved_run={
            "logz": list(recomputed_logz),
            "logvol": list(logvol) + [-9.0] * nlive,
        },
    ), list(running_logz)


def _converged_sampler(ncall=361):
    # The loop's running ln Z at the last dead point (-3.0) differs from the
    # recomputed one (-2.5); the stop check used the running value.
    return _post_run_sampler(
        running_logz=[-6.0, -4.0, -3.0],
        recomputed_logz=[-6.0, -4.0, -2.5] + [-2.4] * 4,
        logvol=[-0.25, -0.5, -0.75],
        live_logl=[-8.0, -7.5, -7.0, -6.5],
        ncall=ncall,
    )


def test_final_dlogz_is_dynestys_stop_check_on_the_running_evidence():
    sampler, running = _converged_sampler()
    record = _dynesty_termination(sampler, running, 50, 0.1, None)
    expected = np.logaddexp(0, np.max(sampler.live_logl) + (-0.75) - (-3.0))
    assert record["dlogz_final"] == float(expected)
    assert record["dlogz_final"] < 0.1
    assert record == {
        "dlogz_final": float(expected),
        "stop_reason": "convergence",
        "ncall": 361,
        "niter": 3,
    }


def test_maxcall_stop_counts_only_the_calls_of_this_run_nested():
    # 311 loop calls after 50 initial live-point calls: dynesty's cap is on
    # the 311.
    sampler, running = _post_run_sampler(
        running_logz=[-6.0, -5.5],
        recomputed_logz=[-6.0, -5.5] + [-5.0] * 4,
        logvol=[-0.25, -0.5],
        live_logl=[-3.0, -2.0, -1.0, 0.0],
        ncall=361,
    )
    record = _dynesty_termination(sampler, running, 50, 0.1, 300)
    assert record["stop_reason"] == "maxcall"
    assert record["dlogz_final"] > 0.1
    assert record["ncall"] == 361 and record["niter"] == 2
    # 350 total calls are only 300 loop calls: the cap did not fire.
    sampler.ncall = 350
    assert _dynesty_termination(sampler, running, 50, 0.1, 300)["stop_reason"] == "unknown"


def test_a_run_that_meets_dlogz_at_the_cap_counts_as_converged():
    sampler, running = _converged_sampler(ncall=1000)
    record = _dynesty_termination(sampler, running, 50, 0.1, 300)
    assert record["stop_reason"] == "convergence"


def test_plateau_stop_is_named():
    sampler, running = _post_run_sampler(
        running_logz=[-6.0],
        recomputed_logz=[-6.0] + [-5.0] * 4,
        logvol=[-0.25],
        live_logl=[-1.0] * 4,
        ncall=10,
    )
    record = _dynesty_termination(sampler, running, 4, 0.1, None)
    assert record["stop_reason"] == "plateau"
    assert record["dlogz_final"] > 0.1


def test_unset_dlogz_uses_dynestys_default_threshold():
    # With add_live=True dynesty stops below 1e-3 * (nlive - 1) + 0.01.
    sampler, running = _converged_sampler()
    dlogz_final = _dynesty_termination(sampler, running, 50, 0.1, None)["dlogz_final"]
    assert 0.013 < dlogz_final < 0.015
    sampler.nlive = 4  # threshold 0.013
    assert _dynesty_termination(sampler, running, 50, None, None)["stop_reason"] == "unknown"
    sampler.nlive = 6  # threshold 0.015
    assert _dynesty_termination(sampler, running, 50, None, None)["stop_reason"] == "convergence"
    # The criterion is strict, as in dynesty.
    assert _dynesty_termination(sampler, running, 50, dlogz_final, None)["stop_reason"] == "unknown"


def test_a_run_that_retired_no_point_uses_the_loops_initial_values():
    sampler, running = _post_run_sampler(
        running_logz=[],
        recomputed_logz=[-5.0] * 4,
        logvol=[],
        live_logl=[-1.0, -0.5, 0.0, 0.5],
        ncall=4,
        logvol_init=0.0,
    )
    record = _dynesty_termination(sampler, running, 4, 0.1, None)
    assert record["niter"] == 0
    assert record["dlogz_final"] == float(np.logaddexp(0, 0.5 + 0.0 - (-1.0e300)))
    assert record["stop_reason"] == "unknown"


def test_an_unfamiliar_sampler_reports_no_termination():
    assert _dynesty_running_logz(SimpleNamespace()) is None
    assert _dynesty_termination(SimpleNamespace(), None, None, 0.1, None) == {
        "dlogz_final": None,
        "stop_reason": None,
        "ncall": None,
        "niter": None,
    }


def test_running_logz_is_the_list_object_dynesty_appends_to():
    logz = [-6.0]
    sampler = SimpleNamespace(saved_run={"logz": logz})
    assert _dynesty_running_logz(sampler) is logz


def test_termination_fields_survive_a_json_round_trip():
    sampler, running = _converged_sampler()
    record = _dynesty_termination(sampler, running, 50, 0.1, None)
    assert tuple(record) == TERMINATION_FIELDS
    back = json.loads(json.dumps(record, allow_nan=False))
    assert back == record
    assert [type(v) for v in back.values()] == [float, str, int, int]
