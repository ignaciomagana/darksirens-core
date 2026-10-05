"""``ds.save_result`` writes one complete HDF5 result in the documented layout."""

from __future__ import annotations

import json

import numpy as np
import pytest

import darksirens as ds
from darksirens.io.results import (
    DEAD_POINT_SEMANTICS,
    RESULT_SCHEMA_VERSION,
    result_is_complete,
)

h5py = pytest.importorskip("h5py")


def _result():
    rng = np.random.default_rng(0)
    return {
        "samples": rng.normal(size=(20, 2)),
        "logZ": -12.5,
        "logZerr": 0.1,
        "log_prior_volume_fraction": 0.0,
        "logZ_corrected": -12.5,
        "dead_points": {
            "logl": np.linspace(-30.0, -10.0, 40),
            "logwt": np.linspace(-40.0, -15.0, 40),
            "n_dead": 40,
            "n_live": 25,
        },
        "nlive_actual": np.int64(25),
        "dlogz_final": 0.08,
        "stop_reason": "convergence",
        "ncall": 900,
        "niter": 15,
        "guard_report": {
            "n_draws": 32,
            "guarded": 3,
            "fired": {"pe_mc_variance": {"count": 3, "ranges": {"H0": [20.5, np.float64(48.0)]}}},
        },
    }


def test_save_result_layout(tmp_path):
    path = tmp_path / "result.h5"
    result = _result()
    ds.save_result(path, result, labels=("H0", "log10n0"))

    assert result_is_complete(path)
    assert not (tmp_path / "result.h5.tmp").exists()
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["samples"][()], result["samples"])
        assert [label.decode() for label in f["labels"][()]] == ["H0", "log10n0"]
        np.testing.assert_array_equal(f["logl_dead"][()], result["dead_points"]["logl"])
        np.testing.assert_array_equal(f["logwt_dead"][()], result["dead_points"]["logwt"])
        assert f.attrs["n_dead"] == 40
        assert f.attrs["n_live"] == 25
        assert f.attrs["dead_points"] == DEAD_POINT_SEMANTICS
        assert f.attrs["logZ"] == -12.5
        assert f.attrs["logZerr"] == 0.1
        assert f.attrs["logZ_corrected"] == -12.5
        assert f.attrs["stop_reason"] == "convergence"
        assert f.attrs["ncall"] == 900
        assert f.attrs["nlive_actual"] == 25
        assert f.attrs["result_schema_version"] == RESULT_SCHEMA_VERSION
        assert "log_likelihood" not in f
        stored = json.loads(f.attrs["result_json"])
    assert stored["guard_report"]["fired"]["pe_mc_variance"]["ranges"]["H0"] == [20.5, 48.0]
    assert stored["logZ"] == -12.5
    assert "samples" not in stored and "dead_points" not in stored


def test_save_result_without_evidence_or_labels(tmp_path):
    # A NumPyro-style result: no evidence, a per-sample log-likelihood.
    path = tmp_path / "numpyro.h5"
    result = {
        "samples": np.ones((5, 1)),
        "logZ": None,
        "logZerr": None,
        "log_likelihood": np.arange(5.0),
        "numpyro_diagnostics": {"mean_accept_prob": float("nan"), "n_divergent": 0},
        "stop_reason": None,
    }
    ds.save_result(path, result)
    with h5py.File(path, "r") as f:
        assert "labels" not in f
        assert "logZ" not in f.attrs and "stop_reason" not in f.attrs
        np.testing.assert_array_equal(f["log_likelihood"][()], np.arange(5.0))
        stored = json.loads(f.attrs["result_json"])
    assert stored["logZ"] is None
    assert stored["numpyro_diagnostics"] == {"mean_accept_prob": None, "n_divergent": 0}


def test_save_result_refuses_mismatched_labels_and_keeps_the_old_file(tmp_path):
    path = tmp_path / "result.h5"
    ds.save_result(path, _result(), labels=("H0", "x"))
    with pytest.raises(ValueError, match="labels"):
        ds.save_result(path, _result(), labels=("H0",))
    assert result_is_complete(path)


def test_save_result_takes_labels_from_the_result(tmp_path):
    path = tmp_path / "result.h5"
    ds.save_result(path, dict(_result(), labels=("a", "b")))
    with h5py.File(path, "r") as f:
        assert [label.decode() for label in f["labels"][()]] == ["a", "b"]
        assert "labels" not in json.loads(f.attrs["result_json"])
