"""Phase 6K tests for TinyNS diagnostic normalization."""

import json

import numpy as np

from darksirens.inference.tinyns_output import normalize_tinyns_diagnostics


class _RuntimeResult:
    logz = -22.31
    logzerr = np.float64(0.07)

    def diagnostics(self):
        return {
            "ncall": np.int64(200),
            "niter": 10,
            "seconds": 5.0,
            "replacement_mean_batches": np.float32(1.5),
            "replacement_failures": 2,
            "replacement_rescue_used": True,
            "replacement_rescue_stage_counts": np.array([1, 2, 3]),
            "ignored_field": "must not leak",
        }

    def summary(self):
        return {"ncall": 250, "message": "summary wins"}


class _BrokenMappings:
    logz = -1.0
    logzerr = 0.2

    def diagnostics(self):
        raise RuntimeError("diagnostics unavailable")

    def summary(self):
        return "not-a-mapping"


class _CallableAttribute:
    ncall = 9

    def niter(self):
        return 11


class _Stringable:
    def __str__(self):
        return "stable-object"


class _NestedValues:
    def diagnostics(self):
        return {
            "replacement_rescue_stage_counts": {
                7: np.array([np.int64(2), np.int64(4)]),
                "tuple": (np.float32(1.25), _Stringable()),
            },
            "seconds": 0.0,
            "ncall": 6,
            "niter": 0,
            "replacement_failures": 3,
        }


def test_diagnostic_sources_precedence_attributes_and_derived_rates():
    diag = normalize_tinyns_diagnostics(_RuntimeResult())
    assert diag["ncall"] == 250
    assert diag["niter"] == 10
    assert diag["seconds"] == 5.0
    assert diag["message"] == "summary wins"
    assert diag["logz"] == -22.31
    assert diag["logzerr"] == 0.07
    assert diag["replacement_mean_batches"] == np.float32(1.5).item()
    assert diag["replacement_rescue_stage_counts"] == [1, 2, 3]
    assert "ignored_field" not in diag
    assert diag["ncall_per_sec"] == 50.0
    assert diag["niter_per_sec"] == 2.0
    assert diag["calls_per_iter"] == 25.0
    assert diag["replacement_failure_rate"] == 0.2
    json.dumps(diag)


def test_broken_or_non_mapping_methods_fall_back_to_attributes():
    diag = normalize_tinyns_diagnostics(_BrokenMappings())
    assert diag == {"logz": -1.0, "logzerr": 0.2}
    json.dumps(diag)


def test_callable_attributes_are_not_serialized():
    diag = normalize_tinyns_diagnostics(_CallableAttribute())
    assert diag == {"ncall": 9}


def test_nested_values_are_json_safe_and_zero_niter_uses_unit_denominator():
    diag = normalize_tinyns_diagnostics(_NestedValues())
    assert diag["replacement_rescue_stage_counts"] == {
        "7": [2, 4],
        "tuple": [np.float32(1.25).item(), "stable-object"],
    }
    assert "ncall_per_sec" not in diag
    assert "niter_per_sec" not in diag
    assert diag["calls_per_iter"] == 6.0
    assert diag["replacement_failure_rate"] == 3.0
    json.dumps(diag)


def test_jax_array_conversion_is_lazy_and_json_safe():
    import jax.numpy as jnp

    class Result:
        def diagnostics(self):
            return {"replacement_batch_ncall": jnp.asarray([3, 5])}

    diag = normalize_tinyns_diagnostics(Result())
    assert diag["replacement_batch_ncall"] == [3, 5]
    json.dumps(diag)
