"""Phase 6I tests for backend-independent dead-point packaging."""

import numpy as np

from darksirens.inference.nested_output import package_dead_points


def test_missing_arrays_return_none_without_output(capsys):
    assert package_dead_points(None, None) is None
    assert package_dead_points(np.zeros(3), None, n_live=5) is None
    assert capsys.readouterr().out == ""


def test_mismatched_arrays_preserve_warning_text(capsys):
    assert package_dead_points(np.zeros(3), np.zeros(4), n_live=5) is None
    assert capsys.readouterr().out == (
        "  [!] dead-point logl/logwt have shapes (3,)/(4,); "
        "not persisting them.\n"
    )


def test_empty_arrays_preserve_warning_text(capsys):
    assert package_dead_points(np.zeros(0), np.zeros(0), n_live=5) is None
    assert capsys.readouterr().out == (
        "  [!] dead-point logl/logwt have shapes (0,)/(0,); "
        "not persisting them.\n"
    )


def test_flattens_to_float64_and_records_lengths():
    block = package_dead_points(
        np.arange(4, dtype=np.int32).reshape(2, 2),
        np.arange(4, dtype=np.float32).reshape(1, 4),
        n_live=np.int64(2),
    )
    assert block["n_dead"] == 4
    assert block["n_live"] == 2
    assert block["logl"].shape == block["logwt"].shape == (4,)
    assert block["logl"].dtype == np.float64
    assert block["logwt"].dtype == np.float64
    np.testing.assert_array_equal(block["logl"], np.arange(4.0))
    np.testing.assert_array_equal(block["logwt"], np.arange(4.0))


def test_n_live_is_optional():
    block = package_dead_points(np.arange(3.0), np.arange(3.0))
    assert block["n_dead"] == 3
    assert "n_live" not in block
