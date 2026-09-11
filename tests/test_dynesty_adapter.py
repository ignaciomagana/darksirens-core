"""Phase 6O/6P tests for the lazy Dynesty execution adapter."""

import numpy as np
import pytest

from darksirens.inference.dynesty_adapter import _normalized_dynesty_weights


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
