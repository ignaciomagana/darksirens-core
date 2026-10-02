"""``Population.fiducial_values``: the values a fixed population evaluates at.

A preset (``fixed=True``) pins a curated fiducial vector whose choices are
easy to miss; the legacy powerlaw+peak one has the merger-rate slope
``gamma = 2.5``, which legacy changed from 0 (commit 0befab7). The property must
be exactly the vector the bound likelihood evaluates.
"""

from __future__ import annotations

import numpy as np
import pytest

import darksirens as ds
from darksirens.runtime_binding import _decode_theta

COSMOLOGY = ds.Cosmology(H0=(20.0, 140.0), Om0=0.3075, w0=-1.0, wa=0.0)


@pytest.mark.parametrize("name, fixed", [
    ("powerlaw+peak", True),
    ("powerlaw+peak", "in_prior_v2"),
    ("brokenpowerlaw+2peaks", "gwtc5"),
])
def test_preset_fiducial_values_are_what_the_likelihood_evaluates(name, fixed):
    population = ds.Population(name, fixed=fixed)
    values = population.fiducial_values
    analysis = ds.model(cosmology=COSMOLOGY, population=population)
    theta = np.array([67.74])
    _, population_vector, _, _ = _decode_theta(analysis, theta, z_depth=None)
    assert list(values.values()) == [float(x) for x in np.asarray(population_vector)]
    assert len(values) == len(population_vector)


def test_legacy_powerlaw_peak_rate_slope_is_visible():
    assert ds.Population("powerlaw+peak", fixed=True).fiducial_values[r"$\gamma$"] == 2.5


def test_mapping_and_sampled_populations():
    mapping = ds.Population("powerlaw+peak", fixed={r"$\gamma$": 0.0})
    assert mapping.fiducial_values == {r"$\gamma$": 0.0}
    assert ds.Population("powerlaw+peak").fiducial_values == {}
