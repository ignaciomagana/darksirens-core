"""The two curated fiducial sets: legacy (out-of-prior) vs in_prior_v2.

``ds.Population(name, fixed='in_prior_v2')`` is part of the public surface, and
the whole point of that set is one invariant: every fiducial value lies inside
its own declared prior.  Three curated compositions inherit their component
blueprints' default fiducials while overriding the priors that would contain
them, so under the LEGACY set their fixed-population truth sits outside the
model's own prior::

    2powerlaws+peak    PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40];
                       G.mu 35 vs [50, 100]
    2powerlaws+2peaks  PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40]
    2powerlaws+3peaks  PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40]

``tests/data/population_registry_golden.json`` pins the LEGACY vector only (its
``_snapshot`` calls ``get_fixed_population_params(name)`` with the default), so
without this file the corrected numbers are recorded nowhere and a retune of
``_IN_PRIOR_FIDUCIAL_PATCHES`` to another in-bounds value would change a public
option's output with a green suite.

The vectors below were verified identical in the frozen reference tree
(``darksirens.gw.populations.get_fixed_population_params`` under its own
interpreter): legacy, in_prior_v2, lows and highs all match element for element,
so pinning them here pins parity as well as behaviour.
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from darksirens.population import (
    FIDUCIAL_SETS,
    FIDUCIAL_SET_IN_PRIOR,
    FIDUCIAL_SET_LEGACY,
    get_fixed_population_params,
    pop_model_prior_parser,
)

#: The three curated entries whose legacy fiducials violate their own priors.
_OFFENDERS = ["2powerlaws+peak", "2powerlaws+2peaks", "2powerlaws+3peaks"]

#: Models whose two sets must be identical: grammar compositions with no patch,
#: and bespoke published vectors (no per-slot priors to patch at all).
_NON_OFFENDERS = [
    "powerlaw+peak",
    "brokenpowerlaw+2peaks",
    "brokenpowerlaw+2peaks+powerlaw",
    "gwtc3_fiducial_plpeak",
    "gwtc5_fiducial_bpl2peaks",
]

#: The corrected vectors, element for element.  Each patched value is the
#: midpoint of the slot's OVERRIDDEN prior (registry ``_IN_PRIOR_FIDUCIAL_PATCHES``
#: documents that rule); nothing else moves.
_IN_PRIOR_VECTORS = {
    "2powerlaws+peak": [
        0.2, 0.125, 2.3, 5.0, 32.5, 3.0, 10.0, 2.3, 30.0, 80.0, 3.0, 10.0,
        75.0, 5.0, 1.0, 0.0, 0.1, 2.5,
    ],
    "2powerlaws+2peaks": [
        0.15, 0.11764705882352942, 0.06666666666666667, 2.3, 5.0, 32.5, 3.0,
        10.0, 2.3, 30.0, 80.0, 3.0, 10.0, 10.0, 3.0, 35.0, 5.0, 1.0, 0.0, 0.1,
        2.5,
    ],
    "2powerlaws+3peaks": [
        0.15, 0.11764705882352942, 0.06666666666666667, 0.04285714285714286,
        2.3, 5.0, 32.5, 3.0, 10.0, 2.3, 30.0, 80.0, 3.0, 10.0, 10.0, 3.0, 35.0,
        5.0, 70.0, 10.0, 1.0, 0.0, 0.1, 2.5,
    ],
}

#: Indices the patch moves, and the legacy values it moves them from.
_PATCHED = {
    "2powerlaws+peak": {4: 80.0, 8: 5.0, 12: 35.0},
    "2powerlaws+2peaks": {5: 80.0, 9: 5.0},
    "2powerlaws+3peaks": {6: 80.0, 10: 5.0},
}


def _fid(name, fiducials):
    return np.asarray(get_fixed_population_params(name, fiducials=fiducials),
                      dtype=float)


def _bounds(name):
    lo, hi, labels, _kinds, _latex = pop_model_prior_parser(name)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), list(labels)


@pytest.mark.parametrize("name", _OFFENDERS)
def test_legacy_fiducials_are_outside_the_declared_priors(name):
    """The defect the corrected set exists for, pinned so it cannot silently go
    away (if it does, the legacy/in_prior split is obsolete and should be
    deleted rather than left as a dead flag)."""
    fid = _fid(name, FIDUCIAL_SET_LEGACY)
    lo, hi, labels = _bounds(name)
    outside = {labels[i]: float(fid[i]) for i in range(len(labels))
               if not (lo[i] <= fid[i] <= hi[i])}
    assert outside, f"{name}: the legacy fiducial is now INSIDE its priors"
    assert len(outside) == len(_PATCHED[name]), (name, outside)


@pytest.mark.parametrize("name", _OFFENDERS + _NON_OFFENDERS)
def test_in_prior_fiducials_lie_inside_every_declared_bound(name):
    """The corrected set's whole contract, for every model it can be asked for."""
    fid = _fid(name, FIDUCIAL_SET_IN_PRIOR)
    lo, hi, labels = _bounds(name)
    outside = [(labels[i], float(fid[i]), float(lo[i]), float(hi[i]))
               for i in range(len(labels))
               if not (lo[i] <= fid[i] <= hi[i])]
    assert not outside, f"{name}: in_prior_v2 fiducial outside bounds: {outside}"


@pytest.mark.parametrize("name", _OFFENDERS)
def test_in_prior_vectors_are_pinned_numerically(name):
    """The numbers themselves, so a retune of the patch table cannot pass."""
    np.testing.assert_array_equal(_fid(name, FIDUCIAL_SET_IN_PRIOR),
                                  np.asarray(_IN_PRIOR_VECTORS[name], dtype=float))


@pytest.mark.parametrize("name", _OFFENDERS)
def test_the_patch_moves_only_the_violating_parameters(name):
    """Same length, and identical to legacy everywhere legacy was already legal."""
    legacy = _fid(name, FIDUCIAL_SET_LEGACY)
    fixed = _fid(name, FIDUCIAL_SET_IN_PRIOR)
    assert legacy.shape == fixed.shape
    moved = {i: float(legacy[i]) for i in range(legacy.size) if legacy[i] != fixed[i]}
    assert moved == _PATCHED[name], (name, moved)


@pytest.mark.parametrize("name", _NON_OFFENDERS)
def test_non_offenders_are_identical_across_the_two_sets(name):
    """No curated tuning outside the three offenders may depend on the set, and
    bespoke published vectors have no per-slot priors to patch at all."""
    np.testing.assert_array_equal(_fid(name, FIDUCIAL_SET_LEGACY),
                                  _fid(name, FIDUCIAL_SET_IN_PRIOR))


def test_legacy_is_the_default_and_the_set_names_are_closed():
    """The default must not move: every archived fixed-population run and every
    mock built from ``get_fixed_population_params`` is pinned to it."""
    for name in _OFFENDERS + _NON_OFFENDERS:
        np.testing.assert_array_equal(
            np.asarray(get_fixed_population_params(name), dtype=float),
            _fid(name, FIDUCIAL_SET_LEGACY),
        )
    assert set(FIDUCIAL_SETS) == {FIDUCIAL_SET_LEGACY, FIDUCIAL_SET_IN_PRIOR}
    with pytest.raises(ValueError, match="unknown fiducial set"):
        get_fixed_population_params("2powerlaws+peak", fiducials="v3")
