"""An anisotropic sky model needs real injection directions, wherever the events come from.

Since selection stores may lack ``ra`` and ``dec`` (PR #76), an isotropic
binding of such a store carries ``make_gw_event``'s zeros placeholder for its
directions. ``bind_analysis`` refuses the store for an anisotropic model, but
two other routes reached the model with the placeholder and returned a wrong
value without any error: a binding rebuilt with ``dataclasses.replace`` and
``make_host_density_target``. A caller that read the store's positions itself
got a bare ``KeyError: 'ra'``. Mock stores only.
"""

from __future__ import annotations

import pickle
from dataclasses import replace

import h5py
import numpy as np
import pytest

from darksirens import Cosmology, Population, model
from darksirens.gw.types import GWEvent, GWStore, SelectionStore
from darksirens.runtime_binding import bind_analysis

FIT = ("m1det", "q", "dL", "chieff")


def _stores(n_sel=96):
    rng = np.random.default_rng(20261107)
    pe_m1 = np.array([36.0, 37.0, 38.0, 39.0])
    pe = GWStore(
        format_version="fixture",
        path="pe-fixture.h5",
        fit_columns=FIT,
        columns={
            "m1det": pe_m1,
            "m2det": 0.8 * pe_m1,
            "dL": np.array([455.0, 475.0, 495.0, 515.0]),
            "chieff": np.array([-0.02, 0.00, 0.02, 0.01]),
            "ra": np.array([0.20, 1.70, 3.20, 5.80]),
            "dec": np.array([-0.50, -0.10, 0.25, 0.65]),
        },
        attrs={},
        n_events=1,
        nsamp=4,
        prior_wt=np.full(4, 0.25),
        event_names=("fixture",),
    )
    sel_m1 = np.linspace(34.0, 41.0, n_sel)
    # More injections in the north, so a dipole moves the selection integral.
    dec = np.arcsin(np.clip(rng.uniform(-0.6, 1.0, n_sel), -1.0, 1.0))
    selection = SelectionStore(
        format_version="fixture",
        path="selection-fixture.h5",
        fit_columns=FIT,
        columns={
            "m1det": sel_m1,
            "m2det": 0.8 * sel_m1,
            "dL": np.linspace(430.0, 540.0, n_sel),
            "chieff": np.linspace(-0.04, 0.04, n_sel),
            "ra": rng.uniform(0.0, 2.0 * np.pi, n_sel),
            "dec": dec,
        },
        attrs={},
        n_injections=n_sel,
        ndraw=n_sel,
        prior_wt=np.ones(n_sel),
    )
    return pe, selection


def _skyless(injections):
    return replace(
        injections,
        columns={k: v for k, v in injections.columns.items() if k not in ("ra", "dec")},
    )


def _analysis(angular="isotropic"):
    return model(
        cosmology=Cosmology(H0=(60.0, 80.0), Om0=0.3075),
        population=Population("powerlaw+peak", fixed=True),
        angular=angular,
    )


def _theta(analysis):
    # The plan's midpoint, with the dipole/multipole coefficients off zero.
    lower = np.asarray(analysis.parameters.lower, dtype=float)
    upper = np.asarray(analysis.parameters.upper, dtype=float)
    theta = 0.5 * (lower + upper)
    for i, label in enumerate(analysis.parameters.labels):
        if label != "H0":
            theta[i] = lower[i] + 0.7 * (upper[i] - lower[i])
    return theta


ANISOTROPIC = ("dipole", "multipole")


# ---------------------------------------------------------------------------
# A binding rebuilt with dataclasses.replace


@pytest.mark.parametrize("angular", ANISOTROPIC)
@pytest.mark.parametrize("sel_batch_size", [None, 48])
def test_a_rebuilt_binding_refuses_the_zeros_placeholder(angular, sel_batch_size):
    events, injections = _stores()
    aniso = _analysis(angular)
    skyless_iso = bind_analysis(
        _analysis(), events=events, injections=_skyless(injections),
        sel_batch_size=sel_batch_size,
    )
    for name in ("nx", "ny", "nz"):
        np.testing.assert_array_equal(np.asarray(getattr(skyless_iso.gw_selection, name)), 0.0)
    positioned = bind_analysis(
        aniso, events=events, injections=injections, sel_batch_size=sel_batch_size
    )
    assert np.isfinite(float(positioned(_theta(aniso))))

    # Another analysis on the skyless binding: the review's first form.
    with pytest.raises(ValueError, match=r"gw_selection carries no sky direction on 96 of its 96"):
        replace(skyless_iso, analysis=aniso)
    # The skyless binding's selection event on an anisotropic binding: the second.
    with pytest.raises(ValueError, match=rf"angular model {angular!r} reads"):
        replace(positioned, gw_selection=skyless_iso.gw_selection)


def test_a_positioned_rebuild_and_the_isotropic_skyless_binding_still_work():
    events, injections = _stores()
    dipole = _analysis("dipole")
    theta = _theta(dipole)
    direct = bind_analysis(dipole, events=events, injections=injections)
    # The same positions through replace: the value is the directly bound one.
    rebuilt = replace(bind_analysis(_analysis(), events=events, injections=injections),
                      analysis=dipole)
    assert float(rebuilt(theta)) == float(direct(theta))
    # Unpickling runs the same check, and passes for real directions.
    assert float(pickle.loads(pickle.dumps(direct))(theta)) == float(direct(theta))
    # An isotropic analysis on the skyless store binds, value unchanged.
    iso = _analysis()
    with_sky = bind_analysis(iso, events=events, injections=injections)
    without = bind_analysis(iso, events=events, injections=_skyless(injections))
    assert float(without(np.array([67.74]))) == float(with_sky(np.array([67.74])))


def test_a_float32_binding_rebuilt_as_anisotropic_keeps_its_own_refusal():
    events, injections = _stores()
    bound = bind_analysis(_analysis(), events=events, injections=injections,
                          compute_dtype="float32")
    with pytest.raises(ValueError, match="float32") as excinfo:
        replace(bound, analysis=_analysis("dipole"))
    assert "carries no sky direction" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# make_host_density_target


class _FlatRedshiftModel:
    def parameter_spec(self):
        from darksirens.analysis import ParameterPlan

        return ParameterPlan(
            labels=("eta",), lower=(0.0,), upper=(1.0,),
            prior_kinds=(("uniform", None, None),), joint_constraints=(),
        )

    def log_density(self, z, pixel, cosmology, parameters, state):
        return 0.0 * z

    def log_auxiliary_likelihood(self, parameters, state):
        return 0.0


def _host_target(base, bound, *, gw_selection=None):
    from darksirens.likelihood.host_density import make_host_density_target

    return make_host_density_target(
        base,
        redshift_model=_FlatRedshiftModel(),
        gw_pe=bound.gw_pe,
        gw_selection=bound.gw_selection if gw_selection is None else gw_selection,
        pe_state=None,
        selection_state=None,
        n_events=bound.n_events,
        nsamp=bound.nsamp,
        n_draw=bound.n_draw,
    )


@pytest.mark.parametrize("angular", ANISOTROPIC)
def test_a_host_density_target_refuses_events_without_directions(angular):
    events, injections = _stores()
    aniso = _analysis(angular)
    skyless_iso = bind_analysis(_analysis(), events=events, injections=_skyless(injections))
    positioned = bind_analysis(aniso, events=events, injections=injections)

    # The natural route: an isotropic binding's events fed to an anisotropic base.
    with pytest.raises(ValueError, match="make_host_density_target's gw_selection carries no sky"):
        _host_target(aniso, skyless_iso)
    # Directions zeroed by hand, and the GWEvent default (None).
    sel = positioned.gw_selection
    zeroed = sel._replace(nx=0.0 * sel.nx, ny=0.0 * sel.ny, nz=0.0 * sel.nz)
    with pytest.raises(ValueError, match="carries no sky direction"):
        _host_target(aniso, positioned, gw_selection=zeroed)
    with pytest.raises(ValueError, match="carries no sky direction"):
        _host_target(aniso, positioned, gw_selection=sel._replace(nx=None, ny=None, nz=None))

    # Real directions build and evaluate.
    target = _host_target(aniso, positioned)
    assert np.isfinite(float(target.log_likelihood(np.append(_theta(aniso), 0.5))))


def test_an_isotropic_host_density_target_still_takes_skyless_events():
    events, injections = _stores()
    iso = _analysis()
    skyless_iso = bind_analysis(iso, events=events, injections=_skyless(injections))
    target = _host_target(iso, skyless_iso)
    assert np.isfinite(float(target.log_likelihood(np.array([67.74, 0.5]))))


def test_padding_rows_are_not_read():
    from darksirens.gw.runtime import make_gw_event, pad_gw_event_to_multiple, require_unit_directions

    ones = np.ones(5)
    event = make_gw_event(30.0 * ones, 25.0 * ones, 500.0 * ones, 0.0 * ones, ones,
                          np.zeros(5, dtype=int), nx=ones, ny=0.0 * ones, nz=0.0 * ones)
    padded, pad = pad_gw_event_to_multiple(event, 8)
    assert pad == 3
    require_unit_directions(padded, what="padded", angular_model="dipole")
    # float32-rounded directions pass.
    angle = np.linspace(0.1, 3.0, 5)
    cast = make_gw_event(30.0 * ones, 25.0 * ones, 500.0 * ones, 0.0 * ones, ones,
                         np.zeros(5, dtype=int),
                         nx=np.cos(angle).astype(np.float32), ny=np.sin(angle).astype(np.float32),
                         nz=0.0 * ones)
    require_unit_directions(cast, what="float32", angular_model="dipole")


# ---------------------------------------------------------------------------
# Callers that read the positions themselves


def test_reading_the_positions_of_a_skyless_store_names_the_file_and_reason():
    from darksirens.gw import MissingSkyPositionsError, require_selection_sky

    _, injections = _stores()
    skyless = _skyless(injections)
    with pytest.raises(KeyError) as excinfo:
        skyless.columns["ra"]
    error = excinfo.value
    assert isinstance(error, MissingSkyPositionsError)
    assert error.args[0] == "ra"
    message = str(error)
    assert "selection-fixture.h5" in message
    assert "missing datasets: ra, dec" in message
    assert "use an injection set with ra and dec" in message
    # Everything else is the plain dict it was.
    assert "ra" not in skyless.columns and skyless.columns.get("dec") is None
    assert isinstance(skyless.columns, dict)
    with pytest.raises(KeyError) as other:
        skyless.columns["no_such_column"]
    assert not isinstance(other.value, MissingSkyPositionsError)
    np.testing.assert_array_equal(skyless.columns["dL"], injections.columns["dL"])
    # A store with positions is untouched.
    assert type(injections.columns) is dict
    # Pickled, the store keeps the message.
    again = pickle.loads(pickle.dumps(skyless))
    with pytest.raises(MissingSkyPositionsError, match="selection-fixture.h5"):
        again.columns["dec"]
    with pytest.raises(RuntimeError, match="carries no injection sky positions"):
        require_selection_sky(skyless, "a test")


def test_a_loaded_skyless_store_raises_the_clear_error(tmp_path):
    from test_gw_store_contract import write_sel

    from darksirens.gw import MissingSkyPositionsError, load_injections

    path = tmp_path / "skyless.h5"
    write_sel(path, basis="chieff_reference")
    with h5py.File(path, "a") as f:
        del f["ra"]
        del f["dec"]
        f.attrs["sky_position_available"] = np.array([False])
    store = load_injections(path)
    with pytest.raises(MissingSkyPositionsError, match="skyless.h5"):
        store.columns["ra"]


def test_the_runtime_event_builder_refuses_a_skyless_store_clearly():
    from darksirens.runtime_binding import _make_runtime_event

    _, injections = _stores()
    skyless = _skyless(injections)
    rows = np.zeros(skyless.n_injections, dtype=np.int32)
    with pytest.raises(RuntimeError, match=r"carries no injection sky positions.*sky=False"):
        _make_runtime_event(skyless, rows, FIT)
    event = _make_runtime_event(skyless, rows, FIT, sky=False)
    assert isinstance(event, GWEvent)
    np.testing.assert_array_equal(np.asarray(event.nx), 0.0)
