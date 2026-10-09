"""``ds.model(..., completeness="selection", selection=TabulatedSelection(...))``: a tabulated completeness.

The completeness of the catalog is given as a table in redshift (nodes ``z``,
values ``completeness`` in [0, 1], linear between nodes) instead of a
magnitude-selection curve. It enters where ``C_sel(z)`` does, so these tests
compare it with the modes it must reduce to: a densely tabulated Schechter
curve against the Schechter family, a table of ones against the complete
catalog, a non-uniform table against a uniform one of the same piecewise
linear function, and a scaled table against a row fraction. They also pin
the payload, the refusals, the plan (no nuisance to sample), the record in
``ParameterPlan.catalog_model`` and the evaluation through ``ds.model`` with
the default and the field sky weighting, under jit, vmap and grad.

The Schechter curve of this code does not depend on H0: the ``+5 log10 h`` of
its h-scaled ``Mstar_hat`` cancels the H0 of the distance modulus
(:mod:`darksirens.selection.catalog`; checked here to rounding). It does
depend on ``Om0``, ``w0`` and ``wa`` through the shape of the distance
modulus, so the tables are built at the fixtures' fixed ``Om0`` and compared
at every sampled H0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens as ds
from darksirens import Population
from darksirens.catalog.types import CatalogParameters
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.inference.run_fingerprint import parameter_plan_semantic
from darksirens.runtime_binding import bind_analysis
from darksirens.selection.catalog import (
    SELECTION_RUNTIME_FORMAT,
    SchechterMagnitudeSelection,
    TabulatedSelection,
    c_sel_tabulated,
    selection_completion_curves,
    selection_curve,
    selection_from_mapping,
    selection_record,
    selection_to_mapping,
    validate_catalog_selection,
    validate_selection_coverage,
)

from _catalog_model_fixtures import (
    COSMOLOGY,
    FIXED_POPULATION,
    MODEL,
    NPIX,
    catalog_store,
    gw_stores,
    theta_of,
)

jax.config.update("jax_enable_x64", True)

SCHECHTER = SchechterMagnitudeSelection(
    m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0
)
POPULATION = Population(MODEL, fixed=FIXED_POPULATION)
STORE = catalog_store()
DEPTH = 0.22
STORE_DEPTH = catalog_store(z_depth=DEPTH)
EVENTS, INJECTIONS = gw_stores()
CAP = 1.0e3
ZGRID = np.asarray(zgrid, dtype=np.float64)
ZTOP = float(ZGRID[-1])
OM0 = 0.3075  # the fixtures' fixed Om0


def _schechter_values(nodes, H0=70.0):
    """The Schechter fixture's clipped curve at ``nodes`` (float64 NumPy)."""
    curve = selection_curve(jnp.asarray(nodes), CosmologyParameters(H0, OM0), SCHECHTER)
    return np.asarray(jnp.clip(curve, 0.0, 1.0), dtype=np.float64)


def _schechter_table(nodes, H0=70.0):
    return TabulatedSelection(z=nodes, completeness=_schechter_values(nodes, H0))


def _analysis(selection, store=STORE, **kwargs):
    return ds.model(
        cosmology=COSMOLOGY,
        population=POPULATION,
        catalog=store,
        completeness="selection",
        selection=selection,
        **kwargs,
    )


def _bind(analysis, **kwargs):
    kwargs.setdefault("max_likelihood_variance", CAP)
    return bind_analysis(analysis, events=EVENTS, injections=INJECTIONS, **kwargs)


def _points(analysis, n=4, seed=7, **values):
    values.setdefault("log10n0", -2.0)
    return [theta_of(analysis, values, seed=seed + i) for i in range(n)]


def _close(a, b, rtol):
    a, b = float(a), float(b)
    assert np.isfinite(a) and np.isfinite(b), (a, b)
    assert abs(a - b) <= rtol * max(1.0, abs(b)), (a, b, a - b)


def _worst(a, b, thetas):
    return max(abs(float(a(theta)) - float(b(theta))) for theta in thetas)


# ---------------------------------------------------------------------------
# The payload, the curve and the refusals


def test_payload_round_trip_and_record():
    table = TabulatedSelection(z=np.array([0.0, 0.1, 0.4, 5.0]), completeness=[1.0, 0.8, 0.1, 0.0])
    payload = selection_to_mapping(table)
    assert payload == {
        "format_version": SELECTION_RUNTIME_FORMAT,
        "family": "tabulated",
        "z": [0.0, 0.1, 0.4, 5.0],
        "completeness": [1.0, 0.8, 0.1, 0.0],
    }
    assert selection_from_mapping(payload) == table
    assert hash(selection_from_mapping(payload)) == hash(table)
    record = selection_record(table)
    assert record["family"] == "tabulated" and record["n_nodes"] == 4
    assert (record["z_min"], record["z_max"]) == (0.0, 5.0)
    assert len(record["table_sha256"]) == 64
    moved = TabulatedSelection(z=table.z, completeness=[1.0, 0.8, 0.1000001, 0.0])
    assert selection_record(moved)["table_sha256"] != record["table_sha256"]
    # A magnitude-selection model's record is its payload, as before.
    assert selection_record(SCHECHTER) == selection_to_mapping(SCHECHTER)


def test_curve_is_linear_between_nodes_clipped_and_zero_outside():
    nodes, values = [0.0, 0.1, 0.4], [1.0, 0.5, 0.2]
    z = jnp.asarray([0.0, 0.05, 0.1, 0.25, 0.4, 0.4000001, 3.0, -0.01])
    got = np.asarray(c_sel_tabulated(z, nodes, values))
    np.testing.assert_allclose(got[:5], [1.0, 0.75, 0.5, 0.35, 0.2], rtol=0, atol=1e-15)
    # Outside the nodes the end values are not held.
    assert np.all(got[5:] == 0.0)
    # No cosmological parameter enters the curve.
    table = TabulatedSelection(z=nodes, completeness=values)
    a = selection_curve(zgrid, CosmologyParameters(50.0, 0.2), table)
    b = selection_curve(zgrid, CosmologyParameters(90.0, 0.4, -0.8, 0.3), table)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_schechter_curve_does_not_depend_on_H0():
    """The premise of the comparisons below, at fixed ``Om0``."""
    curves = [_schechter_values(ZGRID, H0) for H0 in (50.0, 70.0, 90.0)]
    assert np.all(np.isfinite(curves[1]))
    assert max(np.max(np.abs(c - curves[1])) for c in curves) < 1e-12
    # The shape of the distance modulus does enter it.
    other = selection_curve(zgrid, CosmologyParameters(70.0, 0.25), SCHECHTER)
    assert np.max(np.abs(np.asarray(other) - curves[1])) > 1e-4


@pytest.mark.parametrize(
    "z, completeness, match",
    [
        ([0.0, 0.1, 5.0], [1.0, np.nan, 0.0], "must be finite"),
        ([0.0, np.inf], [1.0, 0.0], "must be finite"),
        ([0.0, 0.1, 5.0], [1.0, 1.5, 0.0], r"must lie in \[0, 1\]"),
        ([0.0, 0.1, 5.0], [1.0, -1e-9, 0.0], r"must lie in \[0, 1\]"),
        ([0.0, 0.2, 0.2, 5.0], [1.0, 0.5, 0.4, 0.0], "strictly increasing"),
        ([0.0, 0.3, 0.2, 5.0], [1.0, 0.5, 0.4, 0.0], "strictly increasing"),
        ([0.0], [1.0], "at least 2 nodes"),
        ([0.0, 0.1, 5.0], [1.0, 0.0], "equal lengths"),
        ([[0.0, 5.0]], [[1.0, 0.0]], "one-dimensional"),
    ],
    ids=("nan", "inf-node", "above-one", "below-zero", "repeated-node", "decreasing",
         "one-node", "unequal", "two-dimensional"),
)
def test_structural_refusals_from_the_payload_the_object_and_ds_model(z, completeness, match):
    payload = {
        "format_version": SELECTION_RUNTIME_FORMAT, "family": "tabulated",
        "z": z, "completeness": completeness,
    }
    with pytest.raises(ValueError, match=match):
        selection_from_mapping(payload)
    with pytest.raises(ValueError, match=match):
        validate_catalog_selection(TabulatedSelection(z=z, completeness=completeness))
    with pytest.raises(ValueError, match=match):
        _analysis(payload)


def test_payload_key_refusals():
    base = {"format_version": SELECTION_RUNTIME_FORMAT, "family": "tabulated"}
    with pytest.raises(ValueError, match=r"missing=\['completeness'\]"):
        selection_from_mapping({**base, "z": [0.0, 5.0]})
    with pytest.raises(ValueError, match=r"extra=\['m_lim'\]"):
        selection_from_mapping({**base, "z": [0.0, 5.0], "completeness": [1.0, 0.0], "m_lim": 19.0})
    with pytest.raises(ValueError, match="'gaussian', 'schechter' or 'tabulated'"):
        selection_from_mapping({"format_version": SELECTION_RUNTIME_FORMAT, "family": "table"})


def test_coverage_refusals():
    """No extrapolation: the table must cover every redshift the model reads.

    The curve is read on the model redshift grid, from its lowest node (0) up
    to the catalog's ``z_depth``, or up to the top of the grid for a catalog
    without one (where the magnitude-selection curves are read as well).
    """
    short = TabulatedSelection(z=[0.0, 0.1, 0.2], completeness=[1.0, 0.6, 0.3])
    with pytest.raises(ValueError, match=r"must reach the catalog's z_depth \(0.22\)"):
        _analysis(short, store=STORE_DEPTH)
    with pytest.raises(ValueError, match="must reach the top of the model redshift grid"):
        _analysis(short)
    late = TabulatedSelection(z=[1e-4, 0.1, ZTOP], completeness=[1.0, 0.6, 0.3])
    with pytest.raises(ValueError, match="must start at or below z = 0"):
        _analysis(late)
    with pytest.raises(ValueError, match="never extrapolated"):
        _analysis(selection_to_mapping(late), store=STORE_DEPTH)
    # Exactly covering tables bind: [0, z_depth] with a depth, [0, top] without.
    exact = TabulatedSelection(z=[0.0, 0.1, DEPTH], completeness=[1.0, 0.6, 0.3])
    assert _analysis(exact, store=STORE_DEPTH).redshift.selection == exact
    assert validate_selection_coverage(exact, DEPTH) is exact
    full = TabulatedSelection(z=[0.0, ZTOP], completeness=[1.0, 0.0])
    assert _analysis(full).redshift.selection == full
    # In a mixture each catalog's table is checked against its own depth.
    with pytest.raises(ValueError, match="must reach the top of the model redshift grid"):
        ds.model(
            cosmology=COSMOLOGY, population=POPULATION, catalog=[STORE_DEPTH, STORE],
            catalog_sky_weighting="field", completeness="selection", selection=[exact, exact],
        )
    # The magnitude-selection families are defined everywhere.
    assert validate_selection_coverage(SCHECHTER, None) is SCHECHTER


# ---------------------------------------------------------------------------
# The plan: a fixed family with no nuisance


def test_plan_has_no_selection_nuisance_and_records_the_table():
    table = _schechter_table(ZGRID)
    analysis = _analysis(table)
    incomplete = ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=STORE)
    plan = analysis.parameters
    assert analysis.redshift.selection == table
    assert plan.labels == incomplete.parameters.labels == ("H0", "log10n0", "delta", "sigma_kde")
    assert plan.n_catalog == 3
    assert plan.catalog_model_settings == {
        "completeness": "selection",
        "selection": selection_record(table),
    }
    assert _analysis(selection_to_mapping(table)).parameters == plan
    with pytest.raises(ValueError, match="unknown survey parameter"):
        _analysis(table, survey_priors={"alpha": (-1.5, -0.5)})
    decoded = ds.decode_parameters(analysis, theta_of(analysis))
    assert decoded.catalog.selection is analysis.redshift.selection


def test_fingerprint_distinguishes_two_tables():
    values = _schechter_values(ZGRID)
    a = parameter_plan_semantic(_analysis(TabulatedSelection(ZGRID, values)).parameters)
    b = parameter_plan_semantic(_analysis(TabulatedSelection(ZGRID, 0.999 * values)).parameters)
    c = parameter_plan_semantic(_analysis(SCHECHTER).parameters)
    assert a["catalog_model"]["selection"]["family"] == "tabulated"
    assert a != b and a != c
    assert a["catalog_model"]["selection"]["table_sha256"] != b["catalog_model"]["selection"]["table_sha256"]
    # Existing modes: the record is unchanged.
    assert c["catalog_model"]["selection"] == selection_to_mapping(SCHECHTER)
    default = ds.model(cosmology=COSMOLOGY, population=POPULATION, catalog=STORE)
    assert "catalog_model" not in parameter_plan_semantic(default.parameters)


# ---------------------------------------------------------------------------
# 1. A tabulated Schechter curve against the Schechter family


@pytest.mark.parametrize("store", (STORE, STORE_DEPTH), ids=("no-depth", "depth"))
def test_schechter_tabulated_on_the_model_grid_is_the_schechter_likelihood(store):
    """Nodes on the model grid: no interpolation error is left.

    The table is built at H0 = 70 and compared at H0 drawn over [50, 90]; the
    difference is the rounding of the Schechter curve's H0 cancellation.
    """
    tabulated = _bind(_analysis(_schechter_table(ZGRID), store=store))
    schechter = _bind(_analysis(SCHECHTER, store=store))
    for theta in _points(tabulated.analysis):
        _close(tabulated(theta), schechter(theta), 1e-12)


def test_schechter_tabulated_converges_with_node_spacing():
    """Uniform tables on [0, 0.25] (catalog depth 0.22): second-order convergence.

    Linear interpolation has an error proportional to the square of the node
    spacing, so a four-fold refinement reduces the log-likelihood difference
    about sixteen-fold. Measured here, the largest difference over the four
    points for 9, 33, 129, 513 and 2049 nodes is 2.75e-2, 2.10e-3, 8.80e-5,
    1.06e-5 and 5.6e-7: factors of 13, 24, 8 and 19 (3.5e-8 at 8193 nodes,
    3e-14 with the nodes on the model grid). The bounds are those values with
    a margin of about 2 (the factor) and 3.5 (the finest table).
    """
    schechter = _bind(_analysis(SCHECHTER, store=STORE_DEPTH))
    thetas = _points(schechter.analysis)
    errors = []
    for n in (9, 33, 129, 513, 2049):
        table = _schechter_table(np.linspace(0.0, 0.25, n))
        errors.append(_worst(_bind(_analysis(table, store=STORE_DEPTH)), schechter, thetas))
    print("tabulated Schechter |dlogL| for 9, 33, 129, 513, 2049 nodes:", errors)
    assert errors[0] > 1e-3
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse / 4.0, errors
    assert errors[0] / errors[-1] > 1.0e4, errors
    assert errors[-1] < 2e-6, errors


# ---------------------------------------------------------------------------
# 2. Completeness identically 1 against the complete catalog


@pytest.mark.parametrize("weighting", ("conditional", "field"))
def test_completeness_one_is_the_complete_catalog_without_a_depth(weighting):
    """With ``C = 1`` at every redshift and no ``z_depth`` the whole likelihood is the complete catalog's.

    The selection mode's row density is ``[N_obs,p p_cat(z|p) + dN_miss,p(z)] /
    (N_obs,p + N_miss,p)`` and ``dN_miss = (1 - C) dN_exp``. At ``C = 1``,
    ``dN_miss`` and ``N_miss`` are exactly zero, so the numerator is the
    complete catalog's ``N_obs,p p_cat(z|p)``. The conditional weighting
    divides it by ``N_obs,p``, leaving ``p_cat(z|p)`` (and no hosts in a row
    without galaxies), which is ``completeness="complete"`` with its default
    ``empty_policy="zero"``. The field weighting keeps ``N_obs,p p_cat``, the
    complete catalog's field numerator, and with one catalog the common
    normaliser is not evaluated in either mode. ``log10n0`` multiplies a
    density that is zero, so the value does not depend on it. The equality is
    to rounding (``log N + log p - log N``), not bitwise.

    It does not hold for a catalog with a ``z_depth``: above the depth the
    selection mode takes every host as missing whatever the curve says, and
    the complete catalog has no depth.
    """
    ones = TabulatedSelection(z=[0.0, 1.0, ZTOP], completeness=[1.0, 1.0, 1.0])
    kwargs = dict(catalog_sky_weighting=weighting)
    selection = _bind(_analysis(ones, **kwargs))
    complete = _bind(ds.model(
        cosmology=COSMOLOGY, population=POPULATION, catalog=STORE, completeness="complete", **kwargs
    ))
    assert complete.analysis.parameters.labels == ("H0", "delta", "sigma_kde")
    for theta in _points(complete.analysis, n=4):
        want = float(complete(theta))
        for log10n0 in (-3.0, -1.0):
            _close(selection(np.insert(theta, 1, log10n0)), want, 1e-12)


def test_completeness_one_below_a_depth_is_not_the_complete_catalog():
    ones = TabulatedSelection(z=[0.0, DEPTH], completeness=[1.0, 1.0])
    selection = _bind(_analysis(ones, store=STORE_DEPTH))
    complete = _bind(ds.model(
        cosmology=COSMOLOGY, population=POPULATION, catalog=STORE_DEPTH, completeness="complete"
    ))
    theta = _points(complete.analysis, n=1)[0]
    assert abs(float(selection(np.insert(theta, 1, -2.0))) - float(complete(theta))) > 1e-3


# ---------------------------------------------------------------------------
# 3. Non-uniform nodes


def test_nonuniform_nodes_are_the_uniform_table_of_the_same_function():
    knots = np.array([0.0, 0.03, 0.08, 0.11, 0.19, 0.4, 1.7, ZTOP])
    values = np.array([1.0, 0.95, 0.7, 0.62, 0.2, 0.05, 0.0, 0.0])
    uniform_z = np.linspace(0.0, ZTOP, 501)  # spacing 0.01: every knot is a node
    assert np.all(np.min(np.abs(uniform_z[:, None] - knots[None, :]), axis=0) < 1e-12)
    uniform = TabulatedSelection(z=uniform_z, completeness=np.interp(uniform_z, knots, values))
    sparse = _bind(_analysis(TabulatedSelection(z=knots, completeness=values)))
    dense = _bind(_analysis(uniform))
    for theta in _points(sparse.analysis):
        _close(sparse(theta), dense(theta), 1e-11)
    # The spacing matters where it changes the function.
    coarse = TabulatedSelection(z=knots[[0, 4, 7]], completeness=values[[0, 4, 7]])
    theta = _points(sparse.analysis, n=1)[0]
    assert abs(float(_bind(_analysis(coarse))(theta)) - float(sparse(theta))) > 1e-4


# ---------------------------------------------------------------------------
# 5. The row fraction


def test_row_fraction_scales_the_table_row_by_row():
    """``C(z, row) = row_fraction[row] * C(z)``, as for the magnitude-selection families."""
    values = _schechter_values(ZGRID)
    table = TabulatedSelection(ZGRID, values)
    fraction = np.random.default_rng(4).uniform(0.0, 1.0, NPIX)
    fraction[::7] = 0.0
    schechter = _bind(_analysis(SCHECHTER, store=STORE_DEPTH, row_fraction=fraction))
    analysis = _analysis(table, store=STORE_DEPTH, row_fraction=fraction)
    assert analysis.parameters.catalog_model_settings["row_fraction_sha256"]
    tabulated = _bind(analysis)
    plain = _bind(_analysis(table, store=STORE_DEPTH))
    ones = _bind(_analysis(table, store=STORE_DEPTH, row_fraction=np.ones(NPIX)))
    # A constant fraction on every row is the table scaled by it.
    half = _bind(_analysis(table, store=STORE_DEPTH, row_fraction=np.full(NPIX, 0.5)))
    scaled = _bind(_analysis(TabulatedSelection(ZGRID, 0.5 * values), store=STORE_DEPTH))
    for theta in _points(analysis):
        _close(tabulated(theta), schechter(theta), 1e-11)
        _close(ones(theta), plain(theta), 1e-13)
        _close(half(theta), scaled(theta), 1e-13)
        assert abs(float(tabulated(theta)) - float(plain(theta))) > 1e-6


# ---------------------------------------------------------------------------
# 6. Through ds.model: the curves, the field weighting, jit / vmap / grad


def test_missing_host_curves_follow_the_table_and_the_depth():
    table = TabulatedSelection(z=[0.0, 0.1, 0.25], completeness=[1.0, 0.4, 0.2])
    params = CatalogParameters(n0=10.0**-2.0, delta=0.3, sigma_kde=0.01, z_depth=DEPTH)
    curves = selection_completion_curves(
        CosmologyParameters(70.0, OM0), params, STORE_DEPTH.catalog, table
    )
    C = np.asarray(curves.C[0])
    np.testing.assert_allclose(
        C, np.where(ZGRID <= 0.25, np.interp(ZGRID, table.z, table.completeness), 0.0), atol=1e-15
    )
    # Above the depth every host is missing, as for the other families.
    assert np.all(np.asarray(curves.C_eff[0])[ZGRID > DEPTH] == 0.0)


@pytest.mark.parametrize("store", (STORE, STORE_DEPTH), ids=("no-depth", "depth"))
def test_field_and_default_weighting_with_one_catalog(store):
    """Both weightings through ``ds.model``, each against the Schechter family.

    The two weightings differ in how a row is normalised: the default divides
    each row by its own ``N_obs + N_miss``, the field weighting by one total.
    At ``log10n0 = -2`` the missing hosts are about 1e9 times the catalogued
    galaxies in every row of this fixture, the row normalisers are equal to
    that precision, and the two weightings agree to about 1e-10 for every
    family (measured: Gaussian 3e-10 and 6e-10, Schechter and its table 1e-10
    and 3e-10, without and with a depth). At ``log10n0 = -10`` the two terms
    are comparable and the weightings differ by 0.038 and 0.014, the same for
    the Schechter family and its table.
    """
    table = _schechter_table(ZGRID)
    bound = {
        (name, weighting): _bind(_analysis(sel, store=store, catalog_sky_weighting=weighting))
        for name, sel in (("table", table), ("schechter", SCHECHTER))
        for weighting in ("conditional", "field")
    }
    analysis = bound["table", "field"].analysis
    for log10n0 in (-2.0, -10.0):
        for theta in _points(analysis, n=2, log10n0=log10n0):
            for weighting in ("conditional", "field"):
                _close(bound["table", weighting](theta), bound["schechter", weighting](theta), 1e-12)
    dense = _points(analysis, n=1, log10n0=-2.0)[0]
    sparse = _points(analysis, n=1, log10n0=-10.0)[0]
    for name in ("table", "schechter"):
        _close(bound[name, "field"](dense), bound[name, "conditional"](dense), 1e-8)
        assert abs(float(bound[name, "field"](sparse)) - float(bound[name, "conditional"](sparse))) > 5e-3


def test_field_mixture_takes_one_table_per_catalog():
    """Two catalogs, each with its own table, for both forms of the normaliser."""
    other = catalog_store(23, nside=4, empty=(5, 40, 41, 100), n_max=4, z_depth=DEPTH)
    sel_b = SchechterMagnitudeSelection(m_lim=17.5, Mstar_hat=-21.0, alpha=-0.8, M_faint_offset=5.0)
    curve_b = selection_curve(zgrid, CosmologyParameters(70.0, OM0), sel_b)
    table_a = _schechter_table(ZGRID)
    table_b = TabulatedSelection(ZGRID, np.asarray(jnp.clip(curve_b, 0.0, 1.0)))

    def model(selection, **kwargs):
        return ds.model(
            cosmology=COSMOLOGY, population=POPULATION, catalog=[STORE_DEPTH, other],
            catalog_sky_weighting="field", completeness="selection", selection=selection, **kwargs,
        )

    analysis = model([table_a, table_b])
    assert analysis.parameters.labels == (
        "H0", "log10n0", "delta", "sigma_kde", "log10n0_c2", "delta_c2", "sigma_kde_c2", "fcat_2"
    )
    records = analysis.parameters.catalog_model_settings["catalogs"]
    assert [r["selection"]["family"] for r in records] == ["tabulated", "tabulated"]
    assert records[0]["selection"]["table_sha256"] != records[1]["selection"]["table_sha256"]
    tabulated = _bind(analysis)
    schechter = _bind(model([SCHECHTER, sel_b]))
    direct = _bind(model([table_a, table_b], field_normalizer="direct"))
    mixed = _bind(model([table_a, sel_b]))
    for theta in _points(analysis, n=2, log10n0_c2=-2.4):
        _close(tabulated(theta), schechter(theta), 1e-11)
        _close(direct(theta), tabulated(theta), 1e-11)
        _close(mixed(theta), schechter(theta), 1e-11)


@pytest.mark.parametrize("weighting", ("conditional", "field"))
def test_jit_vmap_and_grad(weighting):
    analysis = _analysis(
        _schechter_table(np.linspace(0.0, 0.25, 65)), store=STORE_DEPTH,
        catalog_sky_weighting=weighting,
    )
    bound = _bind(analysis)
    fn = bound.as_pytree_callable()
    thetas = jnp.asarray(np.stack(_points(analysis, n=3)))
    batch = np.asarray(jax.jit(jax.vmap(lambda f, t: f(t), in_axes=(None, 0)))(fn, thetas))
    assert np.all(np.isfinite(batch))
    for value, theta in zip(batch, np.asarray(thetas)):
        _close(value, bound(theta), 1e-12)
    grad = np.asarray(jax.grad(lambda t: fn(t))(thetas[0]))
    assert np.all(np.isfinite(grad)) and np.all(grad != 0.0)


def test_float32_weights_stay_close():
    analysis = _analysis(_schechter_table(ZGRID), store=STORE_DEPTH)
    ref = _bind(analysis)
    low = _bind(analysis, compute_dtype="float32")
    for theta in _points(analysis, n=3):
        _close(low(theta), ref(theta), 1e-5)
