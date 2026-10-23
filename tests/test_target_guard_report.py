"""``ds.infer`` runs core's guard report for a target that exposes what it reads.

An :class:`~darksirens.InferenceTarget` that exposes a binding's
``diagnostics(theta)``, ``n_events``, ``max_likelihood_variance`` and
``selection_neff_soft_guard`` gets core's report, called unchanged; a target
whose records carry a member axis gets its own ``guard_report`` method's
report, or a warning; a target that exposes none of this runs exactly as
before. A target that records the sampler it was built for warns when it runs
under another with the hard guard.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import pytest

import darksirens as ds
from darksirens.inference import public
from darksirens.inference.guard_report import (
    GUARD_REPORT_DRAWS,
    TARGET_GUARD_REPORT_ATTRIBUTES,
    guard_report,
)
from darksirens.inference.prior import make_prior_transform
from darksirens.inference.target import InferenceTarget

Record = namedtuple("Record", "log_likelihood event_mc_variance n_eff")


def _plan(lower=20.0, upper=140.0):
    return ds.ParameterPlan(
        labels=("H0", "x"),
        lower=(lower, 0.0),
        upper=(upper, 1.0),
        prior_kinds=(("uniform", None, None), ("uniform", None, None)),
        joint_constraints=(),
    )


def _transform(plan):
    return make_prior_transform(
        plan.lower, plan.upper, prior_kinds=plan.prior_kinds,
        joint_constraints=plan.joint_constraints,
    )


def _record(h0):
    """Guards that fire below H0 = 50 (PE variance) and above H0 = 120 (selection)."""
    pe = 2.0 if h0 < 50.0 else 0.0
    n_eff = 1.0 if h0 > 120.0 else 1.0e5
    value = -np.inf if (h0 < 50.0 or h0 > 120.0) else -1.0
    return Record(value, np.full(4, pe / 4), n_eff)


def _log_likelihood(theta):
    return -0.5 * ((theta[0] - 70.0) / 10.0) ** 2


@dataclass(frozen=True)
class DiagnosedTarget(InferenceTarget):
    """A companion target exposing what core's report reads from a binding."""

    n_events: int = field(default=4, kw_only=True)
    max_likelihood_variance: float = field(default=1.0, kw_only=True)
    selection_neff_soft_guard: bool = field(default=False, kw_only=True)
    built_for_sampler: str | None = field(default=None, kw_only=True)
    calls: list = field(default_factory=list, kw_only=True, compare=False)

    def diagnostics(self, theta):
        self.calls.append(np.asarray(theta, dtype=float))
        return _record(float(theta[0]))


@dataclass(frozen=True)
class MemberTarget(DiagnosedTarget):
    """Records with a leading member axis (a likelihood averaged over members)."""

    n_members: int = field(default=3, kw_only=True)

    def diagnostics(self, theta):
        self.calls.append(np.asarray(theta, dtype=float))
        one = _record(float(theta[0]))
        m = self.n_members
        return Record(np.full(m, one.log_likelihood),
                      np.tile(one.event_mc_variance, (m, 1)), np.full(m, one.n_eff))


@dataclass(frozen=True)
class ReportingMemberTarget(MemberTarget):
    reports: list = field(default_factory=list, kw_only=True, compare=False)

    def guard_report(self, prior_transform, labels, *, n_draws, seed):
        self.reports.append((prior_transform, tuple(labels), n_draws, seed))
        return {"n_draws": n_draws, "seed": seed, "members": {"n_members": self.n_members}}


@dataclass(frozen=True)
class PartlyDiagnosedTarget(InferenceTarget):
    """Diagnostics without the rest: not what the report reads."""

    selection_neff_soft_guard: bool = field(default=False, kw_only=True)

    def diagnostics(self, theta):
        raise AssertionError("ds.infer must not evaluate this target's diagnostics")


class _Sampler:
    """Stands in for ``run_sampler``: records each call, returns a fixed result."""

    def __init__(self, result=None):
        self.result = {"logZ": -1.5, "samples": np.zeros((3, 2))} if result is None else result
        self.calls = []

    def __call__(self, method, likelihood, transform, labels, lower, upper, opts, **kwargs):
        self.calls.append(dict(method=method, likelihood=likelihood, transform=transform,
                               labels=labels, lower=lower, upper=upper, opts=opts, **kwargs))
        return dict(self.result)


@pytest.fixture
def sampler(monkeypatch):
    fake = _Sampler()
    monkeypatch.setattr("darksirens.inference.sampling.run_sampler", fake)
    return fake


# ---------------------------------------------------------------------------
# The capability marker


def test_the_capability_marker_names_what_the_report_reads():
    import darksirens.inference.guard_report as module

    assert TARGET_GUARD_REPORT_ATTRIBUTES == (
        "diagnostics", "n_events", "max_likelihood_variance", "selection_neff_soft_guard",
    )
    assert "TARGET_GUARD_REPORT_ATTRIBUTES" in module.__all__


# ---------------------------------------------------------------------------
# A target with diagnostics


def test_a_target_with_diagnostics_gets_cores_report_unchanged(sampler):
    target = DiagnosedTarget(_log_likelihood, _plan())
    with pytest.warns(UserWarning, match="pe_mc_variance") as caught:
        result = ds.infer(target, sampler="dynesty", guard_report=64, seed=5)
    # attributed to the caller of ds.infer, as for an ordinary analysis
    walled = [w for w in caught if "likelihood guard fired" in str(w.message)]
    assert [w.filename for w in walled] == [__file__]
    assert len(target.calls) == 64

    with pytest.warns(UserWarning, match="pe_mc_variance"):
        direct = guard_report(target, _transform(target.parameters), target.parameters.labels,
                              n_draws=64, seed=5)
    assert result["guard_report"] == direct
    assert set(direct["fired"]) == {"pe_mc_variance", "selection_neff"}
    assert result["logZ"] == -1.5

    # The sampler sees the target's likelihood and the plan's transform, once.
    (call,) = sampler.calls
    assert call["likelihood"] is target.log_likelihood
    assert call["labels"] == ("H0", "x")
    u = np.array([0.25, 0.75])
    np.testing.assert_array_equal(np.asarray(call["transform"](u)),
                                  np.asarray(_transform(target.parameters)(u)))
    assert not hasattr(call["opts"], "guard_report")


def test_the_default_report_uses_32_draws_and_the_sampler_seed(sampler):
    target = DiagnosedTarget(_log_likelihood, _plan(60.0, 100.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ds.infer(target, sampler="dynesty")
        direct = guard_report(target, _transform(target.parameters),
                              target.parameters.labels, n_draws=GUARD_REPORT_DRAWS, seed=0)
    assert result["guard_report"] == direct
    assert result["guard_report"]["n_draws"] == GUARD_REPORT_DRAWS
    assert result["guard_report"]["guarded"] == 0

    seeded = ds.infer(DiagnosedTarget(_log_likelihood, _plan(60.0, 100.0)),
                      sampler="dynesty", seed=11, guard_report=8)
    assert seeded["guard_report"] == guard_report(
        target, _transform(target.parameters), target.parameters.labels, n_draws=8, seed=11)
    assert seeded["guard_report"]["probed_ranges"] != guard_report(
        target, _transform(target.parameters), target.parameters.labels, n_draws=8,
        seed=0)["probed_ranges"]


def test_the_soft_guard_is_reported_as_the_target_states_it(sampler):
    target = DiagnosedTarget(_log_likelihood, _plan(60.0, 100.0), selection_neff_soft_guard=True)
    result = ds.infer(target, sampler="numpyro", guard_report=4)
    assert result["guard_report"]["guard_mode"] == "soft"


def test_guard_report_false_skips_the_report(sampler):
    target = DiagnosedTarget(_log_likelihood, _plan())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ds.infer(target, sampler="dynesty", guard_report=False)
    assert "guard_report" not in result
    assert target.calls == []


def test_a_zero_free_target_runs_no_report():
    plan = ds.ParameterPlan(labels=(), lower=(), upper=(), prior_kinds=(),
                            joint_constraints=())
    target = DiagnosedTarget(lambda theta: -2.0, plan, built_for_sampler="numpyro")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ds.infer(target, sampler="dynesty")
    assert result["logZ"] == -2.0
    assert "guard_report" not in result
    assert target.calls == []


def test_a_failing_report_never_stops_the_target_run(sampler):
    @dataclass(frozen=True)
    class Broken(DiagnosedTarget):
        def diagnostics(self, theta):
            raise RuntimeError("no diagnostics here")

    with pytest.warns(RuntimeWarning, match="guard report skipped .RuntimeError"):
        result = ds.infer(Broken(_log_likelihood, _plan()), sampler="dynesty")
    assert len(sampler.calls) == 1
    assert "guard_report" not in result


# ---------------------------------------------------------------------------
# A target without diagnostics runs as before


def _previous_path(target, sampler_name, sampler_options):
    """The target branch of ds.infer before the report (1022429)."""
    return public._execute_target(
        target.log_likelihood, target.parameters, sampler=sampler_name,
        sampler_options=sampler_options, target=target,
    )


@pytest.mark.parametrize("make", [
    lambda: ds.InferenceTarget(_log_likelihood, _plan()),
    lambda: PartlyDiagnosedTarget(_log_likelihood, _plan()),
])
@pytest.mark.parametrize("report", [True, False, 8])
def test_a_target_without_diagnostics_runs_exactly_as_before(monkeypatch, make, report):
    transform = object()
    monkeypatch.setattr("darksirens.inference.prior.make_prior_transform",
                        lambda *args, **kwargs: transform)
    now, before = _Sampler(), _Sampler()
    target = make()

    monkeypatch.setattr("darksirens.inference.sampling.run_sampler", before)
    expected = _previous_path(target, "dynesty", {"seed": 4, "nlive": 50})
    monkeypatch.setattr("darksirens.inference.sampling.run_sampler", now)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ds.infer(target, sampler="dynesty", guard_report=report, seed=4, nlive=50)

    _assert_identical(result, expected)
    assert "guard_report" not in result
    (got,), (want,) = now.calls, before.calls
    assert got.keys() == want.keys()
    for key in got:
        if key == "opts":
            assert vars(got[key]) == vars(want[key])
        else:
            assert got[key] is want[key] or got[key] == want[key], key
    assert got["transform"] is transform


def _dynesty_run(target, **options):
    options = dict(dict(nlive=40, dlogz=0.5, seed=3, show_progress=False), **options)
    return ds.infer(target, sampler="dynesty", **options)


def _assert_identical(left, right):
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict):
            _assert_identical(a, b)
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, key
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b, key


def test_real_dynesty_runs_are_bit_identical_with_and_without_the_report():
    pytest.importorskip("dynesty")
    plain = ds.InferenceTarget(_log_likelihood, _plan(40.0, 100.0))
    reference = public._execute_target(
        plain.log_likelihood, plain.parameters, sampler="dynesty",
        sampler_options=dict(nlive=40, dlogz=0.5, seed=3, show_progress=False),
        target=plain,
    )
    _assert_identical(_dynesty_run(plain), reference)
    _assert_identical(_dynesty_run(plain, guard_report=False), reference)

    # The report draws from an RNG of its own: the sampler's run is unchanged.
    diagnosed = DiagnosedTarget(_log_likelihood, _plan(40.0, 100.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reported = _dynesty_run(diagnosed)
    report = reported.pop("guard_report")
    assert report["n_draws"] == GUARD_REPORT_DRAWS and len(diagnosed.calls) == GUARD_REPORT_DRAWS
    _assert_identical(reported, reference)


# ---------------------------------------------------------------------------
# Member-marginalised targets


def test_a_member_target_with_a_report_method_gets_its_own_report(sampler):
    target = ReportingMemberTarget(_log_likelihood, _plan())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ds.infer(target, sampler="dynesty", guard_report=6, seed=9)
    assert result["guard_report"] == {"n_draws": 6, "seed": 9, "members": {"n_members": 3}}
    ((transform, labels, n_draws, seed),) = target.reports
    assert (labels, n_draws, seed) == (("H0", "x"), 6, 9)
    u = np.array([0.3, 0.6])
    np.testing.assert_array_equal(np.asarray(transform(u)),
                                  np.asarray(_transform(target.parameters)(u)))
    # core read one record, saw the member axis and handed over
    assert len(target.calls) == 1
    (call,) = sampler.calls
    assert call["transform"] is transform


def test_a_member_target_without_a_report_method_is_skipped_with_a_note(sampler):
    target = MemberTarget(_log_likelihood, _plan(), n_members=5)
    with pytest.warns(RuntimeWarning, match=r"leading member axis \(5 members\)") as caught:
        result = ds.infer(target, sampler="dynesty")
    noted = [w for w in caught if "member axis" in str(w.message)]
    assert [w.filename for w in noted] == [__file__]
    assert "guard_report" not in result
    assert len(sampler.calls) == 1


@pytest.mark.parametrize("returned, match", [
    (RuntimeError("member loop failed"), "RuntimeError: member loop failed"),
    ([1, 2], "returned a list, not a mapping"),
])
def test_a_failing_member_report_method_never_stops_the_run(sampler, returned, match):
    @dataclass(frozen=True)
    class Failing(MemberTarget):
        def guard_report(self, prior_transform, labels, *, n_draws, seed):
            if isinstance(returned, Exception):
                raise returned
            return returned

    with pytest.warns(RuntimeWarning, match=match):
        result = ds.infer(Failing(_log_likelihood, _plan()), sampler="dynesty")
    assert "guard_report" not in result
    assert len(sampler.calls) == 1


def test_a_report_method_is_not_called_without_a_member_axis(sampler):
    @dataclass(frozen=True)
    class Scalar(DiagnosedTarget):
        def guard_report(self, prior_transform, labels, *, n_draws, seed):
            raise AssertionError("a target with one record per draw gets core's report")

    target = Scalar(_log_likelihood, _plan(60.0, 100.0))
    result = ds.infer(target, sampler="dynesty", guard_report=4)
    assert result["guard_report"] == guard_report(
        target, _transform(target.parameters), target.parameters.labels, n_draws=4, seed=0)


# ---------------------------------------------------------------------------
# The sampler a target was built for


def _mismatch_warnings(caught):
    return [w for w in caught if "was built for sampler" in str(w.message)]


@pytest.mark.parametrize("built, run, soft, warned", [
    ("dynesty", "numpyro", False, True),
    ("tinyns", "dynesty", False, True),
    ("dynesty", "dynesty", False, False),
    ("numpyro", "dynesty", True, False),
    (None, "numpyro", False, False),
])
def test_a_hard_guard_target_run_under_another_sampler_warns(sampler, built, run, soft, warned):
    target = DiagnosedTarget(_log_likelihood, _plan(60.0, 100.0), built_for_sampler=built,
                             selection_neff_soft_guard=soft)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds.infer(target, sampler=run, guard_report=False)
    found = _mismatch_warnings(caught)
    assert len(found) == int(warned)
    if warned:
        (w,) = found
        assert w.category is UserWarning and w.filename == __file__
        message = str(w.message)
        assert f"built for sampler {built!r}" in message and f"runs it with {run!r}" in message
        assert ("gradient sampler" in message) == (run == "numpyro")


def test_the_sampler_check_needs_the_hard_guard_stated(sampler):
    @dataclass(frozen=True)
    class Unstated(InferenceTarget):
        built_for_sampler: str = field(default="dynesty", kw_only=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ds.infer(Unstated(_log_likelihood, _plan()), sampler="numpyro")
    assert "guard_report" not in result


# ---------------------------------------------------------------------------
# A real darksirens-lss target


def _lss_fixed_table_target():
    """A darksirens-lss fixed-table target on a small synthetic sky (nside 2).

    The fixture of the lss tests (``tests/core_fixtures.py`` there): 48 rows
    with up to 7 galaxies, 6 events of 16 PE samples, 384 injections, and a
    compact table with hot cells (``log Q = 7`` on the first rows at low
    redshift) that walls part of the H0 prior.
    """
    import inspect

    lss = pytest.importorskip("darksirens_lss")
    lss_target = pytest.importorskip("darksirens_lss.target")
    builder = getattr(lss_target, "fixed_table_target", None)
    options = {"sampler", "allow_ungauged_q_table", "guard_report"}
    if (
        not all(hasattr(lss, name) for name in ("QArtifact", "QConditioning", "catalog_identity"))
        or builder is None
        or not options <= set(inspect.signature(builder).parameters)
    ):
        pytest.skip("the installed darksirens-lss predates targets with diagnostics (lss PR #16)")
    from darksirens_lss import QArtifact, QConditioning, catalog_identity

    from darksirens.catalog.io import CatalogStore
    from darksirens.catalog.types import GalaxyCatalog
    from darksirens.cosmology._grid import zgrid
    from darksirens.gw.types import GWStore, SelectionStore
    from darksirens.runtime_binding import bind_analysis

    nside, om0, z_depth, fit = 2, 0.3075, 0.30, ("m1det", "q", "dL", "chieff")
    npix, n_max = 12 * nside**2, 7
    rng = np.random.default_rng(20261001)
    ngals = rng.integers(0, n_max + 1, npix).astype(np.int32)
    ngals[[3, 17, 30, 41]] = 0
    zgals, dzgals, wgals = (np.full((npix, n_max), 100.0), np.ones((npix, n_max)),
                            np.zeros((npix, n_max)))
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.02, 0.28, n))
        dzgals[row, :n] = 0.01 * (1.0 + zgals[row, :n])
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    store = CatalogStore(
        path="synthetic-catalog.h5", nside=nside, z_depth=z_depth,
        catalog=GalaxyCatalog(apix=np.pi / (3.0 * nside**2), zgals=zgals, dzgals=dzgals,
                              wgals=wgals, ngals=ngals, unique_pixels=None),
    )

    def columns(n, per_event, dl_lo, dl_hi):
        k = n // per_event
        rep = lambda a: np.repeat(a, per_event)  # noqa: E731
        m1 = rep(rng.uniform(28.0, 40.0, k)) * np.exp(rng.normal(0.0, 0.04, n))
        q = np.clip(rep(rng.uniform(0.65, 0.95, k)) + rng.normal(0.0, 0.03, n), 0.2, 1.0)
        ra = np.mod(rep(rng.uniform(0.0, 2.0 * np.pi, k)) + rng.normal(0.0, 0.25, n),
                    2.0 * np.pi)
        dec = np.clip(rep(np.arcsin(rng.uniform(-1.0, 1.0, k))) + rng.normal(0.0, 0.25, n),
                      -1.5, 1.5)
        dl = rep(rng.uniform(dl_lo, dl_hi, k)) * np.exp(rng.normal(0.0, 0.12, n))
        return dict(m1det=m1, m2det=q * m1, dL=dl, chieff=rng.normal(0.0, 0.04, n),
                    ra=ra, dec=dec)

    n_events, nsamp, n_sel = 6, 16, 384
    pe, sel = columns(n_events * nsamp, nsamp, 350.0, 1100.0), columns(n_sel, 1, 250.0, 1400.0)
    events = GWStore(
        format_version="fixture", path="pe-fixture.h5", fit_columns=fit, columns=pe, attrs={},
        n_events=n_events, nsamp=nsamp, prior_wt=np.full(n_events * nsamp, 1.0 / nsamp),
        event_names=tuple(f"ev{i}" for i in range(n_events)),
    )
    injections = SelectionStore(
        format_version="fixture", path="sel-fixture.h5", fit_columns=fit, columns=sel,
        attrs={}, n_injections=n_sel, ndraw=2 * n_sel, prior_wt=np.ones(n_sel),
    )
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=om0),
        population=ds.Population("powerlaw+peak", fixed=True), catalog=store,
        fixed_survey={"delta": 0.0, "sigma_kde": 0.0}, count_ratio="row",
    )
    catalog = bind_analysis(analysis, events=events, injections=injections).catalog
    n_rows = int(np.shape(catalog.zgals)[0])
    pixels = catalog.unique_pixels
    row_pixels = (np.arange(n_rows) if pixels is None
                  else np.asarray(pixels, dtype=np.int64).reshape(-1))
    logq = np.zeros((n_rows, int(zgrid.size)))
    logq[: max(1, n_rows // 16), : int(zgrid.size) // 20] = 7.0
    artifact = QArtifact(
        logq_map=logq, zgrid=np.asarray(zgrid), indexing="compact", row_pixels=row_pixels,
        budget_renormalized=False, c_mode="per_pixel",
        conditioning=QConditioning(H0=70.0, Om0=om0, w0=-1.0, wa=0.0, z_depth=z_depth,
                                   **catalog_identity(store)),
    )
    # lss's own build-time report is off: ds.infer runs core's.
    target = builder(
        analysis, events=events, injections=injections, artifact=artifact,
        sampler="dynesty", allow_ungauged_q_table=True, guard_report=False,
    )
    if not public._exposes_guard_diagnostics(target):
        pytest.skip("the installed darksirens-lss target exposes no diagnostics")
    return target


def test_ds_infer_runs_cores_report_for_a_real_lss_fixed_table_target(sampler):
    import inspect

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = _lss_fixed_table_target()
    assert all(inspect.getattr_static(target, name, None) is not None
               for name in TARGET_GUARD_REPORT_ATTRIBUTES)
    assert target.metadata["guard_report"] is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ds.infer(target, sampler="dynesty", guard_report=8, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = guard_report(target, _transform(target.parameters), target.parameters.labels,
                              n_draws=8, seed=2)
    assert result["guard_report"] == direct
    assert result["guard_report"]["guard_mode"] == "hard"
    assert result["guard_report"]["n_draws"] == 8
    walled = [w for w in caught if "likelihood guard fired" in str(w.message)]
    assert len(walled) == int(direct["guarded"] > 0.05 * 8)
    (call,) = sampler.calls
    assert call["likelihood"] is target.log_likelihood
