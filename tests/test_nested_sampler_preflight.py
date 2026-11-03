"""Phase 6G tests for the nested-sampler finite-logL preflight."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import darksirens.inference.preflight as preflight
from darksirens.inference.preflight import nested_sampler_preflight


def _fixed_clock(monkeypatch, elapsed=1.25):
    ticks = iter((100.0, 100.0 + elapsed))
    monkeypatch.setattr(preflight.time, "perf_counter", lambda: next(ticks))


def _counter_likelihood(finite_calls, values=None):
    finite_calls = set(finite_calls)
    values = values or {}
    state = {"calls": 0, "types": []}

    def likelihood(theta):
        state["calls"] += 1
        state["types"].append(isinstance(theta, jax.Array))
        i = state["calls"]
        if i in finite_calls:
            return jnp.asarray(values.get(i, float(i)))
        return jnp.asarray(-jnp.inf)

    return likelihood, state


def test_disabled_preflight_makes_no_calls_or_output(capsys):
    calls = {"prior": 0, "like": 0}

    def prior(u):
        calls["prior"] += 1
        return u

    def like(theta):
        calls["like"] += 1
        return 0.0

    nested_sampler_preflight(like, prior, 2, enabled=False, n_probe=5)
    assert calls == {"prior": 0, "like": 0}
    assert capsys.readouterr().out == ""


def test_all_minus_inf_raises_frozen_error_at_the_legacy_setting(monkeypatch, capsys):
    # max_draws == n_probe is the frozen legacy probe: it stops there.
    _fixed_clock(monkeypatch)
    like, state = _counter_likelihood(set())
    with pytest.raises(RuntimeError) as excinfo:
        nested_sampler_preflight(
            like, lambda u: u, 2, seed=7, nlive=100, n_probe=5, max_draws=5
        )
    assert state["calls"] == 5
    assert capsys.readouterr().out == (
        "[*] preflight: 0/5 prior draws have finite logL [1.25 s]\n"
    )
    assert str(excinfo.value) == (
        "Nested-sampler preflight: the likelihood is -inf on ALL 5 probed prior "
        "draws, so dynesty/tinyns would reject-sample forever without ever "
        "finding an initial live point. The most common cause is the selection "
        "variance guard (GWTC-4.0/5.0 total-log-likelihood variance: -inf when "
        "sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel > "
        "max_likelihood_variance, plus the Vitale Neff > 5*N_obs floor). "
        + CORE_REMEDIES
    )


# The frozen legacy remedy text, kept verbatim so the Phase 6G remedy map is
# checked against it without the legacy tree.
LEGACY_REMEDIES = (
    "Remedies: --selection_neff_guard soft (finite penalized wall — the "
    "sampler initializes and is pushed toward the region satisfying the "
    "criterion); --max_likelihood_variance <cap> (accept a larger MC "
    "variance — the measured sigma^2 at your best-fit point must be below "
    "<cap>); python scripts/diagnose_selection_guard.py -- <your "
    "darksirens_inference args> (measure sigma^2_lnL on your data and "
    "report the smallest admitting cap). --sampler_preflight off skips this "
    "probe."
)

CORE_REMEDIES = (
    'Remedies: infer(..., selection_neff_guard="soft") (finite penalized wall '
    "— the sampler initializes and is pushed toward the region satisfying the "
    "criterion); infer(..., max_likelihood_variance=<cap>) (accept a larger MC "
    "variance — the measured sigma^2 at your best-fit point must be below "
    "<cap>); for an InferenceTarget, whose likelihood options infer refuses, "
    "build its likelihood with selection_neff_soft_guard=True or "
    'max_likelihood_variance=<cap> instead. infer(..., sampler_preflight="off") '
    "skips this probe."
)

LEGACY_FLAG_STRINGS = (
    "--selection_neff_guard",
    "--max_likelihood_variance",
    "--sampler_preflight",
    "scripts/diagnose_selection_guard.py",
    "darksirens_inference",
)


def _load_probe():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "probe_nested_sampler_preflight.py"
    )
    spec = importlib.util.spec_from_file_location("_preflight_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remedies_name_public_infer_keywords_not_legacy_flags(monkeypatch, capsys):
    messages = []
    for max_draws in (5, 40):  # the legacy setting and past n_probe
        _running_clock(monkeypatch)
        like, _ = _counter_likelihood(set())
        with pytest.raises(RuntimeError) as excinfo:
            nested_sampler_preflight(
                like, lambda u: u, 2, seed=7, nlive=100, n_probe=5, max_draws=max_draws
            )
        messages.append(str(excinfo.value))
        assert messages[-1].endswith(CORE_REMEDIES)

    _fixed_clock(monkeypatch)
    capsys.readouterr()
    like, _ = _counter_likelihood({1}, {1: 3.5})
    nested_sampler_preflight(like, lambda u: u, 2, seed=9, nlive=100, n_probe=5)
    warning = capsys.readouterr().out
    assert warning.endswith("If it stalls, consider: " + CORE_REMEDIES + "\n")

    for text in (*messages, warning):
        for legacy in LEGACY_FLAG_STRINGS:
            assert legacy not in text


def test_phase6g_remedy_map_turns_legacy_text_into_core_text_only():
    probe = _load_probe()
    prefix = "If it stalls, consider: "
    legacy = {"case": {"stdout": prefix + LEGACY_REMEDIES + "\n", "error": None}}
    mapped = probe.map_legacy_remedies(legacy)
    assert mapped == {"case": {"stdout": prefix + CORE_REMEDIES + "\n", "error": None}}
    probe.compare(legacy, mapped)

    drifted = {"case": {"stdout": "If it halts, consider: " + CORE_REMEDIES + "\n",
                        "error": None}}
    with pytest.raises(SystemExit, match="parity failure"):
        probe.compare(legacy, drifted)
    with pytest.raises(SystemExit, match="legacy remedy strings"):
        probe.compare(legacy, legacy)
    with pytest.raises(SystemExit, match="absent from legacy record"):
        probe.compare(mapped, mapped)


def test_one_finite_draw_warns_and_estimates_initialization(monkeypatch, capsys):
    _fixed_clock(monkeypatch)
    like, state = _counter_likelihood({1}, {1: 3.5})
    nested_sampler_preflight(like, lambda u: u, 2, seed=9, nlive=100, n_probe=5)
    out = capsys.readouterr().out
    assert state["calls"] == 5
    assert "preflight: 1/5 prior draws have finite logL (finite logL in [3.5, 3.5])" in out
    assert "finite fraction 20.0%" in out
    assert "about ~500 draws to seed 100 live points" in out


def test_three_finite_draws_warn_but_fourth_finite_stops_early(monkeypatch, capsys):
    _fixed_clock(monkeypatch)
    like, state = _counter_likelihood({2, 4, 6, 7})
    nested_sampler_preflight(like, lambda u: u, 2, seed=4, nlive=50, n_probe=10)
    out = capsys.readouterr().out
    assert state["calls"] == 7
    assert "preflight: 4/7 prior draws have finite logL" in out
    assert "preflight WARNING" not in out


def test_all_finite_stops_after_four_and_preserves_range_format(monkeypatch, capsys):
    _fixed_clock(monkeypatch)
    like, state = _counter_likelihood(
        {1, 2, 3, 4}, {1: -12.34567, 2: -1.25, 3: 0.125, 4: 98.7654}
    )
    nested_sampler_preflight(like, lambda u: u, 3, seed=1, nlive=64, n_probe=32)
    out = capsys.readouterr().out
    assert state["calls"] == 4
    assert "4/4" in out
    assert "finite logL in [-12.35, 98.77]" in out


def test_nlive_zero_warning_uses_many(monkeypatch, capsys):
    _fixed_clock(monkeypatch)
    like, _ = _counter_likelihood({2})
    nested_sampler_preflight(like, lambda u: u, 2, seed=2, nlive=0, n_probe=4)
    assert "reject-sample about many draws to seed 0 live points" in capsys.readouterr().out


def test_seed_xor_sequence_and_global_rng_independence(monkeypatch):
    _fixed_clock(monkeypatch)
    seen = []

    def prior(u):
        seen.append(np.asarray(u).copy())
        return u

    like, _ = _counter_likelihood({1, 2, 3, 4})
    seed = 1234
    np.random.seed(98765)
    expected_global = np.random.random(8)
    np.random.seed(98765)
    nested_sampler_preflight(like, prior, 3, seed=seed, nlive=16, n_probe=9)
    observed_global = np.random.random(8)
    np.testing.assert_array_equal(observed_global, expected_global)

    expected_draws = np.random.default_rng(seed ^ 0xC0FFEE).random((4, 3))
    np.testing.assert_array_equal(np.stack(seen), expected_draws)


def test_prior_and_likelihood_receive_jax_arrays(monkeypatch):
    _fixed_clock(monkeypatch)
    prior_types = []

    def prior(u):
        prior_types.append(isinstance(u, jax.Array))
        return np.asarray(u) + 0.5

    like, state = _counter_likelihood({1, 2, 3, 4})
    nested_sampler_preflight(like, prior, 2, seed=5, nlive=8, n_probe=8)
    assert prior_types == [True] * 4
    assert state["types"] == [True] * 4


# --- Past the first n_probe draws (the default since the escalation) ---------


def _running_clock(monkeypatch):
    ticks = itertools.count(100.0, 0.25)
    monkeypatch.setattr(preflight.time, "perf_counter", lambda: next(ticks))


def _one_percent_likelihood():
    # Finite on 1% of the unit-cube prior, like a hard variance guard that
    # leaves little of the prior (the 12AU chains: about 3% at V = 4).
    state = {"calls": 0}

    def likelihood(theta):
        state["calls"] += 1
        return jnp.where(theta[0] < 0.01, -1.5, -jnp.inf)

    return likelihood, state


def _expected_one_percent(seed, ndims, n_probe, max_draws):
    """The probe's own stream, replayed: (draws taken, finite among the first n_probe)."""
    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    hits = first = n = 0
    while hits < 4 and n < max_draws:
        n += 1
        if rng.random(ndims)[0] < 0.01:
            hits += 1
            first += n <= n_probe
    return n, first


def test_likelihood_finite_on_one_percent_of_the_prior_now_passes(monkeypatch, capsys):
    seed, nlive = 11, 1000
    n, first = _expected_one_percent(seed, 2, 32, 4096)
    assert first == 0 and n > 32  # the case the legacy probe refused

    _running_clock(monkeypatch)
    like, state = _one_percent_likelihood()
    nested_sampler_preflight(like, lambda u: u, 2, seed=seed, nlive=nlive)
    out = capsys.readouterr().out
    assert state["calls"] == n
    assert "[*] preflight: 0/32 prior draws have finite logL so far" in out
    assert "drawing on to 64 (max_draws 4096)" in out
    assert f"[*] preflight: 4/{n} prior draws have finite logL (finite logL in [-1.5, -1.5])" in out
    # The slow-initialization warning, with the draws to seed nlive.
    assert f"only 4/{n} prior draws are finite" in out
    assert f"about ~{int(np.ceil(nlive * n / 4)):,} draws to seed {nlive} live points" in out
    assert out.rstrip("\n").endswith("If it stalls, consider: " + CORE_REMEDIES)

    # The legacy setting refuses the same likelihood after 32 draws.
    _fixed_clock(monkeypatch)
    like, state = _one_percent_likelihood()
    with pytest.raises(RuntimeError, match="-inf on ALL 32 probed prior draws, so dynesty"):
        nested_sampler_preflight(like, lambda u: u, 2, seed=seed, nlive=nlive, max_draws=32)
    assert state["calls"] == 32


def test_always_minus_inf_raises_with_the_number_of_draws_probed(monkeypatch, capsys):
    _running_clock(monkeypatch)
    like, state = _counter_likelihood(set())
    with pytest.raises(RuntimeError) as excinfo:
        nested_sampler_preflight(like, lambda u: u, 2, seed=7, nlive=1000)
    assert state["calls"] == 4096
    message = str(excinfo.value)
    assert message.startswith(
        "Nested-sampler preflight: the likelihood is -inf on ALL 4096 probed "
        "prior draws (32 first, then batches doubling the number drawn, up to "
        "max_draws=4096). The finite fraction of the prior is then likely "
        "below 3/4096 (0.073%), so dynesty/tinyns would likely need more than "
        "~1,365,334 draws to seed 1000 live points, and would reject-sample "
        "forever if the likelihood is -inf on the whole prior. "
        "infer(..., sampler_preflight_max_draws=<n>) probes more draws before "
        "refusing. The most common cause is the selection variance guard"
    )
    # "forever" only as the condition the evidence cannot exclude.
    assert "reject-sample forever without ever finding" not in message
    assert message.endswith(CORE_REMEDIES)
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(" [")[0] for line in lines] == [
        f"[*] preflight: 0/{n} prior draws have finite logL so far"
        for n in (32, 64, 128, 256, 512, 1024, 2048)
    ] + ["[*] preflight: 0/4096 prior draws have finite logL"]
    assert lines[0].endswith("drawing on to 64 (max_draws 4096)")
    assert lines[-2].endswith("drawing on to 4096 (max_draws 4096)")


def test_a_ceiling_off_the_doubling_is_reached_exactly(monkeypatch, capsys):
    _running_clock(monkeypatch)
    like, state = _counter_likelihood(set())
    with pytest.raises(RuntimeError, match=r"-inf on ALL 12 probed prior draws \(5 first"):
        nested_sampler_preflight(like, lambda u: u, 2, seed=7, nlive=0, n_probe=5, max_draws=12)
    assert state["calls"] == 12
    assert "drawing on to 10" in capsys.readouterr().out


def test_a_finite_draw_past_n_probe_warns_with_its_fraction(monkeypatch, capsys):
    # One finite draw, the 40th of 4096: the warning gives 1/4096 and the
    # estimate against nlive, not a refusal.
    _running_clock(monkeypatch)
    like, state = _counter_likelihood({40}, {40: -2.0})
    nested_sampler_preflight(like, lambda u: u, 2, seed=1, nlive=500)
    out = capsys.readouterr().out
    assert state["calls"] == 4096
    assert "[*] preflight: 1/4096 prior draws have finite logL (finite logL in [-2, -2])" in out
    assert "finite fraction 0.024%" in out
    assert "about ~2,048,000 draws to seed 500 live points" in out


def test_max_draws_below_n_probe_is_refused():
    like, state = _counter_likelihood(set())
    with pytest.raises(ValueError, match=r"max_draws \(16\) must be at least n_probe \(32\)"):
        nested_sampler_preflight(like, lambda u: u, 2, max_draws=16)
    assert state["calls"] == 0


def test_escalated_draws_continue_the_probe_stream_and_leave_global_rng_alone(monkeypatch):
    _running_clock(monkeypatch)
    seen = []

    def prior(u):
        seen.append(np.asarray(u).copy())
        return u

    like, _ = _counter_likelihood(set())
    seed = 1234
    np.random.seed(98765)
    expected_global = np.random.random(8)
    np.random.seed(98765)
    with pytest.raises(RuntimeError, match="ALL 100 probed"):
        nested_sampler_preflight(like, prior, 3, seed=seed, nlive=16, n_probe=9, max_draws=100)
    observed_global = np.random.random(8)
    np.testing.assert_array_equal(observed_global, expected_global)
    # One stream over the batches (9, 18, 36, 72, 100): no reseed, no skip.
    expected_draws = np.random.default_rng(seed ^ 0xC0FFEE).random((100, 3))
    np.testing.assert_array_equal(np.stack(seen), expected_draws)


def test_max_draws_is_an_infer_sampler_option_threaded_to_the_probe(monkeypatch):
    from types import SimpleNamespace

    import darksirens.inference.sampling as sampling
    from darksirens.inference.public import _check_sampler_options, _sampler_namespace
    from darksirens.inference.run_fingerprint import sampler_semantic

    seen = []
    monkeypatch.setattr(
        preflight, "nested_sampler_preflight", lambda *args, **kwargs: seen.append(kwargs)
    )
    opts = SimpleNamespace(seed=3, nlive=10, sampler_preflight="on")
    sampling._nested_preflight(None, None, 2, opts)
    opts.sampler_preflight_max_draws = 32
    sampling._nested_preflight(None, None, 2, opts)
    assert [k["max_draws"] for k in seen] == [preflight.PREFLIGHT_MAX_DRAWS, 32]
    assert preflight.PREFLIGHT_MAX_DRAWS == 4096

    _check_sampler_options({"sampler_preflight_max_draws": 32})
    default = _sampler_namespace("dynesty", {"nlive": 10})
    legacy = _sampler_namespace("dynesty", {"nlive": 10, "sampler_preflight_max_draws": 32})
    assert default.sampler_preflight_max_draws == preflight.PREFLIGHT_MAX_DRAWS
    # A start-time check, not a sampler setting: a resume may change it.
    assert sampler_semantic("dynesty", legacy) == sampler_semantic("dynesty", default)
