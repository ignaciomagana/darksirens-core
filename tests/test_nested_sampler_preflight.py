"""Phase 6G tests for the nested-sampler finite-logL preflight."""

from __future__ import annotations

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


def test_all_minus_inf_raises_frozen_error(monkeypatch, capsys):
    _fixed_clock(monkeypatch)
    like, state = _counter_likelihood(set())
    with pytest.raises(RuntimeError) as excinfo:
        nested_sampler_preflight(
            like, lambda u: u, 2, seed=7, nlive=100, n_probe=5
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
    '<cap>). infer(..., sampler_preflight="off") skips this probe.'
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
    _fixed_clock(monkeypatch)
    like, _ = _counter_likelihood(set())
    with pytest.raises(RuntimeError) as excinfo:
        nested_sampler_preflight(like, lambda u: u, 2, seed=7, nlive=100, n_probe=5)
    message = str(excinfo.value)
    assert message.endswith(CORE_REMEDIES)

    _fixed_clock(monkeypatch)
    capsys.readouterr()
    like, _ = _counter_likelihood({1}, {1: 3.5})
    nested_sampler_preflight(like, lambda u: u, 2, seed=9, nlive=100, n_probe=5)
    warning = capsys.readouterr().out
    assert warning.endswith("If it stalls, consider: " + CORE_REMEDIES + "\n")

    for text in (message, warning):
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
