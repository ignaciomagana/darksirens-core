"""Phase 6M tests for portable TinyNS configuration resolution."""

from types import SimpleNamespace

import pytest

from darksirens.inference.tinyns_config import (
    TINYNS_RESOLVED_DISPLAY_KEYS,
    build_tinyns_config,
    parse_chain_schedule,
    tinyns_run_kwargs,
    tinyns_sampler_kwargs,
)


def test_recommended_defaults_and_kwargs():
    opts = SimpleNamespace()
    config = build_tinyns_config(opts)
    assert config.preset == "recommended"
    assert config.sample == "rwalk"
    assert config.kernel == "jax"
    assert config.nlive == 1000
    assert config.dlogz == 0.1
    assert config.walks == 5
    assert config.replacement_chains == 1
    assert config.replacement_chain_schedule is None
    assert config.jax_block_size == 32
    assert config.max_attempts == 10000
    assert config.checkpoint_interval == 100
    assert config.explicit == ()
    assert all(hasattr(config, key) for key in TINYNS_RESOLVED_DISPLAY_KEYS)

    sampler = tinyns_sampler_kwargs(config)
    assert sampler["sample"] == "rwalk"
    assert sampler["kernel"] == "jax"
    assert sampler["max_attempts"] == 10000
    assert sampler["jax_block_size"] == 32
    assert "nlive" not in sampler

    run = tinyns_run_kwargs(config)
    assert run == {
        "dlogz": 0.1,
        "maxiter": None,
        "progress": True,
        "progress_interval": 100,
        "checkpoint_interval": 100,
    }


def test_custom_uses_recommended_base_then_explicit_overrides():
    opts = SimpleNamespace(
        tinyns_preset="custom",
        nlive=321,
        dlogz=0.03,
        max_samples=42,
        seed=17,
        show_progress=False,
        tinyns_walks=11,
        tinyns_step_scale=0.02,
        tinyns_replacement_chain_schedule="1,4,16",
        tinyns_jax_block_size=8,
    )
    config = build_tinyns_config(opts)
    assert config.preset == "custom"
    assert config.nlive == 321
    assert config.dlogz == 0.03
    assert config.max_samples == 42
    assert config.seed == 17
    assert config.show_progress is False
    assert config.walks == 11
    assert config.step_scale == 0.02
    assert config.replacement_chain_schedule == (1, 4, 16)
    assert config.replacement_chains == 1
    assert config.jax_block_size == 8
    assert config.max_attempts == 10000
    assert config.explicit == (
        "walks",
        "step_scale",
        "replacement_chain_schedule",
        "jax_block_size",
    )
    assert tinyns_run_kwargs(config)["maxiter"] == 42


def test_nonsemantic_fields_are_not_mirrored_or_fingerprinted_as_explicit():
    opts = SimpleNamespace(
        tinyns_checkpoint_path="checkpoint.npz",
        tinyns_checkpoint_interval=25,
        tinyns_checkpoint_path_out="continued.npz",
        tinyns_progress_interval=7,
        show_progress=False,
    )
    first = build_tinyns_config(opts)
    first_mirror = dict(opts.tinyns_resolved_config)
    second = build_tinyns_config(opts)
    assert first == second
    assert opts.tinyns_resolved_config == first_mirror
    for key in (
        "checkpoint_path",
        "checkpoint_interval",
        "resume_from",
        "checkpoint_path_out",
        "show_progress",
        "progress_interval",
    ):
        assert key not in first_mirror
    assert first_mirror["explicit"] == []
    assert first.checkpoint_interval == 25
    assert first.progress_interval == 7


def test_chain_schedule_parser():
    assert parse_chain_schedule(None) is None
    assert parse_chain_schedule("") is None
    assert parse_chain_schedule(" (1, 4,16) ") == (1, 4, 16)
    assert parse_chain_schedule((1, 3, 9)) == (1, 3, 9)
    with pytest.raises(ValueError, match="only integers"):
        parse_chain_schedule("1,nope,4")
    with pytest.raises(ValueError, match="positive integers"):
        parse_chain_schedule("1,0,4")
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_chain_schedule("1,4,4")


def test_adaptive_preset_and_auto_attempt_budget():
    config = build_tinyns_config(SimpleNamespace(tinyns_preset="adaptive_gpu"))
    assert config.replacement_chain_schedule == (1, 4, 16, 64, 256)
    assert config.replacement_chains == 1
    assert config.walks == 25
    assert config.max_attempts == max(10000, 25 * 256)


def test_zero_max_samples_means_no_tinyns_iteration_cap():
    config = build_tinyns_config(SimpleNamespace(max_samples=0))
    assert config.max_samples == 0
    assert tinyns_run_kwargs(config)["maxiter"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tinyns_preset": "missing"}, "Unknown TinyNS preset"),
        ({"nlive": 0}, "nlive > 0"),
        ({"dlogz": 0.0}, "dlogz > 0"),
        ({"max_samples": -1}, "max_samples >= 0"),
        (
            {"tinyns_preset": "prior", "tinyns_kernel": "jax"},
            "kernel='jax' is only valid",
        ),
        (
            {"tinyns_preset": "prior", "tinyns_walks": 8},
            "rwalk-only options",
        ),
        (
            {"tinyns_kernel": "python", "tinyns_replacement_chains": 2,
             "tinyns_jax_block_size": 1},
            "replacement_chains != 1 requires",
        ),
        (
            {"tinyns_replacement_chains": 2,
             "tinyns_replacement_chain_schedule": "1,4"},
            "mutually exclusive",
        ),
        (
            {"tinyns_max_attempts": 4, "tinyns_walks": 5},
            "max_attempts must be >=",
        ),
        (
            {"tinyns_bound": "single"},
            "bounds are being built but not used",
        ),
        (
            {"tinyns_fused_bound_rwalk": True},
            "fused_bound_rwalk=True requires",
        ),
        (
            {"tinyns_kernel": "python", "tinyns_jax_block_size": 2},
            "jax_block_size > 1 requires",
        ),
        (
            {"tinyns_bound": "single", "tinyns_rwalk_seed": "bound",
             "tinyns_multi_bound_max_ellipsoids": 8,
             "tinyns_jax_block_size": 1},
            r"multi_bound_\* options require",
        ),
        (
            {"tinyns_checkpoint_interval": 0},
            "checkpoint_interval must be a positive",
        ),
        (
            {"tinyns_checkpoint_path": "a.npz", "tinyns_resume_from": "b.npz"},
            "also set checkpoint_path_out",
        ),
    ],
)
def test_invalid_configurations_fail_eagerly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_tinyns_config(SimpleNamespace(**kwargs))


# --- the livecov preset (TinyNS >= 0.2.0) -----------------------------------

from types import SimpleNamespace as _NS  # noqa: E402

from darksirens.inference.tinyns_adapter import require_tinyns_for_config  # noqa: E402
from darksirens.inference.tinyns_config import (  # noqa: E402
    PRESETS as _PRESETS,
    build_tinyns_config as _build,
    tinyns_sampler_kwargs as _kwargs,
    validate_tinyns_config as _validate,
)


def test_livecov_preset_leaves_walks_and_max_attempts_to_tinyns():
    config = _build(_NS(tinyns_preset="livecov"))
    _validate(config)
    assert config.rwalk_proposal == "live-cov" and config.bound == "none"
    assert config.walks is None and config.max_attempts is None
    kw = _kwargs(config)
    # walks=None goes through (TinyNS resolves it); max_attempts is not passed,
    # since TinyNS refuses an explicit None and applies its own default.
    assert kw["walks"] is None and "max_attempts" not in kw and kw["step_scale"] == 0.5
    assert _kwargs(_build(_NS()))["max_attempts"] == max(10000, 5)


def test_livecov_with_explicit_walks_resolves_max_attempts_in_core():
    config = _build(_NS(tinyns_preset="livecov", tinyns_walks=78))
    _validate(config)
    assert config.walks == 78 and config.max_attempts == max(10000, 78)


def test_walks_none_with_a_fixed_max_attempts_is_refused():
    with pytest.raises(ValueError, match="walks=None needs max_attempts=None"):
        _build(_NS(tinyns_preset="livecov", tinyns_max_attempts=5000))


def test_existing_presets_are_unchanged_by_the_livecov_addition():
    assert _PRESETS["recommended"]["walks"] == 5 and _PRESETS["recommended"]["rwalk_proposal"] == "isotropic"
    assert _build(_NS()).max_attempts == max(10000, 5)


@pytest.mark.parametrize("version, ok", [("0.1.0", False), (None, False), ("0.2.0", True), ("0.3.1", True)])
def test_livecov_needs_tinyns_0_2_0(version, ok):
    config = _build(_NS(tinyns_preset="livecov"))
    module = _NS() if version is None else _NS(__version__=version)
    if ok:
        require_tinyns_for_config(config, module)
    else:
        with pytest.raises(ValueError, match="needs tinyns >= 0.2.0"):
            require_tinyns_for_config(config, module)


def test_isotropic_presets_run_on_old_tinyns():
    require_tinyns_for_config(_build(_NS()), _NS(__version__="0.1.0"))
