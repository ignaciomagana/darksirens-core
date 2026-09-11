"""Portable TinyNS configuration resolution and validation.

This module owns the backend configuration contract only.  It deliberately does
not register CLI arguments and imports neither TinyNS nor JAX, so library users
can resolve and inspect sampler configuration without loading an optional
backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PRESETS = {
    "recommended": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=5,
        step_scale=0.1, min_accepts=1, replacement_chains=1,
        replacement_chain_schedule=None, bound="none", jax_block_size=32,
        jax_vectorized=False, vectorized=False, batch_size=128, max_attempts=None,
    ),
    "heavy_darksirens": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=80,
        step_scale=0.015, min_accepts=8, replacement_chains=16,
        replacement_chain_schedule=None, bound="none", jax_block_size=32,
        jax_vectorized=False, vectorized=False, batch_size=128, max_attempts=300000,
    ),
    "heavy_darksirens_strong": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=160,
        step_scale=0.01, min_accepts=12, replacement_chains=16,
        replacement_chain_schedule=None, bound="none", jax_block_size=32,
        jax_vectorized=False, vectorized=False, batch_size=128, max_attempts=300000,
    ),
    "conservative": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=5,
        replacement_chains=1, bound="none", jax_block_size=1,
    ),
    "python_debug": dict(
        sample="rwalk", kernel="python", rwalk_proposal="isotropic", walks=25,
        replacement_chains=1, replacement_chain_schedule=None, bound="none",
        jax_block_size=1,
    ),
    "prior": dict(
        sample="prior", kernel="python", vectorized=False, bound="none",
        jax_block_size=1, replacement_chains=1,
        replacement_chain_schedule=None,
    ),
    "batched_gpu": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=25,
        replacement_chains=16, replacement_chain_schedule=None, bound="none",
        jax_block_size=1,
    ),
    "adaptive_gpu": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=25,
        replacement_chains=1, replacement_chain_schedule=(1, 4, 16, 64, 256),
        bound="none", jax_block_size=1,
    ),
    "bounded_single": dict(
        sample="rwalk", kernel="jax", bound="single", rwalk_seed="bound",
        bound_seed_kernel="python", rwalk_proposal="live-cov", walks=5,
        replacement_chains=16, jax_block_size=1,
    ),
    "bounded_multi": dict(
        sample="rwalk", kernel="jax", bound="multi", rwalk_seed="bound",
        bound_seed_kernel="python", rwalk_proposal="live-cov", walks=5,
        replacement_chains=16, jax_block_size=1,
    ),
    "fused_bounded_multi": dict(
        sample="rwalk", kernel="jax", bound="multi", rwalk_seed="bound",
        bound_seed_kernel="jax", fused_bound_rwalk=True,
        rwalk_proposal="live-cov", walks=5, replacement_chains=16,
        jax_block_size=1,
    ),
    "custom": {},
}

BASE_DEFAULTS = dict(
    sample="rwalk", kernel="jax", vectorized=False, max_attempts=None,
    walks=5, step_scale=0.1, batch_size=128, min_accepts=1,
    replacement_chains=1, replacement_chain_schedule=None,
    rwalk_proposal="isotropic",
    bound="none", bound_enlargement=1.25, bound_update_interval=100,
    bound_jitter=1e-12, bound_max_draws=None, multi_bound_max_ellipsoids=64,
    multi_bound_min_points=None, multi_bound_split_threshold=0.5,
    multi_bound_enlargement=None, multi_bound_overlap_correction=True,
    rwalk_seed="live", rwalk_seed_fallback=True, bound_seed_kernel="python",
    allow_unused_bound=False, fused_bound_rwalk=False,
    bound_rebuild_on_failure=True, bound_failure_rebuild_threshold=10,
    jax_vectorized=False, jax_block_size=32,
    checkpoint_path=None, checkpoint_interval=100, resume_from=None,
    checkpoint_path_out=None, progress_interval=100,
)

TINYNS_RESOLVED_DISPLAY_KEYS = (
    "preset", "sample", "kernel", "rwalk_proposal", "walks", "step_scale",
    "min_accepts", "replacement_chains", "max_attempts", "jax_block_size",
)


@dataclass(frozen=True)
class TinyNSConfig:
    preset: str
    nlive: int
    dlogz: float
    max_samples: int | None
    seed: int
    show_progress: bool
    sample: str
    kernel: str
    vectorized: bool
    max_attempts: int
    walks: int
    step_scale: float
    batch_size: int
    min_accepts: int
    replacement_chains: int
    replacement_chain_schedule: tuple[int, ...] | None
    rwalk_proposal: str
    bound: str
    bound_enlargement: float
    bound_update_interval: int
    bound_jitter: float
    bound_max_draws: int | None
    multi_bound_max_ellipsoids: int
    multi_bound_min_points: int | None
    multi_bound_split_threshold: float
    multi_bound_enlargement: float | None
    multi_bound_overlap_correction: bool
    rwalk_seed: str
    rwalk_seed_fallback: bool
    bound_seed_kernel: str
    allow_unused_bound: bool
    fused_bound_rwalk: bool
    bound_rebuild_on_failure: bool
    bound_failure_rebuild_threshold: int
    jax_vectorized: bool
    jax_block_size: int
    checkpoint_path: str | None
    checkpoint_interval: int | None
    resume_from: str | None
    checkpoint_path_out: str | None
    progress_interval: int
    explicit: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["replacement_chain_schedule"] = (
            list(self.replacement_chain_schedule)
            if self.replacement_chain_schedule
            else None
        )
        data["explicit"] = list(self.explicit)
        return data


_NON_SEMANTIC_RESOLVED_KEYS = (
    "checkpoint_path",
    "checkpoint_interval",
    "resume_from",
    "checkpoint_path_out",
    "show_progress",
    "progress_interval",
)


def parse_chain_schedule(raw):
    if raw in (None, ""):
        return None
    text = str(raw).strip().strip("()")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "--tinyns_replacement_chain_schedule must contain only integers."
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "--tinyns_replacement_chain_schedule must be a nonempty list of "
            "positive integers."
        )
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(
            "--tinyns_replacement_chain_schedule must be strictly increasing."
        )
    return values


def tiny_ns_preset_defaults(preset):
    if preset not in PRESETS:
        raise ValueError(f"Unknown TinyNS preset {preset!r}.")
    values = dict(BASE_DEFAULTS)
    values.update(PRESETS["recommended"] if preset == "custom" else PRESETS[preset])
    return values


def build_tinyns_config(opts):
    preset = getattr(opts, "tinyns_preset", "recommended")
    values = tiny_ns_preset_defaults(preset)
    explicit = []
    for name in BASE_DEFAULTS:
        attr = f"tinyns_{name}"
        if hasattr(opts, attr) and getattr(opts, attr) is not None:
            value = getattr(opts, attr)
            values[name] = (
                parse_chain_schedule(value)
                if name == "replacement_chain_schedule"
                else value
            )
            explicit.append(name)

    values["replacement_chain_schedule"] = parse_chain_schedule(
        values.get("replacement_chain_schedule")
    )
    max_active = (
        max(values["replacement_chain_schedule"])
        if values["replacement_chain_schedule"]
        else int(values["replacement_chains"])
    )
    if values.get("max_attempts") is None:
        values["max_attempts"] = max(10000, int(values["walks"]) * max_active)

    config = TinyNSConfig(
        preset=preset,
        nlive=int(getattr(opts, "nlive", 1000)),
        dlogz=float(getattr(opts, "dlogz", 0.1)),
        max_samples=getattr(opts, "max_samples", None),
        seed=int(getattr(opts, "seed", 0)),
        show_progress=bool(getattr(opts, "show_progress", True)),
        explicit=tuple(explicit),
        **values,
    )
    validate_tinyns_config(config)

    mirror = config.to_json_dict()
    for key in _NON_SEMANTIC_RESOLVED_KEYS:
        mirror.pop(key, None)
    mirror["explicit"] = [
        name
        for name in mirror["explicit"]
        if name not in _NON_SEMANTIC_RESOLVED_KEYS
    ]
    setattr(opts, "tinyns_resolved_config", mirror)
    return config


def _explicit(config, *names):
    return any(name in config.explicit for name in names)


def validate_tinyns_config(config):
    if config.sample not in {"rwalk", "prior"}:
        raise ValueError("TinyNS sample must be 'rwalk' or 'prior'.")
    if config.kernel not in {"python", "jax"}:
        raise ValueError("TinyNS kernel must be 'python' or 'jax'.")
    if (
        config.nlive <= 0
        or config.dlogz <= 0
        or (config.max_samples is not None and config.max_samples < 0)
    ):
        raise ValueError(
            "TinyNS requires nlive > 0, dlogz > 0, and max_samples >= 0."
        )
    if config.sample == "prior":
        if config.kernel == "jax":
            raise ValueError("TinyNS kernel='jax' is only valid with sample='rwalk'.")
        if (
            config.replacement_chains != 1
            or config.replacement_chain_schedule is not None
            or config.bound != "none"
            or config.jax_block_size != 1
        ):
            raise ValueError(
                "TinyNS sample='prior' requires replacement_chains=1, no chain "
                "schedule, bound='none', and jax_block_size=1."
            )
        if _explicit(
            config, "walks", "step_scale", "batch_size", "min_accepts",
            "rwalk_proposal",
        ):
            raise ValueError(
                "TinyNS rwalk-only options cannot be set with sample='prior'."
            )
    if config.kernel == "jax" and config.sample != "rwalk":
        raise ValueError("TinyNS kernel='jax' is only valid with sample='rwalk'.")
    if config.replacement_chains != 1 and not (
        config.sample == "rwalk" and config.kernel == "jax"
    ):
        raise ValueError(
            "replacement_chains != 1 requires sample='rwalk' and kernel='jax'."
        )
    if config.replacement_chain_schedule is not None and not (
        config.sample == "rwalk" and config.kernel == "jax"
    ):
        raise ValueError(
            "replacement_chain_schedule requires sample='rwalk' and kernel='jax'."
        )
    if (
        config.replacement_chain_schedule is not None
        and config.replacement_chains != 1
    ):
        raise ValueError(
            "replacement_chain_schedule and replacement_chains != 1 are mutually "
            "exclusive."
        )
    if (
        config.walks <= 0
        or config.step_scale <= 0
        or config.batch_size <= 0
        or config.min_accepts <= 0
    ):
        raise ValueError(
            "TinyNS walks, step_scale, batch_size, and min_accepts must be positive."
        )
    max_active = (
        max(config.replacement_chain_schedule)
        if config.replacement_chain_schedule
        else config.replacement_chains
    )
    if config.max_attempts < config.walks * max_active:
        raise ValueError(
            "TinyNS max_attempts must be >= walks * max_active_chains."
        )
    if config.bound not in {"none", "single", "multi"}:
        raise ValueError("TinyNS bound must be 'none', 'single', or 'multi'.")
    if (
        config.bound != "none"
        and config.rwalk_seed == "live"
        and not config.allow_unused_bound
    ):
        raise ValueError(
            "TinyNS bounds are being built but not used for rwalk seeding; set "
            "--tinyns_rwalk_seed bound or --tinyns_allow_unused_bound true."
        )
    if config.fused_bound_rwalk and not (
        config.sample == "rwalk"
        and config.kernel == "jax"
        and config.bound in {"single", "multi"}
        and config.rwalk_seed == "bound"
    ):
        raise ValueError(
            "fused_bound_rwalk=True requires sample='rwalk', kernel='jax', bound "
            "single/multi, and rwalk_seed='bound'."
        )
    if config.jax_block_size <= 0:
        raise ValueError("jax_block_size must be positive.")
    block_ok = config.sample == "rwalk" and config.kernel == "jax" and (
        config.bound == "none"
        or (
            config.bound in {"single", "multi"}
            and config.rwalk_seed == "bound"
            and config.bound_seed_kernel == "jax"
            and config.fused_bound_rwalk
        )
    )
    if config.jax_block_size > 1 and not block_ok:
        raise ValueError(
            "jax_block_size > 1 requires unbounded rwalk+jax or fused bounded "
            "rwalk+jax with bound seeding."
        )
    if config.bound_seed_kernel == "jax" and not (
        config.sample == "rwalk" and config.kernel == "jax"
    ):
        raise ValueError(
            "bound_seed_kernel='jax' requires sample='rwalk' and kernel='jax'."
        )
    if config.bound != "multi" and _explicit(
        config,
        "multi_bound_max_ellipsoids",
        "multi_bound_min_points",
        "multi_bound_split_threshold",
        "multi_bound_enlargement",
        "multi_bound_overlap_correction",
    ):
        raise ValueError("multi_bound_* options require --tinyns_bound multi.")
    if (
        config.bound_update_interval <= 0
        or config.bound_enlargement <= 0
        or config.bound_jitter < 0
    ):
        raise ValueError(
            "bound_update_interval and bound_enlargement must be positive; "
            "bound_jitter must be >= 0."
        )
    if config.bound_failure_rebuild_threshold <= 0:
        raise ValueError("bound_failure_rebuild_threshold must be positive.")
    if config.checkpoint_interval is None or config.checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be a positive number of iterations when "
            "checkpointing is enabled."
        )
    if config.checkpoint_path and config.resume_from and not config.checkpoint_path_out:
        raise ValueError(
            "When both resume_from and checkpoint_path are set, also set "
            "checkpoint_path_out to clarify resumed checkpoint output."
        )


def tinyns_sampler_kwargs(config):
    keys = [
        "sample", "kernel", "vectorized", "max_attempts", "walks",
        "step_scale", "batch_size", "min_accepts", "replacement_chains",
        "replacement_chain_schedule", "rwalk_proposal", "bound",
        "bound_enlargement", "bound_update_interval", "bound_jitter",
        "bound_max_draws", "multi_bound_max_ellipsoids",
        "multi_bound_min_points", "multi_bound_split_threshold",
        "multi_bound_enlargement", "multi_bound_overlap_correction",
        "rwalk_seed", "rwalk_seed_fallback", "bound_seed_kernel",
        "allow_unused_bound", "fused_bound_rwalk", "bound_rebuild_on_failure",
        "bound_failure_rebuild_threshold", "jax_vectorized", "jax_block_size",
    ]
    return {key: getattr(config, key) for key in keys}


def tinyns_run_kwargs(config):
    kwargs = dict(
        dlogz=config.dlogz,
        maxiter=(
            None
            if config.max_samples is not None and config.max_samples <= 0
            else config.max_samples
        ),
        progress=config.show_progress,
        progress_interval=config.progress_interval,
    )
    if config.checkpoint_interval is not None:
        kwargs["checkpoint_interval"] = int(config.checkpoint_interval)
    return kwargs


__all__ = [
    "BASE_DEFAULTS",
    "PRESETS",
    "TINYNS_RESOLVED_DISPLAY_KEYS",
    "TinyNSConfig",
    "build_tinyns_config",
    "parse_chain_schedule",
    "tiny_ns_preset_defaults",
    "tinyns_run_kwargs",
    "tinyns_sampler_kwargs",
    "validate_tinyns_config",
]
