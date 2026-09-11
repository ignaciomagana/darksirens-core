#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6M TinyNS config resolution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_legacy(root: Path):
    path = root / "darksirens" / "inference" / "tinyns_config.py"
    spec = importlib.util.spec_from_file_location("frozen_tinyns_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_candidate():
    import darksirens.inference.tinyns_config as module

    return module


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _resolve(module, kwargs):
    opts = SimpleNamespace(**kwargs)
    try:
        config = module.build_tinyns_config(opts)
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}
    return {
        "config": config.to_json_dict(),
        "mirror": _jsonable(opts.tinyns_resolved_config),
        "sampler_kwargs": _jsonable(module.tinyns_sampler_kwargs(config)),
        "run_kwargs": _jsonable(module.tinyns_run_kwargs(config)),
    }


def _parse(module, raw):
    try:
        return {"value": _jsonable(module.parse_chain_schedule(raw))}
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--legacy-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.implementation == "legacy":
        if not args.legacy_root:
            raise SystemExit("--legacy-root is required for legacy")
        module = _load_legacy(Path(args.legacy_root))
    else:
        module = _load_candidate()

    cases = {
        "recommended": {},
        "custom": {
            "tinyns_preset": "custom",
            "nlive": 321,
            "dlogz": 0.03,
            "max_samples": 42,
            "seed": 17,
            "show_progress": False,
            "tinyns_walks": 11,
            "tinyns_step_scale": 0.02,
            "tinyns_replacement_chain_schedule": "1,4,16",
            "tinyns_jax_block_size": 8,
        },
        "adaptive": {"tinyns_preset": "adaptive_gpu"},
        "prior": {"tinyns_preset": "prior"},
        "nonsemantic": {
            "show_progress": False,
            "tinyns_checkpoint_path": "checkpoint.npz",
            "tinyns_checkpoint_interval": 25,
            "tinyns_checkpoint_path_out": "continued.npz",
            "tinyns_progress_interval": 7,
        },
        "zero_max_samples": {"max_samples": 0},
        "bad_prior_jax": {"tinyns_preset": "prior", "tinyns_kernel": "jax"},
        "bad_prior_walks": {"tinyns_preset": "prior", "tinyns_walks": 8},
        "bad_chain_mix": {
            "tinyns_replacement_chains": 2,
            "tinyns_replacement_chain_schedule": "1,4",
        },
        "bad_bound": {"tinyns_bound": "single"},
        "bad_checkpoint_interval": {"tinyns_checkpoint_interval": 0},
        "bad_checkpoint_paths": {
            "tinyns_checkpoint_path": "a.npz",
            "tinyns_resume_from": "b.npz",
        },
    }
    schedules = {
        "none": None,
        "empty": "",
        "tuple_text": " (1, 4,16) ",
        "bad_text": "1,nope,4",
        "bad_zero": "1,0,4",
        "bad_order": "1,4,4",
    }
    behavior = {
        "display_keys": list(module.TINYNS_RESOLVED_DISPLAY_KEYS),
        "cases": {name: _resolve(module, kwargs) for name, kwargs in cases.items()},
        "schedules": {name: _parse(module, raw) for name, raw in schedules.items()},
    }
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
