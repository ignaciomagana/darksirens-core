"""Reusable angular source-population models and registry."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .angular_advanced import (
    MultipoleAngular,
    OverdensityGP3DAngular,
    SphereGPAngular,
    SphereZGPAngular,
    multipole_lm_indices,
)
from .base import ParamSpec, pack_specs

ANGULAR_MODEL_NAMES = (
    "isotropic",
    "dipole",
    "sphere_gp",
    "sphere_gp_z",
    "overdensity_gp",
    "multipole",
    "multipole_l3",
)
ANGULAR_MODEL_LATEX = {
    "isotropic": r"\text{Isotropic}",
    "dipole": r"\text{Dipole}",
    "sphere_gp": r"\text{Sphere GP}",
    "sphere_gp_z": r"\text{Sphere GP }(\hat n, z)",
    "overdensity_gp": r"\text{3D Overdensity GP}",
    "multipole": r"\text{Multipole }(\ell\le2)",
    "multipole_l3": r"\text{Multipole }(\ell\le3)",
}


class IsotropicAngular:
    """Null angular source model: ``g(nhat, z) == 1``."""

    @property
    def param_specs(self):
        return []

    @property
    def constraint_groups(self):
        return ()

    def prior_bounds(self):
        return [], [], []

    def log_prior_volume_correction(self) -> float:
        return 0.0

    def log_g(self, nx, ny, nz, z, theta):
        del ny, nz, z, theta
        return jnp.zeros_like(nx)


class DipoleAngular:
    r"""Mean-one dipole ``g(nhat) = 1 + nhat . d``.

    Positivity on the whole sphere requires ``|d| <= 1``. The three Cartesian
    coordinates therefore declare the frozen ``ball3`` joint-prior map rather
    than sampling a cube and clipping invalid directions pointwise.
    """

    @property
    def param_specs(self):
        return [
            ParamSpec(r"$d_x$", -1.0, 1.0, name="sky_dx"),
            ParamSpec(r"$d_y$", -1.0, 1.0, name="sky_dy"),
            ParamSpec(r"$d_z$", -1.0, 1.0, name="sky_dz"),
        ]

    @property
    def constraint_groups(self):
        return (("ball3", (r"$d_x$", r"$d_y$", r"$d_z$")),)

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_prior_volume_correction(self) -> float:
        return 0.0

    def log_g(self, nx, ny, nz, z, theta):
        del z
        dx, dy, dz = theta[0], theta[1], theta[2]
        g = 1.0 + nx * dx + ny * dy + nz * dz
        valid = (dx * dx + dy * dy + dz * dz) <= 1.0
        logg = jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)
        return jnp.where(valid, logg, -jnp.inf)


_FACTORIES = {
    "isotropic": IsotropicAngular,
    "dipole": DipoleAngular,
    "sphere_gp": SphereGPAngular,
    "sphere_gp_z": SphereZGPAngular,
    "overdensity_gp": OverdensityGP3DAngular,
    "multipole": lambda: MultipoleAngular(lmax=2),
    "multipole_l3": lambda: MultipoleAngular(lmax=3),
}
_REGISTRY: dict[str, object] = {}


def get_angular_model(name: str):
    """Return and cache one angular model with concrete construction constants."""
    if name not in _FACTORIES:
        available = ", ".join(f'"{key}"' for key in ANGULAR_MODEL_NAMES)
        raise ValueError(f"Unknown angular model '{name}'. Available: {available}.")
    if name not in _REGISTRY:
        with jax.ensure_compile_time_eval():
            _REGISTRY[name] = _FACTORIES[name]()
    return _REGISTRY[name]


def angular_model_parser(name: str):
    return get_angular_model(name).log_g


def angular_model_prior_parser(name: str):
    model = get_angular_model(name)
    lows, highs, labels = model.prior_bounds()
    kinds = [
        (spec.prior_kind, spec.prior_loc, spec.prior_scale)
        for spec in model.param_specs
    ]
    return lows, highs, labels, kinds, ANGULAR_MODEL_LATEX[name]


def angular_fiducial(name: str) -> tuple[float, ...]:
    """Return the isotropic-limit parameter vector for one angular model."""
    model = get_angular_model(name)
    values = []
    for spec in model.param_specs:
        if (
            spec.name.startswith("sky_xi")
            or spec.name.startswith("sky_a")
            or spec.name in ("sky_dx", "sky_dy", "sky_dz")
        ):
            values.append(0.0)
        else:
            values.append(0.5 * (spec.low + spec.high))
    return tuple(values)


def get_fixed_angular_params(name: str) -> jnp.ndarray:
    return jnp.array(angular_fiducial(name), dtype=float)


def angular_log_prior_volume_correction(name: str) -> float:
    model = get_angular_model(name)
    fn = getattr(model, "log_prior_volume_correction", None)
    return 0.0 if fn is None else float(fn())


__all__ = [
    "ANGULAR_MODEL_NAMES",
    "ANGULAR_MODEL_LATEX",
    "IsotropicAngular",
    "DipoleAngular",
    "SphereGPAngular",
    "SphereZGPAngular",
    "OverdensityGP3DAngular",
    "MultipoleAngular",
    "multipole_lm_indices",
    "get_angular_model",
    "angular_model_parser",
    "angular_model_prior_parser",
    "angular_fiducial",
    "get_fixed_angular_params",
    "angular_log_prior_volume_correction",
]
