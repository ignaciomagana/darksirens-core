"""Population models and registry for hierarchical siren inference."""

from .registry import (
    FIDUCIAL_SETS,
    FIDUCIAL_SET_IN_PRIOR,
    FIDUCIAL_SET_LEGACY,
    get_fixed_population_params,
    get_model,
    pop_model_parser,
    pop_model_prior_parser,
    population_m1_support_max,
)

__all__ = [
    "get_model",
    "pop_model_parser",
    "pop_model_prior_parser",
    "FIDUCIAL_SETS",
    "FIDUCIAL_SET_IN_PRIOR",
    "FIDUCIAL_SET_LEGACY",
    "get_fixed_population_params",
    "population_m1_support_max",
]
