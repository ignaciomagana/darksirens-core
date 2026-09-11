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
from .angular import (
    ANGULAR_MODEL_NAMES,
    ANGULAR_MODEL_LATEX,
    angular_fiducial,
    angular_log_prior_volume_correction,
    angular_model_parser,
    angular_model_prior_parser,
    get_angular_model,
    get_fixed_angular_params,
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
    "ANGULAR_MODEL_NAMES",
    "ANGULAR_MODEL_LATEX",
    "get_angular_model",
    "angular_model_parser",
    "angular_model_prior_parser",
    "angular_fiducial",
    "get_fixed_angular_params",
    "angular_log_prior_volume_correction",
]
