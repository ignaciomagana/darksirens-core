"""Core hierarchical-likelihood primitives and explicit extension seams."""

from .hierarchical import SpectralLikelihoodDiagnostics, spectral_siren_log_likelihood
from .host_density import (
    HostDensityLikelihoodDiagnostics,
    RedshiftModel,
    host_density_log_likelihood,
    make_host_density_target,
)
from .weights import (
    M1DET_Q_DL_COORDS,
    log_jacobian_dL_to_z,
    log_jacobian_m1src_q_z_to_m1det_q_dL,
    log_sample_weight,
    log_target_density_m1det_q_dL,
)

__all__ = [
    "M1DET_Q_DL_COORDS",
    "SpectralLikelihoodDiagnostics",
    "HostDensityLikelihoodDiagnostics",
    "RedshiftModel",
    "host_density_log_likelihood",
    "make_host_density_target",
    "log_jacobian_dL_to_z",
    "log_jacobian_m1src_q_z_to_m1det_q_dL",
    "log_sample_weight",
    "log_target_density_m1det_q_dL",
    "spectral_siren_log_likelihood",
]
