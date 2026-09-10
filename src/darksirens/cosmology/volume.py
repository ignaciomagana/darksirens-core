"""Comoving-volume operations for the flat-CPL distance model."""

from .distances import E, dV_of_z

expansion_rate = E
differential_comoving_volume = dV_of_z

__all__ = [
    "E",
    "dV_of_z",
    "expansion_rate",
    "differential_comoving_volume",
]
