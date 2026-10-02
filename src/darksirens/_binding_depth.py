"""The default of ``decode_parameters(..., z_depth=...)``, importable without JAX."""

from __future__ import annotations


class _BindingDepth:
    """Marks an omitted ``z_depth``: decode at the depth ``bind_analysis`` uses."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BINDING_DEPTH"

    def __reduce__(self):
        return "BINDING_DEPTH"


#: ``decode_parameters``' default ``z_depth``: the depth the binding uses.
BINDING_DEPTH = _BindingDepth()
