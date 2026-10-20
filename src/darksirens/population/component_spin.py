"""4-D component-spin population model (DS-08).

Independent, identically distributed component-spin magnitudes
``a_i ~ Beta(alpha_chi, beta_chi)`` on [0, 1] and an isotropic plus
preferentially aligned tilt mixture on [-1, 1]^2, in one of two forms
(``ComponentSpinModel.tilt_mixture``):

* ``"per_spin"`` (default, the frozen reference's form): each tilt carries
  its own mixture, ``cos t_i ~ zeta_spin * N_trunc(cos t; 1, sigma_t) +
  (1 - zeta_spin) * 1/2``, and the two are independent;
* ``"binary"`` (opt-in): the GWTC-3 "Default" spin model (Talbot & Thrane
  2017; GWTC-2 and GWTC-3 population papers), one mixture over the binary,
  ``zeta_spin * N_trunc(cos t1) N_trunc(cos t2) + (1 - zeta_spin) / 4``.

The two have the same one-tilt marginal but not the same joint density: the
per-spin form has uncorrelated tilts and cross terms the Default model does
not (at ``zeta_spin = 0.75``, ``sigma_t = 0.9`` it gives two anti-aligned
spins half the Default model's density, and corr(cos t1, cos t2) is 0
against 0.072), so its ``zeta_spin`` and ``sigma_t`` are not the published
model's.

Everything is normalised ANALYTICALLY -- the Beta by ``betaln``, the
truncated Gaussian by ``erf`` -- so no new quadrature enters the likelihood:
the existing 1-D ``SpinModel`` machinery integrates its chi_eff density over
``CHI_GRID`` per proposal, and a 4-D grid twin of that would be the exact
cost this design avoids.

The model consumes the event's ``spin`` block (``GWEvent.spin``, columns
``(a1, a2, cost1, cost2)``) and IGNORES ``chieff``: chi_eff is a deterministic
function of the component spins and q, so a density over the components
already fixes its distribution, and multiplying by an extra chi_eff factor
would double-count.  It therefore only makes sense against a
component-basis store, whose ``p_pe``/``pdraw`` carry the flat component
draw density -- the basis-negotiation gate (DS-09) is what enforces that
pairing.  N.B. the selection-side variance measurement (gwcat GW-24) found
the component basis consumed by a 1-D chi_eff population WORSE than the
chieff basis end-to-end: this model existing is the precondition for the
component basis being usable at all, not an optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.special import betaln, erf

from .base import ParamSpec, SpinModel

#: Column order of the spin block this model consumes (= GWEvent.spin).
COMPONENT_SPIN_COLUMNS = ("a1", "a2", "cost1", "cost2")

#: Forms of the tilt mixture (``ComponentSpinModel.tilt_mixture``).
TILT_MIXTURES = ("per_spin", "binary")

_SQRT_HALF_PI = 1.2533141373155003  # sqrt(pi/2)
_SQRT2 = 1.4142135623730951


@dataclass
class ComponentSpinModel(SpinModel):
    """iid Beta magnitudes x (isotropic + aligned-Gaussian) tilt mixture.

    Parameters (in ``theta`` order): ``alpha_chi``, ``beta_chi`` (Beta shape
    parameters of each magnitude; bounded below by 1 so the density is
    non-singular at the endpoints, matching the LVK convention),
    ``zeta_spin`` (aligned fraction), ``sigma_t`` (width of the aligned
    tilt Gaussian, truncated to [-1, 1] and peaked at cos t = 1).

    ``tilt_mixture`` is ``"per_spin"`` (default: one mixture per tilt, the
    two independent) or ``"binary"`` (one mixture over the binary, the GWTC-3
    Default spin model); see the module docstring.

    This is a :class:`SpinModel` so it slots into every population model's
    ``spin_component`` field and its ``ParamSpec``s reach the flat
    labels/lower/upper vector through the ordinary ``param_specs`` chain --
    but it consumes the (N, 4) spin BLOCK, not chi_eff, so the base class's
    grid normalisation is bypassed (``_norm`` is analytically 1).
    """

    alpha_chi_spec: ParamSpec
    beta_chi_spec: ParamSpec
    zeta_spin_spec: ParamSpec
    sigma_t_spec: ParamSpec
    tilt_mixture: str = "per_spin"

    #: Population models dispatch on this: True means __call__ requires the
    #: event spin block and the paired store must be component-basis.
    consumes_spin_block = True
    spin_columns = COMPONENT_SPIN_COLUMNS

    def __post_init__(self):
        if self.tilt_mixture not in TILT_MIXTURES:
            raise ValueError(
                f"tilt_mixture must be one of {TILT_MIXTURES}, got {self.tilt_mixture!r}"
            )

    @property
    def param_specs(self):
        return [
            self.alpha_chi_spec,
            self.beta_chi_spec,
            self.zeta_spin_spec,
            self.sigma_t_spec,
        ]

    # -- analytic pieces ---------------------------------------------------
    @staticmethod
    def _log_p_magnitude(a, alpha, beta):
        """log Beta(a; alpha, beta) on [0, 1]; -inf outside the open support."""
        in_support = (a > 0.0) & (a < 1.0)
        a_safe = jnp.clip(a, 1e-300, 1.0 - 1e-16)
        lp = (
            (alpha - 1.0) * jnp.log(a_safe)
            + (beta - 1.0) * jnp.log1p(-a_safe)
            - betaln(alpha, beta)
        )
        return jnp.where(in_support, lp, -jnp.inf)

    @staticmethod
    def _log_p_tilt(c, zeta, sigma_t):
        """log of the tilt mixture on [-1, 1]; -inf outside.

        The aligned component is a Gaussian peaked at cos t = 1 truncated to
        [-1, 1], normalised in closed form:
        Z = sigma * sqrt(pi/2) * erf(sqrt(2)/sigma).
        """
        in_support = (c >= -1.0) & (c <= 1.0)
        z_aligned = sigma_t * _SQRT_HALF_PI * erf(_SQRT2 / sigma_t)
        p_aligned = jnp.exp(-0.5 * ((c - 1.0) / sigma_t) ** 2) / z_aligned
        p = zeta * p_aligned + (1.0 - zeta) * 0.5
        return jnp.where(
            in_support, jnp.log(jnp.maximum(p, 1e-300)), -jnp.inf
        )

    @staticmethod
    def _log_p_tilt_binary(c1, c2, zeta, sigma_t):
        """log of the binary tilt mixture on [-1, 1]^2; -inf outside.

        ``zeta N(c1) N(c2) + (1 - zeta) / 4`` with the truncated Gaussian of
        :meth:`_log_p_tilt`, so its one-tilt marginal is that mixture.
        """
        in_support = (c1 >= -1.0) & (c1 <= 1.0) & (c2 >= -1.0) & (c2 <= 1.0)
        z_aligned = sigma_t * _SQRT_HALF_PI * erf(_SQRT2 / sigma_t)
        p_aligned = (jnp.exp(-0.5 * (((c1 - 1.0) / sigma_t) ** 2 + ((c2 - 1.0) / sigma_t) ** 2))
                     / (z_aligned * z_aligned))
        p = zeta * p_aligned + (1.0 - zeta) * 0.25
        return jnp.where(
            in_support, jnp.log(jnp.maximum(p, 1e-300)), -jnp.inf
        )

    def log_prob(self, spin, theta):
        """Normalised log density of the (..., 4) spin block ``(a1, a2, cost1, cost2)``."""
        alpha, beta, zeta, sigma_t = theta[0], theta[1], theta[2], theta[3]
        magnitudes = (
            self._log_p_magnitude(spin[..., 0], alpha, beta)
            + self._log_p_magnitude(spin[..., 1], alpha, beta)
        )
        if self.tilt_mixture == "binary":
            return magnitudes + self._log_p_tilt_binary(spin[..., 2], spin[..., 3], zeta, sigma_t)
        return (
            magnitudes
            + self._log_p_tilt(spin[..., 2], zeta, sigma_t)
            + self._log_p_tilt(spin[..., 3], zeta, sigma_t)
        )

    # -- SpinModel interface -----------------------------------------------
    def _eval_unnorm(self, chieff, theta):
        raise TypeError(
            "ComponentSpinModel has no 1-D chi_eff density: it is a 4-D "
            "density over (a1, a2, cost1, cost2), and its chi_eff marginal "
            "is not analytic. Evaluate log_prob(spin, theta), or use a "
            "chi_eff SpinModel with a chieff-basis store."
        )

    def _norm(self, theta):
        # Analytically normalised; the grid integral the base class would run
        # is both unnecessary and undefined (no 1-D density to integrate).
        return jnp.asarray(1.0)

    def __call__(self, chieff, theta, norm=None, spin=None):
        if spin is None:
            raise TypeError(
                "ComponentSpinModel requires the event spin block "
                "(a1, a2, cost1, cost2): the paired gwcat store must be "
                "exported in the component spin basis. chieff-basis stores "
                "carry no component columns; use a chi_eff spin model with "
                "them instead."
            )
        return jnp.exp(self.log_prob(spin, theta))


def default_component_spin(tilt_mixture: str = "per_spin") -> ComponentSpinModel:
    """The default component-spin block (GWTC-3 Default-model parameters).

    Bounds: aligned fraction in [0, 1] and aligned-tilt width in [0.1, 4],
    the LVK Default ranges; Beta shapes ``alpha_chi``, ``beta_chi`` uniform
    in [1, 10] (non-singular at the endpoints).  The magnitude prior is NOT
    the LVK one: the LVK Default model samples the Beta through its mean and
    variance (``E[a] ~ U(0, 1)``, ``Var[a] ~ U(0, 0.25)``, restricted to
    shapes >= 1), a different density over the shapes, and the [1, 10] box
    also bounds the mean magnitude below by 1/11, which that prior does not.
    ``tilt_mixture`` is passed to :class:`ComponentSpinModel` (``"binary"``
    is the GWTC-3 Default tilt model; the default ``"per_spin"`` is not).
    """
    return ComponentSpinModel(
        ParamSpec(r"$\alpha_\chi$", 1.0, 10.0, name="alpha_chi"),
        ParamSpec(r"$\beta_\chi$", 1.0, 10.0, name="beta_chi"),
        ParamSpec(r"$\zeta_{\rm spin}$", 0.0, 1.0, name="zeta_spin"),
        ParamSpec(r"$\sigma_t$", 0.1, 4.0, name="sigma_t"),
        tilt_mixture=tilt_mixture,
    )
