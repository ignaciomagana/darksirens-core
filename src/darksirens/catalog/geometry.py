"""Minimal host-side HEALPix RING geometry used by ordinary catalog binding.

The installed core intentionally does not depend on ``healpy``.  This module
implements only the one operation the frozen ordinary inference path needs:
RA/Dec (radians) -> zero-based RING pixel index at a catalog NSIDE.
"""

from __future__ import annotations

import operator

import numpy as np

_MAX_NSIDE = 1 << 29
_HALF_PI = 0.5 * np.pi
_INV_HALF_PI = 2.0 / np.pi
_TWO_THIRDS = 2.0 / 3.0
# healpy hands loc2pix a precomputed sin(theta) only inside these theta windows,
# and loc2pix then uses its cancellation-free polar form there. The two limits are
# the literals in the reference HEALPix ang2pix; they are asymmetric on purpose.
_STH_THETA_LO = 0.01
_STH_THETA_HI = 3.13159


def _validated_nside(nside) -> int:
    """Return an integer RING NSIDE accepted by the HEALPix integer layout."""
    if isinstance(nside, (bool, np.bool_)):
        raise TypeError("nside must be a positive integer")
    try:
        value = operator.index(nside)
    except TypeError as exc:
        raise TypeError("nside must be a positive integer") from exc
    value = int(value)
    if value < 1 or value > _MAX_NSIDE:
        raise ValueError(f"nside must lie in [1, {_MAX_NSIDE}], got {value}")
    return value


def ang2pix_ring(nside, ra, dec):
    """Map equatorial sky coordinates to zero-based HEALPix RING pixels.

    Parameters
    ----------
    nside : int
        HEALPix RING NSIDE.  RING indexing does not require power-of-two NSIDE.
    ra, dec : array-like
        Right ascension and declination in radians. Inputs are broadcast using
        NumPy rules. Right ascension is periodic; declination must lie in
        ``[-pi/2, pi/2]``.

    Returns
    -------
    int or numpy.ndarray
        Zero-based RING pixel id(s), matching
        ``healpy.ang2pix(nside, pi/2-dec, ra, nest=False)``.

    Notes
    -----
    The integer construction mirrors the reference HEALPix RING algorithm
    operation by operation, because a one-ulp difference anywhere in it moves a
    point across a ring boundary and shifts the pixel by a whole ring. In
    particular ``theta = pi/2 - dec`` and ``z = cos(theta)`` are evaluated
    explicitly rather than replaced algebraically by ``sin(dec)``; the azimuth
    reduction is HEALPix's ``fmodulo(ra * (2/pi), 4)`` rather than
    ``remainder(ra, 2*pi) / (pi/2)``; the equatorial ``(nside * z) * 0.75``
    keeps HEALPix's left-to-right association; and the polar branch switches to
    the cancellation-free ``nside * sin(theta) / sqrt((1 + |z|) / 3)`` on the
    same theta windows healpy uses.
    """
    ns = _validated_nside(nside)
    ra_arr, dec_arr = np.broadcast_arrays(
        np.asarray(ra, dtype=np.float64),
        np.asarray(dec, dtype=np.float64),
    )
    if not np.all(np.isfinite(ra_arr)) or not np.all(np.isfinite(dec_arr)):
        raise ValueError("ra and dec must be finite")
    if np.any(dec_arr < -_HALF_PI) or np.any(dec_arr > _HALF_PI):
        raise ValueError("dec must lie in [-pi/2, pi/2] radians")

    theta = _HALF_PI - dec_arr
    z = np.cos(theta)
    za = np.abs(z)
    # HEALPix fmodulo: fmod keeps the sign of ra, and a negative ra whose
    # magnitude is below one ulp of the wrap folds to exactly 4.0, which must
    # become 0.0 or the ring index comes out one off at even nside.
    scaled = ra_arr * _INV_HALF_PI
    tt = np.fmod(scaled, 4.0)
    wrapped = tt + 4.0
    tt = np.where(scaled < 0.0, np.where(wrapped >= 4.0, 0.0, wrapped), tt)  # [0, 4)

    npix = 12 * ns * ns
    ncap = 2 * ns * (ns - 1)
    nl4 = 4 * ns
    out = np.empty(ra_arr.shape, dtype=np.int64)

    equatorial = za <= _TWO_THIRDS
    if np.any(equatorial):
        tt_e = tt[equatorial]
        z_e = z[equatorial]
        temp1 = ns * (0.5 + tt_e)
        temp2 = (ns * z_e) * 0.75
        jp = np.floor(temp1 - temp2).astype(np.int64)
        jm = np.floor(temp1 + temp2).astype(np.int64)
        ir = ns + 1 + jp - jm
        kshift = 1 - (ir & 1)
        ip = (jp + jm - ns + kshift + 1) // 2 + 1
        ip = np.where(ip > nl4, ip - nl4, ip)
        ip = np.where(ip < 1, ip + nl4, ip)
        out[equatorial] = ncap + (ir - 1) * nl4 + ip - 1

    polar = ~equatorial
    if np.any(polar):
        tt_p = tt[polar]
        za_p = za[polar]
        z_p = z[polar]
        theta_p = theta[polar]
        tp = tt_p - np.floor(tt_p)
        # 3 * (1 - za) cancels catastrophically near the poles; healpy switches
        # to sin(theta) there, and floors of tp * tmp disagree if we do not.
        stable = (theta_p < _STH_THETA_LO) | (theta_p > _STH_THETA_HI)
        tmp = np.where(
            stable,
            (ns * np.sin(theta_p)) / np.sqrt((1.0 + za_p) / 3.0),
            ns * np.sqrt(3.0 * (1.0 - za_p)),
        )
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.floor(tt_p * ir).astype(np.int64) + 1
        ip = np.where(ip > 4 * ir, ip - 4 * ir, ip)

        north = z_p > 0.0
        polar_pix = np.empty_like(ip)
        polar_pix[north] = 2 * ir[north] * (ir[north] - 1) + ip[north] - 1
        south = ~north
        polar_pix[south] = (
            npix - 2 * ir[south] * (ir[south] + 1) + ip[south] - 1
        )
        out[polar] = polar_pix

    if out.ndim == 0:
        return int(out)
    return out


__all__ = ["ang2pix_ring"]
