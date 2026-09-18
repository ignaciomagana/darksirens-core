import numpy as np
import pytest

from darksirens.catalog.geometry import ang2pix_ring


def test_ang2pix_ring_scalar_and_broadcast_contract():
    scalar = ang2pix_ring(8, 1.2, -0.4)
    assert isinstance(scalar, int)
    assert 0 <= scalar < 12 * 8 * 8

    ra = np.array([[0.0], [0.25 * np.pi]])
    dec = np.array([[-0.4, 0.0, 0.4]])
    pix = ang2pix_ring(7, ra, dec)
    assert pix.shape == (2, 3)
    assert pix.dtype == np.int64
    assert np.all((0 <= pix) & (pix < 12 * 7 * 7))


def test_ang2pix_ring_ra_is_periodic():
    rng = np.random.default_rng(9173)
    ra = rng.uniform(0.0, 2.0 * np.pi, 256)
    dec = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, 256)
    base = ang2pix_ring(32, ra, dec)
    np.testing.assert_array_equal(base, ang2pix_ring(32, ra + 2.0 * np.pi, dec))
    np.testing.assert_array_equal(base, ang2pix_ring(32, ra - 4.0 * np.pi, dec))


def test_ang2pix_ring_accepts_non_power_of_two_ring_nside():
    pix = ang2pix_ring(3, np.linspace(0.0, 2.0 * np.pi, 31, endpoint=False), 0.1)
    assert np.all((0 <= pix) & (pix < 12 * 3 * 3))


@pytest.mark.parametrize("bad", [0, -1, (1 << 29) + 1])
def test_ang2pix_ring_rejects_out_of_range_nside(bad):
    with pytest.raises(ValueError):
        ang2pix_ring(bad, 0.0, 0.0)


@pytest.mark.parametrize("bad", [True, 2.5, "8"])
def test_ang2pix_ring_rejects_non_integer_nside(bad):
    with pytest.raises(TypeError):
        ang2pix_ring(bad, 0.0, 0.0)


@pytest.mark.parametrize("dec", [0.5 * np.pi + 1e-12, -0.5 * np.pi - 1e-12])
def test_ang2pix_ring_rejects_invalid_declination(dec):
    with pytest.raises(ValueError, match="dec must lie"):
        ang2pix_ring(8, 0.0, dec)


@pytest.mark.parametrize("ra,dec", [(np.nan, 0.0), (0.0, np.inf), (-np.inf, 0.0)])
def test_ang2pix_ring_rejects_nonfinite_coordinates(ra, dec):
    with pytest.raises(ValueError, match="finite"):
        ang2pix_ring(8, ra, dec)


_HALF_PI = 0.5 * np.pi
_TWO_PI = 2.0 * np.pi
_BELT_DEC = np.arcsin(2.0 / 3.0)
PARITY_NSIDES = (1, 2, 3, 4, 8, 16, 64, 1024)


def _assert_healpy_parity(nside, ra, dec):
    """Exact integer parity with the frozen path's ``healpy.ang2pix`` call."""
    hp = pytest.importorskip("healpy")
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    expected = np.asarray(
        hp.ang2pix(nside, _HALF_PI - dec, ra, nest=False), dtype=np.int64
    )
    actual = np.asarray(ang2pix_ring(nside, ra, dec), dtype=np.int64)
    bad = np.flatnonzero(actual != expected)
    if bad.size:
        i = int(bad[0])
        raise AssertionError(
            "ang2pix_ring disagrees with healpy at nside=%d for %d/%d points; "
            "first: ra=%.17g dec=%.17g core=%d healpy=%d"
            % (nside, bad.size, actual.size, ra.flat[i], dec.flat[i],
               actual.flat[i], expected.flat[i])
        )


def _ulp_ladder(value, steps=3):
    out = [value]
    up = down = value
    for _ in range(steps):
        up = np.nextafter(up, np.inf)
        down = np.nextafter(down, -np.inf)
        out += [up, down]
    return out


@pytest.mark.parametrize("nside", PARITY_NSIDES)
def test_ang2pix_ring_matches_healpy_over_the_full_sphere(nside):
    rng = np.random.default_rng(20260917)
    ra = rng.uniform(0.0, _TWO_PI, 20000)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, 20000))
    _assert_healpy_parity(nside, ra, dec)


@pytest.mark.parametrize("nside", (16, 64, 256, 1024))
def test_ang2pix_ring_matches_healpy_inside_the_polar_caps(nside):
    # healpy swaps in a cancellation-free polar formula near the poles; both
    # caps must follow it, including on the theta window boundaries themselves.
    rng = np.random.default_rng(4471)
    za = rng.uniform(0.99, 1.0, 4000)
    dec = np.concatenate([np.arcsin(za), -np.arcsin(za)])
    ra = rng.uniform(0.0, _TWO_PI, dec.size)
    _assert_healpy_parity(nside, ra, dec)

    theta = np.array(_ulp_ladder(0.01) + _ulp_ladder(3.13159) + [1e-9, 0.02, 0.14])
    edge_dec = np.clip(_HALF_PI - theta, -_HALF_PI, _HALF_PI)
    edge_ra = np.linspace(0.0, _TWO_PI, 257)
    ra_g, dec_g = np.meshgrid(edge_ra, edge_dec, indexing="ij")
    _assert_healpy_parity(nside, ra_g.ravel(), dec_g.ravel())


def test_ang2pix_ring_matches_healpy_at_the_unstable_polar_point():
    # Measured divergence of the unstable sqrt(3*(1-|z|)) form: core used to
    # return 0 where healpy returns 5.
    _assert_healpy_parity(256, np.array([0.9004013109915235]),
                          np.array([1.565232178147272]))


@pytest.mark.parametrize("nside", (16, 64, 256))
def test_ang2pix_ring_matches_healpy_on_the_two_thirds_belt(nside):
    decs = np.array(_ulp_ladder(_BELT_DEC) + [-d for d in _ulp_ladder(_BELT_DEC)])
    ra = np.concatenate([np.linspace(0.0, _TWO_PI, 4001), [5.6941366846315]])
    ra_g, dec_g = np.meshgrid(ra, decs, indexing="ij")
    _assert_healpy_parity(nside, ra_g.ravel(), dec_g.ravel())


def test_ang2pix_ring_matches_healpy_on_an_equatorial_ring_boundary():
    # Measured point where HEALPix's (nside * z) * 0.75 and the algebraically
    # equal nside * (0.75 * z) floor to different rings.
    _assert_healpy_parity(33, np.array([0.60513360439044406]),
                          np.array([0.51605339366208347]))


@pytest.mark.parametrize("nside", (1, 2, 3, 4, 5, 8, 16, 64, 100, 128, 1024))
def test_ang2pix_ring_matches_healpy_at_wrapping_right_ascensions(nside):
    # np.remainder returns exactly 2*pi just below a wrap; HEALPix folds it to 0.
    ra = np.array([-1e-18, 0.0, -0.0, _TWO_PI - 1e-16, np.nextafter(_TWO_PI, 0.0),
                   _TWO_PI, -_TWO_PI, 4.0 * np.pi - 1e-18, 4.0 * np.pi])
    dec = np.array([0.0, 0.3, -0.3, _BELT_DEC, -_BELT_DEC, 1.2, -1.2,
                    np.arcsin(-31.0 / 48.0), 0.5235987755982989,
                    -0.4761190609117962, -0.7021142756051663])
    ra_g, dec_g = np.meshgrid(ra, dec, indexing="ij")
    _assert_healpy_parity(nside, ra_g.ravel(), dec_g.ravel())


@pytest.mark.parametrize("nside", PARITY_NSIDES)
def test_ang2pix_ring_matches_healpy_at_the_exact_poles(nside):
    ra = np.linspace(0.0, _TWO_PI, 97)
    for dec in (_HALF_PI, -_HALF_PI):
        _assert_healpy_parity(nside, ra, np.full(ra.shape, dec))
