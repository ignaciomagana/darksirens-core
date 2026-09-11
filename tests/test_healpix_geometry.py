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
