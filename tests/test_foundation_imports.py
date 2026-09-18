from __future__ import annotations

import subprocess
import sys


def test_package_root_does_not_import_jax():
    code = "import sys, darksirens; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cosmology_grid_guard_does_not_import_jax():
    # Cosmology validates its bounds against the tabulated distance grid, which
    # only stays light because the grid support lives in _cosmology_support.
    code = (
        "import sys, darksirens as ds; ds.Cosmology(Om0=(0.2, 0.4));"
        " assert 'jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cosmology_public_names():
    import darksirens.cosmology as c

    assert c.H0_FID == 67.74
    assert c.OM0_FID == 0.3075
    assert c.W0_FID == -1.0
    assert c.WA_FID == 0.0
    assert c.luminosity_distance is c.dL_of_z
    assert c.comoving_distance is c.r_of_z
    assert c.redshift_from_luminosity_distance is c.z_of_dL
