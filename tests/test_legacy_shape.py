"""shape_from_tractor: Tractor morphology -> sky-frame Sersic shape."""
import numpy as np
import pytest

from sedphot.catalogs.legacy import shape_from_tractor


# A real ls_dr9.tractor SER source: |e| = 0.552 at PA 80.6 deg east of north.
E = 0.552
PHI = np.radians(2 * 80.6)
E1, E2 = E * np.cos(PHI), E * np.sin(PHI)


def test_ser_uses_sersic_column():
    s = shape_from_tractor('SER', 3.04, 4.80, E1, E2)
    assert s['n'] == pytest.approx(3.04)
    assert s['reff_arcsec'] == pytest.approx(4.80)
    assert s['ellip'] == pytest.approx(1 - (1 - E) / (1 + E), abs=1e-6)
    assert s['pa_deg'] == pytest.approx(80.6, abs=0.01)


def test_type_fixed_indices():
    assert shape_from_tractor('DEV', np.nan, 2.0, 0.1, 0.0)['n'] == 4.0
    assert shape_from_tractor('EXP', np.nan, 2.0, 0.1, 0.0)['n'] == 1.0
    rex = shape_from_tractor('REX', np.nan, 2.0, 0.0, 0.0)
    assert rex['n'] == 1.0
    assert rex['ellip'] == 0.0


def test_pa_range_and_whitespace_type():
    s = shape_from_tractor('SER ', 2.0, 3.0, 0.3, -0.4)
    assert 0.0 <= s['pa_deg'] < 180.0


def test_unusable_types_and_shapes():
    assert shape_from_tractor('PSF', 0.0, 0.0, 0.0, 0.0) is None
    assert shape_from_tractor('DUP', 0.0, 1.0, 0.0, 0.0) is None
    assert shape_from_tractor('SER', np.nan, 4.8, 0.1, 0.0) is None
    assert shape_from_tractor('EXP', 1.0, 0.0, 0.1, 0.0) is None
    assert shape_from_tractor('DEV', 4.0, np.nan, 0.1, 0.0) is None
    assert shape_from_tractor('EXP', 1.0, 2.0, 0.8, 0.7) is None
