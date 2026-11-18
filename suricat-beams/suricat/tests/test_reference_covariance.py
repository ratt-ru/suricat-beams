"""
The calibration-reference covariance of get_rotation_averaged_beam.

Synthetic throughout. The beam is elliptical, so that parallactic rotation makes every
off-axis direction's gain vary in time (a circular beam would give zero variance, and
these tests would prove nothing).
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from suricat.beams import BeamWizard, mdv_beams_to_bds

FREQS_MHZ = np.array([900.0, 1100.0, 1300.0])
NTIMES = 8
# the l/m grid, in degrees: the reference positions below are grid points
GRID = np.linspace(-1.0, 1.0, 9)
REF_LM = (0.5, -0.25)


@pytest.fixture(scope="module")
def bds(tmp_path_factory):
    margin = np.linspace(-5, 5, 41)
    yy, xx = np.meshgrid(margin, margin, indexing="ij")
    beam = np.zeros((4, 2, len(FREQS_MHZ), len(margin), len(margin)), np.complex64)
    for f, freq in enumerate(FREQS_MHZ):
        scale = 1000 / freq
        gain = np.exp(-(xx**2 / (2 * (1.2 * scale)**2) + yy**2 / (2 * (2.4 * scale)**2)))
        beam[0, :, f] = gain           # HH
        beam[3, :, f] = gain           # VV
    npz = tmp_path_factory.mktemp("mdv") / "beams.npz"
    np.savez(npz, beam=beam, pols=np.array([b"HH", b"HV", b"VH", b"VV"]),
             antnames=np.array([b"m000", b"array_average"]),
             freq_MHz=FREQS_MHZ, margin_deg=margin)
    path = str(tmp_path_factory.mktemp("bds") / "beams.bds.zarr")
    mdv_beams_to_bds(str(npz), path)
    return path


@pytest.fixture(scope="module")
def wizard(bds):
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = hdr["NAXIS2"] = 32
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---SIN", "DEC--SIN"
    hdr["CRPIX1"] = hdr["CRPIX2"] = 17
    hdr["CRVAL1"], hdr["CRVAL2"] = 201.365, -43.019
    hdr["CDELT1"], hdr["CDELT2"] = -0.05, 0.05
    hdr["CUNIT1"] = hdr["CUNIT2"] = "deg"
    # six hours around transit: the parallactic angle sweeps far enough to matter
    times = Time(60000.6 + np.arange(NTIMES) * 0.03, format="mjd")
    return BeamWizard(bds, header=hdr, times=times)


def _beam(wizard, **kwargs):
    # pixel_stepping 1: every grid point computed, so a reference on a grid point is compared
    # with that point exactly rather than with an interpolation
    return wizard.get_rotation_averaged_beam(l=GRID, m=GRID, spi=-0.7, num_freq=3, time_stepping=1, pixel_stepping=1,
                                            ncpu=1, **kwargs)


def _grid_index(lm):
    return int(np.argmin(np.abs(GRID - lm[0]))), int(np.argmin(np.abs(GRID - lm[1])))


def test_without_a_reference_nothing_changes(wizard):
    mean, var = _beam(wizard)
    mean_none, var_none = _beam(wizard, reference_lm=None, reference_series=None)
    np.testing.assert_array_equal(mean, mean_none)
    np.testing.assert_array_equal(var, var_none)
    assert var.max() > 0, "the synthetic beam must vary in time, or these tests prove nothing"


def test_a_reference_covaries_with_itself_by_its_own_variance(wizard):
    mean, var = _beam(wizard)
    mean_r, var_r, cov, ref = _beam(wizard, reference_lm=REF_LM)
    # the reference adds outputs; it does not change the mean and variance
    np.testing.assert_allclose(mean_r, mean)
    np.testing.assert_allclose(var_r, var)
    i, j = _grid_index(REF_LM)
    assert ref["series"].shape == (NTIMES,)
    np.testing.assert_allclose(ref["mean"], mean[i, j], rtol=1e-6)
    np.testing.assert_allclose(ref["var"], var[i, j], rtol=1e-6)
    np.testing.assert_allclose(cov[i, j], var[i, j], rtol=1e-6)
    # Cauchy-Schwarz holds everywhere
    assert np.all(np.abs(cov) <= np.sqrt(var * ref["var"]) * (1 + 1e-6) + 1e-12)


def test_an_explicit_series_matches_the_model_reference(wizard):
    _, _, cov_lm, ref = _beam(wizard, reference_lm=REF_LM)
    _, _, cov_series, ref_series = _beam(wizard, reference_series=ref["series"])
    np.testing.assert_allclose(cov_series, cov_lm, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(ref_series["var"], ref["var"], rtol=1e-6)


def test_the_series_is_subsampled_with_the_times(wizard):
    _, _, _, ref = _beam(wizard, reference_lm=REF_LM)
    stepped = wizard.get_rotation_averaged_beam(l=GRID, m=GRID, spi=-0.7, num_freq=3, time_stepping=2, pixel_stepping=1,
                                                ncpu=1, reference_series=ref["series"])
    np.testing.assert_allclose(stepped[3]["series"], ref["series"][::2])


def test_reference_arguments_are_checked(wizard):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _beam(wizard, reference_lm=REF_LM, reference_series=np.ones(NTIMES))
    with pytest.raises(ValueError, match="does not match"):
        _beam(wizard, reference_series=np.ones(NTIMES - 1))


def test_the_reference_arguments_are_keyword_only():
    """Existing positional calls (..., var, i, j) must bind exactly as before the reference arrived."""
    import inspect
    params = inspect.signature(BeamWizard.get_rotation_averaged_beam).parameters
    names = list(params)
    assert names[names.index("var"):names.index("var") + 3] == ["var", "i", "j"]
    for name in ("reference_lm", "reference_series"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
