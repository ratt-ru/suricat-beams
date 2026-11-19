"""
The calibration-reference covariance of get_rotation_averaged_beam.

Synthetic throughout. The beam is an elliptical Gaussian rotated off the axes, with a weak
off-centre lobe: parallactic rotation then makes every off-axis direction's gain vary in
time (a circular beam would give zero variance, and these tests would prove nothing), and
no reflection or point inversion of the sky maps the beam onto itself, so a sign or
orientation slip in the sky-to-beam transform between the pixel tracks and the reference
track changes the results.
"""
import inspect

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
# the two band modes: a spectral index collapses frequency, None keeps one plane per frequency
SPI_MODES = pytest.mark.parametrize("spi", [-0.7, None], ids=["spi", "per-freq"])


@pytest.fixture(scope="module")
def bds(tmp_path_factory):
    margin = np.linspace(-5, 5, 41)
    yy, xx = np.meshgrid(margin, margin, indexing="ij")
    # the ellipse's axes rotated by 30 degrees, and a lobe off the centre and off the axes
    c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    u, v = c * xx + s * yy, -s * xx + c * yy
    beam = np.zeros((4, 2, len(FREQS_MHZ), len(margin), len(margin)), np.complex64)
    for f, freq in enumerate(FREQS_MHZ):
        scale = 1000 / freq
        gain = (np.exp(-(u**2 / (2 * (1.2 * scale)**2) + v**2 / (2 * (2.4 * scale)**2)))
                + 0.15 * np.exp(-((xx - 1.0)**2 + (yy - 0.6)**2) / (2 * (0.5 * scale)**2)))
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


def _beam(wizard, spi=-0.7, pixel_stepping=1, time_stepping=1, **kwargs):
    # pixel_stepping 1 by default: every grid point computed, so a reference on a grid
    # point is compared with that point exactly rather than with an interpolation
    return wizard.get_rotation_averaged_beam(l=GRID, m=GRID, spi=spi, num_freq=3,
                                            time_stepping=time_stepping,
                                            pixel_stepping=pixel_stepping, ncpu=1, **kwargs)


def _grid_index(lm):
    return int(np.argmin(np.abs(GRID - lm[0]))), int(np.argmin(np.abs(GRID - lm[1])))


@SPI_MODES
def test_without_a_reference_nothing_changes(wizard, spi):
    mean, var = _beam(wizard, spi=spi)
    mean_none, var_none = _beam(wizard, spi=spi, reference_lm=None, reference_series=None)
    np.testing.assert_array_equal(mean, mean_none)
    np.testing.assert_array_equal(var, var_none)
    assert var.max() > 0, "the synthetic beam must vary in time, or these tests prove nothing"


@SPI_MODES
def test_a_reference_covaries_with_itself_by_its_own_variance(wizard, spi):
    mean, var = _beam(wizard, spi=spi)
    mean_r, var_r, cov, ref = _beam(wizard, spi=spi, reference_lm=REF_LM)
    # the reference adds outputs; it does not change the mean and variance
    np.testing.assert_allclose(mean_r, mean)
    np.testing.assert_allclose(var_r, var)
    assert cov.shape == var.shape
    i, j = _grid_index(REF_LM)
    # per frequency, the track is (times, frequencies) and the maps carry frequency first
    assert ref["series"].shape == ((NTIMES,) if spi is not None else (NTIMES, len(FREQS_MHZ)))
    np.testing.assert_allclose(ref["mean"], mean[..., i, j], rtol=1e-6)
    np.testing.assert_allclose(ref["var"], var[..., i, j], rtol=1e-6)
    np.testing.assert_allclose(cov[..., i, j], var[..., i, j], rtol=1e-6)


def test_the_fixture_tells_a_direction_from_its_reflections(wizard):
    """Not symmetric under l -> -l, m -> -m or both, so the tests above pin the geometry
    the pixel and reference tracks share, not only their bookkeeping."""
    tracks = [_beam(wizard, reference_lm=(sl * REF_LM[0], sm * REF_LM[1]))[3]["series"]
              for sl, sm in ((1, 1), (-1, 1), (1, -1), (-1, -1))]
    for other in tracks[1:]:
        assert not np.allclose(tracks[0], other, rtol=1e-3)


@SPI_MODES
def test_an_explicit_series_matches_the_model_reference(wizard, spi):
    _, _, cov_lm, ref = _beam(wizard, spi=spi, reference_lm=REF_LM)
    if spi is not None:
        _, _, cov_series, ref_series = _beam(wizard, spi=spi, reference_series=ref["series"])
        np.testing.assert_allclose(cov_series, cov_lm, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(ref_series["var"], ref["var"], rtol=1e-6)
    else:
        # a supplied series is one track for every frequency: the first frequency's model
        # track reproduces that frequency's covariance
        _, _, cov_series, _ = _beam(wizard, spi=spi, reference_series=ref["series"][:, 0])
        np.testing.assert_allclose(cov_series[0], cov_lm[0], rtol=1e-6, atol=1e-12)


@SPI_MODES
def test_pixel_stepping_reproduces_the_computed_nodes(wizard, spi):
    _, var, cov, _ = _beam(wizard, spi=spi, reference_lm=REF_LM)
    _, var_s, cov_s, _ = _beam(wizard, spi=spi, pixel_stepping=2, reference_lm=REF_LM)
    assert cov_s.shape == cov.shape and var_s.shape == var.shape
    # nodes 0, 2, 4, ... are computed; the linear upsampling passes through them
    np.testing.assert_allclose(cov_s[..., ::2, ::2], cov[..., ::2, ::2], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(var_s[..., ::2, ::2], var[..., ::2, ::2], rtol=1e-6, atol=1e-12)


def test_the_series_is_subsampled_with_the_times(wizard):
    _, _, _, ref = _beam(wizard, reference_lm=REF_LM)
    _, _, cov_lm, ref_lm = _beam(wizard, time_stepping=2, reference_lm=REF_LM)
    _, _, cov_series, ref_series = _beam(wizard, time_stepping=2, reference_series=ref["series"])
    # the full-length series is thinned exactly as the times are, so its samples line up
    np.testing.assert_allclose(ref_series["series"], ref["series"][::2])
    np.testing.assert_allclose(ref_series["series"], ref_lm["series"], rtol=1e-6)
    np.testing.assert_allclose(cov_series, cov_lm, rtol=1e-6, atol=1e-12)


def test_reference_arguments_are_checked(wizard):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _beam(wizard, reference_lm=REF_LM, reference_series=np.ones(NTIMES))
    with pytest.raises(ValueError, match="does not match"):
        _beam(wizard, reference_series=np.ones(NTIMES - 1))
    flagged = np.ones(NTIMES)
    flagged[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _beam(wizard, reference_series=flagged)
    with pytest.raises(ValueError, match="positive"):
        _beam(wizard, reference_series=-np.ones(NTIMES))


def test_a_reference_off_the_beam_cube_raises(wizard):
    # the synthetic cube spans +/-5 deg: at 7 deg the interpolation would return 0
    with pytest.raises(ValueError, match="outside the beam cube"):
        _beam(wizard, reference_lm=(7.0, 0.0))


def test_the_reference_arguments_are_keyword_only():
    """Existing positional calls (..., var, i, j) must bind exactly as before the reference arrived."""
    params = inspect.signature(BeamWizard.get_rotation_averaged_beam).parameters
    names = list(params)
    assert names[names.index("var"):names.index("var") + 3] == ["var", "i", "j"]
    for name in ("reference_lm", "reference_series"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
