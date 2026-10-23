"""BeamWizard.interpolate_beam returns the beam dataset's own values at its grid nodes.

Synthetic, so no beam downloads are needed.
"""
import numpy as np
import pytest
import xarray
from astropy.io import fits

from suricat.beams import BeamWizard, mdv_beams_to_bds

FREQS_MHZ = np.array([900.0, 1100.0, 1300.0])
MARGIN = np.linspace(-5, 5, 21)


@pytest.fixture(scope="module")
def bds(tmp_path_factory):
    """A beam dataset from a small MdV-style beam set: a narrow Gaussian main beam,
    sampled coarsely enough that spline interpolation has work to do."""
    yy, xx = np.meshgrid(MARGIN, MARGIN, indexing="ij")
    beam = np.zeros((4, 2, len(FREQS_MHZ), len(MARGIN), len(MARGIN)), np.complex64)
    for f, freq in enumerate(FREQS_MHZ):
        gain = np.exp(-(xx**2 + yy**2) / (2 * (1.2 * 1000 / freq)**2))
        beam[0, :, f] = gain
        beam[3, :, f] = 0.9 * gain
    d = tmp_path_factory.mktemp("bds")
    np.savez(d / "beams.npz", beam=beam, pols=np.array([b"HH", b"HV", b"VH", b"VV"]),
             antnames=np.array([b"m000", b"array_average"]), freq_MHz=FREQS_MHZ, margin_deg=MARGIN)
    mdv_beams_to_bds(str(d / "beams.npz"), str(d / "beams.bds.zarr"))
    return str(d / "beams.bds.zarr")


@pytest.fixture(scope="module")
def wizard(bds, tmp_path_factory):
    hdr = fits.Header()
    hdr.update(NAXIS=2, NAXIS1=16, NAXIS2=16, CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CRPIX1=9,
               CRPIX2=9, CRVAL1=201.365, CRVAL2=-43.019, CDELT1=-0.05, CDELT2=0.05,
               CUNIT1="deg", CUNIT2="deg")
    path = str(tmp_path_factory.mktemp("img") / "image.fits")
    fits.PrimaryHDU(np.zeros((16, 16), np.float32), hdr).writeto(path)
    return BeamWizard(bds, path)


def test_grid_nodes_return_the_data(bds, wizard):
    nstokes = xarray.open_zarr(bds).nstokes.sel(stokes_i="I", stokes_j="I").values  # FREQ, Y, X
    yy, xx = np.meshgrid(np.arange(len(MARGIN)), np.arange(len(MARGIN)), indexing="ij")
    xpyp = np.array([xx.ravel(), yy.ravel()], float)
    freqs = FREQS_MHZ * 1e6
    got = wizard.interpolate_beam(xpyp, freqs).reshape(nstokes.shape)
    np.testing.assert_allclose(got, nstokes, rtol=0, atol=1e-6)


def test_beam_centre_is_unity(wizard):
    x0, y0 = wizard.bds.attrs["x0"], wizard.bds.attrs["y0"]
    got = wizard.interpolate_beam(np.array([[x0], [y0]], float), FREQS_MHZ * 1e6)
    np.testing.assert_allclose(got.ravel(), 1, atol=1e-6)


def test_between_nodes_stays_within_neighbours(wizard):
    """Half a pixel off the peak the value lies between its neighbours' values, below 1."""
    x0, y0 = wizard.bds.attrs["x0"], wizard.bds.attrs["y0"]
    got = wizard.interpolate_beam(np.array([[x0 + 0.5], [y0]], float), FREQS_MHZ * 1e6).ravel()
    right = wizard.interpolate_beam(np.array([[x0 + 1.0], [y0]], float), FREQS_MHZ * 1e6).ravel()
    assert np.all(got < 1) and np.all(got > right)
