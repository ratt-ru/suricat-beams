"""
Zarr stores written by suricat, and the ways BeamWizard learns an image's WCS.

Run under zarr-python 2 or 3: everything is synthetic, so no beam downloads are
needed. Under zarr-python 3 the stores suricat writes are zarr v3.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray
import zarr
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from suricat.beams import BeamWizard, mdv_beams_to_bds
from suricat.xradio_util import bds_to_xradio, mdv_to_xradio
from suricat.zarr_compat import BEAM_COMPRESSOR, BEAM_FILTERS, ZARR3

FREQS_MHZ = np.array([900.0, 1100.0, 1300.0])
TIME0_MJD = 60000.75
NTIMES = 3


def _store_format(path):
    """The on-disk zarr format of a store: 2 or 3."""
    return 3 if (Path(path) / "zarr.json").exists() else 2


@pytest.fixture(scope="module")
def mdv_npz(tmp_path_factory):
    """A small MdV-style beam set: Gaussian main beam with weak leakage, two antennas."""
    margin = np.linspace(-5, 5, 21)
    yy, xx = np.meshgrid(margin, margin, indexing="ij")
    beam = np.zeros((4, 2, len(FREQS_MHZ), len(margin), len(margin)), np.complex64)
    for f, freq in enumerate(FREQS_MHZ):
        gain = np.exp(-(xx**2 + yy**2) / (2 * (2.0 * 1000 / freq)**2))
        beam[0, :, f] = gain           # HH
        beam[1, :, f] = 0.05 * gain    # HV
        beam[2, :, f] = 0.03 * gain    # VH
        beam[3, :, f] = 0.9 * gain     # VV
    path = tmp_path_factory.mktemp("mdv") / "beams.npz"
    np.savez(path, beam=beam, pols=np.array([b"HH", b"HV", b"VH", b"VV"]),
             antnames=np.array([b"m000", b"array_average"]),
             freq_MHz=FREQS_MHZ, margin_deg=margin)
    return str(path)


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "compressed"])
def bds(request, mdv_npz, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("bds") / "beams.bds.zarr")
    mdv_beams_to_bds(mdv_npz, path, compress=request.param)
    return path


def _image_header():
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = hdr["NAXIS2"] = 32
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---SIN", "DEC--SIN"
    hdr["CRPIX1"] = hdr["CRPIX2"] = 17
    hdr["CRVAL1"], hdr["CRVAL2"] = 201.365, -43.019
    hdr["CDELT1"], hdr["CDELT2"] = -0.05, 0.05
    hdr["CUNIT1"] = hdr["CUNIT2"] = "deg"
    return hdr


@pytest.fixture(scope="module")
def times():
    return Time(TIME0_MJD + np.arange(NTIMES) * 0.05, format="mjd")


@pytest.fixture(scope="module")
def fits_image(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("img") / "image.fits")
    fits.PrimaryHDU(np.zeros((32, 32), np.float32), _image_header()).writeto(path)
    return path


@pytest.fixture(scope="module")
def zarr_image(tmp_path_factory, times):
    """A cube in the legacy Breifast/pfb layout: fits_header attribute, TIME in MJD seconds."""
    path = str(tmp_path_factory.mktemp("img") / "cube.zarr")
    ds = xarray.Dataset(
        {"cube": (("TIME", "Y", "X"), np.zeros((NTIMES, 32, 32), np.float32))},
        coords={"TIME": times.mjd * 86400})
    ds.attrs["fits_header"] = list(dict(_image_header()).items())
    ds.to_zarr(path)
    return path


def test_bds_format_follows_zarr_python(bds):
    assert _store_format(bds) == (3 if ZARR3 else 2)
    ds = xarray.open_zarr(bds)
    assert set(ds.data_vars) == {"jones", "njones", "stokes", "nstokes"}
    nstokes = ds.nstokes.sel(stokes_i="I", stokes_j="I").values
    assert np.isfinite(nstokes).all()
    # normalised to unity at the beam centre
    assert np.allclose(nstokes[:, 10, 10], 1, atol=1e-5)


def test_compression_keeps_values(mdv_npz, tmp_path):
    plain, packed = str(tmp_path / "plain.zarr"), str(tmp_path / "packed.zarr")
    mdv_beams_to_bds(mdv_npz, plain, compress=False)
    mdv_beams_to_bds(mdv_npz, packed, compress=True)
    a, b = xarray.open_zarr(plain), xarray.open_zarr(packed)
    for var in ("jones", "njones"):
        # complex beams: Blosc only, lossless
        np.testing.assert_array_equal(a[var].values, b[var].values)
    for var in ("stokes", "nstokes"):
        # the Delta filter differences and re-sums in float32, so not bit-exact
        np.testing.assert_allclose(a[var].values, b[var].values, rtol=0, atol=1e-6)


def test_header_and_times_match_fits_and_zarr_images(bds, fits_image, zarr_image, times):
    by_fits = BeamWizard(bds, fits_image)
    by_zarr = BeamWizard(bds, zarr_image)
    by_header = BeamWizard(bds, header=_image_header(), times=times)
    for bw in (by_zarr, by_header):
        assert bw.wcs.to_header() == by_fits.wcs.to_header()
        assert bw.centre.separation(by_fits.centre).arcsec < 1e-6
        np.testing.assert_array_equal(bw.l_grid, by_fits.l_grid)
        np.testing.assert_array_equal(bw.m_grid, by_fits.m_grid)
    np.testing.assert_allclose(by_zarr.times.mjd, times.mjd, rtol=0, atol=1e-9)
    np.testing.assert_allclose(by_header.times.mjd, times.mjd, rtol=0, atol=1e-9)
    assert by_fits.times is None


def test_header_accepts_a_plain_dict(bds, times):
    bw = BeamWizard(bds, header=dict(_image_header()), times=times)
    assert bw.wcs.wcs.ctype[0] == "RA---SIN"


def test_missing_times_raise_a_clear_error(bds, fits_image):
    bw = BeamWizard(bds, fits_image)
    with pytest.raises(RuntimeError, match="explicit times"):
        bw.get_source_coordinates(bw.centre)


def test_image_name_and_header_are_exclusive(bds, fits_image):
    with pytest.raises(ValueError):
        BeamWizard(bds, fits_image, header=_image_header())
    with pytest.raises(ValueError):
        BeamWizard(bds)


def test_zarr_image_without_fits_header_asks_for_one(bds, tmp_path):
    path = str(tmp_path / "native.zarr")
    xarray.Dataset({"SKY": (("time",), np.zeros(2))}, coords={"time": [0.0, 1.0]}).to_zarr(path)
    with pytest.raises(RuntimeError, match="pass its header and times"):
        BeamWizard(bds, path)


@pytest.mark.parametrize("compress", [False, True], ids=["plain", "compressed"])
def test_time_freq_beam_store(bds, times, tmp_path, compress):
    bw = BeamWizard(bds, header=_image_header(), times=times)
    out = str(tmp_path / "beam.zarr")
    freqs = FREQS_MHZ[:2] * 1e6
    bw.get_time_freq_beam(out, "BEAM", l=bw.l_grid, m=bw.m_grid, freq=freqs,
                          pixel_stepping=1, ncpu=1,
                          compressor=BEAM_COMPRESSOR if compress else None,
                          filters=BEAM_FILTERS if compress else None)
    assert _store_format(out) == (3 if ZARR3 else 2)
    ds = xarray.open_zarr(out)
    assert ds.BEAM.dims == ("time", "freq", "ij", "x", "y")
    assert ds.BEAM.shape == (NTIMES, 2, 1, 32, 32)
    assert list(ds.ij.values) == ["II"]
    np.testing.assert_allclose(ds.time.values, times.mjd)
    beam = ds.BEAM.values
    assert np.isfinite(beam).all()
    # the field centre sits at the beam centre: the stored plane holds what
    # interpolate_beam gives there, at every time and frequency
    centre = bw.interpolate_beam(np.array([[bw.bds.attrs["x0"]], [bw.bds.attrs["y0"]]]), freqs)
    for t in range(NTIMES):
        np.testing.assert_allclose(beam[t, :, 0, 16, 16], centre[:, 0], rtol=1e-6)
    assert (beam[:, :, 0, 0, 0] < beam[:, :, 0, 16, 16]).all()


def test_bds_to_xradio(bds, zarr_image, tmp_path):
    out = str(tmp_path / "pb.zarr")
    bds_to_xradio(bds, zarr_image, out, output_var="PRIMARY_BEAM", pixel_stepping=1,
                  ncpu=1, elements=["II", "QQ"], compress=True)
    ds = xarray.open_zarr(out)
    assert ds.PRIMARY_BEAM.dims == ("time", "frequency", "polarization", "l", "m")
    assert list(ds.polarization.values) == ["I", "Q"]
    # l, m converted to radians
    np.testing.assert_allclose(ds.l.values, np.deg2rad(BeamWizard(bds, zarr_image).l_grid))
    assert ds.attrs["direction"]["projection"] == "SIN"


def test_mdv_to_xradio(mdv_npz, tmp_path):
    out = str(tmp_path / "mdv.zarr")
    mdv_to_xradio(mdv_npz, out, jones="HH", part="abs", compress=True)
    assert _store_format(out) == (3 if ZARR3 else 2)
    ds = xarray.open_zarr(out)
    assert ds.SKY.dims == ("time", "frequency", "polarization", "l", "m")
    assert ds.SKY.shape == (1, 3, 1, 21, 21)
    np.testing.assert_allclose(ds.SKY.values[0, :, 0, 10, 10], 1, atol=1e-6)


def test_beam_gains_from_a_header(bds, fits_image, times):
    from suricat.beamplots import collect_beam_gain_to_source
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    coord = SkyCoord(201.6 * u.deg, -43.1 * u.deg)
    by_image = collect_beam_gain_to_source(bds, fits_image, coord, freq=[1.0e9, 1.2e9], time=times[1])
    by_header = collect_beam_gain_to_source(bds, None, coord, freq=[1.0e9, 1.2e9], time=times[1],
                                            header=_image_header())
    assert by_image.keys() == by_header.keys()
    for name in by_image:
        np.testing.assert_array_equal(by_image[name], by_header[name])
