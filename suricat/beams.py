import pickle
from typing import Any, List, Optional, Union
from dataclasses import dataclass
import os.path
import numpy as np
import numpy.linalg
import scipy.interpolate
import wget
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
from scipy.ndimage import spline_filter, map_coordinates
import xarray

from scabha.schema_utils import clickify_parameters
from .main import cli, schemas

@dataclass
class PowerBeam(object):
    """Power beam info"""
    I: np.ndarray      # Stokes I beam, of shape NFREQ x NDEG x NDEG
    deg: np.ndarray    # coordinates in beam
    freq: np.ndarray   # frequencies

def download_mdv_beams(source: str, dest: Optional[str] = None):
    """Downloads MdV beams from SARAO archive

    Args:
        source (str): Full URL, or filename (e.g. MeerKAT_U_band_primary_beam.npz), or band (e.g. U)
        dest (Optional[str], optional): destination file, defaults to basename of filename
    """
    from . import log
    if "://" in source:
        url = source
    elif source.endswith(".npz"):
        url = f"https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/data/{source}"
    elif source in ("L", "U", "S0", "S1", "S2", "S3", "S4"):
        url = f"https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/data/MeerKAT_{source}_band_primary_beam.npz"
    if dest is None:
        dest = os.path.basename(url)

    log.info(f"downloading {url} to {dest}")
    wget.download(url, out=dest)
    log.info(f"download complete")

@cli.command("download", help=schemas.cabs.get("suricat.download-mdv-beams").info)
@clickify_parameters(schemas.cabs.get("suricat.download-mdv-beams"))
def _download_mdv_beams(*args, **kw):
    return download_mdv_beams(*args, **kw)

def mdv_beams_to_bds(mdv_beams: str, bds: str):
    """
    Converts MdV's npz beamset into a Stokes I power beam
    """
    from . import LOGGER
    LOGGER.info(f"loading MdV beams from {mdv_beams}")
    mdv = np.load(mdv_beams)
    bm = mdv['beam']
    degs = mdv['margin_deg']
    freqs = mdv['freq_MHz']*1e+6
    delta = degs[1] - degs[0]
    i0 = len(degs) // 2

    # form up fits header
    hdr = {}
    hdr['SIMPLE'] = 'T'
    hdr['NAXIS1'] = len(degs) 
    hdr['NAXIS2'] = len(degs)
    hdr['NAXIS3'] = len(freqs)
    hdr['CRPIX1'] = i0+1
    hdr['CRPIX2'] = i0+1
    hdr['CRPIX3'] = 1
    hdr['CRVAL1'] = 0
    hdr['CRVAL2'] = 0
    hdr['CRVAL3'] = freqs[0]
    hdr['CDELT1'] = delta
    hdr['CDELT2'] = delta
    hdr['CDELT3'] = freqs[1] - freqs[0]
    hdr['CTYPE1'] = 'X'
    hdr['CTYPE2'] = 'Y'
    hdr['CTYPE3'] = 'FREQ'
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'
    hdr['CUNIT3'] = 'Hz'

    # See also https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/beam_orientation_diagram.pdf
    # MdV pols are HH, HV, VH, VV, so I think that corresponds to [[HH, HV],[VH,VV]] in the Jones matrix

    LOGGER.info(f"computing normalized beams")
    bm = bm[:,-1]   # select average beam (last antenna index)
    jj = bm.reshape([2,2] + list(bm.shape[1:]))  # reshape to 2x2 to get Jones matrix
    # MdV axes are FREQ,Y,X (probably worth double-checking), so now ROW,COL,FREQ,Y,X
    jjt = jj.transpose((2,3,4,0,1))  # now FREQ,Y,X,ROW,COLUMN
    jj0 = jjt[:, i0, i0, :, :]       # centre beam: FREQ,ROW,COLUMN 
    # linalg.inv() wants last two axes to be matrix row and column, so transpose
    jj0inv = numpy.linalg.inv(jj0)
    # normalized Jones matrix (Jnorm.J)
    jnorm = jj0inv[:, np.newaxis, np.newaxis, :, :] @ jjt

    LOGGER.info(f"computing Stokes beams")
    # S converts Stokes to coherency
    S = np.array([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]])
    # Sinv converts coherency to Stokes
    Sinv = numpy.linalg.inv(S)
    # LOGGER.info(S)
    # LOGGER.info(Sinv)

    # compute Stokes matrices from FREQ,Y,X,ROW,COLUMN Jones matrices
    def stokes(jones):
        # LOGGER.info(f"J={jones[0, i0, i0]}")
        # LOGGER.info(f"JH={np.conj(jones).transpose((0,1,2,4,3))[0, i0, i0]}")
        mshape = list(jones.shape[:-2]) + [4, 4]
        mueller = np.einsum('fyxij,fyxkl->fyxikjl', jones, np.conj(jones).transpose((0,1,2,4,3))).reshape(mshape)
        # LOGGER.info(f"M={mueller[0, i0, i0]}")
        return Sinv @ mueller @ S              

    # compute Stokes and normalized Stokes
    st = stokes(jjt).transpose((3,4,0,1,2)).astype(np.float32)
    stnorm = stokes(jnorm).transpose((3,4,0,1,2)).astype(np.float32)
    # LOGGER.info(stnorm[:,:,0,i0,i0])
    jnorm = jnorm.transpose((3,4,0,1,2))  # back to ROW,COLUMN,FREQ,Y,X

    LOGGER.info(f"saving output dataset {bds}")
    # write to dataset
    jcoords = dict(i=[0,1], j=[0,1], X=degs, Y=degs, FREQ=freqs)
    scoords = dict(i=list("IQUV"), j=list("IQUV"), X=degs, Y=degs, FREQ=freqs)

    xds = xarray.Dataset(dict(
        jones = xarray.DataArray(jj, dims=("i", "j", "FREQ", "Y", "X"), coords=jcoords),
        njones = xarray.DataArray(jnorm, dims=["i", "j", "FREQ", "Y", "X"], coords=jcoords),
        stokes = xarray.DataArray(st, dims=["i", "j", "FREQ", "Y", "X"], coords=scoords),
        nstokes = xarray.DataArray(stnorm, dims=["i", "j", "FREQ", "Y", "X"], coords=scoords),
    ))
    xds.attrs["fits_header"] = hdr
    xds.attrs.update(x0=i0, y0=i0, dx=delta, dy=delta, freqs=freqs)
    xds.to_zarr(bds, mode="w")

@cli.command("mdv2bds", help=schemas.cabs.get("suricat.mdv-beams-to-bds").info)
@clickify_parameters(schemas.cabs.get("suricat.mdv-beams-to-bds"))
def _mdv_beams_to_bds(*args, **kw):
    return mdv_beams_to_bds(*args, **kw)


class BeamWizard(object):
    """Attaches to a BDS and provides various convenienece functions"""
    Eband: np.ndarray          # per-band power beam
    Emean: np.ndarray          # mean MFS beam
    band_weights: np.ndarray   # per-band weights
    freqs: np.ndarray          # band frequencies
    x0: int       # center pixel of beam
    y0: int
    delta: float  # degrees per pixel
    ra0: float    # field centre in degrees
    dec0: float

    def __init__(self, bds_name: str, image_name: str):
        from . import log
        self.log = log
        log.info(f"opening BDS {bds_name}")
        self.bds = xarray.open_zarr(bds_name)
        freqs = self.bds.coords['FREQ'].values
        log.info(f"frequency range is {freqs[0]*1e-6:.0f} to {freqs[-1]*1e-6:.0f} MHz")
        self.index_to_freq = scipy.interpolate.interp1d(np.arange(len(freqs)), freqs)
        self.freq_to_index = scipy.interpolate.interp1d(freqs, np.arange(len(freqs)))
                 
        if image_name.endswith(".fits"):
            log.info(f"obtaining WCS from FITS image {image_name}")
            fitshdr = fits.open(image_name)[0].header
            self.wcs = WCS(fitshdr)
            self.time = None
        elif image_name.endswith(".zarr") or image_name.endswith(".zarr/"):
            log.info(f"obtaining WCS from dataset {image_name}")
            ds = xarray.open_zarr(image_name)
            fitshdr = fits.Header(dict(ds.attrs['fits_header']))
            self.wcs = WCS(fitshdr)
            self.times = Time(ds.coords['TIME'].values/(24*3600), format='mjd')
            log.info(f"time axis is {self.times[0].iso} to {self.times[-1].iso}")
        else:
            raise RuntimeError(f"unable to determine type of image {image_name}")
        # drop WCS axes >2
        while len(self.wcs.axis_type_names) > 2:
            log.debug(f"dropping WCS axis {self.wcs.axis_type_names[-1]}")
            self.wcs = self.wcs.dropaxis(len(self.wcs.axis_type_names) - 1)
        self.centre = self.wcs.pixel_to_world(fitshdr['CRPIX1'] - 1, fitshdr['CRPIX2'] - 1)
        log.info(f"image centre is at {self.centre}")
        # location could be made configurable        
        self.default_location = EarthLocation.of_site("MeerKAT") 
        log.info(f"location is MeerKAT ({self.default_location})")
        self._prefilters = {}

    def _get_prefilter(self, var: str, i: Union[str, int], j: Union[str, int]):
        key = var, i ,j
        if key not in self._prefilters:
            self.log.info(f"computing spline prefilter for {var}[{i},{j}]")
            self._prefilters[key] = spline_filter(self.bds[var].sel(i=i, j=j))
        return self._prefilters[key]

    def get_source_coodinates(self, srcpos: SkyCoord, 
                                    times: Optional[Time] = None, 
                                    loc: EarthLocation = None,
                                    signs=(1,1), swap=False):
        """
        Given a sky position and a list of times, derives the in-beam coordinates of the source (in beam pixels)
        """
        if loc is None:
            loc = loc or self.default_location
        if times is None:
            if self.times is None:
                raise RuntimeError("explicit times must be supplied, since BeamWizard was constructed without observational time info")
            times = self.times
        # convert positions to alt-az
        frame = AltAz(obstime=times, location=loc)       
        altaz_src = srcpos.transform_to(frame)
        altaz_centre = self.centre.transform_to(frame)
        # get angle and separation of source w.r.t. centre
        angles = altaz_centre.position_angle(altaz_src)
        seps = altaz_centre.separation(altaz_src)
        # convert to pixel position 
        # confused about angles, but experiments show that 0 is up and +90 is right 
        x = signs[0] * seps.deg * np.sin(angles.rad)
        y = signs[1] * seps.deg * np.cos(angles.rad)
        if swap:
            x, y = y, x
        xp = x / self.bds.attrs['dx']  + self.bds.attrs['x0']
        yp = y / self.bds.attrs['dy']  + self.bds.attrs['y0']
        return np.array([xp, yp]), seps, angles

    def interpolate_beam(self, xpyp: np.array, freq: List[float], var: str = 'nstokes', i="I", j="I"):
        # beam is I,J,FREQ,Y,X
        freq = self.freq_to_index(freq)
        fx = np.meshgrid(freq, xpyp[0], indexing='ij')   # mesh freq,x
        fy = np.meshgrid(freq, xpyp[1], indexing='ij')   # mesh freq,y 
        coords = np.vstack([fy] + [fx[1:]])  # mesh freq,yx
        return map_coordinates(self._get_prefilter(var, i, j), coords, prefilter=True)

    def get_time_variable_beamgain(self, srcpos: SkyCoord, mjds: List[float], 
                                        loc: EarthLocation = None, spi: Optional[float] = None):
        xpyp = self.get_source_pixel_coodinates(srcpos, mjds, loc)
        if spi is None:
        # interpolate mean beam
            return map_coordinates(self._prefilter_emean, xpyp, prefilter=True), xpyp
        else:
            bg_freq = np.zeros((len(self.freqs), len(mjds)))
            for i, freq in enumerate(self.freqs):
                bg_freq[i] = map_coordinates(self._prefilter_eband[i], xpyp, prefilter=True)
            spectral_weights = (self.freqs/self.freqs[0])**spi
            weights = self.band_weights*spectral_weights
            return (bg_freq*weights[:,np.newaxis]).sum(axis=0) / weights.sum(), xpyp


