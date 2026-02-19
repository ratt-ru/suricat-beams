from typing import Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import os.path
import sys
import numpy as np
import numpy.linalg
import scipy.interpolate
import wget
from urllib.error import HTTPError
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
from scipy.ndimage import spline_filter, map_coordinates
import xarray
from pathlib import Path

from scabha.schema_utils import clickify_parameters
from .main import cli, schemas

@dataclass
class PowerBeam(object):
    """Power beam info"""
    I: np.ndarray      # Stokes I beam, of shape NFREQ x NDEG x NDEG
    deg: np.ndarray    # coordinates in beam
    freq: np.ndarray   # frequencies

def download_mdv_beams(source: str, dest: Optional[str] = None,
                       base_url: List[str] = 
                        ["https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/data/"],
                       exit_on_error: Optional[int] = 1):
    """Downloads MdV beams from SARAO archive

    Args:
        source (str): Full URL, or filename (e.g. MeerKAT_U_band_primary_beam.npz), or band (e.g. U)
        dest (Optional[str], optional): destination file, defaults to basename of filename
    """
    from . import log
    urls = []
    if "://" in source:
        urls = [source]
    elif source.endswith(".npz"):
        urls = [f"{url.rstrip('/')}/{source}" for url in base_url]
    elif source in ("L", "U", "S0", "S1", "S2", "S3", "S4"):
        urls = [f"{url.rstrip('/')}/MeerKAT_{source}_band_primary_beam.npz" for url in base_url]
    else:
        raise RuntimeError(f"unrecognized source argument: {source}")
    
    if not urls:
        raise RuntimeError(f"no download paths -- did you specify base_url?")
    
    if dest is None:
        dest = os.path.basename(urls[0])

    for url in urls:
        log.info(f"downloading {url} to {dest}")
        try:
            wget.download(url, out=dest)
            log.info(f"download complete")
            return 0
        except HTTPError as exc:
            log.warning(f"download failed: {exc}")

    # if we got here, all downloads failed
    log.error(f"all download atempts failed")
    if exit_on_error is not None:
        sys.exit(exit_on_error)
    else:
        raise RuntimeError(f"all download atempts failed")


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
        elif (Path(image_name) / ".zgroup").exists():
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

        # Construct default l/m grid from image pixels
        nx, ny = fitshdr['NAXIS1'], fitshdr['NAXIS2']
        crpix1, crpix2 = fitshdr['CRPIX1'], fitshdr['CRPIX2']
        cdelt1, cdelt2 = fitshdr['CDELT1'], fitshdr['CDELT2']
        # l/m are offsets from center in degrees (l increases to the east, m to the north)
        self.l_grid: np.ndarray = (np.arange(nx) - (crpix1 - 1)) * cdelt1
        self.m_grid: np.ndarray = (np.arange(ny) - (crpix2 - 1)) * cdelt2
        log.info(f"default l/m grid: {nx}x{ny} pixels, "
                 f"l=[{self.l_grid[0]:.4f}, {self.l_grid[-1]:.4f}], "
                 f"m=[{self.m_grid[0]:.4f}, {self.m_grid[-1]:.4f}] deg")

        # location could be made configurable
        self.default_location = EarthLocation.of_site("MeerKAT")
        log.info(f"location is MeerKAT ({self.default_location})")
        self._prefilters = {}

    def _get_prefilter(self, var: str, i: Union[str, int], j: Union[str, int]):
        key = var, i ,j
        if key not in self._prefilters:
            self.log.debug(f"computing spline prefilter for {var}[{i},{j}]")
            self._prefilters[key] = spline_filter(self.bds[var].sel(i=i, j=j))
        return self._prefilters[key]

    def get_source_coordinates(self, srcpos: SkyCoord, 
                                    times: Optional[Time] = None, 
                                    loc: Optional[EarthLocation] = None,
                                    signs=(1,1), swap=False):
        """
        Given a sky position and a list of times, derives the in-beam coordinates of the source (in beam pixels)
        """
        if loc is None:
            loc = self.default_location
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

    def interpolate_beam(self, xpyp: np.ndarray, freq: Union[List[float], np.ndarray], var: str = 'nstokes', i="I", j="I"):
        # beam is I,J,FREQ,Y,X
        freq = self.freq_to_index(freq)
        fx = np.meshgrid(freq, xpyp[0], indexing='ij')   # mesh freq,x
        fy = np.meshgrid(freq, xpyp[1], indexing='ij')   # mesh freq,y 
        coords = np.vstack([fy] + [fx[1:]])  # mesh freq,yx
        return map_coordinates(self._get_prefilter(var, i, j), coords, prefilter=True)

    def _resolve_freqs(self,
                        freq: Optional[np.ndarray] = None,
                        num_freq: Optional[int] = None,
                        spi: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resolve frequency array and compute spectral weights.

        Helper method that determines which frequencies to use and optionally
        computes spectral weights based on a spectral index.

        Args:
            freq: Explicit frequency array in Hz. If None, uses beam dataset frequencies.
            num_freq: Number of linearly spaced frequencies between min and max
                      beam frequencies. Mutually exclusive with freq.
            spi: Spectral index for computing frequency weights as (freq/freq[0])^spi.
                 If None, no weights are computed.

        Returns:
            Tuple of (frequencies, weights) where:
                - frequencies: 1D array of frequencies in Hz
                - weights: 1D array of normalized spectral weights, or None if spi is None

        Raises:
            ValueError: If both freq and num_freq are specified.
        """
        if freq is not None and num_freq is not None:
            raise ValueError("freq and num_freq are mutually exclusive")
        if freq is None:
            bds_freqs = self.bds.coords['FREQ'].values
            if num_freq is not None:
                freq = np.linspace(bds_freqs[0], bds_freqs[-1], num_freq)
            else:
                freq = bds_freqs
        if spi is not None:
            norm_weights = (freq / freq[0]) ** spi
            norm_weights /= norm_weights.sum()
        else:
            norm_weights = None
        return freq, norm_weights


    def get_time_variable_beamgain(self, coord: SkyCoord,
                                   times: Optional[Time] = None,
                                   loc: Optional[EarthLocation] = None,
                                   freq: Optional[np.ndarray] = None,
                                   num_freq: Optional[int] = None,
                                   spi: Optional[float] = None,
                                   var: str = 'nstokes', i: str = "I", j: str = "I") -> np.ndarray:
        """
        Compute time-variable beam gain for a source at fixed sky coordinates.

        Given a source position, computes the beam gain at that source as a function
        of time. As the parallactic angle changes, the source traces a path through
        the beam, and this method returns the beam value along that path.

        Args:
            coord: Source sky coordinate (RA/Dec)
            times: Times to sample. If None, uses times from image/dataset.
            loc: Observer location. If None, uses MeerKAT.
            freq: Explicit frequency array in Hz. If None, uses beam dataset frequencies.
            num_freq: Number of linearly spaced frequencies. Mutually exclusive with freq.
            spi: Spectral index. If provided, returns frequency-averaged beam gain
                 weighted by (freq/freq[0])^spi.
            var: Beam variable to interpolate ('nstokes', 'stokes', 'njones', 'jones')
            i, j: Stokes or Jones indices (e.g., "I", "Q", 0, 1)

        Returns:
            Beam gain values as np.ndarray:
                - If spi is None: shape (NFREQ, NTIME) - beam gain per frequency and time
                - If spi is not None: shape (NTIME,) - frequency-averaged beam gain per time

        Raises:
            RuntimeError: If times are not available and not provided.
            ValueError: If both freq and num_freq are specified.
        """
        xpyp, seps, angles = self.get_source_coordinates(coord, times=times, loc=loc)

        freq, norm_weights = self._resolve_freqs(freq, num_freq, spi)

        beam_vals = self.interpolate_beam(
            xpyp, freq,
            var=var, i=i, j=j)

        if spi is not None:
            beam_vals = (beam_vals * norm_weights[:, np.newaxis]).sum(axis=0)

        return beam_vals


    def get_rotation_averaged_beam(self, l: Optional[np.ndarray] = None,
                                   m: Optional[np.ndarray] = None,
                                   times: Optional[Time] = None,
                                   loc: Optional[EarthLocation] = None,
                                   freq: Optional[np.ndarray] = None,
                                   num_freq: Optional[int] = None,
                                   spi: Optional[float] = None,
                                   ncpu: Optional[int] = None,
                                   time_stepping: int = 4,
                                   pixel_stepping: int = 4,
                                   chunk_size: int = 1024**2,
                                   var: str = 'nstokes', i: str = "I", j: str = "I") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the rotation-averaged beam at specified l/m coordinates.

        Given a grid of l/m coordinates (in degrees, relative to field center),
        computes the average beam value at each position by rotating through
        the parallactic angles corresponding to the given times.

        Args:
            l: 1D or 2D array of l coordinates in degrees (east offset from center).
               If None, uses image grid. If 1D, will be meshed with m.
            m: 1D or 2D array of m coordinates in degrees (north offset from center).
               If None, uses image grid. If 1D, will be meshed with l.
            times: Times to average over. If None, uses times from image/dataset.
            loc: Observer location. If None, uses MeerKAT.
            freq: Explicit frequency array in Hz. If None, uses beam dataset frequencies.
            num_freq: Number of linearly spaced frequencies. Mutually exclusive with freq.
            spi: Spectral index. If provided, averages over frequency with weights
                 (freq/freq[0])^spi, returning 2D spatial arrays.
            ncpu: Number of CPUs for parallel computation. If None, uses half of
                  available cores (physical cores, assuming hyperthreading).
            time_stepping: Use every Nth timeslot (default 4) to reduce computation
                           while maintaining representative parallactic angle coverage.
            pixel_stepping: Compute on every Nth pixel in l and m (default 4), then
                            interpolate back to the full grid. Reduces computation for
                            large images; use 1 to disable.
            chunk_size: Number of spatial pixels to process at once (default 1024²).
                        Controls memory usage for large grids.
            var: Beam variable to interpolate ('nstokes', 'stokes', 'njones', 'jones')
            i, j: Stokes or Jones indices (e.g., "I", "Q", 0, 1)

        Returns:
            Tuple of (mean_beam, variance_beam) as np.ndarray:
                - If spi is None and len(freq) > 1: both arrays have shape (NFREQ, NL, NM)
                - If spi is not None or len(freq) == 1: both arrays have shape (NL, NM)
                Where NL and NM are the dimensions of the l/m grid, corresponding to
                the lengths of the l and m axes respectively (matching indexing='ij').

        Raises:
            RuntimeError: If times are not available and not provided.
            ValueError: If both freq and num_freq are specified.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        if loc is None:
            loc = self.default_location
        if times is None:
            if self.times is None:
                raise RuntimeError("times must be supplied, since BeamWizard was "
                                   "constructed without observational time info")
            times = self.times
        if time_stepping > 1:
            times = times[::time_stepping]

        freq, norm_weights = self._resolve_freqs(freq, num_freq, spi)

        if l is None:
            l = self.l_grid
        if m is None:
            m = self.m_grid
        if ncpu is None:
            ncpu = os.cpu_count() // 2 or 1  # physical cores (assume hyperthreading)

        # Create meshgrid if l and m are 1D
        if l.ndim == 1 and m.ndim == 1:
            ll, mm = np.meshgrid(l, m, indexing='ij')
        elif l.ndim == 2 and m.ndim == 2:
            # When l and m are 2D arrays, ensure they have the same shape
            if l.shape != m.shape:
                raise ValueError(
                    f"Inconsistent shapes for l and m: l.shape={l.shape}, m.shape={m.shape}. "
                    "Both must be either 1D (to form a meshgrid) or 2D (pre-constructed grid).")
            else:
                ll, mm = l, m
        else:
            raise ValueError(
                f"Inconsistent dimensions for l and m: l.ndim={l.ndim}, m.ndim={m.ndim}. "
                "Both must be either 1D (to form a meshgrid) or 2D (pre-constructed grid)."
            )

        full_shape = ll.shape

        # Apply pixel stepping: subsample the grid for cheaper computation
        if pixel_stepping > 1:
            ll_compute = ll[::pixel_stepping, ::pixel_stepping]
            mm_compute = mm[::pixel_stepping, ::pixel_stepping]
        else:
            ll_compute = ll
            mm_compute = mm

        shape = ll_compute.shape
        ll_flat = ll_compute.ravel()
        mm_flat = mm_compute.ravel()

        # Compute parallactic angles at each time for the field center
        frame = AltAz(obstime=times, location=loc)
        altaz_centre = self.centre.transform_to(frame)

        # Get position angle to NCP (north celestial pole) to determine parallactic angle
        ncp = SkyCoord(ra=0*u.deg, dec=90*u.deg)
        altaz_ncp = ncp.transform_to(frame)
        pa = altaz_centre.position_angle(altaz_ncp)

        n_times = len(times)
        n_pixels = len(ll_flat)
        n_chunks = (n_pixels + chunk_size - 1) // chunk_size
        stepping_info = f", pixel_stepping={pixel_stepping}" if pixel_stepping > 1 else ""
        self.log.info(f"computing rotation-averaged beam over {n_times} times, "
                      f"PA range {pa.min().deg:.1f} to {pa.max().deg:.1f} deg, "
                      f"{len(freq)} frequency planes, {n_pixels} pixels in {n_chunks} chunks, "
                      f"using {ncpu} threads{stepping_info}")

        # Precompute the spline filter to ensure it's cached before threading
        self._get_prefilter(var, i, j)

        # Allocate output arrays
        out_shape = (n_pixels,) if spi is not None else (len(freq), n_pixels)
        beam_sum = np.zeros(out_shape)
        beam_sum_sq = np.zeros(out_shape)

        # Process in spatial chunks to limit memory
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_pixels)
            self.log.info(f"processing chunk {chunk_idx + 1}/{n_chunks} "
                          f"(pixels {chunk_start}-{chunk_end})")
            ll_chunk = ll_flat[chunk_start:chunk_end]
            mm_chunk = mm_flat[chunk_start:chunk_end]

            def process_time(t_idx):
                # Convert l/m (RA/Dec frame: l=East, m=North) to beam coordinates.
                # PA = position angle from field centre to NCP in AltAz frame.
                # Empirically: AltAz_angle = PA - ICRS_angle, so expanding:
                #   x_beam = sin(PA - alpha) = m*sin(PA) - l*cos(PA)
                #   y_beam = cos(PA - alpha) = l*sin(PA) + m*cos(PA)
                # (Note: "right" in the AltAz beam = West = negative l, per astronomical convention)
                pa_t = pa[t_idx].rad
                l_rot = mm_chunk * np.sin(pa_t) - ll_chunk * np.cos(pa_t)
                m_rot = ll_chunk * np.sin(pa_t) + mm_chunk * np.cos(pa_t)

                # Convert to beam pixel coordinates
                xp = l_rot / self.bds.attrs['dx'] + self.bds.attrs['x0']
                yp = m_rot / self.bds.attrs['dy'] + self.bds.attrs['y0']

                # Interpolate beam at these coordinates
                # Note: interpolate_beam expects [X, Y] format (uses xpyp[0] as X, xpyp[1] as Y)
                xpyp = np.array([xp, yp])
                beam_vals = self.interpolate_beam(xpyp, freq, var=var, i=i, j=j)

                # Average over frequency if spectral index is given
                if spi is not None:
                    beam_vals = (beam_vals * norm_weights[:, np.newaxis]).sum(axis=0)

                return beam_vals

            # Accumulate over time for this chunk
            chunk_sum = np.zeros((chunk_end - chunk_start,) if spi is not None else (len(freq), chunk_end - chunk_start))
            chunk_sum_sq = np.zeros_like(chunk_sum)

            with ThreadPoolExecutor(max_workers=ncpu) as executor:
                for beam_vals in executor.map(process_time, range(n_times)):
                    chunk_sum += beam_vals
                    chunk_sum_sq += beam_vals ** 2

            # Store chunk results
            if spi is not None:
                beam_sum[chunk_start:chunk_end] = chunk_sum
                beam_sum_sq[chunk_start:chunk_end] = chunk_sum_sq
            else:
                beam_sum[:, chunk_start:chunk_end] = chunk_sum
                beam_sum_sq[:, chunk_start:chunk_end] = chunk_sum_sq

        # Compute mean and variance over time
        beam_mean = beam_sum / n_times
        beam_var = beam_sum_sq / n_times - beam_mean ** 2

        # Reshape to coarse grid
        if spi is not None or len(freq) == 1:
            beam_mean = beam_mean.reshape(shape)
            beam_var = beam_var.reshape(shape)
        else:
            beam_mean = beam_mean.reshape((len(freq),) + shape)
            beam_var = beam_var.reshape((len(freq),) + shape)

        # Interpolate back to full resolution if pixel_stepping was applied
        if pixel_stepping > 1 and shape != full_shape:
            from scipy.ndimage import map_coordinates
            # Fractional coarse-grid coordinates for each full-resolution pixel
            fi = np.arange(full_shape[0]) / pixel_stepping
            fj = np.arange(full_shape[1]) / pixel_stepping
            fi2d, fj2d = np.meshgrid(fi, fj, indexing='ij')
            coords = np.array([fi2d.ravel(), fj2d.ravel()])
            if beam_mean.ndim == 2:
                beam_mean = map_coordinates(beam_mean, coords, order=1, mode='nearest').reshape(full_shape)
                beam_var  = map_coordinates(beam_var,  coords, order=1, mode='nearest').reshape(full_shape)
            else:
                mean_full = np.empty((len(freq),) + full_shape)
                var_full  = np.empty((len(freq),) + full_shape)
                for f_idx in range(len(freq)):
                    mean_full[f_idx] = map_coordinates(beam_mean[f_idx], coords, order=1, mode='nearest').reshape(full_shape)
                    var_full[f_idx]  = map_coordinates(beam_var[f_idx],  coords, order=1, mode='nearest').reshape(full_shape)
                beam_mean = mean_full
                beam_var  = var_full

        return beam_mean, beam_var


