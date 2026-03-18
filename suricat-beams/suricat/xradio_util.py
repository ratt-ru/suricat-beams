"""
Utilities for converting beam data to xradio-compatible zarr datasets.

Produces zarr datasets following the xradio image schema
(https://github.com/casangi/xradio/issues/497), with structure:

    Dimensions:   (time, frequency, polarization, l, m)
    Coordinates:  time (MJD), frequency (Hz), polarization (str), l (rad), m (rad)
    Data vars:    <var_name> (time, frequency, polarization, l, m) float32
    Attributes:   direction (reference, latpole, lonpole, projection, pc)
"""
import numpy as np
from typing import Optional, List
from scabha.schema_utils import clickify_parameters
from .main import cli, schemas
import dask.array as da


def bds_to_xradio(bds_path: str, image_path: str, output: str,
                  output_var: str = "SKY",
                  pixel_stepping: int = 4,
                  time_stepping: int = 1,
                  num_freq: Optional[int] = None,
                  ncpu: Optional[int] = None,
                  chunks_time: int = 1,
                  chunks_freq: Optional[int] = None,
                  chunks_x: int = 256,
                  chunks_y: int = 256,
                  polarizations: Optional[List[str]] = None,
                  beam_type: str = 'nstokes'):
    """
    Render a beam dataset (BDS) to an xradio-compatible zarr image.

    Computes the time- and frequency-resolved beam from a BDS and an
    observational image (for WCS and time info), and writes the result as a
    zarr dataset with xradio-compatible schema.

    Args:
        bds_path: Path to beam dataset (.bds.zarr)
        image_path: Path to image/dataset for WCS and time info
        output: Output path for the zarr dataset
        output_var: Name of the output data variable (default: SKY)
        pixel_stepping: Compute every Nth pixel, interpolate back (default 4)
        time_stepping: Use every Nth timeslot (default 1)
        num_freq: Number of frequency channels (None = use beam dataset freqs)
        ncpu: Number of CPUs (None = auto)
        chunks_time, chunks_freq, chunks_x, chunks_y: Zarr chunk sizes
        polarizations: List of Stokes labels, e.g. ["I"] or ["I", "Q", "U", "V"]
        beam_type: Beam variable ('nstokes', 'stokes', 'njones', 'jones')

    Returns:
        Path to the output zarr dataset.
    """
    from suricat.beams import BeamWizard

    bw = BeamWizard(bds_path, image_path)

    if polarizations is None:
        polarizations = ["I"]

    stokes_ij = {"I": ("I", "I"), "Q": ("Q", "Q"),
                 "U": ("U", "U"), "V": ("V", "V")}
    ij_list = [stokes_ij[p] for p in polarizations]

    bw.get_time_freq_beam(
        filename=output,
        var_name=output_var,
        dim_names=("polarization", "time", "frequency", "l", "m"),
        l=bw.l_grid,
        m=bw.m_grid,
        pixel_stepping=pixel_stepping,
        time_stepping=time_stepping,
        num_freq=num_freq,
        ncpu=ncpu,
        chunks_time=chunks_time,
        chunks_freq=chunks_freq,
        chunks_x=chunks_x,
        chunks_y=chunks_y,
        var=beam_type,
        ij_list=ij_list)

    _enrich_bds_xradio(output, bw, output_var, polarizations)

    bw.log.info(f"xradio-compatible {output_var} written to {output}")
    return output


def _enrich_bds_xradio(zarr_path: str, bw, var_name: str, polarizations: List[str]):
    """Post-process zarr store for xradio compatibility.

    - Converts l/m from degrees to radians
    - Fixes polarization labels
    - Reorders dimensions to (time, frequency, polarization, l, m)
    - Adds direction attributes matching xradio schema
    """
    import zarr

    store = zarr.open(zarr_path, mode='r+')

    # Convert l/m from degrees to radians (xradio convention)
    l_deg = store['l'][:]
    m_deg = store['m'][:]
    store['l'][:] = np.deg2rad(l_deg)
    store['m'][:] = np.deg2rad(m_deg)

    # Fix polarization labels: get_time_freq_beam writes "II", "QQ", etc.
    # xradio expects single-letter Stokes labels "I", "Q", "U", "V"
    pol_arr = store.create_dataset(
        'polarization', data=np.array(polarizations), overwrite=True)
    pol_arr.attrs['_ARRAY_DIMENSIONS'] = ['polarization']

    # Transpose data variable from (polarization, time, frequency, l, m)
    # to xradio order (time, frequency, polarization, l, m) lazily using dask
    # Existing array is chunked in (polarization, time, frequency, l, m)
    dask_arr = da.from_zarr(zarr_path, component=var_name)
    dask_arr = dask_arr.transpose(1, 2, 0, 3, 4)  # -> (time, frequency, polarization, l, m)

    # Read existing chunk info
    old_chunks = store[var_name].chunks
    # old_chunks is (pol, time, freq, l, m) -> remap to (time, freq, pol, l, m)
    new_chunks = (old_chunks[1], old_chunks[2], old_chunks[0],
                  old_chunks[3], old_chunks[4])

    # Rechunk to desired order and write back to the same zarr store chunk-by-chunk
    dask_arr = dask_arr.rechunk(new_chunks)
    dask_arr.to_zarr(zarr_path, component=var_name, overwrite=True)

    store[var_name].attrs['_ARRAY_DIMENSIONS'] = [
        'time', 'frequency', 'polarization', 'l', 'm']

    # Dataset-level attributes: direction block matching reference schema
    ra0 = float(bw.centre.ra.rad)
    dec0 = float(bw.centre.dec.rad)

    existing_attrs = dict(store.attrs)
    existing_attrs['direction'] = {
        'reference': {
            'attrs': {'frame': 'icrs', 'type': 'sky_coord', 'units': 'rad'},
            'data': [ra0, dec0],
            'dims': ['l', 'm'],
        },
        'latpole': {
            'attrs': {'type': 'quantity', 'units': 'rad'},
            'data': dec0,
            'dims': ['l', 'm'],
        },
        'lonpole': {
            'attrs': {'type': 'quantity', 'units': 'rad'},
            'data': float(np.pi),
            'dims': ['l', 'm'],
        },
        'projection': 'SIN',
        'projection_parameters': {
            '_dtype': 'float64', '_type': 'numpy.ndarray',
            '_value': [0.0, 0.0],
        },
        'pc': {
            '_dtype': 'float64', '_type': 'numpy.ndarray',
            '_value': [[1.0, 0.0], [0.0, 1.0]],
        },
    }
    store.attrs.put(existing_attrs)

    # Variable-level attributes
    store[var_name].attrs.update({
        'image_type': 'primary_beam',
        'units': 'dimensionless',
    })

    # Re-consolidate metadata so open_zarr works without consolidated=False
    zarr.consolidate_metadata(zarr_path)


def mdv_to_xradio(npz_path: str, output: str,
                   antenna: int = -1,
                   jones: str = 'HH',
                   part: str = 'real',
                   var_name: str = 'SKY',
                   chunks_freq: int = 64,
                   chunks_x: int = 128,
                   chunks_y: int = 128):
    """
    Convert a raw MdV beam npz file to an xradio-compatible zarr image.

    Renders a single Jones element (real or imaginary part) for a selected
    antenna as a zarr dataset with dimensions (time, frequency, polarization,
    l, m), matching the xradio image schema.

    The MdV npz contains:
        beam: (4, N_ant, N_freq, N_y, N_x) complex64  [pols: HH, HV, VH, VV]
        pols: (4,)           e.g. [b'HH', b'HV', b'VH', b'VV']
        antnames: (N_ant,)   e.g. [b'm000', ..., b'array_average']
        freq_MHz: (N_freq,)
        margin_deg: (N_pix,)

    Args:
        npz_path: Path to MdV .npz beam file
        output: Output zarr path
        antenna: Antenna index (default -1 = array_average, the last entry)
        jones: Jones element to render ('HH', 'HV', 'VH', or 'VV')
        part: 'real', 'imag', 'abs', or 'phase'
        var_name: Data variable name (default: SKY)
        chunks_freq, chunks_x, chunks_y: Zarr chunk sizes
    """
    import xarray
    import zarr

    mdv = np.load(npz_path)
    beam = mdv['beam']       # (4, N_ant, N_freq, N_y, N_x) complex64
    pols = [p.decode() for p in mdv['pols']]
    antnames = [a.decode() for a in mdv['antnames']]
    freqs = mdv['freq_MHz'] * 1e6  # Hz
    margin_deg = mdv['margin_deg']

    # Select polarization
    pol_idx = pols.index(jones)

    # Select antenna
    ant_name = antnames[antenna]
    print(f"Antenna: {ant_name} (index {antenna})")
    print(f"Jones element: {jones}, part: {part}")

    # Extract: (N_freq, N_y, N_x)
    data = beam[pol_idx, antenna]

    if part == 'real':
        data = data.real
    elif part == 'imag':
        data = data.imag
    elif part == 'abs':
        data = np.abs(data)
    elif part == 'phase':
        data = np.angle(data)
    else:
        raise ValueError(f"Unknown part '{part}', expected real/imag/abs/phase")

    data = data.astype(np.float32)

    # Build xradio-compatible dataset
    # Dimensions: (time, frequency, polarization, l, m)
    l_rad = np.deg2rad(margin_deg)
    m_rad = np.deg2rad(margin_deg)
    pol_label = jones

    # Add singleton time dimension, value 0 (no time info in MdV beams)
    # data shape: (freq, y, x) -> (1, freq, 1, y, x) = (time, freq, pol, l, m)
    data_5d = data[np.newaxis, :, np.newaxis, :, :]

    ds = xarray.Dataset({
        var_name: xarray.DataArray(
            data_5d,
            dims=['time', 'frequency', 'polarization', 'l', 'm'],
            coords={
                'time': [0.0],
                'frequency': freqs,
                'polarization': [pol_label],
                'l': l_rad,
                'm': m_rad,
            }
        )
    })

    # Attributes
    ds[var_name].attrs.update({
        'image_type': f'jones_{jones}_{part}',
        'units': 'dimensionless',
    })
    ds.attrs['direction'] = {
        'reference': {
            'attrs': {'frame': 'beam', 'type': 'beam_coordinates', 'units': 'rad'},
            'data': [0.0, 0.0],
            'dims': ['l', 'm'],
        },
        'projection': 'SIN',
        'projection_parameters': {
            '_dtype': 'float64', '_type': 'numpy.ndarray',
            '_value': [0.0, 0.0],
        },
        'pc': {
            '_dtype': 'float64', '_type': 'numpy.ndarray',
            '_value': [[1.0, 0.0], [0.0, 1.0]],
        },
    }
    ds.attrs['antenna'] = ant_name
    ds.attrs['jones_element'] = jones
    ds.attrs['component'] = part

    # Write with chunking
    encoding = {
        var_name: {
            'chunks': (1, chunks_freq, 1, chunks_x, chunks_y),
        }
    }
    ds.to_zarr(output, mode='w', encoding=encoding)
    zarr.consolidate_metadata(output)

    print(f"Written {output}")
    print(f"  Shape: {data_5d.shape} (time, frequency, polarization, l, m)")
    print(f"  Frequencies: {len(freqs)} channels, "
          f"{freqs[0]/1e6:.1f} to {freqs[-1]/1e6:.1f} MHz")
    print(f"  Spatial: {len(margin_deg)}x{len(margin_deg)} pixels, "
          f"{margin_deg[0]:.2f} to {margin_deg[-1]:.2f} deg")

    return output


@cli.command("bds2xradio", help=schemas.cabs.get("suricat.bds-to-xradio").info)
@clickify_parameters(schemas.cabs.get("suricat.bds-to-xradio"))
def _bds_to_xradio(*args, **kw):
    return bds_to_xradio(*args, **kw)


@cli.command("mdv2xradio", help=schemas.cabs.get("suricat.mdv-to-xradio").info)
@clickify_parameters(schemas.cabs.get("suricat.mdv-to-xradio"))
def _mdv_to_xradio(*args, **kw):
    return mdv_to_xradio(*args, **kw)
