#!/usr/bin/env python
"""
CLI for computing PRIMARY_BEAM and writing an xradio-compatible zarr dataset.

Produces a zarr dataset following the xradio image schema proposal
(https://github.com/casangi/xradio/issues/497).

The output dataset has:
    Dimensions:   (time, frequency, polarization, l, m)
    Coordinates:  time (MJD), frequency (Hz), polarization (str),
                  l (rad), m (rad), right_ascension(l,m), declination(l,m)
    Data vars:    PRIMARY_BEAM (time, frequency, polarization, l, m) float32
    Attributes:   coordinate_system_info, type, data_groups
"""
import numpy as np
import click
from typing import Optional, List


def compute_primary_beam_xradio(bds_path: str, image_path: str, output: str,
                                pixel_stepping: int = 4,
                                time_stepping: int = 1,
                                num_freq: Optional[int] = None,
                                ncpu: Optional[int] = None,
                                chunks_time: int = 1,
                                chunks_freq: Optional[int] = None,
                                chunks_x: int = 256,
                                chunks_y: int = 256,
                                polarizations: Optional[List[str]] = None,
                                var: str = 'nstokes'):
    """
    Compute PRIMARY_BEAM and write as xradio-compatible zarr dataset.

    Args:
        bds_path: Path to beam dataset (.bds.zarr)
        image_path: Path to image/dataset for WCS and time info
        output: Output path for the zarr dataset
        pixel_stepping: Compute every Nth pixel, interpolate back (default 4)
        time_stepping: Use every Nth timeslot (default 1)
        num_freq: Number of frequency channels (None = use beam dataset freqs)
        ncpu: Number of CPUs (None = auto)
        chunks_time, chunks_freq, chunks_x, chunks_y: Zarr chunk sizes
        polarizations: List of Stokes labels, e.g. ["I"] or ["I", "Q", "U", "V"]
        var: Beam variable ('nstokes', 'stokes', 'njones', 'jones')

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
        var_name="PRIMARY_BEAM",
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
        var=var,
        ij_list=ij_list)

    _enrich_xradio(output, bw, polarizations)

    bw.log.info(f"xradio-compatible PRIMARY_BEAM written to {output}")
    return output


def _enrich_xradio(zarr_path: str, bw, polarizations: List[str]):
    """Post-process zarr store to add xradio-compatible metadata and coordinates."""
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

    # Add right_ascension(l, m) and declination(l, m) via WCS
    nx, ny = len(l_deg), len(m_deg)
    pix_x, pix_y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    world = bw.wcs.pixel_to_world(pix_x.ravel(), pix_y.ravel())

    ra_ds = store.create_dataset(
        'right_ascension', data=world.ra.rad.reshape(nx, ny),
        dtype='float64', overwrite=True)
    ra_ds.attrs['_ARRAY_DIMENSIONS'] = ['l', 'm']

    dec_ds = store.create_dataset(
        'declination', data=world.dec.rad.reshape(nx, ny),
        dtype='float64', overwrite=True)
    dec_ds.attrs['_ARRAY_DIMENSIONS'] = ['l', 'm']

    # Dataset-level attributes (read-modify-write to preserve xarray metadata)
    ra0 = float(bw.centre.ra.rad)
    dec0 = float(bw.centre.dec.rad)

    existing_attrs = dict(store.attrs)
    existing_attrs.update({
        'type': 'image_dataset',
        'coordinate_system_info': {
            'reference_direction': {
                'attrs': {
                    'frame': 'fk5', 'type': 'sky_coord',
                    'units': 'rad', 'equinox': 'j2000.0',
                },
                'data': [ra0, dec0],
                'dims': 'sky_dir_label',
                'coords': {
                    'sky_dir_label': {
                        'data': ['ra', 'dec'],
                        'dims': 'sky_dir_label',
                    }
                },
            },
            'native_pole_direction': {
                'attrs': {
                    'frame': 'NATIVE_PROJECTION', 'type': 'location',
                    'units': 'rad',
                },
                'data': [float(np.pi), dec0],
                'dims': 'ellipsoid_dir_label',
                'coords': {
                    'ellipsoid_dir_label': {
                        'data': ['lon', 'lat'],
                        'dims': 'ellipsoid_dir_label',
                    }
                },
            },
            'projection': 'SIN',
            'projection_parameters': [0.0, 0.0],
            'pixel_coordinate_transformation_matrix': [[1.0, 0.0], [0.0, 1.0]],
        },
        'data_groups': {
            'base': {'primary_beam': 'PRIMARY_BEAM'},
        },
    })
    store.attrs.put(existing_attrs)

    # Variable-level attributes on PRIMARY_BEAM
    store['PRIMARY_BEAM'].attrs.update({
        'type': 'primary_beam',
        'units': 'dimensionless',
    })

    # Re-consolidate metadata so open_zarr works without consolidated=False
    zarr.consolidate_metadata(zarr_path)


@click.command()
@click.argument('bds_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--pixel-stepping', type=int, default=4,
              help='Spatial subsampling factor (default: 4)')
@click.option('--time-stepping', type=int, default=1,
              help='Use every Nth timeslot (default: 1)')
@click.option('--num-freq', type=int, default=None,
              help='Number of frequency channels (default: all)')
@click.option('--ncpu', type=int, default=None,
              help='Number of CPUs (default: half of cores)')
@click.option('--chunks-time', type=int, default=1)
@click.option('--chunks-freq', type=int, default=None)
@click.option('--chunks-x', type=int, default=256)
@click.option('--chunks-y', type=int, default=256)
@click.option('--pol', '-p', 'polarizations', multiple=True, default=('I',),
              help='Stokes parameters to compute (e.g. -p I -p Q)')
@click.option('--var', type=click.Choice(['nstokes', 'stokes', 'njones', 'jones']),
              default='nstokes', help='Beam variable to use')
def compute_primary_beam_cli(bds_path, image_path, output,
                             pixel_stepping, time_stepping, num_freq,
                             ncpu, chunks_time, chunks_freq, chunks_x, chunks_y,
                             polarizations, var):
    """Compute PRIMARY_BEAM and write as xradio-compatible zarr dataset.

    BDS_PATH: Path to beam dataset (.bds.zarr)

    IMAGE_PATH: Path to image/dataset for WCS and time info

    OUTPUT: Output zarr dataset path
    """
    compute_primary_beam_xradio(
        bds_path=bds_path,
        image_path=image_path,
        output=output,
        pixel_stepping=pixel_stepping,
        time_stepping=time_stepping,
        num_freq=num_freq,
        ncpu=ncpu,
        chunks_time=chunks_time,
        chunks_freq=chunks_freq if chunks_freq else None,
        chunks_x=chunks_x,
        chunks_y=chunks_y,
        polarizations=list(polarizations),
        var=var)


if __name__ == '__main__':
    compute_primary_beam_cli()
