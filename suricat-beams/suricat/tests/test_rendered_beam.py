#!/usr/bin/env python
"""
Test consistency between a rendered beam zarr dataset and get_time_variable_beamgain().

Renders the beam to a zarr dataset via bds_to_xradio, then picks an off-centre
l,m pixel and compares the time/frequency slice from the zarr with the result of
get_time_variable_beamgain() at the same sky coordinate.
"""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import click
import tempfile
import os


def test_rendered_vs_beamgain(bds_path: str, image_path: str,
                               l_index: int = None, m_index: int = None,
                               pixel_stepping: int = 4,
                               time_stepping: int = 4,
                               num_freq: int = 3,
                               ncpu: int = None,
                               beam_type: str = 'nstokes',
                               output_var: str = 'SKY',
                               pol: str = 'I',
                               chunks_time: int = 1,
                               chunks_freq: int = None,
                               output: str = None,
                               threshold: float = 1.0):
    """
    Test that the rendered beam zarr matches get_time_variable_beamgain().

    Args:
        bds_path: Path to beam dataset (.bds.zarr)
        image_path: Path to image/dataset for WCS and time info
        l_index: Pixel index along l axis (default: quarter of grid)
        m_index: Pixel index along m axis (default: quarter of grid)
        pixel_stepping: Spatial subsampling for rendering (default 4)
        time_stepping: Use every Nth timeslot (default 4)
        num_freq: Number of frequency channels (default 3)
        ncpu: Number of CPUs (default: auto)
        beam_type: Beam variable ('nstokes', etc.)
        output_var: Name of the output data variable (default: PRIMARY_BEAM)
        pol: Stokes parameter (default 'I')
        chunks_time: Zarr chunk size along time axis (default 1)
        chunks_freq: Zarr chunk size along frequency axis (default: all)
        output: Path for rendered zarr (default: tempdir)
        threshold: Maximum allowed relative difference in percent (default 1.0)

    Returns:
        Dictionary with test results.
    """
    import xarray
    from suricat.beams import BeamWizard
    from suricat.xradio_util import bds_to_xradio

    bw = BeamWizard(bds_path, image_path)

    # Default to quarter of grid (off-centre)
    nl = len(bw.l_grid)
    nm = len(bw.m_grid)
    if l_index is None:
        l_index = nl // 4
    if m_index is None:
        m_index = nm // 4

    l_deg = bw.l_grid[l_index]
    m_deg = bw.m_grid[m_index]
    bw.log.info(f"Test pixel: l_index={l_index}, m_index={m_index}, "
                f"l={l_deg:.4f} deg, m={m_deg:.4f} deg")

    # Step 1: Render beam to zarr
    cleanup = False
    if output is None:
        output = tempfile.mkdtemp(suffix='.zarr')
        cleanup = True

    bw.log.info(f"Rendering beam to {output}")
    bds_to_xradio(
        bds_path=bds_path,
        image_path=image_path,
        output=output,
        output_var=output_var,
        pixel_stepping=pixel_stepping,
        time_stepping=time_stepping,
        num_freq=num_freq,
        ncpu=ncpu,
        chunks_time=chunks_time,
        chunks_freq=chunks_freq,
        polarizations=[pol],
        beam_type=beam_type)

    # Step 2: Extract rendered beam slice at the test pixel
    ds = xarray.open_zarr(output)
    # Dataset dimensions: (time, frequency, polarization, l, m)
    rendered = ds[output_var].isel(
        polarization=0,
        l=l_index,
        m=m_index,
    ).values  # (ntime, nfreq)
    zarr_times = Time(ds.coords['time'].values, format='mjd')
    zarr_freq = ds.coords['frequency'].values
    bw.log.info(f"Rendered slice shape: {rendered.shape} (ntime, nfreq)")

    # Step 3: Compute sky coordinate from l,m using WCS
    # l,m are relative to field centre; convert pixel index to WCS world coords
    srcpos = bw.wcs.pixel_to_world(l_index, m_index)
    bw.log.info(f"Source sky position: {srcpos.to_string('hmsdms')}")

    # Step 4: Call get_time_variable_beamgain at the same position and times
    beamgain = bw.get_time_variable_beamgain(
        coord=srcpos,
        times=zarr_times,
        freq=zarr_freq,
        var=beam_type,
        i=pol, j=pol)
    # beamgain shape: (nfreq, ntime)
    beamgain = beamgain.T  # -> (ntime, nfreq) to match rendered

    bw.log.info(f"Beamgain slice shape: {beamgain.shape}")

    # Step 5: Compare
    # Use mean across time for the comparison metric (per frequency)
    rendered_mean = rendered.mean(axis=0)
    beamgain_mean = beamgain.mean(axis=0)
    rel_diff = np.abs(rendered_mean - beamgain_mean) / (
        (rendered_mean + beamgain_mean) / 2) * 100

    # Also compare full time series
    full_rel_diff = np.abs(rendered - beamgain) / (
        (np.abs(rendered) + np.abs(beamgain)) / 2 + 1e-30) * 100
    max_rel_diff = full_rel_diff.max()
    mean_rel_diff = full_rel_diff.mean()

    consistent = max_rel_diff < threshold

    bw.log.info("=" * 60)
    bw.log.info("RENDERED BEAM vs BEAMGAIN TEST RESULTS")
    bw.log.info("=" * 60)
    for f_idx, f in enumerate(zarr_freq):
        bw.log.info(f"Frequency {f * 1e-6:.1f} MHz:")
        bw.log.info(f"  Rendered mean:  {rendered_mean[f_idx]:.6f}")
        bw.log.info(f"  Beamgain mean:  {beamgain_mean[f_idx]:.6f}")
        bw.log.info(f"  Rel diff (mean): {rel_diff[f_idx]:.3f}%")
    bw.log.info(f"\nMax relative difference:  {max_rel_diff:.3f}%")
    bw.log.info(f"Mean relative difference: {mean_rel_diff:.3f}%")
    bw.log.info(f"Consistent (< {threshold}% diff): {consistent}")
    bw.log.info("=" * 60)

    results = {
        'rendered': rendered,
        'beamgain': beamgain,
        'rendered_mean': rendered_mean,
        'beamgain_mean': beamgain_mean,
        'rel_diff_per_freq': rel_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'consistent': consistent,
        'l_deg': l_deg,
        'm_deg': m_deg,
        'l_index': l_index,
        'm_index': m_index,
        'source_pos': srcpos,
        'frequencies': zarr_freq,
        'times': zarr_times,
        'zarr_path': output,
    }

    if cleanup:
        import shutil
        shutil.rmtree(output, ignore_errors=True)

    return results


@click.command()
@click.argument('bds_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--l-index', type=int, default=None,
              help='Pixel index along l axis (default: quarter of grid)')
@click.option('--m-index', type=int, default=None,
              help='Pixel index along m axis (default: quarter of grid)')
@click.option('--pixel-stepping', type=int, default=4,
              help='Spatial subsampling factor (default: 4)')
@click.option('--time-stepping', type=int, default=4,
              help='Use every Nth timeslot (default: 4)')
@click.option('--num-freq', type=int, default=3,
              help='Number of frequency channels (default: 3)')
@click.option('--ncpu', type=int, default=None,
              help='Number of CPUs (default: auto)')
@click.option('--beam-type', type=click.Choice(['nstokes', 'stokes', 'njones', 'jones']),
              default='nstokes', help='Beam variable to use')
@click.option('--output-var', type=str, default='SKY',
              help='Name of output data variable (default: SKY)')
@click.option('--pol', '-p', type=str, default='I',
              help='Stokes parameter (default: I)')
@click.option('--chunks-time', type=int, default=1)
@click.option('--chunks-freq', type=int, default=None)
@click.option('--output', type=click.Path(), default=None,
              help='Output zarr path (default: temp directory)')
@click.option('--threshold', type=float, default=1.0,
              help='Max allowed relative difference in percent (default: 1.0)')
def test_rendered_beam_cli(bds_path, image_path, l_index, m_index,
                           pixel_stepping, time_stepping, num_freq,
                           ncpu, beam_type, output_var, pol, chunks_time, chunks_freq,
                           output, threshold):
    """Test rendered beam zarr against get_time_variable_beamgain().

    BDS_PATH: Path to beam dataset (.bds.zarr)

    IMAGE_PATH: Path to image/dataset for WCS and time info
    """
    results = test_rendered_vs_beamgain(
        bds_path=bds_path,
        image_path=image_path,
        l_index=l_index,
        m_index=m_index,
        pixel_stepping=pixel_stepping,
        time_stepping=time_stepping,
        num_freq=num_freq,
        ncpu=ncpu,
        beam_type=beam_type,
        output_var=output_var,
        pol=pol,
        chunks_time=chunks_time,
        chunks_freq=chunks_freq if chunks_freq else None,
        output=output,
        threshold=threshold)

    click.echo(f"\nConsistent: {results['consistent']}")
    click.echo(f"Max relative difference: {results['max_rel_diff']:.3f}%")
    click.echo(f"Mean relative difference: {results['mean_rel_diff']:.3f}%")
    click.echo(f"Test pixel: l={results['l_deg']:.4f} deg, m={results['m_deg']:.4f} deg")
    click.echo(f"Source: {results['source_pos'].to_string('hmsdms')}")


if __name__ == '__main__':
    test_rendered_beam_cli()
