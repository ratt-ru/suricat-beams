#!/usr/bin/env python
"""
Test consistency between different beam computation methods.

This module tests that get_time_variable_beamgain() and get_rotation_averaged_beam()
produce consistent results.
"""
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from typing import Optional, Dict
import click
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def time_variable_vs_rotation_averaged(beam_wizard,
                                        srcpos: SkyCoord,
                                        times: Optional[Time] = None,
                                        loc: Optional[EarthLocation] = None,
                                        freq: Optional[np.ndarray] = None,
                                        num_freq: int = 1,
                                        spi: Optional[float] = None,
                                        time_stepping: int = 4,
                                        var: str = 'nstokes',
                                        i: str = "I",
                                        j: str = "I") -> Dict:
    """
    Test consistency between get_time_variable_beamgain() and get_rotation_averaged_beam().

    The key principle: for a source at a fixed RA/Dec, the time-averaged beam gain
    should match the rotation-averaged beam at the source's l/m position in the image.
    Both methods average over the same range of parallactic angles, just from different
    perspectives:

    - get_time_variable_beamgain(): tracks source through beam over time
    - get_rotation_averaged_beam(): rotates beam at fixed image position over time

    Args:
        beam_wizard: BeamWizard instance to test
        srcpos: Source sky coordinate to test
        times: Times to sample (uses stored times if None)
        loc: Observer location (defaults to MeerKAT)
        freq: Frequencies to sample (if None, uses beam dataset frequencies)
        num_freq: Number of frequencies if freq is None (default 1 for simplicity)
        spi: Spectral index for frequency averaging (None = no averaging)
        time_stepping: Use every Nth timeslot (default 4) to speed up computation
        var: Beam variable to interpolate ('nstokes', 'stokes', 'njones', 'jones')
        i, j: Stokes/Jones indices

    Returns:
        Dictionary with test results including means, standard deviations, and
        consistency flag.
    """
    if loc is None:
        loc = beam_wizard.default_location
    if times is None:
        if beam_wizard.times is None:
            raise RuntimeError("times must be supplied")
        times = beam_wizard.times
    if time_stepping > 1:
        times = times[::time_stepping]

    beam_wizard.log.info(f"Testing consistency for source at {srcpos.to_string('hmsdms')}")
    beam_wizard.log.info(f"Using {len(times)} times (time_stepping={time_stepping})")

    # Method 1: Get time-variable beam gain using the dedicated method
    beam_wizard.log.info("Computing time-variable beam gain...")
    time_variable_beam = beam_wizard.get_time_variable_beamgain(
        coord=srcpos,
        times=times,
        loc=loc,
        freq=freq,
        num_freq=num_freq,
        spi=spi,
        var=var, i=i, j=j
    )

    # Compute statistics over time
    if spi is not None:
        # Already frequency-averaged, shape is (NTIME,)
        tv_mean = np.mean(time_variable_beam)
        tv_var = np.var(time_variable_beam)
        tv_mean = np.array([tv_mean])  # Make 1D for consistent handling
        tv_var = np.array([tv_var])
    else:
        # Shape is (NFREQ, NTIME), average over time
        tv_mean = np.mean(time_variable_beam, axis=1)  # shape: (NFREQ,)
        tv_var = np.var(time_variable_beam, axis=1)    # shape: (NFREQ,)

    # Method 2: Get rotation-averaged beam at source's l/m position
    # Compute l/m in RA/Dec frame: positive l = positive RA (East), positive m = positive Dec (North)
    beam_wizard.log.info("Computing source l/m coordinates in RA/Dec frame...")
    angle = beam_wizard.centre.position_angle(srcpos)  # ICRS position angle (0=North, 90=East)
    sep = beam_wizard.centre.separation(srcpos)

    l_src = sep.deg * np.sin(angle.rad)   # East component
    m_src = sep.deg * np.cos(angle.rad)   # North component

    beam_wizard.log.info(f"Source position: l={l_src:.4f} deg, m={m_src:.4f} deg "
                        f"(angle {angle.deg:.1f} deg, separation {sep.deg:.4f} deg)")

    # Compute rotation-averaged beam at this l/m position
    beam_wizard.log.info("Computing rotation-averaged beam...")
    ra_beam, ra_var = beam_wizard.get_rotation_averaged_beam(
        l=np.array([l_src]),
        m=np.array([m_src]),
        times=times,
        loc=loc,
        freq=freq,
        num_freq=num_freq,
        spi=spi,
        time_stepping=1,  # Already subsampled above
        var=var, i=i, j=j
    )

    # Extract the single spatial pixel result
    if spi is not None:
        # Shape is (1, 1), extract scalar
        ra_mean = np.array([ra_beam[0, 0]])
        ra_variance = np.array([ra_var[0, 0]])
    else:
        # Shape is (NFREQ, 1, 1) if multiple frequencies
        if ra_beam.ndim == 3:
            ra_mean = ra_beam[:, 0, 0]      # shape: (NFREQ,)
            ra_variance = ra_var[:, 0, 0]
        else:
            # Single frequency case
            ra_mean = np.array([ra_beam[0, 0]])
            ra_variance = np.array([ra_var[0, 0]])

    # Compute parallactic angles for diagnostic info
    frame = AltAz(obstime=times, location=loc)
    altaz_centre_all = beam_wizard.centre.transform_to(frame)
    ncp = SkyCoord(ra=0*u.deg, dec=90*u.deg)
    altaz_ncp = ncp.transform_to(frame)
    pa = altaz_centre_all.position_angle(altaz_ncp)

    # Compare results
    rel_diff = np.abs(tv_mean - ra_mean) / ((tv_mean + ra_mean) / 2) * 100
    consistent = np.all(rel_diff < 1.0)  # Less than 1% difference threshold
    assert consistent, f"Inconsistent beam results: max relative difference {np.max(rel_diff):.3f}%"

    # Get frequency array for reporting
    if freq is None:
        bds_freqs = beam_wizard.bds.coords['FREQ'].values
        if spi is not None:
            freq_array = bds_freqs  # Used all frequencies for averaging
            display_freqs = [bds_freqs[0], bds_freqs[-1]]  # Show range
        elif num_freq == 1:
            # When a single frequency is requested, `_resolve_freqs` uses the first
            # frequency in `bds_freqs` (via np.linspace(bds_freqs[0], bds_freqs[-1], 1)).
            # For consistency, report and return that same frequency here.
            freq_array = np.array([bds_freqs[0]])
            display_freqs = freq_array
        else:
            freq_array = np.linspace(bds_freqs[0], bds_freqs[-1], num_freq)
            display_freqs = freq_array
    else:
        freq_array = freq
        display_freqs = freq_array

    # Print results
    beam_wizard.log.info("=" * 60)
    beam_wizard.log.info("CONSISTENCY TEST RESULTS")
    beam_wizard.log.info("=" * 60)

    if spi is not None:
        beam_wizard.log.info(f"Frequency averaging with SPI={spi} "
                            f"({display_freqs[0]*1e-6:.1f}-{display_freqs[-1]*1e-6:.1f} MHz)")
        beam_wizard.log.info(f"  Time-variable mean: {tv_mean[0]:.6f} (var={tv_var[0]:.6f})")
        beam_wizard.log.info(f"  Rotation-averaged:  {ra_mean[0]:.6f} (var={ra_variance[0]:.6f})")
        beam_wizard.log.info(f"  Relative difference: {rel_diff[0]:.3f}%")
    else:
        for f_idx, f in enumerate(display_freqs):
            beam_wizard.log.info(f"Frequency {f*1e-6:.1f} MHz:")
            beam_wizard.log.info(f"  Time-variable mean: {tv_mean[f_idx]:.6f} (var={tv_var[f_idx]:.6f})")
            beam_wizard.log.info(f"  Rotation-averaged:  {ra_mean[f_idx]:.6f} (var={ra_variance[f_idx]:.6f})")
            beam_wizard.log.info(f"  Relative difference: {rel_diff[f_idx]:.3f}%")

    beam_wizard.log.info(f"\nParallactic angle range: {pa.deg.min():.1f} to {pa.deg.max():.1f} deg")
    beam_wizard.log.info(f"Consistent (< 1% diff): {consistent}")
    beam_wizard.log.info("=" * 60)

    return {
        'time_variable_mean': tv_mean,
        'time_variable_var': tv_var,
        'time_variable_values': time_variable_beam,
        'rotation_averaged': ra_mean,
        'rotation_variance': ra_variance,
        'relative_difference': rel_diff,
        'consistent': consistent,
        'source_lm': (l_src, m_src),
        'source_separation_deg': sep.deg,
        'frequencies': freq_array,
        'parallactic_angles': pa.deg,
        'times': times,
        'n_times': len(times),
        'spectral_index': spi
    }


def run_example_test(bds_path: str, image_path: str,
                     ra: Optional[float] = None,
                     dec: Optional[float] = None,
                     time_stepping: int = 4,
                     freq: Optional[np.ndarray] = None,
                     num_freq: int = 1,
                     spi: Optional[float] = None):
    """
    Run a consistency test between time-variable and rotation-averaged beams.

    Args:
        bds_path: Path to beam dataset (.bds.zarr)
        image_path: Path to image or dataset for WCS info
        ra: Source RA in degrees. If None, uses field center.
        dec: Source Dec in degrees. If None, uses field center.
        time_stepping: Use every Nth timeslot (default 4)
        freq: Explicit frequency array in Hz (optional)
        num_freq: Number of frequencies to test (default 1, ignored if freq is provided)
        spi: Spectral index for frequency averaging (None = no averaging)

    Returns:
        Dictionary with test results

    Example:
        >>> from suricat.beams import BeamWizard
        >>> from suricat.tests.test_beam_consistency import run_example_test
        >>>
        >>> # Test at field center with single frequency
        >>> results = run_example_test("beam.bds.zarr", "image.zarr")
        >>> print(f"Consistent: {results['consistent']}")
        >>>
        >>> # Test with specific position and frequency averaging
        >>> results = run_example_test("beam.bds.zarr", "image.zarr",
        ...                             ra=180.0, dec=-30.0, spi=-0.7)
    """
    from suricat.beams import BeamWizard

    # Create BeamWizard
    bw = BeamWizard(bds_path, image_path)

    # Define source position
    if ra is None or dec is None:
        # Use field center
        ra = bw.centre.ra.deg
        dec = bw.centre.dec.deg
        bw.log.info(f"Testing with source at field center")

    srcpos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    # Run test
    results = time_variable_vs_rotation_averaged(
        bw, srcpos,
        time_stepping=time_stepping,
        freq=freq,
        num_freq=num_freq,
        spi=spi
    )

    return results


@click.command()
@click.argument('bds_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--ra', type=float, default=None,
              help='Source RA in degrees (default: field center)')
@click.option('--dec', type=float, default=None,
              help='Source Dec in degrees (default: field center)')
@click.option('--time-stepping', type=int, default=4,
              help='Use every Nth timeslot (default: 4)')
@click.option('--freq', type=str, default=None,
              help='Comma-separated frequency values in Hz (e.g., "1e9,1.5e9,2e9")')
@click.option('--num-freq', type=int, default=3,
              help='Number of frequencies to test (default: 3, ignored if --freq is provided)')
@click.option('--spi', type=float, default=None,
              help='Spectral index for frequency averaging (default: None)')
def test_beam_consistency_cli(bds_path, image_path, ra, dec,
                                time_stepping, freq, num_freq, spi):
    """
    Run beam consistency test between time-variable and rotation-averaged methods.

    BDS_PATH: Path to beam dataset (.bds.zarr)
    IMAGE_PATH: Path to image or dataset for WCS info
    """
    # Parse frequency array if provided
    freq_array = None
    if freq is not None:
        freq_array = np.array([float(f) for f in freq.split(',')])

    # Run the test
    results = run_example_test(
        bds_path=bds_path,
        image_path=image_path,
        ra=ra,
        dec=dec,
        time_stepping=time_stepping,
        freq=freq_array,
        num_freq=num_freq,
        spi=spi
    )

    # Create plot
    times = results['times']
    time_variable_beam = results['time_variable_values']
    frequencies = results['frequencies']

    if plt is None:
        raise click.ClickException(
            "matplotlib is required for plotting but is not installed. "
            "Please install it (e.g., 'pip install matplotlib') and retry."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert times to hours from start for x-axis
    time_hours = (times.mjd - times[0].mjd) * 24

    if spi is not None:
        # Single line for frequency-averaged beam
        ax.plot(time_hours, time_variable_beam, 'b-', linewidth=2, label='Frequency-averaged')
    else:
        # Plot three frequency lines: first, middle, last
        n_freq = len(frequencies)
        if n_freq >= 3:
            freq_indices = [0, n_freq // 2, n_freq - 1]
            colors = ['blue', 'green', 'red']
            labels = ['First', 'Middle', 'Last']
        else:
            freq_indices = list(range(n_freq))
            colors = ['blue', 'green', 'red'][:n_freq]
            labels = [f'Freq {i+1}' for i in range(n_freq)]

        for idx, color, label in zip(freq_indices, colors, labels):
            freq_mhz = frequencies[idx] / 1e6
            ax.plot(time_hours, time_variable_beam[idx, :],
                   color=color, linewidth=2,
                   label=f'{label} ({freq_mhz:.1f} MHz)')

    ax.set_xlabel('Time (hours from start)', fontsize=12)
    ax.set_ylabel('Beam Gain', fontsize=12)
    ax.set_title('Time-Variable Beam Gain', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('beamplot.png', dpi=150, bbox_inches='tight')
    click.echo(f"\nPlot saved to beamplot.png")

    # Print summary
    click.echo("\n" + "="*60)
    click.echo("TEST SUMMARY")
    click.echo("="*60)
    click.echo(f"Consistent: {results['consistent']}")
    click.echo(f"Relative difference: {results['relative_difference'][0]:.3f}%")
    click.echo(f"Number of times: {results['n_times']}")
    click.echo(f"Source separation: {results['source_separation_deg']:.4f} deg")
    if results['spectral_index'] is not None:
        click.echo(f"Spectral index: {results['spectral_index']}")
    click.echo("="*60)


if __name__ == '__main__':
    test_beam_consistency_cli()
