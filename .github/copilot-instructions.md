# Copilot Instructions

## Project Overview

suricat-beams provides MeerKAT primary beam model handling utilities. It downloads, converts, and interpolates primary beam models from the SARAO archive (see https://doi.org/10.48479/wdb0-h061).

The project consists of two packages:
- **suricat-beams** (main): Core beam handling functionality and CLI
- **suricat-beams-cabs**: Stimela cab/recipe definitions for workflow integration

## Architecture

### Core Components

- **`suricat/beams.py`**: Main beam processing module
  - `download_mdv_beams()`: Downloads MdV-format beam files from SARAO mirrors
  - `mdv_beams_to_bds()`: Converts MdV npz to zarr-based Beam Dataset (BDS) with Jones/Stokes matrices
  - `BeamWizard`: Class for beam interpolation given sky coordinates, times, and frequencies
  - `get_time_freq_beam()`: Writes full time-frequency beam cube to zarr
  - `get_rotation_averaged_beam()`: Computes rotation-averaged beam with optional `pixel_stepping` for spatial subsampling

- **`suricat/xradio_util.py`**: xradio-compatible zarr export
  - `bds_to_xradio()`: Renders a BDS to xradio-schema zarr (time, frequency, polarization, l, m)
  - `mdv_to_xradio()`: Converts raw MdV npz to xradio-schema zarr
  - Both registered as Click subcommands (`bds2xradio`, `mdv2xradio`) via `clickify_parameters`

- **`suricat/beamplots.py`**: Beam gain computation utilities (e.g., `collect_beam_gain_to_source()`)

- **`suricat/main.py`**: Click-based CLI entry point; loads cab schemas from `cabs/suricat.yml`

### Data Flow

1. MdV beam files (`.npz`) are downloaded from SARAO archive
2. Converted to Beam Dataset (`.bds.zarr`) containing:
   - `jones`: Raw Jones matrices (2x2 x FREQ x Y x X)
   - `njones`: Normalized Jones matrices
   - `stokes`: Mueller/Stokes matrices (4x4 x FREQ x Y x X)
   - `nstokes`: Normalized Stokes matrices
3. `BeamWizard` interpolates beam values at arbitrary sky positions using spline prefilters
4. xradio export produces zarr with dimensions (time, frequency, polarization, l, m), coordinates in standard units (Hz, radians), and direction attributes

### Stimela Integration

- **`cabs/suricat.yml`**: Defines Stimela cabs (`suricat.download-mdv-beams`, `suricat.mdv-beams-to-bds`, `suricat.bds-to-xradio`, `suricat.mdv-to-xradio`)
- **`recipes/init.yml`**: Recipe to download and convert beams in one step
- **`cargo/`**: Docker container definitions for Stimela cargo
- CLI commands use `clickify_parameters` from `scabha.schema_utils` to generate Click options from cab YAML schemas

## Key Dependencies

- **stimela / scabha**: Workflow framework for radio astronomy; `clickify_parameters` for CLI generation
- **xarray + zarr**: Beam dataset storage format
- **dask[array]**: Lazy array operations (transpose, rechunk) for zarr processing
- **astropy**: Coordinate transforms, WCS handling, time/location
- **scipy**: Spline interpolation for beam values
- **numcodecs**: Optional Delta+Blosc compression for zarr output

## Code Style and Preferences

- Keep code concise and avoid over-engineering. No unnecessary abstractions or helpers for one-off operations.
- CLI subcommands are defined via `clickify_parameters` wrapping functions as `@cli.command`, not as standalone scripts.
- Use `LOGGER.info()` (from `suricat/__init__.py`) for user-facing messages, not `print()`.
- Zarr dimension order follows xradio schema: `(time, frequency, polarization, l, m)`.
- Compression is opt-in (`compress=False` by default).
- Parameter naming: `output_var` for the zarr data variable name, `beam_type` for the beam matrix selection (nstokes/stokes/njones/jones).
- This is a small scientific computing project. Do not suggest adding type annotations, docstrings, or comments to code that already works and is clear in context.
- Do not flag missing unit tests for utility or conversion functions. Testing is done via end-to-end consistency checks (e.g. `test_rendered_beam.py`).
- Do not suggest restructuring working code for stylistic reasons (e.g. extracting helpers, renaming for PEP8, splitting modules).

## Review Guidelines

When reviewing pull requests:

- **Only comment on the diff.** Do not raise issues about pre-existing code that is not part of the changeset.
- **Do not repeat resolved comments.** If a comment has been addressed or dismissed, do not raise it again on re-review.
- **Focus on correctness and bugs**, not style. Only flag issues that would cause incorrect results, data loss, or crashes.
- **Do not suggest adding** error handling, input validation, type hints, or docstrings unless there is a concrete bug.
- **Do not suggest refactoring** unless the current code is broken or unmaintainable. Working code that is slightly verbose is fine.
- **Be concise.** One sentence per comment is preferred. Do not explain what the code does back to the author.
