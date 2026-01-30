# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

suricat-beams provides MeerKAT primary beam model handling utilities. It downloads, converts, and interpolates primary beam models from the SARAO archive (see https://doi.org/10.48479/wdb0-h061).

The project consists of two packages:
- **suricat-beams** (main): Core beam handling functionality and CLI
- **suricat-beams-cabs**: Stimela cab/recipe definitions for workflow integration

## Common Commands

```bash
# Install with Poetry
poetry install

# Run CLI commands
suricat download U                    # Download U-band beam model
suricat mdv2bds input.npz output.bds  # Convert MdV npz to beam dataset

# Run via Stimela recipe
stimela run suricat-beams/suricat/recipes/init.yml suricat-init band=U
```

## Architecture

### Core Components

- **`suricat/beams.py`**: Main beam processing module
  - `download_mdv_beams()`: Downloads MdV-format beam files from SARAO mirrors
  - `mdv_beams_to_bds()`: Converts MdV npz to zarr-based Beam Dataset (BDS) with Jones/Stokes matrices
  - `BeamWizard`: Class for beam interpolation given sky coordinates, times, and frequencies

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

### Stimela Integration

- **`cabs/suricat.yml`**: Defines Stimela cabs (`suricat.download-mdv-beams`, `suricat.mdv-beams-to-bds`)
- **`recipes/init.yml`**: Recipe to download and convert beams in one step
- **`cargo/`**: Docker container definitions for Stimela cargo

## Key Dependencies

- **stimela**: Workflow framework for radio astronomy
- **xarray + zarr**: Beam dataset storage format
- **astropy**: Coordinate transforms, WCS handling, time/location
- **scipy**: Spline interpolation for beam values
