# suricat-beams
MeerKAT primary beam model handling utilities

See https://doi.org/10.48479/wdb0-h061 for documentation on beam models.

The utilities can download the models for you automatically, see suricat-beams/suricat/recipes/init.yml Stimela recipe.

## Installation

Use ``pip install suricat-beams`` for the full package.

Use ``pip install suricat-beams-cabs`` to install Stimela cab definitions only.


## CLI commands

All commands are accessible via the `suricat` CLI:

```bash
suricat download U                      # Download U-band beam model
suricat mdv2bds input.npz output.bds    # Convert MdV npz to beam dataset
suricat bds2xradio beam.bds.zarr image.zarr output.zarr \
    --output-var SKY \
    --pixel-stepping 4 --time-stepping 4 \
    --num-freq 8 --chunks-time 1 --chunks-freq 1 \
    --polarizations I --beam-type nstokes       # Render BDS to xradio zarr
suricat mdv2xradio mdv-beams-L.npz output.zarr \
    --antenna -1 --jones HH --part real \
    --var-name SKY                              # Convert MdV npz to xradio zarr
```

### Output format

Both xradio tools produce zarr datasets following the
[xradio image schema](https://github.com/casangi/xradio/issues/497):

```
Dimensions:   (time, frequency, polarization, l, m)
Coordinates:  time (MJD), frequency (Hz), polarization (str), l (rad), m (rad)
Data vars:    <var_name> (time, frequency, polarization, l, m) float32
Attributes:   direction (reference, latpole, lonpole, projection, pc)
```

## Stimela integration

Stimela cab definitions are in `suricat-beams/suricat/cabs/suricat.yml`:

- `suricat.download-mdv-beams` -- download MdV beams from SARAO archive
- `suricat.mdv-beams-to-bds` -- convert MdV npz to beam dataset
- `suricat.bds-to-xradio` -- render BDS to xradio zarr image
- `suricat.mdv-to-xradio` -- convert MdV npz to xradio zarr image

```bash
stimela run suricat-beams/suricat/recipes/init.yml suricat-init band=U
```

## Python API

```python
from suricat.xradio_util import bds_to_xradio, mdv_to_xradio

# Render a BDS with observational context
bds_to_xradio("beam.bds.zarr", "image.zarr", "output.zarr",
              output_var="SKY", num_freq=8, pixel_stepping=4)

# Convert raw MdV beam
mdv_to_xradio("mdv-beams-L.npz", "output.zarr",
              jones="HH", part="real", antenna=-1)
```
