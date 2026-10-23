"""
Write zarr stores under either zarr-python 2 or zarr-python 3.

zarr-python 3 writes new stores in the zarr v3 format, which configures codecs
differently from v2: ``compressors`` (a list) instead of ``compressor``, and v3
codec classes instead of numcodecs ones. Reading needs no help, since xarray
opens stores of either format. These helpers take the numcodecs codecs suricat
has always used and hand each zarr-python version what it expects.
"""
from typing import Iterable, Optional, Sequence

import numpy as np
import zarr
from numcodecs import Blosc, Delta

ZARR3 = int(zarr.__version__.split(".")[0]) >= 3

# Smooth beam data compresses well with Delta prefiltering + Blosc
BEAM_COMPRESSOR = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
BEAM_FILTERS = [Delta(dtype='float32')]

_BLOSC_SHUFFLE = {Blosc.NOSHUFFLE: "noshuffle", Blosc.SHUFFLE: "shuffle", Blosc.BITSHUFFLE: "bitshuffle"}


def as_v3_codec(codec):
    """Return the zarr v3 equivalent of a numcodecs codec.

    Blosc maps to zarr's own ``BloscCodec``, which is part of the v3 spec. Other
    codecs (e.g. Delta) map to their ``numcodecs.zarr3`` wrappers.
    """
    config = codec.get_config()
    config.pop("id")
    if isinstance(codec, Blosc):
        return zarr.codecs.BloscCodec(cname=config["cname"], clevel=config["clevel"],
                                      shuffle=_BLOSC_SHUFFLE[config["shuffle"]],
                                      blocksize=config["blocksize"])
    try:
        # zarr-python >= 3.1.3 carries the numcodecs wrappers itself
        from zarr.codecs import numcodecs as wrappers
    except ImportError:
        from numcodecs import zarr3 as wrappers
    return getattr(wrappers, type(codec).__name__)(**config)


def variable_encoding(compressor=None, filters: Optional[Iterable] = None) -> dict:
    """Per-variable ``encoding`` for ``xarray.Dataset.to_zarr``, given numcodecs codecs.

    Codecs left as None are left out, so zarr's defaults apply.
    """
    encoding = {}
    if ZARR3:
        if compressor is not None:
            encoding["compressors"] = [as_v3_codec(compressor)]
        if filters:
            encoding["filters"] = [as_v3_codec(f) for f in filters]
    else:
        if compressor is not None:
            encoding["compressor"] = compressor
        if filters:
            encoding["filters"] = list(filters)
    return encoding


def create_array(group, name: str, dims: Sequence[str], *, data=None, shape=None, chunks=None,
                 dtype=None, fill_value=None, compressor=None, filters: Optional[Iterable] = None):
    """Create (or overwrite) array ``name`` in ``group``, labelled with ``dims`` for xarray.

    Give either ``data``, or ``shape`` and ``dtype``. zarr v2 stores carry the
    dimension names in the ``_ARRAY_DIMENSIONS`` attribute, v3 stores in the
    array metadata.
    """
    if ZARR3:
        kw = dict(name=name, dimension_names=list(dims), overwrite=True)
        if compressor is not None:
            kw["compressors"] = [as_v3_codec(compressor)]
        if filters:
            kw["filters"] = [as_v3_codec(f) for f in filters]
        if data is not None:
            data = np.asarray(data)
            array = group.create_array(shape=data.shape, dtype=data.dtype, **kw)
            array[...] = data
        else:
            array = group.create_array(shape=shape, chunks=chunks, dtype=dtype,
                                       fill_value=fill_value, **kw)
        return array
    kw = dict(overwrite=True)
    if compressor is not None:
        kw["compressor"] = compressor
    if filters:
        kw["filters"] = list(filters)
    if data is not None:
        # no fill value: zarr 2 defaults to 0, which xarray would then decode as
        # missing, turning a coordinate value of exactly 0 into NaN
        array = group.create_dataset(name, data=np.asarray(data), fill_value=None, **kw)
    else:
        array = group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype,
                                     fill_value=fill_value, **kw)
    array.attrs['_ARRAY_DIMENSIONS'] = list(dims)
    return array


def is_zarr_store(path) -> bool:
    """True if ``path`` is the root of a zarr store, of either format."""
    from pathlib import Path
    path = Path(path)
    return (path / ".zgroup").exists() or (path / "zarr.json").exists()
