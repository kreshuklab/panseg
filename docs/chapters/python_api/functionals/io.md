# Input/Output

All the input/output operations are handled by the `panseg.io` module.
This module provides functions to read and write data in different formats.
The supported formats are `tiff`, `h5`, and `zarr`, `jpeg`, `png`.
Zarr export writes an OME-Zarr (OME-NGFF v0.5) store.
Segmentations are written as label images below their parent image group
(`<key>/labels/segmentation`).

## Reading

::: panseg.io.smart_load

## Writing

::: panseg.io.create_tiff
::: panseg.io.create_h5
::: panseg.io.create_zarr

## Tiff Utilities

::: panseg.io.tiff.read_tiff_voxel_size

## H5 Utilities

::: panseg.io.h5.list_h5_keys
::: panseg.io.h5.del_h5_key
::: panseg.io.h5.rename_h5_key

## Zarr Utilities

::: panseg.io.zarr.list_zarr_keys
::: panseg.io.zarr.del_zarr_key
::: panseg.io.zarr.rename_zarr_key
