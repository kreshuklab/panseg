# Time Series

PanSeg imports, processes, and exports time series data.
In the current version PanSeg does not take any advantage of the time data,
it only processes the data frame-by-frame. Tracking and other processing
across time points will be added in a future version.

## Supported layouts and formats

The time series layouts are TYX (2D), TZYX (3D), and their multichannel versions
TCYX and TCZYX.

Import supports OME-TIFF, h5, and zarr.
For OME-TIFF, the input tab prefills the stack layout from
the file's OME-XML, and it reads the time spacing from the timing metadata
(`TimeIncrement` or `Plane.DeltaT`) if the file carries it.
OME-Zarr (OME-NGFF) zarr stores are supported as well: the stack
layout, the voxel size, and the time spacing are read from the NGFF
metadata (OME-NGFF v0.4 and v0.5), and a multiscale pyramid is read at its
base level only.

## From multiple files

A time series can also come from a set of independent files, one image per
timepoint. Select several files in the Open File dialog and they stack into
one series in alphanumeric filename order. Each file must be a single image
(no time axis of its own), and the stack layout must be spatial; the import
adds the T axis itself. Multi-channel files stack per channel, with equal
channel counts across files.

The spacing between the files is unknown at import. Set it in the input
tab's Details field once you know the acquisition interval; export then
writes it like any other time spacing.

!!! warning
    A multifile OME-TIFF series is a different thing: one logical image
    whose planes are chained across files through the OME-XML (UUID and
    FileName references), or whose metadata lives in a `.companion.ome`
    file. It looks identical to a set of independent images in a file
    dialog. PanSeg rejects it. The reader would silently fill missing
    planes with zeros when only part of the set is present, and the plane
    order comes from the XML, not from the filenames, so filename order
    would scramble the series anyway.

Export writes the time axis back out. Tiff export of a time series is always
OME-TIFF, named with the `.ome.tiff` extension per the OME-TIFF naming
convention, with `TimeIncrement` when the spacing is known. H5 export carries
the layout and the spacing as PanSeg attributes; zarr export is written as an
OME-Zarr store carrying them in the NGFF metadata. In the
store, the key names the image group (e.g. `ovule`); a segmentation is written
as a label image next to its image (`ovule/labels/segmentation`), so exporting
the image and the segmentation in either order yields the same store.

## Limitations

* Training is currently not supported
* Pre- and Postprocessing steps apply to all timepoints.
* Data must fit into memory, but can be truncated during import (see [Import](../panseg_interactive_napari/import.md))
