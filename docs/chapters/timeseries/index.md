# Time Series

PanSeg imports, processes, and exports time series data.
In the current version Panseg does not take any advantage of the time data,
it only processes the data frame-by-frame. Tracking and other processing
across timepoints will be added in a future version.

## Supported layouts and formats

The time series layouts are TYX (2D), TZYX (3D), and their multichannel versions
TCYX and TCZYX.

Import supports OME-TIFF, h5, and zarr.
For OME-TIFF, the input tab prefills the stack layout from
the file's OME-XML, and it reads the time spacing from the timing metadata
(`TimeIncrement` or `Plane.DeltaT`) if the file carries it.

Export writes the time axis back out. Tiff export of a time series is always
OME-TIFF, named with the `.ome.tiff` extension per the OME-TIFF naming
convention, with `TimeIncrement` when the spacing is known. H5 and zarr export
carry the layout and the spacing with it.

## Limitations

* Training is currently not supported
* Pre- and Postprocessing steps apply to all timepoints.
* Data must fit into memory, but can be truncated during import (see [Import](../panseg_interactive_napari/import.md))
