# Output

Using the Output tab you can save your work as `tiff`, `h5`, or `ome.zarr`.

Labels can be exported as mesh files (`glb`, `obj` or `ply`).
Use `glb` if possible, the other formats are missing the color information,
or merge all segments into a single mesh.

Once you have exported an image, you can create a workflow file to repeat the processing steps you have performed on image batches.

## OME-Zarr

In a zarr store, the `Key` field names the top-level image group.
Segmentation layers are written as a label image inside that group, at
`<key>/labels/segmentation`, the key names the image the segmentation
belongs to. Image and segmentation export in either order into the
same store layout. If the target slot already holds an image or label,
the export asks before overwriting it.

## Time Series

Time series in Tiff format are exported as OME-TIFF files,
with the `TimeIncrement` metadata written
when the time spacing is known. Per the OME-TIFF naming convention,
these files carry the `.ome.tiff` extension.
Zarr is exported as an OME-Zarr (OME-NGFF v0.5) store.
The layout, voxel size, and spacing live in the NGFF metadata.
H5 exports carry the layout (`axis_order` attribute)
and the spacing when it is known.

For a 3D time series segmentation, the mesh export writes one file per
timepoint, named `{name}_t000.{ext}`, `{name}_t001.{ext}`, and so on, with a
zero-based index. Empty timepoints get their (empty) mesh file too, so the file count
always matches the timepoint count.

## Widget: Output

```python exec="1" html="1"
--8<-- "widgets/output_tab/output.py"
```
