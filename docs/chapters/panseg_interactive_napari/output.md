# Output

Using the Output tab you can save your work as `tiff`, `h5`, or `zarr`.

Labels can be exported as mesh files (`glb`, `obj` or `ply`).
Use `glb` if possible, the other formats are missing the color information,
or merge all segments into a single mesh.

Once you have exported an image, you can create a workflow file to repeat the processing steps you have performed on image batches.

## Time Series

Time series in Tiff format are exported as OME-TIFF files,
with the `TimeIncrement` metadata written
when the time spacing is known. Per the OME-TIFF naming convention,
these files carry the `.ome.tiff` extension.
H5 and zarr exports carry the layout
(`axis_order` attribute) and the spacing when it is known, so a re-import
recovers the time series and its spacing.

For a 3D time series segmentation, the mesh export writes one file per
timepoint, named `{name}_t000.{ext}`, `{name}_t001.{ext}`, and so on, with a
zero-based index. Empty timepoints get their (empty) mesh file too, so the file count
always matches the timepoint count.

## Widget: Output

```python exec="1" html="1"
--8<-- "widgets/output_tab/output.py"
```
