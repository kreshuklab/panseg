# Output

Using the Output tab you can save your work as `tiff`, `h5`, or `zarr`.

Labels can be exported as mesh files (`glb`, `obj` or `ply`).
Use `glb` if possible, the other formats are missing the color information,
or merge all segments into a single mesh.

Once you have exported an image, you can create a workflow file to repeat the processing steps you have performed on image batches.

## Widget: Output

```python exec="1" html="1"
--8<-- "widgets/output_tab/output.py"
```
