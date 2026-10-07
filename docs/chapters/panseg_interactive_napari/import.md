# Import Images

## Widget: Open Files

```python exec="1" html="1"
--8<-- "widgets/input_tab/open_file.py"
```

## Stack Layout

When you pick a file, the **Stack layout** field is prefilled. OME-TIFF files
use the axis string from their OME-XML metadata, h5 and zarr files use
the `axis_order` attribute written on export, and everything else falls back
to a guess from the array shape. The layout can be any permutation of the
letters `t`, `c`, `z`, `y`, `x` that match your file, so a 3D time series
gets prefilled as `TZYX`.  
You can invert an axis by prepending it with a minus.
Also you can slice the data by appending, e.g. `tczxy[:10,2]` to only load the
first ten frames of the third channel.  
The slicing functionality can also be used for cropping already during the
import (`CXYZ[1,:50,:50,:10]`).
