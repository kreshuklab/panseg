# Import Images

## Widget: Open Files

```python exec="1" html="1"
--8<-- "widgets/input_tab/open_file.py"
```

## Time Series

Importing time series data is supported in PanSeg. If your file has a time
dimension, just load it and specify the time dimension in the stack layout
as `t`.
Otherwise, you can load a time series by selecting multiple files
during the import. The files are stacked in alphanumeric filename
order, so `a2.tif` comes before `a10.tif`. When importing from multiple files,
slicing is not supported.

## Stack Layout

PanSeg does its best to guess the stack layout of the file you are loading,
but always make sure this guess is correct.
The layout can be any permutation of the
letters `t`, `c`, `z`, `y`, `x` that match your file, so a 3D time series
gets prefilled as `TZYX`.  

You can invert an axis by prepending it with a minus.
Also, slicing is supported: slice the data by appending, e.g. `tczxy[:10,2]`
to only load the first ten frames of the third channel.  
The slicing functionality can also be used for cropping already during the
import (`CXYZ[1,:50,:50,:10]`).
