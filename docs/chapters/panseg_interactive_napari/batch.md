# Batch processing

PanSeg supports processing a whole stack of images at once through the
use of batch scripts.

To generate such a workflow script, first process one image in the intended way,
and export the result. Then, go to the `Batch` tab in Napari and save the workflow
script to a `yaml` file.

You can review and edit workflow scripts using the build-in workflow editor;
open it either directly from the napari `Batch` tab, or from the command
line using `panseg -e` .

Read more about [batch workflows in this chapter](../workflow_gui/index.md).

!!! note ""
    You find the batch processing section in the output tab in napari!

!!! warning
    Some interactive steps can't be recorded into workflows, especially **cropping** and **proofreading**

## Input paths

The `input_path` of a workflow accepts four shapes:

| `input_path` | What runs |
| --- | --- |
| a file | one job, one image |
| a directory of files | one job per file |
| a directory of subdirectories | one job per subdirectory, each is a time series (alphanumeric filename order) |
| a list of files | one job, the files stacked as one time series |

If files and subdirectories are mixed in the same directory, the files
are imported and the subdirectories are ignored.

## Widget: Export Batch Workflow

```python exec="1" html="1"
--8<-- "widgets/batch/batch.py"
```
