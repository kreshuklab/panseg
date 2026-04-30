# PanSeg Interactive - Napari

Interactively apply operations to your images and run the segmentation
pipeline using the GUI of PanSeg.
PanSeg uses napari as its front-end.

Start PanSeg according to your chosen [installation method](../getting_started/installation.md).
From the terminal, you can start PanSeg in the GUI mode with:

```bash
panseg --napari
or
panseg -n
```

![PanSeg 2.0 interface](https://github.com/kreshuklab/panseg/raw/assets/images/plantseg2gui.png)

## Parts of the GUI

### Top left: Settings

* Change your **tool** (pan, brush, boxselect, ...)
* Change **look of image** (opacity, label colors,..)
* Change **tool settings** (brush color & size, select shape)

### Bottom left: Layer selection

Your loaded layers get displayed here. Toggle their visibility with the eye icon.
Below, you can switch to a **grid view** or a **3D view** with the respective icons.

### Right: PanSeg

This is the main interface of PanSeg.
On the bottom you can switch through the different sections of PanSeg.

## Hotkeys

### General

* `Ctrl+i` Increase font size
* `Ctrl+o` Decrease font size
* `Ctrl+w` Quit PanSeg

### Proofreading

* `n` Merge/Split using current scribbles
* `j` Clean scribbles
