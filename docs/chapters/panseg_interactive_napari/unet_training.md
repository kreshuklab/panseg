# UNet Training

With PanSeg you can train bespoke segmentation models!
This is especially useful to get great results on a big dataset for which
the build-in models work not perfectly.  
First proofread some images from the dataset. Then train a new model on this
high-quality data, and run it on the whole dataset.

You can also fine-tune existing models on your data.

## Training from a dataset

For training from an dataset stored on disk, create the directories `train` and `val`.
Your training images must be stored as `h5` files in these directories.
The `h5` files must contain the input image under the `raw` key, and the
segmentation under the `label` key. The label must be channel-less.

```
mydataset/
├── train/
│   ├── first.h5
│   └── second.h5
└── val/
    ├── val_one.h5
    └── val_two.h5 
```

## Training data layouts

Each `h5` file may hold 2D or 3D data; which models can be trained depends on
the layout of the `raw`/`label` pair:

| `raw` | `label` | `2D` model | `3D` model |
| --- | --- | --- | --- |
| `yx` (2D) | `yx` (2D) | ✓ | – |
| `cyx` (2D) | `yx` (2D) | ✓ | – |
| `zyx` (3D) | `zyx` (3D) | ✓, per z-slice | ✓ |
| `czyx` (3D) | `zyx` (3D) | ✓, per z-slice | ✓ |

Any other combination is rejected with a clear error. The label must always be
channel-less.

PanSeg resolves the layout of each dataset from its `axis_order` attribute when
present (PanSeg's own `h5` exports carry it); otherwise it is inferred from the
shapes of the `raw`/`label` pair as listed in the table. See
[Advanced Training](../panseg_models/training.md) for the details.

When training a `2D` model on `zyx`/`czyx` data, every z-slice is an independent
2D sample. The patch z-extent must then be `1` (the z-stride
follows automatically).

## Train from GUI

To train from images loaded in the GUI, you need a layer containing the input
image and one containing the segmentation. Make sure the quality of the
segmentation is as good as possible by using the proofreading tool.

For 2D (`yx`/`cyx`) data only `2D` training is possible. For volumetric
(`zyx`/`czyx`) data, the model settings offer a `2D`/`3D` choice (default `3D`):
`2D` trains a 2D model per z-slice as described above and freezes the patch size
in z to `1`.

## Widgets

```python exec="1" html="1"
--8<-- "widgets/training/training.py"
```
