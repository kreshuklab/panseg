# Custom Datasets

To train own models from multiple images, you need to create a dataset first.
This dataset is just a file structure so PanSeg knows how to load the training data.
It should look like this:

```
mydataset/
├── train/
│   ├── first.h5
│   └── second.h5
└── val/
    ├── val_one.h5
    └── val_two.h5 
```

Each `h5` file must contain the input image under the `raw` key, and the
segmentation under the `label` key. Both 2D and 3D data are supported, see
[Training data layouts](../panseg_interactive_napari/unet_training.md#training-data-layouts).

The recommended way to train new models is through the [napari training GUI](../panseg_interactive_napari/unet_training.md).
