# Data Processing

Basic data processing functions are provided in the `dataprocessing` module. These functions are used to preprocess data before training a model, or to post-process the output of a model.

## Generic Functions

::: panseg.functionals.dataprocessing.dataprocessing.normalize_01
::: panseg.functionals.dataprocessing.dataprocessing.scale_image_to_voxelsize
::: panseg.functionals.dataprocessing.dataprocessing.image_rescale
::: panseg.functionals.dataprocessing.dataprocessing.image_median
::: panseg.functionals.dataprocessing.dataprocessing.image_gaussian_smoothing
::: panseg.functionals.dataprocessing.dataprocessing.process_images

## Segmentation Functions

::: panseg.functionals.dataprocessing.labelprocessing.relabel_segmentation
::: panseg.functionals.dataprocessing.labelprocessing.set_background_to_value

## Advanced Functions

::: panseg.functionals.dataprocessing.advanced_dataprocessing.fix_over_under_segmentation_from_nuclei
::: panseg.functionals.dataprocessing.advanced_dataprocessing.remove_false_positives_by_foreground_probability
