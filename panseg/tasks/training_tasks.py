import tempfile
from functools import reduce
from pathlib import Path
from typing import Literal, Optional

from napari.layers import Image, Labels

from panseg import PATH_PANSEG_MODELS, logger
from panseg.core.image import PanSegImage, save_image
from panseg.functionals.training.train import unet_training
from panseg.tasks.workflow_handler import Task_message, task_tracker
from panseg.viewer_napari import log


@task_tracker
def unet_training_task(
    dataset_dir: Optional[Path],
    image: Optional[Image] | list[Image],
    segmentation: Optional[Labels],
    model_name: str,
    in_channels: int,
    out_channels: int,
    feature_maps: int | list[int] | tuple[int, ...],
    patch_size: tuple[int, int, int],
    max_num_iters: int,
    dimensionality: Literal["2D", "3D"],
    device: str,
    modality: str = "",
    output_type: str = "",
    description: str = "",
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pre_trained: Path | None = None,
    widgets_to_reset: Optional[list] = None,
    _tracker: Optional["PBar_Tracker"] = None,
):
    if dataset_dir is None and (image is None or segmentation is None):
        raise ValueError("dataset_dir or (image and segmentation) must not be None!")
    if dataset_dir is not None and (image is not None or segmentation is not None):
        raise ValueError("dataset_dir or (image and segmentation) must be None!")

    # napari images -> make training dataset on the fly
    tmp_dir = None
    if image is not None and segmentation is not None:
        if isinstance(image, list):
            image_mc = PanSegImage.from_napari_layer(image[0])
            for im in image[1:]:
                image_mc = image_mc.merge_with(PanSegImage.from_napari_layer(im))
            pl_image = image_mc
        else:
            pl_image = PanSegImage.from_napari_layer(image)

        tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        out_path = Path(tmp_dir.name)
        train = out_path / "train"
        train.mkdir()
        val = out_path / "val"
        val.mkdir()
        save_image(
            image=pl_image,
            export_directory=train,
            name_pattern="to_train",
            key="raw",
            scale_to_origin=False,
            export_format="h5",
        )
        save_image(
            image=PanSegImage.from_napari_layer(segmentation),
            export_directory=train,
            name_pattern="to_train",
            key="label",
            scale_to_origin=False,
            export_format="h5",
        )
        dataset_dir = out_path

    assert dataset_dir is not None, (
        "Logic error in unet_training_task"
    )  # make pyright happy
    if not all((dataset_dir / d).exists() for d in ["train", "val"]):
        log(
            "Dataset dir must contain a train and a val directory,\n"
            "each containing h5 files!",
            thread="train_gui_task",
        )
        return

    try:
        unet_training(
            dataset_dir=dataset_dir,
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_maps=feature_maps,
            patch_size=patch_size,
            max_num_iters=max_num_iters,
            dimensionality=dimensionality,
            sparse=False,
            device=device,
            modality=modality,
            output_type=output_type,
            description=description,
            resolution=resolution,
            pre_trained=pre_trained,
        )
    except RuntimeError as e:
        if "Output size is too small" in str(e):
            return Task_message(
                "Patch size too small for model architecture!",
                name="training",
                level="warning",
            )

    checkpoint_dir = PATH_PANSEG_MODELS / model_name
    log(f"Finished training, saved model to {checkpoint_dir}", thread="train_gui_task")
    if widgets_to_reset:
        for widget in widgets_to_reset:
            logger.debug(f"Updating: {widget}")
            widget.reset_choices()

    if tmp_dir:
        tmp_dir.cleanup()
