from pathlib import Path
from typing import Optional

from panseg.core.image import ImageLayout, PanSegImage, SemanticType
from panseg.functionals.dataprocessing import fix_layout
from panseg.functionals.prediction import biio_prediction, unet_prediction
from panseg.tasks import task_tracker


@task_tracker
def unet_prediction_task(
    image: PanSegImage,
    model_name: str | None,
    model_id: str | None,
    suffix: str = "_prediction",
    patch: tuple[int, int, int] | None = None,
    patch_halo: tuple[int, int, int] | None = None,
    single_batch_mode: bool = True,
    device: str = "cuda",
    model_update: bool = False,
    disable_tqdm: bool = False,
    config_path: Path | None = None,
    model_weights_path: Path | None = None,
    _tracker: Optional["PBar_Tracker"] = None,
) -> list[PanSegImage]:
    """
    Apply a trained U-Net model to a PanSegImage object.

    Args:
        image (PanSegImage): input image object
        model_name (str): the name of the model to use
        model_id (str): the ID of the model to use
        suffix (str): suffix to append to the new image name
        patch (tuple[int, int, int]): patch size for prediction
        single_batch_mode (bool): whether to use a single batch for prediction
        device (str): the computation device ('cpu', 'cuda', etc.)
        model_update (bool): whether to update the model to the latest version
        _tracker: Tracker of progress bar. Not used in workflow as it starts with `_`
    """
    data = image.get_data()
    input_layout = image.image_layout

    pmaps = unet_prediction(
        raw=data,
        input_layout=input_layout.value,
        model_name=model_name,
        model_id=model_id,
        patch=patch,
        patch_halo=patch_halo,
        single_batch_mode=single_batch_mode,
        device=device,
        model_update=model_update,
        disable_tqdm=disable_tqdm,
        config_path=config_path,
        model_weights_path=model_weights_path,
        tracker=_tracker,
    )
    assert pmaps.ndim == 4, f"Expected 4D CZXY prediction, got {pmaps.ndim}D"

    new_images = []

    for i, pmap in enumerate(pmaps):
        # Input layout is always ZYX this loop
        pmap = fix_layout(
            pmap, input_layout=ImageLayout.ZYX.value, output_layout=input_layout.value
        )
        new_images.append(
            image.derive_new(
                pmap,
                name=f"{image.name}_{suffix}_{i}",
                semantic_type=SemanticType.PREDICTION,
                image_layout=input_layout,
            )
        )

    return new_images


@task_tracker
def biio_prediction_task(
    image: PanSegImage,
    model_id: str,
    device: str,
    suffix: str = "_prediction",
    _tracker: Optional["PBar_Tracker"] = None,
) -> list[PanSegImage]:
    data = image.get_data()
    input_layout = image.image_layout.value

    named_pmaps = biio_prediction(
        raw=data,
        input_layout=input_layout,
        model_id=model_id,
        device=device,
    )

    new_images = []
    for name, pmap in named_pmaps.items():
        # Input layout is always CZYX this loop
        new_images.append(
            image.derive_new(
                pmap,
                name=f"{image.name}_{suffix}_{name}",
                semantic_type=SemanticType.PREDICTION,
                image_layout="CZYX",
            )
        )
    return new_images
