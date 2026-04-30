from pathlib import Path
from typing import Optional
import numpy as np

from panseg.core.image import PanSegImage, import_image, save_image
from panseg.tasks import task_tracker
from panseg.tasks.workflow_handler import RunTimeInputSchema, Task_message


@task_tracker(
    is_root=True,
    list_inputs={
        "input_path": RunTimeInputSchema(
            description="Path to a file, or a directory containing files (all files will be imported) or list of paths.",
            required=True,
            is_input_file=True,
        ),
    },
)
def import_image_task(
    input_path: Path,
    semantic_type: str,
    stack_layout: str,
    image_name: str | None = None,
    key: str | None = None,
    m_slicing: str | None = None,
) -> PanSegImage | list[PanSegImage] | Task_message:
    """
    Task wrapper creating a PanSegImage object from an image file.

    Args:
        input_path (Path): path to the image file
        semantic_type (str): semantic type of the image (raw, segmentation, prediction)
        stack_layout (str): stack layout of the image (3D, 2D, 2D_time)
        image_name (str): name of the image, if None the name will be the same as the file name
        key (str | None): key for the image (used only for h5 and zarr formats)
        m_slicing (str | None): m_slicing of the image (None, time, z, y, x)
    """

    if image_name is None:
        image_name = input_path.stem

    try:
        return import_image(
            path=input_path,
            key=key,
            image_name=image_name,
            semantic_type=semantic_type,
            stack_layout=stack_layout,
            m_slicing=m_slicing,
        )
    except Exception as e:
        return Task_message(message=str(e), name="import image", level="warning")


@task_tracker(
    is_leaf=True,
    list_inputs={
        "export_directory": RunTimeInputSchema(
            description="Output directory path where the image will be saved",
            required=True,
        ),
        "name_pattern": RunTimeInputSchema(
            description=(
                "Output file name pattern. Use placeholder {image_name} for "
                "the napari layer name or {file_name} for the input file name"
            ),
            required=False,
        ),
    },
)
def export_image_task(
    image: PanSegImage,
    export_directory: Path,
    name_pattern: str = "{file_name}_export",
    key: str | None = None,
    scale_to_origin: bool = True,
    export_format: str = "tiff",
    data_type: str = "uint16",
    export_mesh: Optional[str] = None,
    close_mesh: bool = False,
) -> None:
    """
    Task wrapper for saving an PanSegImage object to disk.

    Args:
        image (PanSegImage): input image to be saved to disk
        export_directory (Path): output directory path where the image will be saved
        name_pattern (str): output file name pattern, can contain the {image_name} or {file_name} tokens
            to be replaced in the final file name.
        key (str | None): key for the image (used only for h5 and zarr formats).
        scale_to_origin (bool): scale the voxel size to the original one
        export_format (str): file format (tiff, h5, zarr)
        data_type (str): data type to save the image.
    """
    save_image(
        image=image,
        export_directory=export_directory,
        name_pattern=name_pattern,
        key=key,
        scale_to_origin=scale_to_origin,
        export_format=export_format,
        data_type=data_type,
        export_mesh=export_mesh,
        close_mesh=close_mesh,
    )
    return None


@task_tracker
def merge_channels_task(**kwargs) -> PanSegImage:
    """Merge an arbitrary number of PanSegImages

    Pass each image as a named argument.
    The images get sorted according to the key
    """
    images: list[PanSegImage] = [kwargs[k] for k in sorted(kwargs.keys())]
    image = images[0].derive_new(images[0].get_data(), images[0].name + "_merged")
    for im in images[1:]:
        image = image.merge_with(im)
    return image
