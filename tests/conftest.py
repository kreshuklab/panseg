# pylint: disable=missing-docstring,import-outside-toplevel

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import skimage.transform as skt
import torch
import yaml
from napari.layers import Image, Labels, Shapes

from panseg.core.image import SemanticType
from panseg.io.io import smart_load

TEST_FILES = Path(__file__).resolve().parent / "resources"
VOXEL_SIZE = (0.235, 0.15, 0.15)
KEY_ZARR = "volumes/new"

IS_CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture
def napari_raw():
    data = np.random.rand(10, 10, 10)
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": SemanticType.RAW,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "ZYX",
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name="test_image_3D")


@pytest.fixture
def napari_raw_2d():
    data = np.random.rand(10, 10)
    voxel_size = None
    metadata = {
        "semantic_type": SemanticType.RAW,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "YX",
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name="test_image_2D")


@pytest.fixture
def napari_raw_4d():
    data = np.random.rand(10, 10, 10, 10)
    voxel_size = None
    metadata = {
        "semantic_type": SemanticType.RAW,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "ZCYX",
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name="test_image_2D")


@pytest.fixture
def napari_prediction():
    data = np.random.rand(10, 10, 10)
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": SemanticType.PREDICTION,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "ZYX",
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name="test_prediction_3D")


@pytest.fixture
def napari_segmentation():
    data = np.random.rand(10, 10, 10)
    data = np.array(data, dtype=np.int8)
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": SemanticType.SEGMENTATION,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "ZYX",
        "id": uuid4(),
    }
    return Labels(data, metadata=metadata, name="test_segmentation_3D")


@pytest.fixture
def napari_no_meta_image():
    data = np.random.rand(10, 10, 10)
    metadata = {}
    return Image(data, metadata=metadata, name="test_image_3D_no_meta")


@pytest.fixture
def napari_no_meta_labels():
    data = np.random.rand(10, 10, 10)
    data = np.array(data, dtype=np.int8)
    metadata = {}
    return Labels(data, metadata=metadata, name="test_label_3D_no_meta")


@pytest.fixture
def napari_shapes():
    return Shapes()


@pytest.fixture
def raw_zcyx_75x2x75x75() -> np.ndarray:
    return smart_load(TEST_FILES / "rgb_3D.tif")


@pytest.fixture
def raw_zcyx_96x2x96x96(raw_zcyx_75x2x75x75):
    return skt.resize(raw_zcyx_75x2x75x75, (96, 2, 96, 96), order=1)


@pytest.fixture
def raw_cell_3d_100x128x128(raw_zcyx_75x2x75x75):
    return skt.resize(raw_zcyx_75x2x75x75[:, 1], (100, 128, 128), order=1)


@pytest.fixture
def raw_cell_2d_96x96(raw_cell_3d_100x128x128):
    return raw_cell_3d_100x128x128[48]


@pytest.fixture
def path_h5(tmpdir) -> Path:
    """Create an HDF5 file using `h5py`'s API with an example dataset for testing purposes."""
    base = Path(tmpdir)
    base.mkdir(exist_ok=True)
    return base / "test.h5"


@pytest.fixture
def path_zarr(tmpdir) -> Path:
    """Create a Zarr file using `zarr`'s API with an example dataset for testing purposes."""
    base = Path(tmpdir)
    base.mkdir(exist_ok=True)
    return base / "test.zarr"


@pytest.fixture
def path_tiff(tmpdir) -> Path:
    """Create a TIFF file using `tifffile`'s API with an example dataset for testing purposes."""
    base = Path(tmpdir)
    base.mkdir(exist_ok=True)
    return base / "test.tiff"


@pytest.fixture
def path_jpg(tmpdir) -> Path:
    """Create a JPG file using `PIL`'s API with an example image for testing purposes."""
    base = Path(tmpdir)
    base.mkdir(exist_ok=True)
    return base / "test.jpg"


@pytest.fixture
def preprocess_config(path_file_hdf5):
    """Create pipeline config with only pre-processing (Gaussian filter) enabled."""
    config_path = TEST_FILES / "test_config.yaml"
    config = yaml.full_load(config_path.read_text())
    # Add the file path to process
    config["path"] = path_file_hdf5
    # Enable Gaussian smoothing for some work
    config["preprocessing"]["state"] = True
    config["preprocessing"]["filter"]["state"] = True
    return config


@pytest.fixture
def prediction_config(tmpdir):
    """Create pipeline config with Unet prediction enabled.

    Prediction will be executed on the `tests/resources/sample_ovules.h5`.
    The `sample_ovules.h5` file is copied to the temporary directory to avoid
    creating unnecessary files in `tests/resources`.
    """
    # Load the test configuration
    config_path = TEST_FILES / "test_config.yaml"
    config = yaml.full_load(config_path.read_text())
    # Enable UNet prediction
    config["cnn_prediction"]["state"] = True
    # Copy `sample_ovule.h5` to the temporary directory
    sample_ovule_path = TEST_FILES / "sample_ovule.h5"
    tmp_path = Path(tmpdir) / "sample_ovule.h5"
    shutil.copy2(sample_ovule_path, tmp_path)
    # Add the temporary path to the config
    config["path"] = str(tmp_path)  # Ensure the path is a string
    return config


@pytest.fixture
def complex_test_data():
    """
    Generates a complex 3D dataset with both under-segmented and over-segmented cells.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: cell segmentation, nuclei segmentation, and boundary probability map.
    """
    # Create a 3D grid of zeros
    cell_seg = np.zeros((10, 10, 10), dtype=np.uint16)
    nuclei_seg = np.zeros_like(cell_seg, dtype=np.uint16)

    # Define cells with under-segmentation (multiple nuclei in one cell)
    # Cell 1: covers (2, 2, 2) to (5, 5, 5), contains two nuclei
    cell_seg[2:6, 2:6, 2:6] = 1
    nuclei_seg[2:4, 2:3, 2:3] = 1
    nuclei_seg[4:6, 5:6, 5:6] = 2

    # Define cells with over-segmentation (one nucleus split into multiple cells)
    # Cell 2 and 3: cover (6, 6, 6) to (8, 8, 8), with one nucleus overlapping both cells
    cell_seg[6:8, 6:10, 6:10] = 2
    cell_seg[8:10, 6:10, 6:10] = 3
    nuclei_seg[7:9, 7:9, 7:9] = 3

    # Define another under-segmented region with a large cell and multiple nuclei
    # Cell 4: covers (1, 1, 6) to (3, 3, 8), contains two nuclei
    cell_seg[1:4, 1:4, 6:9] = 4
    nuclei_seg[1:2, 1:2, 6:7] = 4
    nuclei_seg[3:4, 3:4, 8:9] = 5

    # Generate a boundary probability map with higher values on the edges of the cells
    boundary_pmap = np.ones_like(cell_seg, dtype=np.float32)
    boundary_pmap[2:6, 2:6, 2:6] = 0.2
    boundary_pmap[6:8, 6:8, 6:8] = 0.2
    boundary_pmap[1:4, 1:4, 6:9] = 0.2

    return cell_seg, nuclei_seg, boundary_pmap


@pytest.fixture
def workflow_yaml(tmpdir: Path):
    return Path(shutil.copy2(TEST_FILES / "test_workflow.yaml", tmpdir))


@pytest.fixture
def workflow_complete_yaml(tmpdir: Path):
    return Path(shutil.copy2(TEST_FILES / "test_complete_workflow.yaml", tmpdir))


@pytest.fixture
def workflow_aio_yaml(tmpdir: Path):
    return Path(shutil.copy2(TEST_FILES / "test_workflow_aio.yaml", tmpdir))


@pytest.fixture
def zarr_file_empty():
    return TEST_FILES / "empty.zarr"


@pytest.fixture
def zarr_file_3d():
    return TEST_FILES / "3d.zarr"


@pytest.fixture
def h5_file():
    return TEST_FILES / "sample_ovule.h5"
