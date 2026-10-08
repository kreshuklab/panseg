# pylint: disable=missing-docstring,import-outside-toplevel

import itertools
import shutil
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree

import numpy as np
import pytest
import skimage.transform as skt
import tifffile
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


def _napari_timeseries_layer(t_props: dict) -> Image:
    """RAW TZYX napari layer; the time metadata comes from one of the
    TIMESERIES_PROPS_* dicts below (known vs unknown t_spacing)."""
    data = np.random.rand(4, 5, 16, 16).astype("float32")
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": SemanticType.RAW,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "TZYX",
        "t_unit": "s",
        **t_props,
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name="test_timeseries")


@pytest.fixture
def napari_timeseries():
    """Timeseries napari layer with a known t_spacing (10 s)."""
    return _napari_timeseries_layer(TIMESERIES_PROPS_KNOWN_T_SPACING)


@pytest.fixture
def napari_timeseries_unknown_t_spacing():
    """Timeseries napari layer with unknown t_spacing."""
    return _napari_timeseries_layer(TIMESERIES_PROPS_UNKNOWN_T_SPACING)


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
def workflow_t_spacing_yaml(tmpdir: Path):
    return Path(shutil.copy2(TEST_FILES / "test_workflow_t_spacing.yaml", tmpdir))


@pytest.fixture
def zarr_file_empty():
    return TEST_FILES / "empty.zarr"


@pytest.fixture
def zarr_file_3d():
    return TEST_FILES / "3d.zarr"


@pytest.fixture
def h5_file():
    return TEST_FILES / "sample_ovule.h5"


# --- Time fixtures (time-dimension spec) ---
#
# Synthetic raw timeseriess on one shape skeleton: T=4, C=2, Z=5, Y=X=16.
# Tests build PanSegImage inline from these arrays; known-vs-unknown
# t_spacing are the two property dicts below, not separate fixtures.

TIMESERIES_PROPS_KNOWN_T_SPACING = {"t_spacing": 10.0, "t_unit": "s"}
TIMESERIES_PROPS_UNKNOWN_T_SPACING = {"t_spacing": None}


def _timeseries_raw(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype("float32")


@pytest.fixture
def timeseries_tyx() -> np.ndarray:
    """Raw TYX float32 timeseries, shape (4, 16, 16)."""
    return _timeseries_raw((4, 16, 16), seed=11)


@pytest.fixture
def timeseries_tcyx() -> np.ndarray:
    """Raw TCYX float32 timeseries, shape (4, 2, 16, 16)."""
    return _timeseries_raw((4, 2, 16, 16), seed=12)


@pytest.fixture
def timeseries_tzyx() -> np.ndarray:
    """Raw TZYX float32 timeseries, shape (4, 5, 16, 16)."""
    return _timeseries_raw((4, 5, 16, 16), seed=13)


@pytest.fixture
def timeseries_tczyx() -> np.ndarray:
    """Raw TCZYX float32 timeseries, shape (4, 2, 5, 16, 16)."""
    return _timeseries_raw((4, 2, 5, 16, 16), seed=14)


def _timeseries_segmentation() -> np.ndarray:
    """uint16 TZYX segmentation, shape (4, 5, 16, 16).

    Label IDs are independent across timepoints by construction: timepoint
    t carries the disjoint ID range 3t+1..3t+3, so no label ID ever appears
    in two timepoints.
    """
    t, z, y, x = 4, 5, 16, 16
    blob = 3
    rng = np.random.default_rng(15)
    seg = np.zeros((t, z, y, x), dtype="uint16")
    for t_index in range(t):
        for j in range(3):
            z0 = int(rng.integers(0, z - blob + 1))
            y0 = int(rng.integers(0, y - blob + 1))
            x0 = int(rng.integers(0, x - blob + 1))
            seg[t_index, z0 : z0 + blob, y0 : y0 + blob, x0 : x0 + blob] = (
                t_index * 3 + j + 1
            )
    return seg


@pytest.fixture
def timeseries_segmentation() -> np.ndarray:
    """uint16 TZYX segmentation timeseries; label IDs are independent across timepoints."""
    return _timeseries_segmentation()


def _timeseries_labels_data() -> np.ndarray:
    """Deterministic uint16 TZYX labels, shape (3, 4, 10, 10).

    Label IDs are disjoint per timepoint and the positions are fixed, so
    handler/widget assertions can be exact: t=0 carries {1, 2}, t=1 {3, 4},
    t=2 {5}.
    """
    seg = np.zeros((3, 4, 10, 10), dtype="uint16")
    seg[0, 0:2, 0:2, 0:2] = 1
    seg[0, 0:2, 5:7, 5:7] = 2
    seg[1, 0:2, 0:2, 0:2] = 3
    seg[1, 2:4, 5:7, 5:7] = 4
    seg[2, 0:2, 0:2, 0:2] = 5
    return seg


def _napari_image_layer(
    data: np.ndarray,
    name: str,
    image_layout: str,
    semantic_type: SemanticType,
) -> Image:
    """Image napari layer with panseg metadata; time spacing only for T layouts."""
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": semantic_type,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": image_layout,
        "id": uuid4(),
    }
    if "T" in image_layout:
        metadata["t_spacing"] = 10.0
        metadata["t_unit"] = "s"
    return Image(data, metadata=metadata, name=name)


def _timeseries_labels_layer(data: np.ndarray, name: str, image_layout: str) -> Labels:
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": SemanticType.SEGMENTATION,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": image_layout,
        "t_spacing": 10.0,
        "t_unit": "s",
        "id": uuid4(),
    }
    return Labels(data, metadata=metadata, name=name)


@pytest.fixture
def napari_timeseries_segmentation() -> Labels:
    """Labels napari layer, TZYX layout, deterministic per-timepoint label IDs."""
    return _timeseries_labels_layer(
        _timeseries_labels_data(), "test_segmentation_timeseries", "TZYX"
    )


@pytest.fixture
def napari_timeseries_segmentation_2d() -> Labels:
    """Labels napari layer, TYX layout, deterministic per-timepoint label IDs."""
    data = _timeseries_labels_data().max(axis=1)
    data[1, 5:7, 0:2] = 4
    data[2, 5:7, 5:7] = 6
    return _timeseries_labels_layer(data, "test_segmentation_timeseries_2d", "TYX")


@pytest.fixture
def napari_timeseries_prediction() -> Image:
    """PREDICTION napari layer, TZYX layout, matching the deterministic timeseries labels shape."""
    data = np.random.default_rng(16).random((3, 4, 10, 10)).astype("float32")
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": SemanticType.PREDICTION,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "TZYX",
        "t_spacing": 10.0,
        "t_unit": "s",
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name="test_prediction_timeseries")


@pytest.fixture
def napari_raw_tyx() -> Image:
    """RAW TYX image layer, shape (4, 16, 16)."""
    data = np.random.default_rng(21).random((4, 16, 16)).astype("float32")
    return _napari_image_layer(data, "test_image_tyx", "TYX", SemanticType.RAW)


@pytest.fixture
def napari_raw_cyx() -> Image:
    """RAW CYX image layer, shape (2, 16, 16)."""
    data = np.random.default_rng(22).random((2, 16, 16)).astype("float32")
    return _napari_image_layer(data, "test_image_cyx", "CYX", SemanticType.RAW)


@pytest.fixture
def napari_raw_czyx() -> Image:
    """RAW CZYX image layer, shape (2, 5, 16, 16)."""
    data = np.random.default_rng(23).random((2, 5, 16, 16)).astype("float32")
    return _napari_image_layer(data, "test_image_czyx", "CZYX", SemanticType.RAW)


@pytest.fixture
def napari_raw_tczyx() -> Image:
    """RAW TCZYX image layer, shape (4, 2, 5, 16, 16)."""
    data = np.random.default_rng(24).random((4, 2, 5, 16, 16)).astype("float32")
    return _napari_image_layer(data, "test_image_tczyx", "TCZYX", SemanticType.RAW)


@pytest.fixture
def napari_prediction_tyx() -> Image:
    """PREDICTION TYX image layer, shape (4, 16, 16)."""
    data = np.random.default_rng(25).random((4, 16, 16)).astype("float32")
    return _napari_image_layer(
        data, "test_prediction_tyx", "TYX", SemanticType.PREDICTION
    )


@pytest.fixture
def napari_prediction_czyx() -> Image:
    """PREDICTION CZYX image layer, shape (2, 5, 16, 16)."""
    data = np.random.default_rng(26).random((2, 5, 16, 16)).astype("float32")
    return _napari_image_layer(
        data, "test_prediction_czyx", "CZYX", SemanticType.PREDICTION
    )


@pytest.fixture
def napari_prediction_tczyx() -> Image:
    """PREDICTION TCZYX image layer, shape (4, 2, 5, 16, 16)."""
    data = np.random.default_rng(27).random((4, 2, 5, 16, 16)).astype("float32")
    return _napari_image_layer(
        data, "test_prediction_tczyx", "TCZYX", SemanticType.PREDICTION
    )


# --- Synthetic OME-TIFF builders (time-dimension spec) ---
#
# The committed anchors under tests/resources/ome_tiff_examples/ carry no
# timing metadata, so every timing variant is synthesized into tmp_path at
# test time.

_OME_XML_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"

# The shared shape skeleton (T, C, Z, Y, X); each layout is the projection of
# the canonical order onto its present axes.
TIMESERIES_SHAPE_SKELETON = (4, 2, 5, 16, 16)

_TIMESERIES_OME_SHAPES: dict[str, tuple[int, ...]] = {
    axes: tuple(n for ax, n in zip("TCZYX", TIMESERIES_SHAPE_SKELETON) if ax in axes)
    for axes in ("TYX", "TCYX", "TZYX", "TCZYX")
}

# First-plane DeltaT differences of the nonuniform_plane_delta_t variant;
# deliberately non-uniform so differencing cannot yield a spacing.
_NONUNIFORM_DELTA_T_DIFFS = (1000, 2000, 4000)


def _write_ome_timeseries(
    path: Path,
    axes: str,
    shape: tuple[int, ...],
    seed: int,
    t_increment: float | None = None,
    t_increment_unit: str = "s",
    plane_delta_t: Sequence[int | float | None] | None = None,
    plane_delta_t_unit: str = "ms",
) -> Path:
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) * 4096).astype("uint16")
    metadata: dict = {"axes": axes}
    if t_increment is not None:
        metadata["TimeIncrement"] = t_increment
        metadata["TimeIncrementUnit"] = t_increment_unit
    if plane_delta_t is not None:
        # one attribute dict per plane in raster order; an empty dict
        # omits DeltaT/DeltaTUnit for that plane (tifffile skips absent
        # per-plane attributes but serializes None values as the string
        # "None", so None must be dropped here)
        metadata["Plane"] = [
            {} if value is None else {"DeltaT": value, "DeltaTUnit": plane_delta_t_unit}
            for value in plane_delta_t
        ]
    tifffile.imwrite(path, data, ome=True, photometric="minisblack", metadata=metadata)
    return path


def _plane_delta_t_sequence(
    axes: str, shape: tuple[int, ...], per_timepoint: Sequence[int | float | None]
) -> list[int | float | None]:
    """Per-plane DeltaT values in plane raster order (last page axis fastest).

    Each plane carries the absolute DeltaT given for its timepoint; a None
    entry yields no DeltaT attribute for that timepoint's planes (the
    caller turns None entries into empty per-plane attribute dicts, which
    tifffile writes as attribute-less Plane elements).
    """
    page_shape = shape[:-2]
    t_axis = axes.index("T")
    return [
        per_timepoint[coords[t_axis]]
        for coords in itertools.product(*[range(n) for n in page_shape])
    ]


@pytest.fixture
def make_ome_timeseries(tmp_path):
    """Factory for synthetic OME-TIFF timeseries written into tmp_path.

    Defaults to the TZYX slice of the shared shape skeleton. ``t_increment``
    writes the Pixels TimeIncrement/TimeIncrementUnit attributes.
    ``plane_delta_t`` writes realistic ABSOLUTE Plane.DeltaT times (OME's
    DeltaT is the time of a plane since acquisition start, not an
    increment): a scalar is the uniform time step, so every plane of
    timepoint i carries DeltaT = i * plane_delta_t, while a sequence gives
    the per-timepoint DeltaT directly (a None entry omits the attribute
    for that timepoint's planes, i.e. partially documented timing).
    ``nonuniform_plane_delta_t`` writes per-timepoint absolute DeltaT
    whose first-plane differences are non-uniform (0, 1000, 3000, 7000,
    ... in the given unit). With none of them the file carries no timing
    metadata. Returns the written file path.
    """
    counter = 0

    def _make(
        axes: str = "TZYX",
        shape: tuple[int, ...] | None = None,
        t_increment: float | None = None,
        t_increment_unit: str = "s",
        plane_delta_t: float | Sequence[float | None] | None = None,
        nonuniform_plane_delta_t: bool = False,
        plane_delta_t_unit: str = "ms",
    ) -> Path:
        nonlocal counter
        counter += 1
        if shape is None:
            shape = _TIMESERIES_OME_SHAPES[axes]
        assert len(shape) == len(axes)
        n_t = shape[axes.index("T")]
        if nonuniform_plane_delta_t:
            # non-uniform absolute acquisition times: 0, 1000, 3000, 7000,
            # 8000, ... (first-plane differences 1000, 2000, 4000, ...)
            per_timepoint = [0]
            for diff in itertools.islice(
                itertools.cycle(_NONUNIFORM_DELTA_T_DIFFS), n_t - 1
            ):
                per_timepoint.append(per_timepoint[-1] + diff)
        elif isinstance(plane_delta_t, Sequence):
            # explicit per-timepoint absolute DeltaT; None omits the
            # attribute for that timepoint's planes
            per_timepoint = list(plane_delta_t)
            assert len(per_timepoint) == n_t
        elif plane_delta_t is not None:
            # realistic absolute times for a uniform timelapse: the first
            # plane of timepoint i sits at i * plane_delta_t (constant per
            # timepoint, which is realistic enough; the reader reads only
            # first planes)
            per_timepoint = [i * plane_delta_t for i in range(n_t)]
        else:
            per_timepoint = None
        sequence = (
            None
            if per_timepoint is None
            else _plane_delta_t_sequence(axes, shape, per_timepoint)
        )
        return _write_ome_timeseries(
            tmp_path / f"synthetic_ome_{counter}.ome.tif",
            axes,
            shape,
            seed=1000 + counter,
            t_increment=t_increment,
            t_increment_unit=t_increment_unit,
            plane_delta_t=sequence,
            plane_delta_t_unit=plane_delta_t_unit,
        )

    return _make


def _ome_root(path: Path) -> ElementTree.Element:
    with tifffile.TiffFile(path) as tiff:
        return ElementTree.fromstring(tiff.ome_metadata)


def _save_ome_description(path: Path, root: ElementTree.Element) -> None:
    ElementTree.register_namespace("", _OME_XML_NS)
    xml = '<?xml version="1.0" encoding="UTF-8"?>' + ElementTree.tostring(
        root, encoding="unicode"
    )
    with tifffile.TiffFile(path, mode="r+") as tiff:
        tiff.pages[0].tags["ImageDescription"].overwrite(xml.encode("ascii"))


def _ome_multifile_chain(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    """Two-file OME-TIFF UUID/FileName chain.

    A 4-timepoint TZYX timeseries (T=4, Z=2, Y=X=16) is split 2+2 across two
    files. The first file's OME-XML is patched to describe the full
    timeseries: its TiffData entry gains a UUID child naming the first file,
    and a second TiffData entry (FirstT=2) is appended whose UUID child
    names the second file. The second file stays a plain 2-timepoint
    OME-TIFF.
    """
    t, z, y, x = 4, 2, 16, 16
    axes = "TZYX"
    rng = np.random.default_rng(42)
    data = (rng.random((t, z, y, x)) * 4096).astype("uint16")
    first = tmp_path / "multifile_first.ome.tif"
    second = tmp_path / "multifile_second.ome.tif"
    tifffile.imwrite(
        first,
        data[: t // 2],
        ome=True,
        photometric="minisblack",
        metadata={"axes": axes},
    )
    tifffile.imwrite(
        second,
        data[t // 2 :],
        ome=True,
        photometric="minisblack",
        metadata={"axes": axes},
    )
    uuid_first = _ome_root(first).get("UUID")
    uuid_second = _ome_root(second).get("UUID")

    root_first = _ome_root(first)
    image = next(e for e in root_first if e.tag.endswith("Image"))
    pixels = next(e for e in image if e.tag.endswith("Pixels"))
    pixels.set("SizeT", str(t))
    own_data = next(e for e in pixels if e.tag.endswith("TiffData"))
    own_uuid = ElementTree.SubElement(own_data, f"{{{_OME_XML_NS}}}UUID")
    own_uuid.set("FileName", first.name)
    own_uuid.text = uuid_first
    other_data = ElementTree.Element(f"{{{_OME_XML_NS}}}TiffData")
    other_data.set("FirstT", str(t // 2))
    other_data.set("FirstZ", "0")
    other_data.set("IFD", "0")
    other_data.set("PlaneCount", str(int(np.prod((t // 2, z)))))
    other_uuid = ElementTree.SubElement(other_data, f"{{{_OME_XML_NS}}}UUID")
    other_uuid.set("FileName", second.name)
    other_uuid.text = uuid_second
    pixels.append(other_data)
    _save_ome_description(first, root_first)
    return first, second, data


@pytest.fixture
def ome_timeseries_multifile(tmp_path):
    """Two-file OME-TIFF UUID/FileName chain (see ``_ome_multifile_chain``).

    Returns (first_path, second_path, full_timeseries_data).
    """
    return _ome_multifile_chain(tmp_path)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """TEMPORARY (macOS freeze investigation): per-test resource probe.

    Prints fd/thread/child-process/memory counters straight to the real
    stderr so they survive pytest's capture and land in the CI log even
    when a later test wedges the process.
    """
    yield
    import os
    import sys
    import threading

    try:
        import psutil

        proc = psutil.Process()
        rss = proc.memory_info().rss / (1024 * 1024)
        kids = len(proc.children(recursive=True))
    except Exception:
        rss = kids = -1
    try:
        fds = len(os.listdir("/dev/fd"))
    except Exception:
        fds = -1
    print(
        f"\nRESLOG {item.nodeid} fds={fds} threads={threading.active_count()} "
        f"kids={kids} rss={rss:.0f}MB",
        file=sys.__stderr__,
        flush=True,
    )


def pytest_configure(config):
    """TEMPORARY (macOS freeze investigation): session-wide stack watchdog.

    If 120 s pass without the suite finishing anything, dump the stacks of
    all threads to the GitHub step summary (uploaded by the runner even when
    the step dies via os._exit) and hard-exit so the job fails visibly
    instead of hanging silently. Inert outside GitHub Actions.
    """
    import faulthandler
    import os

    timeout = float(os.environ.get("PANSEG_WATCHDOG_TIMEOUT", "120"))
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path or timeout <= 0:
        return
    handle = open(path, "w")
    faulthandler.dump_traceback_later(timeout, file=handle, exit=True)
