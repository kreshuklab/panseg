import numpy as np
import pytest
import trimesh

from panseg.core.image import (
    ImageLayout,
    ImageProperties,
    PanSegImage,
    SemanticType,
)
from panseg.functionals.dataprocessing.dataprocessing import normalize_01
from panseg.io.voxelsize import VoxelSize
from panseg.tasks.io_tasks import (
    export_image_task,
    import_image_task,
    merge_channels_task,
)
from panseg.tasks.workflow_handler import Task_message


@pytest.mark.parametrize(
    "shape,layout,export_format",
    [
        ((2, 64, 64), ImageLayout.CYX, "tiff"),
        ((2, 64, 64), ImageLayout.CYX, "h5"),
        ((2, 64, 64), ImageLayout.CYX, "zarr"),
        ((2, 64, 32, 32), ImageLayout.CZYX, "h5"),
        ((2, 64, 32, 32), ImageLayout.CZYX, "tiff"),
        ((2, 64, 32, 32), ImageLayout.CZYX, "zarr"),
        # ((64, 64), ImageLayout.YX, "tiff"),
        # ((64, 64), ImageLayout.YX, "h5"),
        # ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_image_io_round_trip_multichannel(tmp_path, shape, layout, export_format):
    mock_data = normalize_01(np.random.rand(*shape).astype("float32"))

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        source_file_name="test",
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="float32",
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
        # tiff alwayes saved as ZCYX
        if layout == ImageLayout.CZYX:
            layout = ImageLayout.ZCYX

    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="test_import",
        semantic_type="raw",
        stack_layout=layout.name,
    )
    assert isinstance(imported_image, list)

    for i in [0, 1]:
        original_data = image.get_data()[i]
        imported_data = imported_image[i].get_data()

        assert np.allclose(original_data, imported_data)
        assert original_data.max() <= 1.0  # check if the normalization is applied
        assert imported_data.max() <= 1.0

        assert image.voxel_size == imported_image[i].voxel_size
        assert image.semantic_type == imported_image[i].semantic_type


@pytest.mark.parametrize(
    "shape,layout,export_format",
    [
        ((32, 64, 64), ImageLayout.ZYX, "tiff"),
        ((32, 64, 64), ImageLayout.ZYX, "h5"),
        ((32, 64, 64), ImageLayout.ZYX, "zarr"),
        ((64, 64), ImageLayout.YX, "tiff"),
        ((64, 64), ImageLayout.YX, "h5"),
        ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_image_io_round_trip(tmp_path, shape, layout, export_format):
    mock_data = np.random.rand(*shape).astype("float32")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        source_file_name="test",
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="float32",
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="raw",
        stack_layout=layout.name,
    )
    assert isinstance(imported_image, PanSegImage)

    # would be normalized during import
    original_data = image.get_data(normalize_01=True)
    imported_data = imported_image.get_data()

    assert np.allclose(original_data, imported_data)
    assert original_data.max() <= 1.0  # check if the normalization is applied
    assert imported_data.max() <= 1.0

    assert image.voxel_size == imported_image.voxel_size
    assert image.semantic_type == imported_image.semantic_type
    assert image.image_layout == imported_image.image_layout


@pytest.mark.parametrize(
    "shape, layout, export_format",
    [
        ((32, 64, 64), ImageLayout.ZYX, "tiff"),
        ((32, 64, 64), ImageLayout.ZYX, "h5"),
        ((32, 64, 64), ImageLayout.ZYX, "zarr"),
        ((64, 64), ImageLayout.YX, "tiff"),
        ((64, 64), ImageLayout.YX, "h5"),
        ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_label_io_round_trip(tmp_path, shape, layout, export_format):
    mock_data = np.random.randint(0, 10, size=[4] * len(shape))
    repeats = np.array(shape) // 4
    for i, rep in enumerate(repeats):
        mock_data = mock_data.repeat(rep, axis=i)
    mock_data = mock_data.astype("uint16")
    assert np.all(mock_data.shape == shape)

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="uint16",
        export_mesh="glb" if len(shape) == 3 else "No",
        close_mesh=False,
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="test_import",
        semantic_type="segmentation",
        stack_layout=layout.name,
    )
    assert isinstance(imported_image, PanSegImage)

    original_data = image.get_data()
    imported_data = imported_image.get_data()

    assert np.allclose(original_data, imported_data)
    assert original_data.max() > 1.0  # check if the normalization is not applied
    assert imported_data.max() > 1.0

    assert image.voxel_size == imported_image.voxel_size
    assert image.semantic_type == imported_image.semantic_type
    assert image.image_layout == imported_image.image_layout


def test_import_image_task_t_layout(make_ome_timeseries):
    path = make_ome_timeseries()
    image = import_image_task(
        input_path=path,
        image_name="timeseries",
        semantic_type="raw",
        stack_layout="TZYX",
    )
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (4, 5, 16, 16)
    assert image.properties.t_spacing is None


def test_import_image_task_tczyx_splits_channels(make_ome_timeseries):
    path = make_ome_timeseries(axes="TCZYX")
    images = import_image_task(
        input_path=path,
        image_name="timeseries",
        semantic_type="raw",
        stack_layout="TCZYX",
    )
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(i.image_layout == ImageLayout.TZYX for i in images)


def test_import_image_task_t_layout_rejects_mismatched_shape(make_ome_timeseries):
    path = make_ome_timeseries()
    result = import_image_task(
        input_path=path,
        image_name="timeseries",
        semantic_type="raw",
        stack_layout="ZYX",
    )
    assert isinstance(result, Task_message)
    assert "incompatible with chosen layout" in result.message


def test_import_image_task_inline_slice_t_first(make_ome_timeseries):
    path = make_ome_timeseries(shape=(4, 5, 20, 60))
    image = import_image_task(
        input_path=path,
        image_name="timeseries",
        semantic_type="raw",
        stack_layout="TZYX[:3,:,:,:50]",
    )
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (3, 5, 20, 50)


def test_import_image_task_length_one_t_slice_squeezes(make_ome_timeseries):
    path = make_ome_timeseries()
    image = import_image_task(
        input_path=path,
        image_name="timeseries",
        semantic_type="raw",
        stack_layout="TZYX[:1,:,:,:]",
    )
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.ZYX
    assert image.properties.t_spacing is None


def test_import_image_task_inline_slicing(make_ome_timeseries):
    path = make_ome_timeseries(shape=(4, 5, 20, 60))
    image = import_image_task(
        input_path=path,
        image_name="timeseries",
        semantic_type="raw",
        stack_layout="tzyx[:3,:,:,:50]",
    )
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (3, 5, 20, 50)


def test_import_image_task_inline_slice_integer_drops_time(make_ome_timeseries):
    path = make_ome_timeseries()
    image = import_image_task(
        input_path=path,
        image_name="timeseries",
        semantic_type="raw",
        stack_layout="TZYX[0,:,:,:]",
    )
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.ZYX
    assert image.properties.t_spacing is None


def test_label_import_image_task_error_message(tmp_path):
    shape = (64, 64)
    layout = ImageLayout.YX
    export_format = "h5"
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="uint16",
    )

    file_path = tmp_path / "test.h5"
    key = "raw"

    result = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="segmentation",
        stack_layout="CYX",
    )
    assert isinstance(result, Task_message)
    assert "Data to import has shape (64, 64)" in result.message


def test_label_io_mesh_error(tmp_path):
    shape = (64, 64)
    export_format = "tiff"

    mock_data = np.random.randint(0, 10, size=[4] * len(shape))
    repeats = np.array(shape) // 4
    for i, rep in enumerate(repeats):
        mock_data = mock_data.repeat(rep, axis=i)
    mock_data = mock_data.astype("uint16")
    assert np.all(mock_data.shape == shape)

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=ImageLayout.YX,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    out = export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="uint16",
        export_mesh="glb",
        close_mesh=True,
    )
    assert isinstance(out, Task_message)
    assert "Mesh export only supported for 3D" in out.message


# --- Time-aware mesh export: a 3D timeseries segmentation writes
# one mesh file per timepoint, empty timepoints included. ---


def _timeseries_mesh_image(seg):
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    return PanSegImage(
        data=seg,
        properties=ImageProperties(
            name="seg",
            semantic_type=SemanticType.SEGMENTATION,
            voxel_size=voxel_size,
            image_layout=ImageLayout.TZYX,
            original_voxel_size=voxel_size,
            t_spacing=10.0,
        ),
    )


@pytest.mark.parametrize("export_mesh", ["glb", "obj", "ply"])
def test_export_mesh_timeseries_one_file_per_timepoint(
    tmp_path, timeseries_segmentation, export_mesh
):
    seg = timeseries_segmentation.copy()
    seg[1] = 0  # an empty timepoint still gets its file
    image = _timeseries_mesh_image(seg)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="seg",
        key="segmentation",
        export_format="tiff",
        data_type="uint16",
        export_mesh=export_mesh,
        close_mesh=False,
    )

    # 1:1 timepoint-file mapping, 0-based, zero-padded to three digits
    expected = [f"seg_t{i:03d}.{export_mesh}" for i in range(4)]
    assert sorted(p.name for p in tmp_path.glob(f"seg_t*.{export_mesh}")) == expected
    assert not (tmp_path / f"seg.{export_mesh}").exists()

    # three labeled blobs per populated timepoint, none in the empty one
    # (glb reloads as a Scene; obj/ply merge into a single mesh)
    for t_index, n_geometries in [(0, 3), (1, 0), (2, 3), (3, 3)]:
        loaded = trimesh.load(tmp_path / f"seg_t{t_index:03d}.{export_mesh}")
        if export_mesh == "glb":
            assert len(loaded.geometry) == n_geometries
        elif n_geometries == 0:
            assert loaded.is_empty
        else:
            assert not loaded.is_empty


def test_export_mesh_still_3d_single_file(tmp_path, timeseries_segmentation):
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image = PanSegImage(
        data=timeseries_segmentation[0],
        properties=ImageProperties(
            name="seg",
            semantic_type=SemanticType.SEGMENTATION,
            voxel_size=voxel_size,
            image_layout=ImageLayout.ZYX,
            original_voxel_size=voxel_size,
        ),
    )
    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="seg",
        key="segmentation",
        export_format="tiff",
        data_type="uint16",
        export_mesh="glb",
        close_mesh=False,
    )
    assert (tmp_path / "seg.glb").exists()
    assert not list(tmp_path.glob("seg_t*.glb"))
    scene = trimesh.load(tmp_path / "seg.glb")
    assert len(scene.geometry) == 3


def test_export_mesh_tyx_segmentation_still_gated(tmp_path):
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image = PanSegImage(
        data=np.zeros((4, 16, 16), dtype="uint16"),
        properties=ImageProperties(
            name="seg",
            semantic_type=SemanticType.SEGMENTATION,
            voxel_size=voxel_size,
            image_layout=ImageLayout.TYX,
            original_voxel_size=voxel_size,
            t_spacing=10.0,
        ),
    )
    out = export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="seg",
        key="segmentation",
        export_format="tiff",
        data_type="uint16",
        export_mesh="glb",
        close_mesh=False,
    )
    assert isinstance(out, Task_message)
    assert "Mesh export only supported for 3D" in out.message
    assert not list(tmp_path.glob("seg_t*.glb"))


def test_io_slicing_trip(tmp_path):
    shape = (32, 64, 64)
    layout = ImageLayout.ZYX
    export_format = "tiff"
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test_raw",
        key="raw",
        export_format=export_format,
        data_type="uint16",
    )

    file_path = tmp_path / "test_raw.tiff"
    key = None

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="segmentation",
        stack_layout="zyx[5:10,:,:50]",
    )
    assert isinstance(imported_image, PanSegImage)

    imported_data = imported_image.get_data()

    assert imported_data.shape == (5, 64, 50)


def test_merge_channels():
    shape = (32, 64, 64)
    layout = ImageLayout.ZYX
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image_1 = PanSegImage(data=mock_data, properties=property)
    image_2 = PanSegImage(data=mock_data, properties=property)
    image_3 = PanSegImage(data=mock_data, properties=property)

    merged = merge_channels_task(image_1=image_1, image_2=image_2, image_3=image_3)
    assert isinstance(merged, PanSegImage)
    assert merged.semantic_type == SemanticType.RAW
    assert merged.image_layout == ImageLayout.CZYX
    assert merged.shape == (3, 32, 64, 64)


def test_merge_channels_one():
    shape = (32, 64, 64)
    layout = ImageLayout.ZYX
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image_1 = PanSegImage(data=mock_data, properties=property)

    merged = merge_channels_task(images=image_1)
    assert isinstance(merged, PanSegImage)
    assert merged.semantic_type == SemanticType.RAW
    assert merged.image_layout == ImageLayout.ZYX
    assert merged.shape == (32, 64, 64)
