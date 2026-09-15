import numpy as np
import pytest
import zarr

from panseg.io.voxelsize import VoxelSize
from panseg.io.zarr import (
    create_zarr,
    del_zarr_key,
    list_zarr_keys,
    load_zarr,
    read_zarr_shape,
    read_zarr_voxel_size,
    rename_zarr_key,
)

SHAPES = (2, 3, 4)


def _fill(group, key, value, dtype="float32"):
    array = group.create_array(name=key, shape=SHAPES, dtype=dtype)
    array[:] = np.full(SHAPES, value, dtype=dtype)
    return array


def test_create_zarr_roundtrip_small(tmp_path):
    out = tmp_path / "out.zarr"
    data = np.random.random(SHAPES).astype("float32")
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.2))
    create_zarr(out, data, "raw", voxel_size, mode="w")
    assert out.exists()
    assert np.array_equal(load_zarr(out, key="raw"), data)
    assert read_zarr_shape(out, key="raw") == SHAPES
    assert read_zarr_voxel_size(out, key="raw") == voxel_size
    assert list_zarr_keys(out) == ["raw"]


def test_create_zarr_mode_a_appends(tmp_path):
    out = tmp_path / "out.zarr"
    data = np.random.random(SHAPES).astype("float32")
    voxel_size = VoxelSize()
    create_zarr(out, data, "raw", voxel_size, mode="w")
    create_zarr(out, data, "prediction", voxel_size, mode="a")
    assert sorted(list_zarr_keys(out)) == ["prediction", "raw"]
    assert np.array_equal(load_zarr(out, key="prediction"), data)


def test_create_zarr_mode_w_wipes(tmp_path):
    out = tmp_path / "out.zarr"
    data = np.random.random(SHAPES).astype("float32")
    voxel_size = VoxelSize()
    create_zarr(out, data, "raw", voxel_size, mode="w")
    create_zarr(out, data, "prediction", voxel_size, mode="a")
    create_zarr(out, data, "raw", voxel_size, mode="w")
    assert list_zarr_keys(out) == ["raw"]


def test_create_zarr_rejects_empty_key(tmp_path):
    out = tmp_path / "out.zarr"
    with pytest.raises(ValueError, match="Key cannot be"):
        create_zarr(out, np.zeros(SHAPES, dtype="float32"), "", VoxelSize())


def test_load_zarr_missing_file(tmp_path):
    with pytest.raises(AssertionError, match="File not found"):
        load_zarr(tmp_path / "missing.zarr", key="raw")


def test_load_zarr_not_a_zarr_file(tmp_path):
    path = tmp_path / "not_zarr.txt"
    path.write_text("hello")
    with pytest.raises(AssertionError, match="not a Zarr file"):
        load_zarr(path, key="raw")


def test_load_zarr_missing_key(tmp_path):
    out = tmp_path / "out.zarr"
    create_zarr(out, np.zeros(SHAPES, dtype="float32"), "raw", VoxelSize(), mode="w")
    with pytest.raises(KeyError):
        load_zarr(out, key="nope")


def test_load_zarr_key_is_group(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    group.create_group("grp")
    _fill(group, "data", 0.0)
    with pytest.raises(ValueError, match="not a zarr.Array"):
        load_zarr(tmp_path / "out.zarr", key="grp")


def test_load_zarr_slicing(tmp_path):
    out = tmp_path / "out.zarr"
    data = np.arange(np.prod(SHAPES), dtype="float32").reshape(SHAPES)
    create_zarr(out, data, "raw", VoxelSize(), mode="w")
    sliced = load_zarr(out, key="raw", slices=slice(0, 1))
    assert sliced.shape == (1, 3, 4)
    assert np.array_equal(sliced, data[:1])


def test_read_zarr_voxel_size_resolution_attr(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "ome", 0.0)
    group["ome"].attrs["resolution"] = (1.0, 2.0, 3.0)
    _fill(group, "noattr", 0.0)
    path = tmp_path / "out.zarr"
    assert read_zarr_voxel_size(path, key="ome") == VoxelSize(
        voxels_size=(1.0, 2.0, 3.0)
    )
    with pytest.warns(UserWarning, match="Voxel size not found"):
        assert read_zarr_voxel_size(path, key="noattr") == VoxelSize()


def test_read_zarr_voxel_size_prefers_element_size(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "data", 0.0)
    group["data"].attrs["element_size_um"] = (1.0, 1.0, 1.0)
    group["data"].attrs["resolution"] = (9.0, 9.0, 9.0)
    assert read_zarr_voxel_size(tmp_path / "out.zarr", key="data") == VoxelSize(
        voxels_size=(1.0, 1.0, 1.0)
    )


def test_auto_key_single_dataset(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "data", 0.0, dtype="uint8")
    group["data"].attrs["element_size_um"] = (0.235, 0.15, 0.15)
    path = tmp_path / "out.zarr"
    assert load_zarr(path, key=None).shape == SHAPES
    assert read_zarr_shape(path, key=None) == SHAPES
    assert read_zarr_voxel_size(path, key=None) == VoxelSize(
        voxels_size=(0.235, 0.15, 0.15)
    )


def test_auto_key_prefers_panseg_key(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    for key, value in (("noise", 1.0), ("prediction", 2.0), ("extra", 3.0)):
        _fill(group, key, value)
    assert load_zarr(tmp_path / "out.zarr", key=None)[0, 0, 0] == 2.0


def test_auto_key_ambiguous_raises(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "alpha", 1.0)
    _fill(group, "beta", 2.0)
    with pytest.raises(RuntimeError, match="Ambiguous datasets"):
        read_zarr_shape(tmp_path / "out.zarr", key=None)


def test_auto_key_no_datasets_raises(zarr_file_empty):
    with pytest.raises(RuntimeError, match="No datasets found"):
        read_zarr_shape(zarr_file_empty, key=None)


def test_auto_key_repo_3d_resource(zarr_file_3d):
    assert read_zarr_shape(zarr_file_3d, key=None) == (10, 10, 10)
    assert read_zarr_voxel_size(zarr_file_3d, key=None) == VoxelSize(
        voxels_size=(0.235, 0.15, 0.15)
    )
    assert load_zarr(zarr_file_3d, key=None).shape == (10, 10, 10)


def test_list_zarr_keys_nested(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "top", 1.0, dtype="uint8")
    sub = group.create_group("sub")
    _fill(sub, "inner", 5.0, dtype="uint8")
    deeper = sub.create_group("deeper")
    _fill(deeper, "bottom", 9.0, dtype="uint8")
    path = tmp_path / "out.zarr"
    assert sorted(list_zarr_keys(path)) == ["sub/deeper/bottom", "sub/inner", "top"]
    assert load_zarr(path, key="sub/inner")[0, 0, 0] == 5
    assert load_zarr(path, key="sub/deeper/bottom")[0, 0, 0] == 9
    # three datasets, none a PanSeg key -> ambiguous
    with pytest.raises(RuntimeError, match="Ambiguous datasets"):
        read_zarr_shape(path, key=None)


def test_del_zarr_key(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "raw", 1.0)
    _fill(group, "prediction", 2.0)
    path = tmp_path / "out.zarr"
    del_zarr_key(path, key="raw")
    assert list_zarr_keys(path) == ["prediction"]
    del_zarr_key(path, key="missing")  # no-op
    assert list_zarr_keys(path) == ["prediction"]


def test_rename_zarr_key(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "raw", 7.0, dtype="uint8")
    group["raw"].attrs["element_size_um"] = (0.235, 0.15, 0.15)
    path = tmp_path / "out.zarr"
    rename_zarr_key(path, old_key="raw", new_key="prediction")
    assert list_zarr_keys(path) == ["prediction"]
    assert load_zarr(path, key="prediction")[0, 0, 0] == 7
    assert read_zarr_voxel_size(path, key="prediction") == VoxelSize(
        voxels_size=(0.235, 0.15, 0.15)
    )


def test_rename_zarr_key_missing_old_key_noop(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "out.zarr"), mode="w")
    _fill(group, "raw", 1.0)
    rename_zarr_key(tmp_path / "out.zarr", old_key="nope", new_key="new")
    assert list_zarr_keys(tmp_path / "out.zarr") == ["raw"]
