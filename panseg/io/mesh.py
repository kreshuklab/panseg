from pathlib import Path

import numpy as np
import trimesh

from panseg import logger
from panseg.io.voxelsize import VoxelSize


def _export_empty_scene(scene: "trimesh.Scene", path: Path) -> None:
    """
    Write a scene without geometry to disk.

    trimesh refuses to export empty scenes ("Can't export empty scenes!").
    An empty timepoint still needs its file to preserve the 1:1
    timepoint-file mapping, so the empty container is written with the
    format's low-level exporter. Formats without an empty-scene writer keep
    trimesh's behavior and raise.
    """
    file_type = path.suffix.lstrip(".").lower()
    if file_type == "glb":
        path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    elif file_type == "gltf":
        data = trimesh.exchange.gltf.export_gltf(scene)
        path.write_bytes(data["model.gltf"])
    elif file_type == "obj":
        data = trimesh.exchange.export.export_obj(scene)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("ascii"))
    elif file_type == "ply":
        path.write_bytes(trimesh.exchange.ply.export_ply(scene.to_mesh()))
    else:
        scene.export(path)


def create_mesh(
    path: Path,
    stack: np.ndarray,
    voxel_size: VoxelSize,
    reduction_factor=2.0,
    close_mesh=False,
):
    try:
        from zmesh import Mesher

    except ImportError:
        logger.error("ERROR: zmesh not installed, please run `pip install zmesh`")
        raise ImportError("ERROR: zmesh not installed, please run `pip install zmesh`")
    assert len(stack.shape) == 3, (
        f"Unsupported data shape of {stack.shape} for meshing."
    )
    mesher = Mesher(voxel_size.voxels_size)
    mesher.mesh(stack, close=close_mesh)

    scene = trimesh.scene.scene.Scene()
    for obj_id in mesher.ids():
        zmesh = mesher.get(
            obj_id,
            normals=False,
            reduction_factor=reduction_factor,
            voxel_centered=False,
            max_error=None,
        )
        vertex_colors = np.empty((zmesh.vertices.shape[0], 4), dtype=int)
        r = np.random.randint(256)
        g = np.random.randint(256)
        b = np.random.randint(256)
        vertex_colors[:] = (r, g, b, 255)
        mesh = trimesh.Trimesh(
            zmesh.vertices, zmesh.faces, vertex_colors=vertex_colors, process=False
        )
        scene.add_geometry(mesh)

    if scene.geometry:
        scene.export(path)
    else:
        _export_empty_scene(scene, path)
    return scene
