from pathlib import Path

import numpy as np
import trimesh

from panseg import logger
from panseg.io.voxelsize import VoxelSize

try:
    from zmesh import Mesher
except ImportError:
    logger.error("ERROR: zmesh not installed, please run `pip install zmesh`")


def create_mesh(
    path: Path,
    stack: np.ndarray,
    voxel_size: VoxelSize,
    reduction_factor=2.0,
    close_mesh=False,
):
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

    scene.export(path)
    return scene
