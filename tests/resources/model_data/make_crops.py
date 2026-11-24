"""Cut the test crops in this folder from the public datasets PanSeg's models
were trained on. See README.md for where each crop comes from.

Run from anywhere with PanSeg's dev environment: `python make_crops.py`.
It reads about 75 MB over the network: one 64.5 MB file from OSF, and two
TIFFs out of a 5.25 GB zip, fetched by HTTP range requests.
"""

import ast
import io
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import requests
import tifffile
from skimage.transform import rescale

HERE = Path(__file__).parent

# Willis et al. 2016, PNAS, doi:10.17863/CAM.7793 (CC-BY 4.0)
PNAS_ZIP = (
    "https://api.repository.cam.ac.uk/server/api/core/bitstreams/"
    "95babfae-ae16-4dae-82f5-8656ceace4d6/content"
)
# A stack emotional-cricket validated on but never trained on
PNAS_STACK = "8hrs_plant2"
PNAS_SEGMENTATION = "8hrs_plant2_trim-acylYFP_hmin_2_asf_1_s_2.20_clean_3.tif"

# Wolny et al. 2020, eLife, OSF project uzq3w: lateral root primordia test set
ROOT_FILE = "https://osf.io/download/63kje/"  # Movie1_t00006_crop_gt.h5
ROOT_VOXEL_SIZE = (0.25, 0.1625, 0.1625)


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, fetched by HTTP range requests."""

    def __init__(self, url: str):
        self.session = requests.Session()
        # some servers leave the length out of HEAD, so ask for one byte
        response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError(f"{url} doesn't serve byte ranges")
        self.url = response.url
        self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        start = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}
        self.pos = start[whence] + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        response = self.session.get(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300
        )
        response.raise_for_status()
        data = response.content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def cell_walls(label: np.ndarray) -> np.ndarray:
    """Voxels of a cell (non-zero label) next to a voxel of another cell."""
    walls = np.zeros(label.shape, dtype=bool)
    for axis in range(label.ndim):
        a = label.take(range(label.shape[axis] - 1), axis=axis)
        b = label.take(range(1, label.shape[axis]), axis=axis)
        between = (a != b) & (a > 0) & (b > 0)
        low = [slice(None)] * label.ndim
        high = [slice(None)] * label.ndim
        low[axis], high[axis] = slice(0, -1), slice(1, None)
        walls[tuple(low)] |= between
        walls[tuple(high)] |= between
    return walls


def densest_window(mask: np.ndarray, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """The window of `shape` with the most voxels set in `mask`."""
    table = np.pad(mask.cumsum(0).cumsum(1).cumsum(2), ((1, 0),) * 3)
    best, start = -1, (0, 0, 0)
    for z in range(0, mask.shape[0] - shape[0] + 1, 4):
        for y in range(0, mask.shape[1] - shape[1] + 1, 8):
            for x in range(0, mask.shape[2] - shape[2] + 1, 8):
                z1, y1, x1 = z + shape[0], y + shape[1], x + shape[2]
                count = (
                    table[z1, y1, x1]
                    - table[z, y1, x1]
                    - table[z1, y, x1]
                    - table[z1, y1, x]
                    + table[z, y, x1]
                    + table[z, y1, x]
                    + table[z1, y, x]
                    - table[z, y, x]
                )
                if count > best:
                    best, start = count, (z, y, x)
    return tuple(slice(s, s + n) for s, n in zip(start, shape))


def write_crop(
    path: Path,
    raw: np.ndarray,
    label: np.ndarray,
    voxel_size: tuple[float, ...],
    **source: str,
) -> None:
    with h5py.File(path, "w") as f:
        for key, data in (("raw", raw), ("label", label)):
            dataset = f.create_dataset(key, data=data, compression="gzip")
            dataset.attrs["element_size_um"] = voxel_size
        f.attrs.update(source)
    print(
        f"wrote {path.name}: raw {raw.shape} {raw.dtype}, {path.stat().st_size / 1e6:.1f} MB"
    )


def shoot_meristem_cells() -> None:
    """Shoot apical meristem cell walls at 0.25 µm, emotional-cricket's resolution."""
    plant = PNAS_STACK.split("_")[1]
    hours = PNAS_STACK.split("_")[0]
    pnas = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(PNAS_ZIP), 1 << 22))
    raw = tifffile.imread(
        io.BytesIO(
            pnas.read(f"PNAS/{plant}/processed_tiffs/{PNAS_STACK}_trim-acylYFP.tif")
        )
    )
    label = tifffile.imread(
        io.BytesIO(pnas.read(f"PNAS/{plant}/segmentation_tiffs/{PNAS_SEGMENTATION}"))
    )
    # the stated z resolution still has to be divided by the stack's
    # z-stretching factor (the dataset's README, the paper's Table S10)
    resolutions = pnas.read(f"PNAS/{plant}/{plant}_tiff_resolutions.txt").decode()
    entry = next(
        ast.literal_eval(line.strip().strip("[],").join("[]"))
        for line in resolutions.splitlines()
        if line.strip().lstrip("[").startswith(f'"{hours}"')
    )
    _, (x_res, y_res, z_res), stretch = entry
    scale = (z_res / stretch / 0.25, y_res / 0.25, x_res / 0.25)

    raw = rescale(raw, scale, order=1, preserve_range=True, anti_aliasing=False)
    label = rescale(label, scale, order=0, preserve_range=True, anti_aliasing=False)
    label = np.where(label == 1, 0, label).astype(np.uint16)  # background is 1
    # emotional-cricket's input size, where the cell walls are densest
    crop = densest_window(cell_walls(label), (100, 128, 128))
    write_crop(
        HERE / "shoot_meristem_cells.h5",
        np.round(raw[crop]).astype(np.uint8),
        label[crop],
        (0.25, 0.25, 0.25),
        source=f"PNAS.zip, {PNAS_STACK} acylYFP and segmentation, rescaled",
        doi="10.17863/CAM.7793",
        license="CC-BY-4.0",
        crop=str([(s.start, s.stop) for s in crop]),
    )


def lateral_root_cells() -> None:
    """Lateral root primordium cell walls, light-sheet, at the training resolution."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Movie1_t00006_crop_gt.h5"
        with requests.get(ROOT_FILE, stream=True, timeout=600) as response:
            response.raise_for_status()
            with path.open("wb") as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
        with h5py.File(path, "r") as f:
            # 1 is the background and 0 the walls between cells; neither is a cell
            label = f["label"][...]
            label = np.where(label <= 1, 0, label)
            # the cells are large, so a wider crop catches more of their walls
            crop = densest_window(cell_walls(label), (32, 192, 192))
            raw = f["raw"][crop]
    write_crop(
        HERE / "lateral_root_cells.h5",
        raw,
        label[crop].astype(np.uint16),
        ROOT_VOXEL_SIZE,
        source="OSF uzq3w, lateral root test set, Movie1_t00006_crop_gt.h5",
        doi="10.17605/OSF.IO/UZQ3W",
        crop=str([(s.start, s.stop) for s in crop]),
    )


if __name__ == "__main__":
    shoot_meristem_cells()
    lateral_root_cells()
