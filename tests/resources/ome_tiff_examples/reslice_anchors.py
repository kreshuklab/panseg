"""One-off script that regenerates the committed OME-TIFF timeseries test anchors.

Input:  ``ome_tiff_examples.tar.xz`` — The Open Microscopy Environment sample
        set (see ``LICENSE.md``), containing at least ``time-series.ome.tif``,
        ``4D-series.ome.tif`` and ``multi-channel-4D-series.ome.tif``.
Output: the three same-named files in this directory, resliced offline to
        T=4, C=3, Z=2, Y=X=32 (cropped from the start along T/C/Z,
        center-cropped along Y/X). The committed files are static artifacts:
        tests load them and never reslice at runtime.

Usage:
    python reslice_anchors.py /path/to/ome_tiff_examples.tar.xz
"""

import sys
import tarfile
import tempfile
from pathlib import Path

import tifffile

OUT_DIR = Path(__file__).resolve().parent

ANCHORS = [
    ("time-series.ome.tif", "TYX", {"T": 4, "Y": 32, "X": 32}),
    ("4D-series.ome.tif", "TZYX", {"T": 4, "Z": 2, "Y": 32, "X": 32}),
    (
        "multi-channel-4D-series.ome.tif",
        "TCZYX",
        {"T": 4, "C": 3, "Z": 2, "Y": 32, "X": 32},
    ),
]


def _centered_slice(n: int, m: int) -> slice:
    start = (n - m) // 2
    return slice(start, start + m)


def reslice(src: Path, axes: str, targets: dict[str, int]) -> None:
    with tifffile.TiffFile(src) as tiff:
        series = tiff.series[0]
        if series.axes != axes:
            raise ValueError(f"{src.name}: expected axes {axes}, got {series.axes}")
        data = tiff.asarray(series=0)
    shape = dict(zip(axes, data.shape))
    if any(shape[ax] < targets[ax] for ax in targets):
        raise ValueError(f"{src.name}: shape {data.shape} too small to reslice")
    slices = []
    for ax, n in zip(axes, data.shape):
        if ax in "YX":
            slices.append(_centered_slice(n, targets[ax]))
        else:
            slices.append(slice(0, targets[ax]))
    sliced = data[tuple(slices)]
    out = OUT_DIR / src.name
    tifffile.imwrite(
        out, sliced, ome=True, photometric="minisblack", metadata={"axes": axes}
    )
    with tifffile.TiffFile(out) as tiff:
        got = tiff.series[0]
        if got.axes != axes or got.shape != sliced.shape:
            raise ValueError(
                f"{out.name}: wrote {got.axes} {got.shape}, expected {axes} {sliced.shape}"
            )
    print(f"wrote {out} ({axes} {sliced.shape} {sliced.dtype})")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            f"usage: python {Path(sys.argv[0]).name} <ome_tiff_examples.tar.xz>"
        )
    tarball = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tarball) as tf:
            tf.extractall(tmp, filter="data")
        for name, axes, targets in ANCHORS:
            reslice(Path(tmp) / name, axes, targets)


if __name__ == "__main__":
    main()
