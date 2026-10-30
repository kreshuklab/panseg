"""
Reading and writing OME-Zarr (OME-NGFF) stores.

OME-Zarr is PanSeg's zarr carrier going forward (issue #622): zarr export
writes an OME-NGFF v0.5 store built with `clearscale`
(https://github.com/ilastik/clearscale), and the readers in ``panseg.io.zarr``
dispatch to this module whenever a store carries NGFF metadata. PanSeg-native
stores (pre-NGFF flat datasets with PanSeg attributes) stay readable as a
fallback.

Notes:
1. `clearscale` is metadata-only: it builds and parses the OME-NGFF group
   attributes; the pixel data is written and read with plain zarr.
2. PanSeg reads the base level (level 0) of a multiscale pyramid only
   (see GLOSSARY: Base level).
3. A store may hold several image groups (e.g. ``raw`` and ``prediction``
   side by side). The key of an image is the path of its image group inside
   the store (see GLOSSARY: Key); foreign stores whose root is the image
   group are read without a key (GLOSSARY: Root image). A segmentation is
   written as a label image below its parent image group (``<key>/labels/
   <name>``, GLOSSARY: Label image) and addressed by that path.
4. `clearscale` is an undeclared dependency while it awaits its first PyPI
   release (#622): it is imported lazily, and only when NGFF data is actually
   loaded or written. Store detection and image-group discovery work without
   it.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from panseg.io.voxelsize import VoxelSize

logger = logging.getLogger(__name__)

# NGFF axis names in canonical order (t, c, z, y, x).
_NGFF_AXIS_ORDER = "tczyx"
_NGFF_AXES = frozenset(_NGFF_AXIS_ORDER)

# Conversion factors to the units PanSeg normalizes to: micrometers for
# space, seconds for time. Spelled out per the NGFF axis-unit names plus
# common aliases.
_SPACE_UNITS_TO_UM = {
    "angstrom": 1e-4,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometers": 1e-3,
    "um": 1.0,
    "\u00b5m": 1.0,  # µm (micro sign)
    "\u03bcm": 1.0,  # µm (Greek small letter mu)
    "micrometer": 1.0,
    "micrometers": 1.0,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimeters": 1e3,
    "cm": 1e4,
    "centimeter": 1e4,
    "centimeters": 1e4,
    "m": 1e6,
    "meter": 1e6,
    "meters": 1e6,
    "km": 1e9,
    "kilometer": 1e9,
    "kilometers": 1e9,
}

_TIME_UNITS_TO_SECONDS = {
    "ns": 1e-9,
    "nanosecond": 1e-9,
    "nanoseconds": 1e-9,
    "us": 1e-6,
    "\u00b5s": 1e-6,  # µs (micro sign)
    "\u03bcs": 1e-6,  # µs (Greek small letter mu)
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}

# PanSeg time units (see panseg.core.image TIME_UNIT_CHOICES) to NGFF axis
# unit names, used on export.
_PANSEG_TIME_UNITS_TO_NGFF = {
    "us": "microsecond",
    "\u00b5s": "microsecond",
    "\u03bcs": "microsecond",
    "ms": "millisecond",
    "s": "second",
    "sec": "second",
    "second": "second",
    "seconds": "second",
    "min": "minute",
    "minute": "minute",
    "minutes": "minute",
    "h": "hour",
    "hr": "hour",
    "hour": "hour",
    "hours": "hour",
    "d": "day",
    "day": "day",
    "days": "day",
}


def _import_clearscale():
    """Import clearscale, with a helpful error when it is missing."""
    try:
        import clearscale
    except ImportError as err:
        raise ImportError(
            "OME-Zarr support requires the 'clearscale' package "
            "(https://github.com/ilastik/clearscale). It is not yet a declared "
            "dependency of PanSeg (pending its first PyPI release, #622): "
            "install it with "
            "'pip install git+https://github.com/ilastik/clearscale.git'."
        ) from err
    return clearscale


def _validate_ome_zarr_file(path: Path) -> None:
    """Check if a file is a Zarr store."""
    assert path.exists(), f"File not found: {path}"
    assert str(path).find(".zarr") != -1, f"File is not a Zarr file: {path}"


def _ngff_metadata(attrs: dict, key: str) -> bool:
    """Whether attrs carry NGFF metadata `key` (v0.4 top-level, v0.5+ under `ome`)."""
    if key in attrs:
        return True
    ome = attrs.get("ome")
    return isinstance(ome, dict) and key in ome


def _is_image_group(group: zarr.Group) -> bool:
    """Check if a group carries OME-NGFF multiscale metadata.

    Labels containers (v0.5 `ome` with `labels` but no `multiscales`) are
    not image groups.
    """
    return _ngff_metadata(dict(group.attrs), "multiscales")


def _is_label_group(group: zarr.Group) -> bool:
    """Check if an image group is a label image (carries `image-label`)."""
    return _ngff_metadata(dict(group.attrs), "image-label")


def _find_image_groups(group: zarr.Group, prefix: str = "") -> list[tuple[str, bool]]:
    """Return (path, is_label) for all image groups at or below `group`.

    The root image group is reported as the empty string; image groups in a
    collection as their path inside the store. Labels containers are not
    image groups, but the label images below them are.
    """
    found = []
    if _is_image_group(group):
        found.append((prefix, _is_label_group(group)))
    for name, node in group.members():
        if isinstance(node, zarr.Group):
            child_prefix = f"{prefix}/{name}" if prefix else name
            found.extend(_find_image_groups(node, child_prefix))
    return found


def is_ome_zarr(path: Path) -> bool:
    """Check if a Zarr store carries OME-NGFF metadata (v0.4 or later).

    True when the store root, or any group inside it, carries NGFF metadata
    (root image group or collection of image groups). False for PanSeg-native
    stores, which fall back to the pre-NGFF readers.
    """
    if not path.exists():
        return False
    try:
        group = zarr.open_group(store=str(path), mode="r")
    except Exception:  # not a (readable) zarr store at all
        return False
    return len(_find_image_groups(group)) > 0


def _display_group_path(group_path: str) -> str:
    return group_path if group_path else "(store root)"


def _resolve_image_group_path(path: Path, key: str | None) -> tuple[zarr.Group, str]:
    """Open the store and return (root group, path of the image group).

    An explicit key is the path of an image group inside the store (label
    images included). Without a key, the unique non-label image group of the
    store is resolved automatically; a labels-only store resolves to its
    label image.
    """
    root = zarr.open_group(store=str(path), mode="r")
    image_groups = _find_image_groups(root)
    group_paths = [group_path for group_path, _ in image_groups]

    if key:
        group_path = str(key).strip("/")
        if group_path not in group_paths:
            raise KeyError(
                f"Key '{group_path}' is not an image group of {path}. "
                f"Available image groups: {[_display_group_path(p) for p in group_paths]}."
            )
        return root, group_path

    candidates = [p for p, is_label in image_groups if not is_label] or [
        p for p, is_label in image_groups if is_label
    ]

    if len(candidates) == 1:
        return root, candidates[0]

    if not candidates:
        raise RuntimeError(f"No image groups found - verify '{root.tree()}'.")

    raise RuntimeError(
        f"Ambiguous image groups '{[_display_group_path(p) for p in candidates]}' in {path}. "
        f"PanSeg expects only one image group in the input OME-Zarr."
    )


def _get_ome_zarr_image(path: Path, key: str | None) -> tuple[zarr.Array, Any]:
    """Return the base-level array and its Multiscale metadata of an image."""
    _import_clearscale()
    from clearscale import OmeZarrGroup

    _validate_ome_zarr_file(path)
    root, group_path = _resolve_image_group_path(path, key)
    image_group = root[group_path] if group_path else root

    multiscale = OmeZarrGroup.from_group(image_group).multiscales[0]
    # The base level is the first dataset of the multiscale (GLOSSARY: Base level).
    base_level = next(iter(multiscale))
    base_array = image_group[base_level]
    if not isinstance(base_array, zarr.Array):
        raise ValueError(f"'{base_level}' is not a zarr.Array.")
    return base_array, multiscale


def load_ome_zarr(
    path: Path,
    key: str | None,
    slices: slice | None = None,
) -> np.ndarray:
    """Load the base level of an image from an OME-Zarr store.

    Args:
        path (Path): The path to the Zarr store.
        key (str | None): The path of the image group, None to resolve the
            single image group of the store automatically.
        slices (slice | None, optional): Slice to load. Defaults to None.

    Returns:
        np.ndarray: The base level of the image.
    """
    data, _ = _get_ome_zarr_image(path, key)
    return data[...] if slices is None else data[slices]


def read_ome_zarr_shape(path: Path, key: str | None = None) -> tuple[int, ...]:
    """Read the shape of the base level of an image in an OME-Zarr store.

    Args:
        path (Path): The path to the Zarr store.
        key (str | None, optional): The path of the image group. Defaults to None.

    Returns:
        tuple: The shape of the base level.
    """
    data, _ = _get_ome_zarr_image(path, key)
    return data.shape


def read_ome_zarr_axis_order(path: Path, key: str | None = None) -> str | None:
    """Read the axis order of an image from its NGFF metadata, e.g. "TZYX".

    Args:
        path (Path): The path to the Zarr store.
        key (str | None, optional): The path of the image group. Defaults to None.

    Returns:
        str | None: The axis order string, or None when the store carries no
        usable NGFF axes (unknown axis names fall back to the shape heuristic).
    """
    _, multiscale = _get_ome_zarr_image(path, key)
    axes = list(multiscale[next(iter(multiscale))].ome_zarr_axes)
    if not axes or not all(axis in _NGFF_AXES for axis in axes):
        warnings.warn(f"Unsupported NGFF axes {axes} in {path}.")
        return None
    return "".join(axes).upper()


def _to_um(value: float, unit: str, axis: str, path: Path) -> float:
    """Convert one space-axis value to micrometers."""
    factor = _SPACE_UNITS_TO_UM.get(unit.lower())
    if factor is None:
        warnings.warn(
            f"Unknown space unit '{unit}' on axis '{axis}' in {path}, assuming micrometers (um)."
        )
        factor = 1.0
    return value * factor


def read_ome_zarr_voxel_size(path: Path, key: str | None) -> VoxelSize:
    """Read the voxel size of an image in an OME-Zarr store.

    The space-axis scales and units are converted to micrometers. Returns the
    default voxel size (1, 1, 1) um when the store carries no units.

    Args:
        path (Path): The path to the Zarr store.
        key (str | None): The path of the image group.

    Returns:
        VoxelSize: The voxel size of the image.
    """
    _, multiscale = _get_ome_zarr_image(path, key)
    scale = multiscale[next(iter(multiscale))]

    axes = scale.ome_zarr_axes
    if not any(axes[axis].unit for axis in axes if axes[axis].type == "space"):
        warnings.warn(f"Voxel size not found in {path}.")
        return VoxelSize()

    sizes = []
    units = []
    for axis in "zyx":
        if axis in axes and axes[axis].type == "space":
            unit = str(axes[axis].unit or "")
            sizes.append(_to_um(float(scale.pixel_size[axis]), unit, axis, path))
            units.append(unit)
        else:
            # PanSeg voxel sizes are always (z, y, x); a missing axis is 1
            sizes.append(1.0)

    if units and not all(unit == units[0] for unit in units):
        warnings.warn(f"Units are not homogeneous: {units}")

    return VoxelSize(voxels_size=tuple(sizes), unit="um")


def read_ome_zarr_time_spacing(
    path: Path, key: str | None = None
) -> tuple[float | None, str]:
    """Read the time spacing of an image from its NGFF metadata.

    The time-axis scale is converted to seconds. Returns (None, "s") when the
    image has no time axis or carries no time unit (unknown spacing).

    Args:
        path (Path): The path to the Zarr store.
        key (str | None, optional): The path of the image group. Defaults to None.

    Returns:
        tuple: (time spacing in seconds or None, unit string)
    """
    _, multiscale = _get_ome_zarr_image(path, key)
    scale = multiscale[next(iter(multiscale))]

    if "t" not in scale.ome_zarr_axes:
        return None, "s"

    unit = str(scale.ome_zarr_axes["t"].unit or "")
    if not unit:
        return None, "s"

    factor = _TIME_UNITS_TO_SECONDS.get(unit.lower())
    if factor is None:
        warnings.warn(f"Unknown time unit '{unit}' in {path}, assuming seconds.")
        factor = 1.0
    return float(scale.pixel_size["t"]) * factor, "s"


def list_ome_zarr_keys(path: Path) -> list[str]:
    """List the keys (image-group paths) of an OME-Zarr store.

    A store whose root is itself an image group returns an empty list: the
    image is addressed without a key. Image groups in a collection return
    their paths inside the store, label images included (e.g.
    ``raw/labels/segmentation``); labels containers are not listed.

    Args:
        path (Path): The path to the Zarr store.

    Returns:
        list: The keys in the file.
    """
    _validate_ome_zarr_file(path)
    root = zarr.open_group(store=str(path), mode="r")
    return [group_path for group_path, _ in _find_image_groups(root) if group_path]


def _slot_occupied(group: zarr.Group) -> bool:
    """Whether a group already holds image data (metadata or a base level)."""
    return _is_image_group(group) or "s0" in group


def _shape_matches_or_1(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Shapes agree per axis, size-1 axes are broadcast-compatible."""
    return len(a) == len(b) and all(x == y or x == 1 or y == 1 for x, y in zip(a, b))


def _counterpart_shapes(
    image_group: zarr.Group, label_name: str | None
) -> list[tuple[str, tuple[int, ...]]]:
    """Shapes of the data that a write at (image_group, label_name) must agree with."""
    shapes = []
    for name, node in image_group.members():
        if label_name is not None:
            # the counterpart of a label image is the image itself
            if name == "s0" and isinstance(node, zarr.Array):
                shapes.append(("image", node.shape))
        elif name == "labels" and isinstance(node, zarr.Group):
            for label, label_group in node.members():
                if not isinstance(label_group, zarr.Group):
                    continue
                for level, array in label_group.members():
                    if level == "s0" and isinstance(array, zarr.Array):
                        shapes.append((f"label '{label}'", array.shape))
    return shapes


def ome_zarr_slot_occupied(path: Path, key: str, label_name: str | None = None) -> bool:
    """Whether a zarr export to (key, label_name) would replace existing data.

    The GUI checks this before exporting and warns the user. A slot is not
    occupied when its group only holds labels (or nothing): writing an image
    to a group that has labels is additive.

    Args:
        path (Path): The path to the Zarr store (absent stores are free).
        key (str): The path of the image group; for a label image, of its
            parent image group.
        label_name (str | None): Label name below ``<key>/labels/``; None
            for an image export.

    Returns:
        bool: True when the slot already holds an image or label.
    """
    if not path.exists():
        return False
    root = zarr.open_group(store=str(path), mode="r")
    group: zarr.Group = root
    parts = [part for part in str(key).strip("/").split("/") if part]
    if label_name is not None:
        parts += ["labels", label_name]
    for part in parts:
        if part not in group:
            return False
        node = group[part]
        if not isinstance(node, zarr.Group):
            return True
        group = node
    return _slot_occupied(group)


def _registered_label_names(group: zarr.Group) -> list[str]:
    """Label names already registered in a labels container (v0.4 or v0.5+)."""
    attrs = dict(group.attrs)
    meta = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    names = meta.get("labels") if isinstance(meta, dict) else None
    return [str(name) for name in names] if isinstance(names, list) else []


def create_ome_zarr(
    path: Path,
    stack: np.ndarray,
    key: str | None,
    voxel_size: VoxelSize,
    mode: str = "a",
    axis_order: str | None = None,
    t_spacing: float | None = None,
    t_spacing_unit: str = "s",
    version: str = "0.5",
    label_name: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Create an OME-Zarr (OME-NGFF) image from a NumPy array.

    The image is written as an image group carrying a single-scale (level 0)
    multiscale metadata block built with clearscale. A non-empty key writes
    the image group to that path inside the store, so several images can
    share one store. With ``label_name``, the stack is written as a label
    image below the parent image group's labels container
    (``<key>/labels/<label_name>``), registered and linked to its source
    image; this is how segmentations are exported.

    Args:
        path (Path): The path to the Zarr store.
        stack (np.ndarray): The NumPy array to save as an image.
        key (str | None): The path of the image group (e.g. "raw"); for a
            label image, the path of its parent image group.
        voxel_size (VoxelSize): The voxel size of the image.
        mode (str): The mode to open the Zarr store ['w', 'a'].
        axis_order (str | None): Layout string of the stack (e.g. "TZYX"),
            defining the NGFF axes. Required to be a subset of tczyx in
            canonical order.
        t_spacing (float | None): Time spacing between timepoints, in
            ``t_spacing_unit``. None (unknown) writes no time unit.
        t_spacing_unit (str): Unit of the time spacing.
        version (str): OME-NGFF version to write ('0.4', '0.5' or '0.6').
            Defaults to '0.5'.
        label_name (str | None): Name of the label image below
            ``<key>/labels/`` (e.g. "segmentation"); None writes an image
            group at ``key``.
        overwrite (bool): Replace an image or label already present at the
            target slot. Without it, writing into an occupied slot is a
            ValueError.
    """
    if not key or not str(key).strip("/"):
        raise ValueError("Key is required for zarr format")
    if label_name is not None and not np.issubdtype(stack.dtype, np.integer):
        raise ValueError(
            f"Label image '{label_name}' must have an integer dtype, got {stack.dtype}."
        )
    _import_clearscale()
    from clearscale import ChildRef, FileRef, GroupKind, Multiscale, OmeZarrGroup, Scale
    from clearscale.ome_zarr import ImageLabel, MultiscaleProperties

    if axis_order is None:
        raise ValueError("Axis order is required to write OME-Zarr metadata.")
    axes = axis_order.lower()
    if (
        len(axes) != len(stack.shape)
        or not set(axes) <= _NGFF_AXES
        or len(set(axes)) != len(axes)
        or axes != "".join(axis for axis in _NGFF_AXIS_ORDER if axis in set(axes))
    ):
        raise ValueError(
            f"Axis order '{axis_order}' with {len(stack.shape)} axes is not a "
            f"valid OME-Zarr layout (a subset of tczyx in canonical order, "
            f"without repeats)."
        )

    root = zarr.open_group(
        store=str(path), mode=mode, zarr_format=2 if version == "0.4" else 3
    )

    image_group = root
    for part in str(key).strip("/").split("/"):
        image_group = image_group.require_group(part)

    group = image_group
    if label_name is not None:
        group = image_group.require_group("labels").require_group(label_name)

    if _slot_occupied(group):
        if not overwrite:
            target = f"{key}/labels/{label_name}" if label_name else str(key)
            raise ValueError(
                f"Key '{target}' already holds a "
                f"{'label' if label_name else 'image'} in {path}."
            )
        if "s0" in group:
            del group["s0"]
        for meta_key in ("multiscales", "ome", "image-label"):
            group.attrs.pop(meta_key, None)

    # the image and its labels must have matching shapes; warn once both
    # sides of the pair exist
    for what, shape in _counterpart_shapes(image_group, label_name):
        if not _shape_matches_or_1(shape, stack.shape):
            warnings.warn(
                f"Label and image shapes do not match in {path}: "
                f"the {what} of '{key}' has {shape}, the written data has {stack.shape}."
            )

    group.create_array(name="s0", data=stack)

    pixel_size = []
    units = []
    for axis in axes:
        if axis == "t":
            # An unknown spacing is written as a unitless scale of 1, which
            # reads back as (None, "s").
            known = t_spacing is not None
            pixel_size.append(float(t_spacing) if known else 1.0)
            units.append(
                _PANSEG_TIME_UNITS_TO_NGFF.get(t_spacing_unit.lower(), t_spacing_unit)
                if known
                else ""
            )
        elif axis == "c":
            pixel_size.append(1.0)
            units.append("")
        elif voxel_size.voxels_size is not None:
            pixel_size.append(
                {"z": voxel_size.z, "y": voxel_size.y, "x": voxel_size.x}[axis]
            )
            units.append("micrometer")
        else:
            pixel_size.append(1.0)
            units.append("")

    scale = Scale.from_lists(
        keys=axes, shape=stack.shape, pixel_size=pixel_size, unit=units
    )
    if label_name is not None:
        # A label image: NGFF multiscales plus the image-label link to its
        # source image (the parent image group, two levels up).
        multiscale = Multiscale(
            {"s0": scale},
            ome=MultiscaleProperties(image_label=ImageLabel(source=FileRef("../.."))),
        )
        labels_container = image_group.require_group("labels")
        names = _registered_label_names(labels_container)
        if label_name not in names:
            names.append(label_name)
        labels_container.attrs.pop("labels", None)
        labels_container.attrs.pop("ome", None)
        labels_container.attrs.update(
            OmeZarrGroup(
                kind=GroupKind.LABELS,
                children=tuple(ChildRef("label", FileRef(name)) for name in names),
            ).to_attrs(version=version)
        )
    else:
        multiscale = Multiscale.from_single(scale, scale_key="s0")
    group.attrs.update(OmeZarrGroup.from_single(multiscale).to_attrs(version=version))
