import logging
from pathlib import Path
from typing import assert_never

import numpy as np
import torch
from bioimageio.core import create_prediction_pipeline
from bioimageio.core.axis import AxisId
from bioimageio.core.prediction import predict
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor
from bioimageio.spec import load_model_description
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

from panseg.core.zoo import model_zoo, resolve_bioimageio_source
from panseg.functionals.dataprocessing.dataprocessing import (
    ImageLayout,
    fix_layout_to_CZYX,
    fix_layout_to_ZYX,
)
from panseg.functionals.model_geometry import ModelGeometry, model_geometry
from panseg.functionals.precision import AmpSetting, validate_amp
from panseg.functionals.prediction.utils.array_dataset import ArrayDataset, pad_to_shape
from panseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from panseg.functionals.prediction.utils.size_finder import (
    find_feasible_patch_and_halo_shapes,
)
from panseg.functionals.prediction.utils.slice_builder import SliceBuilder
from panseg.functionals.training.augs import get_test_augmentations

logger = logging.getLogger(__name__)

# Below WARN_MIN_PATCH_XY computation is inefficient.
WARN_MIN_PATCH_XY = 64


def _warn_inefficient_auto_patch(geometry: ModelGeometry, patch, patch_halo) -> None:
    """Flag an auto-derived patch that is halo-dominated."""
    patch = (int(patch[0]), int(patch[1]), int(patch[2]))
    patch_halo = (int(patch_halo[0]), int(patch_halo[1]), int(patch_halo[2]))
    if patch[1] >= WARN_MIN_PATCH_XY and patch[2] >= WARN_MIN_PATCH_XY:
        return
    if any(patch_halo):
        inefficiency = "Most of the model input is halo, prediction will be slow. "
    else:
        inefficiency = "Small Y/X area, so prediction will be slow. "
    logger.warning(
        f"Patch shape {patch} with halo {patch_halo} gives a model "
        f"input shape {geometry.model_input_shape(patch, patch_halo)}.\n"
        f"The Y/X patch is below {WARN_MIN_PATCH_XY} px, but "
        f"input is above the model's architectural minimum "
        f"{geometry.min_input}.\n{inefficiency}\n"
        "Reduce the model size or halo, or increase patch size."
    )


def biio_prediction(
    raw: np.ndarray,
    input_layout: ImageLayout,
    model_id: str,
    device: str,
) -> dict[str, np.ndarray]:
    """Predict with a BioImage.IO model.

    `model_id` is a bioimage.io model ID such as 'efficient-chipmunk', or the URL
    or path of an RDF or model package.
    """
    assert isinstance(input_layout, str)
    source = resolve_bioimageio_source(model_id)
    logger.info(f"Loading BioImage.IO model {model_id} from {source}")
    model = load_model_description(source, perform_io_checks=False)
    if isinstance(model, v0_4.ModelDescr):
        input_ids = [input_tensor.name for input_tensor in model.inputs]
    elif isinstance(model, v0_5.ModelDescr):
        input_ids = [input_tensor.id for input_tensor in model.inputs]
    else:
        assert_never(model)

    logger.info(f"Model expects these inputs: {input_ids}.")
    if len(input_ids) < 1:
        logger.error("Model needs no input tensor. PanSeg does not support this yet.")
    if len(input_ids) > 1:
        logger.error(
            "Model needs more than one input tensor. PanSeg does not support this yet."
        )

    tensor_id = input_ids[0]
    axes = model.inputs[0].axes  # PanSeg only supports one input tensor for now
    dims = tuple(
        AxisId("channel") if item.lower() == "c" else AxisId(item.lower())
        for item in input_layout
    )  # `AxisId` has to be "channel" not "c"

    pipeline = create_prediction_pipeline(model, devices=[device])
    try:
        if isinstance(
            axes[0], str
        ):  # then it's a <=0.4.10 model, `predict_sample_block` is not implemented
            logger.warning(
                "Model is older than 0.5.0. PanSeg will try to run BioImage.IO core inference, but it is not supported by BioImage.IO core."
            )
            axis_mapping = {"b": "batch", "c": "channel"}
            axes = [AxisId(axis_mapping.get(a, a)) for a in list(axes)]
            members = {
                TensorId(tensor_id): Tensor(array=raw, dims=dims).transpose(
                    [AxisId(a) for a in axes]
                )
            }
            sample = Sample(members=members, stat={}, id="raw")
            sample_out = predict(model=pipeline, inputs=sample)

            # If inference is supported by BioImage.IO core, this is how it should be done in PanSeg:
            #
            # shape = model.inputs[0].shape
            # input_block_shape = {TensorId(tensor_id): {AxisId(a): s for a, s in zip(axes, shape)}}
            # sample_out = predict(model=model, inputs=sample, input_block_shape=input_block_shape)
        else:
            members = {
                TensorId(tensor_id): Tensor(array=raw, dims=dims).transpose(
                    [AxisId(a) if isinstance(a, str) else a.id for a in axes]
                )
            }
            sample = Sample(members=members, stat={}, id="raw")
            sizes_in_rdf = {a.id: a.size for a in axes}
            assert "x" in sizes_in_rdf, "Model does not have 'x' axis in input tensor."
            size_to_check = sizes_in_rdf[AxisId("x")]
            if isinstance(size_to_check, int):  # e.g. 'emotional-cricket'
                # 'emotional-cricket' has {'batch': None, 'channel': 1, 'z': 100, 'y': 128, 'x': 128}
                input_block_shape = {
                    TensorId(tensor_id): {
                        a.id: a.size if isinstance(a.size, int) else 1
                        for a in axes
                        if not isinstance(a, str)  # for a.size/a.id type checking only
                    }
                }
                sample_out = predict(
                    model=pipeline, inputs=sample, input_block_shape=input_block_shape
                )
            elif isinstance(
                size_to_check, v0_5.ParameterizedSize
            ):  # e.g. 'philosophical-panda'
                # 'philosophical-panda' has:
                #   {'z': ParameterizedSize(min=1, step=1),
                #    'channel': 2,
                #    'y': ParameterizedSize(min=16, step=16),
                #    'x': ParameterizedSize(min=16, step=16)}
                blocksize_parameter = {
                    (TensorId(tensor_id), a.id): (
                        (96 - a.size.min) // a.size.step
                        if isinstance(a.size, v0_5.ParameterizedSize)
                        else 1
                    )
                    for a in axes
                    if not isinstance(a, str)  # for a.size/a.id type checking only
                }
                sample_out = predict(
                    model=pipeline,
                    inputs=sample,
                    blocksize_parameter=blocksize_parameter,
                )
            else:
                assert_never(size_to_check)
    except Exception as e:
        pipeline.unload()
        raise e
    pipeline.unload()

    assert isinstance(sample_out, Sample)
    if len(sample_out.members) != 1:
        logger.warning(
            "Model has more than one output tensor. PanSeg does not support this yet."
        )

    desired_axes_short = [AxisId(a) for a in ["b", "c", "z", "y", "x"]]
    desired_axes = [AxisId(a) for a in ["batch", "channel", "z", "y", "x"]]
    t = {
        i: o.transpose(desired_axes_short)
        if "b" in o.dims or "c" in o.dims
        else o.transpose(desired_axes)
        for i, o in sample_out.members.items()
    }

    named_pmaps = {}
    for key, tensor_bczyx in t.items():
        bczyx = tensor_bczyx.data.to_numpy()
        assert bczyx.ndim == 5, (
            f"Expected 5D BCZYX-transposed prediction from `bioimageio.core`, got {bczyx.ndim}D"
        )
        if bczyx.shape[0] == 1:
            named_pmaps[f"{key}"] = bczyx[0]
        else:
            for b, czyx in enumerate(bczyx):
                named_pmaps[f"{key}_{b}"] = czyx
    return named_pmaps  # list of CZYX arrays


def unet_prediction(
    raw: np.ndarray,
    input_layout: ImageLayout,
    model_name: str | None,
    model_id: str | None,
    patch: tuple[int, int, int] | None = None,
    patch_halo: tuple[int, int, int] | None = None,
    single_batch_mode: bool = True,
    device: str = "cuda",
    model_update: bool = False,
    disable_tqdm: bool = False,
    config_path: Path | None = None,
    model_weights_path: Path | None = None,
    tracker=None,
    amp: AmpSetting = "auto",
    channels_last_3d: bool = False,
) -> np.ndarray:
    """Generate prediction from raw data using a specified 3D U-Net model.

    This function handles both single and multi-channel outputs from the model,
    returning appropriately shaped arrays based on the output channel configuration.
    The raw volume is reflect-padded up to the aligned domain and the prediction
    is cropped back, so the returned shape always equals the input shape.

    For Bioimage.IO Model Zoo models, weights are downloaded and loaded into `UNet3D` or `UNet2D`
    in `panseg.functionals.training.model`, i.e. `bioimageio.core` is not used. `biio_prediction()` uses
    `bioimageio.core` for loading and running models.

    Args:
        raw (np.ndarray): Raw input data.
        Input_layout (ImageLayout): The layout of the input data.
        model_name (str | None): The name of the model to use.
        model_id (str | None): The ID of the model from the BioImage.IO model zoo.
        patch (tuple[int, int, int] | None, optional): Patch size for prediction.
            `None` derives a patch aligned to `geometry.increment` and sized so
            `patch + 2 * halo` fits the probed model-input budget. Defaults to None.
        patch_halo (tuple[int, int, int] | None, optional): Halo size around patches.
            `None` uses `geometry.halo`, the minimal halo that makes an aligned patch
            exact; a manual halo is used verbatim (no rounding, no alignment check).
            Defaults to None.
        single_batch_mode (bool, optional): Whether to use a single batch for prediction. Defaults to True.
        device (str, optional): The computation device ('cpu', 'cuda', etc.). Defaults to 'cuda'.
        model_update (bool, optional): Whether to update the model to the latest version. Defaults to False.
        disable_tqdm (bool, optional): If True, disables the tqdm progress bar. Defaults to False.
        config_path (Path | None, optional): Path to the model configuration file. Defaults to None.
        model_weights_path (Path | None, optional): Path to the model weights file. Defaults to None.
        amp (AmpSetting, optional): precision mode (``"auto"``: bf16 autocast on
            supported CUDA devices, else fp32; ``"off"``: always fp32; the
            YAML-boolean spellings of these names also resolve, see
            `validate_amp`). The memory probes size patches in the same mode.
            Outputs are fp32 in either mode. Defaults to "auto".
        channels_last_3d (bool, optional): opt in to running 3D models on CUDA
            in the channels_last_3d memory format (kept off on CPU and for 2D
            models). The memory probes size patches in the same layout.
            Outputs are unaffected. Defaults to False.

    Returns:
        np.ndarray: The predicted boundaries as a 3D (Z, Y, X) or 4D (C, Z, Y, X) array, normalized between 0 and 1.

    Raises:
        ValueError: If neither `model_name`, `model_id`, nor `config_path` are provided.
        ValueError: If the patch or halo shape cannot run on the model.
    """
    amp = validate_amp(amp)
    if config_path is not None:  # Safari mode for custom models outside zoos
        logger.info("Safari prediction: Running model from custom config path.")
        model, model_config, model_path = model_zoo.get_model_by_config_path(
            config_path, model_weights_path
        )
    elif model_id is not None:  # BioImage.IO zoo mode
        logger.info("BioImage.IO prediction: Running model from BioImage.IO model zoo.")
        model, model_config, model_path = model_zoo.get_model_by_id(model_id)
    elif model_name is not None:  # PanSeg zoo mode
        logger.info("Zoo prediction: Running model from PanSeg official zoo.")
        model, model_config, model_path = model_zoo.get_model_by_name(
            model_name, model_update=model_update
        )
    else:
        raise ValueError(
            "Either `model_name` or `model_id` or `model_path` must be provided."
        )
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    if "model_state_dict" in state:  # Model weights format may vary between versions
        state = state["model_state_dict"]
    model.load_state_dict(state)

    # One architecture traversal per prediction: the default halo, the
    # pre-probe checks and the final validation all share this geometry.
    geometry = model_geometry(model)
    user_halo = patch_halo is not None
    if patch_halo is None:
        patch_halo = geometry.halo
        logger.info(f"Computed theoretical minimum halo from model: {patch_halo}")

    if input_layout == "YX":
        raw_shape = (1,) + raw.shape
    elif input_layout == "ZYX":
        raw_shape = raw.shape
    elif input_layout == "CYX":
        raw_shape = (1,) + raw.shape[1:]
    elif input_layout == "CZYX":
        raw_shape = raw.shape[1:]

    assert len(raw_shape) == 3
    # A manual patch is used verbatim, even on a small image; only the derived
    # patch (from an auto-sized domain) must fit the architectural minimum.
    if patch is None:
        geometry.check_input(raw_shape)

    # Predict on an aligned domain: SliceBuilder flushes the last patch start
    # to `i - k`, which stays on the phase grid only when the domain is a
    # multiple of the increment. Reflect-pad the domain up to it; the result is
    # cropped back to `raw_shape` below.
    domain_shape = geometry.aligned_shape(raw_shape)
    auto_patch = patch is None
    if auto_patch:
        if user_halo:
            logger.info(
                f"Using the manually specified halo {patch_halo} while deriving a "
                "phase-aligned patch."
            )
        patch, patch_halo = find_feasible_patch_and_halo_shapes(
            model,
            model_config["in_channels"],
            domain_shape,
            patch_halo,
            device,
            both_sides=False,
            amp=amp,
            channels_last_3d=channels_last_3d,
            geometry=geometry,
        )

    geometry.check_patch(patch, patch_halo)
    if auto_patch:
        _warn_inefficient_auto_patch(geometry, patch, patch_halo)

    oversized_dims = [
        f"{dim}: patch {p} > domain {s}"
        for dim, p, s in zip(("Z", "Y", "X"), patch, domain_shape)
        if p > s
    ]
    if oversized_dims:
        raise ValueError(
            f"Patch shape {tuple(patch)} is bigger than the aligned domain shape "
            f"{tuple(domain_shape)} in: {', '.join(oversized_dims)}. "
            "Reduce the patch size, or leave it unset to derive a feasible one."
        )

    logger.info(
        f"For raw in shape {raw.shape}: set patch shape {patch}, set halo shape {patch_halo}"
    )

    predictor = ArrayPredictor(
        model=model,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        device=device,
        patch=patch,
        patch_halo=patch_halo,
        single_batch_mode=single_batch_mode,
        headless=False,
        verbose_logging=False,
        disable_tqdm=disable_tqdm,
        tracker=tracker,
        amp=amp,
        channels_last_3d=channels_last_3d,
    )

    if int(model_config["in_channels"]) > 1:  # if multi-channel input
        raw = fix_layout_to_CZYX(raw, input_layout)
        multichannel_input = True
    else:
        raw = fix_layout_to_ZYX(raw, input_layout)
        multichannel_input = False
    raw = raw.astype("float32")
    augs = get_test_augmentations(
        raw
    )  # normalization stats of the original raw, before the domain pad
    raw = pad_to_shape(raw, domain_shape)
    stride = geometry.stride_shape(patch)
    slice_builder = SliceBuilder(
        raw, label_dataset=None, patch_shape=patch, stride_shape=stride
    )
    test_dataset = ArrayDataset(
        raw,
        slice_builder,
        augs,
        halo_shape=patch_halo,
        multichannel=multichannel_input,
        verbose_logging=False,
    )

    prediction = predictor(test_dataset)
    # `raw` was reflect-padded up to the phase-aligned domain, so the map covers
    # the padded extent; the public API returns the raw extent. The last three
    # axes are spatial even for 2D models (singleton z), so C stays untouched.
    z, y, x = raw_shape
    return prediction[..., :z, :y, :x]
