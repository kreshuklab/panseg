import gc
import logging
from contextlib import chdir
from pathlib import Path
from typing import Literal, Optional, TypeVar

import numpy as np
import torch
import yaml
from bioimageio.core import test_model
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

from panseg import (
    FILE_CONFIG_TRAIN_YAML,
    PATH_PANSEG_MODELS,
    PATH_TRAIN_TEMPLATE,
)
from panseg.core.zoo import model_zoo
from panseg.functionals.precision import AmpSetting, validate_amp
from panseg.functionals.prediction.utils.size_finder import find_batch_size
from panseg.functionals.training.augs import Augmenter
from panseg.functionals.training.biio import make_model_description
from panseg.functionals.training.h5dataset import HDF5Dataset
from panseg.functionals.training.losses import DiceLoss
from panseg.functionals.training.model import UNet2D, UNet3D
from panseg.functionals.training.trainer import UNetTrainer, load_pretrained_weights

logger = logging.getLogger(__name__)


_T = TypeVar("_T", np.ndarray, torch.Tensor)

ZMUV_EPS = 1e-6
"""Epsilon of the `ZeroMeanUnitVarianceDescr` preprocessing declared for the input."""


def zero_mean_unit_variance(x: _T) -> _T:
    """Apply the input preprocessing declared in the model description.

    Mirrors `ZeroMeanUnitVarianceDescr()` with its default kwargs (normalize all
    axes jointly, `eps=1e-6`), i.e. exactly the normalization `bioimageio.core`
    applies to the test tensor before inference and to raw inputs at prediction
    time.
    """
    mean = x.mean()
    std = ((x - mean) ** 2).mean() ** 0.5
    return (x - mean) / (std + ZMUV_EPS)


def _resolve_unet_class(dimensionality: str) -> type[UNet2D] | type[UNet3D]:
    """Map the configured dimensionality to its UNet class.

    Accepts ``"2D"``/``"2d"``/``"2"`` and ``"3D"``/``"3d"``/``"3"``.

    Raises:
        ValueError: for any other spelling.
    """
    if dimensionality in ["2D", "2d", "2"]:
        return UNet2D
    elif dimensionality in ["3D", "3d", "3"]:
        return UNet3D
    else:
        raise ValueError(f"Unknown dimensionality {dimensionality}")


def create_model(
    dimensionality: Literal["2D", "3D"],
    in_channels: int,
    out_channels: int,
    feature_maps: int | list[int] | tuple[int, ...],
    final_sigmoid: bool,
    layer_order: str = "bcr",
) -> UNet2D | UNet3D:
    """Instantiate the configured UNet architecture.

    Used both for the training model and for the fresh CPU instance that
    generates the bioimageIO test tensors from the packaged weights.
    """
    model_class = _resolve_unet_class(dimensionality)
    return model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        f_maps=feature_maps,
        final_sigmoid=final_sigmoid,
        layer_order=layer_order,
    )


def create_model_config(
    checkpoint_dir: Path,
    in_channels,
    out_channels,
    patch_size,
    dimensionality: Literal["2D", "3D"],
    layer_order: str,
    sparse,
    f_maps,
    max_num_iters,
    pre_trained: Optional[Path] = None,
):
    """Write training config to yaml file."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(PATH_TRAIN_TEMPLATE, "r") as f:
        train_template = yaml.load(f, Loader=yaml.FullLoader)

    train_template["model"]["in_channels"] = in_channels
    train_template["model"]["out_channels"] = out_channels
    train_template["model"]["layer_order"] = layer_order
    train_template["model"]["f_maps"] = f_maps
    train_template["model"]["name"] = _resolve_unet_class(dimensionality).__name__
    train_template["model"]["final_sigmoid"] = not sparse
    train_template["trainer"]["checkpoint_dir"] = str(checkpoint_dir)
    train_template["trainer"]["max_num_iterations"] = max_num_iters
    train_template["trainer"]["pre_trained"] = (
        str(pre_trained) if pre_trained else "null"
    )
    train_template["loaders"]["train"]["slice_builder"]["patch_shape"] = patch_size
    train_template["loaders"]["train"]["slice_builder"]["stride_shape"] = [
        max(i // 2, 1) for i in patch_size
    ]
    train_template["loaders"]["val"]["slice_builder"]["patch_shape"] = patch_size
    train_template["loaders"]["val"]["slice_builder"]["stride_shape"] = patch_size

    out_path = checkpoint_dir / FILE_CONFIG_TRAIN_YAML
    with open(out_path, "w") as yaml_file:
        yaml.dump(train_template, yaml_file, default_flow_style=False)


def unet_training(
    dataset_dir: str | Path,
    model_name: str,
    in_channels: int,
    out_channels: int,
    feature_maps: int | list[int] | tuple[int, ...],
    patch_size: tuple[int, int, int],
    max_num_iters: int,
    dimensionality: Literal["2D", "3D"],
    sparse: bool,
    device: str,
    modality: str = "",
    output_type: str = "",
    description: str = "",
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pre_trained: Optional[Path] = None,
    layer_order: str = "bcr",
    authors: list[str] | None = None,
    additional_citations: list[str] | None = None,
    license: str | None = None,
    documentation: str | None = None,
    amp: AmpSetting = "auto",
) -> None:
    """Train a new UNet model; the main entrypoint behind ``panseg --train``.

    Instantiates the 2D/3D UNet (optionally loading ``pre_trained`` weights for
    fine-tuning), builds train/val loaders from the h5 files in
    ``dataset_dir/<train|val>``, and runs `UNetTrainer` with a `DiceLoss`,
    saving checkpoints under ``PATH_PANSEG_MODELS / model_name``. After training,
    the model is registered in the model zoo and packaged as a bioimage.io model.

    Args:
        dataset_dir: directory containing the training data; must have a
            ``train`` and a ``val`` subdirectory, each holding `h5` files that
            store the input image under ``raw`` and a channel-less segmentation
            under ``label``. Validation is skipped if ``val`` holds no files.
        model_name: name of the new model, shown in the model selection. The
            checkpoints, training config, and packaged model are written to
            ``PATH_PANSEG_MODELS / model_name``, which must not exist yet.
            At least 5 characters.
        in_channels: number of channels of the input images (the data is
            checked against this count).
        out_channels: deprecated, always 1.
            number of predicted output channels, i.e. the number of
            semantic classes the model learns to predict.
        feature_maps: model capacity: number of feature maps in the first
            encoder level; deeper levels follow the geometric progression
            ``f^k`` with k=1..4. A sequence defines each encoder level size explicitly.
        patch_size: training patch shape ``(z, y, x)``. Training patches are
            extracted with half the patch shape as stride (overlapping), val
            patches without overlap. The z-extent must be ``1`` for 2D models.
        max_num_iters: maximum number of training iterations.
        dimensionality: whether to train a 2D or 3D UNet (also accepts
            ``"2d"``/``"2"`` and ``"3d"``/``"3"``). With 2D training on
            volumetric data, training happens per z-slice.
        sparse: whether the targets are sparse (e.g. affinity data); if True,
            the trained model outputs raw values (no final sigmoid).
        device: device to train on, e.g. ``"cuda:0"`` or ``"cpu"``. The batch
            size is fitted automatically to the device's memory; on machines
            with multiple GPUs, data parallel training is used.
        modality: imaging modality of the training data (e.g. ``"confocal"``),
            recorded in the model zoo metadata.
        output_type: what the model predicts (e.g. ``"boundaries"``,
            ``"cell segmentation"``), recorded in the model zoo metadata.
        description: short free-text description recorded in the model zoo metadata
            and in the packaged model description.
        resolution: voxel size ``(z, y, x)`` of the training data in
            micrometers, declared as the spatial scale of the packaged model.
        pre_trained: optional path to a `torch` checkpoint for fine-tuning;
            its architecture (channels, feature maps, layer order) must match
            the configured model.
        layer_order: composition of the network's conv blocks as one-letter
            codes (``b``/``g`` batch/group norm, ``c`` conv, ``r``/``l``/``e``
            ReLU/LeakyReLU/ELU), e.g. ``"bcr"`` = batchnorm+conv+ReLU.
        authors: model authors, one entry per author, either ``"Name"`` or
            ``"Name <email>"``; recorded in the packaged model metadata.
        additional_citations: citations added alongside PanSeg's built-in
            citation, one per entry: ``"<DOI or URL> [free text]"``.
        license: license under which the packaged model is published.
        documentation: markdown text packaged as the model's README
            documentation.
        amp: training precision mode: ``"auto"`` runs bf16 autocast on CUDA
            devices with bf16 support (loss reductions, weights, checkpoints,
            and exported tensors stay fp32) and silently falls back to fp32
            elsewhere; ``"off"`` always trains in fp32.
    """
    amp = validate_amp(amp)
    if dimensionality in ["2D", "2d", "2"] and patch_size[0] != 1:
        raise ValueError(
            "2D models train per z-slice; patch z-extent must be 1, "
            f"but got patch_size {tuple(patch_size)}."
        )

    # Model instantiation and logging
    final_sigmoid = not sparse
    model = create_model(
        dimensionality,
        in_channels,
        out_channels,
        feature_maps,
        final_sigmoid,
        layer_order,
    )
    logger.info(f"Using {model.__class__.__name__} model for training.")

    # Load fine-tuning weights into the unwrapped model before any
    # nn.DataParallel wrapping, otherwise the state dict keys mismatch
    if pre_trained is not None:
        load_pretrained_weights(model, pre_trained)

    # Device configuration
    batch_size = find_batch_size(
        model=model,
        in_channels=in_channels,
        patch_shape=patch_size,
        patch_halo=(4, 4, 4),  # some slack
        device=device,
        for_training=True,  # backward passes need ~2-3x the memory of a forward
        amp=amp,  # probe in the precision training will run in
    )

    if torch.cuda.device_count() > 1 and device != "cpu":
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for prediction.")
        batch_size *= torch.cuda.device_count()
        device = "cuda"

    logger.info(f"Sending model to {device}")
    model = model.to(device)

    # Data loaders setup
    logger.info(f"Creating train/val loaders with batch size {batch_size}")
    # Worker processes only pay off on GPU training: the compute runs
    # on the GPU while the workers prefetch the next batch in parallel. For cpu
    # training they are pure overhead and costs seconds per
    # worker on windows/macOS (spawn, no fork). In-process loading is then
    # also the fastest option. The workers are kept alive across epochs
    # (`persistent_workers`) for the same reason: respawning them per epoch
    # re-imports the whole chain and re-pickles the in-memory dataset.
    if device == "cpu":
        num_workers, pin_memory = 0, False
    else:
        num_workers, pin_memory = 2, True
    train_datasets = create_datasets(
        dataset_dir, "train", patch_size, dimensionality, in_channels=in_channels
    )
    val_datasets = create_datasets(
        dataset_dir, "val", patch_size, dimensionality, in_channels=in_channels
    )
    loaders = {
        "train": DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    }
    if len(val_datasets) > 0:
        loaders["val"] = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    else:
        loaders["val"] = []

    # Optimizer and training environment setup
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    checkpoint_dir = PATH_PANSEG_MODELS / model_name
    logger.info(f"Saving training files in {checkpoint_dir}")
    assert not checkpoint_dir.exists(), (
        f"Checkpoint dir {checkpoint_dir} already exists!"
    )

    create_model_config(
        checkpoint_dir,
        in_channels,
        out_channels,
        patch_size,
        dimensionality,
        layer_order,
        sparse,
        feature_maps,
        max_num_iters,
        pre_trained=pre_trained,
    )

    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=ReduceLROnPlateau(optimizer, factor=0.2, patience=10),
        loss_criterion=DiceLoss(),
        loaders=loaders,
        checkpoint_dir=checkpoint_dir,
        max_num_iterations=max_num_iters,
        device=device,
        amp=amp,
    )

    trainer.train()

    model_zoo.add_custom_model(
        new_model_name=model_name,
        location=checkpoint_dir,
        resolution=resolution,
        description=description,
        dimensionality=dimensionality,
        modality=modality,
        output_type=output_type,
    )

    weights = checkpoint_dir / "best_checkpoint.pytorch"
    if not weights.exists():
        weights = checkpoint_dir / "last_checkpoint.pytorch"

    # Generate the test tensors from the weights that are packaged below, on a
    # fresh instance of the same architecture. Working on CPU under `no_grad`
    # keeps the tensors numerically deterministic and frees the training model
    # and device from any involvement here.
    net = create_model(
        dimensionality,
        in_channels,
        out_channels,
        feature_maps,
        final_sigmoid,
        layer_order,
    )
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()

    # Obtain and save test in- and output for biio.
    # test_in is stored un-normalized: the model description declares
    # ZeroMeanUnitVariance preprocessing, which bioimageio.core applies to the test
    # tensor before inference. Apply the very same preprocessing here so that the
    # stored test_out corresponds to preprocess(test_in).
    # The batch is fetched with a dedicated worker-less loader: iterating the
    # training loader here would (re)spawn its worker processes for one batch.
    test_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size)
    test_in, _ = next(iter(test_loader))
    # bioimageIO needs a single sample; keep the packaged tensors small
    test_in = test_in[:1]

    if isinstance(net, UNet2D):
        # remove the singleton z-dimension from the input
        # Extra dim was only added in the panseg loader for the augmentations
        # Stored data and model are true 2D models
        # Note the loader adds a batch dimension
        test_in = torch.squeeze(test_in, dim=-3)

    with torch.no_grad():
        test_out = net(zero_mean_unit_variance(test_in))

    np.save(checkpoint_dir / "test_in.npy", test_in.numpy())
    np.save(checkpoint_dir / "test_out.npy", test_out.detach().cpu().numpy())

    try:
        axis_min_sizes = np.min(
            (model_zoo.compute_3D_halo_for_pytorch3dunet(net), patch_size), axis=0
        )
    except ValueError:
        axis_min_sizes = patch_size

    # Release everything the training run still holds before bioimageIO
    # validation: `test_model` instantiates a second full model on the training
    # device, which must not sit next to the training model and its optimizer
    # state.
    trainer.cleanup()
    del model, net, optimizer
    del loaders, test_loader, train_datasets, val_datasets, test_in, test_out
    gc.collect()
    torch.cuda.empty_cache()

    with chdir(checkpoint_dir):
        model_desc = make_model_description(
            weights=weights,
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_maps=feature_maps,
            final_sigmoid=final_sigmoid,
            axis_min_sizes=axis_min_sizes,
            dimensionality=dimensionality,
            layer_order=layer_order,
            modality=modality,
            output_type=output_type,
            description=description,
            resolution=resolution,
            test_in=Path("test_in.npy"),
            test_out=Path("test_out.npy"),
            panseg_config=checkpoint_dir / FILE_CONFIG_TRAIN_YAML,
            authors=authors,
            additional_citations=additional_citations,
            license=license,
            documentation=documentation,
        )
        test_model(model_desc, devices=[device]).display()
        model_desc.package(
            checkpoint_dir
            / f"biio_model_{''.join(c for c in model_name if c.isalnum())}.zip"
        )


def create_datasets(
    dataset_dir: str | Path,
    phase: Literal["train", "val"],
    patch_shape: tuple[int, int, int],
    dimensionality: Literal["2D", "3D"],
    in_channels: Optional[int] = None,
):
    """
    Load a dataset for training a unet.

    Args:
        data_dir (str): Must contain a train and a val folder, which inturn contain .h5 files.
        phase: Whether to load train or val.
        patch_shape (tuple): shape of the patch to be extracted from the raw data set
        in_channels (Optional[int]): expected number of raw channels. When given,
            each dataset's raw channel count is checked against it.
    """
    assert phase in ["train", "val"], f"Phase {phase} not supported"
    phase_dir = Path(dataset_dir) / phase
    file_paths = find_h5_files(phase_dir)
    datasets = []
    for file_path in file_paths:
        dataset = HDF5Dataset(
            file_path=file_path,
            augmenter=Augmenter(),
            patch_shape=patch_shape,
            dimensionality=dimensionality,
        )
        num_channels = dataset.raw.shape[0]
        if in_channels is not None and num_channels != in_channels:
            raise ValueError(
                f"Raw dataset '{file_path}' has {num_channels} channels, "
                f"but in_channels={in_channels} was configured."
            )
        datasets.append(dataset)
    return datasets


def find_h5_files(data_dir):
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"Not a directory {data_dir}"
    return (
        list(data_dir.glob("*.h5"))
        + list(data_dir.glob("*.hdf"))
        + list(data_dir.glob("*.hdf5"))
        + list(data_dir.glob("*.hd5"))
    )
