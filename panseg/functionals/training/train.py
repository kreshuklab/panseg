import logging
from contextlib import chdir
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import yaml
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
from panseg.functionals.prediction.utils.size_finder import find_batch_size
from panseg.functionals.training.augs import Augmenter
from panseg.functionals.training.biio import make_model_description
from panseg.functionals.training.h5dataset import HDF5Dataset
from panseg.functionals.training.losses import DiceLoss
from panseg.functionals.training.model import UNet2D, UNet3D
from panseg.functionals.training.trainer import UNetTrainer

logger = logging.getLogger(__name__)


def create_model_config(
    checkpoint_dir: Path,
    in_channels,
    out_channels,
    patch_size,
    dimensionality: Literal["2D", "3D"],
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
    train_template["model"]["f_maps"] = f_maps
    if dimensionality in ["2D", "2d", "2"]:
        train_template["model"]["name"] = "UNet2D"
    elif dimensionality in ["3D", "3d", "3"]:
        train_template["model"]["name"] = "UNet3D"
    else:
        raise ValueError(f"Unknown dimensionality {dimensionality}")
    train_template["model"]["final_sigmoid"] = not sparse
    train_template["trainer"]["checkpoint_dir"] = str(checkpoint_dir)
    train_template["trainer"]["max_num_iterations"] = max_num_iters
    train_template["trainer"]["pre_trained"] = (
        str(pre_trained) if pre_trained else "null"
    )
    train_template["loaders"]["train"]["slice_builder"]["patch_shape"] = patch_size
    train_template["loaders"]["train"]["slice_builder"]["stride_shape"] = list(
        i // 2 for i in patch_size
    )
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
) -> None:
    """
    Main entrypoint for training a new unet model. Gets called when calling `panseg --train` from cli.
    """
    # Model instantiation and logging
    final_sigmoid = not sparse
    if dimensionality in ["2D", "2d", "2"]:
        model = UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            f_maps=feature_maps,
            final_sigmoid=final_sigmoid,
            layer_order="gcr",
        )
    elif dimensionality in ["3D", "3d", "3"]:
        model = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            f_maps=feature_maps,
            final_sigmoid=final_sigmoid,
            layer_order="gcr",
        )
    else:
        raise ValueError(f"Unknown dimensionality {dimensionality}")
    logger.info(f"Using {model.__class__.__name__} model for training.")

    # Device configuration
    batch_size = find_batch_size(
        model=model,
        in_channels=in_channels,
        patch_shape=patch_size,
        patch_halo=(4, 4, 4),  # some slack
        device=device,
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
    train_datasets = create_datasets(dataset_dir, "train", patch_size, dimensionality)
    val_datasets = create_datasets(dataset_dir, "val", patch_size, dimensionality)
    loaders = {
        "train": DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
    }
    if len(val_datasets) > 0:
        loaders["val"] = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
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
        sparse,
        feature_maps,
        max_num_iters,
        pre_trained=pre_trained,
    )

    # Trainer initialization and execution
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=ReduceLROnPlateau(optimizer, factor=0.2, patience=10),
        loss_criterion=DiceLoss(),
        loaders=loaders,
        checkpoint_dir=checkpoint_dir,
        max_num_iterations=max_num_iters,
        device=device,
        pre_trained=pre_trained,
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

    # Obtain and save test in- and output for biio
    test_in, _ = next(iter(loaders["train"]))
    if isinstance(model, UNet2D):
        # remove the singleton z-dimension from the input
        # Extra dim was only added in the panseg loader for the augmentations
        # Stored data and model are true 2D models
        # Note the loader adds a batch dimension
        test_in = torch.squeeze(test_in, dim=-3)

    model.eval()
    test_out = model(test_in.to(device))

    np.save(checkpoint_dir / "test_in.npy", test_in.numpy())
    np.save(checkpoint_dir / "test_out.npy", test_out.detach().cpu().numpy())

    with chdir(checkpoint_dir):
        model_desc = make_model_description(
            weights=weights,
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_maps=feature_maps,
            patch_size=patch_size,
            dimensionality=dimensionality,
            modality=modality,
            output_type=output_type,
            description=description,
            resolution=resolution,
            test_in=Path("test_in.npy"),
            test_out=Path("test_out.npy"),
            panseg_config=checkpoint_dir / FILE_CONFIG_TRAIN_YAML,
        )
        model_desc.package(
            checkpoint_dir
            / f"biio_model_{''.join(c for c in model_name if c.isalnum())}.zip"
        )


def create_datasets(
    dataset_dir: str | Path,
    phase: Literal["train", "val"],
    patch_shape: Tuple[int, int, int],
    dimensionality: Literal["2D", "3D"],
):
    """
    Load a dataset for training a unet.

    Args:
        data_dir (str): Must contain a train and a val folder, which inturn contain .h5 files.
        phase: Whether to load train or val.
        patch_shape (tuple): shape of the patch to be extracted from the raw data set
    """
    assert phase in ["train", "val"], f"Phase {phase} not supported"
    phase_dir = Path(dataset_dir) / phase
    file_paths = find_h5_files(phase_dir)
    return [
        HDF5Dataset(
            file_path=file_path,
            augmenter=Augmenter(),
            patch_shape=patch_shape,
            dimensionality=dimensionality,
        )
        for file_path in file_paths
    ]


def find_h5_files(data_dir):
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"Not a directory {data_dir}"
    return (
        list(data_dir.glob("*.h5"))
        + list(data_dir.glob("*.hdf"))
        + list(data_dir.glob("*.hdf5"))
        + list(data_dir.glob("*.hd5"))
    )
