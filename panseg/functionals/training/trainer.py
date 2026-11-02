import logging
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from panseg.functionals.precision import (
    AmpMode,
    AmpSetting,
    amp_context,
    cudnn_benchmark,
    validate_amp,
)
from panseg.functionals.training.model import UNet2D
from panseg.functionals.training.utils import RunningAverage

logger = logging.getLogger(__name__)


def load_pretrained_weights(model: nn.Module, pre_trained: Path) -> None:
    """Load pre-trained weights into `model` for fine-tuning.

    Tolerates checkpoints wrapped in a ``{"model_state_dict": ...}`` dict (like
    `prediction.py` does) and models wrapped in ``nn.DataParallel`` (the weights
    are loaded into the unwrapped module).

    Args:
        model (nn.Module): model to load the weights into
        pre_trained (Path): path to the pre-trained model weights

    Raises:
        ValueError: if the checkpoint cannot be loaded, likely because its
            architecture or number of channels does not match the model.
    """
    logger.info(f"Loading pre-trained model from '{pre_trained}'...")
    state = torch.load(pre_trained, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:  # Model weights format may vary between versions
        state = state["model_state_dict"]
    net = model.module if isinstance(model, nn.DataParallel) else model
    try:
        net.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError(
            f"Failed to load pre-trained weights from '{pre_trained}': {e}. "
            "The checkpoint likely has a different architecture or a different "
            "number of channels than the configured model."
        ) from e


class UNetTrainer:
    """UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): learning rate scheduler
        loss_criterion (nn.Module): loss function
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_iterations (int): maximum number of iterations
        device (str): device to use for training
        log_after_iters (int): number of iterations before logging to tensorboard
        amp (str): precision mode (``"auto"``: bf16 autocast on supported CUDA
            devices, else fp32; ``"off"``: always fp32). Model weights and
            checkpoints stay fp32 in either mode. Defaults to "auto".
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: ReduceLROnPlateau,
        loss_criterion: nn.Module,
        loaders: dict,
        checkpoint_dir: Path,
        max_num_iterations: int,
        device: str = "cuda",
        log_after_iters: int = 100,
        amp: AmpSetting = "auto",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_iterations = max_num_iterations
        self.max_num_epochs = max_num_iterations // len(loaders["train"]) + 1
        self.device = device
        self.log_after_iters = log_after_iters
        self.amp: AmpMode = validate_amp(amp)
        self.best_eval_loss = float("+inf")

        self.num_iterations = 1

        # init tensorboard logger
        self.writer = SummaryWriter(log_dir=checkpoint_dir / "logs")

    def train(self) -> None:
        # every training/validation patch has the same shape, so cuDNN
        # autotuning pays off (the previous flag value is restored on exit)
        with cudnn_benchmark():
            for epoch in range(self.max_num_epochs):
                logger.info(f"Epoch [{epoch}/~{self.max_num_epochs}]")
                # train for one epoch
                should_terminate = self.train_epoch()

                logger.info("Validating...")
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_loss = self.validate()
                logger.info(f"Val Loss: {eval_loss}.")
                if self.writer is not None:
                    self.writer.add_scalar("Loss/val", eval_loss, self.num_iterations)
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()
                # remember best validation metric
                is_best = eval_loss < self.best_eval_loss
                if is_best:
                    self.best_eval_loss = eval_loss

                # save checkpoint
                self._save_checkpoint(is_best)

                if should_terminate:
                    logger.info("Stopping criterion is satisfied. Finishing training")
                    return

        logger.info(
            f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training..."
        )

    def train_epoch(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()

        # sets the model in training mode
        self.model.train()

        for _input, target in tqdm(self.loaders["train"]):
            _input, target = _input.to(self.device), target.to(self.device)
            output, loss = self._forward_pass(_input, target)

            train_losses.update(loss.item(), self._batch_size(_input))

            # compute gradients and update parameters; the autocast context
            # covers backward and the step too (per the AMP contract; gradients
            # follow the dtypes recorded during the forward pass)
            self.optimizer.zero_grad()
            with amp_context(self.amp, self.device):
                loss.backward()
                self.optimizer.step()

            if self.num_iterations % self.log_after_iters == 0:
                # log stats, params and images
                logger.info(f"Train Loss: {train_losses.avg}.")
                if self.writer is not None:
                    self.writer.add_scalar(
                        "Loss/train", train_losses.avg, self.num_iterations
                    )

                if self.should_stop():
                    return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(
                f"Maximum number of iterations {self.max_num_iterations} exceeded."
            )
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]["lr"]
        if lr < min_lr:
            logger.info(f"Learning rate below the minimum {min_lr}.")
            return True

        return False

    def validate(self):
        val_losses = RunningAverage()

        with torch.no_grad():
            for _input, target in tqdm(self.loaders["val"]):
                _input, target = _input.to(self.device), target.to(self.device)

                output, loss = self._forward_pass(_input, target)
                val_losses.update(loss.item(), self._batch_size(_input))

            return val_losses.avg

    def _forward_pass(
        self, _input: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        with amp_context(self.amp, self.device):
            if isinstance(model, UNet2D):
                # remove the singleton z-dimension from the input
                _input = torch.squeeze(_input, dim=-3)
                # forward pass
                output = self.model(_input)
                # add the singleton z-dimension to the output
                output = torch.unsqueeze(output, dim=-3)
            else:
                # forward pass
                output = self.model(_input)

            # the loss computes in fp32 (see losses.py) even under autocast
            loss = self.loss_criterion(output, target)
        return output, loss

    def _save_checkpoint(self, is_best):
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = self.checkpoint_dir / "last_checkpoint.pytorch"
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        torch.save(state_dict, last_file_path)
        if is_best:
            logger.info("Saving best checkpoint")
            best_file_path = self.checkpoint_dir / "best_checkpoint.pytorch"
            shutil.copyfile(last_file_path, best_file_path)

    def cleanup(self) -> None:
        """Release the resources held after training.

        Closes the tensorboard writer, drops the optimizer state (Adam keeps
        two moment buffers per parameter, doubling the training footprint),
        moves the model back to CPU and drops the loader references. Call this
        before any large allocation that follows training (e.g. the bioimageIO
        model validation); together with dropping the caller's references and
        ``torch.cuda.empty_cache()`` it frees the GPU memory of the training
        run. Idempotent: repeated calls do nothing further.
        """
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.optimizer.state.clear()
        self.model.to("cpu")
        self.loaders = {}

    @staticmethod
    def _batch_size(_input):
        if isinstance(_input, (list, tuple)):
            return _input[0].size(0)
        else:
            return _input.size(0)
