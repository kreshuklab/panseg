from functools import partial
from typing import TYPE_CHECKING, Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_conv(
    in_channels, out_channels, kernel_size, order, num_groups, padding, is3d
):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list of tuple (name, module)
    """
    assert "c" in order, "Conv layer MUST be present"
    assert order[0] not in "rle", (
        "Non-linearity cannot be the first operation in the layer"
    )

    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ("g" in order or "b" in order)
            if is3d:
                conv = nn.Conv3d(
                    in_channels, out_channels, kernel_size, padding=padding, bias=bias
                )
            else:
                conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=padding, bias=bias
                )

            modules.append(("conv", conv))
        elif char == "g":
            is_before_conv = i < order.index("c")
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, (
                f"Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}"
            )
            modules.append(
                (
                    "groupnorm",
                    nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
                )
            )
        elif char == "b":
            is_before_conv = i < order.index("c")
            if is3d:
                bn = nn.BatchNorm3d
            else:
                bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(("batchnorm", bn(in_channels)))
            else:
                modules.append(("batchnorm", bn(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        is3d=True,
    ):
        super(SingleConv, self).__init__()

        for name, module in create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding, is3d
        ):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        encoder,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        is3d=True,
    ):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module(
            "SingleConv1",
            SingleConv(
                conv1_in_channels,
                conv1_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                is3d=is3d,
            ),
        )
        # conv2
        self.add_module(
            "SingleConv2",
            SingleConv(
                conv2_in_channels,
                conv2_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                is3d=is3d,
            ),
        )


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    from the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): use 3d or 2d convolutions/pooling operation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=2,
        pool_type="max",
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        is3d=True,
    ):
        super(Encoder, self).__init__()
        assert pool_type in ["max", "avg"]
        if apply_pooling:
            if pool_type == "max":
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = DoubleConv(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            is3d=is3d,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (bool): should the input be upsampled
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        scale_factor=(2, 2, 2),
        conv_layer_order="gcr",
        num_groups=8,
        mode="nearest",
        padding=1,
        upsample=True,
        is3d=True,
    ):
        super(Decoder, self).__init__()

        if upsample:
            self.upsampling = InterpolateUpsampling(mode=mode)
            # concat joining
            self.joining = partial(self._joining, concat=True)

        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = DoubleConv(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            is3d=is3d,
        )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(
    in_channels,
    f_maps,
    conv_kernel_size,
    conv_padding,
    layer_order,
    num_groups,
    pool_kernel_size,
    is3d,
):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            # apply conv_coord only in the first encoder if any
            encoder = Encoder(
                in_channels,
                out_feature_num,
                conv_kernel_size=conv_kernel_size,
                apply_pooling=False,
                conv_layer_order=layer_order,
                num_groups=num_groups,
                padding=conv_padding,
                is3d=is3d,
            )
        else:
            encoder = Encoder(
                f_maps[i - 1],
                out_feature_num,
                conv_kernel_size=conv_kernel_size,
                pool_kernel_size=pool_kernel_size,
                conv_layer_order=layer_order,
                num_groups=num_groups,
                padding=conv_padding,
                is3d=is3d,
            )

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(
    f_maps, conv_kernel_size, conv_padding, layer_order, num_groups, is3d
):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        out_feature_num = reversed_f_maps[i + 1]

        decoder = Decoder(
            in_feature_num,
            out_feature_num,
            conv_kernel_size=conv_kernel_size,
            conv_layer_order=layer_order,
            num_groups=num_groups,
            padding=conv_padding,
            is3d=is3d,
        )
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode="nearest"):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True `predict`/`activate` apply element-wise nn.Sigmoid to the
            forward logits, otherwise nn.Softmax. `forward` itself always returns logits
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True `predict`/`activate` apply the Sigmoid/Softmax normalization
            to the forward logits; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the conv layer
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool,
        f_maps: int | list[int] | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = True,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        conv_padding: int = 1,
        is3d: bool = True,
    ):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert num_groups is not None, (
                "num_groups must be specified if GroupNorm is used"
            )

        # create encoder path
        self.encoders = create_encoders(
            in_channels,
            f_maps,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            pool_kernel_size,
            is3d,
        )

        # create decoder path
        self.decoders = create_decoders(
            f_maps, conv_kernel_size, conv_padding, layer_order, num_groups, is3d
        )

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # the output contract: forward() returns logits in every mode; the
        # final activation is applied explicitly where the output leaves
        # PanSeg (see `predict` and the `UNet*Probabilities` export classes)
        return x

    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        """Map forward logits to the model's normalized output.

        Applies the configured `final_activation` (Sigmoid or Softmax); a no-op
        for regression models without one. The single place that touches
        `final_activation`.

        Args:
            logits (torch.Tensor): output of `forward`.

        Returns:
            torch.Tensor: the normalized output (probabilities for segmentation
            models, the input unchanged for regression models).
        """
        if self.final_activation is None:
            return logits
        return self.final_activation(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output probabilities for the input `x`.

        `forward` always returns logits; `predict` is the explicit entry for
        the points where probabilities leave PanSeg (inference probability
        maps, packaged bioimageIO test tensors, BioImage.IO export).

        Args:
            x (torch.Tensor): model input.

        Returns:
            torch.Tensor: `activate(forward(x))`.
        """
        return self.activate(self(x))

    def output_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """The activation a consumer must apply to `forward`'s output.

        `forward` returns logits, so consumers that call `forward` and need
        probabilities (e.g. `ArrayPredictor`, which keeps control of the
        forward pass for `torch.compile`) apply this. The identity for models
        whose `forward` is already activated (`UNet*Probabilities`), which makes
        a double activation impossible by construction.
        """
        return self.activate


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool = True,
        f_maps: int | list[int] | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = True,
        conv_padding: int = 1,
        **kwargs,
    ):
        super(UNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            is3d=True,
        )


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool = True,
        f_maps: int | list[int] | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = True,
        conv_padding: int = 1,
        **kwargs,
    ):
        super(UNet2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            is3d=False,
        )


if TYPE_CHECKING:
    # Typing only: the mixin is combined with an `AbstractUNet` subclass, so
    # type checkers must see its API (`activate`, `forward`). At runtime the base
    # is `object`, keeping the MRO of the concrete classes linear.
    _ProbabilitiesBase = AbstractUNet
else:
    _ProbabilitiesBase = object


class ProbabilitiesForward(_ProbabilitiesBase):
    """Mixin for `AbstractUNet` subclasses whose `forward` returns probabilities.

    Mix into the concrete architecture as ``class UNet3DProbabilities(
    ProbabilitiesForward, UNet3D)``. Export-only variant for BioImage.IO:
    published models must emit normalized probabilities when called through
    `bioimageio.core`, which just calls `forward`. State-dict compatible with
    the base architecture — no new parameters or buffers — so trained weights
    load unchanged.

    The mixin owns the whole activation contract, so the subclasses cannot
    diverge: `forward` activates, `predict` therefore does not activate again,
    and `output_activation` is the identity so consumers that activate after
    `forward` (e.g. `ArrayPredictor`) never double-activate either.
    """

    def forward(self, x):
        return self.activate(super().forward(x))

    def predict(self, x):
        # forward() already returns probabilities; never activate twice
        return self(x)

    def output_activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return nn.Identity()


class UNet3DProbabilities(ProbabilitiesForward, UNet3D):
    """`UNet3D` whose `forward` returns probabilities instead of logits.

    See `ProbabilitiesForward`.
    """


class UNet2DProbabilities(ProbabilitiesForward, UNet2D):
    """`UNet2D` whose `forward` returns probabilities instead of logits.

    See `ProbabilitiesForward`.
    """


class SpocoNet(nn.Module):
    """
    Wrapper around the f-network and the moving average g-network.
    """

    def __init__(
        self,
        net_f: UNet2D | UNet3D,
        net_g: UNet2D | UNet3D,
        m: float = 0.999,
        init_equal: bool = True,
    ):
        super().__init__()
        self.net_f = net_f
        self.net_g = net_g
        self.m = m

        if init_equal:
            self._initialize_identical_weights()

    @classmethod
    def from_unet_params(
        cls,
        in_channels: int,
        out_channels: int,
        f_maps: int | list[int] | tuple[int, ...],
        layer_order: str = "bcr",
        m: float = 0.999,
        init_equal: bool = True,
    ) -> Self:
        net_f = UNet2D(
            in_channels, out_channels, f_maps=f_maps, layer_order=layer_order
        )
        net_g = UNet2D(
            in_channels, out_channels, f_maps=f_maps, layer_order=layer_order
        )
        return cls(net_f, net_g, m, init_equal)

    def _initialize_identical_weights(self):
        for param_f, param_g in zip(self.net_f.parameters(), self.net_g.parameters()):
            param_g.data.copy_(param_f.data)  # initialize
            param_g.requires_grad = False  # freeze g parameters

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the g
        """
        for param_f, param_g in zip(self.net_f.parameters(), self.net_g.parameters()):
            param_g.data = param_g.data * self.m + param_f.data * (1.0 - self.m)

    def forward(self, im_f, im_g):
        # compute f-embeddings
        emb_f = self.net_f(im_f)

        # compute g-embeddings
        with torch.no_grad():  # no gradient to g-embeddings
            self._momentum_update()  # momentum update of g
            emb_g = self.net_g(im_g)

        return emb_f, emb_g


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2**k for k in range(num_levels)]


def get_spoco(
    in_channels: int, out_channels: int, f_maps: list[int], layer_order="bcr"
) -> SpocoNet:
    net_f = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        f_maps=f_maps,
        layer_order=layer_order,
    )

    net_g = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        f_maps=f_maps,
        layer_order=layer_order,
    )

    return SpocoNet(net_f, net_g)
