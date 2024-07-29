from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from Utils.nn.parts import TransformerDecoderBottleneck

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        stop_at: int,
        block: Type[BasicBlock],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        maxpool: bool =True
    ) -> None:
        super().__init__()
        if norm_layer is None:
            if stop_at > 0:
                norm_layer = nn.BatchNorm2d
            else:
                norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 512
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        #self.project = TransformerDecoderBottleneck(512, (8, 8), 4, 4, 4, 1024, 0.0, 0.0)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if stop_at > 0:
            self.project = nn.Sequential(
                nn.Unflatten(1, (in_features, 1, 1)),
                nn.ConvTranspose2d(in_features, 512, 8),
            )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer1 = self._make_layer(block, 512, layers[0], stop_at>0)
        self.layer2 = self._make_layer(block, 256, layers[1], stop_at>0)
        self.layer3 = self._make_layer(block, 128, layers[2], stop_at>0)
        self.layer4 = self._make_layer(block, 64, layers[3], stop_at>0)
        self.out_proj = conv1x1(64, out_features) if stop_at>0 else nn.Linear(64, out_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        previous_dilation = self.dilation

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    self.inplanes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        
        x = self.project(x) # (batch, 512, 8, 8) (if self.stop_at > 0, else (batch_size, 512))
        x = self.bn1(x)
        x = self.relu(x)

        if self.stop_at<5:
            x = self.upsample(x) # (batch, 512, 16, 16)

        x = self.layer1(x)
        if self.stop_at<4:
            x = self.upsample(x) # (batch, 256, 32, 32)

        x = self.layer2(x)
        if self.stop_at<3:
            x = self.upsample(x) # (batch, 128, 64, 64)

        x = self.layer3(x)
        if self.stop_at<1:
            x = self.upsample(x) # (batch, 64, 128, 128)

        x = self.layer4(x)

        x = self.out_proj(x)

        return x

    def forward(self, x: Tensor, stop_at=None) -> Tensor:
        return self._forward_impl(x, stop_at)


def _resnet(
    in_shape: Tuple[int, int, int],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(in_shape, BasicBlock, layers, **kwargs)

    return model


def resnet18(in_shape, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        in_shape: Tuple[int, int, int]
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """

    return _resnet(in_shape, [2, 2, 2, 2], **kwargs)

def resnet34(in_shape, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        in_shape: Tuple[int, int, int]
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """

    return _resnet(in_shape, [3, 4, 6, 3], **kwargs)