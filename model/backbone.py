import torch
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n, eps=1e-6):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.zeros(n))
        self.eps = eps

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        var = self.running_var.reshape(1, -1, 1, 1)
        mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (var + self.eps).rsqrt()
        bias = b - mean * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        else:
            return_layers = {'layer4': 'feat4'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, inputs):
        xs = self.body(inputs)
        out = []
        for _, v in xs.items():
            out.append(v)
        return out


class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(pretrained=True,
                                                     replace_stride_with_dilation=[False, False, dilation],
                                                     norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
