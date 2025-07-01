# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from wtconvnext import WTConvNeXt
device = torch.device('cuda')
# 门控融合模块
class GateFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(device)
        self.scale = nn.Parameter(torch.tensor(1.))  # 可学习缩放系数

    def forward(self, resnet_feat, wavelet_feat):

        gate = torch.sigmoid(self.conv(wavelet_feat)).to(device)
        return self.scale* resnet_feat * gate+resnet_feat
# 提取中间层
class FeatureExtractor(nn.Module):
    def __init__(self, model, stage_indices=[0, 1, 2]):
        super().__init__()
        self.model = model
        self.stage_indices = stage_indices

    def forward(self, x):
        outputs = {}
        # for name, layer in self.model.named_children():
        #     x = layer(x)  # 所有层都会执行
        #     if name in return_layers:
        #         outputs[return_layers[name]] = x  # 只保存需要的输出
        # return outputs
        x = self.model.stem(x)
        x = self.model.stages[0](x)
        for i in self.stage_indices:
            x = self.model.stages[i](x)
            outputs[f'{i-1}'] = x
        return outputs

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


from torchvision.utils import make_grid


def visualize_features(tensor, title, nrow=8, normalize=True, cmap='viridis'):
    """
    可视化特征图

    参数:
        tensor (torch.Tensor): 特征图 [C, H, W]
        title (str): 图像标题
        nrow (int): 网格中每行显示的图像数
        normalize (bool): 是否归一化到0-1范围
        cmap (str): 使用的颜色图
    """
    # 转换为CPU numpy数组
    tensor = tensor.detach().cpu()

    # 归一化每个通道
    if normalize:
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-6:
            tensor = (tensor - min_val) / (max_val - min_val)

    # 创建网格
    grid = make_grid(tensor.unsqueeze(1), nrow=nrow, padding=2, normalize=False)

    # 转换为numpy并可视化
    plt.figure(figsize=(15, 10))
    plt.imshow(grid.permute(1, 2, 0), cmap=cmap)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def visualize_comparison(resnet_feat, wavelet_feat, fused_feat, layer_name, max_channels=32):
    """
    可视化三种特征的比较

    参数:
        resnet_feat (torch.Tensor): ResNet特征 [B, C, H, W]
        wavelet_feat (torch.Tensor): 小波特征 [B, C, H, W]
        fused_feat (torch.Tensor): 融合特征 [B, C, H, W]
        layer_name (str): 层名称
        max_channels (int): 最多可视化的通道数
    """
    # 取第一个样本
    resnet_feat = resnet_feat[0]
    wavelet_feat = wavelet_feat[0]
    fused_feat = fused_feat[0]

    # 限制通道数
    if resnet_feat.size(0) > max_channels:
        resnet_feat = resnet_feat[:max_channels]
        wavelet_feat = wavelet_feat[:max_channels]
        fused_feat = fused_feat[:max_channels]

    # 可视化三种特征
    visualize_features(resnet_feat, f"ResNet Features - {layer_name}", cmap='viridis')
    visualize_features(wavelet_feat, f"Wavelet Features - {layer_name}", cmap='plasma')
    visualize_features(fused_feat, f"Fused Features - {layer_name}", cmap='inferno')

    # 可视化通道平均值
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(resnet_feat.mean(0).detach().cpu(), cmap='viridis')
    plt.title(f"ResNet Mean - {layer_name}")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(wavelet_feat.mean(0).detach().cpu(), cmap='plasma')
    plt.title(f"Wavelet Mean - {layer_name}")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(fused_feat.mean(0).detach().cpu(), cmap='inferno')
    plt.title(f"Fused Mean - {layer_name}")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, backbone2: nn.Module, train_backbone: bool, return_interm_layers: bool):

        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        for name, parameter in backbone2.named_parameters():
            print(name, )
            if not train_backbone or 'stages.1' not in name and 'stages.2' not in name and 'stages.3' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]

            # 小波卷积网络
            return_layers2 = {"stages.1": "0", "stages.2": "1", "stages.3": "2"}
            self.num_channels2 = [192, 384, 768]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

            # 小波卷积网络
            return_layers2 = {"stages.3": "0"}
            self.num_channels2 = [768]
        #获得不同层
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # 小波卷积网络
        self.body2 = FeatureExtractor(backbone2, [1, 2, 3])

    def forward(self, tensor_list: NestedTensor):

        resnet_feat = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        wavelet_feat = self.body2(tensor_list.tensors)

        # 存储用于可视化的特征
        vis_features = {}

        for name, _ in resnet_feat.items():
            adjust_conv = nn.Conv2d(wavelet_feat[name].shape[1], resnet_feat[name].shape[1], kernel_size=1).to(device)
            wavelet_feat[name] = adjust_conv(wavelet_feat[name])
            # 获取目标尺寸
            target_size = resnet_feat[name].shape[-2:]
            wavelet_feat[name] = F.interpolate(wavelet_feat[name], size=target_size, mode='bilinear')
            fused_feat = GateFusion(resnet_feat[name].shape[1])(resnet_feat[name], wavelet_feat[name]).to(device)

            m = tensor_list.mask
            assert m is not None

            mask = F.interpolate(m[None].float(), size=resnet_feat[name].shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(fused_feat, mask)


            # visualize_comparison(
            #     features['resnet'],
            #     features['wavelet'],
            #     features['fused'],
            #     layer_name
            # )

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        # resnet50卷积
        backbone1 = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"

        # 小波卷积
        backbone2 = WTConvNeXt(3, 6, wt_levels=(6, 5, 4, 3))
        super().__init__(
        backbone1,backbone2, train_backbone, return_interm_layers)

        self.backbone1=backbone1
        self.backbone2=backbone2

        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    #对于backbone输出特征图进行位置编码，用于后续Transformer部分
    position_embedding = build_position_encoding(args)
    #是否采用预训练的backbone
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
