import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from PIL import Image
import torchvision.transforms as transforms
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from util import wavelet
import os

class SELayer(nn.Module):
    """通道注意力模块（SE Block），适配多频带输入"""

    def __init__(self, channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',
                 se_reduction=4):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # # 添加SE模块到每个小波层级
        # self.se_layers = nn.ModuleList([
        #     SELayer(channels=in_channels * 4, reduction=se_reduction)
        #     for _ in range(wt_levels)
        # ])  # 每个分解层级一个SE模块

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1,
                                   dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding=(kernel_size - 1) // 2, stride=1,
                       dilation=1, groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None
        # 新增：用于存储中间特征的列表
        self.curr_x_tags = []
        self.feature_shapes = []

        self.recon_curr_x = []  # 存储重构过程中的特征
        self.next_x_ll_list=[]

        # 新增：存储中间结果
        self.base_conv_output = None  # 存储 base_conv 后的输出
        self.after_addition = None  # 存储相加后的输出

    def forward(self, x):
        self.curr_x_tags = []
        self.feature_shapes = []
        self.recon_curr_x = []  # 存储重构过程中的特征
        self.next_x_ll_list=[]
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape

            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])

            # 存储中间特征（在reshape之前）
            self.curr_x_tags.append(curr_x_tag.detach().clone())
            self.feature_shapes.append(shape_x)

            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            # # --- 修改点1：在卷积后插入SE模块 ---
            # curr_x_tag = self.wavelet_convs[i](curr_x_tag)
            # curr_x_tag = self.se_layers[i](curr_x_tag)  # 应用通道注意力
            # curr_x_tag = self.wavelet_scale[i](curr_x_tag)
            # # --------------------------------

            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            # 存储重构特征
            self.recon_curr_x.append(curr_x.detach().clone())

            next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)
            # 存储当前层的 next_x_ll
            self.next_x_ll_list.append(next_x_ll.detach().clone())
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        self.base_conv_output=x.detach().clone()
        x = x + x_tag
        self.after_addition =x.detach().clone()


        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


# ================== 可视化函数 ==================
def load_image(image_path, img_size=256):
    """加载并预处理图像"""
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # 添加batch维度


def visualize_features(features, level, save_dir=None, num_channels=4, title_prefix=""):
    """
    可视化特征图

    参数:
        features: 特征张量 [batch, channels*4, H, W]
        level: 小波分解层级
        save_dir: 保存图像的目录
        num_channels: 每行显示的通道数
        title_prefix: 标题前缀
    """
    # 转换为numpy并移除batch维度
    feat_np = features.squeeze(0).cpu().numpy()
    n_channels = feat_np.shape[0]

    # 创建网格
    rows = (n_channels + num_channels - 1) // num_channels
    fig, axes = plt.subplots(rows, num_channels, figsize=(15, 4 * rows))
    fig.suptitle(f'{title_prefix} Level {level + 1} Features (Shape: {features.shape})', fontsize=16)

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_channels):
        row_idx = i // num_channels
        col_idx = i % num_channels

        channel_feat = feat_np[i]

        # 归一化到[0,1]以便可视化
        vmin, vmax = np.percentile(channel_feat, [1, 99])
        channel_feat = np.clip((channel_feat - vmin) / (vmax - vmin), 0, 1)

        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
        ax.imshow(channel_feat, cmap='gray')
        ax.set_title(f'Channel {i}')
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_channels, rows * num_channels):
        row_idx = i // num_channels
        col_idx = i % num_channels
        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
        ax.axis('off')

    plt.tight_layout()

    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{title_prefix.lower()}_level_{level + 1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_recon_curr_x(curr_x, level, save_dir=None):
    """
    可视化重构过程中的curr_x特征

    参数:
        curr_x: 重构特征张量 [batch, in_channels, 4, H, W]
        level: 小波层级
        save_dir: 保存图像的目录
    """
    # 转换为numpy并移除batch维度
    curr_x_np = curr_x.squeeze(0).cpu().numpy()

    # 获取子带名称
    subband_names = ['LL', 'LH', 'HL', 'HH']

    # 创建网格 (通道数 x 子带数)
    in_channels = curr_x_np.shape[0]
    fig, axes = plt.subplots(in_channels, 4, figsize=(15, 4 * in_channels))
    fig.suptitle(f'Reconstruction Input Level {level + 1} (Shape: {curr_x.shape})', fontsize=16)

    if in_channels == 1:
        axes = axes.reshape(1, -1)

    for c in range(in_channels):
        for s in range(4):
            if in_channels > 1:
                ax = axes[c, s]
            else:
                ax = axes[s]

            subband_feat = curr_x_np[c, s]

            # 归一化
            vmin, vmax = np.percentile(subband_feat, [1, 99])
            subband_feat = np.clip((subband_feat - vmin) / (vmax - vmin), 0, 1)

            ax.imshow(subband_feat, cmap='gray')
            ax.set_title(f'Channel {c}, Subband {subband_names[s]}')
            ax.axis('off')

    plt.tight_layout()

    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'recon_curr_x_level_{level + 1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved reconstruction visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_next_x_ll(next_x_ll_list, save_dir=None):
    """
    可视化重构后的低频分量 (next_x_ll)

    参数:
        next_x_ll_list: 存储每层重构低频分量的列表
        save_dir: 保存图像的目录
    """
    for level, next_x_ll in enumerate(next_x_ll_list):
        # 转换为numpy并移除batch维度
        next_x_ll_np = next_x_ll.squeeze(0).cpu().numpy()
        n_channels = next_x_ll_np.shape[0]

        # 创建网格 (通道数 x 1)
        fig, axes = plt.subplots(n_channels, 1, figsize=(6, 6 * n_channels))
        fig.suptitle(f'Reconstructed Low-Frequency (Level {level + 1}) (Shape: {next_x_ll.shape})', fontsize=16)

        if n_channels == 1:
            axes = [axes]

        for c in range(n_channels):
            channel_feat = next_x_ll_np[c]

            # 归一化
            vmin, vmax = np.percentile(channel_feat, [1, 99])
            channel_feat = np.clip((channel_feat - vmin) / (vmax - vmin), 0, 1)

            ax = axes[c]
            ax.imshow(channel_feat, cmap='gray')
            ax.set_title(f'Channel {c}')
            ax.axis('off')

        plt.tight_layout()

        # 保存或显示
        if save_dir:
            save_path = os.path.join(save_dir, f'next_x_ll_level_{level + 1}.png')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved next_x_ll visualization to {save_path}")
        else:
            plt.show()

        plt.close()


def visualize_base_conv_output(feature, save_path=None):
    """
    可视化 base_conv 后的输出

    参数:
        feature: 特征张量 [batch, channels, H, W]
        save_path: 保存图像的完整路径
    """
    # 转换为numpy并移除batch维度
    feat_np = feature.squeeze(0).cpu().numpy()
    n_channels = feat_np.shape[0]

    # 创建网格 (通道数 x 1)
    fig, axes = plt.subplots(n_channels, 1, figsize=(6, 6 * n_channels))
    fig.suptitle(f'Base Conv Output (Shape: {feature.shape})', fontsize=16)

    if n_channels == 1:
        axes = [axes]

    for c in range(n_channels):
        channel_feat = feat_np[c]

        # 归一化
        vmin, vmax = np.percentile(channel_feat, [1, 99])
        channel_feat = np.clip((channel_feat - vmin) / (vmax - vmin), 0, 1)

        ax = axes[c]
        ax.imshow(channel_feat, cmap='gray')
        ax.set_title(f'Channel {c}')
        ax.axis('off')

    plt.tight_layout()

    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved base_conv visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_after_addition(feature, save_path=None):
    """
    可视化相加后的输出

    参数:
        feature: 特征张量 [batch, channels, H, W]
        save_path: 保存图像的完整路径
    """
    # 转换为numpy并移除batch维度
    feat_np = feature.squeeze(0).cpu().numpy()
    n_channels = feat_np.shape[0]

    # 创建网格 (通道数 x 1)
    fig, axes = plt.subplots(n_channels, 1, figsize=(6, 6 * n_channels))
    fig.suptitle(f'After Addition (base_conv + x_tag) (Shape: {feature.shape})', fontsize=16)

    if n_channels == 1:
        axes = [axes]

    for c in range(n_channels):
        channel_feat = feat_np[c]

        # 归一化
        vmin, vmax = np.percentile(channel_feat, [1, 99])
        channel_feat = np.clip((channel_feat - vmin) / (vmax - vmin), 0, 1)

        ax = axes[c]
        ax.imshow(channel_feat, cmap='gray')
        ax.set_title(f'Channel {c}')
        ax.axis('off')

    plt.tight_layout()

    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved after_addition visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# ================== 主函数 ==================
def main():
    parser = argparse.ArgumentParser(description='Visualize Wavelet Features')
    parser.add_argument('--image', type=str, default=r"E:\CZY\WBCDD\train2017\0000.jpg", help='Path to input image')
    parser.add_argument('--wt_levels', type=int, default=2, help='Number of wavelet decomposition levels')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save visualizations')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    args = parser.parse_args()

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载图像
    img_tensor = load_image(args.image, args.img_size).to(device)
    print(f"Input image shape: {img_tensor.shape}")

    # 创建模型
    model = WTConv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=5,
        wt_levels=args.wt_levels,
        wt_type='db1'
    ).to(device)
    model.eval()
    print(f"Model created with {args.wt_levels} wavelet levels")

    # 前向传播（存储中间特征）
    with torch.no_grad():
        output = model(img_tensor)
    print(f"Output shape: {output.shape}")

    # 可视化分解过程中的特征
    print("\nVisualizing decomposition features:")
    for level, features in enumerate(model.curr_x_tags):
        print(f"  Decomposition Level {level + 1}: shape={features.shape}")
        save_path = os.path.join(args.output_dir, f'decomposition_level_{level + 1}.png')
        visualize_features(features, level, args.output_dir, title_prefix="Decomposition")

    # 可视化重构过程中的curr_x
    print("\nVisualizing reconstruction curr_x features:")
    for level, curr_x in enumerate(model.recon_curr_x):
        print(f"  Reconstruction Level {level + 1}: shape={curr_x.shape}")
        save_path = os.path.join(args.output_dir, f'recon_curr_x_level_{level + 1}.png')
        visualize_recon_curr_x(curr_x, level, args.output_dir)

    # 可视化重构后的低频分量 next_x_ll
    print("\nVisualizing reconstructed low-frequency (next_x_ll):")
    visualize_next_x_ll(model.next_x_ll_list, args.output_dir)

    # 新增：可视化 base_conv 后的输出和相加后的输出
    if model.base_conv_output is not None:
        save_path = os.path.join(args.output_dir, 'base_conv_output.png')
        print(f"\nVisualizing base_conv output (shape={model.base_conv_output.shape})")
        visualize_base_conv_output(model.base_conv_output, save_path)

    if model.after_addition is not None:
        save_path = os.path.join(args.output_dir, 'after_addition.png')
        print(f"\nVisualizing after addition (shape={model.after_addition.shape})")
        visualize_after_addition(model.after_addition, save_path)


if __name__ == "__main__":
    main()
    img = Image.open(r"E:\CZY\WBCDD\train2017\0000.jpg").convert('L')
    img.save("ori_img.png")