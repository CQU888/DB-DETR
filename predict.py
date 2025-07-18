import skimage
import math
import numpy as np
import cv2
from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output
# import cmapy
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from torch.nn.functional import dropout, linear, softmax
import torch.nn.functional as F
from modules import build_model
from pathlib import Path
from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
from util.cam_module import CustomGradCAM
torch.set_grad_enabled(False)
import matplotlib


# 在main函数中添加自定义目标类
class DetrBoxScoreTarget:
    def __init__(self, keep_index):
        self.keep_index = keep_index

    def __call__(self, model_outputs):
        # 确保处理的是原始模型输出
        print(type(model_outputs))
        if not isinstance(model_outputs, dict):
            if isinstance(model_outputs, tuple) and len(model_outputs) > 0:
                model_outputs = model_outputs[0]
            else:
                print(model_outputs)
                raise ValueError(f"Unexpected model output type: {type(model_outputs)}")

        scores = model_outputs['pred_logits'].softmax(-1)[0, self.keep_index, :-1]
        return scores.max()

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_fpn_names', default=['top_feature_proj', 'fpn_proj_list',
                                                   'bottomup_conv1', 'align_modules', 'input_proj_list',
                                                   'upbottom_conv', 'bottomup_lateral_conv', 'up_conv',
                                                   'cbam_attention'], type=str, nargs='+')
    parser.add_argument('--lr_fpn_mult', default=1.5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--with_fpn', default=False, action='store_true')
    parser.add_argument('--method_fpn', type=str, choices=["fpn", "bifpn", "pafpn", "fapn", "wbcfpn"])

    # Model parameters
    parser.add_argument('--train_fpn', default=False, action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='E://CZY/WBCDD', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='predict',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume',
                        default='output/checkpoint0095.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def get_reference_points(spatial_shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # 0.5是对应到特征点的中心
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        # 进行归一化
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        # (bs, H_*W_, 2)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    # (bs, h_1 * w_1 + ... + h_4 * w_4, 2) * (bs, 1, n_level, 2)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


def main(args):
    dataset_name = "ZJHospital"
    if dataset_name == "LISC":
        Classes = ['N/A', "Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
        COLORS = [[0.000, 0.447, 0.741], [0.301, 0.745, 0.933], [0.494, 0.184, 0.556],
                  [0.466, 0.674, 0.188], [0.850, 0.325, 0.098], [0.850, 0.325, 0.098]]
    elif dataset_name == "BCCD":
        Classes = ['N/A', 'RBC', 'WBC', 'Platelets']
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125]]
    elif dataset_name == "ZJHospital":
        Classes = ['N/A', 'Neutrophil', 'Monocyte', 'Eosinophil', 'Lymphocyte', 'Basophil']
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    model_without_ddp.eval()

    # 2. 选择目标层（ResNet最后一个卷积层）
    target_layer1 = model_without_ddp.backbone[0].backbone1.layer4[-1].conv3

    # 创建GradCAM对象
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        use_cuda=(args.device == 'cuda')
    )

    # if dataset_name == "LISC":
    #     names = ["Baso_35","eosi_3", "lymp_3", "mono_10", "mono_22", "neut_38"]
    #     img_end = "bmp"
    # elif dataset_name == "ZJHospital":
    names = ['0000', '04-0005',  '0081']


    img_end = "jpg"
    for name in names:
        im = Image.open("E://CZY/WBCDD/train2017/{}.{}".format(name, img_end))
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if
                               not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
        img = transform(im).unsqueeze(0).to(device)

        outputs = model_without_ddp(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9
        keep_indices = keep.nonzero().squeeze(1).cpu().tolist()

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)

        # 为每个检测到的目标生成热力图
        for idx in keep_indices:
            # 创建自定义目标
            targets = [DetrBoxScoreTarget(idx)]

            # 生成热力图
            with torch.enable_grad():
                grayscale_cam = cam(
                    input_tensor=img,
                    targets=targets,
                    # aug_smooth=True,
                    # eigen_smooth=True
                )

            # 转换原始图像用于覆盖
            rgb_img = np.array(im.resize((w, h))) / 255.0  # 调整尺寸匹配模型输入
            rgb_img = np.float32(rgb_img)

            # 创建热力图覆盖
            cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

            # 保存结果
            plt.imshow(cam_image)
            plt.savefig(f"predict/{name}_cam_{idx}.jpg")

        for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero().cpu().numpy(), bboxes_scaled.cpu().numpy()):
            cell_num = torch.argmax(probas[idx], -1)
            cell_name = Classes[cell_num]
            b = plt.Rectangle(
                xy=(xmin, ymin), width=xmax - xmin, height=ymax - ymin,
                fill=False, edgecolor=COLORS[cell_num], linewidth=2)
            plt.gca().add_patch(b)
            plt.text(xmin,
                     ymin, s='%s %.2f' % (cell_name, probas[idx, cell_num]),
                     color='black',
                     verticalalignment='bottom', size=9, family="serif",
                     bbox={'color': COLORS[cell_num], 'pad': 0})
            # plt.scatter(p1[0], p1[1], marker="+", c="k")
        plt.xticks([])
        plt.yticks([])
        plt.savefig("predict/{}.jpg".format(name), bbox_inches='tight',
                    dpi=400)
        # plt.savefig("test_{}.jpg".format(index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
