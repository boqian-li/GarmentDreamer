import torch
import torch.nn.functional as F
import logging
import os
import os.path as osp
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, save_raw_imgs, save_normal_val_imgs
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud

def to_cuda(data: dict):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>=1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data

def align_scale(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    else:
        scale = 1
    pred_scaled = pred * scale
    return pred_scaled, scale

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    target_mask = target[mask].cpu().numpy()
    pred_mask = pred[mask].cpu().numpy()
    if torch.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale

def align_scale_shift_numpy(pred: np.array, target: np.array):
    mask = target > 0
    target_mask = target[mask]
    pred_mask = pred[mask]
    if np.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale


def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map. 
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y = np.arctan(y_center / (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model

def resize_for_input(image, output_shape, intrinsic, canonical_shape, to_canonical_ratio):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    """
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # padding
    image = cv2.copyMakeBorder(
        image, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=padding)
    
    # Resize, adjust principle point
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio

    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(
        cam_model, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=-1)

    pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor=1/to_scale_ratio
    return image, cam_model, pad, label_scale_factor


def get_prediction(
    model: torch.nn.Module,
    input: torch.tensor,
    cam_model: torch.tensor,
    pad_info: torch.tensor,
    scale_info: torch.tensor,
    gt_depth: torch.tensor,
    normalize_scale: float,
    ori_shape: list=[],
):

    data = dict(
        input=input,
        cam_model=cam_model,
    )
    pred_depth, confidence, output_dict = model.module.inference(data)
    pred_depth = pred_depth
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    if gt_depth is not None:
        resize_shape = gt_depth.shape
    elif ori_shape != []:
        resize_shape = ori_shape
    else:
        resize_shape = pred_depth.shape

    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], resize_shape, mode='bilinear').squeeze() # to original size
    pred_depth = pred_depth * normalize_scale / scale_info
    if gt_depth is not None:
        pred_depth_scale, scale = align_scale(pred_depth, gt_depth)
    else:
        pred_depth_scale = None
        scale = None

    return pred_depth, pred_depth_scale, scale, output_dict

def transform_test_data_scalecano(rgb, intrinsic, data_basic):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    # canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # BGR to RGB
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ori_h, ori_w, _ = rgb.shape
    # ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    # canonical_focal = canonical_space['focal_length']
    # cano_label_scale_ratio = canonical_focal / ori_focal

    canonical_intrinsic = [
        intrinsic[0],
        intrinsic[1],
        intrinsic[2],
        intrinsic[3],
    ]

    # resize
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0)

    # label scale factor
    # label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio
    label_scale_factor = 1.0 * resize_label_scale_ratio

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    
    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].cuda()
    cam_model_stacks = [
        torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor

def do_scalecano_test_with_custom_data(
    model: torch.nn.Module,
    cfg: dict,
    test_data: list,
    logger: logging.RootLogger,
    is_distributed: bool = True,
    local_rank: int = 0,
):  
    
    show_dir = cfg.show_dir
    save_interval = 1
    save_imgs_dir = show_dir + '/estimated_normals'
    os.makedirs(save_imgs_dir, exist_ok=True)
    # save_pcd_dir = show_dir + '/pcd'
    # os.makedirs(save_pcd_dir, exist_ok=True)

    normalize_scale = cfg.data_basic.depth_range[1]
    dam = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
    dam_median = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
    dam_global = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
    
    for i, an in tqdm(enumerate(test_data), total=len(test_data)):
        # print(an['rgb'])
        image = cv2.imread(an['rgb'], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        img = np.array(image).astype(np.float32)       # (H, W, RGBA) range(0-255)

        mask = img[:, :, -1:]
        rgb_origin = img[:, :, :-1]

        intrinsic = an['intrinsic'] # None
        if intrinsic is None:
            # intrinsic = [1406.708352764542, 1406.708352764542, rgb_origin.shape[1]/2, rgb_origin.shape[0]/2]
            intrinsic = [731.2116911560281, 731.2116911560281, rgb_origin.shape[1]/2, rgb_origin.shape[0]/2]
            # print(intrinsic)
        rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)

        pred_depth, pred_depth_scale, scale, output = get_prediction(
            model = model,
            input = rgb_input,
            cam_model = cam_models_stacks,
            pad_info = pad,
            scale_info = label_scale_factor,
            gt_depth = None,
            normalize_scale = normalize_scale,
            ori_shape=[rgb_origin.shape[0], rgb_origin.shape[1]],
        )
    
        if "normal_out_list" in output.keys():
            
            normal_out_list = output['normal_out_list'] 
            pred_normal = normal_out_list[0][:, :3, :, :] # (B, 3, H, W)
            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            # print(pred_normal.shape) # torch.Size([1, 3, 1064, 1064])

            if i % save_interval == 0:
                save_normal_val_imgs(iter, 
                                    pred_normal, 
                                    mask, 
                                    an['filename'], 
                                    save_imgs_dir,
                                    )


    # #if gt_depth_flag:
    # if False:
    #     eval_error = dam.get_metrics()
    #     print('w/o match :', eval_error)

    #     eval_error_median = dam_median.get_metrics()
    #     print('median match :', eval_error_median)

    #     eval_error_global = dam_global.get_metrics()
    #     print('global match :', eval_error_global)
    # else:
    #     print('missing gt_depth, only save visualizations...')
