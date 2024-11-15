import matplotlib.pyplot as plt
import os, cv2
import numpy as np
from mono.utils.transform import gray_to_colormap
import shutil
import glob
from mono.utils.running import main_process
import torch
# from html4vision import Col, imagetable

def save_raw_imgs( 
    pred: torch.tensor,  
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str,
    scale: float=200.0, 
    target: torch.tensor=None,
    ):
    """
    Save raw GT, predictions, RGB in the same file.
    """
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), rgb)
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_d.png'), (pred*scale).astype(np.uint16))
    if target is not None:
        cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_gt.png'), (target*scale).astype(np.uint16))
    

def save_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    target: torch.tensor,
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    tb_logger=None
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    rgb, pred_scale, target_scale, pred_color, target_color = get_data_for_log(pred, target, rgb)
    rgb = rgb.transpose((1, 2, 0))
    cat_img = np.concatenate([rgb, pred_color, target_color], axis=0)
    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)

    # save to tensorboard
    if tb_logger is not None:
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)

def save_normal_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    mask,
    filename: str, 
    save_dir: str, 
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    # mean = np.array([123.675, 116.28, 103.53])[np.newaxis, np.newaxis, :]
    # std= np.array([58.395, 57.12, 57.375])[np.newaxis, np.newaxis, :]
    pred = pred.squeeze()

    if pred.size(0) == 3:
        pred = pred.permute(1,2,0)
    else:
        assert 1==0

    pred[:, :, 0] *= -1
    pred[:, :, 1] *= -1
    pred[:, :, 2] *= -1
    pred_normal = vis_surface_normal(pred)
    pred_normal = cv2.resize(pred_normal, (mask.shape[0], mask.shape[1]))
    


    pred_norm_np_withmask = np.concatenate((pred_normal, mask), axis=-1)

    pred_norm_np_withmask = cv2.cvtColor(pred_norm_np_withmask, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'.png'), pred_norm_np_withmask)


def get_data_for_log(pred: torch.tensor, target: torch.tensor, rgb: torch.tensor):
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()

    pred[pred<0] = 0
    target[target<0] = 0
    max_scale = max(pred.max(), target.max())
    pred_scale = (pred/max_scale * 10000).astype(np.uint16)
    target_scale = (target/max_scale * 10000).astype(np.uint16)
    pred_color = gray_to_colormap(pred)
    target_color = gray_to_colormap(target)
    pred_color = cv2.resize(pred_color, (rgb.shape[2], rgb.shape[1]))
    target_color = cv2.resize(target_color, (rgb.shape[2], rgb.shape[1]))

    rgb = ((rgb * std) + mean).astype(np.uint8)
    return rgb, pred_scale, target_scale, pred_color, target_color


# def create_html(name2path, save_path='index.html', size=(256, 384)):
#     # table description
#     cols = []
#     for k, v in name2path.items():
#         col_i =  Col('img', k, v) # specify image content for column
#         cols.append(col_i)
#     # html table generation
#     imagetable(cols, out_file=save_path, imsize=size)

def vis_surface_normal(normal: torch.tensor) -> np.array:
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
        mask (torch.tensor, [h, w]): valid masks
    """
    normal = normal.detach().cpu().numpy().squeeze()

    pred_norm_np = ((normal + 1.0) / 2.0 * 255.0).astype(np.float32)
    return pred_norm_np

