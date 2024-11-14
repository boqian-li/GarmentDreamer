import torch
from typing import Callable, Dict, List
import torch.nn.functional as F

from deformer.core import View


def normal_map_loss_enhanced(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], epsilon=-0.1):
    """ Compute the shading term as the mean difference between the original images and the rendered images from a shader.
    
    Args:
        views (List[View]): Views with color images and masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with 'mask', 'position', and 'normal' channels
        loss_function (Callable): Function for comparing the images or generally a set of pixels
    """
    device = views[0].normal.device
    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):
        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
        normal = gbuffer["normal"] @ view.camera.R.T @ R.T  # range(-1, 1)  # global转camera view
        target_normal = view.normal * 2 - 1  # range(-1, 1) camera view
        normal_errors = 1 - F.cosine_similarity(target_normal, normal, dim=-1)  # (1024, 1024)

        with torch.no_grad(): # TODO： necessary to add `with torch.no_grad()`???
            position = gbuffer['position']
            view_direction = F.normalize(view.camera.center - position, dim=-1) # global view_direction, (1024, 1024, 3)
            view_direction = view_direction @ view.camera.R.T @ R.T # TODO：是否需要* -1？
            view_direction *= -1

            cosines_target = F.cosine_similarity(view_direction, target_normal, dim=-1, eps=1e-6) #(1024, 1024)
            cosines_target[cosines_target > epsilon] = 0
            cosines_view = F.cosine_similarity(view_direction, normal, dim=-1, eps=1e-6) #(1024, 1024) 

        normal_errors = normal_errors * torch.exp(cosines_target.abs()) / torch.exp(cosines_target.abs()).sum() #(1024, 1024)
        

        # Get valid area
        mask = ((view.mask.squeeze() > 0) & (gbuffer["mask"].squeeze() > 0)) & (cosines_view <= 0) & (cosines_target <= epsilon)
        #=====
        # import numpy as np
        # import matplotlib.pyplot as plt

        # array = np.zeros_like(cosines_view.detach().cpu().numpy())
        # # array[(view.mask.squeeze().detach().cpu().numpy() > 0) & (cosines_view.detach().cpu().numpy() <= 0)] = 1
        # array[mask.detach().cpu().numpy()==1] = 1
        # array_gray = array

        # array_normal = (gbuffer["normal"] @ view.camera.R.T @ R.T + 1).detach().cpu().numpy() / 2
        # array_normal_target = ((target_normal + 1) / 2).detach().cpu().numpy()

        # # 创建两个子图
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # # 显示灰度图像
        # axes[0].imshow(array_gray, cmap='gray')
        # axes[0].set_title('Gray Image')

        # # 显示彩色图像
        # axes[1].imshow(array_normal * mask.detach().cpu().unsqueeze(-1).numpy())
        # axes[1].set_title('Normal Image')

        # axes[2].imshow(array_normal_target * mask.detach().unsqueeze(-1).cpu().numpy())
        # axes[2].set_title('Estimated Normal Image')
        # plt.show()
        # =====

        loss += normal_errors[mask].sum()


    return loss / len(views)

def normal_map_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], loss_function=torch.nn.L1Loss()):
    """ Compute the shading term as the mean difference between the original images and the rendered images from a shader.
    
    Args:
        views (List[View]): Views with color images and masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with 'mask', 'position', and 'normal' channels
        loss_function (Callable): Function for comparing the images or generally a set of pixels
    """

    loss = 0
    for view, gbuffer in zip(views, gbuffers):
        # Get valid area
        mask = ((view.mask > 0) & (gbuffer["mask"] > 0)).squeeze()

        target = view.normal[mask] # torch.Size(209529, 3)
        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=mask.device, dtype=torch.float32)
        normal = 0.5 * (gbuffer["normal"] @ view.camera.R.T @ R.T + 1)[mask]

        loss += loss_function(normal, target)

    return loss / len(views)