import torch
from typing import Dict, List
import torch.nn.functional as F

from deformer.core import View

global_execution_count = 0

def hole_mask_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], gbuffers_rf: List[Dict[str, torch.Tensor]], loss_function = torch.nn.MSELoss()):
    """
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """
    device = views[0].normal.device
    loss = 0.0
    for view, gbuffer, gbuffer_rf in zip(views, gbuffers, gbuffers_rf):
        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
        normal = gbuffer["normal"] @ view.camera.R.T @ R.T  # range(-1, 1)  # global转camera view
        normal_rf = gbuffer_rf['normal'] @ view.camera.R.T @ R.T

        # with torch.no_grad(): # TODO： necessary to add `with torch.no_grad()`???
        position = gbuffer['position']
        view_direction = F.normalize(view.camera.center - position, dim=-1) # global view_direction, (1024, 1024, 3)
        view_direction = view_direction @ view.camera.R.T @ R.T # TODO：是否需要* -1？
        view_direction *= -1

        position_rf = gbuffer_rf['position']
        view_direction_rf = F.normalize(view.camera.center - position_rf, dim=-1) # global view_direction, (1024, 1024, 3)
        view_direction_rf = view_direction_rf @ view.camera.R.T @ R.T # TODO：是否需要* -1？
        view_direction_rf *= -1

        cosines = F.cosine_similarity(view_direction, normal, dim=-1, eps=1e-6) #(1024, 1024) 
        cosines_rf = F.cosine_similarity(view_direction_rf, normal_rf, dim=-1, eps=1e-6) #(1024, 1024) 

        cosines.data.masked_fill_(cosines < 0, -1)
        cosines.data.masked_fill_(cosines >= 0, 1)
        cosines_rf.data.masked_fill_(cosines_rf < 0, -1)
        cosines_rf.data.masked_fill_(cosines_rf >= 0, 1)

        sign = cosines
        sign_rf = cosines_rf

        mask =(gbuffer["mask"].squeeze() > 0) & (gbuffer_rf['mask'].squeeze() > 0)
        loss += loss_function(sign[mask], sign_rf[mask])

    return loss / len(views)



def mask_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], loss_function = torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    
    for view, gbuffer in zip(views, gbuffers):
        loss += loss_function(view.mask, gbuffer["mask"])
    return loss / len(views)