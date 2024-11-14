# import torch
# from typing import Dict, List
# from src.core import View

# def mapping_func(x):
#     y = x 
#     return y

# def depth_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], loss_function = torch.nn.MSELoss()):
#     """ Compute the depth loss between estimated depth map and rendered depth map from mesh.

#     Args:
#         views (List[View]): Views with masks
#         gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
#         loss_function (Callable): Function for comparing the masks or generally a set of pixels
#     """
#     loss = 0
#     for view, gbuffer in zip(views, gbuffers):
#         # Get valid area
#         mask = ((view.mask > 0) & (gbuffer["mask"] > 0))

        
#         estimated_depth = view.depth
#         estimated_depth = (estimated_depth - estimated_depth[view.mask != 0].min()) / (estimated_depth[view.mask != 0].max() - estimated_depth[view.mask != 0].min())
#         estimated_depth = mapping_func(estimated_depth)
#         target = estimated_depth[mask]
        

#         depth = gbuffer["depth"]
#         depth = (depth - depth[gbuffer["mask"] != 0].min()) / (depth[gbuffer["mask"] != 0].max() - depth[gbuffer["mask"] != 0].min())
#         depth = -1 * depth + 1
#         depth = depth[mask]

#         loss += loss_function(depth, target)

#     return loss / len(views)
