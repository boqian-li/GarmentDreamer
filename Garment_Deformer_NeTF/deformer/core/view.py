import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import json
import matplotlib.pyplot as plt

from deformer.core import Camera

class View:
    """ A View is a combination of camera and image(s).

    Args:
        # color (tensor): RGB color image (WxHx3)
        normal (tensor): normal map (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, normal, mask, rgb, camera, device='cpu'):
        self.normal = normal.to(device)
        self.mask = mask.to(device)
        self.rgb = rgb.to(device)
        self.camera = camera.to(device)
        self.device = device

    @classmethod
    def load(cls, image_path_normal, image_path_rgb, mode=0, device='cpu'):
        """ Load a view from a given image path.

        The paths of the camera matrices are deduced from the image path. 
        Given an image path `path/to/directory/foo.png`, the paths to the camera matrices
        in numpy readable text format are assumed to be `path/to/directory/foo_k.txt`, 
        `path/to/directory/foo_r.txt`, and `path/to/directory/foo_t.txt`.

        Args:
            image_path (Union[Path, str]): Path to the image file that contains the color and optionally the mask
            device (torch.device): Device where the images and camera are stored
        """       

        image_path_normal = Path(image_path_normal)
        image_path_rgb = Path(image_path_rgb)

        # Load the camera
        if mode == 0:
            assert 1==0
            # K = np.loadtxt(image_path.parent / (image_path.stem + "_k.txt"))
            # R = np.loadtxt(image_path.parent / (image_path.stem + "_r.txt"))
            # t = np.loadtxt(image_path.parent / (image_path.stem + "_t.txt"))
            # camera = Camera(K, R, t)

        elif mode == 1:
            with open(str(image_path_normal.parent.parent) + '/cameras.json') as f:
                unsorted_camera_transforms = json.load(f)
            camera_transforms = sorted(unsorted_camera_transforms.copy(), key = lambda x : x['id'])
            assert (image_path_normal.stem == camera_transforms[int(image_path_normal.stem)]['img_name']) and (image_path_normal.stem == image_path_rgb.stem)

            info = camera_transforms[int(image_path_normal.stem)]
            fx = info['fx']
            fy = info['fy']
            width = info['width']
            height = info['height']
            position = np.array(info['position'])
            rotation = np.array(info['rotation'])
            rotation[:, 0] *= -1

            position[1] = - position[1]
            rotation[1, 0] = - rotation[1, 0]
            rotation[1, 2] = - rotation[1, 2]
            rotation[:, 1] = np.cross(rotation[:, 2], rotation[:, 0])
            rotation[:, 1] = rotation[:, 1] / np.linalg.norm(rotation[:, 1])
            rotation[:, 2] *= -1


            C2W = np.zeros((4,4))
            C2W[:3, :3] = rotation
            C2W[:3, 3] = position
            C2W[3,3] = 1
            W2C = np.linalg.inv(C2W)
            
            R = W2C[:3, :3] 
            t = W2C[:3, 3]
            K = np.array([[fx,   0,    width/2],
                          [0,    fy,   height/2],
                          [0,    0,    1]])

            camera = Camera(K, R, t)

        else:
            assert 1 == 0, "Mode must in int[0, 1]"
        
        # Load the normal
        raw_image = cv2.imread(str(image_path_normal), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2RGBA)
        img = (np.array(image).astype(np.float32) / 255.0).copy()     # (H, W, RGBA) range(0-1)

        img = torch.FloatTensor(img)
        mask = img[:, :, -1:]
        normal = img[:, :, :-1]
        normal = normal * 2 - 1
        normal[:, :, 1] *= -1
        normal = (normal + 1) / 2
        
        # Load the rgb
        raw_image = cv2.imread(str(image_path_rgb), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2RGBA)
        img = (np.array(image).astype(np.float32) / 255.0).copy()     # (H, W, RGBA) range(0-1)

        img = torch.FloatTensor(img)
        # mask = img[:, :, -1:]
        rgb = img[:, :, :-1]

        return cls(normal, mask, rgb, camera, device=device)

    # def to_world(normal_cam, view_angle,):
    #     '''
    #         normal_cam : [4, h, w] tensor. normal map in camera view 
    #         view_angle: yaw angle of camera view in degree
    #     '''
    #     view_angle = torch.deg2rad(torch.tensor(view_angle))
    #     rot = torch.tensor([[torch.cos(view_angle), 0, torch.sin(view_angle)],
    #                         [0, 1, 0],
    #                         [-torch.sin(view_angle), 0, torch.cos(view_angle)]], dtype=normal_cam.dtype, device=normal_cam.device)
    #     mask = normal_cam[3, :, :] >= 0
    #     normal_world = normal_cam.clone()
    #     normal_world[:3, mask] = torch.einsum('ij, jk->ik', rot.T, normal_cam[:3, mask])

    #     return normal_world
    
    def to(self, device: str = "cpu"):
        self.normal = self.normal.to(device)
        self.mask = self.mask.to(device)
        self.camera = self.camera.to(device)
        self.device = device
        return self

    @property
    def resolution(self):
        return (self.normal.shape[0], self.normal.shape[1])
    
    # def scale(self, inverse_factor):
    #     """ Scale the view by a factor.
        
    #     This operation is NOT differentiable in the current state as 
    #     we are using opencv.

    #     Args:
    #         inverse_factor (float): Inverse of the scale factor (e.g. to halve the image size, pass `2`)
    #     """
        
    #     scaled_height = self.depth.shape[0] // inverse_factor
    #     scaled_width = self.depth.shape[1] // inverse_factor

    #     scale_x = scaled_width / self.depth.shape[1]
    #     scale_y = scaled_height / self.depth.shape[0]
        
    #     self.depth = torch.FloatTensor(cv2.resize(self.depth.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)
    #     self.mask = torch.FloatTensor(cv2.resize(self.mask.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)).to(self.device)
    #     self.mask = self.mask.unsqueeze(-1) # Make sure the mask is HxWx1

    #     self.camera.K = torch.FloatTensor(np.diag([scale_x, scale_y, 1])).to(self.device) @ self.camera.K  
    
    def transform(self, A, A_inv=None):
        """ Transform the view pose with an affine mapping.

        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)
        
        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(self.device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(self.device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv.
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix, 
        # and thus make sure that R is a rotation matrix.
        R = self.camera.R @ A_inv[:3, :3]
        t = self.camera.R @ A_inv[:3, 3] + self.camera.t
        P = torch.zeros((3, 4), device=self.device)
        P[:3, :3] = self.camera.K @ R
        P[:3, 3] = self.camera.K @ t
        K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())
        c = c[:3, 0] / c[3]
        t = - R @ c

        # ensure unique scaling of K matrix
        K = K / K[2,2]
        
        self.camera.K = torch.from_numpy(K).to(self.device)
        self.camera.R = torch.from_numpy(R).to(self.device)
        self.camera.t = torch.from_numpy(t).to(self.device)
        
    def project(self, points, depth_as_distance=False):
        """ Project points to the view's image plane according to the equation x = K*(R*X + t).

        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and 
                                   the points' depth relative to the view (A x ... x Z x 3).
        """

        # 
        points_c = points @ torch.transpose(self.camera.R, 0, 1) + self.camera.t
        pixels = points_c @ torch.transpose(self.camera.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)
        return torch.cat([pixels, depths], dim=-1)