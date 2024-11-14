import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import json
import matplotlib.pyplot as plt
from netf.view_core.camera import Camera


def read_views(directory_, device):
    directory_rgb = Path(directory_)
    image_paths_rgb = sorted([path for path in directory_rgb.iterdir() if (path.is_file() and path.suffix == '.png')], key = lambda x : int(x.stem))
    
    views = []
    for image_path_rgb in image_paths_rgb:
        views.append(View.load(image_path_rgb, device))
    print("Found {:d} views".format(len(views)))

    return views


class View:
    def __init__(self, mask, rgb, camera, device='cpu'):
        self.mask = mask.to(device)
        self.rgb = rgb.to(device)
        self.camera = camera
        self.device = device

    @classmethod
    def load(cls, image_path_rgb, device='cpu'):    

        image_path_rgb = Path(image_path_rgb)
        with open(str(image_path_rgb.parent.parent) + '/cameras.json') as f:
            unsorted_camera_transforms = json.load(f)
        camera_transforms = sorted(unsorted_camera_transforms.copy(), key = lambda x : x['id'])
        assert (image_path_rgb.stem == camera_transforms[int(image_path_rgb.stem)]['img_name'])

        info = camera_transforms[int(image_path_rgb.stem)]
        fx = info['fx']
        fy = info['fy']
        width = info['width']
        height = info['height']
        position = np.array(info['position'])
        rotation = np.array(info['rotation'])


        C2W = np.zeros((4,4))
        C2W[:3, :3] = rotation
        C2W[:3, 3] = position
        C2W[3,3] = 1

        K = np.array([[fx,   0,    width/2],
                        [0,    fy,   height/2],
                        [0,    0,    1]])

        camera = Camera(K, C2W)
        
        # Load the rgb
        raw_image = cv2.imread(str(image_path_rgb), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2RGBA)
        img = (np.array(image).astype(np.float32) / 255.0).copy()     # (H, W, RGBA) range(0-1)

        img = torch.FloatTensor(img)
        mask = img[:, :, -1:]
        rgb = img[:, :, :-1]

        return cls(mask, rgb, camera, device=device)
