import yaml
import sys
sys.path.append("..")

from utils import compute_gradients, progress_bar, random_point_sampling

import os
import torch
import numpy as np
from tqdm import tqdm

def main(config, root, target):
    num_points_pcd = config['num_points_pcd']
    num_points_forward = config['num_points_forward']
    max_dist = config['udf_max_dist']

    os.makedirs(target, exist_ok=False)

    for file in tqdm(os.listdir(root)):
        if file.endswith(".npz"):
            npz = np.load(root+"/"+file)
            vertices = npz["vertices"]
            triangles = npz["triangles"]
            pcds = torch.from_numpy(npz["pcd"]).unsqueeze(0)
            coords = torch.from_numpy(npz["coords"]).unsqueeze(0)
            gt_udf = torch.from_numpy(npz["labels"]).unsqueeze(0)
            gt_grad = torch.from_numpy(npz["gradients"]).unsqueeze(0)

            pcds = random_point_sampling(pcds, num_points_pcd)[0] #

            gt_udf = gt_udf / max_dist
            gt_udf = 1 - gt_udf
            c_u_g = torch.cat([coords, gt_udf.unsqueeze(-1), gt_grad], dim=-1)

            selected_c_u_g = random_point_sampling(c_u_g, num_points_forward)
            selected_coords = selected_c_u_g[0, :, :3]
            selected_gt_udf = selected_c_u_g[0, :, 3]
            selected_gt_grad = selected_c_u_g[0, :, 4:]

            np.savez(
                target+"/"+file,
                vertices=vertices,
                triangles=triangles,
                pcds=pcds.numpy(),
                selected_coords=selected_coords.numpy(),
                selected_gt_udf=selected_gt_udf.numpy(),
                selected_gt_grad=selected_gt_grad.numpy(),
                )


if __name__ == "__main__":
    if len(sys.argv) > 3:
        run_cfg_file = sys.argv[1]
        root_dataset = sys.argv[2] # 直接由prepreocess_udf.py提取的npy数据folder，未采样
        target_dataset = sys.argv[3]  # 采样后的npy数据folder
        del sys.argv[1]
    else:
        assert 1==0
    with open(run_cfg_file, 'r') as f:
        config = yaml.safe_load(f)
    main(config, root_dataset, target_dataset)