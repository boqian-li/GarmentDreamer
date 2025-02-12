import sys

sys.path.append("..")

import os
import torch
from models.dgcnn import Dgcnn
import yaml
import numpy as np
import tarfile
import io
from tqdm import tqdm


def main(config) -> None:
    total = 800
    progress_bar = tqdm(total=total)

    device = torch.device(config['device'])
    ckpt = torch.load(config['ckpt_path'])
    root = config['root']
    output_dir = config['output_dir']

    latent_size = config['latent_size']
    num_points_pcd = config['num_points_pcd']

    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.to(device)
    encoder.eval()


    for tar in os.listdir(root):
        if tar.endswith(".tar") and not tar.endswith("000002.tar"):
            tar_path = os.path.join(root, tar)
            with tarfile.open(tar_path, 'r') as tar:
                # 遍历tar包中的文件
                for member in tar.getmembers():
                    # 检查是否为文件以及文件名是否以'pcds.npy'结尾
                    if member.isfile() and member.name.endswith('pcds.npy'):
                        # 从tar包中读取文件内容
                        pcds_f = tar.extractfile(member)

                        # 读取npy文件内容
                        np_array = np.load(io.BytesIO(pcds_f.read()))
                        pcds = torch.from_numpy(np_array).unsqueeze(0) # torch.Size((1, N, 3))
                        pcds = pcds.to(device)

                        item_id = member.name.split('.')[0]
                        assert len(pcds.shape) == 3

                        with torch.no_grad():
                            lat = encoder(pcds)
                        # print(lat.shape) (1, 64)

                        os.makedirs(output_dir, exist_ok=True)
                        torch.save(lat.detach().cpu(), os.path.join(output_dir, f"{item_id}.pt"))
                        progress_bar.update(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python export_codes.py <run_cfg_file>")
        exit(1)

    run_cfg_file = sys.argv[1]
    with open(run_cfg_file, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
