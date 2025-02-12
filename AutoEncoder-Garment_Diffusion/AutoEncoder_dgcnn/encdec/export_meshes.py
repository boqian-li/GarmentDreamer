import sys

sys.path.append("..")

import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from meshudf.meshudf import get_mesh_from_udf
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
import yaml
import numpy as np
from utils import get_o3d_mesh_from_tensors
import open3d as o3d
import tarfile
import io
from tqdm import tqdm


def create_mesh_from_arrays(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def compute_chamfer_distance(mesh1, mesh2, num_points):
    pcd1 = mesh1.sample_points_uniformly(number_of_points=num_points)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=num_points)
    distances_forward = pcd1.compute_point_cloud_distance(pcd2)
    distances_backward = pcd2.compute_point_cloud_distance(pcd1)
    chamfer_distance = (
        np.mean(np.asarray(distances_forward)) + np.mean(np.asarray(distances_backward))
    ) / 2
    return chamfer_distance * 100


def main(config, mode) -> None:
    total = 800
    progress_bar = tqdm(total=total)

    device = torch.device(config['device'])
    ckpt = torch.load(config['ckpt_path'])
    root = config['root']
    output_dir = config['output_dir']

    latent_size = config['latent_size']
    num_points_pcd = config['num_points_pcd']
    udf_max_dist = config['udf_max_dist']
    charmfer_distance_threshold = config['charmfer_distance_threshold']

    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.to(device)
    encoder.eval()

    coords_encoder = CoordsEncoder()

    decoder = CbnDecoder(
            coords_encoder.out_dim,
            latent_size,
            config['decoder']['hidden_dim'],
            config['decoder']['num_hidden_layers']
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.to(device)
    decoder.eval()

    if mode == 0: 
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
                            vertices_f = tar.extractfile(member.name[:member.name.find('.')] + '.vertices.npy')
                            triangles_f = tar.extractfile(member.name[:member.name.find('.')] + '.triangles.npy')

                            # 读取npy文件内容
                            np_array = np.load(io.BytesIO(pcds_f.read()))
                            pcds = torch.from_numpy(np_array).unsqueeze(0) # torch.Size((1, N, 3))
                            pcds = pcds.to(device)

                            vertices = np.load(io.BytesIO(vertices_f.read()))
                            triangles = np.load(io.BytesIO(triangles_f.read()))

                            item_id = member.name.split('.')[0]
                            assert len(pcds.shape) == 3

                            with torch.no_grad():
                                lat = encoder(pcds)

                            def udf_func(c: Tensor) -> Tensor:
                                c = coords_encoder.encode(c.unsqueeze(0))
                                p = decoder(c, lat).squeeze(0)
                                p = torch.sigmoid(p)
                                p = (1 - p) * udf_max_dist
                                return p
                            
                            try:
                                v, t = get_mesh_from_udf(
                                    udf_func,
                                    coords_range=(-1, 1),
                                    max_dist=udf_max_dist,
                                    N=256,
                                    max_batch=2**16,
                                    differentiable=False,
                                )   ## possible RuntimeError

                                pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)
                                gt_mesh_o3d = create_mesh_from_arrays(vertices, triangles)
                                chamfer_distance = compute_chamfer_distance(gt_mesh_o3d, pred_mesh_o3d, config['num_points_compute'])   ## possible IndexError
                                print(f'{member.name}:  {chamfer_distance}')

                                mesh_path = os.path.join(output_dir, f"{item_id}_{chamfer_distance}.obj")
                                os.makedirs(output_dir, exist_ok=True)
                                o3d.io.write_triangle_mesh(mesh_path, pred_mesh_o3d)
                                if chamfer_distance > charmfer_distance_threshold:
                                    with open(os.path.join(output_dir, 'bad.txt'), 'a') as f:
                                        f.write(member.name)
                                        f.write(' :    ')
                                        f.write(str(chamfer_distance))
                                        f.write('\r\n')

                            except (RuntimeError, IndexError) as e:
                                print(e)
                                print(member.name)
                                with open(os.path.join(output_dir, 'fails.txt'), 'a') as f:
                                    f.write(member.name)
                                    f.write(' :    ')
                                    f.write(str(e))
                                    f.write('\r\n')

                            finally:
                                progress_bar.update(1)

    elif mode == 1:
        for tensor in os.listdir(root):
            if tensor.endswith(".pt"):
                tensor_path = os.path.join(root, tensor)
                lat = torch.load(tensor_path)
                lat = lat.to(device)
                assert len(lat) == 2

                item_id = tensor.split('.')[0]

                def udf_func(c: Tensor) -> Tensor:
                    c = coords_encoder.encode(c.unsqueeze(0))
                    p = decoder(c, lat).squeeze(0)
                    p = torch.sigmoid(p)
                    p = (1 - p) * udf_max_dist
                    return p
                
                try:
                    v, t = get_mesh_from_udf(
                        udf_func,
                        coords_range=(-1, 1),
                        max_dist=udf_max_dist,
                        N=256,
                        max_batch=2**16,
                        differentiable=False,
                    )   ## possible runtime error
                
                    pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)

                    mesh_path = os.path.join(output_dir, f"{item_id}.obj")
                    os.makedirs(output_dir, exist_ok=True)
                    o3d.io.write_triangle_mesh(mesh_path, pred_mesh_o3d)

                except RuntimeError as e:
                    print(e)
                    print(member.name)
                    with open(os.path.join(output_dir, 'fails.txt'), 'a') as f:
                        f.write(member.name)
                        f.write(' :    ')
                        f.write(str(e))
                        f.write('\r\n')

                finally:
                    progress_bar.update(1)
                    
    elif mode == 2:
        lat1 = torch.load(os.path.join(root, 'Dress_336.pt')).to(device)
        lat2 = torch.load(os.path.join(root, 'tee_51.pt')).to(device)

        for a in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            lat = lat1 * a + (1 - a) * lat2
        
            item_id = f'{a}_'

            def udf_func(c: Tensor) -> Tensor:
                c = coords_encoder.encode(c.unsqueeze(0))
                p = decoder(c, lat).squeeze(0)
                p = torch.sigmoid(p)
                p = (1 - p) * udf_max_dist
                return p
            
            try:
                v, t = get_mesh_from_udf(
                    udf_func,
                    coords_range=(-1, 1),
                    max_dist=udf_max_dist,
                    N=256,
                    max_batch=2**16,
                    differentiable=False,
                )   ## possible runtime error
            
                pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)

                mesh_path = os.path.join(output_dir, f"{item_id}.obj")
                os.makedirs(output_dir, exist_ok=True)
                o3d.io.write_triangle_mesh(mesh_path, pred_mesh_o3d)

            except RuntimeError as e:
                print(e)
                print(member.name)
                with open(os.path.join(output_dir, 'fails.txt'), 'a') as f:
                    f.write(member.name)
                    f.write(' :    ')
                    f.write(str(e))
                    f.write('\r\n')

            finally:
                progress_bar.update(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_meshes.py <run_cfg_file>")
        exit(1)

    run_cfg_file = sys.argv[1]

    # mode == 0: from pre-resampled item_id.pcds.npy in tar
    # mode == 1: from latents
    mode = int(sys.argv[2]) 

    with open(run_cfg_file, 'r') as f:
        config = yaml.safe_load(f)

    main(config, mode)
