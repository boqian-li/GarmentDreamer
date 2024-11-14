import numpy as np
from pathlib import Path
import trimesh

from deformer.core import Mesh, View
from deformer.tools import decimate_mesh
import os

def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)
    return Mesh(vertices, indices, device)

def write_mesh(path, mesh, post_process=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy()
    indices = mesh.indices.numpy() if mesh.indices is not None else None
    mesh_ = trimesh.Trimesh(vertices=vertices, faces=indices, process=False)
    if post_process:
        # 0. 由于前序步骤deformation之后mesh是横过来的，所以先调整到竖过来：
        angle = np.radians(-90)
        axis = [1, 0, 0]
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
        mesh_.apply_transform(rotation_matrix)


        # simplify source mesh, obtain new mesh
        new_vertices, new_faces = decimate_mesh(mesh_.vertices, mesh_.faces, target=40000, remesh=True, preserveboundary=True, optimalplacement=False)
        mesh_ = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    mesh_.export(path)

def read_views(directory_, mode, device):
    directory_ = Path(directory_)

    directory_rgb = directory_.joinpath("gs_rendered_rgba")
    directory_normal = directory_.joinpath("estimated_normals")

    image_paths_rgb = sorted([path for path in directory_rgb.iterdir() if (path.is_file() and path.suffix == '.png')], key = lambda x : int(x.stem))
    image_paths_normal = sorted([path for path in directory_normal.iterdir() if (path.is_file() and path.suffix == '.png')], key = lambda x : int(x.stem))
    
    views = []
    for image_path_normal, image_path_rgb in zip(image_paths_normal, image_paths_rgb):
        views.append(View.load(image_path_normal, image_path_rgb, mode, device))
    print("Found {:d} views".format(len(views)))

    return views