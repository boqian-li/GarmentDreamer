
# input: mesh with messy vertices and vertex color

import trimesh
# import xatlas
import pymeshlab as pml
import argparse
import os
import numpy as np
from PIL import Image


def decimate_mesh(verts, faces, target, remesh=False, preserveboundary=False, optimalplacement=True):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # filters
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), preserveboundary=preserveboundary, optimalplacement=optimalplacement)

    if remesh:
        print("Smoothing!")
        ms.apply_coord_taubin_smoothing()
        # ms.meshing_isotropic_explicit_remeshing()

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh', type=str, required=True)
    args = parser.parse_args()

    print("Attention: Now the script is used for rotating and simpifying mesh, between Garment_NDS and Garment_finer! \n \
          only vertices and faces will be in the output mesh.")
    source_mesh = trimesh.load_mesh(args.input_mesh)
    # 0. 由于前序步骤NDS之后mesh是横过来的，所以先调整到竖过来：
    # 定义旋转角度（弧度）
    angle = np.radians(-90)
    # 定义旋转轴（这里是z轴）
    axis = [1, 0, 0]
    # 创建旋转矩阵
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
    # 应用旋转变换
    source_mesh.apply_transform(rotation_matrix)
    # source_mesh.export(args.input_mesh[:-4] + "_rotate" + args.input_mesh[-4:])
    # args.input_mesh = args.input_mesh[:-4] + "_rotate" + args.input_mesh[-4:]
    # source_mesh = trimesh.load_mesh(args.input_mesh)

    # tmp_path = os.path.join("./tmp", args.input_mesh.split('/')[-1][:-4], args.input_mesh.split('/')[-1])
    output_path = os.path.join("./output", args.input_mesh.split('/')[-1][:-4], args.input_mesh.split('/')[-1])

    # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # image = Image.open("samples/textures/color_gradient.png")
    # print(" Source Mesh Loaded!")


    # 1. simplify source mesh, obtain new mesh
    new_vertices, new_faces = decimate_mesh(source_mesh.vertices, source_mesh.faces, target=40000, remesh=True, preserveboundary=True, optimalplacement=False)
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    new_mesh.export(output_path)

    # # 2. generate uv map for new mesh
    # new_mesh_with_uv = new_mesh.unwrap(image) # use xatlas for unwrapping
    # print(" UV Map Generation Finished!")

    # # 3. export and re-import
    # new_mesh_with_uv.export(tmp_path)

    # ms = pml.MeshSet()
    # ms.load_new_mesh(args.input_mesh)
    # ms.load_new_mesh(tmp_path)
    
    # # 4. map vertex color from source mesh to uv color of new mesh
    # ms.transfer_attributes_to_texture_per_vertex(sourcemesh=0, targetmesh=1, textw=1024, texth=1024)
    # ms.save_current_mesh(output_path, save_vertex_normal=False)



    

    