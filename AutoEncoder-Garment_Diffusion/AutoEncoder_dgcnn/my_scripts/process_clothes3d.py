import os
import shutil
import open3d as o3d
import numpy as np
import trimesh
from tqdm import tqdm

def process_obj_info(obj_directory, tar_directory):
    categories = {}
    
    # 遍历目录中的所有子文件夹
    for folder_name in os.listdir(obj_directory):
        folder_path = os.path.join(obj_directory, folder_name)
        if os.path.isdir(folder_path):
            # 遍历子文件夹中的所有 OBJ 文件
            for filename in os.listdir(folder_path):
                if filename.endswith(".obj"):
                    # 获取 OBJ 文件的类别名称
                    category_name = filename.split(".")[0]  # 假设类别名称是文件名的前缀
                    if category_name not in categories:
                        categories[category_name] = 1
                        os.mkdir(os.path.join(tar_directory, f"{category_name}"))
                        
                    else:
                        categories[category_name] += 1
                        assert os.path.exists(os.path.join(tar_directory, f"{category_name}"))

                    shutil.copy(os.path.join(folder_path, filename), os.path.join(tar_directory, f"{category_name}", category_name+f"_{categories[category_name] - 1}.obj"))
                    shutil.copy(os.path.join(folder_path, "info.mat"), os.path.join(tar_directory, f"{category_name}", f"info_{categories[category_name] - 1}.mat"))
                    

                    
                    
                    
    return categories

def norm_rotate(tar_directory):
    for folder_name in os.listdir(tar_directory):
        folder_path = os.path.join(tar_directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in tqdm(os.listdir(folder_path)):
                if filename.endswith(".obj"):
                    file = os.path.join(folder_path, filename)

                    # 四面体网格转换三角形网格
                    mesh = trimesh.load_mesh(file)
                    mesh.export(file)

                    # 读取OBJ文件
                    mesh = o3d.io.read_triangle_mesh(file)

                    # 将Mesh绕x轴从90度旋转到0度
                    rotation_matrix = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0.0, 0.0))
                    mesh.rotate(rotation_matrix, center=(0, 0, 0))

                    # 获取顶点坐标
                    vertices = np.asarray(mesh.vertices)

                    # 将顶点坐标对齐到原点
                    center = vertices.mean(axis=0) #mesh.get_center()
                    vertices -= center

                    # 将顶点坐标归一化到单位球内
                    m = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
                    vertices = vertices / m

                    # 更新Mesh的顶点坐标
                    mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    

                    # 保存处理后的Mesh为OBJ文件
                    o3d.io.write_triangle_mesh(file, mesh)


def write_to_txt(categories, output_file):
    with open(output_file, "w") as f:
        for category, count in categories.items():
            f.write(f"{category}: {count}\n")

def main():
    obj_directory = "Cloth3D_train"  # 替换为包含所有 OBJ 文件的目录路径
    tar_directory = "Cloth3D_clean"
    os.makedirs(tar_directory, exist_ok=True)
    
    # 统计每个类别的数量
    categories = process_obj_info(obj_directory, tar_directory)
    write_to_txt(categories, os.path.join(tar_directory, "categories_clothes3D.txt"))

    # 处理mesh，归一化并旋转
    norm_rotate(tar_directory)

if __name__ == "__main__":
    main()
