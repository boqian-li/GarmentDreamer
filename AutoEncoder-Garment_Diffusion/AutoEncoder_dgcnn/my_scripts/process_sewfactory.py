import os
import shutil
import open3d as o3d
import numpy as np
import trimesh
from tqdm import tqdm

def get_catogory_name(filename):
    categories = ["jumpsuit_sleeveless", "wb_dress_sleeveless", "dress_sleeveless",
                  "tee_sleeveless", "tee", "skirt_8_panels", "skirt_4_panels", "skirt_2_panels",
                  "pants_straight_sides", "wb_pants_straight"]
    n = []
    for category in categories:
        if filename.startswith(category):
            n.append(category)
            break
    # assert len(n) == 1, print(n)
    return n[0]


def process_obj_info(obj_directory, tar_directory):
    categories = {}
    
    # 遍历目录中的所有子文件夹
    for folder_name in tqdm(os.listdir(obj_directory)):
        folder_path = os.path.join(obj_directory, folder_name)
        if os.path.isdir(folder_path):
            if not os.path.isdir(os.path.join(folder_path, "static")):
                print(f"No 'static' dir: {folder_name}")
                continue
            for filename in os.listdir(os.path.join(folder_path, "static")):
                if filename.endswith(".obj"):
                    # 获取 OBJ 文件的类别名称
                    category_name = get_catogory_name(filename)
                    if category_name not in categories:
                        categories[category_name] = 1
                        os.mkdir(os.path.join(tar_directory, f"{category_name}"))
                        
                    else:
                        categories[category_name] += 1
                        assert os.path.exists(os.path.join(tar_directory, f"{category_name}"))

                    shutil.copy(os.path.join(folder_path, "static", filename), os.path.join(tar_directory, f"{category_name}", category_name+f"_{categories[category_name] - 1}.obj"))
                    shutil.copy(os.path.join(folder_path, "data_props.json"), os.path.join(tar_directory, f"{category_name}", f"info_{categories[category_name] - 1}.json"))
                    

                    
                    
                    
    return categories

def norm(tar_directory):
    for folder_name in os.listdir(tar_directory):
        folder_path = os.path.join(tar_directory, folder_name)
        if os.path.isdir(folder_path):
            for filename in tqdm(os.listdir(folder_path)):
                if filename.endswith(".obj"):
                    file = os.path.join(folder_path, filename)

                    # 去除无用信息
                    mesh = trimesh.load_mesh(file, skip_materials=True)
                    # 只保留v
                    vertices = mesh.vertices

                    # 将f中的索引指向v
                    faces = mesh.faces

                    # 创建新的Mesh对象，只包含v和新的f
                    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    new_mesh.export(file, digits=6)

                    # 读取OBJ文件
                    mesh = o3d.io.read_triangle_mesh(file)

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
    obj_directory = "sewfactory/sewfactory"  # 替换为包含所有 OBJ 文件的目录路径
    tar_directory = "SewFactory_clean"
    os.makedirs(tar_directory, exist_ok=True)
    
    # 统计每个类别的数量
    categories = process_obj_info(obj_directory, tar_directory)
    write_to_txt(categories, os.path.join(tar_directory, "categories_sewfactory.txt"))

    # 处理mesh，归一化
    norm(tar_directory)

if __name__ == "__main__":
    main()

