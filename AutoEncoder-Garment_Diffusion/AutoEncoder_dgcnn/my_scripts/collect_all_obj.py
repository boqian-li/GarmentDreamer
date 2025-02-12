import shutil
import os
from tqdm import tqdm

mesh_dirs = ["SewFactory_clean", "Cloth3D_clean"]
tar_dir = "ClothesDataset_Mesh"

os.makedirs(tar_dir, exist_ok=True)

for mesh_dir in mesh_dirs: 
    for folder in os.listdir(mesh_dir):
        folder_path = os.path.join(mesh_dir, folder)
        if os.path.isdir(folder_path):
            for obj in tqdm(os.listdir(folder_path)):
                if obj.endswith('.obj'):
                    shutil.copy(os.path.join(folder_path, obj), os.path.join(tar_dir, obj))
                    