import os
import shutil
import random
from tqdm import tqdm

random.seed(123)

categories_0 = ["wb_dress_sleeveless", "dress_sleeveless",
                  "tee_sleeveless", "tee", "skirt_8_panels", "skirt_4_panels", "skirt_2_panels",
                  "wb_pants_straight"]
categories_1 = ['Dress', 'Jumpsuit', 'Tshirt', 'Top', 'Trousers', 'Skirt', "jumpsuit_sleeveless","pants_straight_sides"]

categories = categories_0 + categories_1
root = '/home/boqian/Desktop/ClothesDataset_Mesh_simulated'
tar = '/home/boqian/Desktop/ClothesDataset_Mesh_simulated_reorganized'
num_per_category = 55


for category in categories:
    # 获取目录中的文件列表
    files = os.listdir(root)

    # 筛选满足条件的文件名
    filtered_files = [file for file in files if file[:file.rfind('_')] == category]
    print(category, len(filtered_files))
    random.shuffle(filtered_files)

    sampled_files = random.sample(filtered_files, num_per_category)

    for file in tqdm(sampled_files):
        file_path = os.path.join(root, file)
        tar_path = os.path.join(tar, file)
        shutil.copy(file_path, tar_path)



    