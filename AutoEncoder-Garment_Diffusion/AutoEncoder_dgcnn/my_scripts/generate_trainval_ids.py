import os 
import random
import pickle

random.seed(10)

mesh_dir = "ClothesDataset_Mesh"
ids_all = []
ids_train = []
ids_val = []

for obj in os.listdir(mesh_dir):
    if obj.endswith("obj"):
        ids_all.append(obj.split('.')[0])

assert len(ids_all) == 27777
sample_size = len(ids_all) // 100

random.shuffle(ids_all)

ids_train = ids_all[:-sample_size]
ids_val = ids_all[-sample_size:]

with open("ids_train.pkl", 'wb') as f:
    pickle.dump(ids_train, f)

with open("ids_eval.pkl", 'wb') as f:
    pickle.dump(ids_val, f)

print(sorted(ids_val))