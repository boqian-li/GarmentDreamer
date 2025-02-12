import os 
import pickle
import shutil
from tqdm import tqdm

train_ids = "/devdata/ids_train.pkl"
eval_ids = "/devdata/ids_eval.pkl"


src_dir = "/devdata/ClothesDataset_resampled"
target_dir_train = "/devdata/ClothesDataset_resampled_train"
target_dir_eval = "/devdata/ClothesDataset_resampled_eval"

os.makedirs(target_dir_train, exist_ok=False)
os.makedirs(target_dir_eval, exist_ok=False)

with open(train_ids, "rb") as f:
    ids = sorted(pickle.load(f))
for id in tqdm(ids):
    shutil.copy(src_dir+"/"+f"{id}.npz", target_dir_train+"/"+f"{id}.npz")


with open(eval_ids, "rb") as f:
    ids = sorted(pickle.load(f))
for id in tqdm(ids):
    shutil.copy(src_dir+"/"+f"{id}.npz", target_dir_eval+"/"+f"{id}.npz")