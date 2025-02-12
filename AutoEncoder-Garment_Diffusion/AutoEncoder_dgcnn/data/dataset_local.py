import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

T_ITEM = Tuple[int, str, Tensor, Tensor, Tensor, Tensor]


class LocalDataset(Dataset):
    def __init__(self, ids_file: Path, root: Path, category: str) -> None:
        super().__init__()

        self.root = root

        with open(ids_file, "rb") as f:
            self.ids = sorted(pickle.load(f))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> T_ITEM:
        item_id = self.ids[index]
        npz = np.load(self.root+"/"+f"{item_id}.npz")

        pcds=torch.from_numpy(npz['pcds'])
        selected_coords=torch.from_numpy(npz['selected_coords'])
        selected_gt_udf=torch.from_numpy(npz['selected_gt_udf'])
        selected_gt_grad=torch.from_numpy(npz['selected_gt_grad'])

        return index, item_id, pcds, selected_coords, selected_gt_udf, selected_gt_grad

