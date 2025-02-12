from torch.utils.data import Dataset, DataLoader
import os
import torch
import sys
import yaml

class MyDataset(Dataset):
    def __init__(self, config):
        self.data_dir = config['root']
        self.data = []

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.pt'):
                file_path = os.path.join(self.data_dir, file_name)
                lat = torch.load(file_path)
                lat = lat.view(1, int(lat.shape[1] ** 0.5), int(lat.shape[1] ** 0.5))  # (1, 8, 8)
                self.data.append((file_name, lat))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    


if __name__ == "__main__":
    assert len(sys.argv) == 2
    
    cfg_file = sys.argv[1]
    with open(cfg_file, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = MyDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        ids, lats = batch
        print(ids)
        print(lats.shape)
        