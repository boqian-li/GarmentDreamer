from diffusers import UNet2DConditionModel, DDIMScheduler, UNet2DModel
from accelerate import Accelerator
from tqdm import tqdm
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import yaml
import sys
from dataset.my_dataset import MyDataset
import torch
import torch.nn.functional as F



def main(config):
    category_to_index = config['category_to_index']
    unet = UNet2DModel(sample_size=config['sample_size'], in_channels=config['in_channels'], out_channels=config['out_channels'], 
                                layers_per_block=config['layers_per_block'], 
                                down_block_types=config['down_block_types'], 
                                up_block_types=config['up_block_types'],
                                block_out_channels=config['block_out_channels'],
                                class_embed_type=None,
                                num_class_embeds=len(category_to_index)
                                )

    noise_scheduler = DDIMScheduler(num_train_timesteps=config['train_timesteps'])
    optimizer = torch.optim.Adam(unet.parameters(), lr=config['lr_unet'])
    accelerator = Accelerator(split_batches=True, mixed_precision=config["mixed_precision"])
    train_dataset = MyDataset(config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['train_bs'], shuffle=True)

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    for epoch in range(config['num_epochs']):
        accelerator.print(f"========== EPOCH {epoch} ==========")
        # Train
        for batch in train_dataloader:

            # prepare embeddings
            ids, lats = batch
            indexs = []
            for id in ids:
                category = '_'.join(id.split('_')[:-1])
                index = int(category_to_index[category])
                indexs.append(index)

            indexs = torch.tensor(indexs, dtype=torch.long)
            print(indexs, indexs.type(), indexs.shape)

            # sample noise
            noise = torch.randn(lats.shape, device=lats.device)

            # Sample random timestep for each lat
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (int(lats.shape[0]),),
                dtype=torch.int64, device=lats.device
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            with torch.no_grad():
                noisy_lats = noise_scheduler.add_noise(lats, noise, timesteps)

            optimizer.zero_grad()
            noise_pred = unet(noisy_lats, timesteps, class_labels=indexs, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
        


        # eval and log and save ckpt

    

if __name__ == "__main__":
    assert len(sys.argv) == 2
    
    cfg_file = sys.argv[1]
    with open(cfg_file, 'r') as f:
        config = yaml.safe_load(f)

    
    main(config)
