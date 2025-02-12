from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
# from utils import compute_gradients
from torch import Tensor

from accelerate import Accelerator
from accelerate.logging import get_logger
from datetime import datetime
import os
import numpy as np
# from data.dataset_web import create_train_dataset, create_eval_dataset
from data.dataset_local import LocalDataset

# def collate_wrapper(batch_list):
#     # vertices = [] 
#     # triangles = []
#     pcds = [] 
#     selected_coords = [] 
#     selected_gt_grad = [] 
#     selected_gt_udf = []
#     for item in batch_list:
#         # vertices.append(item[0])
#         # triangles.append(item[1])
#         pcds.append(item[2])
#         selected_coords.append(item[3])
#         selected_gt_grad.append(item[4])
#         selected_gt_udf.append(item[5])
#     if len(batch_list) == 1:
#         return pcds[0].unsqueeze(0), selected_coords[0].unsqueeze(0), selected_gt_grad[0].unsqueeze(0), selected_gt_udf[0].unsqueeze(0)
#     else:
#         return torch.stack(pcds, dim=0), torch.stack(selected_coords, dim=0), torch.stack(selected_gt_grad, dim=0), torch.stack(selected_gt_udf, dim=0) 

def compute_gradients(x: Tensor, y: Tensor) -> Tensor:
    grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True)[0]
    return grads


class EncoderDecoderTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.save_ckpt_f = config['save_ckpt']
        self.save_log_f = config['save_log']

        # set seed
        seed = config["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.root = config['dset']['root']

        train_dset = LocalDataset(self.config)
        train_bs = config['train_bs']
        eval_dset = LocalDataset(self.config)
        eval_bs = config['val_bs']

        self.train_loader = DataLoader(
            train_dset,
            train_bs,
            num_workers=16, #  !!!!
            shuffle=True,
            pin_memory=True,
            # collate_fn=collate_wrapper
        )

        self.eval_loader = DataLoader(
            eval_dset,
            eval_bs,
            num_workers=16, # !!!!
            pin_memory=True,
            shuffle=True
            # collate_fn=collate_wrapper
        )

        self.num_points_pcd = config['num_points_pcd']
        latent_size = config['latent_size']
        self.max_dist = config['udf_max_dist']
        self.num_points_forward = config['num_points_forward']

        self.encoder = Dgcnn(latent_size)

        self.coords_encoder = CoordsEncoder()

        self.decoder = CbnDecoder(
            self.coords_encoder.out_dim,
            latent_size,
            config['decoder']['hidden_dim'],
            config['decoder']['num_hidden_layers'],
        )


        params = list(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        lr = config['lr']
        self.optimizer = Adam(params, lr)


        self.accelerator = Accelerator(split_batches=True, mixed_precision=config["mixed_precision"])
        self.encoder, self.decoder, self.train_loader, self.eval_loader, self.optimizer = self.accelerator.prepare(
            self.encoder, self.decoder, self.train_loader, self.eval_loader, self.optimizer)

        self.epoch = 0
        self.global_step = 0

        now = datetime.now()
        formatted_time = "{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        self.ckpts_path = os.path.join(config["dset"]["log"], formatted_time, "ckpts")
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpts_path, exist_ok=False)


    def train(self) -> None:
        num_epochs = self.config["num_epochs"]
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            self.accelerator.print(f"========== EPOCH {epoch} ==========")
            self.epoch = epoch

            # Train
            self.encoder.train()
            self.decoder.train()
            total_train_loss = 0
            train_loss = 0
            batch_num = len(self.train_loader)
            batch_num_real = 0

            for b, batch in enumerate(self.train_loader):
                self.accelerator.print(f"  BATCH: {b + 1} / {batch_num} ||  ", end='')
                
                self.optimizer.zero_grad()
                keys, urls, pcds, selected_coords, selected_gt_grad, selected_gt_udf = batch.values()
                selected_coords.requires_grad=True  
                
                latent_codes = self.encoder(pcds)
                coords_encoded = self.coords_encoder.encode(selected_coords)
                pred = self.decoder(coords_encoded, latent_codes)

                udf_loss = F.binary_cross_entropy_with_logits(pred, selected_gt_udf)

                udf_pred = torch.sigmoid(pred)
                udf_pred = 1 - udf_pred
                udf_pred *= self.max_dist
                gradients = compute_gradients(selected_coords, udf_pred)

                grad_loss = F.mse_loss(gradients, selected_gt_grad, reduction="none")
                # self.accelerator.print(grad_loss.shape) #torch.Size([4, 20000, 3])
                mask = (selected_gt_udf > 0) & (selected_gt_udf < 1)
                grad_loss = grad_loss[mask].mean()

                loss = udf_loss + 0.1 * grad_loss
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                train_loss = self.accelerator.gather(loss).mean().detach().item()
                total_train_loss += train_loss
                self.accelerator.print(f"train_loss: {train_loss*100}")
                
                self.global_step += 1     
                batch_num_real += 1       

            # Evaluate
            self.accelerator.print(f" EPOCH {epoch}: EVAL ||  ", end='')
            self.encoder.eval()
            self.decoder.eval()
            eval_loss = 0
            eval_batch_num = len(self.eval_loader)
            eval_batch_num_real = 0

            for batch in self.eval_loader:
                keys, urls, pcds, selected_coords, selected_gt_grad, selected_gt_udf = batch.values()
                selected_coords.requires_grad=True

                with torch.no_grad():
                    latent_codes = self.encoder(pcds)
                coords_encoded = self.coords_encoder.encode(selected_coords)
                pred = self.decoder(coords_encoded, latent_codes)
                
                udf_loss = F.binary_cross_entropy_with_logits(self.accelerator.gather(pred), self.accelerator.gather(selected_gt_udf))

                udf_pred = torch.sigmoid(pred)
                udf_pred = 1 - udf_pred
                udf_pred *= self.max_dist
                gradients = compute_gradients(selected_coords, udf_pred)

                grad_loss = F.mse_loss(gradients, selected_gt_grad, reduction="none")
                mask = (selected_gt_udf > 0) & (selected_gt_udf < 1)
                
                grad_loss = grad_loss[mask].mean()
                grad_loss = self.accelerator.gather(grad_loss).mean().detach().item()

                loss = udf_loss + 0.1 * grad_loss
                eval_loss += loss.item()
                eval_batch_num_real += 1

            self.accelerator.print(f"eval_loss: {eval_loss * 100/ eval_batch_num_real}")


            # logging:
            if epoch % int(self.save_log_f) == 0 or epoch == num_epochs - 1:
                if self.accelerator.is_main_process:
                    with open(os.path.join(os.path.dirname(self.ckpts_path), "log.txt"), 'a') as f:
                        w = f"===== Epoch {epoch},  train_loss: {total_train_loss * 100 / batch_num_real},  eval_loss: {eval_loss * 100 / eval_batch_num_real}\n"
                        f.write(w)

            # save ckpt
            if epoch % int(self.save_ckpt_f) == 0 or epoch == num_epochs - 1:
                self.save_ckpt()

    def save_ckpt(self, all: bool = False) -> None:
        self.accelerator.wait_for_everyone()
        encoder = self.accelerator.unwrap_model(self.encoder)
        decoder = self.accelerator.unwrap_model(self.decoder)

        ckpt = {
            "epoch": self.epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            #"optimizer": self.optimizer.state_dict(),
        }

        ckpt_path = os.path.join(self.ckpts_path, f"{self.epoch}.pt")
        self.accelerator.save(ckpt, ckpt_path)

