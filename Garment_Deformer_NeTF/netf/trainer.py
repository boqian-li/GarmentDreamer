import os
import tqdm
import random
import imageio
import numpy as np
from netf.view_core.view import read_views
from netf.view_core.view_sampler import ViewSampler
from netf.view_core.camera import perspective
from kiui.cam import orbit_camera

import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, args):
        self.args = args  # shared with the trainer's args to support in-place modification of rendering parameters.

        self.perspective = perspective(args.fovy)

        self.mode = "image"
        self.seed = 3407
        self.seed_everything()

        os.makedirs(self.args.outdir, exist_ok=True)

        # models
        self.device = torch.device("cuda") # the first visible gpu!
        self.guidance = None

        # views
        self.path_rgba = args.path_rgba
        self.views = read_views(self.path_rgba, self.device)

        self.view_sampler = ViewSampler(views=self.views, mode='several', views_per_iter=1, picked_views=args.reconstruction_picked_views) 

        # renderer
        from netf.render.mesh_renderer import Renderer

        
        self.renderer = Renderer(args, self.view_sampler, self.device).to(self.device)

        # lora_unet
        self.load_lora_unet(self.device)


        # input prompt
        self.prompt = self.args.prompt
        self.negative_prompt = ""

        if self.args.positive_prompt is not None:
            self.prompt = self.prompt +  ', ' + self.args.positive_prompt
        if self.args.negative_prompt is not None:
            self.negative_prompt = self.args.negative_prompt
        
        # training stuff
        self.training = False
        self.optimizer = None
        self.global_step = 0
        self.train_steps = 1  # steps per rendering loop

        # prepare training
        self.prepare_train()


    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        # os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        print(f"== SET SEED ==: {seed}")
        self.last_seed = seed

    def load_lora_unet(self, device):
        # use lora
        from netf.vsd.lora_unet import UNet2DConditionModel     
        from diffusers.loaders import AttnProcsLayers
        from diffusers.models.attention_processor import LoRAAttnProcessor
        import einops

        _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
        _unet.requires_grad_(False)
        lora_attn_procs = {}
        for name in _unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = _unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = _unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        _unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(_unet.attn_processors)

        # text_input = self.guidance.tokenizer(args.text, padding='max_length', max_length=self.guidance.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        # with torch.no_grad():
        #     text_embeddings = self.guidance.text_encoder(text_input.input_ids.to(self.guidance.device))[0]
        
        class LoraUnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = _unet
                self.sample_size = 64
                self.in_channels = 4
                self.dtype = torch.float32
            def forward(self,x,t, text_embeddings, c=None,shading="albedo"):
                textemb = einops.repeat(text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
                return self.unet(x,t,encoder_hidden_states=textemb,c=c,shading=shading)
        
        self._unet = _unet
        self.lora_layers = lora_layers
        self.lora_unet = LoraUnet().to(device)                     


    def prepare_train(self):
        self.global_step = 0

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        params = [
                {'params': self.lora_layers.parameters()},
                {'params': self._unet.camera_emb.parameters()},
                {'params': self._unet.lambertian_emb},
                {'params': self._unet.textureless_emb},
                {'params': self._unet.normal_emb},
            ] 
        
        self.lora_unet_optimizer = torch.optim.Adam(params, lr=self.args.unet_lr)


        # lazy load guidance model
        if self.guidance is None:
            # print(f"[INFO] loading guidance...")
            if self.args.mode == 'SD':
                from netf.guidance.sd_vsd_utils import StableDiffusion
                self.guidance = StableDiffusion(self.device, vram_O=self.args.vram_O)
            elif self.args.mode == 'IF2':
                from netf.guidance.if2_utils import IF2
                self.guidance = IF2(self.device, vram_O=self.args.vram_O)
            else:
                raise NotImplementedError(f"Not implemented: {self.args.mode}!")
            # print(f"[INFO] loaded guidance!")

        # prepare embeddings
        with torch.no_grad():
            self.guidance.get_text_embeds([self.prompt], [self.negative_prompt])

           
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        self.renderer.train()

        for _ in range(self.train_steps):
            self.global_step += 1  #global_step
            loss = 0

            # prepare data
            ### novel view (manual batch)
            images = []
            poses = []
            vers, hors, radii = [], [], []

            assert self.args.batch_size == 1
            for _ in range(self.args.batch_size):
                # render random view
                ver = np.random.randint(-65, 35)
                hor = np.random.randint(-180, 180)
                radius = np.random.uniform()*4 - 3  # [-3, 1]
                pose = orbit_camera(ver, hor, self.args.radius + radius)

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)
                poses.append(torch.from_numpy(pose))

                ssaa = 1
                out = self.renderer.render(pose, self.perspective, self.args.render_resolution, self.args.render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                assert (image.shape[0] == image.shape[1]) and (image.shape[0] == 512)

                # import matplotlib.pyplot as plt
                # plt.imshow(image.detach().cpu())
                # plt.show()

                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
            
            # guidance loss
            pred_rgb = torch.cat(images, dim=0)
            if self.args.text_dir:
                hors = hors
            pose = torch.cat(poses, dim=0).view(self.args.batch_size, 16).to(self.device)
            shading = 'albedo'
            t5 = False
            if self.args.t5_iters != -1 and self.global_step >= self.args.t5_iters:
                if self.global_step == self.args.t5_iters:
                    print("Change into tmax = 500 setting")
                t5 = True

            # train step needs hyperparameter tuning!!! TODO
            if self.args.vds:
                loss, pseudo_loss, latents = self.guidance.train_step(pred_rgb=pred_rgb, guidance_scale=7.5, q_unet=self.lora_unet, pose=pose, shading=shading, as_latent=False, t5=t5)
                loss = self.args.lambda_sd * loss
            else:
                assert 1 == 0, 'please use vds mode'

            # optimize step
            loss.backward()
            if (self.global_step % self.args.batch_size_train == 0) or (self.global_step == self.args.iters):
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Train LoRa:
            assert self.args.q_cond
            if self.global_step % self.args.K2 == 0 and self.args.vds:
                for _ in range(self.args.K):        
                    timesteps = torch.randint(0, 1000, (self.args.unet_bs,), device=self.device).long() # temperarily hard-coded for simplicity
                    with torch.no_grad():
                        if True: #self.buffer_imgs is None or self.buffer_imgs.shape[0]<self.args.buffer_size:
                            latents_clean = latents.expand(self.args.unet_bs, latents.shape[1], latents.shape[2], latents.shape[3]).contiguous()
                            if self.args.q_cond:
                                pose = pose.expand(self.args.unet_bs, 16).contiguous()
                                if random.random() < self.args.uncond_p:
                                    pose = torch.zeros_like(pose)
                        else:
                            assert 1 == 0
                            # latents_clean, pose = self.sample_buffer(self.args.unet_bs)
                            # if random.random() < self.args.uncond_p:
                            #     pose = torch.zeros_like(pose)
                    noise = torch.randn(latents_clean.shape, device=self.device)
                    latents_noisy = self.guidance.scheduler.add_noise(latents_clean, noise, timesteps)
                    if self.args.q_cond:
                        model_output = self.lora_unet(latents_noisy, timesteps, self.guidance.embeddings['pos'].expand(self.args.batch_size, -1, -1), c = pose, shading = shading).sample
                    else:
                        model_output = self.lora_unet(latents_noisy, timesteps).sample
                    if self.args.v_pred:
                        loss_unet = F.mse_loss(model_output, self.guidance.scheduler.get_velocity(latents_clean, noise, timesteps))
                    else:
                        loss_unet = F.mse_loss(model_output, noise)
                    loss_unet.backward()
                    if (self.global_step % self.args.batch_size_train == 0) or (self.global_step == self.args.iters):
                        self.lora_unet_optimizer.step()
                        self.lora_unet_optimizer.zero_grad()
                        


        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

    def save_mesh(self, save_path):
        self.renderer.export_mesh(save_path, texture_resolution=self.args.texture_resolution, reverse=True)
        print(f"save finetuned mesh to {save_path}.")


    def train(self, iters=500):
        for i in tqdm.trange(iters):
            self.train_step()
        # save
        self.save_mesh(os.path.join(self.args.outdir, 'final_mesh' + '_finetuned' + '.obj'))
