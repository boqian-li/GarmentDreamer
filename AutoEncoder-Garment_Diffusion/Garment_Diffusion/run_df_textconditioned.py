# train and test a UNet2DModel (Unconditonal)
from config import TrainingConfig
import os
from make_datasets import MyDataset, Zipper, get_data_generator
from torch.utils.data import Dataset, DataLoader
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
from networks.bert_networks.network import BERTEmbedder
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import notebook_launcher


def evaluate(config, iter_, unet, cond_model, noise_scheduler, device):
    unet.eval()
    with torch.no_grad():
        text_input = "a shirt"
        embeddings = cond_model(text_input).repeat(config.eval_batch_size, 1, 1)
        generator = torch.manual_seed(0) #cpu
        latents = torch.randn(
        (config.eval_batch_size, unet.config.in_channels, config.input_size[0], config.input_size[1]),
        generator=generator,
        ).to(embeddings.device)

        zipper = Zipper()

        noise_scheduler.set_timesteps(config.num_inference_steps)
        for t in tqdm(noise_scheduler.timesteps):

            noise_pred = unet(latents, t, encoder_hidden_states=embeddings).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    for i, latent in enumerate(latents): 
        ordered_dict = zipper.zip(latent)
        os.makedirs(os.path.join('.', f'pred_model_dicts'), exist_ok=True)
        torch.save(ordered_dict, f'./pred_model_dicts/pred_ordereddict_iter_{iter_}_sample_{i}.pth')
    unet.train()
    


def train_loop(config, unet, cond_model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        split_batches=True,
        #gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    #prepare everything
    unet, cond_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, cond_model, optimizer, train_dataloader, lr_scheduler
    )
    data_generator = get_data_generator(train_dataloader)

    if accelerator.is_main_process:
        print("==== Initialization Over! ====")
        print("==== Train Start! ====")


    # train the model
    unet.train()
    # unconditional input
    with torch.no_grad():
        text_input = "a shirt"
        embeddings = BERTEmbedder(text_input)
        embeddings = embeddings.repeat(config.train_batch_size//accelerator.num_processes, 1, 1).to(accelerator.device)
    
    
    progress_bar = tqdm(total=config.total_iters, disable=not accelerator.is_local_main_process)
    for iter_ in range(config.total_iters):
        
        progress_bar.set_description(f"Iter {iter_}")

        clean_z = next(data_generator)
        print(clean_z.shape)
        # Sample noise to add to the z
        noise = torch.randn(clean_z.shape, device=clean_z.device)
        bs = clean_z.shape[0]

        # Sample a random timestep for each z
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_z.device,
            dtype=torch.int64
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        with torch.no_grad():
            noisy_z = noise_scheduler.add_noise(clean_z, noise, timesteps)
        #with accelerator.accumulate(unet):
            
        # Predict the noise residual
        optimizer.zero_grad()
        noise_pred = unet(noisy_z, timesteps, encoder_hidden_states=embeddings, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        accelerator.backward(loss)

        #accelerator.clip_grad_norm_(unet.parameters(), 1.0) 
        optimizer.step()
        lr_scheduler.step()
        

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "iter": iter_}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=iter_)

        if accelerator.is_main_process:
            if iter_ % config.predict_iter == 0 or iter_ == config.total_iters - 1:
                print("==== Start Prediction ====")
                evaluate(config, iter_, accelerator.unwrap_model(unet), cond_model, noise_scheduler, device=accelerator.device)
                print("==== End Prediction ====")




# config
config = TrainingConfig()

# dataset and dataloader


print("==== Data Load Start! ====")
curpath = os.path.dirname(os.path.abspath(__file__))
dict_ls = []
for fn in os.listdir(os.path.join(curpath, "model_dicts")):
    #OrderedDict
    dict = torch.load(os.path.join(curpath, "model_dicts", fn), map_location=torch.device("cpu")) 
    dict_ls.append(dict)

train_dataset = MyDataset(dict_ls)
train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
print("==== Data Load Over! ====")

# model
# unet = UNet2DConditionModel(
#     sample_size=config.input_size,
#     in_channels=1,
#     out_channels=1,
#     layers_per_block=2,
#     block_out_channels=(320, 640, 1280, 1280),
#     down_block_types=("CrossAttnDownBlock2D", 
#                       "CrossAttnDownBlock2D", 
#                       "CrossAttnDownBlock2D", 
#                       "DownBlock2D"
#                       ),
#     up_block_types=("UpBlock2D", 
#                     "CrossAttnUpBlock2D", 
#                     "CrossAttnUpBlock2D", 
#                     "CrossAttnUpBlock2D"
#                     ), 
# )

# for param in cond_model.parameters():
#     param.requires_grad = True
#  # param list
#         trainable_models = [self.df, self.cond_model]
#         trainable_params = []
#         for m in trainable_models:
#             trainable_params += [p for p in m.parameters() if p.requires_grad == True]
#             # print(len(trainable_params))

#         if self.isTrain:
            
#             # initialize optimizers
#             self.optimizer = optim.AdamW(trainable_params, lr=opt.lr)
#             self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)


# train

print("==== Initialisation Start! ====")
unet = UNet2DConditionModel(
    sample_size=config.input_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 32, 64),
    down_block_types=("CrossAttnDownBlock2D", 
                    "CrossAttnDownBlock2D", 
                    "DownBlock2D"
                    ),
    up_block_types=("UpBlock2D", 
                    "CrossAttnUpBlock2D", 
                    "CrossAttnUpBlock2D", 
                    ), 
)
cond_model = BERTEmbedder(n_embed=config.n_embed, n_layer=config.n_layer)
noise_scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps)
optimizer = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, 
    num_warmup_steps=config.lr_warmup_iters,
    num_training_steps=(config.total_iters),
)

train_loop(config, unet, cond_model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
