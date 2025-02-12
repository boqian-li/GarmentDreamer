# AutoEncoder_dgcnn


1. build and install meshudf:

   ```
   cd meshudf
   source setup.sh
   ```

---

## Run train_encdec.py with Accelerate and webdataset

1. make config for accelerate:
   
* Please follow the instruction: https://huggingface.co/docs/accelerate/en/quicktour#launch-your-distributed-script

* To put it simply:

   ```
   # To use it, run a quick configuration setup first on your machine and answer the questions:
   accelerate config  

   # At the end of the setup, a default_config.yaml file will be saved in your cache folder for 🤗 Accelerate

   # Once the configuration setup is complete, you can test your setup by running:
   accelerate test
   ```

2. start train and evaluate

   ```
   cd encdec
   accelerate launch train_encdec.py ../cfg/encdec_web.yaml 
   ```

---

## Config

### cfg/encdec.yaml

* Please set {**path_of_your_data_folder**} firstly. 

```
dset:
  root: {path_of_your_data_folder}/
  log: ../logs                      # Path to save log.txt and ckpts .pt
  dataset_length: 27777
  dataset_length_train: 27500
  dataset_length_eval: 277

num_points_pcd: 10_000 
udf_max_dist: 0.1
latent_size: 64
num_points_forward: 20_000 

decoder:
  hidden_dim: 512
  num_hidden_layers: 5

train_bs: 8
val_bs: 4
lr: 0.0001
num_epochs: 4_000
mixed_precision: 'no'          # Choose from 'no','fp16','bf16 or 'fp8'.
seed: 123
```

* **Please note that at the line 75 of `trainers/encdec.py`we have set `split_batches=True`, which means the batch size will be splitted equally and allocated to gpus.**

