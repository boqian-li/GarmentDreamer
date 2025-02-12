## AutoEncoder-Garment_Diffusion

### AutoEncoder_dgcnn
1. This folder contains the code for training a model that encodes garment meshes into latent codes and decodes them back into garment meshes. 
2. Basic process:

    1. `encdec/preprocess_udf.py` -> (`my_scripts/pre_resample.py`): Prepare training data
    2. `encdec/train_encdec.py`: Train the model. 
    3. `encdec/export_codes.py` or `encdec/export_meshes.py`: Encode garment meshes into latent codes or decode them back into garment meshes. 

3. Since the code is not fully cleaned, you may need to make slight modifications before running it, especially in the dataset-related code because we use web dataset and local dataset in different versions. But it contains all the useful info and necessary code used in our experiments. our requirements. 

### Garment_Diffusion
1. Once you have generated the latent codes from garment meshes, you can use these latent codes as data for this step. The code is used to train a diffusion model (category_conditioned, unconditioned, text_conditioned) on the latent codes allowing you to generate new latent codes conditioned by the input text or category labels. 

2. The code is not fully cleaned. But it contains all the useful info and necessary code used in our experiments for our requirements.  
