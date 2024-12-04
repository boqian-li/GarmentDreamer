<p align="center">

  <h2 align="center">GarmentDreamer<br> 3DGS Guided Garment Synthesis with Diverse Geometry and Texture Details</h2>
  <p align="center">
    <a href="https://boqian-li.github.io/"><strong>Boqian Li</strong></a>
    ·
    <a href="https://xuan-li.github.io/"><strong>Xuan Li</strong></a>
    ·
    <a href="https://yingjiang96.github.io/"><strong>Ying Jiang</strong></a>
    ·
    <a href="https://xpandora.github.io/"><strong>Tianyi Xie</strong></a>
    ·
    <a href="https://fen9.github.io/"><strong>Feng Gao</strong></a>
    ·
    <a href="https://wanghmin.github.io/"><strong>Huamin Wang</strong></a>
    ·
    <a href="https://yangzzzy.github.io/"><strong>Yin Yang</strong></a>
    ·
    <a href="https://www.math.ucla.edu/~cffjiang/"><strong>Chenfanfu Jiang</strong></a>
    <br>
  </p>
  <h3 align="center">International Conference on 3D Vision (3DV) 2025</h3>

  <div align="center">
    <img src="assets/dance.gif">
  </div>

  <p align="center">
  </br>
    <a href="https://arxiv.org/abs/2405.12420">
      <img src='https://img.shields.io/badge/Paper-Arxiv-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://xuan-li.github.io/GarmentDreamerDemo/'>
      <img src='https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://github.com/boqian-li/GarmentDreamer">
      <img src='https://img.shields.io/badge/Code-Github-blue?style=for-the-badge&logo=github&logoColor=white&labelColor=181717' alt='Code Github'></a>
  </p>
</p>


## 📝 TODO

- [x] fix gpu, fix finalmesh reverse problem
- [x] Better configs
- [x] Upload more templates
- [ ] Improve Garment_3DGS to obtain more significant deformation
- [ ] Release the code of AutoEncoder_dgcnn and Garment_Diffusion and release the pretrained models



## 🛠️ Environment Setup

0. Environment:
  - Ubuntu 20.04
  - CUDA 11.8

1. Create Env:

   ```
   conda create -n garmentdreamer python==3.10
   conda activate garmentdreamer
   conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia # specific channel matters
   pip install -r requirements.txt
   pip install xformers==0.0.27 --extra-index-url https://download.pytorch.org/whl/cu118
   ```
2. Install pytorch3d:

  * Download file from https://anaconda.org/pytorch3d/pytorch3d/files, `pytorch3d-0.7.8-py310_cu118_pyt231.tar.bz2` as example.
  * Then install it by:

    ```
    conda install pytorch3d-0.7.8-py310_cu118_pyt231.tar.bz2
    ```
  
3. Install requirements for 3DGS

    ```
    pip install Garment_3DGS/gaussiansplatting/submodules/diff-gaussian-rasterization
    pip install Garment_3DGS/gaussiansplatting/submodules/simple-knn
    ```

4. Huggingface login (国内需要先 export HF_ENDPOINT=https://hf-mirror.com)
    ```
    huggingface-cli login 
    # Then input your huggingface token for authentication
    ```

## 🚀 Get Started

### 🧩 For Normal Estimator
* In this version, we use the normal estimator in [Metric3D](https://github.com/YvanYin/Metric3D) to estimate the normal map of the input garment. You can also use your own normal estimator. 
* As discribed in [Metric3D](https://github.com/YvanYin/Metric3D), download the pretrained normal estimator from [here](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view) and put it in `Garment_3DGS/Normal_estimator_Metric3D/weight/metric_depth_vit_large_800k.pth`.

### 🧩 For mesh templates
* Option 1: Download the mesh templates from [here](https://drive.google.com/drive/folders/1ye9vZ481I-5EstpH6liuswtRVDgrlJvl?usp=sharing) and put them in `Garment_Deformer_NeTF/input_data/`.

* Option 2: You can also use your own mesh templates. But please make sure the direction of the mesh is the same as those in Option 1.

* tips:
  * Our provided mesh templates are unwrinkled, but wrinkled meshes can be better for the final performance.
  * If you got not good results, please try another mesh template, it's very likely that the mesh template is not suitable for your prompt.
  * You may need to translate the mesh template first to ensure the mesh is within the camera's field of view, please check the `outputs/{sample_dir}/gs_check` to see if the position of the mesh is good.

### 🏃‍♂️ Run the Code
* After setting up the environment and downloading necessary files, you can run the main script using the following command:

  ```bash
  CUDA_VISIBLE_DEVICES=/gpu_id/only_one/is_supported python launch_garmentdreamer.py --template_path /path/to/your/mesh/template.obj --prompt "your prompt"
  ```
* tips:
  * I have tried to give the best config as I can but it's still not perfect. All important parameters are in the config files as `launch_garmentdreamer.py` shows and you can modify them to get better results. If you have any questions, please feel free to contact me.
  * A very useful strategy to describe the garment is to use the prompt template: `a {style} {garment type} made of {color} {material}` or a `{color} {material} {garment type}`. For example, 'a traditional royal style dress made of blue silk' or 'a blue denim tee'.

## Acknowledgment

This implementation is built based on [GaussianDreamer](https://github.com/hustvl/GaussianDreamer), [Metric3D](https://github.com/YvanYin/Metric3D), [Neural Deferred Shading](https://github.com/fraunhoferhhi/neural-deferred-shading) and [threefiner](https://github.com/3DTopia/threefiner).



## 📬 Contact

Please contact *Boqian Li* via boqianlihuster@gmail.com



## 📑 Citation

If you find this code or our method useful for your academic research, please cite our paper

```bibtex
@article{li2024garmentdreamer,
  title={GarmentDreamer: 3DGS Guided Garment Synthesis with Diverse Geometry and Texture Details},
  author={Li, Boqian and Li, Xuan and Jiang, Ying and Xie, Tianyi and Gao, Feng and Wang, Huamin and Yang, Yin and Jiang, Chenfanfu},
  journal={arXiv preprint arXiv:2405.12420},
  year={2024}
}
```