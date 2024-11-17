from Garment_3DGS.generate_3dgs import generate_3dgs
from Garment_3DGS.Normal_estimator_Metric3D import estimate_normal
from Garment_Deformer_NeTF.deformation import deformation
from Garment_Deformer_NeTF.reconstruction import reconstruction

import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # configs
    parser.add_argument("--gs_config", type=str, default="Garment_3DGS/configs/gaussiandreamer-sd.yaml", help="path to config file")
    parser.add_argument("--deformer_config", type=str, default="Garment_Deformer_NeTF/configs/garment_deformer_configs.yml", help="path to config file")
    parser.add_argument("--netf_config", type=str, default="Garment_Deformer_NeTF/configs/garment_netf_configs.yml", help="path to config file")

    # inputs and conditions
    parser.add_argument("--template_path", type=str, required=True, help="path to data file")
    parser.add_argument("--prompt", type=str, required=True, help="prompt for the model")
    parser.add_argument("--output_folder", type=str, default="outputs", help="path to output folder")

    # gpu
    print("Please make sure to set CUDA_VISIBLE_DEVICES when running the script")


    args = parser.parse_args()

    print("========== Generating 3DGS... ==========\n")
    args.sample_path, args.bound = generate_3dgs(args)
    print("========== Done! ==========\n")


    print("========== Estimating normals... ==========\n")
    estimate_normal(args)
    print("========== Done! ==========\n")


    print("========== Deforming... ==========\n")
    deformation(args)
    print("========== Done! ==========\n")


    print("========== Reconstructing... ==========\n")
    reconstruction(args)
    print("========== Done! ==========\n")

    
