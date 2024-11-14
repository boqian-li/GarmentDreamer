import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from netf.trainer import Trainer
import yaml
from argparse import Namespace


def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Namespace(**config_dict)

def reconstruction(args):    

    args_reconstruction = load_config(args.netf_config)

    args_reconstruction.mesh = os.path.join(args.sample_path, 'final_mesh.obj')
    args_reconstruction.prompt = args.prompt
    args_reconstruction.path_rgba = os.path.join(args.sample_path, 'gs_rendered_rgba/')
    args_reconstruction.outdir = args.sample_path

    args = args_reconstruction


    trainer = Trainer(args)
    if args.enhance:
        trainer.train(args.iters)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Reconstruction")
#     parser.add_argument("--mesh", type=str, required=True, help="Path to the input mesh")
#     parser.add_argument("--prompt", type=str, required=True, help="Prompt for the reconstruction")
#     parser.add_argument("--path_rgba", type=str, required=True, help="Path to the input rgba image")
#     parser.add_argument("--iters", type=int, required=True, help="Number of iterations")
#     parser.add_argument("--outdir", type=str, required=True, help="Output directory")
#     parser.add_argument("--enhance", type=bool, default=False, help="VSD Enhancement")
#     args = parser.parse_args()
#     reconstruction(args)
