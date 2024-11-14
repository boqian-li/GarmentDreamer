import argparse
import contextlib
import logging
import os
import sys
import shutil
import warnings
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.config import ExperimentConfig, load_config
from threestudio.utils.misc import get_rank

warnings.filterwarnings("ignore", category=UserWarning)

def generate_3dgs(args) -> None:
    extras = ['system.prompt_processor.prompt=' + args.prompt, 'system.load_path=' + args.template_path, 'exp_root_dir=' + args.output_folder]
    
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = 1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU(s): {args.gpu}")

    

    logger = logging.getLogger("pytorch_lightning")
    logger.setLevel(logging.ERROR)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.gs_config, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(cfg.trial_dir)


    trainer = Trainer(
        logger=False,
        checkpoint_callback=False,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )


    trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
    trainer.test(system, datamodule=dm)

    return system.get_save_dir(), cfg.system.scale * cfg.data.eval_camera_distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_config", type=str, default="configs/gaussiandreamer-sd.yaml", help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )


    args, extras = parser.parse_known_args()


    generate_3dgs(args, extras)
