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
    assert env_gpus_str is not None, "CUDA_VISIBLE_DEVICES is not set, please set it before the running command"
    print(f"Using GPU id: {env_gpus_str}")
    

    logger = logging.getLogger("pytorch_lightning")
    logger.setLevel(logging.ERROR)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.gs_config, cli_args=extras, n_gpus=1)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(cfg.trial_dir)


    trainer = Trainer(
        logger=False,
        enable_checkpointing = False,
        enable_model_summary=False,
        inference_mode=False,
        accelerator="gpu",
        devices=1,
        **cfg.trainer,
    )


    trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
    trainer.test(system, datamodule=dm)

    return system.get_save_dir(), cfg.system.scale * cfg.data.eval_camera_distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_config", type=str, default="configs/gaussiandreamer-sd.yaml", help="path to config file")



    args, extras = parser.parse_known_args()


    generate_3dgs(args, extras)
