import os
import os.path as osp
import cv2
import time
import sys
CODE_SPACE=os.path.dirname(os.path.abspath(__file__))
sys.path.append(CODE_SPACE)

import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
import random
import numpy as np
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data


from argparse import Namespace

def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Namespace(**config_dict)

def merge_namespaces(ns1, ns2):
    merged_dict = {**vars(ns1), **vars(ns2)}
    return Namespace(**merged_dict)

def estimate_normal(args):

    args_normal_estimator = load_config('Garment_3DGS/Normal_estimator_Metric3D/metric3d_args.yaml')

    args = merge_namespaces(args_normal_estimator, args)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp

    # os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        
    cfg.show_dir = args.sample_path
    
    # ckpt path
    if args.load_from is None:
        raise RuntimeError('Please set model path!')
    cfg.load_from = args.load_from
    
    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)
    
    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)
    
    # init the logger before other steps
    # cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    # logger = setup_logger(cfg.log_file)
    
    # log some basic info
    # logger.info(f'Config:\n{cfg.pretty_text}')
    
    # init distributed env dirst, since logger depends on the dist info
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_env(args.launcher, cfg)
    # logger.info(f'Distributed training: {cfg.distributed}')
    
    # dump config 
    # cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))
    test_data_path = os.path.join(args.sample_path, 'gs_rendered_rgba')
    

    if 'json' in test_data_path:
        test_data = load_from_annos(test_data_path)
    else:
        test_data = load_data(test_data_path)
    
    if not cfg.distributed:
        main_worker(0, cfg, args.launcher, test_data)
    else:
        # distributed training
        if args.launcher == 'ror':
            local_rank = cfg.dist_params.local_rank
            main_worker(local_rank, cfg, args.launcher, test_data)
        else:
            mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher, test_data))
        
def main_worker(local_rank: int, cfg: dict, launcher: str, test_data: list):
    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank

        if launcher == 'ror':
            init_torch_process_group(use_hvd=False)
        else:
            torch.cuda.set_device(local_rank)
            default_timeout = timedelta(minutes=30)
            dist.init_process_group(
                backend=cfg.dist_params.backend,
                init_method=cfg.dist_params.dist_url,
                world_size=cfg.dist_params.world_size,
                rank=cfg.dist_params.global_rank,
                timeout=default_timeout)
    
    # logger = setup_logger(cfg.log_file)
    # build model
    model = get_configured_monodepth_model(cfg, )
    
    # config distributed training
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    # load ckpt
    model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()
    
    do_scalecano_test_with_custom_data(
        model, 
        cfg,
        test_data,
        # logger,
        cfg.distributed,
        local_rank
    )
    
if __name__ == '__main__':
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    estimate_normal(args)    