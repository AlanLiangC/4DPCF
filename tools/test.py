# Copyright (c) CAIRI AI Lab. All rights reserved
import sys
sys.path.append('/data1/liangao/Projects/3D_Restruction/4DPCF')

import os
import torch
import warnings
warnings.filterwarnings('ignore')
 
from open4dpcf.api import BaseExperiment
from open4dpcf.utils import (create_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

if __name__ == "__main__":

    args = create_parser().parse_args()
    args.test = True
    config = args.__dict__

    ######## Ori ########
    # args.method = 'AL1'
    # args.model_name = 'ori'
    # args.dataname = 'nusc'
    # args.forecasting_time = '3s'
    # args.ex_name = 'Ori_nusc_mini_3s'
    # args.test_from = 'work_dirs/Ori_nusc_mini_3s/checkpoint.pth'

    # args.local_rank = 3
    #######################

    ######## LSTM ########
    args.method = 'AL1'
    args.model_name = 'lstm'
    args.dataname = 'kitti_od'
    args.forecasting_time = '1s'
    args.ex_name = 'Our_kitti_trainval_1s'
    args.test_from = 'work_dirs/Our_kitti_trainval_1s/checkpoint.pth'

    args.local_rank = 3
    #######################

    ########
    if not args.dist:
        os.environ['CUDA_VISIBLE_DEVICES']=f"{args.local_rank}"
        torch.cuda.set_device(args.local_rank)
    ########

    # method
    method = args.method
    method_config_file = os.path.join('configs/methods', method + '.py')
    assert os.path.exists(method_config_file)

    if args.overwrite:
        config = update_config(config, load_config(method_config_file),
                               exclude_keys=['method', 'batch_size', 'warmup_lr', 'opt'])
    else:
        loaded_cfg = load_config(method_config_file)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'batch_size', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])

    # model
    model_name = args.model_name
    model_config_file = os.path.join('configs/models', model_name + '.py')
    assert os.path.exists(model_config_file)
    config = update_config(config, load_config(model_config_file),
                               exclude_keys=['ex_name'])

    # dataset
    dataname = args.dataname
    data_config_file = os.path.join('configs/datasets', dataname + '.py')
    assert os.path.exists(data_config_file)
    config = update_config(config, load_config(data_config_file),
                               exclude_keys=['data_root'],
                               new_key='data_config')

    setup_multi_processes(config)
    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.test()
