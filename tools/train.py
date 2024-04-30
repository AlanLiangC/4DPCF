import sys
sys.path.append('/home/alan/AlanLiang/Projects/3D_Reconstruction/AlanLiang/4DPCF')

import os
from open4dpcf.utils import create_parser, update_config, load_config, setup_multi_processes, get_dist_info
from open4dpcf.api import BaseExperiment

if __name__ == "__main__":
    
    args = create_parser().parse_args()
    config = args.__dict__

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
    exp.train()
