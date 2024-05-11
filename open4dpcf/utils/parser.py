import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='4D Point Cloud Forecasting Training')
    
    # method parameters
    parser.add_argument('--method', '-m', default=None, type=str,
                        help='Name of video prediction method to train (default: "AL1")')
    parser.add_argument('--model_name', default=None, type=str,
                        help='Name of video prediction model to use')
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)

    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--launcher', default='none', type=str,
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        help='job launcher for distributed training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--diff_seed', action='store_true', default=False,
                        help='Whether to set different seeds for different ranks')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                        help='Whether to find unused parameters in forward during DDP training')
    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='Whether to allow overwriting the provided config file with args')
    parser.add_argument('--res_dir', default='work_dirs', type=str)
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
    parser.add_argument('--test_from', type=str, default=None, help='the checkpoint file to test from')
    parser.add_argument('--inference', '-i', action='store_true', default=False, help='Only performs inference')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')
    parser.add_argument('--empty_cache', action='store_true', default=True,
                        help='Whether to empty cuda cache after GPU training')
    
    # Dataset 
    parser.add_argument('--dataname', '-d', default='nusc', type=str,
                        help='Dataset name (default: "mmnist")')
    parser.add_argument('--batch_size', '-b', default=4, type=int, help='Training batch size')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    
    # Training parameters (optimizer)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--early_stop_epoch', default=-1, type=int,
                        help='Check to early stop after this epoch')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=None, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')

    return parser
