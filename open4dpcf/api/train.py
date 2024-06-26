import os
import os.path as osp
import time
import logging
import json
import numpy as np
from typing import Dict, List
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
import torch.distributed as dist

from open4dpcf.core import Hook, metric, Recorder, get_priority, hook_maps
from open4dpcf.utils import (init_dist, get_dist_info, print_log, check_dir, collect_env,
                             init_random_seed, set_seed, get_dataset, weights_to_cpu)
from open4dpcf.methods import method_maps

class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, dataloaders=None):
        """Initialize experiments (non-dist as an example)"""
        self.args = args
        self.config = self.args.__dict__
        self.device = self.args.device
        self.method = None
        self.args.method = self.args.method.lower()
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = self.config['epoch']
        self._max_iters = None
        self._hooks: List[Hook] = []
        self._rank = 0
        self._world_size = 1
        self._dist = self.args.dist
        self._early_stop = self.args.early_stop_epoch

        self._preparation(dataloaders)

    def _acquire_device(self):
        """Setup devices"""
        if self.args.use_gpu:
            self._use_gpu = True
            if self.args.dist:
                device = f'cuda:{self._rank}'
                torch.cuda.set_device(self._rank)
                print_log(f'Use distributed mode with GPUs: local rank={self._rank}')
            else:
                device = torch.device(f'cuda:{self.args.local_rank}')
                print_log(f'Use non-distributed mode with GPU: {device}')
        else:
            self._use_gpu = False
            device = torch.device('cpu')
            print_log('Use CPU')
            if self.args.dist:
                assert False, "Distributed training requires GPUs"
        return device

    def _preparation(self, dataloaders=None):
        """Preparation of environment and basic experiment setups"""
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.args.local_rank)

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher != 'none' or self.args.dist:
            self._dist = True
        if self._dist:
            assert self.args.launcher != 'none'
            dist_params = dict(backend='nccl', init_method='env://')
            if self.args.launcher == 'slurm':
                dist_params['port'] = self.args.port
            init_dist(self.args.launcher, **dist_params)
            self._rank, self._world_size = get_dist_info()
            # re-set gpu_ids with distributed training mode
            self._gpu_ids = range(self._world_size)
        self.device = self._acquire_device()
        if self._early_stop <= self._max_epochs // 5:
            self._early_stop = self._max_epochs * 2

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
            else self.args.ex_name.split(self.args.res_dir+'/')[-1])
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        if self._rank == 0:
            check_dir(self.path)
            check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        if self._rank == 0:
            with open(sv_param, 'w') as file_obj:
                json.dump(self.args.__dict__, file_obj)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            prefix = 'train' 
            if self.args.test:
                prefix = 'test' 
            if self.args.inference:
                prefix = 'inference' 
            logging.basicConfig(level=logging.INFO,
                                filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                                filemode='a', format='%(asctime)s - %(message)s')

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        if self._rank == 0:
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # set random seeds
        if self._dist:
            seed = init_random_seed(self.args.seed)
            seed = seed + dist.get_rank() if self.args.diff_seed else seed
        else:
            seed = self.args.seed
        set_seed(seed)

        # prepare data
        self._get_data(dataloaders)
        # build the method
        self._build_method()

        self._build_hook()
        # resume traing
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:
            self._load(name=self.args.resume_from)
        self.call_hook('before_run')

    def call_hook(self, fn_name: str) -> None:
        """Run hooks by the registered names"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def _get_hook_info(self):
        """Get hook information in each stage"""
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self._hooks:
            priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            self.train_loader, self.vali_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            self.train_loader, self.vali_loader = dataloaders

        self._max_iters = self._max_epochs * len(self.train_loader)

    def _build_method(self):
        self.steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, self.steps_per_epoch)
        self.method.model.eval()
        # setup ddp training
        if self._dist:
            self.method.model.cuda()
            if self.args.torchscript:
                self.method.model = torch.jit.script(self.method.model)
            self.method._init_distributed()

    def _build_hook(self):
        for k in self.args.__dict__:
            if k.lower().endswith('hook'):
                hook_cfg = self.args.__dict__[k].copy()
                priority = get_priority(hook_cfg.pop('priority', 'NORMAL'))
                hook = hook_maps[k.lower()](**hook_cfg)
                if hasattr(hook, 'priority'):
                    raise ValueError('"priority" is a reserved attribute for hooks')
                hook.priority = priority  # type: ignore
                # insert the hook to a sorted list
                inserted = False
                for i in range(len(self._hooks) - 1, -1, -1):
                    if priority >= self._hooks[i].priority:  # type: ignore
                        self._hooks.insert(i + 1, hook)
                        inserted = True
                        break
                if not inserted:
                    self._hooks.insert(0, hook)

    def _save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self._dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        """Loading models from the checkpoint"""
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        # try:
        self._load_from_state_dict(checkpoint['state_dict'])
        # except:
        #     self._load_from_state_dict(checkpoint)
        if checkpoint.get('epoch', None) is not None:
            self._epoch = checkpoint['epoch']
            self.method.model_optim.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def _load_from_state_dict(self, state_dict):
        if self._dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)

    def train(self):
        """Training loops of STL methods"""
        recorder = Recorder(verbose=True, early_stop_time=min(self._max_epochs // 10, 10))
        num_updates = self._epoch * self.steps_per_epoch
        early_stop = False
        self.call_hook('before_train_epoch')

        eta = 1.0  # PredRNN variants
        for epoch in range(self._epoch, self._max_epochs):
            if self._dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean, eta = self.method.train_one_epoch(self, self.train_loader,
                                                                      epoch, num_updates, eta)

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = 0.0
                    # vali_loss = self.vali()
                if self._rank == 0:
                    print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    early_stop = recorder(vali_loss, self.method.model, self.path)
                    self._save(name='latest')
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
            if epoch > self._early_stop and early_stop:  # early stop training
                print_log('Early stop training at f{} epoch'.format(epoch))

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self._load_from_state_dict(torch.load(best_model_path))
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def vali(self):
        """A validation loop during training"""
        self.call_hook('before_val_epoch')
        results, eval_log = self.method.vali_one_epoch(self, self.vali_loader)
        self.call_hook('after_val_epoch')

        if self._rank == 0:
            print_log('val\t '+eval_log)

        return results[self.method.loss_type].mean()
    
    def test(self):
        """A testing loop of STL methods"""
        assert self.args.test
        if self.args.test_from is not None:
            best_model_path = self.args.test_from
        else:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
        checkpoint = torch.load(best_model_path, 
                                map_location=lambda storage, 
                                loc: storage.cuda(self.args.local_rank))
        self._load_from_state_dict(checkpoint)

        self.call_hook('before_val_epoch')
        _, loss_log = self.method.vali_one_epoch(self, self.vali_loader)
        self.call_hook('after_val_epoch')

        if self._rank == 0:
            print_log(loss_log)

        eval_log = ""
        metrics_dict = self.method.model.metrics_dict
        count = metrics_dict.pop('count')
        for k, v in metrics_dict.items():
            v = v / count
            eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
            eval_log += eval_str
        print_log(eval_log)

        # return metrics_dict

    def inference(self):
        """A inference loop of STL methods"""
        assert self.args.inference
        if self.args.model_name != 'rt':
            if isinstance(self.method.model, torch.nn.Module):
                if self.args.test_from is not None:
                    best_model_path = self.args.test_from
                else:
                    best_model_path = osp.join(self.path, 'checkpoint.pth')
                checkpoint = torch.load(best_model_path, 
                                        map_location=lambda storage, 
                                        loc: storage.cuda(self.args.local_rank))
                self._load_from_state_dict(checkpoint)

        batch_data, inference_dict = self.method.inference_one_batch(self, self.vali_loader, self.args.inference_idx)

        save_path = osp.join(self.path.split('/')[0], 'vis_results', self.args.dataname, self.args.forecasting_time, str(self.args.inference_idx), self.args.model_name)
        check_dir(save_path)

        # save results
        batch_size = len(batch_data[0])
        batch_data = self.method._tocpu(batch_data)
        input_points, input_tindex = batch_data[1:3]
        output_origin, output_points, output_tindex = batch_data[3:6]

        # input_points
        save_input_path = osp.join(save_path, 'input')
        check_dir(save_input_path)
        for batch_idx in range(batch_size):
            for i in range(self.args.data_config['n_input']):
                np.savetxt(osp.join(save_input_path, f'{batch_idx}_{i}.txt'), 
                        input_points[batch_idx][input_tindex[batch_idx] == i])
                
        # out_gt_points
        save_output_path = osp.join(save_path, 'gt_output')
        check_dir(save_output_path)
        for batch_idx in range(batch_size):
            for i in range(self.args.data_config['n_output']):
                np.savetxt(osp.join(save_output_path, f'{batch_idx}_{i}.txt'), 
                        output_points[batch_idx][output_tindex[batch_idx] == i])

        # out_pred_points
        save_output_path = osp.join(save_path, 'pred_output')
        check_dir(save_output_path)
        pred_points = inference_dict['pred_pcd']
        for batch_idx in range(batch_size):
            for i in range(self.args.data_config['n_output']):
                np.savetxt(osp.join(save_output_path, f'{batch_idx}_{i}.txt'), 
                        pred_points[batch_idx][i].numpy())

        # output origin
        np.save(osp.join(save_path, 'output_origin.npy'), output_origin)
        # out_pred_occ
        if 'pog' in inference_dict:
            occ = inference_dict['pog'].permute(0,1,4,3,2)
            np.save(osp.join(save_path, 'occ.npy'), occ.detach().cpu().numpy())