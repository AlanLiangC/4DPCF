import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm
import numpy as np
from open4dpcf.models import model_maps
from open4dpcf.utils import reduce_tensor, ProgressBar
from .base_method import Base_method

class AL1(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super().__init__(args, device, steps_per_epoch)
        self.args = args
        self.loss_type = args.data_config['metrics']
        self.model = self._build_model(args)
        if args.model_name != 'rt':
            self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)

    def _build_model(self, args):
        model = model_maps[args.model_name](args)
        model = model.to(self.device)
        return model

    def _predict(self, batch_data, **kwargs):

        batch_data = self._togpu(batch_data, self.device)
        input_points, input_tindex = batch_data[1:3]
        output_origin, output_points, output_tindex = batch_data[3:6]
        if self.args.dataname == "nusc":
            output_labels = batch_data[6]
        else:
            output_labels = None

        ret_dict = self.model(
                    input_points,
                    input_tindex,
                    output_origin,
                    output_points,
                    output_tindex,
                    output_labels=output_labels,
                    **kwargs)
    
        return ret_dict

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.args.sched != "ori_step":
            if self.by_epoch:
                self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        # iter_mem = 0
        for batch_data in train_pbar:
            # if iter_mem < 4170:
            #     iter_mem += 1
            #     continue
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                ret_dict = self._predict(batch_data)
                loss = ret_dict[f"{self.loss_type}_loss"].mean()

            if not self.dist:
                losses_m.update(loss.item(), self.args.batch_size)

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                # loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), self.args.batch_size)

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if self.args.sched == "ori_step":
            assert self.by_epoch
            self.scheduler.step()

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
    
    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length

        for idx, batch_data in enumerate(data_loader):
            try:
                with torch.no_grad():
                    loss_dict, _ = self._predict(batch_data)
                    for k in loss_dict.keys():
                        loss_dict[k] = loss_dict[k].reshape(1)
                    results.append(loss_dict)
            except:
                print(idx)
            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k].detach().cpu().numpy() for batch in results], axis=0)
        return results_all

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)
        else:
            results = self._nondist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)

        eval_log = ""
        for k, v in results.items():
            v = v.mean()
            if k != "loss":
                eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
                eval_log += eval_str

        return results, eval_log

    def inference_one_batch(self, runner, test_loader, inference_idx, **kwargs):
        """Inference one batch with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.
            inference_idx: the index of batch id

        Returns:
            saved results
        """
        self.model.eval()
        length = len(test_loader.dataset)
        if inference_idx >= length:
            inference_idx = inference_idx - length

        for idx, batch_data in enumerate(test_loader):
            if idx != inference_idx:
                continue
            with torch.no_grad():
                inference_dict = self._predict(batch_data, inference_mode = True)
                break        

        return batch_data, inference_dict