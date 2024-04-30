import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm
from open4dpcf.models import AL1_Model
from open4dpcf.utils import reduce_tensor
from .base_method import Base_method

class AL1(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super().__init__(args, device, steps_per_epoch)
        self.args = args
        self.loss_type = args.data_config['metrics']
        self.model = self._build_model(args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)

    def _build_model(self, args):
        model = AL1_Model(args).to(self.device)
        model = nn.DataParallel(model)
        return model

    def _predict(self, batch_data, **kwargs):

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
                    output_labels=output_labels,)
        
        return ret_dict

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_data in train_pbar:
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

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta