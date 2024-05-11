import torch
import torch.nn as nn
import torch.nn.functional as F
from open4dpcf.modules import DenseEncoder, DenseDecoder, dvr
from .utils.al_model_utils import get_grid_mask, get_rendered_pcds, get_clamped_output
from .evals.al_model_evaluation import metrics_dict, compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors


class Ori_Model(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super(Ori_Model, self).__init__()

        self.batch_size = configs.batch_size
        self.loss_type = configs.data_config['metrics']
        self.input_T = configs.data_config['n_input']
        self.output_T = configs.data_config['n_output']
        self.totle_num_frames = self.input_T + self.output_T
        pc_range = configs.data_config['pc_range']
        voxel_size = configs.data_config['voxel_size']

        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)

        self.input_grid = [self.input_T, self.n_height, self.n_length, self.n_width]
        print("input grid:", self.input_grid)

        self.output_grid = [self.output_T, self.n_height, self.n_length, self.n_width]
        print("output grid:", self.output_grid)

        self.pc_range = pc_range
        self.voxel_size = voxel_size

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

        _in_channels = self.input_T * self.n_height
        self.encoder = DenseEncoder(_in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256])

        _out_channels = self.output_T * self.n_height

        self.linear = torch.nn.Conv2d(
            _in_channels, _out_channels, (3, 3), stride=1, padding=1, bias=True
        )
        #
        self.decoder = DenseDecoder(self.encoder.out_channels, _out_channels)

        # eval
        self.metrics_dict = metrics_dict

    def set_threshs(self, threshs):
        self.threshs = torch.nn.parameter.Parameter(
            torch.Tensor(threshs), requires_grad=False
        )

    def forward(self, input_points,
                      input_tindex,
                      output_origin,
                      output_points,
                      output_tindex,
                      output_labels,
                      eval_within_grid=False,
                      eval_outside_grid=False):
        
        if eval_within_grid:
            inner_grid_mask = get_grid_mask(output_points, self.pc_range)
        if eval_outside_grid:
            outer_grid_mask = ~ get_grid_mask(output_points, self.pc_range)

        # preprocess input/output points
        input_points = ((input_points - self.offset) / self.scaler).float()
        output_origin = ((output_origin - self.offset) / self.scaler).float()
        output_points = ((output_points - self.offset) / self.scaler).float()

        # -1: freespace, 0: unknown, 1: occupied
        # N x T1 x H x L x W
        input_occupancy = dvr.init(input_points, input_tindex, self.input_grid) # [4, 2, 45, 700, 700]

        # double check
        N, T_in, H, L, W = input_occupancy.shape
        assert T_in == self.input_T and H == self.n_height

        _input = input_occupancy.reshape(N, -1, L, W)

        # w/ skip connection
        _output = self.linear(_input) + self.decoder(self.encoder(_input))

        #
        output = _output.reshape(N, -1, H, L, W)

        ret_dict = {}
        loss = self.loss_type

        if self.training:
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True).contiguous() # [1, 6, 40, 512, 512]
                if sigma.requires_grad:
                    pred_dist, gt_dist, grad_sigma = dvr.render(
                        sigma,
                        output_origin,
                        output_points,
                        output_tindex,
                        loss
                    )
                    # take care of nans and infs if any
                    invalid = torch.isnan(grad_sigma)
                    grad_sigma[invalid] = 0.0
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    invalid = torch.isinf(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    sigma.backward(grad_sigma)
                else:
                    pred_dist, gt_dist = dvr.render_forward(
                        sigma,
                        output_origin,
                        output_points,
                        output_tindex,
                        self.output_grid,
                        "train"
                    )
                    # take care of nans if any
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0

                pred_dist *= self.voxel_size
                gt_dist *= self.voxel_size

                # compute training losses
                valid = gt_dist >= 0
                count = valid.sum()
                l1_loss = torch.abs(gt_dist - pred_dist)
                l2_loss = ((gt_dist - pred_dist) ** 2) / 2
                absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

                # record training losses
                if count == 0:
                    count = 1
                ret_dict["l1_loss"] = l1_loss[valid].sum() / count
                ret_dict["l2_loss"] = l2_loss[valid].sum() / count
                ret_dict["absrel_loss"] = absrel_loss[valid].sum() / count

            else:
                raise RuntimeError(f"Unknown loss type: {loss}")
            
        else:
            loss_dict = {}
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                pred_dist, gt_dist = dvr.render_forward(
                    sigma, output_origin, output_points, output_tindex, self.output_grid, "test")
                pog = 1 - torch.exp(-sigma)

                pred_dist = pred_dist.detach()
                gt_dist = gt_dist.detach()

            pred_dist *= self.voxel_size
            gt_dist *= self.voxel_size     

            mask = gt_dist > 0
            if eval_within_grid:
                mask = torch.logical_and(mask, inner_grid_mask)
            if eval_outside_grid:
                mask = torch.logical_and(mask, outer_grid_mask)
            count = mask.sum()
            l1_loss = torch.abs(gt_dist - pred_dist)
            l2_loss = ((gt_dist - pred_dist) ** 2) / 2
            absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

            loss_dict["l1_loss"] = l1_loss[mask].sum() / count
            loss_dict["l2_loss"] = l2_loss[mask].sum() / count
            loss_dict["absrel_loss"] = absrel_loss[mask].sum() / count

            ret_dict["gt_dist"] = gt_dist
            ret_dict["pred_dist"] = pred_dist
            ret_dict['pog'] = pog.detach()
            ret_dict["sigma"] = sigma.detach()

            # iterate through the batch
            for j in range(output_points.shape[0]):  # iterate through the batch

                pred_pcds = get_rendered_pcds(
                        output_origin[j].cpu().numpy(),
                        output_points[j].cpu().numpy(),
                        output_tindex[j].cpu().numpy(),
                        ret_dict["gt_dist"][j].cpu().numpy(),
                        ret_dict["pred_dist"][j].cpu().numpy(),
                        self.pc_range,
                        eval_within_grid,
                        eval_outside_grid
                    )

                gt_pcds = get_clamped_output(
                        output_origin[j].cpu().numpy(),
                        output_points[j].cpu().numpy(),
                        output_tindex[j].cpu().numpy(),
                        self.pc_range,
                        ret_dict["gt_dist"][j].cpu().numpy(),
                        eval_within_grid,
                        eval_outside_grid
                    )

                # load predictions
                for k in range(len(gt_pcds)):
                    pred_pcd = pred_pcds[k]
                    gt_pcd = gt_pcds[k]
                    origin = output_origin[j][k].cpu().numpy()

                    # get the metrics
                    self.metrics_dict["count"] += 1
                    self.metrics_dict["chamfer_distance"] += compute_chamfer_distance(pred_pcd, gt_pcd, output_origin.device)
                    self.metrics_dict["chamfer_distance_inner"] += compute_chamfer_distance_inner(pred_pcd, gt_pcd, output_origin.device)
                    l1_error, absrel_error = compute_ray_errors(pred_pcd, gt_pcd, torch.from_numpy(origin), output_origin.device)
                    self.metrics_dict["l1_error"] += l1_error
                    self.metrics_dict["absrel_error"] += absrel_error
            
            return loss_dict, ret_dict

        return ret_dict
        