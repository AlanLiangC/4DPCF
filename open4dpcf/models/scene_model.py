import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from open4dpcf.modules import (DenseEncoder, DenseDecoder, DynamicVoxelVFE, Render_Encoder,
                                SparseClass, dvr)

class Scene_Model(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super(Scene_Model, self).__init__()

        self.batch_size = configs.batch_size
        self.loss_type = configs.data_config['metrics']
        self.input_T = configs.data_config['n_input']
        self.output_T = configs.data_config['n_output']
        self.totle_num_frames = self.input_T + self.output_T
        pc_range = np.array(configs.data_config['pc_range'])
        voxel_size = configs.data_config['voxel_size']

        self.render_encoder = Render_Encoder(configs)

        self.grid_size = (pc_range[3:6] - pc_range[0:3]) / voxel_size
        self.sparse_shape = [int(i) for i in self.grid_size[::-1]]
        dynamic_vfe_config=dict(
            model_cfg = configs.dynamic_vfe_config,
            num_point_features = self.render_encoder.out_feat_dim + 3,
            voxel_size=[voxel_size]*3,
            grid_size=self.grid_size,
            point_cloud_range=pc_range)
        self.voxelize_layer = DynamicVoxelVFE(**dynamic_vfe_config)

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

        _in_channels = self.input_T * self.n_height * (1+self.voxelize_layer.get_output_feature_dim())
        self.encoder = DenseEncoder(_in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256])

        _out_channels = self.output_T * self.n_height

        self.linear = torch.nn.Conv2d(
            _in_channels, _out_channels, (3, 3), stride=1, padding=1, bias=True
        )
        #
        self.decoder = DenseDecoder(self.encoder.out_channels, _out_channels)

    def set_threshs(self, threshs):
        self.threshs = torch.nn.parameter.Parameter(
            torch.Tensor(threshs), requires_grad=False
        )

    def forward(self, input_points,
                      input_tindex,
                      output_origin,
                      output_points,
                      output_tindex,
                      output_labels):

        render_points_list = []
        # get render_features
        for batch_idx in range(self.batch_size):
            sub_input_points = input_points[batch_idx]
            sub_input_tindex = input_tindex[batch_idx]
            for t in range(self.input_T):
                _points = sub_input_points[sub_input_tindex == t]
                render_pw_feature = self.render_encoder(_points, t)
                points = torch.cat([_points, render_pw_feature], dim=-1)
                points = F.pad(points, (1,0), 'constant', self.input_T*batch_idx+t)
                render_points_list.append(points)
        
        rendered_points = torch.vstack(render_points_list)
        sparse_feats, coords = self.voxelize_layer(rendered_points)
        sparse_tensor = SparseClass.to_sparse_tensor(sparse_feats,
                                                     coords.int(),
                                                     sparse_shape=self.sparse_shape,
                                                     batch_size=self.batch_size*self.input_T)

        scene_dense_tensor = sparse_tensor.dense()
        _, C, H, L, W = scene_dense_tensor.shape
        scene_dense_tensor = scene_dense_tensor.view(self.batch_size, self.input_T, C, H, L, W)
        # preprocess input/output points
        input_points = ((input_points - self.offset) / self.scaler).float()
        output_origin = ((output_origin - self.offset) / self.scaler).float()
        output_points = ((output_points - self.offset) / self.scaler).float()

        # -1: freespace, 0: unknown, 1: occupied
        # N x T1 x H x L x W
        input_occupancy = dvr.init(input_points, input_tindex, self.input_grid) # [4, 2, 45, 700, 700]
        input_occupancy = input_occupancy.unsqueeze(dim = 2)
        input_occupancy = torch.cat([input_occupancy, scene_dense_tensor], dim=2)
        # double check
        N, T_in, C, H, L, W = input_occupancy.shape
        assert T_in == self.input_T and H == self.n_height

        _input = input_occupancy.reshape(N, -1, L, W)
        # w/ skip connection
        _output = self.linear(_input) + self.decoder(self.encoder(_input))

        output = _output.reshape(N, -1, H, L, W)

        ret_dict = {}
        if self.training:
            loss = self.loss_type
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
            
            return ret_dict