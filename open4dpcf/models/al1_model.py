import torch
import torch.nn as nn
import torch.nn.functional as F
from open4dpcf.modules import Render_Encoder, Voxelization_Layer, HEDNet, MambaSTL, DenseDecoder

# JIT
from torch.utils.cpp_extension import load

dvr = load("dvr", sources=["open4dpcf/ops/dvr/dvr.cpp", "open4dpcf/ops/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])

class AL1_Model(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super(AL1_Model, self).__init__()

        self.loss_type = configs.data_config['metrics']
        self.input_T = configs.data_config['n_input']
        self.output_T = configs.data_config['n_output']

        # point-wise
        self.render_encoder = Render_Encoder(configs)

        # voxelize
        pc_range = configs.data_config['pc_range']
        voxel_size = configs.data_config['encoding_voxel_size']
        self.voxel_size = configs.data_config['voxel_size']
        self.pc_range = configs.data_config['pc_range']

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=pc_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000])
        self.voxelize_layer = Voxelization_Layer(voxelize_cfg)

        # voxel encoding
        sparse_shape = [int((pc_range[3+i]-pc_range[i]) / voxel_size[i]) for i in range(3)]
        self.input_grid = [int(configs.data_config['n_input'])] + [int((pc_range[3+i]-pc_range[i]) / self.voxel_size) for i in range(3)][::-1]

        sparse_shape[2] = sparse_shape[2] + 1
        vfe_config = configs.vfe_config
        vfe_config.update({'sparse_shape' : sparse_shape})
        self.voxel_encoder = HEDNet(**vfe_config)

        # Spatio-Temporal Predictive Learning
        stl_model_config = configs.stl_model_config

        self.ST = MambaSTL(**stl_model_config)
        self.voxel_decoder = DenseDecoder(in_channels=stl_model_config['embed_dim'],
                                          out_channels=sparse_shape[-1]-1)
        # self.linear = nn.Conv2d(
        #     stl_model_config['embed_dim'], sparse_shape[-1]-1, (3, 3), stride=1, padding=1, bias=True
        # )

    def forward(self, input_points,
                      input_tindex,
                      output_origin,
                      output_points,
                      output_tindex,
                      output_labels):

        input_T = torch.max(input_tindex)
        last_frame = input_points[input_tindex == input_T]
        output_T = torch.max(output_tindex)
        totle_T = input_T + output_T

        # get render_features
        render_points_list = []
        for t in range(self.input_T):
            points = input_points[input_tindex == t]
            render_pw_feature = self.render_encoder(points, t)
            render_points_list.append(torch.cat([points, render_pw_feature], dim=-1))
        for t in range(self.output_T):
            points = output_points[output_tindex == t]
            render_pw_feature = self.render_encoder(points, t + self.input_T)
            render_points_list.append(torch.cat([points, render_pw_feature], dim=-1))
        
        sparse_feats, coords, _ = self.voxelize_layer.voxelize(render_points_list)
        pseudo_batch_size = coords[-1, 0] + 1
        frames = self.voxel_encoder(sparse_feats, coords, pseudo_batch_size) # torch.Size([12, 128, 128, 128, 2])
        frames = frames.dense().permute(0,4,1,2,3)
        L, D, C, H, W = frames.shape
        frames = frames.reshape(L,-1,H,W).unsqueeze(dim=0).contiguous()

        states = None
        next_frames = []
        last_frame = frames[:, -1]
        
        for i in range(self.input_T-1):
            output, states = self.ST(frames[:, i], states) # [1, 128, 128, 128]
            next_frames.append(output)
        for i in range(self.output_T):
            output, states = self.ST(last_frame, states) 
            next_frames.append(output)
            last_frame = output

        next_frames = torch.stack(next_frames, dim=0).squeeze()[-self.output_T:,...]
        voxel_feat = self.voxel_decoder(next_frames).unsqueeze(dim = 0)

        # preprocess input/output points
        input_points = ((input_points - self.offset) / self.scaler).float()
        output_origin = ((output_origin - self.offset) / self.scaler).float()
        output_points = ((output_points - self.offset) / self.scaler).float()
        temp = dvr.init(input_points, input_tindex, self.input_grid) # 

        ret_dict = {}
        if self.training:
            loss = self.loss_type
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(voxel_feat, inplace=True).contiguous() # [1, 6, 40, 512, 512]
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
