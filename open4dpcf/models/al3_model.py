import torch
import torch.nn as nn
import torch.nn.functional as F
from open4dpcf.modules import dvr, DenseEncoder, DenseDecoder

class AL3_Model(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super(AL3_Model, self).__init__()

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

        _in_channels = self.n_height
        self.encoder = DenseEncoder(_in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 128])
        
        stl_model_config = configs.stl_model_config
        self.ST = MambaSTL(**stl_model_config)


        _out_channels = self.n_height

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

        _input = input_occupancy.reshape(-1, H, L, W)

        # w/ skip connection
        frames = self.encoder(_input) # torch.Size([24, 128, 128, 128])
        _, FD, FL, FW = frames.shape
        frames = frames.reshape(N, T_in, FD, FL, FW)
        # With STL
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

        next_frames = torch.stack(next_frames, dim=0).squeeze()[-self.output_T:,...].permute(1,0,2,3,4)
        next_frames = next_frames.reshape(-1, FD, FL, FW)

        _output = self.linear(_input) + self.decoder(next_frames)

        output = _output.reshape(N, T_in, H, L, W)

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