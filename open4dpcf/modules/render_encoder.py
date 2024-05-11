import torch
import torch.nn as nn
from .planes_field import Planes4D
from .hash_field import HashGrid4D
from .flow_field import FlowField

class Render_Encoder(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super(Render_Encoder, self).__init__()

        self.input_num_frames = configs.data_config['n_input']
        self.totle_num_frames = configs.data_config['n_input'] + configs.data_config['n_output']

        self.planes_encoder = Planes4D(
            grid_dimensions=2,
            input_dim=4,
            output_dim=configs.n_features_per_level_plane,
            resolution=[configs.min_resolution] * 3 + [configs.time_resolution],
            multiscale_res=[2**(n) for n in range(configs.n_levels_plane)],
            concat_features=True,
            decompose=True,
        )

        self.hash_encoder = HashGrid4D(
            base_resolution=configs.base_resolution,
            max_resolution=configs.max_resolution,
            time_resolution=configs.time_resolution,
            n_levels=configs.n_levels_hash,
            n_features_per_level=configs.n_features_per_level_hash,
            log2_hashmap_size=configs.log2_hashmap_size,
        )

        self.if_flow = configs.if_flow
        if self.if_flow:
            self.flow_net = FlowField(
                input_dim=4,
                num_layers=configs.num_layers_flow,
                hidden_dim=configs.hidden_dim_flow,
                num_freqs=6,
            )

        self.out_linear = nn.Sequential(
                nn.Linear(4*configs.n_levels_hash*configs.n_features_per_level_hash, 
                          configs.scene_feat_dim),
                nn.BatchNorm1d(configs.scene_feat_dim), 
                nn.ReLU(True))
        
        self.out_feat_dim = configs.scene_feat_dim
        
    def forward(self, points, t):
        if points.dim == 3:
            render_pw_feature_list = []
            for sub_points in points:
                render_pw_feature_list.append(self.forward_single_batch(sub_points, t))
            render_pw_feature = torch.stack(render_pw_feature_list, dim=0)
        else:
            render_pw_feature = self.forward_single_batch(points, t)
            
        return render_pw_feature


    def forward_single_batch(self, points, t):
        
        frame_idx = int(t)
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).cuda().float() / self.totle_num_frames
        hash_feat_s, hash_feat_d = self.hash_encoder(points, t) # static and dynamic
        if t.shape[0] == 1:
            t = t.repeat(points.shape[0], 1)
        xt = torch.cat([points, t], dim=-1)
        plane_feat_s, plane_feat_d = self.planes_encoder(xt)

        if self.if_flow:
            # integrate neighboring dynamic features
            flow = self.flow_net(xt) # N*6
            hash_feat_1 = hash_feat_2 = hash_feat_d
            plane_feat_1 = plane_feat_2 = plane_feat_d
            if frame_idx < self.totle_num_frames - 1:
                x1 = points + flow[:, :3]
                t1 = torch.tensor((frame_idx + 1) / self.totle_num_frames)
                with torch.no_grad():
                    hash_feat_1 = self.hash_encoder.forward_dynamic(x1, t1)
                t1 = t1.repeat(x1.shape[0], 1).to(x1.device)
                xt1 = torch.cat([x1, t1], dim=-1)
                plane_feat_1 = self.planes_encoder.forward_dynamic(xt1)

            if frame_idx > 0:
                x2 = points + flow[:, 3:]
                t2 = torch.tensor((frame_idx - 1) / self.totle_num_frames)
                with torch.no_grad():
                    hash_feat_2 = self.hash_encoder.forward_dynamic(x2, t2)
                t2 = t2.repeat(x2.shape[0], 1).to(x2.device)
                xt2 = torch.cat([x2, t2], dim=-1)
                plane_feat_2 = self.planes_encoder.forward_dynamic(xt2)

            plane_feat_d = 0.5 * plane_feat_d + 0.25 * (plane_feat_1 + plane_feat_2)
            hash_feat_d = 0.5 * hash_feat_d + 0.25 * (hash_feat_1 + hash_feat_2)

        features = torch.cat([plane_feat_s, plane_feat_d,
                              hash_feat_s, hash_feat_d], dim=-1)
        
        return self.out_linear(features)