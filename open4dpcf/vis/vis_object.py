import torch
from typing import List
from .vis_tools.gaussians_common import point2gaussians_ply, point2gaussians_splat
from .vis_tools.points2range import create_rangeview
from open4dpcf.modules import dvr

class VIS_OBJECT(object):
    def __init__(self) -> None:
        self.is_init_grid = False

    def init_grid(self, pc_range, voxel_size):

        assert isinstance(voxel_size, List)
        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size[2])
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size[1])
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size[0])
        self.input_grid = [1, self.n_height, self.n_length, self.n_width]

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor(voxel_size)[None, None, :], requires_grad=False
        )

        self.is_init_grid = True

    # https://playcanvas.com/model-viewer
    @staticmethod
    def point2gaussian_ply(points, colors, path: str = None, cus_scale = 1):

        point2gaussians_ply(points = points,
                            colors = colors,
                            path = path,
                            cus_scale = cus_scale)
        
        print(f"Save points to gaussians .ply at {path}!")

    # https://antimatter15.com/splat/
    @staticmethod
    def point2gaussian_splat(points, colors, path: str = None, cus_scale = 1):

        point2gaussians_splat(points = points,
                            colors = colors,
                            path = path,
                            cus_scale = cus_scale)
        
        print(f"Save points to gaussians .splat at {path}!")

    @staticmethod
    def p2g(self, input_points):
        assert self.is_init_grid
        if input_points.dim() == 2:
            input_points = input_points.unsqueeze(dim = 0)
        input_points = ((input_points - self.offset) / self.scaler).float()
        input_tindex = input_points.new_zeros(input_points.shape[:2])
        input_occupancy = dvr.init(input_points, input_tindex, self.input_grid) # [4, 2, 45, 700, 700]
        input_occupancy = input_occupancy.squeeze().permute(2,1,0)
        return input_occupancy.detach().cpu().numpy()
    
    @staticmethod
    def point2range(lidar_path, save_path, dataname = 'kitti_od', return_fig=False):

        create_rangeview(lidar_path, save_path, dataname, return_fig)

if __name__ == "__main__":
    # import numpy as np
    point_path = '/data1/data_share/KITTI-360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000005.bin'
    # points = np.fromfile(point_path, dtype=np.float32).reshape(-1,4)[:,:3]

    # P2G = PointsAsGrids(pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    #                     voxel_size=[0.2,0.2,0.2])
    # P2G = P2G.cuda()

    # points = torch.from_numpy(points).cuda()
    # occ = P2G.p2g(points)
    # print(occ.shape)
    # np.save('tools/ALTest/temp/occ.npy', occ)
    # print('Save!')