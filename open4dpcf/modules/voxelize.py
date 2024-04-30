import torch
import torch.nn.functional as F
from open4dpcf.ops.voxel import Voxelization

class Voxelization_Layer(object):
    def __init__(self, voxelize_cfg):
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)

            feats = feats.sum(
                dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()

        return feats, coords, sizes

if __name__ == "__main__":

    pass

    # voxelize_cfg=dict(
    #     max_num_points=10,
    #     point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    #     voxel_size=[0.075, 0.075, 0.2],
    #     max_voxels=[120000, 160000])
    
    # points = torch.randn(4,1024,10).cuda()

    # res = voxelize(points, voxelize_cfg)

    # print("OK")