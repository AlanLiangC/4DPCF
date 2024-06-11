from pathlib import Path
import tqdm
import numpy as np
from open4dpcf.datasets import dataset_parameters
from .points2pano import LiDAR_2_Pano

def create_rangeview(lidar_path, save_path, dataname = 'kitti_od', return_fig=False):

    if dataname == 'kitti_od':
        points_dim = 4
    elif dataname == 'nusc':
        points_dim = 5
    else:
        raise NotImplementedError(f'No dataset of: {dataname}')
    
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    if not isinstance(lidar_path, list):
        lidar_paths = [lidar_path]

    for lidar_path in lidar_paths:
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        pano = LiDAR_2_Pano(point_cloud, dataset_parameters[dataname]['vis']['rangeview'])
        frame_name = lidar_path.split("/")[-1]
        suffix = frame_name.split(".")[-1]
        frame_name = frame_name.replace(suffix, "npy")
        np.save(save_path / frame_name, pano)

        if return_fig:
            import cv2
            pano = (pano * 255).astype(np.uint8)
            frame_name = frame_name.split("/")[-1]
            suffix = frame_name.split(".")[-1]
            frame_name = frame_name.replace(suffix, "png")
            cv2.imwrite(
                str(save_path / frame_name),
                cv2.applyColorMap(pano, 20),
            )
    

    

