# Copyright (c) CAIRI AI Lab. All rights reserved

def load_data(dataname, kwargs):

    cfg_dataloader = dict(
        batch_size = kwargs['batch_size'],
        num_workers = kwargs['num_workers'],
        distributed = kwargs['dist']
    )
    data_config = kwargs['data_config']
    data_config.update(**cfg_dataloader)

    if 'nusc' in dataname:
        from .dataloader_nuscenes import load_data
        return load_data(**data_config)
    if 'kitti_od' in dataname:
        from .dataloader_kitti_odometry import load_data
        return load_data(**data_config)

    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
