dataset_parameters = {
    'nusc': {
        '1s' : {
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'encoding_voxel_size': [0.1, 0.1, 0.2],
            'voxel_size': 0.2,
            'n_input': 2,
            'input_step': 1,
            'n_output': 2,
            'input_step': 1,
            'metrics': 'l1'},

        '3s': {
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'encoding_voxel_size': [0.1, 0.1, 0.2],
            'voxel_size': 0.2,
            'n_input': 6,
            'input_step': 1,
            'n_output': 6,
            'input_step': 1,
            'metrics': 'l1'
    }
}
}