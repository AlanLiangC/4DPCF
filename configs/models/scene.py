####### Model #######
scene_feat_dim = 64
time_resolution = 25

# Dynamic VFE
dynamic_vfe_config = dict(
    use_norm = True,
    with_distance = False,
    use_absolute_xyz = True,
    num_filters = [32, 2]
)

# Voxel encoder config
vfe_config = dict(
    input_channels = 64,
    output_channels = 64,
    num_layers = 2,
    num_SBB = [2,1,1],
    down_kernel_size = [3,3,3],
    down_stride = [1,2,2]
)

# Hash encoding
base_resolution = 512
max_resolution = 32768
n_levels_hash = 8
n_features_per_level_hash = 4
log2_hashmap_size = 19

# Plane encoding
n_features_per_level_plane = 8
min_resolution = 64
n_levels_plane = 4

# Flow encoding
if_flow = False
num_layers_flow = 8
hidden_dim_flow = 128