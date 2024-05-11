# Model
eval_within_grid = False
eval_outside_grid = False

with_scene = True
scene_feat_dim = 16
time_resolution = 5
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

# Dynamic VFE
dynamic_vfe_config = dict(
    use_norm = True,
    with_distance = False,
    use_absolute_xyz = True,
    num_filters = [8, 16]
)

# Voxel feature encoder config
encoding_voxel_size = [0.1, 0.1, 0.2]
vfe_config = dict(
    output_channels = 32,
    num_layers = 2,
    num_SBB = [2,1,1],
    down_kernel_size = [3,3,3],
    down_stride = [1,2,2]
)

# STL
# Mamba
# stl_model_config = dict(
#     model_name = 'mamba',
#     img_size = 128,
#     in_chans = 64,
#     patch_size = 2,
#     embed_dim = 32,
#     depths = 2,
#     num_heads = 4,
#     window_size = 4,
#     drop_rate = 0.,
#     attn_drop_rate = 0.,
#     drop_path_rate = 0.1,
# )

# Simvp
stl_model_config = dict(
    model_name = 'simvp',
    in_shape = [time_resolution,64,128,128],
    N_S = 2,
    N_T = 4,
    hid_S = 16,
    hid_T = 64,
    model_type = 'gSTA'
)