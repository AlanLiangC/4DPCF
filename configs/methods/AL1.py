model_name = 'AL1'
# Dataset
forecasting_time = '1s' # for nuscenes

####### Model #######
scene_feat_dim = 64
time_resolution = 25

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
num_layers_flow = 8
hidden_dim_flow = 128

# Voxel encoder config
vfe_config = dict(
    input_channels = scene_feat_dim + 3,
    output_channels = 64,
    num_layers = 2,
    num_SBB = [2,1,1],
    down_kernel_size = [3,3,3],
    down_stride = [1,2,2]
)

# STL model config
stl_model_config = dict(
    img_size = 128,
    in_chans = 128,
    patch_size = 2,
    embed_dim = 128,
    depths = 6,
    num_heads = 8,
    window_size = 4,
    drop_rate = 0.,
    attn_drop_rate = 0.,
    drop_path_rate = 0.1,
)

####### Training parameters #######
batch_size = 1
epoch = 15
opt = "Adam"
lr = 5e-4
weight_decay = 0.1

sched = "step"
decay_epoch = 5
warmup_lr = 1e-6