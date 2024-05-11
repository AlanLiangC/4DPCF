import torch.nn as nn
from functools import partial

from .sparse_class import spconv

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class SEDBlock(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_SBB, norm_fn, indice_key):
        super(SEDBlock, self).__init__()

        first_block = post_act_block(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        block_list = [first_block if stride > 1 else nn.Identity()]
        for _ in range(num_SBB):
            block_list.append(
                SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)


class SEDLayer(spconv.SparseModule):

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, norm_fn, indice_key):
        super().__init__()

        assert down_stride[0] == 1 # hard code
        assert len(down_kernel_size) == len(down_stride) == len(num_SBB)

        self.encoder = nn.ModuleList()
        for idx in range(len(down_stride)):
            self.encoder.append(
                SEDBlock(dim, down_kernel_size[idx], down_stride[idx], num_SBB[idx], norm_fn, f"{indice_key}_{idx}"))

        downsample_times = len(down_stride[1:])
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx, kernel_size in enumerate(down_kernel_size[1:]):
            self.decoder.append(
                post_act_block(
                    dim, dim, kernel_size, norm_fn=norm_fn, conv_type='inverseconv',
                    indice_key=f'spconv_{indice_key}_{downsample_times - idx}'))
            self.decoder_norm.append(norm_fn(dim))

    def forward(self, x):
        features = []
        for conv in self.encoder:
            x = conv(x)
            features.append(x)

        x = features[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, features[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, x.features + up_x.features)
            x = replace_feature(x, norm(x.features))
        return x


class HEDNet(nn.Module):

    def __init__(self, 
                 input_channels, 
                 output_channels,
                 sparse_shape, 
                 num_layers,
                 num_SBB,
                 down_kernel_size,
                 down_stride,
                 **kwargs):
        super().__init__()

        self.sparse_shape = sparse_shape
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        dim = output_channels

        # [1888, 1888, 41] -> [944, 944, 21]
        self.conv1 = spconv.SparseSequential(
            post_act_block(input_channels, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='stem'),
            post_act_block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
        )

        # [944, 944, 21] -> [472, 472, 11]
        self.conv2 = spconv.SparseSequential(
            SEDLayer(32, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer2'),
            post_act_block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        )

        #  [472, 472, 11] -> [236, 236, 11]
        self.conv3 = spconv.SparseSequential(
            SEDLayer(64, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer3'),
            post_act_block(64, dim, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='spconv3', conv_type='spconv'),
        )

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            conv = SEDLayer(dim, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key=f'sedlayer{idx+4}')
            self.layers.append(conv)

        # [236, 236, 11] -> [236, 236, 5] --> [236, 236, 2]
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv4'),
            norm_fn(dim),
            nn.ReLU(),
            spconv.SparseConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv5'),
            norm_fn(dim),
            nn.ReLU(),
        )

        self.out_feat_dim = dim

    def forward(self, voxel_features, coors, batch_size):

        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, 
                                    coors,
                                    self.sparse_shape, 
                                    batch_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for conv in self.layers:
            x = conv(x)
        x = self.conv_out(x)

        return x
