import torch.nn as nn

def deconv3x3(in_channels, out_channels, stride):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=1,
        bias=False,
    )


def relu(inplace=True):
    return nn.ReLU(inplace=inplace)


def bn(num_features):
    return nn.BatchNorm2d(num_features=num_features)

def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )

class DenseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            deconv3x3(in_channels, 128, stride=2),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=2),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, out_channels, bias=True),
        )

    def forward(self, x):
        return self.block(x)