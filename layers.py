"""
Implementation of ReLU-Conv-BN layers
"""
from torch import nn


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=not affine),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ReLUUpConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, output_padding, affine=True):
        super(ReLUUpConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation,
                               output_padding=output_padding, bias=not affine),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)
