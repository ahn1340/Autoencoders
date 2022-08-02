"""
Implementation of ReLU-Conv-BN layers
"""
import torch
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


class VQLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zq, emb_dict):
        N, C, H, W = zq.size()
        zq = zq.view((N, C, H*W)).transpose(1, 2)  # [N, H*W, C]
        dist = torch.cdist(zq, emb_dict)  # [N, H*W, C] x [K, C] -> [N, H*W, K]
        argmins = torch.argmin(dist, dim=2)
        ze = emb_dict[argmins].transpose(1, 2).view(N, C, H, W)  # construct z_e
        return ze

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class ResidualLayer(nn.Module):
    def __init__(self, C_in, C_hidden, C_out):
        super(ResidualLayer, self).__init__()
        self.residual_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_hidden, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(C_hidden, C_out, kernel_size=(1, 1), padding=0),
        )

    def forward(self, x):
        return x + self.residual_layer(x)


