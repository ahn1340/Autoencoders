"""
Implementation of VQ-VAE from scratch using PyTorch
"""
import torch
from torch import nn
import torch.nn.functional as F

from layers import ReLUConvBN, ReLUUpConvBN, VQLayer, ResidualLayer


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            ReLUConvBN(C_in=3, C_out=dim//2, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),
            ReLUConvBN(C_in=dim//2, C_out=dim, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),
            ResidualLayer(C_in=dim, C_hidden=dim, C_out=dim),
            ResidualLayer(C_in=dim, C_hidden=dim, C_out=dim),
        )

    def forward(self, x):
        """
        Return z(x)
        """
        return self.encoder(x)

    def encode(self, x):
        """
        Used for training. Return z(x) and sg(z(x))
        """
        z = self.encoder(x)
        return z


class Decoder(nn.Module):
    def __init__(self, dim=128):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            ResidualLayer(C_in=dim, C_hidden=dim, C_out=dim),
            ResidualLayer(C_in=dim, C_hidden=dim, C_out=dim),
            ReLUUpConvBN(C_in=dim, C_out=dim//2, kernel_size=(4, 4), stride=2,
                         padding=1, dilation=1, output_padding=0, affine=True),
            ReLUUpConvBN(C_in=dim//2, C_out=3, kernel_size=(4, 4), stride=2,
                         padding=1, dilation=1, output_padding=0, affine=True),
        )

    def forward(self, x):
        """
        given embeddings, return reconstruction
        """
        return self.decoder(x)


class VQVAE(nn.Module):
    def __init__(self, k=512, dim=256):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(dim=dim)
        self.decoder = Decoder(dim=dim)
        self.vq_layer = VQLayer
        self.emb_dict = nn.Parameter(torch.randn((k, dim)))
        self.sg = nn.Identity()

    def encode(self, x):
        """
        Encode given image by computing ze
        """
        zq = self.encoder.encode(x)
        return self.vq_layer.apply(zq, self.emb_dict)

    def forward(self, x):
        """
        reconstruct x
        """
        ze = self.encode(x)
        return self.decoder(ze)

    def loss(self, x):
        """
        Compute reconstruction loss, vq loss and commitment loss
        """
        zq = self.encoder.encode(x)
        ze = self.vq_layer.apply(zq, self.emb_dict)
        zq_sg = self.sg(zq)
        ze_sg = self.sg(ze)
        recon = self.decoder(ze)

        recon_loss = F.mse_loss(recon, x)
        vq_loss = F.mse_loss(ze, zq_sg)
        commitment_loss = F.mse_loss(zq, ze_sg)

        # TODO: add beta
        return recon_loss + vq_loss + commitment_loss














