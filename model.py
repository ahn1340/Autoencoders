from torch import nn

from layers import ReLUConvBN, ReLUUpConvBN


##### vanilla autoencoder #####
class AutoEncoder(nn.Module):
    def __init__(self):  #TODO: add arguments later
        """
        Convolutional Autoencoder without pooling
        """
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ReLUConvBN(C_in=3, C_out=32, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 64
            ReLUConvBN(C_in=32, C_out=64, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 32
            ReLUConvBN(C_in=64, C_out=128, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 16
            ReLUConvBN(C_in=128, C_out=256, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 8
            ReLUConvBN(C_in=256, C_out=512, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 4
            ReLUConvBN(C_in=512, C_out=512, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 2
            # nn.Flatten(),
            # nn.Linear(512 * 4, hidden_dim),
        )
        self.decoder = nn.Sequential(
            #nn.Linear(hidden_dim, 512 * 4),
            #nn.Unflatten(1, (512, 2, 2)),
            ReLUUpConvBN(C_in=512, C_out=512, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=512, C_out=256, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=256, C_out=128, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=128, C_out=64, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=64, C_out=32, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=32, C_out=3, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
        )

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)

        return decoding

    def encode(self, x):
        return self.encoder(x)


class ConvAutoEncoder(nn.Module):
    def __init__(self, hidden_dim=1024):  #TODO: add arguments later
        """
        Convolutional Autoencoder with pooling.
        """
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ReLUConvBN(C_in=3, C_out=128, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 64
            ReLUConvBN(C_in=128, C_out=256, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 32
            ReLUConvBN(C_in=256, C_out=512, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 16
            ReLUConvBN(C_in=512, C_out=512, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 8
            ReLUConvBN(C_in=512, C_out=1024, kernel_size=(4, 4), stride=2, padding=1, dilation=1, affine=True),  # 4
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024 * 4 * 4),
            nn.Unflatten(1, (1024, 4, 4)),
            ReLUUpConvBN(C_in=1024, C_out=512, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=512, C_out=512, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=512, C_out=256, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=256, C_out=128, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
            ReLUUpConvBN(C_in=128, C_out=3, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0,
                         affine=True),
        )

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)

        return decoding

    def encode(self, x):
        return self.encoder(x)
