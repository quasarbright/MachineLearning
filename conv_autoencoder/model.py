import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, num_kernels=64, kernel_size=6):
        super(Autoencoder, self).__init__()
        # 3 layers of convolution, with relu activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_kernels,
                      kernel_size=kernel_size),
            nn.ReLU(),

            nn.Conv2d(in_channels=num_kernels,
                      out_channels=num_kernels, kernel_size=kernel_size),
            nn.ReLU(),

            nn.Conv2d(in_channels=num_kernels,
                      out_channels=num_kernels, kernel_size=kernel_size),
            nn.ReLU()
        )
        # inverse 3-layer deconvolution, with relu
        self.restore = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=num_kernels,
                               kernel_size=kernel_size),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=num_kernels,
                               out_channels=num_kernels, kernel_size=kernel_size),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=num_kernels,
                               out_channels=3, kernel_size=kernel_size),
            nn.Tanh()
        )

    def forward(self, img):
        conved = self.conv(img)
        restored = self.restore(conved)
        return restored
    
    def lerp(self, img1, img2, r):
        '''
        linearly interpolate the hidden states
        for img1 and img2 according to r
        r in [0,1]
        '''
        h1 = self.conv(img1)
        h2 = self.conv(img2)
        interpolated = h1 * r + h2 * (1-r)
        return self.restore(interpolated)
