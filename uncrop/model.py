import torch
from torch import nn


class UnCropper(nn.Module):
    def __init__(self, uncropped_size, cropped_size, num_kernels=16, kernel_size=10, pool_size=4):
        '''
        uncropped_size and cropped_size are (height, width) tuples
        '''
        super(UnCropper, self).__init__()
        self.uncropped_width, self.uncropped_height = uncropped_size
        self.cropped_width, self.cropped_height = cropped_size

        # 3 layers of convolution and pooling, with relu activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_kernels,
                      kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.ReLU(),

            nn.Conv2d(in_channels=num_kernels,
                      out_channels=num_kernels, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.ReLU(),

            nn.Conv2d(in_channels=num_kernels,
                      out_channels=num_kernels, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.ReLU()
        )

        # post_conv_width = num_kernels * \
        #     (self.cropped_width - kernel_size*3 - pool_size*3 + 6)
        # post_conv_height = num_kernels * \
        #     (self.cropped_height - kernel_size*3 - pool_size*3 + 6)

        # inverse 3-layer deconvolution and unpooling, with relu
        self.deConv = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=pool_size),
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=num_kernels,
                               kernel_size=kernel_size),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=pool_size),
            nn.ConvTranspose2d(in_channels=num_kernels,
                               out_channels=num_kernels, kernel_size=kernel_size),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=pool_size),
            nn.ConvTranspose2d(in_channels=num_kernels,
                               out_channels=3, kernel_size=kernel_size),
            nn.ReLU()
        )

    def forward(self, img):
        hidden = self.conv(img)
        unCropped = self.deConv(hidden)
        return unCropped
