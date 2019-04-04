import torch
from torch import nn


class UnCropper(nn.Module):
    def __init__(self, uncropped_size, cropped_size, num_kernels=16, kernel_size=4, pool_size=2):
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
            nn.MaxPool2d(kernel_size=pool_size, return_indices=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=num_kernels,
                      out_channels=num_kernels, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=pool_size, return_indices=True),
            nn.ReLU()
        )
        # sigmoid for final activation just to restrict output space to [0,1] explicitly

        # post_conv_width = num_kernels * \
        #     (self.cropped_width - kernel_size*3 - pool_size*3 + 6)
        # post_conv_height = num_kernels * \
        #     (self.cropped_height - kernel_size*3 - pool_size*3 + 6)

        # inverse 3-layer deconvolution and unpooling, with relu
        self.restore = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=pool_size),
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=num_kernels,
                               kernel_size=kernel_size),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=pool_size),
            nn.ConvTranspose2d(in_channels=num_kernels,
                               out_channels=num_kernels, kernel_size=kernel_size),
            nn.ReLU()
        )
        # output is same size as original image, but has num_kernels channels instead of 3

        # uncrop and generate outside of the image
        # TODO make it use multiple deconv layers
        gen_kernel_width = self.uncropped_width-self.cropped_width+1
        gen_kernel_height = self.uncropped_height-self.cropped_height+1
        gen_kernel_size = (gen_kernel_height, gen_kernel_width)
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=3,
                               kernel_size=gen_kernel_size),
            nn.Sigmoid(),
            # use sigmoid to explicitly force output space to be 0 to 1
        )
        # output shape (ih-1+kh, ih-1+kw)

    def forward(self, img):
        # convolve
        hidden = img
        indices_all = []
        # TODO switch to explicit layer type checking
        # or make a conv layer module with conv, pool ind, and relu that returns both
        for layer in self.conv:
            if isinstance(layer, nn.MaxPool2d):
                # this is a pooling layer
                hidden, indices = layer(hidden)
                indices_all.append(indices)
            else:
                hidden = layer(hidden)

        # deconvolve to original size
        restored = hidden
        for layer in self.restore:
            if isinstance(layer, nn.MaxUnpool2d):
                # unpool layer
                indices = indices_all.pop()
                restored = layer(restored, indices)
            else:
                restored = layer(restored)

        # deconvolve beyond original size
        generated = self.generate(restored)
        return generated
