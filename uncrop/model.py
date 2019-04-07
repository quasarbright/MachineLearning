import torch
from torch import nn


class UnCropper(nn.Module):
    def __init__(self, crop_ratio, uncropped_size, cropped_size, num_kernels=64, kernel_size=6, latent_dims=16):
        '''
        uncropped_size and cropped_size are (height, width) tuples
        '''
        self.config = locals()
        super(UnCropper, self).__init__()
        self.crop_ratio = crop_ratio
        self.uncropped_width, self.uncropped_height = uncropped_size
        self.cropped_width, self.cropped_height = cropped_size

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
        
        # linear latent layer
        # self.post_conv_width = (self.cropped_width - kernel_size*3 + 3)
        # self.post_conv_height = (self.cropped_height - kernel_size*3 + 3)
        # self.post_conv_flat_len = num_kernels * self.post_conv_width * self.post_conv_height

        # self.linear = nn.Sequential(
        #     nn.Linear(self.post_conv_flat_len, latent_dims),
        #     nn.Linear(latent_dims, self.post_conv_flat_len)
        # )


        # inverse 3-layer deconvolution, with relu
        self.restore = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_kernels, out_channels=num_kernels,
                               kernel_size=kernel_size),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=num_kernels,
                               out_channels=num_kernels, kernel_size=kernel_size),
            nn.ReLU(),

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
        post_conv = self.conv(img)
        batch_size, post_conv_channels, post_conv_height, post_conv_width = post_conv.shape
        # post_conv_flat = post_conv.view(batch_size, post_conv_channels*post_conv_height*post_conv_width) # flatten out the convolved image
        # post_linear = self.linear(post_conv_flat)
        # post_linear_reshaped = post_linear.view(
        #     batch_size, post_conv_channels, post_conv_height, post_conv_width)
        post_restore = self.restore(post_conv)
        post_generate = self.generate(post_restore)
        return post_generate
