import math
import torch
from torch import nn
from myutils import *
img_width, img_height = img_size

class Generator(nn.Module):
    def __init__(self, noise_size=64, hidden_size_sqrt=10, num_channels=64):
        hidden_size = hidden_size_sqrt ** 2
        self.hidden_size_sqrt = hidden_size_sqrt
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.LeakyReLU()
        )
        # depth of 3
        # need to go from h_sqrt^2 to img_width^2
        # h_sqrt + 3*kernel_size - 3 = img_width
        # kernel_size = (img_width - h_sqrt + 3) / 3
        kernel_size = int((img_width - hidden_size_sqrt + 3) // 3)
        leftover = img_width - (hidden_size_sqrt + 2*kernel_size - 2)
        last_kernel_size = leftover + 1
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=num_channels, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=3, kernel_size=last_kernel_size),
            nn.Tanh(),
        )
    
    def forward(self, noise):
        # noise shape (batch, noise_size)
        batch_size, noise_size = noise.shape
        lineared = self.linear(noise)
        # shape (batch, hidden_size)
        reshaped = lineared.reshape(batch_size, 1, self.hidden_size_sqrt, self.hidden_size_sqrt)
        deconved = reshaped
        deconved = self.deconv(reshaped)
        return deconved


class Discriminator(nn.Module):
    def __init__(self, num_kernels=64, kernel_size=16, stride=1, pool_size=16):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_kernels, kernel_size=kernel_size, stride=stride),
            nn.MaxPool2d(kernel_size=pool_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, stride=stride),
            nn.MaxPool2d(kernel_size=pool_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, stride=stride),
            nn.MaxPool2d(kernel_size=pool_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, stride=stride),
            nn.MaxPool2d(kernel_size=pool_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size, stride=stride),
            nn.MaxPool2d(kernel_size=pool_size, stride=stride),
            nn.ReLU(),
        )
        c = (160 - kernel_size) // stride + 1 # width of output of first convolution
        p = (c - pool_size) // stride + 1 # width of output of second convolution
        for i in range(4): # twice more
            c = (p - kernel_size) // stride + 1 # width of output of next convolution
            p = (c - pool_size) // stride + 1 # width of output of next convolution
        self.flat_conved_size = p**2 * num_kernels

        self.linear = nn.Linear(self.flat_conved_size, 1)
        # sigmoid done by bce with logit loss

    
    def forward(self, img):
        conved = self.conv(img)
        batch_size, num_channels, height, width = conved.shape
        flattened = conved.view(batch_size, num_channels*height*width)
        linear = self.linear(flattened)
        return linear
        
        