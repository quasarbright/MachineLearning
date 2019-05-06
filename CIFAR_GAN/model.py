import math
import torch
from torch import nn
from utils import IMG_SIZE

class Generator(nn.Module):
    def __init__(self, noise_size, num_channels):
        super().__init__()
        assert int(math.sqrt(noise_size)) == math.sqrt(noise_size) # noise size square
        self.noise_size = noise_size
        self.sqrt_noise_size = int(math.sqrt(noise_size))
        '''
        after deconv size is sqrt(noise_size) + kernel_size * 2 - 2 == IMG_SIZE
        kernel_size = (IMG_SIZE - sqrt(noise_size) + 2) / 2
        '''
        kernel_size = int((IMG_SIZE - self.sqrt_noise_size + 2) / 2)
        activation = nn.LeakyReLU(.2, inplace=True)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1, num_channels, kernel_size),
            nn.BatchNorm2d(num_channels, .8),
            activation,
            nn.ConvTranspose2d(num_channels, 3, kernel_size),
            nn.Tanh()
        )

        after_deconv_size = self.sqrt_noise_size + kernel_size * 2 - 2
        extra_needed_size = IMG_SIZE - after_deconv_size
        if extra_needed_size > 0:
            extra_kernel_size = IMG_SIZE - after_deconv_size + 1
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(1, num_channels, kernel_size),
                nn.BatchNorm2d(num_channels, .8),
                activation,
                nn.ConvTranspose2d(num_channels, num_channels, kernel_size),
                nn.BatchNorm2d(num_channels, .8),
                activation,
                nn.ConvTranspose2d(num_channels, 3, extra_kernel_size),
                nn.Tanh()
            )
    
    def forward(self, noise):
        '''
        noise (batch, noise_size)
        out (batch, 3, IMG_SIZE, IMG_SIZE)
        '''
        batch_size, noise_size = noise.shape
        assert noise_size == self.noise_size
        noise_square = noise.view(batch_size, self.sqrt_noise_size, self.sqrt_noise_size)
        noise_square = noise_square.unsqueeze(1) # dummy color channel
        # shape (batch, 1, sqrt_noise_size, sqrt_noise_size)
        return self.deconv(noise_square)


class Discriminator(nn.Module):
    def __init__(self, num_kernels, kernel_size, pool_size, hidden_size):
        super().__init__()
        activation = nn.LeakyReLU(.2, inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(3, num_kernels, kernel_size),
            nn.MaxPool2d(pool_size, stride=1),
            activation,
            nn.Dropout2d(.15),
            nn.BatchNorm2d(num_kernels, .8),
            nn.Conv2d(num_kernels, num_kernels, kernel_size),
            nn.MaxPool2d(pool_size, stride=1),
            activation,
            nn.Dropout2d(.15),
            nn.BatchNorm2d(num_kernels, .8)
        )

        conved_size = IMG_SIZE - 2*(kernel_size + pool_size) + 4
        conved_len = conved_size **2 * num_kernels # number of elements in the after conv

        self.fc = nn.Sequential(
            nn.Linear(conved_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(.15),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(.15),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
            # use BCE with logit loss
        )
    
    def forward(self, img):
        '''
        img (batch, 3, IMG_SIZE, IMG_SIZE)
        out (batch, 1) logits
        '''
        after_conv = self.conv(img)
        batch_size, num_kernels, width, height = after_conv.shape
        after_conv_flat = after_conv.view(batch_size, num_kernels*width*height)
        out = after_conv_flat
        for layer in self.fc:
            out = layer(out)
        return self.fc(after_conv_flat)

if __name__ == '__main__':
    g = Generator(16**2, 128)
    d = Discriminator(128, 10, 2, 100)

    noise = torch.rand((5, 16**2))
    img = torch.rand((5, 3, 32, 32))
    print(g(noise).shape)
    print(d(img)) # not in [0,1] because they're logits
