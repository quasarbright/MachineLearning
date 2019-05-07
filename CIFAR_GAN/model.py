import math
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_size, num_channels, num_classes, embed_dims):
        super().__init__()
        assert int(math.sqrt(noise_size)) == math.sqrt(noise_size) # noise size square
        self.noise_size = noise_size
        self.embed = nn.Embedding(num_classes, embed_dims)
        self.transform_noise = nn.Bilinear(noise_size, embed_dims, noise_size)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_size, num_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels * 8),
            nn.ReLU(True),
            # state size. (num_channels*8) x 4 x 4
            nn.ConvTranspose2d(num_channels * 8, num_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(True),
            # state size. (num_channels*4) x 8 x 8
            nn.ConvTranspose2d(num_channels * 4, num_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(True),
            # state size. (num_channels*2) x 16 x 16
            nn.ConvTranspose2d(num_channels * 2, num_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            # state size. (num_channels) x 32 x 32
            nn.ConvTranspose2d(num_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    
    def forward(self, noise, class_index):
        '''
        noise (batch, noise_size, 1, 1)
        out (batch, 3, img_size, img_size)
        '''
        batch_size = noise.shape[0]
        noise = noise.view(batch_size, -1)
        embedding = self.embed(class_index)
        transformed_noise = self.transform_noise(noise, embedding)
        transformed_noise = transformed_noise.view(batch_size, -1, 1, 1)
        return self.main(transformed_noise)


class Discriminator(nn.Module):
    def __init__(self, num_kernels, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, num_kernels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_kernels) x 32 x 32
            nn.Conv2d(num_kernels, num_kernels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_kernels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_kernels*2) x 16 x 16
            nn.Conv2d(num_kernels * 2, num_kernels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_kernels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_kernels*4) x 8 x 8
            nn.Conv2d(num_kernels * 4, num_kernels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_kernels * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (num_kernels*8) x 4 x 4
        )
        self.realness = nn.Conv2d(num_kernels * 8, 1, 4, 1, 0, bias=False)
        self.classify = nn.Linear(num_kernels*8*4*4, num_classes)
    
    def forward(self, img):
        '''
        img (batch, 3, IMG_SIZE, IMG_SIZE)
        out (batch, 1), (batch, num_classes) realness, class logits
        '''
        batch_size = img.shape[0]

        main = self.main(img)
        realness = self.realness(main)
        realness = realness.view(batch_size, 1)

        main_flat = main.view(batch_size, -1)
        classification = self.classify(main_flat)
        return realness, classification

if __name__ == '__main__':
    g = Generator(100, 128)
    d = Discriminator(128)

    noise = torch.randn(5, 100, 1, 1)
    img = torch.rand((5, 3, 64, 64))
    print(g(noise).shape)
    print(d(img)) # not in [0,1] because they're logits
