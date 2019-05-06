import matplotlib.pyplot as plt
import torch
import torch.distributions
from torchvision.transforms import ToPILImage
import numpy as np
from utils import *
# from load_data import testloader

norm_dist = torch.distributions.normal.Normal(0, 1)


def get_noise(n, size):
    '''
    out (n, size) noise vector
    '''
    return norm_dist.sample((n, size)).to(device)

def to_img(tensor):
    # tensor is -1 to 1
    tensor = tensor + 1
    tensor = tensor / 2
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def to_pil_image(tensor):
    f = ToPILImage()
    img = to_img(tensor)
    img = f(img.cpu().detach())
    return img

def show_generated(n=5):
    g = load_model('generator')
    noise = get_noise(n, g.noise_size)
    imgs = g(noise)
    imgs = [img for img in imgs]
    imgs = torch.cat(imgs, dim=2) # horizontal concatenation
    # imgs = img[0]
    all_imgs = to_pil_image(imgs)
    all_imgs.show()

if __name__ == '__main__':
    show_generated()

