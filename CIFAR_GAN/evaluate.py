import matplotlib.pyplot as plt
import matplotlib.animation
import torch
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils
import numpy as np
from utils import *
# from load_data import testloader


def get_noise(n, size):
    '''
    out (n, size, 1, 1) noise vector
    '''
    return torch.randn(n, size, 1, 1).to(device)

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

def show_generated(nrows, ncols):
    g = load_model('generator')
    noise = get_noise(nrows * ncols, g.noise_size)
    imgs = g(noise)
    imgs = vutils.make_grid(imgs, nrow=nrows, normalize=True)
    plt.figure(figsize=(nrows, ncols))
    plt.axis('off')
    plt.imshow(np.transpose(imgs.detach().cpu().numpy(), (1,2,0)))
    plt.show()

def animated_noise():
    g = load_model('generator')
    grid_size=4
    noise1 = get_noise(grid_size**2, g.noise_size)
    noise2 = get_noise(grid_size**2, g.noise_size)
    frames = 100
    fps = 30
    imgs = g(noise1)
    imgs = vutils.make_grid(imgs, nrow=grid_size, normalize=True)
    fig = plt.figure()
    im = plt.imshow(np.transpose(imgs.detach().cpu().numpy(), (1, 2, 0)))
    plt.axis('off')
    def init():
        nonlocal noise1, noise2
        noise1 = noise2
        noise2 = get_noise(grid_size**2, g.noise_size)
        return im,
    def animate(i):
        noise = noise1 + (i / frames) * (noise2-noise1)
        imgs = g(noise)
        imgs = vutils.make_grid(imgs, nrow=grid_size, normalize=True)
        im.set_data(np.transpose(imgs.detach().cpu().numpy(), (1, 2, 0)))
        return im,
    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init,
                                frames=frames, interval=frames/fps, blit=True)
    plt.show()

if __name__ == '__main__':
    # show_generated(8,8)
    animated_noise()

