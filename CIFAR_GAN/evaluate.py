import random
import matplotlib.pyplot as plt
import matplotlib.animation
import torch
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils
import numpy as np
from utils import *
# from load_data import testloader
classes = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)

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

def animated_noise(class_index):
    class_name = classes[class_index]
    g = load_model('generator')
    grid_size=4
    noise1 = get_noise(grid_size**2, g.noise_size)
    noise2 = get_noise(grid_size**2, g.noise_size)
    class_index = torch.full((grid_size**2,), class_index).long().to(device)
    frames = 100
    fps = 30
    imgs = g(noise1, class_index)
    imgs = vutils.make_grid(imgs, nrow=grid_size, normalize=True)
    fig = plt.figure()
    plt.title(class_name)
    plt.axis('off')
    im = plt.imshow(np.transpose(imgs.detach().cpu().numpy(), (1, 2, 0)))
    def init():
        nonlocal noise1, noise2
        noise1 = noise2
        noise2 = get_noise(grid_size**2, g.noise_size)
        return im,
    def animate(i):
        noise = noise1 + (i / frames) * (noise2-noise1)
        imgs = g(noise, class_index)
        imgs = vutils.make_grid(imgs, nrow=grid_size, normalize=True)
        im.set_data(np.transpose(imgs.detach().cpu().numpy(), (1, 2, 0)))
        return im,
    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init,
                                frames=frames, interval=frames/fps, blit=True)
    plt.show()

if __name__ == '__main__':
    # show_generated(8,8)
    # g = load_model('generator')
    # car = g.embed(torch.LongTensor([[0]]).to(device)).cpu().detach().numpy()
    # bird = g.embed(torch.LongTensor([[1]]).to(device)).cpu().detach().numpy()
    # truck = g.embed(torch.LongTensor([[9]]).to(device)).cpu().detach().numpy()
    # print(car, bird, truck, sep='\n')
    animated_noise(random.randint(0, 9))

