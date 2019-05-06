import torch
import torch.optim
from torch import nn
import torch.distributions
from utils import *
from model import *
from load_data import trainloader, testloader

norm_dist = torch.distributions.normal.Normal(0, 1)
def get_noise(n, size):
    '''
    out (n, size) noise vector
    '''
    return norm_dist.sample((n, size)).to(device)

def train(noise_size=16**2, num_channels=128, num_kernels=128, kernel_size=10, pool_size=2, hidden_size=120, lr=.0005, b1=.5, b2=.999, epochs=200):
    g = Generator(noise_size, num_channels).to(device)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))

    d = Discriminator(num_kernels, kernel_size, pool_size, hidden_size).to(device)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))
    d_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for img, in trainloader:
            g.zero_grad()
            d.zero_grad()
            batch_size = img.shape[0]
            # train d on real images
            real_truth = torch.ones((batch_size, 1)).to(device)
            real_pred = d(img)
            real_loss = d_loss_fn(real_pred, real_truth)

            # generate fakes
            noise = get_noise(batch_size, noise_size)
            fake_img = g(noise)

            # train g on fake images
            fake_truth = torch.zeros((batch_size, 1)).to(device)
            fake_pred = d(fake_img)
            fake_loss = d_loss_fn(fake_pred, fake_truth)

            # backprob losses
            g_loss = d_loss_fn(fake_pred, real_truth)
            g_losses.append(g_loss.item())
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            d_loss = (fake_loss + real_loss) / 2
            d_losses.append(d_loss.item())
            d_loss.backward()
            d_optimizer.step()
        save_model(d, 'discriminator')
        save_model(g, 'generator')
        avg_d_loss = sum(d_losses) / max(1, len(d_losses))
        avg_g_loss = sum(g_losses) / max(1, len(g_losses))
        print('losses at epoch {}:\n\tdiscriminator: {}\n\tgenerator: {}'.format(epoch, avg_d_loss, avg_g_loss))

if __name__ == '__main__':
    train()
