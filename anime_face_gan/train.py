import torch
import torch.optim
from torch import nn
from myutils import *
from model import *
from load_data import trainloader


def get_noise(n, size):
    '''
    out (n, size, 1, 1) noise vector
    '''
    return torch.randn(n, size, 1, 1).to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(noise_size=100, num_kernels=64, lr=.0005, b1=.5, b2=.999, epochs=50):
    g = Generator(noise_size, num_kernels).to(device)
    g.apply(weights_init)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))

    d = Discriminator(num_kernels).to(device)
    d.apply(weights_init)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))
    d_loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        it = iter(enumerate(trainloader))
        i, (img, class_index) = next(it)
        while True:

            img = img.to(device)
            g.zero_grad()
            d.zero_grad()
            batch_size = img.shape[0]
            # train d on real images
            real_truth = torch.ones((batch_size, 1)).to(device)
            real_pred = d(img)
            real_pred_flat = real_pred.view(batch_size, -1)
            real_loss = d_loss_fn(real_pred_flat, real_truth)
            real_loss.backward(retain_graph=True)
            d_optimizer.step()

            # generate fakes
            noise = get_noise(batch_size, noise_size)
            fake_img = g(noise)

            # train g on fake images
            fake_truth = torch.zeros((batch_size, 1)).to(device)
            fake_pred = d(fake_img)
            fake_pred_flat = fake_pred.view(batch_size, -1)
            fake_loss = d_loss_fn(fake_pred_flat, fake_truth)
            fake_loss.backward(retain_graph=True)
            d_optimizer.step()
            d_loss = (fake_loss.item() + real_loss.item()) / 2
            d_losses.append(d_loss)

            # backprob losses
            fake_pred = d(fake_img)
            fake_pred_flat = fake_pred.view(batch_size, -1)
            g_loss = d_loss_fn(fake_pred_flat, real_truth)
            g_losses.append(g_loss.item())
            g_loss.backward()
            g_optimizer.step()

            if i % 50 == 0:
                print('losses at iteration {} in epoch {}:\n\tdiscriminator: {}\n\tgenerator: {}'.format(i, epoch, d_loss, g_loss))
            try:
                i, (img, class_index) = next(it)
            except RuntimeError as e:
                print(e)
                continue
            except StopIteration:
                break

        save_model(d, 'discriminator')
        save_model(g, 'generator')
        avg_d_loss = sum(d_losses) / max(1, len(d_losses))
        avg_g_loss = sum(g_losses) / max(1, len(g_losses))
        print('losses at epoch {}:\n\tdiscriminator: {}\n\tgenerator: {}'.format(epoch, avg_d_loss, avg_g_loss))

if __name__ == '__main__':
    train()
