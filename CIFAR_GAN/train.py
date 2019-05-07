import torch
import torch.optim
from torch import nn
from utils import *
from model import *
from load_data import trainloader, testloader


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

def train(noise_size=100, num_kernels=64, embed_dims=5, lr=.0005, b1=.5, b2=.999, epochs=50):
    g = Generator(noise_size, num_kernels, 10, embed_dims).to(device)
    g.apply(weights_init)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))

    d = Discriminator(num_kernels, 10).to(device)
    d.apply(weights_init)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))
    d_loss_fn = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        d_losses = []
        class_losses = []
        g_losses = []
        for i, (img, class_index) in enumerate(trainloader):
            img = img.to(device)
            class_index = class_index.to(device)
            g.zero_grad()
            d.zero_grad()
            batch_size = img.shape[0]
            # train d on real images
            real_truth = torch.ones((batch_size, 1)).to(device)
            real_pred, real_class_pred = d(img)
            real_pred_flat = real_pred.view(batch_size, -1)
            real_loss = d_loss_fn(real_pred_flat, real_truth)
            real_loss.backward(retain_graph=True)
            d_optimizer.step()

            # real classification loss
            real_class_loss = class_loss_fn(real_class_pred, class_index)
            real_class_loss.backward(retain_graph=True)
            d_optimizer.step()
            class_loss = real_class_loss.item()
            class_losses.append(class_loss)

            # generate fakes
            noise = get_noise(batch_size, noise_size)
            fake_class_index = torch.randint(10, class_index.shape).to(device)
            fake_img = g(noise, fake_class_index)

            # train g on fake images
            fake_truth = torch.zeros((batch_size, 1)).to(device)
            fake_pred, fake_class_pred = d(fake_img)
            fake_pred_flat = fake_pred.view(batch_size, -1)
            fake_loss = d_loss_fn(fake_pred_flat, fake_truth)
            fake_loss.backward(retain_graph=True)
            d_optimizer.step()
            d_loss = (fake_loss.item() + real_loss.item()) / 2
            d_losses.append(d_loss)

            # backprob g losses
            fake_pred, fake_class_pred = d(fake_img)
            fake_pred_flat = fake_pred.view(batch_size, -1)
            fake_class_loss = class_loss_fn(fake_class_pred, fake_class_index)
            fake_class_loss.backward(retain_graph=True)
            g_loss = d_loss_fn(fake_pred_flat, real_truth)
            g_losses.append(g_loss.item())
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            if i % 50 == 0:
                print('losses at iteration {} in epoch {}:\n\tdiscriminator: {}\n\tclass: {}\n\tgenerator: {}'.format(i, epoch, d_loss, class_loss, g_loss))

        save_model(d, 'discriminator')
        save_model(g, 'generator')
        avg_d_loss = sum(d_losses) / max(1, len(d_losses))
        avg_g_loss = sum(g_losses) / max(1, len(g_losses))
        print('losses at epoch {}:\n\tdiscriminator: {}\n\tgenerator: {}'.format(epoch, avg_d_loss, avg_g_loss))

if __name__ == '__main__':
    train()
