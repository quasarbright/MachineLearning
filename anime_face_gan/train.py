import torch
from torch import nn
from myutils import *
from load_data import trainloader
from models import Generator, Discriminator
# noise = torch.rand((1, 64)).to(device)
# gen_ = Generator(noise_size=64).to(device)
# out = gen_(noise)
# print(out.shape)
# discriminator = Discriminator().to(device)
# disc = discriminator(out)
# print(disc)
def train_gan(noise_size=64, epochs=1):
    gen = Generator(noise_size=noise_size).to(device)
    disc = Discriminator().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    gen_optimizer = torch.optim.SGD(gen.parameters(), lr=.1, momentum=.9)
    disc_optimizer = torch.optim.SGD(disc.parameters(), lr=.1, momentum=.9)
    losses = []
    dist = torch.distributions.Normal(0, 1)
    get_noise = lambda batch_size: dist.sample((batch_size, noise_size)).to(device)
    for epoch in range(epochs):
        for img in trainloader:
            disc.zero_grad()
            # img is a batch of images
            img = img.to(device)
            batch_size = img.shape[0]
            # train discriminator with truth
            y_pred = disc(img)
            y_true = torch.ones((batch_size, 1)).to(device)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            disc_optimizer.step()
            del img
            del y_pred
            del y_true
            # train discriminator with fakes
            disc.zero_grad()
            noise = get_noise(batch_size)
            fake = gen(noise)
            y_pred = disc(fake)
            y_true = torch.zeros((batch_size, 1)).to(device)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            disc_optimizer.step()


            # train generator
            gen.zero_grad()
            noise = get_noise(batch_size)
            fake = gen(noise)
            y_pred = disc(fake)
            y_true = torch.zeros((batch_size, 1)).to(device)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            gen_optimizer.step()

            

train_gan()