import torch
from torch import nn
import matplotlib.pyplot as plt

from load_data import trainloader
from model import Autoencoder
from utils import *
# from evaluate import show_examples


def train_model(batch_size=500, lr=.3, momentum=.9, epochs=100, num_kernels=32, kernel_size=8, plot=True):
    config = locals()
    model = Autoencoder(num_kernels, kernel_size).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print("beginning training")
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        num_losses = 0
        for data in trainloader:
            img = data[0]
            model.zero_grad()
            # (batch, 3, height, width)
            restored = model(img)
            assert img.shape == restored.shape
            loss = loss_fn(restored, img)
            loss.backward()
            total_loss += loss.data
            num_losses += 1
            optimizer.step()
        avg_loss = total_loss / num_losses
        losses.append(avg_loss)
        print('loss at epoch {}: {}'.format(epoch, avg_loss))
    print("training complete. final loss after {} epochs: {}".format(
        epochs, losses[-1]))
    plt.plot(losses)
    return model, losses

def train_many(*configs):
    '''
    trains all configs
    configs is list of dictionaries with kwargs for training
    plots losses
    returns models and loss lists in order
    '''
    models = []
    loss_lists = []
    for config in configs:
        # config is a dictionary of kwargs
        print('training config: {}'.format(config))
        model, losses = train_model(**config, plot=False)
        models.append(model)
        loss_lists.append(losses)
        plt.plot(losses)
    plt.legend([str(config) for config in configs], loc='upper right')
    plt.show()
    return models, loss_lists



if __name__ == '__main__':
    # configs = [
    #     {
    #         'num_kernels': 32,
    #         'kernel_size': 7,
    #         'crop_ratio': .95
    #     },
    #     {
    #         'num_kernels': 32,
    #         'kernel_size': 7,
    #         'crop_ratio': .8
    #     },
    #     {
    #         'num_kernels':32,
    #         'kernel_size':7,
    #         'crop_ratio':.95,
    #         'momentum':.8,
    #         'lr':.5
    #     },
    #     {
    #         'num_kernels':64,
    #         'kernel_size':7,
    #         'crop_ratio':.95,
    #     }
    # ]
    # train_many(*configs)
    model, losses = train_model()
    save_model(model, 'model1')