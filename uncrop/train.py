import torch
from torch import nn
import matplotlib.pyplot as plt

from load_data import get_dataloaders
from model import UnCropper
from utils import *
from evaluate import show_examples


def train_model(crop_ratio=.95, batch_size=500, lr=.3, momentum=.9, epochs=100, num_kernels=32, kernel_size=8):
    # cifar
    print("loading data")
    trainloader, testloader, uncropped_shape, cropped_shape = get_dataloaders(
        crop_ratio, batch_size)
    print("data loaded")
    img_width, img_height = uncropped_shape[3], uncropped_shape[2]
    img_size = (img_width, img_height)
    cropped_width, cropped_height = cropped_shape[3], cropped_shape[2]
    cropped_size = (cropped_width, cropped_height)

    model = UnCropper(crop_ratio, img_size, cropped_size,
                      num_kernels, kernel_size).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)

    print("beginning training with image size {} and cropped size {}".format(
        img_size, cropped_size))
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        num_losses = 0
        for index, data in enumerate(trainloader, 0):
            model.zero_grad()
            img, cropped = data
            # (batch, 3, height, width)
            uncropped = model(cropped)
            assert img.shape == uncropped.shape
            loss = loss_fn(uncropped, img)
            loss.backward()
            total_loss += loss.data
            num_losses += 1
            optimizer.step()
        avg_loss = total_loss / num_losses
        losses.append(avg_loss)
        print('loss at epoch {}: {}'.format(epoch, avg_loss))
    print("training complete. final loss after {} epochs: {}".format(
        epochs, losses[-1]))
    
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
        model, losses = train_model(**config)
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