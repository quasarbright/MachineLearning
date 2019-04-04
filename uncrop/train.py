import torch
from torch import nn
from load_data import get_dataloaders
from model import UnCropper
from utils import device
def train_model(crop_ratio=.8, batch_size=300, lr=.1, epochs=100):
    # cifar
    img_width, img_height = 32, 32
    img_size = (img_width, img_height)
    cropped_width = int(img_width * crop_ratio)
    cropped_height = int(img_height * crop_ratio)
    cropped_size = (cropped_width, cropped_height)

    model = UnCropper(img_size, cropped_size).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)

    trainloader, testloader = get_dataloaders(crop_ratio, batch_size)

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
            

if __name__ == '__main__':
    train_model()
