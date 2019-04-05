import torch
import torchvision
import torchvision.transforms as transforms
from utils import device

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# parse loaders and do cropping
default_crop_ratio = .8


def crop_image(img, ratio=default_crop_ratio):
    '''
    img tensor of shape (batch, 3, height, width)
    crops image by reducing width to width*ratio and same for height
    approximately maintains aspect ratio
    '''
    batch, channels, height, width = img.shape
    # distance from center to left and right of cropped image
    x_rad = (width/2) * ratio
    l_ind = int(width/2 - x_rad)
    r_ind = int(width/2 + x_rad)
    # distance from center to top and bottom of cropped image
    y_rad = (height/2) * ratio
    d_ind = int(height/2 + y_rad)
    u_ind = int(height/2 - y_rad)
    return img[:, :, u_ind:d_ind, l_ind:r_ind]


def class_to_crop(loader, crop_ratio=default_crop_ratio):
    uncropped_all = []
    cropped_all = []
    for i, data in enumerate(loader, 0):
        img, label = data
        cropped = crop_image(img, crop_ratio)
        uncropped_all.append(img)
        cropped_all.append(cropped)
    # concatenate and normalize (pixel values from 0 to 1)
    uncropped_all = torch.cat(uncropped_all)
    uncropped_all = (uncropped_all + torch.ones(uncropped_all.shape)) / 2
    uncropped_all = uncropped_all.to(device)
    cropped_all = torch.cat(cropped_all)
    cropped_all = (cropped_all + torch.ones(cropped_all.shape)) / 2
    cropped_all = cropped_all.to(device)
    return uncropped_all, cropped_all


def load_datasets(crop_ratio=default_crop_ratio):
    global trainloader, testloader
    trainset = class_to_crop(trainloader, crop_ratio)
    uncropped, cropped = trainset
    testset = class_to_crop(testloader, crop_ratio)

    trainset = torch.utils.data.TensorDataset(*trainset)
    testset = torch.utils.data.TensorDataset(*testset)

    return trainset, testset, uncropped.shape, cropped.shape

def get_dataloaders(crop_ratio=default_crop_ratio, batch_size=500):
    trainset, testset, uncropped_shape, cropped_shape = load_datasets(crop_ratio)
    trainloader_ = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    testloader_ = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    return trainloader_, testloader_, uncropped_shape, cropped_shape
