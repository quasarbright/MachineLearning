import torch
import torchvision
import torchvision.transforms as transforms
from utils import device

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

batch_size = 500

def get_tensor(cifar_set):
    data = cifar_set.data
    data = torch.Tensor(data)
    data = data.to(device) # move to gpu if possible
    data = data / 255 # 0 to 1
    data = data * 2 - 1
    data = data.permute(0, 3, 1, 2) # color channel needs to be second dimension
    return data

def get_dataloader(cifar_set):
    data = get_tensor(cifar_set)
    dataset =  torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

trainloader = get_dataloader(cifar_trainset)
testloader = get_dataloader(cifar_testset)