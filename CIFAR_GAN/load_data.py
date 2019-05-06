import torch
import torchvision
import torchvision.transforms as transforms
from utils import device

image_size = 64

transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)

cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)

batch_size = 64

def get_dataloader(cifar_set):
    return torch.utils.data.DataLoader(cifar_set, batch_size=batch_size, shuffle=True, num_workers=0)

trainloader = get_dataloader(cifar_trainset)
testloader = get_dataloader(cifar_testset)

if __name__ == '__main__':
    imgs = iter(testloader).next()[0]
    print(imgs.shape)
    print(imgs.min())
    print(imgs.max())
