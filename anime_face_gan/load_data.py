import torch
import torchvision
import torchvision.transforms as transforms
from myutils import *
from PIL import Image

image_size = 64
batch_size = 64

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
print('loading and processing images')
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform, loader=Image.open)
# for some reason, there are 2 images with 4 color channels, so we drop them
# good_shape = torch.zeros((3, 160, 160)).shape
# for i, (image, class_index) in enumerate(dataset):
#     if image.shape != good_shape:
#         print(i) # prints 11659, 11662

#remove 4-channel images
# dataset[11659] = dataset[0]
# dataset[11662] = dataset[0]

print('done')

num_images = len(dataset)
print('generating loaders')
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)
print('done')