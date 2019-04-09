import torch
import torch.utils.data
import torchvision
from PIL import Image
from myutils import *
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((160, 160)),
#     torchvision.transforms.ToTensor()
# ])
# print('loading and processing images')
# dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform, loader=Image.open)
# # drop class information because we don't care
# all_images = [image for image, classIndex in dataset]
# # for some reason, there are 2 images with 4 color channels, so we drop them
# good_shape = torch.zeros((3, 160, 160)).shape
# all_images = list(filter(lambda t: t.shape == good_shape, all_images))
# # now all_images is a tensor of every image. shape is (14490, 3, 160, 160)

# # map values from [0,1] to [-1,1]
# all_images = list(map(lambda t: t * 2 - 1, all_images))
# print('done')

# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, images):
#         self.images = images
#     def __len__(self):
#         return len(self.images)
#     def __getitem__(self, i):
#         return self.images[i]

# print('generating loaders')
# train_test_split = .2 # 20% test data
# num_images = len(all_images)
# split_index = int(num_images*train_test_split)
# train_images = all_images[split_index:]
# test_images = all_images[:split_index]
print('loading datasets')
trainset = load_dataset('trainset')
testset = load_dataset('testset')
print('done')
print('generating loaders')
batch_size = 10
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
print('done')