import os
import torch
import torch.utils.data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i]

data_path = 'D:\\datasets\\animeface-character-dataset\\thumb'
data_save_path = 'D:\\datasets\\animeface-character-dataset\\datasets'
img_size = (160, 160)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def get_data_save_path(name):
    return os.path.join(data_save_path, '{}.pt'.format(name))

def save_dataset(dataset, name):
    torch.save(dataset, get_data_save_path(name))

def load_dataset(name):
    return torch.load(get_data_save_path(name))

def get_save_path(name):
    return 'saves/{}.pt'.format(name)


def save_model(model, name):
    torch.save(model, get_save_path(name))


def load_model(name):
    return torch.load(get_save_path(name), map_location=device)
