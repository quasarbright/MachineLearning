import torch

IMG_SIZE = 32

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def get_save_path(name):
    return 'saves/{}.pt'.format(name)


def save_model(model, name):
    torch.save(model, get_save_path(name))


def load_model(name):
    model = torch.load(get_save_path(name), map_location=device)
    model.eval()
    return model