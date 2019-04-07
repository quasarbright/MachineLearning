import matplotlib.pyplot as plt
from utils import *
from load_data import testloader

def to_img(tensor):
    # tensor is -1 to 1
    tensor = tensor + 1
    tensor = tensor / 2
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def show_examples(model, num_examples=5):
    print('loading data')
    data = iter(testloader).next()
    print('data loaded')
    img = data[0]
    img = img[:num_examples]
    restored = model(img)

    # make color channels last dimension
    imgs = img.permute(0, 2, 3, 1)
    restoreds = restored.permute(0, 2, 3, 1)

    for i, (img, restored) in enumerate(zip(imgs, restoreds)):
        # iterate over batch of examples and show them

        # show original image on top
        ax = plt.subplot(2, num_examples, i + 1)
        img = img.cpu()
        img = to_img(img)
        plt.imshow(img)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        # show cropped in the middle
        ax = plt.subplot(2, num_examples, num_examples + i + 1)
        restored = restored.cpu().detach()
        restored = to_img(restored)
        plt.imshow(restored)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    model = load_model('model1')
    model.eval()
    show_examples(model)
