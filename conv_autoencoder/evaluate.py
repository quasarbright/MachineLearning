import matplotlib.pyplot as plt
from utils import *
from load_data import testloader

def to_img(tensor):
    # tensor is -1 to 1
    tensor = tensor + 1
    tensor = tensor / 2
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def show_reconstructions(model, num_examples=5):
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


def show_lerp(model, dr=.2):
    print('loading data')
    data = iter(testloader).next()
    print('data loaded')
    imgs = data[0]
    imgs = imgs[:2]
    img1, img2 = imgs
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    # how many images of interpolation will there be?
    num_images = int(1 // dr + 1)
    for i in range(num_images):
        r = i*dr
        interpolated_image = model.lerp(img1, img2, r)
        interpolated_image = interpolated_image.cpu().detach()
        interpolated_image = interpolated_image.squeeze(0)
        interpolated_image = interpolated_image.permute(1,2,0)

        interpolated_image = to_img(interpolated_image)
        ax = plt.subplot(1, num_images, i+1)
        plt.imshow(interpolated_image.squeeze(0))
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    model = load_model('model1')
    model.eval()
    # show_reconstructions(model)
    show_lerp(model, .1)
