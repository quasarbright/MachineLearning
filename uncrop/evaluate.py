import matplotlib.pyplot as plt
from utils import *
from load_data import get_dataloaders

def show_examples(model, num_examples=5):
    crop_ratio = model.crop_ratio
    print('loading data')
    trainloader, testloader, uncropped_shape, cropped_shape = get_dataloaders(
        crop_ratio, num_examples)
    print('data loaded')
    data = iter(testloader).next()
    img, cropped = data
    uncropped = model(cropped)

    # make color channels last dimension
    img = img.permute(0, 2, 3, 1)
    cropped = cropped.permute(0, 2, 3, 1)
    uncropped = uncropped.permute(0, 2, 3, 1)

    for i, (img, cropped, uncropped) in enumerate(zip(img, cropped, uncropped)):
        # iterate over batch of examples and show them

        # show original image on top
        ax = plt.subplot(3, num_examples, i + 1)
        plt.imshow(img.cpu())
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        # show cropped in the middle
        ax = plt.subplot(3, num_examples, num_examples + i + 1)
        plt.imshow(cropped.cpu())
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        # show uncropped guess on the bottom
        ax = plt.subplot(3, num_examples, 2*num_examples + i + 1)
        plt.imshow(uncropped.cpu().detach().numpy())
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    model = load_model('model1')
    model.eval()
    show_examples(model)