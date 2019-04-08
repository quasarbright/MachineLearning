# Image Uncropping
Uncrops images using a convolutional autoencoder architecture  
architecture:  
```python
UnCropper(
  (conv): Sequential(
    (0): Conv2d(3, 64, kernel_size=(6, 6), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(6, 6), stride=(1, 1))
    (3): ReLU()
    (4): Conv2d(64, 64, kernel_size=(6, 6), stride=(1, 1))
    (5): ReLU()
  )
  (restore): Sequential(
    (0): ConvTranspose2d(64, 64, kernel_size=(6, 6), stride=(1, 1))
    (1): ReLU()
    (2): ConvTranspose2d(64, 64, kernel_size=(6, 6), stride=(1, 1))
    (3): ReLU()
    (4): ConvTranspose2d(64, 64, kernel_size=(6, 6), stride=(1, 1))
    (5): ReLU()
  )
  (generate): Sequential(
    (0): ConvTranspose2d(64, 3, kernel_size=(8, 8), stride=(1, 1))
    (1): Tanh()
  )
)
```
* trained with mean squared error loss and stochastic gradient descent with momentum
* uses the CIFAR-10 dataset with 32x32 images cropped to 25x25
* the model takes in a cropped 25x25 image and outputs a 32x32 image
loss curve over 200 epochs of training:
[loss curve during training](https://raw.githubusercontent.com/quasarbright/MachineLearning/master/uncrop/figures/25x25%20uncrop%20tanh%20mse%20losses.png)
Here is the model's performance on 10 random test images the model has never seen:
[validation examples](https://raw.githubusercontent.com/quasarbright/MachineLearning/master/uncrop/figures/25x25%20uncrop%20tanh%20mse%20guesses.png)
The top row is the uncropped, ground truth, 32x32 CIFAR-10 test images. The middle row is the cropped, 25x25 images which were the inputs to the model. The bottom row is the model's reconstruction of the original image