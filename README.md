---
title: Machine Learning
---
Machine learning projects I've done
# UnCropper
This is a de-convolutional neural network that uncrops an image and generates the surrounding area. I fed it cropped images and had it guess the original. It doesn't generate unknown parts of the image very well.  
![](https://quasarbright.github.io/MachineLearning/uncrop/figures/25x25%20uncrop%20tanh%20mse%20guesses.png)  
The top row is the original image, the middle is the cropped image, and the bottom is what my NN generated.  
# Convolutional Autoencoder
This Convolves an image to a small vector, and then deconvolves it back to the original image. If you convolve two images, average or interpolate their compressed forms as vectors, and then decompress that, you can continuously morph between images
![](https://quasarbright.github.io/MachineLearning/conv_autoencoder/figures/car%20animal%20lerp.png)
# CIFAR GAN
A neural network that generates fake images, and one that determines if an image is real or one of the other network's fakes compete with each other and get better together. The result is that the faker learns to generate images that look real to humans! That is, unless you look too closely. 
![](https://quasarbright.github.io/MachineLearning/CIFAR_GAN/figures/35%20epochs%208x8.png)
