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
Generative Adversarial Networks: A neural network that generates fake images, and one that determines if an image is real or one of the other network's fakes compete with each other and get better together. The result is that the faker learns to generate images that look real to humans! That is, unless you look too closely. 
![](https://quasarbright.github.io/MachineLearning/CIFAR_GAN/figures/35%20epochs%208x8.png)
# Anime Face GAN
This is the same as CIFAR GAN, but I trained it on images of anime faces. Fake faces look a lot weirder than fake animals, planes, and cars like in the CIFAR GAN.
![](https://quasarbright.github.io/MachineLearning/anime_face_gan/figures/50%20epochs%208x8.png)
# Snake REINFORCE NN
I made a neural network that learns how to play snake throuh the REINFORCE algorithm. I reward the NN when it gets food, and punish it when it dies, and it learned to play pretty well. Granted, it can't think ahead very well. Note: I made this on another repository, [YLUJO](https://github.com/quasarbright/YLUJLO) with my friend Maggie Von Nortwick. There is a failed attempt of this same project on this repository, so look at the one on YLUJLO instead if you want to see how this was made. I have the functionality for actor-critic in the current model, but I shut it off because the snake task is simple enough that actor-critic over-complicates things and makes it perform worse. I also have a double Q model in a separate branch that plays even better!
<iframe width="560" height="315" src="https://www.youtube.com/embed/LL62tmIUtGU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
