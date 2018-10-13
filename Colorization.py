from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

#Import images from the Training folder

# Open a file
path = "C:/Users/Adam/Desktop/HackUC III/color-sensation/Training/"
dirs = os.listdir( path )

X = []
for filename in dirs:
    X.append(img_to_array(load_img(dirs + filename)))
X = np.array(X, dtype=float)

