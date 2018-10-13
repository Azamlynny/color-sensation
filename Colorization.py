from keras.layers import Conv2D, UpSampling2D, InputLayer, Cropping2D
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf
import tensorflowjs as tfjs

num_epochs = 100
batch_size = 80

# Input Data
X = []
for filename in os.listdir('C:/Users/Adam/Desktop/HackUC III/color-sensation/Training/'):
    if filename != '.DS_Store':
        X.append(img_to_array(load_img('C:/Users/Adam/Desktop/HackUC III/color-sensation/Training/' + filename, target_size=(256, 256))))
X = np.array(X, dtype=float)
X *= 1.0 / 255

# Create the structure
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.compile(optimizer='rmsprop', loss='mse')

# Get transformed images for training
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Function to provide training data
def generateTrain(batch_size):
    for batch in datagen.flow(X, batch_size=batch_size):
        batch_lab = rgb2lab(batch)
        batch_x = batch_lab[:,:,:,0]
        batch_y = batch_lab[:,:,:,1:] / 128.0
        yield (batch_x.reshape(batch_x.shape + (1,)), batch_y)

# Train model
tensorboard = TensorBoard()
model.fit_generator(generateTrain(batch_size), callbacks=[tensorboard], epochs=num_epochs, steps_per_epoch=2)

# Save weights
model.save("model.h5")

# Output some tests
tests = X
tests = rgb2lab(tests)[:,:,:,0]
tests = tests.reshape(tests.shape+(1,))

output = model.predict(tests)
output *= 128

# Combine the outputs with the original to form a colored image
for i in range(len(output)):
	cur = np.zeros((256, 256, 4))
	cur[:,:,0] = tests[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave(str(i) + ".png", lab2rgb(cur))
	imsave(str(i) + "b.png", rgb2gray(lab2rgb(cur)))