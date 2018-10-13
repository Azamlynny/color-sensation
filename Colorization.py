from keras.layers import Conv2D, UpSampling2D, InputLayer, Cropping2D
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf

num_epochs = 30
batch_size = 10

X = []
for filename in os.listdir('../virtual/Train/'):
    if filename != '.DS_Store':
        X.append(img_to_array(load_img('../virtual/Train/' + filename, target_size=(256, 256))))
X = np.array(X, dtype=float)
X = 1 / 255.0 * X

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

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
def generateTrain(batch_size):
    for batch in datagen.flow(X, batch_size=batch_size):
        batch_lab = rgb2lab(batch)
        batch_x = batch_lab[:,:,:,0]
        batch_y = batch_lab[:,:,:,1:] / 128
        yield (batch_x.reshape(batch_x.shape + (1,)), batch_y)

# Train model      
tensorboard = TensorBoard()
model.fit_generator(generateTrain(batch_size), callbacks=[tensorboard], epochs=num_epochs, steps_per_epoch=1)

# Save weights
model.save_weights("weights.h5")

# Test
tests = X
tests = rgb2lab(tests)[:,:,:,0]
tests = tests.reshape(tests.shape+(1,))

output = model.predict(tests)
output = output * 128

# Output colorizations
for i in range(len(output)):
	cur = np.zeros((256, 256, 3))
	cur[:,:,0] = tests[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave(str(i) + ".png", lab2rgb(cur))
	imsave(str(i) + "b.png", rgb2gray(lab2rgb(cur)))