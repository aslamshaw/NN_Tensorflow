"""Evaluating a neural network for classification problem"""
import os
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importing the CIFAR-10 dataset of 32x32 pixels colored images
cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images.shape)

# Normalize the images from [0 255] to [0 1]
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Displays the normalized images
def show():
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

show()

# CNN Model using Sequential API
# Doing so would add new features to the dataset
model = keras.models.Sequential()
# Input size is 32x32x3 since images are colored i.e. RGB
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D((2,2)))
# Flatten the multi-dimensional image to 1D vector for the neural network
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# Output would be a probability distribution
model.add(layers.Dense(10))
print(model.summary())
#import sys; sys.exit()

# Loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]
model.compile(optimizer=optim, loss=loss, metrics=metrics)

# Training
batch_size = 64
epochs = 5
model.fit(train_images, train_labels, epochs=epochs,
          batch_size=batch_size, verbose=2)

# Evaulate
model.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)