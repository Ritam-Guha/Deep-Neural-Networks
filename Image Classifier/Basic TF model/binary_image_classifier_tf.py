
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

datasetName = 'chest-xray'


PATH = os.getcwd()
parent = os.pardir
parent_absPATH = os.path.join(PATH, parent)
data = os.path.join(parent_absPATH, 'Data')  
dataset_absPATH = os.path.join(data, datasetName)  
train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')

# train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')  
# train_normal_dir = os.path.join(train_dir, 'NORMAL')  
# test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')  
# test_normal_dir = os.path.join(test_dir, 'NORMAL')

# num_pneumonia_tr = len(os.listdir(train_pneumonia_dir))
# num_normal_tr = len(os.listdir(train_normal_dir))

# num_pneumonia_val = len(os.listdir(test_pneumonia_dir))
# num_normal_val = len(os.listdir(test_normal_dir))

# total_train = num_pneumonia_tr + num_normal_tr
# total_val = num_pneumonia_val + num_normal_val

# print('total training cat images:', num_pneumonia_tr)
# print('total training dog images:', num_normal_tr)

# print('total validation cat images:', num_pneumonia_val)
# print('total validation dog images:', num_normal_val)
# print("--")
# print("Total training images:", total_train)
# print("Total validation images:", total_val)

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) 
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=test_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()