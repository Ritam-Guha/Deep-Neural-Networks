from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

file_name = 'chest_xray'
PATH='/content/drive/My Drive/Datasets/' + file_name
class_1 = 'PNEUMONIA'
class_2 = 'NORMAL'

train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'val')
test_dir = os.path.join(PATH, 'test')

train_class1_dir = os.path.join(train_dir, class_1)  
train_class2_dir = os.path.join(train_dir, class_2)  

val_class1_dir = os.path.join(val_dir, class_1)  
val_class2_dir = os.path.join(val_dir, class_2)  

test_class1_dir = os.path.join(test_dir, class_1)  
test_class2_dir = os.path.join(test_dir, class_2)

num_class1_tr = len(os.listdir(train_class1_dir))
num_class2_tr = len(os.listdir(train_class2_dir))

num_class1_val = len(os.listdir(val_class1_dir))
num_class2_val = len(os.listdir(val_class2_dir))

num_class1_test = len(os.listdir(test_class1_dir))
num_class2_test = len(os.listdir(test_class2_dir))

total_train = num_class1_tr + num_class2_tr
total_val = num_class1_val + num_class2_val
total_test = num_class1_test + num_class2_test

print('total training ' + class_1 + ' images:', num_class1_tr)
print('total training ' + class_2 + ' images:', num_class2_tr)

print('total validation ' + class_1 + ' images:', num_class1_val)
print('total validation ' + class_2 + ' images:', num_class2_val)

print('total test ' + class_1 + ' images:', num_class1_val)
print('total test ' + class_2 + ' images:', num_class2_val)

print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
print("Total test images:", total_test)

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) 
val_image_generator = ImageDataGenerator(rescale=1./255) 
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(batch_size=total_test,
                                                         directory=test_dir,
                                                         target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                         class_mode='binary')

test_data_X, test_data_Y = next(test_data_gen)
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

history = model.fit(
    train_data_gen,
    steps_per_epoch=np.ceil(total_train / batch_size).astype(int),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=np.ceil(total_val / batch_size).astype(int)
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

# print(test_data_Y)
loss, accuracy = model.evaluate(x = test_data_X, y = test_data_Y)

