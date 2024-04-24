import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dataset = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
input_shape = (28, 28, 1)
# Functional CNN model

inputs = tf.keras.layers.Input(input_shape)
x_layer = tf.keras.layers.Conv2D(filters=8,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(inputs)

x_layer = tf.keras.layers.MaxPool2D((2, 2))(x_layer)

x_layer = tf.keras.layers.Conv2D(filters=16,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(x_layer)

x_layer = tf.keras.layers.MaxPool2D((2, 2))(x_layer)

x_layer = tf.keras.layers.Conv2D(filters=32,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(x_layer)

x_layer = tf.keras.layers.MaxPool2D((2, 2))(x_layer)

x_layer = tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu')(x_layer)

x_layer = tf.keras.layers.Flatten()(x_layer)
output_layer = tf.keras.layers.Dense(units=10, activation='softmax')(x_layer)
model = tf.keras.models.Model(inputs, output_layer)
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          epochs=50,
          batch_size=32,
          validation_split=0.15)

# Evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('CNN_classifier.h5')

