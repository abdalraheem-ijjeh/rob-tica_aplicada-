import os
import numpy as np
import tensorflow as tf

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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

##################
# Hyperparameters
##################
epochs = 50
val_split = 0.15
batchs = 32
##################
image_shape = (28, 28, 1)
# Building the purly ANN model Using Functional method
inputs_X = tf.keras.layers.Input(image_shape)
x = tf.keras.layers.Flatten()(inputs_X)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(inputs_X, outputs)
#########################
# Show model summary
model.summary()
#########################

model.compile(optimizer='Adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batchs,
          epochs=epochs,
          validation_split=val_split)

# Evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('ANN_model_MNIST_CLASSIFIER.h5')
